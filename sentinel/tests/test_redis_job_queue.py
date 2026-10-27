"""M4: RedisJobQueue — durable reliable-queue semantics, proven on fakeredis.

No live Redis needed: fakeredis.FakeAsyncRedis is injected through the client
seam. A shared FakeServer stands in for "the same Redis across two processes"
in the crash-recovery test. M7 adds outage log-hygiene/backoff and depth tests.
"""

import asyncio
import json
import logging

import fakeredis
import pytest

from sentinel.infrastructure.redis.redis_job_queue import RedisJobQueue


def _fake_client(server: fakeredis.FakeServer | None = None) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server, decode_responses=True)


def _queue(client: fakeredis.FakeAsyncRedis | None = None) -> RedisJobQueue:
    return RedisJobQueue(
        "redis://unused:6379/0",
        client=client if client is not None else _fake_client(),
        poll_interval=0.01,
    )


def test_constructor_does_not_connect():
    # from_url is lazy; building against a dead address must not raise.
    RedisJobQueue("redis://127.0.0.1:1/0")


def test_enqueue_rejects_non_dict():
    async def _run() -> None:
        queue = _queue()
        with pytest.raises(TypeError):
            await queue.enqueue("bad")  # type: ignore[arg-type]

    asyncio.run(_run())


def test_fifo_round_trip_preserves_nested_payload():
    async def _run() -> None:
        queue = _queue()
        jobs = [
            {
                "owner": "octo",
                "repo": "hello",
                "pr_number": 1,
                "files": ["a.py", "README.md"],
                "file_contents": {"a.py": "+x = 1"},
            },
            {"repo": "hello", "pr_number": 2},
            {"repo": "hello", "pr_number": 3},
        ]
        for job in jobs:
            await queue.enqueue(job)
        dequeued = [await queue.dequeue() for _ in jobs]
        assert dequeued == jobs

    asyncio.run(_run())


def test_dequeue_moves_message_to_processing_and_ack_clears_it():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "hello", "pr_number": 7})

        job = await queue.dequeue()
        assert await client.llen(RedisJobQueue.QUEUE_KEY) == 0
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 1

        await queue.ack(job)
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 0

    asyncio.run(_run())


def test_crash_recovery_requeues_unacked_job():
    async def _run() -> None:
        server = fakeredis.FakeServer()  # one "real" Redis shared by both instances
        first = _queue(_fake_client(server))
        await first.enqueue({"repo": "hello", "pr_number": 42})
        job = await first.dequeue()  # dequeued but never acked -> simulated crash

        second = _queue(_fake_client(server))  # fresh process after restart
        assert await second.recover_pending() == 1
        recovered = await asyncio.wait_for(second.dequeue(), timeout=2.0)
        assert recovered == job

    asyncio.run(_run())


def test_ack_is_noop_for_unknown_job():
    async def _run() -> None:
        queue = _queue()
        await queue.ack({"repo": "never-dequeued", "pr_number": 1})  # must not raise

    asyncio.run(_run())


def test_dequeue_waits_until_a_job_arrives():
    async def _run() -> None:
        queue = _queue()

        async def _enqueue_later() -> None:
            await asyncio.sleep(0.05)
            await queue.enqueue({"repo": "late", "pr_number": 9})

        enqueue_task = asyncio.create_task(_enqueue_later())
        job = await asyncio.wait_for(queue.dequeue(), timeout=2.0)
        await enqueue_task
        assert job == {"repo": "late", "pr_number": 9}

    asyncio.run(_run())


def test_depth_reports_waiting_jobs_and_minus_one_on_error():
    async def _run() -> None:
        queue = _queue()
        assert await queue.depth() == 0
        for pr in (1, 2, 3):
            await queue.enqueue({"repo": "d", "pr_number": pr})
        assert await queue.depth() == 3

        class _Broken:
            async def llen(self, *a):
                raise ConnectionError("down")

        broken = RedisJobQueue("redis://unused", client=_Broken())
        assert await broken.depth() == -1

    asyncio.run(_run())


class _AlwaysFailingClient:
    async def lmove(self, *args, **kwargs):
        raise ConnectionError("redis is down")


_LOGGER_NAME = "sentinel.infrastructure.redis.redis_job_queue"


def _drive_failing_dequeue(queue: RedisJobQueue, min_sleeps: int, monkeypatch) -> list[float]:
    """Run dequeue against a dead client until it has backed off min_sleeps times."""
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def _run() -> None:
        task = asyncio.create_task(queue.dequeue())
        while len(sleeps) < min_sleeps:
            await real_sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())
    return sleeps


def test_outage_backoff_grows_and_traceback_logged_once(caplog, monkeypatch):
    queue = RedisJobQueue(
        "redis://unused", client=_AlwaysFailingClient(), poll_interval=0.01
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        sleeps = _drive_failing_dequeue(queue, min_sleeps=12, monkeypatch=monkeypatch)

    # Backoff: 0.01 * 2**n, capped, never shrinking while failing.
    assert sleeps[0] == pytest.approx(0.02)
    assert max(sleeps) == RedisJobQueue.MAX_BACKOFF_SECONDS
    assert sleeps == sorted(sleeps)

    # One traceback for the whole streak; summaries suppressed inside the window.
    tracebacks = [r for r in caplog.records if r.exc_info]
    assert len(tracebacks) == 1
    assert sum("still unreachable" in r.message for r in caplog.records) == 0


def test_outage_summary_line_after_suppress_window(caplog, monkeypatch):
    monkeypatch.setattr(RedisJobQueue, "LOG_SUPPRESS_SECONDS", 0.0)
    queue = RedisJobQueue("redis://unused", client=_AlwaysFailingClient(), poll_interval=0.01)
    with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
        _drive_failing_dequeue(queue, min_sleeps=5, monkeypatch=monkeypatch)
    assert sum("still unreachable" in r.message for r in caplog.records) >= 1


class _FlakyClient:
    def __init__(self, failures: int, raw: str) -> None:
        self._failures = failures
        self._raw = raw

    async def lmove(self, *args, **kwargs):
        if self._failures > 0:
            self._failures -= 1
            raise ConnectionError("redis is down")
        return self._raw

    async def lrem(self, *args, **kwargs):
        return 1


def test_dequeue_recovers_and_logs_reconnect(caplog, monkeypatch):
    raw = json.dumps({"id": "r1", "job": {"repo": "back", "pr_number": 9}})
    queue = RedisJobQueue(
        "redis://unused", client=_FlakyClient(failures=3, raw=raw), poll_interval=0.01
    )
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        job = asyncio.run(queue.dequeue())

    assert job == {"repo": "back", "pr_number": 9}
    assert len(sleeps) == 3  # one backoff per failure, then success
    assert any("Redis reconnected after 3 failed attempt(s)" in r.message for r in caplog.records)


def test_poison_messages_are_discarded_and_valid_job_still_served():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        # Oldest first: raw garbage, then an envelope whose job isn't a dict.
        await client.lpush(RedisJobQueue.QUEUE_KEY, "not json{")
        await client.lpush(RedisJobQueue.QUEUE_KEY, json.dumps({"id": "x", "job": "flat"}))
        await queue.enqueue({"repo": "ok", "pr_number": 1})

        job = await asyncio.wait_for(queue.dequeue(), timeout=2.0)
        assert job == {"repo": "ok", "pr_number": 1}
        # Both poison messages were dropped from processing; only the live job remains.
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 1
        assert await client.llen(RedisJobQueue.QUEUE_KEY) == 0

    asyncio.run(_run())


# --- Bounded Retries / Dead-Letter (M8) ---


def test_attempts_starts_at_zero_and_survives_a_recovery():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "hello", "pr_number": 1})

        [raw] = await client.lrange(RedisJobQueue.QUEUE_KEY, 0, -1)
        assert json.loads(raw)["attempts"] == 0

        await queue.dequeue()  # unacked -> stranded in processing
        assert await queue.recover_pending() == 1

        [requeued] = await client.lrange(RedisJobQueue.QUEUE_KEY, 0, -1)
        assert json.loads(requeued)["attempts"] == 1

    asyncio.run(_run())


def test_job_is_dead_lettered_after_max_attempts():
    """A job that keeps killing the worker must stop being resurrected."""
    from sentinel.monitoring.metrics import metrics

    metrics.reset()

    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "poison", "pr_number": 66})

        # Each cycle simulates one crashed worker: dequeue, never ack, restart.
        for _ in range(RedisJobQueue.MAX_ATTEMPTS):
            await queue.dequeue()
            await queue.recover_pending()

        assert await client.llen(RedisJobQueue.QUEUE_KEY) == 0
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 0
        assert await client.llen(RedisJobQueue.DEAD_KEY) == 1

        [dead] = await client.lrange(RedisJobQueue.DEAD_KEY, 0, -1)
        envelope = json.loads(dead)
        assert envelope["job"] == {"repo": "poison", "pr_number": 66}
        assert envelope["attempts"] == RedisJobQueue.MAX_ATTEMPTS

        counters = metrics.snapshot()["counters"]
        assert counters["sentinel_jobs_deadlettered_total"] == 1

    asyncio.run(_run())


def test_recover_pending_returns_only_the_requeued_count():
    """Dead-lettered jobs are not 'recovered' — the count must not include them."""

    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)

        # Burn the poison job down to its last life, alone.
        await queue.enqueue({"repo": "poison", "pr_number": 2})
        for _ in range(RedisJobQueue.MAX_ATTEMPTS - 1):
            await queue.dequeue()
            assert await queue.recover_pending() == 1

        # Strand both: the poison job (one attempt left) and a fresh one.
        await queue.dequeue()
        await queue.enqueue({"repo": "healthy", "pr_number": 1})
        await queue.dequeue()

        requeued = await queue.recover_pending()

        assert requeued == 1  # the healthy job only
        assert await client.llen(RedisJobQueue.DEAD_KEY) == 1
        [survivor] = await client.lrange(RedisJobQueue.QUEUE_KEY, 0, -1)
        assert json.loads(survivor)["job"]["repo"] == "healthy"

    asyncio.run(_run())


def test_poison_message_in_processing_is_dropped_not_requeued():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await client.lpush(RedisJobQueue.PROCESSING_KEY, "not-json-at-all")

        assert await queue.recover_pending() == 0
        assert await client.llen(RedisJobQueue.QUEUE_KEY) == 0
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 0
        assert await client.llen(RedisJobQueue.DEAD_KEY) == 0

    asyncio.run(_run())


# --- Visibility timeout: the hung worker, not the crashed one ----------------
#
# recover_pending drains the processing list wholesale at startup, which is only
# safe because nothing is running yet. It cannot help a worker that is still
# alive but wedged — the pipeline runs in a thread via asyncio.to_thread and
# cannot be cancelled, so without a lease that job sits in processing until
# somebody restarts the process.


class _FakeClock:
    """Advanceable clock, matching the seam in test_delivery_dedup.py."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _lease_keys(client: fakeredis.FakeAsyncRedis):
    return client.keys(f"{RedisJobQueue.LEASE_KEY_PREFIX}*")


def test_a_dequeued_job_holds_a_lease_until_it_is_acked():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "r", "pr_number": 1})

        job = await queue.dequeue()
        assert len(await _lease_keys(client)) == 1
        # The lease carries the visibility timeout, not the dedup or quota TTL.
        (key,) = await _lease_keys(client)
        assert 0 < await client.ttl(key) <= RedisJobQueue.VISIBILITY_TIMEOUT_SECONDS

        await queue.ack(job)
        assert await _lease_keys(client) == []

    asyncio.run(_run())


def test_an_expired_lease_returns_the_job_to_the_queue():
    """Deleting the key is how a TTL lapse looks from the outside."""
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "r", "pr_number": 7})
        job = await queue.dequeue()

        for key in await _lease_keys(client):
            await client.delete(key)  # the worker hung; Redis let the lease lapse

        assert await queue.reclaim_expired() == 1
        assert await queue.depth() == 1
        recovered = await asyncio.wait_for(queue.dequeue(), timeout=2.0)
        assert recovered == job

    asyncio.run(_run())


def test_a_live_lease_is_never_reclaimed():
    """The regression that would make every slow review run twice."""
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "r", "pr_number": 2})
        await queue.dequeue()

        assert await queue.reclaim_expired() == 0
        assert await queue.depth() == 0
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 1

    asyncio.run(_run())


def test_the_scan_stops_at_the_first_live_lease():
    """Leases expire in tail order, so a held message shields everything newer.

    Pins the optimisation: if this ever scanned the whole list instead, the
    oldest-first guarantee would be silently doing nothing.
    """
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        for n in (1, 2):
            await queue.enqueue({"repo": "r", "pr_number": n})
        first = await queue.dequeue()   # oldest -> tail of processing
        await queue.dequeue()           # newest -> head

        # Expire only the NEWER one. The older still holds its lease, so the
        # scan must stop before reaching it.
        newer_key = f"{RedisJobQueue.LEASE_KEY_PREFIX}" + json.loads(
            await client.lindex(RedisJobQueue.PROCESSING_KEY, 0)
        )["id"]
        await client.delete(newer_key)

        assert await queue.reclaim_expired() == 0
        assert first["pr_number"] == 1

    asyncio.run(_run())


def test_a_reclaimed_job_survives_the_original_worker_acking_it_late():
    """The double-processing trap, and the reason the lease is a separate key.

    If a reclaim rewrote the envelope in place, the hung worker's ack — an LREM
    by the exact string it was handed — would match nothing and pass silently.
    Here the requeued copy must still be there afterwards.
    """
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "r", "pr_number": 3})
        job = await queue.dequeue()

        for key in await _lease_keys(client):
            await client.delete(key)
        assert await queue.reclaim_expired() == 1

        await queue.ack(job)  # the wedged worker finally wakes up

        assert await queue.depth() == 1, "the reclaimed copy must not be acked away"

    asyncio.run(_run())


def test_reclaiming_bumps_attempts_and_eventually_dead_letters():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await queue.enqueue({"repo": "r", "pr_number": 4})

        for _ in range(RedisJobQueue.MAX_ATTEMPTS):
            await queue.dequeue()
            for key in await _lease_keys(client):
                await client.delete(key)
            await queue.reclaim_expired()

        assert await queue.depth() == 0, "a job that always hangs must stop coming back"
        assert await client.llen(RedisJobQueue.DEAD_KEY) == 1
        parked = json.loads(await client.lindex(RedisJobQueue.DEAD_KEY, 0))
        assert parked["attempts"] == RedisJobQueue.MAX_ATTEMPTS
        assert parked["job"]["pr_number"] == 4

    asyncio.run(_run())


def test_the_sweep_is_rate_limited_by_the_clock():
    async def _run() -> None:
        clock = _FakeClock()
        queue = RedisJobQueue(
            "redis://unused:6379/0", client=_fake_client(), poll_interval=0.01, clock=clock
        )
        calls = {"n": 0}

        async def counting_reclaim() -> int:
            calls["n"] += 1
            return 0

        queue.reclaim_expired = counting_reclaim  # type: ignore[method-assign]

        await queue._reclaim_if_due()
        await queue._reclaim_if_due()
        assert calls["n"] == 1, "a second sweep inside the interval is wasted work"

        clock.now += RedisJobQueue.RECLAIM_INTERVAL_SECONDS
        await queue._reclaim_if_due()
        assert calls["n"] == 2

    asyncio.run(_run())


def test_a_poison_message_in_processing_is_discarded_by_the_sweep():
    async def _run() -> None:
        client = _fake_client()
        queue = _queue(client)
        await client.lpush(RedisJobQueue.PROCESSING_KEY, "{not json")

        assert await queue.reclaim_expired() == 0
        assert await client.llen(RedisJobQueue.PROCESSING_KEY) == 0

    asyncio.run(_run())


def test_a_redis_failure_during_the_sweep_is_contained():
    """Instrumentation must never break the dequeue loop it runs inside."""
    class _Broken:
        async def lindex(self, *a, **k):
            raise RuntimeError("redis down")

    async def _run() -> None:
        queue = _queue(_Broken())
        assert await queue.reclaim_expired() == 0

    asyncio.run(_run())


def test_recover_pending_clears_the_lease_it_finds():
    """A crashed worker's lease outlives it; leaving it would hide the message."""
    async def _run() -> None:
        server = fakeredis.FakeServer()
        client = _fake_client(server)
        first = _queue(client)
        await first.enqueue({"repo": "r", "pr_number": 5})
        await first.dequeue()
        assert len(await _lease_keys(client)) == 1

        second = _queue(_fake_client(server))  # fresh process after the crash
        assert await second.recover_pending() == 1
        assert await _lease_keys(client) == []

    asyncio.run(_run())


def test_in_memory_queue_offers_the_same_method():
    """Interface parity: callers must not need hasattr branching."""
    from sentinel.workers.job_queue import JobQueue

    assert asyncio.run(JobQueue().reclaim_expired()) == 0
