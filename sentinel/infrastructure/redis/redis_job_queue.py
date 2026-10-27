"""Durable FIFO job queue on Redis (LPUSH + LMOVE reliable-queue pattern).

Drop-in counterpart to the in-memory ``sentinel.workers.job_queue.JobQueue``:
same ``enqueue``/``dequeue``/``ack``/``recover_pending`` surface, but jobs live
in a Redis list, so they survive process restarts and are shared across
replicas. Selected by the composition root when ``REDIS_URL`` is set.

Delivery semantics are **at-least-once**: ``dequeue`` atomically moves the
message to a processing list, ``ack`` removes it after the job completes, and
``recover_pending`` (run at worker start) re-queues anything a crashed worker
left behind. A re-run after a crash is safe — the review post is an idempotent
upsert and assessment is deterministic.

Retries are **bounded**: each recovery bumps the envelope's attempt count, and a
job that burns ``MAX_ATTEMPTS`` is parked on ``DEAD_KEY`` rather than re-queued,
so a job that reliably kills the worker cannot be resurrected forever.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from typing import Any

import redis.asyncio as redis_asyncio

from sentinel.monitoring.logger import get_logger
from sentinel.monitoring.metrics import metrics

logger = get_logger(__name__)


class RedisJobQueue:
    QUEUE_KEY = "sentinel:jobs"
    PROCESSING_KEY = "sentinel:jobs:processing"
    # Jobs that exhausted their retries. Recovery is what makes a crash safe, but
    # unbounded recovery makes a poison job immortal: one that kills the process
    # would be re-queued at every start, forever. Parked here instead, for a human.
    DEAD_KEY = "sentinel:jobs:dead"
    MAX_ATTEMPTS = 3
    # A lease proves some worker is still alive on a message. It is a separate
    # key rather than a field on the envelope because ``ack`` removes the
    # message with LREM by its exact serialized value — rewriting the envelope
    # in place would make a merely-slow worker's ack match nothing, silently,
    # and the job would be processed twice.
    LEASE_KEY_PREFIX = "sentinel:jobs:lease:"
    # How long a worker may hold a message before it is presumed wedged. A real
    # review measures ~45s end to end, of which ~18s is the LLM call, plus up to
    # 8s of blocking GitHub retry backoff. An order of magnitude of headroom:
    # reclaiming early causes the duplicate work this exists to prevent.
    VISIBILITY_TIMEOUT_SECONDS = 600.0
    # How often the sweep runs. It costs one LINDEX per idle pass, so it is
    # cheap, but there is no reason to pay it on every poll.
    RECLAIM_INTERVAL_SECONDS = 30.0
    # Outage hygiene: one traceback per failing streak, then at most one summary
    # line per suppress window; retry sleeps back off exponentially to the cap.
    LOG_SUPPRESS_SECONDS = 60.0
    MAX_BACKOFF_SECONDS = 5.0

    def __init__(
        self,
        url: str,
        *,
        client: Any | None = None,
        poll_interval: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # ``client`` is a test seam (fakeredis.FakeAsyncRedis). from_url does not
        # connect — redis-py dials lazily on the first command.
        self._client = client if client is not None else redis_asyncio.from_url(
            url, decode_responses=True
        )
        self._poll_interval = poll_interval
        # Injected so a test can age a lease without sleeping; mirrors the
        # seam DeliveryDeduper already uses.
        self._clock = clock
        self._last_reclaim = 0.0
        self._consecutive_failures = 0
        self._last_failure_log = 0.0
        # (job, raw message) for jobs handed out but not yet acked, keyed by the
        # returned dict's id(). Holding the job itself keeps the dict alive, so
        # its id() can never be reused by a later job while the entry exists;
        # ack pops the entry, bounding this to the jobs in flight.
        self._inflight: dict[int, tuple[dict, str]] = {}

    async def enqueue(self, job: dict) -> None:
        if not isinstance(job, dict):
            raise TypeError("job must be a dictionary")
        # Envelope gives every message a distinct identity, so LREM on ack can
        # never remove a different-but-identical job payload. ``attempts`` lives
        # on the envelope, never inside ``job`` — the dict handed to the worker
        # must stay exactly what the producer enqueued.
        raw = json.dumps({"id": uuid.uuid4().hex, "attempts": 0, "job": job})
        # Redis errors propagate: the webhook route turns enqueue failures into
        # HTTP 500, and GitHub redelivers later.
        await self._client.lpush(self.QUEUE_KEY, raw)
        logger.info("Job enqueued (redis)")

    def _backoff_seconds(self) -> float:
        exponent = min(self._consecutive_failures, 10)  # cap 2**n growth
        return min(self._poll_interval * (2**exponent), self.MAX_BACKOFF_SECONDS)

    def _log_dequeue_failure(self) -> None:
        """Traceback once per failing streak, then one line per suppress window."""
        now = self._clock()
        if self._consecutive_failures == 1:
            logger.exception("Redis dequeue failed; backing off and retrying")
            self._last_failure_log = now
        elif now - self._last_failure_log >= self.LOG_SUPPRESS_SECONDS:
            logger.error(
                "Redis still unreachable (attempt %s); continuing to retry",
                self._consecutive_failures,
            )
            self._last_failure_log = now

    async def dequeue(self) -> dict:
        # Poll with non-blocking LMOVE rather than BLMOVE: cancellation-friendly,
        # resilient to Redis restarts, and exercisable on fakeredis.
        while True:
            try:
                raw = await self._client.lmove(
                    self.QUEUE_KEY, self.PROCESSING_KEY, "RIGHT", "LEFT"
                )
            except Exception:
                self._consecutive_failures += 1
                metrics.counter_inc("sentinel_redis_errors_total", {"op": "dequeue"})
                self._log_dequeue_failure()
                await asyncio.sleep(self._backoff_seconds())
                continue

            if self._consecutive_failures:
                logger.info(
                    "Redis reconnected after %s failed attempt(s)",
                    self._consecutive_failures,
                )
                self._consecutive_failures = 0
                self._last_failure_log = 0.0

            if raw is not None:
                envelope = self._parse_envelope(raw)
                if envelope is None:
                    # Poison pill: drop it from processing so it can't loop forever.
                    await self._discard_processing(raw)
                    continue
                job = envelope["job"]
                envelope_id = envelope.get("id")
                await self._take_lease(envelope_id)
                self._inflight[id(job)] = (job, raw, envelope_id)
                return job

            # Idle: the only moment worth spending on housekeeping, and the
            # queue being empty is exactly when a stranded job matters most.
            await self._reclaim_if_due()
            await asyncio.sleep(self._poll_interval)

    async def ack(self, job: dict) -> None:
        """Mark a dequeued job as done, removing it from the processing list.

        Failure-safe: on Redis error the message stays in processing and is
        re-queued by ``recover_pending`` on the next start — never lost.
        """
        entry = self._inflight.pop(id(job), None)
        if entry is None:
            return
        _, raw, envelope_id = entry
        try:
            # Lease first. If the LREM below fails, the message stays in
            # processing with no lease, so the sweep reclaims it promptly rather
            # than waiting for a restart — the opposite order would leave a
            # finished job holding a lease for the full visibility timeout.
            await self._release_lease(envelope_id)
            await self._client.lrem(self.PROCESSING_KEY, 1, raw)
        except Exception:
            metrics.counter_inc("sentinel_redis_errors_total", {"op": "ack"})
            logger.exception("Redis ack failed; job will be re-queued on next restart")

    async def recover_pending(self) -> int:
        """Re-queue jobs a crashed worker left in the processing list.

        Each recovery bumps the envelope's ``attempts``; a job that has burned
        MAX_ATTEMPTS is moved to DEAD_KEY instead of being re-queued, so a job
        that reliably kills the worker stops taking the process down with it.
        Returns the number **re-queued** (interface parity with the in-memory
        queue's no-op); dead-lettered jobs are logged and counted separately.
        """
        recovered = 0
        dead_lettered = 0
        while True:
            try:
                # LMOVE first, exactly as before: the message is never in flight
                # between two keys. Rewriting it afterwards means a crash mid-
                # recovery leaves the job queued with a stale attempt count —
                # one extra retry at worst, never a lost job.
                raw = await self._client.lmove(
                    self.PROCESSING_KEY, self.QUEUE_KEY, "RIGHT", "LEFT"
                )
                if raw is None:
                    break
                envelope = self._parse_envelope(raw)
                if envelope is None:
                    await self._client.lrem(self.QUEUE_KEY, 1, raw)  # poison, drop it
                    continue
                # A crashed worker's lease outlives it; drop it so the message
                # is not also counted as in-flight by the sweep.
                await self._release_lease(envelope.get("id"))
                if await self._requeue_or_dead_letter(raw, envelope):
                    recovered += 1
                else:
                    dead_lettered += 1
            except Exception:
                metrics.counter_inc("sentinel_redis_errors_total", {"op": "recover"})
                logger.exception("Redis recovery failed; continuing with %s recovered", recovered)
                break
        if recovered:
            logger.info("Recovered %s orphaned job(s) from the processing list", recovered)
        if dead_lettered:
            logger.error(
                "Dead-lettered %s job(s) after %s attempts; inspect %s",
                dead_lettered,
                self.MAX_ATTEMPTS,
                self.DEAD_KEY,
            )
        return recovered

    def _lease_key(self, envelope_id: object) -> str | None:
        return f"{self.LEASE_KEY_PREFIX}{envelope_id}" if isinstance(envelope_id, str) else None

    async def _take_lease(self, envelope_id: object) -> None:
        """Mark a message as held. Failure-safe: no lease just means reclaimable."""
        key = self._lease_key(envelope_id)
        if key is None:
            return
        try:
            await self._client.set(key, "1", ex=int(self.VISIBILITY_TIMEOUT_SECONDS))
        except Exception:
            metrics.counter_inc("sentinel_redis_errors_total", {"op": "lease"})
            logger.exception("Could not take a lease; the job may be reclaimed early")

    async def _release_lease(self, envelope_id: object) -> None:
        key = self._lease_key(envelope_id)
        if key is not None:
            await self._client.delete(key)

    async def _requeue_or_dead_letter(self, raw: str, envelope: dict) -> bool:
        """Bump attempts and put the message back, or park it. True if re-queued.

        The message is already on QUEUE_KEY when this runs — both callers LMOVE
        first so it is never in flight between two keys — so the work here is to
        replace that copy with one carrying the incremented count.
        """
        attempts = envelope.get("attempts")
        envelope["attempts"] = (attempts if isinstance(attempts, int) else 0) + 1
        if envelope["attempts"] >= self.MAX_ATTEMPTS:
            await self._client.lrem(self.QUEUE_KEY, 1, raw)
            await self._client.lpush(self.DEAD_KEY, json.dumps(envelope))
            metrics.counter_inc("sentinel_jobs_deadlettered_total")
            return False
        await self._client.lrem(self.QUEUE_KEY, 1, raw)
        await self._client.lpush(self.QUEUE_KEY, json.dumps(envelope))
        return True

    async def _reclaim_if_due(self) -> int:
        """Run the sweep at most once per RECLAIM_INTERVAL_SECONDS."""
        now = self._clock()
        if now - self._last_reclaim < self.RECLAIM_INTERVAL_SECONDS:
            return 0
        self._last_reclaim = now
        return await self.reclaim_expired()

    async def reclaim_expired(self) -> int:
        """Re-queue in-flight jobs whose lease has expired. Returns the count.

        ``recover_pending`` covers a worker that **crashed** — it drains the
        processing list wholesale at startup, which is only safe because nothing
        is running yet. This covers one that **hung**: the pipeline runs in a
        thread via ``asyncio.to_thread`` and cannot be cancelled, so a wedged job
        would otherwise sit in processing until someone restarted the process.

        Scans from the tail, which is the oldest message: ``dequeue`` LMOVEs
        RIGHT to LEFT, so the list is newest-first and leases expire in tail
        order. The first live lease therefore means every newer one is live too,
        and the scan stops — one LINDEX in the common case.

        At-least-once, like everything else here: a worker that wakes after its
        lease lapsed may still post. That is safe because ``upsert_comment``
        edits Sentinel's existing comment in place rather than adding one.
        """
        reclaimed = 0
        dead_lettered = 0
        while True:
            try:
                raw = await self._client.lindex(self.PROCESSING_KEY, -1)
                if raw is None:
                    break
                envelope = self._parse_envelope(raw)
                if envelope is None:
                    await self._client.lrem(self.PROCESSING_KEY, 1, raw)
                    continue
                key = self._lease_key(envelope.get("id"))
                if key is not None and await self._client.exists(key):
                    break  # still held, and so is everything newer

                # LMOVE rather than LREM+LPUSH so the message is never in flight
                # between two keys. A concurrent ack landing between the LINDEX
                # and this line can move a different message, which re-queues it
                # one cycle early — at-least-once, and the post is idempotent.
                moved = await self._client.lmove(
                    self.PROCESSING_KEY, self.QUEUE_KEY, "RIGHT", "LEFT"
                )
                if moved is None:
                    break
                moved_envelope = self._parse_envelope(moved)
                if moved_envelope is None:
                    await self._client.lrem(self.QUEUE_KEY, 1, moved)
                    continue
                await self._release_lease(moved_envelope.get("id"))
                if await self._requeue_or_dead_letter(moved, moved_envelope):
                    reclaimed += 1
                    metrics.counter_inc("sentinel_jobs_reclaimed_total")
                else:
                    dead_lettered += 1
            except Exception:
                metrics.counter_inc("sentinel_redis_errors_total", {"op": "reclaim"})
                logger.exception("Lease sweep failed; continuing with %s reclaimed", reclaimed)
                break

        if reclaimed:
            logger.warning(
                "Reclaimed %s job(s) whose lease expired after %ss — a worker is hung or died"
                " mid-job",
                reclaimed,
                self.VISIBILITY_TIMEOUT_SECONDS,
            )
        if dead_lettered:
            logger.error(
                "Dead-lettered %s reclaimed job(s) after %s attempts; inspect %s",
                dead_lettered,
                self.MAX_ATTEMPTS,
                self.DEAD_KEY,
            )
        return reclaimed

    async def depth(self) -> int:
        """Jobs currently waiting (the /metrics queue-depth gauge). -1 on error."""
        try:
            return int(await self._client.llen(self.QUEUE_KEY))
        except Exception:
            return -1

    @staticmethod
    def _parse_envelope(raw: str) -> dict | None:
        """Parse a queue message into its envelope dict; None if malformed."""
        try:
            envelope = json.loads(raw)
        except Exception:
            logger.warning("Discarding unparseable queue message")
            return None
        if not isinstance(envelope, dict) or not isinstance(envelope.get("job"), dict):
            logger.warning("Discarding queue message without a job dict")
            return None
        return envelope

    @staticmethod
    def _parse_job(raw: str) -> dict | None:
        """Extract the job dict from an envelope string; None if malformed."""
        envelope = RedisJobQueue._parse_envelope(raw)
        return envelope["job"] if envelope is not None else None

    async def _discard_processing(self, raw: str) -> None:
        try:
            await self._client.lrem(self.PROCESSING_KEY, 1, raw)
        except Exception:
            logger.exception("Failed to discard poison message from processing list")
