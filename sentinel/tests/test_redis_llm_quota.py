"""Per-installation daily LLM budget, proven on fakeredis.

The point of the cap is that exceeding it degrades the review rather than
refusing it: the findings are still real, only the AI-written prose is
withheld. These tests pin both halves — that the budget actually stops being
granted, and that nothing else about the review changes.
"""

import asyncio

import fakeredis

import sentinel.workers.background_worker as bw_module
from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.config.settings import get_settings
from sentinel.domain.entities.finding import Finding
from sentinel.domain.value_objects.severity_level import SeverityLevel
from sentinel.infrastructure.redis.redis_llm_quota import RedisLLMQuota
from sentinel.monitoring.metrics import metrics
from sentinel.workers.background_worker import BackgroundWorker


def _quota(daily_quota: int = 2, client=None) -> RedisLLMQuota:
    return RedisLLMQuota(
        "redis://unused:6379/0",
        daily_quota=daily_quota,
        client=client if client is not None else fakeredis.FakeAsyncRedis(decode_responses=True),
    )


def test_constructor_does_not_connect():
    RedisLLMQuota("redis://127.0.0.1:1/0", daily_quota=5)  # lazy from_url; must not raise


def test_budget_is_granted_until_it_runs_out():
    async def _run() -> None:
        quota = _quota(daily_quota=2)
        assert await quota.consume("111") is True
        assert await quota.consume("111") is True
        assert await quota.consume("111") is False

    asyncio.run(_run())


def test_each_installation_has_its_own_budget():
    """One noisy tenant must not spend everyone else's allowance."""

    async def _run() -> None:
        quota = _quota(daily_quota=1)
        assert await quota.consume("111") is True
        assert await quota.consume("222") is True
        assert await quota.consume("111") is False

    asyncio.run(_run())


def test_zero_quota_means_unlimited():
    """The default. A cap is opt-in, so an existing deployment is unaffected."""

    async def _run() -> None:
        quota = _quota(daily_quota=0)
        for _ in range(50):
            assert await quota.consume("111") is True

    asyncio.run(_run())


def test_jobs_without_an_installation_are_not_metered():
    """Manual triggers and the e2e harness are ours, not a tenant's."""

    async def _run() -> None:
        quota = _quota(daily_quota=1)
        assert await quota.consume(None) is True
        assert await quota.consume("") is True
        assert await quota.consume(None) is True

    asyncio.run(_run())


def test_the_key_is_scoped_to_the_day_and_the_installation():
    """Expiry does the resetting, so there is nothing to run at midnight."""

    async def _run() -> None:
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
        quota = _quota(daily_quota=5, client=client)
        await quota.consume("111")

        keys = await client.keys(f"{RedisLLMQuota.KEY_PREFIX}*")
        assert len(keys) == 1
        assert keys[0].endswith(":111")
        assert await client.ttl(keys[0]) > 0

    asyncio.run(_run())


def test_the_expiry_is_not_slid_forward_on_every_review():
    """Refreshing the TTL per call would mean the key never ages out."""

    async def _run() -> None:
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
        quota = _quota(daily_quota=10, client=client)

        await quota.consume("111")
        key = (await client.keys(f"{RedisLLMQuota.KEY_PREFIX}*"))[0]
        await client.expire(key, 60)
        await quota.consume("111")

        assert await client.ttl(key) <= 60

    asyncio.run(_run())


def test_a_redis_outage_fails_open():
    """A broken budget counter must not silently downgrade every review."""

    class _Broken:
        async def incr(self, key):
            raise ConnectionError("redis is down")

        async def expire(self, key, ttl):
            raise ConnectionError("redis is down")

    async def _run() -> None:
        assert await _quota(daily_quota=1, client=_Broken()).consume("111") is True

    asyncio.run(_run())


# --- wiring -------------------------------------------------------------------


def test_no_quota_object_without_redis(monkeypatch):
    """An in-memory counter would reset the budget on every deploy."""
    monkeypatch.setenv("LLM_DAILY_QUOTA", "10")
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert bw_module._build_llm_quota(get_settings()) is None


def test_no_quota_object_when_the_cap_is_unset(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.delenv("LLM_DAILY_QUOTA", raising=False)
    assert bw_module._build_llm_quota(get_settings()) is None


def test_quota_object_is_built_when_both_are_set(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv("LLM_DAILY_QUOTA", "10")
    assert isinstance(bw_module._build_llm_quota(get_settings()), RedisLLMQuota)


def test_an_exhausted_budget_marks_the_job_and_is_counted():
    async def _run() -> None:
        metrics.reset()
        quota = _quota(daily_quota=1)
        first: dict = {"installation_id": "111"}
        second: dict = {"installation_id": "111"}

        await BackgroundWorker._apply_llm_quota(first, quota)
        await BackgroundWorker._apply_llm_quota(second, quota)

        assert first["llm_allowed"] is True
        assert second["llm_allowed"] is False
        assert metrics.snapshot()["counters"]["sentinel_llm_quota_exhausted_total"] == 1

    asyncio.run(_run())


def test_nothing_is_written_to_the_job_when_nothing_is_metered():
    """Absence is what keeps every other caller, including the Action, allowed."""

    async def _run() -> None:
        job: dict = {"installation_id": "111"}
        await BackgroundWorker._apply_llm_quota(job, None)
        assert "llm_allowed" not in job

    asyncio.run(_run())


def test_a_denied_review_skips_enrichment_but_keeps_the_finding(monkeypatch):
    """The finding survives; only the AI prose is withheld."""
    monkeypatch.setenv("ENABLE_LLM", "true")

    class _LoudLLM:
        def reset_budget(self) -> None:
            raise AssertionError("the LLM must not be consulted past the budget")

        def generate_pr_audit(self, code, findings):
            raise AssertionError("the LLM must not be consulted past the budget")

    finding = Finding(
        rule="hardcoded_secret",
        match='api_key = "sk-..."',
        severity=SeverityLevel.HIGH,
        description="secret",
        recommendation="rotate it",
    )
    orchestrator = AuditOrchestrator(None, llm_service=_LoudLLM())

    unchanged = orchestrator.enrich_findings_with_llm("code", [finding], allow_llm=False)

    assert unchanged == [finding]
