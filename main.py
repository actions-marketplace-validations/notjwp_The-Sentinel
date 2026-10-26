"""The Sentinel — FastAPI composition root.

Wires the job queue (in-memory, or Redis when REDIS_URL is set), the audit
orchestrator, the background worker (started via lifespan), and the routers
(webhook, health, metrics).
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sentinel.api.health_controller import router as health_router
from sentinel.api.metrics_controller import router as metrics_router
from sentinel.api.webhook_controller import get_orchestrator
from sentinel.api.webhook_controller import router as webhook_router
from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.config.settings import get_settings
from sentinel.infrastructure.redis.redis_job_queue import RedisJobQueue
from sentinel.monitoring.logger import get_logger
from sentinel.workers.background_worker import BackgroundWorker
from sentinel.workers.job_queue import JobQueue

logger = get_logger(__name__)

# How long shutdown waits for the worker's in-flight job to finish and ack.
SHUTDOWN_GRACE_SECONDS = 10.0

# Composition root: REDIS_URL set -> durable Redis queue; unset -> in-memory.
_settings = get_settings()
job_queue = RedisJobQueue(_settings.REDIS_URL) if _settings.REDIS_URL else JobQueue()
audit_orchestrator = AuditOrchestrator(job_queue)
background_worker = BackgroundWorker(job_queue)


def verify_startup_config(settings) -> None:
    """Refuse to boot on a configuration that cannot do its job safely.

    Fail-closed only under ENVIRONMENT=production. Anywhere else keeps the
    historical warn-and-continue, so local dev and the test suite (which set
    neither a secret nor credentials) are unaffected. Pure and importable so it
    is testable without booting the app.
    """
    if not settings.ENABLE_GITHUB:
        return
    _verify_webhook_secret(settings)
    _verify_github_credentials(settings)


def _verify_webhook_secret(settings) -> None:
    """An unsigned public /webhook is the one thing that must never start quietly."""
    if settings.GITHUB_WEBHOOK_SECRET:
        return
    if settings.ENVIRONMENT == "production":
        raise RuntimeError(
            "ENVIRONMENT=production with ENABLE_GITHUB but no GITHUB_WEBHOOK_SECRET — "
            "refusing to start: the webhook would accept unsigned requests."
        )
    logger.warning(
        "ENABLE_GITHUB is on but GITHUB_WEBHOOK_SECRET is unset — "
        "webhook signature verification is DISABLED. Set it before public exposure."
    )


def _verify_github_credentials(settings) -> None:
    """Without these the service boots happily and silently posts nothing.

    Every GitHub call is failure-safe by design, so a missing app id or private
    key produces no crash and no failed health check — just reviews that are
    computed, logged, and never delivered. That is worse than not starting:
    the dashboard is green while the product does nothing.

    GITHUB_INSTALLATION_ID is deliberately NOT required. Since multi-tenancy it
    is only the fallback for jobs that carry no installation of their own.
    """
    missing = [
        name
        for name in ("GITHUB_APP_ID", "GITHUB_PRIVATE_KEY")
        if not getattr(settings, name, None)
    ]
    if not missing:
        return

    detail = " and ".join(missing)
    if settings.ENVIRONMENT == "production":
        raise RuntimeError(
            f"ENVIRONMENT=production with ENABLE_GITHUB but no {detail} — refusing to "
            "start: reviews would run and silently never be posted."
        )
    logger.warning(
        "ENABLE_GITHUB is on but %s is unset — reviews will run but cannot be posted.",
        detail,
    )


def warn_unreachable_config(settings) -> None:
    """Flag settings that are switched on but cannot take effect as configured."""
    # Enrichment spends the shared budget before translation is reached, so
    # anything below 2 guarantees translations are skipped no matter what.
    if settings.ENABLE_TRANSLATION and settings.LLM_MAX_CALLS < 2:
        logger.warning(
            "ENABLE_TRANSLATION is on but LLM_MAX_CALLS=%s — enrichment spends the "
            "budget first, so translations will never run. Raise it to at least 3 "
            "(1 enrichment + 1 per language).",
            settings.LLM_MAX_CALLS,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Validate BEFORE starting the worker: raising afterwards would orphan its task.
    verify_startup_config(settings)
    warn_unreachable_config(settings)

    logger.info("Starting background worker")
    app.state.worker_task = asyncio.create_task(background_worker.start())
    logger.info(
        "Queue/dedup backend: %s",
        "redis (durable)" if settings.REDIS_URL else "in-memory (non-durable)",
    )
    try:
        yield
    finally:
        worker_task = getattr(app.state, "worker_task", None)
        if worker_task:
            # The worker shields an in-flight job, so cancelling lets that job
            # finish and ack rather than abandoning it. Bound the wait so a
            # wedged job cannot hold shutdown open indefinitely.
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(worker_task, timeout=SHUTDOWN_GRACE_SECONDS)


app = FastAPI(title="The Sentinel", lifespan=lifespan)

app.dependency_overrides[get_orchestrator] = lambda: audit_orchestrator
app.state.job_queue = job_queue  # /metrics reads live queue depth from here

app.include_router(webhook_router)
app.include_router(health_router)
app.include_router(metrics_router)


@app.get("/")
def root():
    return {"message": "The Sentinel is running"}
