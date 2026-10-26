from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from sentinel.api.delivery_dedup import DeliveryDeduper
from sentinel.api.webhook_events import should_process
from sentinel.api.webhook_security import verify_webhook_signature
from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.config.settings import get_settings
from sentinel.infrastructure.redis.redis_delivery_dedup import RedisDeliveryDeduper
from sentinel.monitoring.logger import get_logger
from sentinel.monitoring.metrics import metrics

router = APIRouter()
logger = get_logger(__name__)


def _build_deduper() -> DeliveryDeduper | RedisDeliveryDeduper:
    """Pick the dedup backend: Redis when REDIS_URL is set, else in-memory.

    Module-scope seam (mirrors the worker's _build_github_client) so tests can
    monkeypatch webhook_controller._deduper with a fresh instance.
    """
    settings = get_settings()
    if settings.REDIS_URL:
        return RedisDeliveryDeduper(settings.REDIS_URL)
    return DeliveryDeduper()


# Remembers recent X-GitHub-Delivery ids so re-sent deliveries (redeliveries,
# double-sends) don't re-run the whole pipeline. Module state, like the router;
# tests swap in a fresh instance via monkeypatch.
_deduper = _build_deduper()


class WebhookPayload(BaseModel):
    """The fields a caller may supply directly.

    A real GitHub delivery sets none of these — everything is read out of the
    nested event body by the extractors below. They exist for manual triggering
    ("review owner/name#12") and are treated as overrides when present.

    There is deliberately no ``code`` field: the endpoint never analyzes
    caller-supplied source. See the module note on the route.
    """

    model_config = ConfigDict(extra="ignore")

    repo: str | None = Field(default=None, min_length=1, max_length=1024)
    pr_number: int | None = None
    author: str | None = Field(default=None, max_length=256)
    files: list[str] | None = None


def get_orchestrator() -> AuditOrchestrator:
    raise RuntimeError("AuditOrchestrator dependency is not configured")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


async def _raw_json(request: Request) -> dict[str, Any]:
    """Best-effort parse of the request body as a JSON object; {} on anything else.

    Starlette caches the body, so calling this alongside the signature
    dependency reads it only once.
    """
    try:
        parsed = await request.json()
    except Exception:
        return {}
    return _as_dict(parsed)


def _extract_repo_name(raw_payload: dict[str, Any], fallback_repo: str | None) -> str | None:
    if fallback_repo:
        return fallback_repo

    repository = _as_dict(raw_payload.get("repository"))
    full_name = repository.get("full_name")
    if isinstance(full_name, str) and "/" in full_name:
        return full_name.split("/", 1)[1]

    name = repository.get("name")
    if isinstance(name, str) and name.strip() != "":
        return name

    return None


def _extract_owner(raw_payload: dict[str, Any], fallback_repo: str | None) -> str | None:
    repository = _as_dict(raw_payload.get("repository"))
    owner_info = _as_dict(repository.get("owner"))
    owner = owner_info.get("login")
    if isinstance(owner, str) and owner.strip() != "":
        return owner

    full_name = repository.get("full_name")
    if isinstance(full_name, str) and "/" in full_name:
        return full_name.split("/", 1)[0]

    if isinstance(fallback_repo, str) and "/" in fallback_repo:
        return fallback_repo.split("/", 1)[0]

    return None


def _extract_pr_number(raw_payload: dict[str, Any], fallback_pr_number: int | None) -> int | None:
    if isinstance(fallback_pr_number, int):
        return fallback_pr_number

    pull_request = _as_dict(raw_payload.get("pull_request"))
    number = pull_request.get("number")
    if isinstance(number, int):
        return number

    top_level_number = raw_payload.get("number")
    if isinstance(top_level_number, int):
        return top_level_number

    return None


def _extract_author(raw_payload: dict[str, Any], fallback_author: str | None) -> str | None:
    if isinstance(fallback_author, str) and fallback_author.strip() != "":
        return fallback_author

    pull_request = _as_dict(raw_payload.get("pull_request"))
    user = _as_dict(pull_request.get("user"))
    login = user.get("login")
    if isinstance(login, str) and login.strip() != "":
        return login

    sender = _as_dict(raw_payload.get("sender"))
    sender_login = sender.get("login")
    if isinstance(sender_login, str) and sender_login.strip() != "":
        return sender_login

    return None


def _extract_files(raw_payload: dict[str, Any], fallback_files: list[str] | None) -> list[str]:
    if isinstance(fallback_files, list) and fallback_files:
        return [file_name for file_name in fallback_files if isinstance(file_name, str)]

    raw_files = raw_payload.get("files")
    if not isinstance(raw_files, list):
        return []

    files: list[str] = []
    for item in raw_files:
        if isinstance(item, str):
            files.append(item)
            continue

        if isinstance(item, dict):
            file_name = item.get("filename") or item.get("path")
            if isinstance(file_name, str):
                files.append(file_name)

    return files


def _extract_installation_id(raw_payload: dict[str, Any]) -> str | None:
    """Which App installation this delivery belongs to.

    Every GitHub App delivery carries it, and it decides WHOSE credentials the
    worker reviews with. Without it the worker falls back to the single id in
    the environment, which is correct only for the one installation that
    happens to be configured — anyone else's pull request would be fetched with
    the wrong token and fail every call, silently.

    GitHub sends it as an int and the client interpolates it into a URL, so it
    is normalized to ``str`` here. ``bool`` is rejected explicitly because it is
    a subclass of ``int``.
    """
    installation = _as_dict(raw_payload.get("installation"))
    installation_id = installation.get("id")

    if isinstance(installation_id, bool):
        return None
    if isinstance(installation_id, int):
        return str(installation_id)
    if isinstance(installation_id, str) and installation_id.strip() != "":
        return installation_id.strip()
    return None


def _build_job(raw_payload: dict[str, Any], payload: WebhookPayload) -> dict[str, Any]:
    """Resolve a delivery into the job the worker needs: who, which repo, which PR.

    Directly-supplied fields win; everything else is read out of the nested
    GitHub event body. ``owner`` is threaded separately because ``repo`` may
    arrive either as ``owner/name`` or bare, and the worker's ``_identity``
    needs the owner to build a well-formed API URL.
    """
    job: dict[str, Any] = {}
    repo_name = _extract_repo_name(raw_payload, payload.repo)
    owner = _extract_owner(raw_payload, payload.repo)
    pr_number = _extract_pr_number(raw_payload, payload.pr_number)
    author = _extract_author(raw_payload, payload.author)
    files = _extract_files(raw_payload, payload.files)
    installation_id = _extract_installation_id(raw_payload)

    if repo_name is not None:
        job["repo"] = repo_name
    if owner is not None:
        job["owner"] = owner
    if pr_number is not None:
        job["pr_number"] = pr_number
    if author is not None:
        job["author"] = author
    if files:
        job["files"] = files
    if installation_id is not None:
        job["installation_id"] = installation_id
    return job


@router.post("/webhook", dependencies=[Depends(verify_webhook_signature)])
async def webhook(
    request: Request,
    payload: WebhookPayload = Body(
        ...,
        examples={
            "pull_request_opened": {
                "summary": "Manual trigger for a pull request",
                "value": {"repo": "octo/hello", "pr_number": 12},
            }
        },
    ),
    orchestrator: AuditOrchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Accept a delivery and queue a review. Never analyzes caller-supplied code.

    The endpoint identifies a pull request and hands it to the worker, which
    fetches the real diff from the GitHub API itself. An earlier synchronous
    mode analyzed a ``code`` field posted in the request body; it was removed
    because it made a public endpoint an arbitrary-code-analysis service, and
    real GitHub deliveries never carried that field.

    Order matters: signature (route dependency) -> event filter -> dedup ->
    enqueue. Filtering ahead of dedup means an ignored event never consumes a
    dedup slot, and every outcome answers 200 because GitHub records any
    non-2xx as a failed delivery.
    """
    raw_payload = await _raw_json(request)

    event = request.headers.get("X-GitHub-Event")
    allowed, reason = should_process(event, raw_payload)
    if not allowed:
        logger.info("Ignoring webhook delivery (%s)", reason)
        metrics.counter_inc("sentinel_webhooks_total", {"mode": "ignored"})
        return {"status": "ignored", "event": event}

    delivery_id = request.headers.get("X-GitHub-Delivery")
    if await _deduper.is_duplicate(delivery_id):
        logger.info("Duplicate delivery %s skipped", delivery_id)
        metrics.counter_inc("sentinel_webhooks_total", {"mode": "duplicate"})
        return {"status": "duplicate"}

    job = _build_job(raw_payload, payload)
    logger.info(
        "Queueing review for repo=%s pr_number=%s", job.get("repo"), job.get("pr_number")
    )
    try:
        await orchestrator.enqueue_pull_request(job)
    except Exception:
        logger.exception("Failed to enqueue webhook payload")
        raise HTTPException(status_code=500, detail="Failed to queue webhook payload")

    metrics.counter_inc("sentinel_webhooks_total", {"mode": "queued"})
    return {"status": "queued"}
