import hashlib
import hmac

from fastapi import HTTPException, Request

from sentinel.config.settings import get_settings
from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)

_SIGNATURE_HEADER = "X-Hub-Signature-256"
_PREFIX = "sha256="


def compute_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{_PREFIX}{digest}"


def is_valid_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    if not secret:
        return False
    if not signature_header or not signature_header.startswith(_PREFIX):
        return False
    expected = compute_signature(secret, body)
    return hmac.compare_digest(expected.encode("utf-8"), signature_header.encode("utf-8"))


async def verify_webhook_signature(request: Request) -> None:
    """FastAPI dependency. Enforces HMAC only when a secret is configured.

    No secret configured -> verification is skipped (dev/test/local). Production
    must set GITHUB_WEBHOOK_SECRET; see the startup warning in main.py.

    Note this cannot be where the body size is bounded: FastAPI reads and
    caches the whole body before it solves a route's dependencies, so by the
    time this runs the allocation has already happened. The cap lives in
    ``BodySizeLimitMiddleware`` instead — see ``sentinel/api/body_limit.py``.
    """
    secret = get_settings().GITHUB_WEBHOOK_SECRET
    if not secret:
        return
    body = await request.body()
    signature = request.headers.get(_SIGNATURE_HEADER)
    if not is_valid_signature(secret, body, signature):
        logger.warning("Rejected webhook: invalid or missing %s", _SIGNATURE_HEADER)
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
