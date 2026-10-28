"""Refuse an oversized request body before anything buffers it.

This is ASGI middleware rather than a route dependency for a measured reason:
FastAPI reads and caches the entire body *before* it solves a route's
dependencies, so a check inside ``verify_webhook_signature`` runs after the
allocation it was meant to prevent.

It matters because the HMAC cannot be computed without the body, so the read
happens while the caller is still unauthenticated. Anyone who can reach the
port can make the process allocate: four concurrent 200 MB unsigned posts took
a running Sentinel from 167 MB to 1.1 GB of RSS in under three seconds. On the
free-tier instance ``docs/DEPLOYMENT.md`` targets, that is an OOM kill from a
stranger with curl.

GitHub caps its own webhook deliveries at 25 MB and refuses to send anything
larger, so nothing legitimate is lost.
"""

import json

from sentinel.monitoring.logger import get_logger

logger = get_logger(__name__)

MAX_BODY_BYTES = 25 * 1024 * 1024

_RESPONSE_BODY = json.dumps({"detail": "Request body too large"}).encode("utf-8")


def _content_length(scope) -> int | None:
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


async def _send_too_large(send) -> None:
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(_RESPONSE_BODY)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": _RESPONSE_BODY})


class BodySizeLimitMiddleware:
    """Reject bodies over ``max_bytes``, by declared length or by counting."""

    def __init__(self, app, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # The cheap path: a declared Content-Length over the cap is refused
        # without reading a single byte of the body.
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            logger.warning("Rejected a %s-byte request body (cap %s)",
                           declared, self.max_bytes)
            await _send_too_large(send)
            return

        state = {"received": 0, "over": False, "responded": False}

        async def counting_receive():
            # A chunked upload declares no length, so it is counted instead and
            # cut off the moment it crosses the cap.
            message = await receive()
            if message.get("type") == "http.request":
                state["received"] += len(message.get("body") or b"")
                if state["received"] > self.max_bytes:
                    state["over"] = True
                    logger.warning("Rejected a streamed body over the %s-byte cap",
                                   self.max_bytes)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            # Once the cap is blown the application's own response (whatever it
            # made of a truncated body) is discarded in favour of a 413.
            if state["over"]:
                if not state["responded"]:
                    state["responded"] = True
                    await _send_too_large(send)
                return
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

        if state["over"] and not state["responded"]:
            state["responded"] = True
            await _send_too_large(send)
