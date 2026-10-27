"""The body cap, at the only layer where it can work.

A route dependency cannot do this job: FastAPI reads and caches the whole body
before it solves dependencies, so a check there runs after the allocation it
was supposed to prevent. test_the_cap_must_run_before_fastapi_reads_the_body
pins that fact, so nobody "simplifies" the middleware back into a Depends.
"""

from fastapi import Body, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from sentinel.api.body_limit import MAX_BODY_BYTES, BodySizeLimitMiddleware


class _Payload(BaseModel):
    pad: str | None = None
    value: int | None = None


def _app(max_bytes: int = 1024) -> FastAPI:
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(payload: _Payload = Body(...)):
        return {"value": payload.value}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_declared_length_over_the_cap_is_refused():
    body = b'{"pad":"' + b"A" * 4096 + b'"}'
    response = TestClient(_app()).post(
        "/echo", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"


def test_chunked_body_over_the_cap_is_refused():
    """No Content-Length to check, so the bytes are counted instead."""

    def chunks():
        for _ in range(8):
            yield b"A" * 512

    response = TestClient(_app()).post(
        "/echo", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_a_normal_body_passes_through_untouched():
    response = TestClient(_app()).post("/echo", json={"value": 7})
    assert response.status_code == 200
    assert response.json() == {"value": 7}


def test_a_body_exactly_at_the_cap_is_allowed():
    app = _app(max_bytes=64)
    body = b'{"value":1}'
    assert len(body) < 64
    response = TestClient(app).post(
        "/echo", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200


def test_requests_without_a_body_are_unaffected():
    assert TestClient(_app()).get("/health").status_code == 200


def test_unparseable_content_length_falls_through_to_counting():
    response = TestClient(_app(max_bytes=64)).post(
        "/echo",
        content=b'{"pad":"' + b"A" * 512 + b'"}',
        headers={"Content-Type": "application/json", "Content-Length": "not-a-number"},
    )
    assert response.status_code in (413, 400)


def test_default_cap_matches_githubs_own_delivery_limit():
    assert MAX_BODY_BYTES == 25 * 1024 * 1024


def test_the_cap_must_run_before_fastapi_reads_the_body():
    """Why this is middleware and not a dependency. Pinned, not assumed."""
    seen = {}

    async def dependency(request: Request):
        seen["body_already_cached"] = hasattr(request, "_body")

    app = FastAPI()

    @app.post("/x", dependencies=[Depends(dependency)])
    async def route(payload: _Payload = Body(...)):
        return {}

    TestClient(app).post("/x", json={"value": 1})
    assert seen["body_already_cached"] is True
