"""Webhook route behavior: queueing, validation, and enqueue failure.

The synchronous ``code``-in-body mode was removed in M9 — the endpoint no longer
analyzes caller-supplied source, it only identifies a pull request and queues it.
The tests that drove that mode went with it; what remains covers the single path.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api.webhook_controller import get_orchestrator
from sentinel.api.webhook_controller import router as webhook_router


class _DummyOrchestrator:
    def __init__(self) -> None:
        self.enqueued: list[dict] = []
        self.raise_on_enqueue = False

    async def enqueue_pull_request(self, payload: dict) -> None:
        if self.raise_on_enqueue:
            raise RuntimeError("queue is down")
        self.enqueued.append(payload)


def _build_client(orchestrator: _DummyOrchestrator) -> TestClient:
    app = FastAPI(title="Webhook Advanced Test")
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.include_router(webhook_router)
    return TestClient(app)


def test_webhook_queues_the_identified_pull_request():
    orchestrator = _DummyOrchestrator()
    client = _build_client(orchestrator)

    response = client.post("/webhook", json={"repo": "demo", "pr_number": 2, "author": "alice"})

    assert response.status_code == 200
    assert response.json() == {"status": "queued"}
    assert orchestrator.enqueued == [{"repo": "demo", "pr_number": 2, "author": "alice"}]


def test_webhook_ignores_a_code_field_instead_of_analyzing_it():
    """M9 regression guard: posting source must not get it analyzed.

    ``code`` used to switch the endpoint into a synchronous mode that ran the
    engines over whatever the caller sent. It is now dropped by the payload
    model, and the request queues a review of the identified PR like any other.
    """
    orchestrator = _DummyOrchestrator()
    client = _build_client(orchestrator)

    response = client.post(
        "/webhook",
        json={"repo": "demo", "pr_number": 3, "code": "password = 'hunter2'"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "queued"}
    assert orchestrator.enqueued == [{"repo": "demo", "pr_number": 3}]
    assert "code" not in orchestrator.enqueued[0]


def test_webhook_owner_is_split_from_a_full_name_repo():
    orchestrator = _DummyOrchestrator()
    client = _build_client(orchestrator)

    client.post("/webhook", json={"repo": "octo/hello", "pr_number": 7})

    assert orchestrator.enqueued == [{"repo": "octo/hello", "owner": "octo", "pr_number": 7}]


def test_webhook_invalid_body_returns_422():
    client = _build_client(_DummyOrchestrator())

    response = client.post("/webhook", json=["bad", "payload"])

    assert response.status_code == 422


def test_webhook_queue_exception_returns_500_http_error():
    orchestrator = _DummyOrchestrator()
    orchestrator.raise_on_enqueue = True
    client = _build_client(orchestrator)

    response = client.post("/webhook", json={"repo": "demo", "pr_number": 4})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to queue webhook payload"}


def test_pr_number_rejects_booleans():
    """bool is a subclass of int, so `{"number": true}` reached /pulls/True.

    _extract_installation_id and action/context.py both guard this already;
    this extractor was the one that did not.
    """
    from sentinel.api.webhook_controller import _extract_pr_number

    assert _extract_pr_number({"number": True}, None) is None
    assert _extract_pr_number({"number": False}, None) is None
    assert _extract_pr_number({"pull_request": {"number": True}}, None) is None
    assert _extract_pr_number({}, True) is None

    # Real numbers still work, including the falsy one.
    assert _extract_pr_number({"number": 7}, None) == 7
    assert _extract_pr_number({"pull_request": {"number": 9}}, None) == 9
    assert _extract_pr_number({}, 3) == 3
    assert _extract_pr_number({"number": 0}, None) == 0
