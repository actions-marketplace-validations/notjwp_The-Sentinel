"""Multi-tenancy: one deployment serving many GitHub App installations.

The installation id decides WHOSE credentials a review runs with. Before this,
it came from the environment, so a second installation's pull request was
fetched with the first installation's token — every call 404s, including the
"review skipped" check run, and the user sees nothing at all. Silence is the
worst possible failure mode, so these tests pin the whole path: the id is read
off the delivery, survives the orchestrator's job whitelist, and selects the
client the worker actually uses.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sentinel.api.webhook_controller as webhook_controller
import sentinel.workers.background_worker as bw_module
from sentinel.api.delivery_dedup import DeliveryDeduper
from sentinel.api.webhook_controller import get_orchestrator
from sentinel.api.webhook_controller import router as webhook_router
from sentinel.application.audit_orchestrator import AuditOrchestrator
from sentinel.config.settings import get_settings
from sentinel.workers.background_worker import BackgroundWorker
from sentinel.workers.job_queue import JobQueue


@pytest.fixture
def github_enabled(monkeypatch):
    """Settings with GitHub on and no configured installation."""
    monkeypatch.setenv("ENABLE_GITHUB", "true")
    return get_settings()


def _worker() -> BackgroundWorker:
    return BackgroundWorker(JobQueue())


# --- the id is read off the delivery -----------------------------------------


def test_installation_id_is_extracted_from_a_github_delivery():
    raw = {"installation": {"id": 12345678}}
    assert webhook_controller._extract_installation_id(raw) == "12345678"


def test_installation_id_is_normalized_and_bad_shapes_are_rejected():
    """It is interpolated into a URL, so only a usable value may come back."""
    assert webhook_controller._extract_installation_id({"installation": {"id": "42"}}) == "42"
    assert webhook_controller._extract_installation_id({"installation": {"id": "  7 "}}) == "7"
    assert webhook_controller._extract_installation_id({}) is None
    assert webhook_controller._extract_installation_id({"installation": {}}) is None
    assert webhook_controller._extract_installation_id({"installation": {"id": ""}}) is None
    assert webhook_controller._extract_installation_id({"installation": None}) is None
    # bool is a subclass of int; True must not become the installation "True".
    assert webhook_controller._extract_installation_id({"installation": {"id": True}}) is None


# --- it survives the orchestrator's whitelist --------------------------------


def test_installation_id_survives_the_enqueue_whitelist():
    """enqueue_pull_request rebuilds the job from named keys only.

    A key added at the webhook and not here would vanish between the two with
    nothing logged — the one hop in this path that fails silently.
    """
    captured: list[dict] = []

    class _Queue:
        async def enqueue(self, job):
            captured.append(job)

    orchestrator = AuditOrchestrator(_Queue())
    payload = {"repo": "hello", "owner": "octo", "pr_number": 7, "installation_id": "555"}
    asyncio.run(orchestrator.enqueue_pull_request(payload))

    assert captured[0]["installation_id"] == "555"


def test_webhook_threads_the_installation_id_end_to_end(monkeypatch):
    """A GitHub-shaped body reaches the queue with the installation intact."""
    monkeypatch.setattr(webhook_controller, "_deduper", DeliveryDeduper())

    class _DummyOrchestrator:
        def __init__(self) -> None:
            self.received: list[dict] = []

        async def enqueue_pull_request(self, payload) -> None:
            self.received.append(payload)

    app = FastAPI(title="The Sentinel Multi-Tenancy Test App")
    orchestrator = _DummyOrchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.include_router(webhook_router)

    response = TestClient(app).post(
        "/webhook",
        json={
            "action": "opened",
            "installation": {"id": 987654},
            "repository": {"full_name": "octo/hello", "owner": {"login": "octo"}},
            "pull_request": {"number": 3, "user": {"login": "someone"}},
        },
        headers={"X-GitHub-Event": "pull_request"},
    )

    assert response.status_code == 200
    assert orchestrator.received[0]["installation_id"] == "987654"


# --- it selects the client the worker uses -----------------------------------


def test_client_is_built_for_the_job_s_installation(github_enabled):
    client = _worker()._client_for(github_enabled, "246")
    assert client.installation_id == "246"


def test_each_installation_gets_its_own_client(github_enabled):
    """Separate tokens are cached per client, so sharing one would cross tenants."""
    worker = _worker()

    first = worker._client_for(github_enabled, "111")
    second = worker._client_for(github_enabled, "222")

    assert first is not second
    assert (first.installation_id, second.installation_id) == ("111", "222")


def test_a_repeat_installation_reuses_its_client(github_enabled):
    """Reuse is the point: the token is cached ON the client.

    A fresh client per job would mint a new installation token for every single
    pull request.
    """
    worker = _worker()

    first = worker._client_for(github_enabled, "111")
    again = worker._client_for(github_enabled, "111")

    assert again is first


def test_a_job_without_an_installation_falls_back_to_the_configured_one(monkeypatch):
    """Manual triggers and the e2e harness carry no delivery, and must still work."""
    monkeypatch.setenv("ENABLE_GITHUB", "true")
    monkeypatch.setenv("GITHUB_INSTALLATION_ID", "fallback-id")
    settings = get_settings()

    client = _worker()._client_for(settings, None)

    assert client.installation_id == "fallback-id"


def test_the_client_cache_is_bounded(monkeypatch, github_enabled):
    """A widely installed App must not grow the cache without limit."""
    monkeypatch.setattr(BackgroundWorker, "MAX_CACHED_CLIENTS", 2)
    worker = _worker()

    worker._client_for(github_enabled, "1")
    worker._client_for(github_enabled, "2")
    worker._client_for(github_enabled, "3")

    assert len(worker._clients) == 2
    assert "1" not in worker._clients  # oldest evicted first
    assert set(worker._clients) == {"2", "3"}


def test_no_client_when_github_is_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_GITHUB", "false")
    assert _worker()._client_for(get_settings(), "111") is None


def test_the_worker_loop_resolves_the_client_per_job(monkeypatch):
    """The whole point, proven through a real start() iteration."""
    monkeypatch.setenv("ENABLE_GITHUB", "true")
    seen: list[str | None] = []

    class _Recorder:
        def __init__(self) -> None:
            self.posted: list[tuple] = []

        def get_pull_request_data(self, owner, repo, pr_number):
            return {"code": "", "files": [], "file_contents": {}, "line_map": []}

        def get_pull_request_refs(self, owner, repo, pr_number):
            return {"head_sha": "head-sha", "base_sha": "base-sha"}

        def get_repo_code_corpus(self, owner, repo, ref, prefer_paths=None):
            return []

        def create_check_run(self, *args, **kwargs):
            return True

        def upsert_comment(self, owner, repo, pr_number, body):
            self.posted.append((owner, repo, pr_number, body))
            return True

    recorder = _Recorder()

    def _fake_build(settings, installation_id=None):
        seen.append(installation_id)
        return recorder

    monkeypatch.setattr(bw_module, "_build_github_client", _fake_build)

    queue = JobQueue()
    worker = BackgroundWorker(queue)

    async def _run() -> None:
        await queue.enqueue({"owner": "octo", "repo": "hello", "pr_number": 7,
                             "installation_id": "424242"})
        task = asyncio.create_task(worker.start())
        for _ in range(500):
            if recorder.posted:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    assert seen == ["424242"]
    assert recorder.posted, "the review should have been posted with the per-installation client"
