from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.api.health_controller import router as health_router
from sentinel.api.webhook_controller import get_orchestrator
from sentinel.api.webhook_controller import router as webhook_router
from sentinel.api.webhook_security import compute_signature, is_valid_signature


class _DummyOrchestrator:
    def __init__(self) -> None:
        self.received: list[dict] = []

    async def enqueue_pull_request(self, payload: dict) -> None:
        self.received.append(payload)


def _client() -> TestClient:
    app = FastAPI()
    app.dependency_overrides[get_orchestrator] = lambda: _DummyOrchestrator()
    app.include_router(webhook_router)
    app.include_router(health_router)
    return TestClient(app)


def test_is_valid_signature_roundtrip():
    body = b'{"repo":"r","pr_number":1}'
    sig = compute_signature("s3cret", body)
    assert is_valid_signature("s3cret", body, sig) is True
    assert is_valid_signature("s3cret", body, "sha256=deadbeef") is False
    assert is_valid_signature("s3cret", body, None) is False
    assert is_valid_signature("", body, sig) is False


def test_webhook_without_secret_skips_verification(monkeypatch):
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    resp = _client().post("/webhook", json={"repo": "r", "pr_number": 1})
    assert resp.status_code == 200


def test_webhook_with_secret_rejects_missing_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    resp = _client().post("/webhook", json={"repo": "r", "pr_number": 1})
    assert resp.status_code == 401


def test_webhook_with_secret_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    resp = _client().post(
        "/webhook",
        content=b'{"repo":"r","pr_number":1}',
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": "sha256=bad"},
    )
    assert resp.status_code == 401


def test_webhook_with_secret_accepts_valid_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    body = b'{"repo":"r","pr_number":1}'
    sig = compute_signature("s3cret", body)
    resp = _client().post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
    )
    assert resp.status_code == 200


def test_is_valid_signature_non_ascii_header_returns_false():
    # A non-ASCII signature header must not raise (would 500); it must be rejected.
    assert is_valid_signature("s3cret", b'{"repo":"r"}', "sha256=\xff") is False


# --- M9: fail-closed startup in production ---


def _settings(**overrides):
    """Build a Settings from the real parser with env overrides applied."""
    import os

    from sentinel.config.settings import get_settings

    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return get_settings()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_production_without_webhook_secret_refuses_to_start():
    """The one misconfiguration that must never boot quietly."""
    import pytest

    from main import verify_startup_config

    settings = _settings(ENVIRONMENT="production", ENABLE_GITHUB="true", GITHUB_WEBHOOK_SECRET=None)
    assert settings.ENVIRONMENT == "production"

    with pytest.raises(RuntimeError, match="refusing to start"):
        verify_startup_config(settings)


def test_production_with_webhook_secret_starts():
    from main import verify_startup_config

    settings = _settings(
        ENVIRONMENT="production",
        ENABLE_GITHUB="true",
        GITHUB_WEBHOOK_SECRET="s3cret",
        # Startup also refuses to boot without credentials, so a test asserting
        # that it STARTS has to supply them. They are not ambient: conftest
        # clears them precisely so this cannot pass on a developer's .env.
        GITHUB_APP_ID="12345",
        GITHUB_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nx\n",
    )
    verify_startup_config(settings)  # must not raise


def test_development_without_secret_only_warns():
    """Regression guard for the whole pre-M9 suite, which sets no secret."""
    from main import verify_startup_config

    settings = _settings(ENVIRONMENT=None, ENABLE_GITHUB="true", GITHUB_WEBHOOK_SECRET=None)
    assert settings.ENVIRONMENT == "development"

    verify_startup_config(settings)  # warns, does not raise


def test_production_with_github_disabled_starts_without_a_secret():
    """No GitHub traffic means no unsigned-webhook exposure to guard against."""
    from main import verify_startup_config

    settings = _settings(
        ENVIRONMENT="production", ENABLE_GITHUB="false", GITHUB_WEBHOOK_SECRET=None
    )
    verify_startup_config(settings)  # must not raise

