"""Startup refuses configurations that would run but not work.

The webhook-secret gate is covered in test_webhook_signature.py. This file
covers the credential gate, which exists because every GitHub call in this
codebase is failure-safe: without an app id or a private key the service
boots, answers /health, computes reviews, and posts none of them. Nothing goes
red. That is a worse outcome than refusing to start.
"""

import pytest

from main import verify_startup_config
from sentinel.config.settings import get_settings

_CREDENTIALS = ("GITHUB_APP_ID", "GITHUB_PRIVATE_KEY")


@pytest.fixture
def clean_credentials(monkeypatch):
    """Credentials absent unless a test sets them.

    conftest's posture fixture does not clear these, and a developer's .env
    very likely has them — without this the tests below would pass locally for
    the wrong reason and fail in CI.
    """
    for name in _CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENABLE_GITHUB", "true")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    return monkeypatch


def _with_credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nx\n")


def test_production_boots_with_a_secret_and_credentials(clean_credentials):
    _with_credentials(clean_credentials)
    clean_credentials.setenv("ENVIRONMENT", "production")

    verify_startup_config(get_settings())  # must not raise


def test_production_refuses_to_start_without_an_app_id(clean_credentials):
    clean_credentials.setenv("ENVIRONMENT", "production")
    clean_credentials.setenv("GITHUB_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nx\n")

    with pytest.raises(RuntimeError, match="GITHUB_APP_ID"):
        verify_startup_config(get_settings())


def test_production_refuses_to_start_without_a_private_key(clean_credentials):
    clean_credentials.setenv("ENVIRONMENT", "production")
    clean_credentials.setenv("GITHUB_APP_ID", "12345")

    with pytest.raises(RuntimeError, match="GITHUB_PRIVATE_KEY"):
        verify_startup_config(get_settings())


def test_the_error_names_everything_that_is_missing(clean_credentials):
    clean_credentials.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError) as excinfo:
        verify_startup_config(get_settings())

    message = str(excinfo.value)
    assert "GITHUB_APP_ID" in message and "GITHUB_PRIVATE_KEY" in message


def test_the_installation_id_is_not_required(clean_credentials):
    """Since multi-tenancy it is only a fallback, not a requirement."""
    _with_credentials(clean_credentials)
    clean_credentials.setenv("ENVIRONMENT", "production")
    clean_credentials.delenv("GITHUB_INSTALLATION_ID", raising=False)

    verify_startup_config(get_settings())  # must not raise


def test_development_warns_and_continues(clean_credentials, caplog):
    """Local dev and the test suite must never be held to production shape."""
    clean_credentials.setenv("ENVIRONMENT", "development")

    with caplog.at_level("WARNING"):
        verify_startup_config(get_settings())

    assert "cannot be posted" in caplog.text


def test_github_disabled_skips_every_check(clean_credentials):
    clean_credentials.setenv("ENABLE_GITHUB", "false")
    clean_credentials.setenv("ENVIRONMENT", "production")
    clean_credentials.delenv("GITHUB_WEBHOOK_SECRET", raising=False)

    verify_startup_config(get_settings())  # must not raise


def test_the_webhook_secret_gate_still_fires_first(clean_credentials):
    """Guards the restructure: an unsigned endpoint is the worse failure."""
    _with_credentials(clean_credentials)
    clean_credentials.setenv("ENVIRONMENT", "production")
    clean_credentials.delenv("GITHUB_WEBHOOK_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="GITHUB_WEBHOOK_SECRET"):
        verify_startup_config(get_settings())
