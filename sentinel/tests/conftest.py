"""Test-suite isolation from the developer's local `.env`.

`sentinel.config.settings` calls `load_dotenv()` at import, so whatever sits in a
developer's `.env` leaks into every `get_settings()` call the suite makes. That
was harmless while nobody had a webhook secret configured — the M0 gate only
enforces signatures "when a secret is set", and no secret was ever set locally.

Preparing for production breaks that assumption: the moment `GITHUB_WEBHOOK_SECRET`
lands in `.env`, ~46 webhook tests start getting 401 instead of their expected
status, because they post unsigned bodies. CI never saw it (no `.env` in the
runner), so the suite passed there and failed on the machine that was closest to
shipping — the worst way round.

The fixture below pins the deployment-posture variables to a known-empty state for
every test. Tests that need one set it themselves with `monkeypatch.setenv`, which
still wins: this runs at setup, before the test body.
"""

import pytest

# Variables that change how the app gates requests, rather than what it computes.
# Anything here must be absent by default or the suite stops being hermetic.
_POSTURE_VARS = (
    "GITHUB_WEBHOOK_SECRET",
    "METRICS_TOKEN",
    "ENVIRONMENT",
    # Not a gate, but the same trap: it is the fallback used when a job carries
    # no installation of its own, so a developer with one in `.env` would see
    # the fallback tests pass locally and fail in CI, where it is unset.
    "GITHUB_INSTALLATION_ID",
    # These two became posture the moment startup began refusing to boot
    # without them. Leaving them ambient meant a developer's .env satisfied
    # the check and CI, which has no .env, did not — the suite passing locally
    # for a reason that does not exist on the runner.
    "GITHUB_APP_ID",
    "GITHUB_PRIVATE_KEY",
)


@pytest.fixture(autouse=True)
def _hermetic_posture_env(monkeypatch):
    """Clear deployment-posture env vars so tests never depend on a local .env."""
    for name in _POSTURE_VARS:
        monkeypatch.delenv(name, raising=False)
