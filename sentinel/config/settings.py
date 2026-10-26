import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, parsed)


@dataclass(frozen=True)
class Settings:
    ENVIRONMENT: str
    NVIDIA_API_KEY: str | None
    ENABLE_LLM: bool
    LLM_MAX_CALLS: int
    # Reviews per installation per day that may use the LLM. 0 = unlimited.
    # Caps what one API key costs when the App is public; past the cap reviews
    # still run, with static recommendations.
    LLM_DAILY_QUOTA: int
    LLM_TIMEOUT: float
    LLM_BASE_URL: str
    LLM_MODEL: str
    LLM_API_KEY: str | None
    ENABLE_GITHUB: bool
    ENABLE_CHECKS: bool
    ENABLE_TRANSLATION: bool
    ENABLE_DOC_REVIEW: bool
    GITHUB_APP_ID: str | None
    GITHUB_INSTALLATION_ID: str | None
    GITHUB_PRIVATE_KEY: str | None
    GITHUB_API_BASE_URL: str
    GITHUB_WEBHOOK_SECRET: str | None
    METRICS_TOKEN: str | None
    REDIS_URL: str | None


def get_settings() -> Settings:
    # Deployment posture. "production" makes the webhook-secret requirement
    # fail-closed at startup (see main.py); anything else preserves the
    # historical warn-and-continue behavior for local dev and the test suite.
    raw_environment = os.getenv("ENVIRONMENT")
    environment = raw_environment.strip().lower() if raw_environment is not None else ""
    if environment == "":
        environment = "development"

    raw_key = os.getenv("NVIDIA_API_KEY")
    nvidia_key = raw_key.strip() if raw_key is not None else None
    if nvidia_key == "":
        nvidia_key = None

    # Provider-agnostic LLM config (OpenAI-compatible). Defaults reproduce today's
    # NVIDIA behavior; LLM_API_KEY falls back to NVIDIA_API_KEY for back-compat.
    raw_llm_base = os.getenv("LLM_BASE_URL")
    llm_base_url = raw_llm_base.strip() if raw_llm_base is not None else ""
    if llm_base_url == "":
        llm_base_url = "https://integrate.api.nvidia.com/v1"

    raw_llm_model = os.getenv("LLM_MODEL")
    llm_model = raw_llm_model.strip() if raw_llm_model is not None else ""
    if llm_model == "":
        llm_model = "deepseek-ai/deepseek-v4-flash"

    raw_llm_key = os.getenv("LLM_API_KEY")
    llm_api_key = raw_llm_key.strip() if raw_llm_key is not None else None
    if not llm_api_key:
        llm_api_key = nvidia_key

    raw_private_key = os.getenv("GITHUB_PRIVATE_KEY")
    github_private_key = raw_private_key.strip() if raw_private_key is not None else None
    if github_private_key == "":
        github_private_key = None
    elif github_private_key is not None:
        github_private_key = github_private_key.replace("\\n", "\n")

    github_app_id = os.getenv("GITHUB_APP_ID")
    if github_app_id is not None:
        github_app_id = github_app_id.strip() or None

    github_installation_id = os.getenv("GITHUB_INSTALLATION_ID")
    if github_installation_id is not None:
        github_installation_id = github_installation_id.strip() or None

    github_api_base_url = os.getenv("GITHUB_API_BASE_URL")
    if github_api_base_url is None:
        github_api_base_url = "https://api.github.com"
    github_api_base_url = github_api_base_url.strip() or "https://api.github.com"

    raw_webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    github_webhook_secret = raw_webhook_secret.strip() if raw_webhook_secret is not None else None
    if github_webhook_secret == "":
        github_webhook_secret = None

    # Bearer token guarding GET /metrics. Unset -> endpoint stays open (dev default).
    raw_metrics_token = os.getenv("METRICS_TOKEN")
    metrics_token = raw_metrics_token.strip() if raw_metrics_token is not None else None
    if metrics_token == "":
        metrics_token = None

    # Durable queue/dedup backend. Unset -> in-memory (single-process, non-durable).
    raw_redis_url = os.getenv("REDIS_URL")
    redis_url = raw_redis_url.strip() if raw_redis_url is not None else None
    if redis_url == "":
        redis_url = None

    return Settings(
        ENVIRONMENT=environment,
        NVIDIA_API_KEY=nvidia_key,
        ENABLE_LLM=_to_bool(os.getenv("ENABLE_LLM"), True),
        LLM_MAX_CALLS=_to_int(os.getenv("LLM_MAX_CALLS"), 1),
        LLM_DAILY_QUOTA=_to_int(os.getenv("LLM_DAILY_QUOTA"), 0),
        LLM_TIMEOUT=_to_float(os.getenv("LLM_TIMEOUT"), 5.0),
        LLM_BASE_URL=llm_base_url,
        LLM_MODEL=llm_model,
        LLM_API_KEY=llm_api_key,
        ENABLE_GITHUB=_to_bool(os.getenv("ENABLE_GITHUB"), True),
        ENABLE_CHECKS=_to_bool(os.getenv("ENABLE_CHECKS"), True),
        ENABLE_TRANSLATION=_to_bool(os.getenv("ENABLE_TRANSLATION"), False),
        ENABLE_DOC_REVIEW=_to_bool(os.getenv("ENABLE_DOC_REVIEW"), True),
        GITHUB_APP_ID=github_app_id,
        GITHUB_INSTALLATION_ID=github_installation_id,
        GITHUB_PRIVATE_KEY=github_private_key,
        GITHUB_API_BASE_URL=github_api_base_url,
        GITHUB_WEBHOOK_SECRET=github_webhook_secret,
        METRICS_TOKEN=metrics_token,
        REDIS_URL=redis_url,
    )
