import builtins
import importlib

import sentinel.config.settings as settings_module


def test_to_bool_handles_true_false_and_default():
    assert settings_module._to_bool("true", False) is True
    assert settings_module._to_bool("ON", False) is True
    assert settings_module._to_bool("false", True) is False
    assert settings_module._to_bool(None, True) is True


def test_to_int_and_to_float_handle_invalid_and_negative_values():
    assert settings_module._to_int("7", 5) == 7
    assert settings_module._to_int("-3", 5) == 0
    assert settings_module._to_int("x", 5) == 5

    assert settings_module._to_float("2.5", 1.0) == 2.5
    assert settings_module._to_float("-2.5", 1.0) == 0.0
    assert settings_module._to_float("x", 1.0) == 1.0


def test_get_settings_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_LLM", raising=False)
    monkeypatch.delenv("LLM_MAX_CALLS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_GITHUB", raising=False)
    monkeypatch.delenv("ENABLE_TRANSLATION", raising=False)
    monkeypatch.delenv("ENABLE_DOC_REVIEW", raising=False)
    monkeypatch.delenv("GITHUB_APP_ID", raising=False)
    monkeypatch.delenv("GITHUB_INSTALLATION_ID", raising=False)
    monkeypatch.delenv("GITHUB_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("GITHUB_API_BASE_URL", raising=False)

    settings = settings_module.get_settings()

    assert settings.NVIDIA_API_KEY is None
    assert settings.ENABLE_LLM is True
    assert settings.LLM_MAX_CALLS == 1
    assert settings.LLM_TIMEOUT == 45.0
    assert settings.LLM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert settings.LLM_MODEL == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert settings.LLM_API_KEY is None
    assert settings.ENABLE_GITHUB is True
    assert settings.ENABLE_TRANSLATION is False
    assert settings.ENABLE_DOC_REVIEW is True
    assert settings.GITHUB_APP_ID is None
    assert settings.GITHUB_INSTALLATION_ID is None
    assert settings.GITHUB_PRIVATE_KEY is None
    assert settings.GITHUB_API_BASE_URL == "https://api.github.com"


def test_get_settings_reads_env_values_and_empty_key_handling(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "  nv-key  ")
    monkeypatch.setenv("ENABLE_LLM", "false")
    monkeypatch.setenv("LLM_MAX_CALLS", "12")
    monkeypatch.setenv("LLM_TIMEOUT", "4.5")
    monkeypatch.setenv("LLM_BASE_URL", " https://api.groq.com/openai/v1 ")
    monkeypatch.setenv("LLM_MODEL", " llama-3.3-70b-versatile ")
    monkeypatch.setenv("LLM_API_KEY", " gsk-key ")
    monkeypatch.setenv("ENABLE_GITHUB", "false")
    monkeypatch.setenv("ENABLE_TRANSLATION", "false")
    monkeypatch.setenv("ENABLE_DOC_REVIEW", "false")
    monkeypatch.setenv("GITHUB_APP_ID", " 12345 ")
    monkeypatch.setenv("GITHUB_INSTALLATION_ID", " 999 ")
    monkeypatch.setenv("GITHUB_PRIVATE_KEY", "line1\\nline2")
    monkeypatch.setenv("GITHUB_API_BASE_URL", " https://ghe.local/api/v3 ")

    settings = settings_module.get_settings()

    assert settings.NVIDIA_API_KEY == "nv-key"
    assert settings.ENABLE_LLM is False
    assert settings.LLM_MAX_CALLS == 12
    assert settings.LLM_TIMEOUT == 4.5
    assert settings.LLM_BASE_URL == "https://api.groq.com/openai/v1"
    assert settings.LLM_MODEL == "llama-3.3-70b-versatile"
    assert settings.LLM_API_KEY == "gsk-key"  # explicit LLM_API_KEY wins over NVIDIA_API_KEY
    assert settings.ENABLE_GITHUB is False
    assert settings.ENABLE_TRANSLATION is False
    assert settings.ENABLE_DOC_REVIEW is False
    assert settings.GITHUB_APP_ID == "12345"
    assert settings.GITHUB_INSTALLATION_ID == "999"
    assert settings.GITHUB_PRIVATE_KEY == "line1\nline2"
    assert settings.GITHUB_API_BASE_URL == "https://ghe.local/api/v3"

    monkeypatch.setenv("NVIDIA_API_KEY", "   ")
    settings_empty = settings_module.get_settings()
    assert settings_empty.NVIDIA_API_KEY is None


def test_llm_api_key_falls_back_to_nvidia_key(monkeypatch):
    # Back-compat: existing setups only set NVIDIA_API_KEY.
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-fallback")
    settings = settings_module.get_settings()
    assert settings.LLM_API_KEY == "nv-fallback"

    # An empty LLM_API_KEY also falls back, not the empty string.
    monkeypatch.setenv("LLM_API_KEY", "   ")
    settings_empty = settings_module.get_settings()
    assert settings_empty.LLM_API_KEY == "nv-fallback"


def test_get_settings_invalid_numeric_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS", "invalid")
    monkeypatch.setenv("LLM_TIMEOUT", "invalid")

    settings = settings_module.get_settings()

    assert settings.LLM_MAX_CALLS == 1
    assert settings.LLM_TIMEOUT == 45.0


def test_settings_module_import_handles_missing_dotenv(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dotenv":
            raise ImportError("dotenv unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reloaded = importlib.reload(settings_module)
    assert reloaded.load_dotenv is None

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(settings_module)


def test_credentials_do_not_appear_in_the_settings_repr(monkeypatch):
    """A dataclass repr reaches pytest output, log lines and tracebacks.

    This is not hypothetical: a failing assertion in this very module printed a
    live NVIDIA key into the test output. Rendering Settings anywhere must not
    disclose a credential, while the attributes stay readable to the code that
    needs them.
    """
    secrets = {
        "NVIDIA_API_KEY": "nvapi-SENTINEL-TEST-SECRET",
        "LLM_API_KEY": "llm-SENTINEL-TEST-SECRET",
        "GITHUB_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----SENTINEL-TEST-SECRET",
        "GITHUB_WEBHOOK_SECRET": "whsec-SENTINEL-TEST-SECRET",
        "METRICS_TOKEN": "metrics-SENTINEL-TEST-SECRET",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    settings = settings_module.get_settings()
    rendered = repr(settings)

    for name, value in secrets.items():
        assert value not in rendered, f"{name} leaked into repr()"
        assert getattr(settings, name) == value, f"{name} is no longer readable"

    # Non-secret configuration must still be visible, or the repr is useless.
    assert "ENVIRONMENT=" in rendered
    assert "LLM_MODEL=" in rendered
