"""Tests for the FlintAI SDK library."""

from unittest.mock import MagicMock

import flintai_sdk
import pytest
from flintai_sdk.plugins import FlintAIPlugin


def test_actions_before_init_raise():
    flintai_sdk.shutdown()
    with pytest.raises(RuntimeError, match="Call flintai_sdk.init"):
        flintai_sdk.register_plugin(FlintAIPlugin())


def test_init_configuration():
    client = flintai_sdk.init(provider="anthropic")
    assert client.provider == "anthropic"


def test_double_init_shuts_down_previous():
    first = flintai_sdk.init()
    plugin = MagicMock()
    first.register_plugin(plugin)

    flintai_sdk.init()

    plugin.on_shutdown.assert_called_once()


def test_version_is_available():
    assert isinstance(flintai_sdk.__version__, str)
    assert len(flintai_sdk.__version__) > 0


def test_shutdown_idempotent():
    flintai_sdk.init()
    flintai_sdk.shutdown()
    flintai_sdk.shutdown()


def test_reinit_after_shutdown():
    first = flintai_sdk.init()
    assert first is not None

    flintai_sdk.shutdown()

    second = flintai_sdk.init()
    assert second is not None


def test_init_with_guardrails_params():
    client = flintai_sdk.init(
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    assert client.guardrails_config is not None
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "grl_key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "sk-test"
    assert client.guardrails_config.gateway_url == "https://gw.example.com"


def test_init_with_guardrails_and_provider():
    client = flintai_sdk.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    assert client.guardrails_config.provider == "openai"
    assert client.guardrails_config.base_url == "https://gw.example.com/openai"


def test_init_with_guardrails_policy_id():
    client = flintai_sdk.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
        policy_id="pol-1",
    )
    assert client.guardrails_config.headers["X-Guardrails-Policy-Id"] == "pol-1"


def test_init_partial_guardrails_params_raises():
    with pytest.raises(
        ValueError, match="gateway_url, api_key, and llm_api_key are all required"
    ):
        flintai_sdk.init(gateway_url="https://gw.example.com")


def test_init_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        flintai_sdk.init(provider="cohere")


def test_version_fallback_when_package_not_installed(monkeypatch):
    import importlib
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )
    importlib.reload(flintai_sdk)
    assert flintai_sdk.__version__ == "0.0.0-dev"
    monkeypatch.undo()
    importlib.reload(flintai_sdk)


# --- Environment variable configuration ---


def test_init_from_env_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = flintai_sdk.init()
    assert client.guardrails_config is not None
    assert client.guardrails_config.gateway_url == "https://gw.env.com"
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "env-api-key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "env-llm-key"


def test_init_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = flintai_sdk.init(
        gateway_url="https://explicit.com",
        api_key="explicit-key",
        llm_api_key="explicit-llm-key",
    )
    assert client.guardrails_config.gateway_url == "https://explicit.com"
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "explicit-key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "explicit-llm-key"


def test_init_no_env_no_params_skips_guardrails():
    client = flintai_sdk.init()
    assert client.guardrails_config is None


def test_init_partial_env_raises(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    with pytest.raises(
        ValueError, match="gateway_url, api_key, and llm_api_key are all required"
    ):
        flintai_sdk.init()


def test_init_policy_id_from_env(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("FLINTAI_POLICY_ID", "env-pol-1")
    client = flintai_sdk.init()
    assert client.guardrails_config.headers["X-Guardrails-Policy-Id"] == "env-pol-1"
