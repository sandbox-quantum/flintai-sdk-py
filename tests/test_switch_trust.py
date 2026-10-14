"""Tests for the Switch Trust SDK library."""

import logging
import warnings
from unittest.mock import MagicMock

import pytest

import switch_trust
from switch_trust.guardrails import InsecureGatewayWarning, SwitchTrustGuardrailsError
from switch_trust.plugins import SwitchTrustPlugin


def test_actions_before_init_raise():
    switch_trust.shutdown()
    with pytest.raises(RuntimeError, match="Call switch_trust.init"):
        switch_trust.register_plugin(SwitchTrustPlugin())


def test_init_configuration():
    client = switch_trust.init(provider="anthropic", require_guardrails=False)
    assert client.provider == "anthropic"


def test_double_init_shuts_down_previous():
    first = switch_trust.init(require_guardrails=False)
    plugin = MagicMock()
    first.register_plugin(plugin)

    switch_trust.init(require_guardrails=False)

    plugin.on_shutdown.assert_called_once()


def test_version_is_available():
    assert isinstance(switch_trust.__version__, str)
    assert len(switch_trust.__version__) > 0


def test_shutdown_idempotent():
    switch_trust.init(require_guardrails=False)
    switch_trust.shutdown()
    switch_trust.shutdown()


def test_reinit_after_shutdown():
    first = switch_trust.init(require_guardrails=False)
    assert first is not None

    switch_trust.shutdown()

    second = switch_trust.init(require_guardrails=False)
    assert second is not None


def test_init_with_guardrails_params():
    client = switch_trust.init(
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    assert client.guardrails_config is not None
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "grl_key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "sk-test"
    assert client.guardrails_config.gateway_url == "https://gw.example.com"


def test_init_with_guardrails_and_provider():
    client = switch_trust.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    assert client.guardrails_config.provider == "openai"
    assert client.guardrails_config.base_url == "https://gw.example.com/openai"


def test_init_with_guardrails_policy_id():
    client = switch_trust.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
        policy_id="pol-1",
    )
    assert client.guardrails_config.headers["X-Guardrails-Policy-Id"] == "pol-1"


def test_init_partial_guardrails_params_raises():
    with pytest.raises(ValueError, match="gateway_url and api_key are both required"):
        switch_trust.init(gateway_url="https://gw.example.com")


def test_init_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        switch_trust.init(provider="cohere")


def test_version_fallback_when_package_not_installed(monkeypatch):
    import importlib
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)),
    )
    importlib.reload(switch_trust)
    assert switch_trust.__version__ == "0.0.0-dev"
    monkeypatch.undo()
    importlib.reload(switch_trust)


# --- Environment variable configuration ---


def test_init_from_env_vars(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("SWITCH_TRUST_LLM_API_KEY", "env-llm-key")
    client = switch_trust.init()
    assert client.guardrails_config is not None
    assert client.guardrails_config.gateway_url == "https://gw.env.com"
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "env-api-key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "env-llm-key"


def test_init_from_flintai_fallback_env_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = switch_trust.init()
    assert client.guardrails_config is not None
    assert client.guardrails_config.gateway_url == "https://gw.env.com"
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "env-api-key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "env-llm-key"


def test_init_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = switch_trust.init(
        gateway_url="https://explicit.com",
        api_key="explicit-key",
        llm_api_key="explicit-llm-key",
    )
    assert client.guardrails_config.gateway_url == "https://explicit.com"
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "explicit-key"
    assert client.guardrails_config.headers["X-LLM-API-Key"] == "explicit-llm-key"


def test_init_no_config_raises_by_default():
    with pytest.raises(
        SwitchTrustGuardrailsError, match="Guardrails configuration is required"
    ):
        switch_trust.init()


def test_init_no_env_no_params_skips_guardrails():
    client = switch_trust.init(require_guardrails=False)
    assert client.guardrails_config is None


def test_init_partial_env_raises(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    with pytest.raises(ValueError, match="gateway_url and api_key are both required"):
        switch_trust.init()


def test_init_policy_id_from_env(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("FLINTAI_POLICY_ID", "env-pol-1")
    client = switch_trust.init()
    assert client.guardrails_config.headers["X-Guardrails-Policy-Id"] == "env-pol-1"


def test_init_without_llm_api_key():
    client = switch_trust.init(
        gateway_url="https://gw.example.com",
        api_key="grl_key",
    )
    assert client.guardrails_config is not None
    assert client.guardrails_config.headers["X-FlintAI-API-Key"] == "grl_key"
    assert "X-LLM-API-Key" not in client.guardrails_config.headers


# --- plaintext HTTP rejection ---


def test_init_rejects_http_non_loopback():
    with pytest.raises(ValueError, match="must use https://"):
        switch_trust.init(
            gateway_url="http://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )


def test_init_http_loopback_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = switch_trust.init(
            gateway_url="http://localhost:8080",
            api_key="key",
            llm_api_key="llm_key",
            allow_insecure_gateway=True,
        )
    assert client.guardrails_config is not None
    assert client.guardrails_config.gateway_url == "http://localhost:8080"
    assert any(issubclass(x.category, InsecureGatewayWarning) for x in w)


_OPT_OUT_MARKER = "SECURITY CONTROL DISABLED"


def _opt_out_records(caplog):
    return [r for r in caplog.records if _OPT_OUT_MARKER in r.getMessage()]


def test_init_opt_out_emits_audit_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.init(require_guardrails=False)
    assert len(_opt_out_records(caplog)) == 1


def test_init_guarded_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.init(
            gateway_url="https://gw.example.com", api_key="key", llm_api_key="llm_key"
        )
    assert _opt_out_records(caplog) == []


def test_opt_out_warning_is_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.init(require_guardrails=False)
        switch_trust.init(require_guardrails=False)
        switch_trust.init(require_guardrails=False)
    assert len(_opt_out_records(caplog)) == 1


def test_dangerously_disable_guardrails_equivalent_to_false():
    client = switch_trust.init(dangerously_disable_guardrails=True)
    assert client.require_guardrails is False
    assert client.guardrails_config is None


def test_dangerously_disable_guardrails_emits_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.init(dangerously_disable_guardrails=True)
    assert len(_opt_out_records(caplog)) == 1


def test_dangerously_disable_conflicts_with_require_true():
    with pytest.raises(ValueError, match="Conflicting options"):
        switch_trust.init(dangerously_disable_guardrails=True, require_guardrails=True)


def test_dangerously_disable_false_keeps_guardrails_required():
    client = switch_trust.init(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
        dangerously_disable_guardrails=False,
    )
    assert client.require_guardrails is True


def test_status_none_before_init():
    switch_trust.shutdown()
    assert switch_trust.status() is None


def test_status_active_when_guarded(monkeypatch):
    monkeypatch.setenv("AGENT_ID", "a1")
    monkeypatch.setenv("AGENT_NAME", "agent-one")
    switch_trust.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    st = switch_trust.status()
    assert st.active is True
    assert st.require_guardrails is True
    assert st.provider == "openai"
    assert st.agent_id == "a1"
    assert st.agent_name == "agent-one"


def test_status_inactive_when_opted_out():
    switch_trust.init(provider="openai", dangerously_disable_guardrails=True)
    st = switch_trust.status()
    assert st.active is False
    assert st.require_guardrails is False
    assert st.provider == "openai"
