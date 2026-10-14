"""Tests for error paths in switch_trust.guardrails."""

import logging
import warnings

import pytest

from switch_trust import core
from switch_trust.core import SwitchTrustClient
from switch_trust.guardrails import (
    GuardrailsConfig,
    InsecureGatewayWarning,
    _validate_guardrails_params,
    build_guardrails_config,
    configure_guardrails,
    detect_provider,
    resolve_from_env,
)


def test_no_provider_defers_to_wrap_time(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider=None))
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert config.provider is None
    assert config.base_url == "https://gw.example.com"


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        configure_guardrails(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
            provider="cohere",
        )


def test_configure_without_client(monkeypatch):
    monkeypatch.setattr(core, "_client", None)
    with pytest.warns(UserWarning, match="called before switch_trust.init"):
        config = configure_guardrails(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
            provider="openai",
        )
    assert config.provider == "openai"
    assert config.base_url == "https://gw.example.com/openai"


@pytest.mark.parametrize(
    "env_var,expected",
    [
        ("GOOGLE_API_KEY", "google"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        pytest.param(None, None, id="none"),
    ],
)
def test_detect_provider(monkeypatch, env_var, expected):
    if env_var:
        monkeypatch.setenv(env_var, "fake-key")
    for other in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        if other != env_var:
            monkeypatch.delenv(other, raising=False)
    assert detect_provider() == expected


@pytest.mark.parametrize(
    "policy_id,expect_header",
    [
        pytest.param("pol-1", True, id="with-policy"),
        pytest.param(None, False, id="without-policy"),
    ],
)
def test_policy_id_header_behavior(monkeypatch, policy_id, expect_header):
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider="openai"))
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
        provider="openai",
        policy_id=policy_id,
    )
    if expect_header:
        assert config.headers["X-Guardrails-Policy-Id"] == policy_id
        assert config.policy_id == policy_id
    else:
        assert "X-Guardrails-Policy-Id" not in config.headers
        assert config.policy_id is None


def test_configure_stores_config_on_client(monkeypatch):
    client = SwitchTrustClient(provider="openai")
    monkeypatch.setattr(core, "_client", client)
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert core._client.guardrails_config is config


def test_configure_uses_client_provider(monkeypatch):
    client = SwitchTrustClient(provider="anthropic")
    monkeypatch.setattr(core, "_client", client)
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert config.provider == "anthropic"
    assert config.base_url == "https://gw.example.com/anthropic"


def test_configure_falls_through_to_detect_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = SwitchTrustClient(provider=None)
    monkeypatch.setattr(core, "_client", client)
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert config.provider == "openai"
    assert config.base_url == "https://gw.example.com/openai"


def test_trailing_slash_normalization(monkeypatch):
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider="openai"))
    config = configure_guardrails(
        gateway_url="https://gw.example.com/",
        api_key="key",
        llm_api_key="llm_key",
        provider="openai",
    )
    assert config.base_url == "https://gw.example.com/openai"


def test_guardrails_config_clear():
    config = GuardrailsConfig(
        base_url="https://gw.example.com/openai",
        headers={
            "X-FlintAI-API-Key": "grl_sk_1234567890",
            "X-LLM-API-Key": "sk-proj-abc123def456",
        },
        provider="openai",
        gateway_url="https://gw.example.com",
        policy_id="pol-1",
    )
    config.clear()
    assert config.headers == {}
    assert config.base_url == ""
    assert config.gateway_url == ""
    assert config.provider is None
    assert config.policy_id is None


@pytest.mark.parametrize(
    "headers,expected_in,expected_not_in",
    [
        pytest.param(
            {
                "X-FlintAI-API-Key": "grl_sk_1234567890",
                "X-LLM-API-Key": "sk-proj-abc123def456",
            },
            ["gr***", "sk***"],
            ["grl_sk_1234567890", "sk-proj-abc123def456"],
            id="long-values",
        ),
        pytest.param(
            {"X-FlintAI-API-Key": "ab"},
            ["'X-FlintAI-API-Key': '***'"],
            [],
            id="short-values",
        ),
    ],
)
def test_guardrails_config_repr_masks_credentials(
    headers, expected_in, expected_not_in
):
    config = GuardrailsConfig(
        base_url="https://gw.example.com/openai",
        headers=headers,
        provider="openai",
        gateway_url="https://gw.example.com",
    )
    result = repr(config)
    for s in expected_in:
        assert s in result
    for s in expected_not_in:
        assert s not in result


@pytest.mark.parametrize(
    "provider,expected_path",
    [
        ("openai", "/openai"),
        ("anthropic", "/anthropic"),
        ("google", "/gemini"),
    ],
)
def test_build_guardrails_config(provider, expected_path):
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
        provider=provider,
    )
    assert config.base_url == f"https://gw.example.com{expected_path}"
    assert config.headers["X-FlintAI-API-Key"] == "key"
    assert config.headers["X-LLM-API-Key"] == "llm_key"
    assert config.provider == provider


def test_build_guardrails_config_with_policy_id():
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
        provider="openai",
        policy_id="pol-1",
    )
    assert config.headers["X-Guardrails-Policy-Id"] == "pol-1"
    assert config.policy_id == "pol-1"


def test_build_guardrails_config_adds_agent_identity_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_ID", "agn_123")
    monkeypatch.setenv("AGENT_NAME", "env-agent")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        provider="openai",
    )
    assert config.headers["X-Agent-Id"] == "agn_123"
    assert config.headers["X-Agent-Name"] == "env-agent"


def test_build_guardrails_config_omits_whitespace_only_agent_identity(monkeypatch):
    """A whitespace-only AGENT_ID or AGENT_NAME is treated as unset.

    httpx's HTTP/1.1 layer (h11) rejects a header value with leading or
    trailing whitespace. Sending one would fail every call on the client side
    with LocalProtocolError, before the call reaches the gateway.
    """
    monkeypatch.setenv("AGENT_ID", "   ")
    monkeypatch.setenv("AGENT_NAME", "   ")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        provider="openai",
    )
    assert "X-Agent-Id" not in config.headers
    assert "X-Agent-Name" not in config.headers


def test_build_guardrails_config_strips_agent_identity(monkeypatch):
    monkeypatch.setenv("AGENT_ID", " agn_123\t")
    monkeypatch.setenv("AGENT_NAME", " agent ")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        provider="openai",
    )
    assert config.headers["X-Agent-Id"] == "agn_123"
    assert config.headers["X-Agent-Name"] == "agent"


@pytest.mark.parametrize(
    "var,header,other_var,other_header",
    [
        pytest.param("AGENT_NAME", "X-Agent-Name", "AGENT_ID", "X-Agent-Id", id="name"),
        pytest.param("AGENT_ID", "X-Agent-Id", "AGENT_NAME", "X-Agent-Name", id="id"),
    ],
)
def test_build_guardrails_config_skips_non_ascii_agent_identity(
    monkeypatch, caplog, var, header, other_var, other_header
):
    """httpx encodes header values as ASCII, so a non-ASCII value would make
    every call raise UnicodeEncodeError. The warning must not echo the value."""
    monkeypatch.setenv(var, " café-bot ")
    monkeypatch.setenv(other_var, "agent-1")
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        config = build_guardrails_config(
            gateway_url="https://gw.example.com",
            api_key="key",
            provider="openai",
        )
    assert header not in config.headers
    assert config.headers[other_header] == "agent-1"
    assert var in caplog.text
    assert "café" not in caplog.text


def test_build_guardrails_config_does_not_touch_global_state(monkeypatch):
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider="openai"))
    build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
        provider="openai",
    )
    assert core._client.guardrails_config is None


def test_build_guardrails_config_without_llm_api_key():
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        provider="openai",
    )
    assert config.headers["X-FlintAI-API-Key"] == "key"
    assert "X-LLM-API-Key" not in config.headers
    assert config.provider == "openai"


def test_check_optional_params_without_llm_api_key():
    from switch_trust.guardrails import check_optional_guardrails_params

    result = check_optional_guardrails_params(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key=None,
    )
    assert result is True


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        pytest.param(
            {
                "gateway_url": "https://gw.example.com",
                "api_key": "key",
                "llm_api_key": "llm_key",
                "provider": "cohere",
            },
            "Unsupported provider",
            id="unsupported-provider",
        ),
        pytest.param(
            {
                "gateway_url": "not-a-url",
                "api_key": "key",
                "llm_api_key": "llm_key",
                "provider": "openai",
            },
            "gateway_url must use https://",
            id="bad-url",
        ),
    ],
)
def test_build_guardrails_config_rejects_invalid(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        build_guardrails_config(**kwargs)


def test_detect_provider_multiple_keys_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Multiple provider API keys detected"):
        detect_provider()


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        pytest.param(
            {"gateway_url": "bad", "api_key": "key", "llm_api_key": "llm"},
            "gateway_url must use https://",
            id="bad-url",
        ),
        pytest.param(
            {
                "gateway_url": "https://gw.example.com",
                "api_key": "",
                "llm_api_key": "llm",
            },
            "api_key must be a non-empty string",
            id="empty-api-key",
        ),
    ],
)
def test_configure_no_provider_rejects_invalid_params(monkeypatch, kwargs, error_match):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider=None))
    with pytest.raises(ValueError, match=error_match):
        configure_guardrails(**kwargs)


def test_configure_no_provider_with_policy_id(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider=None))
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm",
        policy_id="pol-99",
    )
    assert config.headers["X-Guardrails-Policy-Id"] == "pol-99"
    assert config.policy_id == "pol-99"
    assert config.provider is None


# --- resolve_from_env ---


def test_resolve_from_env_reads_switch_trust_vars(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("SWITCH_TRUST_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("SWITCH_TRUST_POLICY_ID", "env-pol-1")
    gw, api, llm, pol = resolve_from_env()
    assert gw == "https://gw.env.com"
    assert api == "env-api-key"
    assert llm == "env-llm-key"
    assert pol == "env-pol-1"


def test_resolve_from_env_reads_flintai_fallback_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("FLINTAI_POLICY_ID", "env-pol-1")
    gw, api, llm, pol = resolve_from_env()
    assert gw == "https://gw.env.com"
    assert api == "env-api-key"
    assert llm == "env-llm-key"
    assert pol == "env-pol-1"


def test_resolve_from_env_switch_trust_takes_precedence_over_flintai(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://legacy.env.com")
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    gw, _, _, _ = resolve_from_env()
    assert gw == "https://gw.env.com"


def test_resolve_from_env_explicit_wins(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("FLINTAI_POLICY_ID", "env-pol-1")
    gw, api, llm, pol = resolve_from_env(
        gateway_url="https://explicit.com",
        api_key="explicit-key",
        llm_api_key="explicit-llm",
        policy_id="explicit-pol",
    )
    assert gw == "https://explicit.com"
    assert api == "explicit-key"
    assert llm == "explicit-llm"
    assert pol == "explicit-pol"


def test_resolve_from_env_no_vars_returns_none():
    gw, api, llm, pol = resolve_from_env()
    assert all(v is None for v in (gw, api, llm, pol))


def test_resolve_from_env_partial_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    gw, api, llm, pol = resolve_from_env()
    assert gw == "https://gw.env.com"
    assert api is None
    assert llm is None
    assert pol is None


def test_resolve_from_env_does_not_load_dotenv_by_default(monkeypatch, tmp_path):
    """A cwd .env must be ignored unless load_dotenv is explicitly requested."""
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    env_file = tmp_path / ".env"
    env_file.write_text("FLINTAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    from switch_trust import guardrails

    guardrails._dotenv_loaded = False
    _, api, _, _ = resolve_from_env()
    assert api is None


def test_resolve_from_env_loads_dotenv_when_opted_in(monkeypatch, tmp_path):
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    env_file = tmp_path / ".env"
    env_file.write_text("FLINTAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    from switch_trust import guardrails

    guardrails._dotenv_loaded = False
    _, api, _, _ = resolve_from_env(load_dotenv=True)
    assert api == "from-dotenv"


def test_resolve_from_env_loads_dotenv_from_path(monkeypatch, tmp_path):
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    env_file = tmp_path / "trusted.env"
    env_file.write_text("FLINTAI_API_KEY=from-path\n")
    from switch_trust import guardrails

    guardrails._dotenv_loaded = False
    _, api, _, _ = resolve_from_env(load_dotenv=str(env_file))
    assert api == "from-path"


def test_resolve_from_env_ignores_parent_dotenv(monkeypatch, tmp_path):
    """Parent directory .env must NOT be loaded — only cwd, and only opt-in."""
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / ".env").write_text("FLINTAI_API_KEY=from-parent\n")
    monkeypatch.chdir(child)
    from switch_trust import guardrails

    guardrails._dotenv_loaded = False
    _, api, _, _ = resolve_from_env(load_dotenv=True)
    assert api is None


# --- gateway host allowlist ---


def test_validate_rejects_empty_hostname():
    with pytest.raises(ValueError, match="valid hostname"):
        build_guardrails_config(
            gateway_url="https:///path",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_allowlist_accepts_matching_host(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "gw.example.com")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw.example.com"


def test_validate_allowlist_switch_trust_takes_precedence_over_flintai(monkeypatch):
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "gw.example.com")
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "gw.corp.com")
    with pytest.raises(ValueError, match="not in the allowed gateway hosts"):
        build_guardrails_config(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_allowlist_rejects_non_matching_host(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "gw.example.com")
    with pytest.raises(ValueError, match="not in the allowed gateway hosts"):
        build_guardrails_config(
            gateway_url="https://evil.example.com",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_allowlist_comma_separated(monkeypatch):
    monkeypatch.setenv(
        "SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "gw1.example.com, gw2.example.com"
    )
    config = build_guardrails_config(
        gateway_url="https://gw2.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw2.example.com"


def test_validate_allowlist_case_insensitive(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "GW.Example.COM")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw.example.com"


def test_validate_wildcard_accepts_any_with_flag(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "*")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureGatewayWarning)
        config = build_guardrails_config(
            gateway_url="https://any-host.example.com",
            api_key="key",
            llm_api_key="llm",
            allow_insecure_gateway=True,
        )
    assert config.gateway_url == "https://any-host.example.com"


def test_validate_wildcard_errors_without_flag(monkeypatch):
    """'*' requires allow_insecure_gateway=True and cannot be enabled via env."""
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "*")
    with pytest.raises(ValueError, match="requires allow_insecure_gateway=True"):
        _validate_guardrails_params("https://any-host.example.com", "key")


def test_validate_wildcard_warns_with_flag(monkeypatch):
    """Arbitrary-host mode ('*') emits a high-visibility warning when enabled."""
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "*")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _validate_guardrails_params(
            "https://any-host.example.com", "key", allow_insecure_gateway=True
        )
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)
    assert "disables the gateway host allowlist" in str(w[0].message)


def test_validate_default_allowlist_rejects_non_default(monkeypatch):
    """With no env allowlist, only the shipped default host is accepted."""
    monkeypatch.delenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", raising=False)
    with pytest.raises(ValueError, match="not in the allowed gateway hosts"):
        build_guardrails_config(
            gateway_url="https://evil.example.com",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_default_allowlist_accepts_app_flintai_dev(monkeypatch):
    monkeypatch.delenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", raising=False)
    config = build_guardrails_config(
        gateway_url="https://api.flintai.dev",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://api.flintai.dev"


def test_validate_default_allowlist_accepts_api_switchagents_ai(monkeypatch):
    monkeypatch.delenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", raising=False)
    config = build_guardrails_config(
        gateway_url="https://api.switchagents.ai",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://api.switchagents.ai"


def test_validate_loopback_exempt_from_allowlist_with_flag(monkeypatch):
    """Loopback is allowed against a restrictive allowlist when the flag is set."""
    monkeypatch.setenv("SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS", "api.flintai.dev")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = build_guardrails_config(
            gateway_url="http://localhost:8080",
            api_key="key",
            llm_api_key="llm",
            allow_insecure_gateway=True,
        )
    assert config.gateway_url == "http://localhost:8080"


def test_validate_loopback_errors_without_flag():
    """Loopback gateway_url requires allow_insecure_gateway=True."""
    with pytest.raises(ValueError, match="requires allow_insecure_gateway=True"):
        _validate_guardrails_params("http://localhost:8080", "key")


def test_validate_https_localhost_errors_without_flag():
    """https://localhost is no longer silently accepted without the flag."""
    with pytest.raises(ValueError, match="requires allow_insecure_gateway=True"):
        _validate_guardrails_params("https://localhost:8080", "key")


# --- plaintext HTTP rejection ---


def test_validate_rejects_http_non_loopback():
    with pytest.raises(ValueError, match="must use https://"):
        _validate_guardrails_params("http://gw.example.com", "key")


def test_validate_http_loopback_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params(
            "http://localhost:8080", "key", allow_insecure_gateway=True
        )
    assert result == "http://localhost:8080"
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)
    assert "plaintext HTTP" in str(w[0].message)


def test_validate_http_127_0_0_1_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params(
            "http://127.0.0.1:8080", "key", allow_insecure_gateway=True
        )
    assert result == "http://127.0.0.1:8080"
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)


def test_validate_http_ipv6_loopback_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params(
            "http://[::1]:8080", "key", allow_insecure_gateway=True
        )
    assert result == "http://[::1]:8080"
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)


def test_validate_https_always_accepted():
    result = _validate_guardrails_params("https://gw.example.com", "key")
    assert result == "https://gw.example.com"


def test_validate_rejects_bad_scheme():
    with pytest.raises(ValueError, match="must use https://"):
        _validate_guardrails_params("ftp://gw.example.com", "key")


def test_validate_rejects_empty_url():
    with pytest.raises(ValueError, match="must use https://"):
        _validate_guardrails_params("", "key")


def test_build_config_rejects_http_non_loopback():
    with pytest.raises(ValueError, match="must use https://"):
        build_guardrails_config(
            gateway_url="http://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )


def test_build_config_http_loopback_accepted():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = build_guardrails_config(
            gateway_url="http://localhost:8080",
            api_key="key",
            llm_api_key="llm_key",
            allow_insecure_gateway=True,
        )
    assert config.gateway_url == "http://localhost:8080"


def test_configure_rejects_http_non_loopback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider=None))
    with pytest.raises(ValueError, match="must use https://"):
        configure_guardrails(
            gateway_url="http://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )
