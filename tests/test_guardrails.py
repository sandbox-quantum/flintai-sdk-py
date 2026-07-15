"""Tests for error paths in flintai.guardrails."""

import warnings

import pytest
from flintai import core
from flintai.core import FlintAIClient
from flintai.guardrails import (
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

    monkeypatch.setattr(core, "_client", FlintAIClient(provider=None))
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
    with pytest.warns(UserWarning, match="called before flintai.init"):
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
    monkeypatch.setattr(core, "_client", FlintAIClient(provider="openai"))
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
    client = FlintAIClient(provider="openai")
    monkeypatch.setattr(core, "_client", client)
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert core._client.guardrails_config is config


def test_configure_uses_client_provider(monkeypatch):
    client = FlintAIClient(provider="anthropic")
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

    client = FlintAIClient(provider=None)
    monkeypatch.setattr(core, "_client", client)
    config = configure_guardrails(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm_key",
    )
    assert config.provider == "openai"
    assert config.base_url == "https://gw.example.com/openai"


def test_trailing_slash_normalization(monkeypatch):
    monkeypatch.setattr(core, "_client", FlintAIClient(provider="openai"))
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


def test_build_guardrails_config_does_not_touch_global_state(monkeypatch):
    monkeypatch.setattr(core, "_client", FlintAIClient(provider="openai"))
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
    from flintai.guardrails import check_optional_guardrails_params

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
            dict(
                gateway_url="https://gw.example.com",
                api_key="key",
                llm_api_key="llm_key",
                provider="cohere",
            ),
            "Unsupported provider",
            id="unsupported-provider",
        ),
        pytest.param(
            dict(
                gateway_url="not-a-url",
                api_key="key",
                llm_api_key="llm_key",
                provider="openai",
            ),
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
            dict(gateway_url="bad", api_key="key", llm_api_key="llm"),
            "gateway_url must use https://",
            id="bad-url",
        ),
        pytest.param(
            dict(gateway_url="https://gw.example.com", api_key="", llm_api_key="llm"),
            "api_key must be a non-empty string",
            id="empty-api-key",
        ),
    ],
)
def test_configure_no_provider_rejects_invalid_params(monkeypatch, kwargs, error_match):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", FlintAIClient(provider=None))
    with pytest.raises(ValueError, match=error_match):
        configure_guardrails(**kwargs)


def test_configure_no_provider_with_policy_id(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", FlintAIClient(provider=None))
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


def test_resolve_from_env_reads_all_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    monkeypatch.setenv("FLINTAI_POLICY_ID", "env-pol-1")
    gw, api, llm, pol = resolve_from_env()
    assert gw == "https://gw.env.com"
    assert api == "env-api-key"
    assert llm == "env-llm-key"
    assert pol == "env-pol-1"


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


def test_resolve_from_env_loads_dotenv(monkeypatch, tmp_path):
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    env_file = tmp_path / ".env"
    env_file.write_text("FLINTAI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    from flintai import guardrails

    guardrails._dotenv_loaded = False
    gw, api, llm, pol = resolve_from_env()
    assert api == "from-dotenv"


def test_resolve_from_env_ignores_parent_dotenv(monkeypatch, tmp_path):
    """Parent directory .env must NOT be loaded — only cwd."""
    pytest.importorskip("dotenv", reason="python-dotenv not installed")
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / ".env").write_text("FLINTAI_API_KEY=from-parent\n")
    monkeypatch.chdir(child)
    from flintai import guardrails

    guardrails._dotenv_loaded = False
    _, api, _, _ = resolve_from_env()
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
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "gw.example.com")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw.example.com"


def test_validate_allowlist_rejects_non_matching_host(monkeypatch):
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "gw.example.com")
    with pytest.raises(ValueError, match="not in the allowed gateway hosts"):
        build_guardrails_config(
            gateway_url="https://evil.example.com",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_allowlist_comma_separated(monkeypatch):
    monkeypatch.setenv(
        "FLINTAI_ALLOWED_GATEWAY_HOSTS", "gw1.example.com, gw2.example.com"
    )
    config = build_guardrails_config(
        gateway_url="https://gw2.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw2.example.com"


def test_validate_allowlist_case_insensitive(monkeypatch):
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "GW.Example.COM")
    config = build_guardrails_config(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://gw.example.com"


def test_validate_wildcard_accepts_any(monkeypatch):
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "*")
    config = build_guardrails_config(
        gateway_url="https://any-host.example.com",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://any-host.example.com"


def test_validate_default_allowlist_rejects_non_default(monkeypatch):
    """With no env allowlist, only the shipped default host is accepted."""
    monkeypatch.delenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", raising=False)
    with pytest.raises(ValueError, match="not in the allowed gateway hosts"):
        build_guardrails_config(
            gateway_url="https://evil.example.com",
            api_key="key",
            llm_api_key="llm",
        )


def test_validate_default_allowlist_accepts_app_flintai_dev(monkeypatch):
    monkeypatch.delenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", raising=False)
    config = build_guardrails_config(
        gateway_url="https://app.flintai.dev",
        api_key="key",
        llm_api_key="llm",
    )
    assert config.gateway_url == "https://app.flintai.dev"


def test_validate_loopback_exempt_from_allowlist(monkeypatch):
    """Loopback is always allowed, even against a restrictive allowlist."""
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "app.flintai.dev")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        config = build_guardrails_config(
            gateway_url="http://localhost:8080",
            api_key="key",
            llm_api_key="llm",
        )
    assert config.gateway_url == "http://localhost:8080"


# --- plaintext HTTP rejection ---


def test_validate_rejects_http_non_loopback():
    with pytest.raises(ValueError, match="must use https://"):
        _validate_guardrails_params("http://gw.example.com", "key")


def test_validate_http_loopback_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params("http://localhost:8080", "key")
    assert result == "http://localhost:8080"
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)
    assert "plaintext HTTP" in str(w[0].message)


def test_validate_http_127_0_0_1_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params("http://127.0.0.1:8080", "key")
    assert result == "http://127.0.0.1:8080"
    assert len(w) == 1
    assert issubclass(w[0].category, InsecureGatewayWarning)


def test_validate_http_ipv6_loopback_accepted():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _validate_guardrails_params("http://[::1]:8080", "key")
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
        )
    assert config.gateway_url == "http://localhost:8080"


def test_configure_rejects_http_non_loopback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(core, "_client", FlintAIClient(provider=None))
    with pytest.raises(ValueError, match="must use https://"):
        configure_guardrails(
            gateway_url="http://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )
