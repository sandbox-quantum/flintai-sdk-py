"""Tests for switch_trust.wrap() — wrapping user-created LLM clients."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import switch_trust
from switch_trust import core
from switch_trust.core import SwitchTrustClient
from switch_trust.guardrails import GuardrailsConfig, SwitchTrustGuardrailsError
from switch_trust.plugins._llm_wrapper import (
    _apply_guardrails_config,
    _check_sdk_version,
    _clear_wrapped,
    _detect_client_type,
    _get_sdk_headers,
    _is_wrapped,
    _mark_wrapped,
    _merge_sdk_headers,
    _parse_version,
    _unwrap_langchain_model,
    wrap_client,
)


def _make_mock_client(module: str, class_name: str, provider: str = "openai"):
    """Create a fake client whose type appears to come from the given module."""
    cls = type(class_name, (), {"__module__": module})
    obj = cls()
    obj._base_url = None
    obj._custom_headers = {}
    if provider == "openai":
        chat = MagicMock()
        chat.completions.create.__name__ = "create"
        obj.chat = chat
    elif provider == "google":
        models = MagicMock()
        models.generate_content.__name__ = "generate_content"
        obj.models = models
        http_options = MagicMock()
        http_options.base_url = "https://generativelanguage.googleapis.com/"
        http_options.headers = {"Content-Type": "application/json"}
        api_client = MagicMock()
        api_client._http_options = http_options
        obj._api_client = api_client
    else:
        messages = MagicMock()
        messages.create.__name__ = "create"
        obj.messages = messages
    return obj


def _guardrails_config(provider="openai"):
    return GuardrailsConfig(
        base_url=f"https://gw.example.com/{provider}",
        headers={"X-FlintAI-API-Key": "grl_key", "X-LLM-API-Key": "sk-test"},
        provider=provider,
        gateway_url="https://gw.example.com",
    )


# --- _detect_client_type ---


@pytest.mark.parametrize(
    "module,class_name,provider,expected",
    [
        pytest.param(
            "anthropic._client", "Anthropic", "anthropic", "anthropic", id="anthropic"
        ),
        pytest.param("openai._client", "OpenAI", "openai", "openai", id="openai"),
        pytest.param("google.genai.client", "Client", "google", "google", id="google"),
    ],
)
def test_detect_sync_client(module, class_name, provider, expected):
    client = _make_mock_client(module, class_name, provider)
    assert _detect_client_type(client) == expected


@pytest.mark.parametrize(
    "module,class_name,provider",
    [
        pytest.param(
            "anthropic._client", "AsyncAnthropic", "anthropic", id="anthropic"
        ),
        pytest.param("openai._client", "AsyncOpenAI", "openai", id="openai"),
    ],
)
def test_detect_async_client_raises(module, class_name, provider):
    client = _make_mock_client(module, class_name, provider)
    with pytest.raises(TypeError, match="Async clients are not supported"):
        _detect_client_type(client)


def test_detect_unknown_raises():
    cls = type("Client", (), {"__module__": "cohere.client"})
    client = cls()
    with pytest.raises(TypeError, match="Unrecognized client type"):
        _detect_client_type(client)


# --- _apply_guardrails_config ---


def test_apply_guardrails_anthropic():
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"
    config = _guardrails_config("anthropic")

    _apply_guardrails_config(client, "anthropic", config)

    assert client._base_url == "https://gw.example.com/anthropic/"
    # A missing trailing slash makes httpx.URL.join() replace the last path
    # segment instead of appending after it, silently dropping the provider
    # prefix -- assert on the actual joined request URL, not just the stored
    # string, so that regression can't hide behind a string that "looks" right.
    assert (
        str(client._base_url.join("v1/messages"))
        == "https://gw.example.com/anthropic/v1/messages"
    )
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert client._custom_headers["X-LLM-API-Key"] == "sk-test"


def test_apply_guardrails_openai_property():
    """OpenAI client with a base_url property setter."""

    class FakeOpenAI:
        __module__ = "openai._client"
        __name__ = "OpenAI"

        def __init__(self):
            self._internal_url = "https://api.openai.com"
            self._custom_headers = {}

        @property
        def base_url(self):
            return self._internal_url

        @base_url.setter
        def base_url(self, val):
            self._internal_url = val

    client = FakeOpenAI()
    config = _guardrails_config("openai")

    _apply_guardrails_config(client, "openai", config)

    assert client.base_url == "https://gw.example.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"


def test_apply_guardrails_no_base_url_raises():
    cls = type("Anthropic", (), {"__module__": "anthropic._client"})
    client = cls()
    config = _guardrails_config("anthropic")

    with pytest.raises(TypeError, match="Cannot set base_url"):
        _apply_guardrails_config(client, "anthropic", config)


# --- wrap_client ---


def test_wrap_applies_guardrails_when_configured(monkeypatch):
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"

    sc = SwitchTrustClient(provider="anthropic")
    sc.guardrails_config = _guardrails_config("anthropic")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)

    assert client._base_url == "https://gw.example.com/anthropic/"
    assert (
        str(client._base_url.join("v1/messages"))
        == "https://gw.example.com/anthropic/v1/messages"
    )
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"


def test_wrap_client_raises_when_guardrails_required_but_no_config(monkeypatch):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"

    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider="openai"))

    with pytest.raises(SwitchTrustGuardrailsError, match="no config was found"):
        wrap_client(client)


def test_wrap_skips_guardrails_when_not_configured(monkeypatch):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"

    monkeypatch.setattr(
        core, "_client", SwitchTrustClient(provider="openai", require_guardrails=False)
    )

    wrap_client(client)

    assert client._base_url == "https://api.openai.com"
    assert client._custom_headers == {}


def test_wrap_double_wrap_noop(monkeypatch, caplog):
    client = _make_mock_client("openai._client", "OpenAI", "openai")

    monkeypatch.setattr(
        core, "_client", SwitchTrustClient(provider="openai", require_guardrails=False)
    )

    wrap_client(client)
    wrap_client(client)
    assert "already wrapped" in caplog.text


# --- switch_trust.wrap() public API ---


def test_wrap_raises_when_no_config():
    switch_trust.shutdown()
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    with pytest.raises(
        SwitchTrustGuardrailsError, match="Guardrails configuration is required"
    ):
        switch_trust.wrap(client)


def test_wrap_does_not_mutate_existing_client_require_guardrails(monkeypatch):
    switch_trust.shutdown()
    switch_trust.init(require_guardrails=False, provider="openai")
    assert core._client.require_guardrails is False

    client = _make_mock_client("openai._client", "OpenAI", "openai")
    with pytest.raises(SwitchTrustGuardrailsError):
        switch_trust.wrap(client, require_guardrails=True)

    assert core._client.require_guardrails is False


def test_wrap_auto_inits_when_no_client():
    switch_trust.shutdown()
    assert core._client is None
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    switch_trust.wrap(client, require_guardrails=False)
    assert core._client is not None


def test_wrap_auto_init_registers_atexit(monkeypatch):
    switch_trust.shutdown()
    monkeypatch.setattr(switch_trust, "_atexit_registered", False)
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    switch_trust.wrap(client, require_guardrails=False)
    assert core._client is not None
    assert switch_trust._atexit_registered is True


def test_wrap_with_guardrails_params():
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-test"
    switch_trust.wrap(
        client,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        forward_llm_key=True,
    )
    assert client._base_url == "https://gw.example.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert client._custom_headers["X-LLM-API-Key"] == "sk-test"


def test_wrap_auto_extracts_llm_api_key():
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"
    client.api_key = "sk-ant-key"
    switch_trust.wrap(
        client,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        forward_llm_key=True,
    )
    assert client._custom_headers["X-LLM-API-Key"] == "sk-ant-key"


def test_wrap_auto_extract_logs_warning(caplog):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-test"
    with caplog.at_level(logging.WARNING, logger="switch_trust"):
        switch_trust.wrap(
            client,
            gateway_url="https://gw.example.com",
            api_key="grl_key",
            forward_llm_key=True,
        )
    assert "llm_api_key auto-extracted" in caplog.text


def test_wrap_explicit_llm_key_no_warning(caplog):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-test"
    with caplog.at_level(logging.WARNING, logger="switch_trust"):
        switch_trust.wrap(
            client,
            gateway_url="https://gw.example.com",
            api_key="grl_key",
            llm_api_key="sk-explicit",
        )
    assert "llm_api_key auto-extracted" not in caplog.text


def test_wrap_does_not_forward_client_key_by_default():
    """Without forward_llm_key, a key present on the client is NOT forwarded."""
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-should-not-leak"
    switch_trust.wrap(client, gateway_url="https://gw.example.com", api_key="grl_key")
    assert "X-LLM-API-Key" not in client._custom_headers


def test_wrap_explicit_llm_api_key_wins():
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-from-client"
    switch_trust.wrap(
        client,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-explicit",
    )
    assert client._custom_headers["X-LLM-API-Key"] == "sk-explicit"


def test_wrap_no_api_key_on_client_succeeds_without_llm_key():
    """When the client has no api_key and llm_api_key is not provided, wrap succeeds
    and X-LLM-API-Key is not forwarded — the proxy uses its own upstream credentials."""
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(client, gateway_url="https://gw.example.com", api_key="grl_key")
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert "X-LLM-API-Key" not in client._custom_headers


def test_wrap_with_guardrails_params_auto_inits():
    switch_trust.shutdown()
    assert core._client is None
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-test"
    switch_trust.wrap(client, gateway_url="https://gw.example.com", api_key="grl_key")
    assert core._client is not None
    assert client._base_url == "https://gw.example.com/openai"


def test_wrap_partial_guardrails_params_raises():
    with pytest.raises(ValueError, match="gateway_url and api_key are both required"):
        switch_trust.wrap(
            _make_mock_client("openai._client", "OpenAI", "openai"),
            gateway_url="https://gw.example.com",
        )


def test_wrap_unknown_client_raises():
    switch_trust.init(require_guardrails=False)
    with pytest.raises(TypeError, match="Unrecognized client type"):
        switch_trust.wrap({"not": "a client"}, require_guardrails=False)


def test_wrap_preserves_user_api_key(monkeypatch):
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client.api_key = "sk-real-key"
    client._base_url = "https://api.anthropic.com"

    sc = SwitchTrustClient(provider="anthropic")
    sc.guardrails_config = _guardrails_config("anthropic")
    monkeypatch.setattr(core, "_client", sc)

    switch_trust.wrap(client)

    assert client.api_key == "sk-real-key"


# --- Google/Gemini ---


def test_wrap_google_applies_guardrails(monkeypatch):
    client = _make_mock_client("google.genai.client", "Client", "google")

    sc = SwitchTrustClient(provider="google")
    sc.guardrails_config = _guardrails_config("google")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)

    assert client._api_client._http_options.base_url == "https://gw.example.com/google/"
    assert client._api_client._http_options.headers["X-FlintAI-API-Key"] == "grl_key"


# --- Deferred provider ---


def _guardrails_config_deferred():
    return GuardrailsConfig(
        base_url="https://gw.example.com",
        headers={"X-FlintAI-API-Key": "grl_key", "X-LLM-API-Key": "sk-test"},
        provider=None,
        gateway_url="https://gw.example.com",
    )


def _get_effective_url(client, provider):
    """Extract the effective base URL from a wrapped client."""
    if provider == "google":
        return client._api_client._http_options.base_url
    return client._base_url


@pytest.mark.parametrize(
    "module,class_name,provider,expected_url",
    [
        pytest.param(
            "openai._client",
            "OpenAI",
            "openai",
            "https://gw.example.com/openai",
            id="openai",
        ),
        pytest.param(
            "anthropic._client",
            "Anthropic",
            "anthropic",
            "https://gw.example.com/anthropic/",
            id="anthropic",
        ),
        pytest.param(
            "google.genai.client",
            "Client",
            "google",
            "https://gw.example.com/gemini/",
            id="google",
        ),
    ],
)
def test_wrap_with_deferred_provider_resolves_url(
    monkeypatch, module, class_name, provider, expected_url
):
    client = _make_mock_client(module, class_name, provider)
    if provider != "google":
        client._base_url = f"https://api.{provider}.com"

    sc = SwitchTrustClient(provider=None)
    sc.guardrails_config = _guardrails_config_deferred()
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)

    assert _get_effective_url(client, provider) == expected_url


# --- _apply_guardrails_config edge cases ---


def test_apply_guardrails_anthropic_without_httpx(monkeypatch):
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"
    config = _guardrails_config("anthropic")

    monkeypatch.setitem(sys.modules, "httpx", None)
    _apply_guardrails_config(client, "anthropic", config)

    assert client._base_url == "https://gw.example.com/anthropic/"
    assert isinstance(client._base_url, str)


def test_apply_guardrails_openai_no_base_url_raises():
    cls = type("OpenAI", (), {"__module__": "openai._client"})
    client = cls()
    config = _guardrails_config("openai")
    with pytest.raises(TypeError, match="no known attribute found"):
        _apply_guardrails_config(client, "openai", config)


def test_apply_guardrails_google_no_api_client_raises():
    cls = type("Client", (), {"__module__": "google.genai.client"})
    client = cls()
    config = _guardrails_config("google")
    with pytest.raises(TypeError, match="_api_client attribute not found"):
        _apply_guardrails_config(client, "google", config)


def test_apply_guardrails_google_no_http_options_raises():
    cls = type("Client", (), {"__module__": "google.genai.client"})
    client = cls()
    api_client = MagicMock(spec=[])
    client._api_client = api_client
    config = _guardrails_config("google")
    with pytest.raises(TypeError, match="_http_options attribute not found"):
        _apply_guardrails_config(client, "google", config)


def test_apply_guardrails_throws_missing_custom_headers():
    cls = type("Anthropic", (), {"__module__": "anthropic._client"})
    client = cls()
    client._base_url = "https://api.anthropic.com"
    config = _guardrails_config("anthropic")
    with pytest.raises(TypeError, match="Cannot set custom headers"):
        _apply_guardrails_config(client, "anthropic", config)


# --- LangChain wrapping ---


def _make_langchain_model(
    langchain_module: str, class_name: str, inner_attr: str, inner_provider: str
):
    """Create a fake LangChain chat model with an underlying SDK client."""
    inner_client = _make_mock_client(
        {
            "openai": "openai._client",
            "anthropic": "anthropic._client",
            "google": "google.genai.client",
        }[inner_provider],
        {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Client"}[
            inner_provider
        ],
        inner_provider,
    )
    inner_client._base_url = f"https://api.{inner_provider}.com"
    inner_client._custom_headers = {}
    if inner_provider == "openai":
        inner_client.api_key = "sk-openai-key"
    elif inner_provider == "anthropic":
        inner_client.api_key = "sk-ant-key"

    cls = type(class_name, (), {"__module__": langchain_module})
    model = cls()
    setattr(model, inner_attr, inner_client)
    return model, inner_client


@pytest.mark.parametrize(
    "langchain_module,class_name,inner_attr,inner_provider",
    [
        pytest.param(
            "langchain_openai.chat_models",
            "ChatOpenAI",
            "root_client",
            "openai",
            id="openai",
        ),
        pytest.param(
            "langchain_anthropic._chat_models",
            "ChatAnthropic",
            "_client",
            "anthropic",
            id="anthropic",
        ),
        pytest.param(
            "langchain_google_genai._chat_models",
            "ChatGoogleGenerativeAI",
            "client",
            "google",
            id="google",
        ),
    ],
)
def test_unwrap_langchain_model(
    langchain_module, class_name, inner_attr, inner_provider
):
    model, inner_client = _make_langchain_model(
        langchain_module, class_name, inner_attr, inner_provider
    )
    result = _unwrap_langchain_model(model)
    assert result is not None
    sdk_client, provider = result
    assert sdk_client is inner_client
    assert provider == inner_provider


def test_unwrap_langchain_returns_none_for_non_langchain():
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    assert _unwrap_langchain_model(client) is None


def test_unwrap_langchain_missing_attr_raises():
    cls = type("ChatOpenAI", (), {"__module__": "langchain_openai.chat_models"})
    model = cls()
    with pytest.raises(TypeError, match="attribute 'root_client' not found"):
        _unwrap_langchain_model(model)


# --- _get_sdk_headers ---


def test_get_sdk_headers_openai():
    client = MagicMock()
    client._custom_headers = {"existing": "header"}
    assert _get_sdk_headers(client, "openai") is client._custom_headers


def test_get_sdk_headers_anthropic():
    client = MagicMock()
    client._custom_headers = {}
    assert _get_sdk_headers(client, "anthropic") is client._custom_headers


def test_get_sdk_headers_google():
    client = MagicMock()
    client._api_client._http_options.headers = {"X-API-Key": "test"}
    assert (
        _get_sdk_headers(client, "google") is client._api_client._http_options.headers
    )


def test_get_sdk_headers_unknown_provider():
    assert _get_sdk_headers(MagicMock(), "unknown") is None


@pytest.mark.parametrize(
    "langchain_module,class_name,inner_attr,inner_provider",
    [
        pytest.param(
            "langchain_openai.chat_models",
            "ChatOpenAI",
            "root_client",
            "openai",
            id="openai",
        ),
        pytest.param(
            "langchain_anthropic._chat_models",
            "ChatAnthropic",
            "_client",
            "anthropic",
            id="anthropic",
        ),
        pytest.param(
            "langchain_google_genai._chat_models",
            "ChatGoogleGenerativeAI",
            "client",
            "google",
            id="google",
        ),
    ],
)
def test_wrap_langchain_applies_guardrails(
    monkeypatch, langchain_module, class_name, inner_attr, inner_provider
):
    model, inner_client = _make_langchain_model(
        langchain_module, class_name, inner_attr, inner_provider
    )

    sc = SwitchTrustClient(provider=inner_provider)
    sc.guardrails_config = _guardrails_config(inner_provider)
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(model)

    expected_url = f"https://gw.example.com/{inner_provider}"
    if inner_provider in ("google", "anthropic"):
        expected_url += "/"
    if inner_provider == "google":
        assert inner_client._api_client._http_options.base_url == expected_url
        assert (
            inner_client._api_client._http_options.headers["X-FlintAI-API-Key"]
            == "grl_key"
        )
    else:
        assert inner_client._base_url == expected_url
        assert inner_client._custom_headers["X-FlintAI-API-Key"] == "grl_key"


def test_wrap_langchain_double_wrap_noop(monkeypatch, caplog):
    model, _ = _make_langchain_model(
        "langchain_openai.chat_models", "ChatOpenAI", "root_client", "openai"
    )
    monkeypatch.setattr(
        core, "_client", SwitchTrustClient(provider="openai", require_guardrails=False)
    )

    wrap_client(model)
    wrap_client(model)
    assert "already wrapped" in caplog.text


def test_wrap_langchain_auto_extracts_api_key():
    model, inner_client = _make_langchain_model(
        "langchain_openai.chat_models",
        "ChatOpenAI",
        "root_client",
        "openai",
    )
    inner_client._base_url = "https://api.openai.com"
    inner_client.api_key = "sk-from-langchain"

    switch_trust.wrap(
        model,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        forward_llm_key=True,
    )
    assert inner_client._custom_headers["X-LLM-API-Key"] == "sk-from-langchain"


def test_wrap_langchain_with_policy_id():
    model, inner_client = _make_langchain_model(
        "langchain_openai.chat_models",
        "ChatOpenAI",
        "root_client",
        "openai",
    )
    inner_client._base_url = "https://api.openai.com"
    inner_client.api_key = "sk-test"

    switch_trust.wrap(
        model,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        policy_id="pol-1",
    )
    assert inner_client._custom_headers["X-Guardrails-Policy-Id"] == "pol-1"


# --- ADK agent rejection ---


def test_detect_adk_agent_raises():
    cls = type("LlmAgent", (), {"__module__": "google.adk.agents"})
    agent = cls()
    with pytest.raises(TypeError, match="ADK agents cannot be wrapped"):
        _detect_client_type(agent)


def test_shutdown_clears_wrapped_tracking(monkeypatch):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    sc = SwitchTrustClient(provider="openai", require_guardrails=False)
    monkeypatch.setattr(core, "_client", sc)
    wrap_client(client)
    assert _is_wrapped(client)

    switch_trust.shutdown()

    monkeypatch.setattr(
        core, "_client", SwitchTrustClient(provider="openai", require_guardrails=False)
    )
    assert not _is_wrapped(client)


def test_shutdown_scrubs_anthropic_client(monkeypatch):
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"

    sc = SwitchTrustClient(provider="anthropic")
    sc.guardrails_config = _guardrails_config("anthropic")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert client._custom_headers["X-LLM-API-Key"] == "sk-test"

    switch_trust.shutdown()

    assert "X-FlintAI-API-Key" not in client._custom_headers
    assert "X-LLM-API-Key" not in client._custom_headers
    assert client._base_url == "https://api.anthropic.com"


def test_shutdown_scrubs_openai_client(monkeypatch):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"

    sc = SwitchTrustClient(provider="openai")
    sc.guardrails_config = _guardrails_config("openai")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"

    switch_trust.shutdown()

    assert "X-FlintAI-API-Key" not in client._custom_headers
    assert "X-LLM-API-Key" not in client._custom_headers
    assert client._base_url == "https://api.openai.com"


def test_shutdown_scrubs_google_client(monkeypatch):
    client = _make_mock_client("google.genai.client", "Client", "google")
    orig_url = client._api_client._http_options.base_url

    sc = SwitchTrustClient(provider="google")
    sc.guardrails_config = _guardrails_config("google")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    assert client._api_client._http_options.headers["X-FlintAI-API-Key"] == "grl_key"

    switch_trust.shutdown()

    assert "X-FlintAI-API-Key" not in client._api_client._http_options.headers
    assert "X-LLM-API-Key" not in client._api_client._http_options.headers
    assert client._api_client._http_options.base_url == orig_url


def test_shutdown_scrubs_langchain_inner_client(monkeypatch):
    model, inner_client = _make_langchain_model(
        "langchain_anthropic._chat_models", "ChatAnthropic", "_client", "anthropic"
    )
    inner_client._base_url = "https://api.anthropic.com"

    sc = SwitchTrustClient(provider="anthropic")
    sc.guardrails_config = _guardrails_config("anthropic")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(model)
    assert inner_client._custom_headers["X-FlintAI-API-Key"] == "grl_key"

    switch_trust.shutdown()

    assert "X-FlintAI-API-Key" not in inner_client._custom_headers
    assert "X-LLM-API-Key" not in inner_client._custom_headers
    assert inner_client._base_url == "https://api.anthropic.com"


# --- Restoration failure handling (fault injection) ---


def _make_faulty_openai_client():
    """OpenAI client whose base_url setter can be flipped to raise during restore."""

    class FaultyOpenAI:
        __module__ = "openai._client"
        __name__ = "OpenAI"

        def __init__(self):
            self._internal_url = "https://api.openai.com"
            self._custom_headers = {}
            self._fault = False

        @property
        def base_url(self):
            return self._internal_url

        @base_url.setter
        def base_url(self, val):
            if self._fault:
                raise RuntimeError(
                    "restore blew up referencing "
                    "https://gw.example.com/openai and grl_key"
                )
            self._internal_url = val

    return FaultyOpenAI()


def test_shutdown_restore_failure_logs_error_without_secrets(monkeypatch, caplog):
    client = _make_faulty_openai_client()
    sc = SwitchTrustClient(provider="openai")
    sc.guardrails_config = _guardrails_config("openai")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    assert _is_wrapped(client)

    client._fault = True
    with caplog.at_level(logging.ERROR, logger="switch_trust"):
        _clear_wrapped()

    assert "failed to restore" in caplog.text
    assert "openai" in caplog.text
    assert "OpenAI" in caplog.text
    # No secrets in the log.
    assert "gw.example.com" not in caplog.text
    assert "grl_key" not in caplog.text
    assert "api.openai.com" not in caplog.text
    assert "blew up" not in caplog.text

    client._fault = False


def test_shutdown_restore_failure_preserves_state_for_retry(monkeypatch, caplog):
    client = _make_faulty_openai_client()
    sc = SwitchTrustClient(provider="openai")
    sc.guardrails_config = _guardrails_config("openai")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    client._fault = True
    with caplog.at_level(logging.ERROR, logger="switch_trust"):
        _clear_wrapped()

    # Restore raised before headers were scrubbed: still gateway-routed, still
    # tracked for retry.
    assert client.base_url == "https://gw.example.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert _is_wrapped(client)

    client._fault = False


def test_shutdown_restore_retries_and_clears_once_successful(monkeypatch, caplog):
    client = _make_faulty_openai_client()
    sc = SwitchTrustClient(provider="openai")
    sc.guardrails_config = _guardrails_config("openai")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(client)
    client._fault = True
    with caplog.at_level(logging.ERROR, logger="switch_trust"):
        _clear_wrapped()
    assert _is_wrapped(client)

    client._fault = False
    _clear_wrapped()

    assert client.base_url == "https://api.openai.com"
    assert "X-FlintAI-API-Key" not in client._custom_headers
    assert not _is_wrapped(client)


def test_shutdown_restore_failure_does_not_block_other_clients(monkeypatch, caplog):
    faulty = _make_faulty_openai_client()
    healthy = _make_mock_client("openai._client", "OpenAI", "openai")
    healthy._base_url = "https://api.openai.com"
    sc = SwitchTrustClient(provider="openai")
    sc.guardrails_config = _guardrails_config("openai")
    monkeypatch.setattr(core, "_client", sc)

    wrap_client(faulty)
    wrap_client(healthy)
    faulty._fault = True
    with caplog.at_level(logging.ERROR, logger="switch_trust"):
        _clear_wrapped()

    # Healthy client fully restored and untracked; faulty one retained.
    assert healthy._base_url == "https://api.openai.com"
    assert "X-FlintAI-API-Key" not in healthy._custom_headers
    assert not _is_wrapped(healthy)
    assert _is_wrapped(faulty)

    faulty._fault = False


# --- Unhashable client tracking ---


def _make_unhashable_client(module: str, class_name: str):
    """Create an unhashable client whose type appears to come from the given module."""
    cls = type(class_name, (), {"__hash__": None, "__module__": module})
    obj = cls()
    obj._base_url = "https://api.openai.com"
    obj._custom_headers = {}
    return obj


def test_unhashable_client_is_wrapped_false_before_mark():
    _clear_wrapped()
    client = _make_unhashable_client("openai._client", "OpenAI")
    assert not _is_wrapped(client)


def test_unhashable_client_mark_and_check():
    _clear_wrapped()
    client = _make_unhashable_client("openai._client", "OpenAI")
    _mark_wrapped(client)
    assert _is_wrapped(client)


def test_unhashable_client_expired_ref():
    import weakref

    _clear_wrapped()
    client = _make_unhashable_client("openai._client", "OpenAI")
    _mark_wrapped(client)

    from switch_trust.plugins._llm_wrapper import _wrapped_client_id_refs

    client_id = id(client)
    assert client_id in _wrapped_client_id_refs
    _wrapped_client_id_refs[client_id] = weakref.ref(type("Dead", (), {})())
    assert not _is_wrapped(client)


def test_unweakrefable_client_mark_is_noop():
    _clear_wrapped()

    class NoWeakRef:
        __hash__ = None
        __slots__ = ()

    client = NoWeakRef()
    _mark_wrapped(client)
    assert not _is_wrapped(client)


# --- _apply_guardrails_config edge cases ---


def test_apply_guardrails_unknown_provider_raises():
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    config = _guardrails_config("openai")
    with pytest.raises(TypeError, match="No guardrails applier for provider"):
        _apply_guardrails_config(client, "cohere", config)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("httpx"),
    reason="httpx not installed",
)
def test_apply_guardrails_anthropic_with_httpx():
    client = _make_mock_client("anthropic._client", "Anthropic", "anthropic")
    client._base_url = "https://api.anthropic.com"
    config = _guardrails_config("anthropic")

    _apply_guardrails_config(client, "anthropic", config)

    from httpx import URL

    assert isinstance(client._base_url, URL)
    assert str(client._base_url) == "https://gw.example.com/anthropic/"


# --- Environment variable configuration ---


def test_wrap_from_env_vars(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(client)
    assert client._base_url == "https://gw.env.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "env-api-key"
    assert client._custom_headers["X-LLM-API-Key"] == "env-llm-key"


def _wrap_target(kind):
    """Return (object to wrap, SDK client that receives the headers, provider)."""
    if kind == "chat_openai":
        model, inner_client = _make_langchain_model(
            "langchain_openai.chat_models", "ChatOpenAI", "root_client", "openai"
        )
        return model, inner_client, "openai"
    module, class_name, provider = {
        "openai": ("openai._client", "OpenAI", "openai"),
        "anthropic": ("anthropic._client", "Anthropic", "anthropic"),
        "google": ("google.genai.client", "Client", "google"),
    }[kind]
    client = _make_mock_client(module, class_name, provider)
    if provider != "google":
        client._base_url = f"https://api.{provider}.com"
    return client, client, provider


@pytest.mark.parametrize("kind", ["openai", "anthropic", "google", "chat_openai"])
def test_wrap_sends_agent_identity_from_env(monkeypatch, kind):
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("AGENT_ID", "agn_123")
    monkeypatch.setenv("AGENT_NAME", "env-agent")
    target, sdk_client, provider = _wrap_target(kind)
    switch_trust.wrap(target)
    headers = _get_sdk_headers(sdk_client, provider)
    assert headers["X-Agent-Id"] == "agn_123"
    assert headers["X-Agent-Name"] == "env-agent"


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_wrap_omits_agent_identity_when_env_unset(monkeypatch, value):
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    if value is not None:
        monkeypatch.setenv("AGENT_ID", value)
        monkeypatch.setenv("AGENT_NAME", value)
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(client)
    assert "X-Agent-Id" not in client._custom_headers
    assert "X-Agent-Name" not in client._custom_headers


def test_wrap_skips_non_ascii_agent_name(monkeypatch):
    """Every header wrap() sets must be ASCII, the encoding httpx uses."""
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("AGENT_ID", "agn_123")
    monkeypatch.setenv("AGENT_NAME", "café-bot")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(client)
    assert "X-Agent-Name" not in client._custom_headers
    assert client._custom_headers["X-Agent-Id"] == "agn_123"
    for value in client._custom_headers.values():
        value.encode("ascii")


def test_wrap_keeps_caller_agent_identity_when_env_unset(monkeypatch):
    """Identity headers the caller set in default_headers survive wrap().

    A caller can name the agent this way instead of through the environment.
    """
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client._custom_headers.update(
        {"X-Agent-Id": "agn_manual", "X-Agent-Name": "manual-agent"}
    )
    switch_trust.wrap(client)
    assert client._custom_headers["X-Agent-Id"] == "agn_manual"
    assert client._custom_headers["X-Agent-Name"] == "manual-agent"


@pytest.mark.parametrize("header", ["X-Agent-Name", "x-agent-name"])
def test_wrap_caller_agent_name_beats_env_and_survives_shutdown(monkeypatch, header):
    """An X-Agent-Name from default_headers wins over AGENT_NAME, in either letter case.

    shutdown() removes only the headers the SDK added, so the caller's header stays.
    """
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("AGENT_ID", "agn_env")
    monkeypatch.setenv("AGENT_NAME", "from-env")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client._custom_headers[header] = "from-default-headers"
    switch_trust.wrap(client)
    names = {
        k: v for k, v in client._custom_headers.items() if k.lower() == "x-agent-name"
    }
    assert names == {header: "from-default-headers"}
    assert client._custom_headers["X-Agent-Id"] == "agn_env"

    switch_trust.shutdown()

    assert client._custom_headers[header] == "from-default-headers"
    assert "X-Agent-Id" not in client._custom_headers


def test_shutdown_removes_env_agent_identity(monkeypatch):
    monkeypatch.setenv("SWITCH_TRUST_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("SWITCH_TRUST_API_KEY", "env-api-key")
    monkeypatch.setenv("AGENT_ID", "agn_123")
    monkeypatch.setenv("AGENT_NAME", "env-agent")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(client)

    switch_trust.shutdown()

    assert "X-Agent-Id" not in client._custom_headers
    assert "X-Agent-Name" not in client._custom_headers


def test_wrap_env_llm_key_beats_client_auto_extract(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-from-client"
    switch_trust.wrap(client, forward_llm_key=True)
    assert client._custom_headers["X-LLM-API-Key"] == "env-llm-key"


def test_wrap_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(
        client,
        gateway_url="https://explicit.com",
        api_key="explicit-key",
        llm_api_key="explicit-llm-key",
    )
    assert client._base_url == "https://explicit.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "explicit-key"
    assert client._custom_headers["X-LLM-API-Key"] == "explicit-llm-key"


def test_wrap_auto_extract_when_no_env_llm_key(monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://gw.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    client.api_key = "sk-from-client"
    switch_trust.wrap(client, forward_llm_key=True)
    assert client._custom_headers["X-LLM-API-Key"] == "sk-from-client"


def test_wrap_without_llm_api_key():
    """Wrapping with gateway_url and api_key but no llm_api_key succeeds.
    X-LLM-API-Key is not sent — the proxy uses its own upstream credentials."""
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    client._base_url = "https://api.openai.com"
    switch_trust.wrap(
        client,
        gateway_url="https://gw.example.com",
        api_key="grl_key",
    )
    assert client._base_url == "https://gw.example.com/openai"
    assert client._custom_headers["X-FlintAI-API-Key"] == "grl_key"
    assert "X-LLM-API-Key" not in client._custom_headers


# --- require_guardrails enforcement ---


def test_merge_sdk_headers_raises_when_required():
    cls = type("Anthropic", (), {"__module__": "anthropic._client"})
    client = cls()
    client._base_url = "https://api.anthropic.com"
    with pytest.raises(
        SwitchTrustGuardrailsError, match="Cannot set guardrails headers"
    ):
        _merge_sdk_headers(
            client, "anthropic", {"X-FlintAI-API-Key": "key"}, require_guardrails=True
        )


def test_merge_sdk_headers_raises_when_not_required():
    cls = type("Anthropic", (), {"__module__": "anthropic._client"})
    client = cls()
    client._base_url = "https://api.anthropic.com"
    with pytest.raises(TypeError, match="Cannot set custom headers"):
        _merge_sdk_headers(
            client,
            "anthropic",
            {"X-FlintAI-API-Key": "key"},
            require_guardrails=False,
        )


def test_wrap_langchain_client_raises_when_no_config_required(monkeypatch):
    model, _ = _make_langchain_model(
        "langchain_openai.chat_models", "ChatOpenAI", "root_client", "openai"
    )
    monkeypatch.setattr(core, "_client", SwitchTrustClient(provider="openai"))

    with pytest.raises(SwitchTrustGuardrailsError, match="no config was found"):
        wrap_client(model)


# --- _parse_version ---


@pytest.mark.parametrize(
    "version_str,expected",
    [
        pytest.param("2.40.0", (2, 40, 0), id="semver"),
        pytest.param("0.105.2", (0, 105, 2), id="pre-1.0"),
        pytest.param("3", (3,), id="major-only"),
        pytest.param("2.7", (2, 7), id="major-minor"),
        pytest.param("1.0.0rc1", (1, 0, 0), id="prerelease-stripped"),
        pytest.param("abc", None, id="garbage"),
    ],
)
def test_parse_version(version_str, expected):
    assert _parse_version(version_str) == expected


# --- _check_sdk_version ---


def test_check_sdk_version_within_range(caplog):
    with patch("importlib.metadata.version", return_value="2.45.0") as mock_ver:
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("openai")
        mock_ver.assert_called_once_with("openai")
    assert "compatibility" not in caplog.text.lower()


def test_check_sdk_version_too_old(caplog):
    with patch("importlib.metadata.version", return_value="2.30.0"):
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("openai")
    assert "Private API compatibility is not guaranteed" in caplog.text
    assert "openai 2.30.0" in caplog.text


def test_check_sdk_version_too_new(caplog):
    with patch("importlib.metadata.version", return_value="3.0.0"):
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("openai")
    assert "Private API compatibility is not guaranteed" in caplog.text
    assert "openai 3.0.0" in caplog.text


def test_check_sdk_version_package_not_found(caplog):
    from importlib.metadata import PackageNotFoundError

    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("openai"),
    ):
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("openai")
    assert caplog.text == ""


def test_check_sdk_version_unknown_provider(caplog):
    with caplog.at_level(logging.WARNING):
        _check_sdk_version("cohere")
    assert caplog.text == ""


def test_check_sdk_version_anthropic_within_range(caplog):
    with patch("importlib.metadata.version", return_value="0.110.0"):
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("anthropic")
    assert "compatibility" not in caplog.text.lower()


def test_check_sdk_version_anthropic_too_new(caplog):
    with patch("importlib.metadata.version", return_value="1.0.0"):
        with caplog.at_level(logging.WARNING):
            _check_sdk_version("anthropic")
    assert "Private API compatibility is not guaranteed" in caplog.text


_OPT_OUT_MARKER = "SECURITY CONTROL DISABLED"


def _opt_out_records(caplog):
    return [r for r in caplog.records if _OPT_OUT_MARKER in r.getMessage()]


@pytest.mark.parametrize(
    "module,class_name,provider",
    [
        pytest.param("openai._client", "OpenAI", "openai", id="openai"),
        pytest.param("anthropic._client", "Anthropic", "anthropic", id="anthropic"),
        pytest.param("google.genai.client", "Client", "google", id="google"),
    ],
)
def test_wrap_opt_out_emits_audit_warning(module, class_name, provider, caplog):
    client = _make_mock_client(module, class_name, provider)
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.wrap(client, require_guardrails=False)
    assert len(_opt_out_records(caplog)) == 1


def test_wrap_dangerously_disable_emits_warning(caplog):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.wrap(client, dangerously_disable_guardrails=True)
    assert len(_opt_out_records(caplog)) == 1


def test_wrap_dangerously_disable_conflict_raises():
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    with pytest.raises(ValueError, match="Conflicting options"):
        switch_trust.wrap(
            client, dangerously_disable_guardrails=True, require_guardrails=True
        )


def test_wrap_guarded_does_not_warn(caplog):
    client = _make_mock_client("openai._client", "OpenAI", "openai")
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.wrap(
            client,
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )
    assert _opt_out_records(caplog) == []
