"""Tests for the ADK guardrails plugin."""

import sys
from unittest.mock import MagicMock, patch

import flintai_sdk
import pytest
from flintai_sdk.plugins.adk import ADKGuardrailsPlugin


def _mock_genai():
    """Mock google.genai.types so the plugin can import without real dependencies."""
    mock_types = MagicMock()

    def fake_config(**kwargs):
        obj = MagicMock()
        http_opts = kwargs.get("http_options")
        obj.http_options = http_opts
        return obj

    def fake_http_options(**kwargs):
        obj = MagicMock()
        obj.base_url = kwargs.get("base_url")
        obj.headers = kwargs.get("headers", {})
        return obj

    mock_types.GenerateContentConfig = fake_config
    mock_types.HttpOptions = fake_http_options
    return mock_types


@pytest.fixture
def mock_genai_modules():
    """Patch google.genai modules so ADKGuardrailsPlugin can import without real deps."""
    with patch.dict(
        sys.modules,
        {
            "google.genai.types": _mock_genai(),
            "google.genai": MagicMock(),
            "google": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def mock_adk_modules():
    """Patch google.genai + google.adk modules for on_model_error tests."""
    mock_types = MagicMock()
    mock_llm_response_module = MagicMock()
    mock_llm_response_module.LlmResponse = lambda **kwargs: MagicMock(**kwargs)
    with patch.dict(
        sys.modules,
        {
            "google.genai": MagicMock(),
            "google.genai.types": mock_types,
            "google": MagicMock(),
            "google.adk": MagicMock(),
            "google.adk.models": MagicMock(),
            "google.adk.models.llm_response": mock_llm_response_module,
        },
    ):
        yield


def test_adk_plugin_creates_content_config(mock_genai_modules):
    flintai_sdk.init(
        provider="google",
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    plugin = ADKGuardrailsPlugin()
    flintai_sdk.register_plugin(plugin)

    assert plugin.content_config is not None
    assert (
        plugin.content_config.http_options.base_url
        == "https://guardrails.example.com/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )
    assert plugin.content_config.http_options.headers["X-LLM-API-Key"] == "AIzaSy_test"


def test_adk_plugin_simplified_flow(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    assert plugin.content_config is not None
    assert (
        plugin.content_config.http_options.base_url
        == "https://guardrails.example.com/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )


def test_adk_plugin_merges_user_content_config(mock_genai_modules):
    user_config = MagicMock()
    user_config.temperature = 0.7
    user_config.top_p = 0.9

    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
        content_config=user_config,
    )
    assert plugin.content_config is user_config
    assert (
        plugin.content_config.http_options.base_url
        == "https://guardrails.example.com/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )
    assert plugin.content_config.temperature == 0.7
    assert plugin.content_config.top_p == 0.9


def test_adk_plugin_simplified_with_policy_id(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
        policy_id="pol-1",
    )
    assert (
        plugin.content_config.http_options.headers["X-Guardrails-Policy-Id"] == "pol-1"
    )


def test_adk_plugin_on_init_with_constructor_config(mock_genai_modules, caplog):
    import logging

    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    client = flintai_sdk.init(provider="google")
    with caplog.at_level(logging.INFO):
        flintai_sdk.register_plugin(plugin)
    assert client.guardrails_config is plugin._config
    assert "will route through" in caplog.text


def test_adk_plugin_partial_params_raises():
    with pytest.raises(
        ValueError, match="gateway_url, api_key, and llm_api_key are all required"
    ):
        ADKGuardrailsPlugin(gateway_url="https://gw.example.com")


def test_adk_plugin_no_guardrails_config(caplog):
    flintai_sdk.init(provider="google")
    plugin = ADKGuardrailsPlugin()
    flintai_sdk.register_plugin(plugin)

    assert plugin.content_config is None
    assert "No guardrails config found" in caplog.text


def test_adk_plugin_openai_provider(mock_genai_modules):
    flintai_sdk.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    plugin = ADKGuardrailsPlugin()
    flintai_sdk.register_plugin(plugin)

    assert (
        plugin.content_config.http_options.base_url == "https://gw.example.com/openai/"
    )


@pytest.mark.parametrize(
    "error_msg,status_code",
    [
        pytest.param("Request blocked by policy", None, id="blocked-msg"),
        pytest.param("guardrail violation detected", None, id="guardrail-msg"),
        pytest.param("request failed", 403, id="403-status"),
    ],
)
def test_on_model_error_blocked_variants(mock_adk_modules, error_msg, status_code):
    error = Exception(error_msg)
    if status_code is not None:
        error.status_code = status_code
    result = ADKGuardrailsPlugin.on_model_error(
        callback_context=MagicMock(),
        llm_request=MagicMock(),
        error=error,
    )
    assert result.error_code == "GUARDRAIL_BLOCKED"


def test_on_model_error_missing_deps_raises():
    with patch.dict(
        sys.modules,
        {
            "google.adk.models.llm_response": None,
            "google.genai": None,
            "google.genai.types": None,
        },
    ):
        with pytest.raises(
            RuntimeError, match="google-adk and google-genai are required"
        ):
            ADKGuardrailsPlugin.on_model_error(
                callback_context=MagicMock(),
                llm_request=MagicMock(),
                error=Exception("blocked by policy"),
            )


def test_on_model_error_non_blocked_reraises(mock_adk_modules):
    error = Exception("connection timeout")
    with pytest.raises(Exception, match="connection timeout"):
        ADKGuardrailsPlugin.on_model_error(
            callback_context=MagicMock(),
            llm_request=MagicMock(),
            error=error,
        )


# --- before_model_callback ---


def test_before_model_callback_injects_agent_id_and_session_id(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {"X-FlintAI-API-Key": "grl_sk_test"}

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None
    headers = llm_request.config.http_options.headers
    assert headers["X-Agent-Session-Id"] == "sess-abc-123"
    assert headers["X-Agent-Id"] == "search_agent"
    assert headers["X-FlintAI-API-Key"] == "grl_sk_test"


def test_before_model_callback_no_agent_name(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["session"])
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None
    assert "X-Agent-Id" not in llm_request.config.http_options.headers
    assert (
        llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-abc-123"
    )


def test_before_model_callback_static_agent_id_takes_precedence(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {
        "X-FlintAI-API-Key": "grl_sk_test",
        "X-Agent-Id": "static_override",
    }

    plugin.before_model_callback(ctx, llm_request)

    assert llm_request.config.http_options.headers["X-Agent-Id"] == "static_override"


def test_before_model_callback_no_session(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name"])
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None
    assert "X-Agent-Session-Id" not in llm_request.config.http_options.headers
    assert llm_request.config.http_options.headers["X-Agent-Id"] == "search_agent"


def test_before_model_callback_config_none(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config = None

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None


def test_before_model_callback_http_options_none(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options = None

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None


def test_before_model_callback_headers_none(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = None

    result = plugin.before_model_callback(ctx, llm_request)

    assert result is None
    headers = llm_request.config.http_options.headers
    assert headers["X-Agent-Session-Id"] == "sess-abc-123"
    assert headers["X-Agent-Id"] == "search_agent"


def test_before_model_callback_agent_id_env_var_takes_priority(
    mock_genai_modules, monkeypatch
):
    monkeypatch.setenv("AGENT_ID", "env-agent-id")
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-1"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    assert llm_request.config.http_options.headers["X-Agent-Id"] == "env-agent-id"


# --- Environment variable configuration ---


def test_adk_plugin_from_env_vars(mock_genai_modules, monkeypatch):
    monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://guardrails.env.com")
    monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
    monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
    plugin = ADKGuardrailsPlugin()
    assert plugin.content_config is not None
    assert (
        plugin.content_config.http_options.base_url
        == "https://guardrails.env.com/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "env-api-key"
    )
    assert plugin.content_config.http_options.headers["X-LLM-API-Key"] == "env-llm-key"
