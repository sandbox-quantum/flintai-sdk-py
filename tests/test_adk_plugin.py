"""Tests for the ADK guardrails plugin."""

import sys
from unittest.mock import MagicMock, patch

import flintai
import pytest
from flintai.guardrails import FlintAIGuardrailsError, InsecureGatewayWarning
from flintai.plugins.adk import ADKGuardrailsPlugin


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
    flintai.init(
        provider="google",
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    plugin = ADKGuardrailsPlugin()
    flintai.register_plugin(plugin)

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
    client = flintai.init(provider="google", require_guardrails=False)
    with caplog.at_level(logging.INFO):
        flintai.register_plugin(plugin)
    assert client.guardrails_config is plugin._config
    assert "will route through" in caplog.text


def test_adk_plugin_partial_params_raises():
    with pytest.raises(ValueError, match="gateway_url and api_key are both required"):
        ADKGuardrailsPlugin(gateway_url="https://gw.example.com")


def test_adk_plugin_no_guardrails_config(caplog):
    flintai.init(provider="google", require_guardrails=False)
    plugin = ADKGuardrailsPlugin()
    flintai.register_plugin(plugin)

    assert plugin.content_config is None
    assert "No guardrails config found" in caplog.text


def test_adk_plugin_no_config_raises_when_required():
    flintai.init(provider="google", require_guardrails=False)
    plugin = ADKGuardrailsPlugin(require_guardrails=True)
    with pytest.raises(FlintAIGuardrailsError, match="No guardrails config found"):
        flintai.register_plugin(plugin)


def test_adk_plugin_inherits_require_guardrails_from_client():
    client = flintai.init(provider="google", require_guardrails=False)
    client.require_guardrails = True
    plugin = ADKGuardrailsPlugin()
    with pytest.raises(FlintAIGuardrailsError, match="No guardrails config found"):
        flintai.register_plugin(plugin)


def test_adk_plugin_explicit_override_honored_over_client(caplog):
    client = flintai.init(provider="google", require_guardrails=False)
    client.require_guardrails = True
    plugin = ADKGuardrailsPlugin(require_guardrails=False)
    flintai.register_plugin(plugin)
    assert "No guardrails config found" in caplog.text


def test_adk_plugin_openai_provider(mock_genai_modules):
    flintai.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    plugin = ADKGuardrailsPlugin()
    flintai.register_plugin(plugin)

    assert (
        plugin.content_config.http_options.base_url == "https://gw.example.com/openai/"
    )


def test_on_model_error_structured_code_in_details(mock_adk_modules):
    """Detects block via error.details (google-genai APIError production path)."""
    error = Exception("400 None. {'code': 'GUARDRAIL_BLOCKED'}")
    error.details = {
        "error": "Blocked by SandboxAQ Guardrail Service",
        "code": "GUARDRAIL_BLOCKED",
        "policy_id": "pol-123",
        "policy_name": "Test Policy",
        "findings": [
            {"category": "pii", "severity": "High", "message": "Sensitive data"}
        ],
    }
    result = ADKGuardrailsPlugin.on_model_error(
        callback_context=MagicMock(),
        llm_request=MagicMock(),
        error=error,
    )
    assert result.error_code == "GUARDRAIL_BLOCKED"
    assert result.custom_metadata is not None
    assert result.custom_metadata["policy_id"] == "pol-123"
    assert result.custom_metadata["policy_name"] == "Test Policy"
    assert len(result.custom_metadata["findings"]) == 1
    assert result.custom_metadata["findings"][0]["category"] == "pii"


def test_on_model_error_structured_code_in_body(mock_adk_modules):
    """Detects block via error.body (fallback for non-google-genai errors)."""
    error = Exception("400 Bad Request")
    error.body = {
        "error": "Blocked by SandboxAQ Guardrail Service",
        "code": "GUARDRAIL_BLOCKED",
        "policy_id": "pol-456",
        "policy_name": "Another Policy",
    }
    result = ADKGuardrailsPlugin.on_model_error(
        callback_context=MagicMock(),
        llm_request=MagicMock(),
        error=error,
    )
    assert result.error_code == "GUARDRAIL_BLOCKED"
    assert result.custom_metadata["policy_id"] == "pol-456"


def test_on_model_error_details_takes_priority_over_body(mock_adk_modules):
    """error.details is checked before error.body."""
    error = Exception("400 Bad Request")
    error.details = {"code": "GUARDRAIL_BLOCKED", "policy_id": "from-details"}
    error.body = {"code": "GUARDRAIL_BLOCKED", "policy_id": "from-body"}
    result = ADKGuardrailsPlugin.on_model_error(
        callback_context=MagicMock(),
        llm_request=MagicMock(),
        error=error,
    )
    assert result.custom_metadata["policy_id"] == "from-details"


def test_on_model_error_structured_code_in_string(mock_adk_modules):
    """String fallback works but produces no custom_metadata."""
    error = Exception(
        '{"error": "Blocked by SandboxAQ Guardrail Service", "code": "GUARDRAIL_BLOCKED"}'
    )
    result = ADKGuardrailsPlugin.on_model_error(
        callback_context=MagicMock(),
        llm_request=MagicMock(),
        error=error,
    )
    assert result.error_code == "GUARDRAIL_BLOCKED"
    assert result.custom_metadata is None


def test_on_model_error_403_no_longer_matches(mock_adk_modules):
    """A 403 auth error without GUARDRAIL_BLOCKED code is re-raised, not swallowed."""
    error = Exception("invalid API key")
    error.status_code = 403
    with pytest.raises(Exception, match="invalid API key"):
        ADKGuardrailsPlugin.on_model_error(
            callback_context=MagicMock(),
            llm_request=MagicMock(),
            error=error,
        )


def test_on_model_error_generic_blocked_keyword_no_longer_matches(mock_adk_modules):
    """Generic 'blocked' in error message without structured code is re-raised."""
    error = Exception("connection blocked by firewall")
    with pytest.raises(Exception, match="connection blocked by firewall"):
        ADKGuardrailsPlugin.on_model_error(
            callback_context=MagicMock(),
            llm_request=MagicMock(),
            error=error,
        )


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
                error=Exception("some error"),
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


def test_before_model_callback_injects_agent_name_and_session_id(mock_genai_modules):
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
    assert headers["X-Agent-Name"] == "search_agent"
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
    assert "X-Agent-Name" not in llm_request.config.http_options.headers
    assert (
        llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-abc-123"
    )


def test_before_model_callback_static_agent_name_takes_precedence(mock_genai_modules):
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
        "X-Agent-Name": "static_override",
    }

    plugin.before_model_callback(ctx, llm_request)

    assert llm_request.config.http_options.headers["X-Agent-Name"] == "static_override"


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
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_config_none_raises_when_required(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config = None

    with pytest.raises(FlintAIGuardrailsError, match="no http_options"):
        plugin.before_model_callback(ctx, llm_request)


def test_before_model_callback_config_none_best_effort(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
        require_guardrails=False,
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config = None

    assert plugin.before_model_callback(ctx, llm_request) is None


def test_before_model_callback_http_options_none_raises_when_required(
    mock_genai_modules,
):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options = None

    with pytest.raises(FlintAIGuardrailsError, match="no http_options"):
        plugin.before_model_callback(ctx, llm_request)


def test_before_model_callback_http_options_none_best_effort(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://guardrails.example.com",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
        require_guardrails=False,
    )
    ctx = MagicMock()
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options = None

    assert plugin.before_model_callback(ctx, llm_request) is None


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
    assert headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_agent_name_env_var_takes_priority(
    mock_genai_modules, monkeypatch
):
    monkeypatch.setenv("AGENT_NAME", "env-agent-name")
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

    assert llm_request.config.http_options.headers["X-Agent-Name"] == "env-agent-name"


def test_before_model_callback_agent_id_env_var_in_header_when_provided(
    mock_genai_modules, monkeypatch
):
    monkeypatch.setenv("AGENT_ID", "custom-agent-id")
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

    assert llm_request.config.http_options.headers["X-Agent-Id"] == "custom-agent-id"
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_agent_id_defaults_to_plugin_name(
    mock_genai_modules, monkeypatch
):
    # With no AGENT_ID env var, X-Agent-Id falls back to the plugin name so
    # agent identity is never silently dropped (symmetric with LangChain).
    monkeypatch.delenv("AGENT_ID", raising=False)
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

    assert llm_request.config.http_options.headers["X-Agent-Id"] == "adk-guardrails"
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


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


# ---------------------------------------------------------------------------
# plaintext HTTP rejection
# ---------------------------------------------------------------------------


def test_adk_plugin_rejects_http_non_loopback():
    with pytest.raises(ValueError, match="must use https://"):
        ADKGuardrailsPlugin(
            gateway_url="http://gw.example.com",
            api_key="key",
            llm_api_key="llm_key",
        )


def test_adk_plugin_http_loopback_accepted(mock_genai_modules):
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plugin = ADKGuardrailsPlugin(
            gateway_url="http://localhost:8080",
            api_key="key",
            llm_api_key="llm_key",
        )
    assert plugin._config is not None
    assert plugin._config.gateway_url == "http://localhost:8080"
    assert any(issubclass(x.category, InsecureGatewayWarning) for x in w)
