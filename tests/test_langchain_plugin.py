"""Tests for the LangChain guardrails middleware."""

from unittest.mock import MagicMock, patch

import pytest

import flintai
from flintai.guardrails import FlintAIGuardrailsError, InsecureGatewayWarning
from flintai.plugins.langchain import LangChainGuardrailsMiddleware, _extract_thread_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_model():
    """Create a mock LangChain ChatOpenAI model with an underlying SDK client."""
    model = MagicMock()
    type(model).__module__ = "langchain_openai.chat_models"
    model.root_client = MagicMock()
    model.root_client._custom_headers = {}
    return model


def _make_anthropic_model():
    """Create a mock LangChain ChatAnthropic model."""
    model = MagicMock()
    type(model).__module__ = "langchain_anthropic.chat_models"
    model._client = MagicMock()
    model._client._custom_headers = {}
    return model


def _make_google_model():
    """Create a mock LangChain ChatGoogleGenerativeAI model."""
    model = MagicMock()
    type(model).__module__ = "langchain_google_genai.chat_models"
    model.client = MagicMock()
    model.client._api_client = MagicMock()
    model.client._api_client._http_options = MagicMock()
    model.client._api_client._http_options.headers = {}
    return model


def _make_request(model, *, thread_id=None, use_exec_info=True):
    """Build a mock ModelRequest with runtime."""
    request = MagicMock()
    request.model = model

    runtime = MagicMock()

    if thread_id is not None and use_exec_info:
        runtime.execution_info = MagicMock()
        runtime.execution_info.thread_id = thread_id
        runtime.config = None
    elif thread_id is not None:
        runtime.execution_info = None
        runtime.config = MagicMock()
        runtime.config.thread_id = thread_id
    else:
        runtime.execution_info = None
        runtime.config = None

    request.runtime = runtime
    return request


# ---------------------------------------------------------------------------
# _extract_thread_id
# ---------------------------------------------------------------------------


class TestExtractThreadId:
    def test_from_execution_info(self):
        request = MagicMock()
        request.runtime.execution_info.thread_id = "thread-abc"
        request.runtime.config = None
        assert _extract_thread_id(request) == "thread-abc"

    def test_from_config_object(self):
        request = MagicMock()
        request.runtime.execution_info = None
        request.runtime.config.thread_id = "thread-from-config"
        assert _extract_thread_id(request) == "thread-from-config"

    def test_from_config_dict(self):
        request = MagicMock()
        request.runtime.execution_info = None
        request.runtime.config = {"configurable": {"thread_id": "thread-from-dict"}}
        assert _extract_thread_id(request) == "thread-from-dict"

    def test_no_runtime(self):
        request = MagicMock(spec=[])
        assert _extract_thread_id(request) is None

    def test_no_thread_id(self):
        request = MagicMock()
        request.runtime.execution_info = None
        request.runtime.config = None
        assert _extract_thread_id(request) is None

    def test_non_string_thread_id_is_stringified(self):
        request = MagicMock()
        request.runtime.execution_info.thread_id = 42
        request.runtime.config = None
        assert _extract_thread_id(request) == "42"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_with_explicit_params(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        assert middleware._config is not None
        assert middleware._config.gateway_url == "https://guardrails.example.com"
        assert middleware._config.headers["X-FlintAI-API-Key"] == "grl_sk_test"
        assert middleware._config.headers["X-LLM-API-Key"] == "sk-test"

    def test_with_policy_id(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
            policy_id="pol-1",
        )
        assert middleware._config.headers["X-Guardrails-Policy-Id"] == "pol-1"

    def test_partial_params_raises(self):
        with pytest.raises(
            ValueError, match="gateway_url and api_key are both required"
        ):
            LangChainGuardrailsMiddleware(gateway_url="https://gw.example.com")

    def test_no_params_no_config(self):
        middleware = LangChainGuardrailsMiddleware()
        assert middleware._config is None

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("FLINTAI_GATEWAY_URL", "https://guardrails.env.com")
        monkeypatch.setenv("FLINTAI_API_KEY", "env-api-key")
        monkeypatch.setenv("FLINTAI_LLM_API_KEY", "env-llm-key")
        middleware = LangChainGuardrailsMiddleware()
        assert middleware._config is not None
        assert middleware._config.gateway_url == "https://guardrails.env.com"


# ---------------------------------------------------------------------------
# on_init
# ---------------------------------------------------------------------------


class TestOnInit:
    def test_with_constructor_config(self, caplog):
        import logging

        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        client = flintai.init(require_guardrails=False)
        with caplog.at_level(logging.INFO):
            flintai.register_plugin(middleware)
        assert client.guardrails_config is middleware._config
        assert "will route through" in caplog.text

    def test_no_config_warns(self, caplog):
        flintai.init(require_guardrails=False)
        middleware = LangChainGuardrailsMiddleware()
        flintai.register_plugin(middleware)
        assert "No guardrails config found" in caplog.text

    def test_no_config_raises_when_required(self):
        flintai.init(require_guardrails=False)
        middleware = LangChainGuardrailsMiddleware(require_guardrails=True)
        with pytest.raises(FlintAIGuardrailsError, match="No guardrails config found"):
            flintai.register_plugin(middleware)

    def test_inherits_require_guardrails_from_client(self):
        client = flintai.init(require_guardrails=False)
        client.require_guardrails = True
        middleware = LangChainGuardrailsMiddleware()
        with pytest.raises(FlintAIGuardrailsError, match="No guardrails config found"):
            flintai.register_plugin(middleware)

    def test_explicit_override_honored_over_client(self, caplog):
        client = flintai.init(require_guardrails=False)
        client.require_guardrails = True
        middleware = LangChainGuardrailsMiddleware(require_guardrails=False)
        flintai.register_plugin(middleware)
        assert "No guardrails config found" in caplog.text


# ---------------------------------------------------------------------------
# wrap_model_call — routing
# ---------------------------------------------------------------------------


class TestWrapModelCallRouting:
    def test_applies_guardrails_config_on_first_call(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="response")

        with patch(
            "flintai.plugins._llm_wrapper._apply_guardrails_config"
        ) as mock_apply:
            result = middleware.wrap_model_call(request, handler)

        mock_apply.assert_called_once_with(
            model.root_client, "openai", middleware._config, True
        )
        handler.assert_called_once_with(request)
        assert result == "response"

    def test_routing_applied_only_once(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = _make_openai_model()
        handler = MagicMock(return_value="ok")

        with patch(
            "flintai.plugins._llm_wrapper._apply_guardrails_config"
        ) as mock_apply:
            middleware.wrap_model_call(_make_request(model), handler)
            middleware.wrap_model_call(_make_request(model), handler)

        mock_apply.assert_called_once()

    def test_unknown_model_skips_routing_when_not_required(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
            require_guardrails=False,
        )
        model = MagicMock()
        type(model).__module__ = "unknown_module"
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        result = middleware.wrap_model_call(request, handler)

        assert result == "ok"
        assert middleware._routed is False
        assert middleware._sdk_client is None

    def test_unknown_model_fails_closed_by_default_standalone(self):
        # No flintai.init()/register_plugin => on_init never runs =>
        # require_guardrails stays None and must default to fail-closed.
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = MagicMock()
        type(model).__module__ = "unknown_module"
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(FlintAIGuardrailsError, match="Cannot extract SDK client"):
            middleware.wrap_model_call(request, handler)

    def test_known_model_no_config_raises_when_guardrails_required(self):
        middleware = LangChainGuardrailsMiddleware(require_guardrails=True)
        model = _make_openai_model()
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(
            FlintAIGuardrailsError,
            match="Guardrails configuration is required",
        ):
            middleware.wrap_model_call(request, handler)

    def test_unknown_model_raises_when_guardrails_required(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://guardrails.example.com",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
            require_guardrails=True,
        )
        model = MagicMock()
        type(model).__module__ = "unknown_module"
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(FlintAIGuardrailsError, match="Cannot extract SDK client"):
            middleware.wrap_model_call(request, handler)


# ---------------------------------------------------------------------------
# wrap_model_call — session header injection
# ---------------------------------------------------------------------------


class TestWrapModelCallHeaders:
    def test_injects_session_id_from_execution_info(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-abc-123", use_exec_info=True)
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        headers = model.root_client._custom_headers
        assert headers["X-Agent-Session-Id"] == "sess-abc-123"
        assert headers["X-Agent-Id"] == "langchain-guardrails"

    def test_injects_session_id_from_config(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(
            model, thread_id="sess-from-config", use_exec_info=False
        )
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert (
            model.root_client._custom_headers["X-Agent-Session-Id"]
            == "sess-from-config"
        )

    def test_no_thread_id_skips_session_header(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id=None)
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert "X-Agent-Session-Id" not in model.root_client._custom_headers

    def test_static_agent_id_not_overwritten(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        model.root_client._custom_headers["X-Agent-Id"] = "agn_static_override"
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert model.root_client._custom_headers["X-Agent-Id"] == "agn_static_override"

    def test_anthropic_headers(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_anthropic_model()
        request = _make_request(model, thread_id="sess-anthropic")
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert model._client._custom_headers["X-Agent-Session-Id"] == "sess-anthropic"

    def test_google_headers(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_google_model()
        request = _make_request(model, thread_id="sess-google")
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        headers = model.client._api_client._http_options.headers
        assert headers["X-Agent-Session-Id"] == "sess-google"


# ---------------------------------------------------------------------------
# Agent ID from env var
# ---------------------------------------------------------------------------


class TestAgentId:
    def test_agent_id_from_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENT_ID", "my-custom-agent")
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert model.root_client._custom_headers["X-Agent-Id"] == "my-custom-agent"

    def test_agent_id_defaults_to_plugin_name(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="ok")

        with patch("flintai.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert model.root_client._custom_headers["X-Agent-Id"] == "langchain-guardrails"


# ---------------------------------------------------------------------------
# plaintext HTTP rejection
# ---------------------------------------------------------------------------


class TestHTTPRejection:
    def test_rejects_http_non_loopback(self):
        with pytest.raises(ValueError, match="must use https://"):
            LangChainGuardrailsMiddleware(
                gateway_url="http://gw.example.com",
                api_key="key",
                llm_api_key="llm_key",
            )

    def test_http_loopback_accepted(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            middleware = LangChainGuardrailsMiddleware(
                gateway_url="http://localhost:8080",
                api_key="key",
                llm_api_key="llm_key",
            )
        assert middleware._config is not None
        assert middleware._config.gateway_url == "http://localhost:8080"
        assert any(issubclass(x.category, InsecureGatewayWarning) for x in w)
