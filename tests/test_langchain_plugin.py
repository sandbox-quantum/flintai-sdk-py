"""Tests for the LangChain guardrails middleware."""

import asyncio
import dataclasses
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import switch_trust
from switch_trust.guardrails import InsecureGatewayWarning, SwitchTrustGuardrailsError
from switch_trust.plugins.langchain import (
    LangChainGuardrailsMiddleware,
    _extract_thread_id,
    _extract_turn_id,
)

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


@dataclasses.dataclass
class _FakeModelRequest:
    """Stand-in for langchain's ModelRequest: ``override`` returns a modified copy."""

    model: Any
    runtime: Any
    state: dict = dataclasses.field(default_factory=dict)
    model_settings: dict = dataclasses.field(default_factory=dict)

    def override(self, **overrides):
        return dataclasses.replace(self, **overrides)


def _make_request(model, *, thread_id=None, use_exec_info=True, turn_id=None):
    """Build a fake ModelRequest with runtime (and agent state carrying turn_id)."""
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

    state = {"switch_trust_turn_id": turn_id} if turn_id is not None else {}
    return _FakeModelRequest(model=model, runtime=runtime, state=state)


def _sent_request(handler):
    """The request the middleware handed to the model handler."""
    handler.assert_called_once()
    return handler.call_args.args[0]


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
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        assert middleware._config is not None
        assert middleware._config.gateway_url == "https://api.flintai.dev"
        assert middleware._config.headers["X-FlintAI-API-Key"] == "grl_sk_test"
        assert middleware._config.headers["X-LLM-API-Key"] == "sk-test"

    def test_with_policy_id(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
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
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        client = switch_trust.init(require_guardrails=False)
        with caplog.at_level(logging.INFO):
            switch_trust.register_plugin(middleware)
        assert client.guardrails_config is middleware._config
        assert "will route through" in caplog.text

    def test_no_config_warns(self, caplog):
        switch_trust.init(require_guardrails=False)
        middleware = LangChainGuardrailsMiddleware()
        switch_trust.register_plugin(middleware)
        assert "No guardrails config found" in caplog.text

    def test_no_config_raises_when_required(self):
        switch_trust.init(require_guardrails=False)
        middleware = LangChainGuardrailsMiddleware(require_guardrails=True)
        with pytest.raises(
            SwitchTrustGuardrailsError, match="No guardrails config found"
        ):
            switch_trust.register_plugin(middleware)

    def test_inherits_require_guardrails_from_client(self):
        client = switch_trust.init(require_guardrails=False)
        client.require_guardrails = True
        middleware = LangChainGuardrailsMiddleware()
        with pytest.raises(
            SwitchTrustGuardrailsError, match="No guardrails config found"
        ):
            switch_trust.register_plugin(middleware)

    def test_explicit_override_honored_over_client(self, caplog):
        client = switch_trust.init(require_guardrails=False)
        client.require_guardrails = True
        middleware = LangChainGuardrailsMiddleware(require_guardrails=False)
        switch_trust.register_plugin(middleware)
        assert "No guardrails config found" in caplog.text


# ---------------------------------------------------------------------------
# wrap_model_call — routing
# ---------------------------------------------------------------------------


class TestWrapModelCallRouting:
    def test_applies_guardrails_config_on_first_call(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="response")

        with patch(
            "switch_trust.plugins._llm_wrapper._apply_guardrails_config"
        ) as mock_apply:
            result = middleware.wrap_model_call(request, handler)

        mock_apply.assert_called_once_with(
            model.root_client, "openai", middleware._config, True
        )
        # The handler gets the same call, carrying this request's session header.
        assert _sent_request(handler).model is model
        assert result == "response"

    def test_routing_applied_only_once(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = _make_openai_model()
        handler = MagicMock(return_value="ok")

        with patch(
            "switch_trust.plugins._llm_wrapper._apply_guardrails_config"
        ) as mock_apply:
            middleware.wrap_model_call(_make_request(model), handler)
            middleware.wrap_model_call(_make_request(model), handler)

        mock_apply.assert_called_once()

    def test_unknown_model_skips_routing_when_not_required(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
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
        # No switch_trust.init()/register_plugin => on_init never runs =>
        # require_guardrails stays None and must default to fail-closed.
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
        )
        model = MagicMock()
        type(model).__module__ = "unknown_module"
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(
            SwitchTrustGuardrailsError, match="Cannot extract SDK client"
        ):
            middleware.wrap_model_call(request, handler)

    def test_known_model_no_config_raises_when_guardrails_required(self):
        middleware = LangChainGuardrailsMiddleware(require_guardrails=True)
        model = _make_openai_model()
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(
            SwitchTrustGuardrailsError,
            match="Guardrails configuration is required",
        ):
            middleware.wrap_model_call(request, handler)

    def test_unknown_model_raises_when_guardrails_required(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
            api_key="grl_sk_test",
            llm_api_key="sk-test",
            require_guardrails=True,
        )
        model = MagicMock()
        type(model).__module__ = "unknown_module"
        request = _make_request(model)
        handler = MagicMock(return_value="ok")

        with pytest.raises(
            SwitchTrustGuardrailsError, match="Cannot extract SDK client"
        ):
            middleware.wrap_model_call(request, handler)


# ---------------------------------------------------------------------------
# wrap_model_call — session header injection
# ---------------------------------------------------------------------------


def _middleware():
    return LangChainGuardrailsMiddleware(
        gateway_url="https://gw.example.com",
        api_key="key",
        llm_api_key="llm-key",
    )


class TestWrapModelCallHeaders:
    """Session and turn headers ride on each request, never the shared client."""

    def test_injects_session_id_from_execution_info(self):
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-abc-123", use_exec_info=True)
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        sent = _sent_request(handler)
        assert (
            sent.model_settings["extra_headers"]["X-Agent-Session-Id"] == "sess-abc-123"
        )

    def test_injects_session_id_from_config(self):
        model = _make_openai_model()
        request = _make_request(
            model, thread_id="sess-from-config", use_exec_info=False
        )
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        sent = _sent_request(handler)
        assert (
            sent.model_settings["extra_headers"]["X-Agent-Session-Id"]
            == "sess-from-config"
        )

    def test_session_id_not_written_to_shared_client(self):
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1", turn_id="turn-1")
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        assert "X-Agent-Session-Id" not in model.root_client._custom_headers
        assert "X-Agent-Turn-Id" not in model.root_client._custom_headers

    def test_no_thread_id_or_turn_id_leaves_request_untouched(self):
        model = _make_openai_model()
        request = _make_request(model, thread_id=None)
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        assert _sent_request(handler) is request

    def test_injects_turn_id_from_agent_state(self):
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1", turn_id="turn-abc")
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        extra = _sent_request(handler).model_settings["extra_headers"]
        assert extra == {"X-Agent-Session-Id": "sess-1", "X-Agent-Turn-Id": "turn-abc"}

    def test_merges_with_caller_extra_headers(self):
        model = _make_openai_model()
        request = _make_request(model, turn_id="turn-abc")
        request.model_settings = {"extra_headers": {"X-Custom": "1"}, "temperature": 0}
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        settings = _sent_request(handler).model_settings
        assert settings["extra_headers"] == {
            "X-Custom": "1",
            "X-Agent-Turn-Id": "turn-abc",
        }
        assert settings["temperature"] == 0
        assert request.model_settings["extra_headers"] == {"X-Custom": "1"}, (
            "caller's settings must not be mutated"
        )

    def test_concurrent_invocations_do_not_share_headers(self):
        """Two invocations interleaving on one agent each send their own ids."""
        middleware = _middleware()
        model = _make_openai_model()
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(
                _make_request(model, thread_id="sess-A", turn_id="turn-A"), handler
            )
            middleware.wrap_model_call(
                _make_request(model, thread_id="sess-B", turn_id="turn-B"), handler
            )

        sent = [
            c.args[0].model_settings["extra_headers"] for c in handler.call_args_list
        ]
        assert sent == [
            {"X-Agent-Session-Id": "sess-A", "X-Agent-Turn-Id": "turn-A"},
            {"X-Agent-Session-Id": "sess-B", "X-Agent-Turn-Id": "turn-B"},
        ]

    def test_anthropic_headers(self):
        model = _make_anthropic_model()
        request = _make_request(model, thread_id="sess-anthropic", turn_id="turn-1")
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        extra = _sent_request(handler).model_settings["extra_headers"]
        assert extra == {
            "X-Agent-Session-Id": "sess-anthropic",
            "X-Agent-Turn-Id": "turn-1",
        }

    def test_google_headers(self):
        model = _make_google_model()
        request = _make_request(model, thread_id="sess-google", turn_id="turn-1")
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        settings = _sent_request(handler).model_settings
        assert "extra_headers" not in settings
        assert settings["http_options"] == {
            "headers": {
                "X-Agent-Session-Id": "sess-google",
                "X-Agent-Turn-Id": "turn-1",
            }
        }

    def test_google_merges_caller_http_options(self):
        model = _make_google_model()
        request = _make_request(model, turn_id="turn-1")
        request.model_settings = {
            "http_options": {"timeout": 30, "headers": {"X-Custom": "1"}}
        }
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            _middleware().wrap_model_call(request, handler)

        assert _sent_request(handler).model_settings["http_options"] == {
            "timeout": 30,
            "headers": {"X-Custom": "1", "X-Agent-Turn-Id": "turn-1"},
        }


class TestTurnId:
    def test_before_agent_starts_a_new_turn_each_invocation(self):
        middleware = _middleware()
        first = middleware.before_agent({}, MagicMock())
        second = middleware.before_agent({}, MagicMock())

        assert first["switch_trust_turn_id"]
        assert second["switch_trust_turn_id"]
        assert first["switch_trust_turn_id"] != second["switch_trust_turn_id"]

    def test_abefore_agent_matches_sync(self):
        update = asyncio.run(_middleware().abefore_agent({}, MagicMock()))
        assert update["switch_trust_turn_id"]

    def test_turn_id_is_valid_for_the_proxy(self):
        import re

        turn_id = _middleware().before_agent({}, MagicMock())["switch_trust_turn_id"]
        # Mirrors the guardrails proxy's accepted shape.
        assert re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", turn_id)

    def test_state_schema_declares_private_turn_id(self):
        pytest.importorskip("langchain")
        schema = LangChainGuardrailsMiddleware.state_schema
        assert "switch_trust_turn_id" in schema.__annotations__

    def test_extract_turn_id(self):
        assert _extract_turn_id(_make_request(MagicMock(), turn_id="t-1")) == "t-1"
        assert _extract_turn_id(_make_request(MagicMock())) is None
        assert _extract_turn_id(MagicMock(state=None)) is None


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

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert model.root_client._custom_headers["X-Agent-Id"] == "my-custom-agent"

    def test_no_agent_id_header_without_env_var(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        request = _make_request(model, thread_id="sess-1")
        handler = MagicMock(return_value="ok")

        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(request, handler)

        assert "X-Agent-Id" not in model.root_client._custom_headers


class TestAgentName:
    def _run(self):
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        handler = MagicMock(return_value="ok")
        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(
                _make_request(model, thread_id="sess-1"), handler
            )
        return model.root_client._custom_headers

    def test_agent_name_from_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENT_ID", "agn_123")
        monkeypatch.setenv("AGENT_NAME", "new-langchain-weather")
        headers = self._run()
        assert headers["X-Agent-Id"] == "agn_123"
        assert headers["X-Agent-Name"] == "new-langchain-weather"

    def test_no_agent_name_header_without_env_var(self, monkeypatch):
        monkeypatch.delenv("AGENT_NAME", raising=False)
        assert "X-Agent-Name" not in self._run()

    def test_caller_set_agent_name_is_kept(self, monkeypatch):
        monkeypatch.setenv("AGENT_NAME", "from-env")
        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://gw.example.com",
            api_key="key",
            llm_api_key="llm-key",
        )
        model = _make_openai_model()
        model.root_client._custom_headers["X-Agent-Name"] = "from-default-headers"
        handler = MagicMock(return_value="ok")
        with patch("switch_trust.plugins._llm_wrapper._apply_guardrails_config"):
            middleware.wrap_model_call(
                _make_request(model, thread_id="sess-1"), handler
            )
        assert (
            model.root_client._custom_headers["X-Agent-Name"] == "from-default-headers"
        )


def _agent_identity_headers(headers):
    """Return the id and name headers, matched case-insensitively as in HTTP."""
    return {
        k: v for k, v in headers.items() if k.lower() in ("x-agent-id", "x-agent-name")
    }


class TestAgentIdentityThroughRouting:
    """Test identity headers with the real _apply_guardrails_config, unpatched.

    Routing merges the config's headers into the SDK client before the
    middleware sets identity, so the two paths can collide.
    """

    def _run(self, model):
        handler = MagicMock(return_value="ok")
        _middleware().wrap_model_call(_make_request(model, thread_id="sess-1"), handler)
        return model.root_client._custom_headers

    def test_env_identity_is_sent_once(self, monkeypatch):
        monkeypatch.setenv("AGENT_ID", "agn_123")
        monkeypatch.setenv("AGENT_NAME", "env-agent")
        headers = self._run(_make_openai_model())
        assert _agent_identity_headers(headers) == {
            "X-Agent-Id": "agn_123",
            "X-Agent-Name": "env-agent",
        }

    def test_caller_set_agent_name_is_kept(self, monkeypatch):
        monkeypatch.setenv("AGENT_NAME", "from-env")
        model = _make_openai_model()
        model.root_client._custom_headers["X-Agent-Name"] = "from-default-headers"
        headers = self._run(model)
        assert headers["X-Agent-Name"] == "from-default-headers"


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
                allow_insecure_gateway=True,
            )
        assert middleware._config is not None
        assert middleware._config.gateway_url == "http://localhost:8080"
        assert any(issubclass(x.category, InsecureGatewayWarning) for x in w)


# ---------------------------------------------------------------------------
# opt-out audit warning
# ---------------------------------------------------------------------------

_OPT_OUT_MARKER = "SECURITY CONTROL DISABLED"


def _opt_out_records(caplog):
    return [r for r in caplog.records if _OPT_OUT_MARKER in r.getMessage()]


def test_langchain_plugin_opt_out_emits_audit_warning(caplog):
    import logging

    # Guarded init (no warning) so the asserted warning comes from the plugin path.
    switch_trust.init(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="sk-test",
    )
    middleware = LangChainGuardrailsMiddleware(dangerously_disable_guardrails=True)
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.register_plugin(middleware)
    assert len(_opt_out_records(caplog)) == 1


def test_langchain_plugin_dangerously_disable_conflict_raises():
    with pytest.raises(ValueError, match="Conflicting options"):
        LangChainGuardrailsMiddleware(
            dangerously_disable_guardrails=True, require_guardrails=True
        )


def test_langchain_plugin_dangerously_disable_resolves_to_false():
    switch_trust.init(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="sk-test",
    )
    middleware = LangChainGuardrailsMiddleware(dangerously_disable_guardrails=True)
    switch_trust.register_plugin(middleware)
    assert middleware._require_guardrails is False
