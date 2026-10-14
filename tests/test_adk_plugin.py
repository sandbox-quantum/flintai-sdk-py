"""Tests for the ADK guardrails plugin."""

import sys
from unittest.mock import MagicMock, patch

import pytest

import switch_trust
from switch_trust.guardrails import InsecureGatewayWarning, SwitchTrustGuardrailsError
from switch_trust.plugins.adk import ADKGuardrailsPlugin


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
    switch_trust.init(
        provider="google",
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    plugin = ADKGuardrailsPlugin()
    switch_trust.register_plugin(plugin)

    assert plugin.content_config is not None
    assert (
        plugin.content_config.http_options.base_url == "https://api.flintai.dev/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )
    assert plugin.content_config.http_options.headers["X-LLM-API-Key"] == "AIzaSy_test"


def test_adk_plugin_simplified_flow(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    assert plugin.content_config is not None
    assert (
        plugin.content_config.http_options.base_url == "https://api.flintai.dev/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )


class _UserConfig:
    """Minimal stand-in for GenerateContentConfig that copy.copy can clone."""


def test_adk_plugin_merges_user_content_config_into_clone(mock_genai_modules):
    user_config = _UserConfig()
    user_config.temperature = 0.7
    user_config.top_p = 0.9

    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
        content_config=user_config,
    )
    # The clone carries the user settings plus guardrails http_options...
    assert plugin.content_config is not user_config
    assert (
        plugin.content_config.http_options.base_url == "https://api.flintai.dev/gemini/"
    )
    assert (
        plugin.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"
    )
    assert plugin.content_config.temperature == 0.7
    assert plugin.content_config.top_p == 0.9
    # ...but the caller's object is left untouched.
    assert not hasattr(user_config, "http_options")


def test_adk_plugin_reused_content_config_not_cross_contaminated(mock_genai_modules):
    shared = _UserConfig()
    plugin_a = ADKGuardrailsPlugin(
        gateway_url="https://gw1.example.com",
        api_key="grl_a",
        llm_api_key="AIzaSy_test",
        content_config=shared,
    )
    plugin_b = ADKGuardrailsPlugin(
        gateway_url="https://gw2.example.com",
        api_key="grl_b",
        llm_api_key="AIzaSy_test",
        content_config=shared,
    )
    assert plugin_a.content_config is not plugin_b.content_config
    assert (
        plugin_a.content_config.http_options.base_url
        == "https://gw1.example.com/gemini/"
    )
    assert (
        plugin_b.content_config.http_options.base_url
        == "https://gw2.example.com/gemini/"
    )
    assert plugin_a.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_a"
    assert plugin_b.content_config.http_options.headers["X-FlintAI-API-Key"] == "grl_b"
    assert not hasattr(shared, "http_options")


def test_adk_plugin_simplified_with_policy_id(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
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
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    client = switch_trust.init(provider="google", require_guardrails=False)
    with caplog.at_level(logging.INFO):
        switch_trust.register_plugin(plugin)
    assert client.guardrails_config is plugin._config
    assert "will route through" in caplog.text


def test_adk_plugin_partial_params_raises():
    with pytest.raises(ValueError, match="gateway_url and api_key are both required"):
        ADKGuardrailsPlugin(gateway_url="https://gw.example.com")


def test_adk_plugin_no_guardrails_config(caplog):
    switch_trust.init(provider="google", require_guardrails=False)
    plugin = ADKGuardrailsPlugin()
    switch_trust.register_plugin(plugin)

    assert plugin.content_config is None
    assert "No guardrails config found" in caplog.text


def test_adk_plugin_no_config_raises_when_required():
    switch_trust.init(provider="google", require_guardrails=False)
    plugin = ADKGuardrailsPlugin(require_guardrails=True)
    with pytest.raises(SwitchTrustGuardrailsError, match="No guardrails config found"):
        switch_trust.register_plugin(plugin)


def test_adk_plugin_inherits_require_guardrails_from_client():
    client = switch_trust.init(provider="google", require_guardrails=False)
    client.require_guardrails = True
    plugin = ADKGuardrailsPlugin()
    with pytest.raises(SwitchTrustGuardrailsError, match="No guardrails config found"):
        switch_trust.register_plugin(plugin)


def test_adk_plugin_explicit_override_honored_over_client(caplog):
    client = switch_trust.init(provider="google", require_guardrails=False)
    client.require_guardrails = True
    plugin = ADKGuardrailsPlugin(require_guardrails=False)
    switch_trust.register_plugin(plugin)
    assert "No guardrails config found" in caplog.text


def test_adk_plugin_openai_provider(mock_genai_modules):
    switch_trust.init(
        provider="openai",
        gateway_url="https://gw.example.com",
        api_key="grl_key",
        llm_api_key="sk-test",
    )
    plugin = ADKGuardrailsPlugin()
    switch_trust.register_plugin(plugin)

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
        gateway_url="https://api.flintai.dev",
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


def test_before_model_callback_injects_turn_id_from_invocation_id(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name", "session", "invocation_id"])
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    ctx.invocation_id = "e-0f8fad5b-d9cb-469f-a165-70867728950e"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    headers = llm_request.config.http_options.headers
    assert headers["X-Agent-Turn-Id"] == "e-0f8fad5b-d9cb-469f-a165-70867728950e"


def test_before_model_callback_turn_id_changes_per_invocation(mock_genai_modules):
    """Every model call of one invocation shares its id; the next user message
    gets a new one — the rebuilt headers never carry the previous turn's id."""
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    sent = []
    for invocation_id in ["e-turn-1", "e-turn-1", "e-turn-2"]:
        ctx = MagicMock(spec=["agent_name", "session", "invocation_id"])
        ctx.agent_name = "search_agent"
        ctx.session.id = "sess-abc-123"
        ctx.invocation_id = invocation_id
        llm_request = MagicMock()
        llm_request.config.http_options.headers = {}
        plugin.before_model_callback(ctx, llm_request)
        sent.append(llm_request.config.http_options.headers["X-Agent-Turn-Id"])

    assert sent == ["e-turn-1", "e-turn-1", "e-turn-2"]


def test_before_model_callback_no_invocation_id_skips_turn_header(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name", "session"])
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    assert "X-Agent-Turn-Id" not in llm_request.config.http_options.headers


def test_before_model_callback_no_agent_name(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
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


def test_before_model_callback_request_supplied_identity_not_trusted(
    mock_genai_modules, monkeypatch
):
    """A request-level X-Agent-Name is not honored.

    The callback rebuilds identity from the build-time snapshot (which carries
    only the plugin's own base headers) plus the callback context, so a value
    injected onto the incoming request can't override the resolved agent name.
    """
    monkeypatch.delenv("AGENT_NAME", raising=False)
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name", "session"])
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {
        "X-FlintAI-API-Key": "grl_sk_test",
        "X-Agent-Name": "request_injected",
    }

    plugin.before_model_callback(ctx, llm_request)

    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_no_session(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
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


def test_before_model_callback_sequential_reuse_clears_stale_session(
    mock_genai_modules,
):
    """A later request with no session must not inherit an earlier session id."""
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    # One shared config object, reused across turns (the ADK case where the
    # agent's generate_content_config is reused for every request).
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {"X-FlintAI-API-Key": "grl_sk_test"}

    ctx1 = MagicMock(spec=["session"])
    ctx1.session.id = "sess-1"
    plugin.before_model_callback(ctx1, llm_request)
    assert llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-1"

    ctx2 = MagicMock(spec=[])  # no session, no agent_name
    plugin.before_model_callback(ctx2, llm_request)
    assert "X-Agent-Session-Id" not in llm_request.config.http_options.headers
    assert llm_request.config.http_options.headers["X-FlintAI-API-Key"] == "grl_sk_test"


def test_before_model_callback_mixed_session_reuse_overwrites(mock_genai_modules):
    """A later session id replaces an earlier one on a reused config."""
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    ctx1 = MagicMock(spec=["session"])
    ctx1.session.id = "sess-a"
    plugin.before_model_callback(ctx1, llm_request)
    assert llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-a"

    ctx2 = MagicMock(spec=["session"])
    ctx2.session.id = "sess-b"
    plugin.before_model_callback(ctx2, llm_request)
    assert llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-b"


def test_before_model_callback_concurrent_requests_isolated(mock_genai_modules):
    """Interleaved requests sharing one http_options object stay isolated.

    Root agent + sub-agents share a single content_config; ADK hands the same
    http_options object to each turn, so mutating it in place would let the last
    writer win for everyone. Each request must get its own http_options instead.
    """
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    shared_http_options = plugin.content_config.http_options

    req_a = MagicMock()
    req_a.config.http_options = shared_http_options
    req_b = MagicMock()
    req_b.config.http_options = shared_http_options

    ctx_a = MagicMock(spec=["session"])
    ctx_a.session.id = "sess-a"
    ctx_b = MagicMock(spec=["session"])
    ctx_b.session.id = "sess-b"

    plugin.before_model_callback(ctx_a, req_a)
    plugin.before_model_callback(ctx_b, req_b)

    assert req_a.config.http_options.headers["X-Agent-Session-Id"] == "sess-a"
    assert req_b.config.http_options.headers["X-Agent-Session-Id"] == "sess-b"
    # Each request received its own http_options; the shared base is untouched.
    assert req_a.config.http_options is not shared_http_options
    assert req_b.config.http_options is not shared_http_options
    assert req_a.config.http_options is not req_b.config.http_options
    assert "X-Agent-Session-Id" not in shared_http_options.headers


def test_before_model_callback_reused_config_agent_name_not_carried_over(
    mock_genai_modules, monkeypatch
):
    """A reused config must not carry an earlier turn's agent name forward."""
    monkeypatch.delenv("AGENT_NAME", raising=False)
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    # One shared config object threaded through two turns (root + sub-agent).
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {"X-FlintAI-API-Key": "grl_sk_test"}

    ctx1 = MagicMock(spec=["agent_name"])
    ctx1.agent_name = "root_agent"
    plugin.before_model_callback(ctx1, llm_request)
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "root_agent"

    ctx2 = MagicMock(spec=["agent_name"])
    ctx2.agent_name = "search_agent"
    plugin.before_model_callback(ctx2, llm_request)
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_does_not_mutate_content_config_in_place(
    mock_genai_modules,
):
    """Feeding the plugin's own content_config.http_options must not mutate it.

    This is the one path where the write-back actually matters: the callback
    builds a fresh http_options for the request and leaves the plugin's base
    object (and its headers) untouched.
    """
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    base_http_options = plugin.content_config.http_options
    base_headers = base_http_options.headers

    llm_request = MagicMock()
    llm_request.config.http_options = base_http_options

    ctx = MagicMock(spec=["session"])
    ctx.session.id = "sess-1"
    plugin.before_model_callback(ctx, llm_request)

    # The plugin's base object and its headers are untouched...
    assert plugin.content_config.http_options is base_http_options
    assert base_http_options.headers is base_headers
    assert "X-Agent-Session-Id" not in base_headers
    # ...while the session id lands on the fresh per-request http_options.
    assert llm_request.config.http_options is not base_http_options
    assert llm_request.config.http_options.headers["X-Agent-Session-Id"] == "sess-1"


def test_before_model_callback_config_none_raises_when_required(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config = None

    with pytest.raises(SwitchTrustGuardrailsError, match="no http_options"):
        plugin.before_model_callback(ctx, llm_request)


def test_before_model_callback_config_none_best_effort(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
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
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.session.id = "sess-abc-123"
    llm_request = MagicMock()
    llm_request.config.http_options = None

    with pytest.raises(SwitchTrustGuardrailsError, match="no http_options"):
        plugin.before_model_callback(ctx, llm_request)


def test_before_model_callback_http_options_none_best_effort(mock_genai_modules):
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
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
        gateway_url="https://api.flintai.dev",
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
        gateway_url="https://api.flintai.dev",
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
        gateway_url="https://api.flintai.dev",
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


def test_before_model_callback_no_agent_id_header_without_env_var(
    mock_genai_modules, monkeypatch
):
    monkeypatch.delenv("AGENT_ID", raising=False)
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock()
    ctx.agent_name = "search_agent"
    ctx.session.id = "sess-1"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    assert "X-Agent-Id" not in llm_request.config.http_options.headers
    assert llm_request.config.http_options.headers["X-Agent-Name"] == "search_agent"


def test_before_model_callback_env_identity_is_sent_once(
    mock_genai_modules, monkeypatch
):
    """Each identity header is sent once.

    The config snapshot and the callback both read the env.
    """
    monkeypatch.setenv("AGENT_ID", "agn_123")
    monkeypatch.setenv("AGENT_NAME", "env-agent")
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name"])
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    headers = llm_request.config.http_options.headers
    identity = {
        k: v for k, v in headers.items() if k.lower() in ("x-agent-id", "x-agent-name")
    }
    assert identity == {"X-Agent-Id": "agn_123", "X-Agent-Name": "env-agent"}


def test_before_model_callback_empty_env_falls_back_to_agent_name(
    mock_genai_modules, monkeypatch
):
    """An empty AGENT_NAME keeps the ADK fallback to the agent name."""
    monkeypatch.setenv("AGENT_ID", "")
    monkeypatch.setenv("AGENT_NAME", "")
    plugin = ADKGuardrailsPlugin(
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    ctx = MagicMock(spec=["agent_name"])
    ctx.agent_name = "search_agent"
    llm_request = MagicMock()
    llm_request.config.http_options.headers = {}

    plugin.before_model_callback(ctx, llm_request)

    headers = llm_request.config.http_options.headers
    assert headers["X-Agent-Name"] == "search_agent"
    assert "X-Agent-Id" not in headers


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
            allow_insecure_gateway=True,
        )
    assert plugin._config is not None
    assert plugin._config.gateway_url == "http://localhost:8080"
    assert any(issubclass(x.category, InsecureGatewayWarning) for x in w)


_OPT_OUT_MARKER = "SECURITY CONTROL DISABLED"


def _opt_out_records(caplog):
    return [r for r in caplog.records if _OPT_OUT_MARKER in r.getMessage()]


def test_adk_plugin_opt_out_emits_audit_warning(mock_genai_modules, caplog):
    import logging

    # Guarded init (no warning) so the asserted warning comes from the plugin path.
    switch_trust.init(
        provider="google",
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    plugin = ADKGuardrailsPlugin(dangerously_disable_guardrails=True)
    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.register_plugin(plugin)
    assert len(_opt_out_records(caplog)) == 1


def test_adk_plugin_inherited_opt_out_warns_once(mock_genai_modules, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="switch_trust.guardrails"):
        switch_trust.init(provider="google", require_guardrails=False)
        plugin = ADKGuardrailsPlugin()
        switch_trust.register_plugin(plugin)
    # init already warned; once-per-process keeps the total at one.
    assert len(_opt_out_records(caplog)) == 1


def test_adk_plugin_dangerously_disable_conflict_raises():
    with pytest.raises(ValueError, match="Conflicting options"):
        ADKGuardrailsPlugin(
            dangerously_disable_guardrails=True, require_guardrails=True
        )


def test_adk_plugin_dangerously_disable_resolves_to_false(mock_genai_modules):
    switch_trust.init(
        provider="google",
        gateway_url="https://api.flintai.dev",
        api_key="grl_sk_test",
        llm_api_key="AIzaSy_test",
    )
    plugin = ADKGuardrailsPlugin(dangerously_disable_guardrails=True)
    switch_trust.register_plugin(plugin)
    assert plugin._require_guardrails is False
