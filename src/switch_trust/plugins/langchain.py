"""LangChain guardrails middleware — routes LLM traffic through the guardrails proxy."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any

from switch_trust.guardrails import SwitchTrustGuardrailsError, warn_guardrails_disabled
from switch_trust.plugins import SwitchTrustPlugin
from switch_trust.plugins._llm_wrapper import _get_sdk_headers
from switch_trust.plugins._provider_base import (
    _resolve_on_init,
    _validate_and_build_config,
    effective_require_guardrails,
)

if TYPE_CHECKING:
    from switch_trust.core import SwitchTrustClient
    from switch_trust.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)

# Agent-state key holding the current turn id: one per agent invocation, shared by
# every model call of it (tool loop included), sent as X-Agent-Turn-Id.
_TURN_ID_STATE_KEY = "switch_trust_turn_id"

try:
    from langchain.agents.middleware import AgentMiddleware as _LangChainMiddleware
    from langchain.agents.middleware.types import AgentState, PrivateStateAttr

    # total=False rather than NotRequired: typing.NotRequired is 3.11+ and the SDK
    # supports 3.10.
    class _SwitchTrustAgentState(AgentState, total=False):  # type: ignore[misc]
        """Agent state with the middleware's private turn id.

        Checkpointed (not UntrackedValue) so a run resumed after an interrupt keeps
        its turn id; the next invocation overwrites it in ``before_agent``.
        """

        switch_trust_turn_id: Annotated[str, PrivateStateAttr]

except ImportError:
    _SwitchTrustAgentState = None  # type: ignore[assignment,misc]

    class _LangChainMiddleware:  # type: ignore[no-redef]
        """Stub when langchain is not installed."""


def _extract_thread_id(request: Any) -> str | None:
    """Extract thread_id from a LangChain ModelRequest's runtime."""
    runtime = getattr(request, "runtime", None)
    if runtime is None:
        return None

    exec_info = getattr(runtime, "execution_info", None)
    if exec_info is not None:
        tid = getattr(exec_info, "thread_id", None)
        if tid is not None:
            return str(tid)

    config = getattr(runtime, "config", None)
    if config is not None:
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict):
                tid = configurable.get("thread_id")
                if tid is not None:
                    return str(tid)
        else:
            tid = getattr(config, "thread_id", None)
            if tid is not None:
                return str(tid)

    return None


def _extract_turn_id(request: Any) -> str | None:
    """Extract the turn id ``before_agent`` stored in the request's agent state."""
    state = getattr(request, "state", None)
    if not isinstance(state, Mapping):
        return None
    turn_id = state.get(_TURN_ID_STATE_KEY)
    if isinstance(turn_id, str) and turn_id:
        return turn_id
    return None


def _with_http_options_headers(http_options: Any, headers: dict[str, str]) -> Any:
    """Merge headers into a per-request google-genai ``http_options`` value."""
    if http_options is None:
        return {"headers": dict(headers)}
    if isinstance(http_options, Mapping):
        merged = dict(http_options)
        merged["headers"] = {**(merged.get("headers") or {}), **headers}
        return merged
    existing = getattr(http_options, "headers", None) or {}
    return http_options.model_copy(update={"headers": {**existing, **headers}})


class LangChainGuardrailsMiddleware(SwitchTrustPlugin, _LangChainMiddleware):
    """Routes LLM traffic through the guardrails proxy for LangChain agents.

    Usage::

        from switch_trust.plugins.langchain import LangChainGuardrailsMiddleware
        from langchain.agents import create_agent

        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://api.flintai.dev",
            api_key="your-guardrails-api-key",
            llm_api_key="your-openai-api-key",
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[...],
            middleware=[middleware],
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello!"}]},
            config={"configurable": {"thread_id": "session-123"}},
        )
    """

    name = "langchain-guardrails"
    if _SwitchTrustAgentState is not None:
        state_schema = _SwitchTrustAgentState

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        api_key: str | None = None,
        llm_api_key: str | None = None,
        policy_id: str | None = None,
        require_guardrails: bool | None = None,
        dangerously_disable_guardrails: bool = False,
        allow_insecure_gateway: bool = False,
        load_dotenv: bool | str = False,
    ) -> None:
        self._config: GuardrailsConfig | None
        self._config, self._config_from_constructor = _validate_and_build_config(
            gateway_url=gateway_url,
            api_key=api_key,
            llm_api_key=llm_api_key,
            provider=None,
            policy_id=policy_id,
            allow_insecure_gateway=allow_insecure_gateway,
            load_dotenv=load_dotenv,
        )
        if dangerously_disable_guardrails:
            if require_guardrails is True:
                raise ValueError(
                    "Conflicting options: dangerously_disable_guardrails=True "
                    "cannot be combined with require_guardrails=True."
                )
            self._require_guardrails: bool | None = False
        else:
            self._require_guardrails = require_guardrails
        self._routed = False
        self._sdk_client: Any = None
        self._provider: str | None = None

    def on_init(self, client: SwitchTrustClient) -> None:
        if self._require_guardrails is None:
            self._require_guardrails = client.require_guardrails
        self._config = _resolve_on_init(
            client,
            self._config,
            self._config_from_constructor,
            self._require_guardrails,
        )
        if not effective_require_guardrails(self._require_guardrails):
            warn_guardrails_disabled("LangChainGuardrailsMiddleware", self._config)

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Start a new turn: every model call of this invocation shares its id."""
        return {_TURN_ID_STATE_KEY: str(uuid.uuid4())}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Intercept model calls to inject guardrails routing, session and turn headers."""
        model = getattr(request, "model", None)

        if not self._routed and model is not None:
            self._setup_routing(model)

        if self._sdk_client is not None and self._provider is not None:
            self._inject_agent_identity()
            request = self._with_request_headers(request)

        return handler(request)

    def _with_request_headers(self, request: Any) -> Any:
        """Attach session and turn headers to this model call only.

        They go on the request (``model_settings``), not the shared SDK client:
        the client is shared by every concurrent invocation of the agent, so
        per-call values written there would leak across them.
        """
        headers: dict[str, str] = {}
        thread_id = _extract_thread_id(request)
        if thread_id is not None:
            headers["X-Agent-Session-Id"] = thread_id
        turn_id = _extract_turn_id(request)
        if turn_id is not None:
            headers["X-Agent-Turn-Id"] = turn_id
        if not headers:
            return request

        settings = dict(getattr(request, "model_settings", None) or {})
        if self._provider == "google":
            settings["http_options"] = _with_http_options_headers(
                settings.get("http_options"), headers
            )
        else:
            settings["extra_headers"] = {
                **(settings.get("extra_headers") or {}),
                **headers,
            }
        return request.override(model_settings=settings)

    def _setup_routing(self, model: Any) -> None:
        """Apply guardrails proxy routing to the underlying SDK client (once).

        Raises ``TypeError`` if the underlying SDK client cannot be extracted
        or configured — fail-closed so that guardrails are never silently
        bypassed.
        """
        from switch_trust.plugins._llm_wrapper import (  # noqa: PLC0415 - breaks an import cycle
            _apply_guardrails_config,
            _unwrap_langchain_model,
        )

        require = effective_require_guardrails(self._require_guardrails)

        result = _unwrap_langchain_model(model)

        if result is None:
            if require:
                raise SwitchTrustGuardrailsError(
                    f"Cannot extract SDK client from {type(model).__name__}; guardrails routing cannot be applied."
                )
            logger.debug(
                "Cannot extract SDK client from %s; guardrails routing and session headers will not be applied.",
                type(model).__name__,
            )
            return

        self._sdk_client, self._provider = result

        if self._config is None:
            if require:
                raise SwitchTrustGuardrailsError(
                    "Guardrails configuration is required but no config was found. "
                    "Pass gateway_url and api_key to the middleware "
                    "constructor or to switch_trust.init()."
                )
            logger.warning(
                "No guardrails config found; routing not applied for %s.",
                type(model).__name__,
            )
        else:
            _apply_guardrails_config(
                self._sdk_client,
                self._provider,
                self._config,
                require,
            )

        self._routed = True

    def _inject_agent_identity(self) -> None:
        """Set the static agent identity headers (id and name) on the SDK client.

        Raises ``TypeError`` if the headers dict cannot be located — fail-closed
        so that agent identity is never silently dropped.
        """
        # Only invoked once routing is set up, so the provider is resolved.
        assert self._provider is not None
        headers = _get_sdk_headers(self._sdk_client, self._provider)
        if headers is None:
            raise TypeError(
                f"Cannot access headers on {self._provider} SDK client "
                f"({type(self._sdk_client).__name__}): headers dict not found. "
                f"The SDK version may be incompatible."
            )

        if "X-Agent-Id" not in headers:
            agent_id = os.environ.get("AGENT_ID")
            if agent_id:
                headers["X-Agent-Id"] = agent_id

        # Same as the ADK plugin. The platform uses it to name an agent it hasn't
        # seen before; an existing agent keeps its name.
        if "X-Agent-Name" not in headers:
            agent_name = os.environ.get("AGENT_NAME")
            if agent_name:
                headers["X-Agent-Name"] = agent_name
