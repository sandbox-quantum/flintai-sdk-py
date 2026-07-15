"""LangChain guardrails middleware — routes LLM traffic through the guardrails proxy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from flintai.guardrails import FlintAIGuardrailsError
from flintai.plugins import FlintAIPlugin
from flintai.plugins._llm_wrapper import _get_sdk_headers
from flintai.plugins._provider_base import (
    _resolve_on_init,
    _validate_and_build_config,
    effective_require_guardrails,
)

if TYPE_CHECKING:
    from flintai.core import FlintAIClient
    from flintai.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)

try:
    from langchain.agents.middleware import AgentMiddleware as _LangChainMiddleware
except ImportError:

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


class LangChainGuardrailsMiddleware(FlintAIPlugin, _LangChainMiddleware):
    """Routes LLM traffic through the guardrails proxy for LangChain agents.

    Usage::

        from flintai.plugins.langchain import LangChainGuardrailsMiddleware
        from langchain.agents import create_agent

        middleware = LangChainGuardrailsMiddleware(
            gateway_url="https://app.flintai.dev",
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

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        api_key: str | None = None,
        llm_api_key: str | None = None,
        policy_id: str | None = None,
        require_guardrails: bool | None = None,
    ) -> None:
        self._config: GuardrailsConfig | None
        self._config, self._config_from_constructor = _validate_and_build_config(
            gateway_url=gateway_url,
            api_key=api_key,
            llm_api_key=llm_api_key,
            provider=None,
            policy_id=policy_id,
        )
        self._require_guardrails = require_guardrails
        self._routed = False
        self._sdk_client: Any = None
        self._provider: str | None = None

    def on_init(self, client: FlintAIClient) -> None:
        if self._require_guardrails is None:
            self._require_guardrails = client.require_guardrails
        self._config = _resolve_on_init(
            client,
            self._config,
            self._config_from_constructor,
            self._require_guardrails,
        )

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Intercept model calls to inject guardrails routing and session headers."""
        model = getattr(request, "model", None)

        if not self._routed and model is not None:
            self._setup_routing(model)

        if self._sdk_client is not None and self._provider is not None:
            thread_id = _extract_thread_id(request)
            self._inject_headers(thread_id)

        return handler(request)

    def _setup_routing(self, model: Any) -> None:
        """Apply guardrails proxy routing to the underlying SDK client (once).

        Raises ``TypeError`` if the underlying SDK client cannot be extracted
        or configured — fail-closed so that guardrails are never silently
        bypassed.
        """
        from flintai.plugins._llm_wrapper import (
            _apply_guardrails_config,
            _unwrap_langchain_model,
        )

        require = effective_require_guardrails(self._require_guardrails)

        result = _unwrap_langchain_model(model)

        if result is None:
            if require:
                raise FlintAIGuardrailsError(
                    f"Cannot extract SDK client from "
                    f"{type(model).__name__}; "
                    "guardrails routing cannot be applied."
                )
            logger.debug(
                "Cannot extract SDK client from %s; guardrails "
                "routing and session headers will not be applied.",
                type(model).__name__,
            )
            return

        self._sdk_client, self._provider = result

        if self._config is None:
            if require:
                raise FlintAIGuardrailsError(
                    "Guardrails configuration is required but no config was found. "
                    "Pass gateway_url and api_key to the middleware "
                    "constructor or to flintai.init()."
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

    def _inject_headers(self, thread_id: str | None) -> None:
        """Set session and agent identity headers on the SDK client.

        Raises ``TypeError`` if the headers dict cannot be located — fail-closed
        so that session/agent identity is never silently dropped.
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

        if thread_id is not None:
            headers["X-Agent-Session-Id"] = thread_id

        if "X-Agent-Id" not in headers:
            agent_id = os.environ.get("AGENT_ID") or self.name
            headers["X-Agent-Id"] = agent_id
