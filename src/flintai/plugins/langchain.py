"""LangChain guardrails middleware — routes LLM traffic through the guardrails proxy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from flintai.plugins import FlintAIPlugin
from flintai.plugins._llm_wrapper import _get_sdk_headers
from flintai.plugins._provider_base import _resolve_on_init, _validate_and_build_config

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
            gateway_url="https://guardrails.example.com",
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
    ) -> None:
        self._config: GuardrailsConfig | None
        self._config, self._config_from_constructor = _validate_and_build_config(
            gateway_url=gateway_url,
            api_key=api_key,
            llm_api_key=llm_api_key,
            provider=None,
            policy_id=policy_id,
        )
        self._routed = False
        self._sdk_client: Any = None
        self._provider: str | None = None

    def on_init(self, client: FlintAIClient) -> None:
        self._config = _resolve_on_init(
            client, self._config, self._config_from_constructor
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
        """Apply guardrails proxy routing to the underlying SDK client (once)."""
        from flintai.plugins._llm_wrapper import (
            _apply_guardrails_config,
            _unwrap_langchain_model,
        )

        try:
            result = _unwrap_langchain_model(model)
        except TypeError:
            result = None

        if result is None:
            logger.debug(
                "Cannot extract SDK client from %s; "
                "guardrails routing and session headers will not be applied",
                type(model).__name__,
            )
            self._routed = True
            return

        self._sdk_client, self._provider = result

        if self._config is not None:
            _apply_guardrails_config(self._sdk_client, self._provider, self._config)

        self._routed = True

    def _inject_headers(self, thread_id: str | None) -> None:
        """Set session and agent identity headers on the SDK client."""
        headers = _get_sdk_headers(self._sdk_client, self._provider)
        if headers is None:
            logger.debug(
                "Cannot access headers on %s SDK client",
                self._provider,
            )
            return

        if thread_id is not None:
            headers["X-Agent-Session-Id"] = thread_id

        if "X-Agent-Id" not in headers:
            agent_id = os.environ.get("AGENT_ID") or self.name
            headers["X-Agent-Id"] = agent_id
