"""Google ADK guardrails plugin — routes LLM traffic through the FlintAI guardrails proxy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from flintai_sdk.plugins import FlintAIPlugin
from flintai_sdk.plugins._provider_base import (
    _resolve_on_init,
    _validate_and_build_config,
)

if TYPE_CHECKING:
    from flintai_sdk.core import FlintAIClient
    from flintai_sdk.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


class ADKGuardrailsPlugin(FlintAIPlugin):
    """Applies guardrails proxy config to a Google ADK agent.

    Usage:
        from flintai_sdk.plugins.adk import ADKGuardrailsPlugin
        from google.adk import Agent

        plugin = ADKGuardrailsPlugin(
            gateway_url="https://guardrails.example.com",
            api_key="your-guardrails-api-key",
            llm_api_key="your-gemini-api-key",
        )

        agent = Agent(
            model="gemini-2.5-flash",
            generate_content_config=plugin.content_config,
            before_model_callback=plugin.before_model_callback,
            on_model_error_callback=plugin.on_model_error,
        )
    """

    name = "adk-guardrails"
    _PROVIDER = "google"
    # Best-effort heuristic for detecting guardrails blocks from error messages.
    # TODO(CHOO-245): Replace with structured error code from proxy.
    _BLOCK_KEYWORDS = frozenset({"blocked", "guardrail"})

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        api_key: str | None = None,
        llm_api_key: str | None = None,
        policy_id: str | None = None,
        content_config: Any = None,
    ) -> None:
        self._user_content_config = content_config
        self.content_config: Any = None
        self._config, self._config_from_constructor = _validate_and_build_config(
            gateway_url=gateway_url,
            api_key=api_key,
            llm_api_key=llm_api_key,
            provider=self._PROVIDER,
            policy_id=policy_id,
        )
        if self._config is not None:
            self._build_content_config(self._config)

    def _build_content_config(self, config: GuardrailsConfig) -> None:
        from google.genai.types import GenerateContentConfig, HttpOptions

        http_options = HttpOptions(
            base_url=config.base_url.rstrip("/") + "/",
            headers=config.headers,
        )

        if self._user_content_config is not None:
            self._user_content_config.http_options = http_options
            self.content_config = self._user_content_config
        else:
            self.content_config = GenerateContentConfig(
                http_options=http_options,
            )

    def on_init(self, client: FlintAIClient) -> None:
        self._config = _resolve_on_init(
            client, self._config, self._config_from_constructor
        )
        if self._config is not None and self.content_config is None:
            self._build_content_config(self._config)

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        """Injects agent identity and session ID as request headers."""
        config = llm_request.config
        if config is None or getattr(config, "http_options", None) is None:
            logger.debug("llm request config doesn't contain http_options struct")
            return None

        if config.http_options.headers is None:
            config.http_options.headers = {}

        headers = config.http_options.headers

        if "X-Agent-Id" not in headers:
            agent_id = os.environ.get("AGENT_ID")
            if not agent_id:
                try:
                    agent_id = callback_context.agent_name
                except (AttributeError, TypeError):
                    logger.debug("failed to extract agent name from callback context")
            if agent_id:
                headers["X-Agent-Id"] = agent_id

        try:
            session_id = callback_context.session.id
        except (AttributeError, TypeError):
            logger.debug("failed to extract session ID from callback context")
        else:
            headers["X-Agent-Session-Id"] = session_id

        return None

    @classmethod
    def on_model_error(
        cls, callback_context: Any, llm_request: Any, error: Exception
    ) -> Any:
        try:
            from google.adk.models.llm_response import LlmResponse
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-adk and google-genai are required for on_model_error. "
                'Install with: pip install "flintai-sdk-py[adk]"'
            ) from exc

        error_msg = str(error).lower()
        status_code = getattr(error, "status_code", None) or getattr(
            error, "code", None
        )
        is_block = (status_code is not None and status_code == 403) or any(
            kw in error_msg for kw in cls._BLOCK_KEYWORDS
        )

        if is_block:
            logger.warning("Request blocked: %s", error)
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Request blocked by guardrails policy.")],
                ),
                error_code="GUARDRAIL_BLOCKED",
                error_message=str(error),
                turn_complete=True,
            )
        raise error
