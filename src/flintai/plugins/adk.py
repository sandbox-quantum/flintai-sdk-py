"""Google ADK guardrails plugin — routes LLM traffic through the FlintAI guardrails proxy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from flintai.guardrails import FlintAIGuardrailsError
from flintai.plugins import FlintAIPlugin
from flintai.plugins._provider_base import (
    _resolve_on_init,
    _validate_and_build_config,
    effective_require_guardrails,
)

if TYPE_CHECKING:
    from flintai.core import FlintAIClient
    from flintai.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


class ADKGuardrailsPlugin(FlintAIPlugin):
    """Applies guardrails proxy config to a Google ADK agent.

    Usage:
        from flintai.plugins.adk import ADKGuardrailsPlugin
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
    _GUARDRAIL_BLOCKED_CODE = "GUARDRAIL_BLOCKED"

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        api_key: str | None = None,
        llm_api_key: str | None = None,
        policy_id: str | None = None,
        content_config: Any = None,
        require_guardrails: bool | None = None,
    ) -> None:
        self._user_content_config = content_config
        self.content_config: Any = None
        self._require_guardrails = require_guardrails
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
        if self._require_guardrails is None:
            self._require_guardrails = client.require_guardrails
        self._config = _resolve_on_init(
            client,
            self._config,
            self._config_from_constructor,
            self._require_guardrails,
        )
        if self._config is not None and self.content_config is None:
            self._build_content_config(self._config)

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        """Inject agent identity and session ID as request headers.

        Fails closed (symmetric with the LangChain middleware's header
        injection): if the request's header dict cannot be located — ``config``
        or ``http_options`` is missing — identity/session headers would be
        silently dropped, so this raises ``FlintAIGuardrailsError`` when
        guardrails are required. Pass ``require_guardrails=False`` for
        best-effort behavior.
        """
        require = effective_require_guardrails(self._require_guardrails)
        config = llm_request.config
        if config is None or getattr(config, "http_options", None) is None:
            if require:
                raise FlintAIGuardrailsError(
                    "Cannot inject guardrails identity/session headers: the ADK "
                    "llm_request has no http_options. Pass the plugin's "
                    "content_config to the Agent (generate_content_config=...), "
                    "or set require_guardrails=False for best-effort operation."
                )
            logger.debug("llm request config doesn't contain http_options struct")
            return None

        if config.http_options.headers is None:
            config.http_options.headers = {}

        headers = config.http_options.headers

        if "X-Agent-Name" not in headers:
            agent_name = os.environ.get("AGENT_NAME")
            if not agent_name:
                try:
                    agent_name = callback_context.agent_name
                except (AttributeError, TypeError):
                    logger.debug("failed to extract agent name from callback context")

            if agent_name:
                headers["X-Agent-Name"] = agent_name

        if "X-Agent-Id" not in headers:
            headers["X-Agent-Id"] = os.environ.get("AGENT_ID") or self.name

        try:
            session_id = callback_context.session.id
        except (AttributeError, TypeError):
            logger.debug("failed to extract session ID from callback context")
        else:
            headers["X-Agent-Session-Id"] = session_id

        return None

    @classmethod
    def _detect_guardrail_block(cls, error: Exception) -> dict[str, Any] | None:
        for attr in ("details", "body"):
            data = getattr(error, attr, None)
            if (
                isinstance(data, dict)
                and data.get("code") == cls._GUARDRAIL_BLOCKED_CODE
            ):
                return data
        if cls._GUARDRAIL_BLOCKED_CODE in str(error):
            return {}
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

        block_data = cls._detect_guardrail_block(error)

        if block_data is not None:
            logger.warning("Request blocked: %s", error)

            metadata = None
            if block_data:
                metadata = {
                    k: block_data[k]
                    for k in ("policy_id", "policy_name", "findings")
                    if k in block_data
                } or None

            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Request blocked by guardrails policy.")],
                ),
                error_code=cls._GUARDRAIL_BLOCKED_CODE,
                error_message=str(error),
                turn_complete=True,
                custom_metadata=metadata,
            )
        raise error
