"""Google ADK guardrails plugin — routes LLM traffic through the Switch Trust guardrails proxy."""

from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING, Any

from switch_trust.guardrails import SwitchTrustGuardrailsError, warn_guardrails_disabled
from switch_trust.plugins import SwitchTrustPlugin
from switch_trust.plugins._provider_base import (
    _resolve_on_init,
    _validate_and_build_config,
    effective_require_guardrails,
)

if TYPE_CHECKING:
    from switch_trust.core import SwitchTrustClient
    from switch_trust.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


class ADKGuardrailsPlugin(SwitchTrustPlugin):
    """Applies guardrails proxy config to a Google ADK agent.

    Usage:
        from switch_trust.plugins.adk import ADKGuardrailsPlugin
        from google.adk import Agent

        plugin = ADKGuardrailsPlugin(
            gateway_url="https://api.flintai.dev",
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
        dangerously_disable_guardrails: bool = False,
        allow_insecure_gateway: bool = False,
        load_dotenv: bool | str = False,
    ) -> None:
        self._user_content_config = content_config
        self.content_config: Any = None
        if dangerously_disable_guardrails:
            if require_guardrails is True:
                raise ValueError(
                    "Conflicting options: dangerously_disable_guardrails=True "
                    "cannot be combined with require_guardrails=True."
                )
            self._require_guardrails: bool | None = False
        else:
            self._require_guardrails = require_guardrails
        self._config, self._config_from_constructor = _validate_and_build_config(
            gateway_url=gateway_url,
            api_key=api_key,
            llm_api_key=llm_api_key,
            provider=self._PROVIDER,
            policy_id=policy_id,
            allow_insecure_gateway=allow_insecure_gateway,
            load_dotenv=load_dotenv,
        )
        if self._config is not None:
            self._build_content_config(self._config)

    def _build_content_config(self, config: GuardrailsConfig) -> None:
        from google.genai.types import (  # noqa: PLC0415 - optional extra
            GenerateContentConfig,
            HttpOptions,
        )

        # Snapshot base headers at build time so each request can rebuild from a
        # known-good baseline instead of the previous request's dict.
        self._base_headers = dict(config.headers)

        http_options = HttpOptions(
            base_url=config.base_url.rstrip("/") + "/",
            headers=dict(config.headers),
        )

        if self._user_content_config is not None:
            # Clone so a config reused across agents/sessions isn't mutated.
            clone = copy.copy(self._user_content_config)
            clone.http_options = http_options
            self.content_config = clone
        else:
            self.content_config = GenerateContentConfig(
                http_options=http_options,
            )

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
            warn_guardrails_disabled("ADKGuardrailsPlugin", self._config)
        if self._config is not None and self.content_config is None:
            self._build_content_config(self._config)

    def before_model_callback(self, callback_context: Any, llm_request: Any) -> None:
        """Inject agent identity, session ID and turn ID as request headers.

        Fails closed (symmetric with the LangChain middleware's header
        injection): if the request's header dict cannot be located — ``config``
        or ``http_options`` is missing — identity/session headers would be
        silently dropped, so this raises ``SwitchTrustGuardrailsError`` when
        guardrails are required. Pass ``require_guardrails=False`` for
        best-effort behavior.
        """
        require = effective_require_guardrails(self._require_guardrails)
        config = llm_request.config
        if config is None or getattr(config, "http_options", None) is None:
            if require:
                raise SwitchTrustGuardrailsError(
                    "Cannot inject guardrails identity/session headers: the ADK "
                    "llm_request has no http_options. Pass the plugin's "
                    "content_config to the Agent (generate_content_config=...), "
                    "or set require_guardrails=False for best-effort operation."
                )
            logger.debug("llm request config doesn't contain http_options struct")
            return None

        http_options = config.http_options

        # Rebuild request headers from the build-time snapshot so identity/session
        # values from a prior turn can't leak in on a reused/shared config. Fall
        # back to the request's own headers only when there is no snapshot.
        base = (
            self._base_headers
            if self._base_headers is not None
            else (http_options.headers or {})
        )
        headers = dict(base)

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
            agent_id = os.environ.get("AGENT_ID")
            if agent_id:
                headers["X-Agent-Id"] = agent_id

        # Set the current session id if present; the snapshot handles the
        # no-session case, so there is nothing stale to clear.
        try:
            session_id = callback_context.session.id
        except (AttributeError, TypeError):
            logger.debug("failed to extract session ID from callback context")
        else:
            if session_id:
                headers["X-Agent-Session-Id"] = session_id

        # The invocation id is one per user message and shared by every model and
        # tool step of it, so it correlates all LLM calls of a turn.
        try:
            turn_id = callback_context.invocation_id
        except (AttributeError, TypeError):
            logger.debug("failed to extract invocation ID from callback context")
        else:
            if isinstance(turn_id, str) and turn_id:
                headers["X-Agent-Turn-Id"] = turn_id

        # Assign a fresh http_options to the per-request config; mutating the
        # shared one would leak identity/session across interleaved turns.
        request_http_options = type(http_options)(
            base_url=getattr(http_options, "base_url", None),
            headers=headers,
        )
        config.http_options = request_http_options

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
            from google.adk.models.llm_response import (  # noqa: PLC0415 - optional extra
                LlmResponse,
            )
            from google.genai import types  # noqa: PLC0415 - optional extra
        except ImportError as exc:
            raise RuntimeError(
                "google-adk and google-genai are required for on_model_error. "
                'Install with: pip install "switch-trust-sdk-py[adk]"'
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
