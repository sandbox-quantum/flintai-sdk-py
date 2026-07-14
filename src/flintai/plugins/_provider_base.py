"""Helpers for provider-specific guardrails plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flintai.guardrails import FlintAIGuardrailsError

if TYPE_CHECKING:
    from flintai.core import FlintAIClient
    from flintai.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


def effective_require_guardrails(value: bool | None) -> bool:
    """Resolve a plugin's ``require_guardrails`` to a concrete fail-closed default.

    A plugin's ``require_guardrails`` is ``None`` until ``on_init`` runs and
    inherits the client's value. ``on_init`` only runs when the plugin is
    attached via ``flintai.init()`` + ``register_plugin()``; the documented
    standalone usage (handing the plugin straight to ``create_agent``/``Agent``)
    never calls it. So an unresolved ``None`` must default to **fail-closed**
    (``True``) — otherwise standalone plugins would silently bypass guardrails.
    An explicit ``False`` remains a valid opt-out.
    """
    return True if value is None else value


def _validate_and_build_config(
    *,
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
    provider: str | None,
    policy_id: str | None,
) -> tuple[GuardrailsConfig | None, bool]:
    """Shared constructor validation. Returns (config, config_from_constructor)."""
    from flintai.guardrails import check_optional_guardrails_params, resolve_from_env

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
    )

    if not check_optional_guardrails_params(gateway_url, api_key, llm_api_key):
        return None, False

    # Guard above returns early unless both are non-None.
    assert gateway_url is not None and api_key is not None

    from flintai.guardrails import build_guardrails_config

    config = build_guardrails_config(
        gateway_url=gateway_url,
        api_key=api_key,
        llm_api_key=llm_api_key,
        provider=provider,
        policy_id=policy_id,
    )
    return config, True


def _resolve_on_init(
    client: FlintAIClient,
    config: GuardrailsConfig | None,
    config_from_constructor: bool,
    require_guardrails: bool = True,
) -> GuardrailsConfig | None:
    """Shared on_init logic. Returns resolved config."""
    if config_from_constructor:
        # config_from_constructor is only True when a config was built.
        assert config is not None
        client.guardrails_config = config
        logger.info("LLM traffic will route through: %s", config.base_url)
        return config
    config = client.guardrails_config
    if config is None:
        if require_guardrails:
            raise FlintAIGuardrailsError(
                "No guardrails config found. Pass gateway_url, api_key, and "
                "llm_api_key to flintai.init() or the plugin constructor."
            )
        logger.warning(
            "No guardrails config found. Pass gateway_url, api_key, and "
            "llm_api_key to flintai.init() or the plugin constructor."
        )
        return None
    logger.info("LLM traffic will route through: %s", config.base_url)
    return config
