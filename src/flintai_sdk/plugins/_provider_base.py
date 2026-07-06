"""Helpers for provider-specific guardrails plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flintai_sdk.core import FlintAIClient
    from flintai_sdk.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


def _validate_and_build_config(
    *,
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
    provider: str,
    policy_id: str | None,
) -> tuple[GuardrailsConfig | None, bool]:
    """Shared constructor validation. Returns (config, config_from_constructor)."""
    from flintai_sdk.guardrails import (
        check_optional_guardrails_params,
        resolve_from_env,
    )

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
    )

    if not check_optional_guardrails_params(gateway_url, api_key, llm_api_key):
        return None, False

    from flintai_sdk.guardrails import build_guardrails_config

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
) -> GuardrailsConfig | None:
    """Shared on_init logic. Returns resolved config."""
    if config_from_constructor:
        client.guardrails_config = config
        logger.info("LLM traffic will route through: %s", config.base_url)
        return config
    config = client.guardrails_config
    if config is None:
        logger.warning(
            "No guardrails config found. Pass gateway_url, api_key, and "
            "llm_api_key to flintai_sdk.init() or the plugin constructor."
        )
        return None
    logger.info("LLM traffic will route through: %s", config.base_url)
    return config
