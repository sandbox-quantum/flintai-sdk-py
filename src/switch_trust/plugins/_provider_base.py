"""Helpers for provider-specific guardrails plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from switch_trust.guardrails import SwitchTrustGuardrailsError

if TYPE_CHECKING:
    from switch_trust.core import SwitchTrustClient
    from switch_trust.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)


def effective_require_guardrails(value: bool | None) -> bool:
    """Resolve a plugin's ``require_guardrails`` to a concrete fail-closed default.

    A plugin's ``require_guardrails`` is ``None`` until ``on_init`` runs and
    inherits the client's value. ``on_init`` only runs when the plugin is
    attached via ``switch_trust.init()`` + ``register_plugin()``; the documented
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
    allow_insecure_gateway: bool = False,
    load_dotenv: bool | str = False,
) -> tuple[GuardrailsConfig | None, bool]:
    """Shared constructor validation. Returns (config, config_from_constructor)."""
    from switch_trust.guardrails import (  # noqa: PLC0415 - breaks an import cycle
        check_optional_guardrails_params,
        resolve_from_env,
    )

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
        load_dotenv=load_dotenv,
    )

    if not check_optional_guardrails_params(gateway_url, api_key, llm_api_key):
        return None, False

    # Guard above returns early unless both are non-None.
    assert gateway_url is not None and api_key is not None

    from switch_trust.guardrails import (  # noqa: PLC0415 - breaks an import cycle
        build_guardrails_config,
    )

    config = build_guardrails_config(
        gateway_url=gateway_url,
        api_key=api_key,
        llm_api_key=llm_api_key,
        provider=provider,
        policy_id=policy_id,
        allow_insecure_gateway=allow_insecure_gateway,
    )
    return config, True


def _resolve_on_init(
    client: SwitchTrustClient,
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
            raise SwitchTrustGuardrailsError(
                "No guardrails config found. Pass gateway_url, api_key, and "
                "llm_api_key to switch_trust.init() or the plugin constructor."
            )
        logger.warning(
            "No guardrails config found. Pass gateway_url, api_key, and "
            "llm_api_key to switch_trust.init() or the plugin constructor."
        )
        return None
    logger.info("LLM traffic will route through: %s", config.base_url)
    return config
