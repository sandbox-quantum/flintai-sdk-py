"""Switch Trust SDK client — central registry for plugins."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from switch_trust.plugins import SwitchTrustPlugin

if TYPE_CHECKING:
    from switch_trust.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)

__all__ = ["SwitchTrustClient", "GuardrailsStatus"]

_client: SwitchTrustClient | None = None


@dataclass
class GuardrailsStatus:
    """Effective guardrails posture of a client."""

    active: bool
    """Traffic is routed through the guardrails gateway (config resolved)."""
    require_guardrails: bool
    """Fail-closed enforcement is on (throws/verifies rather than opting out)."""
    provider: str | None
    agent_id: str | None
    agent_name: str | None


class SwitchTrustClient:
    """Central Switch Trust SDK client that holds guardrails config and plugins.

    Created by :func:`switch_trust.init`; a single active instance is tracked as a
    module-global singleton. Holds the resolved :class:`GuardrailsConfig` (if any)
    and the list of registered plugins, and fans out lifecycle events to them.
    """

    def __init__(
        self,
        provider: str | None = None,
        require_guardrails: bool = True,
    ) -> None:
        """Initialize a client.

        Args:
            provider: Optional provider hint (e.g. ``"openai"``, ``"anthropic"``,
                ``"google"``) used when building the guardrails config.
            require_guardrails: When True, downstream code fails closed if no
                guardrails configuration can be resolved.
        """
        self.plugins: list[SwitchTrustPlugin] = []
        self.provider = provider
        self.require_guardrails = require_guardrails
        self.guardrails_config: GuardrailsConfig | None = None

    def __repr__(self) -> str:
        return f"SwitchTrustClient(provider={self.provider!r}, plugins={len(self.plugins)})"

    def guardrails_status(self) -> GuardrailsStatus:
        """Report the effective guardrails posture of this client."""
        return GuardrailsStatus(
            active=self.guardrails_config is not None,
            require_guardrails=self.require_guardrails,
            provider=self.provider,
            agent_id=os.environ.get("AGENT_ID"),
            agent_name=os.environ.get("AGENT_NAME"),
        )

    def register_plugin(self, plugin: SwitchTrustPlugin) -> None:
        """Register a plugin and invoke its ``on_init`` hook with this client."""
        self.plugins.append(plugin)
        plugin.on_init(self)

    def notify(self, method: str, **kwargs: Any) -> None:
        """Dispatch a lifecycle event to every plugin that implements it.

        Looks up ``method`` on each plugin and calls it with ``kwargs``.
        Exceptions raised by a plugin handler are caught and logged so one
        misbehaving plugin cannot break dispatch to the others.
        """
        for plugin in self.plugins:
            handler = getattr(plugin, method, None)
            if handler:
                try:
                    handler(**kwargs)
                except Exception as exc:
                    logger.warning(
                        "Plugin '%s' error in %s: %s", plugin.name, method, exc
                    )

    def shutdown(self) -> None:
        """Notify plugins of shutdown and clear any active guardrails config."""
        self.notify("on_shutdown")
        if self.guardrails_config is not None:
            self.guardrails_config.clear()
            self.guardrails_config = None
