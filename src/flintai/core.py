"""FlintAI SDK client — central registry for plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from flintai.plugins import FlintAIPlugin

if TYPE_CHECKING:
    from flintai.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)

__all__ = ["FlintAIClient"]

_client: FlintAIClient | None = None


class FlintAIClient:
    """Central FlintAI SDK client that holds guardrails config and plugins.

    Created by :func:`flintai.init`; a single active instance is tracked as a
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
        self.plugins: list[FlintAIPlugin] = []
        self.provider = provider
        self.require_guardrails = require_guardrails
        self.guardrails_config: GuardrailsConfig | None = None

    def __repr__(self) -> str:
        return (
            f"FlintAIClient(provider={self.provider!r}, "
            f"plugins={len(self.plugins)})"
        )

    def register_plugin(self, plugin: FlintAIPlugin) -> None:
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
