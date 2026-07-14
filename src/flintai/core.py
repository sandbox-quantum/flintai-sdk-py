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
    def __init__(
        self,
        provider: str | None = None,
    ) -> None:
        self.plugins: list[FlintAIPlugin] = []
        self.provider = provider
        self.guardrails_config: GuardrailsConfig | None = None

    def __repr__(self) -> str:
        return (
            f"FlintAIClient(provider={self.provider!r}, "
            f"plugins={len(self.plugins)})"
        )

    def register_plugin(self, plugin: FlintAIPlugin) -> None:
        self.plugins.append(plugin)
        plugin.on_init(self)

    def notify(self, method: str, **kwargs: Any) -> None:
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
        self.notify("on_shutdown")
        if self.guardrails_config is not None:
            self.guardrails_config.clear()
            self.guardrails_config = None
