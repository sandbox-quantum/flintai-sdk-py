"""Plugin system for FlintAI SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flintai_sdk.core import FlintAIClient

__all__ = ["FlintAIPlugin"]


class FlintAIPlugin:
    """Base class for FlintAI SDK plugins.

    Subclass this and override the methods you care about.
    """

    name: str = "unnamed"

    def on_init(self, client: FlintAIClient) -> None:
        pass

    def on_shutdown(self) -> None:
        pass
