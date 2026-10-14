"""Plugin system for Switch Trust SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from switch_trust.core import SwitchTrustClient

__all__ = ["SwitchTrustPlugin"]


class SwitchTrustPlugin:
    """Base class for Switch Trust SDK plugins.

    Subclass this and override the methods you care about.
    """

    name: str = "unnamed"

    def on_init(self, client: SwitchTrustClient) -> None:
        pass

    def on_shutdown(self) -> None:
        pass
