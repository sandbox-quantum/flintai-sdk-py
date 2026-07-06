"""FlintAI SDK — lightweight guardrails routing for AI agents."""

from __future__ import annotations

import atexit
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from flintai_sdk.core import FlintAIClient
from flintai_sdk.plugins import FlintAIPlugin

try:
    __version__ = version("flintai-sdk-py")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "__version__",
    "init",
    "shutdown",
    "register_plugin",
    "FlintAIClient",
    "FlintAIPlugin",
    "wrap",
]

_atexit_registered = False


def _ensure_client() -> None:
    """Ensure a FlintAIClient exists without configuring guardrails."""
    from flintai_sdk import core

    if core._client is not None:
        return

    core._client = FlintAIClient()

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True


def _configure_guardrails_from_params(
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
    policy_id: str | None,
) -> None:
    """Validate and apply guardrails params if any are provided."""
    from flintai_sdk.guardrails import (
        check_optional_guardrails_params,
        configure_guardrails,
        resolve_from_env,
    )

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
    )

    if not check_optional_guardrails_params(gateway_url, api_key, llm_api_key):
        return

    configure_guardrails(
        gateway_url=gateway_url,
        api_key=api_key,
        llm_api_key=llm_api_key,
        policy_id=policy_id,
    )


def init(
    provider: str | None = None,
    *,
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
) -> FlintAIClient:
    """Initialize FlintAI SDK and optionally configure guardrails.

    Not thread-safe. Call from the main thread only.
    """
    from flintai_sdk import core
    from flintai_sdk.guardrails import PROVIDER_PATH_MAP

    if provider is not None and provider not in PROVIDER_PATH_MAP:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported: {list(PROVIDER_PATH_MAP.keys())}"
        )

    if core._client is not None:
        core._client.shutdown()

    client = FlintAIClient(provider=provider)
    core._client = client

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True

    _configure_guardrails_from_params(gateway_url, api_key, llm_api_key, policy_id)

    return client


def register_plugin(plugin: FlintAIPlugin) -> None:
    """Register a plugin with the active FlintAI SDK client."""
    from flintai_sdk import core

    if core._client is None:
        raise RuntimeError("Call flintai_sdk.init() before registering plugins")
    core._client.register_plugin(plugin)


def wrap(
    client: Any,
    *,
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
) -> Any:
    """Wrap an LLM client with guardrails routing.

    Auto-detects the client type (Anthropic, OpenAI, Google GenAI, or
    LangChain chat models wrapping one of these).
    Auto-initializes FlintAI SDK if not already initialized.
    Returns the same client instance, mutated in place.
    """
    from flintai_sdk.guardrails import resolve_from_env
    from flintai_sdk.plugins._llm_wrapper import wrap_client

    _ensure_client()

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
    )

    if llm_api_key is None and gateway_url is not None:
        llm_api_key = getattr(client, "api_key", None)
        if llm_api_key is None:
            for attr in ("root_client", "_client", "client"):
                inner = getattr(client, attr, None)
                if inner is not None:
                    llm_api_key = getattr(inner, "api_key", None)
                    if llm_api_key is not None:
                        break

    _configure_guardrails_from_params(gateway_url, api_key, llm_api_key, policy_id)

    return wrap_client(client)


def shutdown() -> None:
    """Shutdown FlintAI SDK, notify all plugins."""
    from flintai_sdk import core
    from flintai_sdk.plugins._llm_wrapper import _clear_wrapped

    if core._client:
        core._client.shutdown()
        core._client = None
    _clear_wrapped()
