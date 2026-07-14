"""FlintAI SDK — lightweight guardrails routing for AI agents."""

from __future__ import annotations

import atexit
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from flintai.core import FlintAIClient
from flintai.guardrails import FlintAIGuardrailsError
from flintai.plugins import FlintAIPlugin

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
    "FlintAIGuardrailsError",
    "FlintAIPlugin",
    "wrap",
]

_atexit_registered = False


def _ensure_client() -> None:
    """Ensure a FlintAIClient exists without configuring guardrails."""
    from flintai import core

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
    from flintai.guardrails import (
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

    # Guard above returns early unless both are non-None.
    assert gateway_url is not None and api_key is not None

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
    require_guardrails: bool = True,
) -> FlintAIClient:
    """Initialize FlintAI SDK and optionally configure guardrails.

    Not thread-safe. Call from the main thread only.
    """
    from flintai import core
    from flintai.guardrails import PROVIDER_PATH_MAP

    if provider is not None and provider not in PROVIDER_PATH_MAP:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_PATH_MAP.keys())}"
        )

    if core._client is not None:
        core._client.shutdown()

    client = FlintAIClient(provider=provider, require_guardrails=require_guardrails)
    core._client = client

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True

    _configure_guardrails_from_params(gateway_url, api_key, llm_api_key, policy_id)

    if require_guardrails and client.guardrails_config is None:
        raise FlintAIGuardrailsError(
            "Guardrails configuration is required but no config was found. "
            "Provide gateway_url and api_key to flintai.init(), "
            "or set the FLINTAI_GATEWAY_URL and FLINTAI_API_KEY "
            "environment variables. "
            "Pass require_guardrails=False to allow operation without guardrails."
        )

    return client


def register_plugin(plugin: FlintAIPlugin) -> None:
    """Register a plugin with the active FlintAI SDK client."""
    from flintai import core

    if core._client is None:
        raise RuntimeError("Call flintai.init() before registering plugins")
    core._client.register_plugin(plugin)


def wrap(
    client: Any,
    *,
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
    require_guardrails: bool = True,
    forward_llm_key: bool = False,
) -> Any:
    """Wrap an LLM client with guardrails routing.

    Auto-detects the client type (Anthropic, OpenAI, Google GenAI, or
    LangChain chat models wrapping one of these).
    Auto-initializes FlintAI SDK if not already initialized.
    Returns the same client instance, mutated in place.

    By default the upstream provider key is NOT forwarded to the gateway; the
    gateway supplies its own upstream credentials. Set ``forward_llm_key=True``
    to auto-extract the key from the client and forward it as ``X-LLM-API-Key``.
    Passing ``llm_api_key=`` or setting ``FLINTAI_LLM_API_KEY`` still forwards an
    explicit key regardless of this flag.
    """
    from flintai import core
    from flintai.guardrails import resolve_from_env
    from flintai.plugins._llm_wrapper import wrap_client

    created = core._client is None
    _ensure_client()
    # _ensure_client() guarantees the global client is set.
    assert core._client is not None
    active_client = core._client

    if created:
        active_client.require_guardrails = require_guardrails

    gateway_url, api_key, llm_api_key, policy_id = resolve_from_env(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
    )

    if forward_llm_key and llm_api_key is None and gateway_url is not None:
        llm_api_key = getattr(client, "api_key", None)
        if llm_api_key is None:
            for attr in ("root_client", "_client", "client"):
                inner = getattr(client, attr, None)
                if inner is not None:
                    llm_api_key = getattr(inner, "api_key", None)
                    if llm_api_key is not None:
                        break
        if llm_api_key is not None:
            logging.getLogger(__name__).warning(
                "llm_api_key auto-extracted from client object. "
                "Set FLINTAI_LLM_API_KEY or pass llm_api_key= explicitly "
                "to avoid implicit credential forwarding."
            )

    _configure_guardrails_from_params(gateway_url, api_key, llm_api_key, policy_id)

    if require_guardrails and active_client.guardrails_config is None:
        raise FlintAIGuardrailsError(
            "Guardrails configuration is required but no config was found. "
            "Provide gateway_url and api_key to flintai.wrap(), "
            "or set the FLINTAI_GATEWAY_URL and FLINTAI_API_KEY "
            "environment variables. "
            "Pass require_guardrails=False to allow operation without guardrails."
        )

    return wrap_client(client)


def shutdown() -> None:
    """Shutdown FlintAI SDK, notify all plugins."""
    from flintai import core
    from flintai.plugins._llm_wrapper import _clear_wrapped

    if core._client:
        core._client.shutdown()
        core._client = None
    _clear_wrapped()
