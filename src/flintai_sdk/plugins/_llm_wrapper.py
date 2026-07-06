"""Shared wrapping logic for LLM client guardrails routing."""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from typing import Any

from flintai_sdk.guardrails import GuardrailsConfig

logger = logging.getLogger(__name__)

__all__ = ["wrap_client"]

_wrapped_lock = threading.Lock()
_wrapped_clients: weakref.WeakSet[Any] = weakref.WeakSet()
_wrapped_client_id_refs: dict[int, weakref.ref[Any]] = {}


def _is_wrapped(client: Any) -> bool:
    """Return True if the client has already been wrapped by FlintAI SDK."""
    with _wrapped_lock:
        try:
            return client in _wrapped_clients
        except TypeError:
            ref = _wrapped_client_id_refs.get(id(client))
            if ref is None:
                return False
            referent = ref()
            if referent is None:
                _wrapped_client_id_refs.pop(id(client), None)
                return False
            return referent is client


def _mark_wrapped(client: Any) -> None:
    """Record that a client has been wrapped, to prevent double-wrapping."""
    with _wrapped_lock:
        try:
            _wrapped_clients.add(client)
        except TypeError:
            try:
                client_id = id(client)
                _wrapped_client_id_refs[client_id] = weakref.ref(
                    client,
                    lambda _ref, _id=client_id: _wrapped_client_id_refs.pop(_id, None),
                )
            except TypeError:
                pass


def _clear_wrapped() -> None:
    """Reset the set of wrapped clients (called on shutdown)."""
    with _wrapped_lock:
        _wrapped_clients.clear()
        _wrapped_client_id_refs.clear()


_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "google": ("google.genai", "google_genai"),
}


_LANGCHAIN_PREFIXES: dict[str, tuple[str, str]] = {
    "langchain_openai": ("root_client", "openai"),
    "langchain_anthropic": ("_client", "anthropic"),
    "langchain_google_genai": ("client", "google"),
}


def _unwrap_langchain_model(client: Any) -> tuple[Any, str] | None:
    """If *client* is a LangChain chat model, extract the underlying SDK client.

    Returns ``(sdk_client, provider)`` or ``None``.
    """
    module = type(client).__module__ or ""
    for prefix, (attr, provider) in _LANGCHAIN_PREFIXES.items():
        if module.startswith(prefix):
            sdk_client = getattr(client, attr, None)
            if sdk_client is None:
                raise TypeError(
                    f"Cannot unwrap LangChain model {type(client).__name__}: "
                    f"attribute '{attr}' not found. "
                    f"Ensure the model has been fully initialized."
                )
            return (sdk_client, provider)
    return None


def _detect_client_type(client: Any) -> str:
    """Detect the provider of an LLM client.

    Returns the provider string. Raises TypeError for unrecognized or async clients.
    """
    module = type(client).__module__ or ""
    class_name = type(client).__name__

    if class_name.startswith("Async"):
        raise TypeError(
            f"Async clients are not supported: {module}.{class_name}. "
            f"Use the sync client instead (e.g. Anthropic, OpenAI, genai.Client)."
        )

    if module.startswith("google.adk"):
        raise TypeError(
            "ADK agents cannot be wrapped with flintai_sdk.wrap(). "
            "Use ADKGuardrailsPlugin instead:\n"
            "    from flintai_sdk.plugins.adk import ADKGuardrailsPlugin\n"
            "    plugin = ADKGuardrailsPlugin(gateway_url=..., api_key=..., llm_api_key=...)\n"
            "    agent = Agent(model=..., generate_content_config=plugin.content_config, "
            "    before_model_callback=plugin.before_model_callback, "
            "    on_model_error_callback=plugin.on_model_error)"
        )

    for provider, prefixes in _PROVIDER_PREFIXES.items():
        if any(module.startswith(p) for p in prefixes):
            return provider

    raise TypeError(
        f"Unrecognized client type: {module}.{class_name}. "
        f"Supported: anthropic.Anthropic, openai.OpenAI, google.genai.Client, "
        f"or LangChain chat models (ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI)"
    )


def _resolve_base_url(provider: str, config: GuardrailsConfig) -> str:
    """Compute the provider-specific base URL from the guardrails config.

    When provider was known at configure_guardrails() time, config.base_url
    already contains the path prefix.  When it was deferred, we compute it now.
    """
    if config.provider:
        return config.base_url

    from flintai_sdk.guardrails import PROVIDER_PATH_MAP

    path_prefix = PROVIDER_PATH_MAP.get(provider, "")
    return config.gateway_url + path_prefix


def _update_custom_headers(client: Any, headers: dict[str, str]) -> None:
    """Merge guardrails headers into the client's ``_custom_headers``."""
    if hasattr(client, "_custom_headers"):
        client._custom_headers.update(headers)
    else:
        logger.warning(
            "Cannot set custom headers on %s: _custom_headers not found. "
            "Guardrails auth headers will not be applied.",
            type(client).__name__,
        )


def _apply_anthropic_config(
    client: Any, base_url: str, headers: dict[str, str]
) -> None:
    """Rewrite an Anthropic client's base URL and inject guardrails headers."""
    if not hasattr(client, "_base_url"):
        raise TypeError(
            f"Cannot set base_url on {type(client).__name__}: "
            f"_base_url attribute not found"
        )
    try:
        from httpx import URL

        client._base_url = URL(base_url)
    except ImportError:
        client._base_url = base_url
    _update_custom_headers(client, headers)


def _apply_openai_config(client: Any, base_url: str, headers: dict[str, str]) -> None:
    """Rewrite an OpenAI client's base URL and inject guardrails headers."""
    if hasattr(type(client), "base_url") and isinstance(
        getattr(type(client), "base_url", None), property
    ):
        client.base_url = base_url
    elif hasattr(client, "_base_url"):
        client._base_url = base_url
    else:
        raise TypeError(
            f"Cannot set base_url on {type(client).__name__}: "
            f"no known attribute found"
        )
    _update_custom_headers(client, headers)


def _apply_google_config(client: Any, base_url: str, headers: dict[str, str]) -> None:
    """Rewrite a Google GenAI client's HTTP options for guardrails routing."""
    api_client = getattr(client, "_api_client", None)
    if api_client is None:
        raise TypeError(
            f"Cannot set base_url on {type(client).__name__}: "
            f"_api_client attribute not found"
        )
    http_options = getattr(api_client, "_http_options", None)
    if http_options is None:
        raise TypeError(
            f"Cannot set base_url on {type(client).__name__}: "
            f"_http_options attribute not found"
        )
    # Google GenAI SDK requires trailing slash; without it, URL joining drops the last segment
    http_options.base_url = base_url.rstrip("/") + "/"
    http_options.headers.update(headers)


_PROVIDER_APPLIERS: dict[str, Callable[[Any, str, dict[str, str]], None]] = {
    "anthropic": _apply_anthropic_config,
    "openai": _apply_openai_config,
    "google": _apply_google_config,
}


def _apply_guardrails_config(
    client: Any, provider: str, config: GuardrailsConfig
) -> None:
    """Mutate an existing client's base_url and headers for guardrails routing."""
    base_url = _resolve_base_url(provider, config)
    applier = _PROVIDER_APPLIERS.get(provider)
    if applier is None:
        raise TypeError(f"No guardrails applier for provider: {provider!r}")
    applier(client, base_url, config.headers)


def wrap_client(client: Any) -> Any:
    """Wrap a user-created LLM client with guardrails routing.

    Auto-detects the client type (Anthropic, OpenAI, Google GenAI, or
    LangChain chat models wrapping one of these).
    Returns the same client instance, mutated in place.
    """
    from flintai_sdk.core import _client as flintai_client

    if _is_wrapped(client):
        logger.warning("Client is already wrapped by FlintAI SDK; skipping.")
        return client

    langchain_result = _unwrap_langchain_model(client)
    if langchain_result is not None:
        sdk_client, provider = langchain_result
        if flintai_client and flintai_client.guardrails_config:
            _apply_guardrails_config(
                sdk_client, provider, flintai_client.guardrails_config
            )
        _mark_wrapped(client)
        return client

    provider = _detect_client_type(client)

    if flintai_client and flintai_client.guardrails_config:
        _apply_guardrails_config(client, provider, flintai_client.guardrails_config)

    _mark_wrapped(client)
    return client
