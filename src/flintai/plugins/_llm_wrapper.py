"""Shared wrapping logic for LLM client guardrails routing."""

from __future__ import annotations

import logging
import re
import threading
import weakref
from collections.abc import Callable
from typing import Any

from flintai.guardrails import FlintAIGuardrailsError, GuardrailsConfig

logger = logging.getLogger(__name__)

__all__ = ["wrap_client"]

_wrapped_lock = threading.Lock()
_wrapped_clients: weakref.WeakSet[Any] = weakref.WeakSet()
_wrapped_client_id_refs: dict[int, weakref.ref[Any]] = {}
_wrapped_client_originals: dict[
    int, tuple[weakref.ref[Any], str, Any, frozenset[str]]
] = {}


def _make_id_ref_finalizer(client_id: int) -> Callable[[weakref.ref[Any]], None]:
    """Build a weakref finalizer that drops the client from _wrapped_client_id_refs."""

    def _finalizer(_ref: weakref.ref[Any]) -> None:
        _wrapped_client_id_refs.pop(client_id, None)

    return _finalizer


def _make_originals_finalizer(client_id: int) -> Callable[[weakref.ref[Any]], None]:
    """Build a weakref finalizer that drops the client from _wrapped_client_originals."""

    def _finalizer(_ref: weakref.ref[Any]) -> None:
        _wrapped_client_originals.pop(client_id, None)

    return _finalizer


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
                    _make_id_ref_finalizer(client_id),
                )
            except TypeError:
                pass


def _clear_wrapped() -> None:
    """Restore original config on all wrapped clients, then clear tracking."""
    with _wrapped_lock:
        for _client_id, (ref, provider, orig_url, header_keys) in list(
            _wrapped_client_originals.items()
        ):
            client = ref()
            if client is not None:
                _restore_client_config(client, provider, orig_url, header_keys)
        _wrapped_client_originals.clear()
        _wrapped_clients.clear()
        _wrapped_client_id_refs.clear()


def _get_base_url(client: Any, provider: str) -> Any:
    """Read the current base URL from an SDK client."""
    if provider == "anthropic":
        return getattr(client, "_base_url", None)
    if provider == "openai":
        cls = type(client)
        if hasattr(cls, "base_url") and isinstance(
            getattr(cls, "base_url", None), property
        ):
            return client.base_url
        return getattr(client, "_base_url", None)
    if provider == "google":
        api_client = getattr(client, "_api_client", None)
        if api_client is not None:
            http_options = getattr(api_client, "_http_options", None)
            if http_options is not None:
                return getattr(http_options, "base_url", None)
    return None


def _save_client_originals(
    client: Any, provider: str, original_base_url: Any, header_keys: frozenset[str]
) -> None:
    """Store original client state so it can be restored on shutdown."""
    client_id = id(client)
    with _wrapped_lock:
        existing = _wrapped_client_originals.get(client_id)
        if existing is not None:
            ref, _, orig_url, existing_keys = existing
            if ref() is client:
                _wrapped_client_originals[client_id] = (
                    ref,
                    provider,
                    orig_url,
                    existing_keys | header_keys,
                )
                return
        try:
            ref = weakref.ref(
                client,
                _make_originals_finalizer(client_id),
            )
        except TypeError:
            return
        _wrapped_client_originals[client_id] = (
            ref,
            provider,
            original_base_url,
            header_keys,
        )


def _restore_client_config(
    client: Any, provider: str, original_base_url: Any, header_keys: frozenset[str]
) -> None:
    """Undo guardrails mutations on a wrapped client."""
    if provider == "anthropic":
        if hasattr(client, "_base_url"):
            client._base_url = original_base_url
    elif provider == "openai":
        cls = type(client)
        if hasattr(cls, "base_url") and isinstance(
            getattr(cls, "base_url", None), property
        ):
            client.base_url = original_base_url
        elif hasattr(client, "_base_url"):
            client._base_url = original_base_url
    elif provider == "google":
        api_client = getattr(client, "_api_client", None)
        if api_client is not None:
            http_options = getattr(api_client, "_http_options", None)
            if http_options is not None:
                http_options.base_url = original_base_url

    sdk_headers = _get_sdk_headers(client, provider)
    if sdk_headers is not None:
        for key in header_keys:
            sdk_headers.pop(key, None)


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

_SUPPORTED_SDK_VERSIONS: dict[str, tuple[str, str, str]] = {
    "openai": ("openai", "2.40.0", "3"),
    "anthropic": ("anthropic", "0.105.2", "1"),
    "google": ("google-genai", "2.7.0", "3"),
}

_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def _parse_version(version_str: str) -> tuple[int, ...] | None:
    """Parse a PEP 440 version string into a comparable tuple of integers."""
    match = _VERSION_RE.match(version_str)
    if match is None:
        return None
    return tuple(int(x) for x in match.group(1).split("."))


def _check_sdk_version(provider: str) -> None:
    """Warn if the installed provider SDK version is outside the tested range.

    This is a soft check — it logs a warning but does not raise. The private
    attributes the SDK relies on *may* still work in untested versions, but
    compatibility is not guaranteed.
    """
    entry = _SUPPORTED_SDK_VERSIONS.get(provider)
    if entry is None:
        return

    package_name, min_version_str, max_major_str = entry

    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version(package_name)
    except (ImportError, PackageNotFoundError):
        return

    installed_parts = _parse_version(installed)
    if installed_parts is None:
        return

    min_parts = _parse_version(min_version_str)
    max_major = int(max_major_str)

    too_old = min_parts is not None and installed_parts < min_parts
    too_new = installed_parts[0] >= max_major
    if too_old or too_new:
        logger.warning(
            "%s %s is installed but flintai-sdk-py supports %s >=%s,<%s. "
            "Private API compatibility is not guaranteed.",
            package_name,
            installed,
            package_name,
            min_version_str,
            max_major_str,
        )


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
            "ADK agents cannot be wrapped with flintai.wrap(). "
            "Use ADKGuardrailsPlugin instead:\n"
            "    from flintai.plugins.adk import ADKGuardrailsPlugin\n"
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

    from flintai.guardrails import PROVIDER_PATH_MAP

    path_prefix = PROVIDER_PATH_MAP.get(provider, "")
    return config.gateway_url + path_prefix


def _get_sdk_headers(sdk_client: Any, provider: str) -> dict[str, str] | None:
    """Get the mutable headers dict from an SDK client.

    Single source of truth for where each provider stores HTTP headers.
    """
    if provider in ("openai", "anthropic"):
        return getattr(sdk_client, "_custom_headers", None)
    if provider == "google":
        api_client = getattr(sdk_client, "_api_client", None)
        if api_client is None:
            return None
        http_options = getattr(api_client, "_http_options", None)
        if http_options is None:
            return None
        return getattr(http_options, "headers", None)
    return None


def _merge_sdk_headers(
    client: Any,
    provider: str,
    headers: dict[str, str],
    require_guardrails: bool = False,
) -> None:
    """Merge *headers* into the SDK client's header dict.

    Raises on failure — fail-closed by design so that a provider SDK change
    never silently bypasses guardrails authentication.
    """
    sdk_headers = _get_sdk_headers(client, provider)
    if sdk_headers is not None:
        sdk_headers.update(headers)
        return
    if require_guardrails:
        raise FlintAIGuardrailsError(
            f"Cannot set guardrails headers on {type(client).__name__}: "
            f"headers dict not found. Traffic would route through the "
            f"guardrails proxy without authentication."
        )
    raise TypeError(
        f"Cannot set custom headers on {type(client).__name__}: "
        f"headers dict not found. This usually means the {provider} SDK "
        f"version is incompatible — check SUPPORTED_SDK_VERSIONS."
    )


def _apply_anthropic_config(
    client: Any,
    base_url: str,
    headers: dict[str, str],
    require_guardrails: bool = False,
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
    _merge_sdk_headers(client, "anthropic", headers, require_guardrails)


def _apply_openai_config(
    client: Any,
    base_url: str,
    headers: dict[str, str],
    require_guardrails: bool = False,
) -> None:
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
    _merge_sdk_headers(client, "openai", headers, require_guardrails)


def _apply_google_config(
    client: Any,
    base_url: str,
    headers: dict[str, str],
    require_guardrails: bool = False,
) -> None:
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
    _merge_sdk_headers(client, "google", headers, require_guardrails)


_PROVIDER_APPLIERS: dict[str, Callable[[Any, str, dict[str, str], bool], None]] = {
    "anthropic": _apply_anthropic_config,
    "openai": _apply_openai_config,
    "google": _apply_google_config,
}


def _apply_guardrails_config(
    client: Any,
    provider: str,
    config: GuardrailsConfig,
    require_guardrails: bool = False,
) -> None:
    """Mutate an existing client's base_url and headers for guardrails routing."""
    original_base_url = _get_base_url(client, provider)
    base_url = _resolve_base_url(provider, config)
    applier = _PROVIDER_APPLIERS.get(provider)
    if applier is None:
        raise TypeError(f"No guardrails applier for provider: {provider!r}")
    applier(client, base_url, config.headers, require_guardrails)
    _save_client_originals(
        client, provider, original_base_url, frozenset(config.headers)
    )


def wrap_client(client: Any) -> Any:
    """Wrap a user-created LLM client with guardrails routing.

    Auto-detects the client type (Anthropic, OpenAI, Google GenAI, or
    LangChain chat models wrapping one of these).
    Returns the same client instance, mutated in place.
    """
    from flintai.core import _client as flintai_client

    if _is_wrapped(client):
        logger.warning("Client is already wrapped by FlintAI SDK; skipping.")
        return client

    require = flintai_client.require_guardrails if flintai_client else False

    langchain_result = _unwrap_langchain_model(client)
    if langchain_result is not None:
        sdk_client, provider = langchain_result
        _check_sdk_version(provider)
        if flintai_client and flintai_client.guardrails_config:
            _apply_guardrails_config(
                sdk_client, provider, flintai_client.guardrails_config, require
            )
        elif require:
            raise FlintAIGuardrailsError(
                "Guardrails configuration is required but no config was found. "
                "Configure guardrails via flintai.init() or flintai.wrap() before "
                "wrapping LLM clients."
            )
        else:
            logger.warning(
                "No guardrails config found; client wrapped without guardrails routing."
            )
        _mark_wrapped(client)
        return client

    provider = _detect_client_type(client)
    _check_sdk_version(provider)

    if flintai_client and flintai_client.guardrails_config:
        _apply_guardrails_config(
            client, provider, flintai_client.guardrails_config, require
        )
    elif require:
        raise FlintAIGuardrailsError(
            "Guardrails configuration is required but no config was found. "
            "Configure guardrails via flintai.init() or flintai.wrap() before "
            "wrapping LLM clients."
        )
    else:
        logger.warning(
            "No guardrails config found; client wrapped without guardrails routing."
        )

    _mark_wrapped(client)
    return client
