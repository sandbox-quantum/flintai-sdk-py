"""Switch Trust SDK — lightweight guardrails routing for AI agents."""

from __future__ import annotations

import atexit
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from switch_trust.core import GuardrailsStatus, SwitchTrustClient
from switch_trust.guardrails import SwitchTrustGuardrailsError, warn_guardrails_disabled
from switch_trust.plugins import SwitchTrustPlugin

try:
    __version__ = version("switch-trust-sdk-py")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "__version__",
    "init",
    "shutdown",
    "register_plugin",
    "status",
    "SwitchTrustClient",
    "SwitchTrustGuardrailsError",
    "SwitchTrustPlugin",
    "GuardrailsStatus",
    "wrap",
]

_atexit_registered = False


def _resolve_require_guardrails(
    require_guardrails: bool | None, dangerously_disable_guardrails: bool
) -> bool:
    """require_guardrails and dangerously_disable_guardrails must not contradict."""
    if dangerously_disable_guardrails:
        if require_guardrails is True:
            raise ValueError(
                "Conflicting options: dangerously_disable_guardrails=True cannot "
                "be combined with require_guardrails=True."
            )
        return False
    return True if require_guardrails is None else require_guardrails


def _ensure_client() -> None:
    """Ensure a SwitchTrustClient exists without configuring guardrails."""
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle

    if core._client is not None:
        return

    core._client = SwitchTrustClient()

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True


def _configure_guardrails_from_params(
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
    policy_id: str | None,
    allow_insecure_gateway: bool = False,
    load_dotenv: bool | str = False,
) -> None:
    """Validate and apply guardrails params if any are provided."""
    from switch_trust.guardrails import (  # noqa: PLC0415 - breaks an import cycle
        check_optional_guardrails_params,
        configure_guardrails,
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
        return

    # Guard above returns early unless both are non-None.
    assert gateway_url is not None and api_key is not None

    configure_guardrails(
        gateway_url=gateway_url,
        api_key=api_key,
        llm_api_key=llm_api_key,
        policy_id=policy_id,
        allow_insecure_gateway=allow_insecure_gateway,
    )


def init(
    provider: str | None = None,
    *,
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
    require_guardrails: bool | None = None,
    dangerously_disable_guardrails: bool = False,
    allow_insecure_gateway: bool = False,
    load_dotenv: bool | str = False,
) -> SwitchTrustClient:
    """Initialize Switch Trust SDK and optionally configure guardrails.

    Not thread-safe. Call from the main thread only.

    Set ``allow_insecure_gateway=True`` (local development only) to permit a
    loopback ``gateway_url`` or the ``SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS='*'``
    wildcard. Set ``load_dotenv=True`` to read ``<cwd>/.env`` (or pass a path);
    by default no ``.env`` file is read.
    """
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle
    from switch_trust.guardrails import (  # noqa: PLC0415 - breaks an import cycle
        PROVIDER_PATH_MAP,
    )

    require_guardrails = _resolve_require_guardrails(
        require_guardrails, dangerously_disable_guardrails
    )

    if provider is not None and provider not in PROVIDER_PATH_MAP:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_PATH_MAP.keys())}"
        )

    if core._client is not None:
        core._client.shutdown()

    client = SwitchTrustClient(provider=provider, require_guardrails=require_guardrails)
    core._client = client

    global _atexit_registered
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True

    _configure_guardrails_from_params(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
        allow_insecure_gateway=allow_insecure_gateway,
        load_dotenv=load_dotenv,
    )

    if require_guardrails and client.guardrails_config is None:
        raise SwitchTrustGuardrailsError(
            "Guardrails configuration is required but no config was found. "
            "Provide gateway_url and api_key to switch_trust.init(), "
            "or set the SWITCH_TRUST_GATEWAY_URL and SWITCH_TRUST_API_KEY "
            "environment variables. "
            "Pass require_guardrails=False to allow operation without guardrails."
        )

    if not require_guardrails:
        warn_guardrails_disabled("switch_trust.init()", client.guardrails_config)

    return client


def register_plugin(plugin: SwitchTrustPlugin) -> None:
    """Register a plugin with the active Switch Trust SDK client."""
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle

    if core._client is None:
        raise RuntimeError("Call switch_trust.init() before registering plugins")
    core._client.register_plugin(plugin)


def wrap(
    client: Any,
    *,
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
    require_guardrails: bool | None = None,
    dangerously_disable_guardrails: bool = False,
    forward_llm_key: bool = False,
    allow_insecure_gateway: bool = False,
    load_dotenv: bool | str = False,
) -> Any:
    """Wrap an LLM client with guardrails routing.

    Auto-detects the client type (Anthropic, OpenAI, Google GenAI, or
    LangChain chat models wrapping one of these).
    Auto-initializes Switch Trust SDK if not already initialized.
    Returns the same client instance, mutated in place.

    By default the upstream provider key is NOT forwarded to the gateway; the
    gateway supplies its own upstream credentials. Set ``forward_llm_key=True``
    to auto-extract the key from the client and forward it as ``X-LLM-API-Key``.
    Passing ``llm_api_key=`` or setting ``SWITCH_TRUST_LLM_API_KEY`` still forwards an
    explicit key regardless of this flag.
    """
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle
    from switch_trust.guardrails import (  # noqa: PLC0415 - breaks an import cycle
        resolve_from_env,
    )
    from switch_trust.plugins._llm_wrapper import (  # noqa: PLC0415 - breaks an import cycle
        wrap_client,
    )

    require_guardrails = _resolve_require_guardrails(
        require_guardrails, dangerously_disable_guardrails
    )

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
        load_dotenv=load_dotenv,
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
                "Set SWITCH_TRUST_LLM_API_KEY or pass llm_api_key= explicitly "
                "to avoid implicit credential forwarding."
            )

    _configure_guardrails_from_params(
        gateway_url,
        api_key,
        llm_api_key,
        policy_id,
        allow_insecure_gateway=allow_insecure_gateway,
        load_dotenv=load_dotenv,
    )

    if require_guardrails and active_client.guardrails_config is None:
        raise SwitchTrustGuardrailsError(
            "Guardrails configuration is required but no config was found. "
            "Provide gateway_url and api_key to switch_trust.wrap(), "
            "or set the SWITCH_TRUST_GATEWAY_URL and SWITCH_TRUST_API_KEY "
            "environment variables. "
            "Pass require_guardrails=False to allow operation without guardrails."
        )

    if not require_guardrails:
        warn_guardrails_disabled("switch_trust.wrap()", active_client.guardrails_config)

    return wrap_client(client)


def status() -> GuardrailsStatus | None:
    """Effective guardrails posture of the current client, or None before init()."""
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle

    if core._client is None:
        return None
    return core._client.guardrails_status()


def shutdown() -> None:
    """Shutdown Switch Trust SDK, notify all plugins."""
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle
    from switch_trust.plugins._llm_wrapper import (  # noqa: PLC0415 - breaks an import cycle
        _clear_wrapped,
    )

    if core._client:
        core._client.shutdown()
        core._client = None
    _clear_wrapped()
