"""Guardrails proxy configuration."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "SwitchTrustGuardrailsError",
    "GuardrailsConfig",
    "InsecureGatewayWarning",
    "PROVIDER_PATH_MAP",
    "build_guardrails_config",
    "check_optional_guardrails_params",
    "configure_guardrails",
    "detect_provider",
    "resolve_from_env",
    "warn_guardrails_disabled",
]


class SwitchTrustGuardrailsError(RuntimeError):
    """Raised when guardrails enforcement is required but cannot be applied."""


_dotenv_loaded = False

_opt_out_warned = False


def reset_opt_out_warned(warned: bool = False) -> None:
    """Test hook: reset the once-per-process latch."""
    global _opt_out_warned
    _opt_out_warned = warned


def warn_guardrails_disabled(source: str, config: GuardrailsConfig | None) -> None:
    """Audit signal (once per process) so the opt-out is not a silent fail-open."""
    global _opt_out_warned
    if _opt_out_warned:
        return
    _opt_out_warned = True
    agent = os.environ.get("AGENT_NAME") or os.environ.get("AGENT_ID") or "unknown"
    posture = (
        "guardrails config is present, but enforcement checks (fail-closed, "
        "base-URL verification) are OFF"
        if config is not None
        else "NO guardrails config is active — LLM traffic is NOT protected by the gateway"
    )
    logger.warning(
        "Switch Trust SDK: SECURITY CONTROL DISABLED — %s invoked with "
        "require_guardrails=False. %s. agent=%r. This deployment is running "
        "without the SDK guardrails guarantee.",
        source,
        posture,
        agent,
    )


class InsecureGatewayWarning(UserWarning):
    """Emitted when gateway_url uses plaintext HTTP."""


PROVIDER_PATH_MAP = {
    "openai": "/openai",
    "google": "/gemini",
    "anthropic": "/anthropic",
}


@dataclass
class GuardrailsConfig:
    """Resolved routing configuration for the Switch Trust guardrails proxy.

    Attributes:
        base_url: Provider-specific proxy URL that SDK traffic is routed to.
        headers: Auth/routing headers injected into outbound requests
            (e.g. ``X-FlintAI-API-Key``). Masked in ``repr`` to avoid leaking
            secrets.
        provider: Provider this config targets, or None if provider-agnostic.
        gateway_url: Base gateway URL before the provider path prefix.
        policy_id: Optional guardrails policy identifier.
    """

    base_url: str
    headers: dict[str, str]
    provider: str | None
    gateway_url: str
    policy_id: str | None = None

    def clear(self) -> None:
        """Zero out all fields so a released config cannot route traffic."""
        self.headers.clear()
        self.base_url = ""
        self.gateway_url = ""
        self.provider = None
        self.policy_id = None

    def __repr__(self) -> str:
        masked = {
            k: (v[:2] + "***" if len(v) > 2 else "***") for k, v in self.headers.items()
        }
        return (
            f"GuardrailsConfig(provider={self.provider!r}, "
            f"gateway_url={self.gateway_url!r}, "
            f"headers={masked}, "
            f"policy_id={self.policy_id!r})"
        )


def detect_provider() -> str | None:
    env_to_provider = {
        "GOOGLE_API_KEY": "google",
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
    }
    found = [p for env_var, p in env_to_provider.items() if os.getenv(env_var)]
    if len(found) > 1:
        raise ValueError(
            f"Multiple provider API keys detected ({', '.join(sorted(found))}). "
            f"Pass provider= explicitly to disambiguate."
        )
    return found[0] if found else None


_ALLOWED_HOSTS_ENV = "SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS"
_ALLOWED_HOSTS_ENV_FALLBACK = "FLINTAI_ALLOWED_GATEWAY_HOSTS"

# Applied when SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS is unset, so an arbitrary gateway_url
# cannot silently receive provider/guardrails keys. Override via the env var
# (comma-separated hosts). Loopback and the "*" wildcard are only honored when
# allow_insecure_gateway=True is passed programmatically (never from env/.env).
_DEFAULT_ALLOWED_GATEWAY_HOSTS: frozenset[str] = frozenset(
    {"api.switchagents.ai", "api.flintai.dev", "app.flintai.dev"}
)


def _validate_guardrails_params(
    gateway_url: str,
    api_key: str,
    allow_insecure_gateway: bool = False,
) -> str:
    """Validate raw guardrails params. Returns normalized gateway_url.

    ``allow_insecure_gateway`` is a programmatic-only opt-in (it cannot be set
    via environment variables). When False (the default), loopback gateway URLs
    and the ``SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS='*'`` wildcard are rejected, since
    both let a caller redirect credentials and traffic to an unintended host.
    """
    parsed = urlparse(gateway_url or "")
    is_loopback = parsed.hostname in ("localhost", "127.0.0.1", "::1")

    # Loopback redirects credentials to a local endpoint; only allow it under an
    # explicit programmatic opt-in so a .env-supplied localhost URL can't do it.
    if is_loopback and not allow_insecure_gateway:
        raise ValueError(
            f"gateway_url {gateway_url!r} targets a loopback host; this requires "
            f"allow_insecure_gateway=True (local development only)."
        )

    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and is_loopback:
        warnings.warn(
            f"gateway_url {gateway_url!r} uses plaintext HTTP; API keys will be sent unencrypted.",
            InsecureGatewayWarning,
            stacklevel=3,
        )
    else:
        raise ValueError(
            f"gateway_url must use https:// (http:// is allowed only for localhost); got: {gateway_url!r}"
        )
    parsed = urlparse(gateway_url)
    if not parsed.hostname:
        raise ValueError(
            f"gateway_url must contain a valid hostname, got: {gateway_url!r}"
        )
    allowed_raw = os.getenv(_ALLOWED_HOSTS_ENV) or os.getenv(
        _ALLOWED_HOSTS_ENV_FALLBACK
    )
    if allowed_raw is not None:
        allowed = {h.strip().lower() for h in allowed_raw.split(",") if h.strip()}
    else:
        allowed = {h.lower() for h in _DEFAULT_ALLOWED_GATEWAY_HOSTS}

    if "*" in allowed:
        if not allow_insecure_gateway:
            raise ValueError(
                f"{_ALLOWED_HOSTS_ENV}='*' disables the gateway host allowlist "
                f"and requires allow_insecure_gateway=True to be set in code. "
                f"Set an explicit comma-separated host list instead, or pass "
                f"allow_insecure_gateway=True for local development."
            )
        if not is_loopback:
            warnings.warn(
                f"{_ALLOWED_HOSTS_ENV}='*' disables the gateway host allowlist: "
                f"gateway_url {parsed.hostname!r} will receive the Switch Trust API key, "
                f"provider credentials, and all prompt/response traffic. Set an "
                f"explicit host allowlist outside local development.",
                InsecureGatewayWarning,
                stacklevel=3,
            )
    elif not is_loopback and parsed.hostname.lower() not in allowed:
        raise ValueError(
            f"gateway_url hostname {parsed.hostname!r} is not in the allowed "
            f"gateway hosts {sorted(allowed)}. Set {_ALLOWED_HOSTS_ENV} to "
            f"override (comma-separated hosts)."
        )
    if not api_key:
        raise ValueError("api_key must be a non-empty string")
    return gateway_url.rstrip("/")


def check_optional_guardrails_params(
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
) -> bool:
    """Return True if guardrails params are provided, False if all are None.

    Raises ValueError if gateway_url or api_key is provided without the other.
    llm_api_key is independently optional.
    """
    if gateway_url is None and api_key is None and llm_api_key is None:
        return False
    missing = [
        name
        for name, val in [
            ("gateway_url", gateway_url),
            ("api_key", api_key),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            "gateway_url and api_key are both required when providing guardrails config."
        )
    return True


_ENV_VARS = {
    "gateway_url": "SWITCH_TRUST_GATEWAY_URL",
    "api_key": "SWITCH_TRUST_API_KEY",
    "llm_api_key": "SWITCH_TRUST_LLM_API_KEY",
    "policy_id": "SWITCH_TRUST_POLICY_ID",
}

# Read when the SWITCH_TRUST_* variable above is unset, so existing FLINTAI_*
# deployments keep working. TODO: Remove as fast-follow
_ENV_VARS_FALLBACK = {
    "gateway_url": "FLINTAI_GATEWAY_URL",
    "api_key": "FLINTAI_API_KEY",
    "llm_api_key": "FLINTAI_LLM_API_KEY",
    "policy_id": "FLINTAI_POLICY_ID",
}


def resolve_from_env(
    gateway_url: str | None = None,
    api_key: str | None = None,
    llm_api_key: str | None = None,
    policy_id: str | None = None,
    load_dotenv: bool | str = False,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fill in None params from SWITCH_TRUST_* (or legacy FLINTAI_*) environment variables.

    Explicit (non-None) values take precedence over env vars, and SWITCH_TRUST_*
    takes precedence over FLINTAI_* when both are set. A ``.env`` file is only
    read when ``load_dotenv`` is truthy: ``True`` loads ``<cwd>/.env`` and a
    string loads that path. This is off by default so an untrusted ``.env`` in
    the working directory cannot influence gateway configuration.
    """
    global _dotenv_loaded
    if load_dotenv and _dotenv_load is not None and not _dotenv_loaded:
        dotenv_path = (
            load_dotenv
            if isinstance(load_dotenv, str)
            else os.path.join(os.getcwd(), ".env")
        )
        _dotenv_load(dotenv_path)
        _dotenv_loaded = True
    if gateway_url is None:
        gateway_url = (
            os.getenv(_ENV_VARS["gateway_url"])
            or os.getenv(_ENV_VARS_FALLBACK["gateway_url"])
            or None
        )
    if api_key is None:
        api_key = (
            os.getenv(_ENV_VARS["api_key"])
            or os.getenv(_ENV_VARS_FALLBACK["api_key"])
            or None
        )
    if llm_api_key is None:
        llm_api_key = (
            os.getenv(_ENV_VARS["llm_api_key"])
            or os.getenv(_ENV_VARS_FALLBACK["llm_api_key"])
            or None
        )
    if policy_id is None:
        policy_id = (
            os.getenv(_ENV_VARS["policy_id"])
            or os.getenv(_ENV_VARS_FALLBACK["policy_id"])
            or None
        )
    return gateway_url, api_key, llm_api_key, policy_id


def _agent_identity_from_env(var: str) -> str:
    """Read an agent identity variable for use as a header value.

    httpx encodes header values as ASCII, so a non-ASCII value would make every
    call raise UnicodeEncodeError. Such a value is skipped rather than encoded,
    and the warning names the variable without echoing the value.
    """
    value = os.environ.get(var, "").strip()
    if value and not value.isascii():
        logger.warning(
            "Switch Trust SDK: %s contains non-ASCII characters, so it is not "
            "sent as a request header. HTTP header values must be ASCII.",
            var,
        )
        return ""
    return value


def build_guardrails_config(
    gateway_url: str,
    api_key: str,
    llm_api_key: str | None = None,
    provider: str | None = None,
    policy_id: str | None = None,
    allow_insecure_gateway: bool = False,
) -> GuardrailsConfig:
    """Build a GuardrailsConfig without touching global state.

    Adds ``X-Agent-Id`` and ``X-Agent-Name`` from the ``AGENT_ID`` and
    ``AGENT_NAME`` environment variables, so the gateway can attribute the call
    to an agent. Each value is stripped, because h11 (httpx's HTTP/1.1 layer)
    rejects a header value with surrounding whitespace. A value that is empty
    after the strip counts as unset, and a non-ASCII value is skipped with a
    warning (see ``_agent_identity_from_env``).
    """
    gateway_url = _validate_guardrails_params(
        gateway_url, api_key, allow_insecure_gateway
    )

    if provider is not None:
        path_prefix = PROVIDER_PATH_MAP.get(provider)
        if path_prefix is None:
            raise ValueError(
                f"Unsupported provider: '{provider}'. Supported: {list(PROVIDER_PATH_MAP.keys())}"
            )
        base_url = gateway_url + path_prefix
    else:
        base_url = gateway_url

    headers: dict[str, str] = {
        "X-FlintAI-API-Key": api_key,
    }
    if llm_api_key is not None:
        headers["X-LLM-API-Key"] = llm_api_key

    if policy_id is not None:
        headers["X-Guardrails-Policy-Id"] = policy_id

    agent_id = _agent_identity_from_env("AGENT_ID")
    if agent_id:
        headers["X-Agent-Id"] = agent_id

    agent_name = _agent_identity_from_env("AGENT_NAME")
    if agent_name:
        headers["X-Agent-Name"] = agent_name

    return GuardrailsConfig(
        base_url=base_url,
        headers=headers,
        provider=provider,
        gateway_url=gateway_url,
        policy_id=policy_id,
    )


def configure_guardrails(
    gateway_url: str,
    api_key: str,
    llm_api_key: str | None = None,
    provider: str | None = None,
    policy_id: str | None = None,
    allow_insecure_gateway: bool = False,
) -> GuardrailsConfig:
    from switch_trust import core  # noqa: PLC0415 - breaks an import cycle

    if provider is None and core._client:
        provider = core._client.provider

    if provider is None:
        provider = detect_provider()

    config = build_guardrails_config(
        gateway_url=gateway_url,
        api_key=api_key,
        llm_api_key=llm_api_key,
        provider=provider,
        policy_id=policy_id,
        allow_insecure_gateway=allow_insecure_gateway,
    )

    if core._client:
        core._client.guardrails_config = config
    else:
        warnings.warn(
            "configure_guardrails() called before switch_trust.init(). "
            "The config will not be attached to any client. "
            "Call switch_trust.init() first, or pass this config manually.",
            stacklevel=2,
        )

    return config
