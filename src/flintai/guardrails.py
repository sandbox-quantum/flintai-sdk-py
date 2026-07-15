"""Guardrails proxy configuration."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

__all__ = [
    "FlintAIGuardrailsError",
    "GuardrailsConfig",
    "InsecureGatewayWarning",
    "PROVIDER_PATH_MAP",
    "build_guardrails_config",
    "check_optional_guardrails_params",
    "configure_guardrails",
    "detect_provider",
    "resolve_from_env",
]


class FlintAIGuardrailsError(RuntimeError):
    """Raised when guardrails enforcement is required but cannot be applied."""


_dotenv_loaded = False


class InsecureGatewayWarning(UserWarning):
    """Emitted when gateway_url uses plaintext HTTP."""


PROVIDER_PATH_MAP = {
    "openai": "/openai",
    "google": "/gemini",
    "anthropic": "/anthropic",
}


@dataclass
class GuardrailsConfig:
    """Resolved routing configuration for the FlintAI guardrails proxy.

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


_ALLOWED_HOSTS_ENV = "FLINTAI_ALLOWED_GATEWAY_HOSTS"

# Applied when FLINTAI_ALLOWED_GATEWAY_HOSTS is unset, so an arbitrary gateway_url
# cannot silently receive provider/guardrails keys. Override via the env var
# (comma-separated hosts, or "*" to allow any host). Loopback is always allowed.
_DEFAULT_ALLOWED_GATEWAY_HOSTS: frozenset[str] = frozenset({"app.flintai.dev"})


def _validate_guardrails_params(
    gateway_url: str,
    api_key: str,
) -> str:
    """Validate raw guardrails params. Returns normalized gateway_url."""
    parsed = urlparse(gateway_url or "")
    is_loopback = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and is_loopback:
        warnings.warn(
            f"gateway_url {gateway_url!r} uses plaintext HTTP; "
            "API keys will be sent unencrypted.",
            InsecureGatewayWarning,
            stacklevel=3,
        )
    else:
        raise ValueError(
            "gateway_url must use https:// (http:// is allowed "
            f"only for localhost); got: {gateway_url!r}"
        )
    parsed = urlparse(gateway_url)
    if not parsed.hostname:
        raise ValueError(
            f"gateway_url must contain a valid hostname, got: {gateway_url!r}"
        )
    allowed_raw = os.getenv(_ALLOWED_HOSTS_ENV)
    if allowed_raw is not None:
        allowed = {h.strip().lower() for h in allowed_raw.split(",") if h.strip()}
    else:
        allowed = {h.lower() for h in _DEFAULT_ALLOWED_GATEWAY_HOSTS}
    # Loopback is always permitted (local dev); "*" is an explicit opt-out that
    # allows any host (e.g. a self-hosted gateway).
    if (
        not is_loopback
        and "*" not in allowed
        and parsed.hostname.lower() not in allowed
    ):
        raise ValueError(
            f"gateway_url hostname {parsed.hostname!r} is not in the allowed "
            f"gateway hosts {sorted(allowed)}. Set {_ALLOWED_HOSTS_ENV} to "
            f"override (comma-separated hosts, or '*' to allow any host)."
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
) -> tuple[str | None, str | None, str | None, str | None]:
    """Fill in None params from FLINTAI_* environment variables or .env file.

    Explicit (non-None) values take precedence over env vars.
    """
    global _dotenv_loaded
    if load_dotenv is not None and not _dotenv_loaded:
        load_dotenv(os.path.join(os.getcwd(), ".env"))
        _dotenv_loaded = True
    if gateway_url is None:
        gateway_url = os.getenv(_ENV_VARS["gateway_url"]) or None
    if api_key is None:
        api_key = os.getenv(_ENV_VARS["api_key"]) or None
    if llm_api_key is None:
        llm_api_key = os.getenv(_ENV_VARS["llm_api_key"]) or None
    if policy_id is None:
        policy_id = os.getenv(_ENV_VARS["policy_id"]) or None
    return gateway_url, api_key, llm_api_key, policy_id


def build_guardrails_config(
    gateway_url: str,
    api_key: str,
    llm_api_key: str | None = None,
    provider: str | None = None,
    policy_id: str | None = None,
) -> GuardrailsConfig:
    """Build a GuardrailsConfig without touching global state."""
    gateway_url = _validate_guardrails_params(gateway_url, api_key)

    if provider is not None:
        path_prefix = PROVIDER_PATH_MAP.get(provider)
        if path_prefix is None:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Supported: {list(PROVIDER_PATH_MAP.keys())}"
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
) -> GuardrailsConfig:
    from flintai import core

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
    )

    if core._client:
        core._client.guardrails_config = config
    else:
        warnings.warn(
            "configure_guardrails() called before flintai.init(). "
            "The config will not be attached to any client. "
            "Call flintai.init() first, or pass this config manually.",
            stacklevel=2,
        )

    return config
