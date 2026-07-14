"""Guardrails proxy configuration."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:
    find_dotenv = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]

__all__ = [
    "GuardrailsConfig",
    "PROVIDER_PATH_MAP",
    "build_guardrails_config",
    "check_optional_guardrails_params",
    "configure_guardrails",
    "detect_provider",
    "resolve_from_env",
]

_dotenv_loaded = False

PROVIDER_PATH_MAP = {
    "openai": "/openai",
    "google": "/gemini",
    "anthropic": "/anthropic",
}


@dataclass
class GuardrailsConfig:
    base_url: str
    headers: dict[str, str]
    provider: str | None
    gateway_url: str
    policy_id: str | None = None

    def clear(self) -> None:
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


def _validate_guardrails_params(
    gateway_url: str, api_key: str, llm_api_key: str
) -> str:
    """Validate raw guardrails params. Returns normalized gateway_url."""
    if not gateway_url or not gateway_url.startswith(("http://", "https://")):
        raise ValueError(
            f"gateway_url must start with http:// or https://, got: {gateway_url!r}"
        )
    if not api_key:
        raise ValueError("api_key must be a non-empty string")
    if not llm_api_key:
        raise ValueError("llm_api_key must be a non-empty string")
    return gateway_url.rstrip("/")


def check_optional_guardrails_params(
    gateway_url: str | None,
    api_key: str | None,
    llm_api_key: str | None,
) -> bool:
    """Return True if guardrails params are provided, False if all are None.

    Raises ValueError if only some params are provided.
    """
    if gateway_url is None and api_key is None and llm_api_key is None:
        return False
    missing = [
        name
        for name, val in [
            ("gateway_url", gateway_url),
            ("api_key", api_key),
            ("llm_api_key", llm_api_key),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            "gateway_url, api_key, and llm_api_key are all required "
            "when providing guardrails config."
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
        load_dotenv(find_dotenv(usecwd=True))
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
    llm_api_key: str,
    provider: str | None = None,
    policy_id: str | None = None,
) -> GuardrailsConfig:
    """Build a GuardrailsConfig without touching global state."""
    gateway_url = _validate_guardrails_params(gateway_url, api_key, llm_api_key)

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

    headers = {
        "X-FlintAI-API-Key": api_key,
        "X-LLM-API-Key": llm_api_key,
    }

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
    llm_api_key: str,
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
