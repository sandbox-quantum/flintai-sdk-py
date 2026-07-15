"""Shared fixtures for FlintAI SDK tests."""

import pytest

import flintai
from flintai import guardrails

_FLINTAI_ENV_VARS = [
    "FLINTAI_GATEWAY_URL",
    "FLINTAI_API_KEY",
    "FLINTAI_LLM_API_KEY",
    "FLINTAI_POLICY_ID",
    "FLINTAI_ALLOWED_GATEWAY_HOSTS",
]


@pytest.fixture(autouse=True)
def _reset_flintai(monkeypatch):
    for var in _FLINTAI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Most tests use example.com gateway hosts; allow any host by default so the
    # default allowlist doesn't reject them. Related tests override or delete this.
    monkeypatch.setenv("FLINTAI_ALLOWED_GATEWAY_HOSTS", "*")
    guardrails._dotenv_loaded = True
    yield
    flintai.shutdown()
    guardrails._dotenv_loaded = False
