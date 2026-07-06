"""Shared fixtures for FlintAI SDK tests."""

import flintai_sdk
import pytest
from flintai_sdk import guardrails

_FLINTAI_ENV_VARS = [
    "FLINTAI_GATEWAY_URL",
    "FLINTAI_API_KEY",
    "FLINTAI_LLM_API_KEY",
    "FLINTAI_POLICY_ID",
]


@pytest.fixture(autouse=True)
def _reset_flintai(monkeypatch):
    for var in _FLINTAI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    guardrails._dotenv_loaded = True
    yield
    flintai_sdk.shutdown()
    guardrails._dotenv_loaded = False
