"""Shared fixtures for FlintAI SDK tests."""

import flintai
import pytest
from flintai import guardrails

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
    flintai.shutdown()
    guardrails._dotenv_loaded = False
