"""Shared fixtures for Switch Trust SDK tests."""

import pytest

import switch_trust
from switch_trust import guardrails

_SWITCH_TRUST_ENV_VARS = [
    "SWITCH_TRUST_GATEWAY_URL",
    "SWITCH_TRUST_API_KEY",
    "SWITCH_TRUST_LLM_API_KEY",
    "SWITCH_TRUST_POLICY_ID",
    "SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS",
    "FLINTAI_GATEWAY_URL",
    "FLINTAI_API_KEY",
    "FLINTAI_LLM_API_KEY",
    "FLINTAI_POLICY_ID",
    "FLINTAI_ALLOWED_GATEWAY_HOSTS",
    "AGENT_ID",
    "AGENT_NAME",
]


@pytest.fixture(autouse=True)
def _reset_switch_trust(monkeypatch):
    for var in _SWITCH_TRUST_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Most tests use example.com gateway hosts. Allowlist them explicitly so the
    # default allowlist doesn't reject them. We deliberately avoid "*" here: the
    # wildcard now requires allow_insecure_gateway=True and would otherwise make
    # every test raise. Tests that exercise "*" or loopback set their own env /
    # pass allow_insecure_gateway=True.
    monkeypatch.setenv(
        "SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS",
        "api.flintai.dev,gw.example.com,gw1.example.com,gw2.example.com,"
        "custom.host,any-host.example.com,gw.corp.com,gw.env.com,"
        "guardrails.env.com,explicit.com",
    )
    guardrails._dotenv_loaded = True
    guardrails.reset_opt_out_warned()
    yield
    switch_trust.shutdown()
    guardrails._dotenv_loaded = False
    guardrails.reset_opt_out_warned()
