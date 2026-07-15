"""Integration tests verifying private-attribute compatibility with real provider SDKs.

These tests import real provider SDK clients and verify that ``_get_sdk_headers``
can locate their internal headers dict.  They are skipped automatically when the
provider SDK is not installed (run ``pip install flintai-sdk-py[all]`` to enable).
"""

from __future__ import annotations

import pytest

from flintai.plugins._llm_wrapper import _get_sdk_headers

# --- OpenAI ---


openai = pytest.importorskip("openai", reason="openai SDK not installed")


class TestOpenAICompat:
    def test_get_sdk_headers_returns_dict(self):
        client = openai.OpenAI(api_key="sk-test")
        headers = _get_sdk_headers(client, "openai")
        assert isinstance(headers, dict)

    def test_headers_dict_is_mutable(self):
        client = openai.OpenAI(api_key="sk-test")
        headers = _get_sdk_headers(client, "openai")
        headers["X-Test"] = "value"
        assert _get_sdk_headers(client, "openai")["X-Test"] == "value"


# --- Anthropic ---


anthropic = pytest.importorskip("anthropic", reason="anthropic SDK not installed")


class TestAnthropicCompat:
    def test_get_sdk_headers_returns_dict(self):
        client = anthropic.Anthropic(api_key="sk-test")
        headers = _get_sdk_headers(client, "anthropic")
        assert isinstance(headers, dict)

    def test_headers_dict_is_mutable(self):
        client = anthropic.Anthropic(api_key="sk-test")
        headers = _get_sdk_headers(client, "anthropic")
        headers["X-Test"] = "value"
        assert _get_sdk_headers(client, "anthropic")["X-Test"] == "value"


# --- Google GenAI ---


genai = pytest.importorskip("google.genai", reason="google-genai SDK not installed")


class TestGoogleGenAICompat:
    def test_get_sdk_headers_returns_dict(self):
        client = genai.Client(api_key="fake-key")
        headers = _get_sdk_headers(client, "google")
        assert isinstance(headers, dict)

    def test_headers_dict_is_mutable(self):
        client = genai.Client(api_key="fake-key")
        headers = _get_sdk_headers(client, "google")
        headers["X-Test"] = "value"
        assert _get_sdk_headers(client, "google")["X-Test"] == "value"
