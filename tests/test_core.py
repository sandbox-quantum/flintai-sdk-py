"""Unit tests for flintai.core.FlintAIClient."""

from unittest.mock import MagicMock

from flintai.core import FlintAIClient


def test_construction_defaults():
    client = FlintAIClient()
    assert client.plugins == []
    assert client.provider is None
    assert client.guardrails_config is None


def test_construction_with_provider():
    client = FlintAIClient(provider="openai")
    assert client.provider == "openai"


def test_register_plugin():
    client = FlintAIClient()
    plugin_a = MagicMock()
    plugin_b = MagicMock()

    client.register_plugin(plugin_a)
    plugin_a.on_init.assert_called_once_with(client)

    client.register_plugin(plugin_b)
    assert len(client.plugins) == 2
    assert client.plugins[0] is plugin_a
    assert client.plugins[1] is plugin_b


def test_notify_behavior():
    client = FlintAIClient()
    plugin_a = MagicMock()
    plugin_b = MagicMock(spec=[])
    client.plugins = [plugin_a, plugin_b]

    client.notify("on_shutdown")

    plugin_a.on_shutdown.assert_called_once_with()


def test_shutdown_full_behavior():
    from flintai.guardrails import GuardrailsConfig

    client = FlintAIClient()
    plugin = MagicMock()
    client.plugins = [plugin]
    client.guardrails_config = GuardrailsConfig(
        base_url="https://gw.example.com/openai",
        headers={
            "X-FlintAI-API-Key": "grl_sk_1234567890",
            "X-LLM-API-Key": "sk-proj-abc123def456",
        },
        provider="openai",
        gateway_url="https://gw.example.com",
    )
    client.shutdown()
    plugin.on_shutdown.assert_called_once_with()
    assert client.guardrails_config is None


def test_flintai_client_repr():
    from flintai.plugins import FlintAIPlugin

    client = FlintAIClient(provider="openai")
    plugin = MagicMock(spec=FlintAIPlugin)
    plugin.name = "mock"
    client.plugins = [plugin]

    r = repr(client)
    assert "provider='openai'" in r
    assert "plugins=1" in r


def test_bare_plugin_noop_methods():
    from flintai.plugins import FlintAIPlugin

    plugin = FlintAIPlugin()
    plugin.on_init(MagicMock())
    plugin.on_shutdown()

    client = FlintAIClient()
    client.register_plugin(plugin)
    client.shutdown()


def test_notify_exception_is_caught(caplog):
    import logging

    from flintai.plugins import FlintAIPlugin

    class FailingPlugin(FlintAIPlugin):
        name = "failing"

        def on_shutdown(self):
            raise ValueError("plugin error")

    client = FlintAIClient()
    plugin_ok = MagicMock()
    client.plugins = [FailingPlugin(), plugin_ok]

    with caplog.at_level(logging.WARNING, logger="flintai.core"):
        client.notify("on_shutdown")

    plugin_ok.on_shutdown.assert_called_once()
    assert "plugin error" in caplog.text
