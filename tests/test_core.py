"""Unit tests for switch_trust.core.SwitchTrustClient."""

from unittest.mock import MagicMock

from switch_trust.core import SwitchTrustClient


def test_construction_defaults():
    client = SwitchTrustClient()
    assert client.plugins == []
    assert client.provider is None
    assert client.guardrails_config is None


def test_construction_with_provider():
    client = SwitchTrustClient(provider="openai")
    assert client.provider == "openai"


def test_register_plugin():
    client = SwitchTrustClient()
    plugin_a = MagicMock()
    plugin_b = MagicMock()

    client.register_plugin(plugin_a)
    plugin_a.on_init.assert_called_once_with(client)

    client.register_plugin(plugin_b)
    assert len(client.plugins) == 2
    assert client.plugins[0] is plugin_a
    assert client.plugins[1] is plugin_b


def test_notify_behavior():
    client = SwitchTrustClient()
    plugin_a = MagicMock()
    plugin_b = MagicMock(spec=[])
    client.plugins = [plugin_a, plugin_b]

    client.notify("on_shutdown")

    plugin_a.on_shutdown.assert_called_once_with()


def test_shutdown_full_behavior():
    from switch_trust.guardrails import GuardrailsConfig

    client = SwitchTrustClient()
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


def test_switch_trust_client_repr():
    from switch_trust.plugins import SwitchTrustPlugin

    client = SwitchTrustClient(provider="openai")
    plugin = MagicMock(spec=SwitchTrustPlugin)
    plugin.name = "mock"
    client.plugins = [plugin]

    r = repr(client)
    assert "provider='openai'" in r
    assert "plugins=1" in r


def test_bare_plugin_noop_methods():
    from switch_trust.plugins import SwitchTrustPlugin

    plugin = SwitchTrustPlugin()
    plugin.on_init(MagicMock())
    plugin.on_shutdown()

    client = SwitchTrustClient()
    client.register_plugin(plugin)
    client.shutdown()


def test_notify_exception_is_caught(caplog):
    import logging

    from switch_trust.plugins import SwitchTrustPlugin

    class FailingPlugin(SwitchTrustPlugin):
        name = "failing"

        def on_shutdown(self):
            raise ValueError("plugin error")

    client = SwitchTrustClient()
    plugin_ok = MagicMock()
    client.plugins = [FailingPlugin(), plugin_ok]

    with caplog.at_level(logging.WARNING, logger="switch_trust.core"):
        client.notify("on_shutdown")

    plugin_ok.on_shutdown.assert_called_once()
    assert "plugin error" in caplog.text
