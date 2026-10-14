# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file covers two package names: `switch-trust-sdk-py`, the current name,
and `flintai-sdk-py`, the retired name it was published under for versions
1.0.0 to 1.0.2. Versioning starts at 1.1.0 under the new name.

## switch-trust-sdk-py

### 1.1.0 - 2026-10-14

#### Security

- In `flintai-sdk-py` 1.0.2 and earlier, a `.env` file in the working directory
  was loaded automatically whenever `python-dotenv` was installed, and the `*`
  allowlist wildcard or a loopback gateway URL was accepted without an in-code
  opt-in. Either could redirect the Switch Trust API key, any forwarded provider
  API key, and prompt and response traffic to an unintended host. `.env` is now
  read only when `load_dotenv` is set, and the wildcard and loopback hosts
  (`localhost`, `127.0.0.1`, `::1`) require `allow_insecure_gateway=True` in
  code. This fix is not available under the old package name.

#### Changed

- Renamed the distribution `flintai-sdk-py` to `switch-trust-sdk-py`, and the
  import package `flintai` to `switch_trust`.
- Renamed `FlintAIClient`, `FlintAIPlugin` and `FlintAIGuardrailsError` to
  `SwitchTrustClient`, `SwitchTrustPlugin` and `SwitchTrustGuardrailsError`.
- Environment variables are now `SWITCH_TRUST_GATEWAY_URL`,
  `SWITCH_TRUST_API_KEY`, `SWITCH_TRUST_LLM_API_KEY`, `SWITCH_TRUST_POLICY_ID`
  and `SWITCH_TRUST_ALLOWED_GATEWAY_HOSTS`. The matching `FLINTAI_*` variables
  are still read when the new ones are unset.
- `.env` is no longer loaded automatically; pass `load_dotenv=True` or a path.
- The default gateway allowlist now contains `api.flintai.dev` as well as
  `app.flintai.dev`.

#### Added

- `dangerously_disable_guardrails` as an explicit opt-out. Any guardrails
  opt-out logs a `SECURITY CONTROL DISABLED` warning once per process.
- `status()` and `client.guardrails_status()` report the effective guardrails
  posture.
- The ADK plugin and LangChain middleware send an `X-Agent-Turn-Id` header.

#### Fixed

- The ADK plugin no longer carries stale session metadata across requests.
- The LangChain middleware sends `X-Agent-Name` from `AGENT_NAME`, as the ADK
  plugin does, so an agent first seen through LangChain gets a name.
- Failures while restoring wrapped clients on shutdown are now surfaced.
- `wrap()` sends `X-Agent-Id` and `X-Agent-Name` from `AGENT_ID` and
  `AGENT_NAME`, as the ADK plugin and LangChain middleware do. Before, `wrap()`
  traffic reached the gateway with no agent identity. The gateway can now
  attribute that traffic, so `wrap()` agents count toward agents under
  management (AUM). The SDK reads both variables when it builds the config, in
  `init()` or `wrap()`, not on each request. A value that is not ASCII is
  skipped with a warning, because HTTP header values must be ASCII. An identity
  header that the client already sets, for example through `default_headers`,
  takes precedence, and `shutdown()` leaves it in place.

#### Deprecated

- `flintai-sdk-py` is retired and receives no further releases, including
  security fixes. Its releases are yanked on PyPI and the project is archived.
  Use `switch-trust-sdk-py`.

## flintai-sdk-py (retired)

### flintai-sdk-py 1.0.2 - 2026-07-17

#### Changed

- The ADK plugin and LangChain middleware send `X-Agent-Id` only when the
  `AGENT_ID` environment variable is set. It no longer falls back to the plugin
  name (`adk-guardrails` or `langchain-guardrails`).

### flintai-sdk-py 1.0.1 - 2026-07-15

#### Changed

- Documentation: docstring examples use `https://app.flintai.dev` instead of a
  placeholder host.

### flintai-sdk-py 1.0.0 - 2026-07-15

#### Added

- First public release. `wrap()` routes OpenAI, Anthropic, Google GenAI and
  LangChain chat model clients through the guardrails proxy, with a Google ADK
  plugin and a plugin system for extensions.
