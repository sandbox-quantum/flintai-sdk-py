# FlintAI SDK (Python)

Lightweight SDK for routing LLM traffic through a guardrails proxy. Wraps OpenAI, Anthropic, Google GenAI, and LangChain LLM clients, with agent framework plugins for Google ADK.

## Features

- **LLM SDK wrapping** — route LLM traffic through the FlintAI guardrails proxy with a single `flintai.wrap()` call. Supports OpenAI, Anthropic, Google GenAI, and LangChain chat models.
- **Agent framework plugins** — deeper integration with agent lifecycle hooks for metadata extraction and guardrails routing. Currently supports Google ADK.
- **Plugin system** for extensibility

## Supported Integrations

| Integration | Category | Guardrails Routing |
|---|---|---|
| OpenAI SDK | LLM SDK | `flintai.wrap()` |
| Anthropic SDK | LLM SDK | `flintai.wrap()` |
| Google GenAI SDK | LLM SDK | `flintai.wrap()` |
| LangChain (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`) | LLM SDK | `flintai.wrap()` |
| Google ADK | Agent Framework | `ADKGuardrailsPlugin` |

## Install

```bash
pip install flintai-sdk-py

# With optional extras for your LLM provider
pip install "flintai-sdk-py[openai]"     # OpenAI SDK
pip install "flintai-sdk-py[anthropic]"  # Anthropic SDK
pip install "flintai-sdk-py[genai]"      # Google GenAI SDK
pip install "flintai-sdk-py[adk]"        # Google ADK (includes GenAI)
pip install "flintai-sdk-py[langchain]"  # LangChain integrations
pip install "flintai-sdk-py[all]"        # Everything
```

## Guardrails

Route all LLM traffic through the FlintAI guardrails proxy. Create your sync client as usual, then call `flintai.wrap()` — it auto-detects the provider and applies guardrails routing. Async clients (`AsyncOpenAI`, `AsyncAnthropic`) are not supported.

### OpenAI

```python
import openai
import flintai

client = openai.OpenAI(api_key="your-openai-api-key")
client = flintai.wrap(
    client,
    gateway_url="https://app.flintai.dev",
    api_key="your-guardrails-api-key",
    policy_id="your-policy-id",  # optional
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Anthropic

```python
import anthropic
import flintai

client = anthropic.Anthropic(api_key="your-anthropic-api-key")
client = flintai.wrap(
    client,
    gateway_url="https://app.flintai.dev",
    api_key="your-guardrails-api-key",
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Google GenAI

Google GenAI clients don't expose `api_key` as an attribute, so `llm_api_key` must be passed explicitly:

```python
import google.genai
import flintai

client = google.genai.Client(api_key="your-gemini-api-key")
client = flintai.wrap(
    client,
    gateway_url="https://app.flintai.dev",
    api_key="your-guardrails-api-key",
    llm_api_key="your-gemini-api-key",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello",
)
```

### LangChain

Create your LangChain chat model as usual, then call `flintai.wrap()` — it detects the LangChain model, finds the underlying SDK client, and applies guardrails routing:

```python
from langchain_openai import ChatOpenAI
import flintai

llm = ChatOpenAI(model="gpt-4", api_key="your-openai-api-key")
llm = flintai.wrap(
    llm,
    gateway_url="https://app.flintai.dev",
    api_key="your-guardrails-api-key",
)

response = llm.invoke("Hello")
```

Works with `ChatOpenAI`, `ChatAnthropic`, and `ChatGoogleGenerativeAI`.

### Google ADK

ADK agents lazily create their GenAI client at runtime and use `generate_content_config` for per-request routing, so `flintai.wrap()` cannot be used. Use the ADK plugin instead:

```python
from flintai.plugins.adk import ADKGuardrailsPlugin
from google.adk import Agent

plugin = ADKGuardrailsPlugin(
    gateway_url="https://app.flintai.dev",
    api_key="your-guardrails-api-key",
    llm_api_key="your-gemini-api-key",
)

agent = Agent(
    model="gemini-2.5-flash",
    generate_content_config=plugin.content_config,
    before_model_callback=plugin.before_model_callback,
    on_model_error_callback=plugin.on_model_error,
)
```

`before_model_callback` automatically extracts agent identity and session metadata on each LLM call — it reads the agent name from the ADK callback context and attaches `X-Agent-Id` and `X-Agent-Session-Id` headers to the guardrails request. It fails closed: if guardrails routing was never applied (the request has no `http_options`), it raises `FlintAIGuardrailsError` rather than sending an unguarded request — pass `require_guardrails=False` to the plugin for best-effort behavior. `on_model_error_callback` converts guardrails blocks (detected via the `GUARDRAIL_BLOCKED` error code) into an `LlmResponse` so the agent can handle them gracefully.

## Configuration Reference

`flintai.wrap()` and `flintai.init()` accept these guardrails parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gateway_url` | `str` | Guardrails proxy URL |
| `api_key` | `str` | Your guardrails API key |
| `llm_api_key` | `str \| None` | Your upstream LLM provider API key, forwarded to the gateway as `X-LLM-API-Key`. Optional — by default the gateway supplies its own upstream credentials, so the key is not sent. |
| `forward_llm_key` | `bool` | `wrap()` only. When `True`, auto-extract the provider key from the client (`client.api_key`) and forward it. Default `False` (opt-in) — the caller's key is not sent to the gateway unless requested. |
| `policy_id` | `str \| None` | Guardrails policy ID to apply (optional) |

**Fail-closed by default:** `require_guardrails` defaults to `True` on `init()`, `wrap()`, and the plugins. If guardrails can't be applied — missing config, an unrecognized client, or a plugin used standalone (e.g. passed straight to `create_agent`/`Agent` without `flintai.init()`) — a `FlintAIGuardrailsError` is raised instead of sending traffic unguarded. Pass `require_guardrails=False` to opt out.

## Environment Variables

Instead of passing credentials in code, you can set them as environment variables:

| Environment Variable | Parameter | Description |
|---|---|---|
| `FLINTAI_GATEWAY_URL` | `gateway_url` | Guardrails proxy URL |
| `FLINTAI_API_KEY` | `api_key` | Your guardrails API key |
| `FLINTAI_LLM_API_KEY` | `llm_api_key` | Your upstream LLM provider API key |
| `FLINTAI_POLICY_ID` | `policy_id` | Guardrails policy ID (optional) |
| `AGENT_ID` | `agent_id` | Agent identifier attached to guardrails requests (overrides `agent_name`) |
| `FLINTAI_ALLOWED_GATEWAY_HOSTS` | — | Comma-separated allowlist of permitted gateway hostnames. Defaults to `app.flintai.dev`; set to override (or `*` to allow any host). Loopback is always allowed. |

Set the variables in your shell or in a `.env` file (requires `pip install "flintai-sdk-py[dotenv]"`):

```bash
# .env
FLINTAI_GATEWAY_URL=https://app.flintai.dev
FLINTAI_API_KEY=your-guardrails-api-key
FLINTAI_LLM_API_KEY=your-llm-api-key
```

Then use the FlintAI SDK without passing credentials:

```python
import openai
import flintai

client = openai.OpenAI()
client = flintai.wrap(client)  # reads from env vars / .env file
```

**Precedence:** Explicit parameters > environment variables > auto-extract from `client.api_key` (only in `wrap()`, and only when `forward_llm_key=True`).

**Multiple provider keys:** If multiple provider API keys are set in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`), auto-detection will fail. Pass `provider` explicitly to `flintai.init()` to disambiguate.

## Gateway Allowlist

By default, `gateway_url` must resolve to `app.flintai.dev` (the hosted FlintAI guardrails gateway); any other host is rejected so provider keys can't be sent to an unintended endpoint. To use a self-hosted or alternate gateway, set `FLINTAI_ALLOWED_GATEWAY_HOSTS` to a comma-separated list of allowed hostnames (or `*` to disable the check). Loopback hosts (`localhost`, `127.0.0.1`, `::1`) are always allowed for local development. Plaintext `http://` is rejected except for loopback.

## How It Works

1. `flintai.wrap(client, gateway_url=..., ...)` auto-detects the provider from the client type, computes the provider-specific path prefix (`/openai`, `/anthropic`, `/gemini`), rewrites the client's base URL, and injects custom headers (`X-FlintAI-API-Key`, `X-Guardrails-Policy-Id`, and `X-LLM-API-Key` only when a provider key is provided or `forward_llm_key=True`)
2. The guardrails proxy intercepts the request, applies policy checks (detectors), and strips the custom headers
3. The proxy injects the appropriate upstream auth credentials and forwards the request to the LLM provider

**Global state:** Each `flintai.wrap()` call updates the global FlintAI SDK client's guardrails config. If you wrap multiple clients with different parameters, each client retains its own headers and base URL, but only the last `wrap()` call's config is stored globally. For most applications (single provider, shared credentials), this is transparent.

## Known Limitations

**Private attribute mutation:** `flintai.wrap()` modifies internal attributes of LLM SDK clients (`_base_url`, `_custom_headers`, `_api_client._http_options`) to redirect traffic through the guardrails proxy. These are not part of the public API of the respective SDKs and may change without notice. Pin your SDK versions to the tested ranges:

- `openai>=2.40.0,<3`
- `anthropic>=0.105.2,<1`
- `google-genai>=2.7,<3`

**Google GenAI `api_key`:** Google GenAI clients do not expose `api_key` as an attribute. You must pass `llm_api_key=` explicitly when calling `flintai.wrap()`.

**Google GenAI URL normalization:** The Google GenAI SDK requires a trailing slash on the base URL. The FlintAI SDK normalizes this automatically — do not add a trailing slash to `gateway_url`.

**Async clients:** Async clients (`AsyncOpenAI`, `AsyncAnthropic`) are not supported. Use sync clients only.

**Thread safety:** `flintai.init()`, `flintai.wrap()`, and `flintai.shutdown()` are not thread-safe. Call them from the main thread during application startup. Once initialized, wrapped clients can be used from any thread — the underlying SDK clients handle their own thread safety.

## Plugins

Plugins handle events from the FlintAI SDK lifecycle. Subclass `FlintAIPlugin` and override the methods you care about:

```python
from flintai.plugins import FlintAIPlugin

class MyPlugin(FlintAIPlugin):
    name = "my-plugin"

    def on_init(self, client):
        print(f"Plugin initialized with {client}")

    def on_shutdown(self):
        print("Shutting down")

# Register after init
flintai.init()
flintai.register_plugin(MyPlugin())
```

### Plugin Methods

| Method | Called when |
|--------|-----------|
| `on_init(client)` | Plugin is registered |
| `on_shutdown()` | `flintai.shutdown()` is called |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
