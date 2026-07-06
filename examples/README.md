# Example agents

## Env setup

- `python3.13 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `bazel build //sdk/flintai-sdk-py:flintai_sdk_wheel`
- `pip install ../../../bazel-bin/sdk/flintai-sdk-py/flintai_sdk-0.1.0-py3-none-any.whl`
- Bring up the Envoy+Guardrails+ExtProc stack

## ADK

- `cd adk`
- Configure an `.env` file that contains `FLINTAI_GATEWAY_URL`, `FLINTAI_API_KEY`, and `FLINTAI_LLM_API_KEY` per README instructions of the FlintAI SDK.
- `adk web . --port 8501`
- Try interacting with the agents in the ADK UI, everything should flow through the AI gateway stack.
