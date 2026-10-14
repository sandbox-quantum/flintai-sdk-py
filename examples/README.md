# Example agents

## Env setup

- `python3.13 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install -e ..`
- Configure a `.env` file with `SWITCH_TRUST_GATEWAY_URL`, `SWITCH_TRUST_API_KEY`, and `SWITCH_TRUST_LLM_API_KEY` (see `../.env.example`). These examples read it because they pass `load_dotenv=True`; the SDK does not read `.env` unless asked (export the vars in your shell to skip the file).
- To attach the runs to an agent you added in the platform, also set `AGENT_ID` (the agent's `agn_…` id). The agent keeps the name you gave it there; `AGENT_NAME` only names an agent the platform hasn't seen before.

## ADK

- `cd adk`
- Configure an `.env` file that contains `SWITCH_TRUST_GATEWAY_URL`, `SWITCH_TRUST_API_KEY`, and `SWITCH_TRUST_LLM_API_KEY` per README instructions of the Switch Trust SDK.
- `adk web . --port 8501`
- Try interacting with the agents in the ADK UI, everything should flow through the AI gateway stack.

## LangChain

- `cd langchain`
- Configure an `.env` file that contains `SWITCH_TRUST_GATEWAY_URL`, `SWITCH_TRUST_API_KEY`, and `SWITCH_TRUST_LLM_API_KEY` per README instructions of the Switch Trust SDK.
- Simple agent use case: `python weather/weather.py`
