# Example agents

## Env setup

- `python3.13 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install -e ..`
- Configure a `.env` file with `FLINTAI_GATEWAY_URL`, `FLINTAI_API_KEY`, and `FLINTAI_LLM_API_KEY` (see `../.env.example`)

## ADK

- `cd adk`
- Configure an `.env` file that contains `FLINTAI_GATEWAY_URL`, `FLINTAI_API_KEY`, and `FLINTAI_LLM_API_KEY` per README instructions of the FlintAI SDK.
- `adk web . --port 8501`
- Try interacting with the agents in the ADK UI, everything should flow through the AI gateway stack.

## LangChain

- `cd langchain`
- Configure an `.env` file that contains `FLINTAI_GATEWAY_URL`, `FLINTAI_API_KEY`, and `FLINTAI_LLM_API_KEY` per README instructions of the FlintAI SDK.
- Simple agent use case: `python weather/weather.py`
