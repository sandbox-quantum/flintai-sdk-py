from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from switch_trust.plugins.langchain import LangChainGuardrailsMiddleware

middleware = LangChainGuardrailsMiddleware(load_dotenv=True)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


llm = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    middleware=[middleware],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": "session-456"}},
)
print(result["messages"][-1].content_blocks)
