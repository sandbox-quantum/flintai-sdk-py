from flintai.plugins.langchain import LangChainGuardrailsMiddleware
from langchain.agents import create_agent

middleware = LangChainGuardrailsMiddleware()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    middleware=[middleware],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config={"configurable": {"thread_id": "session-456"}},
)
print(result["messages"][-1].content_blocks)
