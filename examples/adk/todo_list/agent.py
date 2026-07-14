import asyncio
import logging

from flintai.plugins.adk import ADKGuardrailsPlugin
from google import adk
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Set up logging to catch ADK internal events
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


def add_todo_items(tasks: list[str]):
    """
    Writes a list of detailed TODO tasks to a storage.

    Args:
        tasks (list[str]): a list of detailed tasks to be added.

    """
    print("Items added:", tasks)


def notify_user(message: str):
    """
    Notify user any text messages

    :param message: text message to send to user
    :type message: str
    """
    print("User sees:", message)
    return {"status": "success"}


plugin = ADKGuardrailsPlugin()


root_agent = LlmAgent(
    model="gemini-3-flash-preview",
    name="todo_list_agent",
    description="generate todo list items for the user",
    instruction="""
You are a project manager. When a user gives you a task,
break it down into 5-10 actionable steps.
Once you have the steps, use the 'add_todo_items' tool to save them.
Call the tool with 1 task at a time to optimize updating frequency.

The user will only see messages from you if you notify them via
'notify_user' tool. Notify the user while you are adding tasks.
    """,
    tools=[add_todo_items, notify_user],
    generate_content_config=plugin.content_config,
    before_model_callback=plugin.before_model_callback,
    on_model_error_callback=plugin.on_model_error,
)

if __name__ == "__main__":
    session_service = InMemorySessionService()

    runner = adk.Runner(
        agent=root_agent, app_name="todo_app", session_service=session_service
    )

    async def new_session():
        user_id = "user_123"
        session_id = "session_456"

        await session_service.create_session(
            app_name="todo_app", user_id=user_id, session_id=session_id
        )

    async def run_planner(user_query: str):
        print(f"User: {user_query}\n" + "-" * 20)

        # Create the message content
        content = types.Content(role="user", parts=[types.Part(text=user_query)])

        user_id = "user_123"
        session_id = "session_456"

        # Run the agent (this handles the tool calling automatically)
        events = runner.run(user_id=user_id, session_id=session_id, new_message=content)

        for event in events:
            # The runner streams events (Thinking, Tool Calls, etc.)
            if event.is_final_response():
                print(f"Agent Response: {event.content.parts[0].text}")

    asyncio.run(new_session())
    asyncio.run(run_planner("I need to clean up my garage it's a mess."))
