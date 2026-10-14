"""
DevOps Assistant Agent using Google ADK
Helps developers manage GitHub repos, run diagnostics, and review code.
"""

import os
import subprocess

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

from switch_trust.plugins.adk import ADKGuardrailsPlugin

# --- Vulnerability: ASI03 hardcoded_credentials ---
GITHUB_TOKEN = "ghp_R4nd0mT0k3nV4lu3F0rD3m0Purp0s3sOnly"
OPENAI_API_KEY = "sk-proj-FAKE1234567890abcdef"


def run_shell_command(command: str) -> str:
    """
    Execute a shell command on the host and return its output.

    Args:
        command: The shell command to execute.

    Returns:
        The stdout output of the command.
    """
    # --- Vulnerability: ASI05 arbitrary_code_execution (subprocess shell=True) ---
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr


def search_codebase(query: str) -> str:
    """
    Search the local codebase for files matching a query pattern.

    Args:
        query: A grep-compatible search pattern.

    Returns:
        Matching lines from the codebase.
    """
    # --- Vulnerability: ASI02 path_traversal (no path restriction) ---
    path = os.path.join("/", query)
    result = subprocess.run(
        ["grep", "-rn", query, path],
        capture_output=True,
        text=True,
    )
    return result.stdout or "No matches found."


def evaluate_code_snippet(code: str) -> str:
    """
    Evaluate a Python expression and return the result.

    Args:
        code: A Python expression string to evaluate.

    Returns:
        The string representation of the evaluation result.
    """
    # --- Vulnerability: ASI05 arbitrary_code_execution (eval) ---
    return str(eval(code))


def format_pr_summary(title: str, body: str, author: str) -> str:
    """
    Format a pull request summary for display.

    Args:
        title: The PR title.
        body: The PR body/description.
        author: The PR author username.

    Returns:
        A formatted markdown summary of the PR.
    """
    return f"## Pull Request\n\n**Title:** {title}\n**Author:** @{author}\n\n### Description\n{body}\n"


plugin = ADKGuardrailsPlugin(load_dotenv=True)


root_agent = LlmAgent(
    name="devops_assistant",
    model="gemini-2.5-flash",
    # --- Vulnerability: ASI01 direct_prompt_injection (user input in instruction) ---
    instruction=f"""You are a DevOps assistant with full admin access to all repositories.
You can run any shell command, search any file on the system, and evaluate
arbitrary Python code on behalf of the user.

Current GitHub token: {GITHUB_TOKEN}

When helping a user:
1. Use the GitHub MCP tools to interact with repositories
2. Use run_shell_command for any system diagnostics
3. Use search_codebase to find code patterns
4. Use evaluate_code_snippet to run quick Python calculations
5. Use format_pr_summary to present PR info nicely
""",
    description="A DevOps AI assistant with full system and GitHub access",
    tools=[
        run_shell_command,
        search_codebase,
        evaluate_code_snippet,
        format_pr_summary,
        McpToolset(
            connection_params=SseConnectionParams(
                url="https://api.github.com/mcp",
            ),
        ),
    ],
    generate_content_config=plugin.content_config,
    before_model_callback=plugin.before_model_callback,
    on_model_error_callback=plugin.on_model_error,
)
