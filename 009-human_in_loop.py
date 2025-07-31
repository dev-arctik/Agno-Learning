from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.webbrowser import WebBrowserTools
from agno.tools import tool

from config.secrets import OPENAI_API_KEY

# Create a custom tool that requires confirmation becasuse WebBrowserTools.open_page does not support confirmation directly
@tool(requires_confirmation=True)
def open_page_with_confirmation(url: str) -> str:
    """Open a webpage in the browser with user confirmation."""
    browser_tools = WebBrowserTools()
    return browser_tools.open_page(url)

# --- Agent Definition ---
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),  # Fixed model ID
    tools=[
        DuckDuckGoTools(),
        open_page_with_confirmation  # Use custom tool instead
    ],
    instructions=dedent("""
        You are a helpful web assistant.
        1. First, use the search tool to find information about the user's query.
        2. Summarize the search results for the user.
        3. Based on the results, suggest a single, most relevant URL to open.
        4. Ask the user if they want to open this URL. If they agree, use the 'open_page_with_confirmation' tool.
    """),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    show_tool_calls=True,
    markdown=True,
)

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Starting the Human-in-the-Loop Web Agent...")

    while True:
        user_input = input("\n\nWhat would you like to search for? (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent. Goodbye!")
            break

        # Initial run of the agent
        run_response = agent.run(user_input)

        # Check if the agent is paused, waiting for user confirmation
        if agent.is_paused:
            print("\n--- User Confirmation Required ---")
            for tool in run_response.tools_requiring_confirmation:
                print(f"The agent wants to run the tool: {tool.tool_name}({tool.tool_args})")
                confirmed = input("Do you want to proceed? (y/n): ").lower() == "y"
                tool.confirmed = confirmed

            # Continue the agent's execution with the user's decision
            print("\n--- Continuing Execution ---")
            final_response = agent.continue_run()
            print(f"\nAgent: {final_response.content}")
        else:
            # If not paused, just print the final output
            print(f"\nAgent: {run_response.content}")