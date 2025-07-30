from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.tools.webbrowser import WebBrowserTools
from agno.tools.duckduckgo import DuckDuckGoTools

from config.secrets import OPENAI_API_KEY

memory = Memory(
    # Use any model for creating and managing memories
    model=OpenAIChat(
        id="gpt-4.1-mini",
    ),
    # Store memories in a SQLite database
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db"),
    # We disable deletion by default, enable it if needed
    # delete_memories=True,
    # clear_memories=True,
)

agent = Agent(
    name="Web Browser Agent",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.5,
        api_key=OPENAI_API_KEY,
    ),
    tools=[
        WebBrowserTools(),
        DuckDuckGoTools(),
    ],
    instructions=[
        "You are a web browser agent.",
        "You can open web pages and search the web.",
        "You can answer questions based on the current conversation.",
        "You can also answer questions based on the current date.",
        "You search for information on the web for user queries. Then ask the user if they want to open the page in a web browser, based on the answer open the page in a new window or tab.",
        "If the user wants to open the page, use the WebBrowserTools to do so.",
        "If the user does not want to open the page, just answer the question based on the search results.",
    ],
    show_tool_calls=True,
    memory=memory,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    print("Starting the Web Browser Agent...")

    agent.user_id="temporary_user_id"  # Set a temporary user ID for the session
    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent.")
            break

        agent.print_response(user_input, show_tool_calls=True, stream=True, stream_intermediate_steps=True)