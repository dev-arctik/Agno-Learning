from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage

from config.secrets import OPENAI_API_KEY

# --- Tool Definitions ---
def add_item(agent: Agent, item: str) -> str:
    """Add an item to the shopping list."""
    if item.lower() not in [i.lower() for i in agent.session_state["shopping_list"]]:
        agent.session_state["shopping_list"].append(item)
        return f"Added '{item}' to the shopping list."
    else:
        return f"'{item}' is already in the shopping list."

def remove_item(agent: Agent, item: str) -> str:
    """Remove an item from the shopping list by name."""
    # Case-insensitive search
    for i, list_item in enumerate(agent.session_state["shopping_list"]):
        if list_item.lower() == item.lower():
            agent.session_state["shopping_list"].pop(i)
            return f"Removed '{list_item}' from the shopping list."
    return f"'{item}' was not found in the shopping list."

def rename_item(agent: Agent, old_item: str, new_item: str) -> str:
    """Rename an item in the shopping list."""
    for i, list_item in enumerate(agent.session_state["shopping_list"]):
        if list_item.lower() == old_item.lower():
            agent.session_state["shopping_list"][i] = new_item
            return f"Renamed '{old_item}' to '{new_item}'."
    return f"'{old_item}' was not found in the shopping list."

def list_items(agent: Agent) -> str:
    """List all items in the shopping list."""
    shopping_list = agent.session_state["shopping_list"]
    if not shopping_list:
        return "The shopping list is empty."
    items_text = "\n".join([f"- {item}" for item in shopping_list])
    return f"Current shopping list:\n{items_text}"

# --- Agent Definition ---
agent = Agent(
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    session_id="shopping_list_demo",
    # AGENT STATE is stored in session_state itself, which is a dictionary
    session_state={"shopping_list": []},
    tools=[add_item, remove_item, rename_item, list_items],
    storage=SqliteStorage(table_name="shopping_list_sessions", db_file="tmp/shopping_list.db"),
    instructions=('''
        You are a shopping list assistant.
        You can add, remove, rename, and list items in the shopping list.
        
        Current shopping list: {shopping_list}
    '''),
    add_state_in_messages=True,
    markdown=True,
)

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Starting the Shopping List Agent...")
    print(f"Current list: {agent.session_state['shopping_list']}")

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent. Your list is saved.")
            break

        agent.print_response(user_input, stream=True)
        print(f"\nSession state: {agent.session_state}")