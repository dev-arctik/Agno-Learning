from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType

from config.secrets import OPENAI_API_KEY

# create knowledge base
knowledgebase = UrlKnowledge(
    urls=["https://docs.agno.com/llms-full.txt"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="knowledgebase",
        search_type=SearchType.hybrid,
        # embedding using OpenAI
        embedder=OpenAIEmbedder(
            id="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
            dimensions=1536
        ),
    ),
)

# Store agent session in SQLite database
storage = SqliteStorage(table_name="001_first_agent", db_file="tmp/001_first_agent.db")

agent = Agent(
    name="Simple Rag Agent",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.5,
        api_key=OPENAI_API_KEY,
    ),
    instructions=[
        "Your name is Amy. You are the supreme RAG agent.",
        "You are able to answer questions based on the knowledge base.",
        "You can also answer questions based on the current conversation.",
        "You can also answer questions based on the current date.",
    ],
    knowledge=knowledgebase,
    storage=storage,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    markdown=True,
)

# Run the agent
if __name__ == "__main__":
    print("Starting the agent...")
    
    agent.knowledge.load(recreate=False)  # Load knowledge base from LanceDB

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agent.")
            break
        agent.print_response(message=user_input, stream=True)