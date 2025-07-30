from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools

from config.secrets import OPENAI_API_KEY

# Website Agent
website_agent = Agent(
    name="Web Search Agent",
    role="Handles web searches and retrieves information from the internet.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are a web search agent.",
        "You can search the web for information and answer questions based on the search results.",
        "You can also answer questions based on the current date.",
        "If you find relevant information, provide it to the user.",
    ],
    add_datetime_to_instructions=True
)

# Wikipedia Agent
wikipedia_agent = Agent(
    name="Wikipedia Agent",
    role="Retrieves information from Wikipedia.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[WikipediaTools()],
    instructions=[
        "You are a Wikipedia agent.",
        "You can search Wikipedia for information and answer questions based on the search results.",
        "If you find relevant information, provide it to the user.",
    ],
    add_datetime_to_instructions=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Retrieves financial information and stock market data.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[YFinanceTools()],
    instructions=[
        "You are a finance agent.",
        "You can search for financial information and stock market data.",
        "If you find relevant information, provide it to the user.",
    ],
    add_datetime_to_instructions=True
)

# Team of Agents with Reasoning
agent_team = Team(
    name="Multi-Agent Team",
    mode="coordinate",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
    ),
    members=[
        website_agent,
        wikipedia_agent,
        finance_agent
    ],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "You are a team of agents working together to answer questions.",
        "Each agent has a specific role and can access different tools.",
        "Coordinate with each other to provide the best answers.",
        "Use reasoning to combine information from different agents if necessary.",
    ],
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    add_datetime_to_instructions=True,
    success_criteria= "Provide accurate and relevant information based on the user's query.",
)


# run the team 
if __name__ == "__main__":
    print("Multi-Agent Team is starting...")

    while True:
        user_input = input("\n\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the Multi-Agent Team.")
            break
        
        agent_team.print_response(user_input, stream=True, show_full_reasoning=True, stream_intermediate_steps=True)