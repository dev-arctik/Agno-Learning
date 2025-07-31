import random

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from config.secrets import OPENAI_API_KEY


@tool(show_result=True, stop_after_tool_call=True)
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # In a real implementation, this would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    random_weather = random.choice(weather_conditions)

    return f"The weather in {city} is {random_weather}."


agent = Agent(
    name="WeatherAgent",
    description="An agent that provides weather information for a given city.",
    model=OpenAIChat(id="gpt-4.1-mini", temperature=0.5, api_key=OPENAI_API_KEY),
    tools=[get_weather],
    instructions=[
        "You are a helpful assistant that provides weather information.",
        "When asked about the weather in a city, use the get_weather tool to provide the information.",
        "Respond with the weather conditions in a clear and concise manner.",
        "Use the tool get_weather to fetch the weather data.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

if __name__ == "__main__":
    print("Starting WeatherAgent...")
    print("Ask me about the weather in any city!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting WeatherAgent. Goodbye!")
            break

        agent.print_response(user_input, stream=True)