from typing import List
from rich.pretty import pprint
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from config.secrets import OPENAI_API_KEY

# --- Pydantic Model for Structured Output ---
class Joke(BaseModel):
    """Defines the structure for a joke."""
    topic: str = Field(..., description="The topic the joke is about.")
    category: str = Field(..., description="The category of the joke, e.g., Pun, Dad Joke, Knock-knock.")
    setup: str = Field(..., description="The setup or question part of the joke.")
    punchline: str = Field(..., description="The punchline or answer part of the joke.")

# --- Agent Definition ---
# This agent is configured to always return a response that matches the Joke model.
structured_joke_agent = Agent(
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    description="You are a world-class comedian who tells jokes based on a given topic.",
    # Set the response model to our Pydantic class
    response_model=Joke,
    # use_json_mode=True, # JSON mode works with models that don't support structured outputs
    markdown=True,
)


# --- Main Execution Loop ---
if __name__ == "__main__":
    print("Starting the Structured Joke Agent ...")
    print("I can tell you a joke on any topic using the structured output agent.")

    while True:
        user_input = input("\n\nWhat topic do you want a joke about? (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the agents. Hope you had a laugh!")
            break

        print("\n" + "="*60)
        print("STRUCTURED JOKE AGENT (response_model)")
        print("="*60)
        
        # Get the structured response from the first agent
        response = structured_joke_agent.run(user_input)

        # The output is a Pydantic model instance
        joke_object = response.content

        print("\nStructured joke object:")
        pprint(joke_object)

        print("\nThe joke:")
        print(f"Topic: {joke_object.topic}")
        print(f"Category: {joke_object.category}")
        print(f"Setup: {joke_object.setup}")
        print(f"Punchline: {joke_object.punchline}")