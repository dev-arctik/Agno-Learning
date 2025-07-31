from typing import List
from agno.agent import Agent
from agno.tools import Toolkit
from agno.models.openai import OpenAIChat

from config.secrets import OPENAI_API_KEY

class MathTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(
            name="math_tools", 
            tools=[self.add, self.subtract, self.multiply, self.divide], 
            **kwargs
        )

    def add(self, a: float, b: float) -> float:
        """Add two numbers together.
        
        Args:
            a (float): First number
            b (float): Second number
            
        Returns:
            float: Sum of a and b
        """
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Subtract second number from first number.
        
        Args:
            a (float): First number
            b (float): Second number
            
        Returns:
            float: Difference of a and b
        """
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a (float): First number
            b (float): Second number
            
        Returns:
            float: Product of a and b
        """
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Divide first number by second number.
        
        Args:
            a (float): First number (dividend)
            b (float): Second number (divisor)
            
        Returns:
            float: Quotient of a divided by b
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# Create agent with your math toolkit
agent = Agent(
    name="MathAgent",
    description="An agent that performs mathematical operations.",
    model=OpenAIChat(id="gpt-4.1-mini", temperature=0.5, api_key=OPENAI_API_KEY),
    tools=[MathTools()], 
    instructions=[
        "You are a helpful assistant that performs mathematical calculations.",
        "When asked to perform a calculation, use the appropriate math tool.",
        "Respond with the result of the calculation in a clear and concise manner.",
    ],
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=15,
    markdown=True
)

if __name__ == "__main__":
    print("Starting MathAgent...")
    print("You can ask me to perform mathematical operations like addition, subtraction, multiplication, and division.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting MathAgent. Goodbye!")
            break

        agent.print_response(user_input, stream=True)