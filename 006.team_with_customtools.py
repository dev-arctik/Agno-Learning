from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.tools import Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.reasoning import ReasoningTools

from config.secrets import OPENAI_API_KEY

# --- Custom Math Tools ---
class MathTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(
            name="math_tools",
            tools=[self.add, self.subtract, self.multiply, self.divide],
            **kwargs
        )

    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Subtract second number from first number."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Divide first number by second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

# --- Teacher Agents ---

# Math Teacher Agent
math_teacher = Agent(
    name="Math Teacher",
    role="Solves mathematical problems and explains mathematical concepts.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[MathTools()],
    instructions=[
        "You are an expert math teacher.",
        "You solve math problems step-by-step and explain the reasoning.",
        "Use your tools to perform calculations and provide clear explanations.",
    ],
    add_datetime_to_instructions=True
)

# Science Teacher Agent
science_teacher = Agent(
    name="Science Teacher",
    role="Answers questions about scientific topics using reliable sources.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[WikipediaTools(), DuckDuckGoTools()],
    instructions=[
        "You are a knowledgeable science teacher.",
        "You answer questions about biology, chemistry, physics, and other scientific fields.",
        "Use your tools to find accurate information and cite your sources.",
    ],
    add_datetime_to_instructions=True
)

# History Teacher Agent
history_teacher = Agent(
    name="History Teacher",
    role="Provides information and explanations about historical events and figures.",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    tools=[WikipediaTools(), DuckDuckGoTools()],
    instructions=[
        "You are an engaging history teacher.",
        "You bring historical events to life and provide context for your answers.",
        "Use your tools to find accurate information and cite your sources.",
    ],
    add_datetime_to_instructions=True
)

# --- Team of Teachers ---
teacher_team = Team(
    name="Teaching Team",
    mode="coordinate",
    model=OpenAIChat(
        id="gpt-4.1-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
    ),
    members=[
        math_teacher,
        science_teacher,
        history_teacher
    ],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "You are the Head Teacher, coordinating a team of expert teachers.",
        "Delegate questions to the appropriate teacher (Math, Science, or History).",
        "If a question involves multiple subjects, coordinate with the relevant teachers to form a comprehensive answer.",
        "Synthesize the information from your team members to provide a final, clear answer to the student.",
    ],
    markdown=True,
    show_members_responses=True,
    enable_agentic_context=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=15,
    success_criteria="Provide an accurate, well-explained, and comprehensive answer based on the user's query by leveraging the expert teachers.",
)

# --- Run the Team ---
if __name__ == "__main__":
    print("Starting the Teaching Team...")
    print("You can ask questions about Math, Science, or History.")

    while True:
        user_input = input("\n\nStudent: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the Teaching Team. Goodbye!")
            break
        
        teacher_team.print_response(user_input, stream=True, show_full_reasoning=True, stream_intermediate_steps=True)