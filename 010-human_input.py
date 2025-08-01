from typing import Any, Dict, List
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.toolkit import Toolkit
from agno.tools.user_control_flow import UserControlFlowTools
from agno.utils import pprint

from config.secrets import OPENAI_API_KEY

# --- Custom Toolkit for Medical Appointments ---
class MedicalAppointmentTools(Toolkit):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="MedicalAppointmentTools",
            tools=[self.schedule_medical_appointment, self.check_doctor_availability, self.get_appointment_details],
            *args,
            **kwargs
        )

    def schedule_medical_appointment(
        self, 
        patient_name: str, 
        patient_phone: str, 
        doctor_name: str, 
        appointment_type: str, 
        date: str, 
        time: str,
        symptoms_or_reason: str = None
    ) -> str:
        """Schedules a medical appointment after collecting all necessary details.

        Args:
            patient_name (str): The full name of the patient.
            patient_phone (str): The patient's phone number for contact.
            doctor_name (str): The name of the doctor (e.g., 'Dr. Smith', 'Dr. Johnson').
            appointment_type (str): Type of appointment (e.g., 'General Checkup', 'Consultation', 'Follow-up').
            date (str): The desired date for the appointment (e.g., '2025-08-15').
            time (str): The desired time for the appointment (e.g., '10:00 AM').
            symptoms_or_reason (str, optional): Brief description of symptoms or reason for visit.
        """
        appointment_details = f"""
Medical Appointment Confirmed!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient: {patient_name}
Phone: {patient_phone}
Doctor: {doctor_name}
Type: {appointment_type}
Date: {date}
Time: {time}
{f'Reason: {symptoms_or_reason}' if symptoms_or_reason else ''}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Please arrive 15 minutes early for check-in.
"""
        return appointment_details

    def check_doctor_availability(self, doctor_name: str, date: str) -> str:
        """Check doctor's availability for a given date.

        Args:
            doctor_name (str): The name of the doctor to check.
            date (str): The date to check availability for.
        """
        # Simulated availability data
        available_times = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM", "5:00 PM"]
        return f"Dr. {doctor_name} is available on {date} at: {', '.join(available_times)}"

    def get_appointment_details(self, patient_name: str, date: str) -> str:
        """Get existing appointment details for a patient.

        Args:
            patient_name (str): The name of the patient.
            date (str): The date to check for appointments.
        """
        return f"Checking appointments for {patient_name} on {date}... No existing appointments found."

# --- Agent Definition ---
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
    # The agent gets medical appointment tools AND the tools to ask for user input
    tools=[MedicalAppointmentTools(), UserControlFlowTools()],
    instructions=dedent("""
        You are a medical appointment scheduling assistant for a healthcare clinic.
        Your goal is to help patients schedule medical appointments using the 'schedule_medical_appointment' tool.
        
        To schedule an appointment, you need to collect the following information:
        - Patient's full name
        - Patient's phone number
        - Doctor name (ask which doctor they prefer or what type of specialist they need)
        - Appointment type (General Checkup, Consultation, Follow-up, Specialist Visit, etc.)
        - Preferred date
        - Preferred time
        - Reason for visit or symptoms (optional but helpful)
        
        Use the 'request_user_input' tool to ask the user for any missing information.
        You can also use 'check_doctor_availability' to help find suitable appointment times.
        Always be professional, empathetic, and helpful when dealing with medical appointment requests.
        Do not make up any information - always ask the patient for accurate details.
    """),
    markdown=True
)

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("🏥 Starting the Medical Appointment Booking System...")
    print("Welcome to our healthcare clinic appointment scheduler!")

    while True:
        user_input = input("\n\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using our medical appointment system. Take care!")
            break

        # Initial run of the agent
        run_response = agent.run(user_input)

        # We use a while loop to continue the running until the agent is satisfied with the user input
        while run_response.is_paused:
            print("\n--- 📋 Additional Information Required ---")
            for tool in run_response.tools_requiring_user_input:
                input_schema = tool.user_input_schema

                for field in input_schema:
                    # Display field information to the user
                    print(f"\nField: {field.name} ({field.field_type.__name__}) -> {field.description}")

                    # Get user input (if the value is not set, it means the user needs to provide the value)
                    if field.value is None:
                        user_value = input(f"Please enter a value for {field.name}: ")
                        field.value = user_value
                    else:
                        print(f"Value provided by the agent: {field.value}")

            # Continue the agent's execution with the new information
            run_response = agent.continue_run(run_response=run_response)

        # If the agent is not paused for input, we are done
        if not run_response.is_paused:
            print(f"\n🏥 Medical Assistant: ")
            pprint.pprint_run_response(run_response)
