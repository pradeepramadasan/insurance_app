from autogen import AssistantAgent
from config import config_list_gpt4o
import json
from db.cosmos_db import get_mandatory_questions, save_underwriting_responses
# Update the import to use the correct class name
from agents.hera import HeraAgent

# Initialize Hera when needed with correct class name
hera_agent = HeraAgent()

# Call Hera with standardized data format
def get_recommendations(customer_data):
    # The source parameter identifies which agent is making the call
    result = hera_agent.get_recommendations(customer_data, source="mnemosyne")
    return result

def create_mnemosyne_agent():
    """Create and return the Mnemosyne (Profile) agent"""
    return AssistantAgent(
        name="Mnemosyne (ProfileAgent)",
        system_message="""
You are MNEMOSYNE, the profile building agent in the insurance policy workflow.

Your job is to:
1. Review initial customer data collected by Iris
2. Query the Cosmos DB 'autopm' container to retrieve mandatory underwriting questions
3. Work with Iris to ask these questions to the end user
4. Process responses and build a comprehensive customer profile
5. Determine eligibility based on underwriting responses (any "No" answer makes the customer ineligible)
6. Save the profile data regardless of eligibility outcome

IMPORTANT UNDERWRITING RULE:
- If ANY underwriting question receives a "No" answer, the customer is ineligible
- The quote should still be saved to the database with eligibility=false
- The process must end with a message that coverage cannot be provided at this time

Always return your response in clean JSON format with the following structure:
{
    "customerProfile": {
        "personal": {
            "name": "string",
            "dateOfBirth": "string",
            "address": "string",
            "contactInfo": {
                "phone": "string",
                "email": "string"
            }
        },
        "vehicle": {
            "make": "string",
            "model": "string",
            "year": "number",
            "vin": "string"
        },
        "underwriting": {
            "questionId1": "response",
            "questionId2": "response"
        },
        "eligibility": boolean,
        "eligibilityReason": "string"
    }
}
""",
        llm_config={"config_list": config_list_gpt4o}
    )