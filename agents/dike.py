from autogen import AssistantAgent
from config import config_list_gpt4o

def create_dike_agent():
    """Create and return the Dike (Regulatory) agent"""
    return AssistantAgent(
        name="Dike (RegulatoryAgent)",
        system_message="You are Dike. Ensure the policy complies with all regulatory requirements.",
        llm_config={"config_list": config_list_gpt4o}
    )