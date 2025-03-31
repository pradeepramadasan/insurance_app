from autogen import AssistantAgent
from config import config_list_gpt4o

def create_orpheus_agent():
    """Create and return the Orpheus (Presentation) agent"""
    return AssistantAgent(
        name="Orpheus (PresentationAgent)",
        system_message="You are Orpheus. Present policy proposals to customers in a persuasive and clear manner.",
        llm_config={"config_list": config_list_gpt4o}
    )