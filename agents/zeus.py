from autogen import AssistantAgent
from config import config_list_gpt4o

def create_zeus_agent():
    """Create and return the Zeus (Coordinating) agent"""
    return AssistantAgent(
        name="Zeus (CoordinatingAgent)",
        system_message="You are Zeus. Coordinate the entire insurance policy creation process and ensure smooth handoffs between agents.",
        llm_config={"config_list": config_list_gpt4o}
    )