from autogen import AssistantAgent
from config import config_list_gpt4o

def create_apollo_agent():
    """Create and return the Apollo (Policy Drafting) agent"""
    return AssistantAgent(
        name="Apollo (PolicyDraftingAgent)",
        system_message="You are Apollo. Draft compliant policy language based on the coverage model.",
        llm_config={"config_list": config_list_gpt4o}
    )