from autogen import AssistantAgent
from config import config_list_gpt4o

def create_eirene_agent():
    """Create and return the Eirene (Issuance) agent"""
    return AssistantAgent(
        name="Eirene (IssuanceAgent)",
        system_message="You are Eirene. Finalize the issuance of policies and assign policy numbers.",
        llm_config={"config_list": config_list_gpt4o}
    )