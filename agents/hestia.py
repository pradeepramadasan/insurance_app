from autogen import AssistantAgent
from config import config_list_gpt4o

def create_hestia_agent():
    """Create and return the Hestia (Internal Approval) agent"""
    return AssistantAgent(
        name="Hestia (InternalApprovalAgent)",
        system_message="You are Hestia. Conduct internal reviews and approve finalized policy drafts.",
        llm_config={"config_list": config_list_gpt4o}
    )