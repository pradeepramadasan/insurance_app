from autogen import AssistantAgent
from config import config_list_gpt4o

def create_calliope_agent():
    """Create and return the Calliope (Document Drafting) agent"""
    return AssistantAgent(
        name="Calliope (DocumentDraftingAgent)",
        system_message="You are Calliope. Refine and polish the policy draft into a finalized document.",
        llm_config={"config_list": config_list_gpt4o}
    )