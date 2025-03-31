from autogen import AssistantAgent
from config import config_list_gpt4o

def create_tyche_agent():
    """Create and return the Tyche (Quote) agent"""
    return AssistantAgent(
        name="Tyche (QuoteAgent)",
        system_message="You are Tyche. Generate a detailed, customer-friendly quote based on pricing data.",
        llm_config={"config_list": config_list_gpt4o}
    )