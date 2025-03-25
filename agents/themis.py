from autogen import AssistantAgent
from config import config_list_gpt4o

def create_themis_agent():
    """Create and return the Themis (Monitoring) agent"""
    return AssistantAgent(
        name="Themis (MonitoringAgent)",
        system_message="You are Themis. Monitor issued policies and report on performance metrics.",
        llm_config={"config_list": config_list_gpt4o}
    )