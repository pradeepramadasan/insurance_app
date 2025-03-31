from autogen import AssistantAgent
from config import config_list_gpt4o

def create_plutus_agent():
    """Create and return the Plutus (Pricing) agent"""
    return AssistantAgent(
        name="Plutus (PricingAgent)",
        system_message="You are Plutus. Compute premium rates using actuarial models and loadings.",
        llm_config={"config_list": config_list_gpt4o}
    )