from autogen import AssistantAgent
from config import config_list_gpt4o

def create_ares_agent():
    """Create and return the Ares (Risk Evaluation) agent"""
    return AssistantAgent(
        name="Ares (RiskEvaluationAgent)",
        system_message="You are Ares. Analyze risk factors (vehicle, driving history, etc.) and produce a risk score.",
        llm_config={"config_list": config_list_gpt4o}
    )