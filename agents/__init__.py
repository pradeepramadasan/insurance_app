from .iris import create_iris_agent
from .mnemosyne import create_mnemosyne_agent
from .ares import create_ares_agent
from .hera import create_hera_agent
from .demeter import create_demeter_agent
from .apollo import create_apollo_agent
from .calliope import create_calliope_agent
from .plutus import create_plutus_agent
from .tyche import create_tyche_agent
from .orpheus import create_orpheus_agent
from .hestia import create_hestia_agent
from .dike import create_dike_agent
from .eirene import create_eirene_agent
from .themis import create_themis_agent
from .zeus import create_zeus_agent
from autogen import AssistantAgent, UserProxyAgent
from config import config_list_from_model
import os
import logging
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

# Configure logging
logger = logging.getLogger("insurance_agents")
# Read the specific deployment name for o1-mini from environment variables
# Load environment variables from x1.env - ADD THIS LINE
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "x1.env"))

o1mini_deployment_name = os.getenv("AZURE_OPENAI_O1MINI_DEPLOYMENT")

if not o1mini_deployment_name:
    # Handle error or log a warning if the deployment name isn't set
    print("WARNING: AZURE_OPENAI_O1MINI_DEPLOYMENT environment variable not set. Demeter agent may fail.")
    # You might want to raise an error or use a default/fallback if appropriate
    o1mini_deployment_name = "o1-mini" # Fallback to potentially incorrect name, but log warning

demeter_config_list = [{
    "model": o1mini_deployment_name, # <-- Use the variable holding the deployment name
    "api_key": os.getenv("AZURE_OPENAI_API_KEY_X1"),
    "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"), # Read version from env
    "api_type": "azure"
}]


# --- Agent Creation ---
import os
import logging
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

# Configure logging
logger = logging.getLogger("insurance_agents")

# Load environment variables from x1.env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "x1.env"))

def initialize_agents(model=None):
    """
    Initialize all agents required for the insurance workflow.
    
    Args:
        model (str, optional): The model deployment name to use. If None, uses the value from environment.
        
    Returns:
        dict: Dictionary of initialized agents
    """
    # Get deployment names from environment variables or use provided model
    gpt4o_deployment = model or os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
    o1mini_deployment_name = os.getenv("AZURE_OPENAI_O1MINI_DEPLOYMENT")
    
    if not gpt4o_deployment:
        logger.warning("No GPT-4o deployment specified - using default 'gpt-4o'")
        gpt4o_deployment = "gpt-4o"  # Fallback name
    
    if not o1mini_deployment_name:
        logger.warning("AZURE_OPENAI_O1MINI_DEPLOYMENT environment variable not set. Demeter agent may fail.")
        o1mini_deployment_name = "o1-mini"  # Fallback name, but this will likely cause a 404 error
    
    # Log the deployment names being used
    logger.info(f"Initializing agents with GPT-4o deployment: {gpt4o_deployment}")
    logger.info(f"Demeter will use o1-mini deployment: {o1mini_deployment_name}")
    
    # Common configuration for GPT-4o agents
    gpt4o_config_list = [{
        "model": gpt4o_deployment,
        "api_key": os.getenv("AZURE_OPENAI_API_KEY_X1"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "api_type": "azure"
    }]
    
    # Configuration for Demeter agent (using o1-mini)
    demeter_config_list = [{
        "model": o1mini_deployment_name,  # Using the environment variable for o1-mini
        "api_key": os.getenv("AZURE_OPENAI_API_KEY_X1"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "api_type": "azure"
    }]
    
    # Initialize all agents
    agents = {
        # Primary agents using GPT-4o
        "iris": AssistantAgent("iris", llm_config={"config_list": gpt4o_config_list}),
        "mnemosyne": AssistantAgent("mnemosyne", llm_config={"config_list": gpt4o_config_list}),
        "ares": AssistantAgent("ares", llm_config={"config_list": gpt4o_config_list}),
        "hera": AssistantAgent("hera", llm_config={"config_list": gpt4o_config_list}),
        "apollo": AssistantAgent("apollo", llm_config={"config_list": gpt4o_config_list}),
        "calliope": AssistantAgent("calliope", llm_config={"config_list": gpt4o_config_list}),
        "plutus": AssistantAgent("plutus", llm_config={"config_list": gpt4o_config_list}),
        "tyche": AssistantAgent("tyche", llm_config={"config_list": gpt4o_config_list}),
        "orpheus": AssistantAgent("orpheus", llm_config={"config_list": gpt4o_config_list}),
        "hestia": AssistantAgent("hestia", llm_config={"config_list": gpt4o_config_list}),
        "dike": AssistantAgent("dike", llm_config={"config_list": gpt4o_config_list}),
        "eirene": AssistantAgent("eirene", llm_config={"config_list": gpt4o_config_list}),
        "themis": AssistantAgent("themis", llm_config={"config_list": gpt4o_config_list}),
        "zeus": AssistantAgent("zeus", llm_config={"config_list": gpt4o_config_list}),
        
        # Demeter using o1-mini via separate config
        "demeter": AssistantAgent("demeter", llm_config={"config_list": gpt4o_config_list}),
        
        # User proxy agent
        "user_proxy": UserProxyAgent(
            "user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False
        )
    }
    
    logger.info("Agents initialized successfully")
    print("Agents initialized successfully")
    return agents        
# You might call initialize_agents() elsewhere in your app startup