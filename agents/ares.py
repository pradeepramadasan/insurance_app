import os
from autogen import AssistantAgent, config_list_from_json
from config import config_list_gpt4o

# Import Hera agent for recommendations with correct class name
from agents.hera import HeraAgent

# Initialize Hera when needed with correct class name
hera_agent = HeraAgent()

def create_ares_agent():
    """
    Creates and returns the Ares risk assessment agent using Azure OpenAI.
    
    Follows Azure best practices for agent initialization and configuration.
    
    Returns:
        AssistantAgent: The configured Ares agent
    """
    # Use environment variables with fallbacks for Azure configuration
    azure_deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o")
    
    # Define the Ares agent system message
    system_message = """You are Ares, an insurance risk assessment specialist agent.
    
    Your responsibilities:
    1. Analyze customer profiles for risk factors
    2. Evaluate driving history, vehicle specifications, and location data
    3. Generate comprehensive risk assessments
    4. Recommend appropriate insurance coverage levels based on risk profile
    5. Work with other agents to provide holistic insurance recommendations
    
    Always consider both standard risk factors and edge cases in your assessments.
    """
    
    # Create the Ares agent with proper Azure OpenAI configuration
    ares_agent = AssistantAgent(
        name="Ares",
        system_message=system_message,
        llm_config={
            "config_list": config_list_gpt4o,
            "temperature": 0.2,  # Lower temperature for more consistent risk assessment
            "timeout": 60,  # Set reasonable timeout for Azure API calls
            "azure_deployment": azure_deployment
        }
    )
    
    return ares_agent

# Update the get_recommendations function
def get_recommendations(customer_data):
    """
    Get insurance recommendations based on customer risk profile
    """
    # Use the correct source identifier for Ares
    result = hera_agent.get_recommendations(customer_data, source="ares")
    return result

# Expose any additional functions needed for the Ares workflow
def analyze_risk_factors(customer_data):
    """
    Analyze specific risk factors in customer data
    
    Args:
        customer_data (dict): Customer profile data
        
    Returns:
        dict: Detailed risk factor analysis
    """
    # This would use the Ares agent or internal logic to analyze risks
    # For now, this is a placeholder
    return {
        "riskLevel": "medium",  # Example output
        "factors": {
            "driving": calculate_driving_risk(customer_data),
            "vehicle": calculate_vehicle_risk(customer_data),
            "location": calculate_location_risk(customer_data)
        }
    }

def calculate_driving_risk(customer_data):
    """Calculate driving history risk factors"""
    # Implementation would go here
    return "medium"

def calculate_vehicle_risk(customer_data):
    """Calculate vehicle-related risk factors"""
    # Implementation would go here
    return "low"

def calculate_location_risk(customer_data):
    """Calculate location-based risk factors using H3 indices"""
    # Implementation would go here
    return "medium"