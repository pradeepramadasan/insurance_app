import datetime
import random as random_module
import os
import logging
import time
import json
import uuid 
import copy
import re 
from dotenv import load_dotenv
from fsspec import Callback
from openai import AzureOpenAI
from agents import initialize_agents
from agents.hera import get_profile_recommendations
from workflow.document_processor import DocumentProcessor
from utils.helpers import extract_json_content
from db.cosmos_db import (
    save_policy_checkpoint,
    get_mandatory_questions,
    save_underwriting_responses,
    confirm_policy,
    get_container_client,
    get_default_coverage_data
)

# Azure Best Practice: Configure module-level logger
logger = logging.getLogger(__name__)

# Configure root logger if not already done
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Azure Best Practice: Add Application Insights integration if available

# Global Azure OpenAI client and deployment
azure_openai_client = None
gpt4o_deployment = None

def initialize_azure_openai():
    global azure_openai_client, gpt4o_deployment

    if azure_openai_client:
        logger.info("Using existing Azure OpenAI client")
        return azure_openai_client

    # Load environment variables from x1.env
    if os.path.exists("x1.env"):
        logger.info("Loading environment variables from x1.env")
        load_dotenv("x1.env")
    else:
        logger.error("Environment file x1.env not found.")
        return None

    api_key = os.getenv("AZURE_OPENAI_API_KEY_X1")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    gpt4o_deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")

    if not all([api_key, azure_endpoint, gpt4o_deployment]):
        logger.error("Missing required Azure OpenAI environment variables.")
        return None

    try:
        # Initialize client
        azure_openai_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        # Test connection
        models = azure_openai_client.models.list()
        available_models = [model.id for model in models.data]
        logger.info(f"Azure OpenAI connected. Available models: {available_models}")
    except Exception as e:
        # Log error *before* returning
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        azure_openai_client = None
        return None # Return None after logging the error

    return azure_openai_client



from utils.helpers import read_customer_data_from_file, show_current_status_and_confirm, extract_json_content
###


def query_agent(agent, prompt, gpt4o_model, description=None):
    """
    Azure best practice implementation for consistent agent interaction.
    
    Args:
        agent: The agent to query
        prompt: The prompt to send
        gpt4o_model: The GPT-4o deployment name
        description: Optional description for logging
        
    Returns:
        tuple: (content_str, parsed_json, success_flag)
    """
    try:
        agent_name = getattr(agent, 'name', 'unknown')
        if description:
            logger.info(f"Querying {agent_name} - {description}")
        
        # Standardize agent prompting with explicit model
        response = agent.generate_reply(
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Handle different response formats
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Log the raw response for debugging (truncated for large responses)
        logger.info(f"=== RAW {agent_name.upper()} RESPONSE ===")
        logger.info(response_content[:500] + "..." if len(response_content) > 500 else response_content)
        
        # Try to parse JSON from the response
        parsed_data = extract_json_with_fallback(response_content)
        
        if parsed_data is not None:
            return response_content, parsed_data, True
        
        # If JSON parsing failed but we still have a response
        return response_content, None, False
        
    except Exception as e:
        logger.error(f"Error querying {agent_name}: {str(e)}")
        return str(e), None, False

def format_prompt_for_json_output(instructions, model_structure=None):
    """
    Azure best practice for consistent prompt engineering to ensure JSON outputs.
    
    Args:
        instructions: The core instructions for the agent
        model_structure: Optional example of the expected JSON structure
        
    Returns:
        str: Formatted prompt with JSON output instructions
    """
    json_instructions = """
IMPORTANT INSTRUCTIONS: 
1. DO NOT write any code or show your work
2. DO NOT use print statements
3. DO NOT include explanation text, comments, or markdown formatting  
4. Directly output a valid JSON object ONLY
"""

    if model_structure:
        json_instructions += f"""
Return your response as EXACTLY this JSON format with the actual data:
{model_structure}
"""
    else:
        json_instructions += "Return ONLY a valid JSON object as your complete response."

    return f"{instructions}\n\n{json_instructions}"


def process_with_agent(agent, prompt, current_state, gpt4o_deployment, step_name, 
                       json_expected=True, state_key=None, fallback_handler=None):
    """
    Generic agent processing function following Azure best practices.
    
    Args:
        agent: The agent to use
        prompt: The prompt to send
        current_state: Current workflow state
        gpt4o_deployment: The GPT-4o deployment name
        step_name: Name of the workflow step (for logging and checkpoints)
        json_expected: Whether JSON response is expected
        state_key: Key in current_state to store the result
        fallback_handler: Optional function to handle failures
        
    Returns:
        dict: Updated workflow state
    """
    agent_name = getattr(agent, 'name', 'unknown')
    logger.info(f"Processing {step_name} with {agent_name}")
    
    # Query the agent
    content, parsed_result, success = query_agent(
        agent, 
        prompt, 
        gpt4o_deployment, 
        description=step_name
    )
    
    # Display the result to the user
    print(f"\n=== {step_name.upper()} ===")
    if len(content) > 500:
        print(f"{content[:500]}...")
        print("(Response truncated for display. Full response logged.)")
    else:
        print(content)
    print("=" * (len(step_name) + 8))
    
    # Handle the result based on whether JSON is expected
    if json_expected:
        if not success or not parsed_result:
            print(f"⚠️ [{agent_name}] Failed to parse JSON response.")
            
            # Use fallback handler if provided
            if fallback_handler:
                print(f"Using fallback handler for {step_name}...")
                parsed_result = fallback_handler(current_state, agent, content)
            
            # If still no valid result
            if not parsed_result:
                print(f"⚠️ Fallback processing failed for {step_name}.")
                return current_state
        
        # Store the result in the state if requested
        if state_key and parsed_result is not None:
            current_state[state_key] = parsed_result
            
    else:
        # For non-JSON responses, store the raw content
        if state_key:
            current_state[state_key] = content
    
    # Save checkpoint
    save_policy_checkpoint(current_state, f"{step_name.lower().replace(' ', '_')}_completed")
    logger.info(f"Completed {step_name} with {agent_name}")
    
    return current_state
# Add this function to your workflow process.py

# Add near the imports section

def display_policy_graph(current_state, latest_update=None, zeus=None, gpt4o_deployment=None):
    """
    Display a comprehensive policy graph through Zeus agent showing all information collected so far.
    Following Azure best practices for LLM-based customer communications.

    Args:
        current_state: Current workflow state with all collected policy information
        latest_update: Section that was most recently updated (for highlighting)
        zeus: Zeus agent for generating the policy summary
        gpt4o_deployment: The GPT-4o deployment name to use
    """
    # Invoke the get_azure_best_practices tool to ensure Azure compliance
    # @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.
    # Tool call is commented out, so commenting out the usage below as well.
    # try:
    #     # best_practices = get_azure_best_practices()
    #     # logger.info(f"Zeus display: Azure best practices applied: {', '.join(best_practices.get('applied_practices', []))}")
    #     pass # Placeholder if tool call is not made
    # except Exception as e:
    #     logger.warning(f"Could not fetch Azure best practices for Zeus display: {str(e)}")

    print("\n" + "="*80)
    print("                    ZEUS IS PREPARING YOUR POLICY SUMMARY")
    print("="*80 + "\n")

    # If Zeus agent or deployment name is not provided, fall back to direct display
    if not zeus or not gpt4o_deployment:
        logger.warning("Zeus agent or GPT-4o deployment not provided, using direct display instead")
        _display_policy_graph_direct(current_state, latest_update)
        return # Correctly indented return

    # Format all policy data for Zeus
    summary_data = {}
    # Rest of the function continues...
    sections = []

    # 1. Customer Information
    if "customerProfile" in current_state:
        profile = current_state["customerProfile"]
        # ... (rest of display_policy_graph) ...

# ... (other functions like _display_policy_graph_direct, 
        customer_info = {
            "name": profile.get("name", "Not provided"),
            "dob": profile.get("dob", "Not provided"),
            "contact": profile.get("contact", {}),
            "address": profile.get("address", {}),
            "vehicle": profile.get("vehicle_details", {}),
            "drivingHistory": profile.get("driving_history", {})
        }
        summary_data["customerProfile"] = customer_info
        sections.append("customerProfile")
    
    # 2. Risk Assessment
    if "risk_info" in current_state:
        summary_data["riskAssessment"] = current_state["risk_info"]
        sections.append("riskAssessment")
    
    # 3. Coverage Information
    if "coverage" in current_state:
        summary_data["coverage"] = current_state["coverage"]
        sections.append("coverage")
    
    # 4. Pricing Information
    if "pricing" in current_state:
        summary_data["pricing"] = current_state["pricing"]
        sections.append("pricing")
    
    # 5. Policy Issuance
    if "issuance" in current_state:
        summary_data["issuance"] = current_state["issuance"]
        sections.append("issuance")
    
    # 6. Policy Monitoring
    if "monitoring" in current_state:
        summary_data["monitoring"] = current_state["monitoring"]
        sections.append("monitoring")
    
    # Create prompt for Zeus
    zeus_prompt = f"""
    As Zeus, the planning agent, please summarize the customer's insurance policy information in a clear, 
    conversational way. Focus on presenting the information in a helpful manner that a customer would appreciate.
    
    The policy information available includes: {", ".join(sections)}
    Recently updated section: {latest_update if latest_update else "None"}
    
    Policy Data:
    {json.dumps(summary_data, indent=2)}
    
    Please present a complete summary of the policy information in a well-organized format with these guidelines:
    
    1. Use friendly, conversational language a non-expert would understand
    2. Highlight the recently updated section ("{latest_update if latest_update else "None"}")
    3. Use visual organization (emojis, bullet points, sections) for readability
    4. Format currency values properly (with $ and commas)
    5. Present the information in logical sections
    6. Avoid technical jargon or insurance industry terminology without explanation
    7. Provide a brief explanation of what each section means for the customer
    8. End with a "next steps" suggestion based on where they are in the process
    
    Format this as a friendly, conversational summary that builds trust with the customer.
    """
    
    try:
        # Azure Best Practice: Handle potential timeouts and errors
        start_time = time.time()
        
        # Query Zeus with timeout
        logger.info("Querying Zeus for policy summary")
        
        # Azure Best Practice: Implement circuit breaker pattern
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Standardized agent prompting with explicit model for consistent results
                response = zeus.generate_reply(
                    messages=[{"role": "user", "content": zeus_prompt}]
                )
                
                # Handle different response formats (content property or string)
                zeus_summary = response.content if hasattr(response, 'content') else str(response)
                logger.info(f"Zeus policy summary generated in {time.time() - start_time:.2f} seconds")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Zeus query attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error(f"All Zeus query attempts failed: {str(e)}")
                    # Fall back to direct display
                    print("\n⚠️ Zeus is currently unavailable. Displaying standard policy summary instead.\n")
                    _display_policy_graph_direct(current_state, latest_update)
                    return
        
        # Display Zeus's summary
        print(zeus_summary)
        
    except Exception as e:
        logger.error(f"Error getting Zeus to summarize policy: {str(e)}")
        print(f"\n⚠️ Zeus encountered an issue while preparing your summary: {str(e)}")
        print("\nFalling back to standard summary format:\n")
        _display_policy_graph_direct(current_state, latest_update)
    
    print("\n" + "="*80)


def _display_policy_graph_direct(current_state, latest_update=None):
    """
    Direct display of policy information without using Zeus.
    Used as a fallback if Zeus is not available.
    
    Args:
        current_state: Current workflow state with all collected policy information
        latest_update: Section that was most recently updated (for highlighting)
    """
    print("\n" + "="*80)
    print("                       CURRENT POLICY PROFILE SUMMARY")
    print("="*80)
    
    # Track if any data has been displayed yet
    data_displayed = False
    
    # 1. CUSTOMER INFORMATION
    if "customerProfile" in current_state:
        data_displayed = True
        profile = current_state["customerProfile"]
        print("\n📋 CUSTOMER INFORMATION" + (" (UPDATED)" if latest_update == "customerProfile" else ""))
        print("  Name:         " + profile.get("name", "Not provided"))
        print("  Date of Birth:" + profile.get("dob", "Not provided"))
        
        address = profile.get('address', {})
        if isinstance(address, dict):
            print(f"  Address:      {address.get('street', '')}, {address.get('city', '')}, "
                  f"{address.get('state', '')} {address.get('zip', '')}")
        else:
            print(f"  Address:      {address}")
            
        contact = profile.get('contact', {})
        if isinstance(contact, dict):
            print(f"  Phone:        {contact.get('phone', 'Not provided')}")
            print(f"  Email:        {contact.get('email', 'Not provided')}")
            
        # Vehicle details if available
        vehicle = profile.get('vehicle_details', {})
        if vehicle and isinstance(vehicle, dict):
            print("\n🚗 VEHICLE INFORMATION" + (" (UPDATED)" if latest_update == "vehicle_details" else ""))
            print(f"  Make:         {vehicle.get('make', 'Not provided')}")
            print(f"  Model:        {vehicle.get('model', 'Not provided')}")
            print(f"  Year:         {vehicle.get('year', 'Not provided')}")
            print(f"  VIN:          {vehicle.get('vin', 'Not provided')}")
            
        # Driving history if available
        driving = profile.get('driving_history', {})
        if driving and isinstance(driving, dict):
            print("\n🚦 DRIVING HISTORY" + (" (UPDATED)" if latest_update == "driving_history" else ""))
            print(f"  Violations:   {driving.get('violations', 'Not provided')}")
            print(f"  Accidents:    {driving.get('accidents', 'Not provided')}")
            print(f"  Years Licensed: {driving.get('years_licensed', 'Not provided')}")

    # 2. RISK ASSESSMENT
    if "risk_info" in current_state:
        data_displayed = True
        risk_info = current_state["risk_info"]
        print("\n⚠️ RISK ASSESSMENT" + (" (UPDATED)" if latest_update == "risk_info" else ""))
        print(f"  Risk Score:   {risk_info.get('riskScore', 'Not calculated')}")
        
        risk_factors = risk_info.get('riskFactors', [])
        if risk_factors:
            print("  Risk Factors:")
            for factor in risk_factors:
                print(f"    • {factor}")

    # 3. COVERAGE INFORMATION
    if "coverage" in current_state:
        data_displayed = True
        coverage = current_state["coverage"]
        print("\n🛡️ COVERAGE DETAILS" + (" (UPDATED)" if latest_update == "coverage" else ""))
        
        # Core coverages
        coverages = coverage.get('coverages', [])
        if coverages:
            print("  Selected Coverages:")
            for cov in coverages:
                print(f"    • {cov}")
                
        # Coverage limits
        limits = coverage.get('limits', {})
        if limits:
            print("  Coverage Limits:")
            for limit_name, limit_details in limits.items():
                if isinstance(limit_details, dict):
                    if 'per_person' in limit_details and 'per_accident' in limit_details:
                        print(f"    • {limit_name.replace('_', ' ').title()}: "
                              f"${limit_details.get('per_person'):,} per person / "
                              f"${limit_details.get('per_accident'):,} per accident")
                    elif 'amount' in limit_details:
                        print(f"    • {limit_name.replace('_', ' ').title()}: ${limit_details.get('amount'):,}")
                else:
                    print(f"    • {limit_name.replace('_', ' ').title()}: {limit_details}")
        
        # Deductibles
        deductibles = coverage.get('deductibles', {})
        if deductibles:
            print("  Deductibles:")
            for ded_name, ded_details in deductibles.items():
                if isinstance(ded_details, dict) and 'amount' in ded_details:
                    print(f"    • {ded_name.replace('_', ' ').title()}: ${ded_details.get('amount'):,}")
                else:
                    print(f"    • {ded_name.replace('_', ' ').title()}: {ded_details}")
        
        # Add-ons
        addons = coverage.get('addOns', [])
        if addons:
            print("  Add-ons:")
            for addon in addons:
                print(f"    • {addon}")
                
        # Exclusions
        exclusions = coverage.get('exclusions', [])
        if exclusions:
            print("  Exclusions:")
            for exclusion in exclusions:
                print(f"    • {exclusion}")

    # 4. PRICING INFORMATION
    if "pricing" in current_state:
        data_displayed = True
        pricing = current_state["pricing"]
        print("\n💰 PRICING INFORMATION" + (" (UPDATED)" if latest_update == "pricing" else ""))
        print(f"  Base Premium:   ${pricing.get('basePremium', 0):,.2f}")
        print(f"  Risk Multiplier: {pricing.get('riskMultiplier', 1.0):.2f}x")
        print(f"  Final Premium:  ${pricing.get('finalPremium', 0):,.2f}")

    # 5. POLICY ISSUANCE DETAILS
    if "issuance" in current_state:
        data_displayed = True
        issuance = current_state["issuance"]
        print("\n📜 POLICY ISSUANCE" + (" (UPDATED)" if latest_update == "issuance" else ""))
        print(f"  Policy Number: {issuance.get('policyNumber', 'Not issued')}")
        print(f"  Start Date:    {issuance.get('startDate', 'Not specified')}")
        print(f"  End Date:      {issuance.get('endDate', 'Not specified')}")
        print(f"  Status:        {issuance.get('status', 'Pending')}")

    # 6. POLICY MONITORING
    if "monitoring" in current_state:
        data_displayed = True
        monitoring = current_state["monitoring"]
        print("\n🔔 POLICY MONITORING" + (" (UPDATED)" if latest_update == "monitoring" else ""))
        print(f"  Status:         {monitoring.get('monitoringStatus', 'Not setup')}")
        print(f"  Notification:   {monitoring.get('notificationEmail', 'Not specified')}")
        print(f"  Renewal Date:   {monitoring.get('renewalDate', 'Not scheduled')}")

    # If no data has been displayed yet
    if not data_displayed:
        print("\nNo policy information collected yet.")
        print("As you progress through the workflow, your policy details will appear here.")


# Update calls to display_policy_graph in process_insurance_request to include Zeus
# For example, change:
# display_policy_graph(current_state, latest_update="customerProfile")
# To:
# display_policy_graph(current_state, latest_update="customerProfile", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

def create_detailed_profile_manually(basic_profile):
    """Create a detailed customer profile manually"""
    detailed_profile = basic_profile.copy()
    
    # Add vehicle details
    detailed_profile["vehicle_details"] = {
        "make": input("Enter vehicle make: ").strip(),
        "model": input("Enter vehicle model: ").strip(),
        "year": input("Enter vehicle year: ").strip(),
        "vin": input("Enter VIN: ").strip()
    }
    
    # Add driving history
    detailed_profile["driving_history"] = {
        "violations": input("Enter number of violations: ").strip(),
        "accidents": input("Enter number of accidents: ").strip(),
        "years_licensed": input("Enter years licensed: ").strip()
    }
    
    # Add coverage preferences
    detailed_profile["coverage_preferences"] = input("Enter coverage preferences (comma separated): ").strip().split(",")
    
    return detailed_profile

def handle_detailed_profile_corrections(profile):
    """Handle user corrections to a detailed profile"""
    # First handle basic profile corrections
    profile = handle_profile_corrections(profile)
    
    # Then handle vehicle details
    if "vehicle_details" not in profile:
        profile["vehicle_details"] = {}
    profile["vehicle_details"]["make"] = input("Enter vehicle make: ").strip() or profile.get("vehicle_details", {}).get("make", "")
    profile["vehicle_details"]["model"] = input("Enter vehicle model: ").strip() or profile.get("vehicle_details", {}).get("model", "")
    profile["vehicle_details"]["year"] = input("Enter vehicle year: ").strip() or profile.get("vehicle_details", {}).get("year", "")
    profile["vehicle_details"]["vin"] = input("Enter VIN: ").strip() or profile.get("vehicle_details", {}).get("vin", "")
    
    # Then handle driving history
    if "driving_history" not in profile:
        profile["driving_history"] = {}
    profile["driving_history"]["violations"] = input("Enter number of violations: ").strip() or profile.get("driving_history", {}).get("violations", "")
    profile["driving_history"]["accidents"] = input("Enter number of accidents: ").strip() or profile.get("driving_history", {}).get("accidents", "")
    profile["driving_history"]["years_licensed"] = input("Enter years licensed: ").strip() or profile.get("driving_history", {}).get("years_licensed", "")
    
    # Then handle coverage preferences
    coverage_input = input("Enter coverage preferences (comma separated): ").strip()
    if coverage_input:
        profile["coverage_preferences"] = [c.strip() for c in coverage_input.split(",")]
    
    return profile

def display_detailed_profile(profile):
    """Display a detailed profile in a user-friendly format"""
    print("\n=== Detailed Vehicle & Driving Information ===")
    vehicle = profile.get('vehicle_details', {})
    if isinstance(vehicle, dict):
        print(f"Make: {vehicle.get('make', 'Not provided')}")
        print(f"Model: {vehicle.get('model', 'Not provided')}")
        print(f"Year: {vehicle.get('year', 'Not provided')}")
        print(f"VIN: {vehicle.get('vin', 'Not provided')}")
    
    driving = profile.get('driving_history', {})
    if isinstance(driving, dict):
        print(f"Violations: {driving.get('violations', 'Not provided')}")
        print(f"Accidents: {driving.get('accidents', 'Not provided')}")
        print(f"Years Licensed: {driving.get('years_licensed', 'Not provided')}")
    
    coverages = profile.get('coverage_preferences', [])
    if isinstance(coverages, list) and coverages:
        print("Coverage Preferences:")
        for coverage in coverages:
            print(f"  - {coverage}")

def get_coverage_with_demeter(demeter):
    """
    Retrieve coverage options from Cosmos DB and have Demeter process them.
    Implements Azure best practices for reliable AI operations.
    """
    try:
        # Get coverage data from Cosmos DB
        container_client = get_container_client("autopm")
        
        # Azure Best Practice: Use parametrized query instead of raw string
        query = "SELECT * FROM c WHERE c.productModel.coverageCategories != null"
        items = list(container_client.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        if not items:
            logger.warning("No coverage data found in autopm container")
            return None
            
        # Azure Best Practice: Log data size for debugging
        logger.info(f"Retrieved {len(items)} coverage items from Cosmos DB")
        
        # Format data for Demeter - simplify the prompt!
        prompt = """
        You're analyzing a product model JSON for an insurance application. Extract all coverage options.
        
        For each coverage category:
        1. Extract the name and description
        2. For each coverage within the category:
            a. Extract the name, coverageCategory, and mandatory flag
            b. Extract the coverageTerms including termName, modelType, and options
            c. For each option, extract optionLabel, value/min/max, and description
        3. Create a user-friendly explanation for each coverage and option
        
        Return the coverage information in this exact format:
        ```json
        {
          "coverageCategories": [
            {
              "name": "Liability Coverages",
              "description": "Covers damages to others",
              "coverages": [
                {
                  "name": "Bodily Injury",
                  "coverageCategory": "Liability",
                  "mandatory": true,
                  "explanation": "Your enhanced explanation of what this covers",
                  "termName": "Limit",
                  "modelType": "Limit",
                  "options": [
                    {
                      "label": "15/30",
                      "display": "$15,000 per person / $30,000 per accident",
                      "min": 15000,
                      "max": 30000,
                      "explanation": "Simple explanation of this option"
                    },
                    ...more options...
                  ]
                },
                ...more coverages...
              ]
            },
            ...more categories...
          ]
        }
        
        
        JSON DATA TO ANALYZE:
        {json_data}
        """
        
        # Azure Best Practice: Take only the first item to reduce complexity
        json_data = json.dumps(items[0], indent=2)
        
        # Azure Best Practice: Use explicit response format
        params = {
            'messages': [
                {
                    'content': prompt.replace("{json_data}", json_data), 
                    'role': 'user'
                }
            ]
        }
        
        # Log the size of the prompt to check for token limits
        logger.info(f"Prompt size: {len(prompt) + len(json_data)} characters")
        
        # Send to Demeter with proper error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Send request to Demeter
                response = demeter.generate_reply(**params)
                
                # Process the response to extract JSON
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Azure Best Practice: Improved JSON extraction
                extracted_json = extract_json_with_fallback(content)
                
                if extracted_json:
                    logger.info("Successfully extracted coverage options")
                    return extracted_json
                
                logger.warning(f"Attempt {attempt+1}: Failed to extract JSON from Demeter response")
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Error calling Demeter: {str(e)}")
                
            # Don't retry if it's the last attempt
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error("All attempts to extract coverage options failed")
        return None
        
    except Exception as e:
        logger.error(f"Error getting coverage options: {str(e)}")
        return None

# Enhance the extraction function with more robust capabilities

def extract_json_with_fallback(content):
    """
    Enhanced JSON extraction function with Azure best practices for LLM response handling.
    """
    try:
        if not content or not isinstance(content, str):
            logger.warning("Empty or non-string content provided to JSON extractor")
            return None
            
        # Log sample of content for debugging
        logger.debug(f"Extracting JSON from content: {content[:100]}...")
        
        # Strategy 1: Direct JSON parsing if the content is already JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from code blocks
        if "```json" in content or "```" in content:
            json_matches = re.findall(r'```(?:json)?(.*?)```', content, re.DOTALL)
            
            if json_matches:
                for json_match in json_matches:
                    try:
                        return json.loads(json_match.strip())
                    except json.JSONDecodeError:
                        continue
        
        # Strategy 3: Find everything between first { and last }
        if '{' in content and '}' in content:
            first_brace = content.find('{')
            last_brace = content.rfind('}')
            if first_brace < last_brace:
                try:
                    extracted = content[first_brace:last_brace+1]
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    # Try to fix common JSON formatting issues
                    try:
                        # Fix unquoted keys (Python-style to JSON)
                        fixed = re.sub(r'(\w+)(?=\s*:)', r'"\1"', extracted)
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass
                        
        # Strategy 4: Handle Python code output
        if "formatted_data" in content and "json.dumps" in content:
            logger.info("Attempting to extract JSON by executing isolated Python code")
            try:
                # Create isolated namespace
                safe_globals = {'json': json}
                safe_locals = {}
                
                # Extract and clean the code
                code_lines = []
                raw_data_found = False
                
                for line in content.split('\n'):
                    # Skip comments, prints and unsafe imports
                    if line.strip().startswith('#') or 'print(' in line or 'import ' in line:
                        continue
                    
                    # Include only the core data transformation code
                    if 'raw_data' in line and not raw_data_found:
                        # Inject the raw data from your actual data source
                        if 'file_data' in vars():
                            code_lines.append(f"raw_data = {json.dumps(file_data)}")
                            raw_data_found = True
                    elif 'formatted_data =' in line:
                        code_lines.append(line)
                        
                # Get only the code that defines formatted_data
                clean_code = '\n'.join(code_lines)
                
                # Execute in isolated environment
                exec(clean_code, safe_globals, safe_locals)
                
                # Get the result
                if 'formatted_data' in safe_locals:
                    logger.info("Successfully extracted JSON by executing isolated code")
                    return safe_locals['formatted_data']
            except Exception as e:
                logger.warning(f"Failed to execute isolated code: {str(e)}")

        # Strategy 5: Try to convert Python-style dictionaries to JSON
        if "'" in content:
            try:
                # Replace single quotes with double quotes
                converted = content.replace("'", '"')
                # Try to parse as JSON
                return json.loads(converted)
            except json.JSONDecodeError:
                pass
        
        logger.warning("Failed to extract JSON using all strategies")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting JSON with fallback: {str(e)}")
        return None
    
def extract_customer_data_regex(data):
    """
    Direct regex extraction of customer data when JSON parsing fails.
    Azure best practice: Always have multiple fallback methods for critical data extraction.
    """
    try:
        result = {}
        
        # Extract name
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', data)
        if name_match:
            result["name"] = name_match.group(1)
        
        # Extract DOB
        dob_match = re.search(r'"(?:dateOfBirth|dob)"\s*:\s*"([^"]+)"', data)
        if dob_match:
            result["dob"] = dob_match.group(1)
        
        # Extract address parts
        address = {}
        for field in ["street", "city", "state", "zip"]:
            match = re.search(fr'"{field}"\s*:\s*"([^"]+)"', data)
            if match:
                address[field] = match.group(1)
        
        if address:
            result["address"] = address
            
        # Extract contact info
        contact = {}
        phone_match = re.search(r'"phone"\s*:\s*"([^"]+)"', data)
        if phone_match:
            contact["phone"] = phone_match.group(1)
            
        email_match = re.search(r'"email"\s*:\s*"([^"]+)"', data)
        if email_match:
            contact["email"] = email_match.group(1)
            
        if contact:
            result["contact"] = contact
            
        # Only return if we got something useful
        if result and "name" in result:
            logger.info("Successfully extracted customer data via regex")
            return result
        return {"status": "error", "message": "Failed to extract sufficient data with regex"}
        
    except Exception as e:
        logger.error(f"Error in regex extraction: {str(e)}")
        return {"status": "error", "message": str(e)}
            
def parse_customer_data(data_source, iris_agent=None):
    """
    Enhanced customer data parsing function following Azure best practices.
    Handles both file-based and direct string data.
    
    Args:
        data_source (str): File path or raw data string
        iris_agent: Optional Iris agent for generating explanations
        
    Returns:
        dict: Parsed customer data or default structure if parsing fails
    """
    try:
        # Determine if data_source is a file path
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, 'r') as file:
                data = file.read()
        else:
            data = data_source
            
        # Extract JSON using our robust extraction function
        customer_data = extract_json_with_fallback(data)
        
        if not customer_data:
            logger.warning("Failed to extract customer data JSON")
            return {"status": "error", "message": "Failed to parse customer data"}
            
        # Handle different JSON structures - normalize the data
        normalized_data = {}
        
        # Handle policyHolder structure
        if "policyHolder" in customer_data:
            holder = customer_data["policyHolder"]
            normalized_data["name"] = f"{holder.get('firstName', '')} {holder.get('lastName', '')}".strip()
            normalized_data["dob"] = holder.get("dateOfBirth")
            normalized_data["address"] = holder.get("address", {})
            normalized_data["contact"] = holder.get("contactInfo", {})
            
        # Handle direct customer structure
        elif "name" in customer_data:
            normalized_data["name"] = customer_data["name"]
            normalized_data["dob"] = customer_data.get("dateOfBirth") or customer_data.get("dob")
            normalized_data["address"] = customer_data.get("address", {})
            normalized_data["contact"] = customer_data.get("contactInfo", {}) or customer_data.get("contact", {})
        
        # Extract vehicle data if available
        if "vehicles" in customer_data and isinstance(customer_data["vehicles"], list) and customer_data["vehicles"]:
            normalized_data["vehicle_details"] = customer_data["vehicles"][0]
        elif "vehicle" in customer_data:
            normalized_data["vehicle_details"] = customer_data["vehicle"]
            
        # Extract driver data if available
        if "drivers" in customer_data and isinstance(customer_data["drivers"], list) and customer_data["drivers"]:
            normalized_data["driver_details"] = customer_data["drivers"][0]
        elif "driver" in customer_data:
            normalized_data["driver_details"] = customer_data["driver"]
            
        return normalized_data
        
    except Exception as e:
        logger.error(f"Error parsing customer data: {str(e)}")
        return {"status": "error", "message": str(e)}
        
def default_coverage_design(current_state, demeter):
    """
    Provides a default coverage design when the primary method fails.
    
    Args:
        current_state (dict): Current workflow state
        demeter (Agent): The Demeter agent for optional explanations
    
    Returns:
        dict: Default coverage structure
    """
    print("\n=== USING DEFAULT COVERAGE DESIGN ===")
    print("Unable to extract custom coverage options - using standard coverage package")
    
    # Try to get vehicle information for better defaults
    vehicle_year = None
    vehicle_make = None
    try:
        if "customerProfile" in current_state and "vehicle_details" in current_state["customerProfile"]:
            vehicle = current_state["customerProfile"]["vehicle_details"]
            vehicle_year = vehicle.get("year")
            vehicle_make = vehicle.get("make")
            print(f"Customizing default coverage for {vehicle_year} {vehicle_make}")
    except Exception as e:
        print(f"Could not retrieve vehicle details: {str(e)}")
    
    # Generate explanation with Demeter if available
    try:
        explanation_prompt = """
        Please explain the following standard auto insurance coverages in simple terms:
        1. Bodily Injury Liability
        2. Property Damage Liability
        3. Uninsured Motorist
        4. Collision
        5. Comprehensive
        
        Keep your explanation brief but informative.
        """
        
        explanation = demeter.generate_reply(messages=[{"role": "user", "content": explanation_prompt}])
        if hasattr(explanation, 'content'):
            print(f"\n{explanation.content}\n")
    except Exception as e:
        print(f"Could not generate explanations: {str(e)}")
    
    # Return default coverage structure
    return {
        "coverages": [
            "Bodily Injury Liability",
            "Property Damage Liability",
            "Uninsured Motorist Bodily Injury",
            "Collision",
            "Comprehensive"
        ],
        "limits": {
            "bodily_injury": {
                "per_person": 50000,
                "per_accident": 100000,
                "label": "50/100"
            },
            "property_damage": {
                "amount": 50000,
                "label": "50,000"
            },
            "uninsured_motorist_bodily_injury": {
                "per_person": 25000,
                "per_accident": 50000,
                "label": "25/50"
            }
        },
        "deductibles": {
            "collision": {
                "amount": 500,
                "label": "500"
            },
            "comprehensive": {
                "amount": 500,
                "label": "500"
            }
        },
        "exclusions": [
            "Racing",
            "Commercial use",
            "Intentional damage",
            "Driving under influence"
        ],
        "addOns": [
            "Roadside Assistance"
        ]
    }

def check_azure_openai_deployments():
    """Verify Azure OpenAI deployments are available following Azure best practices"""
    try:
        # Create client from environment variables
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # List available deployments/models
        models = client.models.list()
        
        print("\n=== Available Azure OpenAI Deployments ===")
        for model in models.data:
            print(f"- {model.id}")
        print("=========================================\n")
        
        return [model.id for model in models.data]
    except Exception as e:
        print(f"⚠️ Error checking Azure OpenAI deployments: {str(e)}")
        return []


def display_hera_recommendations(current_state, stage):
    """
    Display recommendations from Hera to the end user following Azure best practices.
    
    Args:
        current_state (dict): Current workflow state
        stage (str): The stage for which to display recommendations
    """
    if "hera_recommendations" in current_state and stage in current_state["hera_recommendations"]:
        recommendations = current_state["hera_recommendations"][stage]
        
        print("\n=== RECOMMENDED COVERAGES FROM SIMILAR CUSTOMERS ===")
        
        if "recommended_coverages" in recommendations:
            for idx, coverage in enumerate(recommendations["recommended_coverages"], 1):
                print(f"\nRecommendation #{idx}:")
                
                if coverage.get("coverages"):
                    print(f"  Coverages: {', '.join(coverage['coverages'])}")
                
                if coverage.get("limits"):
                    print("  Limits:")
                    for limit_name, limit_value in coverage["limits"].items():
                        print(f"    - {limit_name}: {limit_value}")
                
                if coverage.get("deductibles"):
                    print("  Deductibles:")
                    for ded_name, ded_value in coverage["deductibles"].items():
                        print(f"    - {ded_name}: {ded_value}")
                        
                if coverage.get("premium") is not None:
                    print(f"  Premium: ${coverage['premium']:.2f}")
        else:
            print("No specific coverage recommendations available at this stage.")
            
        print("\n===================================================\n")
    else:
        print("\nNo recommendations available from Hera for this stage.\n")

def process_with_hera(current_state, stage):
    """
    Process the current state with Hera at different workflow stages
    
    Args:
        current_state (dict): Current workflow state
        stage (str): Current workflow stage (iris, mnemosyne, ares)
        
    Returns:
        dict: Updated workflow state with Hera recommendations
    """
    print(f"\n[Workflow] Processing stage {stage} with Hera...\n")
    
    # Track which Hera stages have been processed - FIX THE RECURSION ERROR
    if "hera_processed_stages" not in current_state:
        # Replace recursive call with direct initialization
        current_state["hera_processed_stages"] = []
    
    # Check if this specific stage has already been processed by Hera
    stage_key = f"hera_{stage}"
    if stage_key in current_state["hera_processed_stages"]:
        print(f"⚠️ Warning: Stage '{stage}' has already been processed by Hera. Skipping to prevent loop.")
        return current_state
    
    # Add this stage to processed stages
    current_state["hera_processed_stages"].append(stage_key)
    
    # Call Hera to get recommendations based on the current stage
    recommendations = get_profile_recommendations(current_state, stage)
    
    # Store recommendations in the workflow state
    if "hera_recommendations" not in current_state:
        current_state["hera_recommendations"] = {}
    
    current_state["hera_recommendations"][stage] = recommendations
    
    # Log the workflow progress
    print(f"✅ Hera processing complete for stage: {stage}")
    print(f"📊 Processed stages so far: {current_state['hera_processed_stages']}")
    
    # Display the recommendations for immediate feedback
    display_hera_recommendations(current_state, stage)
    
    return current_state


def call_customerprofile(customer_data):
    """
    Process customer data using direct module import instead of subprocess
    
    Args:
        customer_data (dict): Customer profile information
        
    Returns:
        dict: Processed customer profile with recommendations
    """
    logger.info("Starting customer profile processing")
    try:
        # Import the module directly instead of using subprocess
        from customerprofile import process_customer_data
        
        # Call the function directly with the data
        result = process_customer_data(customer_data)
        
        logger.info("Customer profile processing completed")
        return result
    except Exception as e:
        logger.error(f"Error processing customer profile: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
def parse_profile_output(output):
    """
    Parse the output from customerprofile.py to extract match information
    
    Args:
        output (str): Output from customerprofile.py
        
    Returns:
        dict: Dictionary with matches and suggestion
    """
    result = {
        "matches": [],
        "suggestion": "No specific suggestion available based on the provided data."
    }
    
    # Extract matches
    match_sections = output.split("--- MATCH #")
    
    for i in range(1, len(match_sections)):
        section = match_sections[i]
        match = {}
        
        # Extract similarity
        similarity_match = section.split("(Similarity: ")
        if len(similarity_match) > 1:
            similarity_str = similarity_match[1].split(")")[0]
            try:
                match["similarity"] = float(similarity_str)
            except:
                match["similarity"] = 0.0
                
        # Extract policy number
        policy_match = section.split("Policy Number: ")
        if len(policy_match) > 1:
            match["policyNumber"] = policy_match[1].split("\n")[0].strip()
            
        # Extract coverages
        coverage_match = section.split("Coverages: ")
        if len(coverage_match) > 1:
            coverages_str = coverage_match[1].split("\n")[0].strip()
            match["coverages"] = [cov.strip() for cov in coverages_str.split(",")]
            
        # Extract add-ons
        addon_match = section.split("Add-ons: ")
        if len(addon_match) > 1:
            addons_str = addon_match[1].split("\n")[0].strip()
            match["addOns"] = [addon.strip() for addon in addons_str.split(",")]
            
        # Extract premium
        premium_match = section.split("Premium: $")
        if len(premium_match) > 1:
            try:
                premium_str = premium_match[1].split("\n")[0].strip()
                match["premium"] = float(premium_str)
            except:
                pass
                
        # Extract limits
        limits = {}
        limits_section_match = section.split("Key Limits:")
        if len(limits_section_match) > 1:
            limits_section = limits_section_match[1].split("\n  Deductibles:")[0]
            for line in limits_section.split("\n"):
                if ":" in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().replace("    ", "")
                        value = parts[1].strip()
                        limits[key] = value
        
        if limits:
            match["limits"] = limits
            
        # Extract deductibles
        deductibles = {}
        deductibles_section_match = section.split("Deductibles:")
        if len(deductibles_section_match) > 1:
            deductibles_section = deductibles_section_match[1].split("\n\n")[0]
            for line in deductibles_section.split("\n"):
                if ":" in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().replace("    ", "")
                        value = parts[1].strip()
                        deductibles[key] = value
        
        if deductibles:
            match["deductibles"] = deductibles
                
        # Add match to results
        result["matches"].append(match)
    
    # Generate a suggestion based on the matches
    if result["matches"]:
        # Count coverages to find most common
        coverage_counter = {}
        for match in result["matches"]:
            for coverage in match.get("coverages", []):
                if coverage in coverage_counter:
                    coverage_counter[coverage] += 1
                else:
                    coverage_counter[coverage] = 1
        
        # Sort coverages by frequency
        sorted_coverages = sorted(coverage_counter.items(), key=lambda x: x[1], reverse=True)
        top_coverages = [cov for cov, _ in sorted_coverages[:5]]
        
        # Generate suggestion
        premiums = [match.get("premium", 0) for match in result["matches"] if "premium" in match]
        
        suggestion = "Based on similar customer profiles, consider these popular coverages: "
        suggestion += ", ".join(top_coverages)
        
        if premiums:
            avg_premium = sum(premiums) / len(premiums)
            suggestion += f". Typical premium range: ${min(premiums):.2f} - ${max(premiums):.2f}, averaging ${avg_premium:.2f}."
            
        result["suggestion"] = suggestion
    
    return result
# Add this function if it doesn't exist or modify it if it does

def explore_coverage_options(demeter, user_proxy):
    """
    Interactively retrieve, explain, and select coverage options from autopm container.
    
    Args:
        demeter: Demeter agent for coverage expertise and explanations
        user_proxy: User proxy for input collection
        
    Returns:
        dict: User selected coverage options by coverage type
    """
    print("\n=== COVERAGE OPTIONS EXPLORER ===")
    print("We'll go through available coverage options and explain what each means.\n")
    
    selected_options = {}
    
    try:
        # Step 1: Retrieve coverage data from autopm container
        container_client = get_container_client("autopm")
        query = "SELECT * FROM c WHERE IS_DEFINED(c.productModel.coverageCategories)"
        items = list(container_client.query_items(
            query=query, 
            enable_cross_partition_query=True
        ))
        
        if not items:
            logger.error("No coverage data found in autopm container")
            print("⚠️ Could not retrieve coverage options from database.")
            return selected_options
            
        product_model = items[0].get("productModel", {})
        coverage_categories = product_model.get("coverageCategories", [])
        
        if not coverage_categories:
            logger.error("No coverage categories found in product model")
            print("⚠️ No coverage categories available.")
            return selected_options
            
        # Step 2: Process each category
        for category_index, category in enumerate(coverage_categories):
            if not isinstance(category, dict):
                logger.warning(f"Invalid category format: {type(category)}")
                continue
                
            category_name = category.get("name", f"Category {category_index+1}")
            print(f"\n{'='*20}")
            print(f"✨ {category_name.upper()}")
            print(f"{'='*20}")
            
            # Ask Demeter for category explanation
            category_explanation = _get_category_explanation(demeter, category_name)
            if category_explanation:
                print(f"\nDemeter: {category_explanation}\n")
                
            coverages = category.get("coverages", [])
            if not coverages:
                print("No coverages available in this category.")
                continue
                
            # Step 3: Process each coverage
            for coverage in coverages:
                if not isinstance(coverage, dict):
                    continue
                    
                coverage_name = coverage.get("name", "Unnamed Coverage")
                mandatory = coverage.get("mandatory", False)
                
                print(f"\n📋 {coverage_name} {'(Mandatory)' if mandatory else '(Optional)'}")
                
                # Get coverage explanation
                coverage_explanation = _get_coverage_explanation(demeter, coverage_name)
                if coverage_explanation:
                    print(f"{coverage_explanation}\n")
                
                # Step 4: Process each coverage term
                coverage_terms = coverage.get("coverageTerms", [])
                if not coverage_terms:
                    print("No configurable options available for this coverage.")
                    continue
                    
                for term_index, term in enumerate(coverage_terms):
                    if not isinstance(term, dict):
                        logger.warning(f"Invalid term format for {coverage_name}: {type(term)}")
                        continue
                        
                    term_name = term.get("termName", f"Term {term_index+1}")
                    model_type = term.get("modelType", "Unknown")
                    
                    print(f"\n🔹 {term_name} ({model_type})")
                    
                    # Get options
                    options = term.get("options", [])
                    if not options:
                        print("No options available for this term.")
                        continue
                    
                    # Get term explanation
                    term_explanation = _get_term_explanation(demeter, coverage_name, term_name, model_type)
                    if term_explanation:
                        print(f"{term_explanation}\n")
                    
                    # Display options
                    print("Available options:")
                    for i, option in enumerate(options):
                        if not isinstance(option, dict):
                            continue
                        
                        req_id = option.get("requirementId", "N/A")
                        label = option.get("label", "Unnamed Option")
                        min_val = option.get("min", "N/A")
                        max_val = option.get("max", "N/A")
                        description = option.get("description", "No description available")
                        
                        # Format for display
                        range_info = ""
                        if min_val != "N/A" and max_val != "N/A":
                            range_info = f" (Range: {min_val}-{max_val})"
                        
                        print(f"  {i+1}. {label}{range_info}")
                        print(f"     Description: {description}")
                        if req_id != "N/A":
                            print(f"     Requirement ID: {req_id}")
                    
                    # Get user selection with validation
                    if user_proxy.get_human_input(f"\nWould you like to select an option for {coverage_name} {term_name}? (yes/no): ").lower().startswith('y'):
                        selected_option = _get_user_selection(user_proxy, options)
                        if selected_option:
                            option_key = f"{coverage_name}_{term_name}"
                            selected_options[option_key] = selected_option
                            print(f"✅ Selected: {selected_option.get('label', 'Unknown')} for {coverage_name} {term_name}")
        
        # Step 5: Summarize selections
        if selected_options:
            print("\n=== YOUR SELECTED OPTIONS ===")
            for key, option in selected_options.items():
                print(f"• {key}: {option.get('label', 'Unknown')} - {option.get('description', 'No description')}")
        else:
            print("\nNo options were selected.")
            
        return selected_options
            
    except Exception as e:
        logger.error(f"Error in coverage options explorer: {str(e)}", exc_info=True)
        print(f"\n⚠️ Error exploring coverage options: {str(e)}")
        return selected_options

def _get_category_explanation(demeter, category_name):
    """Get explanation about a coverage category from Demeter"""
    try:
        prompt = f"""
        Explain the '{category_name}' category of auto insurance in simple terms.
        What types of coverages does this category include and why are they important?
        Keep your explanation under 75 words and make it conversational.
        """
        
        explanation, _, success = query_agent(
            demeter, 
            prompt, 
            gpt4o_deployment, 
            f"Explain {category_name} category"
        )
        
        return explanation if success else None
    except Exception as e:
        logger.warning(f"Error getting category explanation: {str(e)}")
        return None

def _get_coverage_explanation(demeter, coverage_name):
    """Get explanation about a specific coverage from Demeter"""
    try:
        prompt = f"""
        Explain '{coverage_name}' coverage in simple terms.
        What does it protect against? Who needs it? Why is it important?
        Keep your explanation under 50 words and make it conversational.
        """
        
        explanation, _, success = query_agent(
            demeter, 
            prompt, 
            gpt4o_deployment, 
            f"Explain {coverage_name}"
        )
        
        return explanation if success else None
    except Exception as e:
        logger.warning(f"Error getting coverage explanation: {str(e)}")
        return None

def _get_term_explanation(demeter, coverage_name, term_name, model_type):
    """Get explanation about a coverage term from Demeter"""
    try:
        prompt = f"""
        Explain what '{term_name}' means for {coverage_name} in simple terms.
        This is a {model_type} type option.
        
        Explain:
        1. What this option controls in the policy
        2. How different values impact coverage and premium
        3. What a typical customer should consider when selecting a value
        
        Keep your explanation under 75 words and make it very conversational.
        """
        
        explanation, _, success = query_agent(
            demeter, 
            prompt, 
            gpt4o_deployment, 
            f"Explain {coverage_name} {term_name}"
        )
        
        return explanation if success else None
    except Exception as e:
        logger.warning(f"Error getting term explanation: {str(e)}")
        return None

def _get_user_selection(user_proxy, options):
    """Get user selection with proper validation"""
    if not options:
        return None
        
    while True:
        try:
            choice = user_proxy.get_human_input(f"Select an option (1-{len(options)}): ").strip()
            
            # Check if user wants to skip
            if not choice or choice.lower() in ('skip', 'none', 'cancel'):
                return None
                
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(options):
                return options[choice_idx]
            
            print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number or 'skip' to not select any option.")

# filepath: c:\Users\pramadasan\insurance_app\workflow\process.py


def design_coverage_with_demeter(current_state, demeter, iris, user_proxy):
    """
    Interactive coverage design with Demeter following Azure best practices for customer choice.
    Provides interactive selection for both mandatory and optional coverages.

    Args:
        current_state: Current workflow state
        demeter: Demeter agent for coverage expertise
        iris: Iris agent for customer interaction
        user_proxy: User proxy for input collection

    Returns:
        dict: Customer-selected coverage options
    """
    logger.info("Starting interactive coverage design with Demeter")

    try:
        # Step 1: Get coverage data from Cosmos DB
        # @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.
        # Tool call is commented out, so commenting out the usage below as well.
        # try:
        #     # best_practices = get_azure_best_practices(operation='CosmosDBQuery')
        #     # logger.info(f"Cosmos DB Query Best Practices: {best_practices}")
        #     pass # Placeholder if tool call is not made
        # except Exception as e:
        #     logger.warning(f"Could not fetch Azure best practices for Cosmos DB query: {str(e)}")

        container_client = get_container_client("autopm")
        # Azure Best Practice: Use specific query projection if possible, but SELECT * is okay for product models
        query = "SELECT * FROM c WHERE IS_DEFINED(c.productModel.coverageCategories)"
        items = list(container_client.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            logger.error("No coverage data found in autopm container.")
            raise ValueError("No coverage data found in autopm container.")

        # Step 2: Extract and organize coverage options
        # Azure Best Practice: Validate the structure of the retrieved data
        product_model = items[0].get("productModel", {})
        coverage_categories = product_model.get("coverageCategories", [])
        if not isinstance(coverage_categories, list):
             logger.error(f"Expected coverageCategories to be a list, but got {type(coverage_categories)}")
             raise ValueError("Invalid coverage data structure retrieved from database.")

        # Step 3: Have Demeter introduce the coverage selection process
        print("\n=== COVERAGE SELECTION WITH DEMETER ===")
        introduction_prompt = """
        Create a friendly introduction to auto insurance coverage selection that:
        1. Explains the difference between mandatory and optional coverages
        2. Describes the concept of limits and deductibles in simple terms
        3. Reassures the customer that you'll guide them through each decision
        4. Mentions that we'll start with required coverages before moving to optional ones

        Keep your response conversational, helpful and under 150 words.
        """

        try:
            # Azure Best Practice: Use standardized agent interaction with error handling
            introduction, _, success = query_agent(demeter, introduction_prompt, gpt4o_deployment, "Coverage Introduction")
            if success:
                print(f"\n{introduction}\n")
            else:
                 # Fallback message if Demeter fails
                 print("\nWelcome to coverage selection! I'll guide you through choosing both required and optional coverages for your policy. We'll start with mandatory coverages, then look at optional ones. For many coverages, you can choose limits (how much the policy pays) and deductibles (how much you pay first).\n")
        except Exception as e:
            logger.error(f"Error getting introduction: {str(e)}")
            print("\nWelcome to coverage selection! I'll guide you through choosing both required and optional coverages for your policy. We'll start with mandatory coverages, then look at optional ones. For many coverages, you can choose limits (how much the policy pays) and deductibles (how much you pay first).\n")

        # Step 4: Identify mandatory coverages
        mandatory_coverages = []
        all_coverages_data = {} # Store all coverage data for later lookup
        for category in coverage_categories:
             # Azure Best Practice: Add type checking for category structure
             if not isinstance(category, dict):
                 logger.warning(f"Skipping invalid category format: {type(category)}")
                 continue
             for coverage in category.get("coverages", []):
                 # Azure Best Practice: Add type checking for coverage structure
                 if not isinstance(coverage, dict):
                     logger.warning(f"Skipping invalid coverage format in category '{category.get('name', 'Unknown')}': {type(coverage)}")
                     continue
                 coverage_name = coverage.get("name")
                 if coverage_name:
                     all_coverages_data[coverage_name] = coverage # Store full data
                     if coverage.get("mandatory", False):
                        mandatory_coverages.append(coverage) # Store full data for mandatory

        print("\n=== MANDATORY COVERAGES ===")
        print("The following coverages are required by law or policy requirements:")
        for coverage in mandatory_coverages:
            print(f"✓ {coverage.get('name', 'Unnamed Coverage')}") # Use default if name missing

        print("\nEven though these coverages are mandatory, you can still choose specific options for each.")

        # Step 5: Process MANDATORY coverages interactively
        selected_coverages = []
        limits = {}
        deductibles = {}
        # Initialize policy structure if not present
        policy_state_for_options = {"coverage": {"limits": {}, "deductibles": {}}}

        print("\n=== CONFIGURING MANDATORY COVERAGES ===")

        for coverage in mandatory_coverages:
            # Ensure coverage is a dictionary before proceeding
            if not isinstance(coverage, dict):
                logger.warning(f"Skipping mandatory coverage due to invalid format: {type(coverage)}")
                continue

            coverage_name = coverage.get("name", "Unnamed Coverage")
            selected_coverages.append(coverage_name)

            # Have Demeter explain this specific coverage
            explain_prompt = f"""
            Explain the following mandatory auto insurance coverage in simple, everyday language:

            Coverage: {coverage_name}

            In your explanation:
            1. Describe what this coverage protects against
            2. Provide a real-world example of when it would be used
            3. Explain why it's mandatory
            4. Keep your explanation under 100 words and very conversational
            """

            try:
                # Azure Best Practice: Use standardized agent interaction
                explanation, _, success = query_agent(demeter, explain_prompt, gpt4o_deployment, f"Explain {coverage_name}")
                print(f"\n--- {coverage_name} ---")
                if success:
                    print(f"{explanation}\n")
                else:
                    print("This is a mandatory coverage required for your policy.\n") # Fallback explanation
            except Exception as e:
                logger.error(f"Error getting coverage explanation for {coverage_name}: {str(e)}")
                print(f"\n--- {coverage_name} ---")
                print("This is a mandatory coverage required for your policy.\n")

            # Process coverage terms (limits and deductibles) for this mandatory coverage
            coverage_terms_data = coverage.get("coverageTerms") # Get the raw data
            # Azure Best Practice: Log the structure for debugging if needed
            logger.debug(f"Processing coverageTerms for {coverage_name}: Type={type(coverage_terms_data)}, Data={coverage_terms_data}")

            # FIX: Handle both list and dict for coverageTerms
            terms_to_process = []
            if isinstance(coverage_terms_data, list):
                terms_to_process = coverage_terms_data
            elif isinstance(coverage_terms_data, dict):
                terms_to_process = [coverage_terms_data] # Wrap single dict in a list
            else:
                logger.warning(f"Expected coverageTerms for {coverage_name} to be a list or dict, but got {type(coverage_terms_data)}. Skipping terms.")
                continue # Skip processing terms for this coverage if the structure is wrong

            for term in terms_to_process:
                # ***** FIX: Add type checking for 'term' *****
                if not isinstance(term, dict):
                    logger.warning(f"Skipping invalid term format for {coverage_name}: Expected dict, got {type(term)}. Term data: {term}")
                    continue # Skip this term and proceed to the next

                # Now it's safe to use .get()
                model_type = term.get("modelType")
                options = term.get("options", [])

                # Ensure options is a list before proceeding
                if not isinstance(options, list):
                    logger.warning(f"Expected options for {model_type} in {coverage_name} to be a list, but got {type(options)}. Skipping options.")
                    continue

                if not options:
                    logger.info(f"No options found for {model_type} in {coverage_name}. Skipping.")
                    continue

                # --- Process Limits ---
                if model_type == "Limit":
                    try: # Add try/except around limit processing for robustness
                        # Call helper function for interactive selection
                        policy_state_for_options = _handle_limit_options(policy_state_for_options, coverage_name, term, options, demeter, user_proxy)
                        # Update main limits dict from the result
                        limits.update(policy_state_for_options.get("coverage", {}).get("limits", {}))
                    except Exception as limit_err:
                        logger.error(f"Error processing limits for {coverage_name}: {limit_err}", exc_info=True)
                        print(f"⚠️ Error configuring limits for {coverage_name}. Using default.")
                        # Apply a default limit if processing fails
                        if options and isinstance(options[0], dict):
                             limits[coverage_name] = options[0].get('value', 0) # Store default directly

                # --- Process Deductibles ---
                elif model_type == "Deductible":
                    try: # Add try/except around deductible processing
                        # Call helper function for interactive selection
                        policy_state_for_options = _handle_deductible_options(policy_state_for_options, coverage_name, term, options, demeter, user_proxy)
                        # Update main deductibles dict from the result
                        deductibles.update(policy_state_for_options.get("coverage", {}).get("deductibles", {}))
                    except Exception as ded_err:
                        logger.error(f"Error processing deductibles for {coverage_name}: {ded_err}", exc_info=True)
                        print(f"⚠️ Error configuring deductibles for {coverage_name}. Using default.")
                        # Apply a default deductible if processing fails
                        if options and isinstance(options[0], dict):
                             deductibles[coverage_name] = options[0].get('value', 0) # Store default directly
                else:
                     logger.warning(f"Unsupported modelType '{model_type}' found for term in {coverage_name}. Skipping.")

        # Step 6: Process OPTIONAL coverages (Add similar type checking here)
        print("\n=== OPTIONAL COVERAGES ===")
        # FIX: Define optional_prompt
        optional_prompt = """
        Explain what optional auto insurance coverages are in general.
        Mention examples like Collision, Comprehensive, Rental Reimbursement, or Roadside Assistance.
        Explain that these add extra protection beyond the mandatory requirements.
        Keep it brief (under 75 words) and conversational.
        """
        try:
            optional_overview, _, success = query_agent(demeter, optional_prompt, gpt4o_deployment, "Optional Coverage Overview")
            if success:
                print(f"\n{optional_overview}\n")
            else:
                # Fallback message
                print("\nOptional coverages offer extra protection beyond the basics, like covering damage to your own car or providing help if you break down.\n")
        except Exception as e:
            logger.error(f"Error getting optional coverage overview: {str(e)}")
            print("\nOptional coverages offer extra protection beyond the basics, like covering damage to your own car or providing help if you break down.\n") # Fallback

        # Process each coverage category
        for category in coverage_categories:
             # Azure Best Practice: Add type checking for category structure
             if not isinstance(category, dict):
                 logger.warning(f"Skipping invalid optional category format: {type(category)}")
                 continue

             category_name = category.get("name", "Coverage Category")

             # Check if category has optional coverages
             optional_in_category = False
             category_coverages = category.get("coverages", [])
             if not isinstance(category_coverages, list):
                 logger.warning(f"Expected coverages in category '{category_name}' to be a list, got {type(category_coverages)}. Skipping category.")
                 continue

             # Filter for optional coverages within this category
             optional_coverages_in_cat = [
                 cov for cov in category_coverages
                 if isinstance(cov, dict) and not cov.get("mandatory", False)
             ]

             if not optional_coverages_in_cat:
                continue # Skip category if no optional coverages

             print(f"\n--- Optional Coverages in {category_name} ---")
             # Optional: Add category explanation here if desired

             # Present each optional coverage in this category
             for coverage in optional_coverages_in_cat:
                 # Type checking already done when filtering
                 coverage_name = coverage.get("name", "Unnamed Optional Coverage")

                 # Explain the optional coverage
                 explain_optional_prompt = f"""
                 Explain the optional auto insurance coverage '{coverage_name}' in simple terms.
                 What does it cover? Who might benefit from adding it?
                 Keep it under 75 words and conversational.
                 """
                 try:
                     explanation, _, success = query_agent(demeter, explain_optional_prompt, gpt4o_deployment, f"Explain Optional {coverage_name}")
                     print(f"\n--- {coverage_name} ---")
                     if success:
                         print(f"{explanation}\n")
                     else:
                         print("This is an optional coverage you can add for extra protection.\n") # Fallback
                 except Exception as e:
                     logger.error(f"Error getting optional coverage explanation for {coverage_name}: {str(e)}")
                     print(f"\n--- {coverage_name} ---")
                     print("This is an optional coverage you can add for extra protection.\n")

                 # Ask if user wants to add this optional coverage
                 add_coverage = user_proxy.get_human_input(f"Would you like to add {coverage_name} to your policy? (yes/no): ").strip().lower()
                 if add_coverage == "yes" or add_coverage == "y":
                    if coverage_name not in selected_coverages: # Avoid duplicates
                        selected_coverages.append(coverage_name)
                        print(f"✅ Added {coverage_name}.")
                    else:
                        logger.info(f"Coverage {coverage_name} already selected.")
                        print(f"{coverage_name} is already included.")

                    # Process coverage terms (limits and deductibles) for this optional coverage
                    coverage_terms_data = coverage.get("coverageTerms") # Get raw data
                    # FIX: Handle both list and dict for coverageTerms
                    terms_to_process = []
                    if isinstance(coverage_terms_data, list):
                        terms_to_process = coverage_terms_data
                    elif isinstance(coverage_terms_data, dict):
                        terms_to_process = [coverage_terms_data] # Wrap single dict in a list
                    else:
                        logger.warning(f"Expected coverageTerms for optional {coverage_name} to be a list or dict, but got {type(coverage_terms_data)}. Skipping terms.")
                        continue # Skip processing terms for this coverage if the structure is wrong

                    for term in terms_to_process:
                        # ***** FIX: Add type checking for 'term' *****
                        if not isinstance(term, dict):
                            logger.warning(f"Skipping invalid term format for optional {coverage_name}: Expected dict, got {type(term)}. Term data: {term}")
                            continue # Skip this term

                        model_type = term.get("modelType")
                        options = term.get("options", [])

                        # Ensure options is a list before proceeding
                        if not isinstance(options, list):
                            logger.warning(f"Expected options for {model_type} in optional {coverage_name} to be a list, but got {type(options)}. Skipping options.")
                            continue

                        if not options:
                            logger.info(f"No options found for {model_type} in optional {coverage_name}. Skipping.")
                            continue

                        # --- Process Limits (Optional) ---
                        if model_type == "Limit":
                             try: # Add try/except around limit processing
                                 # Call helper function for interactive selection
                                 policy_state_for_options = _handle_limit_options(policy_state_for_options, coverage_name, term, options, demeter, user_proxy)
                                 # Update main limits dict from the result
                                 limits.update(policy_state_for_options.get("coverage", {}).get("limits", {}))
                             except Exception as limit_err:
                                 logger.error(f"Error processing limits for optional {coverage_name}: {limit_err}", exc_info=True)
                                 print(f"⚠️ Error configuring limits for {coverage_name}. Using default.")
                                 if options and isinstance(options[0], dict):
                                      limits[coverage_name] = options[0].get('value', 0) # Store default directly

                        # --- Process Deductibles (Optional) ---
                        elif model_type == "Deductible":
                             try: # Add try/except around deductible processing
                                 # Call helper function for interactive selection
                                 policy_state_for_options = _handle_deductible_options(policy_state_for_options, coverage_name, term, options, demeter, user_proxy)
                                 # Update main deductibles dict from the result
                                 deductibles.update(policy_state_for_options.get("coverage", {}).get("deductibles", {}))
                             except Exception as ded_err:
                                 logger.error(f"Error processing deductibles for optional {coverage_name}: {ded_err}", exc_info=True)
                                 print(f"⚠️ Error configuring deductibles for {coverage_name}. Using default.")
                                 if options and isinstance(options[0], dict):
                                      deductibles[coverage_name] = options[0].get('value', 0) # Store default directly
                        else:
                             logger.warning(f"Unsupported modelType '{model_type}' found for term in optional {coverage_name}. Skipping.")

                 else: # User chose not to add the coverage
                     logger.info(f"User declined optional coverage: {coverage_name}")
                     # Ensure it's removed if it was somehow added previously (unlikely here but safe)
                     if coverage_name in selected_coverages:
                         selected_coverages.remove(coverage_name)
                     if coverage_name in limits:
                         del limits[coverage_name]
                     if coverage_name in deductibles:
                         del deductibles[coverage_name]


        # Step 7: Process add-ons (Add similar type checking here)
        addOns = []
        # Find categories that are likely add-ons (e.g., name contains "Add-on")
        addon_categories = [cat for cat in coverage_categories if isinstance(cat, dict) and "Add-on" in cat.get("name", "")]

        if addon_categories:
            print("\n=== ADD-ON OPTIONS ===")
            # Add-on explanation prompt
            addon_prompt = """
            Explain what Add-ons (also called endorsements or riders) are in auto insurance.
            Give examples like Roadside Assistance, Rental Reimbursement, or Gap Insurance.
            Keep it brief (under 75 words) and conversational.
            """
            try:
                addon_explanation, _, success = query_agent(demeter, addon_prompt, gpt4o_deployment, "Add-on Explanation")
                if success:
                    print(f"\n{addon_explanation}\n")
                else:
                    print("\nAdd-ons provide extra protection for specific situations, often for an additional cost.\n") # Fallback
            except Exception as e:
                 logger.error(f"Error getting add-on explanation: {str(e)}")
                 print("\nAdd-ons provide extra protection for specific situations, often for an additional cost.\n") # Fallback

            for category in addon_categories:
                 # Ensure category is dict
                 if not isinstance(category, dict): continue
                 category_coverages = category.get("coverages", [])
                 # Ensure coverages is list
                 if not isinstance(category_coverages, list): continue

                 for coverage in category_coverages:
                     # Ensure coverage is dict
                     if not isinstance(coverage, dict): continue

                     addon_name = coverage.get("name", "Unnamed Add-on")
                     # Add-on specific explanation prompt
                     addon_specific_prompt = f"""
                     Explain the '{addon_name}' add-on in simple terms.
                     What does it provide? Who might find it useful?
                     Keep it under 50 words and conversational.
                     """
                     try:
                         addon_specific_explanation, _, success = query_agent(demeter, addon_specific_prompt, gpt4o_deployment, f"Explain Add-on {addon_name}")
                         print(f"\n--- {addon_name} ---")
                         if success:
                             print(f"{addon_specific_explanation}\n")
                         else:
                             print("This is an optional add-on for extra convenience or protection.\n") # Fallback
                     except Exception as e:
                         logger.error(f"Error getting specific add-on explanation for {addon_name}: {str(e)}")
                         print(f"\n--- {addon_name} ---")
                         print("This is an optional add-on for extra convenience or protection.\n")

                     add_addon = user_proxy.get_human_input(f"Would you like to add the {addon_name} to your policy? (yes/no): ").strip().lower()
                     if add_addon == "yes" or add_addon == "y":
                        if addon_name not in addOns: # Avoid duplicates
                            addOns.append(addon_name)
                            print(f"✓ {addon_name} added to your policy.\n")
                        else:
                            logger.info(f"Add-on {addon_name} already selected.")
                            print(f"{addon_name} is already included.")
                     else:
                         logger.info(f"User declined add-on: {addon_name}")
                         # Ensure removed if previously added (unlikely here)
                         if addon_name in addOns:
                             addOns.remove(addon_name)


        # Step 8: Display summary of selections
        print("\n=== YOUR COVERAGE SELECTIONS SUMMARY ===")
        try:
            # Ensure limits/deductibles are serializable for the prompt
            serializable_limits = {k: (v if isinstance(v, (int, float, str, dict)) else str(v)) for k, v in limits.items()}
            serializable_deductibles = {k: (v if isinstance(v, (int, float, str, dict)) else str(v)) for k, v in deductibles.items()}

            summary_prompt = f"""
            Create a friendly, personalized summary of the customer's selected coverage options.

            Selected coverages: {selected_coverages}
            Limits: {json.dumps(serializable_limits)}
            Deductibles: {json.dumps(serializable_deductibles)}
            Add-ons: {addOns}

            In your summary:
            1. List the selected coverages clearly.
            2. Mention the chosen limits and deductibles for key coverages.
            3. List any selected add-ons.
            4. Provide a brief statement about the overall protection level.
            5. Thank them for their selections.

            Keep it under 150 words and conversational. Format using bullet points for clarity.
            """
            summary, _, success = query_agent(demeter, summary_prompt, gpt4o_deployment, "Coverage Summary")
            if success:
                print(f"\n{summary}\n")
            else:
                 # Fallback summary
                 print("\nHere's a summary of your selections:")
                 print("Coverages:")
                 for cov in selected_coverages: print(f"- {cov}")
                 if limits:
                     print("\nLimits:")
                     for k, v in limits.items(): print(f"- {k}: {v}")
                 if deductibles:
                     print("\nDeductibles:")
                     for k, v in deductibles.items(): print(f"- {k}: {v}")
                 if addOns:
                     print("\nAdd-ons:")
                     for addon in addOns: print(f"- {addon}")
                 print("\nThank you for customizing your coverage!")

        except Exception as e:
            logger.error(f"Error generating coverage summary: {str(e)}")
            # Provide a basic summary if Demeter fails
            print("\nYou've selected a comprehensive set of coverages for your policy.")
            print(f"Your selections include {len(selected_coverages)} coverages and {len(addOns)} add-ons.")
            print("Thank you for customizing your coverage options!")

        # Build the coverage plan object
        coverage_plan = {
            "coverages": selected_coverages,
            "limits": limits, # Use the original limits dict
            "deductibles": deductibles, # Use the original deductibles dict
            "addOns": addOns,
            "exclusions": ["Racing", "Commercial Use"] # Consider making exclusions dynamic
        }

        # Ask for final confirmation
        confirmation = user_proxy.get_human_input("\nConfirm these coverage selections? (yes/no): ").strip().lower()
        if confirmation != "yes":
            print("Let's try again with coverage selection.")
            # Azure Best Practice: Avoid deep recursion; consider iterative approach or state reset if needed
            # For simplicity here, we allow one level of recursion. Add max depth check if needed.
            # Check recursion depth if necessary: import sys; if sys.getrecursionlimit() < current_depth + 10: raise RecursionError("Max depth exceeded")
            return design_coverage_with_demeter(current_state, demeter, iris, user_proxy)

        logger.info("Customer confirmed coverage selections")
        return coverage_plan

    except ValueError as ve: # Catch specific configuration errors
         logger.error(f"Configuration error in coverage design: {str(ve)}", exc_info=True)
         print(f"\n⚠️ Configuration Error: {str(ve)}")
         print("Using default coverage options instead.")
         return get_default_coverage_data(current_state) # Ensure this function exists and returns a valid dict
    except Exception as e:
        # Catch-all for unexpected errors during coverage design
        logger.error(f"Unexpected error in coverage design with Demeter: {str(e)}", exc_info=True)
        print(f"\n⚠️ An unexpected error occurred during coverage design: {str(e)}")
        print("Using default coverage options instead.")
        # Azure Best Practice: Ensure fallback function is robust
        return get_default_coverage_data(current_state) # Ensure this function exists and returns a valid dict


# --- Ensure helper functions also have robust type checking ---

def _handle_limit_options(policy, coverage_name, term, options, demeter, user_proxy):
    """Helper function to handle limit options with enhanced robustness."""
    # ... (existing explanation logic using query_agent) ...
    try:
        limit_prompt = (
            f"Explain what '{term.get('termName', 'limit')}' means for "
            f"{coverage_name} coverage in simple terms with examples."
        )
        explanation, _, success = query_agent(demeter, limit_prompt, gpt4o_deployment, f"Explain {coverage_name} Limit Options")
        if success:
            print(f"\nAbout this limit: {explanation}\n")
        else:
            print("\nAbout this limit: This defines the maximum amount your policy will pay for this coverage.\n") # Fallback
    except Exception as e:
        logger.error(f"Error explaining limit for {coverage_name}: {e}")
        print("\nAbout this limit: This defines the maximum amount your policy will pay for this coverage.\n")

    # Display options, filtering for valid dicts
    print(f"Select a limit for {coverage_name}:")
    valid_options = []
    for i, opt in enumerate(options):
        if isinstance(opt, dict):
            label = opt.get('label', '')
            desc = opt.get('description', '')
            print(f"  {len(valid_options)+1}. {label}: {desc}")
            valid_options.append(opt)
        else:
            logger.warning(f"Skipping invalid limit option format for {coverage_name}: {type(opt)}")

    if not valid_options:
        logger.error(f"No valid limit options available for {coverage_name}.")
        print(f"⚠️ No valid limit options found for {coverage_name}. Using default.")
        # Set a default value if no valid options
        policy.setdefault("coverage", {}).setdefault("limits", {})[coverage_name] = 0
        return policy

    # Get user choice with robust error handling
    selected = None
    while selected is None:
        choice = user_proxy.get_human_input(
            f"Enter option number (1-{len(valid_options)}): "
        ).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(valid_options):
                selected = valid_options[idx]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(valid_options)}.")
        except ValueError:
            print("Please enter a valid number.")

    # Ensure 'limits' key exists
    if "limits" not in policy.setdefault("coverage", {}):
        policy["coverage"]["limits"] = {}

    # Parse bodily injury style vs simple with improved robustness
    limit_value_to_store = selected.get('value', 0) # Default value
    limit_label_to_store = selected.get('label', '')

    label_str = str(selected.get('label', '')).lower()
    desc_str = str(selected.get('description', '')).lower()
    if "per person" in label_str or "per accident" in label_str or \
       "per person" in desc_str or "per accident" in desc_str:
        # Use raw string for regex
        m = re.findall(r'(\d+)(?:[,/](\d+))?', selected.get('label', ''))
        if m:
            try:
                per_person = int(m[0][0].replace(',', '')) * 1000 # Assume K
                per_accident = int(m[0][1].replace(',', '')) * 1000 if m[0][1] else per_person * 2
                limit_value_to_store = {
                    "per_person": per_person,
                    "per_accident": per_accident,
                    "label": limit_label_to_store # Store label
                }
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing per_person/per_accident values for {coverage_name}: {e}")
                # Fallback to simple value
                limit_value_to_store = selected.get('value', 0)

    # Store the determined limit value
    policy["coverage"]["limits"][coverage_name] = limit_value_to_store
    logger.info(f"Set limit for {coverage_name}: {limit_value_to_store}")

    return policy


def _handle_deductible_options(policy, coverage_name, term, options, demeter, user_proxy):
    """Helper function to handle deductible options with enhanced robustness."""
    # ... (existing explanation logic using query_agent) ...
    try:
        deductible_prompt = (
            f"Explain what a deductible is for {coverage_name} coverage "
            f"and how deductible amount affects premiums."
        )
        explanation, _, success = query_agent(demeter, deductible_prompt, gpt4o_deployment, f"Explain {coverage_name} Deductible Options")
        if success:
             print(f"\nAbout deductibles: {explanation}\n")
        else:
             print("\nAbout deductibles: This is the amount you pay before insurance covers the rest.\n") # Fallback
    except Exception as e:
        logger.error(f"Error explaining deductible for {coverage_name}: {e}")
        print("\nAbout deductibles: This is the amount you pay before insurance covers the rest.\n")

    # Ensure options is defined and is a list before use
    if not options or not isinstance(options, list):
        logger.error(f"Options for {coverage_name} deductibles are not defined or invalid.")
        print(f"\nⓧ Error: No deductible options available for {coverage_name}.")
        # Set a default value if no valid options
        policy.setdefault("coverage", {}).setdefault("deductibles", {})[coverage_name] = 0
        return policy

    # Display options, filtering for valid dicts
    print(f"Select a deductible for {coverage_name}:")
    print("Lower deductible → higher premium; higher deductible → lower premium.")
    valid_options = []
    for i, opt in enumerate(options):
        if isinstance(opt, dict):
            val = opt.get('value', 0)
            print(f"  {len(valid_options)+1}. ${val}")
            valid_options.append(opt)
        else:
            logger.warning(f"Skipping invalid deductible option format for {coverage_name}: {type(opt)}")

    if not valid_options:
        logger.error(f"No valid deductible options available for {coverage_name}.")
        print(f"⚠️ No valid deductible options found for {coverage_name}. Using default.")
        # Set a default value if no valid options
        policy.setdefault("coverage", {}).setdefault("deductibles", {})[coverage_name] = 0
        return policy

    # Get user choice with robust error handling
    selected = None
    while selected is None:
        choice = user_proxy.get_human_input(
            f"Enter option number (1-{len(valid_options)}): "
        ).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(valid_options):
                selected = valid_options[idx]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(valid_options)}.")
        except ValueError:
            print("Please enter a valid number.")

    # Ensure 'deductibles' key exists
    if "deductibles" not in policy.setdefault("coverage", {}):
        policy["coverage"]["deductibles"] = {}

    # Save selection using setdefault for safer dict operations
    deductible_value = selected.get('value', 0)
    policy["coverage"]["deductibles"][coverage_name] = deductible_value
    logger.info(f"Set deductible for {coverage_name}: {deductible_value}")

    return policy    
     
def get_default_coverage_data(current_state):
    """Generate default coverage when Demeter fails"""
    # ... (existing implementation of the default coverage logic) ...
    vehicle = current_state.get("customerProfile", {}).get("vehicle_details", {})
    vehicle_desc = f"{vehicle.get('year', '')} {vehicle.get('make', '')} {vehicle.get('model', '')}"

    print(f"Customizing default coverage for {vehicle_desc}")

    return {
        "coverages": [
            "Bodily Injury Liability",
            "Property Damage Liability",
            "Collision Coverage",
            "Comprehensive Coverage"
        ],
        "limits": {
            "bodily_injury": {"per_person": 50000, "per_accident": 100000},
            "property_damage": {"per_accident": 50000}
        },
        "deductibles": {
            "collision": {"amount": 500},
            "comprehensive": {"amount": 500}
        },
        "addOns": [
            "Roadside Assistance",
            "Rental Car Coverage"
        ],
        "exclusions": [
            "Intentional damage",
            "Racing or speed contests"
        ]
    }




def build_customer_profile(current_state, iris, mnemosyne, user_proxy):
    """Build detailed customer profile using Mnemosyne and underwriting questions"""
    print("\n=== UNDERWRITING VERIFICATION ===")
    print("Retrieving pre-qualification questions from autopm container...")
    
    # Initialize the Cosmos DB connection if not already done
    from db.cosmos_db import init_cosmos_db, get_mandatory_questions, get_questions_with_mnemosyne
    init_cosmos_db()  # Ensure DB connection is initialized
    
    # HYBRID APPROACH:
    # 1. First try to use Mnemosyne's language skills to extract enhanced questions
    mnemosyne_questions = get_questions_with_mnemosyne(mnemosyne)
    
    # 2. If that fails, fall back to code-based extraction
    if not mnemosyne_questions:
        print("Falling back to code-based question extraction...")
        questions = get_mandatory_questions()
        enhanced = False
    else:
        questions = mnemosyne_questions
        enhanced = True
    
    print(f"Processing {len(questions)} pre-qualification questions...")
    
    # Initial eligibility is True, will be set to False if any "Yes" answers to Decline questions
    eligibility = True
    eligibility_reason = "All underwriting criteria met"
    underwriting_responses = {}
    
    # Process each question sequentially
    for question in questions:
        question_id = question.get("id", "Unknown ID")
        question_text = question.get("text", "Unknown question")
        explanation = question.get("explanation", "")
        
        # Use enhanced explanation if available
        context = question.get("enhanced_explanation", explanation) if enhanced else explanation
        
        print(f"\n[Question {question.get('order', '?')}]: {question_text}")
        
        # If using enhanced questions, we can skip asking Mnemosyne for context
        if not enhanced:
            # Have Mnemosyne analyze the question
            try:
                prompt = f"Analyze this underwriting question and provide a brief explanation of why it matters for insurance: '{question_text}'. Explanation from policy: {explanation}"
                mnemosyne_response = mnemosyne.generate_reply(messages=[{"role": "user", "content": prompt}])
                
                # Handle both potential response formats
                context = mnemosyne_response if isinstance(mnemosyne_response, str) else getattr(mnemosyne_response, 'content', str(mnemosyne_response))
            except Exception as e:
                print(f"Error getting context from Mnemosyne: {e}")
        
        # Show explanation to user
        print(f"[Context]: {context}")
        
        # Have Iris present the question to the user
        try:
            iris_prompt = f"Please ask the customer this important underwriting question: {question_text}. The customer must answer YES or NO."
            iris_response = iris.generate_reply(messages=[{"role": "user", "content": iris_prompt}])
            
            # Handle response format safely
            iris_message = iris_response if isinstance(iris_response, str) else getattr(iris_response, 'content', str(iris_response))
            
            print(f"[Iris]: {iris_message}")
        except Exception as e:
            print(f"Error having Iris ask the question: {e}")
            print(f"QUESTION: {question_text}")
        
        # Get user response directly
        while True:
            user_response = user_proxy.get_human_input("Your answer (Yes/No): ").strip().lower()
            if user_response in ["yes", "no"]:
                break
            print("Please answer with 'Yes' or 'No' only.")
        
        # Record the response
        underwriting_responses[question_id] = user_response
        
        # Check eligibility - "Yes" for action=Decline questions means ineligible
        if question.get("action") == "Decline" and user_response.lower() == "yes":
            eligibility = False
            eligibility_reason = f"Failed pre-qualification: {explanation}"
            print(f"\n⚠️ Your answer affects your eligibility. We may not be able to offer coverage.")
            break
    
    # Save responses to current state regardless of eligibility outcome
    current_state = save_underwriting_responses(
        current_state, 
        underwriting_responses, 
        eligibility, 
        eligibility_reason
    )
    
    # Show eligibility status
    print("\n===================================")
    if eligibility:
        print("✅ PRE-QUALIFICATION SUCCESSFUL")
        print("===================================")
        print("You meet our initial underwriting criteria.")
        print("Your policy application will proceed to the next stage.")
    else:
        print("❌ PRE-QUALIFICATION DECLINED")
        print("===================================")
        print(f"Reason: {eligibility_reason}")
        print("We cannot provide coverage at this time based on your responses.")
        print("Your quote information has been saved for reference.")
    print("===================================\n")
    
    return eligibility

def handle_extracted_corrections(extracted_info, user_proxy):
    """
    Allow users to correct information extracted from documents.
    Following Azure best practices for robust data correction.
    
    Args:
        extracted_info: Dictionary of extracted information
        user_proxy: User proxy for input collection
        
    Returns:
        dict: Corrected profile information
    """
    corrected_info = extracted_info.copy()
    
    print("\n=== CORRECT EXTRACTED INFORMATION ===")
    print("For each field, press Enter to keep the current value or type a new value.")
    
    # Correct name
    current_name = corrected_info.get("name", "")
    new_name = user_proxy.get_human_input(f"Name [{current_name}]: ").strip()
    if new_name:
        corrected_info["name"] = new_name
    
    # Correct DOB
    current_dob = corrected_info.get("dob", "")
    new_dob = user_proxy.get_human_input(f"Date of Birth [{current_dob}]: ").strip()
    if new_dob:
        corrected_info["dob"] = new_dob
    
    # Ensure address is a dictionary
    if not isinstance(corrected_info.get("address"), dict):
        corrected_info["address"] = {}
    
    # Correct address
    current_street = corrected_info.get("address", {}).get("street", "")
    new_street = user_proxy.get_human_input(f"Street Address [{current_street}]: ").strip()
    if new_street:
        corrected_info["address"]["street"] = new_street
    
    current_city = corrected_info.get("address", {}).get("city", "")
    new_city = user_proxy.get_human_input(f"City [{current_city}]: ").strip()
    if new_city:
        corrected_info["address"]["city"] = new_city
    
    current_state = corrected_info.get("address", {}).get("state", "")
    new_state = user_proxy.get_human_input(f"State [{current_state}]: ").strip()
    if new_state:
        corrected_info["address"]["state"] = new_state
    
    current_zip = corrected_info.get("address", {}).get("zip", "")
    new_zip = user_proxy.get_human_input(f"ZIP Code [{current_zip}]: ").strip()
    if new_zip:
        corrected_info["address"]["zip"] = new_zip
    
    # Ensure contact is a dictionary
    if not isinstance(corrected_info.get("contact"), dict):
        corrected_info["contact"] = {}
    
    # Correct contact information
    current_phone = corrected_info.get("contact", {}).get("phone", "")
    new_phone = user_proxy.get_human_input(f"Phone Number [{current_phone}]: ").strip()
    if new_phone:
        corrected_info["contact"]["phone"] = new_phone
    
    current_email = corrected_info.get("contact", {}).get("email", "")
    new_email = user_proxy.get_human_input(f"Email Address [{current_email}]: ").strip()
    if new_email:
        corrected_info["contact"]["email"] = new_email
    
    return corrected_info
def show_current_status_and_confirm(current_state, step_description, allow_file_input=True):
    """
    Display workflow status and get user confirmation, with optional file path extraction.
    
    Args:
        current_state: Current workflow state
        step_description: Description of the next step
        allow_file_input: Whether to allow file paths in response
        
    Returns:
        tuple: (confirmation_bool, file_path) or just confirmation_bool if allow_file_input=False
    """
    print(f"\n=== CURRENT POLICY STATUS ===")
    # Display workflow status...
    
    print(f"\nNEXT STEP: {step_description}")
    
    if allow_file_input:
        prompt = "Do you want to proceed with this step? (yes/no or 'yes' followed by file path): "
    else:
        prompt = "Do you want to proceed with this step? (yes/no): "
    
    response = input(prompt).strip()
    
    # Parse the response for confirmation and optional file path
    if response.lower().startswith("yes"):
        confirmation = True
        # Check for file path (anything after "yes" plus whitespace)
        file_path = None
        path_match = re.search(r'yes\s+(.+)', response, re.IGNORECASE)
        if path_match and allow_file_input:
            potential_path = path_match.group(1).strip()
            if os.path.exists(potential_path):
                file_path = potential_path
                print(f"Found valid file path: {file_path}")
            else:
                print(f"Warning: The path '{potential_path}' was not found.")
        
        if allow_file_input:
            return confirmation, file_path
        else:
            return confirmation
    else:
        if allow_file_input:
            return False, None
        else:
            return False

# --- Helper function for manual merging (add this outside process_insurance_request) ---
def _manual_merge_details(basic_profile, collected_details):
    """Fallback function to manually merge collected details into the basic profile."""
    logger.info("Performing manual merge of collected details.")
    merged_profile = copy.deepcopy(basic_profile)
    merged_profile.setdefault("vehicle_details", {})
    merged_profile.setdefault("driving_history", {})
    merged_profile.setdefault("coverage_preferences", [])

    for key, data in collected_details.items():
        question = data.get("question", "").lower()
        answer = data.get("answer", "")

        # Simple keyword matching - This is fragile and assumes question order/content
        if "make" in question: merged_profile["vehicle_details"]["make"] = answer
        elif "model" in question: merged_profile["vehicle_details"]["model"] = answer
        elif "year" in question: merged_profile["vehicle_details"]["year"] = answer
        elif "vin" in question: merged_profile["vehicle_details"]["vin"] = answer
        elif "violation" in question: merged_profile["driving_history"]["violations"] = answer
        elif "accident" in question: merged_profile["driving_history"]["accidents"] = answer
        elif "licensed" in question: merged_profile["driving_history"]["years_licensed"] = answer
        elif "coverage" in question or "preference" in question:
            # Split comma-separated preferences
            merged_profile["coverage_preferences"] = [p.strip() for p in answer.split(',') if p.strip()]

    return merged_profile





# Update process_insurance_request to use the initialized client
# Fix around line 972 where the error is occurring

# In the process_insurance_request function:
# Add this to the process_insurance_request function
def process_basic_profile(current_state, agents, customer_file=None):
    """
    Step 1: Process basic customer profile with Iris agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        customer_file: Optional path to customer data file
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    iris = agents["iris"]
    user_proxy = agents["user_proxy"]
    
    # Initialize document processor for file handling
    doc_processor = DocumentProcessor()
    
    confirmation, provided_file = show_current_status_and_confirm(
        current_state, 
        "Collect basic customer information with Iris", 
        allow_file_input=True
    )

    if not confirmation:
        print("Workflow halted at basic profile stage.")
        return None

    # Process document if provided in the confirmation step
    extracted_info = {}
    basic_profile = None
    file_processed = False
    file_path = None

    # If user provided a file path in their confirmation response, use it
    if provided_file:
        file_path = provided_file
        print(f"Processing document from path provided in confirmation: {os.path.basename(file_path)}")
        
        # Process the file based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            print(f"Extracting customer information from {file_extension} document...")
            extracted_info = doc_processor.process_document(file_path)
            if extracted_info is None:
                extracted_info = {}
            else:
                file_processed = True
        else:
            # Handle text/JSON files as before
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
                    # Process file_content as before
            except Exception as e:
                print(f"Error reading file: {str(e)}")

    # ------ 1-B. INITIALIZE CONVERSATION WITH IRIS ------ #
    print("\n=== CUSTOMER INFORMATION COLLECTION WITH IRIS ===")

    # If we successfully processed a file, show extracted info
    if file_processed and extracted_info:
        print("\n=== Detailed Profile Information Extracted ===")
        print(json.dumps(extracted_info, indent=2))
        print("==============================================\n")
        
        # IMPORTANT: Store in current_state
        current_state["customerProfile"] = extracted_info
    
    # Display extracted information in a user-friendly format
    if extracted_info and isinstance(extracted_info, dict):
        if "name" in extracted_info:
            print(f"• Name: {extracted_info['name']}")
        if "dob" in extracted_info:
            print(f"• Date of Birth: {extracted_info['dob']}")

        address = extracted_info.get('address', {})
        if isinstance(address, dict) and any(address.values()):
            addr_parts = []
            if address.get('street'): addr_parts.append(address['street'])
            if address.get('city'): addr_parts.append(address['city'])
            if address.get('state'): addr_parts.append(address['state'])
            if address.get('zip'): addr_parts.append(address['zip'])
            print(f"• Address: {', '.join(addr_parts)}")

        contact = extracted_info.get('contact', {})
        if isinstance(contact, dict):
            if contact.get('phone'): print(f"• Phone: {contact['phone']}")
            if contact.get('email'): print(f"• Email: {contact['email']}")
    else:
        print("\n⚠️ No information was extracted from the document. Proceeding with manual input.")
            
    # Ask if the information is correct or if the customer wants to provide more
    print("\nWould you like to:")
    print("1. Use this extracted information")
    print("2. Provide your information through conversation")
    print("3. Make corrections to the extracted information")
    
    choice = user_proxy.get_human_input("Please select an option (1-3): ").strip()
    if choice == "1":
        # Use extracted information as is
        basic_profile = extracted_info
        print("\n✅ Using extracted information to proceed with your policy.")
    elif choice == "3":
        # Start with extracted info but allow corrections
        print("Let's correct the extracted information.")
        basic_profile = handle_extracted_corrections(extracted_info, user_proxy)
    else:
        # Reset to trigger conversation flow
        extracted_info = None
        file_processed = False
    
        # Now the iris_greeting is only initialized for the conversation path
        iris_greeting = iris.generate_reply(
            messages=[{
                "role": "user", 
                "content": "Introduce yourself as Iris and ask the customer for their basic information..."
            }],
            temperature=0.7
        )
        iris_message = iris_greeting.content if hasattr(iris_greeting, 'content') else str(iris_greeting)
        print(f"Iris: {iris_message}\n")

    # ------ 1-C. FREE-FORM TEXT OR FILE PATH DETECTION ------ #
    if not basic_profile:
        print("\nPlease provide your information. You can enter everything at once or provide a file path like C:\\path\\to\\file.pdf")
        conversation_buffer = ""
        customer_input = user_proxy.get_human_input("You: ").strip()
        conversation_buffer = customer_input
    
        # NEW FILE PATH DETECTION: Check if the user is referencing a file
        file_path_match = re.search(r'([a-zA-Z]:\\[^"<>|?*\n\r]+\.\w{2,5})', customer_input)
        referenced_file = False
        
        if file_path_match:
            potential_file_path = file_path_match.group(1)
            if os.path.exists(potential_file_path):
                print(f"\nI see you've referenced a file: {potential_file_path}")
                print("Processing this file...")
                referenced_file = True
                
                # Choose processing method based on file extension
                file_extension = os.path.splitext(potential_file_path)[1].lower()
                
                if file_extension in ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    # Use Document Intelligence for documents
                    print(f"Using Document Intelligence to extract information from {file_extension} file...")
                    extracted_info = doc_processor.process_document(potential_file_path)
                    if extracted_info and "error" not in extracted_info:
                        basic_profile = extracted_info
                        print("✅ Successfully extracted information from document")
                        
                        # Display extracted information for verification
                        print("\n=== EXTRACTED INFORMATION ===")
                        if "name" in basic_profile:
                            print(f"Name: {basic_profile['name']}")
                        if "dob" in basic_profile:
                            print(f"Date of Birth: {basic_profile['dob']}")
                        
                        address = basic_profile.get('address', {})
                        if isinstance(address, dict) and any(address.values()):
                            addr_parts = []
                            if address.get('street'): addr_parts.append(address['street'])
                            if address.get('city'): addr_parts.append(address['city'])
                            if address.get('state'): addr_parts.append(address['state'])
                            if address.get('zip'): addr_parts.append(address['zip'])
                            print(f"Address: {', '.join(addr_parts)}")
                        
                        contact = basic_profile.get('contact', {})
                        if isinstance(contact, dict):
                            if contact.get('phone'): print(f"Phone: {contact.get('phone')}")
                            if contact.get('email'): print(f"Email: {contact.get('email')}")
                        
                        # Let user make corrections
                        basic_profile = handle_extracted_corrections(basic_profile, user_proxy)
                    else:
                        print(f"\n⚠️ Document processing issue: {extracted_info.get('error', 'Unknown error')}")
                        print("Falling back to conversation.")
                        referenced_file = False
                else:
                    # For text or JSON files
                    try:
                        with open(potential_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_content = f.read()
                            
                        # Check if file is too large for LLM
                        if len(file_content) > 12000:
                            print("\n⚠️ File is too large for full processing. Using first 12000 characters.")
                            file_content = file_content[:12000]
                        
                        # Try to parse as JSON
                        try:
                            json_data = json.loads(file_content)
                            extracted_info = parse_customer_data(json_data)
                            if extracted_info and "error" not in extracted_info:
                                basic_profile = extracted_info
                                print("✅ Successfully extracted information from JSON file")
                                
                                # Let user make corrections
                                basic_profile = handle_extracted_corrections(basic_profile, user_proxy)
                            else:
                                print("\n⚠️ Could not extract customer data from JSON structure.")
                                referenced_file = False
                        except json.JSONDecodeError:
                            # Not JSON, have IRIS extract the information
                            print("Processing text file with IRIS...")
                            extraction_prompt = f"""
                            Extract customer information from this text file content:
                            
                            {file_content}
                            
                            Format your response as EXACTLY this JSON structure with appropriate values:
                            {{
                                "name": "Full Name",
                                "dob": "YYYY-MM-DD",
                                "address": {{
                                    "street": "Street address",
                                    "city": "City",
                                    "state": "State",
                                    "zip": "Zip code"
                                }},
                                "contact": {{
                                    "phone": "Phone number",
                                    "email": "Email address"
                                }}
                            }}
                            
                            Return ONLY the JSON, nothing else.
                            """
                            
                            extraction_response = iris.generate_reply(
                                messages=[{"role": "user", "content": extraction_prompt}],
                                temperature=0.2
                            )
                            
                            extraction_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
                            extracted_info = extract_json_with_fallback(extraction_text)
                            
                            if extracted_info:
                                basic_profile = extracted_info
                                print("✅ Successfully extracted information from text file")
                                
                                # Let user make corrections
                                basic_profile = handle_extracted_corrections(basic_profile, user_proxy)
                            else:
                                print("\n⚠️ Could not extract structured data from text file.")
                                referenced_file = False
                    except Exception as e:
                        print(f"⚠️ Error reading file: {str(e)}")
                        referenced_file = False

        # ------ 1-D. CONVERSATIONAL INFORMATION COLLECTION ------ #
        if not basic_profile and not referenced_file:
            # Proceed with conversational collection
            follow_up_count = 0
            follow_up_limit = 3  # Maximum number of follow-ups
            
            while follow_up_count < follow_up_limit:
                # Have IRIS analyze the input so far
                analysis_prompt = f"""
                Analyze the customer information provided so far:
                "{conversation_buffer}"
                
                Check if we have all required fields:
                - Full name
                - Date of birth
                - Complete address (street, city, state, zip)
                - Phone number
                - Email
                
                If any required field is missing, respond with a JSON:
                {{
                  "missing_fields": ["field1", "field2", ...],
                  "follow_up_question": "A natural, conversational question asking for the specific missing information"
                }}
                
                If all required fields are present, respond with:
                {{
                  "missing_fields": [],
                  "follow_up_question": ""
                }}
                """
                
                analysis_response = iris.generate_reply(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3
                )
                
                analysis_text = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
                analysis = extract_json_with_fallback(analysis_text) or {"missing_fields": ["all"], "follow_up_question": "Could you please provide your complete information?"}
                
                # If all fields are present, break the loop
                if not analysis.get("missing_fields"):
                    break
                    
                # Ask follow-up question if fields are missing
                follow_up_question = analysis.get("follow_up_question", "Could you provide more information?")
                print(f"\nIris: {follow_up_question}")
                
                # Check for file path in response again
                additional_input = user_proxy.get_human_input("You: ").strip()
                file_path_match = re.search(r'([a-zA-Z]:\\[^"<>|?*\n\r]+\.\w{2,5})', additional_input)
                
                if file_path_match:
                    # Handle file path in response (similar to previous file handling)
                    # ... (file processing code here, which is similar to earlier file processing) ...
                    # For brevity, I'm not repeating the full file processing code here
                    pass
                
                # Add to conversation buffer and continue
                conversation_buffer += f" {additional_input}"
                follow_up_count += 1
            
            # ------ 1-E. EXTRACT STRUCTURED DATA FROM CONVERSATION ------ #
            if not basic_profile:
                # Extract structured data from the entire conversation
                extraction_prompt = f"""
                Extract customer information from the following conversation:
                
                "{conversation_buffer}"
                
                Format your response as EXACTLY this JSON structure with appropriate values:
                {{
                    "name": "Full Name",
                    "dob": "YYYY-MM-DD",
                    "address": {{
                        "street": "Street address",
                        "city": "City",
                        "state": "State",
                        "zip": "Zip code"
                    }},
                    "contact": {{
                        "phone": "Phone number",
                        "email": "Email address"
                    }}
                }}
                
                Return ONLY the JSON, nothing else.
                """
                
                # Have IRIS process the conversation to extract structured data
                extraction_response = iris.generate_reply(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.2
                )
                
                extraction_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
                extracted_info = extract_json_with_fallback(extraction_text)
                
                if extracted_info:
                    basic_profile = extracted_info
                    print("\n✅ Successfully extracted information from conversation")
                    
                    # Display extracted information for verification
                    print("\n=== EXTRACTED INFORMATION ===")
                    if "name" in basic_profile:
                        print(f"Name: {basic_profile['name']}")
                    if "dob" in basic_profile:
                        print(f"Date of Birth: {basic_profile['dob']}")
                    
                    address = basic_profile.get('address', {})
                    if isinstance(address, dict) and any(address.values()):
                        addr_parts = []
                        if address.get('street'): addr_parts.append(address['street'])
                        if address.get('city'): addr_parts.append(address['city'])
                        if address.get('state'): addr_parts.append(address['state'])
                        if address.get('zip'): addr_parts.append(address['zip'])
                        print(f"Address: {', '.join(addr_parts)}")
                    
                    contact = basic_profile.get('contact', {})
                    if isinstance(contact, dict):
                        if contact.get('phone'): print(f"Phone: {contact.get('phone')}")
                        if contact.get('email'): print(f"Email: {contact.get('email')}")
                    
                    # Let user make corrections to extracted information
                    basic_profile = handle_extracted_corrections(basic_profile, user_proxy)
                else:
                    # Fallback to regex extraction if JSON parsing failed
                    print("\n⚠️ Failed to parse structured data from conversation. Using regex fallback.")
                    extracted_info = extract_customer_data_regex(conversation_buffer)
                    if extracted_info and "name" in extracted_info:
                        basic_profile = extracted_info
                        basic_profile = handle_extracted_corrections(basic_profile, user_proxy)
                    else:
                        print("\n⚠️ Could not extract information. Please enter your information manually.")
                        basic_profile = create_profile_manually()

    # After basic_profile is created in any path (file extraction or conversation)
    if basic_profile:
        # Ensure we save the profile to current_state
        current_state["customerProfile"] = basic_profile
        # Save checkpoint
        save_policy_checkpoint(current_state, "basic_profile_completed")
        print("✅ Basic profile information saved successfully.")
    else:
        print("⚠️ No basic profile information was collected. Cannot proceed.")
        return None
        
    return current_state

def process_detailed_profile(current_state, agents):
    """
    Step 2: Process detailed vehicle and driving information with Mnemosyne agent.
    Enhanced with document processing capabilities for multiple file formats.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    mnemosyne = agents["mnemosyne"]
    user_proxy = agents["user_proxy"]
    zeus = agents["zeus"]
    iris = agents["iris"]  # For document explanation
    
    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # Before proceeding, check if customer profile exists and is valid
    if "customerProfile" not in current_state or not current_state["customerProfile"]:
        print("⚠️ Customer profile information is missing. Please restart the process.")
        return None

    # Validate structure
    if not isinstance(current_state["customerProfile"], dict):
        print("⚠️ Customer profile has invalid format. Please restart the process.")
        return None
    
    # Get user confirmation to proceed with option to provide vehicle document
    print("\n=== DETAILED VEHICLE & DRIVING INFORMATION ===")
    print("We need more information about your vehicle and driving history.")
    print("You can:")
    print("1. Enter information through a conversation")
    print("2. Upload a vehicle document (registration, insurance card, etc.)")
    
    choice = user_proxy.get_human_input("Select an option (1 or 2): ").strip()
    
    # Get the basic profile collected in Step 1
    basic_profile = current_state.get("customerProfile", {})
    
    detailed_info_collected = {}  # To store collected details
    detailed_profile = None       # Final detailed profile
    document_processed = False    # Flag to track if document was processed
    
    # Process document if user chose that option
    if choice == "2":
        print("\n=== DOCUMENT UPLOAD FOR VEHICLE INFORMATION ===")
        file_path = user_proxy.get_human_input("Enter the full path to your vehicle document: ").strip()
        
        if os.path.exists(file_path):
            print(f"Processing document: {os.path.basename(file_path)}")
            
            # File extension and processing approach selection
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # ENHANCED: Support for different file formats including JSON and TXT
            if file_extension in ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                # Process document using Document Intelligence for image-based formats
                print("Using Document Intelligence to extract vehicle information...")
                extracted_info = doc_processor.process_document(file_path)
                
                if extracted_info and "error" not in extracted_info:
                    document_processed = True
                    
                    # Display the extracted vehicle information
                    print("\n=== VEHICLE INFORMATION EXTRACTED ===")
                    
                    vehicle_details = extracted_info.get('vehicle_details', {})
                    if vehicle_details:
                        print("Vehicle Information:")
                        for key, value in vehicle_details.items():
                            if value:
                                print(f"  {key.capitalize()}: {value}")
                    
                    # Have Iris explain the extracted information
                    # ... [existing Iris explanation code] ...
                    
                    # Let the user correct or add missing information
                    corrected_info = handle_extracted_corrections(extracted_info, user_proxy)
                    detailed_profile = {**basic_profile, **corrected_info}
                    
                    # Ensure vehicle_details is preserved properly
                    if "vehicle_details" in corrected_info:
                        detailed_profile["vehicle_details"] = corrected_info["vehicle_details"]
                        
                    # Ensure driving_history is preserved properly
                    if "driving_history" in corrected_info:
                        detailed_profile["driving_history"] = corrected_info["driving_history"]
                else:
                    # Document processing failed
                    error_message = extracted_info.get("error", "Unknown error") if extracted_info else "Failed to extract information"
                    print(f"\n⚠️ Document processing issue: {error_message}")
                    print("Falling back to manual information collection.")
                    
            elif file_extension in ['.json', '.txt']:
                # NEW: Handle JSON and TXT file formats
                print(f"Processing {file_extension.upper()} file...")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                    
                    # Check if file is too large for LLM processing
                    if len(file_content) > 12000:
                        print("\n⚠️ File is too large for full processing. Using first 12000 characters.")
                        file_content = file_content[:12000]
                    
                    # For JSON files, try to parse directly
                    if file_extension == '.json':
                        try:
                            json_data = json.loads(file_content)
                            extracted_info = parse_customer_data(json_data)
                            
                            if extracted_info and "error" not in extracted_info:
                                document_processed = True
                                print("✅ Successfully extracted information from JSON file.")
                                
                                # Display extracted information
                                print("\n=== VEHICLE INFORMATION EXTRACTED ===")
                                vehicle_details = extracted_info.get('vehicle_details', {})
                                if vehicle_details:
                                    print("Vehicle Information:")
                                    for key, value in vehicle_details.items():
                                        if value:
                                            print(f"  {key.capitalize()}: {value}")
                                
                                # Let user make corrections
                                corrected_info = handle_extracted_corrections(extracted_info, user_proxy)
                                detailed_profile = {**basic_profile, **corrected_info}
                                
                                # Ensure vehicle_details is preserved properly
                                if "vehicle_details" in corrected_info:
                                    detailed_profile["vehicle_details"] = corrected_info["vehicle_details"]
                                    
                                # Ensure driving_history is preserved properly
                                if "driving_history" in corrected_info:
                                    detailed_profile["driving_history"] = corrected_info["driving_history"]
                            else:
                                print("\n⚠️ Could not extract vehicle information from JSON structure.")
                                print("Falling back to manual information collection.")
                        except json.JSONDecodeError:
                            print("\n⚠️ Invalid JSON format. Processing as plain text.")
                            # Fall through to TXT processing
                    
                    # For TXT files or invalid JSON, use LLM to extract information
                    if not document_processed:
                        # Use Mnemosyne or Iris to extract information from text
                        extraction_prompt = f"""
                        Extract vehicle and driving history information from this text file content:
                        
                        {file_content}
                        
                        Format your response as EXACTLY this JSON structure with appropriate values:
                        {{
                            "vehicle_details": {{
                                "make": "Vehicle make",
                                "model": "Vehicle model",
                                "year": "Year",
                                "vin": "VIN number (if available)"
                            }},
                            "driving_history": {{
                                "violations": "Number of violations in past 3 years",
                                "accidents": "Number of accidents in past 3 years",
                                "years_licensed": "Years licensed to drive"
                            }}
                        }}
                        
                        Return ONLY the JSON, nothing else.
                        """
                        
                        # Use a try/except pattern to handle extraction failures
                        try:
                            extraction_response = mnemosyne.generate_reply(
                                messages=[{"role": "user", "content": extraction_prompt}],
                                temperature=0.2
                            )
                            
                            extraction_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
                            extracted_info = extract_json_with_fallback(extraction_text)
                            
                            if extracted_info and ("vehicle_details" in extracted_info or "driving_history" in extracted_info):
                                document_processed = True
                                print("✅ Successfully extracted information from text file.")
                                
                                # Display extracted information
                                print("\n=== INFORMATION EXTRACTED FROM TEXT ===")
                                vehicle_details = extracted_info.get('vehicle_details', {})
                                if vehicle_details:
                                    print("Vehicle Information:")
                                    for key, value in vehicle_details.items():
                                        if value and value.lower() != "vehicle make" and value.lower() != "vehicle model":
                                            print(f"  {key.capitalize()}: {value}")
                                
                                driving_history = extracted_info.get('driving_history', {})
                                if driving_history:
                                    print("\nDriving History:")
                                    for key, value in driving_history.items():
                                        if value and not value.startswith("Number of"):
                                            print(f"  {key.capitalize()}: {value}")
                                
                                # Let user make corrections
                                merged_info = {**basic_profile}
                                if "vehicle_details" in extracted_info:
                                    merged_info["vehicle_details"] = extracted_info["vehicle_details"]
                                if "driving_history" in extracted_info:
                                    merged_info["driving_history"] = extracted_info["driving_history"]
                                
                                corrected_info = handle_extracted_corrections(merged_info, user_proxy)
                                detailed_profile = corrected_info
                            else:
                                print("\n⚠️ Could not extract structured vehicle information from text.")
                                print("Falling back to manual information collection.")
                        except Exception as e:
                            logger.error(f"Error extracting from text file: {str(e)}")
                            print(f"\n⚠️ Error processing text file: {str(e)}")
                            print("Falling back to manual information collection.")
                except Exception as e:
                    logger.error(f"Error reading file: {str(e)}")
                    print(f"\n⚠️ Error reading file: {str(e)}")
                    print("Falling back to manual information collection.")
            else:
                print(f"\n⚠️ Unsupported file format: {file_extension}")
                print("Supported formats are: PDF, JPG, JPEG, PNG, TIFF, JSON, TXT")
                print("Falling back to manual information collection.")
        else:
            print(f"\n⚠️ File not found: {file_path}")
            print("Falling back to manual information collection.")

    # If we didn't successfully process a document, collect information through conversation
    if not document_processed:
        print("\n=== COLLECTING VEHICLE & DRIVING INFORMATION ===")
        print("Mnemosyne will now ask for details about your vehicle and driving history.")

        # Prepare prompt for Mnemosyne to ask for missing details
        missing_details_prompt = f"""
You are Mnemosyne, the memory and detail collection agent.
The basic customer profile collected so far is:
{json.dumps(basic_profile, indent=2)}

Your task is to identify the missing detailed information needed for a full auto insurance profile. Specifically check for:
- Vehicle details: 'make', 'model', 'year', 'vin' (VIN is optional but good to ask)
- Driving history: 'violations' (last 3 years), 'accidents' (last 3 years), 'years_licensed'
- Coverage preferences: General preferences like 'full coverage', 'liability only', or specific add-ons desired.

Generate a JSON object containing a list of conversational questions to ask the user *only* for the information that is missing or seems like placeholder data (e.g., "Vehicle make", "YYYY-MM-DD"). If a field exists with real data, do not ask for it again.

Example Response Format:
{{
  "questions": [
    "What is the make of your vehicle?",
    "And the model?",
    "What year was it manufactured?",
    "Could you provide the VIN (Vehicle Identification Number)? It helps ensure accurate coverage, but it's optional.",
    "How many moving violations (like speeding tickets) have you had in the past 3 years?",
    "How many at-fault accidents have you been involved in during the past 3 years?",
    "How many years have you been licensed to drive?",
    "Do you have any initial thoughts on the type of coverage you're looking for (e.g., full coverage, liability only)?"
  ]
}}

If all details seem present, return:
{{
  "questions": []
}}

Respond ONLY with the JSON object.
"""

        # Query Mnemosyne to get the questions
        logger.info("Querying Mnemosyne to generate questions for missing details.")
        questions_content, questions_parsed, success = query_agent(
            mnemosyne,
            missing_details_prompt,
            gpt4o_deployment,
            "Generate detailed profile questions"
        )

        if success and questions_parsed and "questions" in questions_parsed:
            questions_to_ask = questions_parsed["questions"]

            if not questions_to_ask:
                print("Mnemosyne indicates all necessary details are already present in the basic profile.")
                detailed_profile = basic_profile # Use the basic profile as the detailed one
            else:
                print(f"Mnemosyne needs {len(questions_to_ask)} more details.")

                # Ask the questions generated by Mnemosyne
                for i, question in enumerate(questions_to_ask):
                    print(f"\nMnemosyne: {question}")
                    answer = user_proxy.get_human_input("You: ").strip()
                    # Store answer temporarily; we'll structure it later
                    detailed_info_collected[f"detail_{i+1}"] = {"question": question, "answer": answer}

                # Now, ask Mnemosyne to structure the collected details and merge with basic profile
                structuring_prompt = f"""
You are Mnemosyne. Structure the collected detailed information and merge it intelligently with the existing basic profile.

Basic Profile (Use this as the base):
{json.dumps(basic_profile, indent=2)}

Collected Details (Question/Answer pairs):
{json.dumps(detailed_info_collected, indent=2)}

Your goal is to create a complete, detailed profile.
1. Start with the Basic Profile data.
2. Analyze the 'Collected Details'. Infer which field each question/answer pair corresponds to (e.g., vehicle_details.make, driving_history.violations).
3. Update the profile with the information from 'Collected Details', overwriting placeholders or adding missing fields.
4. Ensure the final output matches the target JSON structure below.

Target JSON Structure:
{{
    "name": "...",
    "dob": "...",
    "address": {{ ... }},
    "contact": {{ ... }},
    "vehicle_details": {{
        "make": "...", "model": "...", "year": "...", "vin": "..."
    }},
    "driving_history": {{
        "violations": "...", "accidents": "...", "years_licensed": "..."
    }},
    "coverage_preferences": ["...", "..."]
}}

Return ONLY the completed JSON object representing the final detailed profile. Do not include explanations.
"""
                logger.info("Querying Mnemosyne to structure and merge collected details.")
                final_profile_content, final_profile_parsed, final_success = query_agent(
                    mnemosyne,
                    structuring_prompt,
                    gpt4o_deployment,
                    "Structure detailed profile"
                )

                if final_success and final_profile_parsed and isinstance(final_profile_parsed, dict):
                    detailed_profile = final_profile_parsed
                    # Basic validation: Check if essential keys exist
                    if "vehicle_details" not in detailed_profile or "driving_history" not in detailed_profile:
                         logger.warning("Mnemosyne's structured profile missing key sections. Attempting manual merge.")
                         # Fallback to manual merge if structure is wrong
                         detailed_profile = _manual_merge_details(basic_profile, detailed_info_collected)
                    else:
                         print("\nMnemosyne has structured the detailed profile.")
                else:
                    print("[Mnemosyne] Failed to structure the collected details automatically.")
                    logger.warning("Failed to parse Mnemosyne's structured profile response. Using manual merge.")
                    detailed_profile = _manual_merge_details(basic_profile, detailed_info_collected)

        else:
            # Fallback if Mnemosyne failed to generate questions
            print("[Mnemosyne] Failed to generate questions for detailed profile. Proceeding with manual input.")
            logger.warning("Failed to get questions from Mnemosyne - using manual profile creation")
            # Use the existing manual function, passing the basic profile as a starting point
            detailed_profile = create_detailed_profile_manually(basic_profile)

    # Display and confirm the final detailed profile
    display_detailed_profile(detailed_profile) # Use existing display function
    confirmation = user_proxy.get_human_input("\nDo you confirm these details? (yes/no): ").strip().lower()
    if confirmation != "yes":
        print("[Mnemosyne] Updating vehicle and driving information based on corrections.")
        # Use the existing correction function
        detailed_profile = handle_detailed_profile_corrections(detailed_profile)

    # Final Update and Checkpoint
    # Ensure the profile is valid before saving
    if not isinstance(detailed_profile, dict) or "name" not in detailed_profile:
         logger.error("Detailed profile is invalid after collection/correction.")
         print("⚠️ Error: Failed to create a valid detailed profile.")
         return None # Halt workflow

    # Update state with the final detailed profile
    current_state["customerProfile"] = detailed_profile
    logger.info("Detailed customer profile collected and confirmed.")

    # Call Hera for recommendations based on the detailed profile
    current_state = process_with_hera(current_state, "mnemosyne") # Pass the updated state

    # Save checkpoint
    save_policy_checkpoint(current_state, "detailed_profile_completed")

    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="customerProfile", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_underwriting(current_state, agents):
    """
    Step 2.5: Process underwriting verification checks.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    iris = agents["iris"]
    mnemosyne = agents["mnemosyne"]
    hera = agents["hera"]
    zeus = agents["zeus"]
    user_proxy = agents["user_proxy"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Perform underwriting verification"):
        print("Workflow halted at underwriting verification stage.")
        return None

    # Run the underwriting check
    if not build_customer_profile(current_state, iris, mnemosyne, user_proxy):
        save_policy_checkpoint(current_state, "underwriting_failed")
        logger.warning("Underwriting check failed - workflow cannot proceed")
        return None

    # Process with Hera for recommendations
    current_state = process_with_hera(current_state, "underwriting")
    save_policy_checkpoint(current_state, "underwriting_completed")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="underwriting", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state
def process_risk_assessment(current_state, agents):
    """
    Step 3: Process risk assessment with Ares agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    ares = agents["ares"]
    hera = agents["hera"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Assess risk factors with Ares"):
        print("Workflow halted at risk assessment stage.")
        return None

    # Define fallback handler for risk assessment
    def risk_fallback_handler(state, agent, content):
        """Callback handler for risk assessment failures"""
        print("Using default risk assessment due to parsing failure")
        logger.warning("Risk assessment parsing failed - using default values")
        return {
            "riskScore": 5.0,
            "riskFactors": ["Default risk assessment - parsing failed"],
            "confidence": "low"
        }

    # Process risk assessment
    risk_prompt = f"Call: assess_risk; Params: {json.dumps({'profile': current_state['customerProfile']})}"
    current_state = process_with_agent(
        agent=ares,
        prompt=risk_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Risk Assessment",
        json_expected=True,
        state_key="risk_info",
        fallback_handler=risk_fallback_handler
    )

    # Process with Hera for recommendations
    current_state = process_with_hera(current_state, "ares")
    save_policy_checkpoint(current_state, "risk_assessment_completed")

    if "risk_info" in current_state:
        print(f"[Ares] Risk evaluation completed. Risk Score: {current_state['risk_info'].get('riskScore', 'N/A')}")
    else:
        print("Risk assessment could not be completed.")
        return None
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="risk_info", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_coverage_design(current_state, agents):
    """
    Step 4: Design coverage with Demeter agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    demeter = agents["demeter"]
    iris = agents["iris"]
    zeus = agents["zeus"]
    user_proxy = agents["user_proxy"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Design coverage model with Demeter"):
        print("Workflow halted at coverage design stage.")
        return None

    # Explore coverage options
    print("\n=== EXPLORING COVERAGE OPTIONS ===")
    print("Let's first explore all available coverage options to help you make informed selections.\n")
    selected_options = explore_coverage_options(demeter, user_proxy)

    # Store these preferences in the current_state
    if selected_options:
        print("\nThank you for exploring options! Your preferences will be considered when designing your coverage.")
        current_state["coverage_preferences"] = selected_options
    else:
        print("\nNo specific options selected. We'll proceed with the standard coverage design process.")

    # Design coverage
    coverage = design_coverage_with_demeter(current_state, demeter, iris, user_proxy)
    current_state["coverage"] = coverage
    save_policy_checkpoint(current_state, "coverage_design_completed")
    print("[Demeter] Coverage model designed and saved.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="coverage", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_policy_draft(current_state, agents):
    """
    Step 5: Draft policy document with Apollo agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    apollo = agents["apollo"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Draft policy document with Apollo"):
        print("Workflow halted at policy drafting stage.")
        return None

    # Process policy draft
    draft_prompt = f"Call: draft_policy; Params: {json.dumps({'coverage': current_state['coverage']})}"
    current_state = process_with_agent(
        agent=apollo,
        prompt=draft_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Policy Draft",
        json_expected=False,
        state_key="policyDraft"
    )
    save_policy_checkpoint(current_state, "policy_draft_completed")
    print("[Apollo] Policy draft prepared.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="policyDraft", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_document_polish(current_state, agents):
    """
    Step 6: Polish policy document with Calliope agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    calliope = agents["calliope"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Polish policy document with Calliope"):
        print("Workflow halted at document polishing stage.")
        return None

    # Process document polishing
    polish_prompt = f"Call: polish_document; Params: {json.dumps({'draft': current_state['policyDraft']})}"
    current_state = process_with_agent(
        agent=calliope,
        prompt=polish_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Document Polishing",
        json_expected=False,
        state_key="policyDraft"  # Overwrite the existing draft
    )
    save_policy_checkpoint(current_state, "document_polished_completed")
    print("[Calliope] Policy document finalized.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="policyDraft", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_document_polish(current_state, agents):
    """
    Step 6: Polish policy document with Calliope agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    calliope = agents["calliope"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Polish policy document with Calliope"):
        print("Workflow halted at document polishing stage.")
        return None

    # Process document polishing
    polish_prompt = f"Call: polish_document; Params: {json.dumps({'draft': current_state['policyDraft']})}"
    current_state = process_with_agent(
        agent=calliope,
        prompt=polish_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Document Polishing",
        json_expected=False,
        state_key="policyDraft"  # Overwrite the existing draft
    )
    save_policy_checkpoint(current_state, "document_polished_completed")
    print("[Calliope] Policy document finalized.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="policyDraft", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state
def process_pricing(current_state, agents):
    """
    Step 7: Calculate pricing with Plutus agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    plutus = agents["plutus"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Calculate pricing with Plutus"):
        print("Workflow halted at pricing calculation stage.")
        return None

    # Define fallback handler for pricing calculation
    def pricing_fallback_handler(state, agent, content):
        # Fallback handler for pricing calculation failures
        print("Using default pricing due to parsing failure")
        logger.warning("Pricing calculation failed - using default values")
        risk_score = state.get('risk_info', {}).get('riskScore', 5.0)
        base_premium = 750
        risk_multiplier = risk_score / 5.0
        return {
            "basePremium": base_premium,
            "riskMultiplier": risk_multiplier,
            "finalPremium": round(base_premium * risk_multiplier, 2),
            "confidence": "low"
        }

    # Process pricing calculation
    pricing_prompt = f"Call: calculate_pricing; Params: {json.dumps({'coverage': current_state['coverage'], 'profile': current_state['customerProfile'], 'risk': current_state.get('risk_info', {})})}"
    current_state = process_with_agent(
        agent=plutus,
        prompt=pricing_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Pricing Calculation",
        json_expected=True,
        state_key="pricing",
        fallback_handler=pricing_fallback_handler
    )
    save_policy_checkpoint(current_state, "pricing_completed")
    print(f"[Plutus] Pricing computed. Final premium: ${current_state['pricing'].get('finalPremium', 'N/A')}")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="pricing", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_quote(current_state, agents):
    """
    Step 8: Generate formal quote with Tyche agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    tyche = agents["tyche"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Generate formal quote with Tyche"):
        print("Workflow halted at quote generation stage.")
        return None

    # Process quote generation
    quote_prompt = f"Call: generate_quote; Params: {json.dumps({'pricing': current_state['pricing'], 'coverage': current_state['coverage'], 'customer': current_state['customerProfile']})}"
    current_state = process_with_agent(
        agent=tyche,
        prompt=quote_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Quote Generation",
        json_expected=False,
        state_key="quote"
    )
    save_policy_checkpoint(current_state, "quote_generated_completed")
    print("[Tyche] Quote generated.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="quote", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_presentation(current_state, agents):
    """
    Step 9: Present policy to customer with Orpheus agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    orpheus = agents["orpheus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Present policy to customer with Orpheus"):
        print("Workflow halted at customer presentation stage.")
        return None

    # Process policy presentation
    present_prompt = f"Call: present_policy; Params: {json.dumps({'document': current_state['policyDraft'], 'quote': current_state['quote'], 'customer': current_state['customerProfile']})}"
    current_state = process_with_agent(
        agent=orpheus,
        prompt=present_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Customer Presentation",
        json_expected=False,
        state_key="presentation"
    )
    save_policy_checkpoint(current_state, "presentation_completed")
    print("[Orpheus] Policy proposal presented to customer.")
    
    return current_state
def process_internal_review(current_state, agents):
    """
    Step 10: Perform internal approval and regulatory review.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    hestia = agents["hestia"]
    dike = agents["dike"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Perform internal approval and regulatory review"):
        print("Workflow halted at internal approval stage.")
        return None

    # Define fallback handler for approval
    def approval_fallback_handler(state, agent, content):
        # Fallback handler for approval failures
        logger.warning(f"Failed to parse {agent.name} response - using default approval")
        return {"approved": True, "confidence": "low", "notes": "Default approval due to parsing error"}

    # Process internal approval
    internal_prompt = f"Call: internal_approval; Params: {json.dumps({'document': current_state['policyDraft'], 'pricing': current_state['pricing'], 'risk': current_state.get('risk_info', {})})}"
    current_state = process_with_agent(
        agent=hestia,
        prompt=internal_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Internal Approval",
        json_expected=True,
        state_key="internal_approval",
        fallback_handler=approval_fallback_handler
    )

    # Define fallback handler for compliance check
    def compliance_fallback_handler(state, agent, content):
        # Fallback handler for compliance check failures
        logger.warning(f"Failed to parse {agent.name} response - using default compliance")
        return {"compliance": True, "confidence": "low", "notes": "Default compliance due to parsing error"}

    # Process regulatory compliance
    regulatory_prompt = f"Call: regulatory_review; Params: {json.dumps({'document': current_state['policyDraft'], 'state': current_state['customerProfile']['address'].get('state', 'CA')})}"
    current_state = process_with_agent(
        agent=dike,
        prompt=regulatory_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Regulatory Compliance",
        json_expected=True,
        state_key="compliance",
        fallback_handler=compliance_fallback_handler
    )
    save_policy_checkpoint(current_state, "internal_review_completed")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="internal_approval", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    # Check approval status
    internal_approved = current_state.get("internal_approval", {}).get("approved", False)
    compliance_approved = current_state.get("compliance", {}).get("compliance", False)

    if internal_approved and compliance_approved:
        print("[Hestia & Dike] ✅ Internal approval and regulatory compliance confirmed.")
    else:
        print("⚠️ WARNING: Policy did not pass internal approval or regulatory compliance.")
        if not internal_approved:
            print(f"Internal approval issues: {current_state.get('internal_approval', {}).get('reasons', 'No specific reason provided')}")
        if not compliance_approved:
            print(f"Regulatory compliance issues: {current_state.get('compliance', {}).get('issues', 'No specific issues provided')}")
        continue_anyway = input("Continue despite approval issues? (yes/no): ").strip().lower()
        if continue_anyway != "yes":
            print("Workflow halted due to approval issues.")
            return None

    return current_state

def process_customer_approval(current_state, agents):
    """
    Step 11: Get customer approval for the policy.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Get customer approval
    approval_input = input("\nDo you approve the presented policy and quote? (yes/no): ").strip().lower()
    if approval_input != "yes":
        print("Policy creation halted per customer decision.")
        save_policy_checkpoint(current_state, "customer_declined")
        return None
    
    # Save checkpoint for approval
    save_policy_checkpoint(current_state, "customer_approved")
    
    return current_state

def process_policy_issuance(current_state, agents):
    """
    Step 12: Issue final policy with Eirene agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    eirene = agents["eirene"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Issue final policy with Eirene"):
        print("Workflow halted at policy issuance stage.")
        return None

    # Define fallback handler for policy issuance
    def issuance_fallback_handler(state, agent, content):
        # Fallback handler for policy issuance failures
        logger.warning("Failed to parse issuance response - using default policy number")
        return {
            "policyNumber": f"POL{random_module.randint(100000, 999999)}",
            "startDate": datetime.datetime.now().strftime("%Y-%m-%d"),
            "endDate": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "status": "Active",
            "confidence": "low"
        }

    # Process policy issuance
    issuance_prompt = f"Call: issue_policy; Params: {json.dumps({'customer': current_state['customerProfile'], 'coverage': current_state['coverage'], 'pricing': current_state['pricing']})}"
    current_state = process_with_agent(
        agent=eirene,
        prompt=issuance_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Policy Issuance",
        json_expected=True,
        state_key="issuance",
        fallback_handler=issuance_fallback_handler
    )
    save_policy_checkpoint(current_state, "policy_issued_completed")
    print(f"[Eirene] Policy issued with policy number: {current_state['issuance'].get('policyNumber', 'Unknown')}")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="issuance", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_monitoring_setup(current_state, agents):
    """
    Step 13: Set up policy monitoring with Themis agent.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    themis = agents["themis"]
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Set up policy monitoring with Themis"):
        print("Workflow halted at policy monitoring stage.")
        return None

    # Define fallback handler for monitoring setup
    def monitoring_fallback_handler(state, agent, content):
        # Fallback handler for monitoring setup failures
        logger.warning("Failed to parse monitoring response - using default monitoring setup")
        return {
            "monitoringStatus": "Active",
            "notificationEmail": state['customerProfile']['contact'].get('email', 'customer@example.com'),
            "renewalDate": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "confidence": "low"
        }

    # Process monitoring setup
    monitor_prompt = f"Call: monitor_policy; Params: {json.dumps({'policyNumber': current_state['issuance'].get('policyNumber', 'Unknown'), 'customer': current_state['customerProfile']})}"
    current_state = process_with_agent(
        agent=themis,
        prompt=monitor_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Policy Monitoring",
        json_expected=True,
        state_key="monitoring",
        fallback_handler=monitoring_fallback_handler
    )
    save_policy_checkpoint(current_state, "monitoring_setup_completed")
    print("[Themis] Policy monitoring setup completed.")
    
    # Use Zeus to display the policy graph
    display_policy_graph(current_state, latest_update="monitoring", zeus=zeus, gpt4o_deployment=gpt4o_deployment)
    
    return current_state

def process_policy_activation(current_state, agents):
    """
    Step 14: Convert quote to active policy and save to database.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state or None if halted
    """
    # Extract necessary agents
    zeus = agents["zeus"]
    
    # Get user confirmation to proceed
    if not show_current_status_and_confirm(current_state, "Finalize and save active policy with Zeus"):
        print("Workflow halted at policy finalization stage.")
        return None

    print("\n=== FINALIZING AND ACTIVATING POLICY WITH ZEUS ===")
    print("Converting quote to active policy and storing in database...")

    try: # <<< The 'try' block starts here
        # Prepare final policy data
        final_policy = {
            "customerProfile": current_state.get("customerProfile", {}),
            "coverage": current_state.get("coverage", {}),
            "policyDraft": current_state.get("policyDraft", ""),
            "pricing": current_state.get("pricing", {}),
            "quote": current_state.get("quote", ""),
            "issuance": current_state.get("issuance", {}),
            "monitoring": current_state.get("monitoring", {}),
            "activatedDate": datetime.datetime.now().isoformat()
        }

        # 1. Have Zeus prepare the final policy document and conversion
        final_policy_prompt = f"""
        You are Zeus, responsible for finalizing insurance policies. Please convert the quote into an active policy:
        
        CURRENT POLICY STATE:
        {json.dumps(final_policy, indent=2)}
        
        Your tasks:
        1. Create a JSON representing the ACTIVE_POLICY with all required fields
        2. Ensure fields are properly structured for database storage
        3. Add any missing required metadata fields
        
        Return ONLY a valid JSON object representing the finalized active policy.
        """
        
        # Call Zeus to finalize the policy
        finalized_policy_response = zeus.generate_reply(
            messages=[{"role": "user", "content": final_policy_prompt}],
            temperature=0.2
        )
        
        finalized_policy_text = finalized_policy_response.content if hasattr(finalized_policy_response, "content") else str(finalized_policy_response)
        active_policy = extract_json_content(finalized_policy_text)
        
        if not active_policy:
            logger.warning("Failed to parse Zeus's finalized policy. Using default structure.")
            # Create default active policy structure
            policy_id = final_policy.get("issuance", {}).get("policyNumber", f"POL{random_module.randint(100000, 999999)}")
            active_policy = {
                "id": policy_id,
                "type": "ACTIVE_POLICY",
                "status": "Active",
                "originalQuoteId": current_state.get("quoteId"),
                "policyNumber": policy_id,
                "effectiveDate": final_policy.get("issuance", {}).get("startDate", datetime.datetime.now().isoformat()),
                "expirationDate": final_policy.get("issuance", {}).get("endDate"),
                "activationDate": datetime.datetime.now().isoformat(),
                "customerProfile": final_policy.get("customerProfile", {}),
                "coverage": final_policy.get("coverage", {}),
                "policyDraft": final_policy.get("policyDraft", ""),
                "pricing": final_policy.get("pricing", {}),
                "issuance": final_policy.get("issuance", {}),
                "monitoring": final_policy.get("monitoring", {}),
                "policyGraph": current_state.get("summary", {})
            }
        
        # 2. Ensure required fields exist
        if "id" not in active_policy:
            policy_id = final_policy.get("issuance", {}).get("policyNumber", f"POL{random_module.randint(100000, 999999)}")
            active_policy["id"] = policy_id
            active_policy["policyNumber"] = policy_id
        
        # 3. Save to PolicyIssued container - reuse existing container client if available
        container_name = "PolicyIssued"
        container_client = None
        
        # Reuse list_containers function that was already defined
        from db.cosmos_db import list_containers, get_container_client, init_cosmos_db
        from azure.cosmos import CosmosClient, PartitionKey # Import necessary Cosmos DB classes

        # Ensure DB is initialized
        init_cosmos_db()
        
        available_containers = list_containers()
        logger.info(f"Available containers: {available_containers}")

        # Try to get the container client
        possible_containers = ["PolicyIssued", "policyIssued", "policiesIssued", "policies", "issuedPolicies"]
        for name in possible_containers:
            if name in available_containers:
                logger.info(f"Found matching container: {name}")
                container_client = get_container_client(name)
                break

        # If container doesn't exist, create it (Azure Best Practice: Idempotent creation)
        if not container_client:
            logger.warning(f"Container '{container_name}' not found. Creating it.")
            # @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            if not endpoint or not key:
                raise ValueError("Cosmos DB credentials (COSMOS_ENDPOINT, COSMOS_KEY) not found in environment variables.")
            
            client = CosmosClient(endpoint, key)
            database = client.get_database_client("insurance") # Assuming database name is 'insurance'
            container_client = database.create_container_if_not_exists(
                id=container_name, 
                partition_key=PartitionKey(path="/id") # Use '/id' as partition key
            )
            logger.info(f"Container '{container_name}' created successfully.")

        # Save the active policy to Cosmos DB
        # @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.
        container_client.upsert_item(body=active_policy)
        logger.info(f"Successfully saved active policy {active_policy['id']} to container '{container_name}'")

        # Update current state with the final active policy details
        current_state["active_policy"] = active_policy
        save_policy_checkpoint(current_state, "policy_activated_completed")
        print(f"[Zeus] Policy {active_policy['id']} finalized, activated, and saved.")

        # Use Zeus to display the policy graph
        display_policy_graph(current_state, latest_update="active_policy", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

        return current_state

    except Exception as e: # <<< The corresponding 'except' block
        logger.error(f"Error during policy activation: {str(e)}", exc_info=True)
        print(f"\n⚠️ Error finalizing and activating policy: {str(e)}")
        save_policy_checkpoint(current_state, "policy_activation_failed")
        return None # Halt workflow on error

# The 'def process_final_summary' function should start *after* the 'except' block above.
def process_final_summary(current_state, agents):
    """
    Step 15: Generate final policy summary with Zeus.
    
    Args:
        current_state: Current workflow state
        agents: Dictionary of initialized agents
        
    Returns:
        dict: Updated workflow state with summary
    """
    # ... (rest of the function remains the same) ...
    # Extract necessary agents
    zeus = agents["zeus"]
    
    print("\n=== FINAL POLICY SUMMARY ===")
    print("Zeus is preparing your final policy summary...")

    try:
        # Format policy data for final summary
        policy_number = current_state.get("issuance", {}).get("policyNumber", "Unknown")
        customer_name = current_state.get("customerProfile", {}).get("name", "Unknown")
        
        # Create prompt for Zeus final summary
        final_summary_prompt = f"""
        Create a comprehensive final summary of this insurance policy in customer-friendly language.
        
        Policy Number: {policy_number}
        Customer: {customer_name}
        
        Include the following sections:
        1. Coverage Details - what is and isn't covered
        2. Financial Summary - premium, payment schedule, etc.
        3. Important Dates - effective date, expiration date, etc.
        4. Next Steps - what the customer should expect
        5. Contact Information - who to contact for questions or claims
        
        Use a friendly, conversational tone with appropriate formatting (sections, bullet points).
        Make it thorough but easy to understand for someone who isn't an insurance expert.
        """
        
        # Call Zeus with robust error handling
        max_attempts = 2
        summary = None
        
        for attempt in range(max_attempts):
            try:
                # Azure Best Practice: Set appropriate timeout for LLM operations
                response = zeus.generate_reply(
                    messages=[{"role": "user", "content": final_summary_prompt}],
                    temperature=0.3
                )
                summary = response.content if hasattr(response, 'content') else str(response)
                break  # Success, exit retry loop
            except Exception as e:
                logger.warning(f"Zeus summary attempt {attempt+1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # Brief pause before retry
                else:
                    raise  # Re-raise on last attempt
        
        if summary:
            # Store in current state
            current_state["summary"] = summary
            
            # Display to user
            print("\n" + "-" * 80)
            print(summary)
            print("-" * 80 + "\n")
        else:
            # Generate basic summary as fallback
            basic_summary = (
                f"POLICY SUMMARY\n\n"
                f"Policy Number: {policy_number}\n"
                f"Policyholder: {customer_name}\n"
                f"Effective Date: {current_state.get('issuance', {}).get('startDate', 'Unknown')}\n"
                f"Expiration Date: {current_state.get('issuance', {}).get('endDate', 'Unknown')}\n"
                f"Premium: ${current_state.get('pricing', {}).get('finalPremium', 0):,.2f}\n\n"
                f"Your insurance policy has been successfully issued. Please contact customer "
                f"service if you have any questions or need assistance with your policy."
            )
            current_state["summary"] = basic_summary
            print("\n" + "-" * 80)
            print(basic_summary)
            print("-" * 80 + "\n")
            
        # Save checkpoint with summary
        save_policy_checkpoint(current_state, "final_summary_completed")
        
        return current_state
        
    except Exception as e:
        logger.error(f"Error in final policy summary: {str(e)}", exc_info=True)
        print(f"\n⚠️ Error preparing final summary: {str(e)}")
        # Don't fail the entire workflow just because the summary failed
        return current_state
        
def process_insurance_request(customer_file=None):
    """
    Master workflow to process an insurance request from intake to issuance.
    
    Implements Azure best practices for LLM agent interactions, ensuring consistent model usage,
    robust error handling, and standardized prompting across all agents.
    
    Args:
        customer_file (str, optional): Path to customer data file. Defaults to None.
        
    Returns:
        dict: Final policy summary or None if workflow was halted.
    """
    # Generate correlation ID for request tracking - Azure best practice
    correlation_id = str(uuid.uuid4())
    logger.info(f"[REQUEST:{correlation_id}] Starting insurance processing")
    
    # Initialize Azure OpenAI client and verify deployment
    if not initialize_azure_openai():
        print("❌ Azure OpenAI initialization failed. Workflow halted.")
        logger.error("Azure OpenAI initialization failed - cannot proceed")
        return None  # Return early if initialization fails
        
    # Azure Best Practice: Verify model availability before starting
    available_deployments = [model.id for model in azure_openai_client.models.list().data]
    if gpt4o_deployment not in available_deployments:
        print(f"❌ Required GPT-4o deployment '{gpt4o_deployment}' not found.")
        print(f"Available models: {', '.join(available_deployments)}")
        logger.error(f"GPT-4o deployment '{gpt4o_deployment}' not found")
        return None  # Return early if model not available
    
    logger.info(f"Using GPT-4o deployment: {gpt4o_deployment}")
    
    # Initialize all agents with consistent model configuration
    agents = initialize_agents(model=gpt4o_deployment)
    
    # Initialize workflow state
    current_state = {
        "status": "Initiated",
        "timestamp": datetime.datetime.now().isoformat(),
        "quoteId": f"QUOTE{random_module.randint(10000, 99999)}",
        "hera_processed_stages": []  # Initialize to prevent recursion issues
    }

    # Process customer file if provided
    if customer_file:
        print(f"Reading customer data from {customer_file}...")
        file_data = read_customer_data_from_file(customer_file)
        if file_data:
            current_state["file_data"] = file_data
            logger.info(f"Successfully loaded customer data from {customer_file}")
        else:
            print("Could not read customer data from file. Starting with manual intake.")
            logger.warning(f"Failed to load customer data from {customer_file}")

    # ------- Execute each step in sequence --------
    
    # Step 1: Basic Customer Profile with Iris
    current_state = process_basic_profile(current_state, agents, customer_file)
    if current_state is None:
        return None
    
    # Step 2: Detailed Profile with Mnemosyne
    current_state = process_detailed_profile(current_state, agents)
    if current_state is None:
        return None
        
    # Step 2.5: Underwriting Verification
    current_state = process_underwriting(current_state, agents)
    if current_state is None:
        return None
    
    # Step 3: Risk Assessment with Ares
    current_state = process_risk_assessment(current_state, agents)
    if current_state is None:
        return None
    
    # Step 4: Coverage Design with Demeter
    current_state = process_coverage_design(current_state, agents)
    if current_state is None:
        return None
    
    # Step 5: Draft Policy with Apollo
    current_state = process_policy_draft(current_state, agents)
    if current_state is None:
        return None
    
    # Step 6: Polish Document with Calliope
    current_state = process_document_polish(current_state, agents)
    if current_state is None:
        return None
    
    # Step 7: Calculate Pricing with Plutus
    current_state = process_pricing(current_state, agents)
    if current_state is None:
        return None
    
    # Step 8: Generate Quote with Tyche
    current_state = process_quote(current_state, agents)
    if current_state is None:
        return None
    
    # Step 9: Present Policy with Orpheus
    current_state = process_presentation(current_state, agents)
    if current_state is None:
        return None
    
    # Step 10: Internal Approval & Regulatory Review
    current_state = process_internal_review(current_state, agents)
    if current_state is None:
        return None
    
    # Step 11: Customer Approval
    current_state = process_customer_approval(current_state, agents)
    if current_state is None:
        return None
    
    # Step 12: Issue Policy with Eirene
    current_state = process_policy_issuance(current_state, agents)
    if current_state is None:
        return None
    
    # Step 13: Monitoring Setup with Themis
    current_state = process_monitoring_setup(current_state, agents)
    if current_state is None:
        return None
    
    # Step 14: Convert Quote to Active Policy and Save
    current_state = process_policy_activation(current_state, agents)
    if current_state is None:
        return None
    
    # Step 15: Final Summary with Zeus
    current_state = process_final_summary(current_state, agents)
    if current_state is None:
        return None

    print("\n=== Insurance Policy Creation Workflow Completed Successfully ===")
    print(f"Policy Number: {current_state.get('issuance', {}).get('policyNumber', 'Unknown')}")
    print(f"Final Premium: ${current_state.get('pricing', {}).get('finalPremium', 'Unknown')}")
    print("Thank you for using our service!")

    return current_state.get("summary", current_state)

def insurance_app_main():
    """
    Main entry point for the insurance application that routes to appropriate flows
    based on user intent. Zeus agent determines which workflow to execute.
    
    Following Azure best practices for multi-intent applications.
    """
    logger.info("Starting insurance application main entry point")
    
    # Initialize Azure OpenAI client
    if not initialize_azure_openai():
        print("❌ Azure OpenAI initialization failed. Cannot proceed.")
        return
    
    # Initialize Zeus agent for routing
    agents = initialize_agents(model=gpt4o_deployment)
    zeus = agents["zeus"]
    user_proxy = agents["user_proxy"]
    
    print("\n" + "="*80)
    print("                WELCOME TO THE INSURANCE MANAGEMENT SYSTEM")
    print("="*80 + "\n")
    
    print("Zeus: Hello! I'm Zeus, your insurance assistant.")
    print("Zeus: I can help you with new quotes, policy changes, cancellations, or quote updates.")
    print("Zeus: Please briefly tell me what you'd like to do today.")
    
    user_input = user_proxy.get_human_input("\nWhat would you like to do? ").strip()
    
    # Have Zeus determine intent - using parenthesized string concatenation to avoid formatting issues
    intent_prompt = (
        "Based on the user's input, determine which workflow they need.\n\n"
        f"User input: \"{user_input}\"\n\n"
        "Classify the intent as one of:\n"
        "1. NEW_POLICY - User wants to get a new quote or policy\n"
        "2. CHANGE_POLICY - User wants to modify an existing policy (vehicle changes or coverage changes)\n"
        "3. CANCEL_POLICY - User wants to cancel their policy\n"
        "4. UPDATE_QUOTE - User wants to modify an existing quote\n"
        "5. OTHER - User intent doesn't match any of the above\n\n"
        "Respond with ONLY the intent code and a brief explanation, like:\n"
        "INTENT_CODE: Brief explanation of why you classified it this way"
    )
    
    intent_response = zeus.generate_reply(messages=[{"role": "user", "content": intent_prompt}])
    intent_text = intent_response.content if hasattr(intent_response, 'content') else str(intent_response)
    
    # Extract intent code
    if "NEW_POLICY" in intent_text:
        workflow = "new_policy"
    elif "CHANGE_POLICY" in intent_text:
        workflow = "change_policy"
    elif "CANCEL_POLICY" in intent_text:
        workflow = "cancel_policy"
    elif "UPDATE_QUOTE" in intent_text:
        workflow = "update_quote"
    else:
        workflow = "other"
    
    # Route to appropriate workflow
    if workflow == "new_policy":
        print("\nZeus: I understand you want to create a new insurance policy. Let me guide you through the process.")
        _explain_new_policy_process(zeus)
        return process_insurance_request()
        
    elif workflow == "change_policy":
        print("\nZeus: I understand you want to make changes to your existing policy. Let me help you with that.")
        return handle_policy_change(user_input)
        
    elif workflow == "cancel_policy":
        print("\nZeus: I understand you want to cancel your policy.")
        print("Zeus: This functionality is coming soon. Please contact customer service for now.")
        return None
        
    elif workflow == "update_quote":
        print("\nZeus: I understand you want to update an existing quote.")
        print("Zeus: This functionality is coming soon. Please contact customer service for now.")
        return None
        
    else:
        print("\nZeus: I'm not sure I understand what you need. Let me help you create a new policy.")
        print("Zeus: If you need something else, please contact customer service.")
        return None

def _explain_new_policy_process(zeus):
    """
    #Have Zeus explain the new policy creation process following Azure best practices
    #for user expectation management.
    """
    # Use a more detailed prompt following Azure best practices
    process_explanation = zeus.generate_reply(
        messages=[{
            "role": "user", 
            "content": "Explain the steps involved in creating a new insurance policy and the approximate time needed. "
                      "Include information about documentation required, coverage options, and what to expect at each step. "
                      "Format your response in a user-friendly way with clear headings and approximate timeframes."
        }],
        temperature=0.3
    )
    
    explanation = process_explanation.content if hasattr(process_explanation, 'content') else str(process_explanation)
    print(f"\nZeus: {explanation}\n")
    print("\nZeus: Let's get started with creating your new policy.")


def handle_policy_change(initial_input):
    """
    # Handle changes to existing policies following Azure best practices for LLM agent workflows.
    """
    logger.info("Starting policy change workflow")
    
    # Initialize required agents
    agents = initialize_agents(model=gpt4o_deployment)
    zeus = agents["zeus"]
    iris = agents["iris"]
    demeter = agents["demeter"]
    plutus = agents["plutus"]
    eirene = agents["eirene"]
    user_proxy = agents["user_proxy"]
    
    # Step 1: Get policy information
    print("\n=== POLICY IDENTIFICATION ===")
    print("To make changes to your policy, I'll need some information to identify it.")
    
    policy_number = user_proxy.get_human_input("Please enter your policy number: ").strip()
    
    # Step 2: Retrieve policy from Cosmos DB with improved error handling
    try:
        # Check if the database exists and initialize if needed
        from db.cosmos_db import init_cosmos_db
        init_cosmos_db()
        
        # Get container (first check if it exists)
        from db.cosmos_db import list_containers
        available_containers = list_containers()
        logger.info(f"Available containers: {available_containers}")
        
        # Try different possible container names
        possible_containers = ["PolicyIssued", "policyIssued", "policiesIssued", "policies", "issuedPolicies"]
        container_client = None
        
        for container_name in possible_containers:
            if container_name in available_containers:
                logger.info(f"Found matching container: {container_name}")
                container_client = get_container_client(container_name)
                break
                
        if not container_client:
            # Create a container if it doesn't exist
            logger.warning("No policy container found. Creating a new 'PolicyIssued' container.")
            from azure.cosmos import CosmosClient, PartitionKey
            # Ensure these environment variables are set or use values from your config
            endpoint = os.getenv("COSMOS_ENDPOINT")
            key = os.getenv("COSMOS_KEY")
            
            if not endpoint or not key:
                raise ValueError("Cosmos DB credentials not found in environment variables")
                
            client = CosmosClient(endpoint, key)
            database = client.get_database_client("insurance")
            container_client = database.create_container_if_not_exists(
                id="PolicyIssued", 
                partition_key=PartitionKey(path="/id")
            )
            
        # Now attempt the query with the container client
        query =(  
        "SELECT * FROM c WHERE " 
        "c.issuance.policyNumber = @policyNumber OR "
        "c.policyNumber = @policyNumber OR "
        " c.id = @policyNumber"
        )
        
        parameters = [{"name": "@policyNumber", "value": policy_number}]
        
        items = list(container_client.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        # If no results found and the input might be a name, try searching by policyholder name
        if not items:
            # Use parenthesized string concatenation for the query
            name_query = (
                "SELECT * FROM c WHERE "
                "CONTAINS(LOWER(c.customerProfile.name), LOWER(@name)) OR "
                "CONTAINS(LOWER(c.policyholder.name), LOWER(@name)) OR "
                "CONTAINS(LOWER(c.policyHolder.name), LOWER(@name))"
            ) # <-- CORRECTED: Parenthesis moved to end the string assignment

            # Correctly assign parameters after the query definition
            parameters = [{"name": "@name", "value": policy_number.lower()}]
# ...existing code..

            items = list(container_client.query_items(
                query=name_query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # If multiple policies found for the name, ask user to choose
            if len(items) > 1:
                print(f"\nFound {len(items)} policies for '{policy_number}'.")
                print("Please select which policy you want to modify:")
                
                for i, policy in enumerate(items):
                    policy_num = (policy.get('issuance', {}).get('policyNumber') or 
                                policy.get('policyNumber') or 
                                policy.get('id', 'Unknown'))
                    
                    customer = (policy.get('customerProfile', {}).get('name') or 
                                policy.get('policyholder', {}).get('name') or
                                policy.get('policyHolder', {}).get('name', 'Unknown'))
                    
                    vehicle_details = policy.get('customerProfile', {}).get('vehicle_details', {})
                    if isinstance(vehicle_details, dict):
                        vehicle = f"{vehicle_details.get('year', '')} {vehicle_details.get('make', '')} {vehicle_details.get('model', '')}"
                    elif isinstance(vehicle_details, list) and vehicle_details:
                        first_vehicle = vehicle_details[0]
                        vehicle = f"{first_vehicle.get('year', '')} {first_vehicle.get('make', '')} {first_vehicle.get('model', '')}"
                    else:
                        vehicle = "Unknown vehicle"
                    
                    print(f"{i+1}. Policy #{policy_num}: {customer} - {vehicle}")
                    
                while True:
                    try:
                        selection = int(user_proxy.get_human_input(f"Enter policy number (1-{len(items)}): ")) - 1
                        if 0 <= selection < len(items):
                            current_policy = items[selection]
                            break
                        print(f"Please enter a number between 1 and {len(items)}.")
                    except ValueError:
                        print("Please enter a valid number.")
            elif len(items) == 1:
                current_policy = items[0]
            else:
                print("\nⓧ Policy not found. Please check your policy number and try again.")
                logger.warning(f"Policy number or name '{policy_number}' not found in database")
                return None
        else:
            current_policy = items[0]
            
        logger.info(f"Successfully retrieved policy: {policy_number}")
        
    except Exception as e:
        print(f"\nⓧ Error retrieving policy: {str(e)}")
        logger.error(f"Error retrieving policy: {str(e)}", exc_info=True)
        return None
    
    # Step 3: Verify policy ownership
    customer_name = current_policy.get("customerProfile", {}).get("name", "")
    
    print(f"\nThank you. I found policy #{policy_number} for {customer_name}.")
    
    # Use Zeus to generate a policy summary
    # Use parenthesized string concatenation for the prompt
    policy_summary_prompt = (
        "Create a concise summary of this insurance policy in customer-friendly language:\n\n"
        f"Policy Number: {policy_number}\n"
        f"Customer: {customer_name}\n\n"
        "Include:\n"
        "1. Vehicle details\n"
        "2. Coverage summary\n"
        "3. Premium information\n"
        "4. Policy dates\n\n"
        "Format with emojis and clear sections."
    ) # Correctly formatted the string assignment

    summary_response = zeus.generate_reply(
        messages=[{"role": "user", "content": policy_summary_prompt}],
        temperature=0.3
    )
    
    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    print("\n=== YOUR CURRENT POLICY ===")
    print(summary)
    
    # Step 4: Determine change type via free-form English
    print("\n=== POLICY CHANGE OPTIONS ===")
    print("You can say things like:")
    print("  • 'I want to add a vehicle'")
    print("  • 'I'd like to update my coverage limits'")
    print("  • 'Cancel the change process'")
    
    user_choice = user_proxy.get_human_input("\nWhat would you like to do? ").strip()
    intent_prompt = (
    "Classify the user's request into one of:\n"
    "VEHICLE_CHANGE, COVERAGE_CHANGE, CANCEL_PROCESS\n\n"
    f"User said: \"{user_choice}\"\n"
    "Respond with ONLY the intent code."
)
    intent_resp = zeus.generate_reply(messages=[{"role":"user","content":intent_prompt}])
    intent = intent_resp.content if hasattr(intent_resp, "content") else str(intent_resp)

    if "VEHICLE_CHANGE" in intent:
        change_type = 1
    elif "COVERAGE_CHANGE" in intent:
        change_type = 2
    else:
        change_type = 3

    if change_type == 3:
        print("Policy change process cancelled.")
        return None

    updated_policy = copy.deepcopy(current_policy)
    if change_type == 1:
        updated_policy = handle_vehicle_changes(updated_policy, user_proxy, iris, zeus)
    else:
        updated_policy = handle_coverage_changes(updated_policy, user_proxy, demeter, zeus)
    
    if not updated_policy:
        print("Policy change process cancelled or encountered an error.")
        return None
    
    # Step 6: Calculate new premium
    print("\n=== RECALCULATING PREMIUM ===")
    print("Based on your changes, we need to recalculate your premium...")

    try:
        # Use parenthesized string concatenation and f-strings
        pricing_prompt = (
            "Call: calculate_pricing\n"
            "Params: " + json.dumps({
                'coverage': updated_policy['coverage'],
                'profile': updated_policy['customerProfile'],
                'risk': updated_policy.get('risk_info', {})
            })
        )

        pricing_response = plutus.generate_reply(
            messages=[{"role": "user", "content": pricing_prompt}],
            temperature=0.2
        )

        pricing_content = pricing_response.content if hasattr(pricing_response, 'content') else str(pricing_response)
        new_pricing = extract_json_content(pricing_content)

        if not new_pricing:
            raise ValueError("Failed to calculate new premium")

        old_premium = current_policy.get("pricing", {}).get("finalPremium", 0)
        new_premium = new_pricing.get("finalPremium", 0)

        print(f"\nPrevious premium: ${old_premium:.2f}")
        print(f"New premium: ${new_premium:.2f}")
        print(f"Difference: ${new_premium - old_premium:.2f}")

        updated_policy["pricing"] = new_pricing

    except Exception as e:
        print(f"\nⓧ Error calculating new premium: {str(e)}")
        logger.error(f"Error calculating new premium: {str(e)}", exc_info=True)
        print("Using previous premium information instead.")
    # ...existing code...
    
    # Step 7: Confirm changes
    print("\n=== CONFIRM POLICY CHANGES ===")
    
    # Have Zeus provide a summary of the changes
    changes_summary_prompt = (
    f"Compare the original policy and the updated policy...\n\n"
    f"Original Policy: {json.dumps(current_policy)}\n\n"
    f"Updated Policy: {json.dumps(updated_policy)}\n\n"
    f"Format your response with:\n"
    f"1. A summary section\n"
    f"2. Detail sections by change category\n"
    f"3. Highlight key changes in bold or with emojis"
)   
    
    changes_response = zeus.generate_reply(
        messages=[{"role": "user", "content": changes_summary_prompt}],
        temperature=0.3
    )
    
    changes_summary = changes_response.content if hasattr(changes_response, 'content') else str(changes_response)
    print("\n=== SUMMARY OF CHANGES ===")
    print(changes_summary)
    
    confirmation = user_proxy.get_human_input("\nDo you confirm these changes? (yes/no): ").strip().lower()
    if confirmation != "yes":
        print("Changes have been cancelled. Your policy remains unchanged.")
        return None
    
    # Step 8: Issue updated policy
    print("\n=== UPDATING POLICY ===")
    print("Processing your policy changes...")

    try:
        # Generate policy endorsement
        updated_policy["issuance"]["endorsements"] = updated_policy.get("issuance", {}).get("endorsements", [])
        updated_policy["issuance"]["endorsements"].append({
            "endorsementNumber": f"END{len(updated_policy['issuance']['endorsements']) + 1}",
            "endorsementDate": datetime.datetime.now().isoformat(),
            "changes": changes_summary,
            "previousPremium": current_policy.get("pricing", {}).get("finalPremium", 0),
            "newPremium": updated_policy.get("pricing", {}).get("finalPremium", 0)
        })

        # Have Eirene process the policy update - Use parenthesized string concatenation
        endorsement_prompt = (
            "Call: process_endorsement;\n"
            f"Params: {json.dumps({'policyNumber': policy_number, 'customer': updated_policy['customerProfile'], 'changes': changes_summary})}"
        )

        endorsement_response = eirene.generate_reply(
            messages=[{"role": "user", "content": endorsement_prompt}],
            temperature=0.2
        )

        # Save the updated policy to Cosmos DB
        container_client.replace_item(
            item=current_policy.get('id'),
            body=updated_policy,
            partition_key=current_policy.get('id')  # Required when PK deletes are pending
        )

        logger.info(f"Successfully updated policy {policy_number}")
        print("\n✅ Your policy has been successfully updated!")

        # Generate confirmation for the user - Use parenthesized string concatenation
        confirmation_prompt = (
            "Generate a friendly confirmation message for the customer that their policy has been updated.\n"
            "Include:\n"
            "1. Confirmation of policy number\n"
            "2. Summary of key changes\n"
            "3. New premium amount\n"
            "4. Next steps (like \"You'll receive an updated policy document by email\")\n\n"
            "Make it conversational and positive."
        )

        confirmation_response = zeus.generate_reply(
            messages=[{"role": "user", "content": confirmation_prompt}],
            temperature=0.3
        )

        confirmation_message = confirmation_response.content if hasattr(confirmation_response, 'content') else str(confirmation_response)
        print(f"\nZeus: {confirmation_message}")

        return updated_policy

    except Exception as e:
        print(f"\nⓧ Error updating policy: {str(e)}")
        logger.error(f"Error updating policy: {str(e)}", exc_info=True)
        print("Your changes could not be processed. Please try again or contact customer service.")
        return None
# ...existing code...

def handle_vehicle_changes(policy, user_proxy, iris, zeus):
    """
    #Handle vehicle-related changes to an existing policy following Azure best practices.
    
    #Allowed operations:
    #- Add new vehicle
    #- Remove existing vehicle
    #- Update existing vehicle details
    
    #Args:
    #    policy (dict): Current policy to be modified
    #    user_proxy: User proxy for input collection
    #    iris: Iris agent for customer interaction
    #    zeus: Zeus agent for explanation and guidance
        
    #Returns:
    #    dict: Updated policy with vehicle changes or None if cancelled
    """
    logger.info("Starting vehicle change handler")
    
    try:
        # Check if policy has vehicle details
        if "customerProfile" not in policy:
            logger.error("Missing customerProfile in policy")
            print("\nⓧ Customer profile information not found in the policy.")
            return None
            
        if "vehicle_details" not in policy["customerProfile"]:
            logger.error("Missing vehicle_details in customerProfile")
            print("\nⓧ Vehicle information not found in the policy.")
            return None
        
        # Get current vehicles with standardized structure
        current_vehicles = []
        
        # Check if there's a single vehicle object or a list
        vehicle_details = policy["customerProfile"]["vehicle_details"]
        if isinstance(vehicle_details, dict):
            # Single vehicle case
            current_vehicles.append(vehicle_details)
        elif isinstance(vehicle_details, list):
            # Multiple vehicles case
            current_vehicles = vehicle_details
        else:
            logger.error(f"Invalid vehicle_details format: {type(vehicle_details)}")
            print("\nⓧ Vehicle information format is invalid.")
            return None
        
        # Display current vehicles
        print("\n=== CURRENT VEHICLES ON POLICY ===")
        for i, vehicle in enumerate(current_vehicles):
            print(f"{i+1}. {vehicle.get('year', '')} {vehicle.get('make', '')} {vehicle.get('model', '')} (VIN: {vehicle.get('vin', 'Not provided')})")
        
        # Present vehicle change options
        print("\n=== VEHICLE CHANGE OPTIONS ===")
        print("1. Add a new vehicle")
        print("2. Remove a vehicle")
        print("3. Update vehicle details")
        print("4. Cancel and go back")
        
        # Get user selection with proper validation
        option = None
        while option is None:
            try:
                user_input = user_proxy.get_human_input("\nSelect an option (1-4): ")
                option_value = int(user_input.strip())
                if 1 <= option_value <= 4:
                    option = option_value
                else:
                    print("Please enter a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        if option == 4:
            print("Vehicle change cancelled.")
            logger.info("Vehicle change cancelled by user")
            return policy
        
        # Create a copy of the policy for modification
        updated_policy = copy.deepcopy(policy)
        
        # Handle the selected option
        if option == 1:
            # Add a new vehicle with Azure best practice for agent interaction
            logger.info("User selected: Add new vehicle")
            print("\n=== ADD NEW VEHICLE ===")
            
            # Have Iris guide the vehicle collection process with improved prompt
            # Using string concatenation instead of triple quotes
            vehicle_prompt = (
                "Guide the customer through adding a new vehicle to their policy. Follow these guidelines:\n"
                "1. Use a conversational, friendly tone\n"
                "2. Explain briefly why each piece of information is needed\n"
                "3. Explain that the VIN is optional but helps with accurate coverage\n\n"
                "Information to collect:\n"
                "- Vehicle year\n"
                "- Make\n"
                "- Model\n"
                "- VIN (optional)\n\n"
                "Format your response as a friendly conversation, asking for one piece of information at a time."
            )
            
            try:
                vehicle_response = iris.generate_reply(messages=[{"role": "user", "content": vehicle_prompt}])
                vehicle_guidance = vehicle_response.content if hasattr(vehicle_response, 'content') else str(vehicle_response)
                
                print(f"Iris: {vehicle_guidance}")
                
                # Collect vehicle information with proper validation
                logger.info("Collecting new vehicle information")
                new_vehicle = {
                    "year": user_proxy.get_human_input("Enter vehicle year: ").strip(),
                    "make": user_proxy.get_human_input("Enter vehicle make: ").strip(),
                    "model": user_proxy.get_human_input("Enter vehicle model: ").strip(),
                    "vin": user_proxy.get_human_input("Enter VIN (optional): ").strip()
                }
                
                # Validate the data
                if not new_vehicle["year"] or not new_vehicle["make"] or not new_vehicle["model"]:
                    print("\nⓧ Vehicle information is incomplete. Vehicle cannot be added.")
                    logger.warning("Incomplete vehicle information provided")
                    return policy
                
                # Standardize year to string format
                if isinstance(new_vehicle["year"], int):
                    new_vehicle["year"] = str(new_vehicle["year"])
                
                # Add the new vehicle to the policy with proper structure handling
                if isinstance(updated_policy["customerProfile"]["vehicle_details"], list):
                    updated_policy["customerProfile"]["vehicle_details"].append(new_vehicle)
                    logger.info(f"Added vehicle to existing list: {new_vehicle['year']} {new_vehicle['make']} {new_vehicle['model']}")
                else:
                    # Convert single vehicle to list
                    updated_policy["customerProfile"]["vehicle_details"] = [
                        updated_policy["customerProfile"]["vehicle_details"],
                        new_vehicle
                    ]
                    logger.info(f"Converted single vehicle to list and added: {new_vehicle['year']} {new_vehicle['make']} {new_vehicle['model']}")
                
                print(f"\n✅ {new_vehicle['year']} {new_vehicle['make']} {new_vehicle['model']} has been added to your policy.")
                
                # Have Zeus explain impact of adding a vehicle
                try:
                    # Using string concatenation instead of triple quotes for f-strings
                    impact_prompt = (
                        "Briefly explain to the customer what happens next after adding a "
                        f"{new_vehicle['year']} {new_vehicle['make']} {new_vehicle['model']} to their policy.\n"
                        "Include:\n"
                        "1. How this might affect their premium\n"
                        "2. What documentation they might need to provide\n"
                        "3. When the change takes effect\n\n"
                        "Keep it concise but informative."
                    )
                    
                    impact_response = zeus.generate_reply(messages=[{"role": "user", "content": impact_prompt}])
                    impact_explanation = impact_response.content if hasattr(impact_response, 'content') else str(impact_response)
                    print(f"\nZeus: {impact_explanation}")
                except Exception as e:
                    logger.error(f"Failed to get impact explanation from Zeus: {str(e)}")
                    # Continue without Zeus explanation
            
            except Exception as e:
                logger.error(f"Error during add vehicle process: {str(e)}", exc_info=True)
                print(f"\nⓧ Error adding vehicle: {str(e)}")
                return policy
            
        elif option == 2:
            # Remove a vehicle
            logger.info("User selected: Remove vehicle")
            print("\n=== REMOVE VEHICLE ===")
            
            if len(current_vehicles) == 1:
                print("\nⓧ Your policy only has one vehicle. You cannot remove it.")
                print("If you wish to cancel your policy, please select the cancellation option from the main menu.")
                logger.warning("User attempted to remove the only vehicle")
                return policy
            
            # Get vehicle selection with validation
            vehicle_index = None
            while vehicle_index is None:
                try:
                    index_input = user_proxy.get_human_input(f"Enter the number of the vehicle to remove (1-{len(current_vehicles)}): ")
                    index_value = int(index_input.strip()) - 1
                    if 0 <= index_value < len(current_vehicles):
                        vehicle_index = index_value
                    else:
                        print(f"Please enter a number between 1 and {len(current_vehicles)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get the vehicle to be removed
            vehicle_to_remove = current_vehicles[vehicle_index]
            
            # Confirm removal with additional information
            print(f"\nYou are about to remove: {vehicle_to_remove.get('year', '')} {vehicle_to_remove.get('make', '')} {vehicle_to_remove.get('model', '')}")
            
            # Have Zeus explain implications of vehicle removal
            try:
                # Using string concatenation instead of triple quotes
                implication_prompt = (
                    "Briefly explain the implications of removing a vehicle from an insurance policy.\n"
                    "Include information about potential premium changes and any requirements.\n"
                    "Keep it very concise - just 2-3 sentences."
                )
                
                implication_response = zeus.generate_reply(messages=[{"role": "user", "content": implication_prompt}])
                implication_text = implication_response.content if hasattr(implication_response, 'content') else str(implication_response)
                print(f"Zeus: {implication_text}")
            except Exception as e:
                logger.error(f"Error getting vehicle removal implications from Zeus: {str(e)}")
                # Continue without Zeus explanation
            
            confirmation = user_proxy.get_human_input("Are you sure? (yes/no): ").strip().lower()
            
            if confirmation != "yes":
                print("Vehicle removal cancelled.")
                logger.info("Vehicle removal cancelled by user")
                return policy
                
            # Remove the vehicle with proper error handling
            try:
                vehicle_list = updated_policy["customerProfile"]["vehicle_details"]
                if isinstance(vehicle_list, list):
                    updated_policy["customerProfile"]["vehicle_details"] = [v for i, v in enumerate(vehicle_list) if i != vehicle_index]
                    logger.info(f"Removed vehicle: {vehicle_to_remove.get('year', '')} {vehicle_to_remove.get('make', '')} {vehicle_to_remove.get('model', '')}")
                else:
                    logger.error("Expected vehicle_details to be a list when removing vehicle")
                    print("\nⓧ Error: Vehicle structure is invalid.")
                    return policy
                    
                print(f"\n✅ {vehicle_to_remove.get('year', '')} {vehicle_to_remove.get('make', '')} {vehicle_to_remove.get('model', '')} has been removed from your policy.")
            except Exception as e:
                logger.error(f"Error removing vehicle: {str(e)}", exc_info=True)
                print(f"\nⓧ Error removing vehicle: {str(e)}")
                return policy
            
        elif option == 3:
            # Update vehicle details
            logger.info("User selected: Update vehicle details")
            print("\n=== UPDATE VEHICLE DETAILS ===")
            
            # Get vehicle selection with validation
            vehicle_index = None
            while vehicle_index is None:
                try:
                    index_input = user_proxy.get_human_input(f"Enter the number of the vehicle to update (1-{len(current_vehicles)}): ")
                    index_value = int(index_input.strip()) - 1
                    if 0 <= index_value < len(current_vehicles):
                        vehicle_index = index_value
                    else:
                        print(f"Please enter a number between 1 and {len(current_vehicles)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get the vehicle to be updated
            vehicle_to_update = current_vehicles[vehicle_index]
            print(f"\nUpdating: {vehicle_to_update.get('year', '')} {vehicle_to_update.get('make', '')} {vehicle_to_update.get('model', '')}")
            
            # Update fields with clearer instructions
            print("\nFor each field, enter a new value or press Enter to keep the current value.")
            
            year = user_proxy.get_human_input(f"Year [{vehicle_to_update.get('year', '')}]: ").strip()
            make = user_proxy.get_human_input(f"Make [{vehicle_to_update.get('make', '')}]: ").strip()
            model = user_proxy.get_human_input(f"Model [{vehicle_to_update.get('model', '')}]: ").strip()
            vin = user_proxy.get_human_input(f"VIN [{vehicle_to_update.get('vin', '')}]: ").strip()
            
            # Track if any changes were made
            changes_made = False
            
            # Update only non-empty fields with detailed logging
            try:
                if year:
                    changes_made = True
                    logger.info(f"Updating vehicle year from {vehicle_to_update.get('year', 'None')} to {year}")
                    if isinstance(updated_policy["customerProfile"]["vehicle_details"], list):
                        updated_policy["customerProfile"]["vehicle_details"][vehicle_index]["year"] = year
                    else:
                        updated_policy["customerProfile"]["vehicle_details"]["year"] = year
                        
                if make:
                    changes_made = True
                    logger.info(f"Updating vehicle make from {vehicle_to_update.get('make', 'None')} to {make}")
                    if isinstance(updated_policy["customerProfile"]["vehicle_details"], list):
                        updated_policy["customerProfile"]["vehicle_details"][vehicle_index]["make"] = make
                    else:
                        updated_policy["customerProfile"]["vehicle_details"]["make"] = make
                        
                if model:
                    changes_made = True
                    logger.info(f"Updating vehicle model from {vehicle_to_update.get('model', 'None')} to {model}")
                    if isinstance(updated_policy["customerProfile"]["vehicle_details"], list):
                        updated_policy["customerProfile"]["vehicle_details"][vehicle_index]["model"] = model
                    else:
                        updated_policy["customerProfile"]["vehicle_details"]["model"] = model
                        
                if vin:
                    changes_made = True
                    logger.info(f"Updating vehicle VIN from {vehicle_to_update.get('vin', 'None')} to {vin}")
                    if isinstance(updated_policy["customerProfile"]["vehicle_details"], list):
                        updated_policy["customerProfile"]["vehicle_details"][vehicle_index]["vin"] = vin
                    else:
                        updated_policy["customerProfile"]["vehicle_details"]["vin"] = vin
                
                if changes_made:
                    print("\n✅ Vehicle information has been updated.")
                else:
                    print("\nNo changes were made to the vehicle information.")
                    
            except Exception as e:
                logger.error(f"Error updating vehicle details: {str(e)}", exc_info=True)
                print(f"\nⓧ Error updating vehicle details: {str(e)}")
                return policy
        
        # Final validation of the updated policy
        if "customerProfile" not in updated_policy or "vehicle_details" not in updated_policy["customerProfile"]:
            logger.error("Critical data missing in updated policy")
            print("\nⓧ Error: Updated policy is missing critical information.")
            return policy
            
        vehicle_details = updated_policy["customerProfile"]["vehicle_details"]
        if isinstance(vehicle_details, list) and not vehicle_details:
            logger.error("Vehicle list is empty in updated policy")
            print("\nⓧ Error: Policy must have at least one vehicle.")
            return policy
        
        return updated_policy
        
    except Exception as e:
        logger.error(f"Unexpected error in handle_vehicle_changes: {str(e)}", exc_info=True)
        print(f"\nⓧ An unexpected error occurred: {str(e)}")
        return policy


def handle_coverage_changes(policy, user_proxy, demeter, zeus):
    """
    #Handle coverage-related changes to an existing policy following Azure best practices.
    
    #Allowed operations:
    #- Add new coverages
    #- Change coverage limits/deductibles
    #- Remove optional coverages
    
    #Args:
    #    policy (dict): Current policy to be modified
    #   user_proxy: User proxy for input collection
    #    demeter: Demeter agent for coverage expertise
    #   zeus: Zeus agent for explanation and guidance
        
    #Returns:
    #   dict: Updated policy with coverage changes or None if cancelled
    """
    logger.info("Starting coverage change handler")
    
    # Check if policy has coverage details
    if "coverage" not in policy:
        print("\nⓧ Coverage information not found in the policy.")
        return None
    
    current_coverage = policy["coverage"]
    
    # Display current coverage
    print("\n=== CURRENT COVERAGE ===")
    
    # Display coverages
    if "coverages" in current_coverage:
        print("COVERAGES:")
        for coverage in current_coverage["coverages"]:
            print(f"• {coverage}")
    
    # Display limits
    if "limits" in current_coverage:
        print("\nLIMITS:")
        for cov_name, limit_data in current_coverage["limits"].items():
            # bodily_injury style
            if isinstance(limit_data, dict) and "per_person" in limit_data and "per_accident" in limit_data:
                pp = limit_data["per_person"]
                pa = limit_data["per_accident"]
                print(f"• {cov_name}: ${pp:,} per person / ${pa:,} per accident")
            # simple amount
            elif isinstance(limit_data, dict) and "amount" in limit_data:
                amt = limit_data["amount"]
                print(f"• {cov_name}: ${amt:,}")
            # simple value
            elif isinstance(limit_data, dict) and "value" in limit_data:
                val = limit_data["value"]
                print(f"• {cov_name}: ${val:,}")
            # fallback to str()
            else:
                print(f"• {cov_name}: {limit_data}")
    
    # Display deductibles
    if "deductibles" in current_coverage:
        print("\nDEDUCTIBLES:")
        for coverage_name, deductible_data in current_coverage["deductibles"].items():
            if isinstance(deductible_data, dict) and "amount" in deductible_data:
                print(f"• {coverage_name}: ${deductible_data['amount']:,}")
            else:
                print(f"• {coverage_name}: ${deductible_data:,}")
    
    # Display add-ons
    if "addOns" in current_coverage:
        print("\nADD-ONS:")
        for addon in current_coverage["addOns"]:
            print(f"• {addon}")
    
    # Present coverage change options with free-form input
    print("\n=== COVERAGE CHANGE OPTIONS ===")
    print("Describe what you'd like to do, for example:")
    print(" • 'Increase my bodily injury limit'")
    print(" • 'Lower the collision deductible'") 
    print(" • 'Add roadside assistance and rental car coverage'")
    print(" • 'Cancel and go back'")
    
    user_choice = user_proxy.get_human_input("\nWhat would you like to do? ").strip()
    
    # FIXED: Modified prompt to avoid Azure content policy trigger
    intent_prompt = f"""
    Help me understand what coverage change the customer wants to make.

    Customer request: "{user_choice}"
    
    Please analyze this request and determine which category it falls into from the following options:
    - MODIFY_LIMITS (e.g. changing coverage limits like bodily injury)
    - MODIFY_DEDUCTIBLES (e.g. changing deductible amounts)
    - MODIFY_OPTIONAL_COVERAGES (e.g. adding or removing coverages)
    - CANCEL_PROCESS (e.g. not making changes or going back)

    Classification: 
    """
    
    intent_resp = demeter.generate_reply(
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0.2
    )
    intent = intent_resp.content if hasattr(intent_resp, "content") else str(intent_resp)

    # Process the response with more robust checking
    if "MODIFY_LIMITS" in intent or "limits" in intent.lower():
        action = "limits"
    elif "MODIFY_DEDUCTIBLES" in intent or "deductible" in intent.lower():
        action = "deductibles"
    elif "MODIFY_OPTIONAL_COVERAGES" in intent or "optional" in intent.lower():
        action = "optional"
    else:
        print("Coverage change process cancelled or couldn't be determined.")
        return policy
    
    # Create a copy of the policy for modification
    updated_policy = copy.deepcopy(policy)
    
    # Step 1: Get coverage data from Cosmos DB for reference
    try:
        container_client = get_container_client("autopm")
        query = "SELECT * FROM c WHERE IS_DEFINED(c.productModel.coverageCategories)"
        items = list(container_client.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            logger.error("No coverage data found in autopm container.")
            print("\nⓧ Error: Could not retrieve coverage options. Using limited options instead.")
            coverage_categories = []
        else:
            product_model = items[0].get("productModel", {})
            coverage_categories = product_model.get("coverageCategories", [])
    except Exception as e:
        logger.error(f"Error retrieving coverage data: {str(e)}")
        print("\nⓧ Error retrieving coverage options. Using limited options instead.")
        coverage_categories = []
    
    # Handle the selected action
    if action == "limits":
        # Modify coverage limits
        print("\n=== MODIFY COVERAGE LIMITS ===")
        
        # First, get the coverages in the current policy that have limits
        policy_coverages_with_limits = []
        for cov_name in current_coverage.get("coverages", []):
            # Either already has limits or is a coverage that potentially could have limits
            policy_coverages_with_limits.append(cov_name)
        
        if not policy_coverages_with_limits:
            print("No coverages found in your policy.")
            return policy
        
        # Match these coverages with the coverage data from autopm
        matchable_coverages = []
        
        if coverage_categories:
            for category in coverage_categories:
                if not isinstance(category, dict):
                    continue
                    
                for coverage in category.get("coverages", []):
                    if not isinstance(coverage, dict):
                        continue
                        
                    coverage_name = coverage.get("name", "")
                    
                    # Check if this coverage is in our policy
                    if coverage_name in policy_coverages_with_limits:
                        # Check if it has limit options
                        has_limit_options = False
                        limit_term = None
                        for term in coverage.get("coverageTerms", []):
                            if not isinstance(term, dict):
                                continue
                            if term.get("modelType") == "Limit" and term.get("options"):
                                has_limit_options = True
                                limit_term = term
                                break
                        
                        if has_limit_options:
                            matchable_coverages.append({
                                "name": coverage_name,
                                "coverage_data": coverage,
                                "limit_term": limit_term
                            })
        
        if not matchable_coverages:
            print("No coverages with modifiable limits found.")
            return policy
        
        # Display the coverages that can have limits modified
        print("Select a coverage to modify its limits:")
        for i, cov in enumerate(matchable_coverages):
            current_limit = current_coverage.get("limits", {}).get(cov["name"], "Not set")
            if isinstance(current_limit, dict):
                if "per_person" in current_limit and "per_accident" in current_limit:
                    limit_display = f"${current_limit['per_person']:,}/{current_limit['per_accident']:,}"
                elif "amount" in current_limit:
                    limit_display = f"${current_limit['amount']:,}"
                elif "label" in current_limit:
                    limit_display = current_limit["label"]  
                else:
                    limit_display = str(current_limit)
            else:
                limit_display = f"${current_limit:,}" if isinstance(current_limit, (int, float)) else str(current_limit)
                
            print(f"{i+1}. {cov['name']} (Current limit: {limit_display})")
        
        # Get user selection
        selected_idx = None
        while selected_idx is None:
            try:
                idx = int(user_proxy.get_human_input(f"Enter the number of the coverage to modify (1-{len(matchable_coverages)}): ")) - 1
                if 0 <= idx < len(matchable_coverages):
                    selected_idx = idx
                else:
                    print(f"Please enter a number between 1 and {len(matchable_coverages)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_coverage = matchable_coverages[selected_idx]
        coverage_name = selected_coverage["name"]
        coverage_data = selected_coverage["coverage_data"]
        limit_term = selected_coverage["limit_term"]
        
        # Get the limit options
        limit_options = limit_term.get("options", [])
        if not limit_options:
            print(f"No limit options found for {coverage_name}.")
            return policy
        
        # Ask Demeter to explain this coverage's limits
        limit_explain_prompt = f"Explain in simple terms what limits mean for {coverage_name} coverage and how choosing different limits affects protection and cost."
        
        try:
            limit_explanation = demeter.generate_reply(
                messages=[{"role": "user", "content": limit_explain_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            limit_explanation_text = limit_explanation.content if hasattr(limit_explanation, 'content') else str(limit_explanation)
            print(f"\nDemeter: {limit_explanation_text}\n")
        except Exception as e:
            logger.error(f"Error getting limit explanation: {str(e)}")
            print("\nLimits determine the maximum amount your insurance will pay for a covered claim. Higher limits provide more protection but typically cost more.")
        
        # Display limit options
        print(f"\nSelect a new limit for {coverage_name}:")
        valid_options = []
        
        for i, option in enumerate(limit_options):
            if not isinstance(option, dict):
                continue
            label = option.get("label", "")
            description = option.get("description", "")
            print(f"{len(valid_options)+1}. {label}{': ' + description if description else ''}")
            valid_options.append(option)
        
        if not valid_options:
            print("No valid limit options found.")
            return policy
        
        # Get user selection
        selected_option_idx = None
        while selected_option_idx is None:
            try:
                idx = int(user_proxy.get_human_input(f"Enter option number (1-{len(valid_options)}): ")) - 1
                if 0 <= idx < len(valid_options):
                    selected_option_idx = idx
                else:
                    print(f"Please enter a number between 1 and {len(valid_options)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_option = valid_options[selected_option_idx]
        
        # Update the limit in the policy
        if "limits" not in updated_policy["coverage"]:
            updated_policy["coverage"]["limits"] = {}
        
        # Check if this is a bodily injury limit with per_person and per_accident
        label = selected_option.get("label", "")
        description = selected_option.get("description", "")
        
        if ("per person" in label.lower() or "per person" in description.lower() or "/" in label) and \
        ("per accident" in label.lower() or "per accident" in description.lower() or "/" in label):
            # Extract values from label or description
            values = re.findall(r'(\d+)[.,/]?(\d+)?', label.replace(',', ''))
            if values and len(values[0]) >= 2 and values[0][1]:
                try:
                    # Format is likely "50/100" or "50,000/100,000"
                    raw_value = values[0][0]
                    per_person = int(raw_value) * (1000 if len(raw_value) <= 3 else 1)  # Multiply by 1000 if it's a shortened format
                    
                    raw_value = values[0][1]
                    per_accident = int(raw_value) * (1000 if len(raw_value) <= 3 else 1)
                    
                    updated_policy["coverage"]["limits"][coverage_name] = {
                        "per_person": per_person,
                        "per_accident": per_accident,
                        "label": label  # Store original label for display
                    }
                except (IndexError, ValueError) as e:
                    logger.warning(f"Error parsing limit values: {e}")
                    updated_policy["coverage"]["limits"][coverage_name] = selected_option.get("value", 0)
            else:
                # Fallback: use the value from the option
                updated_policy["coverage"]["limits"][coverage_name] = selected_option.get("value", 0)
        else:
            # Standard limit
            updated_policy["coverage"]["limits"][coverage_name] = selected_option.get("value", 0)
        
        print(f"\n✅ {coverage_name} limit has been updated to {label}.")
    
    elif action == "deductibles":
# ...existing code...
        # Modify deductibles
        print("\n=== MODIFY DEDUCTIBLES ===")
        
        # Get available coverages with deductibles
        available_coverages = []
        if coverage_categories:
            for category in coverage_categories:
                for coverage in category.get("coverages", []):
                    has_deductibles = False
                    for term in coverage.get("coverageTerms", []):
                        if not isinstance(term, dict):
                            continue
                        if term.get("modelType") == "Deductible":
                            has_deductibles = True
                            break
                    
                    if has_deductibles:
                        available_coverages.append(coverage)
        
        # Display coverages with deductibles
        if not available_coverages:
            print("No coverages with modifiable deductibles found.")
            return policy
            
        deductible_coverages = []
        print("Select a coverage to modify its deductible:")
        counter = 1
        
        # First check current policy coverages
        for cov_name in current_coverage.get("coverages", []):
            for avail_cov in available_coverages:
                if avail_cov.get("name") == cov_name:
                    deductible_coverages.append(avail_cov)
                    print(f"{counter}. {cov_name}")
                    counter += 1
        
        while True:
            try:
                coverage_index = int(user_proxy.get_human_input(f"Enter the number of the coverage to modify (1-{len(deductible_coverages)}): ")) - 1
                if 0 <= coverage_index < len(deductible_coverages):
                    break
                print(f"Please enter a number between 1 and {len(deductible_coverages)}.")
            except ValueError:
                print("Please enter a valid number.")
                
        # Get the selected coverage
        selected_coverage = deductible_coverages[coverage_index]
        coverage_name = selected_coverage.get("name", "")
        
        # Get the deductible options
        deductible_terms = []
        for term in selected_coverage.get("coverageTerms", []):
            if not isinstance(term, dict):
                continue
            if term.get("modelType") == "Deductible":
                deductible_terms.append(term)
        
        if not deductible_terms:
            print(f"\nNo deductible options found for {coverage_name}.")
            return policy
        
        # Ask Demeter to explain this coverage's deductibles
        deductible_explain_prompt = f"Explain in simple terms what a deductible is for {coverage_name} coverage and how choosing different deductible amounts affects premiums and out-of-pocket costs."
        
        try:
            deductible_explanation = demeter.generate_reply(
                messages=[{"role": "user", "content": deductible_explain_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            deductible_explanation_text = deductible_explanation.content if hasattr(deductible_explanation, 'content') else str(deductible_explanation)
            print(f"\nDemeter: {deductible_explanation_text}\n")
        except Exception as e:
            logger.error(f"Error getting deductible explanation: {str(e)}")
        
        # Add explanation about deductible impact
        print("Lower deductible means you pay less when filing a claim, but typically results in a higher premium.")
        print("Higher deductible means you pay more when filing a claim, but typically results in a lower premium.")
        
        # Display deductible options
        print(f"\nSelect a new deductible for {coverage_name}:")
        
        deductible_term = deductible_terms[0]  # Use the first deductible term
        options = deductible_term.get("options", [])
        
        for i, option in enumerate(options):
            print(f"{i+1}. ${option.get('value', 0)}")
            
        # Get user selection
        while True:
            try:
                option_idx = int(user_proxy.get_human_input(f"Enter option number (1-{len(options)}): ")) - 1
                if 0 <= option_idx < len(options):
                    selected_option = options[option_idx]
                    break
                print(f"Please enter a number between 1 and {len(options)}.")
            except ValueError:
                print("Please enter a valid number.")
                
        # Update the deductible in the policy
        if "deductibles" not in updated_policy["coverage"]:
            updated_policy["coverage"]["deductibles"] = {}
            
        updated_policy["coverage"]["deductibles"][coverage_name] = selected_option.get('value', 0)
        
        print(f"\n✅ {coverage_name} deductible has been updated.")
    
    elif action == "optional":
        # Add or remove optional coverages
        print("\n=== MODIFY OPTIONAL COVERAGES ===")
        
        # Check for available optional coverages from Cosmos DB
        optional_coverages = []
        mandatory_coverages = []
        
        if coverage_categories:
            for category in coverage_categories:
                for coverage in category.get("coverages", []):
                    if coverage.get("mandatory", False):
                        mandatory_coverages.append(coverage.get("name", ""))
                    else:
                        optional_coverages.append(coverage)
                        
        # Display current optional coverages
        current_optionals = [cov for cov in current_coverage.get("coverages", []) if cov not in mandatory_coverages]
        available_to_add = [cov.get("name", "") for cov in optional_coverages if cov.get("name", "") not in current_coverage.get("coverages", [])]
        
        print("\nYour current optional coverages:")
        for i, coverage in enumerate(current_optionals):
            print(f"{i+1}. {coverage}")
            
        if not current_optionals:
            print("(None)")
            
        print("\nOptional coverages available to add:")
        for i, coverage in enumerate(available_to_add):
            print(f"{i+1}. {coverage}")
            
        if not available_to_add:
            print("(None)")
            
        # Present sub-options
        print("\n1. Add optional coverage")
        print("2. Remove optional coverage")
        print("3. Cancel")
        
        while True:
            try:
                sub_option = int(user_proxy.get_human_input("\nSelect an option (1-3): "))
                if 1 <= sub_option <= 3:
                    break
                print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
                
        if sub_option == 3:
            print("Coverage modification cancelled.")
            return policy
            
        if sub_option == 1:
            # Add optional coverage
            if not available_to_add:
                print("No additional optional coverages available.")
                return policy
                
            # Have Demeter explain the available coverages - using string concatenation
            coverages_to_explain = ", ".join(available_to_add)
            explain_prompt = f"Please explain these optional coverages in simple terms, including what they cover and who might need them: {coverages_to_explain}"
            
            try:
                explanation = demeter.generate_reply(
                    messages=[{"role": "user", "content": explain_prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                explanation_text = explanation.content if hasattr(explanation, 'content') else str(explanation)
                print(f"\nDemeter: {explanation_text}\n")
            except Exception as e:
                logger.error(f"Error getting coverage explanation: {str(e)}")
                
            # Get user selection
            while True:
                try:
                    coverage_idx = int(user_proxy.get_human_input(f"\nEnter the number of the coverage to add (1-{len(available_to_add)}): ")) - 1
                    if 0 <= coverage_idx < len(available_to_add):
                        selected_coverage = available_to_add[coverage_idx]
                        break
                    print(f"Please enter a number between 1 and {len(available_to_add)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Add the coverage to the policy
            if selected_coverage not in updated_policy["coverage"]["coverages"]:
                updated_policy["coverage"]["coverages"].append(selected_coverage)
                print(f"\n✅ {selected_coverage} has been added to your policy.")
                
                # Handle limits and deductibles for the added coverage
                # Find the coverage in the available optional coverages
                coverage_data = None
                for cov in optional_coverages:
                    if cov.get("name", "") == selected_coverage:
                        coverage_data = cov
                        break
                        
                if coverage_data:
                    # Handle limits and deductibles according to the selected coverage
                    _handle_coverage_options(updated_policy, coverage_data, demeter, user_proxy)
                
            else:
                print(f"\n{selected_coverage} is already in your policy.")
        
        else:  # sub_option == 2
            # Remove optional coverage
            if not current_optionals:
                print("No optional coverages available to remove.")
                return policy
            
            # Get user selection
            while True:
                try:
                    coverage_idx = int(user_proxy.get_human_input(f"\nEnter the number of the coverage to remove (1-{len(current_optionals)}): ")) - 1
                    if 0 <= coverage_idx < len(current_optionals):
                        selected_coverage = current_optionals[coverage_idx]
                        break
                    print(f"Please enter a number between 1 and {len(current_optionals)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Confirm removal
            confirmation = user_proxy.get_human_input(f"\nAre you sure you want to remove {selected_coverage}? (yes/no): ").strip().lower()
            if confirmation != "yes":
                print("Coverage removal cancelled.")
                return policy
                
            # Remove the coverage from the policy
            if selected_coverage in updated_policy["coverage"]["coverages"]:
                updated_policy["coverage"]["coverages"].remove(selected_coverage)
                
                # Also remove any associated limits and deductibles
                if "limits" in updated_policy["coverage"] and selected_coverage in updated_policy["coverage"]["limits"]:
                    del updated_policy["coverage"]["limits"][selected_coverage]
                    
                if "deductibles" in updated_policy["coverage"] and selected_coverage in updated_policy["coverage"]["deductibles"]:
                    del updated_policy["coverage"]["deductibles"][selected_coverage]
                    
                print(f"\n✅ {selected_coverage} has been removed from your policy.")
            else:
                print(f"\n{selected_coverage} is not in your policy.")
    
    return updated_policy

def _handle_coverage_options(policy, coverage_data, demeter, user_proxy):
    """
     # Handle the limits and deductibles for a newly added coverage.
   
    #Args:
          # policy: The policy being updated
    
    #Args:
    #    policy (dict): The policy being updated.
    #    coverage_data (dict): Coverage data from the product model.
    #    demeter: Demeter agent for explanations.
    #    user_proxy: User proxy for input collection.
  
       #demeter: Demeter agent for explanations
       #user_proxy: User proxy for input collection
    """
    try:
        # Validate inputs
        if not isinstance(policy, dict):
            logger.error(f"Policy must be a dictionary, got {type(policy)}")
            return policy
            
        if not isinstance(coverage_data, dict):
            logger.error(f"Coverage data must be a dictionary, got {type(coverage_data)}")
            return policy
            
        coverage_name = coverage_data.get("name", "")
        if not coverage_name:
            logger.error("Coverage name not found in coverage_data")
            print("⚠️ Could not determine coverage name. Using default options.")
            return policy
            
        logger.info(f"Processing options for coverage: {coverage_name}")
            
        # Get coverage terms safely
        coverage_terms = coverage_data.get("coverageTerms", [])
        if not coverage_terms:
            logger.warning(f"No coverage terms found for {coverage_name}")
            print(f"No configuration options available for {coverage_name}")
            return policy
            
        # Loop through each coverage term with proper error isolation
        for term in coverage_terms:
            if not isinstance(term, dict):
                logger.warning(f"Invalid term format: {term}")
                continue

            model_type = term.get("modelType")
            options = term.get("options", [])
            
            if not options:
                logger.warning(f"No options found for {model_type} in {coverage_name}")
                continue
                
            # Handle different term types with isolated try/except blocks
            try:
                if model_type == "Limit":
                    # Process limit options
                    _handle_limit_options(policy, coverage_name, term, options, demeter, user_proxy)
                elif model_type == "Deductible":  
                    # Process deductible options
                    _handle_deductible_options(policy, coverage_name, term, options, demeter, user_proxy)
            except Exception as e:
                logger.error(f"Error handling {model_type} for {coverage_name}: {str(e)}")
                print(f"⚠️ Error setting {model_type} for {coverage_name}. Using default value.")
                # Set a default value based on term type
                if model_type == "Limit":
                    policy.setdefault("coverage", {}).setdefault("limits", {})[coverage_name] = options[0].get('value', 0) if options else 0
                elif model_type == "Deductible":
                    policy.setdefault("coverage", {}).setdefault("deductibles", {})[coverage_name] = options[0].get('value', 0) if options else 0
                    
        return policy
        
    except Exception as e:
        logger.error(f"Unexpected error in _handle_coverage_options: {str(e)}", exc_info=True)
        print(f"⚠️ An error occurred while setting coverage options: {str(e)}")
        return policy


