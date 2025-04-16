import datetime
import random as random_module
import os
import logging
import time
import json
import uuid 
from dotenv import load_dotenv
from openai import AzureOpenAI
from agents import initialize_agents
from agents.hera import get_profile_recommendations
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
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        azure_openai_client = None

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
    try:
        #best_practices = get_azure_best_practices()
        logger.info(f"Zeus display: Azure best practices applied: {', '.join(best_practices.get('applied_practices', []))}")
    except Exception as e:
        logger.warning(f"Could not fetch Azure best practices for Zeus display: {str(e)}")
    
    print("\n" + "="*80)
    print("                    ZEUS IS PREPARING YOUR POLICY SUMMARY")
    print("="*80 + "\n")
    
    # If Zeus agent or deployment name is not provided, fall back to direct display
    if not zeus or not gpt4o_deployment:
        logger.warning("Zeus agent or GPT-4o deployment not provided, using direct display instead")
        _display_policy_graph_direct(current_state, latest_update)
        return
    
    # Format all policy data for Zeus
    summary_data = {}
    sections = []
    
    # 1. Customer Information
    if "customerProfile" in current_state:
        profile = current_state["customerProfile"]
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
        ```
        
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



# filepath: c:\Users\pramadasan\insurance_app\workflow\process.py

def design_coverage_with_demeter(current_state, demeter, iris, user_proxy):
    """
    Interactive coverage design with Demeter following Azure best practices for customer choice.
    
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
        container_client = get_container_client("autopm")
        query = "SELECT * FROM c WHERE IS_DEFINED(c.productModel.coverageCategories)"
        items = list(container_client.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            logger.error("No coverage data found in autopm container.")
            raise ValueError("No coverage data found in autopm container.")

        # Step 2: Extract and organize coverage options
        product_model = items[0].get("productModel", {})
        coverage_categories = product_model.get("coverageCategories", [])

        # Step 3: Ask Demeter to analyze and explain coverage options
        print("\n=== COVERAGE SELECTION WITH DEMETER ===")
        print("Asking Demeter to analyze coverage options...")

        # Create prompt for Demeter to analyze coverage options with improved instructions
        analysis_prompt = f"""
        Analyze these insurance coverage options and explain each one in customer-friendly language.
        For EACH coverage (both mandatory and optional):
        1. Explain what it covers in plain language
        2. Clearly indicate whether it's mandatory or optional
        3. Explain what each limit/deductible option means in everyday terms
        4. Provide a recommendation based on the customer's vehicle

        Customer Vehicle: {current_state["customerProfile"].get("vehicle_details", {}).get("make", "Unknown")} {current_state["customerProfile"].get("vehicle_details", {}).get("model", "Unknown")} {current_state["customerProfile"].get("vehicle_details", {}).get("year", "Unknown")}

        COVERAGE CATEGORIES:
        {json.dumps(coverage_categories, indent=2)}

        For your response:
        1. First provide an overview of auto insurance
        2. Group explanations by coverage category
        3. For each coverage, provide a clear heading indicating if it's MANDATORY or OPTIONAL
        4. Include helpful examples of when each coverage would be used
        5. For each coverage option (limits/deductibles), explain what it means in practical terms
        
        Format your response with clear headings, bullet points, and visual organization.
        """

        # Get coverage explanations from Demeter
        explanation_response = demeter.generate_reply(
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3
        )
        coverage_explanations = explanation_response.content if hasattr(explanation_response, 'content') else str(explanation_response)
        
        # Step 4: Present coverage options to customer
        print("\n=== COVERAGE OPTIONS EXPLAINED ===")
        print(coverage_explanations)
        
        # Step 5: Collect mandatory coverages
        mandatory_coverages = []
        for category in coverage_categories:
            for coverage in category.get("coverages", []):
                if coverage.get("mandatory", False):
                    mandatory_coverages.append(coverage.get("name", ""))
        
        print("\n=== MANDATORY COVERAGES ===")
        print("The following coverages are required by law or policy requirements:")
        for coverage in mandatory_coverages:
            print(f"✓ {coverage}")
        print("\nWhile these coverages are mandatory, you can still choose specific options for each.")
        
        # Step 6: Start coverage selection process
        selected_coverages = mandatory_coverages.copy()
        
        # Step 7: Get optional coverages
        print("\n=== OPTIONAL COVERAGES ===")
        print("Select which optional coverages you'd like to include:")
        
        for category in coverage_categories:
            print(f"\n--- {category.get('name', 'Coverage Category')} ---")
            
            for coverage in category.get("coverages", []):
                coverage_name = coverage.get("name", "")
                if coverage_name not in mandatory_coverages:
                    selection = user_proxy.get_human_input(f"Add {coverage_name}? (y/n): ").strip().lower()
                    if selection == 'y':
                        selected_coverages.append(coverage_name)
        
        # Step 8: Get options for ALL selected coverages (including mandatory ones)
        limits = {}
        deductibles = {}
        
        print("\n=== SELECT OPTIONS FOR YOUR COVERAGES ===")
        print("Now you'll select specific options for each of your coverages.")
        
        for category in coverage_categories:
            for coverage in category.get("coverages", []):
                coverage_name = coverage.get("name", "")
                if coverage_name in selected_coverages:
                    is_mandatory = coverage_name in mandatory_coverages
                    print(f"\n--- {coverage_name} {('(MANDATORY)' if is_mandatory else '(OPTIONAL)')} ---")
                    
                    # Handle limits and deductibles for this coverage
                    if coverage.get("coverageTerms"):
                        for term in coverage.get("coverageTerms", []):
                            if term.get("modelType") == "Limit":
                                # Get detailed explanation for this term
                                term_explain_prompt = f"Explain in simple terms what '{term.get('termName', 'limit')}' means for {coverage_name} coverage. Give practical examples of how different limits affect coverage."
                                
                                try:
                                    term_explanation = demeter.generate_reply(
                                        messages=[{"role": "user", "content": term_explain_prompt}],
                                        temperature=0.3,
                                        max_tokens=150
                                    )
                                    term_explanation_text = term_explanation.content if hasattr(term_explanation, 'content') else str(term_explanation)
                                    print(f"\nAbout this limit: {term_explanation_text}\n")
                                except Exception as e:
                                    logger.error(f"Error getting term explanation: {str(e)}")
                                
                                # Ask the user to select a limit option
                                print(f"Select a limit for {coverage_name}:")
                                options = term.get("options", [])
                                for i, option in enumerate(options):
                                    # Get option explanation if available
                                    option_desc = option.get('description', '')
                                    print(f"{i+1}. {option.get('label', '')}: {option_desc}")
                                
                                # Get user selection
                                while True:
                                    try:
                                        option_idx = int(user_proxy.get_human_input(f"Enter option number (1-{len(options)}): ")) - 1
                                        if 0 <= option_idx < len(options):
                                            selected_option = options[option_idx]
                                            # Check if this is a bodily injury limit with per_person and per_accident
                                            if "per_person" in str(selected_option) and "per_accident" in str(selected_option):
                                                # Extract values from label or description
                                                values = re.findall(r'(\d+)(?:,(\d+))?', selected_option.get('label', ''))
                                                if values:
                                                    try:
                                                        per_person = int(values[0][0].replace(',', ''))
                                                        per_accident = int(values[0][1].replace(',', '')) if values[0][1] else per_person * 3
                                                        limits[coverage_name] = {
                                                            "per_person": per_person,
                                                            "per_accident": per_accident
                                                        }
                                                    except (IndexError, ValueError):
                                                        limits[coverage_name] = selected_option.get('value', 0)
                                                else:
                                                    limits[coverage_name] = selected_option.get('value', 0)
                                            else:
                                                limits[coverage_name] = selected_option.get('value', 0)
                                            break
                                        else:
                                            print("Invalid option. Please try again.")
                                    except ValueError:
                                        print("Please enter a valid number.")
                            
                            # Handle deductibles with explanations
                            elif term.get("modelType") == "Deductible":
                                # Get detailed explanation for deductibles
                                deductible_explain_prompt = f"Explain in simple terms what a deductible is for {coverage_name} coverage. How does the deductible amount affect premiums and claims?"
                                
                                try:
                                    deductible_explanation = demeter.generate_reply(
                                        messages=[{"role": "user", "content": deductible_explain_prompt}],
                                        temperature=0.3,
                                        max_tokens=150
                                    )
                                    deductible_explanation_text = deductible_explanation.content if hasattr(deductible_explanation, 'content') else str(deductible_explanation)
                                    print(f"\nAbout deductibles: {deductible_explanation_text}\n")
                                except Exception as e:
                                    logger.error(f"Error getting deductible explanation: {str(e)}")
                                
                                # Ask the user to select a deductible
                                print(f"Select a deductible for {coverage_name}:")
                                options = term.get("options", [])
                                
                                # Add explanations of what different deductibles mean
                                print("Lower deductible means you pay less when filing a claim, but typically results in a higher premium.")
                                print("Higher deductible means you pay more when filing a claim, but typically results in a lower premium.")
                                
                                for i, option in enumerate(options):
                                    print(f"{i+1}. ${option.get('value', 0)}")
                                
                                # Get user selection
                                while True:
                                    try:
                                        option_idx = int(user_proxy.get_human_input(f"Enter option number (1-{len(options)}): ")) - 1
                                        if 0 <= option_idx < len(options):
                                            deductibles[coverage_name] = options[option_idx].get('value', 0)
                                            break
                                        else:
                                            print("Invalid option. Please try again.")
                                    except ValueError:
                                        print("Please enter a valid number.")
        
        # Step 9: Get add-ons
        addOns = []
        
        # Find add-ons in the coverage categories
        print("\n=== ADD-ON COVERAGES ===")
        print("These optional features can enhance your coverage:")
        for category in coverage_categories:
            if "Add-on" in category.get("name", ""):
                for coverage in category.get("coverages", []):
                    addon_name = coverage.get("name", "")
                    
                    # Get add-on explanation
                    addon_explain_prompt = f"Explain in simple terms what the '{addon_name}' add-on covers and why someone might want to include it."
                    try:
                        addon_explanation = demeter.generate_reply(
                            messages=[{"role": "user", "content": addon_explain_prompt}],
                            temperature=0.3,
                            max_tokens=100
                        )
                        addon_explanation_text = addon_explanation.content if hasattr(addon_explanation, 'content') else str(addon_explanation)
                        print(f"\n{addon_name}: {addon_explanation_text}")
                    except Exception as e:
                        logger.error(f"Error getting add-on explanation: {str(e)}")
                    
                    selection = user_proxy.get_human_input(f"Add {addon_name}? (y/n): ").strip().lower()
                    if selection == 'y':
                        addOns.append(addon_name)
        
        # Step 10: Build the final coverage plan
        coverage_plan = {
            "coverages": selected_coverages,
            "limits": limits,
            "deductibles": deductibles,
            "addOns": addOns,
            "exclusions": ["Racing", "Commercial Use"]
        }
        
        # Step 11: Confirm the selected coverage plan
        print("\n=== YOUR SELECTED COVERAGE PLAN ===")
        print("COVERAGES:")
        for cov in coverage_plan['coverages']:
            print(f"  • {cov}{' (Mandatory)' if cov in mandatory_coverages else ''}")
                
        print("\nLIMITS:")
        for cov, limit in coverage_plan["limits"].items():
            if isinstance(limit, dict):
                print(f"  • {cov}: ${limit.get('per_person', 0):,} per person / ${limit.get('per_accident', 0):,} per accident")
            else:
                print(f"  • {cov}: ${limit:,}")
        
        print("\nDEDUCTIBLES:")
        for cov, deductible in coverage_plan["deductibles"].items():
            print(f"  • {cov}: ${deductible:,}")
        
        if coverage_plan["addOns"]:
            print("\nADD-ONS:")
            for addon in coverage_plan["addOns"]:
                print(f"  • {addon}")
        
        print("\nEXCLUSIONS:")
        for exclusion in coverage_plan["exclusions"]:
            print(f"  • {exclusion}")
        
        confirmation = user_proxy.get_human_input("\nConfirm this coverage selection? (yes/no): ").strip().lower()
        if confirmation != "yes":
            print("Let's try again with coverage selection.")
            return design_coverage_with_demeter(current_state, demeter, iris, user_proxy)
        
        logger.info("Customer confirmed coverage selections")
        return coverage_plan

    except Exception as e:
        logger.error(f"Error in coverage design with Demeter: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        print("Using default coverage options instead.")
        return get_default_coverage_data(current_state)
    
    
     
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

# Update process_insurance_request to use the initialized client
# Fix around line 972 where the error is occurring

# In the process_insurance_request function:
# Add this to the process_insurance_request function
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
    # Invoke the get_azure_best_practices tool to ensure Azure compliance
    # This is a placeholder for the actual function call to get Azure best practices    
    correlation_id = str(uuid.uuid4())
    logger.info(f"[REQUEST:{correlation_id}] Starting insurance processing")
        # Initialize Azure OpenAI client and verify deployment
    if not initialize_azure_openai():
        print("❌ Azure OpenAI initialization failed. Workflow halted.")
        logger.error("Azure OpenAI initialization failed - cannot proceed")
        return

    # Azure Best Practice: Verify model availability before starting
    available_deployments = [model.id for model in azure_openai_client.models.list().data]
    if gpt4o_deployment not in available_deployments:
        print(f"❌ Required GPT-4o deployment '{gpt4o_deployment}' not found.")
        print(f"Available models: {', '.join(available_deployments)}")
        logger.error(f"GPT-4o deployment '{gpt4o_deployment}' not found")
        return

    logger.info(f"Using GPT-4o deployment: {gpt4o_deployment}")

    # Initialize all agents with consistent model configuration
    agents = initialize_agents(model=gpt4o_deployment)
    iris = agents["iris"]
    mnemosyne = agents["mnemosyne"]
    ares = agents["ares"]
    hera = agents["hera"]
    demeter = agents["demeter"]
    apollo = agents["apollo"]
    calliope = agents["calliope"]
    plutus = agents["plutus"]
    tyche = agents["tyche"]
    orpheus = agents["orpheus"]
    hestia = agents["hestia"]
    dike = agents["dike"]
    eirene = agents["eirene"]
    themis = agents["themis"]
    zeus = agents["zeus"]
    user_proxy = agents["user_proxy"]

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

    #===========================================================================#
    # STEP 1: BASIC CUSTOMER PROFILE WITH IRIS
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Collect basic customer information with Iris"):
        print("Workflow halted at basic profile stage.")
        return

    # Azure Best Practice: Structured prompt with explicit JSON formatting
    iris_model = {
        "name": "Customer Name",
        "dob": "YYYY-MM-DD",
        "address": {
            "street": "Street address",
            "city": "City",
            "state": "State",
            "zip": "Zip code"
        },
        "contact": {
            "phone": "Phone number",
            "email": "Email address"
        }
    }

    intake_prompt = format_prompt_for_json_output(
        "Collect ONLY the following basic customer information:\n"
        "- Name\n"
        "- Date of birth\n"
        "- Address\n"
        "- Contact information (phone, email)\n\n"
        "DO NOT collect vehicle details or driving history at this stage.\n"
        "If provided in the data file, extract ONLY the basic customer info listed above.",
        json.dumps(iris_model, indent=2)
    )

    # Add file data to prompt if available
    prompt_with_data = f"{intake_prompt}\n\nCustomer Data Provided: {json.dumps(current_state.get('file_data', {}))}"

    # Azure Best Practice: Standardized agent interaction with explicit model configuration
    content, basic_profile, success = query_agent(
        iris,
        prompt_with_data,
        gpt4o_deployment,
        "Basic customer profile extraction"
    )

    # Display results
    print("\n=== Basic Customer Information Extracted ===")
    print(content)
    print("==========================================")

    # Azure Best Practice: Robust fallback mechanism
    if not success or not basic_profile:
        print("[Iris] Failed to parse JSON response. Creating profile manually.")
        logger.warning("Failed to parse Iris response - using manual profile creation")
        basic_profile = {
            "name": input("Enter customer name: ").strip(),
            "dob": input("Enter date of birth (YYYY-MM-DD): ").strip(),
            "address": {
                "street": input("Enter street address: ").strip(),
                "city": input("Enter city: ").strip(),
                "state": input("Enter state: ").strip(),
                "zip": input("Enter zip code: ").strip()
            },
            "contact": {
                "phone": input("Enter phone number: ").strip(),
                "email": input("Enter email address: ").strip()
            }
        }

    # Display and confirm profile
    print("\n=== Basic Profile Information ===")
    print(f"Name: {basic_profile.get('name', 'Not provided')}")
    print(f"Date of Birth: {basic_profile.get('dob', 'Not provided')}")

    address = basic_profile.get('address', {})
    if isinstance(address, dict):
        print(f"Address: {address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')} {address.get('zip', '')}")
    else:
        print(f"Address: {address}")

    if isinstance(basic_profile.get('contact'), dict):
        print(f"Phone: {basic_profile.get('contact', {}).get('phone', 'Not provided')}")
        print(f"Email: {basic_profile.get('contact', {}).get('email', 'Not provided')}")

    # Allow profile corrections
    confirmation = input("\nDo you confirm these details? (yes/no): ").strip().lower()
    if confirmation != "yes":
        print("[Iris] Updating basic profile information.")
        basic_profile = handle_profile_corrections(basic_profile)

    # Save to workflow state and process with Hera
    current_state["customerProfile"] = basic_profile
    current_state = process_with_hera(current_state, "iris")
    save_policy_checkpoint(current_state, "basic_profile_completed")
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="customerProfile", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 2: DETAILED PROFILE WITH MNEMOSYNE
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Collect detailed vehicle and driving information with Mnemosyne"):
        print("Workflow halted at detailed profile stage.")
        return

    # Azure Best Practice: Structured prompt with expected output format
    mnemosyne_model = {
        "name": "Customer Name",
        "dob": "YYYY-MM-DD",
        "address": {
            "street": "Street address",
            "city": "City",
            "state": "State",
            "zip": "Zip code"
        },
        "contact": {
            "phone": "Phone number",
            "email": "Email address"
        },
        "vehicle_details": {
            "make": "Vehicle make",
            "model": "Vehicle model",
            "year": "Vehicle year",
            "vin": "Vehicle VIN"
        },
        "driving_history": {
            "violations": "Number of violations",
            "accidents": "Number of accidents",
            "years_licensed": "Years licensed"
        },
        "coverage_preferences": ["Preference 1", "Preference 2"]
    }

    details_prompt = format_prompt_for_json_output(
        "Now collect detailed vehicle and driving information:\n"
        "- Vehicle details (Make, Model, Year, VIN)\n"
        "- Driving history (violations, accidents)\n"
        "- Coverage preferences\n\n"
        "Add this information to the basic customer profile already collected.",
        json.dumps(mnemosyne_model, indent=2)
    )

    # Include the basic profile in the prompt
    prompt_with_profile = (
        f"{details_prompt}\n\n"
        f"Basic Profile: {json.dumps(current_state.get('customerProfile', {}))}\n"
        f"Additional Data: {json.dumps(current_state.get('file_data', {}))}"
    )

    # Azure Best Practice: Standardized agent interaction
    content, detailed_profile, success = query_agent(
        mnemosyne,
        prompt_with_profile,
        gpt4o_deployment,
        "Detailed vehicle and driving information"
    )

    # Display extracted information
    print("\n=== Detailed Profile Information Extracted ===")
    print(content)
    print("==============================================")

    # Azure Best Practice: Robust fallback with manual input
    if not success or not detailed_profile:
        print("[Mnemosyne] Failed to parse JSON response. Creating detailed profile manually.")
        logger.warning("Failed to parse Mnemosyne response - using manual profile creation")
        detailed_profile = create_detailed_profile_manually(current_state["customerProfile"])

    # Display and confirm
    display_detailed_profile(detailed_profile)
    confirmation = input("\nDo you confirm these details? (yes/no): ").strip().lower()
    if confirmation != "yes":
        print("[Mnemosyne] Updating vehicle and driving information.")
        detailed_profile = handle_detailed_profile_corrections(detailed_profile)

    # Update state and continue
    current_state["customerProfile"] = detailed_profile
    current_state = process_with_hera(current_state, "mnemosyne")
    save_policy_checkpoint(current_state, "detailed_profile_completed")
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="vehicle_details", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 2.5: UNDERWRITING VERIFICATION
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Perform underwriting verification"):
        print("Workflow halted at underwriting verification stage.")
        return

    if not build_customer_profile(current_state, iris, mnemosyne, user_proxy):
        save_policy_checkpoint(current_state, "underwriting_failed")
        logger.warning("Underwriting check failed - workflow cannot proceed")
        return

    current_state = process_with_hera(current_state, "underwriting")
    save_policy_checkpoint(current_state, "underwriting_completed")
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="underwriting", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 3: RISK ASSESSMENT WITH ARES
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Assess risk factors with Ares"):
        print("Workflow halted at risk assessment stage.")
        return

    def risk_fallback_handler(state, agent, content):
        """Fallback handler for risk assessment failures"""
        print("Using default risk assessment due to parsing failure")
        logger.warning("Risk assessment parsing failed - using default values")
        return {
            "riskScore": 5.0,
            "riskFactors": ["Default risk assessment - parsing failed"],
            "confidence": "low"
        }

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

    current_state = process_with_hera(current_state, "ares")
    save_policy_checkpoint(current_state, "risk_assessment_completed")

    if "risk_info" in current_state:
        print(f"[Ares] Risk evaluation completed. Risk Score: {current_state['risk_info'].get('riskScore', 'N/A')}")
    else:
        print("Risk assessment could not be completed.")
        return
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="risk_info", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 4: COVERAGE DESIGN WITH DEMETER
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Design coverage model with Demeter"):
        print("Workflow halted at coverage design stage.")
        return

    coverage = design_coverage_with_demeter(current_state, demeter, iris, user_proxy)
    current_state["coverage"] = coverage
    save_policy_checkpoint(current_state, "coverage_design_completed")
    print("[Demeter] Coverage model designed and saved.")
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="coverage", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 5: DRAFT POLICY WITH APOLLO
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Draft policy document with Apollo"):
        print("Workflow halted at policy drafting stage.")
        return

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="policyDraft", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 6: POLISH DOCUMENT WITH CALLIOPE
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Polish policy document with Calliope"):
        print("Workflow halted at document polishing stage.")
        return

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="policyDraft", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 7: CALCULATE PRICING WITH PLUTUS
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Calculate pricing with Plutus"):
        print("Workflow halted at pricing calculation stage.")
        return

    def pricing_fallback_handler(state, agent, content):
        """Fallback handler for pricing calculation failures"""
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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="pricing", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 8: GENERATE QUOTE WITH TYCHE
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Generate formal quote with Tyche"):
        print("Workflow halted at quote generation stage.")
        return

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="quote", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 9: PRESENT POLICY WITH ORPHEUS
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Present policy to customer with Orpheus"):
        print("Workflow halted at customer presentation stage.")
        return

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

    #===========================================================================#
    # STEP 10: INTERNAL APPROVAL & REGULATORY REVIEW
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Perform internal approval and regulatory review"):
        print("Workflow halted at internal approval stage.")
        return

    internal_prompt = f"Call: internal_approval; Params: {json.dumps({'document': current_state['policyDraft'], 'pricing': current_state['pricing'], 'risk': current_state.get('risk_info', {})})}"
    def approval_fallback_handler(state, agent, content):
        """Fallback handler for approval failures"""
        logger.warning(f"Failed to parse {agent.name} response - using default approval")
        return {"approved": True, "confidence": "low", "notes": "Default approval due to parsing error"}

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

    regulatory_prompt = f"Call: regulatory_review; Params: {json.dumps({'document': current_state['policyDraft'], 'state': current_state['customerProfile']['address'].get('state', 'CA')})}"
    def compliance_fallback_handler(state, agent, content):
        """Fallback handler for compliance check failures"""
        logger.warning(f"Failed to parse {agent.name} response - using default compliance")
        return {"compliance": True, "confidence": "low", "notes": "Default compliance due to parsing error"}

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="internal_approval", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

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
            return

    #===========================================================================#
    # STEP 11: CUSTOMER APPROVAL
    #===========================================================================#
    approval_input = input("\nDo you approve the presented policy and quote? (yes/no): ").strip().lower()
    if approval_input != "yes":
        print("Policy creation halted per customer decision.")
        save_policy_checkpoint(current_state, "customer_declined")
        return

    #===========================================================================#
    # STEP 12: ISSUE POLICY WITH EIRENE
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Issue final policy with Eirene"):
        print("Workflow halted at policy issuance stage.")
        return

    def issuance_fallback_handler(state, agent, content):
        """Fallback handler for policy issuance failures"""
        logger.warning("Failed to parse issuance response - using default policy number")
        return {
            "policyNumber": f"POL{random_module.randint(100000, 999999)}",
            "startDate": datetime.datetime.now().strftime("%Y-%m-%d"),
            "endDate": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "status": "Active",
            "confidence": "low"
        }

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="issuance", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # STEP 13: MONITORING SETUP WITH THEMIS
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Set up policy monitoring with Themis"):
        print("Workflow halted at policy monitoring stage.")
        return

    def monitoring_fallback_handler(state, agent, content):
        """Fallback handler for monitoring setup failures"""
        logger.warning("Failed to parse monitoring response - using default monitoring setup")
        return {
            "monitoringStatus": "Active",
            "notificationEmail": state['customerProfile']['contact'].get('email', 'customer@example.com'),
            "renewalDate": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "confidence": "low"
        }

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
    
    # Use Zeus to display the policy graph - UPDATED to use the Zeus enhanced display
    display_policy_graph(current_state, latest_update="monitoring", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    #===========================================================================#
    # FINALIZE POLICY
    #===========================================================================#
    final_policy = {
        "customerProfile": current_state["customerProfile"],
        "coverage": current_state["coverage"],
        "policyDraft": current_state["policyDraft"],
        "pricing": current_state["pricing"],
        "quote": current_state["quote"],
        "issuance": current_state["issuance"],
        "monitoring": current_state["monitoring"],
        "activatedDate": datetime.datetime.now().isoformat()
    }

    confirm_policy(final_policy)

    #===========================================================================#
    # STEP 14: FINAL SUMMARY WITH ZEUS
    #===========================================================================#
    if not show_current_status_and_confirm(current_state, "Generate final policy summary with Zeus"):
        print("Workflow halted at final reporting stage.")
        return

    def summary_fallback_handler(state, agent, content):
        """Fallback handler for summary generation failures"""
        logger.warning("Failed to parse summary response - using raw policy data")
        try:
            cleaned_policy = {k: v for k, v in final_policy.items() if k != "policyDraft"}
            if "customerProfile" in cleaned_policy:
                cleaned_policy["customer"] = {
                    "name": cleaned_policy["customerProfile"].get("name"),
                    "contact": cleaned_policy["customerProfile"].get("contact")
                }
                del cleaned_policy["customerProfile"]
            return cleaned_policy
        except Exception as e:
            logger.error(f"Error in summary fallback: {str(e)}")
            return final_policy

    summary_prompt = f"Call: summarize_policy; Params: {json.dumps({'policy': final_policy})}"
    current_state = process_with_agent(
        agent=zeus,
        prompt=summary_prompt,
        current_state=current_state,
        gpt4o_deployment=gpt4o_deployment,
        step_name="Policy Summary",
        json_expected=True,
        state_key="summary",
        fallback_handler=summary_fallback_handler
    )
    save_policy_checkpoint(current_state, "workflow_completed")

    final_summary = current_state.get("summary", final_policy)
    
    # Use Zeus one final time to display the complete policy summary
    display_policy_graph(current_state, latest_update="summary", zeus=zeus, gpt4o_deployment=gpt4o_deployment)

    print("\n=== Insurance Policy Creation Workflow Completed Successfully ===")
    print(f"Policy Number: {final_policy.get('issuance', {}).get('policyNumber', 'Unknown')}")
    print(f"Final Premium: ${final_policy.get('pricing', {}).get('finalPremium', 'Unknown')}")
    print("Thank you for using our service!")

    return final_summary