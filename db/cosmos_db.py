import datetime
import os
import random
import json
import logging
from azure.cosmos import exceptions
# Import the connection manager
from .cosmos_connection import CosmosConnectionManager

logger = logging.getLogger("cosmos_db")

# Only declare these once
use_cosmos = False
autopm_container = None
drafts_container = None
issued_container = None

# Container names - define only once
AUTOPM_CONTAINER_NAME = "autopm"
DRAFTS_CONTAINER_NAME = "PolicyDrafts"
ISSUED_CONTAINER_NAME = "PolicyIssued"

# Database settings
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "insurance")

# In-memory storage as fallback
in_memory_db = {
    "policies": [],
    "quotes": [],
    "underwriting_questions": [  # Fallback underwriting questions
        {
            "id": "UW001",
            "text": "Has the driver had a license for at least 3 years?",
            "mandatory": True,
            "type": "underwriting_question"
        },
        {
            "id": "UW002",
            "text": "Is the vehicle free of modifications not approved by the manufacturer?",
            "mandatory": True,
            "type": "underwriting_question"
        },
        {
            "id": "UW003",
            "text": "Has the driver been free of DUI convictions in the past 5 years?",
            "mandatory": True,
            "type": "underwriting_question"
        }
    ]
}

def init_cosmos_db():
    """Initialize Cosmos DB connection using the connection manager"""
    try:
        # Initialize the connection manager
        connection_manager = CosmosConnectionManager.get_instance()
        client = connection_manager.get_client()
        
        # Try to initialize the specific containers we need
        global autopm_container, drafts_container, issued_container, use_cosmos
        
        try:
            autopm_container = connection_manager.get_container(AUTOPM_CONTAINER_NAME)
        except Exception as e:
            logger.warning(f"Could not connect to {AUTOPM_CONTAINER_NAME}: {str(e)}")
            autopm_container = None
            
        try:
            drafts_container = connection_manager.get_container(DRAFTS_CONTAINER_NAME)
        except Exception as e:
            logger.warning(f"Could not connect to {DRAFTS_CONTAINER_NAME}: {str(e)}")
            drafts_container = None
            
        try:
            issued_container = connection_manager.get_container(ISSUED_CONTAINER_NAME)
        except Exception as e:
            logger.warning(f"Could not connect to {ISSUED_CONTAINER_NAME}: {str(e)}")
            issued_container = None
            
        # Set cosmos flag if any container is available
        use_cosmos = any([autopm_container, drafts_container, issued_container])
        
        if use_cosmos:
            logger.info("Connected to Cosmos DB successfully.")
            print("Connected to Cosmos DB successfully.")
        else:
            logger.warning("Could not connect to any containers.")
            print("Could not connect to any containers.")
            
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Cosmos DB: {str(e)}")
        use_cosmos = False
        return None

def get_container_client(container_name):
    """
    Get a container client for the specified container using the connection manager.
    Following Azure best practices for connection management.
    
    Args:
        container_name (str): The name of the container to access
        
    Returns:
        ContainerProxy: The container client
    """
    try:
        connection_manager = CosmosConnectionManager.get_instance()
        return connection_manager.get_container(container_name)
    except Exception as e:
        logger.error(f"Failed to get container {container_name}: {str(e)}")
        return None

# Rest of your functions (get_next_number, save_policy_draft, etc.)
# ...rest of the file continues...
#            
def get_next_number(field_name, container_ref=None, increment=10, default_start=100000):
    """Get the next sequential number for a field"""
    if use_cosmos and container_ref:
        try:
            query = f"SELECT VALUE MAX(c.{field_name}) FROM c"
            items = list(container_ref.query_items(query=query, enable_cross_partition_query=True))
            max_number = items[0] if items and items[0] else 0
            return int(max_number) + increment if max_number else default_start
        except Exception as e:
            print(f"Error querying Cosmos DB: {e}")
    
    if field_name == "quoteNumber":
        quotes = in_memory_db["quotes"]
        max_number = max([q.get("quoteNumber", 0) for q in quotes]) if quotes else 0
    else:
        policies = in_memory_db["policies"]
        max_number = max([int(p.get("policyNumber", "0").replace("MV", "")) for p in policies]) if policies else 0
    
    return max_number + increment if max_number else default_start

def save_policy_draft(policy):
    """Save a policy draft to the database"""
    policy["status"] = "Draft"
    policy["quoteNumber"] = get_next_number("quoteNumber", drafts_container)
    policy["id"] = f"QUOTE{policy['quoteNumber']}"
    
    if use_cosmos:
        try:
            drafts_container.upsert_item(policy)
            print(f"Policy draft saved to Cosmos DB with Quote Number: {policy['quoteNumber']}")
            return policy
        except Exception as e:
            print(f"Error saving to Cosmos DB: {e}")
            print("Falling back to in-memory storage")
    
    in_memory_db["quotes"].append(policy)
    print(f"Policy draft saved in memory with Quote Number: {policy['quoteNumber']}")
    return policy

def confirm_policy(policy):
    """Confirm and activate a policy"""
    policy["status"] = "Active"
    if "policyNumber" not in policy:
        quote_number = policy.get("quoteNumber", get_next_number("quoteNumber", drafts_container))
        policy["policyNumber"] = f"MV{quote_number}"
    policy["id"] = policy["policyNumber"]
    
    if use_cosmos:
        try:
            issued_container.upsert_item(policy)
            print(f"Policy confirmed and activated in PolicyIssued container with Policy Number: {policy['policyNumber']}")
            return policy
        except Exception as e:
            print(f"Error saving to Cosmos DB PolicyIssued container: {e}")
            print("Falling back to in-memory storage")
    
    in_memory_db["policies"].append(policy)
    print(f"Policy confirmed and activated in memory with Policy Number: {policy['policyNumber']}")
    return policy

def save_policy_checkpoint(current_state, stage):
    """
    Save a checkpoint of the current policy creation process.
    This ensures data isn't lost if the customer logs out mid-process.
    
    Args:
        current_state (dict): Current state from the workflow.
        stage (str): The name of the completed stage.
    """
    if "customerProfile" not in current_state:
        return
    
    policy_draft = {
        "customerProfile": current_state.get("customerProfile", {}),
        "stage": stage,
        "lastUpdated": datetime.datetime.utcnow().isoformat(),
        "status": "InProgress"
    }
    
    if "risk_info" in current_state:
        policy_draft["riskInfo"] = current_state["risk_info"]
    if "coverage" in current_state:
        policy_draft["coverage"] = current_state["coverage"]
    if "policyDraft" in current_state:
        policy_draft["policyDocument"] = current_state["policyDraft"]
    if "pricing" in current_state:
        policy_draft["pricing"] = current_state["pricing"]
    if "quote" in current_state:
        policy_draft["quoteDetails"] = current_state["quote"]
    
    quote_number = current_state.get("quoteNumber")
    if quote_number:
        policy_draft["quoteNumber"] = quote_number
        policy_draft["id"] = f"QUOTE{quote_number}"
    
    saved_policy = save_policy_draft(policy_draft)
    if "quoteNumber" not in current_state and "quoteNumber" in saved_policy:
        current_state["quoteNumber"] = saved_policy["quoteNumber"]
    
    print(f"\n✅ Progress saved with Quote Number: {saved_policy.get('quoteNumber', 'N/A')}")
    print("   You can resume this quote later using this number.\n")

# New functions for underwriting questions
def query_cosmos(container_ref, query):
    """Query the specified Cosmos DB container with the given query"""
    if use_cosmos and container_ref:
        try:
            items = list(container_ref.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return {"status": "success", "data": items}
        except exceptions.CosmosHttpResponseError as e:
            print(f"Error querying Cosmos DB: {e}")
    
    # Fallback for autopm container queries
    if "underwriting_question" in query:
        return {"status": "success", "data": in_memory_db["underwriting_questions"]}
    
    return {"status": "error", "message": "Cosmos DB unavailable and no fallback data"}

def get_underwriting_questions():
    """Retrieve all underwriting questions from the autopm container"""
    query = "SELECT * FROM c WHERE c.type = 'underwriting_question'"
    result = query_cosmos(autopm_container, query)
    
    if result["status"] == "success":
        return result["data"]
    else:
        print(f"Warning: {result['message']}")
        return in_memory_db["underwriting_questions"]

def get_mandatory_questions():
    """Retrieve Pre-Qualification questions from the autopm container"""
    # Simplified query - get the whole document first
    query = "SELECT * FROM c"
    
    # For debugging
    print("Executing simplified query to get product model document...")
    
    result = query_cosmos(autopm_container, query)
    
    if result["status"] == "success":
        # Extract pre-qualification questions from the results manually
        pre_qual_questions = []
        
        for doc in result["data"]:
            if "productModel" in doc and "questions" in doc["productModel"]:
                for question in doc["productModel"]["questions"]:
                    if question.get("questionType") == "Pre-Qualification":
                        # Format question for our needs
                        pre_qual_questions.append({
                            "id": question.get("requirementId"),
                            "text": question.get("question"),
                            "action": question.get("action"),
                            "explanation": question.get("explanation"),
                            "order": question.get("order", 999),
                            "possibleAnswers": question.get("possibleAnswers", ["Yes", "No"])
                        })
        
        # Sort questions by order field
        pre_qual_questions.sort(key=lambda q: q.get("order", 999))
        
        print(f"Found {len(pre_qual_questions)} pre-qualification questions")
        return pre_qual_questions
    else:
        print(f"Warning: {result.get('message', 'Unknown error')}")
        print("Using fallback underwriting questions from memory")
        return [q for q in in_memory_db["underwriting_questions"] if q.get("mandatory", False)]

def get_coverage_with_demeter(demeter_agent, raw_json=None):
    """
    Use Demeter's language skills to extract and enhance coverage options.
    Following Azure AI best practices for robust AI integration.
    
    Args:
        demeter_agent: The agent to use for coverage extraction
        raw_json: Optional pre-loaded JSON data
        
    Returns:
        dict: Extracted coverage information, or None if extraction fails
    """
    # Get the raw JSON if not provided
    if not raw_json:
        query = "SELECT * FROM c"
        result = query_cosmos(autopm_container, query)
        if result["status"] != "success":
            logger.error("Failed to get raw data from Cosmos DB")
            print("Failed to get raw data from Cosmos DB")
            return None
        raw_json = result["data"]
    
    # Format the prompt with clear instructions
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
    
    IMPORTANT: Your response should contain ONLY the JSON with no additional text before or after. The JSON must be valid and properly formatted.
    
    JSON DATA TO ANALYZE:
    """ + json.dumps(raw_json, indent=2)
    
    # Send to Demeter agent with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Demeter agent (attempt {attempt + 1}/{max_retries})")
            response = demeter_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
            
            # Extract the content from the response object
            if isinstance(response, str):
                response_content = response
            else:
                response_content = getattr(response, 'content', str(response))
            
            # Use the robust extraction function
            coverage_data = extract_json_from_llm_response(response_content)
            
            if coverage_data and isinstance(coverage_data, dict) and "coverageCategories" in coverage_data:
                logger.info(f"Demeter successfully extracted coverage options")
                print(f"Demeter successfully extracted coverage options")
                return coverage_data
            else:
                logger.warning(f"Invalid response format from Demeter (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    logger.error("Failed to extract valid JSON after all retry attempts")
                    print("Failed to extract valid JSON from Demeter's response")
                    # Log a sample of the response for debugging
                    sample = response_content[:200] + "..." if len(response_content) > 200 else response_content
                    logger.error(f"Response sample: {sample}")
        except Exception as e:
            logger.error(f"Error using Demeter for coverage extraction (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Error using Demeter for coverage extraction: {str(e)}")
                return None
    
    # Fallback to default coverage if extraction fails
    logger.warning("Using fallback coverage due to extraction failure")
    return get_default_coverage_data()

def get_default_coverage_data():
    """Return default coverage data when extraction fails"""
    return {
        "coverageCategories": [
            {
                "name": "Liability Coverages",
                "description": "Protects you financially when you're responsible for injury to others or damage to their property",
                "coverages": [
                    {
                        "name": "Bodily Injury Liability",
                        "coverageCategory": "Liability",
                        "mandatory": True,
                        "explanation": "Covers medical expenses, lost wages, and legal fees if you injure someone in an accident",
                        "termName": "Limit",
                        "modelType": "Limit",
                        "options": [
                            {
                                "label": "25/50",
                                "display": "$25,000 per person / $50,000 per accident",
                                "min": 25000,
                                "max": 50000,
                                "explanation": "Minimum coverage in most states"
                            },
                            {
                                "label": "50/100",
                                "display": "$50,000 per person / $100,000 per accident",
                                "min": 50000,
                                "max": 100000,
                                "explanation": "Recommended standard coverage"
                            }
                        ]
                    }
                ]
            }
        ]
    }


def extract_json_from_llm_response(response_content):
    """
    Extract JSON from an LLM response using multiple strategies.
    Following Azure best practices for robust parsing of AI responses.
    
    Args:
        response_content (str): The raw response content from the LLM
        
    Returns:
        dict or list: The extracted JSON object, or None if extraction fails
    """
    import re
    import json
    
    # Strategy 1: Look for code fence markers
    json_regex = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_regex, response_content)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Look for JSON objects enclosed in brackets
    try:
        # Try to find JSON object pattern
        object_match = re.search(r"\{[\s\S]*\}", response_content)
        if object_match:
            return json.loads(object_match.group(0))
        
        # Try to find JSON array pattern
        array_match = re.search(r"\[[\s\S]*\]", response_content)
        if array_match:
            return json.loads(array_match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try parsing the entire response as JSON
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        pass
    
    # If all strategies fail, return None
    return None
def get_questions_with_mnemosyne(mnemosyne_agent, raw_json=None):
    """
    Use Mnemosyne's language skills to extract and enhance underwriting questions.
    Following Azure AI best practices for robust AI integration.
    
    Args:
        mnemosyne_agent: The agent to use for question extraction
        raw_json: Optional pre-loaded JSON data
        
    Returns:
        list: Extracted and enhanced questions, or None if extraction fails
    """
    # Get the raw JSON if not provided
    if not raw_json:
        query = "SELECT * FROM c"
        result = query_cosmos(autopm_container, query)
        if result["status"] != "success":
            logger.error("Failed to get raw data from Cosmos DB")
            print("Failed to get raw data from Cosmos DB")
            return None
        raw_json = result["data"]
    
    # Format the prompt with clear instructions
    prompt = """
    You're analyzing a product model JSON for an insurance application. Extract all Pre-Qualification questions.
    
    For each question:
    1. Extract the exact question text
    2. Extract the requirementId as the question ID
    3. Extract the order, action, and explanation
    4. Create an enhanced explanation that makes the question clearer for the customer
    5. Format each question in easily parseable JSON
    
    Return the questions in order by their 'order' field, in this exact format:
    ```json
    [
      {
        "id": "SC-AUT-082-003",
        "text": "Has any policy or coverage ever been declined...",
        "order": 1,
        "action": "Decline",
        "explanation": "Original explanation from product model",
        "enhanced_explanation": "Your enhanced explanation making this clearer", 
        "possibleAnswers": ["Yes", "No"]
      },
      ...more questions...
    ]
    ```
    
    IMPORTANT: Your response should contain ONLY the JSON array with no additional text before or after. The JSON must be valid and properly formatted.
    
    JSON DATA TO ANALYZE:
    """ + json.dumps(raw_json, indent=2)
    
    # Send to Mnemosyne agent with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending request to Mnemosyne agent (attempt {attempt + 1}/{max_retries})")
            response = mnemosyne_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
            
            # Extract the content from the response object
            if isinstance(response, str):
                response_content = response
            else:
                response_content = getattr(response, 'content', str(response))
            
            # Use the robust extraction function
            mnemosyne_questions = extract_json_from_llm_response(response_content)
            
            if mnemosyne_questions and isinstance(mnemosyne_questions, list):
                logger.info(f"Mnemosyne successfully extracted {len(mnemosyne_questions)} questions")
                print(f"Mnemosyne successfully extracted {len(mnemosyne_questions)} questions")
                return mnemosyne_questions
            else:
                logger.warning(f"Invalid response format from Mnemosyne (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    logger.error("Failed to extract valid JSON after all retry attempts")
                    print("Failed to extract valid JSON from Mnemosyne's response")
                    # Log a sample of the response for debugging
                    sample = response_content[:200] + "..." if len(response_content) > 200 else response_content
                    logger.error(f"Response sample: {sample}")
        except Exception as e:
            logger.error(f"Error using Mnemosyne for question extraction (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Error using Mnemosyne for question extraction: {str(e)}")
                return None
    
    # Fallback to default questions if extraction fails
    logger.warning("Using fallback questions due to extraction failure")
    return get_default_questions()

def get_default_questions():
    """Return default questions when extraction fails"""
    return [
        {
            "id": "UW001",
            "text": "Has the driver had a license for at least 3 years?",
            "order": 1,
            "action": "Review",
            "explanation": "Drivers with less experience may present higher risk",
            "enhanced_explanation": "Having at least 3 years of driving experience helps establish a driving history and may qualify you for better rates.",
            "possibleAnswers": ["Yes", "No"]
        },
        {
            "id": "UW002",
            "text": "Is the vehicle free of modifications not approved by the manufacturer?",
            "order": 2,
            "action": "Review",
            "explanation": "Modified vehicles may have different risk profiles",
            "enhanced_explanation": "Vehicle modifications can affect performance, safety, and repair costs. Factory-standard vehicles are easier to insure.",
            "possibleAnswers": ["Yes", "No"]
        },
        {
            "id": "UW003", 
            "text": "Has the driver been free of DUI convictions in the past 5 years?",
            "order": 3,
            "action": "Decline",
            "explanation": "DUI convictions indicate higher risk",
            "enhanced_explanation": "A history free of driving under the influence convictions is important for insurance eligibility.",
            "possibleAnswers": ["Yes", "No"]
        }
    ]

def save_with_eligibility(policy_data, eligibility, reason=None):
    """Save policy data with eligibility information"""
    # Add eligibility information to policy data
    policy_data["eligibility"] = eligibility
    if reason and not eligibility:
        policy_data["eligibilityReason"] = reason
    
    # Set status based on eligibility
    policy_data["status"] = "Eligible" if eligibility else "Ineligible"
    
    # Save to database
    return save_policy_draft(policy_data)

def save_underwriting_responses(current_state, responses, eligibility, reason=None):
    """Save underwriting responses to the current state"""
    if "customerProfile" not in current_state:
        current_state["customerProfile"] = {}
    
    # Add underwriting data to customer profile
    current_state["customerProfile"]["underwriting"] = responses
    current_state["customerProfile"]["eligibility"] = eligibility
    if reason:
        current_state["customerProfile"]["eligibilityReason"] = reason
    
    # Save to database
    policy_data = {
        "customerProfile": current_state["customerProfile"],
        "status": "Eligible" if eligibility else "Ineligible",
        "lastUpdated": datetime.datetime.utcnow().isoformat()
    }
    
    if not eligibility and reason:
        policy_data["ineligibilityReason"] = reason
        
    # Use existing policy draft save function
    quote_number = current_state.get("quoteNumber")
    if quote_number:
        policy_data["quoteNumber"] = quote_number
    
    save_policy_draft(policy_data)
    
    return current_state