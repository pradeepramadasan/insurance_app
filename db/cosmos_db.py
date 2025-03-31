from azure.identity import AzureCliCredential
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import datetime
import os
import random
import json
# To this:
from config import COSMOS_ENDPOINT, DATABASE_NAME, DRAFTS_CONTAINER, ISSUED_CONTAINER

# Add autopm container name
AUTOPM_CONTAINER = "autopm"

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
use_cosmos = False  # Will be set to True if connection succeeds

# Global container references
drafts_container = None
issued_container = None
autopm_container = None  # Add reference to autopm container

def init_cosmos_db():
    """Initialize connection to Cosmos DB"""
    global use_cosmos, drafts_container, issued_container, autopm_container
    
    try:
        if not COSMOS_ENDPOINT:
            raise ValueError("COSMOS_ENDPOINT not set in environment variables")
        
        print("Attempting to connect to Cosmos DB...")
        credential = AzureCliCredential()
        cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        
        # Test the connection by listing databases
        list(cosmos_client.list_databases())
        print("Successfully authenticated with Cosmos DB")
        
        # Create database and containers if they don't exist
        database = cosmos_client.create_database_if_not_exists(id=DATABASE_NAME)
        
        drafts_container = database.create_container_if_not_exists(
            id=DRAFTS_CONTAINER,
            partition_key=PartitionKey(path="/policyNumber")
        )
        
        issued_container = database.create_container_if_not_exists(
            id=ISSUED_CONTAINER,
            partition_key=PartitionKey(path="/policyNumber")
        )
        
        # Add reference to autopm container (don't create if it doesn't exist)
        try:
            autopm_container = database.get_container_client(AUTOPM_CONTAINER)
            print(f"Connected to {AUTOPM_CONTAINER} container successfully.")
        except exceptions.CosmosResourceNotFoundError:
            print(f"Warning: {AUTOPM_CONTAINER} container not found. Using in-memory fallback for underwriting questions.")
        
        print("Connected to Cosmos DB containers successfully.")
        use_cosmos = True
        
    except exceptions.CosmosHttpResponseError as e:
        print(f"Cosmos DB Error: {e}")
        print("Firewall may be blocking your IP. Using in-memory storage instead.")
    except Exception as e:
        print(f"Failed to connect to Cosmos DB: {e}")
        print("Using in-memory storage instead.")

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
    """Use Demeter's language skills to extract and enhance coverage options"""
    
    # Get the raw JSON if not provided
    if not raw_json:
        query = "SELECT * FROM c"
        result = query_cosmos(autopm_container, query)
        if result["status"] != "success":
            print("Failed to get raw data from Cosmos DB")
            return None
        raw_json = result["data"]
    
    # Format the prompt for Demeter
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
    """ + json.dumps(raw_json, indent=2)
    
    # Send to Demeter agent
    try:
        print("Asking Demeter to analyze coverage options...")
        response = demeter_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        
        # Extract the JSON content from response
        if isinstance(response, str):
            response_content = response
        else:
            response_content = getattr(response, 'content', str(response))
            
        json_start = response_content.find("```json")
        json_end = response_content.rfind("```")
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_content[json_start + 7:json_end].strip()
            coverage_data = json.loads(json_content)
            print(f"Demeter successfully extracted coverage options")
            return coverage_data
        else:
            print("Failed to extract valid JSON from Demeter's response")
            return None
    except Exception as e:
        print(f"Error using Demeter for coverage extraction: {e}")
        return None   

def get_questions_with_mnemosyne(mnemosyne_agent, raw_json=None):
    """Use Mnemosyne's language skills to extract and enhance underwriting questions"""
    
    # Get the raw JSON if not provided
    if not raw_json:
        query = "SELECT * FROM c"
        result = query_cosmos(autopm_container, query)
        if result["status"] != "success":
            print("Failed to get raw data from Cosmos DB")
            return None
        raw_json = result["data"]
    
    # Format the prompt for Mnemosyne
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
    
    JSON DATA TO ANALYZE:
    """ + json.dumps(raw_json, indent=2)
    
    # Send to Mnemosyne agent
    try:
        response = mnemosyne_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        
        # Extract the JSON content from response
        if isinstance(response, str):
            response_content = response
        else:
            response_content = getattr(response, 'content', str(response))
            
        json_start = response_content.find("```json")
        json_end = response_content.rfind("```")
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_content[json_start + 7:json_end].strip()
            mnemosyne_questions = json.loads(json_content)
            print(f"Mnemosyne successfully extracted {len(mnemosyne_questions)} questions")
            return mnemosyne_questions
        else:
            print("Failed to extract valid JSON from Mnemosyne's response")
            return None
    except Exception as e:
        print(f"Error using Mnemosyne for question extraction: {e}")
        return None
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