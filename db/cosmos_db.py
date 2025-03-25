from azure.identity import AzureCliCredential
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import datetime
import os
import random
# To this:
from config import COSMOS_ENDPOINT, DATABASE_NAME, DRAFTS_CONTAINER, ISSUED_CONTAINER
# In-memory storage as fallback
in_memory_db = {
    "policies": [],
    "quotes": []
}
use_cosmos = False  # Will be set to True if connection succeeds

# Global container references
drafts_container = None
issued_container = None

def init_cosmos_db():
    """Initialize connection to Cosmos DB"""
    global use_cosmos, drafts_container, issued_container
    
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