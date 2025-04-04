import os
import argparse
import json
import numpy as np
import datetime
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from azure.identity import AzureCliCredential, ClientSecretCredential
from openai import AzureOpenAI

# ----------------------------
# Command Line Argument Handling
# ----------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Customer profile matching system")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to input customer data file (JSON)")
    parser.add_argument("--top", "-t", type=int, default=3, 
                        help="Number of top matches to return")
    parser.add_argument("--quiet", "-q", action="store_true", 
                        help="Suppress verbose output")
    return parser.parse_args()

# ----------------------------
# Initialize Global Configurations
# ----------------------------
def initialize_configs():
    # Load environment variables
    load_dotenv("x1.env")
    
    config = {}
    
    # Azure OpenAI configuration
    config["openai_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    config["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
    config["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    config["embedding_model"] = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large") 
    config["embedding_deployment"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    config["gpt4o_deployment"] = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
    
    # Cosmos DB configuration
    config["cosmos_endpoint"] = os.getenv("COSMOS_ENDPOINT")
    config["database_name"] = os.getenv("COSMOS_DATABASE_NAME", "insurance")
    config["tenant_id"] = os.getenv("AZURE_TENANT_ID")
    config["client_id"] = os.getenv("AZURE_CLIENT_ID")
    config["client_secret"] = os.getenv("AZURE_CLIENT_SECRET")
    
    return config

# ----------------------------
# Connect to Azure Services
# ----------------------------
def connect_to_services(config, verbose=True):
    connections = {}
    
    # Initialize Azure OpenAI client
    connections["openai"] = AzureOpenAI(
        api_key=config["openai_api_key"],
        api_version=config["api_version"],
        azure_endpoint=config["openai_endpoint"]
    )
    
    if verbose:
        print(f"Connected to Azure OpenAI at: {config['openai_endpoint']}")
        print(f"Using embedding deployment: {config['embedding_deployment']}")
        print(f"Using GPT-4o deployment: {config['gpt4o_deployment']}")
    
    # Connect to Cosmos DB (try AzureCliCredential first, then fall back to ClientSecretCredential)
    try:
        credential = AzureCliCredential()
        cosmos_client = CosmosClient(config["cosmos_endpoint"], credential=credential)
        
        # Test connection
        list(cosmos_client.list_databases())
        if verbose:
            print("Connected to Cosmos DB using AzureCliCredential")
            
    except Exception as e:
        if verbose:
            print(f"AzureCliCredential failed: {str(e)[:100]}...")
            
        try:
            credential = ClientSecretCredential(
                config["tenant_id"], 
                config["client_id"], 
                config["client_secret"]
            )
            cosmos_client = CosmosClient(config["cosmos_endpoint"], credential=credential)
            
            # Test connection
            list(cosmos_client.list_databases())
            if verbose:
                print("Connected to Cosmos DB using ClientSecretCredential")
                
        except Exception as e:
            print(f"Error connecting to Cosmos DB: {str(e)[:100]}...")
            raise
    
    connections["cosmos"] = cosmos_client
    
    # Get database and containers
    database = cosmos_client.get_database_client(config["database_name"])
    connections["database"] = database
    connections["segments_container"] = database.get_container_client("CustomerSegments")
    connections["policy_container"] = database.get_container_client("Policy")
    
    if verbose:
        print(f"Using database: {config['database_name']}")
    
    return connections

# ----------------------------
# Process Customer Input File
# ----------------------------
def process_input_file(file_path, verbose=True):
    """Process customer input file and convert to structured data"""
    if verbose:
        print(f"Processing input file: {file_path}")
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Try parsing as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: The file is not valid JSON. Will attempt to extract information using GPT-4o.")
            return {"raw_text": content}
            
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return {}

# ----------------------------
# Extract Customer Data Fields
# ----------------------------
def extract_customer_fields(customer_data, openai_client, gpt4o_deployment, verbose=True):
    """Extract standardized fields from customer data using GPT-4o"""
    # If already structured properly, just return
    if isinstance(customer_data, dict) and "dateOfBirth" in customer_data:
        return customer_data
        
    current_year = datetime.datetime.now().year
    
    # Convert to JSON string for the prompt
    if isinstance(customer_data, dict) and "raw_text" in customer_data:
        prompt_text = customer_data["raw_text"]
    else:
        prompt_text = json.dumps(customer_data, indent=2)
    
    # Create system prompt
    system_prompt = """
    You are an AI assistant specialized in extracting insurance customer information.
    Extract ONLY the following fields (if present) from the provided text:
    
    1. Policyholder date of birth
    2. Vehicle information: make, model, year, and calculate the age of vehicle
    3. Covered drivers: date of birth and relationship
    4. Policy effective date
    
    Format your response as a clean JSON object with these fields:
    {
      "dateOfBirth": "YYYY-MM-DD",
      "insuredVehicles": [
        {
          "make": "string",
          "model": "string", 
          "year": number,
          "ageOfVehicle": number
        }
      ],
      "coveredDrivers": [
        {
          "dateOfBirth": "YYYY-MM-DD",
          "relationship": "string"
        }
      ],
      "policyEffectiveDate": "YYYY-MM-DD"
    }
    
    IMPORTANT: Return ONLY the raw JSON object. No code blocks or explanations.
    Include ONLY fields that you can find. If a field is missing, omit it.
    """
    
    user_prompt = f"Extract customer information from this text:\n\n{prompt_text}"
    
    try:
        # Call GPT-4o only once - reduce the logging spam
        if verbose:
            print("Extracting structured data with GPT-4o...")
            
        response = openai_client.chat.completions.create(
            model=gpt4o_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        # Process and clean the response
        content = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if present
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if part.strip().startswith("json") and i+1 < len(parts):
                    content = parts[i+1].strip()
                    break
                elif part.strip().startswith("{"):
                    content = part.strip()
                    break
        
        # Remove trailing backticks
        content = content.rstrip('`')
        
        # Find the JSON content if it doesn't start with '{'
        if not content.startswith("{"):
            start_idx = content.find("{")
            if start_idx >= 0:
                content = content[start_idx:]
        
        try:
            extracted_data = json.loads(content)
            if verbose:
                print("Successfully extracted structured customer data")
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT-4o response: {str(e)}")
            print(f"Response preview: {content[:150]}...")
            return {}
            
    except Exception as e:
        print(f"Error calling GPT-4o: {str(e)}")
        return {}

# ----------------------------
# Generate Customer Text & Embedding
# ----------------------------
def format_customer_text(customer_data, verbose=True):
    """Format customer data as text for embedding"""
    text = "Customer profile. "
    
    # Add DOB if available
    if "dateOfBirth" in customer_data:
        text += f"DOB: {customer_data['dateOfBirth']}. "
    
    # Add vehicle information
    if "insuredVehicles" in customer_data and customer_data["insuredVehicles"]:
        text += "Vehicles: "
        for i, vehicle in enumerate(customer_data["insuredVehicles"]):
            details = []
            if "make" in vehicle:
                details.append(f"{vehicle['make']}")
            if "model" in vehicle:
                details.append(f"{vehicle['model']}")
            if "year" in vehicle:
                details.append(f"{vehicle['year']}")
            if "ageOfVehicle" in vehicle:
                details.append(f"{vehicle['ageOfVehicle']} years old")
            
            text += f"{', '.join(details)}. "
    
    # Add driver information
    if "coveredDrivers" in customer_data and customer_data["coveredDrivers"]:
        text += "Drivers: "
        for i, driver in enumerate(customer_data["coveredDrivers"]):
            details = []
            if "dateOfBirth" in driver:
                details.append(f"born {driver['dateOfBirth']}")
            if "relationship" in driver:
                details.append(f"{driver['relationship']}")
            
            text += f"{', '.join(details)}. "
    
    # Add policy effective date
    if "policyEffectiveDate" in customer_data:
        text += f"Effective: {customer_data['policyEffectiveDate']}. "
    
    if verbose:
        print(f"\nGenerated customer text profile:")
        print(f"{text}")
        
    return text

def get_embedding(text, openai_client, embedding_deployment, verbose=True):
    """Generate embedding for text"""
    try:
        if verbose:
            print("Generating text embedding...")
            
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_deployment
        )
        return response.data[0].embedding
        
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0.0] * 3072  # text-embedding-3-large has 3072 dimensions

# ----------------------------
# Find Similar Customers
# ----------------------------
def find_similar_customers(customer_embedding, segments_container, top_n=3, verbose=True):
    """Find most similar customer segments using cosine similarity"""
    if verbose:
        print(f"\nSearching for similar customer profiles...")
    
    # Retrieve all customer segments with embeddings
    try:
        segments = list(segments_container.read_all_items(max_item_count=1000))
        if verbose:
            print(f"Retrieved {len(segments)} customer segments from database")
            
    except Exception as e:
        print(f"Error retrieving customer segments: {str(e)}")
        return []
    
    # Calculate similarity for each segment
    similarities = []
    for segment in segments:
        # Skip segments without embeddings
        if "embedding" not in segment:
            continue
            
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(customer_embedding, segment["embedding"])
        
        # Add to results
        similarities.append({
            "segment": segment,
            "similarity": similarity,
            "policy_id": segment.get("policyId")
        })
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top N matches
    top_matches = similarities[:top_n]
    
    if verbose:
        print(f"Found {len(top_matches)} similar customer profiles")
        
    return top_matches

# ----------------------------
# Retrieve Policy Details
# ----------------------------
def get_policy_details(policy_id, policy_container, verbose=True):
    """Retrieve full policy information from policy container"""
    if verbose:
        print(f"Retrieving policy details for ID: {policy_id}")
        
    try:
        # Try to retrieve by ID directly
        policy = policy_container.read_item(item=policy_id, partition_key=policy_id)
        return policy
    except Exception:
        # If direct lookup fails, try cross-partition query
        try:
            query = f"SELECT * FROM c WHERE c.id = '{policy_id}'"
            items = list(policy_container.query_items(query=query, enable_cross_partition_query=True))
            
            if items:
                return items[0]
        except Exception as e:
            if verbose:
                print(f"Error querying by ID: {str(e)}")
                
        # Try looking up by policyNumber
        try:
            query = f"SELECT * FROM c WHERE c.policyNumber = '{policy_id}'"
            items = list(policy_container.query_items(query=query, enable_cross_partition_query=True))
            
            if items:
                return items[0]
                
        except Exception as e:
            if verbose:
                print(f"Error querying by policyNumber: {str(e)}")
                
        return None

def extract_coverage_details(policy):
    """Extract coverage information from policy"""
    if not policy:
        return {}
        
    coverage_info = {
        "policyNumber": policy.get("policyNumber", "Unknown"),
        "coverages": [],
        "limits": {},
        "deductibles": {},
        "addOns": [],
        "premium": None
    }
    
    # Extract coverages
    if "coverage" in policy:
        coverage = policy["coverage"]
        
        # Extract coverages list
        if "coverages" in coverage:
            coverage_info["coverages"] = coverage["coverages"]
        
        # Extract limits
        if "limits" in coverage:
            coverage_info["limits"] = coverage["limits"]
        
        # Extract deductibles
        if "deductibles" in coverage:
            coverage_info["deductibles"] = coverage["deductibles"]
        
        # Extract add-ons
        if "addOns" in coverage:
            coverage_info["addOns"] = coverage["addOns"]
    
    # Extract premium information
    if "pricing" in policy and "finalPremium" in policy["pricing"]:
        coverage_info["premium"] = policy["pricing"]["finalPremium"]
        
    return coverage_info

# ----------------------------
# Display Results
# ----------------------------
def display_match(match, policy_details, idx):
    """Display a single matching customer profile with coverage details"""
    print(f"\n--- MATCH #{idx+1} (Similarity: {match['similarity']:.4f}) ---")
    
    # Display policy number
    segment = match["segment"]
    print(f"Profile: {segment.get('feedback', 'No profile available')[:150]}...")
    
    if policy_details:
        # Display policy number
        print(f"Policy Number: {policy_details.get('policyNumber', 'Unknown')}")
        
        # Display coverages
        if policy_details.get("coverages"):
            print(f"Coverages: {', '.join(policy_details['coverages'])}")
        
        # Display add-ons
        if policy_details.get("addOns"):
            print(f"Add-ons: {', '.join(policy_details['addOns'])}")
        
        # Display premium
        if policy_details.get("premium"):
            print(f"Premium: ${policy_details['premium']:.2f}")
        
        # Display limits
        if policy_details.get("limits"):
            print("\nLimits:")
            for k, v in policy_details["limits"].items():
                print(f"  {k}: {v}")
                
        # Display deductibles
        if policy_details.get("deductibles"):
            print("\nDeductibles:")
            for k, v in policy_details["deductibles"].items():
                print(f"  {k}: {v}")
    else:
        print("No policy details available")

def summarize_recommendations(matches):
    """Generate a summary of recommendations from the matches"""
    # Collect all coverages and add-ons
    all_coverages = []
    all_addons = []
    all_premiums = []
    
    for match in matches:
        if "coverage_details" in match:
            coverage = match["coverage_details"]
            
            if "coverages" in coverage:
                all_coverages.extend(coverage["coverages"])
                
            if "addOns" in coverage:
                all_addons.extend(coverage["addOns"])
                
            if "premium" in coverage and coverage["premium"]:
                all_premiums.append(coverage["premium"])
    
    # Count occurrences
    coverage_counts = {}
    for item in all_coverages:
        if item in coverage_counts:
            coverage_counts[item] += 1
        else:
            coverage_counts[item] = 1
            
    addon_counts = {}
    for item in all_addons:
        if item in addon_counts:
            addon_counts[item] += 1
        else:
            addon_counts[item] = 1
    
    # Sort by frequency
    sorted_coverages = sorted(coverage_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_addons = sorted(addon_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Display recommendations
    print("\n=== RECOMMENDED COVERAGES ===")
    for coverage, count in sorted_coverages:
        percentage = (count / len(matches)) * 100
        print(f"- {coverage} ({percentage:.0f}% of similar customers)")
    
    if sorted_addons:
        print("\n=== RECOMMENDED ADD-ONS ===")
        for addon, count in sorted_addons:
            percentage = (count / len(matches)) * 100
            print(f"- {addon} ({percentage:.0f}% of similar customers)")
    
    if all_premiums:
        avg_premium = sum(all_premiums) / len(all_premiums)
        print(f"\nPremium range: ${min(all_premiums):.2f} - ${max(all_premiums):.2f}")
        print(f"Average premium: ${avg_premium:.2f}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    # Parse command line arguments
    args = parse_arguments()
    verbose = not args.quiet
    
    # Initialize configurations
    config = initialize_configs()
    
    # Connect to Azure services
    connections = connect_to_services(config, verbose)
    
    # Process input file
    customer_data = process_input_file(args.input, verbose)
    if not customer_data:
        print("Error: Could not process input file")
        return
    
    # Extract customer fields with GPT-4o (only called once to prevent log spam)
    extracted_data = extract_customer_fields(
        customer_data, 
        connections["openai"], 
        config["gpt4o_deployment"],
        verbose
    )
    
    # Format customer text for embedding
    customer_text = format_customer_text(extracted_data, verbose)
    
    # Generate embedding
    customer_embedding = get_embedding(
        customer_text, 
        connections["openai"], 
        config["embedding_deployment"],
        verbose
    )
    
    # Find similar customer segments
    similar_customers = find_similar_customers(
        customer_embedding, 
        connections["segments_container"], 
        args.top,
        verbose
    )
    
    if not similar_customers:
        print("No similar customers found")
        return
    
    # Retrieve policy details for each similar customer
    print("\n=== MATCHING CUSTOMER PROFILES ===")
    for idx, match in enumerate(similar_customers):
        policy_id = match["policy_id"]
        policy = get_policy_details(policy_id, connections["policy_container"], verbose=False)
        coverage_details = extract_coverage_details(policy)
        
        # Add coverage details to match for summarization
        match["coverage_details"] = coverage_details
        
        # Display match details
        display_match(match, coverage_details, idx)
    
    # Generate recommendation summary
    summarize_recommendations(similar_customers)

# Run the script
if __name__ == "__main__":
    main()