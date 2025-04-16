# This script retrieves policy data from Azure Cosmos DB, 
# generates text embeddings using Azure OpenAI, clusters the data using KMeans, 
# and stores the segmented data back in Cosmos DB

import os
from dotenv import load_dotenv
import numpy as np
import json
import datetime
from azure.cosmos import CosmosClient, PartitionKey
from sklearn.cluster import KMeans
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ClientSecretCredential

# ----------------------------
# 1. Load Configuration from Environment Variables
# ----------------------------

# Load variables from x1.env file
load_dotenv("x1.env")

# Azure OpenAI configuration
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY_X1")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# GPT-4o configuration from x1.env
gpt4o_deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o") 
gpt4o_model = os.getenv("AZURE_OPENAI_GPT4O_MODEL", "gpt-4o")

# Replace the existing print statements with OpenAI-focused output
print("\n=== AZURE OPENAI CONFIGURATION ===")
print(f"Endpoint:               {openai_endpoint if openai_endpoint else 'NOT SET'}")
print(f"API Version:            {api_version}")
print(f"API Key:                {'Configured' if openai_api_key else 'NOT CONFIGURED'}")
print(f"Embedding Model:        {embedding_model}")
print(f"Embedding Deployment:   {embedding_deployment}")
print(f"GPT-4o Model:           {gpt4o_model}")
print(f"GPT-4o Deployment:      {gpt4o_deployment}")
print("================================\n")

# Cosmos DB configuration
cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
cosmos_key = os.getenv("COSMOS_KEY")
database_name = os.getenv("COSMOS_DATABASE_NAME", "insurance")
policy_container_name = os.getenv("COSMOS_CONTAINER_NAME", "Policy")

# Azure AD credentials
tenant_id = os.getenv("AZURE_TENANT_ID")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")

print(f"Loading configuration from x1.env")
print(f"Using database: {database_name}, container: {policy_container_name}")
print(f"Azure OpenAI endpoint: {openai_endpoint}")
print(f"Embedding model: {embedding_model}, deployment: {embedding_deployment}")
print(f"GPT-4o model: {gpt4o_model}, deployment: {gpt4o_deployment}")

# ----------------------------
# 2. Initialize Azure OpenAI Client
# ----------------------------

# Initialize Azure OpenAI client
azure_openai_client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=api_version,
    azure_endpoint=openai_endpoint
)

# ----------------------------
# 3. Initialize Cosmos DB with Azure AD Authentication
# ----------------------------

try:
    # Option 1: Try using AzureCliCredential first (if you're logged in via CLI)
    print("Attempting to connect with AzureCliCredential...")
    credential = AzureCliCredential()
    client = CosmosClient(cosmos_endpoint, credential=credential)
    
    # Test connection
    list(client.list_databases())
    print("Successfully connected using AzureCliCredential")
    
except Exception as e:
    print(f"AzureCliCredential failed: {e}")
    # Option 2: Fall back to ClientSecretCredential
    print("Attempting to connect with ClientSecretCredential...")
    try:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        client = CosmosClient(cosmos_endpoint, credential=credential)
        
        # Test connection
        list(client.list_databases())
        print("Successfully connected using ClientSecretCredential")
        
    except Exception as e:
        print(f"ClientSecretCredential also failed: {e}")
        print("Could not connect to Cosmos DB. Exiting.")
        exit(1)

# ----------------------------
# 4. Define Helper Functions
# ----------------------------

def read_embedding_config(config_file="embedding_config.txt"):
    """
    Read embedding fields configuration from a text file.
    
    Expected format in config file:
    dateOfBirth
    insuredVehicles.make
    insuredVehicles.model
    insuredVehicles.year
    coveredDrivers.dateOfBirth
    coveredDrivers.relationship
    policyEffectiveDate
    
    Returns:
        dict: Configuration with field paths to extract
    """
    config = {
        "fields": []
    }
    
    try:
        with open(config_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            field = line.strip()
            if field and not field.startswith('#') and '=' not in field:  # Skip comments, empty lines, and settings
                config["fields"].append(field)
                
        print(f"Loaded {len(config['fields'])} fields from config file: {config_file}")
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found. Using default fields.")
        # Default fields if config file not found
        config["fields"] = [
            "dateOfBirth",
            "insuredVehicles.make",
            "insuredVehicles.model", 
            "insuredVehicles.year",
            "coveredDrivers.dateOfBirth",
            "coveredDrivers.relationship",
            "policyEffectiveDate"
        ]
        return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        return config

def read_segment_config(config_file="customerprofilingfields.txt"):
    """
    Read segment configuration from a text file.
    
    Expected format in config file (can include these parameters):
    num_segments=5
    algorithm=kmeans
    random_seed=42
    
    Returns:
        dict: Configuration with segment parameters
    """
    config = {
        "num_segments": 3,  # Default to 3 if not specified
        "algorithm": "kmeans",
        "random_seed": 42
    }
    
    try:
        with open(config_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments and empty lines
                # Try to parse parameters in the form key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle specific parameters
                    if key == "num_segments":
                        try:
                            config["num_segments"] = int(value)
                            print(f"Using {config['num_segments']} segments from config file")
                        except ValueError:
                            print(f"Invalid num_segments value: {value}, using default")
                    elif key == "algorithm":
                        config["algorithm"] = value
                    elif key == "random_seed":
                        try:
                            config["random_seed"] = int(value)
                        except ValueError:
                            pass
        
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found. Using default segment settings.")
        return config
    except Exception as e:
        print(f"Error reading segment config file: {e}")
        return config

def format_policy_for_embedding(policy):
    """
    Format policy data as text for embedding
    """
    # Get basic policy information
    policy_number = policy.get("policyNumber", "unknown")
    status = policy.get("status", "unknown")
    
    # Start with basic policy info
    policy_text = f"Policy {policy_number}. Status: {status}. "
    
    # Add customer profile information if available
    if "customerProfile" in policy:
        profile = policy["customerProfile"]
        
        # Add personal info
        if "personal" in profile:
            personal = profile["personal"]
            if "name" in personal:
                policy_text += f"Customer: {personal['name']}. "
            if "dateOfBirth" in personal:
                policy_text += f"DOB: {personal['dateOfBirth']}. "
            if "address" in personal and "zip" in personal["address"]:
                policy_text += f"ZIP: {personal['address']['zip']}. "
        
        # Add vehicle info
        if "vehicle" in profile:
            vehicle = profile["vehicle"]
            vehicle_info = []
            if "make" in vehicle:
                vehicle_info.append(vehicle["make"])
            if "model" in vehicle:
                vehicle_info.append(vehicle["model"])
            if "year" in vehicle:
                vehicle_info.append(str(vehicle["year"]))
                
            if vehicle_info:
                policy_text += f"Vehicle: {' '.join(vehicle_info)}. "
        
        # Add coverage preferences
        if "coveragePreferences" in profile:
            prefs = profile["coveragePreferences"]
            
            # Vehicle types
            if "insuredVehicles" in prefs and prefs["insuredVehicles"]:
                vehicle_types = []
                for v in prefs["insuredVehicles"]:
                    if "vehicleType" in v:
                        vehicle_types.append(v["vehicleType"])
                    if "mileage" in v:
                        policy_text += f"Mileage: {v['mileage']}. "
                        
                if vehicle_types:
                    policy_text += f"Vehicle types: {', '.join(vehicle_types)}. "
            
            # Covered drivers
            if "coveredDrivers" in prefs and prefs["coveredDrivers"]:
                driver_count = len(prefs["coveredDrivers"])
                policy_text += f"Drivers: {driver_count}. "
    
    # Add coverage information
    if "coverage" in policy:
        coverage = policy["coverage"]
        
        # Add coverages list
        if "coverages" in coverage and coverage["coverages"]:
            policy_text += f"Coverages: {', '.join(coverage['coverages'])}. "
        
        # Add add-ons
        if "addOns" in coverage and coverage["addOns"]:
            policy_text += f"Add-ons: {', '.join(coverage['addOns'])}. "
    
    # Add pricing information
    if "pricing" in policy and "finalPremium" in policy["pricing"]:
        policy_text += f"Premium: ${policy['pricing']['finalPremium']}. "
    
    return policy_text

def get_text_embedding(text):
    """
    Use Azure OpenAI service to generate embeddings for a given text.
    """
    try:
        # Using the Azure OpenAI client
        response = azure_openai_client.embeddings.create(
            input=text,
            model=embedding_deployment
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a zero vector as fallback (adjust dimension for text-embedding-3-large)
        return [0.0] * 3072  # text-embedding-3-large has 3072 dimensions

# ----------------------------
# 5. Access Database and Retrieve Data
# ----------------------------

# Now access the database and container
database = client.get_database_client(database_name)
policy_container = database.get_container_client(policy_container_name)

print("Retrieving policy data from Cosmos DB...")
try:
    # Try the container specified in environment variables
    policy_container = database.get_container_client(policy_container_name)
    policy_items = list(policy_container.read_all_items(max_item_count=100))
    print(f"Successfully retrieved {len(policy_items)} records from {policy_container_name} container")
except Exception as e:
    print(f"Error accessing {policy_container_name} container: {str(e)}")
    
    # Try 'Policy' container as fallback
    try:
        print("Trying 'Policy' container instead...")
        policy_container = database.get_container_client("Policy")
        policy_items = list(policy_container.read_all_items(max_item_count=100))
        print(f"Successfully retrieved {len(policy_items)} records from 'Policy' container")
    except Exception as e2:
        print(f"Error accessing 'Policy' container: {str(e2)}")
        
        # Create mock data if both attempts fail
        print("Creating mock data for demonstration...")
        policy_items = [
            {
                "id": "mock1",
                "policyNumber": "POL123456",
                "status": "Active",
                "customerProfile": {
                    "personal": {
                        "name": "John Smith",
                        "dateOfBirth": "1980-05-15",
                        "address": {"zip": "95123"}
                    },
                    "vehicle": {
                        "make": "Honda", 
                        "model": "Civic", 
                        "year": 2018
                    },
                    "coveragePreferences": {
                        "insuredVehicles": [{"vehicleType": "Sedan", "mileage": 35000}],
                        "coveredDrivers": [{"name": "John Smith"}]
                    }
                },
                "coverage": {
                    "coverages": ["Bodily Injury", "Property Damage"],
                    "limits": {"bodily_injury": {"per_person": 50000}},
                    "deductibles": {"collision": {"amount": 500}},
                    "addOns": ["Roadside Assistance"]
                },
                "pricing": {"finalPremium": 950.75}
            },
            # Add a second mock item with different values
            {
                "id": "mock2",
                "policyNumber": "POL789012",
                "status": "Active",
                "customerProfile": {
                    "personal": {
                        "name": "Emma Johnson",
                        "dateOfBirth": "1992-11-23",
                        "address": {"zip": "94103"}
                    },
                    "vehicle": {
                        "make": "Toyota", 
                        "model": "RAV4", 
                        "year": 2020
                    },
                    "coveragePreferences": {
                        "insuredVehicles": [{"vehicleType": "SUV", "mileage": 15000}],
                        "coveredDrivers": [{"name": "Emma Johnson"}]
                    }
                },
                "coverage": {
                    "coverages": ["Bodily Injury", "Property Damage", "Comprehensive", "Collision"],
                    "limits": {"bodily_injury": {"per_person": 100000}},
                    "deductibles": {"collision": {"amount": 250}, "comprehensive": {"amount": 100}},
                    "addOns": ["Rental Car Reimbursement", "Gap Insurance"]
                },
                "pricing": {"finalPremium": 1250.50}
            }
        ]
        print(f"Created {len(policy_items)} mock policy records")

# ----------------------------
# 6. Process Data
# ----------------------------

# Format policy data as text for embeddings
print("\nFormatting policy data for embeddings...")
policy_texts = [format_policy_for_embedding(item) for item in policy_items]

# Display a sample of the formatted policies
if policy_texts:
    print("\nSample of formatted policy text:")
    print(policy_texts[0][:150] + "...")  # Print first 150 chars

# Generate embeddings for policy texts
print("\nGenerating embeddings for policy data...")
embeddings = np.array([get_text_embedding(text) for text in policy_texts])
print(f"Generated embeddings with shape: {embeddings.shape}")

# ----------------------------
# 7. Clustering using KMeans
# ----------------------------

# Load segment configuration
segment_config = read_segment_config("customerprofilingfields.txt")
configured_num_segments = segment_config["num_segments"]

# Define the number of clusters based on configuration and data size
num_clusters = min(configured_num_segments, len(policy_texts))  # At least one item per cluster
print(f"\nClustering policies into {num_clusters} segments (from config)...")

kmeans = KMeans(n_clusters=num_clusters, random_state=segment_config["random_seed"])
kmeans.fit(embeddings)
labels = kmeans.labels_  # Cluster assignments for each policy

# Count customers in each segment
segment_counts = {}
for label in labels:
    if label not in segment_counts:
        segment_counts[label] = 0
    segment_counts[label] += 1

print("Customer segments distribution:")
for segment, count in segment_counts.items():
    print(f"  Segment {segment}: {count} customers ({count/len(labels)*100:.1f}%)")

# ----------------------------
# 8. Store Results in Cosmos DB
# ----------------------------

# Create (or get) a container for storing customer segments
segments_container_name = "CustomerSegments"
try:
    # Check if container exists by listing containers and looking for our container
    container_list = list(database.list_containers())
    container_exists = any(container['id'] == segments_container_name for container in container_list)
    
    if container_exists:
        segments_container = database.get_container_client(segments_container_name)
        print(f"\nUsing existing container: {segments_container_name}")
    else:
        print(f"\nCreating new container: {segments_container_name}")
        segments_container = database.create_container(
            id=segments_container_name,
            partition_key=PartitionKey(path="/segment"),
            default_ttl=None  # No automatic expiration for embedding data
        )
        print(f"Container {segments_container_name} created successfully with partition key '/segment'")
except Exception as e:
    print(f"Error accessing or creating container: {str(e)}")
    raise

# Store each segmented customer policy
print("Storing customer segments in Cosmos DB...")
for idx, (policy_text, segment) in enumerate(zip(policy_texts, labels)):
    # Get the original policy item for reference ID
    original_id = policy_items[idx].get("id", str(idx))
    
    # Convert numpy array to list for JSON serialization
    embedding_vector = embeddings[idx].tolist()
    
    item = {
        "id": f"segment_{original_id}",
        "policyId": original_id,
        "policyText": policy_text,
        "segment": int(segment),
        "segmentInfo": f"Customer Group {int(segment)}",
        "embedding": embedding_vector
    }
    segments_container.upsert_item(item)

print("\nCustomer segmentation completed and stored in Cosmos DB!")