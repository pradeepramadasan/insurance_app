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
import sys
from autogen import AssistantAgent
from config import config_list_gpt4o
import h3  # H3 geospatial indexing library
from geopy.geocoders import Nominatim  # For geocoding addresses
import logging
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
import os
# Suppress httpx INFO logs
# Configure logging once at the top of the file
# Configure logging once at the top of the file
logger = logging.getLogger(__name__)




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
def validate_openai_connection(openai_client):
    """Validate Azure OpenAI connection following Azure best practices"""
    try:
        # Method 1: List available models (lightweight API call)
        models = openai_client.models.list()
        model_names = [model.id for model in models.data]
        print("✅ Connection successful! Available models:")
        for name in model_names:
            print(f"  - {name}")
        
        # Verify specific deployments exist
        if "gpt-4o" in model_names:
            print("✅ GPT-4o deployment available")
        else:
            print("⚠️ Warning: GPT-4o deployment not found")
            
        if "text-embedding-3-large" in model_names:
            print("✅ Embedding model available")
        else:
            print("⚠️ Warning: text-embedding-3-large model not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        
        # Check common issues
        if "401" in str(e):
            print("❌ Authentication error: Check your API key")
        elif "404" in str(e):
            print("❌ Resource not found: Check your endpoint URL")
        elif "429" in str(e):
            print("❌ Rate limit exceeded: Your requests are being throttled")
        
        return False

def test_embedding_functionality(openai_client, embedding_deployment):
    """Test if the embedding model works properly"""
    try:
        response = openai_client.embeddings.create(
            input="This is a test sentence to check if embeddings work.",
            model=embedding_deployment
        )
        
        if response and response.data and len(response.data[0].embedding) > 0:
            print(f"✅ Embedding generation successful!")
            print(f"   Vector dimensions: {len(response.data[0].embedding)}")
            return True
        else:
            print("❌ Empty embedding response")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test failed: {str(e)}")
        return False

def test_gpt4o_functionality(openai_client, gpt4o_deployment):
    """Test if the GPT-4o deployment works properly"""
    try:
        response = openai_client.chat.completions.create(
            model=gpt4o_deployment,
            messages=[{"role": "user", "content": "Say 'Azure OpenAI connection is working properly'"}],
            max_tokens=30
        )
        
        if response and response.choices and response.choices[0].message.content:
            print(f"✅ GPT-4o test successful!")
            print(f"   Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ Empty GPT-4o response")
            return False
            
    except Exception as e:
        print(f"❌ GPT-4o test failed: {str(e)}")
        return False
def validate_azure_openai_configuration(config, connections):
    """Comprehensive validation of Azure OpenAI configuration"""
    print("\n===== AZURE OPENAI VALIDATION =====")
    print(f"Endpoint: {config['openai_endpoint']}")
    print(f"API Version: {config['api_version']}")
    print(f"API Version: {config['openai_api_key']}")
    print(f"GPT-4o Deployment: {config['gpt4o_deployment']}")
    print(f"Embedding Deployment: {config['embedding_deployment']}")
    print("\nRunning validation checks...")
    
    # Get the OpenAI client
    openai_client = connections.get('openai')
    
    if not openai_client:
        print("❌ OpenAI client object not found in connections dictionary")
        return False
    
    # Step 1: Basic connection test
    connection_valid = validate_openai_connection(openai_client)
    if not connection_valid:
        return False
    
    # Step 2: Test embedding functionality
    embedding_valid = test_embedding_functionality(
        openai_client, 
        config['embedding_deployment']
    )
    print("tthhh")
    # Step 3: Test GPT-4o functionality
    gpt4o_valid = test_gpt4o_functionality(
        openai_client,
        config['gpt4o_deployment']
    )
    
    # Final result
    if connection_valid and embedding_valid and gpt4o_valid:
        print("\n✅ All Azure OpenAI validation checks passed!")
        return True
    else:
        print("\n⚠️ Some validation checks failed. See details above.")
        return False
            
# ----------------------------
# Initialize Global Configurations
# ----------------------------
def initialize_configs(verbose=True):
    # Load environment variables
    load_dotenv("x1.env")
    
    config = {}
    
    # Azure OpenAI configuration
    config["openai_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    config["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY_X1")
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
    
    try:
        connections["openai"] = AzureOpenAI(
            api_key=config["openai_api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["openai_endpoint"]
        )

        # Test Azure OpenAI connection
        if verbose:
            print("Successfully connected to Azure OpenAI")

    except Exception as e:
        print(f"Azure OpenAI connection failed: {str(e)}")
        raise

    # Cosmos DB connection - Use AzureCliCredential instead of DefaultAzureCredential
    try:
        # Use AzureCliCredential like in cosmos_db.py
        credential = AzureCliCredential()
        cosmos_client = CosmosClient(config["cosmos_endpoint"], credential=credential)
        
        # Test connection
        list(cosmos_client.list_databases())
        if verbose:
            print("Connected to Cosmos DB using AzureCliCredential")
    except Exception as e:
        print(f"Cosmos DB connection failed: {str(e)}")
        

    connections["cosmos"] = cosmos_client
    database = cosmos_client.get_database_client(config["database_name"])
    connections["database"] = database
    connections["segments_container"] = database.get_container_client("CustomerSegments")
    connections["policy_container"] = database.get_container_client("Policy")

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
# ...existing code...

# ----------------------------
# Geospatial Indexing Functions
# ----------------------------
def geocode_address(address, verbose=True):
    """Geocode address to get latitude and longitude coordinates"""
    # Build address string from components
    address_str = ""
    
    # Add street address if available
    if "streetAddress" in address:
        address_str += address["streetAddress"] + ", "
    elif "street" in address:
        address_str += address["street"] + ", "
    
    # Add city, state, postal code
    if "city" in address:
        address_str += address["city"] + ", "
    if "state" in address:
        address_str += address["state"] + ", "
    if "postalCode" in address:
        address_str += address["postalCode"] + ", "
    elif "zipCode" in address:
        address_str += address["zipCode"] + ", "
    
    # Add country if available
    if "country" in address:
        address_str += address["country"]
    
    # Trim trailing comma and space
    address_str = address_str.rstrip(", ")
    
    if not address_str:
        if verbose:
            print("Could not construct address string for geocoding")
        return None, None
    
    try:
        # Use Nominatim for geocoding
        geolocator = Nominatim(user_agent="insurance_app")
        location = geolocator.geocode(address_str)
        
        if location:
            if verbose:
                print(f"Geocoded address: {address_str} -> ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            if verbose:
                print(f"Could not geocode address: {address_str}")
            return None, None
            
    except Exception as e:
        if verbose:
            print(f"Error during geocoding: {str(e)}")
        return None, None

def calculate_h3_index(customer_data, resolution=8, verbose=True):
    """Calculate H3 geonspatial index for customer address data"""
    try:
        # Extract address information
        address = customer_data.get("address")
        
        if not address or not isinstance(address, dict):
            if verbose:
                print("No valid address found in customer data")
            return None
        
        # Try to extract latitude and longitude
        lat = address.get("latitude") or address.get("lat")
        lng = address.get("longitude") or address.get("lng") or address.get("lon")
        
        # If lat/lng not directly available, try to geocode the address
        if lat is None or lng is None:
            if verbose:
                print("Latitude/longitude not found, attempting to geocode address...")
            
            lat, lng = geocode_address(address, verbose)
            
        # If we have valid coordinates, generate H3 index
        if lat is not None and lng is not None:
            try:
                lat = float(lat)
                lng = float(lng)
                h3_index = h3.geo_to_h3(lat, lng, resolution)
                
                if verbose:
                    print(f"Generated H3 index: {h3_index} (resolution: {resolution})")
                
                return h3_index
            except Exception as e:
                if verbose:
                    print(f"Error generating H3 index: {str(e)}")
                return None
        else:
            if verbose:
                print("Could not determine latitude/longitude for H3 indexing")
            return None
            
    except ImportError:
        print("H3 library not installed. Install with: pip install h3")
        return None
# ----------------------------
# Extract Customer Data Fields
# ----------------------------
def extract_customer_fields(customer_data, openai_client, gpt4o_deployment, verbose=True):
    """Extract standardized fields from customer data for embedding and comparison"""
    # If already structured properly, just return
    if isinstance(customer_data, dict) and "dateOfBirth" in customer_data:
        return customer_data
        
    current_year = datetime.datetime.now().year
    
    # Convert to JSON string for the prompt
    if isinstance(customer_data, dict) and "raw_text" in customer_data:
        prompt_text = customer_data["raw_text"]
    else:
        prompt_text = json.dumps(customer_data, indent=2)
    
    # Create comprehensive system prompt based on customerprofilingfields.txt
    system_prompt = """
    You are an AI assistant specialized in extracting insurance customer information.
    Extract the following fields (if present) from the provided text:
    
    1. Personal Information:
       - Full name
       - Date of birth
       - Gender
       - Marital status
       - Occupation
       - Contact information (email, phone)
    
    2. Address Information:
       - Street address
       - City
       - State/province
       - Postal/zip code
       - Country
    
    3. Vehicle Information:
       - Make
       - Model
       - Year
       - VIN
       - Vehicle usage type (personal/commercial)
       - Annual mileage
       - Vehicle age based on current year
    
    4. Policy Information:
       - Policy type
       - Policy effective date
       - Policy expiration date
       - Policy number
    
    5. Coverage Information:
       - Liability limits
       - Deductibles
       - Coverage types
       - Special endorsements
    
    6. Driver Information:
       - Additional drivers
       - Each driver's DOB
       - Each driver's relationship to policyholder
       - Driving history (accidents, violations)
    
    7. Risk Factors:
       - Prior claims
       - Credit score range (if available)
       - Property features (for home insurance)
       - Safety devices
    
    Format your response as a clean JSON object with these categories.
    IMPORTANT: Return ONLY the raw JSON object. No code blocks or explanations.
    Include ONLY fields that you can find in the text. If a field is missing, omit it.
    """
    
    user_prompt = f"Extract insurance customer profile information from this text:\n\n{prompt_text}"
    
    try:
        # Azure Best Practice: Validate OpenAI client before using
        if openai_client is None:
            raise ValueError("OpenAI client is None. Check your Azure OpenAI configuration.")
            
        # Call Azure OpenAI GPT-4o with best practices
        if verbose:
            print("Extracting structured data with Azure OpenAI GPT-4o...")
            print(f"Using deployment: {gpt4o_deployment}")
            
        # Azure Best Practice: Add validation check for deployment name
        if not gpt4o_deployment:
            raise ValueError("GPT-4o deployment name is empty. Check AZURE_OPENAI_GPT4O_DEPLOYMENT in your .env file.")
            
        # Using Azure OpenAI best practices
        response = openai_client.chat.completions.create(
            model=gpt4o_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"},  # Changed from "json" to "json_object"
            timeout=30  # Azure best practice - set reasonable timeout
        )
        
        # Rest of the function remains the same...
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
            
            # Calculate vehicle age if year is provided but age is not
            if "insuredVehicles" in extracted_data and isinstance(extracted_data["insuredVehicles"], list):
                for vehicle in extracted_data["insuredVehicles"]:
                    if "year" in vehicle and "ageOfVehicle" not in vehicle:
                        try:
                            vehicle_year = int(vehicle["year"])
                            vehicle["ageOfVehicle"] = current_year - vehicle_year
                        except (ValueError, TypeError):
                            pass
            
            # Calculate H3 geospatial index if address information is present
            if "address" in extracted_data and isinstance(extracted_data["address"], dict):
                h3_index = calculate_h3_index(extracted_data, resolution=8, verbose=verbose)
                if h3_index:
                    extracted_data["h3_index"] = h3_index
                    
                    # Find neighboring H3 indices for proximity search
                    try:
                        extracted_data["h3_neighbors"] = h3.k_ring(h3_index, 1)
                    except Exception as e:
                        if verbose:
                            print(f"Error calculating H3 neighbors: {str(e)}")
            
            if verbose:
                print("Successfully extracted structured customer data")
                print(f"Extracted {len(extracted_data)} top-level fields")
                if "h3_index" in extracted_data:
                    print(f"H3 index: {extracted_data['h3_index']}")
            
            return extracted_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Azure OpenAI response: {str(e)}")
            print(f"Response preview: {content[:150]}...")
            return {}
            
    except Exception as e:
        print(f"Error calling Azure OpenAI GPT-4o: {str(e)}")
        return {}
    
# ----------------------------
# Generate Customer Text & Embedding
# ----------------------------
def format_customer_text(customer_data, verbose=True):
    """Format customer data as text for embedding"""
    text = "Customer profile. "
    
    # Add personal information
    if "fullName" in customer_data:
        text += f"Name: {customer_data['fullName']}. "
    
    if "dateOfBirth" in customer_data:
        text += f"DOB: {customer_data['dateOfBirth']}. "
        
    if "gender" in customer_data:
        text += f"Gender: {customer_data['gender']}. "
        
    if "maritalStatus" in customer_data:
        text += f"Marital status: {customer_data['maritalStatus']}. "
        
    if "occupation" in customer_data:
        text += f"Occupation: {customer_data['occupation']}. "
    
    # Add contact info
    if "email" in customer_data:
        text += f"Email: {customer_data['email']}. "
        
    if "phone" in customer_data:
        text += f"Phone: {customer_data['phone']}. "
    
    # Add address information with H3 index
    if "address" in customer_data:
        addr = customer_data["address"]
        address_parts = []
        
        if "streetAddress" in addr:
            address_parts.append(addr["streetAddress"])
        if "city" in addr:
            address_parts.append(addr["city"])
        if "state" in addr:
            address_parts.append(addr["state"])
        if "postalCode" in addr:
            address_parts.append(addr["postalCode"])
            
        if address_parts:
            text += f"Address: {', '.join(address_parts)}. "
    
    # Include H3 geospatial index if available
    if "h3_index" in customer_data:
        text += f"Location H3: {customer_data['h3_index']}. "
    
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
            if "vehicleUsage" in vehicle:
                details.append(f"{vehicle['vehicleUsage']} usage")
            if "annualMileage" in vehicle:
                details.append(f"{vehicle['annualMileage']} miles/year")
            
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
            if "drivingHistory" in driver:
                details.append(f"history: {driver['drivingHistory']}")
            
            text += f"{', '.join(details)}. "
    
    # Add policy information
    if "policyType" in customer_data:
        text += f"Policy type: {customer_data['policyType']}. "
        
    if "policyNumber" in customer_data:
        text += f"Policy number: {customer_data['policyNumber']}. "
        
    if "policyEffectiveDate" in customer_data:
        text += f"Effective: {customer_data['policyEffectiveDate']}. "
        
    if "policyExpirationDate" in customer_data:
        text += f"Expiration: {customer_data['policyExpirationDate']}. "
    
    # Add coverage information
    if "coverage" in customer_data:
        coverage = customer_data["coverage"]
        if isinstance(coverage, dict):
            if "coverageTypes" in coverage:
                text += f"Coverage types: {', '.join(coverage['coverageTypes'])}. "
            if "liabilityLimits" in coverage:
                text += f"Liability limits: {coverage['liabilityLimits']}. "
            if "deductibles" in coverage:
                text += f"Deductibles: {coverage['deductibles']}. "
    
    # Add risk factors
    if "riskFactors" in customer_data:
        risk = customer_data["riskFactors"]
        if isinstance(risk, dict):
            if "priorClaims" in risk:
                text += f"Prior claims: {risk['priorClaims']}. "
            if "creditScore" in risk:
                text += f"Credit score range: {risk['creditScore']}. "
    
    if verbose:
        print(f"\nGenerated customer text profile:")
        print(f"{text}")
        
    return text

def get_embedding(text, openai_client, embedding_deployment, verbose=False):
    """
    Generate embeddings for text using Azure OpenAI.
    
    Parameters:
        text (str): The text to generate embeddings for
        openai_client: Azure OpenAI client instance
        embedding_deployment (str): Name of the embedding model deployment
        verbose (bool): Whether to print verbose output
        
    Returns:
        list: The embedding vector
    """
    try:
        # Log only when needed
        if verbose:
            logger.info("Generating text embedding with Azure OpenAI")
        
        # Azure best practice: Set timeout and clear parameters
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_deployment,
            dimensions=3072  # Explicitly set dimensions for text-embedding-3-large
        )
        
        # Azure best practice: Validate response
        if not response or not response.data or not response.data[0].embedding:
            raise ValueError("Empty embedding response received")
            
        return response.data[0].embedding
        
    except Exception as e:
        # Azure best practice: Structured error logging
        logger.error(f"Azure OpenAI embedding error: {str(e)}")
        
        # Return fallback embedding vector with correct dimensions
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
        if "embedding" not in segment or not segment["embedding"]:
            if verbose:
                print(f"Skipping segment with ID {segment.get('id', 'unknown')}: no embedding")
            continue
        
        # Validate embeddings to prevent NaN results
        if not all(isinstance(x, (int, float)) for x in customer_embedding) or \
           not all(isinstance(x, (int, float)) for x in segment["embedding"]):
            if verbose:
                print(f"Skipping segment with ID {segment.get('id', 'unknown')}: invalid embedding type")
            continue
            
        # Prevent division by zero in cosine calculation
        if np.all(np.array(customer_embedding) == 0) or np.all(np.array(segment["embedding"]) == 0):
            if verbose:
                print(f"Skipping segment with ID {segment.get('id', 'unknown')}: zero embedding")
            continue
            
        try:
            # Calculate cosine similarity (1 - cosine distance)
            # Use try-except to handle potential numerical issues
            similarity = 1 - cosine(customer_embedding, segment["embedding"])
            
            # Check if similarity is a valid number
            if np.isnan(similarity) or np.isinf(similarity):
                if verbose:
                    print(f"Warning: Invalid similarity result for segment {segment.get('id', 'unknown')}")
                continue
                
            # Add to results
            similarities.append({
                "segment": segment,
                "similarity": similarity,
                "policy_id": segment.get("policyId")
            })
            
        except Exception as e:
            if verbose:
                print(f"Error calculating similarity for segment {segment.get('id', 'unknown')}: {str(e)}")
            continue
    
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
    # Format similarity with proper error handling
    similarity_str = f"{match['similarity']:.4f}" if isinstance(match['similarity'], (int, float)) and not np.isnan(match['similarity']) else "N/A"
    
    print(f"\n--- MATCH #{idx+1} (Similarity: {similarity_str}) ---")
    
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
        if policy_details.get("premium") is not None:
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

def display_all_similarity_scores(customer_embedding, segments_container, verbose=True):
    """Display all similarity scores for debugging purposes"""
    print("\n=== DETAILED SIMILARITY ANALYSIS ===")
    
    # Retrieve all customer segments with embeddings
    try:
        segments = list(segments_container.read_all_items(max_item_count=1000))
        if verbose:
            print(f"Retrieved {len(segments)} total customer segments from database")
            
        if not segments:
            print("WARNING: No customer segments found in database!")
            return
            
    except Exception as e:
        print(f"Error retrieving customer segments: {str(e)}")
        return
        
    # Track statistics
    stats = {
        "total": len(segments),
        "no_embedding": 0,
        "invalid_embedding": 0,
        "zero_embedding": 0,
        "nan_result": 0,
        "error": 0,
        "valid": 0
    }
    
    # Calculate similarity for all segments
    print("\nSimilarity Scores:")
    print("-----------------------------------------------------------")
    print("| Policy ID               | Similarity | Status            |")
    print("-----------------------------------------------------------")
    
    valid_similarities = []
    
    for segment in segments:
        policy_id = segment.get("policyId", "unknown")
        status = "Unknown"
        similarity_value = None
        
        # Check if segment has embedding
        if "embedding" not in segment or not segment["embedding"]:
            status = "No embedding"
            stats["no_embedding"] += 1
        
        # Validate embeddings
        elif not all(isinstance(x, (int, float)) for x in customer_embedding) or \
             not all(isinstance(x, (int, float)) for x in segment["embedding"]):
            status = "Invalid embedding"
            stats["invalid_embedding"] += 1
            
        # Check for zero embeddings
        elif np.all(np.array(customer_embedding) == 0) or np.all(np.array(segment["embedding"]) == 0):
            status = "Zero embedding"
            stats["zero_embedding"] += 1
            
        else:
            try:
                # Calculate cosine similarity
                similarity_value = 1 - cosine(customer_embedding, segment["embedding"])
                
                # Check if result is valid
                if np.isnan(similarity_value) or np.isinf(similarity_value):
                    status = "NaN/Inf result"
                    stats["nan_result"] += 1
                else:
                    status = "Valid"
                    stats["valid"] += 1
                    valid_similarities.append(similarity_value)
                    
            except Exception as e:
                status = f"Error: {str(e)[:20]}..."
                stats["error"] += 1
        
        # Format similarity string
        if similarity_value is not None:
            similarity_str = f"{similarity_value:.4f}"
        else:
            similarity_str = "N/A"
            
        # Print result
        print(f"| {policy_id[:20]:<20} | {similarity_str:>10} | {status:<16} |")
    
    print("-----------------------------------------------------------")
    
    # Print statistics
    print("\nSimilarity Statistics:")
    print(f"  Total segments: {stats['total']}")
    print(f"  Valid similarities: {stats['valid']}")
    print(f"  No embedding: {stats['no_embedding']}")
    print(f"  Invalid embedding: {stats['invalid_embedding']}")
    print(f"  Zero embedding: {stats['zero_embedding']}")
    print(f"  NaN/Inf results: {stats['nan_result']}")
    print(f"  Calculation errors: {stats['error']}")
    
    if valid_similarities:
        print(f"\nValid similarity range: {min(valid_similarities):.4f} to {max(valid_similarities):.4f}")
        print(f"Average similarity: {sum(valid_similarities)/len(valid_similarities):.4f}")
    else:
        print("\nNo valid similarities found!")

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
    #print(config)
    
    # Connect to Azure services
    print("xx")
    print(config["gpt4o_deployment"])
    connections = connect_to_services(config, verbose)

    print("xx")
    print(config["gpt4o_deployment"])
    
    # Process input file
    customer_data = process_input_file(args.input, verbose)
    print("xxx")
    print(config["gpt4o_deployment"])
    
    if not customer_data:
        print("Error: Could not process input file")
        return
    
    # Extract customer fields with GPT-4o
    extracted_data = extract_customer_fields(
        customer_data, 
        connections["openai"], 
        config["gpt4o_deployment"],
        verbose
    )
    print("xxxx")
    print(config["gpt4o_deployment"])
    
    # Format customer text for embedding
    customer_text = format_customer_text(extracted_data, verbose)
    
    # Generate embedding
    customer_embedding = get_embedding(
        customer_text, 
        connections["openai"], 
        config["embedding_deployment"],
        verbose
    )
    
    # NEW: If debug mode enabled, show all similarity scores
    if args.debug:
        display_all_similarity_scores(customer_embedding, connections["segments_container"], verbose=True)
    
    # Find similar customer segments
    similar_customers = find_similar_customers(
        customer_embedding, 
        connections["segments_container"], 
        args.top,
        verbose
    )
    
    # Rest of function continues as before...    
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

def call_customerprofile(customer_data: dict) -> dict:
  
    logger.info("Starting customer profile processing")
    try:
        verbose = False
        config = initialize_configs(verbose=False)
        connections = connect_to_services(config, verbose=False)
        
        # Validate Azure OpenAI configuration silently
        is_valid = validate_azure_openai_configuration(config, connections)

        # Calculate garaging H3 index if ZIP code exists
        vehicles = customer_data.get("vehicles", [])
        for vehicle in vehicles:
            garaging_zip = vehicle.get("garagingZip")
            if garaging_zip:
                lat, lng = geocode_address({"postalCode": garaging_zip}, verbose=False)
                if lat and lng:
                    vehicle["garagingH3Index"] = h3.geo_to_h3(lat, lng, resolution=8)

        # Extract structured fields
        extracted_data = extract_customer_fields(
            customer_data,
            connections["openai"],
            config["gpt4o_deployment"],
            verbose=False
        )
        print("I a done with extracting data with gpt4o)")
        
        # Generate embedding
        customer_text = format_customer_text(extracted_data, verbose=False)
        customer_embedding = get_embedding(
            customer_text,
            connections["openai"],
            config["embedding_deployment"],
            verbose=False
        )
        
        # Find top 3 similar customers
        similar_customers = find_similar_customers(
            customer_embedding,
            connections["segments_container"],
            top_n=3,
            verbose=False
        )

        result = {"TOP_3_CLOSEST_POLICIES": []}

        for match in similar_customers:
            policy_id = match["policy_id"]
            policy = get_policy_details(policy_id, connections["policy_container"], verbose=False)
            coverage_details = extract_coverage_details(policy)

            result_entry = {
                "policyId": policy_id,
                "similarityScore": match["similarity"],
                "coverages": coverage_details.get("coverages", []),
                "limits": coverage_details.get("limits", {}),
                "deductibles": coverage_details.get("deductibles", {}),
                "addOns": coverage_details.get("addOns", []),
                "premium": coverage_details.get("premium")
            }

            result["TOP_3_CLOSEST_POLICIES"].append(result_entry)

        return result

    except Exception as e:
        return {
            "TOP_3_CLOSEST_POLICIES": [],
            "error": str(e)
        }
    
if __name__ == "__main__":
    # Parse command line arguments properly
    args = parse_arguments()
    
    # Use the parsed input file argument
    input_file = args.input
    
    # Load customer data from the provided JSON file
    with open(input_file, "r") as f:
        customer_data = json.load(f)
    
    # Call the main processing function
    output = call_customerprofile(customer_data)
    
    # Print the output JSON
    print(json.dumps(output, indent=4))