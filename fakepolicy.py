import os
import uuid
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from faker import Faker
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import AzureCliCredential, ClientSecretCredential
import h3  # Import H3 library
from geopy.geocoders import Nominatim
import time  # For sleep between geocoding requests

# Initialize Faker for synthetic data generation
fake = Faker()

# Load environment variables from x1.env
load_dotenv("x1.env")

# Get CosmosDB settings from environment variables
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
DATABASE_NAME = os.getenv("COSMOS_DATABASE_NAME", "insurance")
CONTAINER_NAME = "Policy"  # Use Policy as the container name

# Azure AD credentials
tenant_id = os.getenv("AZURE_TENANT_ID")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")

print(f"Connecting to Cosmos DB at {COSMOS_ENDPOINT}")
print(f"Using database: {DATABASE_NAME}, container: {CONTAINER_NAME}")

# Try to authenticate using Azure CLI credentials first
try:
    print("Attempting to connect with AzureCliCredential...")
    credential = AzureCliCredential()
    client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
    
    # Test connection
    list(client.list_databases())
    print("Successfully connected using AzureCliCredential")
    
except Exception as e:
    print(f"AzureCliCredential failed: {e}")
    # Fall back to ClientSecretCredential
    print("Attempting to connect with ClientSecretCredential...")
    try:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        client = CosmosClient(COSMOS_ENDPOINT, credential=credential)
        
        # Test connection
        list(client.list_databases())
        print("Successfully connected using ClientSecretCredential")
        
    except Exception as e:
        print(f"ClientSecretCredential also failed: {e}")
        print("Could not connect to Cosmos DB. Exiting.")
        exit(1)

# Get or create the database
database = client.get_database_client(DATABASE_NAME)

# Create or get the container
try:
    container = database.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path="/policyNumber")
    )
    print(f"Successfully accessed container: {CONTAINER_NAME}")
except Exception as e:
    print(f"Error creating/accessing container: {e}")
    container = database.get_container_client(CONTAINER_NAME)
    print(f"Using existing container: {CONTAINER_NAME}")

# California zip codes (major regions)
CA_ZIP_RANGES = [
    (90001, 91755),  # Los Angeles County
    (91901, 92899),  # San Diego, Orange County
    (93001, 93999),  # Central Coast
    (94001, 94999),  # San Francisco Bay Area
    (95001, 96162),  # Northern California
]

# California cities
CA_CITIES = [
    "Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento",
    "Fresno", "Long Beach", "Oakland", "Bakersfield", "Anaheim", 
    "Santa Ana", "Riverside", "Stockton", "Irvine", "Chula Vista",
    "Fremont", "San Bernardino", "Modesto", "Fontana", "Oxnard",
    "Moreno Valley", "Huntington Beach", "Glendale", "Santa Clarita"
]

# Define the effective date range for the policies
start_date = datetime(2024, 1, 10)
end_date = datetime(2026, 3, 31)

# Initialize geocoder with a user agent
geolocator = Nominatim(user_agent="insurance_app_generator")

# Cache for ZIP code to coordinates (to minimize API calls)
zip_to_coords_cache = {}

def get_h3_for_zip(zipcode, resolution=8):
    """
    Convert a ZIP code to an H3 hexagon index
    
    Args:
        zipcode (str): ZIP code to convert
        resolution (int): H3 resolution (0-15), higher is more precise
        
    Returns:
        str: H3 index or None if geocoding fails
    """
    # Check cache first
    if zipcode in zip_to_coords_cache:
        lat, lon = zip_to_coords_cache[zipcode]
    else:
        try:
            # Use California to narrow down the search
            location = geolocator.geocode(f"{zipcode}, CA, USA")
            
            if location:
                lat, lon = location.latitude, location.longitude
                zip_to_coords_cache[zipcode] = (lat, lon)
                # Be nice to the geocoding service
                time.sleep(0.5)
            else:
                # If not found, use approximate coordinates for California
                print(f"Geocoding failed for {zipcode}, using approximate location")
                # Central California coordinates as fallback
                lat, lon = 36.7783, -119.4179  
                zip_to_coords_cache[zipcode] = (lat, lon)
                
        except Exception as e:
            print(f"Error geocoding ZIP {zipcode}: {e}")
            # Central California coordinates as fallback
            lat, lon = 36.7783, -119.4179
            zip_to_coords_cache[zipcode] = (lat, lon)
    
    # Convert to H3 index
    try:
        hex_index = hex_index = h3.latlng_to_cell(lat, lon, resolution)
        return hex_index
    except Exception as e:
        print(f"Error converting to H3: {e}")
        return None

def generate_ca_zip():
    """Generate a random California zip code"""
    zip_range = random.choice(CA_ZIP_RANGES)
    return str(random.randint(zip_range[0], zip_range[1]))

def generate_ca_address():
    """Generate a California address"""
    city = random.choice(CA_CITIES)
    return f"{fake.street_address()}\n{city}, CA {generate_ca_zip()}"

def extract_zip_from_address(address):
    """Extract zip code from address string"""
    parts = address.split()
    if parts and len(parts) > 0:
        # Try to find the ZIP code (last numeric part)
        for part in reversed(parts):
            if part.isdigit() and len(part) == 5:
                return part
    # Default zip if not found
    return generate_ca_zip()

def generate_random_effective_date():
    """Generates a random effective date between start_date and end_date."""
    delta_days = (end_date - start_date).days
    random_offset = random.randint(0, delta_days)
    return start_date + timedelta(days=random_offset)

def generate_policy():
    """Generates a single auto insurance policy document with synthetic data."""
    policy = {}
    
    # Generate a unique policy ID and number
    policy['id'] = str(uuid.uuid4())
    policy['policyNumber'] = f"POL-{random.randint(10000, 99999)}"
    policy['policyType'] = "Auto"
    policy['status'] = random.choice(["Active", "Pending", "Expired", "Cancelled"])

    # Generate current timestamp for metadata
    now = datetime.now()
    policy['createdAt'] = now.isoformat()
    policy['updatedAt'] = now.isoformat()

    # Policyholder details
    ca_address = generate_ca_address()
    zip_code = extract_zip_from_address(ca_address)
    
    # Generate H3 index for policyholder address
    address_h3 = get_h3_for_zip(zip_code)
    
    policy['policyHolder'] = {
        "firstName": fake.first_name(),
        "lastName": fake.last_name(),
        "dateOfBirth": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d"),
        "address": {
            "street": fake.street_address(),
            "city": random.choice(CA_CITIES),
            "state": "CA",
            "zip": zip_code,
            "h3Index": address_h3  # Add H3 index to address
        },
        "contactInfo": {
            "phone": fake.phone_number(),
            "email": fake.email()
        }
    }
    
    # Add spatial indices at policy level for easier querying
    policy['spatialIndices'] = {
        "addressH3": address_h3,
        "garagingH3": []  # Will be filled with vehicle garaging H3 indices
    }
    
    # Insured Drivers
    num_drivers = random.randint(1, 3)
    drivers = []
    for i in range(num_drivers):
        # First driver is the policyholder
        if i == 0:
            name = f"{policy['policyHolder']['firstName']} {policy['policyHolder']['lastName']}"
            age = datetime.now().year - int(policy['policyHolder']['dateOfBirth'].split('-')[0])
        else:
            name = fake.name()
            age = random.randint(18, 80)
            
        driver = {
            "name": name,
            "age": age,
            "licenseNumber": fake.bothify(text='?######'),
            "licenseState": "CA",
            "drivingHistory": {
                "violationsLast3Years": random.randint(0, 3),
                "accidentsLast3Years": random.randint(0, 2)
            },
            "relationship": "Self" if i == 0 else random.choice(["Spouse", "Child", "Parent", "Sibling", "Other"])
        }
        drivers.append(driver)
    
    policy['drivers'] = drivers
    
    # Covered Vehicles
    num_vehicles = random.randint(1, 3)
    vehicles = []
    
    car_makes = ["Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW", "Mercedes", "Tesla", "Audi", "Lexus", "Subaru"]
    models_by_make = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
        "Ford": ["F-150", "Escape", "Explorer", "Mustang", "Focus"],
        "Chevrolet": ["Silverado", "Equinox", "Malibu", "Traverse", "Tahoe"],
        "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "Maxima"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
        "Tesla": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck"],
        "Audi": ["A4", "Q5", "A6", "Q7", "e-tron"],
        "Lexus": ["RX", "ES", "NX", "IS", "GX"],
        "Subaru": ["Outback", "Forester", "Crosstrek", "Impreza", "Legacy"]
    }
    
    for _ in range(num_vehicles):
        make = random.choice(car_makes)
        model = random.choice(models_by_make.get(make, ["Unknown"]))
        year = random.randint(2010, 2024)
        
        # For garaging ZIP, either use same as address or generate a nearby one
        if random.random() < 0.8:  # 80% chance to use same ZIP
            garaging_zip = zip_code
        else:
            garaging_zip = generate_ca_zip()
            
        # Generate H3 index for garaging address
        garaging_h3 = get_h3_for_zip(garaging_zip)
        
        # Add to the spatial indices list
        if garaging_h3 and garaging_h3 not in policy['spatialIndices']['garagingH3']:
            policy['spatialIndices']['garagingH3'].append(garaging_h3)
        
        vehicle = {
            "make": make,
            "model": model,
            "year": year,
            "vin": fake.bothify(text='?#?#?#?#?#?#?#?#?#?'),
            "primaryUse": random.choice(["Commute", "Pleasure", "Business"]),
            "annualMileage": random.randint(5000, 25000),
            "garagingZip": garaging_zip,
            "garagingH3Index": garaging_h3  # Add H3 index to vehicle
        }
        vehicles.append(vehicle)
    
    policy['vehicles'] = vehicles
    
    # Coverage details using realistic auto insurance structure
# Implementation based on the provided product model
    
    # List of all coverages - we'll select mandatory ones and randomly select optional ones
    all_coverages = []
    coverage_limits = {}
    coverage_deductibles = {}
    
    # Mandatory Liability Coverages - always include these
    all_coverages.append("Bodily Injury Liability")
    # Randomly select one of the BI options
    if random.random() < 0.6:  # 60% chance for lower option
        bi_option = "15/30"
        bi_min = 15000
        bi_max = 30000
    else:
        bi_option = "50/100"
        bi_min = 50000
        bi_max = 100000
    coverage_limits["bodilyInjury"] = f"{bi_min/1000:.0f}k/{bi_max/1000:.0f}k"
    
    all_coverages.append("Property Damage Liability")
    # Randomly select one of the PD options
    if random.random() < 0.7:  # 70% chance for lower option
        pd_value = 10000
    else:
        pd_value = 15000
    coverage_limits["propertyDamage"] = f"{pd_value/1000:.0f}k"
    
    all_coverages.append("Uninsured Motorist Bodily Injury")
    # Randomly select one of the UMBI options
    if random.random() < 0.6:  # 60% chance for lower option
        umbi_option = "15/30"
        umbi_min = 15000
        umbi_max = 30000
    else:
        umbi_option = "25/50"
        umbi_min = 25000
        umbi_max = 50000
    coverage_limits["uninsuredMotorist"] = f"{umbi_min/1000:.0f}k/{umbi_max/1000:.0f}k"
    
    # Optional Liability Coverages - randomly select these
    if random.random() < 0.6:  # 60% chance to include Underinsured Motorist
        all_coverages.append("Underinsured Motorist Bodily Injury")
        # Randomly select one of the UIMBI options
        if random.random() < 0.7:  # 70% chance for lower option
            uimbi_option = "50/100"
            uimbi_min = 50000
            uimbi_max = 100000
        else:
            uimbi_option = "100/300"
            uimbi_min = 100000
            uimbi_max = 300000
        coverage_limits["underinsuredMotorist"] = f"{uimbi_min/1000:.0f}k/{uimbi_max/1000:.0f}k"
    
    if random.random() < 0.7:  # 70% chance to include Medical Payments
        all_coverages.append("Medical Payments")
        # Randomly select one of the Medical Payment options
        med_value = random.choice([5000, 10000, 25000, 50000])
        coverage_limits["medicalPayments"] = f"{med_value/1000:.0f}k"
    
    # Optional Physical Damage Coverages - randomly select these
    if random.random() < 0.8:  # 80% chance to include Collision
        all_coverages.append("Collision")
        # Randomly select one of the Collision deductible options
        collision_deductible = random.choice([100, 250, 500])
        coverage_deductibles["collision"] = collision_deductible
    
    if random.random() < 0.75:  # 75% chance to include Comprehensive
        all_coverages.append("Comprehensive")
        # Randomly select one of the Comprehensive deductible options
        comprehensive_deductible = random.choice([100, 250, 500, 750])
        coverage_deductibles["comprehensive"] = comprehensive_deductible
    
    # Optional Add-ons
    possible_addons = [
        "Roadside Assistance",
        "Rental Car Reimbursement",
        "Gap Insurance",
        "New Car Replacement",
        "Accident Forgiveness"
    ]
    
    # Randomly select 0-3 add-ons
    selected_addons = []
    if random.random() < 0.8:  # 80% chance to have at least one add-on
        num_addons = random.randint(1, 3)
        selected_addons = random.sample(possible_addons, k=num_addons)
    
    # Construct the coverage structure
    coverage = {
        "coverages": all_coverages,
        "limits": coverage_limits,
        "deductibles": coverage_deductibles,
        "addOns": selected_addons
    }
    
    policy['coverage'] = coverage    
    
    
    # Policy dates
    effective_date = generate_random_effective_date()
    term_months = random.choice([6, 12])
    expiration_date = effective_date + relativedelta(months=term_months)
    
    policy['policyDates'] = {
        "effectiveDate": effective_date.strftime("%Y-%m-%d"),
        "expirationDate": expiration_date.strftime("%Y-%m-%d"),
        "term": f"{term_months} months"
    }
    
    # Pricing information
    base_premium = random.uniform(500, 2500)
    discounts = random.uniform(0, base_premium * 0.3)  # Up to 30% discount
    fees = random.uniform(20, 150)
    final_premium = base_premium - discounts + fees
    
    policy['pricing'] = {
        "basePremium": round(base_premium, 2),
        "discounts": round(discounts, 2),
        "fees": round(fees, 2),
        "finalPremium": round(final_premium, 2)
    }
    
    # Payment information
    policy['payment'] = {
        "method": random.choice(["Credit Card", "Bank Transfer", "Check", "Debit Card"]),
        "schedule": random.choice(["Monthly", "Quarterly", "Semi-Annual", "Annual"]),
        "autopay": random.choice([True, False])
    }
    
    return policy

# Number of policies to generate
NUM_POLICIES = 50  # Default to 50, change as needed

# Generate policy documents
print(f"Generating {NUM_POLICIES} policy documents...")
policies_to_insert = [generate_policy() for _ in range(NUM_POLICIES)]

# Insert each policy into Cosmos DB
success_count = 0
for policy in policies_to_insert:
    try:
        container.upsert_item(body=policy)
        print(f"Inserted policy: {policy['policyNumber']} - {policy['policyHolder']['address']['zip']} - H3: {policy['spatialIndices']['addressH3']}")
        success_count += 1
    except Exception as e:
        print(f"Error inserting policy {policy['policyNumber']}: {str(e)}")

print(f"Successfully inserted {success_count} of {NUM_POLICIES} policies.")
print("Policy generation complete.")