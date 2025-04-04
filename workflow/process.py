import datetime
import random as random_module  # Rename the import to avoid potential shadowing.Also needed for policy number generation
import json
from agents import initialize_agents
from agents.hera import get_profile_recommendations



from db.cosmos_db import (
    save_policy_checkpoint, 
    get_mandatory_questions, 
    save_underwriting_responses,
    confirm_policy
)
from utils.helpers import read_customer_data_from_file, show_current_status_and_confirm, extract_json_content
# Add this function to your workflow process.py
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
    
    # Call Hera to get recommendations based on the current stage
    recommendations = get_profile_recommendations(current_state, stage)
    
    # Store recommendations in the workflow state
    if "hera_recommendations" not in current_state:
        current_state["hera_recommendations"] = {}
    
    current_state["hera_recommendations"][stage] = recommendations
    
    return current_state
def call_customerprofile(customer_data):
    """
    Call customerprofile.py with customer data
    
    Args:
        customer_data (dict): Customer profile data
        
    Returns:
        str: Output from customerprofile.py execution
    """
    try:
        # Create a temporary file to store the customer data
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
            temp_file = temp.name
            json.dump(customer_data, temp)
        
        # Call customerprofile.py with the temp file
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "customerprofile.py")
        result = subprocess.run(
            [sys.executable, script_path, "--input", temp_file],
            capture_output=True,
            text=True
        )
        
        # Clean up the temp file
        try:
            os.remove(temp_file)
        except:
            pass
        
        return result.stdout
    except Exception as e:
        return f"Error calling customerprofile.py: {str(e)}"

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
def design_coverage_with_demeter(current_state, demeter, iris, user_proxy):
    """Design coverage using Demeter's language capabilities"""
    print("\n=== COVERAGE DESIGN WITH DEMETER ===")
    print("Retrieving coverage options from autopm container...")
    
    # Initialize the Cosmos DB connection if not already done
    from db.cosmos_db import init_cosmos_db, get_coverage_with_demeter
    init_cosmos_db()  # Ensure DB connection is initialized
    
    # Have Demeter analyze the coverage options
    coverage_data = get_coverage_with_demeter(demeter)
    
    if not coverage_data or "coverageCategories" not in coverage_data:
        print("❌ Failed to extract coverage options. Using default coverage design.")
        # Add your fallback logic here
        return default_coverage_design(current_state, demeter)
    
    # Initialize the coverage structure
    selected_coverages = {
        "coverages": [],
        "limits": {},
        "deductibles": {},
        "exclusions": ["Racing", "Intentional damage"],
        "addOns": []
    }
    
    # Have Iris introduce the coverage design process
    iris_prompt = """
    Introduce the coverage selection process to the customer. Explain:
    1. They will select coverages for their policy
    2. Some coverages are mandatory for legal reasons
    3. For each coverage, they'll need to select limits or deductibles
    4. The process will walk them through each coverage category
    
    Keep your introduction brief but friendly.
    """
    
    iris_response = iris.generate_reply(messages=[{"role": "user", "content": iris_prompt}])
    if hasattr(iris_response, 'content'):
        print(f"\n[Iris]: {iris_response.content}")
    else:
        print(f"\n[Iris]: {iris_response}")
        
    # Process each coverage category using Demeter's intelligence
    for category in coverage_data.get("coverageCategories", []):
        category_name = category.get("name", "Unknown Category")
        category_description = category.get("description", "")
        
        # Have Demeter explain the category
        demeter_prompt = f"""
        Explain this insurance coverage category to the customer:
        Category: {category_name}
        Description: {category_description}
        
        Keep your explanation concise, conversational and valuable.
        """
        
        demeter_response = demeter.generate_reply(messages=[{"role": "user", "content": demeter_prompt}])
        category_explanation = demeter_response if isinstance(demeter_response, str) else getattr(demeter_response, 'content', str(demeter_response))
        
        print(f"\n=== {category_name.upper()} ===")
        print(f"{category_explanation}")
        
        # Process each coverage in the category
        for coverage in category.get("coverages", []):
            coverage_name = coverage.get("name", "Unknown Coverage")
            is_mandatory = coverage.get("mandatory", False)
            explanation = coverage.get("explanation", "")
            term_name = coverage.get("termName", "")
            model_type = coverage.get("modelType", "")
            
            print(f"\n📋 COVERAGE: {coverage_name}")
            print(f"{explanation}")
            
            # Handle mandatory vs. optional coverages
            if is_mandatory:
                print(f"⚠️ Note: {coverage_name} is mandatory and will be automatically included in your policy.")
                user_choice = "yes"
            else:
                # Ask if user wants this coverage
                user_choice = user_proxy.get_human_input(f"Would you like to include {coverage_name}? (yes/no): ").strip().lower()
            
            # Process user's coverage choice
            if user_choice == "yes" or is_mandatory:
                # Add to selected coverages
                selected_coverages["coverages"].append(coverage_name)
                
                # Present options for the coverage if available
                options = coverage.get("options", [])
                if options:
                    print(f"\nSelect {term_name} for {coverage_name}:")
                    
                    # Have Demeter explain the options
                    options_prompt = f"""
                    Explain these {term_name} options for {coverage_name} to the customer in an easy-to-understand way:
                    {json.dumps(options, indent=2)}
                    
                    Explain how these choices affect their coverage and premium.
                    """
                    
                    demeter_options_response = demeter.generate_reply(messages=[{"role": "user", "content": options_prompt}])
                    options_explanation = demeter_options_response if isinstance(demeter_options_response, str) else getattr(demeter_options_response, 'content', str(demeter_options_response))
                    
                    print(f"\n{options_explanation}")
                    
                    # Display options
                    for i, option in enumerate(options):
                        label = option.get("label", "")
                        display = option.get("display", "")
                        
                        print(f"{i+1}. {label} - {display}")
                    
                    # Get user selection
                    while True:
                        try:
                            option_choice = int(user_proxy.get_human_input(f"Enter option number (1-{len(options)}): "))
                            if 1 <= option_choice <= len(options):
                                break
                            print(f"Please enter a number between 1 and {len(options)}")
                        except ValueError:
                            print("Please enter a valid number")
                    
                    selected_option = options[option_choice - 1]
                    
                    # Store the selected limit or deductible
                    option_key = coverage_name.lower().replace(" ", "_")
                    if model_type.lower() == "limit":
                        if "min" in selected_option and "max" in selected_option:
                            selected_coverages["limits"][option_key] = {
                                "per_person": selected_option.get("min"),
                                "per_accident": selected_option.get("max"),
                                "label": selected_option.get("label")
                            }
                        else:
                            selected_coverages["limits"][option_key] = {
                                "amount": selected_option.get("value", 0),
                                "label": selected_option.get("label")
                            }
                    elif model_type.lower() == "deductible":
                        selected_coverages["deductibles"][option_key] = {
                            "amount": selected_option.get("value", 0),
                            "label": selected_option.get("label")
                        }
    
    # Ask about add-ons using Demeter's reasoning
    print("\n=== OPTIONAL ADD-ONS ===")
    
    addon_prompt = """
    Suggest 3 common insurance add-ons that customers often find valuable for auto insurance:
    1. Describe each add-on in a sentence
    2. Explain its benefit
    3. Format your response as a list
    
    Return in this format:
    ```json
    [
      {"name": "Roadside Assistance", "description": "Provides emergency services like towing and battery jumps."},
      {"name": "Rental Coverage", "description": "Pays for a rental car while yours is being repaired."},
      {"name": "Gap Insurance", "description": "Covers the difference between your car's value and what you owe."}
    ]
    ```
    """
    
    demeter_addons_response = demeter.generate_reply(messages=[{"role": "user", "content": addon_prompt}])
    response_content = demeter_addons_response if isinstance(demeter_addons_response, str) else getattr(demeter_addons_response, 'content', str(demeter_addons_response))
    
    # Extract add-ons JSON
    try:
        json_start = response_content.find("```json")
        json_end = response_content.rfind("```")
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_content[json_start + 7:json_end].strip()
            add_ons = json.loads(json_content)
        else:
            # Fallback add-ons
            add_ons = [
                {"name": "Roadside Assistance", "description": "Provides help with emergencies like flat tires, lockouts, and towing"},
                {"name": "Rental Car Coverage", "description": "Covers the cost of a rental car if your vehicle is being repaired after a covered accident"},
                {"name": "Gap Insurance", "description": "Covers the difference between what you owe on your vehicle and its actual cash value if it's totaled"}
            ]
    except:
        # Fallback add-ons
        add_ons = [
            {"name": "Roadside Assistance", "description": "Provides help with emergencies like flat tires, lockouts, and towing"},
            {"name": "Rental Car Coverage", "description": "Covers the cost of a rental car if your vehicle is being repaired after a covered accident"},
            {"name": "Gap Insurance", "description": "Covers the difference between what you owe on your vehicle and its actual cash value if it's totaled"}
        ]
    
    # Present add-ons to user
    for add_on in add_ons:
        add_on_choice = user_proxy.get_human_input(f"\nAdd {add_on['name']}? ({add_on['description']}) (yes/no): ").strip().lower()
        if add_on_choice == "yes":
            selected_coverages["addOns"].append(add_on["name"])
    
    # Have Demeter summarize the coverage
    summary_prompt = f"""
    Create a concise summary of the customer's selected coverage:
    
    Selected Coverages: {", ".join(selected_coverages["coverages"])}
    Limits: {json.dumps(selected_coverages["limits"])}
    Deductibles: {json.dumps(selected_coverages["deductibles"])}
    Add-ons: {", ".join(selected_coverages["addOns"]) if selected_coverages["addOns"] else "None"}
    
    Format this as a friendly, conversational summary explaining what's covered.
    """
    
    demeter_summary = demeter.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
    summary = demeter_summary if isinstance(demeter_summary, str) else getattr(demeter_summary, 'content', str(demeter_summary))
    
    print("\n=== COVERAGE SUMMARY ===")
    print(summary)
    
    return selected_coverages

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

def process_insurance_request(customer_file=None):
    """
    Master workflow to process an insurance request from intake to issuance.
    Each step uses the function-call feature by sending structured messages to agents.
    Progress is checkpointed after each step.
    
    Args:
        customer_file: Optional path to a file containing customer data.
    """
    print("=== Insurance Policy Workflow Initiated ===")
    
    # Initialize agents
    agents = initialize_agents()
    iris = agents["iris"]
    mnemosyne = agents["mnemosyne"]
    ares = agents["ares"]
    hera = agents["hera"]  # Make sure Hera is included
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
    user_proxy = agents["user_proxy"]  # ADD THIS LINE
    
        # Initialize state
    current_state = {
    "status": "Initiated",
    "timestamp": datetime.datetime.now().isoformat(),
    "quoteId": f"QUOTE{random_module.randint(10000, 99999)}"  # Use random_module instead
    }
    file_data = None
    if customer_file:
        print(f"Reading customer data from {customer_file}...")
        file_data = read_customer_data_from_file(customer_file)
        if not file_data:
            print("Could not read customer data from file. Starting with manual intake.")
        else:
            current_state["file_data"] = file_data
    
    # Step 1: Basic Customer Profile using Iris (only basic info)
    if show_current_status_and_confirm(current_state, "Collect basic customer information with Iris"):
        intake_prompt = """Collect ONLY the following basic customer information:
- Name
- Date of birth
- Address
- Contact information (phone, email)

DO NOT collect vehicle details or driving history at this stage.
If provided in the data file, extract ONLY the basic customer info listed above.

IMPORTANT: Return your response as valid JSON format with all extracted information.
"""
        prompt_with_data = f"{intake_prompt}\nCustomer Data Provided: {json.dumps(current_state.get('file_data', {}))}"
        
        # Send to Iris and get response
        response = iris.generate_reply(messages=[{"role": "user", "content": prompt_with_data}])
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Display the extracted information
        print("\n=== Basic Customer Information Extracted ===")
        print(response_content)
        print("==========================================")
        
        # Extract JSON content from response
        basic_profile = extract_json_content(response_content)
        if not isinstance(basic_profile, dict):
            print("[Iris] Failed to parse JSON response. Creating profile manually.")
            basic_profile = {
                "name": input("Enter customer name: ").strip(),
                "dob": input("Enter date of birth (YYYY-MM-DD): ").strip(),
                "address": input("Enter address: ").strip(),
                "contact": {"phone": input("Enter phone number: ").strip(), 
                           "email": input("Enter email address: ").strip()}
            }
        
        # Show and confirm
        print("\n=== Basic Profile Information ===")
        print(f"Name: {basic_profile.get('name', 'Not provided')}")
        print(f"Date of Birth: {basic_profile.get('dob', 'Not provided')}")
        print(f"Address: {basic_profile.get('address', 'Not provided')}")
        if isinstance(basic_profile.get('contact'), dict):
            print(f"Phone: {basic_profile.get('contact', {}).get('phone', 'Not provided')}")
            print(f"Email: {basic_profile.get('contact', {}).get('email', 'Not provided')}")
        
        confirmation = input("\nDo you confirm these details? (yes/no): ").strip().lower()
        current_state = process_with_hera(current_state, "iris")
        if confirmation != "yes":
            print("[Iris] Updating basic profile information.")
            basic_profile["name"] = input("Enter customer name: ").strip() or basic_profile.get("name", "")
            basic_profile["dob"] = input("Enter date of birth (YYYY-MM-DD): ").strip() or basic_profile.get("dob", "")
            basic_profile["address"] = input("Enter address: ").strip() or basic_profile.get("address", "")
            if "contact" not in basic_profile:
                basic_profile["contact"] = {}
            basic_profile["contact"]["phone"] = input("Enter phone number: ").strip() or basic_profile.get("contact", {}).get("phone", "")
            basic_profile["contact"]["email"] = input("Enter email address: ").strip() or basic_profile.get("contact", {}).get("email", "")
        
        current_state["customerProfile"] = basic_profile
        save_policy_checkpoint(current_state, "basic_profile_completed")
    else:
        print("Workflow halted at basic profile stage.")
        return
    
    # Step 2: Detailed Profile using Mnemosyne (vehicle details, driving history)
    if show_current_status_and_confirm(current_state, "Collect detailed vehicle and driving information with Mnemosyne"):
        details_prompt = """Now collect detailed vehicle and driving information:
- Vehicle details (Make, Model, Year, VIN)
- Driving history (violations, accidents)
- Coverage preferences

Add this information to the basic customer profile already collected.
Return the complete customer profile as JSON.
"""
        # Include the basic profile in the prompt
        prompt_with_profile = f"{details_prompt}\nBasic Profile: {json.dumps(current_state.get('customerProfile', {}))}\nAdditional Data: {json.dumps(current_state.get('file_data', {}))}"
        
        response = mnemosyne.generate_reply(messages=[{"role": "user", "content": prompt_with_profile}])
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Display extracted information
        print("\n=== Detailed Profile Information Extracted ===")
        print(response_content)
        print("==============================================")
        
        # Extract JSON content from response
        detailed_profile = extract_json_content(response_content)
        if not isinstance(detailed_profile, dict):
            print("[Mnemosyne] Failed to parse JSON response. Creating detailed profile manually.")
            # Start with basic profile
            detailed_profile = current_state["customerProfile"].copy()
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
        
        # Show and confirm
        print("\n=== Detailed Vehicle & Driving Information ===")
        vehicle = detailed_profile.get('vehicle_details', {})
        if isinstance(vehicle, dict):
            print(f"Make: {vehicle.get('make', 'Not provided')}")
            print(f"Model: {vehicle.get('model', 'Not provided')}")
            print(f"Year: {vehicle.get('year', 'Not provided')}")
            print(f"VIN: {vehicle.get('vin', 'Not provided')}")
        
        driving = detailed_profile.get('driving_history', {})
        if isinstance(driving, dict):
            print(f"Violations: {driving.get('violations', 'Not provided')}")
            print(f"Accidents: {driving.get('accidents', 'Not provided')}")
            print(f"Years Licensed: {driving.get('years_licensed', 'Not provided')}")
        
        coverages = detailed_profile.get('coverage_preferences', [])
        if isinstance(coverages, list) and coverages:
            print("Coverage Preferences:")
            for coverage in coverages:
                print(f"  - {coverage}")
        
        confirmation = input("\nDo you confirm these details? (yes/no): ").strip().lower()
        if confirmation != "yes":
            print("[Mnemosyne] Updating vehicle and driving information.")
            if "vehicle_details" not in detailed_profile:
                detailed_profile["vehicle_details"] = {}
            detailed_profile["vehicle_details"]["make"] = input("Enter vehicle make: ").strip() or detailed_profile.get("vehicle_details", {}).get("make", "")
            detailed_profile["vehicle_details"]["model"] = input("Enter vehicle model: ").strip() or detailed_profile.get("vehicle_details", {}).get("model", "")
            detailed_profile["vehicle_details"]["year"] = input("Enter vehicle year: ").strip() or detailed_profile.get("vehicle_details", {}).get("year", "")
            detailed_profile["vehicle_details"]["vin"] = input("Enter VIN: ").strip() or detailed_profile.get("vehicle_details", {}).get("vin", "")
            
            if "driving_history" not in detailed_profile:
                detailed_profile["driving_history"] = {}
            detailed_profile["driving_history"]["violations"] = input("Enter number of violations: ").strip() or detailed_profile.get("driving_history", {}).get("violations", "")
            detailed_profile["driving_history"]["accidents"] = input("Enter number of accidents: ").strip() or detailed_profile.get("driving_history", {}).get("accidents", "")
            detailed_profile["driving_history"]["years_licensed"] = input("Enter years licensed: ").strip() or detailed_profile.get("driving_history", {}).get("years_licensed", "")
            
            coverage_input = input("Enter coverage preferences (comma separated): ").strip()
            
            if coverage_input:
                detailed_profile["coverage_preferences"] = [c.strip() for c in coverage_input.split(",")]
            
        # Update profile with detailed information
        current_state["customerProfile"] = detailed_profile
        current_state = process_with_hera(current_state, "mnemosyne")
        save_policy_checkpoint(current_state, "detailed_profile_completed")
    else:
        print("Workflow halted at detailed profile stage.")
        return
    # MOVE UNDERWRITING HERE - Create a new Step 2.5: Underwriting Verification
    if show_current_status_and_confirm(current_state, "Perform underwriting verification"):
        # Run underwriting questions
        if not build_customer_profile(current_state, iris, mnemosyne, user_proxy):
            # Exit workflow if underwriting fails
            save_policy_checkpoint(current_state, "underwriting_failed")
            return
        current_state = process_with_hera(current_state, "mnemosyne")
        save_policy_checkpoint(current_state, "underwriting_completed")
    else:
        print("Workflow halted at underwriting verification stage.")
        return





    # Step 3: Assess Risk using Ares
    if show_current_status_and_confirm(current_state, "Assess risk factors with Ares"):
        risk_prompt = f"Call: assess_risk; Params: {json.dumps({'profile': current_state['customerProfile']})}"
        response = ares.generate_reply(messages=[{"role": "user", "content": risk_prompt}])
        risk_info = extract_json_content(response.content) if hasattr(response, 'content') else None
        
        if not isinstance(risk_info, dict) or "riskScore" not in risk_info:
            print("[Ares] Could not parse response. Defaulting risk score.")
            risk_info = {"riskScore": 5.0, "riskFactors": ["Default risk assessment"]}
        
        current_state["risk_info"] = risk_info
        current_state = process_with_hera(current_state, "mnemosyne")
        save_policy_checkpoint(current_state, "risk_assessment_completed")
        print(f"[Ares] Risk evaluation completed. Risk Score: {risk_info.get('riskScore', 'N/A')}")
    else:
        print("Workflow halted at risk assessment stage.")
        return
    
    # Step 4: Design Coverage using Demeter
    if show_current_status_and_confirm(current_state, "Design coverage model with Demeter"):
        # Use the new LLM-powered coverage design function
        coverage = design_coverage_with_demeter(current_state, demeter, iris, user_proxy)
        
        current_state["coverage"] = coverage
        save_policy_checkpoint(current_state, "coverage_design_completed")
        
        print("[Demeter] Coverage model designed and saved.")
    else:
        print("Workflow halted at coverage design stage.")
        return
    
    # Step 5: Draft Policy using Apollo
    if show_current_status_and_confirm(current_state, "Draft policy document with Apollo"):
        draft_prompt = f"Call: draft_policy; Params: {json.dumps({'coverage': current_state['coverage']})}"
        response = apollo.generate_reply(messages=[{"role": "user", "content": draft_prompt}])
        draft = response.content.strip() if hasattr(response, 'content') else "Standard policy language."
        
        current_state["policyDraft"] = draft
        save_policy_checkpoint(current_state, "policy_draft_completed")
        print("[Apollo] Policy draft prepared.")
    else:
        print("Workflow halted at policy drafting stage.")
        return
    
    # Step 6: Polish Document using Calliope
    if show_current_status_and_confirm(current_state, "Polish policy document with Calliope"):
        polish_prompt = f"Call: polish_document; Params: {json.dumps({'draft': current_state['policyDraft']})}"
        response = calliope.generate_reply(messages=[{"role": "user", "content": polish_prompt}])
        polished = response.content.strip() if hasattr(response, 'content') else current_state["policyDraft"]
        
        current_state["policyDraft"] = polished
        save_policy_checkpoint(current_state, "document_polished_completed")
        print("[Calliope] Policy document finalized.")
    else:
        print("Workflow halted at document polishing stage.")
        return
    
    # Step 7: Calculate Pricing using Plutus
    if show_current_status_and_confirm(current_state, "Calculate pricing with Plutus"):
        pricing_prompt = f"Call: calculate_pricing; Params: {json.dumps({'coverage': current_state['coverage'], 'profile': current_state['customerProfile']})}"
        response = plutus.generate_reply(messages=[{"role": "user", "content": pricing_prompt}])
        pricing = extract_json_content(response.content) if hasattr(response, 'content') else None
        
        if not isinstance(pricing, dict):
            print("[Plutus] Could not parse response. Using default pricing.")
            pricing = {"basePremium": 750, "finalPremium": 825.50}
        
        current_state["pricing"] = pricing
        save_policy_checkpoint(current_state, "pricing_completed")
        print(f"[Plutus] Pricing computed. Final premium: ${pricing.get('finalPremium', 'N/A')}")
    else:
        print("Workflow halted at pricing calculation stage.")
        return
    
    # Step 8: Generate Quote using Tyche
    if show_current_status_and_confirm(current_state, "Generate formal quote with Tyche"):
        quote_prompt = f"Call: generate_quote; Params: {json.dumps({'pricing': current_state['pricing']})}"
        response = tyche.generate_reply(messages=[{"role": "user", "content": quote_prompt}])
        quote = response.content.strip() if hasattr(response, 'content') else f"Final premium: ${current_state['pricing'].get('finalPremium', 0)}"
        
        current_state["quote"] = quote
        save_policy_checkpoint(current_state, "quote_generated_completed")
        print("[Tyche] Quote generated.")
    else:
        print("Workflow halted at quote generation stage.")
        return
    
    # Step 9: Present Policy using Orpheus
    if show_current_status_and_confirm(current_state, "Present policy to customer with Orpheus"):
        present_prompt = f"Call: present_policy; Params: {json.dumps({'document': current_state['policyDraft'], 'quote': current_state['quote']})}"
        response = orpheus.generate_reply(messages=[{"role": "user", "content": present_prompt}])
        presentation = response.content.strip() if hasattr(response, 'content') else "Presentation details unavailable."
        
        current_state["presentation"] = presentation
        save_policy_checkpoint(current_state, "presentation_completed")
        print("[Orpheus] Policy proposal presented to customer.")
    else:
        print("Workflow halted at customer presentation stage.")
        return
    
    # Step 10: Internal Approval & Regulatory Review using Hestia and Dike
    if show_current_status_and_confirm(current_state, "Perform internal approval with Hestia and regulatory review with Dike"):
        # Internal approval
        internal_prompt = f"Call: internal_approval; Params: {json.dumps({'document': current_state['policyDraft'], 'pricing': current_state['pricing']})}"
        response = hestia.generate_reply(messages=[{"role": "user", "content": internal_prompt}])
        internal = extract_json_content(response.content) if hasattr(response, 'content') else {"approved": True}
        
        # Regulatory review
        regulatory_prompt = f"Call: regulatory_review; Params: {json.dumps({'document': current_state['policyDraft']})}"
        response = dike.generate_reply(messages=[{"role": "user", "content": regulatory_prompt}])
        compliance = extract_json_content(response.content) if hasattr(response, 'content') else {"compliance": True}
        
        current_state["internal_approval"] = internal
        current_state["compliance"] = compliance
        save_policy_checkpoint(current_state, "internal_review_completed")
        
        # Check if both approved
        if internal.get("approved", False) and compliance.get("compliance", False):
            print("[Hestia & Dike] Internal approval and regulatory compliance confirmed.")
        else:
            print("WARNING: Policy did not pass internal approval or regulatory compliance.")
            if not internal.get("approved", False):
                print(f"Internal approval issues: {internal.get('reasons', 'No specific reason provided')}")
            if not compliance.get("compliance", False):
                print(f"Regulatory compliance issues: {compliance.get('issues', 'No specific issues provided')}")
            
            continue_anyway = input("Continue despite approval issues? (yes/no): ").strip().lower()
            if continue_anyway != "yes":
                print("Workflow halted due to approval issues.")
                return
    else:
        print("Workflow halted at internal approval stage.")
        return
    
    # Step 11: Customer Approval
    approval_input = input("\nDo you approve the presented policy and quote? (yes/no): ").strip().lower()
    if approval_input != "yes":
        print("Policy creation halted per customer decision.")
        return
    
    # Step 12: Issue Policy using Eirene
    if show_current_status_and_confirm(current_state, "Issue final policy with Eirene"):
        issuance_prompt = f"Call: issue_policy; Params: {json.dumps({'document': current_state['policyDraft'], 'pricing': current_state['pricing']})}"
        response = eirene.generate_reply(messages=[{"role": "user", "content": issuance_prompt}])
        issuance = extract_json_content(response.content) if hasattr(response, 'content') else None
        
        if not isinstance(issuance, dict):
            
            
            issuance = {
                "policyNumber": f"POL{random_module.randint(100000, 999999)}",
                "startDate": datetime.datetime.now().strftime("%Y-%m-%d"),
                "endDate": (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            }
        
        current_state["issuance"] = issuance
        save_policy_checkpoint(current_state, "policy_issued_completed")
        print(f"[Eirene] Policy issued with policy number: {issuance.get('policyNumber', 'Unknown')}")
    else:
        print("Workflow halted at policy issuance stage.")
        return
    
    # Step 13: Set Up Monitoring using Themis
    if show_current_status_and_confirm(current_state, "Set up policy monitoring with Themis"):
        monitor_prompt = f"Call: monitor_policy; Params: {json.dumps({'policyNumber': current_state['issuance'].get('policyNumber', 'Unknown')})}"
        response = themis.generate_reply(messages=[{"role": "user", "content": monitor_prompt}])
        monitoring = extract_json_content(response.content) if hasattr(response, 'content') else {"monitoringStatus": "Active"}
        
        current_state["monitoring"] = monitoring
        save_policy_checkpoint(current_state, "monitoring_setup_completed")
        print("[Themis] Policy monitoring setup completed.")
    else:
        print("Workflow halted at policy monitoring stage.")
        return
    
    # Finalize & Confirm the Policy
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
    
    # Step 14: Generate Final Report using Zeus
    if show_current_status_and_confirm(current_state, "Generate final policy summary with Zeus"):
        summary_prompt = f"Call: summarize_policy; Params: {json.dumps(final_policy)}"
        response = zeus.generate_reply(messages=[{"role": "user", "content": summary_prompt}])
        final_summary = extract_json_content(response.content) if hasattr(response, 'content') else final_policy
        
        save_policy_checkpoint(current_state, "workflow_completed")
        
        print("\n=== Final Policy Summary ===")
        if isinstance(final_summary, dict):
            try:
                print(json.dumps(final_summary, indent=2))
            except:
                print(str(final_summary))
        else:
            print(final_summary)
    else:
        print("Workflow halted at final reporting stage.")
        return
    
    print("\n=== Insurance Policy Creation Workflow Completed Successfully ===")
    print(f"Policy Number: {final_policy.get('issuance', {}).get('policyNumber', 'Unknown')}")
    print(f"Final Premium: ${final_policy.get('pricing', {}).get('finalPremium', 'Unknown')}")
    print("Thank you for using our service!")
    
    return final_summary