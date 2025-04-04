import os
import sys
import json
import subprocess
import tempfile
from autogen import AssistantAgent
from config import config_list_gpt4o

def create_hera_agent():
    """Create and return the Hera (Customer Profiling) agent"""
    return AssistantAgent(
        name="Hera (CustomerProfilingAgent)",
        system_message="""
You are HERA, the customer profiling and recommendation agent in the insurance policy workflow.

Your job is to:
1. Analyze customer data at different workflow stages (Iris, Mnemosyne, Ares)
2. Suggest coverage options based on similar existing customers
3. Display the top 3 matches with detailed coverage information
4. Provide recommendations without interrupting the main workflow

When receiving customer data, you will:
1. Process the data using customerprofile.py with GPT-4o
2. Generate embeddings using text-embedding-3-large
3. Find similar customer profiles via cosine similarity
4. Present the top 3 matching profiles at each workflow stage
5. Highlight the most common coverage options from these matches

Format your response with clear sections:
- STAGE INFORMATION (which stage data was collected from)
- TOP 3 MATCHING PROFILES (with similarity scores and policy details)
- RECOMMENDED COVERAGES (based on frequency in similar profiles)
- SUGGESTED NEXT STEPS (what the customer might consider)

Your recommendations will improve as you receive more data at later workflow stages.
""",
        llm_config={"config_list": config_list_gpt4o}
    )

def get_profile_recommendations(customer_data, workflow_stage="unknown"):
    """
    Get profile-based recommendations at different workflow stages
    
    Args:
        customer_data (dict): Customer profile data collected at the current stage
        workflow_stage (str): Current stage in the workflow (iris/mnemosyne/ares)
        
    Returns:
        dict: Recommendations and suggestion
    """
    # Create a log prefix for debugging
    log_prefix = f"[HERA - {workflow_stage.upper()}]"
    print(f"{log_prefix} Processing customer data...")
    
    try:
        # Format the data according to current workflow stage
        formatted_data = format_data_for_stage(customer_data, workflow_stage)
        
        # Call customerprofile.py and get the output
        print(f"{log_prefix} Calling customerprofile.py for similarity analysis...")
        profile_output = call_customerprofile(formatted_data)
        
        # Parse the output to extract the top 3 matches
        print(f"{log_prefix} Parsing results...")
        recommendations = parse_customerprofile_output(profile_output)
        
        # Add the workflow stage information
        result = {
            "stage": workflow_stage,
            "recommendations": recommendations,
            "suggestion": generate_suggestion(recommendations, workflow_stage),
            "proceed": True  # Always allow workflow to continue
        }
        
        # Display results to user
        display_results_to_user(result, workflow_stage)
        
        return result
    except Exception as e:
        print(f"{log_prefix} Error: {str(e)}")
        # Return a minimal response that allows continuation on error
        return {
            "stage": workflow_stage,
            "error": str(e),
            "recommendations": [],
            "suggestion": "Unable to generate recommendations. Proceed with standard process.",
            "proceed": True
        }

def format_data_for_stage(data, stage):
    """
    Format data based on the current workflow stage
    
    Args:
        data (dict): Raw data from the workflow stage
        stage (str): Current workflow stage
    
    Returns:
        dict: Formatted data optimized for customerprofile.py
    """
    formatted_data = {}
    
    # Format based on stage
    if stage.lower() == "iris":
        # Basic customer information and demographic data
        formatted_data = {
            "customerInfo": data.get("basic_info", {}),
            "stage": "iris"
        }
    elif stage.lower() == "mnemosyne":
        # Full customer profile with preferences
        formatted_data = {
            "customerProfile": data.get("customerProfile", {}),
            "stage": "mnemosyne"
        }
    elif stage.lower() == "ares":
        # Customer profile with additional risk information
        formatted_data = {
            "customerProfile": data.get("customerProfile", {}),
            "riskAssessment": data.get("risk_info", {}),
            "stage": "ares" 
        }
    else:
        # Unknown stage, just pass the data as is
        formatted_data = data
    
    return formatted_data

def call_customerprofile(customer_data):
    """
    Call customerprofile.py with customer data
    
    Args:
        customer_data (dict): Customer profile data in JSON format
        
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

def parse_customerprofile_output(output):
    """
    Parse the output from customerprofile.py to extract recommendations
    
    Args:
        output (str): Output from customerprofile.py
        
    Returns:
        list: List of recommendation dictionaries
    """
    recommendations = []
    
    # Find all matches in the output
    match_sections = output.split("--- MATCH #")
    
    for i in range(1, min(4, len(match_sections))):  # Up to 3 matches
        section = match_sections[i]
        
        # Extract similarity score
        similarity = 0.0
        similarity_match = section.split("(Similarity: ")
        if len(similarity_match) > 1:
            similarity_str = similarity_match[1].split(")")[0]
            try:
                similarity = float(similarity_str)
            except:
                pass
                
        # Extract policy number
        policy_number = "Unknown"
        policy_match = section.split("Policy Number: ")
        if len(policy_match) > 1:
            policy_number = policy_match[1].split("\n")[0].strip()
            
        # Extract coverages
        coverages = []
        coverage_match = section.split("Coverages: ")
        if len(coverage_match) > 1:
            coverages_str = coverage_match[1].split("\n")[0].strip()
            coverages = [c.strip() for c in coverages_str.split(",")]
            
        # Extract add-ons
        addons = []
        addon_match = section.split("Add-ons: ")
        if len(addon_match) > 1:
            addons_str = addon_match[1].split("\n")[0].strip()
            addons = [a.strip() for a in addons_str.split(",")]
            
        # Extract premium
        premium = None
        premium_match = section.split("Premium: $")
        if len(premium_match) > 1:
            try:
                premium = float(premium_match[1].split("\n")[0].strip())
            except:
                pass
                
        # Extract limits
        limits = {}
        limits_section = None
        if "Key Limits:" in section:
            limits_section = section.split("Key Limits:")[1].split("Deductibles:")[0] if "Deductibles:" in section else section.split("Key Limits:")[1]
            for line in limits_section.split("\n"):
                if ":" in line and line.strip():
                    parts = line.strip().split(":", 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    if key:
                        limits[key] = value
        
        # Extract deductibles
        deductibles = {}
        deductibles_section = None
        if "Deductibles:" in section:
            deductibles_section = section.split("Deductibles:")[1]
            for line in deductibles_section.split("\n"):
                if ":" in line and line.strip():
                    parts = line.strip().split(":", 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    if key:
                        deductibles[key] = value
                
        # Create recommendation object
        recommendation = {
            "rank": i,
            "similarity": similarity,
            "policyNumber": policy_number,
            "coverages": coverages,
            "addOns": addons
        }
        
        if premium is not None:
            recommendation["premium"] = premium
            
        if limits:
            recommendation["limits"] = limits
            
        if deductibles:
            recommendation["deductibles"] = deductibles
            
        recommendations.append(recommendation)
    
    return recommendations

def generate_suggestion(recommendations, stage):
    """
    Generate a suggestion based on the recommendations and current stage
    
    Args:
        recommendations (list): List of recommendation dictionaries
        stage (str): Current workflow stage
        
    Returns:
        str: Suggestion text
    """
    if not recommendations:
        return f"No similar customer profiles found at the {stage} stage. Consider standard coverage options."
        
    # Extract most common coverages from top matches
    all_coverages = []
    for rec in recommendations:
        all_coverages.extend(rec.get("coverages", []))
        
    coverage_counts = {}
    for coverage in all_coverages:
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
        
    # Sort coverages by frequency
    common_coverages = sorted(coverage_counts.keys(), key=lambda k: coverage_counts[k], reverse=True)
    top_coverages = common_coverages[:min(5, len(common_coverages))]
    
    # Generate stage-specific suggestions
    stage_prefix = {
        "iris": "Based on initial customer information,",
        "mnemosyne": "With the complete customer profile,",
        "ares": "Considering both profile and risk assessment,"
    }
    
    prefix = stage_prefix.get(stage.lower(), "Based on similar customer profiles,")
    
    # Generate suggestion
    suggestion = f"{prefix} consider these popular coverages: {', '.join(top_coverages)}."
    
    # Add premium range if available
    premiums = [rec.get("premium") for rec in recommendations if "premium" in rec]
    if premiums:
        avg_premium = sum(premiums) / len(premiums)
        suggestion += f" Typical premium range: ${min(premiums):.2f} - ${max(premiums):.2f}, averaging ${avg_premium:.2f}."
    
    return suggestion

def display_results_to_user(result, stage):
    """
    Display the results to the user in a friendly format
    
    Args:
        result (dict): The recommendation result
        stage (str): The workflow stage
    """
    recommendations = result.get("recommendations", [])
    
    print("\n" + "="*80)
    print(f"  HERA RECOMMENDATIONS ({stage.upper()} STAGE)")
    print("="*80)
    
    if not recommendations:
        print("\nNo similar customer profiles found at this stage.")
        print("\nPlease continue with standard coverage options.")
    else:
        # Display summary suggestion
        print(f"\n{result.get('suggestion')}")
        
        # Display each recommendation
        print(f"\nTOP {len(recommendations)} MATCHING PROFILES:")
        for i, rec in enumerate(recommendations):
            print(f"\n--- MATCH #{i+1} (Similarity: {rec['similarity']:.4f}) ---")
            print(f"Policy Number: {rec['policyNumber']}")
            
            if rec.get("coverages"):
                print(f"Coverages: {', '.join(rec['coverages'])}")
            
            if rec.get("addOns"):
                print(f"Add-ons: {', '.join(rec['addOns'])}")
            
            if rec.get("premium"):
                print(f"Premium: ${rec['premium']:.2f}")
            
            if rec.get("limits"):
                print("\nKey Limits:")
                for k, v in rec["limits"].items():
                    print(f"  {k}: {v}")
            
            if rec.get("deductibles"):
                print("\nDeductibles:")
                for k, v in rec["deductibles"].items():
                    print(f"  {k}: {v}")
    
    print("\n" + "="*80)
    print("  Continue with workflow for complete coverage design")
    print("="*80 + "\n")