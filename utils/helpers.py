import json

def read_customer_data_from_file(file_path):
    """Read customer data from a file (JSON or text)"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw_input": content}
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def show_current_status_and_confirm(current_state, next_step):
    """
    Displays the current state and asks the user to confirm proceeding with the next step.
    
    Args:
        current_state (dict): Current policy details.
        next_step (str): Description of the upcoming step.
        
    Returns:
        bool: True if the user confirms, False otherwise.
    """
    print("\n=== CURRENT POLICY STATUS ===")
    if "customerProfile" in current_state and current_state["customerProfile"]:
        print("\nCUSTOMER PROFILE:")
        profile = current_state["customerProfile"]
        if isinstance(profile, dict):
            for key, value in profile.items():
                if key in ["name", "dob", "address", "contact"]:
                    print(f"  {key.capitalize()}: {value}")
        else:
            print(f"  {profile}")
    
    if "risk_info" in current_state and current_state["risk_info"]:
        print("\nRISK ASSESSMENT:")
        risk = current_state["risk_info"]
        if isinstance(risk, dict) and "riskScore" in risk:
            print(f"  Risk Score: {risk['riskScore']}")
        else:
            print(f"  {risk}")
    
    if "coverage" in current_state and current_state["coverage"]:
        print("\nCOVERAGE DETAILS:")
        coverage = current_state["coverage"]
        if isinstance(coverage, dict):
            if "coverages" in coverage:
                print(f"  Coverages: {', '.join(coverage['coverages'])}")
            if "limits" in coverage:
                print(f"  Limits: {coverage['limits']}")
            if "deductibles" in coverage:
                print(f"  Deductibles: {coverage['deductibles']}")
        else:
            print(f"  {coverage}")
    
    if "policyDraft" in current_state and current_state["policyDraft"]:
        print("\nPOLICY DRAFT:")
        draft = current_state["policyDraft"]
        print(f"  {draft[:200]}..." if len(draft) > 200 else f"  {draft}")
    
    if "pricing" in current_state and current_state["pricing"]:
        print("\nPRICING:")
        pricing = current_state["pricing"]
        if isinstance(pricing, dict):
            for key, value in pricing.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {pricing}")
    
    if "quote" in current_state and current_state["quote"]:
        print("\nQUOTE:")
        quote = current_state["quote"]
        print(f"  {quote[:200]}..." if len(quote) > 200 else f"  {quote}")
    
    if "issuance" in current_state and current_state["issuance"]:
        print("\nISSUANCE DETAILS:")
        issuance = current_state["issuance"]
        if isinstance(issuance, dict):
            for key, value in issuance.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {issuance}")
    
    print(f"\nNEXT STEP: {next_step}")
    response = input("Do you want to proceed with this step? (yes/no): ").strip().lower()
    return response == "yes"

def extract_json_content(text):
    """Extract JSON content from a text string that might contain non-JSON content"""
    try:
        # First try direct JSON parsing
        return json.loads(text)
    except Exception:
        # Try to extract JSON from code blocks if present
        if "```json" in text:
            try:
                # Extract content between ```json and ``` markers
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_content = text[json_start:json_end].strip()
                return json.loads(json_content)
            except Exception:
                pass
        
        # Try to find JSON-like content with curly braces
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                json_content = text[start:end+1]
                return json.loads(json_content)
        except Exception:
            pass
    
    return None