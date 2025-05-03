import json
import logging
from typing import Dict, Any, Optional, List, Callable

# Azure Best Practice: Configure module-level logger
logger = logging.getLogger(__name__)

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

def extract_json_content(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from text using multiple strategies with robust error handling.
    
    Args:
        text (str): Text that may contain JSON
        
    Returns:
        Optional[Dict[str, Any]]: Extracted JSON object or None if extraction fails
    """
    if not text:
        logger.warning("Empty text provided to extract_json_content")
        return None
    
    # Log the first part of the content for debugging
    logger.info(f"Extracting JSON from content (truncated): {text[:200]}...")
        
    # Define extraction strategies from most strict to most lenient
    strategies = [
        # Strategy 1: Direct JSON parsing
        lambda t: json.loads(t),
        
        # Strategy 2: Find content between triple backticks with json marker
        lambda t: json.loads(re.search(r'```json\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        
        # Strategy 3: Find content between triple backticks (any language or no marker)
        lambda t: json.loads(re.search(r'```(?:\w+)?\s*(.*?)\s*```', t, re.DOTALL).group(1)),
        
        # Strategy 4: Find content between curly braces (most common LLM response format)
        lambda t: json.loads(re.search(r'(\{.*\})', t, re.DOTALL).group(1)),
        
        # Strategy 5: Fix and parse JSON with unquoted keys (common LLM error)
        lambda t: json.loads(re.sub(r'(\w+)(?=\s*:)', r'"\1"', re.search(r'(\{.*\})', t, re.DOTALL).group(1))),
        
        # Strategy 6: Remove any leading/trailing text and parse
        lambda t: json.loads(t.strip().split('\n', 1)[-1].rsplit('\n', 1)[0] if '\n' in t else t),
        
        # Strategy 7: Handle nested JSON structures
        lambda t: json.loads(re.search(r'"({.*})"', t.replace('\\"', '"'), re.DOTALL).group(1)),
        
        # Strategy 8: Try to parse JSON by finding and fixing missing quotes around keys
        lambda t: json.loads(re.sub(r'(?<!\w)"(?!\s*[,\:}])(.+?)(?<![,\:\{])"', r'"\1"', re.search(r'(\{.*\})', t, re.DOTALL).group(1)))
    ]
    
    # Try each strategy
    for i, strategy in enumerate(strategies):
        try:
            result = strategy(text)
            logger.info(f"Successfully extracted JSON using strategy {i+1}")
            return result
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {str(e)}")
    
    # All strategies failed - log full text for debugging (truncated if very long)
    if len(text) > 1000:
        logger.warning(f"Failed to extract JSON from long text. First 500 chars: {text[:500]}, Last 500 chars: {text[-500:]}")
    else:
        logger.warning(f"Failed to extract JSON from text: {text}")
    
    return None


def extract_from_code_blocks(text):
    """Extract JSON from markdown code blocks"""
    patterns = [
        r"```(?:json)?\s*([\s\S]*?)```",  # Standard markdown code blocks
        r"`{3,}(?:json)?\s*([\s\S]*?)`{3,}"  # Handle variations in backtick count
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                clean_json = match.strip()
                return json.loads(clean_json)
            except json.JSONDecodeError:
                continue
    
    return None

def extract_between_braces(text):
    """Extract JSON between outermost curly braces"""
    # Find text between first { and last }
    match = re.search(r"\{([\s\S]*)\}", text)
    if match:
        try:
            # Add the braces back and try to parse
            json_str = "{" + match.group(1) + "}"
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    return None

# Alias for backward compatibility if needed
extract_json_with_fallback = extract_json_content

# Add this helper function

