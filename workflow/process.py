import datetime
import random  # Also needed for policy number generation
import json
from agents import initialize_agents
from db.cosmos_db import save_policy_checkpoint, save_policy_draft, confirm_policy
from utils.helpers import read_customer_data_from_file, show_current_status_and_confirm, extract_json_content

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
    
    current_state = {}
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
        save_policy_checkpoint(current_state, "detailed_profile_completed")
    else:
        print("Workflow halted at detailed profile stage.")
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
        save_policy_checkpoint(current_state, "risk_assessment_completed")
        print(f"[Ares] Risk evaluation completed. Risk Score: {risk_info.get('riskScore', 'N/A')}")
    else:
        print("Workflow halted at risk assessment stage.")
        return
    
    # Step 4: Design Coverage using Demeter
    if show_current_status_and_confirm(current_state, "Design coverage model with Demeter"):
        coverage_prompt = f"Call: design_coverage; Params: {json.dumps({'profile': current_state['customerProfile'], 'risk_info': current_state['risk_info']})}"
        response = demeter.generate_reply(messages=[{"role": "user", "content": coverage_prompt}])
        coverage = extract_json_content(response.content) if hasattr(response, 'content') else None
        
        if not isinstance(coverage, dict):
            print("[Demeter] Could not parse response. Using default coverage model.")
            coverage = {
                "coverages": ["Collision", "Liability", "Comprehensive"],
                "limits": {"collision": 40000, "liability": 100000},
                "deductibles": {"collision": 1000, "comprehensive": 500},
                "exclusions": ["Wear and Tear"],
                "addOns": ["Roadside Assistance"]
            }
        
        current_state["coverage"] = coverage
        save_policy_checkpoint(current_state, "coverage_design_completed")
        print("[Demeter] Coverage model designed.")
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
            import random
            import datetime
            issuance = {
                "policyNumber": f"POL{random.randint(100000, 999999)}",
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