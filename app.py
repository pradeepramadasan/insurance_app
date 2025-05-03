import os
from flask import Flask, render_template, request, jsonify, session
import logging
import uuid
import tempfile
from werkzeug.utils import secure_filename
import time
from datetime import datetime
from workflow.process import (
    process_insurance_request, 
    handle_policy_change
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Landing page with options for new policy or change policy"""
    return render_template('index.html')

@app.route('/new-policy', methods=['GET', 'POST'])
def new_policy():
    """New policy creation flow"""
    if request.method == 'GET':
        # Initialize new policy workflow
        session['workflow_state'] = 'new_policy_started'
        session['current_step'] = 1
        session['agent'] = 'iris'
        return render_template(
            'new_policy.html', 
            step=1, 
            agent='iris', 
            agent_message="Welcome! I'm Iris, your personal insurance assistant. Let's get started with your new policy."
        )
    
    # Handle POST requests (form submissions)
    action = request.form.get('action')
    current_step = session.get('current_step', 1)
    
    # Process form data through the backend workflow
    if action == 'next':
        try:
            # Store correlation ID for error tracking
            correlation_id = session.get('correlation_id', str(uuid.uuid4()))
            
            # Process the step with proper error context
            result = process_step(current_step, request.form)
            
            # Update session with new state
            next_step = current_step + 1
            agent, message = get_agent_for_step(next_step)
            
            session['current_step'] = next_step
            session['agent'] = agent
            
            # Azure best practice: Log successful step completion
            logger.info(f"Successfully completed step {current_step}, advancing to step {next_step}",
                      extra={"correlation_id": correlation_id})
            
            return render_template(
                'new_policy.html', 
                step=next_step, 
                agent=agent,
                agent_message=message,
                data=result
            )
        except ValueError as e:
            # Azure best practice: Differentiate between expected validation errors and unexpected exceptions
            correlation_id = session.get('correlation_id', 'unknown')
            logger.warning(f"Validation error in step {current_step}: {str(e)}",
                         extra={"correlation_id": correlation_id})
            
            return render_template(
                'new_policy.html',
                step=current_step,
                agent=session.get('agent', 'iris'),
                error=str(e),
                correlation_id=correlation_id
            )
        except Exception as e:
            # Azure best practice: Log detailed context for unexpected errors
            correlation_id = session.get('correlation_id', 'unknown')
            logger.error(f"Unexpected error in step {current_step}: {str(e)}",
                       extra={"correlation_id": correlation_id}, exc_info=True)
            
            return render_template(
                'new_policy.html',
                step=current_step,
                agent=session.get('agent', 'iris'),
                error="An unexpected error occurred. Please try again or contact support.",
                correlation_id=correlation_id
            )

@app.route('/change-policy', methods=['GET', 'POST'])
def change_policy():
    """Policy change flow"""
    if request.method == 'GET':
        # Initialize change policy workflow
        session['workflow_state'] = 'change_policy_started'
        session['current_step'] = 1
        session['agent'] = 'zeus'
        return render_template(
            'change_policy.html', 
            step=1, 
            agent='zeus', 
            agent_message="Welcome back! I'm Zeus, your policy management expert. What would you like to change on your policy today?"
        )
    
    # Similar logic as new_policy but for the change workflow
    action = request.form.get('action')
    current_step = session.get('current_step', 1)
    
    if action == 'next':
        try:
            # This would integrate with your existing backend process for policy changes
            result = process_change_step(current_step, request.form)
            
            next_step = current_step + 1
            agent, message = get_change_agent_for_step(next_step)
            
            session['current_step'] = next_step
            session['agent'] = agent
            
            return render_template(
                'change_policy.html', 
                step=next_step, 
                agent=agent,
                agent_message=message,
                data=result
            )
        except Exception as e:
            logger.error(f"Error processing change step {current_step}: {str(e)}")
            return render_template(
                'change_policy.html',
                step=current_step,
                agent=session.get('agent', 'zeus'),
                error=str(e)
            )
    
    elif action == 'back':
        prev_step = max(1, current_step - 1)
        agent, message = get_change_agent_for_step(prev_step)
        session['current_step'] = prev_step
        session['agent'] = agent
        return render_template(
            'change_policy.html', 
            step=prev_step, 
            agent=agent,
            agent_message=message
        )



@app.route('/api/agent-status', methods=['GET'])
def agent_status():
    """API endpoint to get real-time agent status with Azure monitoring integration
    
    Returns agent status, current activity, and progress information
    """
    agent = request.args.get('agent')
    step = request.args.get('step')
    workflow_id = session.get('workflow_id', str(uuid.uuid4()))
    
    if not agent or not step:
        return jsonify({"error": "Missing parameters"}), 400
    
    try:
        # Azure best practice: Add correlation ID for cross-service tracing
        correlation_id = request.args.get('correlation_id', str(uuid.uuid4()))
        
        # Call the backend function with proper tracing
        status = get_agent_status(
            agent_name=agent,
            step=step,
            workflow_id=workflow_id,
            correlation_id=correlation_id
        )
        
        # If we didn't get a real status from backend, provide useful defaults
        if not status:
            # Default responses based on agent and step
            default_status = _get_default_agent_status(agent, step)
            
            # Azure best practice: Log when falling back to defaults
            logger.warning(
                f"No real-time status available for agent={agent}, step={step}. " +
                f"Using default response. Correlation ID: {correlation_id}"
            )
            
            return jsonify({
                "agent": agent,
                "status": default_status.get("status", "active"),
                "activity": default_status.get("activity", "Processing request"),
                "progress": default_status.get("progress", 50),
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "is_default": True
            })
        
        # Add timestamp and correlation ID to the response
        if isinstance(status, dict):
            status.update({
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id
            })
        
        return jsonify(status)
    
    except Exception as e:
        # Azure best practice: Log detailed error with context
        logger.error(f"Error fetching agent status: {str(e)}", exc_info=True)
        
        # Return a 500 response with error details
        return jsonify({
            "error": f"Error fetching agent status: {str(e)}",
            "agent": agent,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": request.args.get('correlation_id', 'unknown')
        }), 500

def _get_default_agent_status(agent, step):
    """Get default agent status when real-time status isn't available"""
    # Default activities by agent
    default_activities = {
        'iris': "Collecting customer information",
        'mnemosyne': "Analyzing vehicle and driving history",
        'hera': "Performing risk assessment",
        'demeter': "Designing coverage options",
        'apollo': "Creating policy documents",
        'calliope': "Enhancing policy clarity",
        'charon': "Processing payment options",
        'hermes': "Finalizing policy issuance",
        'zeus': "Overseeing policy creation"
    }
    
    # Default progress by step (1-9)
    step_progress = {
        '1': 20, '2': 30, '3': 40, '4': 50, 
        '5': 65, '6': 75, '7': 85, '8': 95, '9': 100
    }
    
    return {
        "status": "active",
        "activity": default_activities.get(agent, "Processing your request"),
        "progress": step_progress.get(str(step), 50),
        "details": f"{agent.title()} is handling step {step} of your request",
        "is_default": True
    }

# Helper functions to determine agents and messages for each step
def get_agent_for_step(step):
    """Map workflow step to appropriate agent and initial message"""
    agents = {
        1: ('iris', "I'll help collect your basic information."),
        2: ('mnemosyne', "Let's gather detailed information about your vehicle and driving history."),
        3: ('hera', "I'll analyze your data to recommend the best coverages for your needs."),
        4: ('demeter', "I'll help you design your coverage package."),
        5: ('apollo', "I'm preparing your policy draft based on your selections."),
        6: ('calliope', "I'll polish your policy documents to ensure clarity and accuracy."),
        7: ('charon', "I'm processing your payment options."),
        8: ('hermes', "I'll handle the final issuance of your policy."),
        9: ('zeus', "Congratulations on your new policy! Here's a summary of your coverage.")
    }
    return agents.get(step, ('iris', "Let me assist you with the next step."))
def _validate_file(file):
    """
    Validate uploaded files for security and compatibility.
    
    Args:
        file: The uploaded file object from request
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    # Azure best practice: Whitelist allowed extensions
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'json', 'txt', 'csv'}
    
    # Check file extension
    valid_extension = '.' in file.filename and \
                     file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Azure best practice: Check file size (max 10MB)
    MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    valid_size = file_size <= MAX_SIZE_BYTES
    
    if not valid_extension:
        logger.warning(f"Invalid file extension: {file.filename}")
    if not valid_size:
        logger.warning(f"File too large: {file_size} bytes (max {MAX_SIZE_BYTES})")
    
    return valid_extension and valid_size

def _cleanup_temp_file(file_path):
    """
    Safely remove temporary files.
    
    Args:
        file_path: Path to the file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
def get_change_agent_for_step(step):
    """Map policy change workflow step to appropriate agent and initial message"""
    agents = {
        1: ('zeus', "What would you like to change on your policy?"),
        2: ('demeter', "I'll help you adjust your coverage options."),
        3: ('hera', "I'm recalculating your premium based on these changes."),
        4: ('apollo', "I'm updating your policy documents."),
        5: ('hermes', "I'm processing these changes for your policy."),
        6: ('zeus', "Your policy has been successfully updated!")
    }
    return agents.get(step, ('zeus', "Let me assist you with the next step."))

def process_step(step, form_data):
    """Process steps of new policy creation using existing backend functions
    
    Follows Azure best practices for workflow state management and error handling.
    """
    # Get current state from session or create new state dict
    current_state = session.get('policy_state', {})
    
    try:
        # Step 1: Basic Information (Iris)
        if step == 1:
            # Azure best practice: Add correlation ID for end-to-end tracing
            correlation_id = str(uuid.uuid4())
            logger.info(f"Starting step 1 processing with correlation ID: {correlation_id}")
            
            # Handle file upload if present
            customer_file = None
            if 'customerFile' in request.files and request.files['customerFile'].filename:
                file = request.files['customerFile']
                
                # Azure best practice: Validate file before processing
                if _validate_file(file):
                    # Azure best practice: Use unique filenames with secure handling
                    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                    temp_path = os.path.join(tempfile.gettempdir(), unique_filename)
                    file.save(temp_path)
                    customer_file = temp_path
                    
                    logger.info(f"File saved to temporary location, correlation ID: {correlation_id}")
                    
                    # Azure best practice: Direct integration with existing workflow
                    try:
                        # Use the existing process_insurance_request with the file path
                        # This delegates file parsing and extraction to the existing process
                        policy_data = process_insurance_request(customer_file=customer_file)
                        
                        # Store the processed data in session for future steps
                        current_state.update(policy_data)
                        session['policy_state'] = current_state
                        session['correlation_id'] = correlation_id
                        
                        return {
                            "processed": True,
                            "step": step,
                            "state": current_state,
                            "correlation_id": correlation_id,
                            "from_file": True
                        }
                    except Exception as proc_error:
                        # Azure best practice: Structured error logging with context
                        logger.error(f"Error in file processing: {str(proc_error)}", 
                                     extra={"correlation_id": correlation_id}, exc_info=True)
                        # Cleanup temp file on error
                        _cleanup_temp_file(temp_path)
                        # Fall through to form-based processing
                else:
                    logger.warning(f"Invalid file uploaded: {file.filename}", 
                                  extra={"correlation_id": correlation_id})
                    raise ValueError("Invalid file format. Please upload a supported file type.")
            
            # If no file uploaded or file processing failed, use form data
            current_state['customerProfile'] = {
                "name": form_data.get('fullName'),
                "dob": form_data.get('dateOfBirth'),
                "email": form_data.get('email'),
                "phone": form_data.get('phoneNumber'),
                "address": form_data.get('address'),
            }
            
            # Store the session data
            session['policy_state'] = current_state
            session['correlation_id'] = correlation_id
            
            return {
                "processed": True,
                "step": step,
                "state": current_state,
                "correlation_id": correlation_id,
                "from_file": False
            }
        
        # Remaining steps (2-9) stay the same
        # ...        
        # Step 2: Vehicle & Driving Details (Mnemosyne)
        elif step == 2:
            # Add vehicle information to state
            current_state['vehicleDetails'] = {
                "make": form_data.get('vehicleMake'),
                "model": form_data.get('vehicleModel'),
                "year": form_data.get('vehicleYear'),
                "vin": form_data.get('vin'),
            }
            
            current_state['drivingRecord'] = {
                "violations": form_data.get('violations'),
                "accidents": form_data.get('accidents'),
                "yearsLicensed": form_data.get('yearsLicensed')
            }
            
            # Update session state
            session['policy_state'] = current_state
            return {
                "processed": True,
                "step": step,
                "state": current_state
            }
        
        # Step 3: Risk Assessment (Hera)
        elif step == 3:
            # Azure best practice: Use an Azure Function for compute-intensive operations
            # This would be an async call with a status check in real implementation
            from workflow.process import assess_risk
            risk_assessment = assess_risk(current_state)
            current_state['riskAssessment'] = risk_assessment
            
            # Update session state
            session['policy_state'] = current_state
            return {
                "processed": True,
                "step": step,
                "riskLevel": risk_assessment.get('riskLevel', 'Medium'),
                "riskFactors": risk_assessment.get('factors', []),
                "state": current_state
            }
        
        # Step 4: Coverage Design (Demeter)
        elif step == 4:
            # Extract coverage selections from form
            coverage = {
                "coverages": [],
                "limits": {},
                "deductibles": {}
            }
            
            # Mandatory coverages - always included
            mandatory = ["Bodily Injury", "Property Damage", "Uninsured Motorist Bodily Injury"]
            for cov in mandatory:
                coverage["coverages"].append(cov)
            
            # Get limits from form data
            coverage["limits"]["Bodily Injury"] = {
                "per_person": int(form_data.get('bodilyInjuryLimit').split('/')[0]),
                "per_accident": int(form_data.get('bodilyInjuryLimit').split('/')[1])
            }
            
            coverage["limits"]["Property Damage"] = {
                "amount": int(form_data.get('propertyDamageLimit'))
            }
            
            coverage["limits"]["Uninsured Motorist"] = {
                "per_person": int(form_data.get('uninsuredMotoristLimit').split('/')[0]),
                "per_accident": int(form_data.get('uninsuredMotoristLimit').split('/')[1])
            }
            
            # Optional coverages
            if form_data.get('collision'):
                coverage["coverages"].append("Collision")
                coverage["deductibles"]["Collision"] = int(form_data.get('collisionDeductible', 500))
            
            if form_data.get('comprehensive'):
                coverage["coverages"].append("Comprehensive")
                coverage["deductibles"]["Comprehensive"] = int(form_data.get('comprehensiveDeductible', 500))
            
            if form_data.get('rentalReimbursement'):
                coverage["coverages"].append("Rental Reimbursement")
            
            if form_data.get('roadside'):
                coverage["coverages"].append("Roadside Assistance")
            
            # Store coverage in state
            current_state['coverage'] = coverage
            
            # Call backend to design coverage with demeter
            from workflow.process import design_coverage_with_demeter
            # In the real implementation, you'd use the agents from process.py
            # For now, we'll just use the coverage object
            current_state['coverage'] = coverage
            
            # Update session state
            session['policy_state'] = current_state
            return {
                "processed": True,
                "step": step,
                "coverage": coverage,
                "state": current_state
            }
        
        # Step 5-8: These would follow similar patterns, calling appropriate backend functions
        # For brevity, I'll skip the implementation details but would follow same pattern
        
        # Step 9: Final Policy (Zeus) - Complete the process
        elif step == 9:
            # Final step - Call process_insurance_request with complete state
            try:
                # Azure best practice: Add traceability for long-running operations
                correlation_id = str(uuid.uuid4())
                logger.info(f"Starting policy creation with correlation ID: {correlation_id}")
                
                # Call the main process function with our accumulated state
                complete_policy = process_insurance_request(
                    customer_data=current_state,
                    correlation_id=correlation_id
                )
                
                # Reset session state now that we've completed
                session.pop('policy_state', None)
                
                # Store completed policy ID for reference
                session['completed_policy_id'] = complete_policy.get('policyNumber', 'Unknown')
                
                return {
                    "processed": True,
                    "step": step,
                    "complete": True,
                    "policy": complete_policy
                }
                
            except Exception as e:
                # Azure best practice: Log detailed errors to Application Insights
                logger.error(f"Policy creation failed: {str(e)}", exc_info=True)
                # Reuse Azure's error codes/messages where applicable
                raise ValueError(f"Policy creation failed: {str(e)}")
        
        # Default case - for steps we haven't fully implemented
        else:
            # Update session with current state
            session['policy_state'] = current_state
            return {
                "processed": True,
                "step": step,
                "state": current_state
            }
    
    except Exception as e:
        # Azure best practice: Application Insights for error logging
        logger.error(f"Error in process_step at step {step}: {str(e)}", exc_info=True)
        # Add telemetry correlation
        if hasattr(e, 'trace_id'):
            logger.info(f"Error trace ID: {e.trace_id}")
            
        raise ValueError(f"Error processing step {step}: {str(e)}")

def process_change_step(step, form_data):
    """Process policy change workflow using the backend handle_policy_change function
    
    Implements Azure best practices for error handling and state management.
    """
    # Get current state from session or initialize
    change_state = session.get('change_state', {})
    
    try:
        # Step 1: Get policy details and change request
        if step == 1:
            # Get policy number
            policy_number = form_data.get('policyNumber')
            change_description = form_data.get('changeDescription')
            
            if not policy_number:
                raise ValueError("Policy number is required")
            
            # Store policy number and change request in session
            change_state['policyNumber'] = policy_number
            change_state['changeRequest'] = change_description
            session['change_state'] = change_state
            
            # Try to fetch policy details as validation
            from workflow.process import get_policy_details
            policy = get_policy_details(policy_number)
            
            if not policy:
                raise ValueError(f"Policy {policy_number} not found")
            
            change_state['currentPolicy'] = policy
            session['change_state'] = change_state
            
            return {
                "processed": True,
                "step": step,
                "policy": policy
            }
        
        # Step 2: Coverage modifications (Demeter)
        elif step == 2:
            # Extract coverage changes
            coverage_changes = {
                "action": form_data.get('coverageAction', 'modify'),  # add, remove, modify
                "type": form_data.get('coverageType'),                # limits, deductibles, optional
            }
            
            if coverage_changes["type"] == "limits":
                coverage_name = form_data.get('coverageName')
                new_limit = form_data.get('newLimit')
                coverage_changes["coverage"] = coverage_name
                coverage_changes["newValue"] = new_limit
            
            elif coverage_changes["type"] == "deductibles":
                coverage_name = form_data.get('coverageName')
                new_deductible = form_data.get('newDeductible')
                coverage_changes["coverage"] = coverage_name
                coverage_changes["newValue"] = new_deductible
            
            elif coverage_changes["type"] == "optional":
                coverage_name = form_data.get('coverageName')
                action = form_data.get('optionAction')  # add or remove
                coverage_changes["coverage"] = coverage_name
                coverage_changes["optionAction"] = action
            
            # Store the changes in session
            change_state['coverageChanges'] = coverage_changes
            session['change_state'] = change_state
            
            return {
                "processed": True,
                "step": step,
                "changes": coverage_changes
            }
        
        # Step 3: Premium recalculation (Hera)
        elif step == 3:
            # Azure best practice: Use Azure Functions for compute-intensive operations
            policy = change_state.get('currentPolicy', {})
            changes = change_state.get('coverageChanges', {})
            
            # Call backend for premium calculation
            from workflow.process import recalculate_premium
            new_premium = recalculate_premium(policy, changes)
            
            change_state['newPremium'] = new_premium
            session['change_state'] = change_state
            
            return {
                "processed": True,
                "step": step,
                "premium": new_premium
            }
        
        # Step 4-5: Document updates and processing (Apollo, Hermes)
        # ...Implementation similar to above...
        
        # Step 6: Final policy change (Zeus)
        elif step == 6:
            # Process the complete change with the backend
            try:
                # Azure best practice: Add correlation ID for tracing
                correlation_id = str(uuid.uuid4())
                logger.info(f"Processing policy change with correlation ID: {correlation_id}")
                
                # Call the backend function to handle the policy change
                updated_policy = handle_policy_change(
                    policy_id=change_state.get('policyNumber'),
                    change_request=change_state.get('changeRequest'),
                    coverage_changes=change_state.get('coverageChanges'),
                    correlation_id=correlation_id
                )
                
                # Reset change state now that we're done
                session.pop('change_state', None)
                
                # Store updated policy info
                session['updated_policy_id'] = updated_policy.get('policyNumber', 'Unknown')
                
                return {
                    "processed": True,
                    "step": step,
                    "complete": True,
                    "policy": updated_policy
                }
                
            except Exception as e:
                logger.error(f"Policy change failed: {str(e)}", exc_info=True)
                raise ValueError(f"Policy change failed: {str(e)}")
        
        # Default for other steps
        else:
            # Store state and pass it along
            session['change_state'] = change_state
            return {
                "processed": True,
                "step": step,
                "state": change_state
            }
    
    except Exception as e:
        # Azure best practice: Add structured error logging
        logger.error(f"Error in process_change_step at step {step}: {str(e)}", exc_info=True)
        if hasattr(e, 'status_code'):
            logger.info(f"Error status code: {e.status_code}")
        
        raise ValueError(f"Error processing change step {step}: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)