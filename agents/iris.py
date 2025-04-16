import json
import logging
import uuid
import os
from autogen import AssistantAgent
from config import config_list_gpt4o
from agents.hera import HeraAgent

# Azure Best Practice: Configure module-level logger with consistent naming
logger = logging.getLogger(__name__)


class IrisAgent:
    """
    IrisAgent processes customer data and coordinates with other agents in the workflow.
    """
    def __init__(self):
        # Azure Best Practice: Generate correlation ID for request tracing
        self.correlation_id = str(uuid.uuid4())
        self.hera_agent = HeraAgent()
        # Use module-level logger, not self.logger
        logger.info(f"[IRIS][{self.correlation_id}] IrisAgent initialized")

    def process_customer_data(self, customer_data):
        """Process customer data and get recommendations from Hera"""
        try:
            # Call Hera to get recommendations
            logger.info(f"[IRIS][{self.correlation_id}] Requesting recommendations from Hera agent")
            hera_response = self.hera_agent.get_recommendations(customer_data, source="iris")
            
            # Process and display Hera's response
            self.process_hera_response(hera_response)
            
            return hera_response
        except Exception as e:
            # Azure Best Practice: Include exc_info for better diagnostics
            logger.error(f"[IRIS][{self.correlation_id}] Error processing customer data: {str(e)}", exc_info=True)
            return {"error": str(e), "correlation_id": self.correlation_id}

    def process_hera_response(self, hera_response):
        """Process and display the recommendations from Hera"""
        try:
            # Check for recommended_coverages in response
            if "recommended_coverages" in hera_response and hera_response["recommended_coverages"]:
                logger.info(f"[IRIS][{self.correlation_id}] Processing {len(hera_response['recommended_coverages'])} coverage recommendations")
                print("\n=== RECOMMENDED COVERAGES ===")
                for idx, coverage in enumerate(hera_response["recommended_coverages"], 1):
                    print(f"\nOption {idx}:")
                    if coverage.get("coverages"):
                        print(f"  Coverages: {', '.join(coverage['coverages'])}")
                    if coverage.get("limits"):
                        print("  Limits:")
                        for limit_name, limit_value in coverage["limits"].items():
                            print(f"    - {limit_name}: {limit_value}")
                    if coverage.get("deductibles"):
                        print("  Deductibles:")
                        for ded_name, ded_value in coverage["deductibles"].items():
                            print(f"    - {ded_name}: {ded_value}")
                    if coverage.get("premium") is not None:
                        print(f"  Premium: ${coverage['premium']:.2f}")
                print("\n")
            else:
                logger.warning(f"[IRIS][{self.correlation_id}] No coverage recommendations available")
                print("\nNo coverage recommendations available")
        except Exception as e:
            # Azure Best Practice: Include exc_info for better diagnostics
            logger.error(f"[IRIS][{self.correlation_id}] Error processing Hera response: {str(e)}", exc_info=True)
            print("\nError displaying recommendations")

# Create a wrapper function to maintain backward compatibility
def present_recommendations_to_user(customer_data):
    """Wrapper function to maintain backward compatibility"""
    # Azure Best Practice: Log operations at appropriate places
    logger.info("[IRIS] Starting presentation of recommendations to user")
    iris = IrisAgent()
    return iris.process_customer_data(customer_data)

# Create the autogen assistant agent
def create_iris_agent():
    """Create and return the Iris (Intake) agent"""
    # Azure Best Practice: Log agent creation
    logger.info("[IRIS] Creating Iris autogen agent")
    return AssistantAgent(
        name="Iris (IntakeAgent)",
        system_message="""
You are IRIS, the master planner and coordinator agent for the insurance policy creation workflow.
[rest of system message...]
""",
        llm_config={"config_list": config_list_gpt4o}
    )