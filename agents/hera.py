import logging
import uuid
import os
from customerprofile import call_customerprofile

# Azure Best Practice: Configure module-level logger with Azure Monitor integration
logger = logging.getLogger(__name__)



class HeraAgent:
    """
    HeraAgent receives data from IRIS, calls customerprofile.py, and returns only the top 3 recommended coverages.
    """
  
    def __init__(self):
        # Azure Best Practice: Use module-level logger, not self.logger
        logger.info("Starting customer profile processing")
        logger.info("HeraAgent initialized")
                         
    def get_recommendations(self, customer_data, source="iris"):
        """
        Receives customer data from IRIS, calls customerprofile.py, and returns only the top 3 recommended coverages.
        """
        # Azure Best Practice: Add correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        
        try:
            # Use module-level logger, not self.logger
            logger.info(f"[HERA - {source.upper()}][{correlation_id}] Calling customerprofile.py")
            
            # Call customerprofile.py directly with the data as-is
            profile_result = call_customerprofile(customer_data)

            # Extract only the coverages from the top 3 closest policies
            top_policies = profile_result.get("TOP_3_CLOSEST_POLICIES", [])
            recommended_coverages = []

            for policy in top_policies:
                coverage_entry = {
                    "coverages": policy.get("coverages", []),
                    "limits": policy.get("limits", {}),
                    "deductibles": policy.get("deductibles", {}),
                    "addOns": policy.get("addOns", []),
                    "premium": policy.get("premium")
                }
                recommended_coverages.append(coverage_entry)

            result = {
                "source": source,
                "recommended_coverages": recommended_coverages,
                "next_agent": "Mnemosyne",
                "proceed": True,
                "correlation_id": correlation_id  # Azure Best Practice: Include correlation ID
            }

            # Use module-level logger, not self.logger
            logger.info(f"[HERA - {source.upper()}][{correlation_id}] Successfully retrieved recommendations")
            return result

        except Exception as e:
            # Azure Best Practice: Include exc_info for better diagnostics
            logger.error(f"[HERA - {source.upper()}][{correlation_id}] Error: {str(e)}", exc_info=True)
            return {
                "source": source,
                "error": str(e),
                "recommended_coverages": [],
                "next_agent": "Mnemosyne",
                "proceed": True,
                "correlation_id": correlation_id
            }

# Factory function to create HeraAgent instance (remove duplicate definition)
def create_hera_agent():
    """
    Factory function to create a HeraAgent instance.
    """
    return HeraAgent()

def get_profile_recommendations(customer_data, workflow_stage="iris"):
    """
    Wrapper function to get recommendations from HeraAgent.
    """
    # Use module-level logger, not self.logger
    logger.info(f"[HERA - {workflow_stage.upper()}] Processing customer data...")
    try:
        agent = create_hera_agent()
        return agent.get_recommendations(customer_data, source=workflow_stage)
    except Exception as e:
        # Azure Best Practice: Include exc_info for better diagnostics
        logger.error(f"[HERA - {workflow_stage.upper()}] Error: {str(e)}", exc_info=True)
        return {
            "source": workflow_stage,
            "error": str(e),
            "recommended_coverages": [],
            "proceed": True
        }