import logging
import os
import sys
from db.cosmos_db import init_cosmos_db

# Remove duplicate imports and use only one import statement
# If you added insurance_app_main to process.py (Solution 1)
from workflow.process import insurance_app_main

# Azure Best Practice: Configure logging once at application startup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)



def main():
    """
    Main application entry point.
    
    Following Azure best practices:
    - Initialize resources: Sets up Cosmos DB
    - Error handling: Catches and logs initialization errors
    - Clear entry point: Delegates to domain-specific handler
    """
    # Initialize required resources
    try:
        init_cosmos_db()
    except Exception as e:
        logging.error(f"Failed to initialize Cosmos DB: {str(e)}", exc_info=True)
        print(f"Error initializing database: {str(e)}")
        return
    
    # Run the application
    insurance_app_main()

if __name__ == "__main__":
    main()