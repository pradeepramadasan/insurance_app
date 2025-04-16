import sys
import logging
from workflow.process import process_insurance_request  # Changed from insurance_app.workflow.process
from db.cosmos_db import init_cosmos_db  # Changed from insurance_app.db.cosmos_db
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point for the insurance application"""
    print("Initializing Insurance Policy Creation System...")
    
    # Initialize Cosmos DB
    init_cosmos_db()
    
    # Get customer file path if provided
    customer_file = None
    if len(sys.argv) > 1:
        customer_file = sys.argv[1]
    else:
        file_input = input("Do you want to provide a customer data file? (yes/no): ").strip().lower()
        if file_input == "yes":
            customer_file = input("Enter the path to the customer data file: ").strip()
    
    # Process the insurance request
    process_insurance_request(customer_file)

if __name__ == "__main__":
    main()