"""
Configuration for Demeter agent using Azure OpenAI o1-mini model.
"""

import os
from dotenv import load_dotenv

# Load environment variables
if os.path.exists("x1.env"):
    load_dotenv("x1.env")

# Demeter-specific Azure OpenAI configuration
DEMETER_CONFIG = {
    "endpoint": "https://azureopenaipr1.openai.azure.com/",
    "model_name": "o1-mini",
    "deployment": "o1-mini",
    "api_version": "2024-09-12",  # Updated model version
    "api_key": os.getenv("AZURE_OPENAI_API_KEY_X1")
}