from azure.identity import DefaultAzureCredential, AzureCliCredential
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Load environment variables
load_dotenv('x.env')

# Initialize Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv('ENDPOINT_URL'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version="2024-12-01-preview"
)

# Set up deployment configurations
gpt4o_deployment = os.getenv('GPT4O_DEPLOYMENT_NAME')
assert gpt4o_deployment, "GPT4O deployment name missing in environment variables"
config_list_gpt4o = [{
    "model": gpt4o_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "base_url": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]

o3_deployment = os.getenv('DEPLOYMENT_NAME')
assert o3_deployment, "o3-mini deployment name missing in environment variables"
config_list_o3 = [{
    "model": o3_deployment,
    "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
    "base_url": os.getenv('ENDPOINT_URL'),
    "api_type": "azure",
    "api_version": "2024-12-01-preview"
}]

# Cosmos DB configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
DATABASE_NAME = os.getenv("COSMOS_DATABASE", "InsuranceDB")
DRAFTS_CONTAINER = os.getenv("COSMOS_CONTAINER", "PolicyDrafts")
ISSUED_CONTAINER = "PolicyIssued"