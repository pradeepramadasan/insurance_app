import os
import logging
import time
from functools import wraps
from azure.cosmos import CosmosClient, exceptions
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    ClientSecretCredential,
    AzureCliCredential
)

logger = logging.getLogger("cosmos_connection")

class CosmosConnectionManager:
    """
    Singleton manager for Cosmos DB connections following Azure best practices.
    Prioritizes Managed Identity authentication which typically bypasses
    Conditional Access policy restrictions.
    """
    _instance = None
    _client = None
    _containers = {}
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = CosmosConnectionManager()
        return cls._instance
    
    def __init__(self):
        """Initialize connection parameters"""
        self.endpoint = os.getenv("COSMOS_ENDPOINT")
        self.database_name = os.getenv("COSMOS_DATABASE_NAME", "insurance")
        
        # Authentication parameters - used as fallback only
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        
        # Validate essential parameters
        if not self.endpoint:
            logger.error("COSMOS_ENDPOINT environment variable not set")
            raise ValueError("COSMOS_ENDPOINT environment variable not set")
    
    def get_client(self):
        """
        Get or initialize the Cosmos DB client with authentication.
        Prioritizes Managed Identity to bypass Conditional Access policies.
        """
        if self._client is not None:
            return self._client
            
        logger.info("Initializing Cosmos DB client with optimal authentication...")
        
        # Try authentication methods in order of preference
        
        # 1. Managed Identity - best for Azure environments and bypasses many Conditional Access policies
        try:
            logger.info("Attempting connection with ManagedIdentityCredential")
            print("Attempting connection with Managed Identity (recommended for Azure environments)...")
            credential = ManagedIdentityCredential()
            self._client = CosmosClient(self.endpoint, credential=credential)
            
            # Test the connection with a lightweight operation
            db = self._client.get_database_client(self.database_name)
            db.read() # Verify database access
            
            logger.info("Successfully connected to Cosmos DB using ManagedIdentityCredential")
            print("✅ Successfully connected to Cosmos DB using Managed Identity")
            return self._client
        except Exception as e:
            logger.warning(f"Managed Identity authentication failed: {type(e).__name__}: {str(e)}")
            print(f"Managed Identity authentication failed: {type(e).__name__}")
            # Continue to next method
        
        # 2. Azure CLI - good for development environments
        try:
            logger.info("Attempting connection with AzureCliCredential")
            print("Attempting connection with Azure CLI credentials...")
            credential = AzureCliCredential()
            self._client = CosmosClient(self.endpoint, credential=credential)
            
            # Test the connection
            db = self._client.get_database_client(self.database_name)
            db.read()
            
            logger.info("Successfully connected to Cosmos DB using AzureCliCredential")
            print("✅ Successfully connected to Cosmos DB using Azure CLI credentials")
            return self._client
        except Exception as e:
            logger.warning(f"Azure CLI authentication failed: {type(e).__name__}: {str(e)}")
            # Continue to next method
            
        # 3. Service Principal (Client Secret) - explicitly controlled
        if self.tenant_id and self.client_id and self.client_secret:
            try:
                logger.info("Attempting connection with ClientSecretCredential")
                print("Attempting connection with Service Principal credentials...")
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self._client = CosmosClient(self.endpoint, credential=credential)
                
                # Test the connection
                db = self._client.get_database_client(self.database_name)
                db.read()
                
                logger.info("Successfully connected to Cosmos DB using ClientSecretCredential")
                print("✅ Successfully connected to Cosmos DB using Service Principal")
                return self._client
            except Exception as e:
                logger.warning(f"Service Principal authentication failed: {type(e).__name__}: {str(e)}")
                # Continue to next method
        
        # 4. Last attempt with DefaultAzureCredential (tries multiple methods)
        try:
            logger.info("Attempting connection with DefaultAzureCredential")
            print("Attempting connection with Default Azure Credential (multiple methods)...")
            # Exclude methods we've already tried
            credential = DefaultAzureCredential(
                exclude_managed_identity_credential=True,
                exclude_cli_credential=True
            )
            self._client = CosmosClient(self.endpoint, credential=credential)
            
            # Test the connection
            db = self._client.get_database_client(self.database_name)
            db.read()
            
            logger.info("Successfully connected to Cosmos DB using DefaultAzureCredential")
            print("✅ Successfully connected to Cosmos DB using Default Azure Credential")
            return self._client
        except Exception as e:
            error_msg = f"All authentication methods failed. Last error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise ConnectionError(error_msg) from e
    
    def get_container(self, container_name):
        """
        Get a container client by name, with caching.
        Following Azure best practices for connection efficiency.
        
        Args:
            container_name (str): The name of the container to access
            
        Returns:
            ContainerProxy or None: The container client if successful, None otherwise
        """
        # Return cached container if available
        if container_name in self._containers and self._containers[container_name] is not None:
            return self._containers[container_name]
            
        logger.info(f"Retrieving container: {container_name}")
        
        try:
            # Ensure client is initialized
            client = self.get_client()
            
            # Get database and container
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(container_name)
            
            # Test container access
            container.read()
            
            # Cache for future use
            self._containers[container_name] = container
            
            logger.info(f"Successfully connected to container: {container_name}")
            return container
            
        except exceptions.CosmosResourceNotFoundError:
            logger.error(f"Container '{container_name}' not found in database '{self.database_name}'")
            self._containers[container_name] = None
            return None
            
        except Exception as e:
            logger.error(f"Error accessing container '{container_name}': {type(e).__name__}: {str(e)}")
            self._containers[container_name] = None
            return None