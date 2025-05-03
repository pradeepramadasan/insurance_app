import os
import logging
from azure.cosmos import CosmosClient, exceptions
from azure.identity import (
    ManagedIdentityCredential,
    AzureCliCredential,
    ClientSecretCredential,
    DefaultAzureCredential
)

logger = logging.getLogger("cosmos_connection")


class CosmosConnectionManager:
    """
    Singleton manager for Cosmos DB connections following Azure best practices.
    Prioritizes Managed Identity, then Azure CLI, then Service Principal,
    then DefaultAzureCredential. Caches clients and containers.
    """
    _instance = None
    _client = None
    _containers = {}

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = CosmosConnectionManager()
        return cls._instance

    def __init__(self):
        """Initialize connection parameters from environment."""
        if CosmosConnectionManager._instance is not None:
            # Prevent direct instantiation
            raise RuntimeError("Use get_instance() to get CosmosConnectionManager")

        self.endpoint = os.getenv("COSMOS_ENDPOINT")
        self.database_name = os.getenv("COSMOS_DATABASE_NAME", "insurance")
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")

        if not self.endpoint:
            logger.error("COSMOS_ENDPOINT environment variable not set")
            raise ValueError("COSMOS_ENDPOINT environment variable not set")

    def get_client(self):
        """
        Get or initialize the CosmosClient with AAD authentication.
        """
        if CosmosConnectionManager._client:
            return CosmosConnectionManager._client

        logger.info("Initializing Cosmos DB client with AAD authentication")

        # 1. Managed Identity
        try:
            credential = ManagedIdentityCredential()
            client = CosmosClient(self.endpoint, credential=credential)
            # Test with a lightweight call
            list(client.list_databases())
            logger.info("Connected using ManagedIdentityCredential")
            CosmosConnectionManager._client = client
            return client
        except Exception as e:
            logger.warning(f"ManagedIdentityCredential failed: {e}")

        # 2. Azure CLI
        try:
            credential = AzureCliCredential()
            client = CosmosClient(self.endpoint, credential=credential)
            list(client.list_databases())
            logger.info("Connected using AzureCliCredential")
            CosmosConnectionManager._client = client
            return client
        except Exception as e:
            logger.warning(f"AzureCliCredential failed: {e}")

        # 3. Service Principal (Client Secret)
        if self.tenant_id and self.client_id and self.client_secret:
            try:
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                client = CosmosClient(self.endpoint, credential=credential)
                list(client.list_databases())
                logger.info("Connected using ClientSecretCredential")
                CosmosConnectionManager._client = client
                return client
            except Exception as e:
                logger.warning(f"ClientSecretCredential failed: {e}")

        # 4. Default Azure Credential
        try:
            credential = DefaultAzureCredential(
                exclude_managed_identity_credential=True,
                exclude_cli_credential=True
            )
            client = CosmosClient(self.endpoint, credential=credential)
            list(client.list_databases())
            logger.info("Connected using DefaultAzureCredential")
            CosmosConnectionManager._client = client
            return client
        except Exception as e:
            logger.error(f"All AAD authentication methods failed: {e}")
            raise

    def get_container(self, container_name: str):
        """
        Get a container client by exact name, with caching.
        Returns None if the container does not exist or access fails.
        """
        if container_name in self._containers:
            return self._containers[container_name]

        try:
            client = self.get_client()
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(container_name)
            container.read()
            logger.info(f"Connected to container: {container_name}")
            self._containers[container_name] = container
            return container
        except exceptions.CosmosResourceNotFoundError:
            logger.error(f"Container '{container_name}' not found")
        except Exception as e:
            logger.error(f"Error accessing container '{container_name}': {e}")

        self._containers[container_name] = None
        return None

    def list_containers(self) -> list:
        """
        List all containers in the database.
        Returns a list of container IDs (strings).
        """
        try:
            client = self.get_client()
            database = client.get_database_client(self.database_name)
            containers = list(database.list_containers())
            names = [c["id"] for c in containers]
            logger.info(f"Listed containers: {names}")
            return names
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []

    def get_container_case_insensitive(self, container_name: str):
        """
        Case-insensitive container lookup.
        """
        # Try exact match first
        container = self.get_container(container_name)
        if container:
            return container

        # Try case-insensitive search
        for existing in self.list_containers():
            if existing.lower() == container_name.lower():
                return self.get_container(existing)

        logger.warning(f"Container '{container_name}' not found (case‑insensitive)")
        return None

    def check_connection_health(self) -> bool:
        """
        Perform a lightweight check to verify the client is still valid.
        """
        try:
            client = self.get_client()
            list(client.list_databases())
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            CosmosConnectionManager._client = None
            return False

    def close(self):
        """
        Clean up cached client and containers.
        """
        logger.info("Closing CosmosConnectionManager")
        CosmosConnectionManager._client = None
        self._containers.clear()