# insurance_app
Insurance App
A comprehensive auto insurance policy generation and management system powered by AI agents, Azure OpenAI, and Cosmos DB.

Project Overview
The Insurance App is an intelligent insurance workflow system that uses multiple specialized AI agents to process customer data, analyze risk profiles, generate policy quotes, and issue policy documents. By leveraging Azure services including Azure OpenAI with GPT-4o and Cosmos DB, the system delivers personalized insurance recommendations and automated policy documentation.

System Architecture
The application implements a multi-agent architecture where each agent serves a specialized role in the insurance policy workflow:

┌───────────────────┐       ┌─────────────────┐       ┌───────────────────┐
│   Customer Data   │──────►│  Agent Workflow  │──────►│  Policy Issuance  │
└───────────────────┘       └─────────────────┘       └───────────────────┘
                                    │
                                    ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   Iris   │  │ Mnemosyne│  │   Ares   │  │   Hera   │  │  Demeter │
│  (Input) │  │ (Profile)│  │  (Risk)  │  │(Similar) │  │(Coverage)│
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                                              │
                                                              ▼
                                                        ┌──────────┐  ┌──────────┐
                                                        │  Apollo  │  │   Zeus   │
                                                        │ (Policy) │  │(Storage) │
                                                        └──────────┘  └──────────┘

Agent Descriptions
Core Workflow Agents
Agent	Role	Description
Zeus	Coordinating Agent	Orchestrates the entire insurance process, ensuring smooth handoffs between specialized agents and making final policy decisions. Also responsible for storing finalized policy documents in Cosmos DB.
Iris	Input Processing	Handles initial customer data collection and validates required information for policy generation.
Mnemosyne	Customer Profile Builder	Creates comprehensive customer profiles by analyzing customer data and identifying key characteristics.
Ares	Risk Assessment	Evaluates risk factors based on customer profile and vehicle information to influence coverage and premium options.
Hera	Customer Profiling	Analyzes customer data at different workflow stages, finds similar existing customers via embedding similarity, and suggests coverage options based on similar profiles.
Demeter	Coverage Designer	Designs optimal coverage models based on customer profile, risk assessment, and similar customer recommendations.
Apollo	Policy Document Generation	Generates comprehensive policy documents with vehicle, insured, coverage, and policy information using GPT-4o.


File Structure and Descriptions
insurance_app/
├── agents/                      # AI agent implementations
│   ├── __init__.py              # Package initialization
│   ├── apollo.py                # Policy document generation agent
│   ├── ares.py                  # Risk assessment agent
│   ├── demeter.py               # Coverage design agent
│   ├── hera.py                  # Customer profiling and recommendation agent
│   ├── iris.py                  # Input processing agent
│   ├── mnemosyne.py             # Customer profile builder agent
│   └── zeus.py                  # Coordination agent and document storage
├── utils/                       # Utility functions
│   ├── __init__.py              # Package initialization
│   ├── cosmos_db.py             # Cosmos DB interaction utilities
│   └── document_gen.py          # Document generation utilities
├── workflow/                    # Workflow orchestration
│   ├── __init__.py              # Package initialization
│   └── process.py               # Main workflow processing logic
├── customerprofile.py           # Customer profile analysis functionality
├── customerprofiling.py         # Customer segmentation and embedding
├── customerprofilingfields.txt  # Configuration for profiling fields
├── fakepolicy.py                # Synthetic policy generation for testing
├── x1.env                       # Environment variables (API keys, endpoints)
└── README.md                    # This documentation file

Key Files Explained

ile	Description
agents/zeus.py	Implements the Zeus agent that coordinates the workflow and stores policy documents in Cosmos DB.
agents/hera.py	Implements the Hera agent that performs customer profiling and recommendations at each workflow stage.
agents/apollo.py	Implements the Apollo agent that generates comprehensive policy documents.
customerprofile.py	Core functionality for analyzing customer profiles and finding similar customers through embeddings.
customerprofiling.py	Handles customer segmentation using embeddings and clustering.
fakepolicy.py	Generates synthetic policy data for testing and development.
utils/cosmos_db.py	Utilities for interacting with Azure Cosmos DB, handling connections and data storage.
utils/document_gen.py	Utilities for document generation using GPT-4o.
workflow/process.py	Implements the main workflow process, orchestrating agent interactions.

Setup Instructions
Clone the repository
Install required packages: pip install -r requirements.txt
Configure Azure services:
Azure OpenAI (GPT-4o and text-embedding-3-large models)
Azure Cosmos DB
Set up environment variables in x1.env