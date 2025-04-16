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
Insurance Workflow Agents and Their Detailed Roles
The insurance system features a team of AI agents named after Greek deities, each handling a specialized aspect of the insurance policy lifecycle:
1. Iris (IntakeAgent)
•	Primary Role: First point of contact with customers
•	Key Tasks:
o	Collects initial customer information and policy requirements
o	Validates input data from files or manual entry
o	Ensures all required fields are captured (name, DOB, address, vehicle details, etc.)
•	Function: intake_data() - Processes files or interactively requests customer information
•	System Message: "You are Iris. Gather initial customer details for an insurance request."
2. Mnemosyne (ProfileAgent)
•	Primary Role: Customer profile development
•	Key Tasks:
o	Cross-references customer details with historical data (if available)
o	Creates comprehensive customer profiles for risk assessment
o	Organizes customer information into a structured format
•	Function: build_profile() - Transforms raw intake data into a complete profile
•	System Message: "You are Mnemosyne. Compile and cross-reference customer historical data to build a full profile."
3. Ares (RiskEvaluationAgent)
•	Primary Role: Risk assessment specialist
•	Key Tasks:
o	Analyzes vehicle condition, driving history, and other risk factors
o	Calculates a numerical risk score (out of 10)
o	Identifies specific risk factors requiring coverage attention
•	Function: assess_risk() - Evaluates customer profile for insurance risks
•	System Message: "You are Ares. Analyze risk factors (vehicle, driving history, etc.) and produce a risk score."
4. Demeter (CoverageModelAgent)
•	Primary Role: Coverage design expert
•	Key Tasks:
o	Creates tailored insurance coverage models based on risk assessment
o	Determines appropriate limits, deductibles, and exclusions
o	Suggests optional add-ons relevant to the customer's needs
•	Function: design_coverage() - Designs a custom coverage model
•	System Message: "You are Demeter. Design a tailored coverage model with limits, deductibles, and exclusions."
5. Apollo (PolicyDraftingAgent)
•	Primary Role: Policy language specialist
•	Key Tasks:
o	Drafts initial policy document using precise insurance terminology
o	Ensures policy language reflects the coverage model
o	Creates legally compliant policy documentation
•	Function: draft_policy() - Creates the first draft of policy language
•	System Message: "You are Apollo. Draft compliant policy language based on the coverage model."
6. Calliope (DocumentDraftingAgent)
•	Primary Role: Policy document refiner
•	Key Tasks:
o	Polishes policy drafts for clarity and professionalism
o	Enhances readability while maintaining legal precision
o	Final formatting and document preparation
•	Function: polish_document() - Refines raw policy drafts into final documents
•	System Message: "You are Calliope. Refine and polish the policy draft into a finalized document."
7. Plutus (PricingAgent)
•	Primary Role: Actuarial specialist
•	Key Tasks:
o	Calculates base and final premiums based on coverage model
o	Applies risk factors to pricing calculations
o	Determines discounts, surcharges, and fee structures
•	Function: calculate_pricing() - Determines the appropriate premium for coverage
•	System Message: "You are Plutus. Compute premium rates using actuarial models and loadings."
8. Tyche (QuoteAgent)
•	Primary Role: Quote generation specialist
•	Key Tasks:
o	Transforms pricing calculations into formal customer quotes
o	Creates customer-friendly quote presentations
o	Formats pricing information for customer review
•	Function: generate_quote() - Creates formal, customer-facing quotes
•	System Message: "You are Tyche. Generate a detailed, customer-friendly quote based on pricing data."
9. Orpheus (PresentationAgent)
•	Primary Role: Customer communication expert
•	Key Tasks:
o	Presents policy proposals to customers persuasively
o	Explains coverage details and pricing in accessible language
o	Highlights policy benefits and value propositions
•	Function: present_to_customer() - Creates customer presentations of policies and quotes
•	System Message: "You are Orpheus. Present policy proposals to customers in a persuasive and clear manner."
10. Hestia (InternalApprovalAgent)
•	Primary Role: Internal review specialist
•	Key Tasks:
o	Reviews policy drafts for company standards
o	Ensures pricing aligns with company guidelines
o	Provides internal approval for policy issuance
•	Function: internal_approval() - Reviews and approves policies
•	System Message: "You are Hestia. Conduct internal reviews and approve finalized policy drafts."
11. Dike (RegulatoryAgent)
•	Primary Role: Compliance specialist
•	Key Tasks:
o	Ensures policies comply with all relevant regulations
o	Verifies legal language and required disclosures
o	Confirms policies meet industry standards
•	Function: regulatory_review() - Verifies regulatory compliance
•	System Message: "You are Dike. Ensure the policy complies with all regulatory requirements."
12. Eirene (IssuanceAgent)
•	Primary Role: Policy issuance specialist
•	Key Tasks:
o	Finalizes policies for official issuance
o	Assigns unique policy numbers with "MV" prefix
o	Sets policy effective dates and terms
•	Function: issue_policy() - Finalizes and issues approved policies
•	System Message: "You are Eirene. Finalize the issuance of policies and assign policy numbers."
13. Themis (MonitoringAgent)
•	Primary Role: Policy performance analyst
•	Key Tasks:
o	Monitors issued policies for performance metrics
o	Tracks claims ratios, sales volumes, and other KPIs
o	Provides ongoing policy evaluation and feedback
•	Function: monitor_policy() - Tracks policy performance metrics
•	System Message: "You are Themis. Monitor issued policies and report on performance metrics."
14. Zeus (CoordinatingAgent)
•	Primary Role: Workflow coordinator
•	Key Tasks:
o	Oversees the end-to-end insurance policy creation process
o	Facilitates handoffs between specialized agents
o	Generates final policy summaries and reports
•	Function: Coordinates final reporting and overall workflow
•	System Message: "You are Zeus. Coordinate the entire insurance policy creation process and ensure smooth handoffs between agents."



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