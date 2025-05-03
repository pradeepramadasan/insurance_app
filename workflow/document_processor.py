import os
import time
import logging
import json
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceResponseError, ClientAuthenticationError

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Azure Document Intelligence processor that extracts vehicle and insurance information
    from documents using Azure's advanced AI models.
    
    Features:
    - Vehicle information extraction from registration documents, insurance cards, etc.
    - Driver's license information extraction
    - Insurance policy document parsing
    - Robust error handling with graceful fallbacks
    """
    
    def __init__(self):
        """Initialize Document Intelligence client using Azure environment variables"""
        self.endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self.client = None
        
        if self.endpoint and self.key:
            try:
                self.client = DocumentAnalysisClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.key)
                )
                logger.info("Document Intelligence client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Document Intelligence client: {str(e)}")
                self.client = None
        else:
            logger.warning("Azure Document Intelligence credentials not found in environment variables")
    
    def process_document(self, file_path):
        """
        Process a document to extract vehicle and insurance information.
        
        Args:
            file_path: Path to the document file (PDF, JPEG, PNG, TIFF)
            
        Returns:
            dict: Extracted information or None if processing failed
        """
        if not self.client:
            logger.error("Document Intelligence client not initialized")
            return {"error": "Document Intelligence service not configured"}
        
        # Validate file exists and has supported extension
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found"}
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            logger.error(f"Unsupported file format: {file_extension}")
            return {"error": f"Unsupported file format: {file_extension}"}
        
        try:
            # Choose the appropriate pre-built model based on document type
            logger.info(f"Processing document: {file_path}")
            
            # Start with the general document model
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document("prebuilt-document", f)
                
            # Wait for processing to complete with timeout
            max_wait_sec = 30
            poll_interval_sec = 2
            start_time = time.time()
            
            while not poller.done() and time.time() - start_time < max_wait_sec:
                time.sleep(poll_interval_sec)
                
            if not poller.done():
                logger.warning(f"Document analysis timed out after {max_wait_sec} seconds")
                return {"error": "Document analysis timed out"}
                
            result = poller.result()
            logger.info(f"Document analysis completed with status: {poller.status()}")
            
            # Process the results and extract relevant information
            extracted_info = self._extract_vehicle_insurance_data(result)
            
            # If we didn't get much data, try specialized model for vehicle registration
            if not extracted_info.get("vehicle_details") and file_extension in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                logger.info("Attempting specialized vehicle document model")
                with open(file_path, "rb") as f:
                    poller = self.client.begin_analyze_document("prebuilt-idDocument", f)
                    
                result = poller.result()
                id_extracted = self._extract_from_id_document(result)
                
                # Merge the results with priority to ID document extraction
                extracted_info.update(id_extracted)
            
            return extracted_info
            
        except HttpResponseError as e:
            logger.error(f"HTTP error processing document: {str(e)}")
            return {"error": f"Service error: {str(e)}"}
        except ServiceResponseError as e:
            logger.error(f"Azure service error: {str(e)}")
            return {"error": f"Service response error: {str(e)}"}
        except ClientAuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            return {"error": "Authentication error with Azure services"}
        except Exception as e:
            logger.error(f"Unexpected error processing document: {str(e)}", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _extract_vehicle_insurance_data(self, result):
        """
        Extract vehicle and insurance information from Document Intelligence results
        
        Args:
            result: Document Intelligence result object
            
        Returns:
            dict: Extracted information with standardized structure
        """
        extracted_info = {
            "name": "",
            "dob": "",
            "address": {},
            "contact": {},
            "vehicle_details": {},
            "driving_history": {}
        }
        
        if not result or not hasattr(result, 'content'):
            return extracted_info
            
        try:
            # Go through each key-value pair in the document
            for kv_pair in result.key_value_pairs:
                if not kv_pair.key or not kv_pair.value:
                    continue
                
                key = kv_pair.key.content.lower()
                value = kv_pair.value.content
                
                # Name extraction
                if any(name_key in key for name_key in ["full name", "name", "owner", "insured"]):
                    extracted_info["name"] = value
                
                # DOB extraction
                elif any(dob_key in key for dob_key in ["birth", "dob", "date of birth"]):
                    extracted_info["dob"] = value
                
                # Vehicle info extraction
                elif "make" in key or "manufacturer" in key:
                    extracted_info["vehicle_details"]["make"] = value
                elif "model" in key:
                    extracted_info["vehicle_details"]["model"] = value
                elif "year" in key:
                    extracted_info["vehicle_details"]["year"] = value
                elif any(vin_key in key for vin_key in ["vin", "vehicle id", "vehicle identification"]):
                    extracted_info["vehicle_details"]["vin"] = value
                    
                # Address extraction - look for address components
                elif any(addr_key in key for addr_key in ["street", "address"]):
                    extracted_info["address"]["street"] = value
                elif "city" in key:
                    extracted_info["address"]["city"] = value
                elif "state" in key or "province" in key:
                    extracted_info["address"]["state"] = value
                elif "zip" in key or "postal" in key:
                    extracted_info["address"]["zip"] = value
                    
                # Contact info
                elif any(phone_key in key for phone_key in ["phone", "telephone", "mobile"]):
                    extracted_info["contact"]["phone"] = value
                elif "email" in key:
                    extracted_info["contact"]["email"] = value
                    
                # Driving history - look for indicators
                elif "violations" in key or "infractions" in key:
                    extracted_info["driving_history"]["violations"] = value
                elif "accident" in key:
                    extracted_info["driving_history"]["accidents"] = value
                elif "licensed" in key or "years driving" in key:
                    extracted_info["driving_history"]["years_licensed"] = value
            
            # Try to extract VIN from the full content if not found in key-value pairs
            if not extracted_info["vehicle_details"].get("vin"):
                vin_pattern = r'\b[A-HJ-NPR-Z0-9]{17}\b'
                import re
                vin_matches = re.findall(vin_pattern, result.content)
                if vin_matches:
                    extracted_info["vehicle_details"]["vin"] = vin_matches[0]
                    
            # If we have address components in the document content but not extracted
            if not any(extracted_info["address"].values()):
                self._extract_address_from_content(result.content, extracted_info)
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting data from document: {str(e)}")
            return extracted_info
            
    def _extract_from_id_document(self, result):
        """
        Extract information from ID document model results
        
        Args:
            result: ID Document Intelligence result
            
        Returns:
            dict: Extracted information
        """
        extracted_info = {
            "name": "",
            "dob": "",
            "address": {},
            "vehicle_details": {}
        }
        
        try:
            if hasattr(result, 'documents') and result.documents:
                for doc in result.documents:
                    # Extract driver's license info
                    if doc.doc_type == "idDocument.driverLicense":
                        for field_name, field in doc.fields.items():
                            if field_name == "FullName" and field.value:
                                extracted_info["name"] = field.value
                            elif field_name == "DateOfBirth" and field.value:
                                extracted_info["dob"] = field.value
                            elif field_name == "DocumentId" and field.value:
                                # Some vehicle registrations have VIN in document ID
                                if len(field.value) == 17:  # VIN length
                                    extracted_info["vehicle_details"]["vin"] = field.value
                            elif field_name == "Address" and field.value:
                                extracted_info["address"]["street"] = field.value
                            # Add more field extractions as needed
                                
        except Exception as e:
            logger.error(f"Error extracting from ID document: {str(e)}")
            
        return extracted_info
        
    def _extract_address_from_content(self, content, extracted_info):
        """
        Attempt to extract address components from document content
        
        Args:
            content: Full document text content
            extracted_info: Dictionary to update with extracted address
        """
        try:
            # Simple pattern matching for US addresses
            import re
            
            # Look for street address (with number)
            street_pattern = r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Place|Pl|Way|Court|Ct)\b'
            street_match = re.search(street_pattern, content)
            if street_match:
                extracted_info["address"]["street"] = street_match.group(0)
            
            # Look for City, State ZIP pattern
            address_pattern = r'\b([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)\b'
            address_match = re.search(address_pattern, content)
            if address_match:
                extracted_info["address"]["city"] = address_match.group(1).strip()
                extracted_info["address"]["state"] = address_match.group(2)
                extracted_info["address"]["zip"] = address_match.group(3)
                
        except Exception as e:
            logger.error(f"Error extracting address from content: {str(e)}")