
import json
import uuid
import datetime
import logging
import re
from typing import Dict, Optional, Any, List, Union
from urllib.parse import urlparse

from packageurl import PackageURL
from huggingface_hub import HfApi, ModelCard
from huggingface_hub.repocard_data import EvalResult

from .extractor import EnhancedExtractor
from .scoring import calculate_completeness_score
from .registry import get_field_registry_manager
from .schemas import AIBOMResponse, EnhancementReport
from ..utils.validation import validate_aibom, get_validation_summary
from ..utils.license_utils import normalize_license_id, get_license_url

logger = logging.getLogger(__name__)

class AIBOMService:
    """
    Service layer for AI SBOM generation.
    Orchestrates metadata extraction, AI SBOM structure creation, and scoring.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        inference_model_url: Optional[str] = None,
        use_inference: bool = True,
        use_best_practices: bool = True,
    ):
        self.hf_api = HfApi(token=hf_token)
        self.inference_model_url = inference_model_url
        self.use_inference = use_inference
        self.use_best_practices = use_best_practices
        self.enhancement_report = None
        self.extraction_results = {}
        
        # Initialize registry manager
        try:
            self.registry_manager = get_field_registry_manager()
            logger.info("✅ Registry manager initialized in service")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize registry manager: {e}")
            self.registry_manager = None

    def get_extraction_results(self):
        """Return the enhanced extraction results from the last extraction"""
        return self.extraction_results

    def get_enhancement_report(self):
        """Return the enhancement report from the last generation"""
        return self.enhancement_report

    def generate_aibom(
        self,
        model_id: str,
        include_inference: Optional[bool] = None,
        use_best_practices: Optional[bool] = None,
        enable_summarization: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate an AIBOM for the specified Hugging Face model.
        """
        try:
            model_id = self._normalise_model_id(model_id)
            use_inference = include_inference if include_inference is not None else self.use_inference
            use_best_practices = use_best_practices if use_best_practices is not None else self.use_best_practices
            
            logger.info(f"Generating AIBOM for {model_id}")
            
            # Fetch generic info
            model_info = self._fetch_model_info(model_id)
            model_card = self._fetch_model_card(model_id)
            
            # 1. Extract Metadata
            original_metadata = self._extract_metadata(model_id, model_info, model_card, enable_summarization)
            
            # 2. Create Initial AIBOM
            original_aibom = self._create_aibom_structure(model_id, original_metadata)
            
            # 3. Initial Score
            original_score = calculate_completeness_score(
                original_aibom, 
                validate=True, 
                extraction_results=self.extraction_results   # Using results from _extract_metadata
            )
            
            # 4. AI Enhancement (Placeholder for now as in original)
            final_metadata = original_metadata.copy()
            ai_enhanced = False
            ai_model_name = None
            
            if use_inference and self.inference_model_url:
                # Placeholder for AI enhancement logic
                pass
            
            # 5. Create Final AIBOM
            aibom = self._create_aibom_structure(model_id, final_metadata)
            
            # Validate Schema
            is_valid, validation_errors = validate_aibom(aibom)
            if not is_valid:
                logger.warning(f"AIBOM schema validation failed with {len(validation_errors)} errors")
            
            # 6. Final Score
            final_score = calculate_completeness_score(
                aibom, 
                validate=True,
                extraction_results=self.extraction_results
            )
            
            # 7. Store Report
            self.enhancement_report = {
                "ai_enhanced": ai_enhanced,
                "ai_model": ai_model_name,
                "original_score": original_score,
                "final_score": final_score,
                "improvement": round(final_score["total_score"] - original_score["total_score"], 2) if ai_enhanced else 0,
                "schema_validation": {
                    "valid": is_valid,
                    "error_count": len(validation_errors),
                    "errors": validation_errors[:10] if not is_valid else []
                }
            }
            
            return aibom
            
        except Exception as e:
            logger.error(f"Error generating AIBOM: {e}", exc_info=True)
            return self._create_minimal_aibom(model_id)

    def _extract_metadata(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard], enable_summarization: bool = False) -> Dict[str, Any]:
        """Wrapper around EnhancedExtractor"""
        extractor = EnhancedExtractor(self.hf_api) # Pass hfapi instance
        # Ideally we reuse the registry manager
        if self.registry_manager:
            extractor.registry_manager = self.registry_manager
            extractor.registry_fields = self.registry_manager.get_field_definitions()

        metadata = extractor.extract_metadata(model_id, model_info, model_card, enable_summarization=enable_summarization)
        self.extraction_results = extractor.extraction_results
        return metadata

    def _create_minimal_aibom(self, model_id: str) -> Dict[str, Any]:
        """Create a minimal valid AIBOM structure in case of errors"""
        _hf_purl = self._generate_hf_purl(model_id, "1.0")
        
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{str(uuid.uuid4())}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                "tools": {
                    "components": [{
                        "bom-ref": "pkg:generic/owasp-genai/owasp-aibom-generator@1.0.0",
                        "type": "application",
                        "name": "OWASP AIBOM Generator",
                        "version": "1.0.0"
                    }]
                },
                "component": {
                    "bom-ref": f"pkg:generic/{model_id.replace('/', '%2F')}@1.0",
                    "type": "application",
                    "name": model_id.split("/")[-1],
                    "version": "1.0"
                }
            },
            "components": [{
                "bom-ref": _hf_purl,
                "type": "machine-learning-model",
                "name": model_id.split("/")[-1],
                "version": "1.0",
                "purl": _hf_purl
            }]
        }

    def _fetch_model_info(self, model_id: str) -> Dict[str, Any]:
        try:
            return self.hf_api.model_info(model_id)
        except Exception as e:
            logger.warning(f"Error fetching model info for {model_id}: {e}")
            return {}

    def _fetch_model_card(self, model_id: str) -> Optional[ModelCard]:
        try:
            return ModelCard.load(model_id)
        except Exception as e:
            logger.warning(f"Error fetching model card for {model_id}: {e}")
            return None

    def _generate_hf_purl(self, model_id: str, version: str) -> str:
        """Generate HuggingFace PURL using packageurl-python library."""
        parts = model_id.split("/", 1)
        namespace = parts[0] if len(parts) == 2 else None
        name = parts[1] if len(parts) == 2 else parts[0]
        return str(PackageURL(type="huggingface", namespace=namespace, name=name, version=version))

    @staticmethod
    def _normalise_model_id(raw_id: str) -> str:
        if raw_id.startswith(("http://", "https://")):
            path = urlparse(raw_id).path.lstrip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
            return path
        return raw_id

    def _create_aibom_structure(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        version = metadata.get("commit", "1.0")
        _hf_purl = self._generate_hf_purl(model_id, version)
        
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{str(uuid.uuid4())}",
            "version": 1,
            "metadata": self._create_metadata_section(model_id, metadata),
            "components": [self._create_component_section(model_id, metadata)],
            "dependencies": [
                {
                    "ref": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}",
                    "dependsOn": [_hf_purl]
                }
            ]
        }
        
        # Add root external references
        aibom["externalReferences"] = [{
            "type": "distribution",
            "url": f"https://huggingface.co/{model_id}"
        }]
        if metadata and "commit_url" in metadata:
            aibom["externalReferences"].append({
                "type": "vcs",
                "url": metadata["commit_url"]
            })
            
        return aibom

    def _create_metadata_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        version = metadata.get("commit", "1.0")
        
        tools = {
            "components": [{
                "bom-ref": "pkg:generic/owasp-genai/owasp-aibom-generator@1.0.0",
                "type": "application",
                "name": "OWASP AIBOM Generator",
                "version": "1.0.0",
                "manufacturer": {"name": "OWASP GenAI Security Project"}
            }]
        }
        
        authors = []
        if "author" in metadata and metadata["author"]:
            authors.append({"name": metadata["author"]})
            
        component = {
            "bom-ref": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}",
            "type": "application",
            "name": metadata.get("name", model_id.split("/")[-1]),
            "description": metadata.get("description", f"AI model {model_id}"),
            "version": version,
            "purl": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}"
        }
        if authors:
            component["authors"] = authors

        # Manufacturer and Supplier (from Group/Org)
        parts = model_id.split("/")
        if len(parts) > 1:
            group = parts[0]
            org_entity = {
                "name": group,
                "url": [f"https://huggingface.co/{group}"]
            }
            component["manufacturer"] = org_entity
            component["supplier"] = org_entity
            
        return {
            "timestamp": timestamp,
            "tools": tools,
            "component": component
        }

    def _create_component_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        parts = model_id.split("/")
        group = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        version = metadata.get("commit", "1.0")
        purl = self._generate_hf_purl(model_id, version)

        component = {
            "bom-ref": purl,
            "type": "machine-learning-model",
            "group": group,
            "name": name,
            "version": version,
            "purl": purl,
            "description": metadata.get("description", f"AI model {model_id}")
        }
        
        # License
        raw_license = metadata.get("licenses") or metadata.get("license")
        if raw_license:
            # Handle list input (e.g. from regex text extraction)
            if isinstance(raw_license, list):
                if len(raw_license) > 0:
                    raw_license = raw_license[0] # Take the first match
                else:
                    raw_license = None
            
            if raw_license:
                norm_license = normalize_license_id(raw_license)
            # User request: treat NOASSERTION as name to be safe
            if norm_license == "NOASSERTION":
                 component["licenses"] = [{"license": {"name": "NOASSERTION"}}]
            elif norm_license and norm_license.lower() != "other":
                # Check if it looks like a valid SPDX ID (simple heuristic: no spaces, usually short)
                # But our normalize_license_id might return long URLs or names if mapped 
                # (e.g. nvidia-open-model-license is not a standard SPDX ID but we treat it as key)
                
                # If it's the NVIDIA license, putting it in ID fails schema validation because it's not in the enum.
                # So we put it in name, and add the URL.
                if "nvidia" in norm_license.lower():
                     component["licenses"] = [{
                        "license": {
                            "name": norm_license,
                            "url": get_license_url(norm_license)
                        }
                    }]
                else:
                    component["licenses"] = [{
                        "license": {
                            "id": norm_license,
                            "url": get_license_url(norm_license)
                        }
                    }]
            else:
                # Fallback if normalization fails or is 'other', use name
                component["licenses"] = [{"license": {"name": raw_license}}] 
        else:
             # Default fallback per user request to use name for NOASSERTION
             component["licenses"] = [{"license": {"name": "NOASSERTION"}}]
             
        # Authors
        author = metadata.get("author", group)
        if author and author != "unknown":
            component["authors"] = [{"name": author}]
            
            # Manufacturer and Supplier
            # Use the group (org name) as the manufacturer and supplier if available
            if group:
                org_entity = {
                    "name": group,
                    "url": [f"https://huggingface.co/{group}"]
                }
                component["manufacturer"] = org_entity
                component["supplier"] = org_entity
            
        # Technical Properties
        tech_props = []
        for field in ["model_type", "tokenizer_class", "architectures", "library_name"]:
            if field in metadata:
                val = metadata[field]
                if isinstance(val, list): val = ", ".join(val)
                tech_props.append({"name": field, "value": str(val)})
        if tech_props:
            component["properties"] = tech_props
            
        # External Refs
        component["externalReferences"] = [{
            "type": "website", "url": f"https://huggingface.co/{model_id}"
        }]
        
        # Model Card
        component["modelCard"] = self._create_model_card_section(metadata)
        
        return component

    def _create_model_card_section(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        section = {}
        
        # Quantitative Analysis
        if "eval_results" in metadata:
             metrics = []
             for res in metadata["eval_results"]:
                 if hasattr(res, "metric_type") and hasattr(res, "metric_value"):
                     metrics.append({"type": res.metric_type, "value": str(res.metric_value)})
             section["quantitativeAnalysis"] = {"performanceMetrics": metrics}
        
        # Model Parameters
        params = {
            "task": metadata.get("pipeline_tag", "text-generation"),
            "modelArchitecture": f"{metadata.get('name', 'Unknown')}Model"
        }
        if "datasets" in metadata:
            ds_val = metadata["datasets"]
            datasets = []
            if isinstance(ds_val, list):
                for d in ds_val:
                    if isinstance(d, str): datasets.append({"type": "dataset", "name": d})
                    elif isinstance(d, dict) and "name" in d: datasets.append(d)
            elif isinstance(ds_val, str):
                datasets.append({"type": "dataset", "name": ds_val})
            if datasets:
                params["datasets"] = datasets
        
        section["modelParameters"] = params
        
        # Considerations (could map limitation/safety/energy here)
        considerations = {}
        # ... map fields ...
        
        # Properties (everything else)
        props = []
        exclude = ["name", "author", "license", "description", "commit", "bomFormat", "specVersion", "version", "licenses"]
        for k, v in metadata.items():
            if k not in exclude and v is not None:
                val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                props.append({"name": k, "value": val})
        section["properties"] = props
        
        return section
