
import json
import uuid
import datetime
import logging
import re
from typing import Dict, Optional, Any, List, Union
from urllib.parse import urlparse, quote

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
        include_inference: bool = False,
        use_best_practices: Optional[bool] = None,
        enable_summarization: bool = False,
        spec_version: str = "1.6",
        metadata_overrides: Optional[Dict[str, str]] = None,
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
            original_aibom = self._create_aibom_structure(model_id, original_metadata, spec_version)
            
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
            aibom = self._create_aibom_structure(model_id, final_metadata, spec_version=spec_version, metadata_overrides=metadata_overrides)
            
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
            return self._create_minimal_aibom(model_id, spec_version)

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

    def _generate_hf_purl(self, model_id: str, version: str) -> str:
        """Helper to generate consistent Hugging Face PURLs"""
        parts = model_id.split("/")
        if len(parts) > 1:
            group = parts[0]
            name = "/".join(parts[1:])
            # Preserve case and URL encode parts
            purl_path = f"{quote(group)}/{quote(name)}"
        else:
            purl_path = quote(model_id)
            
        return f"pkg:huggingface/{purl_path}@{version}"

    def _create_minimal_aibom(self, model_id: str, spec_version: str = "1.6") -> Dict[str, Any]:
        """Create a minimal valid AIBOM structure in case of errors"""
        hf_purl = self._generate_hf_purl(model_id, "1.0")
        
        return {
            "bomFormat": "CycloneDX",
            "specVersion": spec_version,
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
                "bom-ref": hf_purl,
                "type": "machine-learning-model",
                "name": model_id.split("/")[-1],
                "version": "1.0",
                "purl": hf_purl
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

    @staticmethod
    def _normalise_model_id(raw_id: str) -> str:
        if raw_id.startswith(("http://", "https://")):
            path = urlparse(raw_id).path.lstrip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
            return path
        return raw_id

    def _create_aibom_structure(self, model_id: str, metadata: Dict[str, Any], spec_version: str = "1.6",
                              metadata_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        full_commit = metadata.get("commit")
        version = full_commit[:8] if full_commit else "1.0"
        
        aibom = {
            "bomFormat": "CycloneDX",
            "specVersion": spec_version,
            "serialNumber": f"urn:uuid:{str(uuid.uuid4())}",
            "version": 1,
            "metadata": self._create_metadata_section(model_id, metadata, overrides=metadata_overrides),
            "components": [self._create_component_section(model_id, metadata)],
            "dependencies": [
                {
                    "ref": f"pkg:generic/{model_id.replace('/', '%2F')}@{version}",
                    # Must match the component PURL format
                    "dependsOn": [self._generate_hf_purl(model_id, version)]
                }
            ]
        }
        

            
        return aibom

    def _create_metadata_section(self, model_id: str, metadata: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        
        # Defaults
        default_timestamp = datetime.datetime.now().strftime("job-%Y-%m-%d-%H:%M:%S")
        default_version = str(int(datetime.datetime.now().timestamp()))
        default_mfr = "OWASP AIBOM Generator"
        
        # Apply oveerides or defaults
        overrides = overrides or {}
        comp_name = overrides.get("name") or default_timestamp
        comp_version = overrides.get("version") or default_version
        comp_mfr = overrides.get("manufacturer") or default_mfr
        
        # Normalize for PURL (replace spaces with - or similar if needed, but minimal change is best)
        # PURL: pkg:generic/namespace/name@version
        # namespace = manufacturer
        purl_ns = comp_mfr.replace(" ", "-") # simplistic sanitation
        purl_name = comp_name.replace(" ", "-")
        purl = f"pkg:generic/{purl_ns}/{purl_name}@{comp_version}"
        
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
            "bom-ref": purl,
            "type": "application",
            "name": comp_name,
            "description": f"Generating SBOM for {model_id}",
            "version": comp_version,
            "purl": purl,
            "manufacturer": {"name": comp_mfr},
            "supplier": {"name": comp_mfr}
        }
        if authors:
            component["authors"] = authors
            
        return {
            "timestamp": timestamp,
            "tools": tools,
            "component": component
        }

    def _create_component_section(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        parts = model_id.split("/")
        group = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else parts[0]
        full_commit = metadata.get("commit")
        version = full_commit[:8] if full_commit else "1.0"
        
        # PURL Construction: pkg:huggingface/<group>/<name>@<version>
        # Ensure group and name are URL encoded, separated by /, and preserve case
        if group:
            purl_path = f"{quote(group)}/{quote(name)}"
        else:
            purl_path = quote(name)
            
        purl = f"pkg:huggingface/{purl_path}"
        
        # Use shortened version in PURL
        purl += f"@{version}"
            
        component = {
            "bom-ref": purl,
            "type": "machine-learning-model",
            "group": group,
            "name": name,
            "version": version,
            "purl": purl,
            "description": metadata.get("description", f"AI model {model_id}")
        }
        
        # 1. Licenses
        licenses = self._process_licenses(metadata)
        if licenses:
            component["licenses"] = licenses

        # 2. Authors, Manufacturer, Supplier
        # Note: logic inferred from group and metadata
        authors, manufacturer, supplier = self._process_authors_and_suppliers(metadata, group)
        if authors:
            component["authors"] = authors
        if manufacturer:
            component["manufacturer"] = manufacturer
        if supplier:
            component["supplier"] = supplier
            
        # 3. Technical Properties
        tech_props = self._process_technical_properties(metadata)
        if tech_props:
            component["properties"] = tech_props
            
        # 4. External References
        external_refs = self._process_external_references(model_id, metadata)
        if external_refs:
            component["externalReferences"] = external_refs
        
        # 5. Model Card
        component["modelCard"] = self._create_model_card_section(metadata)
        
        # Defined order for better readability: bom-ref, type, group, name, version, purl, description, modelCard, manufacturer, supplier, authors
        # We also need to preserve: licenses, properties, externalReferences (placing them logically)
        ordered_keys = [
            "bom-ref", "type", "group", "name", "version", "purl", 
            "description", "licenses", "modelCard", 
            "manufacturer", "supplier", "authors", 
            "properties", "externalReferences"
        ]
        
        ordered_component = {}
        for key in ordered_keys:
            if key in component:
                ordered_component[key] = component[key]
                
        # Ensure we didn't miss anything (though we shouldn't have extra keys usually)
        for k, v in component.items():
            if k not in ordered_component:
                ordered_component[k] = v
                
        return ordered_component

    def _process_licenses(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and normalize license information."""
        raw_license = metadata.get("licenses") or metadata.get("license")
        if not raw_license:
            # Default fallback per user request to use name for NOASSERTION
            return [{"license": {"name": "NOASSERTION"}}]

        # Handle list input (e.g. from regex text extraction)
        if isinstance(raw_license, list):
            if len(raw_license) > 0:
                raw_license = raw_license[0] # Take the first match
            else:
                return [{"license": {"name": "NOASSERTION"}}]
        
        norm_license = normalize_license_id(raw_license)
        
        # User request: treat NOASSERTION as name to be safe
        if norm_license == "NOASSERTION":
             return [{"license": {"name": "NOASSERTION"}}]
        elif norm_license and norm_license.lower() != "other":
            # Check if it looks like a valid SPDX ID (simple heuristic: no spaces, usually short)
            # But our normalize_license_id might return long URLs or names if mapped 
            # (e.g. nvidia-open-model-license is not a standard SPDX ID but we treat it as key)
            
            # If it's the NVIDIA license, putting it in ID fails schema validation because it's not in the enum.
            # So we put it in name, and add the URL.
            if "nvidia" in norm_license.lower():
                 return [{
                    "license": {
                        "name": norm_license,
                        "url": get_license_url(norm_license)
                    }
                }]
            else:
                return [{
                    "license": {
                        "id": norm_license,
                        "url": get_license_url(norm_license)
                    }
                }]
        else:
            # Fallback if normalization fails or is 'other', use name
            return [{"license": {"name": raw_license}}]

    def _process_authors_and_suppliers(self, metadata: Dict[str, Any], group: str) -> tuple:
        """
        Process authors, manufacturer, and supplier information.
        Returns: (authors, manufacturer, supplier)
        """
        authors = []
        raw_author = metadata.get("author", group)
        if raw_author and raw_author != "unknown":
            if isinstance(raw_author, str):
                 authors.append({"name": raw_author})
            elif isinstance(raw_author, list):
                 for a in raw_author:
                      authors.append({"name": a})

        manufacturer = None
        supplier = None
        
        # Manufacturer and Supplier
        # Use the group (org name) as the manufacturer and supplier if available
        # If 'suppliedBy' extracted from README, overwrite supplier
        supplier_entity = None
        if group:
            supplier_entity = {
                "name": group,
                "url": [f"https://huggingface.co/{group}"]
            }
        
        if "suppliedBy" in metadata and metadata["suppliedBy"]:
             # If we have explicit suppliedBy, use it for supplier
             supplier_entity = {"name": metadata["suppliedBy"]}
             
        if supplier_entity:
            supplier = supplier_entity
            # Manufacturer often implies the creator/fine-tuner. 
            # If we have a group, we assume they manufactured it too unless specified.
            if group:
                 manufacturer = {
                    "name": group,
                    "url": [f"https://huggingface.co/{group}"]
                }
        
        return authors, manufacturer, supplier

    def _process_technical_properties(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract technical properties from metadata."""
        tech_props = []
        for field in ["model_type", "tokenizer_class", "architectures", "library_name"]:
            if field in metadata:
                val = metadata[field]
                if isinstance(val, list): val = ", ".join(val)
                tech_props.append({"name": field, "value": str(val)})
        return tech_props

    def _process_external_references(self, model_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process external references including Hugging Face URLs and papers."""
        # Start with generic website reference
        generic_ref = {"type": "website", "url": f"https://huggingface.co/{model_id}"}
        external_refs = [generic_ref]
        
        if "external_references" in metadata and isinstance(metadata["external_references"], list):
            for ref in metadata["external_references"]:
                if isinstance(ref, dict) and "url" in ref:
                    rtype = ref.get("type", "website")
                    # Check if URL already exists in our list
                    existing_idx = next((i for i, r in enumerate(external_refs) if r["url"] == ref["url"]), -1)
                    
                    new_ref = {"type": rtype, "url": ref["url"], "comment": ref.get("comment")}
                    
                    if existing_idx != -1:
                        # If existing is generic (no comment) and new one has comment, replace it
                        if not external_refs[existing_idx].get("comment") and new_ref.get("comment"):
                            external_refs[existing_idx] = new_ref
                    else:
                        external_refs.append(new_ref)
        
        # Paper (ArXiv or other documentation)
        if "paper" in metadata and metadata["paper"]:
            papers = metadata["paper"]
            if isinstance(papers, str):
                papers = [papers]
            
            for p in papers:
                # Check for duplicates
                if not any(r["url"] == p for r in external_refs):
                    # Try to infer if it's arxiv for comment
                    comment = "Research Paper"
                    if "arxiv.org" in p:
                        comment = "ArXiv Paper"
                    
                    external_refs.append({
                        "type": "documentation", 
                        "url": p,
                        "comment": comment
                    })
                    
        return external_refs

    def _create_model_card_section(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        section = {}
        
        # 1. Model Parameters
        params = {}
        # primaryPurpose -> task
        if "primaryPurpose" in metadata:
             params["task"] = metadata["primaryPurpose"]
        elif "pipeline_tag" in metadata:
             params["task"] = metadata["pipeline_tag"]
             
        # typeOfModel -> modelArchitecture
        if "typeOfModel" in metadata:
             params["modelArchitecture"] = metadata["typeOfModel"]
        else:
             params["modelArchitecture"] = f"{metadata.get('name', 'Unknown')}Model"
             
        # Datasets
        if "datasets" in metadata:
            ds_val = metadata["datasets"]
            datasets = []
            if isinstance(ds_val, list):
                for d in ds_val:
                    if isinstance(d, str): 
                        # CycloneDX 1.7 compliant componentData
                        datasets.append({
                            "type": "dataset", 
                            "name": d,
                            "contents": {
                                "url": f"https://huggingface.co/datasets/{d}"
                            }
                        })
                    elif isinstance(d, dict) and "name" in d: 
                        datasets.append({"type": "dataset", "name": d.get("name"), "url": d.get("url")})
            elif isinstance(ds_val, str):
                datasets.append({
                    "type": "dataset", 
                    "name": ds_val,
                    "contents": {
                        "url": f"https://huggingface.co/datasets/{ds_val}"
                    }
                })
            
            if datasets:
                params["datasets"] = datasets
        
        # Inputs / Outputs (Inferred from task)
        task = params.get("task")
        if task:
            inputs, outputs = self._infer_io_formats(task)
            if inputs:
                params["inputs"] = [{"format": i} for i in inputs]
            if outputs:
                params["outputs"] = [{"format": o} for o in outputs]
        
        if params:
            section["modelParameters"] = params

        # 2. Quantitative Analysis
        if "eval_results" in metadata:
             metrics = []
             raw_results = metadata["eval_results"]
             if isinstance(raw_results, list):
                 for res in raw_results:
                     # Handle object or dict
                     if hasattr(res, "metric_type") and hasattr(res, "metric_value"):
                         metrics.append({"type": str(res.metric_type), "value": str(res.metric_value)})
                     elif isinstance(res, dict) and "metric_type" in res and "metric_value" in res:
                         metrics.append({"type": str(res["metric_type"]), "value": str(res["metric_value"])})
             
             if metrics:
                 section["quantitativeAnalysis"] = {"performanceMetrics": metrics}

        # 3. Considerations
        considerations = {}
        # intendedUse -> useCases
        if "intendedUse" in metadata:
             considerations["useCases"] = [metadata["intendedUse"]]
        # limitations -> technicalLimitations
        if "limitations" in metadata:
             considerations["technicalLimitations"] = [metadata["limitations"]]
        # ethicalConsiderations
        if "ethicalConsiderations" in metadata:
             considerations["ethicalConsiderations"] = [{"name": "Ethical Considerations", "description": metadata["ethicalConsiderations"]}]
        
        if considerations:
            section["considerations"] = considerations

        # 4. Properties (Generic Bag for leftovers)
        props = []
        # Fields we've already mapped to structured homes
        mapped_fields = [
            "primaryPurpose", "typeOfModel", "suppliedBy", "intendedUse", 
            "limitations", "ethicalConsiderations", "datasets", "eval_results", 
            "pipeline_tag", "name", "author", "license", "description", 
            "commit", "bomFormat", "specVersion", "version", "licenses", 
            "external_references", "tags", "library_name", "paper"
        ]
        
        for k, v in metadata.items():
            if k not in mapped_fields and v is not None:
                # Basic types only for properties
                if isinstance(v, (str, int, float, bool)):
                    props.append({"name": k, "value": str(v)})
                elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                    props.append({"name": k, "value": ", ".join(map(str, v))})
                    
        if props:
            section["properties"] = props
            
        return section

    def _infer_io_formats(self, task: str) -> tuple:
        """
        Infer input and output formats based on the pipeline task.
        Returns (inputs: list, outputs: list)
        """
        task = task.lower().strip()
        
        # Text to Text
        if task in ["text-generation", "text2text-generation", "summarization", "translation", 
                   "conversational", "question-answering", "text-classification", "token-classification"]:
            return (["string"], ["string"])
            
        # Image to Text/Label
        if task in ["image-classification", "object-detection", "image-segmentation"]:
            return (["image"], ["string", "json"])
            
        # Text to Image
        if task in ["text-to-image"]:
            return (["string"], ["image"])
        
        # Audio
        if task in ["automatic-speech-recognition", "audio-classification"]:
            return (["audio"], ["string"])
        if task in ["text-to-speech"]:
            return (["string"], ["audio"])
            
        # Multimodal
        if task in ["visual-question-answering"]:
            return (["image", "string"], ["string"])
            
        # Tabular
        if task in ["tabular-classification", "tabular-regression"]:
            return (["csv", "json"], ["string", "number"])
            
        return ([], [])
