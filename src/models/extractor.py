

import logging
import re
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from urllib.parse import urlparse, urljoin

from huggingface_hub import HfApi, ModelCard, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

from .schemas import DataSource, ConfidenceLevel, ExtractionResult
from .registry import get_field_registry_manager
from .model_file_extractors import ModelFileExtractor, default_extractors

logger = logging.getLogger(__name__)

class EnhancedExtractor:
    """
    Registry-integrated enhanced extractor that automatically picks up new fields
    from the JSON registry (field_registry.json) without requiring code changes.
    """
    
    # SPDX mappings for common licences
    LICENSE_MAPPINGS = {
        "mit": "MIT",
        "mit license": "MIT",
        "apache license version 2.0": "Apache-2.0",
        "apache license 2.0": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache license, version 2.0": "Apache-2.0",
        "bsd 3-clause": "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        "bsd-2-clause": "BSD-2-Clause",
        "gnu general public license v3": "GPL-3.0-only",
        "gplv3": "GPL-3.0-only",
        "gnu general public license v2": "GPL-2.0-only",
        "gplv2": "GPL-2.0-only",
    }

    def __init__(self, hf_api: Optional[HfApi] = None):
        """
        Initialize the enhanced extractor with registry integration.
        
        Args:
            hf_api: Optional HuggingFace API instance (will create if not provided)
        """
        self.hf_api = hf_api or HfApi()
        self.extraction_results = {}
        
        # Initialize registry manager
        try:
            self.registry_manager = get_field_registry_manager()
            logger.info("✅ Registry manager initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize registry manager: {e}")
            self.registry_manager = None
        
        # Load registry fields
        self.registry_fields = {}
        if self.registry_manager:
            try:
                self.registry_fields = self.registry_manager.get_field_definitions()
                logger.info(f"✅ Loaded {len(self.registry_fields)} fields from registry")
            except Exception as e:
                logger.error(f"❌ Error loading registry fields: {e}")
                self.registry_fields = {}
        
    # Compiled regex patterns for text extraction
    # Moved to class level to avoid recompilation on every request
    PATTERNS = {
        'license': [
            re.compile(r'license[:\s]+([a-zA-Z0-9\-\.\s\n]+)', re.IGNORECASE | re.DOTALL),
            re.compile(r'licensed under[:\s]+([a-zA-Z0-9\-\.\s\n]+)', re.IGNORECASE | re.DOTALL),
            # Robust capture for markdown links [License Name](...)
            re.compile(r'governed by[:\s]+(?:the\s+)?\[([^\]]+)\]', re.IGNORECASE | re.DOTALL),
            re.compile(r'governed by[:\s]+(?:the\s+)?([a-zA-Z0-9\-\.\s\n]+)', re.IGNORECASE | re.DOTALL),
            re.compile(r'governed by the[:\s]+\[([^\]]+)\]', re.IGNORECASE | re.DOTALL),
        ],
        'datasets': [
            re.compile(r'trained on[:\s]+([a-zA-Z0-9\-\_\/]+)', re.IGNORECASE),
            re.compile(r'dataset[:\s]+([a-zA-Z0-9\-\_\/]+)', re.IGNORECASE),
            re.compile(r'using[:\s]+([a-zA-Z0-9\-\_\/]+)\s+dataset', re.IGNORECASE),
        ],
        'metrics': [
            re.compile(r'([a-zA-Z]+)[:\s]+([0-9\.]+)', re.IGNORECASE),
            re.compile(r'achieves[:\s]+([0-9\.]+)[:\s]+([a-zA-Z]+)', re.IGNORECASE),
        ],
        'model_type': [
            re.compile(r'model type[:\s]+([a-zA-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'architecture[:\s]+([a-zA-Z0-9\-]+)', re.IGNORECASE),
        ],
        'energy': [
            re.compile(r'energy[:\s]+([0-9\.]+)\s*([a-zA-Z]+)', re.IGNORECASE),
            re.compile(r'power[:\s]+([0-9\.]+)\s*([a-zA-Z]+)', re.IGNORECASE),
            re.compile(r'consumption[:\s]+([0-9\.]+)\s*([a-zA-Z]+)', re.IGNORECASE),
        ],
        'limitations': [
            re.compile(r'limitation[s]?[:\s]+([^\.]+)', re.IGNORECASE),
            re.compile(r'known issue[s]?[:\s]+([^\.]+)', re.IGNORECASE),
            re.compile(r'constraint[s]?[:\s]+([^\.]+)', re.IGNORECASE),
        ],
        'safety': [
            re.compile(r'safety[:\s]+([^\.]+)', re.IGNORECASE),
            re.compile(r'risk[s]?[:\s]+([^\.]+)', re.IGNORECASE),
            re.compile(r'bias[:\s]+([^\.]+)', re.IGNORECASE),
        ]
    }

    def __init__(
        self,
        hf_api: Optional[HfApi] = None,
        model_file_extractors: Optional[List[ModelFileExtractor]] = None,
    ):
        self.hf_api = hf_api or HfApi()
        self.extraction_results = {}
        self.model_file_extractors = (
            model_file_extractors if model_file_extractors is not None
            else default_extractors()
        )

        # Initialize registry manager
        try:
            self.registry_manager = get_field_registry_manager()
            logger.info("✅ Registry manager initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize registry manager: {e}")
            self.registry_manager = None

        # Load registry fields
        self.registry_fields = {}
        if self.registry_manager:
            try:
                self.registry_fields = self.registry_manager.get_field_definitions()
                logger.info(f"✅ Loaded {len(self.registry_fields)} fields from registry")
            except Exception as e:
                logger.error(f"❌ Error loading registry fields: {e}")
                self.registry_fields = {}

        logger.info(f"Enhanced extractor initialized (registry-driven: {bool(self.registry_fields)})")
    
    # def _compile_patterns(self):  - Removed
       # ...

    def _detect_license_from_file(self, model_id: str) -> Optional[str]:
        """
        Attempt to detect a licence by looking at repository files.
        Downloads common licence filenames (e.g. LICENSE, LICENSE.md),
        reads a small snippet, and returns the matching SPDX identifier,
        or None if none match.
        """
        license_filenames = ["LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "COPYING"]
        for filename in license_filenames:
            try:
                file_path = hf_hub_download(repo_id=model_id, filename=filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    snippet = f.read(4096).lower()
                for header, spdx_id in self.LICENSE_MAPPINGS.items():
                    if header in snippet:
                        return spdx_id
            except (RepositoryNotFoundError, EntryNotFoundError):
                # file doesn’t exist; continue
                continue
            except Exception as e:
                logger.debug(f"Licence detection error reading {filename}: {e}")
                continue
        return None

    def extract_metadata(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard], enable_summarization: bool = False) -> Dict[str, Any]:
        """
        Main extraction method with full registry integration.
        """
        logger.info(f"🚀 Starting registry-driven extraction for model: {model_id}")
        
        # Initialize extraction results tracking
        self.extraction_results = {}
        metadata = {}
        
        if self.registry_fields:
            # Registry-driven extraction
            logger.info(f"📋 Registry-driven mode: Attempting extraction for {len(self.registry_fields)} fields")
            metadata = self._registry_driven_extraction(model_id, model_info, model_card, enable_summarization)
        else:
            # Fallback to legacy extraction
            logger.warning("⚠️ Registry not available, falling back to legacy extraction")
            metadata = self._legacy_extraction(model_id, model_info, model_card)
        
        # Return metadata in the same format as original method
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _registry_driven_extraction(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard], enable_summarization: bool = False) -> Dict[str, Any]:
        """
        Registry-driven extraction that automatically processes all registry fields.
        """
        metadata = {}
        
        # Prepare extraction context
        extraction_context = {
            'model_id': model_id,
            'model_info': model_info,
            'model_card': model_card,
            'readme_content': self._get_readme_content(model_card, model_id),
            'config_data': self._download_and_parse_config(model_id, "config.json"),
            'tokenizer_config': self._download_and_parse_config(model_id, "tokenizer_config.json"),
            'enable_summarization': enable_summarization
        }
        
        # Process each field from the registry
        successful_extractions = 0
        failed_extractions = 0
        
        for field_name, field_config in self.registry_fields.items():
            try:
                logger.info(f"🔍 Attempting extraction for field: {field_name}")
                
                # Extract field using registry configuration
                extracted_value = self._extract_registry_field(field_name, field_config, extraction_context)
                
                if extracted_value is not None:
                    metadata[field_name] = extracted_value
                    successful_extractions += 1
                else:
                    failed_extractions += 1
                    
            except Exception as e:
                failed_extractions += 1
                logger.error(f"❌ Error extracting {field_name}: {e}")
                continue
        
        logger.info(f"📊 Registry extraction complete: {successful_extractions} successful, {failed_extractions} failed")

        model_file_metadata = self._extract_model_file_metadata(model_id)
        if model_file_metadata:
            for key, value in model_file_metadata.items():
                if value is not None:
                    metadata[key] = value
                    self.extraction_results[key] = ExtractionResult(
                        value=value,
                        source=DataSource.REPOSITORY_FILES,
                        confidence=ConfidenceLevel.HIGH,
                        extraction_method="model_file_header",
                    )

        # Always extract commit SHA if available (vital for BOM versioning)
        if 'commit' not in metadata:
             commit_sha = getattr(model_info, 'sha', None)
             if commit_sha:
                 metadata['commit'] = commit_sha

        # Add external references (always needed)
        metadata.update(self._generate_external_references(model_id, metadata))
        
        return metadata
    
    def _extract_model_file_metadata(self, model_id: str) -> Dict[str, Any]:
        for extractor in self.model_file_extractors:
            try:
                if extractor.can_extract(model_id):
                    metadata = extractor.extract_metadata(model_id)
                    if metadata:
                        logger.info(
                            f"{type(extractor).__name__} returned {len(metadata)} fields"
                        )
                        return metadata
            except Exception as e:
                logger.warning(
                    f"Model file extraction failed ({type(extractor).__name__}): {e}"
                )
                continue
        return {}

    def _extract_registry_field(self, field_name: str, field_config: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Extract a single field based on its registry configuration.
        """
        if field_name == 'license':
             logger.warning(f"DEBUG: Extracting license...")

        extraction_methods = []
        
        # Strategy 1: Direct API extraction
        api_value = self._try_api_extraction(field_name, context)
        if api_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=api_value,
                source=DataSource.HF_API,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="api_direct"
            )
            return api_value
        
        # Strategy 2: Model card YAML extraction
        yaml_value = self._try_model_card_extraction(field_name, context)
        if yaml_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=yaml_value,
                source=DataSource.MODEL_CARD,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="model_card_yaml"
            )
            return yaml_value
        
        # Strategy 3: Configuration file extraction
        config_value = self._try_config_extraction(field_name, context)
        if config_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=config_value,
                source=DataSource.CONFIG_FILE,
                confidence=ConfidenceLevel.HIGH,
                extraction_method="config_file"
            )
            return config_value
        
        # Strategy 4: Text pattern extraction
        text_value = self._try_text_pattern_extraction(field_name, context)
        if text_value is not None:
             # ...
            self.extraction_results[field_name] = ExtractionResult(
                value=text_value,
                source=DataSource.README_TEXT,
                confidence=ConfidenceLevel.MEDIUM,
                extraction_method="text_pattern"
            )
            return text_value
        
        # Strategy 5: Intelligent inference
        inferred_value = self._try_intelligent_inference(field_name, context)
        if inferred_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=inferred_value,
                source=DataSource.INTELLIGENT_DEFAULT,
                confidence=ConfidenceLevel.MEDIUM,
                extraction_method="intelligent_inference"
            )
            return inferred_value

        # detect licence from repository files if the field is licence/ licences
        if field_name in {"license", "licenses"}:
            detected = self._detect_license_from_file(context["model_id"])
            if detected:
                self.extraction_results[field_name] = ExtractionResult(
                    value=detected,
                    source=DataSource.REPOSITORY_FILES,
                    confidence=ConfidenceLevel.MEDIUM,
                    extraction_method="license_file",
                    fallback_chain=extraction_methods,
                )
                return detected
        
        if field_name == "description":
            # Try intelligent summarization if description is missing AND enabled
            if context.get('enable_summarization', False):
                try:
                    from ..utils.summarizer import LocalSummarizer
                    readme = context.get('readme_content')
                    if readme:
                        summary = LocalSummarizer.summarize(readme, model_id=context.get('model_id', ''))
                        if summary:
                            self.extraction_results[field_name] = ExtractionResult(
                                value=summary,
                                source=DataSource.INTELLIGENT_DEFAULT,
                                confidence=ConfidenceLevel.MEDIUM,
                                extraction_method="llm_summarization",
                                fallback_chain=extraction_methods
                            )
                            return summary
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"Summarization processing failed: {e}")

        # Strategy 6: Fallback value (if configured)
        fallback_value = self._try_fallback_value(field_name, field_config)
        if fallback_value is not None:
            self.extraction_results[field_name] = ExtractionResult(
                value=fallback_value,
                source=DataSource.PLACEHOLDER,
                confidence=ConfidenceLevel.NONE,
                extraction_method="fallback_placeholder",
                fallback_chain=extraction_methods
            )
            return fallback_value
        
        # No extraction successful
        self.extraction_results[field_name] = ExtractionResult(
            value=None,
            source=DataSource.PLACEHOLDER,
            confidence=ConfidenceLevel.NONE,
            extraction_method="extraction_failed",
            fallback_chain=extraction_methods
        )
        return None
    
    def _extract_paper_link(self, info: Any) -> Union[str, List[str], None]:
        # 1. Check card_data for explicit paper field
        if hasattr(info, 'card_data') and info.card_data:
            paper = getattr(info.card_data, 'paper', None)
            if paper:
                return paper
        
        # 2. Check tags for arxiv: ID
        papers = []
        if hasattr(info, 'tags') and info.tags:
            for tag in info.tags:
                if isinstance(tag, str) and tag.startswith('arxiv:'):
                    papers.append(f"https://arxiv.org/abs/{tag.split(':', 1)[1]}")
        
        return papers if papers else None

    def _try_api_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from HuggingFace API data"""
        model_info = context.get('model_info')
        if not model_info:
            return None
        
        # Field mapping for API extraction
        api_mappings = {
            'author': lambda info: getattr(info, 'author', None) or context['model_id'].split('/')[0],
            'name': lambda info: getattr(info, 'modelId', context['model_id']).split('/')[-1],
            'tags': lambda info: getattr(info, 'tags', []),
            'pipeline_tag': lambda info: getattr(info, 'pipeline_tag', None),
            'downloads': lambda info: getattr(info, 'downloads', 0),
            'commit': lambda info: getattr(info, 'sha', '') if getattr(info, 'sha', None) else None,
            'suppliedBy': lambda info: getattr(info, 'author', None) or context['model_id'].split('/')[0],
            'primaryPurpose': lambda info: getattr(info, 'pipeline_tag', 'text-generation'),
            'downloadLocation': lambda info: f"https://huggingface.co/{context['model_id']}/tree/main",
            'license': lambda info: getattr(info.card_data, 'license', None) if hasattr(info, 'card_data') and info.card_data else None,
            'licenses': lambda info: getattr(info.card_data, 'license', None) if hasattr(info, 'card_data') and info.card_data else None,
            'datasets': lambda info: getattr(info.card_data, 'datasets', []) if hasattr(info, 'card_data') and info.card_data else [],
            'paper': self._extract_paper_link
        }
        
        if field_name in api_mappings:
            try:
                val = api_mappings[field_name](model_info)
                # If valid value found, return it (filtering out "other")
                if val:
                    # Special handling for lists (datasets, tags, paper) - don't lowercase/string convert immmediately
                    if field_name in ["datasets", "tags", "external_references", "paper"]:
                         return val

                    str_val = str(val).lower()
                    if isinstance(val, list) and len(val) > 0:
                        str_val = str(val[0]).lower()
                    
                    # Enhanced filtering for "other" variants
                    ignored_values = {"other", "['other']", "other license", "other-license", "unknown"}
                    if str_val not in ignored_values:
                        return val
                return None
            except Exception as e:
                logger.debug(f"API extraction failed for {field_name}: {e}")
                return None
        
        return None
    
    def _try_model_card_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from model card YAML frontmatter"""
        model_card = context.get('model_card')
        if not model_card or not hasattr(model_card, 'data') or not model_card.data:
            return None
        
        try:
            card_data = model_card.data.to_dict() if hasattr(model_card.data, 'to_dict') else {}
            
            # Field mapping for model card extraction
            card_mappings = {
                'license': 'license',
                'language': 'language',
                'library_name': 'library_name',
                'base_model': 'base_model',
                'datasets': 'datasets',
                'description': ['model_summary', 'description'],
                'typeOfModel': 'model_type',
                'licenses': 'license'  # Alternative mapping
            }
            
            if field_name in card_mappings:
                mapping = card_mappings[field_name]
                if isinstance(mapping, list):
                    # Try multiple keys
                    for key in mapping:
                        value = card_data.get(key)
                        if value:
                            return value
                else:
                    val = card_data.get(mapping)
                    if val:
                        str_val = str(val).lower()
                        if isinstance(val, list) and len(val) > 0:
                            str_val = str(val[0]).lower()
                        
                        ignored_values = {"other", "['other']", "other license", "other-license", "unknown"}
                        return val if str_val not in ignored_values else None
                    return None
            
            # Direct field name lookup
            val = card_data.get(field_name)
            if val:
                str_val = str(val).lower()
                if isinstance(val, list) and len(val) > 0:
                    str_val = str(val[0]).lower()
                return val if str_val != "other" else None
            return None
            
        except Exception as e:
            logger.debug(f"Model card extraction failed for {field_name}: {e}")
            return None
    
    def _try_config_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field from configuration files"""
        # Config file mappings
        config_mappings = {
            'model_type': ('config_data', 'model_type'),
            'architectures': ('config_data', 'architectures'),
            'vocab_size': ('config_data', 'vocab_size'),
            'tokenizer_class': ('tokenizer_config', 'tokenizer_class'),
            'typeOfModel': ('config_data', 'model_type')
        }
        
        if field_name in config_mappings:
            config_type, config_key = config_mappings[field_name]
            config_source = context.get(config_type)
            if config_source:
                return config_source.get(config_key)
        
        return None
    
    def _try_text_pattern_extraction(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to extract field using text pattern matching"""
        readme_content = context.get('readme_content')
        if not readme_content:
            return None
        
        # Pattern mappings for different fields
        pattern_mappings = {
            'license': 'license',
            'licenses': 'license', # Fix: Handle plural key
            'datasets': 'datasets',
            'energyConsumption': 'energy',
            'limitation': 'limitations',
            'safetyRiskAssessment': 'safety',
            'model_type': 'model_type'
        }
        
        if field_name in pattern_mappings:
            pattern_key = pattern_mappings[field_name]
            if pattern_key in self.PATTERNS:
                # Need to implement _find_pattern_matches which was missing in original snippet but used
                matches = self._find_pattern_matches(readme_content, self.PATTERNS[pattern_key])
                if matches:
                    # Prefer longest match for critical fields where "the" or short noise might appear
                    if field_name in ['license', 'licenses']:
                         return max(matches, key=len)
                    # Prefer string for critical fields
                    if field_name in ['model_type']: 
                        return matches[0]
                    return matches[0] if len(matches) == 1 else matches
        
        return None

    def _find_pattern_matches(self, content: str, patterns: List[re.Pattern]) -> List[str]:
        """Find matches for a list of patterns in content"""
        matches = []
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                # Replace newlines/tabs with single space
                val = re.sub(r'\s+', ' ', match.group(1)).strip()
                # Filtering: 'the' is never a license, and generic "other" values
                ignored_values = {
                    "the", "other", "other license", "other-license", "unknown",
                    "vision", "text", "audio", "image", "video", "data", "dataset", "datasets",
                    "training", "eval", "evaluation"
                }
                if val.lower() in ignored_values:
                    continue
                matches.append(val)
        return list(set(matches)) # Return unique matches
    
    def _try_intelligent_inference(self, field_name: str, context: Dict[str, Any]) -> Any:
        """Try to infer field value from other available data"""
        model_id = context['model_id']
        
        # Intelligent inference rules
        inference_rules = {
            'author': lambda: model_id.split('/')[0] if '/' in model_id else 'unknown',
            'suppliedBy': lambda: model_id.split('/')[0] if '/' in model_id else 'unknown',
            'name': lambda: model_id.split('/')[-1],
            'primaryPurpose': lambda: 'text-generation',  # Default for most HF models
            'typeOfModel': lambda: 'transformer',  # Default for most HF models
            'downloadLocation': lambda: f"https://huggingface.co/{model_id}/tree/main",
            'bomFormat': lambda: 'CycloneDX',
            'specVersion': lambda: '1.6',
            'serialNumber': lambda: f"urn:uuid:{model_id.replace('/', '-')}",
            'version': lambda: '1.0.0'
        }
        
        if field_name in inference_rules:
            try:
                return inference_rules[field_name]()
            except Exception as e:
                logger.debug(f"Intelligent inference failed for {field_name}: {e}")
                return None
        
        return None
    
    def _try_fallback_value(self, field_name: str, field_config: Dict[str, Any]) -> Any:
        """Try to get fallback value from field configuration"""
        # Check if field config has fallback value
        if isinstance(field_config, dict):
            fallback = field_config.get('fallback_value')
            if fallback:
                return fallback
        
        # Standard fallback values for common fields
        standard_fallbacks = {
            'license': 'NOASSERTION',
            'description': 'No description available',
            'version': '1.0.0',
            'bomFormat': 'CycloneDX',
            'specVersion': '1.6'
        }
        
        return standard_fallbacks.get(field_name)
    
    def _legacy_extraction(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard]) -> Dict[str, Any]:
        """
        Fallback to legacy extraction when registry is not available.
        This maintains backward compatibility.
        """
        logger.info("🔄 Executing legacy extraction mode")
        metadata = {}
        
        # Execute legacy extraction layers
        metadata.update(self._layer1_structured_api(model_id, model_info, model_card))
        metadata.update(self._layer2_repository_files(model_id))
        metadata.update(self._layer3_stp_extraction(model_card, model_id))
        metadata.update(self._layer4_external_references(model_id, metadata))
        metadata.update(self._layer5_intelligent_defaults(model_id, metadata))
        
        return metadata
    
    def _generate_external_references(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate external references for the model"""
        external_refs = []
        
        # Model repository
        repo_url = f"https://huggingface.co/{model_id}"
        external_refs.append({
            "type": "website",
            "url": repo_url,
            "comment": "Model repository"
        })
        
        # Model files
        files_url = f"https://huggingface.co/{model_id}/tree/main"
        external_refs.append({
            "type": "distribution",
            "url": files_url,
            "comment": "Model files"
        })
        
        # Commit URL if available
        if 'commit' in metadata:
            commit_url = f"https://huggingface.co/{model_id}/commit/{metadata['commit']}"
            external_refs.append({
                "type": "vcs",
                "url": commit_url,
                "comment": "Specific commit"
            })
        
        # Dataset references
        if 'datasets' in metadata:
            datasets = metadata['datasets']
            if isinstance(datasets, list):
                for dataset in datasets:
                    if isinstance(dataset, str):
                        dataset_url = f"https://huggingface.co/datasets/{dataset}"
                        external_refs.append({
                            "type": "distribution",
                            "url": dataset_url,
                            "comment": f"Training dataset: {dataset}"
                        })
        
        # In current structure, we don't store into self.extraction_results here as a side effect if we can avoid it.
        # But for tracing, we might want to.
        
        return {'external_references': external_refs}
    
    # Legacy methods for backward compatibility
    def _layer1_structured_api(self, model_id: str, model_info: Dict[str, Any], model_card: Optional[ModelCard]) -> Dict[str, Any]:
        """Legacy Layer 1: Enhanced structured data extraction from HF API and model card."""
        metadata = {}
        # Enhanced model info extraction
        if model_info:
            try:
                author = getattr(model_info, "author", None)
                if not author or author.strip() == "":
                    parts = model_id.split("/")
                    author = parts[0] if len(parts) > 1 else "unknown"
                
                metadata['author'] = author
                metadata['name'] = getattr(model_info, "modelId", model_id).split("/")[-1]
                metadata['tags'] = getattr(model_info, "tags", [])
                metadata['pipeline_tag'] = getattr(model_info, "pipeline_tag", None)
                metadata['downloads'] = getattr(model_info, "downloads", 0)
                
                commit_sha = getattr(model_info, "sha", None)
                if commit_sha:
                    metadata['commit'] = commit_sha
            except Exception:
                pass
        
        if model_card and hasattr(model_card, "data") and model_card.data:
            try:
                card_data = model_card.data.to_dict() if hasattr(model_card.data, "to_dict") else {}
                metadata['license'] = card_data.get("license")
                metadata['language'] = card_data.get("language")
                metadata['library_name'] = card_data.get("library_name")
                metadata['base_model'] = card_data.get("base_model")
                metadata['datasets'] = card_data.get("datasets")
                metadata['description'] = card_data.get("model_summary") or card_data.get("description")
            except Exception:
                pass
        
        metadata["primaryPurpose"] = metadata.get("pipeline_tag", "text-generation")
        metadata["suppliedBy"] = metadata.get("author", "unknown")
        metadata["typeOfModel"] = "transformer"
        return metadata
    
    def _layer2_repository_files(self, model_id: str) -> Dict[str, Any]:
        """Legacy Layer 2: Repository file analysis"""
        metadata = {}
        try:
            config_data = self._download_and_parse_config(model_id, "config.json")
            if config_data:
                metadata['model_type'] = config_data.get("model_type")
                metadata['architectures'] = config_data.get("architectures", [])
                metadata['vocab_size'] = config_data.get("vocab_size")
            
            tokenizer_config = self._download_and_parse_config(model_id, "tokenizer_config.json")
            if tokenizer_config:
                metadata['tokenizer_class'] = tokenizer_config.get("tokenizer_class")

            if "license" not in metadata or not metadata["license"]:
                detected_license = self._detect_license_from_file(model_id)
                if detected_license:
                    metadata["license"] = detected_license
        except Exception:
            pass
        return metadata
    
    def _layer3_stp_extraction(self, model_card: Optional[ModelCard], model_id: str) -> Dict[str, Any]:
        """Legacy Layer 3: Smart Text Parsing"""
        metadata = {}
        try:
            readme_content = self._get_readme_content(model_card, model_id)
            if readme_content:
                extracted_info = self._extract_from_text(readme_content)
                metadata.update(extracted_info)

                license_from_text = extracted_info.get("license_from_text")
                if license_from_text and not metadata.get("license"):
                    if isinstance(license_from_text, list):
                        metadata["license"] = license_from_text[0]
                    else:
                        metadata["license"] = license_from_text
        except Exception:
            pass
        return metadata
    
    def _layer4_external_references(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy Layer 4: External reference generation"""
        return self._generate_external_references(model_id, metadata)
    
    def _layer5_intelligent_defaults(self, model_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy Layer 5: Intelligent default generation"""
        if 'author' not in metadata or not metadata['author']:
            parts = model_id.split("/")
            metadata['author'] = parts[0] if len(parts) > 1 else "unknown"
        if 'license' not in metadata or not metadata['license']:
            metadata['license'] = "NOASSERTION"
        return metadata
    
    def _fetch_with_backoff(self, fetch_func, *args, max_retries=3, initial_backoff=1.0, **kwargs):
        import time
        for attempt in range(max_retries):
            try:
                return fetch_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "404" in error_msg:  # Auth or not found don't retry
                    raise e
                if attempt == max_retries - 1:
                    raise e
                time.sleep(initial_backoff * (2 ** attempt))

    def _download_and_parse_config(self, model_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Download and parse a JSON config file from the model repository"""
        import json
        try:
            file_path = self._fetch_with_backoff(hf_hub_download, repo_id=model_id, filename=filename)
            with open(file_path, 'r') as f:
                return json.load(f)
        except (RepositoryNotFoundError, EntryNotFoundError, json.JSONDecodeError):
            return None
        except Exception:
            return None
    
    def _get_readme_content(self, model_card: Optional[ModelCard], model_id: str) -> Optional[str]:
        """Get README content from model card or by downloading"""
        try:
            if model_card and hasattr(model_card, 'content'):
                return model_card.content
            readme_path = self._fetch_with_backoff(hf_hub_download, repo_id=model_id, filename="README.md")
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured information from unstructured text (Legacy Helper)"""
        # Minimal implementation for legacy support, utilizing the patterns we already have
        metadata = {}
        for category, patterns in self.PATTERNS.items():
            matches = self._find_pattern_matches(text, patterns)
            if matches:
                metadata[category] = matches[0] if len(matches) == 1 else matches
        return metadata
