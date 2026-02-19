"""
Field Registry Manager for AI SBOM Generator
Combines registry loading, configuration generation, and field detection functionality
"""

import json
import os
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

class FieldRegistryManager:
    """
    Field registry manager that handles:
    1. Registry loading and validation
    2. Configuration generation for utils.py compatibility
    3. Field detection and JSONPath parsing
    4. AIBOM completeness analysis
    5. Scoring calculations
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the field registry manager
        
        Args:
            registry_path: Path to the field registry JSON file. If None, auto-detects.
        """
        if registry_path is None:
            # Auto-detect registry path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            registry_path = os.path.join(current_dir, "field_registry.json")
        
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
        # Cache for performance
        self._field_classification = None
        self._completeness_profiles = None
        self._validation_messages = None
        self._scoring_weights = None
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load the complete field registry from JSON file"""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # Validate basic structure
            required_sections = ["fields"]
            missing_sections = [section for section in required_sections if section not in registry]
            
            if missing_sections:
                raise ValueError(f"Registry missing required sections: {missing_sections}")
            
            # Validate fields structure
            fields = registry.get('fields', {})
            if not fields:
                raise ValueError("Registry 'fields' section is empty")
            
            logger.info(f"✅ Field registry loaded: {len(fields)} fields from {self.registry_path}")
            return registry
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Field registry not found at: {self.registry_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in field registry: {e}")
        except Exception as e:
            raise Exception(f"Failed to load field registry: {e}")
    
    # =============================================================================
    # CONFIGURATION GENERATION
    # =============================================================================
    
    @lru_cache(maxsize=1)
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration from registry"""
        return self.registry.get('scoring_config', {})
    
    @lru_cache(maxsize=1)
    def get_aibom_config(self) -> Dict[str, Any]:
        """Get AIBOM generation configuration from registry"""
        return self.registry.get('aibom_config', {})
    
    @lru_cache(maxsize=1)
    def get_field_definitions(self) -> Dict[str, Any]:
        """Get all field definitions from registry"""
        return self.registry.get('fields', {})
    
    def generate_field_classification(self) -> Dict[str, Any]:
        """
        Generate FIELD_CLASSIFICATION dictionary from registry
        """
        if self._field_classification is not None:
            return self._field_classification
            
        fields = self.get_field_definitions()
        classification = {}
        
        for field_name, field_config in fields.items():
            classification[field_name] = {
                "tier": field_config.get("tier", "supplementary"),
                "weight": field_config.get("weight", 1),
                "category": field_config.get("category", "unknown")
            }
        
        self._field_classification = classification
        return classification
    
    def generate_completeness_profiles(self) -> Dict[str, Any]:
        """
        Generate COMPLETENESS_PROFILES dictionary from registry
        """
        if self._completeness_profiles is not None:
            return self._completeness_profiles
            
        scoring_config = self.get_scoring_config()
        profiles = scoring_config.get("scoring_profiles", {})
        
        # Convert to utils.py format
        completeness_profiles = {}
        for profile_name, profile_config in profiles.items():
            completeness_profiles[profile_name] = {
                "description": profile_config.get("description", f"{profile_name.title()} completeness profile"),
                "required_fields": profile_config.get("required_fields", []),
                "minimum_score": profile_config.get("minimum_score", 50)
            }
        
        # Fallback profiles if none defined in registry
        if not completeness_profiles:
            completeness_profiles = {
                "basic": {
                    "description": "Minimal fields required for identification",
                    "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name"],
                    "minimum_score": 40
                },
                "standard": {
                    "description": "Comprehensive fields for proper documentation",
                    "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name", 
                                       "downloadLocation", "primaryPurpose", "suppliedBy"],
                    "minimum_score": 70
                },
                "advanced": {
                    "description": "Extensive documentation for maximum transparency",
                    "required_fields": ["bomFormat", "specVersion", "serialNumber", "version", "name", 
                                       "downloadLocation", "primaryPurpose", "suppliedBy",
                                       "type", "purl", "description", "licenses", "hyperparameter", "limitation", 
                                       "energyConsumption", "safetyRiskAssessment", "typeOfModel"],
                    "minimum_score": 85
                }
            }
        
        self._completeness_profiles = completeness_profiles
        return completeness_profiles
    
    def generate_validation_messages(self) -> Dict[str, Any]:
        """
        Generate VALIDATION_MESSAGES dictionary from registry
        """
        if self._validation_messages is not None:
            return self._validation_messages
            
        fields = self.get_field_definitions()
        validation_messages = {}
        
        for field_name, field_config in fields.items():
            validation_msg = field_config.get("validation_message", {})
            if validation_msg:
                validation_messages[field_name] = {
                    "missing": validation_msg.get("missing", f"Missing field: {field_name}"),
                    "recommendation": validation_msg.get("recommendation", f"Consider adding {field_name} field")
                }
        
        self._validation_messages = validation_messages
        return validation_messages
    
    def get_configurable_scoring_weights(self) -> Dict[str, Any]:
        """Get configurable scoring weights from registry"""
        if self._scoring_weights is not None:
            return self._scoring_weights
            
        scoring_config = self.get_scoring_config()
        
        weights = {
            "tier_weights": scoring_config.get("tier_weights", {
                "critical": 3,
                "important": 2,
                "supplementary": 1
            }),
            "category_weights": scoring_config.get("category_weights", {
                "required_fields": 20,
                "metadata": 20,
                "component_basic": 20,
                "component_model_card": 30,
                "external_references": 10
            }),
            "algorithm_config": scoring_config.get("algorithm_config", {
                "type": "weighted_sum",
                "max_score": 100,
                "normalization": "category_based"
            })
        }
        
        self._scoring_weights = weights
        return weights
    
    # =============================================================================
    # FIELD DETECTION
    # =============================================================================
    
    def _get_nested_value(self, data: dict, path: str) -> Tuple[bool, Any]:
        """
        Get value from nested dictionary using dot notation and array filters
        Supports paths like: $.components[0].name, $.metadata.properties[?(@.name=='primaryPurpose')].value
        """
        try:
            # Remove leading $. if present
            if path.startswith('$.'):
                path = path[2:]
            
            # Handle special JSONPath-like syntax for property/array filtering
            # Supports [?(@.field=='value')]
            if '[?(@.' in path:
                return self._handle_property_array_path(data, path)
            
            # Split path and traverse
            parts = self._split_path(path)
            current = data
            
            for part in parts:
                if '[' in part and ']' in part:
                    # Handle array access like components[0]
                    key, index_str = part.split('[')
                    index = int(index_str.rstrip(']'))
                    
                    if key and key in current:
                        current = current[key]
                    
                    if isinstance(current, list) and 0 <= index < len(current):
                        current = current[index]
                    else:
                        return False, None
                else:
                    # Regular key access
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False, None
            
            # Check if value is meaningful
            if current is not None and current != "" and current != []:
                return True, current
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error getting value at path {path}: {e}")
            return False, None
    
    def _handle_property_array_path(self, data: dict, path: str) -> Tuple[bool, Any]:
        """
        Handle generic JSONPath-like syntax for array filtering
        Supports: base_path[?(@.key=='value')].optional_suffix
        Example: metadata.component.externalReferences[?(@.type=='documentation')]
        Example: metadata.properties[?(@.name=='primaryPurpose')].value
        """
        try:
            # Regex to capture: Base Path, Filter Key, Filter Value, Optional Suffix
            # matches: something[?(@.key=='val')] or something[?(@.key=='val')].sub
            pattern = r'(.+)\[\?\(@\.(\w+)==\'([^\']+)\'\)\](.*)'
            match = re.search(pattern, path)
            
            if not match:
                return False, None
            
            base_path, filter_key, filter_val, suffix = match.groups()
            
            # Get the list at base_path
            base_found, base_list = self._get_nested_value(data, base_path)
            if not base_found or not isinstance(base_list, list):
                return False, None
            
            # Find matching item
            found_item = None
            for item in base_list:
                if isinstance(item, dict) and str(item.get(filter_key)) == filter_val:
                    found_item = item
                    break
            
            if found_item is None:
                return False, None
                
            # If there's a suffix (e.g., .value), traverse it
            if suffix:
                if suffix.startswith('.'):
                    suffix = suffix[1:]
                return self._get_nested_value(found_item, suffix)
            
            # No suffix, return the item itself
            return True, found_item
            
        except Exception as e:
            logger.error(f"Error handling array path {path}: {e}")
            return False, None
            
        except Exception as e:
            logger.error(f"Error handling property array path {path}: {e}")
            return False, None
    
    def _split_path(self, path: str) -> List[str]:
        """Split path into parts, handling array notation"""
        parts = []
        current_part = ""
        in_brackets = False
        
        for char in path:
            if char == '[':
                in_brackets = True
                current_part += char
            elif char == ']':
                in_brackets = False
                current_part += char
            elif char == '.' and not in_brackets:
                if current_part:
                    parts.append(current_part)
                current_part = ""
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def detect_field_presence(self, aibom: dict, field_path: str) -> Tuple[bool, Any]:
        """
        Detect if a field exists at the given path in the AIBOM
        Returns: (field_exists, field_value)
        """
        return self._get_nested_value(aibom, field_path)
    
    def analyze_aibom_completeness(self, aibom: dict) -> Dict[str, Any]:
        """
        Analyze AIBOM completeness against the enhanced field registry
        Compatible with enhanced registry structure: registry['fields'][field_name]
        """
        results = {
            'category_scores': {},
            'total_score': 0,
            'field_details': {},
            'summary': {}
        }
        
        # Get fields from enhanced registry structure
        fields = self.get_field_definitions()
        if not fields:
            logger.warning("❌ No fields found in registry")
            return results
        
        # Get scoring configuration
        scoring_weights = self.get_configurable_scoring_weights()
        category_weights = scoring_weights.get('category_weights', {})
        
        # Group fields by category
        categories = {}
        for field_name, field_config in fields.items():
            category = field_config.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append((field_name, field_config))
        
        logger.info(f"🔍 Analyzing {len(fields)} fields across {len(categories)} categories")
        
        total_weighted_score = 0
        
        for category_name, category_fields in categories.items():
            category_weight = category_weights.get(category_name, 20)
            
            present_fields = 0
            total_fields = len(category_fields)
            field_details = {}
            
            for field_name, field_config in category_fields:
                field_path = field_config.get('jsonpath', '')
                tier = field_config.get('tier', 'supplementary')
                weight = field_config.get('weight', 1)
                
                if not field_path:
                    field_details[field_name] = {
                        'present': False,
                        'value': None,
                        'path': field_path,
                        'tier': tier,
                        'weight': weight,
                        'error': 'No jsonpath defined'
                    }
                    continue
                
                is_present, value = self.detect_field_presence(aibom, field_path)
                
                field_details[field_name] = {
                    'present': is_present,
                    'value': value,
                    'path': field_path,
                    'tier': tier,
                    'weight': weight
                }
                
                if is_present:
                    present_fields += 1
            
            # Calculate category score
            category_percentage = (present_fields / total_fields) * 100 if total_fields > 0 else 0
            category_score = (category_percentage / 100) * category_weight
            
            results['category_scores'][category_name] = category_score
            results['field_details'][category_name] = field_details
            results['summary'][category_name] = {
                'present': present_fields,
                'total': total_fields,
                'percentage': category_percentage,
                'weight': category_weight
            }
            
            total_weighted_score += category_score
            
        results['total_score'] = total_weighted_score
        
        return results
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_field_info(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get complete information for a specific field"""
        fields = self.get_field_definitions()
        return fields.get(field_name)
    
    def get_field_jsonpath(self, field_name: str) -> Optional[str]:
        """Get JSONPath expression for a specific field"""
        field_info = self.get_field_info(field_name)
        return field_info.get("jsonpath") if field_info else None
    
    def get_fields_by_category(self, category: str) -> List[str]:
        """Get all field names in a specific category"""
        fields = self.get_field_definitions()
        return [
            field_name for field_name, field_config in fields.items()
            if field_config.get("category") == category
        ]
    
    def get_fields_by_tier(self, tier: str) -> List[str]:
        """Get all field names in a specific tier"""
        fields = self.get_field_definitions()
        return [
            field_name for field_name, field_config in fields.items()
            if field_config.get("tier") == tier
        ]
    
    def validate_registry_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the loaded registry"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "field_count": 0,
            "category_distribution": {},
            "tier_distribution": {}
        }
        
        try:
            fields = self.get_field_definitions()
            validation_results["field_count"] = len(fields)
            
            # Check category and tier distribution
            categories = {}
            tiers = {}
            
            for field_name, field_config in fields.items():
                # Check required field properties
                required_props = ["tier", "weight", "category", "jsonpath"]
                missing_props = [prop for prop in required_props if prop not in field_config]
                
                if missing_props:
                    validation_results["errors"].append(
                        f"Field '{field_name}' missing properties: {missing_props}"
                    )
                    validation_results["valid"] = False
                
                # Count categories and tiers
                category = field_config.get("category", "unknown")
                tier = field_config.get("tier", "unknown")
                
                categories[category] = categories.get(category, 0) + 1
                tiers[tier] = tiers.get(tier, 0) + 1
            
            validation_results["category_distribution"] = categories
            validation_results["tier_distribution"] = tiers
            
            # Check scoring configuration
            scoring_config = self.get_scoring_config()
            if not scoring_config.get("tier_weights"):
                validation_results["warnings"].append("Missing tier_weights in scoring_config")
            
            if not scoring_config.get("category_weights"):
                validation_results["warnings"].append("Missing category_weights in scoring_config")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Registry validation error: {e}")
        
        return validation_results

# Global Instance
_registry_manager = None

def get_field_registry_manager() -> FieldRegistryManager:
    """Get the global field registry manager instance (singleton pattern)"""
    global _registry_manager
    if _registry_manager is None:
        _registry_manager = FieldRegistryManager()
    return _registry_manager
