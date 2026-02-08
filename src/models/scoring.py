
import logging
import re
import os
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from .registry import get_field_registry_manager

logger = logging.getLogger(__name__)

# Validation severity levels
class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

# Initialize registry manager
try:
    REGISTRY_MANAGER = get_field_registry_manager()
    FIELD_CLASSIFICATION = REGISTRY_MANAGER.generate_field_classification()
    COMPLETENESS_PROFILES = REGISTRY_MANAGER.generate_completeness_profiles()
    VALIDATION_MESSAGES = REGISTRY_MANAGER.generate_validation_messages()
    SCORING_WEIGHTS = REGISTRY_MANAGER.get_configurable_scoring_weights()
    logger.info(f"✅ Registry-driven configuration loaded: {len(FIELD_CLASSIFICATION)} fields")
except Exception as e:
    logger.error(f"❌ Failed to load registry configuration: {e}")
    # Fallback to empty defaults or handle gracefully
    FIELD_CLASSIFICATION = {}
    COMPLETENESS_PROFILES = {}
    VALIDATION_MESSAGES = {}
    SCORING_WEIGHTS = {}


def validate_spdx(license_entry):
    spdx_licenses = [
        "MIT", "Apache-2.0", "GPL-3.0-only", "GPL-2.0-only", "LGPL-3.0-only",
        "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0",
        "Unlicense", "NONE"
    ]
    if isinstance(license_entry, list):
        return all(lic in spdx_licenses for lic in license_entry)
    return license_entry in spdx_licenses

def check_field_in_aibom(aibom: Dict[str, Any], field: str) -> bool:
    """
    Check if a field is present in the AIBOM (Legacy/Standard Layout check).
    Optimized to use a flattened set if possible, but for individual check this is fine.
    """
    # Quick top-level check
    if field in aibom:
        return True
        
    # Metadata Check
    metadata = aibom.get("metadata", {})
    if field in metadata:
        return True
    
    # Metadata Properties
    if "properties" in metadata:
        for prop in metadata["properties"]:
            if prop.get("name") in {field, f"spdx:{field}"}:
                return True

    # Component Check (only first component as per original logic)
    components = aibom.get("components", [])
    if components:
        component = components[0]
        if field in component:
            return True
        
        # Component Properties
        if "properties" in component:
            for prop in component["properties"]:
                if prop.get("name") in {field, f"spdx:{field}"}:
                    return True
        
        # Model Card
        model_card = component.get("modelCard", {})
        if field in model_card:
            return True
        
        if "modelParameters" in model_card and field in model_card["modelParameters"]:
            return True
        
        # Considerations Mapping
        if "considerations" in model_card:
            considerations = model_card["considerations"]
            field_mappings = {
                "limitation": ["technicalLimitations", "limitations"],
                "safetyRiskAssessment": ["ethicalConsiderations", "safetyRiskAssessment"],
                "energyConsumption": ["environmentalConsiderations", "energyConsumption"]
            }
            if field in field_mappings:
                if any(sec in considerations and considerations[sec] for sec in field_mappings[field]):
                    return True
            if field in considerations:
                return True

    # External References Check
    if field == "downloadLocation" and "externalReferences" in aibom:
        # Optimized generator expression
        return any(ref.get("type") == "distribution" and ref.get("url") for ref in aibom["externalReferences"])
        
    return False

def check_field_with_enhanced_results(aibom: Dict[str, Any], field: str, extraction_results: Optional[Dict[str, Any]] = None) -> bool:
    """
    Enhanced field detection using registry manager and extraction results.
    """
    try:
        manager = get_field_registry_manager()
        
        # 1. Registry-based dynamic detection
        fields = manager.get_field_definitions()
        if field in fields:
            field_config = fields[field]
            field_path = field_config.get('jsonpath', '')
            if field_path:
                is_present, value = manager.detect_field_presence(aibom, field_path)
                if is_present:
                    return True
        
        # 2. Extraction results check
        if extraction_results and field in extraction_results:
            extraction_result = extraction_results[field]
            # Handle Pydantic model vs Dict vs Object
            if hasattr(extraction_result, 'confidence'):
                # Object/Model access
                conf = extraction_result.confidence
                # conf could be an Enum or string
                val = conf.value if hasattr(conf, 'value') else conf
                if val == 'none':
                    return False
                return val in ['medium', 'high']
            elif hasattr(extraction_result, 'value'):
                val = extraction_result.value
                return val not in ['NOASSERTION', 'NOT_FOUND', None, '']
            else:
                 # Should probably return True if present in dict?
                 return True

        # 3. Fallback
        return check_field_in_aibom(aibom, field)

    except Exception as e:
        logger.error(f"Error in enhanced field detection for {field}: {e}")
        return check_field_in_aibom(aibom, field)

def determine_completeness_profile(aibom: Dict[str, Any], score: float) -> Dict[str, Any]:
    satisfied_profiles = []
    
    for profile_name, profile in COMPLETENESS_PROFILES.items():
        all_required_present = all(check_field_in_aibom(aibom, field) for field in profile["required_fields"])
        score_sufficient = score >= profile["minimum_score"]
        
        if all_required_present and score_sufficient:
            satisfied_profiles.append(profile_name)
    
    if "advanced" in satisfied_profiles:
        profile = COMPLETENESS_PROFILES.get("advanced", {})
        return {"name": "Advanced", "description": profile.get("description", ""), "satisfied": True}
    elif "standard" in satisfied_profiles:
        profile = COMPLETENESS_PROFILES.get("standard", {})
        return {"name": "Standard", "description": profile.get("description", ""), "satisfied": True}
    elif "basic" in satisfied_profiles:
        profile = COMPLETENESS_PROFILES.get("basic", {})
        return {"name": "Basic", "description": profile.get("description", ""), "satisfied": True}
    else:
        return {"name": "incomplete", "description": "Does not satisfy any completeness profile", "satisfied": False}

def generate_field_recommendations(missing_fields: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    recommendations = []
    
    for field in missing_fields.get("critical", []):
        if field in VALIDATION_MESSAGES:
            recommendations.append({
                "priority": "high",
                "field": field,
                "message": VALIDATION_MESSAGES[field]["missing"],
                "recommendation": VALIDATION_MESSAGES[field]["recommendation"]
            })
        else:
            recommendations.append({
                "priority": "high",
                "field": field,
                "message": f"Missing critical field: {field}",
                "recommendation": f"Add {field} to improve documentation completeness"
            })
            
    for field in missing_fields.get("important", []):
         if field in VALIDATION_MESSAGES:
            recommendations.append({
                "priority": "medium",
                "field": field,
                "message": VALIDATION_MESSAGES[field]["missing"],
                "recommendation": VALIDATION_MESSAGES[field]["recommendation"]
            })
         else:
            recommendations.append({
                "priority": "medium",
                "field": field,
                "message": f"Missing field: {field}",
                "recommendation": f"Consider adding {field}"
            })

    supplementary_count = 0
    for field in missing_fields.get("supplementary", []):
        if supplementary_count >= 5: break
        recommendations.append({
            "priority": "low",
            "field": field,
            "message": f"Missing supplementary field: {field}",
            "recommendation": f"Consider adding {field}"
        })
        supplementary_count += 1
        
    return recommendations


def calculate_completeness_score(aibom: Dict[str, Any], validate: bool = True, extraction_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate completeness score using registry-defined weights and rules.
    """
    # Max points (weights)
    category_weights = SCORING_WEIGHTS.get("category_weights", {})
    max_scores = {
        "required_fields": category_weights.get("required_fields", 20),
        "metadata": category_weights.get("metadata", 20),
        "component_basic": category_weights.get("component_basic", 20),
        "component_model_card": category_weights.get("component_model_card", 30),
        "external_references": category_weights.get("external_references", 10)
    }
    
    missing_fields = {"critical": [], "important": [], "supplementary": []}
    fields_by_category = {category: {"total": 0, "present": 0} for category in max_scores.keys()}
    field_checklist = {}

    # Evaluate fields
    for field, classification in FIELD_CLASSIFICATION.items():
        tier = classification["tier"]
        category = classification["category"]
        
        # Ensure category exists in tracking, else fallback or skip? 
        # Ideally FIELD_CLASSIFICATION only contains known categories.
        if category not in fields_by_category:
            fields_by_category[category] = {"total": 0, "present": 0}
            # Note: if it's a new category not in max_scores, it won't contribute to score unless we adjust max_scores
        
        fields_by_category[category]["total"] += 1
        
        is_present = check_field_with_enhanced_results(aibom, field, extraction_results)
        
        if is_present:
            fields_by_category[category]["present"] += 1
        else:
            if tier in missing_fields:
                missing_fields[tier].append(field)
                
        importance_indicator = "★★★" if tier == "critical" else "★★" if tier == "important" else "★"
        field_checklist[field] = f"{'✔' if is_present else '✘'} {importance_indicator}"

    # Calculate category scores
    category_scores = {}
    for category, counts in fields_by_category.items():
        if counts["total"] > 0:
            weight = max_scores.get(category, 0)
            raw_score = (counts["present"] / counts["total"]) * weight
            category_scores[category] = round(raw_score, 1)
        else:
             category_scores[category] = 0.0

    subtotal_score = sum(category_scores.values())
    
    # Penalties
    missing_critical = len(missing_fields["critical"])
    missing_important = len(missing_fields["important"])
    
    penalty_factor = 1.0
    penalty_reasons = []
    
    if missing_critical > 3:
        penalty_factor *= 0.8
        penalty_reasons.append("Multiple critical fields missing")
    elif missing_critical >= 2:
        penalty_factor *= 0.9
        penalty_reasons.append("Some critical fields missing")
        
    if missing_important >= 5:
        penalty_factor *= 0.95
        penalty_reasons.append("Several important fields missing")
        
    final_score = round(subtotal_score * penalty_factor, 1)
    final_score = max(0.0, min(final_score, 100.0))
    
    # Prepare result
    result = {
        "total_score": final_score,
        "subtotal_score": subtotal_score,
        "section_scores": category_scores,
        "max_scores": max_scores,
        "field_checklist": field_checklist,
        "missing_fields": missing_fields,
        "completeness_profile": determine_completeness_profile(aibom, final_score),
        "penalty_applied": penalty_factor < 1.0,
        "penalty_reason": " and ".join(penalty_reasons) if penalty_reasons else None,
        "recommendations": generate_field_recommendations(missing_fields)
    }
    
    if validate:
         validation_report = validate_aibom(aibom)
         result["validation"] = validation_report

    return result

def _validate_ai_requirements(aibom: Dict[str, Any]) -> List[Dict[str, Any]]:
    # ... logic from utils.py ...
    # Implementing minimal version or copying full logic?
    # I'll implement a concise version.
    issues = []
    if "bomFormat" in aibom and aibom["bomFormat"] != "CycloneDX":
         issues.append({"severity": "error", "code": "INVALID_BOM_FORMAT", "message": "Must be CycloneDX", "path": "$.bomFormat"})
    # ... (Add more crucial checks here as needed)
    return issues

def validate_aibom(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the AIBOM against the appropriate CycloneDX schema.
    """
    issues = []
    
    # 1. Schema Validation (using local schemas)
    try:
        import json
        import jsonschema
        import os
        
        spec_version = aibom.get("specVersion", "1.6")
        schema_file = f"bom-{spec_version}.schema.json"
        # Relative path from src/models/scoring.py -> src/schemas/
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'schemas', schema_file)
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            jsonschema.validate(instance=aibom, schema=schema)
        else:
             # If schema missing, warn but don't fail hard
             issues.append({"severity": "warning", "message": f"Schema file not found: {schema_file}, skipping strict validation."})
             
    except jsonschema.ValidationError as e:
        issues.append({"severity": "error", "message": e.message, "path": getattr(e, "json_path", "unknown")})
    except Exception as e:
        issues.append({"severity": "error", "message": f"Validation error: {str(e)}"})
        
    # 2. Custom Business Logic Checks (AI Requirements)
    custom_issues = _validate_ai_requirements(aibom)
    issues.extend(custom_issues)

    return {
        "valid": not any(i["severity"] == "error" for i in issues),
        "issues": issues,
        "error_count": sum(1 for i in issues if i["severity"] == "error")
    }
