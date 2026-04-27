"""
CycloneDX Schema Validation for AIBOM Generator.

Delegates to cyclonedx-python-lib's built-in JsonValidator which bundles the
official CycloneDX SNAPSHOT schemas and handles all sub-schema references
"""
import json
import logging
from typing import Any, Dict, List, Tuple

from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonValidator

logger = logging.getLogger(__name__)

_validator = JsonValidator(SchemaVersion.V1_6)


def validate_aibom(aibom: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate an AIBOM against the CycloneDX 1.6 schema.

    Args:
        aibom: The AIBOM dictionary to validate.
        strict: Unused — kept for interface compatibility.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    try:
        errors = _validator.validate_str(json.dumps(aibom), all_errors=True)
        if errors is None:
            return True, []
        messages = [f"[{'.'.join(str(p) for p in e.data_path) or 'root'}] {e.message}" for e in errors]
        return False, messages
    except Exception as e:
        logger.warning("Validation failed unexpectedly: %s", e)
        return False, [str(e)]


def get_validation_summary(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of schema validation results."""
    is_valid, errors = validate_aibom(aibom)
    return {
        "valid": is_valid,
        "error_count": len(errors),
        "errors": errors[:10] if not is_valid else [],
    }
