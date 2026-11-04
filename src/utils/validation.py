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

_SCHEMA_VERSIONS = {
    "1.6": SchemaVersion.V1_6,
    "1.7": SchemaVersion.V1_7,
}
_validator = JsonValidator(SchemaVersion.V1_6)
_validators: Dict[str, JsonValidator] = {"1.6": _validator}


def _validator_for(spec_version: str) -> JsonValidator:
    schema_version = _SCHEMA_VERSIONS.get(spec_version)
    if schema_version is None:
        raise ValueError(f"Unsupported CycloneDX schema version: {spec_version}")
    if spec_version not in _validators:
        _validators[spec_version] = JsonValidator(schema_version)
    return _validators[spec_version]


def _format_validation_error(error: Any) -> str:
    message = getattr(error, "message", None)
    if message is None:
        message = str(error)
    else:
        message = str(message)

    for path_attr in ("data_path", "path", "json_path", "absolute_path", "relative_path"):
        try:
            path = getattr(error, path_attr)
        except Exception:
            continue

        if path is None:
            continue

        if isinstance(path, str):
            location = path
        else:
            try:
                location = ".".join(str(part) for part in path)
            except TypeError:
                location = str(path)

        return f"[{location or 'root'}] {message}"

    return message


def validate_aibom(aibom: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate an AIBOM against its declared CycloneDX schema.

    Args:
        aibom: The AIBOM dictionary to validate.
        strict: Unused — kept for interface compatibility.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    try:
        spec_version = str(aibom.get("specVersion", "1.6"))
        errors = _validator_for(spec_version).validate_str(
            json.dumps(aibom),
            all_errors=True,
        )
        if errors is None:
            return True, []
        messages = [_format_validation_error(e) for e in errors]
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
