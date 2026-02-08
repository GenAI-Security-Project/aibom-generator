"""
CycloneDX 1.6 Schema Validation for AIBOM Generator.

This module provides validation of generated AIBOMs against the official
CycloneDX 1.6 JSON schema to ensure compliance and interoperability.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure to handle requests import if it's not a core dependency (it is in my project)
import requests
import jsonschema
from jsonschema import Draft7Validator, ValidationError
from referencing import Registry, Resource

# Module-level logger
logger = logging.getLogger(__name__)

# CycloneDX schema configuration
CYCLONEDX_1_6_SCHEMA_URL = "https://raw.githubusercontent.com/CycloneDX/specification/master/schema/bom-1.6.schema.json"
# Correct path relative to this file: src/utils/../schemas -> src/schemas
SCHEMA_CACHE_DIR = Path(__file__).parent.parent / "schemas"
SCHEMA_CACHE_FILE = SCHEMA_CACHE_DIR / "bom-1.6.schema.json"

# Global schema cache
_cached_schema: Optional[Dict[str, Any]] = None


def _ensure_cache_dir() -> None:
    """Ensure the schema cache directory exists."""
    SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_schema_from_cache() -> Optional[Dict[str, Any]]:
    """Load schema from local cache if available."""
    if SCHEMA_CACHE_FILE.exists():
        try:
            with open(SCHEMA_CACHE_FILE, "r", encoding="utf-8") as f:
                schema = json.load(f)
                logger.debug("Loaded CycloneDX 1.6 schema from cache")
                return schema
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load cached schema: %s", e)
    return None


def _download_schema() -> Optional[Dict[str, Any]]:
    """Download the CycloneDX 1.6 schema from the official repository."""
    try:
        logger.info("Downloading CycloneDX 1.6 schema from %s", CYCLONEDX_1_6_SCHEMA_URL)
        response = requests.get(CYCLONEDX_1_6_SCHEMA_URL, timeout=30)
        response.raise_for_status()
        schema = response.json()

        # Cache the schema locally
        _ensure_cache_dir()
        with open(SCHEMA_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        logger.info("CycloneDX 1.6 schema downloaded and cached")

        return schema
    except requests.RequestException as e:
        logger.error("Failed to download CycloneDX schema: %s", e)
        return None
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to parse or cache schema: %s", e)
        return None


def load_schema(force_download: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load the CycloneDX 1.6 JSON schema.

    Uses in-memory cache first, then file cache, then downloads if needed.

    Args:
        force_download: If True, download fresh schema even if cached.

    Returns:
        The schema dictionary, or None if loading failed.
    """
    global _cached_schema

    # Return in-memory cache if available
    if _cached_schema is not None and not force_download:
        return _cached_schema

    # Try loading from file cache
    if not force_download:
        schema = _load_schema_from_cache()
        if schema:
            _cached_schema = schema
            return schema

    # Download fresh schema
    schema = _download_schema()
    if schema:
        _cached_schema = schema

    return schema


def _format_validation_error(error: ValidationError) -> str:
    """Format a validation error into a readable message."""
    path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
    return f"[{path}] {error.message}"


def validate_aibom(aibom: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate an AIBOM against the CycloneDX 1.6 schema.

    Args:
        aibom: The AIBOM dictionary to validate.
        strict: If True, fail on any schema deviation. If False, collect all errors.

    Returns:
        Tuple of (is_valid, list of error messages).
        If valid, returns (True, []).
        If invalid, returns (False, [error1, error2, ...]).
    """
    schema = load_schema()

    if schema is None:
        logger.warning("Could not load CycloneDX schema - skipping validation")
        return True, ["Schema unavailable"]
    
    # Load SPDX schema for reference resolution
    spdx_path = SCHEMA_CACHE_DIR / "spdx.schema.json"
    registry = Registry()
    if spdx_path.exists():
        try:
            with open(spdx_path, "r", encoding="utf-8") as f:
                spdx_schema = json.load(f)
            resource = Resource.from_contents(spdx_schema)
            registry = registry.with_resource(uri="spdx.schema.json", resource=resource)
        except Exception as e:
            logger.warning("Failed to load SPDX schema for validation: %s", e)

    validator = Draft7Validator(schema, registry=registry)
    errors = sorted(validator.iter_errors(aibom), key=lambda e: e.path)
    
    if not errors:
        return True, []
        
    error_messages = [_format_validation_error(e) for e in errors]
    return False, error_messages

def get_validation_summary(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """Get a summary of schema validation results."""
    is_valid, errors = validate_aibom(aibom)
    return {
        "valid": is_valid,
        "error_count": len(errors),
        "errors": errors[:10] if not is_valid else [] # Limit to first 10
    }
