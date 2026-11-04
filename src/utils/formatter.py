import json
import copy
from typing import Dict, Any

from .integrity import add_integrity_attestation

CYCLONEDX_SCHEMA_BY_SPEC_VERSION = {
    "1.6": "http://cyclonedx.org/schema/bom-1.6.schema.json",
    "1.7": "http://cyclonedx.org/schema/bom-1.7.schema.json",
}

def export_aibom(aibom: Dict[str, Any], bom_type: str = "cyclonedx", spec_version: str = "1.6") -> str:
    """
    Exports the internal AIBOM object into a specified format and specification version.
    Returns the generated SBOM as a formatted JSON string.
    """
    # Create a deep copy to avoid modifying the original unified object
    output = copy.deepcopy(aibom)
    
    if bom_type.lower() == "cyclonedx":
        output["bomFormat"] = "CycloneDX"
        output["specVersion"] = spec_version
        if spec_version in CYCLONEDX_SCHEMA_BY_SPEC_VERSION:
            output["$schema"] = CYCLONEDX_SCHEMA_BY_SPEC_VERSION[spec_version]
        # Any specific CycloneDX mappings or adjustments can be placed here over time.
        
    elif bom_type.lower() == "spdx":
        # Placeholder for future SPDX generation logic
        output["bomFormat"] = "SPDX"
        output["specVersion"] = spec_version
        # Since spdx mapping logic to AIBOM isn't fully built yet, this serves as the routing hook
        pass
        
    return json.dumps(add_integrity_attestation(output), indent=2)
