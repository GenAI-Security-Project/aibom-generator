import json
import copy
from typing import Dict, Any

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
        # Any specific CycloneDX mappings or adjustments can be placed here over time.
        
    elif bom_type.lower() == "spdx":
        # Placeholder for future SPDX generation logic
        output["bomFormat"] = "SPDX"
        output["specVersion"] = spec_version
        # Since spdx mapping logic to AIBOM isn't fully built yet, this serves as the routing hook
        pass
        
    return json.dumps(output, indent=2)
