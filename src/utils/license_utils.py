"""
License utility functions for normalising and verifying SPDX license IDs.
"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Common mapping of license names or incomplete IDs to generic URLs or valid SPDX
LICENSE_URLS: Dict[str, str] = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "MIT": "https://opensource.org/licenses/MIT",
    "BSD-3-Clause": "https://opensource.org/licenses/BSD-3-Clause",
    "BSD-2-Clause": "https://opensource.org/licenses/BSD-2-Clause",
    "GPL-3.0-only": "https://www.gnu.org/licenses/gpl-3.0.txt",
    "GPL-2.0-only": "https://www.gnu.org/licenses/gpl-2.0.txt",
    "LGPL-3.0-only": "https://www.gnu.org/licenses/lgpl-3.0.txt",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/legalcode",
    "CC-BY-ND-4.0": "https://creativecommons.org/licenses/by-nd/4.0/legalcode",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
    "CC-BY-NC-ND-4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
    "MPL-2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "Unlicense": "https://unlicense.org/",
    "nvidia-open-model-license": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
}

# Mapping common variations to valid SPDX IDs
LICENSE_MAPPING: Dict[str, str] = {
    "apache license 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "mit license": "MIT",
    "bsd-3-clause": "BSD-3-Clause",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc0-1.0": "CC0-1.0",
    "gpl-3.0": "GPL-3.0-only",
    "nvidia open model license agreement": "nvidia-open-model-license",
    # Add more as needed
}

def normalize_license_id(license_id: str) -> Optional[str]:
    """
    Normalize a license string to a valid SPDX ID if possible.
    Returns None if no clear mapping is found.
    """
    if not license_id:
        return None
        
    # Check if exact match in our known list
    if license_id in LICENSE_URLS:
        return license_id
        
    lower_id = license_id.lower()
    
    # Check mapping
    if lower_id in LICENSE_MAPPING:
        return LICENSE_MAPPING[lower_id]
        
    # Check if any key in URLS (case-insensitive) matches
    for valid_id in LICENSE_URLS:
        if valid_id.lower() == lower_id:
            return valid_id
            
    # Simple heuristic: if it looks like an ID, return it (e.g. contains hyphens/dots, no spaces)
    if " " not in license_id and len(license_id) < 50:
         # Might be valid, might not. Let's return it and rely on validation warnings.
         return license_id
         
    return None

def get_license_url(license_id: str) -> str:
    """Get the URL for a license based on its ID."""
    if license_id in LICENSE_URLS:
        return LICENSE_URLS[license_id]
    
    # Case insensitive fallback
    lower_id = license_id.lower()
    for valid_id, url in LICENSE_URLS.items():
        if valid_id.lower() == lower_id:
            return url
            
    return f"https://spdx.org/licenses/{license_id}.html"
