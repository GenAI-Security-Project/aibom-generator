import copy
import hashlib
import hmac
import json
import re
from typing import Any

import rfc8785

INTEGRITY_PROPERTY_PREFIX = "genai:aibom:integrity:"
INTEGRITY_CANONICALIZATION_PROPERTY = f"{INTEGRITY_PROPERTY_PREFIX}canonicalization"
INTEGRITY_ALGORITHM_PROPERTY = f"{INTEGRITY_PROPERTY_PREFIX}algorithm"
INTEGRITY_DIGEST_PROPERTY = f"{INTEGRITY_PROPERTY_PREFIX}digest"
SPDX_ATTESTATION_PREFIX = "OWASP-BOM-INTEGRITY:"
CANONICALIZATION = "RFC8785"
ALGORITHM = "SHA-256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def detect_bom_format(aibom: dict[str, Any]) -> str:
    bom_format = str(aibom.get("bomFormat", "")).lower()
    if bom_format == "cyclonedx":
        return "cyclonedx"
    if bom_format == "spdx" or str(aibom.get("spdxVersion", "")).startswith("SPDX-"):
        return "spdx"
    raise ValueError("Unsupported BOM format")


def _strip_integrity_attestation(aibom: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(aibom)
    bom_format = detect_bom_format(unsigned)

    if bom_format == "cyclonedx":
        properties = unsigned.get("properties")
        if isinstance(properties, list):
            filtered = [
                prop for prop in properties
                if not (
                    isinstance(prop, dict)
                    and str(prop.get("name", "")).startswith(INTEGRITY_PROPERTY_PREFIX)
                )
            ]
            if filtered:
                unsigned["properties"] = filtered
            else:
                unsigned.pop("properties", None)
    else:
        annotations = unsigned.get("annotations")
        if isinstance(annotations, list):
            filtered = [
                annotation for annotation in annotations
                if not (
                    isinstance(annotation, dict)
                    and str(annotation.get("comment", "")).startswith(SPDX_ATTESTATION_PREFIX)
                )
            ]
            if filtered:
                unsigned["annotations"] = filtered
            else:
                unsigned.pop("annotations", None)

    return unsigned


def calculate_integrity_digest(aibom: dict[str, Any]) -> str:
    unsigned = _strip_integrity_attestation(aibom)
    canonical_bytes = rfc8785.dumps(unsigned)
    return hashlib.sha256(canonical_bytes).hexdigest()


def add_integrity_attestation(aibom: dict[str, Any]) -> dict[str, Any]:
    attested = _strip_integrity_attestation(aibom)
    digest = calculate_integrity_digest(attested)
    bom_format = detect_bom_format(attested)

    if bom_format == "cyclonedx":
        properties = list(attested.get("properties", []))
        properties.extend([
            {"name": INTEGRITY_CANONICALIZATION_PROPERTY, "value": CANONICALIZATION},
            {"name": INTEGRITY_ALGORITHM_PROPERTY, "value": ALGORITHM},
            {"name": INTEGRITY_DIGEST_PROPERTY, "value": digest},
        ])
        attested["properties"] = properties
    else:
        created = (
            attested.get("creationInfo", {}).get("created")
            or "1970-01-01T00:00:00Z"
        )
        payload = json.dumps(
            {
                "algorithm": ALGORITHM,
                "canonicalization": CANONICALIZATION,
                "digest": digest,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        annotations = list(attested.get("annotations", []))
        annotations.append({
            "annotationDate": created,
            "annotationType": "OTHER",
            "annotator": "Tool: OWASP BOM Generator",
            "comment": f"{SPDX_ATTESTATION_PREFIX}{payload}",
        })
        attested["annotations"] = annotations

    return attested


def _cyclonedx_attestation(aibom: dict[str, Any]) -> tuple[bool, dict | None, list[str]]:
    properties = aibom.get("properties", [])
    if not isinstance(properties, list):
        return False, None, ["CycloneDX properties must be an array"]

    values: dict[str, list[Any]] = {}
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        name = str(prop.get("name", ""))
        if name.startswith(INTEGRITY_PROPERTY_PREFIX):
            values.setdefault(name, []).append(prop.get("value"))

    if not values:
        return False, None, []

    required = {
        INTEGRITY_CANONICALIZATION_PROPERTY,
        INTEGRITY_ALGORITHM_PROPERTY,
        INTEGRITY_DIGEST_PROPERTY,
    }
    errors = []
    missing = required - values.keys()
    if missing:
        errors.append(f"Missing integrity properties: {', '.join(sorted(missing))}")
    duplicates = sorted(name for name, entries in values.items() if len(entries) != 1)
    if duplicates:
        errors.append(f"Duplicate integrity properties: {', '.join(duplicates)}")
    if errors:
        return True, None, errors

    return True, {
        "canonicalization": values[INTEGRITY_CANONICALIZATION_PROPERTY][0],
        "algorithm": values[INTEGRITY_ALGORITHM_PROPERTY][0],
        "digest": values[INTEGRITY_DIGEST_PROPERTY][0],
    }, []


def _spdx_attestation(aibom: dict[str, Any]) -> tuple[bool, dict | None, list[str]]:
    annotations = aibom.get("annotations", [])
    if not isinstance(annotations, list):
        return False, None, ["SPDX annotations must be an array"]

    matching = [
        annotation for annotation in annotations
        if isinstance(annotation, dict)
        and str(annotation.get("comment", "")).startswith(SPDX_ATTESTATION_PREFIX)
    ]
    if not matching:
        return False, None, []
    if len(matching) != 1:
        return True, None, ["Expected exactly one integrity annotation"]

    comment = str(matching[0]["comment"])
    try:
        payload = json.loads(comment[len(SPDX_ATTESTATION_PREFIX):])
    except (TypeError, ValueError) as exc:
        return True, None, [f"Invalid integrity annotation: {exc}"]
    if not isinstance(payload, dict):
        return True, None, ["Integrity annotation payload must be an object"]
    return True, payload, []


def verify_integrity_attestation(aibom: dict[str, Any]) -> dict[str, Any]:
    try:
        bom_format = detect_bom_format(aibom)
    except ValueError as exc:
        return {
            "present": False,
            "valid": False,
            "status": "invalid",
            "errors": [str(exc)],
        }

    present, attestation, errors = (
        _cyclonedx_attestation(aibom)
        if bom_format == "cyclonedx"
        else _spdx_attestation(aibom)
    )
    if not present:
        return {
            "present": False,
            "valid": None,
            "status": "unattested",
            "errors": errors,
        }
    if errors or attestation is None:
        return {
            "present": True,
            "valid": False,
            "status": "invalid",
            "errors": errors,
        }

    canonicalization = attestation.get("canonicalization")
    algorithm = attestation.get("algorithm")
    expected = attestation.get("digest")
    errors = []
    if canonicalization != CANONICALIZATION:
        errors.append(f"Unsupported canonicalization: {canonicalization}")
    if algorithm != ALGORITHM:
        errors.append(f"Unsupported digest algorithm: {algorithm}")
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        errors.append("Integrity digest must be 64 lowercase hexadecimal characters")
    if errors:
        return {
            "present": True,
            "valid": False,
            "status": "invalid",
            "expected": expected,
            "errors": errors,
        }

    try:
        actual = calculate_integrity_digest(aibom)
    except (TypeError, ValueError, rfc8785.CanonicalizationError) as exc:
        return {
            "present": True,
            "valid": False,
            "status": "invalid",
            "expected": expected,
            "errors": [f"Canonicalization failed: {exc}"],
        }

    valid = hmac.compare_digest(expected, actual)
    return {
        "present": True,
        "valid": valid,
        "status": "verified" if valid else "mismatch",
        "expected": expected,
        "actual": actual,
        "errors": [] if valid else ["Integrity digest mismatch"],
    }
