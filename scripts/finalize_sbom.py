#!/usr/bin/env python3
"""Finalize the release SBOM's root component metadata.

`cyclonedx-py environment --pyproject` populates the root component's name,
version, type, description and license, but not its group, PURL, supplier or
manufacturer. This script fills those in so the released SBOM clearly
identifies the project and its owning organization (GenAI-Security-Project).

Usage:
    python scripts/finalize_sbom.py <sbom.json> [version]

If ``version`` is given (e.g. the release tag) it overrides the root
component's version and is reflected in the PURL.
"""
from __future__ import annotations

import json
import sys

GROUP = "GenAI-Security-Project"
ORG_URL = "https://github.com/GenAI-Security-Project"
REPO_URL = "https://github.com/GenAI-Security-Project/aibom-generator"


def finalize(bom: dict, version_override: str | None = None) -> dict:
    component = bom.setdefault("metadata", {}).setdefault("component", {})

    name = component.get("name") or "owasp-aibom-generator"
    if version_override:
        # Release tags look like "v1.0.3"; store clean semver "1.0.3".
        if len(version_override) > 1 and version_override[0] in "vV" and version_override[1].isdigit():
            version_override = version_override[1:]
        component["version"] = version_override
    version = component.get("version")

    # Root component identity.
    component["type"] = "application"
    component["group"] = GROUP

    # PURL with the organization as the namespace/group.
    purl = f"pkg:pypi/{GROUP}/{name}"
    if version:
        purl = f"{purl}@{version}"
    component["purl"] = purl

    # Who made it (manufacturer) and who distributes it (supplier).
    org = {"name": GROUP, "url": [ORG_URL, REPO_URL]}
    component["manufacturer"] = dict(org)
    component["supplier"] = dict(org)

    return bom


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: finalize_sbom.py <sbom.json> [version]")
    path = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else None

    with open(path, encoding="utf-8") as fh:
        bom = json.load(fh)

    finalize(bom, version)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bom, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
