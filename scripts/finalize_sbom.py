#!/usr/bin/env python3
"""Finalize the release SBOM's root component metadata and record artifact hashes.

`cyclonedx-py environment --pyproject` populates the root component's name,
version, type, description and license, but not its group, PURL, supplier or
manufacturer. This script fills those in so the released SBOM clearly
identifies the project and its owning organization (GenAI-Security-Project).

It also records the produced release artifacts (wheel / sdist) on the root
component as `distribution` external references, each carrying the artifact's
SHA-256 and SHA-512 hashes, so the SBOM captures exactly what was shipped.

Usage:
    python scripts/finalize_sbom.py <sbom.json> \
        [--version <tag>] \
        [--download-base-url <url>] \
        [--artifact <file> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

GROUP = "GenAI-Security-Project"
ORG_URL = "https://github.com/GenAI-Security-Project"
REPO_URL = "https://github.com/GenAI-Security-Project/aibom-generator"

# (CycloneDX hash alg name, hashlib constructor name)
HASH_ALGS = (("SHA-256", "sha256"), ("SHA-512", "sha512"))


def _normalize_version(version: str) -> str:
    # Release tags look like "v1.0.3"; store clean semver "1.0.3".
    if len(version) > 1 and version[0] in "vV" and version[1].isdigit():
        return version[1:]
    return version


def _file_hashes(path: str) -> list[dict]:
    hashers = {alg: hashlib.new(constructor) for alg, constructor in HASH_ALGS}
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            for h in hashers.values():
                h.update(chunk)
    return [{"alg": alg, "content": hashers[alg].hexdigest()} for alg, _ in HASH_ALGS]


def finalize(
    bom: dict,
    version_override: str | None = None,
    artifacts: list[str] | None = None,
    download_base_url: str | None = None,
) -> dict:
    component = bom.setdefault("metadata", {}).setdefault("component", {})

    name = component.get("name") or "owasp-aibom-generator"
    if version_override:
        component["version"] = _normalize_version(version_override)
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

    # Record each produced release artifact with its hashes.
    for artifact in artifacts or []:
        filename = os.path.basename(artifact)
        if download_base_url:
            url = f"{download_base_url.rstrip('/')}/{filename}"
        else:
            url = filename
        ref = {
            "type": "distribution",
            "url": url,
            "comment": f"Release artifact: {filename}",
            "hashes": _file_hashes(artifact),
        }
        component.setdefault("externalReferences", []).append(ref)

    return bom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sbom", help="Path to the CycloneDX JSON SBOM to finalize.")
    parser.add_argument("--version", help="Release version/tag for the root component.")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Path to a produced release artifact (repeatable).",
    )
    parser.add_argument(
        "--download-base-url",
        help="Base URL the artifacts will be downloadable from (e.g. the release assets URL).",
    )
    args = parser.parse_args()

    with open(args.sbom, encoding="utf-8") as fh:
        bom = json.load(fh)

    finalize(bom, args.version, args.artifact, args.download_base_url)

    with open(args.sbom, "w", encoding="utf-8") as fh:
        json.dump(bom, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
