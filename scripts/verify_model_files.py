#!/usr/bin/env python3
"""Verify the model files a host loads against the SHA-256 hashes recorded in an AIBOM.

Usage:
    python scripts/verify_model_files.py AIBOM_JSON MODEL_DIR

MODEL_DIR holds the files as the host loads them. Every file in it is hashed
with SHA-256 and looked up among the file components of the
machine-learning-model component in the AIBOM. Only the Python standard
library is used, so the script can run on the host itself.

Each local file gets one status:
  MATCH       declared, and the SHA-256 equals the recorded one
  MISMATCH    declared, but the content differs from what the AIBOM records
  UNVERIFIED  declared without a hash (hashSource unknown), so it can't be checked
  UNDECLARED  not in the AIBOM at all

Declared files that are not in MODEL_DIR are only counted: an AIBOM lists
everything in the repository, and a host usually loads one variant.
Hidden files and directories (for example the .cache directory that
`hf download --local-dir` creates) are skipped.

Exit code: 0 when every local file is MATCH, 1 otherwise, 2 when the AIBOM
records no model file hashes.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_files(aibom: dict) -> Dict[str, Optional[str]]:
    """Map each declared file path to its recorded SHA-256 (None when unknown)."""
    files: Dict[str, Optional[str]] = {}
    for component in aibom.get("components", []):
        if component.get("type") != "machine-learning-model":
            continue
        for child in component.get("components", []):
            if child.get("type") != "file" or not child.get("name"):
                continue
            sha = None
            for h in child.get("hashes", []):
                if h.get("alg") == "SHA-256" and h.get("content"):
                    sha = h["content"].lower()
            files[child["name"]] = sha
    return files


def local_files(model_dir: str) -> Iterator[str]:
    for root, dirs, names in os.walk(model_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            if name.startswith("."):
                continue
            yield os.path.relpath(os.path.join(root, name), model_dir).replace(os.sep, "/")


def verify(aibom: dict, model_dir: str) -> Tuple[List[Tuple[str, str]], int]:
    """Return (status, path) for every local file and the number of declared files not present."""
    declared = declared_files(aibom)
    results = []
    seen = set()
    for path in local_files(model_dir):
        seen.add(path)
        if path not in declared:
            results.append(("UNDECLARED", path))
        elif declared[path] is None:
            results.append(("UNVERIFIED", path))
        elif sha256_of_file(os.path.join(model_dir, *path.split("/"))) == declared[path]:
            results.append(("MATCH", path))
        else:
            results.append(("MISMATCH", path))
    not_present = sum(1 for path in declared if path not in seen)
    return results, not_present


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify model files against the hashes in an AIBOM.")
    parser.add_argument("aibom", help="AIBOM JSON file produced by the generator")
    parser.add_argument("model_dir", help="directory with the model files as the host loads them")
    args = parser.parse_args(argv)

    with open(args.aibom, encoding="utf-8") as fh:
        aibom = json.load(fh)

    if not declared_files(aibom):
        print("The AIBOM records no model file hashes.")
        return 2

    results, not_present = verify(aibom, args.model_dir)
    for status, path in results:
        print(f"{status:<11} {path}")

    counts = {s: sum(1 for st, _ in results if st == s) for s in ("MATCH", "MISMATCH", "UNVERIFIED", "UNDECLARED")}
    summary = ", ".join(f"{k.lower()} {v}" for k, v in counts.items())
    print(f"{summary}; declared but not present {not_present}")
    if not results:
        print("No files found in the model directory.")
        return 1
    return 0 if counts["MATCH"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
