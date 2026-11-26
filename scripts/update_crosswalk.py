"""Vendor the OWASP GenAI Crosswalk AIBOM data at a pinned commit.

Downloads the two files the risk section of the AIBOM is built from:

  data/exports/aibom-index.json      generated: risks, frameworks, controls, links
  data/aibom-applicability.json      hand-authored (DRAFT): which risk lists apply

from https://github.com/GenAI-Security-Project/crosswalk at one commit, writes
them to src/data/owasp_crosswalk/, and records the commit and each file's
SHA-256 in SOURCE.json. The generator never fetches the crosswalk at run time.

Usage:
    python scripts/update_crosswalk.py <40-character crosswalk commit SHA>
    python scripts/update_crosswalk.py --verify     # check files against SOURCE.json
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "GenAI-Security-Project/crosswalk"
FILES = {
    "aibom-index.json": "data/exports/aibom-index.json",
    "aibom-applicability.json": "data/aibom-applicability.json",
}
DEST = Path(__file__).resolve().parent.parent / "src" / "data" / "owasp_crosswalk"
SOURCE = DEST / "SOURCE.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        sys.exit("give the full 40-character commit SHA, not a branch or tag")
    DEST.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, path in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"
        with urllib.request.urlopen(url, timeout=60) as resp:  # fixed https host, pinned commit
            data = resp.read()
        json.loads(data)  # refuse to vendor anything that is not JSON
        (DEST / name).write_bytes(data)
        files[name] = {"path": path, "sha256": _sha256(data)}
        print(f"{name}: {len(data):,} bytes")
    record = {
        "repository": f"https://github.com/{REPO}",
        "commit": commit,
        "files": files,
        "license": "CC BY-SA 4.0",
    }
    # Bytes, so the file has LF line endings on every platform.
    SOURCE.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    print(f"pinned to {commit}")


def verify() -> int:
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    bad = [n for n, f in record["files"].items() if _sha256((DEST / n).read_bytes()) != f["sha256"]]
    for n in bad:
        print(f"{n}: SHA-256 does not match SOURCE.json")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commit", nargs="?", help="crosswalk commit SHA (40 hex characters)")
    parser.add_argument("--verify", action="store_true", help="check vendored files against SOURCE.json")
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify())
    if not args.commit:
        parser.error("a commit SHA is required unless --verify is given")
    fetch(args.commit)


if __name__ == "__main__":
    main()
