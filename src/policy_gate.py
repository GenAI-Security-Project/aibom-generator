"""CLI for the local CycloneDX AI-BOM policy gate.

Usage:
    python -m src.policy_gate --bom sbom.json --policy policy.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .models.policy_gate import evaluate_aibom_policy


def load_document(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed policy gate for a CycloneDX AI-BOM")
    parser.add_argument("--bom", required=True, help="Path to a CycloneDX AI-BOM JSON document")
    parser.add_argument("--policy", required=True, help="Path to a JSON or YAML policy file")
    args = parser.parse_args(argv)

    try:
        aibom = load_document(Path(args.bom))
        policy = load_document(Path(args.policy))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"policy gate could not read input: {exc}", file=sys.stderr)
        return 2

    decision = evaluate_aibom_policy(aibom, policy)
    print(json.dumps(decision.to_dict(), indent=2))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
