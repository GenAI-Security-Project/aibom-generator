"""Fail-closed policy gate for a CycloneDX AI-BOM.

The gate is intentionally local: it evaluates a document the generator already
produced. It does not fetch model metadata and it does not treat a missing or
unknown policy key as success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .scoring import check_field_in_aibom
from ..utils.validation import validate_aibom

POLICY_KEYS = frozenset(
    {
        "minimum_score",
        "required_fields",
        "denied_licenses",
        "require_license",
        "spec_version",
        "require_schema_valid",
        "allowed_tasks",
    }
)
_EXPRESSION_OPERATORS = frozenset({"AND", "OR", "WITH"})
_LICENSE_TOKEN = re.compile(r"[A-Za-z0-9.+-]+")


@dataclass(frozen=True)
class PolicyDecision:
    """Result of evaluating one BOM against one policy."""

    passed: bool
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "violations": list(self.violations)}


def evaluate_aibom_policy(
    aibom: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    score: float | None = None,
) -> PolicyDecision:
    """Return every policy violation. An empty or unknown policy fails closed."""
    if not isinstance(aibom, Mapping):
        return PolicyDecision(False, ["aibom must be an object"])
    if not isinstance(policy, Mapping) or not policy:
        return PolicyDecision(False, ["policy must be a non-empty object"])

    violations: list[str] = []
    unknown = sorted(set(policy) - POLICY_KEYS)
    if unknown:
        violations.append(f"unknown policy keys: {', '.join(unknown)}")

    spec_version = policy.get("spec_version")
    if spec_version is not None:
        if not isinstance(spec_version, str) or not spec_version.strip():
            violations.append("spec_version must be a non-empty string")
        elif aibom.get("specVersion") != spec_version:
            violations.append(
                f"specVersion is {aibom.get('specVersion')!r}, policy requires {spec_version!r}"
            )

    required_fields = policy.get("required_fields")
    if required_fields is not None:
        if not isinstance(required_fields, list) or not all(
            isinstance(item, str) and item.strip() for item in required_fields
        ):
            violations.append("required_fields must be a list of non-empty strings")
        else:
            missing = [
                item
                for item in required_fields
                if not check_field_in_aibom(dict(aibom), item)
            ]
            if missing:
                violations.append("missing required fields: " + ", ".join(missing))

    licenses = _license_tokens(aibom)
    if policy.get("require_license") is True and not licenses:
        violations.append("policy requires a license and none was found")
    elif "require_license" in policy and policy.get("require_license") is not True:
        if policy.get("require_license") is not False:
            violations.append("require_license must be a boolean")

    denied = policy.get("denied_licenses")
    if denied is not None:
        if not isinstance(denied, list) or not all(
            isinstance(item, str) and item.strip() for item in denied
        ):
            violations.append("denied_licenses must be a list of non-empty strings")
        else:
            denied_norm = {item.casefold() for item in denied}
            present = sorted(token for token in licenses if token.casefold() in denied_norm)
            if present:
                violations.append("denied licenses present: " + ", ".join(present))

    allowed_tasks = policy.get("allowed_tasks")
    if allowed_tasks is not None:
        if not isinstance(allowed_tasks, list) or not all(
            isinstance(item, str) and item.strip() for item in allowed_tasks
        ):
            violations.append("allowed_tasks must be a list of non-empty strings")
        else:
            task = _model_task(aibom)
            allowed = {item.casefold() for item in allowed_tasks}
            if task is None or task.casefold() not in allowed:
                violations.append(
                    f"model task {task!r} is not in the allowed task list"
                )

    minimum_score = policy.get("minimum_score")
    if minimum_score is not None:
        if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)):
            violations.append("minimum_score must be a number")
        else:
            observed = score if score is not None else _completeness_score(aibom)
            if observed is None or observed < float(minimum_score):
                violations.append(
                    f"completeness score {observed} is below minimum {minimum_score}"
                )

    if policy.get("require_schema_valid") is True:
        valid, errors = validate_aibom(dict(aibom))
        if not valid:
            detail = errors[0] if errors else "schema validation failed"
            violations.append(f"schema validation failed: {detail}")
    elif "require_schema_valid" in policy and policy.get("require_schema_valid") is not False:
        violations.append("require_schema_valid must be a boolean")

    return PolicyDecision(not violations, violations)


def _completeness_score(aibom: Mapping[str, Any]) -> float | None:
    from .scoring import calculate_completeness_score

    try:
        report = calculate_completeness_score(dict(aibom), validate=False)
    except Exception:
        return None
    total = report.get("total_score") if isinstance(report, dict) else None
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return None
    return float(total)


def _model_task(aibom: Mapping[str, Any]) -> str | None:
    components = aibom.get("components")
    if not isinstance(components, list) or not components:
        return None
    component = components[0]
    if not isinstance(component, dict):
        return None
    card = component.get("modelCard")
    if not isinstance(card, dict):
        return None
    parameters = card.get("modelParameters")
    if not isinstance(parameters, dict):
        return None
    task = parameters.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    return None


def _license_tokens(aibom: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    components = aibom.get("components")
    if not isinstance(components, list):
        return tokens
    for component in components:
        if not isinstance(component, dict):
            continue
        entries = component.get("licenses")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            license_obj = entry.get("license")
            if isinstance(license_obj, dict):
                for key in ("id", "name"):
                    value = license_obj.get(key)
                    if isinstance(value, str) and value.strip():
                        tokens.add(value.strip())
            expression = entry.get("expression")
            if isinstance(expression, str):
                for token in _LICENSE_TOKEN.findall(expression):
                    if token.upper() not in _EXPRESSION_OPERATORS:
                        tokens.add(token)
    return tokens
