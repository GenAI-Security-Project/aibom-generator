"""Conservative, evidence-scoped G7 documentation coverage for CycloneDX 1.7."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator


PROFILE_PATH = Path(__file__).with_name("g7_cyclonedx_1_7_profile.json")
_VALIDATOR = JsonStrictValidator(SchemaVersion.V1_7)
_UNKNOWN = {"unknown", "noassertion", "not_found", "n/a", "not available", "unspecified"}
_STATUSES = ("present", "partial", "missing", "review", "unknown", "not_assessed")
_SOFTWARE_TYPES = {"application", "framework", "library", "container", "platform", "operating-system", "firmware"}


@lru_cache(maxsize=1)
def load_g7_profile() -> Dict[str, Any]:
    with PROFILE_PATH.open(encoding="utf-8") as profile_file:
        return json.load(profile_file)


@dataclass
class _Subject:
    path: str
    value: Dict[str, Any]


def _values(value: Any, segments: List[str], path: str) -> Iterable[Tuple[str, Any]]:
    if not segments:
        yield path, value
    elif segments[0] == "*" and isinstance(value, list):
        for index, child in enumerate(value):
            yield from _values(child, segments[1:], f"{path}/{index}")
    elif isinstance(value, dict) and segments[0] in value:
        key = segments[0]
        escaped = key.replace("~", "~0").replace("/", "~1")
        yield from _values(value[key], segments[1:], f"{path}/{escaped}")


def _meaningful(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in _UNKNOWN
    if isinstance(value, dict):
        return any(_meaningful(child) for child in value.values())
    if isinstance(value, list):
        return any(_meaningful(child) for child in value)
    # Only explicit text evidence is used by this profile, except numeric BOM version.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _components(document: Dict[str, Any]) -> List[_Subject]:
    result = []

    def visit(component: Dict[str, Any], path: str) -> None:
        result.append(_Subject(path, component))
        for index, child in enumerate(component.get("components", [])):
            visit(child, f"{path}/components/{index}")

    if "component" in document.get("metadata", {}):
        visit(document["metadata"]["component"], "/metadata/component")
    for index, component in enumerate(document.get("components", [])):
        visit(component, f"/components/{index}")
    return result


def _cohorts(document: Dict[str, Any]) -> Dict[str, List[_Subject]]:
    components = _components(document)
    by_ref = {c.value["bom-ref"]: c for c in components if c.value.get("bom-ref")}
    models = [c for c in components if c.value.get("type") == "machine-learning-model"]
    datasets = {
        c.path: c for c in components if c.value.get("type") == "data"
        and any(data.get("type") == "dataset" for data in c.value.get("data", []))
    }
    for model in models:
        for path, data in _values(model.value, ["modelCard", "modelParameters", "datasets", "*"], model.path):
            if "ref" in data:
                target = by_ref.get(data["ref"])
                if target and target.value.get("type") == "data":
                    datasets[target.path] = target
                else:
                    # Keep unresolved/external references visible; do not invent a dataset.
                    datasets[path] = _Subject(path, {"unresolved_ref": data["ref"]})
            elif data.get("type") == "dataset":
                datasets[path] = _Subject(path, data)
            else:
                datasets[path] = _Subject(path, {"unresolved_ref": None})

    # A BOM about a model is not necessarily about its deployment system. Only
    # select an application subject that actually contains or depends on a model.
    systems = []
    subject = next((c for c in components if c.path == "/metadata/component"), None)
    if subject and subject.value.get("type") == "application":
        edges = {edge["ref"]: edge.get("dependsOn", []) for edge in document.get("dependencies", [])}
        pending = [subject.value.get("bom-ref")]
        reached = set()
        while pending:
            ref = pending.pop()
            if ref and ref not in reached:
                reached.add(ref)
                pending.extend(edges.get(ref, []))
        if any(m.path.startswith(subject.path + "/components/") or m.value.get("bom-ref") in reached for m in models):
            systems.append(subject)
    tools = document.get("metadata", {}).get("tools", {})
    if isinstance(tools, list):
        tool_subjects = [_Subject(f"/metadata/tools/{index}", tool) for index, tool in enumerate(tools)]
    else:
        tool_subjects = [
            _Subject(f"/metadata/tools/{kind}/{index}", tool)
            for kind in ("components", "services") for index, tool in enumerate(tools.get(kind, []))
        ]
    return {"document": [_Subject("", document)], "system": systems, "model": models,
            "dataset": list(datasets.values()), "tool": tool_subjects}


def _relationships(document: Dict[str, Any], subject: _Subject, check: str) -> List[str]:
    components = _components(document)
    by_ref = {c.value["bom-ref"]: c.value for c in components if c.value.get("bom-ref")}
    cohorts = _cohorts(document)
    ai_refs = {c.value.get("bom-ref") for c in cohorts["model"] + cohorts["system"]}
    evidence = []
    for index, edge in enumerate(document.get("dependencies", [])):
        source = by_ref.get(edge["ref"])
        if not source:
            continue
        if check in {"dataset_dependencies", "subject_dependencies"} and edge["ref"] != subject.value.get("bom-ref"):
            continue
        if check == "software_dependencies" and edge["ref"] not in ai_refs:
            continue
        for target_index, ref in enumerate(edge.get("dependsOn", [])):
            target = by_ref.get(ref)
            if not target or ref == edge["ref"]:
                continue
            if check == "software_dependencies" and target.get("type") not in _SOFTWARE_TYPES:
                continue
            if check == "dataset_dependencies" and target.get("type") not in _SOFTWARE_TYPES | {"data"}:
                continue
            evidence.append(f"/dependencies/{index}/dependsOn/{target_index}")
    return evidence


def _evidence(document: Dict[str, Any], subject: _Subject, element: Dict[str, Any]) -> Tuple[List[str], bool, bool]:
    """Return JSON pointers, whether all required groups matched, and unknown presence."""
    check = element.get("check")
    if check in {"dependencies", "dataset_dependencies", "software_dependencies", "subject_dependencies"}:
        evidence = _relationships(document, subject, check)
        return evidence, bool(evidence), False
    evidence = []
    unknown = False
    matched = []
    for alternatives in element.get("evidence_groups", []):
        group_hits = []
        for path in alternatives:
            for pointer, value in _values(subject.value, path.split("."), subject.path):
                if isinstance(value, str) and value.strip().casefold() in _UNKNOWN:
                    unknown = True
                if _meaningful(value):
                    if check == "security_txt" and not (
                        isinstance(value, str) and urlparse(value).scheme in {"https", "http"}
                        and urlparse(value).path in {"/.well-known/security.txt", "/security.txt"}
                    ):
                        continue
                    group_hits.append(pointer)
        evidence.extend(group_hits)
        matched.append(bool(group_hits))
    return sorted(set(evidence)), bool(matched) and all(matched), unknown


def _evaluate(document: Dict[str, Any], subjects: List[_Subject], element: Dict[str, Any]) -> Dict[str, Any]:
    details = []
    for subject in subjects:
        evidence, complete, unknown = _evidence(document, subject, element)
        if "unresolved_ref" in subject.value:
            status = "review"
        elif element["coverage"] == "manual":
            status = "review"
        elif complete:
            status = "partial" if element["coverage"] == "partial" else "present"
        elif evidence:
            status = "partial"
        else:
            status = "unknown" if unknown else "missing"
        details.append({"path": subject.path or "/", "status": status, "evidence": evidence})
    statuses = {detail["status"] for detail in details}
    if not details:
        status = "review"  # Absence does not establish non-applicability.
    elif len(statuses) == 1:
        status = statuses.pop()
    elif "present" in statuses or "partial" in statuses:
        status = "partial"
    elif "review" in statuses:
        status = "review"
    else:
        status = "missing"
    return {"status": status, "subjects": details}


def calculate_g7_completeness(aibom: Dict[str, Any]) -> Dict[str, Any]:
    """Assess known, scoped evidence, preserving incomplete and review-only findings.

    Schema-invalid, non-1.7 and non-AI documents receive no percentage. Even a
    'present' result means documented evidence, not verified facts or compliance.
    """
    profile = load_g7_profile()
    report_status, reason = "assessed", None
    cohorts: Dict[str, List[_Subject]] = {}
    if not isinstance(aibom, dict) or aibom.get("bomFormat") != "CycloneDX" or aibom.get("specVersion") != "1.7":
        report_status, reason = "unsupported", "This profile requires a CycloneDX 1.7 JSON document."
    else:
        try:
            error = _VALIDATOR.validate_str(json.dumps(aibom, allow_nan=False))
        except (TypeError, ValueError, RecursionError) as exc:
            error = str(exc)
        if error is not None:
            report_status, reason = "invalid", f"CycloneDX 1.7 schema validation failed: {error}"
        else:
            components = _components(aibom)
            refs = [c.value["bom-ref"] for c in components if c.value.get("bom-ref")]
            if len(refs) != len(set(refs)):
                report_status, reason = "invalid", "Component bom-ref values must be unique."
            else:
                cohorts = _cohorts(aibom)
                if not cohorts["model"] and not cohorts["dataset"]:
                    report_status, reason = "not_applicable", "No typed AI models or datasets were found."
    results = []
    clusters: Dict[str, Dict[str, int]] = {}
    for element in profile["elements"]:
        result = {key: element[key] for key in ("id", "cluster", "name", "coverage", "cyclonedx", "source_section", "source_page", "notes")}
        result.update(
            _evaluate(aibom, cohorts[element["scope"]], element)
            if report_status == "assessed" else {"status": "not_assessed", "subjects": []}
        )
        results.append(result)
        counts = clusters.setdefault(element["cluster"], {"total": 0, **dict.fromkeys(_STATUSES, 0)})
        counts["total"] += 1
        counts[result["status"]] += 1
    counts = {status: sum(result["status"] == status for result in results) for status in _STATUSES}
    return {
        "profile": profile["profile"], "profile_version": profile["profile_version"],
        "spec_version": "1.7", "input_spec_version": aibom.get("specVersion") if isinstance(aibom, dict) else None,
        "status": report_status, "reason": reason, "advisory": True,
        "source": profile["source"], "total": len(results), **counts,
        "percentage": round(counts["present"] / len(results) * 100, 1) if report_status == "assessed" else None,
        "clusters": clusters, "elements": results,
    }
