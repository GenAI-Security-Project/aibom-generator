"""
Candidate OWASP GenAI risks and mapped framework controls for an AIBOM.

Data: the OWASP GenAI Crosswalk AIBOM index and applicability profiles,
vendored in src/data/owasp_crosswalk/ at a pinned commit (see SOURCE.json and
scripts/update_crosswalk.py). Nothing is fetched at run time.

The one rule: these are *candidate* risks for the kind of model, chosen by
whole OWASP list from the model's task and what the user says about the
deployment. Nothing here says the model has a risk or meets a control, so the
output is labelled "Candidate risk — not assessed" and no field that asserts
something about the model is filled from it. Every control link carries the
crosswalk's own review status (reviewed / draft / unreviewed).

Output, all optional and off by default:
  - components[0].modelCard.considerations.ethicalConsiderations[]: one
    {name, mitigationStrategy} per candidate risk
  - definitions.standards[]: one standard per selected OWASP list and per
    selected framework (control text only where the framework's licence
    allows redistribution)
  - metadata.properties: owasp:aibom:risk:* (source, status, profiles, ...)
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "owasp_crosswalk"

# Major version of the crosswalk files' format_version this module reads.
SUPPORTED_FORMAT_MAJOR = 1

PROP_PREFIX = "owasp:aibom:risk:"
STATUS_VALUE = "candidate-not-assessed"
CANDIDATE_LABEL = "Candidate risk — not assessed."
# Controls named per framework inside one mitigationStrategy string; the full
# set is always in definitions.standards.
MAX_CONTROLS_PER_FRAMEWORK = 5
OWASP_OWNER = "OWASP GenAI Security Project"

# Flag name → applicability profile id. --rag and --training-data both select
# the data list (GenAI Data Security).
FLAG_PROFILES = {
    "llm": "llm",
    "agentic": "agentic",
    "rag": "data",
    "training_data": "data",
    "skills": "skills",
}


class OwaspRiskError(ValueError):
    """Raised for an unknown profile or framework."""


@dataclass
class RiskOptions:
    """What the user asked for. Profiles are applicability profile ids."""
    enabled: bool = False
    profiles: List[str] = field(default_factory=list)
    frameworks: Optional[List[str]] = None  # ids or names; None = defaults

    @classmethod
    def from_flags(cls, owasp_risks: bool = False, llm: bool = False, agentic: bool = False,
                   rag: bool = False, training_data: bool = False, skills: bool = False,
                   frameworks: Optional[str] = None) -> "RiskOptions":
        flags = {"llm": llm, "agentic": agentic, "rag": rag,
                 "training_data": training_data, "skills": skills}
        profiles = sorted({FLAG_PROFILES[k] for k, v in flags.items() if v})
        fw = [f.strip() for f in frameworks.split(",") if f.strip()] if frameworks else None
        # Any profile or framework choice implies the section is wanted.
        return cls(enabled=bool(owasp_risks or profiles or fw), profiles=profiles, frameworks=fw)


def check_format_version(version: Optional[str], what: str) -> None:
    """Refuse a crosswalk file whose format this code was not written for.

    The crosswalk versions both files with semver: a minor version only adds
    fields (safe to read), a major version changes or removes them.
    """
    major = str(version or "").split(".")[0]
    if major != str(SUPPORTED_FORMAT_MAJOR):
        raise OwaspRiskError(
            f"{what} has format_version {version!r}; this code reads "
            f"{SUPPORTED_FORMAT_MAJOR}.x. Update src/models/owasp_risks.py "
            "before vendoring a new major version."
        )


@lru_cache(maxsize=1)
def load_index() -> Dict[str, Any]:
    data = json.loads((DATA_DIR / "aibom-index.json").read_text(encoding="utf-8"))
    check_format_version(data.get("meta", {}).get("format_version"), "aibom-index.json")
    return data


@lru_cache(maxsize=1)
def load_applicability() -> Dict[str, Any]:
    data = json.loads((DATA_DIR / "aibom-applicability.json").read_text(encoding="utf-8"))
    check_format_version(data.get("format_version"), "aibom-applicability.json")
    return data


@lru_cache(maxsize=1)
def load_source() -> Dict[str, Any]:
    return json.loads((DATA_DIR / "SOURCE.json").read_text(encoding="utf-8"))


def select_profiles(task: Optional[str], opted: Iterable[str] = ()) -> List[str]:
    """Profiles that apply: those whose default_for_tasks names the task, plus opted ones."""
    app = load_applicability()
    known = {p["id"] for p in app["profiles"]}
    unknown = sorted(set(opted) - known)
    if unknown:
        raise OwaspRiskError(f"unknown risk profile(s): {', '.join(unknown)}")
    chosen = set(opted)
    if task:
        chosen |= {p["id"] for p in app["profiles"] if task in p["default_for_tasks"]}
    # Keep the applicability file's order.
    return [p["id"] for p in app["profiles"] if p["id"] in chosen]


def resolve_frameworks(requested: Optional[Sequence[str]]) -> List[str]:
    """Framework ids for the request (ids or names, case-insensitive); defaults when None."""
    index = load_index()
    if not requested:
        return list(load_applicability()["default_frameworks"]["framework_ids"])
    by_key = {}
    for f in index["frameworks"]:
        by_key[f["id"].lower()] = f["id"]
        by_key[f["name"].lower()] = f["id"]
    out, unknown = [], []
    for r in requested:
        fid = by_key.get(r.strip().lower())
        if fid is None:
            unknown.append(r)
        elif fid not in out:
            out.append(fid)
    if unknown:
        raise OwaspRiskError(f"unknown framework(s): {', '.join(unknown)}")
    return out


def select_risks(profiles: Sequence[str]) -> List[Dict[str, Any]]:
    """Entries of the source lists the profiles add: lists in profile order, entries in index order."""
    app = load_applicability()
    lists: List[str] = []
    for p in app["profiles"]:
        if p["id"] in profiles:
            lists.extend(s for s in p["source_lists"] if s not in lists)
    entries = load_index()["entries"]
    return [e for sid in lists for e in entries if e["source_list"] == sid]


def select_links(risks: Sequence[Dict[str, Any]], framework_ids: Sequence[str]) -> List[Dict[str, Any]]:
    ids = {r["id"] for r in risks}
    fws = set(framework_ids)
    return [l for l in load_index()["links"] if l["entry"] in ids and l["framework_id"] in fws]


def check_options(options: Optional[RiskOptions]) -> None:
    """Raise OwaspRiskError for an unknown profile or framework; no-op when off."""
    if options is None or not options.enabled:
        return
    select_profiles(None, options.profiles)
    resolve_frameworks(options.frameworks)


def build_risk_section(task: Optional[str], options: RiskOptions) -> Optional[Dict[str, Any]]:
    """Everything the BOM and the UI need, or None when off or nothing applies."""
    if not options.enabled:
        return None
    profiles = select_profiles(task, options.profiles)
    framework_ids = resolve_frameworks(options.frameworks)
    risks = select_risks(profiles)
    if not risks:
        return {"profiles": profiles, "framework_ids": framework_ids, "risks": [], "links": []}
    return {
        "profiles": profiles,
        "framework_ids": framework_ids,
        "risks": risks,
        "links": select_links(risks, framework_ids),
    }


# --- CycloneDX builders ------------------------------------------------------

def _maps():
    index = load_index()
    return (
        {s["id"]: s for s in index["source_lists"]},
        {f["id"]: f for f in index["frameworks"]},
        {(c["framework_id"], c["control_id"]): c for c in index["controls"]},
    )


def _entry_ref(entry_id: str) -> str:
    return f"owasp-crosswalk:entry:{entry_id}"


def _control_ref(framework_id: str, control_id: str) -> str:
    return f"owasp-crosswalk:control:{framework_id}:{control_id}"


def _distinct(links: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One link per (control, review status), in order. The crosswalk can map the
    same risk to the same control on more than one row (different evidence);
    a reader needs the control once per status, not once per row."""
    seen, out = set(), []
    for link in links:
        key = (link["control_id"], link["review_status"])
        if key not in seen:
            seen.add(key)
            out.append(link)
    return out


def ethical_considerations(section: Dict[str, Any]) -> List[Dict[str, str]]:
    lists, frameworks, _ = _maps()
    out = []
    for risk in section["risks"]:
        by_fw: Dict[str, List[Dict[str, Any]]] = {}
        for link in section["links"]:
            if link["entry"] == risk["id"]:
                by_fw.setdefault(link["framework_id"], []).append(link)
        parts = []
        for fid in section["framework_ids"]:
            links = _distinct(by_fw.get(fid, []))
            if not links:
                continue
            shown = ", ".join(f"{l['control_id']} ({l['review_status']})"
                              for l in links[:MAX_CONTROLS_PER_FRAMEWORK])
            more = len(links) - MAX_CONTROLS_PER_FRAMEWORK
            if more > 0:
                shown += f", and {more} more"
            parts.append(f"{frameworks[fid]['name']}: {shown}")
        mapped = ("Mapped controls — " + "; ".join(parts) + ". Full list in definitions.standards."
                  if parts else "No controls mapped in the selected frameworks.")
        out.append({
            "name": f"{risk['id']} {risk['name']} ({lists[risk['source_list']]['name']})",
            "mitigationStrategy": f"{CANDIDATE_LABEL} {mapped} Source: OWASP GenAI Crosswalk.",
        })
    return out


def _crosswalk_ref() -> Dict[str, str]:
    src = load_source()
    return {
        "type": "website",
        "url": f"{src['repository']}/tree/{src['commit']}",
        "comment": "OWASP GenAI Crosswalk at the pinned commit (CC BY-SA 4.0)",
    }


def standards(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    lists, frameworks, controls = _maps()
    out = []
    selected_lists = []
    for r in section["risks"]:
        if r["source_list"] not in selected_lists:
            selected_lists.append(r["source_list"])
    for sid in selected_lists:
        s = lists[sid]
        out.append({
            "bom-ref": f"owasp-crosswalk:standard:{sid}",
            "name": s["name"],
            "owner": OWASP_OWNER,
            "description": f"{s['name']}. Listed as candidate risks for this model — not assessed.",
            "requirements": [
                {"bom-ref": _entry_ref(r["id"]), "identifier": r["id"], "title": r["name"]}
                for r in section["risks"] if r["source_list"] == sid
            ],
            "externalReferences": [{"type": "documentation", "url": s["url"]}, _crosswalk_ref()],
        })
    for fid in section["framework_ids"]:
        fw = frameworks[fid]
        reqs: Dict[str, Dict[str, Any]] = {}
        for link in section["links"]:
            if link["framework_id"] != fid:
                continue
            cid = link["control_id"]
            if cid not in reqs:
                ctl = controls.get((fid, cid), {})
                req: Dict[str, Any] = {"bom-ref": _control_ref(fid, cid), "identifier": cid}
                if fw["redistributable"]:
                    if ctl.get("title"):
                        req["title"] = ctl["title"]
                    if ctl.get("description"):
                        req["text"] = ctl["description"]
                if ctl.get("url"):
                    req["externalReferences"] = [{"type": "documentation", "url": ctl["url"]}]
                req["properties"] = []
                reqs[cid] = req
            rel = link.get("relationship", "")
            reqs[cid]["properties"].append({
                "name": f"{PROP_PREFIX}link",
                "value": f"{link['entry']}|{rel}|{link['review_status']}",
            })
        if not reqs:
            continue
        std: Dict[str, Any] = {
            "bom-ref": f"owasp-crosswalk:standard:{fid}",
            "name": fw["name"],
            "version": fw["version"],
            "description": (f"{fw['name']} controls mapped to the candidate risks by the OWASP GenAI "
                            "Crosswalk. Each owasp:aibom:risk:link property is "
                            "<risk>|<relationship>|<review status>; a mapping is not a claim the "
                            "model meets the control."),
            "requirements": list(reqs.values()),
            "externalReferences": [_crosswalk_ref()],
        }
        if not fw["redistributable"]:
            std["description"] += f" Control text omitted: {fw['license']} licence."
        if fw.get("url"):
            std["externalReferences"].insert(0, {"type": "website", "url": fw["url"]})
        out.append(std)
    return out


def metadata_properties(section: Dict[str, Any]) -> List[Dict[str, str]]:
    src = load_source()
    index = load_index()
    app = load_applicability()
    counts: Dict[str, int] = {}
    for l in section["links"]:
        counts[l["review_status"]] = counts.get(l["review_status"], 0) + 1
    return [
        {"name": f"{PROP_PREFIX}status", "value": STATUS_VALUE},
        {"name": f"{PROP_PREFIX}source",
         "value": f"OWASP GenAI Crosswalk {index['meta']['crosswalk_version']} @ {src['commit']}"},
        {"name": f"{PROP_PREFIX}profiles", "value": ",".join(section["profiles"])},
        {"name": f"{PROP_PREFIX}frameworks", "value": ",".join(section["framework_ids"])},
        {"name": f"{PROP_PREFIX}applicabilityStatus", "value": app["status"]},
        {"name": f"{PROP_PREFIX}linkReviewStatus",
         "value": ",".join(f"{k}={counts.get(k, 0)}" for k in ("reviewed", "draft", "unreviewed"))},
        {"name": f"{PROP_PREFIX}attribution", "value": index["meta"]["attribution"]},
    ]


def apply_to_aibom(aibom: Dict[str, Any], section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Add the risk section to a serialized AIBOM dict in place; no-op for None or no risks."""
    if not section or not section["risks"]:
        return aibom
    components = aibom.get("components") or []
    if components:
        card = components[0].setdefault("modelCard", {})
        considerations = card.setdefault("considerations", {})
        considerations.setdefault("ethicalConsiderations", []).extend(ethical_considerations(section))
    stds = standards(section)
    if stds:
        aibom.setdefault("definitions", {}).setdefault("standards", []).extend(stds)
    aibom.setdefault("metadata", {}).setdefault("properties", []).extend(metadata_properties(section))
    return aibom


def summary_for_ui(section: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A template-friendly view: risks with their controls grouped by framework."""
    if not section:
        return None
    lists, frameworks, _ = _maps()
    risks = []
    for r in section["risks"]:
        groups = []
        for fid in section["framework_ids"]:
            ls = _distinct([l for l in section["links"] if l["entry"] == r["id"] and l["framework_id"] == fid])
            if ls:
                groups.append({"framework": frameworks[fid]["name"],
                               "controls": [{"id": l["control_id"], "status": l["review_status"]} for l in ls]})
        risks.append({"id": r["id"], "name": r["name"], "list": lists[r["source_list"]]["name"],
                      "groups": groups})
    return {
        "profiles": section["profiles"],
        "frameworks": [frameworks[f]["name"] for f in section["framework_ids"]],
        "risks": risks,
        "applicability_status": load_applicability()["status"],
        "source": load_source()["commit"],
    }
