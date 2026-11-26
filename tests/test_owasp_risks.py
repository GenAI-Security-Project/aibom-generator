"""Optional OWASP GenAI risk section (src/models/owasp_risks.py).

Guards: the vendored crosswalk data matches its pin; risk lists are chosen
whole, from the task or an explicit flag; licence-restricted control text is
never emitted; every item says "candidate, not assessed" and carries the
crosswalk's review status; output stays CycloneDX 1.6 and 1.7 valid; and with
the option off, the BOM is exactly what it was before.
"""
import hashlib
import json

import pytest
from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator

from src.models import owasp_risks as risks
from src.models.service import AIBOMService
from src.utils.formatter import export_aibom

REVIEW_STATUSES = {"reviewed", "draft", "unreviewed"}


def _section(task="text-generation", **flags):
    opts = risks.RiskOptions.from_flags(owasp_risks=True, **flags)
    return risks.build_risk_section(task, opts)


def _props(aibom):
    return [p for p in aibom.get("metadata", {}).get("properties", [])
            if p["name"].startswith(risks.PROP_PREFIX)]


# --- data and selection ------------------------------------------------------

def test_vendored_files_match_source_pin():
    src = risks.load_source()
    assert len(src["commit"]) == 40
    for name, meta in src["files"].items():
        digest = hashlib.sha256((risks.DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == meta["sha256"], name


def test_generative_task_gets_llm_list_by_default():
    assert risks.select_profiles("text-generation") == ["llm"]
    assert risks.select_profiles("image-classification") == []
    assert risks.select_profiles("image-classification", ["agentic"]) == ["agentic"]


def test_flags_map_to_profiles_and_imply_enabled():
    o = risks.RiskOptions.from_flags(rag=True, training_data=True)
    assert o.enabled and o.profiles == ["data"]
    assert risks.RiskOptions.from_flags(frameworks="NIST AI RMF 1.0").enabled
    assert not risks.RiskOptions.from_flags().enabled


def test_unknown_profile_or_framework_is_an_error():
    with pytest.raises(risks.OwaspRiskError):
        risks.select_profiles(None, ["nope"])
    with pytest.raises(risks.OwaspRiskError):
        risks.resolve_frameworks(["Not A Framework"])
    assert risks.resolve_frameworks(["nist ai rmf 1.0", "owasp-aisvs"]) == ["nist-ai-rmf", "owasp-aisvs"]


def test_lists_are_added_whole():
    s = _section()
    assert [r["id"] for r in s["risks"]] == [f"LLM{n:02d}" for n in range(1, 11)]
    s = _section(agentic=True)
    assert {r["source_list"] for r in s["risks"]} == {"LLM-Top10-2026", "Agentic-Top10-2026"}
    assert len(s["risks"]) == 20


def test_default_frameworks_and_links_stay_in_selection():
    s = _section()
    assert s["framework_ids"] == risks.load_applicability()["default_frameworks"]["framework_ids"]
    ids = {r["id"] for r in s["risks"]}
    assert s["links"]
    for link in s["links"]:
        assert link["entry"] in ids and link["framework_id"] in s["framework_ids"]
        assert link["review_status"] in REVIEW_STATUSES


# --- builders ----------------------------------------------------------------

def test_ethical_considerations_are_labelled_candidates():
    items = risks.ethical_considerations(_section())
    assert len(items) == 10
    for item in items:
        assert set(item) == {"name", "mitigationStrategy"}
        assert item["mitigationStrategy"].startswith(risks.CANDIDATE_LABEL)
    assert items[0]["name"] == "LLM01 Prompt Injection (OWASP Top 10 for LLM Applications 2026)"


def test_restricted_licences_ship_no_control_text():
    restricted = [f["id"] for f in risks.load_index()["frameworks"] if not f["redistributable"]]
    assert restricted
    s = _section(frameworks=",".join(restricted))
    stds = risks.standards(s)
    fw_stds = [x for x in stds if x["bom-ref"].split(":")[-1] in restricted]
    assert fw_stds
    for std in fw_stds:
        assert "omitted" in std["description"]
        for req in std["requirements"]:
            assert "title" not in req and "text" not in req


def test_open_licence_control_text_is_carried():
    s = _section(frameworks="owasp-aisvs")
    reqs = [r for std in risks.standards(s) if std["bom-ref"].endswith("owasp-aisvs") for r in std["requirements"]]
    assert reqs and any("text" in r for r in reqs)
    for r in reqs:
        for p in r["properties"]:
            entry, rel, status = p["value"].split("|")
            assert entry.startswith("LLM") and status in REVIEW_STATUSES


def test_reviewed_status_is_never_invented():
    index = risks.load_index()
    # The same entry/control pair can appear on more than one mapping row, each
    # with its own status, so compare rows, not keys: every selected link is an
    # index row, unchanged.
    rows = {json.dumps(l, sort_keys=True) for l in index["links"]}
    s = _section(frameworks=",".join(f["id"] for f in index["frameworks"]), agentic=True, rag=True, skills=True)
    assert s["links"]
    for l in s["links"]:
        assert json.dumps(l, sort_keys=True) in rows
    for item in risks.ethical_considerations(s):
        assert "(reviewed)" not in item["mitigationStrategy"] or any(
            l["review_status"] == "reviewed" for l in s["links"] if item["name"].startswith(l["entry"] + " "))


def test_bom_refs_are_unique():
    s = _section(agentic=True, frameworks=",".join(f["id"] for f in risks.load_index()["frameworks"]))
    refs = []
    for std in risks.standards(s):
        refs.append(std["bom-ref"])
        refs.extend(r["bom-ref"] for r in std["requirements"])
    assert len(refs) == len(set(refs))


def test_no_applicable_list_adds_nothing():
    aibom = {"metadata": {}, "components": [{"modelCard": {}}]}
    s = risks.build_risk_section("image-classification", risks.RiskOptions(enabled=True))
    assert s["risks"] == []
    assert risks.apply_to_aibom(aibom, s) == {"metadata": {}, "components": [{"modelCard": {}}]}


# --- through the service -----------------------------------------------------

@pytest.fixture
def service(patch_service_io):
    return AIBOMService(use_inference=False)


def _strict_valid(aibom, version):
    doc = export_aibom(aibom, bom_type="cyclonedx", spec_version=version)
    schema = {"1.6": SchemaVersion.V1_6, "1.7": SchemaVersion.V1_7}[version]
    return JsonStrictValidator(schema).validate_str(doc, all_errors=True)


@pytest.mark.parametrize("version", ["1.6", "1.7"])
def test_generated_bom_with_risks_is_schema_valid(service, version):
    opts = risks.RiskOptions.from_flags(owasp_risks=True, agentic=True)
    aibom = service.generate_aibom("test-owner/test-model", risk_options=opts)
    assert "definitions" in aibom
    errors = _strict_valid(aibom, version)
    assert errors is None, [str(e) for e in errors][:5]
    assert service.get_enhancement_report()["schema_validation"]["valid"]


def test_generated_bom_places_each_part(service):
    opts = risks.RiskOptions.from_flags(owasp_risks=True)
    aibom = service.generate_aibom("test-owner/test-model", risk_options=opts)
    ec = aibom["components"][0]["modelCard"]["considerations"]["ethicalConsiderations"]
    assert sum(1 for e in ec if e.get("mitigationStrategy", "").startswith(risks.CANDIDATE_LABEL)) == 10
    props = {p["name"]: p["value"] for p in _props(aibom)}
    assert props[f"{risks.PROP_PREFIX}status"] == risks.STATUS_VALUE
    assert risks.load_source()["commit"] in props[f"{risks.PROP_PREFIX}source"]
    assert props[f"{risks.PROP_PREFIX}applicabilityStatus"].startswith("DRAFT")
    for p in aibom["metadata"]["properties"]:
        assert p["name"].startswith(("genai:aibom:", "owasp:aibom:")), p["name"]


def test_off_by_default_changes_nothing(service):
    plain = service.generate_aibom("test-owner/test-model")
    off = service.generate_aibom("test-owner/test-model", risk_options=risks.RiskOptions())
    assert service.risk_section is None
    for doc in (plain, off):
        assert "definitions" not in doc
        assert not _props(doc)
    # Serial numbers and dependency refs carry run-time values; compare the
    # parts the risk section would touch.
    assert set(plain) == set(off)
    assert plain["components"][0]["modelCard"] == off["components"][0]["modelCard"]
    assert plain["metadata"].get("properties") == off["metadata"].get("properties")


def test_completeness_score_ignores_the_risk_section(service):
    service.generate_aibom("test-owner/test-model")
    before = service.get_enhancement_report()["final_score"]["total_score"]
    service.generate_aibom("test-owner/test-model", risk_options=risks.RiskOptions.from_flags(owasp_risks=True))
    assert service.get_enhancement_report()["final_score"]["total_score"] == before


def test_bad_framework_is_not_swallowed(service):
    with pytest.raises(risks.OwaspRiskError):
        service.generate_aibom("test-owner/test-model",
                               risk_options=risks.RiskOptions.from_flags(frameworks="nope"))


# --- web and CLI -------------------------------------------------------------

@pytest.fixture
def web_client(patch_service_io, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.controllers import web_controller

    class _FakeHfApi:
        def model_info(self, model_id):
            return {"modelId": model_id}

    monkeypatch.setattr(web_controller, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(web_controller, "HfApi", _FakeHfApi)
    monkeypatch.setattr(web_controller, "log_sbom_generation", lambda model_id: None)
    monkeypatch.setattr(web_controller, "get_sbom_count", lambda: "0")
    app = FastAPI()
    app.include_router(web_controller.router)
    return TestClient(app)


def test_home_page_offers_the_option_off_with_default_frameworks(web_client):
    page = web_client.get("/").text
    assert 'name="owasp_risks"' in page
    assert 'name="owasp_risks" value="true" form="sbom-form" checked' not in page
    assert 'value="owasp-aisvs" form="sbom-form" checked' in page
    assert 'value="iso-27001" form="sbom-form" >' in page


def test_web_generate_with_risks_shows_the_risks_tab(web_client):
    resp = web_client.post("/generate", data={
        "model_id": "test-owner/test-model", "owasp_risks": "true", "risk_agentic": "true",
        "risk_frameworks": ["owasp-aisvs", "mitre-atlas"],
    })
    assert resp.status_code == 200
    assert resp.template.name == "result.html"
    assert "switchTab('risks-view')" in resp.text
    assert "Candidate OWASP GenAI Risks" in resp.text and "ASI01" in resp.text
    assert "MITRE ATLAS" in resp.text


def test_web_generate_without_risks_has_no_tab(web_client):
    resp = web_client.post("/generate", data={"model_id": "test-owner/test-model"})
    assert resp.template.name == "result.html"
    assert "risks-view" not in resp.text


def test_web_unknown_framework_is_an_error_page(web_client):
    resp = web_client.post("/generate", data={
        "model_id": "test-owner/test-model", "owasp_risks": "true", "risk_frameworks": ["nope"]})
    assert resp.template.name == "error.html"
    assert "unknown framework" in resp.text


def test_cli_flags_reach_the_controller(monkeypatch):
    import sys
    from src import cli
    seen = {}

    class _Ctl:
        def generate(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(cli, "CLIController", _Ctl)
    monkeypatch.setattr(sys, "argv", ["aibom", "owner/model", "--rag", "--frameworks", "NIST AI RMF 1.0,owasp-aisvs"])
    cli.main()
    opts = seen["risk_options"]
    assert opts.enabled and opts.profiles == ["data"]
    assert opts.frameworks == ["NIST AI RMF 1.0", "owasp-aisvs"]


def test_cli_unknown_framework_exits_with_usage_error(monkeypatch):
    import sys
    from src import cli
    monkeypatch.setattr(sys, "argv", ["aibom", "owner/model", "--frameworks", "nope"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
