import copy
import json
from collections import Counter

import pytest
from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator

from src.models.g7_profile import PROFILE_PATH, calculate_g7_completeness, load_g7_profile
from src.models.scoring import calculate_completeness_score


def model(name="model-a", **fields):
    return {"type": "machine-learning-model", "name": name, "bom-ref": name, **fields}


def bom(*components, **fields):
    return {"bomFormat": "CycloneDX", "specVersion": "1.7", "version": 1,
            "components": list(components), **fields}


def report(document):
    # Positive fixtures must actually conform to the targeted schema.
    errors = JsonStrictValidator(SchemaVersion.V1_7).validate_str(json.dumps(document), all_errors=True)
    assert list(errors or []) == []
    return calculate_g7_completeness(document)


def element(result, element_id):
    return next(item for item in result["elements"] if item["id"] == "g7-" + element_id)


def status(result, element_id):
    return element(result, element_id)["status"]


def test_registry_is_traceable_to_original_document_and_has_50_elements():
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert len(profile["elements"]) == len({e["id"] for e in profile["elements"]}) == 50
    assert Counter(e["cluster"] for e in profile["elements"]) == {
        "metadata": 10, "system-level-properties": 9, "models": 13,
        "dataset-properties": 10, "infrastructure": 2,
        "security-properties": 4, "key-performance-indicators": 2,
    }
    assert profile["source"]["sha256"] == "d0b2f41c5baa2bcb2c8bea5b9d610fe9bafef202bbc42798b0a562eb0db5e77c"
    assert all(6 <= e["source_page"] <= 22 and e["source_section"].startswith("2.") for e in profile["elements"])
    assert next(e for e in profile["elements"] if e["id"] == "g7-model-license")["name"] == "Model license"


@pytest.mark.parametrize("document", [None, {}, {"bomFormat": "SPDX", "specVersion": "bogus"},
                                      {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1}])
def test_unsupported_inputs_have_no_percentage(document):
    result = calculate_g7_completeness(document)
    assert result["status"] == "unsupported"
    assert result["percentage"] is None
    assert result["not_assessed"] == 50


@pytest.mark.parametrize("bad_fields", [{"version": 0}, {"unexpected": "field"},
                                       {"components": [{"type": "machine-learning-model", "name": "m", "licenses": [{}]}]}])
def test_schema_invalid_boms_are_not_scored(bad_fields):
    result = calculate_g7_completeness(bom(model(), **bad_fields))
    assert result["status"] == "invalid"
    assert result["present"] == 0
    assert result["percentage"] is None


def test_arbitrary_extension_pairs_cannot_create_full_coverage():
    properties = [{"name": "genai:aibom:g7:" + e["id"], "value": "yes"} for e in load_g7_profile()["elements"]]
    invalid = calculate_g7_completeness({"anything": properties})
    assert invalid["percentage"] is None
    baseline = report(bom(model()))
    document = bom(model(properties=properties), properties=properties, metadata={"properties": properties})
    augmented = report(document)
    assert augmented["percentage"] == baseline["percentage"]
    assert status(augmented, "meta-signature") == "review"
    assert status(augmented, "security-controls") == "review"


def test_non_ai_library_cannot_earn_model_or_dataset_credit():
    library = {"type": "library", "name": "unrelated", "bom-ref": "lib", "version": "1",
               "description": "library", "hashes": [{"alg": "SHA-256", "content": "a" * 64}],
               "licenses": [{"license": {"id": "MIT"}}]}
    result = report(bom(library))
    assert result["status"] == "not_applicable"
    assert result["percentage"] is None
    result = report(bom(model(), library))
    assert status(result, "model-version") == "missing"
    assert status(result, "model-hash-value") == "missing"
    assert status(result, "dataset-license") == "review"


def test_all_models_are_checked_including_metadata_and_nested_components():
    document = bom({"type": "application", "name": "container", "components": [model("nested")]},
                   metadata={"component": model("subject", version="1")})
    result = report(document)
    version = element(result, "model-version")
    assert version["status"] == "partial"
    assert {s["path"] for s in version["subjects"]} == {"/metadata/component", "/components/0/components/0"}
    assert version["subjects"][0]["evidence"] == ["/metadata/component/version"]


def test_both_input_and_output_are_required_on_each_model():
    document = bom(model(modelCard={"modelParameters": {"inputs": [{"format": "text"}]}}))
    assert status(report(document), "model-input-output") == "partial"
    document["components"][0]["modelCard"]["modelParameters"]["outputs"] = [{"format": "text"}]
    assert status(report(document), "model-input-output") == "present"
    document["components"].append(model("undocumented"))
    assert status(report(document), "model-input-output") == "partial"


def test_empty_model_card_and_input_formats_do_not_establish_training_or_architecture():
    document = bom(model(modelCard={"modelParameters": {}}))
    result = report(document)
    assert status(result, "model-properties") == "missing"
    assert status(result, "model-training") == "missing"
    parameters = document["components"][0]["modelCard"]["modelParameters"]
    parameters["inputs"] = [{"format": "text"}]
    parameters["modelArchitecture"] = "transformer"
    assert status(report(document), "model-properties") == "present"
    assert status(report(document), "model-training") == "missing"
    parameters["approach"] = {"type": "self-supervised"}
    assert status(report(document), "model-training") == "partial"


@pytest.mark.parametrize("value", [" UNKNOWN ", "noassertion", " NOT_FOUND ", "N/A"])
def test_explicit_unknown_values_are_visible_without_full_credit(value):
    result = report(bom(model(version=value)))
    assert status(result, "model-version") == "unknown"


def test_blank_values_do_not_earn_credit():
    result = report(bom(model(version="   ", modelCard={"modelParameters": {"architectureFamily": " "}})))
    assert status(result, "model-version") == "missing"
    assert status(result, "model-properties") == "missing"


def test_dataset_arrays_and_governance_evidence_use_actual_json_pointers():
    dataset = {"type": "data", "name": "training", "bom-ref": "dataset", "data": [
        {"type": "dataset", "contents": {"url": "https://example.org/data"},
         "governance": {"owners": [{"organization": {"name": "Data owner"}}]}}
    ]}
    result = report(bom(dataset))
    assert status(result, "dataset-name") == "present"
    assert status(result, "dataset-content") == "partial"
    assert status(result, "dataset-provenance") == "partial"
    assert element(result, "dataset-content")["subjects"][0]["evidence"] == ["/components/0/data/0/contents/url"]
    assert element(result, "dataset-provenance")["subjects"][0]["evidence"] == ["/components/0/data/0/governance/owners/0/organization/name"]


def test_inline_datasets_and_resolved_and_unresolved_references():
    datasets = [{"ref": "dataset"}, {"ref": "absent"}, {"type": "dataset", "name": "inline"}]
    dataset = {"type": "data", "name": "resolved", "bom-ref": "dataset", "data": [{"type": "dataset"}]}
    result = report(bom(model(modelCard={"modelParameters": {"datasets": datasets}}), dataset))
    names = element(result, "dataset-name")
    assert names["status"] == "partial"
    assert len(names["subjects"]) == 3  # Resolved component is not double-counted.
    assert Counter(s["status"] for s in names["subjects"]) == {"present": 2, "review": 1}


def test_empty_dataset_reference_does_not_disappear_from_the_cohort():
    document = bom(model(modelCard={"modelParameters": {"datasets": [
        {"type": "dataset", "name": "named"}, {}]}}))
    names = element(report(document), "dataset-name")
    assert names["status"] == "partial"
    assert len(names["subjects"]) == 2


def test_unrelated_application_dependencies_do_not_earn_infrastructure_credit():
    document = bom(model(), {"type": "application", "name": "unrelated", "bom-ref": "app"},
                   {"type": "library", "name": "library", "bom-ref": "lib"},
                   dependencies=[{"ref": "app", "dependsOn": ["lib"]}])
    assert status(report(document), "infrastructure-software") == "missing"


def test_dataset_does_not_inherit_model_license_or_hashes():
    result = report(bom(model(licenses=[{"license": {"id": "MIT"}}],
                              hashes=[{"alg": "SHA-256", "content": "a" * 64}],
                              modelCard={"modelParameters": {"datasets": [{"type": "dataset", "name": "inline"}]}})))
    assert status(result, "model-license") == "present"
    assert status(result, "model-hash-value") == "present"
    assert status(result, "dataset-license") == "missing"
    assert status(result, "dataset-hash") == "missing"
    assert "does not infer openness" in element(result, "model-license")["notes"]


def test_relationships_must_resolve_and_match_the_element_scope():
    dataset = {"type": "data", "name": "data", "bom-ref": "data", "data": [{"type": "dataset"}]}
    library = {"type": "library", "name": "runtime", "bom-ref": "runtime"}
    document = bom(model(), dataset, library, dependencies=[{"ref": "model-a", "dependsOn": ["data", "missing"]}])
    result = report(document)
    assert status(result, "meta-dependency") == "partial"
    assert status(result, "infrastructure-software") == "missing"
    assert status(result, "dataset-dependency") == "missing"
    document["dependencies"].append({"ref": "data", "dependsOn": ["runtime"]})
    assert status(report(document), "dataset-dependency") == "partial"
    document["dependencies"][0]["dependsOn"].append("runtime")
    assert status(report(document), "infrastructure-software") == "partial"


def test_bare_dependency_ref_and_self_edge_are_not_relationship_evidence():
    document = bom(model(), dependencies=[{"ref": "model-a", "dependsOn": ["model-a"]}])
    assert status(report(document), "meta-dependency") == "missing"


def test_system_identity_requires_a_connected_application_subject():
    document = bom(model(), metadata={"timestamp": "2026-10-18T00:00:00Z", "component": {
        "type": "application", "name": "system", "bom-ref": "system", "version": "1"}})
    assert status(report(document), "slp-name") == "review"
    document["dependencies"] = [{"ref": "system", "dependsOn": ["model-a"]}]
    result = report(document)
    assert status(result, "slp-name") == "present"
    assert status(result, "slp-timestamp") == "missing"
    assert status(result, "model-timestamp") == "missing"


def test_supplier_is_not_producer_and_tool_is_not_bom_author():
    document = bom(model(supplier={"name": "Distributor"}), metadata={"manufacturer": {"name": "Producer"},
                   "tools": {"components": [{"type": "application", "name": "tool", "version": "1"}]}})
    result = report(document)
    assert status(result, "meta-author") == "missing"
    assert status(result, "model-producer") == "missing"
    assert status(result, "meta-tool-version") == "present"
    document["metadata"]["tools"]["components"].append({"type": "application", "name": "another"})
    assert status(report(document), "meta-tool-version") == "partial"


def test_lifecycle_is_native_generation_context():
    result = report(bom(model(), metadata={"lifecycles": [{"phase": "post-build"}]}))
    assert status(result, "meta-generation-context") == "present"


def test_signature_presence_still_requires_verification():
    result = report(bom(model(), signature={"algorithm": "ES256", "value": "YWJj"}))
    signature = element(result, "meta-signature")
    assert signature["status"] == "review"
    assert signature["subjects"][0]["evidence"] == ["/signature/algorithm", "/signature/value"]


def test_security_policy_requires_security_txt_candidate_not_contact_email():
    document = bom(model(externalReferences=[{"type": "security-contact", "url": "mailto:security@example.org"}]))
    assert status(report(document), "security-policy") == "missing"
    document["components"][0]["externalReferences"][0]["url"] = "https://example.org/.well-known/security.txt"
    assert status(report(document), "security-policy") == "partial"


def test_accuracy_does_not_satisfy_operational_or_security_metrics():
    result = report(bom(model(modelCard={"quantitativeAnalysis": {"performanceMetrics": [
        {"type": "accuracy", "value": "0.99"}]}})))
    assert status(result, "kpi-operational") == "review"
    assert status(result, "kpi-security") == "review"
    assert not element(result, "kpi-operational")["subjects"][0]["evidence"]


def test_duplicate_component_references_are_rejected():
    result = calculate_g7_completeness(bom(model(version="1"), model(version="2")))
    assert result["status"] == "invalid"
    assert result["percentage"] is None


def test_counts_reconcile_without_mutating_input():
    document = bom(model(version="unknown"), model("second", version="1"))
    original = copy.deepcopy(document)
    result = report(document)
    statuses = ("present", "partial", "missing", "review", "unknown", "not_assessed")
    assert sum(result[key] for key in statuses) == result["total"] == 50
    assert result["percentage"] == result["present"] * 2
    assert all(sum(cluster[key] for key in statuses) == cluster["total"] for cluster in result["clusters"].values())
    assert document == original


def test_existing_score_integration_preserves_legacy_score_and_reports_unsupported_1_6():
    document = bom(model())
    result = calculate_completeness_score(document, validate=False)
    assert result["g7_profile"]["status"] == "assessed"
    assert result["g7_profile"]["total"] == 50
    document["specVersion"] = "1.6"
    legacy = calculate_completeness_score(document, validate=False)
    assert legacy["g7_profile"]["status"] == "unsupported"
    assert legacy["g7_profile"]["percentage"] is None
    assert legacy["total_score"] == result["total_score"]
