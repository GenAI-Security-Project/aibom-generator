import logging
import unittest

from src.models.scoring import (
    calculate_completeness_score,
    check_field_in_aibom,
    check_field_with_enhanced_results,
    determine_completeness_profile,
)
from src.models.service import AIBOMService


def _build_aibom(extra=None):
    service = AIBOMService.__new__(AIBOMService)
    service.model_file_extractors = []
    metadata = {
        "commit": "a" * 40,
        "author": "acme",
        "license": "apache-2.0",
        "description": "d",
        "primaryPurpose": "text-generation",
        "suppliedBy": "acme",
        "typeOfModel": "llama",
        "downloadLocation": "https://huggingface.co/acme/foo/tree/main",
    }
    metadata.update(extra or {})
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return service._create_aibom_structure("acme/foo", metadata)
    finally:
        logging.disable(previous)


class TestCompletenessProfile(unittest.TestCase):
    def test_emitted_values_are_detected(self):
        aibom = _build_aibom()
        component = aibom["components"][0]
        self.assertEqual(component["modelCard"]["modelParameters"]["task"], "text-generation")
        self.assertEqual(component["supplier"]["name"], "acme")

        self.assertTrue(check_field_in_aibom(aibom, "primaryPurpose"))
        self.assertTrue(check_field_in_aibom(aibom, "suppliedBy"))
        self.assertTrue(check_field_with_enhanced_results(aibom, "primaryPurpose"))
        self.assertTrue(check_field_with_enhanced_results(aibom, "suppliedBy"))

    def test_missing_values_are_not_detected(self):
        aibom = _build_aibom()
        component = aibom["components"][0]
        del component["modelCard"]["modelParameters"]["task"]
        del component["supplier"]
        self.assertFalse(check_field_in_aibom(aibom, "primaryPurpose"))
        self.assertFalse(check_field_in_aibom(aibom, "suppliedBy"))
        self.assertFalse(check_field_with_enhanced_results(aibom, "primaryPurpose"))
        self.assertFalse(check_field_with_enhanced_results(aibom, "suppliedBy"))

    def test_standard_profile_reachable(self):
        aibom = _build_aibom()
        profile = determine_completeness_profile(aibom, 100.0)
        self.assertEqual(profile["name"], "Standard")
        self.assertTrue(profile["satisfied"])

    def test_standard_profile_requires_minimum_score(self):
        aibom = _build_aibom()
        self.assertEqual(determine_completeness_profile(aibom, 50.0)["name"], "Basic")

    def test_standard_profile_requires_primary_purpose_and_supplier(self):
        aibom = _build_aibom()
        del aibom["components"][0]["supplier"]
        self.assertEqual(determine_completeness_profile(aibom, 100.0)["name"], "Basic")

    def test_advanced_profile_reachable(self):
        aibom = _build_aibom({
            "technicalLimitations": "may hallucinate",
            "ethicalConsiderations": "bias risk",
            "intendedUse": "chat",
        })
        component = aibom["components"][0]
        properties = component["modelCard"].setdefault("properties", [])
        metadata_properties = aibom["metadata"].setdefault("properties", [])
        for name in ("energyConsumption", "safetyRiskAssessment"):
            metadata_properties.append({"name": name, "value": "x"})
        properties.append({"name": "genai:aibom:modelcard:hyperparameter", "value": "{}"})
        profile = determine_completeness_profile(aibom, 100.0)
        self.assertEqual(profile["name"], "Advanced")

    def test_calculate_completeness_score_uses_registry_paths(self):
        aibom = _build_aibom()
        result = calculate_completeness_score(aibom, validate=False)
        self.assertNotIn("primaryPurpose", result["missing_fields"]["critical"])
        self.assertNotIn("suppliedBy", result["missing_fields"]["critical"])
        self.assertTrue(result["field_checklist"]["primaryPurpose"].startswith("✔"))
        self.assertTrue(result["field_checklist"]["suppliedBy"].startswith("✔"))


if __name__ == "__main__":
    unittest.main()
