import unittest
from src.models.registry import get_field_registry_manager
from src.models.service import AIBOMService
from src.models.scoring import (
    calculate_completeness_score,
    check_field_in_aibom,
    check_field_with_enhanced_results,
    ValidationSeverity,
)

class TestScoring(unittest.TestCase):
    def test_basic_completeness(self):
        aibom = {
            "bomFormat": "CycloneDX",
            "metadata": {
                "component": {
                    "name": "test-model"
                },
                "properties": []
            },
            "components": []
        }
        score = calculate_completeness_score(aibom, validate=False)
        self.assertIn("total_score", score)
        self.assertGreaterEqual(score["total_score"], 0)
        self.assertLessEqual(score["total_score"], 100)
    
    def test_completeness_with_fields(self):
        # A somewhat populated AIBOM
        aibom = {
            "metadata": {
                "properties": [
                    {"name": "primaryPurpose", "value": "text-generation"},
                    {"name": "suppliedBy", "value": "test"}
                ]
            }
        }
        score = calculate_completeness_score(aibom, validate=False)
        # Should have some score
        self.assertGreater(score["total_score"], 0)
        
    def test_registry_fallback(self):
        # Ensure it doesn't crash if registry logic is used
        aibom = {}
        score = calculate_completeness_score(aibom)
        self.assertIsNotNone(score)

    def test_external_references_detected_under_components_array(self):
        # Regression for #76: externalReferences lives under components[0]
        # (plural, camelCase) in CycloneDX 1.6/1.7, so a populated field must be
        # detected as present rather than scored as missing.
        aibom = {
            "bomFormat": "CycloneDX",
            "components": [
                {
                    "name": "test-model",
                    "type": "machine-learning-model",
                    "externalReferences": [
                        {"type": "website", "url": "https://example.com"},
                        {"type": "vcs", "url": "https://github.com/example/model"},
                    ],
                }
            ],
        }
        # registry jsonpath detection ($.components[0].externalReferences)
        self.assertTrue(check_field_with_enhanced_results(aibom, "external_references"))
        # fallback presence check resolves the snake_case -> camelCase alias
        self.assertTrue(check_field_in_aibom(aibom, "external_references"))

    def test_external_references_absent_is_not_reported_present(self):
        # Negative guard: with no externalReferences, detection stays False so
        # the fix cannot regress into always-present.
        aibom = {
            "bomFormat": "CycloneDX",
            "components": [{"name": "m", "type": "machine-learning-model"}],
        }
        self.assertFalse(check_field_with_enhanced_results(aibom, "external_references"))
        self.assertFalse(check_field_in_aibom(aibom, "external_references"))

def _component_basic_aibom(with_external_references: bool) -> dict:
    component = {
        "type": "machine-learning-model",
        "name": "test-model",
        "version": "1.0",
        "purl": "pkg:huggingface/acme/test-model@1.0",
        "description": "A test model",
        "licenses": [{"license": {"id": "Apache-2.0"}}],
    }
    if with_external_references:
        component["externalReferences"] = [
            {"type": "website", "url": "https://huggingface.co/acme/test-model"},
            {"type": "vcs", "url": "https://github.com/acme/test-model"},
            {"type": "documentation", "url": "https://arxiv.org/abs/0000.00000"},
        ]
    return {"bomFormat": "CycloneDX", "components": [component]}


class TestExternalReferencesScoring(unittest.TestCase):
    """Regression tests for #76 (component_basic/external_references scored as missing)."""

    def test_registry_jsonpath_uses_components_array(self):
        manager = get_field_registry_manager()
        fields = manager.get_field_definitions()

        self.assertEqual(
            fields["external_references"]["jsonpath"], "$.components[0].externalReferences"
        )
        # Every externalReferences-based path must use the plural `components[0]` root.
        for name, definition in fields.items():
            for path in (
                definition.get("jsonpath", ""),
                definition.get("aibom_generation", {}).get("location", ""),
            ):
                if "externalReferences" in path:
                    self.assertTrue(
                        path.startswith("$.components[0].externalReferences"),
                        f"{name}: stale externalReferences path {path}",
                    )

    def test_registry_detects_populated_external_references(self):
        manager = get_field_registry_manager()
        aibom = _component_basic_aibom(with_external_references=True)

        present, value = manager.detect_field_presence(
            aibom, manager.get_field_jsonpath("external_references")
        )

        self.assertTrue(present)
        self.assertEqual(len(value), 3)

    def test_paper_documentation_reference_detected_under_components_array(self):
        manager = get_field_registry_manager()
        present, _ = manager.detect_field_presence(
            _component_basic_aibom(with_external_references=True),
            manager.get_field_jsonpath("paper"),
        )
        self.assertTrue(present)

        present, _ = manager.detect_field_presence(
            _component_basic_aibom(with_external_references=False),
            manager.get_field_jsonpath("paper"),
        )
        self.assertFalse(present)

    def test_populated_external_references_awards_full_component_basic_points(self):
        result = calculate_completeness_score(
            _component_basic_aibom(with_external_references=True), validate=False
        )

        self.assertTrue(result["field_checklist"]["external_references"].startswith("✔"))
        self.assertNotIn("external_references", result["missing_fields"]["supplementary"])
        details = result["category_details"]["component_basic"]
        self.assertEqual(details["present_fields"], details["total_fields"])
        self.assertEqual(
            result["section_scores"]["component_basic"], result["max_scores"]["component_basic"]
        )

    def test_missing_external_references_loses_only_its_share(self):
        with_refs = calculate_completeness_score(
            _component_basic_aibom(with_external_references=True), validate=False
        )
        without_refs = calculate_completeness_score(
            _component_basic_aibom(with_external_references=False), validate=False
        )

        self.assertTrue(without_refs["field_checklist"]["external_references"].startswith("✘"))
        self.assertIn("external_references", without_refs["missing_fields"]["supplementary"])
        details = without_refs["category_details"]["component_basic"]
        self.assertEqual(details["present_fields"], details["total_fields"] - 1)
        self.assertLess(
            without_refs["section_scores"]["component_basic"],
            with_refs["section_scores"]["component_basic"],
        )

    def test_generated_component_scores_external_references_present(self):
        service = AIBOMService(hf_token="fake_token")
        component = service._create_component_section(
            "acme/test-model", {"license": "apache-2.0", "description": "A test model"}
        )
        self.assertIn("externalReferences", component)

        result = calculate_completeness_score(
            {"bomFormat": "CycloneDX", "components": [component]}, validate=False
        )

        self.assertTrue(result["field_checklist"]["external_references"].startswith("✔"))


if __name__ == '__main__':
    unittest.main()
