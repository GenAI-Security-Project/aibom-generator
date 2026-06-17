import unittest
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

if __name__ == '__main__':
    unittest.main()
