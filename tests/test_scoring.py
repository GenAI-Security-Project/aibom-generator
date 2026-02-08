import unittest
from src.models.scoring import calculate_completeness_score, ValidationSeverity

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

if __name__ == '__main__':
    unittest.main()
