import unittest
from src.models.service import AIBOMService

class TestPurlConstruction(unittest.TestCase):
    def setUp(self):
        self.service = AIBOMService(hf_token="fake_token")

    def test_standard_purl(self):
        """Test standard group/name PURL generation."""
        model_id = "AI45Research/AgentDoG-Qwen3-4B"
        version = "1.0"
        expected = "pkg:huggingface/AI45Research/AgentDoG-Qwen3-4B@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_spaces_in_group_and_name(self):
        """Test PURL generation with spaces in group and name."""
        model_id = "My Group/My Model"
        version = "1.0"
        # Spaces should be encoded as %20
        expected = "pkg:huggingface/My%20Group/My%20Model@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_special_chars_in_group(self):
        """Test PURL generation with special characters in group."""
        model_id = "c++-lib/cpp-model"
        version = "1.0"
        # + should be encoded as %2B
        expected = "pkg:huggingface/c%2B%2B-lib/cpp-model@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_special_chars_in_name(self):
        """Test PURL generation with special characters in name."""
        model_id = "user/model@v1"
        version = "1.0"
        # @ should be encoded as %40
        expected = "pkg:huggingface/user/model%40v1@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group(self):
        """Test PURL generation with no group (just model name)."""
        model_id = "justname"
        version = "1.0"
        expected = "pkg:huggingface/justname@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group_with_spaces(self):
        """Test PURL generation with no group and spaces in name."""
        model_id = "just name with spaces"
        version = "1.0"
        expected = "pkg:huggingface/just%20name%20with%20spaces@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group_with_special_chars(self):
        """Test PURL generation with no group and special chars in name."""
        model_id = "name@v1"
        version = "1.0"
        expected = "pkg:huggingface/name%40v1@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_mixed_case_preservation(self):
        """Test that mixed case is preserved in group and name."""
        model_id = "MixedCaseGroup/MixedCaseModel"
        version = "1.0"
        expected = "pkg:huggingface/MixedCaseGroup/MixedCaseModel@1.0"
        result = self.service._generate_hf_purl(model_id, version)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
