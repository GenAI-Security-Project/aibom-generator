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
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_spaces_in_group_and_name(self):
        """Test PURL generation with spaces in group and name."""
        model_id = "My Group/My Model"
        version = "1.0"
        # Spaces should be encoded as %20
        expected = "pkg:huggingface/My%20Group/My%20Model@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_special_chars_in_group(self):
        """Test PURL generation with special characters in group."""
        model_id = "c++-lib/cpp-model"
        version = "1.0"
        # + should be encoded as %2B
        expected = "pkg:huggingface/c%2B%2B-lib/cpp-model@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_special_chars_in_name(self):
        """Test PURL generation with special characters in name."""
        model_id = "user/model@v1"
        version = "1.0"
        # @ should be encoded as %40
        expected = "pkg:huggingface/user/model%40v1@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group(self):
        """Test PURL generation with no group (just model name)."""
        model_id = "justname"
        version = "1.0"
        expected = "pkg:huggingface/justname@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group_with_spaces(self):
        """Test PURL generation with no group and spaces in name."""
        model_id = "just name with spaces"
        version = "1.0"
        expected = "pkg:huggingface/just%20name%20with%20spaces@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_no_group_with_special_chars(self):
        """Test PURL generation with no group and special chars in name."""
        model_id = "name@v1"
        version = "1.0"
        expected = "pkg:huggingface/name%40v1@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_mixed_case_preservation(self):
        """Test that mixed case is preserved in group and name."""
        model_id = "MixedCaseGroup/MixedCaseModel"
        version = "1.0"
        expected = "pkg:huggingface/MixedCaseGroup/MixedCaseModel@1.0"
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

    def test_generic_purl_with_namespace(self):
        """Test generic PURL generation with namespace."""
        model_id = "owner/model"
        version = "1.0"
        expected = "pkg:generic/owner/model@1.0"
        result = self.service._generate_purl(model_id, version, purl_type="generic")
        self.assertEqual(result, expected)

    def test_generic_purl_without_namespace(self):
        """Test generic PURL generation without namespace."""
        model_id = "model"
        version = "1.0"
        expected = "pkg:generic/model@1.0"
        result = self.service._generate_purl(model_id, version, purl_type="generic")
        self.assertEqual(result, expected)

    def test_owner_name_separator_is_not_percent_encoded(self):
        """Regression for #13: the owner/name separator is a purl namespace
        delimiter, so it must stay a literal "/" and never become "%2F"."""
        purl = self.service._generate_purl("owner/model", "1.0")
        self.assertNotIn("%2F", purl.upper())
        self.assertEqual(purl, "pkg:huggingface/owner/model@1.0")

    def test_purl_round_trips_into_namespace_and_name(self):
        from packageurl import PackageURL

        parsed = PackageURL.from_string(self.service._generate_purl("owner/model", "1.0"))
        self.assertEqual(parsed.type, "huggingface")
        self.assertEqual(parsed.namespace, "owner")
        self.assertEqual(parsed.name, "model")
        self.assertEqual(parsed.version, "1.0")

    def test_component_purl_and_bom_ref_keep_literal_separator(self):
        component = self.service._create_component_section("owner/model", {"commit": "a" * 40})
        self.assertEqual(component["purl"], "pkg:huggingface/owner/model@aaaaaaaa")
        self.assertEqual(component["bom-ref"], component["purl"])

    def test_default_purl_type_is_huggingface(self):
        """Test that default purl_type is 'huggingface'."""
        model_id = "owner/model"
        version = "1.0"
        expected = "pkg:huggingface/owner/model@1.0"
        # Call without purl_type parameter - should default to huggingface
        result = self.service._generate_purl(model_id, version)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
