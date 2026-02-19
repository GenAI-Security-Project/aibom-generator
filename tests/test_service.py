import unittest
from unittest.mock import MagicMock, patch
from src.models.service import AIBOMService

class TestService(unittest.TestCase):
    def setUp(self):
        self.service = AIBOMService(hf_token="fake_token")
        self.service.hf_api = MagicMock()
        
    def test_normalise_model_id(self):
        self.assertEqual(AIBOMService._normalise_model_id("owner/model"), "owner/model")
        self.assertEqual(AIBOMService._normalise_model_id("https://huggingface.co/owner/model"), "owner/model")
        self.assertEqual(AIBOMService._normalise_model_id("https://huggingface.co/owner/model/tree/main"), "owner/model")

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_basic(self, mock_extractor_cls, mock_score):
        # Mock dependencies
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "tester"}
        mock_extractor.extraction_results = {}
        
        mock_score.return_value = {"total_score": 50}
        
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        self.service.hf_api.model_card.return_value = MagicMock(data=MagicMock(to_dict=lambda: {}))
        
        aibom = self.service.generate_aibom("owner/test-model")
        
        self.assertIsNotNone(aibom)
        self.assertEqual(aibom["metadata"]["component"]["name"], "test-model")
        self.assertEqual(aibom["bomFormat"], "CycloneDX")

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_purl_format(self, mock_extractor_cls, mock_score):
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "tester"}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}
        
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        
        # Action
        model_id = "owner/model"
        aibom = self.service.generate_aibom(model_id)
        
        # Verify PURL format using packageurl-python library
        # Expected: pkg:huggingface/owner/model@... (namespace/name format)
        
        # Check components section (ML model) - uses packageurl-python
        ml_cmp = aibom["components"][0]
        self.assertIn("pkg:huggingface/owner/model@", ml_cmp["bom-ref"])
        self.assertIn("pkg:huggingface/owner/model@", ml_cmp["purl"])
        
        # Verify namespace and name are correctly separated
        self.assertEqual(ml_cmp["group"], "owner")
        self.assertEqual(ml_cmp["name"], "model")

    def test_generate_hf_purl_with_namespace(self):
        """Test PURL generation for model with namespace (owner/model)"""
        purl = self.service._generate_hf_purl("deepseek-ai/DeepSeek-R1", "abc1234")
        self.assertEqual(purl, "pkg:huggingface/deepseek-ai/DeepSeek-R1@abc1234")
        
    def test_generate_hf_purl_without_namespace(self):
        """Test PURL generation for model without namespace (just model name)"""
        purl = self.service._generate_hf_purl("gpt2", "1.0")
        self.assertEqual(purl, "pkg:huggingface/gpt2@1.0")
        
    def test_generate_hf_purl_special_characters(self):
        """Test PURL generation handles special characters correctly"""
        purl = self.service._generate_hf_purl("owner/model-name_v2", "1.0.0")
        self.assertEqual(purl, "pkg:huggingface/owner/model-name_v2@1.0.0")
        
    def test_generate_hf_purl_preserves_case(self):
        """Test PURL generation preserves case sensitivity"""
        purl = self.service._generate_hf_purl("OpenAI/GPT-4", "v1.0")
        self.assertEqual(purl, "pkg:huggingface/OpenAI/GPT-4@v1.0")

if __name__ == '__main__':
    unittest.main()
