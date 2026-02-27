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
        # Metadata component name is timestamp by default, check ML component instead
        self.assertEqual(aibom["components"][0]["name"], "test-model")
        self.assertEqual(aibom["bomFormat"], "CycloneDX")

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_purl_encoding(self, mock_extractor_cls, mock_score):
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "tester"}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}
        
        self.service.hf_api.model_info.return_value = MagicMock(sha="123456")
        
        # Action
        model_id = "owner/model"
        aibom = self.service.generate_aibom(model_id)
        
        # Verify PURL encoding (slash should be / now, case preserved)
        # Expected: pkg:huggingface/owner/model@12345678 (truncated hash)
        
        # Check components section (ML model)
        ml_cmp = aibom["components"][0]
        self.assertIn("pkg:huggingface/owner/model@123456", ml_cmp["bom-ref"])
        self.assertEqual(ml_cmp["purl"], ml_cmp["bom-ref"])

        # Check dependency links use matching purls
        self.assertIn("pkg:generic/owner/model@123456", aibom["dependencies"][0]["ref"])
        self.assertIn("pkg:huggingface/owner/model@123456", aibom["dependencies"][0]["dependsOn"][0])
        
    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_generate_aibom_version_truncation(self, mock_extractor_cls, mock_score):
        # Setup
        mock_extractor = mock_extractor_cls.return_value
        long_sha = "a" * 40
        # Extractor typically puts commit in metadata if available
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "commit": long_sha}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}
        
        self.service.hf_api.model_info.return_value = MagicMock(sha=long_sha)
        
        # Action
        aibom = self.service.generate_aibom("owner/model")
        
        # Verify
        ml_cmp = aibom["components"][0]
        expected_version = "aaaaaaaa" # First 8 chars
        
        self.assertEqual(ml_cmp["version"], expected_version)
        self.assertIn(f"@{expected_version}", ml_cmp["purl"])
        self.assertIn(f"@{expected_version}", ml_cmp["bom-ref"])
        
        # Verify dependencies
        self.assertIn(f"@{expected_version}", aibom["dependencies"][0]["ref"])
        self.assertIn(f"@{expected_version}", aibom["dependencies"][0]["dependsOn"][0])

    def test_infer_io_formats(self):
        # Test Text Classification
        inputs, outputs = self.service._infer_io_formats("text-classification")
        self.assertEqual(inputs, ["string"])
        self.assertEqual(outputs, ["string"])
        
        # Test Image Classification
        inputs, outputs = self.service._infer_io_formats("image-classification")
        self.assertEqual(inputs, ["image"])
        self.assertEqual(outputs, ["string", "json"])
        
        # Test ASR (Audio)
        inputs, outputs = self.service._infer_io_formats("automatic-speech-recognition")
        self.assertEqual(inputs, ["audio"])
        self.assertEqual(outputs, ["string"])
        
        # Test Unknown
        inputs, outputs = self.service._infer_io_formats("unknown-task")
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])

    def test_generate_purl_huggingface_default(self):
        """Test _generate_purl with default huggingface type"""
        purl = self.service._generate_purl("owner/model", "1.0")
        self.assertEqual(purl, "pkg:huggingface/owner/model@1.0")
    
    def test_generate_purl_generic_type(self):
        """Test _generate_purl with generic type"""
        purl = self.service._generate_purl("owner/model", "1.0", purl_type="generic")
        self.assertEqual(purl, "pkg:generic/owner/model@1.0")
    
    def test_generate_purl_no_namespace(self):
        """Test _generate_purl without namespace"""
        purl = self.service._generate_purl("model", "1.0")
        self.assertEqual(purl, "pkg:huggingface/model@1.0")

if __name__ == '__main__':
    unittest.main()
