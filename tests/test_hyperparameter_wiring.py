"""
Tests that hyperparameter extraction is wired into the enhanced extractor pipeline.

The enhanced extractor should extract hyperparameters from any model format via
duck-typed model file extractors (GGUF, safetensors, future formats).
"""
import json
from unittest.mock import MagicMock, patch


class TestHyperparameterWiring:
    """Verify hyperparameter field flows through the enhanced extractor regardless of format."""

    def _make_extractor(self):
        """Create an EnhancedExtractor with mocked registry and no real extractors."""
        from src.models.extractor import EnhancedExtractor

        mock_api = MagicMock()
        mock_registry_manager = MagicMock()
        mock_registry_manager.get_field_definitions.return_value = {
            "hyperparameter": {
                "tier": "important",
                "category": "component_model_card",
                "aibom_generation": {
                    "source_fields": ["hyperparameter"],
                },
            }
        }

        with patch("src.models.extractor.get_field_registry_manager", return_value=mock_registry_manager):
            ext = EnhancedExtractor(hf_api=mock_api, model_file_extractors=[])
        return ext

    # --- config.json path (safetensors repos, HF repos with config.json) ---

    def test_config_extraction_returns_hyperparameters(self):
        """config.json present → hyperparameters extracted via key fallback chains."""
        extractor = self._make_extractor()
        config_data = {
            "model_type": "llama",
            "num_hidden_layers": 16,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "vocab_size": 128256,
        }
        context = {
            "model_id": "org/model",
            "model_info": {},
            "model_card": None,
            "config_data": config_data,
            "tokenizer_config": None,
            "readme_content": None,
        }

        result = extractor._try_config_extraction("hyperparameter", context)

        assert result is not None
        hp = json.loads(result) if isinstance(result, str) else result
        assert hp["block_count"] == 16
        assert hp["embedding_length"] == 2048
        assert hp["attention_head_count"] == 32

    def test_config_extraction_hyperparameter_no_config_returns_none(self):
        """No config.json → config strategy returns None (other strategies can fill in)."""
        extractor = self._make_extractor()
        context = {
            "model_id": "org/model",
            "model_info": {},
            "model_card": None,
            "config_data": None,
            "tokenizer_config": None,
            "readme_content": None,
        }

        result = extractor._try_config_extraction("hyperparameter", context)
        assert result is None

    def test_existing_config_mappings_still_work(self):
        """Adding hyperparameter support doesn't break model_type, vocab_size extraction."""
        extractor = self._make_extractor()
        config_data = {"model_type": "llama", "vocab_size": 32000}
        context = {
            "model_id": "org/model",
            "model_info": {},
            "model_card": None,
            "config_data": config_data,
            "tokenizer_config": None,
            "readme_content": None,
        }

        assert extractor._try_config_extraction("model_type", context) == "llama"
        assert extractor._try_config_extraction("vocab_size", context) == 32000

    # --- model file extractor path (GGUF repos without config.json, any format) ---

    def test_model_file_extraction_called_for_hyperparameter(self):
        """When config.json is absent, _extract_model_file_metadata returns hyperparameters."""
        extractor = self._make_extractor()

        # Mock an extractor that can handle this model
        mock_ext = MagicMock()
        mock_ext.can_extract.return_value = True
        mock_ext.extract_metadata.return_value = {
            "hyperparameter": {
                "context_length": 4096,
                "embedding_length": 3072,
                "block_count": 32,
            },
            "model_type": "phi3",
        }
        extractor.model_file_extractors = [mock_ext]

        result = extractor._extract_model_file_metadata("org/gguf-only-model")

        assert result is not None
        assert result["hyperparameter"]["block_count"] == 32
        assert result["model_type"] == "phi3"

    def test_model_file_extraction_returns_empty_when_no_extractor_matches(self):
        """No extractor can handle this model → returns empty dict."""
        extractor = self._make_extractor()

        mock_ext = MagicMock()
        mock_ext.can_extract.return_value = False
        extractor.model_file_extractors = [mock_ext]

        result = extractor._extract_model_file_metadata("org/unknown-format")
        assert result == {}

    def test_model_file_extraction_skipped_for_empty_extractors(self):
        """No extractors registered → returns empty dict."""
        extractor = self._make_extractor()
        extractor.model_file_extractors = []

        result = extractor._extract_model_file_metadata("org/model")
        assert result == {}

    def test_full_extraction_merges_model_file_metadata(self):
        """Registry extraction + model file metadata are merged in _registry_driven_extraction."""
        extractor = self._make_extractor()

        mock_ext = MagicMock()
        mock_ext.can_extract.return_value = True
        mock_ext.extract_metadata.return_value = {
            "hyperparameter": {
                "context_length": 4096,
                "block_count": 32,
            },
        }
        extractor.model_file_extractors = [mock_ext]

        mock_model_info = MagicMock()
        mock_model_info.sha = "abc123"

        metadata = extractor._registry_driven_extraction(
            "org/gguf-only-model", mock_model_info, None
        )

        # Model file metadata should be merged in
        assert "hyperparameter" in metadata
        hp = metadata["hyperparameter"]
        assert hp["block_count"] == 32

    def test_model_file_extraction_error_returns_empty(self):
        """Extractor errors are caught gracefully."""
        extractor = self._make_extractor()

        mock_ext = MagicMock()
        mock_ext.can_extract.side_effect = Exception("network error")
        extractor.model_file_extractors = [mock_ext]

        result = extractor._extract_model_file_metadata("org/model")
        assert result == {}
