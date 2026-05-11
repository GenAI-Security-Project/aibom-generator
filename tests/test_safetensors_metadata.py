"""
Unit tests for safetensors metadata extraction module.
"""
import json
import struct
from unittest.mock import patch, MagicMock

from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import TensorInfo

from src.models.config_parsing import parse_config
from src.models.safetensors_metadata import (
    SafetensorsModelInfo,
    _extract_tensor_info,
    map_to_metadata,
    fetch_safetensors_metadata,
)
from tests.fixtures import build_safetensors_fixture


class TestParseConfig:
    """config.json → hyperparameter extraction, mirroring llama.cpp's find_hparam()."""

    def test_standard_llama_config(self):
        config = {
            "model_type": "llama",
            "num_hidden_layers": 16,
            "max_position_embeddings": 131072,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
        }
        result = parse_config(config)
        assert result["architecture"] == "llama"
        assert result["block_count"] == 16
        assert result["context_length"] == 131072
        assert result["embedding_length"] == 2048
        assert result["feed_forward_length"] == 8192
        assert result["attention_head_count"] == 32
        assert result["attention_head_count_kv"] == 8
        assert result["vocab_size"] == 128256

    def test_gpt2_alt_key_names(self):
        config = {
            "model_type": "gpt2",
            "n_layer": 12,
            "n_positions": 1024,
            "n_embd": 768,
            "n_inner": 3072,
            "n_head": 12,
            "vocab_size": 50257,
        }
        result = parse_config(config)
        assert result["architecture"] == "gpt2"
        assert result["block_count"] == 12
        assert result["context_length"] == 1024
        assert result["embedding_length"] == 768
        assert result["feed_forward_length"] == 3072
        assert result["attention_head_count"] == 12

    def test_bert_config(self):
        config = {
            "model_type": "bert",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "max_position_embeddings": 512,
            "vocab_size": 30522,
        }
        result = parse_config(config)
        assert result["architecture"] == "bert"
        assert result["block_count"] == 12
        assert result["embedding_length"] == 768
        assert result["attention_head_count"] == 12
        assert result["context_length"] == 512

    def test_vlm_nested_text_config(self):
        config = {
            "model_type": "llava",
            "text_config": {
                "model_type": "llama",
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "max_position_embeddings": 4096,
                "vocab_size": 32000,
            },
        }
        result = parse_config(config)
        assert result["architecture"] == "llama"
        assert result["block_count"] == 32
        assert result["embedding_length"] == 4096

    def test_missing_optional_fields(self):
        config = {"model_type": "custom", "hidden_size": 512}
        result = parse_config(config)
        assert result["architecture"] == "custom"
        assert result["embedding_length"] == 512
        assert result.get("block_count") is None
        assert result.get("attention_head_count") is None
        assert result.get("context_length") is None

    def test_first_matching_key_wins(self):
        config = {
            "n_layers": 10,
            "num_hidden_layers": 20,
        }
        result = parse_config(config)
        assert result["block_count"] == 10

    def test_empty_config(self):
        result = parse_config({})
        assert result.get("architecture") is None
        assert result.get("block_count") is None

    def test_rope_dimension_from_rotary_dim(self):
        """GPT-NeoX style: explicit rotary_dim key."""
        config = {"model_type": "gpt_neox", "rotary_dim": 32}
        result = parse_config(config)
        assert result["rope_dimension_count"] == 32

    def test_rope_dimension_from_rope_dim(self):
        """Some models use rope_dim directly."""
        config = {"model_type": "llama", "rope_dim": 64}
        result = parse_config(config)
        assert result["rope_dimension_count"] == 64

    def test_rope_dimension_none_when_absent(self):
        config = {"model_type": "llama", "hidden_size": 2048}
        result = parse_config(config)
        assert result.get("rope_dimension_count") is None


class TestMapToMetadata:
    """Output dict must match GGUF's map_to_metadata() format."""

    def test_full_model_info_produces_gguf_compatible_output(self):
        info = SafetensorsModelInfo(
            architecture="llama",
            context_length=131072,
            embedding_length=2048,
            block_count=16,
            attention_head_count=32,
            attention_head_count_kv=8,
            feed_forward_length=8192,
            vocab_size=128256,
            total_parameters=1_235_814_400,
        )
        result = map_to_metadata(info)
        assert result["model_type"] == "llama"
        assert result["typeOfModel"] == "llama"
        assert result["vocab_size"] == 128256
        assert result["context_length"] == 131072
        assert result["hyperparameter"]["context_length"] == 131072
        assert result["hyperparameter"]["embedding_length"] == 2048
        assert result["hyperparameter"]["block_count"] == 16
        assert result["hyperparameter"]["attention_head_count"] == 32
        assert result["hyperparameter"]["attention_head_count_kv"] == 8
        assert result["hyperparameter"]["feed_forward_length"] == 8192
        assert result["safetensors_total_parameters"] == 1_235_814_400

    def test_minimal_model_info(self):
        info = SafetensorsModelInfo(architecture="bert", embedding_length=768)
        result = map_to_metadata(info)
        assert result["model_type"] == "bert"
        assert result["hyperparameter"]["embedding_length"] == 768
        assert "attention_head_count" not in result["hyperparameter"]

    def test_none_fields_excluded_from_hyperparameters(self):
        info = SafetensorsModelInfo(architecture="gpt2", block_count=12)
        result = map_to_metadata(info)
        hp = result["hyperparameter"]
        assert hp["block_count"] == 12
        for key in [
            "context_length", "embedding_length", "feed_forward_length",
            "attention_head_count", "attention_head_count_kv",
        ]:
            assert key not in hp

    def test_no_architecture_produces_minimal_output(self):
        info = SafetensorsModelInfo(total_parameters=1000)
        result = map_to_metadata(info)
        assert "model_type" not in result
        assert result["safetensors_total_parameters"] == 1000

    def test_rope_dimension_count_in_hyperparameters(self):
        info = SafetensorsModelInfo(
            architecture="llama", block_count=16, rope_dimension_count=64,
        )
        result = map_to_metadata(info)
        assert result["hyperparameter"]["rope_dimension_count"] == 64

    def test_tokenizer_class_in_output(self):
        info = SafetensorsModelInfo(
            architecture="llama", tokenizer_class="LlamaTokenizerFast",
        )
        result = map_to_metadata(info)
        assert result["tokenizer_class"] == "LlamaTokenizerFast"

    def test_tokenizer_class_absent_when_none(self):
        info = SafetensorsModelInfo(architecture="llama")
        result = map_to_metadata(info)
        assert "tokenizer_class" not in result


class TestExtractTensorInfo:
    """Tensor info from safetensors headers."""

    def test_parameter_count_from_shapes(self):
        tensors = {
            "model.embed_tokens.weight": TensorInfo(
                dtype="BF16", shape=[128256, 2048], data_offsets=(0, 100)
            ),
            "model.layers.0.self_attn.q_proj.weight": TensorInfo(
                dtype="BF16", shape=[2048, 2048], data_offsets=(100, 200)
            ),
        }
        result = _extract_tensor_info(tensors)
        expected = (128256 * 2048) + (2048 * 2048)
        assert result["total_parameters"] == expected

    def test_dtype_distribution(self):
        tensors = {
            "a": TensorInfo(dtype="BF16", shape=[10, 10], data_offsets=(0, 1)),
            "b": TensorInfo(dtype="BF16", shape=[20, 20], data_offsets=(1, 2)),
            "c": TensorInfo(dtype="F32", shape=[5, 5], data_offsets=(2, 3)),
        }
        result = _extract_tensor_info(tensors)
        assert result["dtype_counts"]["BF16"] == 2
        assert result["dtype_counts"]["F32"] == 1

    def test_empty_tensors(self):
        result = _extract_tensor_info({})
        assert result["total_parameters"] == 0
        assert result["dtype_counts"] == {}


class TestFetchSafetensorsMetadata:
    """Integration: config.json + safetensors headers → SafetensorsModelInfo."""

    @patch("src.models.safetensors_metadata.HfApi")
    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_combines_config_and_tensor_info(self, mock_download, mock_hf_api_cls):
        import tempfile, os
        config = {
            "model_type": "llama",
            "num_hidden_layers": 16,
            "hidden_size": 2048,
            "max_position_embeddings": 131072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, tmp)
        tmp.close()

        def download_side_effect(repo_id, filename, **kwargs):
            if filename == "config.json":
                return tmp.name
            raise EntryNotFoundError(f"{filename} not found")

        mock_download.side_effect = download_side_effect

        # Mock safetensors repo metadata
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_file_meta = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.dtype = "BF16"
        mock_tensor.shape = [128256, 2048]
        mock_file_meta.tensors = {"model.embed_tokens.weight": mock_tensor}
        mock_repo_meta = MagicMock()
        mock_repo_meta.files_metadata = {"model.safetensors": mock_file_meta}
        mock_api.get_safetensors_metadata.return_value = mock_repo_meta

        info = fetch_safetensors_metadata("meta-llama/Llama-3.2-1B")
        os.unlink(tmp.name)

        assert info is not None
        assert info.architecture == "llama"
        assert info.block_count == 16
        assert info.embedding_length == 2048
        assert info.total_parameters == 128256 * 2048
        assert "BF16" in info.dtype_counts

    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_config_json_missing_returns_none(self, mock_download):
        mock_download.side_effect = EntryNotFoundError("config.json not found")

        result = fetch_safetensors_metadata("org/model")
        assert result is None

    @patch("src.models.safetensors_metadata.HfApi")
    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_safetensors_missing_still_returns_config_data(self, mock_download, mock_hf_api_cls):
        import tempfile, os
        config = {"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12}
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, tmp)
        tmp.close()
        mock_download.return_value = tmp.name

        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.get_safetensors_metadata.side_effect = Exception("not a safetensors repo")

        info = fetch_safetensors_metadata("org/model")
        os.unlink(tmp.name)

        assert info is not None
        assert info.architecture == "bert"
        assert info.embedding_length == 768
        assert info.total_parameters is None

    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_network_error_returns_none(self, mock_download):
        mock_download.side_effect = ConnectionError("network error")

        result = fetch_safetensors_metadata("org/model")
        assert result is None

    @patch("src.models.safetensors_metadata.HfApi")
    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_tokenizer_class_extracted_from_tokenizer_config(self, mock_download, mock_hf_api_cls):
        import tempfile, os
        config = {"model_type": "llama", "hidden_size": 2048}
        tokenizer_config = {"tokenizer_class": "LlamaTokenizerFast"}

        config_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, config_tmp)
        config_tmp.close()

        tok_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(tokenizer_config, tok_tmp)
        tok_tmp.close()

        def download_side_effect(repo_id, filename, **kwargs):
            if filename == "config.json":
                return config_tmp.name
            elif filename == "tokenizer_config.json":
                return tok_tmp.name
            raise EntryNotFoundError(f"{filename} not found")

        mock_download.side_effect = download_side_effect
        mock_hf_api_cls.return_value.get_safetensors_metadata.side_effect = Exception("skip")

        info = fetch_safetensors_metadata("org/model")
        os.unlink(config_tmp.name)
        os.unlink(tok_tmp.name)

        assert info is not None
        assert info.tokenizer_class == "LlamaTokenizerFast"

    @patch("src.models.safetensors_metadata.HfApi")
    @patch("src.models.safetensors_metadata.hf_hub_download")
    def test_missing_tokenizer_config_still_works(self, mock_download, mock_hf_api_cls):
        import tempfile, os
        config = {"model_type": "llama", "hidden_size": 2048}
        config_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config, config_tmp)
        config_tmp.close()

        def download_side_effect(repo_id, filename, **kwargs):
            if filename == "config.json":
                return config_tmp.name
            raise EntryNotFoundError(f"{filename} not found")

        mock_download.side_effect = download_side_effect
        mock_hf_api_cls.return_value.get_safetensors_metadata.side_effect = Exception("skip")

        info = fetch_safetensors_metadata("org/model")
        os.unlink(config_tmp.name)

        assert info is not None
        assert info.tokenizer_class is None
        assert info.architecture == "llama"


class TestSafetensorsFileExtractor:
    """SafetensorsFileExtractor implements ModelFileExtractor protocol."""

    def _make_extractor(self):
        from src.models.model_file_extractors import SafetensorsFileExtractor
        return SafetensorsFileExtractor()

    @patch("src.models.model_file_extractors.list_repo_files")
    def test_can_extract_true_when_single_safetensors(self, mock_list):
        mock_list.return_value = ["model.safetensors", "config.json"]
        assert self._make_extractor().can_extract("org/model") is True

    @patch("src.models.model_file_extractors.list_repo_files")
    def test_can_extract_true_for_sharded(self, mock_list):
        mock_list.return_value = [
            "model-00001-of-00003.safetensors",
            "model.safetensors.index.json",
        ]
        assert self._make_extractor().can_extract("org/model") is True

    @patch("src.models.model_file_extractors.list_repo_files")
    def test_can_extract_false_when_no_safetensors(self, mock_list):
        mock_list.return_value = ["pytorch_model.bin", "config.json"]
        assert self._make_extractor().can_extract("org/model") is False

    @patch("src.models.model_file_extractors.list_repo_files")
    def test_can_extract_false_on_network_error(self, mock_list):
        mock_list.side_effect = Exception("network error")
        assert self._make_extractor().can_extract("org/model") is False

    @patch("src.models.model_file_extractors.fetch_safetensors_metadata")
    def test_extract_metadata_returns_dict(self, mock_fetch):
        mock_fetch.return_value = SafetensorsModelInfo(
            architecture="llama", block_count=16, embedding_length=2048,
        )

        result = self._make_extractor().extract_metadata("org/model")
        assert result["model_type"] == "llama"
        assert result["hyperparameter"]["block_count"] == 16
        assert result["hyperparameter"]["embedding_length"] == 2048

    @patch("src.models.model_file_extractors.fetch_safetensors_metadata")
    def test_extract_metadata_failure_returns_empty_dict(self, mock_fetch):
        mock_fetch.side_effect = Exception("something broke")

        result = self._make_extractor().extract_metadata("org/model")
        assert result == {}

    @patch("src.models.model_file_extractors.fetch_safetensors_metadata")
    def test_extract_metadata_returns_empty_when_fetch_returns_none(self, mock_fetch):
        mock_fetch.return_value = None

        result = self._make_extractor().extract_metadata("org/model")
        assert result == {}


class TestExtractorIntegration:
    """Test messages crossing the EnhancedExtractor ↔ ModelFileExtractor boundary.

    Sandi Metz: test incoming messages (return values) and outgoing command messages
    (side effects), not internal implementation details.
    """

    def _make_extractor(self, model_file_extractors):
        from src.models.extractor import EnhancedExtractor
        mock_api = MagicMock()
        mock_registry_manager = MagicMock()
        mock_registry_manager.get_field_definitions.return_value = {
            "hyperparameter": {"tier": "important"},
        }
        with patch("src.models.extractor.get_field_registry_manager", return_value=mock_registry_manager):
            return EnhancedExtractor(hf_api=mock_api, model_file_extractors=model_file_extractors)

    def test_default_extractors_returns_safetensors_then_gguf(self):
        """Composition root: safetensors before GGUF (original source takes precedence)."""
        from src.models.model_file_extractors import default_extractors, GGUFFileExtractor, SafetensorsFileExtractor
        extractors = default_extractors()
        assert len(extractors) == 2
        assert isinstance(extractors[0], SafetensorsFileExtractor)
        assert isinstance(extractors[1], GGUFFileExtractor)

    def test_first_successful_extractor_wins(self):
        """_extract_model_file_metadata returns the first extractor's result, not both."""
        ext = self._make_extractor([])

        first = MagicMock()
        first.can_extract.return_value = True
        first.extract_metadata.return_value = {"model_type": "llama", "hyperparameter": {"block_count": 32}}

        second = MagicMock()
        second.can_extract.return_value = True
        second.extract_metadata.return_value = {"model_type": "bert", "hyperparameter": {"block_count": 12}}

        ext.model_file_extractors = [first, second]
        result = ext._extract_model_file_metadata("org/model")

        assert result["model_type"] == "llama"
        assert result["hyperparameter"]["block_count"] == 32
        second.extract_metadata.assert_not_called()

    def test_falls_through_to_second_extractor(self):
        """When first extractor can't handle the repo, second gets a chance."""
        ext = self._make_extractor([])

        first = MagicMock()
        first.can_extract.return_value = False

        second = MagicMock()
        second.can_extract.return_value = True
        second.extract_metadata.return_value = {"model_type": "llama"}

        ext.model_file_extractors = [first, second]
        result = ext._extract_model_file_metadata("org/model")

        assert result["model_type"] == "llama"
        first.extract_metadata.assert_not_called()

    def test_model_file_metadata_overwrites_registry_extraction(self):
        """Model file metadata merges after registry extraction — last write wins."""
        ext = self._make_extractor([])

        mock_ext = MagicMock()
        mock_ext.can_extract.return_value = True
        mock_ext.extract_metadata.return_value = {
            "hyperparameter": {"block_count": 32, "embedding_length": 4096},
            "model_type": "phi3",
        }
        ext.model_file_extractors = [mock_ext]

        mock_model_info = MagicMock()
        mock_model_info.sha = "abc123"
        mock_model_info.card_data = None
        mock_model_info.tags = []

        metadata = ext._registry_driven_extraction("org/model", mock_model_info, None)

        # model file extractor's values should be present
        assert metadata["hyperparameter"]["block_count"] == 32
        assert metadata["model_type"] == "phi3"

    def test_config_json_and_model_file_both_produce_hyperparameters(self):
        """When config.json AND model file both have hyperparameters, model file wins.

        This happens because _registry_driven_extraction runs config extraction first
        (via _extract_registry_field), then model file metadata overwrites (lines 246-248).
        """
        ext = self._make_extractor([])

        # Model file extractor returns different hyperparameters
        mock_ext = MagicMock()
        mock_ext.can_extract.return_value = True
        mock_ext.extract_metadata.return_value = {
            "hyperparameter": {"block_count": 99, "embedding_length": 999},
        }
        ext.model_file_extractors = [mock_ext]

        mock_model_info = MagicMock()
        mock_model_info.sha = None
        mock_model_info.card_data = None
        mock_model_info.tags = []

        # Stub _download_and_parse_config to return a config.json with different values
        ext._download_and_parse_config = MagicMock(side_effect=lambda mid, fname: (
            {"num_hidden_layers": 16, "hidden_size": 2048} if fname == "config.json" else None
        ))

        metadata = ext._registry_driven_extraction("org/model", mock_model_info, None)

        # Model file extractor's values win (later write)
        hp = metadata["hyperparameter"]
        assert hp["block_count"] == 99
        assert hp["embedding_length"] == 999


class TestFixtureEndToEnd:
    """End-to-end tests with real safetensors binary from fixture."""

    def test_parse_real_safetensors_header(self):
        st_bytes, config = build_safetensors_fixture(
            vocab_size=256, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_kv_heads=2, intermediate_size=128,
        )
        header_size = struct.unpack("<Q", st_bytes[:8])[0]
        header = json.loads(st_bytes[8 : 8 + header_size])

        assert "model.embed_tokens.weight" in header
        assert header["model.embed_tokens.weight"]["shape"] == [256, 64]
        assert header["model.embed_tokens.weight"]["dtype"] == "BF16"
        assert "model.layers.0.self_attn.q_proj.weight" in header
        assert header["model.layers.0.self_attn.q_proj.weight"]["shape"] == [64, 64]

    def test_full_pipeline_with_fixture(self):
        _, config = build_safetensors_fixture(
            vocab_size=256, hidden_size=64, num_layers=2,
            num_attention_heads=4, num_kv_heads=2, intermediate_size=128,
            model_type="llama",
        )
        parsed = parse_config(config)
        info = SafetensorsModelInfo(
            architecture=parsed.get("architecture"),
            block_count=parsed.get("block_count"),
            embedding_length=parsed.get("embedding_length"),
            attention_head_count=parsed.get("attention_head_count"),
            attention_head_count_kv=parsed.get("attention_head_count_kv"),
            feed_forward_length=parsed.get("feed_forward_length"),
            vocab_size=parsed.get("vocab_size"),
            context_length=parsed.get("context_length"),
        )
        result = map_to_metadata(info)
        assert result["model_type"] == "llama"
        assert result["hyperparameter"]["block_count"] == 2
        assert result["hyperparameter"]["embedding_length"] == 64
        assert result["hyperparameter"]["attention_head_count"] == 4
        assert result["hyperparameter"]["attention_head_count_kv"] == 2
        assert result["hyperparameter"]["feed_forward_length"] == 128
        assert result["vocab_size"] == 256

    def test_fixture_tensor_shapes_match_config(self):
        st_bytes, config = build_safetensors_fixture(vocab_size=512, hidden_size=128)
        header_size = struct.unpack("<Q", st_bytes[:8])[0]
        header = json.loads(st_bytes[8 : 8 + header_size])

        embed_shape = header["model.embed_tokens.weight"]["shape"]
        assert embed_shape[0] == config["vocab_size"]
        assert embed_shape[1] == config["hidden_size"]

        gate_shape = header["model.layers.0.mlp.gate_proj.weight"]["shape"]
        assert gate_shape[0] == config["intermediate_size"]
