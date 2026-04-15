from src.models.extractor import EnhancedExtractor
from tests.doubles import FakeModelFileExtractor, FailingModelFileExtractor
from tests.conftest import SAMPLE_GGUF_METADATA


class TestGGUFBytesProduceMetadata:
    """Real GGUF binary data, parsed through the full chain, produces
    the expected AIBOM metadata fields."""

    def test_full_chain_bytes_to_metadata(self, full_gguf_bytes):
        from src.models.gguf_metadata import parse_gguf_metadata, extract_model_info, map_to_metadata

        parsed = parse_gguf_metadata(full_gguf_bytes, filename="model.gguf")
        model_info = extract_model_info(parsed)
        metadata = map_to_metadata(model_info)

        assert metadata["hyperparameter"]["context_length"] == 4096
        assert metadata["hyperparameter"]["block_count"] == 32
        assert metadata["quantization"]["version"] == 2
        assert metadata["model_type"] == "llama"
        assert metadata["gguf_filename"] == "model.gguf"

    def test_minimal_gguf_without_hyperparameters(self):
        from tests.fixtures import build_gguf_bytes
        from src.models.gguf_metadata import parse_gguf_metadata, extract_model_info, map_to_metadata

        data = build_gguf_bytes(architecture="bert", model_name="bert-base")
        parsed = parse_gguf_metadata(data)
        model_info = extract_model_info(parsed)
        metadata = map_to_metadata(model_info)

        assert "hyperparameter" not in metadata
        assert metadata["model_type"] == "bert"


class TestGGUFFileExtractorWithMockedAPI:
    """GGUFFileExtractor correctly interacts with HF API to find
    and fetch GGUF files."""

    def test_can_extract_true_when_gguf_files_present(self, monkeypatch):
        from src.models.model_file_extractors import GGUFFileExtractor

        monkeypatch.setattr(
            "huggingface_hub.list_repo_files",
            lambda model_id, **kw: ["README.md", "model-q4.gguf"],
        )
        assert GGUFFileExtractor().can_extract("owner/model") is True

    def test_can_extract_false_when_no_gguf_files(self, monkeypatch):
        from src.models.model_file_extractors import GGUFFileExtractor

        monkeypatch.setattr(
            "huggingface_hub.list_repo_files",
            lambda model_id, **kw: ["README.md", "model.safetensors"],
        )
        assert GGUFFileExtractor().can_extract("owner/model") is False

    def test_can_extract_false_when_api_errors(self, monkeypatch):
        from src.models.model_file_extractors import GGUFFileExtractor

        def raise_error(*a, **kw):
            raise RuntimeError("network")

        monkeypatch.setattr("huggingface_hub.list_repo_files", raise_error)
        assert GGUFFileExtractor().can_extract("owner/model") is False

    def test_extract_metadata_selects_first_gguf_file(self, monkeypatch, full_gguf_bytes):
        from src.models.model_file_extractors import GGUFFileExtractor

        monkeypatch.setattr(
            "huggingface_hub.list_repo_files",
            lambda model_id, **kw: ["a.gguf", "b.gguf"],
        )
        captured = {}

        def fake_fetch(repo_id, filename, **kw):
            captured["filename"] = filename
            from src.models.gguf_metadata import parse_gguf_metadata, extract_model_info
            parsed = parse_gguf_metadata(full_gguf_bytes, filename=filename)
            return extract_model_info(parsed)

        monkeypatch.setattr(
            "src.models.model_file_extractors.fetch_gguf_metadata_from_repo", fake_fetch
        )
        result = GGUFFileExtractor().extract_metadata("owner/model")

        assert captured["filename"] == "a.gguf"
        assert result["hyperparameter"]["context_length"] == 4096

    def test_extract_metadata_returns_empty_on_fetch_failure(self, monkeypatch):
        from src.models.model_file_extractors import GGUFFileExtractor

        monkeypatch.setattr(
            "huggingface_hub.list_repo_files",
            lambda model_id, **kw: ["model.gguf"],
        )

        def raise_error(*a, **kw):
            raise RuntimeError("timeout")

        monkeypatch.setattr(
            "src.models.model_file_extractors.fetch_gguf_metadata_from_repo", raise_error
        )
        assert GGUFFileExtractor().extract_metadata("owner/model") == {}


class TestGGUFModelEnrichesExtractedMetadata:
    """When a model has GGUF files, hyperparameters from the GGUF
    header appear in the extracted metadata."""

    def test_hyperparameters_present(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        fake = FakeModelFileExtractor(can=True, metadata=SAMPLE_GGUF_METADATA)
        extractor = EnhancedExtractor(model_file_extractors=[fake])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert metadata["hyperparameter"]["context_length"] == 4096
        assert metadata["hyperparameter"]["block_count"] == 32

    def test_quantization_present(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        fake = FakeModelFileExtractor(can=True, metadata={
            "quantization": {"version": 2, "file_type": "Q4_0"},
        })
        extractor = EnhancedExtractor(model_file_extractors=[fake])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert metadata["quantization"]["file_type"] == "Q4_0"


class TestModelWithoutModelFilesStillExtracts:
    """When no model file extractor matches, extraction still succeeds
    using API/card/config strategies."""

    def test_no_matching_extractor(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        fake = FakeModelFileExtractor(can=False)
        extractor = EnhancedExtractor(model_file_extractors=[fake])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert "hyperparameter" not in metadata

    def test_no_extractors_configured(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        extractor = EnhancedExtractor(model_file_extractors=[])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert "hyperparameter" not in metadata


class TestModelFileMetadataTakesPrecedence:
    """Model file header data is ground truth — it overwrites values
    from earlier extraction strategies (API, config, model card)."""

    def test_type_of_model_from_gguf_wins(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        fake = FakeModelFileExtractor(can=True, metadata={"typeOfModel": "mistral"})
        extractor = EnhancedExtractor(model_file_extractors=[fake])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert metadata["typeOfModel"] == "mistral"


class TestExtractionIsResilientToModelFileFailures:
    """A failing or erroring model file extractor does not prevent
    metadata extraction from succeeding."""

    def test_failing_extractor_skipped(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        extractor = EnhancedExtractor(model_file_extractors=[FailingModelFileExtractor()])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert metadata is not None
        assert "hyperparameter" not in metadata

    def test_failing_extractor_followed_by_working_extractor(
        self, model_id, model_info, model_card, patch_extractor_io
    ):
        fake = FakeModelFileExtractor(can=True, metadata={"hyperparameter": {"block_count": 32}})
        extractor = EnhancedExtractor(model_file_extractors=[FailingModelFileExtractor(), fake])
        metadata = extractor.extract_metadata(model_id, model_info, model_card)
        assert metadata["hyperparameter"]["block_count"] == 32
