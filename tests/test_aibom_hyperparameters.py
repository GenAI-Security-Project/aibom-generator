from src.models.service import AIBOMService
from tests.doubles import FakeModelFileExtractor
from tests.conftest import SAMPLE_GGUF_METADATA


class TestHyperparametersAppearInAIBOM:
    """When a model has GGUF files with hyperparameters, the generated
    AIBOM contains them in component properties."""

    def test_hyperparameters_in_component_properties(self, patch_service_io):
        fake = FakeModelFileExtractor(can=True, metadata=SAMPLE_GGUF_METADATA)
        service = AIBOMService(model_file_extractors=[fake])
        aibom = service.generate_aibom("owner/model")
        props = aibom["components"][0]["modelCard"]["properties"]
        prop_names = [p["name"] for p in props]
        assert "genai:aibom:modelcard:hyperparameter" in prop_names


class TestQuantizationAppearsInAIBOM:
    """When a model has GGUF files with quantization info, the generated
    AIBOM contains it in component properties."""

    def test_quantization_in_component_properties(self, patch_service_io):
        fake = FakeModelFileExtractor(can=True, metadata=SAMPLE_GGUF_METADATA)
        service = AIBOMService(model_file_extractors=[fake])
        aibom = service.generate_aibom("owner/model")
        props = aibom["components"][0]["modelCard"]["properties"]
        prop_names = [p["name"] for p in props]
        assert "genai:aibom:modelcard:quantizationVersion" in prop_names
        assert "genai:aibom:modelcard:quantizationFileType" in prop_names


class TestAIBOMWithoutModelFileData:
    """When no model file extractor matches, the AIBOM is still
    generated successfully -- just without hyperparameter fields."""

    def test_no_hyperparameters_when_no_gguf(self, patch_service_io):
        fake = FakeModelFileExtractor(can=False)
        service = AIBOMService(model_file_extractors=[fake])
        aibom = service.generate_aibom("owner/model")
        props = aibom["components"][0].get("modelCard", {}).get("properties", [])
        prop_names = [p["name"] for p in props]
        assert not any(name == "genai:aibom:modelcard:hyperparameter" for name in prop_names)
