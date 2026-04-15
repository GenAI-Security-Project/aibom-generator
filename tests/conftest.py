import pytest

from src.models.extractor import EnhancedExtractor
from src.models.service import AIBOMService


SAMPLE_GGUF_METADATA = {
    "hyperparameter": {
        "context_length": 4096,
        "embedding_length": 4096,
        "block_count": 32,
        "attention_head_count": 32,
        "attention_head_count_kv": 8,
    },
    "model_type": "llama",
    "typeOfModel": "llama",
    "quantization": {"version": 2, "file_type": "Q4_0"},
    "context_length": 4096,
    "vocab_size": 32000,
    "tokenizer_class": "gpt2",
    "gguf_filename": "model.gguf",
}


STUB_CONFIG = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}


@pytest.fixture
def model_id():
    return "test-owner/test-model"


@pytest.fixture
def model_info():
    return {
        "id": "test-owner/test-model",
        "modelId": "test-owner/test-model",
        "pipeline_tag": "text-generation",
        "tags": ["llama"],
        "library_name": "transformers",
    }


@pytest.fixture
def model_card():
    return None


@pytest.fixture
def patch_extractor_io(monkeypatch):
    def fake_download(self, model_id, filename, **kw):
        if filename == "config.json":
            return STUB_CONFIG
        return None

    monkeypatch.setattr(EnhancedExtractor, "_download_and_parse_config", fake_download)
    monkeypatch.setattr(
        EnhancedExtractor, "_get_readme_content", lambda self, *a, **kw: None
    )


@pytest.fixture
def patch_service_io(monkeypatch):
    monkeypatch.setattr(
        AIBOMService, "_fetch_model_info", lambda self, model_id: {
            "id": model_id,
            "modelId": model_id,
            "pipeline_tag": "text-generation",
            "tags": ["llama"],
            "library_name": "transformers",
        }
    )
    monkeypatch.setattr(
        AIBOMService, "_fetch_model_card", lambda self, model_id: None
    )
    monkeypatch.setattr(
        EnhancedExtractor, "_download_and_parse_config",
        lambda self, model_id, filename, **kw: STUB_CONFIG if filename == "config.json" else None
    )
    monkeypatch.setattr(
        EnhancedExtractor, "_get_readme_content", lambda self, *a, **kw: None
    )


@pytest.fixture
def full_gguf_bytes():
    from tests.fixtures import get_full_gguf_bytes
    return get_full_gguf_bytes()
