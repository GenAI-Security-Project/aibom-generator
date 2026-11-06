"""Public generation must not attach an ambient Hugging Face token."""

from huggingface_hub.utils import build_hf_headers

from src.models.model_file_extractors import GGUFFileExtractor, SafetensorsFileExtractor


def _has_authorization(headers: dict) -> bool:
    return any(key.lower() == "authorization" for key in headers)


def test_omitted_token_attaches_ambient_credential(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_private_operator_token")
    headers = build_hf_headers(token=None)
    assert _has_authorization(headers)
    assert not _has_authorization(build_hf_headers(token=False))


def test_file_extractors_forward_disabled_token(monkeypatch):
    seen = []

    def record_list(repo_id, token=None):
        seen.append(token)
        return ["model.gguf", "model.safetensors"]

    monkeypatch.setattr(
        "src.models.model_file_extractors.list_repo_files",
        record_list,
    )
    assert GGUFFileExtractor(hub_token=False).can_extract("owner/model") is True
    assert SafetensorsFileExtractor(hub_token=False).can_extract("owner/model") is True
    assert seen == [False, False]
