import logging
from typing import Protocol, Dict, Any, List, runtime_checkable

from .gguf_metadata import fetch_gguf_metadata_from_repo, map_to_metadata

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelFileExtractor(Protocol):
    def can_extract(self, model_id: str) -> bool: ...
    def extract_metadata(self, model_id: str) -> Dict[str, Any]: ...


class GGUFFileExtractor:

    def can_extract(self, model_id: str) -> bool:
        try:
            from huggingface_hub import list_repo_files
            return any(f.endswith(".gguf") for f in list_repo_files(model_id))
        except Exception:
            return False

    def extract_metadata(self, model_id: str) -> Dict[str, Any]:
        from huggingface_hub import list_repo_files

        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                return {}

            model_info = fetch_gguf_metadata_from_repo(model_id, gguf_files[0])
            if model_info is None:
                return {}

            return map_to_metadata(model_info)
        except Exception as e:
            logger.warning(f"GGUF extraction failed for {model_id}: {e}")
            return {}


def default_extractors() -> List[ModelFileExtractor]:
    return [GGUFFileExtractor()]
