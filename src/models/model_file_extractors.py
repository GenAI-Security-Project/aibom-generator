import logging
from typing import Protocol, Dict, List, Union, runtime_checkable

from huggingface_hub import list_repo_files

from .gguf_metadata import fetch_gguf_metadata_from_repo, map_to_metadata as gguf_map_to_metadata
from .safetensors_metadata import fetch_safetensors_metadata, map_to_metadata as st_map_to_metadata

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelFileExtractor(Protocol):
    def can_extract(self, model_id: str) -> bool: ...
    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]: ...


class GGUFFileExtractor:

    def can_extract(self, model_id: str) -> bool:
        try:
            return any(f.endswith(".gguf") for f in list_repo_files(model_id))
        except Exception:
            return False

    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]:
        try:
            files = list_repo_files(model_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                return {}

            model_info = fetch_gguf_metadata_from_repo(model_id, gguf_files[0])
            if model_info is None:
                return {}

            return gguf_map_to_metadata(model_info)
        except Exception as e:
            logger.warning(f"GGUF extraction failed for {model_id}: {e}")
            return {}


class SafetensorsFileExtractor:

    def can_extract(self, model_id: str) -> bool:
        try:
            return any(f.endswith(".safetensors") for f in list_repo_files(model_id))
        except Exception:
            return False

    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]:
        try:
            info = fetch_safetensors_metadata(model_id)
            if info is None:
                return {}
            return st_map_to_metadata(info)
        except Exception as e:
            logger.warning(f"Safetensors extraction failed for {model_id}: {e}")
            return {}


def default_extractors() -> List[ModelFileExtractor]:
    return [SafetensorsFileExtractor(), GGUFFileExtractor()]
