import logging
import os
import re
from typing import Optional, Protocol, Dict, List, Union, runtime_checkable

from huggingface_hub import HfApi, list_repo_files

from .gguf_metadata import fetch_gguf_metadata_from_repo, map_to_metadata as gguf_map_to_metadata
from .safetensors_metadata import fetch_safetensors_metadata, map_to_metadata as st_map_to_metadata

logger = logging.getLogger(__name__)

# Weight-file formats loadable directly from a Python ML stack
# (transformers / PyTorch / TensorFlow / ONNX Runtime, etc.).
PYTHON_WEIGHT_EXTENSIONS = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
    ".h5", ".msgpack", ".onnx", ".tflite", ".pb",
}
# Weight-file formats that require a dedicated LLM inference stack
# (llama.cpp / Ollama / LM Studio, etc.) rather than a Python import.
LLM_STACK_WEIGHT_EXTENSIONS = {".gguf", ".ggml"}

# Taxonomy values for the runtime_requirement attribute.
RUNTIME_PYTHON = "python"
RUNTIME_LLM_STACK = "llm-stack"

# Matches HuggingFace shard suffixes like "-00001-of-00002" or ".00001-of-00003"
# so shards of the same model/quant variant can be grouped together.
_SHARD_SUFFIX_RE = re.compile(r"[-.]?\d{3,5}-of-\d{3,5}")


@runtime_checkable
class ModelFileExtractor(Protocol):
    def can_extract(self, model_id: str) -> bool: ...
    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]: ...


class GGUFFileExtractor:
    def __init__(self, hub_token: str | bool | None = None):
        self.hub_token = hub_token

    def can_extract(self, model_id: str) -> bool:
        try:
            return any(
                f.endswith(".gguf")
                for f in list_repo_files(model_id, token=self.hub_token)
            )
        except Exception:
            return False

    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]:
        try:
            files = list_repo_files(model_id, token=self.hub_token)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                return {}

            model_info = fetch_gguf_metadata_from_repo(
                model_id,
                gguf_files[0],
                hf_token=self.hub_token,
            )
            if model_info is None:
                return {}

            return gguf_map_to_metadata(model_info)
        except Exception as e:
            logger.warning(f"GGUF extraction failed for {model_id}: {e}")
            return {}


class SafetensorsFileExtractor:
    def __init__(self, hub_token: str | bool | None = None):
        self.hub_token = hub_token

    def can_extract(self, model_id: str) -> bool:
        try:
            return any(
                f.endswith(".safetensors")
                for f in list_repo_files(model_id, token=self.hub_token)
            )
        except Exception:
            return False

    def extract_metadata(self, model_id: str) -> Dict[str, Union[str, int, dict]]:
        try:
            info = fetch_safetensors_metadata(model_id, hf_token=self.hub_token)
            if info is None:
                return {}
            return st_map_to_metadata(info)
        except Exception as e:
            logger.warning(f"Safetensors extraction failed for {model_id}: {e}")
            return {}


def default_extractors(
    hub_token: str | bool | None = None,
) -> List[ModelFileExtractor]:
    return [
        SafetensorsFileExtractor(hub_token=hub_token),
        GGUFFileExtractor(hub_token=hub_token),
    ]


def extract_distribution_metadata(
    model_id: str, hf_api: Optional[HfApi] = None
) -> Dict[str, Union[str, int]]:
    """Extract format-agnostic distribution metadata for a model repo.

    Computes two attributes from the repository file listing (with sizes):
      * ``model_file_size``     — total size in bytes of all model weight files
      * ``runtime_requirement`` — ``python`` if the weights load from a Python ML
        stack (e.g. safetensors/PyTorch), else ``llm-stack`` if the only weights
        are GGUF/GGML (require llama.cpp/Ollama-style runtimes).

    Unlike the per-format ``ModelFileExtractor`` implementations, this runs for
    every repo regardless of weight format. Returns an empty dict on failure.
    """
    api = hf_api or HfApi(token=False)
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:
        logger.warning(f"Distribution metadata fetch failed for {model_id}: {e}")
        return {}

    total_size = 0
    has_python_weights = False
    has_llm_stack_weights = False

    for sibling in getattr(info, "siblings", None) or []:
        filename = getattr(sibling, "rfilename", "") or ""
        ext = os.path.splitext(filename.lower())[1]
        size = getattr(sibling, "size", None)

        if ext in PYTHON_WEIGHT_EXTENSIONS:
            has_python_weights = True
        elif ext in LLM_STACK_WEIGHT_EXTENSIONS:
            has_llm_stack_weights = True
        else:
            continue  # not a weight file — ignore for size + runtime classification

        if isinstance(size, int) and size > 0:
            total_size += size

    metadata: Dict[str, Union[str, int]] = {}
    if total_size > 0:
        metadata["model_file_size"] = total_size
    # A repo that ships Python-loadable weights can run in Python even if it also
    # ships GGUF; only GGUF/GGML-exclusive repos require an LLM stack.
    if has_python_weights:
        metadata["runtime_requirement"] = RUNTIME_PYTHON
    elif has_llm_stack_weights:
        metadata["runtime_requirement"] = RUNTIME_LLM_STACK

    return metadata
