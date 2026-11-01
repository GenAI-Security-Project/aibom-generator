import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Protocol, Dict, List, Union, runtime_checkable

from huggingface_hub import HfApi, list_repo_files

from .gguf_metadata import (
    GGUFModelInfo,
    fetch_gguf_metadata_from_repo,
    map_to_metadata as gguf_map_to_metadata,
    summarize_chat_template_consistency,
)
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
_SHARD_INDEX_RE = re.compile(r"(\d{3,5})-of-\d{3,5}")

# Upper bound on GGUF files whose header is fetched to compare chat templates
# (each fetch reads the first few MB of the file, never the weights).
MAX_GGUF_VARIANTS_EXAMINED = 8
_GGUF_FETCH_WORKERS = 4


def select_gguf_variants(gguf_files: List[str]) -> List[str]:
    """One file per variant: later shards of a split GGUF carry no tokenizer metadata."""
    variants = []
    for name in gguf_files:
        match = _SHARD_INDEX_RE.search(os.path.basename(name))
        if match and int(match.group(1)) != 1:
            continue
        variants.append(name)
    return variants or list(gguf_files)


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

            variants = select_gguf_variants(gguf_files)
            model_info = fetch_gguf_metadata_from_repo(model_id, variants[0])
            if model_info is None:
                return {}

            metadata = gguf_map_to_metadata(model_info)
            examined = [model_info] + self._fetch_other_variants(
                model_id, variants[1:MAX_GGUF_VARIANTS_EXAMINED]
            )
            metadata.update(summarize_chat_template_consistency(examined, len(variants)))
            return metadata
        except Exception as e:
            logger.warning(f"GGUF extraction failed for {model_id}: {e}")
            return {}

    @staticmethod
    def _fetch_other_variants(model_id: str, filenames: List[str]) -> List[GGUFModelInfo]:
        """Best effort: a variant that cannot be read is skipped, not fatal."""
        if not filenames:
            return []

        def fetch(filename: str) -> Optional[GGUFModelInfo]:
            try:
                return fetch_gguf_metadata_from_repo(model_id, filename)
            except Exception as e:
                logger.warning(f"GGUF variant {filename} unreadable for {model_id}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=_GGUF_FETCH_WORKERS) as pool:
            return [info for info in pool.map(fetch, filenames) if info is not None]


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
    api = hf_api or HfApi()
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
