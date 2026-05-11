"""
Safetensors Metadata Extraction for AIBOM Generator

Extracts hyperparameters from safetensors repos by combining:
1. config.json — all hyperparameters (mirroring llama.cpp's find_hparam approach)
2. Safetensors headers — tensor info (parameter count, dtype distribution)

See research.md section 12 for full format specification and design rationale.
"""
import json
import math
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError

from .config_parsing import parse_config

logger = logging.getLogger(__name__)


@dataclass
class SafetensorsModelInfo:
    """Model information extracted from safetensors repo for AIBOM.

    Parallels GGUFModelInfo but sources hyperparameters from config.json
    (not from the safetensors header, which contains only tensor definitions).
    """
    # From config.json (via parse_config, same source as llama.cpp)
    architecture: Optional[str] = None
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    block_count: Optional[int] = None
    attention_head_count: Optional[int] = None
    attention_head_count_kv: Optional[int] = None
    feed_forward_length: Optional[int] = None
    rope_dimension_count: Optional[int] = None
    vocab_size: Optional[int] = None
    # From tokenizer_config.json
    tokenizer_class: Optional[str] = None
    # From safetensors headers (via get_safetensors_metadata)
    total_parameters: Optional[int] = None
    dtype_counts: Dict[str, int] = field(default_factory=dict)
    user_metadata: Dict[str, str] = field(default_factory=dict)


TensorInfoResult = Dict[str, Union[int, Dict[str, int]]]


def _extract_tensor_info(tensors: dict) -> TensorInfoResult:
    """Extract parameter count and dtype distribution from safetensors tensor metadata.

    tensors: dict mapping tensor name → object with .dtype and .shape attributes
    (from huggingface_hub's SafetensorsFileMetadata.tensors or compatible mock).
    """
    total_parameters = 0
    dtype_counter: Counter = Counter()

    for _, tensor in tensors.items():
        shape = tensor.shape
        param_count = math.prod(shape) if shape else 0
        total_parameters += param_count
        dtype_counter[tensor.dtype] += 1

    return {
        "total_parameters": total_parameters,
        "dtype_counts": dict(dtype_counter),
    }


MetadataValue = Union[str, int, Dict[str, int]]
MetadataDict = Dict[str, MetadataValue]


def map_to_metadata(info: SafetensorsModelInfo) -> MetadataDict:
    """Map SafetensorsModelInfo to the same dict format as gguf_metadata.map_to_metadata().

    Output structure mirrors GGUF: model_type, typeOfModel, vocab_size, context_length
    at top level; hyperparameter dict with non-None hyperparams; safetensors-specific fields.
    """
    metadata: MetadataDict = {}

    # Core fields (same as gguf_metadata._map_core_fields)
    if info.architecture is not None:
        metadata["model_type"] = info.architecture
        metadata["typeOfModel"] = info.architecture

    if info.vocab_size is not None:
        metadata["vocab_size"] = info.vocab_size

    if info.context_length is not None:
        metadata["context_length"] = info.context_length

    if info.tokenizer_class is not None:
        metadata["tokenizer_class"] = info.tokenizer_class

    # Hyperparameter dict (same as gguf_metadata._map_hyperparameters)
    hyperparams: Dict[str, int] = {}
    for field_name in (
        "context_length", "embedding_length", "block_count",
        "attention_head_count", "attention_head_count_kv",
        "feed_forward_length", "rope_dimension_count",
    ):
        value = getattr(info, field_name)
        if value is not None:
            hyperparams[field_name] = value

    if hyperparams:
        metadata["hyperparameter"] = hyperparams

    # Safetensors-specific
    if info.total_parameters is not None:
        metadata["safetensors_total_parameters"] = info.total_parameters

    return metadata


def fetch_safetensors_metadata(
    repo_id: str, *, hf_token: Optional[str] = None
) -> Optional[SafetensorsModelInfo]:
    """Fetch config.json + safetensors headers from a HuggingFace repo.

    Returns None if config.json is missing (can't extract hyperparameters).
    Returns partial info if safetensors headers are unavailable.
    """
    # Step 1: Fetch config.json (required)
    try:
        config_path = hf_hub_download(repo_id, "config.json", token=hf_token)
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.warning(f"Could not fetch config.json for {repo_id}: {e}")
        return None

    parsed = parse_config(config)

    info = SafetensorsModelInfo(
        architecture=parsed.get("architecture"),
        context_length=parsed.get("context_length"),
        embedding_length=parsed.get("embedding_length"),
        block_count=parsed.get("block_count"),
        attention_head_count=parsed.get("attention_head_count"),
        attention_head_count_kv=parsed.get("attention_head_count_kv"),
        feed_forward_length=parsed.get("feed_forward_length"),
        rope_dimension_count=parsed.get("rope_dimension_count"),
        vocab_size=parsed.get("vocab_size"),
    )

    # Step 2: Fetch tokenizer_config.json (optional — adds tokenizer_class)
    try:
        tok_path = hf_hub_download(repo_id, "tokenizer_config.json", token=hf_token)
        with open(tok_path) as f:
            tok_config = json.load(f)
        info.tokenizer_class = tok_config.get("tokenizer_class")
    except Exception:
        pass

    # Step 3: Fetch safetensors headers (optional — adds tensor info)
    try:
        api = HfApi()
        repo_meta = api.get_safetensors_metadata(repo_id, token=hf_token)

        # Aggregate tensors across all shard files
        all_tensors = {}
        for file_meta in repo_meta.files_metadata.values():
            all_tensors.update(file_meta.tensors)

        tensor_info = _extract_tensor_info(all_tensors)
        info.total_parameters = tensor_info["total_parameters"]
        info.dtype_counts = tensor_info["dtype_counts"]
    except Exception as e:
        logger.info(f"No safetensors metadata for {repo_id}: {e}")

    return info
