"""
config.json parsing for HuggingFace model repositories.

Extracts hyperparameters using llama.cpp's find_hparam key fallback chains.
Works for any model format (safetensors, GGUF, pytorch, etc.) — the config.json
schema is format-agnostic.
"""
from typing import Dict, List, Optional, Union

# Exact key fallback order from llama.cpp convert_hf_to_gguf.py
# (see research.md section 12.4 for source line references)
HPARAM_KEYS: Dict[str, List[str]] = {
    "block_count": ["n_layers", "num_hidden_layers", "n_layer", "num_layers"],
    "context_length": ["max_position_embeddings", "n_ctx", "n_positions",
                        "max_length", "max_sequence_length", "model_max_length"],
    "embedding_length": ["hidden_size", "n_embd", "dim"],
    "feed_forward_length": ["intermediate_size", "n_inner", "hidden_dim"],
    "attention_head_count": ["num_attention_heads", "n_head", "n_heads"],
    "attention_head_count_kv": ["num_key_value_heads", "n_kv_heads"],
    "rope_dimension_count": ["rotary_dim", "rope_dim"],
    "vocab_size": ["vocab_size"],
    "architecture": ["model_type"],
}


ParsedConfig = Dict[str, Optional[Union[str, int]]]


def parse_config(config: dict) -> ParsedConfig:
    """Extract hyperparameters from config.json using llama.cpp's find_hparam key fallback chains.

    Handles VLM models that nest text params under text_config (llama.cpp L800-802).

    Returns a dict with canonical keys (block_count, embedding_length, etc.)
    and None for any fields not found in the config.
    """
    # VLM merge: text_config values override root, mirroring llama.cpp
    if "text_config" in config:
        merged = dict(config)
        merged.update(config["text_config"])
        config = merged

    result: ParsedConfig = {}
    for canonical_name, candidate_keys in HPARAM_KEYS.items():
        value = None
        for key in candidate_keys:
            if key in config:
                value = config[key]
                break
        result[canonical_name] = value

    return result
