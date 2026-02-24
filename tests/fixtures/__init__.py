import hashlib
import os
import tempfile
from typing import Dict, Optional

from gguf import GGUFWriter

from src.models.gguf_metadata import (
    GGUFModelInfo,
    GGUFChatTemplateInfo,
    HashValue,
)


def _write_gguf_to_bytes(writer: GGUFWriter, path: str) -> bytes:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with open(path, "rb") as f:
        data = f.read()
    os.unlink(path)
    return data


_KV_TYPE_WRITERS = {
    "uint32": "add_uint32",
    "int32": "add_int32",
    "float32": "add_float32",
    "bool": "add_bool",
    "string": "add_string",
}


def build_gguf_bytes(
    *,
    architecture: str = "test-arch",
    model_name: str = "test-model",
    chat_template: Optional[str] = None,
    context_length: Optional[int] = None,
    embedding_length: Optional[int] = None,
    block_count: Optional[int] = None,
    head_count: Optional[int] = None,
    head_count_kv: Optional[int] = None,
    feed_forward_length: Optional[int] = None,
    rope_dimension_count: Optional[int] = None,
    quantization_version: Optional[int] = None,
    file_type: Optional[int] = None,
    tokenizer_model: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
    license: Optional[str] = None,
    extra_kv: Optional[Dict[str, tuple]] = None,
) -> bytes:
    """Build a valid GGUF binary using the canonical gguf package.

    extra_kv accepts arbitrary key-value pairs as:
        {"key": ("type_name", value)}
    where type_name is one of: uint32, int32, float32, bool, string.
    """
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name

    writer = GGUFWriter(path, architecture)
    writer.add_name(model_name)

    if chat_template is not None:
        writer.add_chat_template(chat_template)
    if context_length is not None:
        writer.add_context_length(context_length)
    if embedding_length is not None:
        writer.add_embedding_length(embedding_length)
    if block_count is not None:
        writer.add_block_count(block_count)
    if head_count is not None:
        writer.add_head_count(head_count)
    if head_count_kv is not None:
        writer.add_head_count_kv(head_count_kv)
    if feed_forward_length is not None:
        writer.add_feed_forward_length(feed_forward_length)
    if rope_dimension_count is not None:
        writer.add_rope_dimension_count(rope_dimension_count)
    if quantization_version is not None:
        writer.add_quantization_version(quantization_version)
    if file_type is not None:
        writer.add_file_type(file_type)
    if tokenizer_model is not None:
        writer.add_tokenizer_model(tokenizer_model)
    if description is not None:
        writer.add_description(description)
    if author is not None:
        writer.add_author(author)
    if license is not None:
        writer.add_license(license)

    if extra_kv:
        for key, (type_name, value) in extra_kv.items():
            method = _KV_TYPE_WRITERS.get(type_name)
            if method is None:
                raise ValueError(f"unsupported extra_kv type: {type_name}")
            getattr(writer, method)(key, value)

    return _write_gguf_to_bytes(writer, path)


def build_gguf_model_info(
    *,
    filename: str = "test.gguf",
    architecture: str = "llama",
    name: str = "Test Model",
    chat_template: Optional[str] = None,
    template_hash: Optional[str] = None,
    named_templates: Optional[Dict[str, str]] = None,
) -> GGUFModelInfo:
    ct_info = None
    if chat_template is not None:
        template_hash_structured = HashValue.from_content(chat_template)
        if template_hash is None:
            template_hash = template_hash_structured.to_prefixed()

        named_template_hashes = {}
        named_template_hashes_structured = {}
        if named_templates:
            for tname, tcontent in named_templates.items():
                h = HashValue.from_content(tcontent)
                named_template_hashes[tname] = h.to_prefixed()
                named_template_hashes_structured[tname] = h

        ct_info = GGUFChatTemplateInfo(
            has_template=True,
            default_template=chat_template,
            named_templates=named_templates or {},
            template_names=list(named_templates.keys()) if named_templates else [],
            template_hash=template_hash,
            template_hash_structured=template_hash_structured,
            named_template_hashes=named_template_hashes,
            named_template_hashes_structured=named_template_hashes_structured,
        )

    return GGUFModelInfo(
        filename=filename,
        architecture=architecture,
        name=name,
        chat_template=ct_info,
    )


SAMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{{ '<|' ~ message['role'] ~ '|>' ~ message['content'] }}\n"
    "{% endfor %}\n"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def get_sample_chat_template_hash() -> str:
    return f"sha256:{hashlib.sha256(SAMPLE_CHAT_TEMPLATE.encode('utf-8')).hexdigest()}"


def get_minimal_gguf_bytes() -> bytes:
    return build_gguf_bytes(architecture="llama", model_name="test-model")


def get_gguf_bytes_with_chat_template() -> bytes:
    return build_gguf_bytes(
        architecture="llama",
        model_name="test-model",
        chat_template=SAMPLE_CHAT_TEMPLATE,
    )


def get_full_gguf_bytes() -> bytes:
    return build_gguf_bytes(
        architecture="llama",
        model_name="Llama-2-7B-Chat",
        chat_template=SAMPLE_CHAT_TEMPLATE,
        tokenizer_model="gpt2",
        context_length=4096,
        embedding_length=4096,
        block_count=32,
        head_count=32,
        head_count_kv=8,
        quantization_version=2,
        file_type=7,
    )
