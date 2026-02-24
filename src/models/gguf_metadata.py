"""
GGUF Metadata Extraction for AIBOM Generator

This module extracts metadata from GGUF files without downloading the full file.
It uses HTTP range requests to fetch only the header portion (typically 2-8MB)
of potentially multi-GB model files.

GGUF files bundle model weights, tokenizer, and chat template together,
making them a critical attack surface for chat template poisoning.
"""

import struct
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, OrderedDict
from collections import OrderedDict as OrderedDictType
from datetime import datetime, timezone
from urllib.parse import quote

logger = logging.getLogger(__name__)

GGUF_MAGIC = 0x46554747

_STRUCT_UINT8 = struct.Struct("<B")
_STRUCT_INT8 = struct.Struct("<b")
_STRUCT_UINT16 = struct.Struct("<H")
_STRUCT_INT16 = struct.Struct("<h")
_STRUCT_UINT32 = struct.Struct("<I")
_STRUCT_INT32 = struct.Struct("<i")
_STRUCT_UINT64 = struct.Struct("<Q")
_STRUCT_INT64 = struct.Struct("<q")
_STRUCT_FLOAT32 = struct.Struct("<f")
_STRUCT_FLOAT64 = struct.Struct("<d")


class GGUFParseError(Exception):
    """Base exception for GGUF parsing errors."""
    pass


class BufferUnderrunError(GGUFParseError):
    """Raised when buffer doesn't contain enough data to parse."""
    def __init__(self, message: str = "buffer underrun", *, required_bytes: Optional[int] = None):
        super().__init__(message)
        self.required_bytes = required_bytes


class InvalidMagicError(GGUFParseError):
    """Raised when file doesn't have valid GGUF magic number."""
    pass


class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class HashValue:
    """Hash value in both CycloneDX structured and prefixed string formats."""
    algorithm: str
    value: str

    def to_cyclonedx(self) -> Dict[str, str]:
        """CycloneDX 1.6 structured format for component.hashes[]."""
        return {"alg": self.algorithm, "content": self.value}

    def to_prefixed(self) -> str:
        """Prefixed string format (e.g., 'sha256:abc123...')."""
        return f"{self.algorithm.lower().replace('-', '')}:{self.value}"

    @classmethod
    def from_content(cls, content: str, algorithm: str = "SHA-256") -> "HashValue":
        """Create hash from content string."""
        algo_for_hashlib = algorithm.lower().replace("-", "")
        hasher = getattr(hashlib, algo_for_hashlib, None)
        if not hasher:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        digest = hasher(content.encode("utf-8")).hexdigest()
        return cls(algorithm=algorithm.upper(), value=digest)


@dataclass
class GGUFMetadata:
    """Parsed GGUF file metadata."""
    version: int
    tensor_count: int
    kv_count: int
    metadata: Dict[str, Any]
    header_length: int
    filename: str = ""


@dataclass
class GGUFChatTemplateInfo:
    """Chat template information extracted from GGUF file."""
    has_template: bool
    default_template: Optional[str]
    named_templates: Dict[str, str]
    template_names: List[str]
    template_hash: Optional[str] = None
    template_hash_structured: Optional[HashValue] = None
    named_template_hashes: Dict[str, str] = field(default_factory=dict)
    named_template_hashes_structured: Dict[str, HashValue] = field(default_factory=dict)


@dataclass
class GGUFModelInfo:
    """Model information extracted from GGUF metadata for AIBOM."""
    filename: str
    architecture: Optional[str] = None
    name: Optional[str] = None
    quantization_version: Optional[int] = None
    file_type: Optional[int] = None
    tokenizer_model: Optional[str] = None
    vocab_size: Optional[int] = None
    chat_template: Optional[GGUFChatTemplateInfo] = None
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    block_count: Optional[int] = None
    attention_head_count: Optional[int] = None
    attention_head_count_kv: Optional[int] = None
    feed_forward_length: Optional[int] = None
    rope_dimension_count: Optional[int] = None
    description: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class _ByteReader:
    """Helper for reading structured binary data from a buffer."""

    __slots__ = ("_view", "_offset")

    def __init__(self, buffer: bytes) -> None:
        self._view = memoryview(buffer)
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def _require(self, size: int) -> None:
        if self._offset + size > len(self._view):
            raise BufferUnderrunError(
                f"need {size} bytes at offset {self._offset}, but only {len(self._view) - self._offset} available",
                required_bytes=self._offset + size
            )

    def read(self, size: int) -> memoryview:
        self._require(size)
        start = self._offset
        self._offset += size
        return self._view[start:self._offset]

    def read_uint8(self) -> int:
        return _STRUCT_UINT8.unpack_from(self.read(_STRUCT_UINT8.size))[0]

    def read_int8(self) -> int:
        return _STRUCT_INT8.unpack_from(self.read(_STRUCT_INT8.size))[0]

    def read_uint16(self) -> int:
        return _STRUCT_UINT16.unpack_from(self.read(_STRUCT_UINT16.size))[0]

    def read_int16(self) -> int:
        return _STRUCT_INT16.unpack_from(self.read(_STRUCT_INT16.size))[0]

    def read_uint32(self) -> int:
        return _STRUCT_UINT32.unpack_from(self.read(_STRUCT_UINT32.size))[0]

    def read_int32(self) -> int:
        return _STRUCT_INT32.unpack_from(self.read(_STRUCT_INT32.size))[0]

    def read_uint64(self) -> int:
        return _STRUCT_UINT64.unpack_from(self.read(_STRUCT_UINT64.size))[0]

    def read_int64(self) -> int:
        return _STRUCT_INT64.unpack_from(self.read(_STRUCT_INT64.size))[0]

    def read_float32(self) -> float:
        return _STRUCT_FLOAT32.unpack_from(self.read(_STRUCT_FLOAT32.size))[0]

    def read_float64(self) -> float:
        return _STRUCT_FLOAT64.unpack_from(self.read(_STRUCT_FLOAT64.size))[0]

    def read_bool(self) -> bool:
        return self.read_uint8() != 0

    def read_string(self) -> str:
        length = self.read_uint64()
        if length > 10_000_000:
            raise GGUFParseError(f"string length {length} exceeds sanity limit")
        raw = self.read(length)
        return raw.tobytes().decode("utf-8")


def _read_value(reader: _ByteReader, value_type: int) -> Any:
    """Parse a GGUF metadata value based on its type."""
    if value_type == GGUFValueType.UINT8:
        return reader.read_uint8()
    elif value_type == GGUFValueType.INT8:
        return reader.read_int8()
    elif value_type == GGUFValueType.UINT16:
        return reader.read_uint16()
    elif value_type == GGUFValueType.INT16:
        return reader.read_int16()
    elif value_type == GGUFValueType.UINT32:
        return reader.read_uint32()
    elif value_type == GGUFValueType.INT32:
        return reader.read_int32()
    elif value_type == GGUFValueType.UINT64:
        return reader.read_uint64()
    elif value_type == GGUFValueType.INT64:
        return reader.read_int64()
    elif value_type == GGUFValueType.FLOAT32:
        return reader.read_float32()
    elif value_type == GGUFValueType.FLOAT64:
        return reader.read_float64()
    elif value_type == GGUFValueType.BOOL:
        return reader.read_bool()
    elif value_type == GGUFValueType.STRING:
        return reader.read_string()
    elif value_type == GGUFValueType.ARRAY:
        element_type = reader.read_uint32()
        count = reader.read_uint64()
        if count > 1_000_000:
            raise GGUFParseError(f"array count {count} exceeds sanity limit")
        return [_read_value(reader, element_type) for _ in range(count)]
    else:
        raise GGUFParseError(f"unknown GGUF value type: {value_type}")


def parse_gguf_metadata(buffer: bytes, filename: str = "") -> GGUFMetadata:
    """Parse GGUF metadata from a byte buffer."""
    reader = _ByteReader(buffer)

    magic = reader.read_uint32()
    if magic != GGUF_MAGIC:
        raise InvalidMagicError(f"invalid magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}")

    version = reader.read_uint32()
    tensor_count = reader.read_uint64()
    kv_count = reader.read_uint64()

    if kv_count > 100_000:
        raise GGUFParseError(f"kv_count {kv_count} exceeds sanity limit")

    metadata: OrderedDictType[str, Any] = OrderedDictType()

    for _ in range(kv_count):
        key = reader.read_string()
        value_type = reader.read_uint32()
        value = _read_value(reader, value_type)
        metadata[key] = value

    return GGUFMetadata(
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        metadata=metadata,
        header_length=reader.offset,
        filename=filename
    )


def extract_chat_template_info(metadata: Dict[str, Any]) -> GGUFChatTemplateInfo:
    """Extract chat template information from GGUF metadata."""
    default_template = metadata.get("tokenizer.chat_template")

    template_names_raw = metadata.get("tokenizer.chat_templates", [])
    template_names: List[str] = []

    if isinstance(template_names_raw, (list, tuple)):
        for entry in template_names_raw:
            if isinstance(entry, str):
                template_names.append(entry)

    named_templates: Dict[str, str] = {}
    prefix = "tokenizer.chat_template."

    for name in template_names:
        key = prefix + name
        value = metadata.get(key)
        if isinstance(value, str):
            named_templates[name] = value

    if not template_names:
        for key, value in metadata.items():
            if key.startswith(prefix) and isinstance(value, str):
                suffix = key[len(prefix):]
                if suffix and suffix not in named_templates:
                    named_templates[suffix] = value
                    template_names.append(suffix)

    has_template = bool(default_template or named_templates)

    template_hash = None
    template_hash_structured = None
    if default_template:
        template_hash_structured = HashValue.from_content(default_template)
        template_hash = template_hash_structured.to_prefixed()

    named_template_hashes = {}
    named_template_hashes_structured = {}
    for name, content in named_templates.items():
        hash_obj = HashValue.from_content(content)
        named_template_hashes[name] = hash_obj.to_prefixed()
        named_template_hashes_structured[name] = hash_obj

    return GGUFChatTemplateInfo(
        has_template=has_template,
        default_template=default_template,
        named_templates=named_templates,
        template_names=template_names,
        template_hash=template_hash,
        template_hash_structured=template_hash_structured,
        named_template_hashes=named_template_hashes,
        named_template_hashes_structured=named_template_hashes_structured,
    )


def extract_model_info(gguf_metadata: GGUFMetadata) -> GGUFModelInfo:
    """Extract AIBOM-relevant model information from GGUF metadata."""
    meta = gguf_metadata.metadata
    chat_template = extract_chat_template_info(meta)
    arch = meta.get("general.architecture", "")

    def get_arch_key(suffix: str) -> Optional[Any]:
        if arch:
            val = meta.get(f"{arch}.{suffix}")
            if val is not None:
                return val
        return None

    return GGUFModelInfo(
        filename=gguf_metadata.filename,
        architecture=arch or None,
        name=meta.get("general.name"),
        quantization_version=meta.get("general.quantization_version"),
        file_type=meta.get("general.file_type"),
        tokenizer_model=meta.get("tokenizer.ggml.model"),
        vocab_size=len(meta.get("tokenizer.ggml.tokens", [])) or None,
        chat_template=chat_template,
        context_length=get_arch_key("context_length"),
        embedding_length=get_arch_key("embedding_length"),
        block_count=get_arch_key("block_count"),
        attention_head_count=get_arch_key("attention.head_count"),
        attention_head_count_kv=get_arch_key("attention.head_count_kv"),
        feed_forward_length=get_arch_key("feed_forward_length"),
        rope_dimension_count=get_arch_key("rope.dimension_count"),
        description=meta.get("general.description"),
        license=meta.get("general.license"),
        author=meta.get("general.author"),
        raw_metadata=dict(meta)
    )


def build_huggingface_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Build a HuggingFace download URL for a file."""
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must be in format 'owner/repo'")

    owner, repo = repo_id.split("/", 1)
    owner_quoted = quote(owner, safe="-_.~")
    repo_quoted = quote(repo, safe="-_.~")
    revision_quoted = quote(revision, safe="-_.~")
    filename_quoted = "/".join(quote(part, safe="-_.~/") for part in filename.split("/"))

    return f"https://huggingface.co/{owner_quoted}/{repo_quoted}/resolve/{revision_quoted}/{filename_quoted}"


def fetch_gguf_metadata_from_url(
    url: str,
    filename: str = "",
    *,
    hf_token: Optional[str] = None,
    initial_request_size: int = 8 * 1024 * 1024,
    max_request_size: int = 64 * 1024 * 1024,
    timeout: float = 60.0,
) -> GGUFMetadata:
    """Fetch and parse GGUF metadata from a URL using HTTP range requests."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required for remote GGUF fetching. Install with: pip install httpx")

    headers = {
        "User-Agent": "OWASP-AIBOM-Generator/1.0",
        "Accept": "application/octet-stream",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        current_url = url
        for _ in range(5):
            response = client.head(current_url, headers=headers)
            if response.status_code in (301, 302, 303, 307, 308):
                current_url = response.headers.get("location", current_url)
                logger.debug(f"Redirecting to: {current_url}")
            else:
                break
        actual_url = current_url

    buffer = bytearray()
    request_size = initial_request_size

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        range_header = f"bytes=0-{request_size - 1}"
        request_headers = {**headers, "Range": range_header}

        logger.info(f"Fetching first {request_size // (1024*1024)}MB of GGUF metadata...")
        response = client.get(actual_url, headers=request_headers)
        response.raise_for_status()
        buffer.extend(response.content)

        max_retries = 5
        for retry in range(max_retries):
            try:
                return parse_gguf_metadata(bytes(buffer), filename)
            except BufferUnderrunError as exc:
                if retry >= max_retries - 1:
                    raise

                if exc.required_bytes:
                    needed = exc.required_bytes + 1024
                else:
                    needed = len(buffer) * 2

                additional_size = min(needed - len(buffer), max_request_size - len(buffer))

                if additional_size <= 0 or len(buffer) >= max_request_size:
                    raise GGUFParseError(f"unable to parse metadata within {max_request_size} bytes")

                logger.info(f"Need more data (retry {retry + 1}), fetching additional {additional_size // 1024}KB...")

                range_header = f"bytes={len(buffer)}-{len(buffer) + additional_size - 1}"
                request_headers = {**headers, "Range": range_header}
                response = client.get(actual_url, headers=request_headers)
                response.raise_for_status()
                buffer.extend(response.content)
                logger.info(f"Buffer now {len(buffer) // 1024}KB")


def fetch_gguf_metadata_from_repo(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    hf_token: Optional[str] = None,
    **kwargs
) -> GGUFModelInfo:
    """Fetch and extract AIBOM-relevant metadata from a GGUF file in a HuggingFace repo."""
    url = build_huggingface_url(repo_id, filename, revision)
    logger.info(f"Fetching GGUF metadata from {repo_id}/{filename}")

    gguf_metadata = fetch_gguf_metadata_from_url(
        url,
        filename=filename,
        hf_token=hf_token,
        **kwargs
    )

    return extract_model_info(gguf_metadata)


def list_gguf_files(repo_id: str, hf_token: Optional[str] = None) -> List[str]:
    """List GGUF files in a HuggingFace repository."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, token=hf_token)
    return [f for f in files if f.endswith('.gguf')]


def extract_all_gguf_metadata(
    repo_id: str,
    *,
    hf_token: Optional[str] = None,
    **kwargs
) -> List[GGUFModelInfo]:
    """Extract metadata from all GGUF files in a repository."""
    gguf_files = list_gguf_files(repo_id, hf_token)

    if not gguf_files:
        logger.debug(f"No GGUF files found in {repo_id}")
        return []

    logger.info(f"Found {len(gguf_files)} GGUF files in {repo_id}")

    results = []
    for filename in gguf_files:
        try:
            info = fetch_gguf_metadata_from_repo(
                repo_id,
                filename,
                hf_token=hf_token,
                **kwargs
            )
            results.append(info)
            logger.info(f"  {filename}: architecture={info.architecture}, has_chat_template={info.chat_template.has_template if info.chat_template else False}")
        except Exception as e:
            logger.warning(f"  {filename}: failed to extract metadata: {e}")

    return results


AIBOM_NAMESPACE = "aibom"


def _map_core_fields(gguf_info: GGUFModelInfo) -> Dict[str, Any]:
    """Map basic model identity and tokenizer fields."""
    metadata = {}

    if gguf_info.architecture:
        metadata["model_type"] = gguf_info.architecture
        metadata["typeOfModel"] = gguf_info.architecture

    if gguf_info.name:
        metadata["name"] = gguf_info.name

    if gguf_info.tokenizer_model:
        metadata["tokenizer_class"] = gguf_info.tokenizer_model

    if gguf_info.vocab_size:
        metadata["vocab_size"] = gguf_info.vocab_size

    if gguf_info.context_length:
        metadata["context_length"] = gguf_info.context_length

    metadata["gguf_filename"] = gguf_info.filename

    return metadata


def _map_supplementary_fields(gguf_info: GGUFModelInfo) -> Dict[str, Any]:
    """Map optional descriptive fields from GGUF."""
    metadata = {}

    if gguf_info.description:
        metadata["description"] = gguf_info.description

    if gguf_info.author:
        metadata["suppliedBy"] = gguf_info.author

    if gguf_info.license:
        metadata["gguf_license"] = gguf_info.license

    return metadata


def _map_quantization(gguf_info: GGUFModelInfo) -> Dict[str, Any]:
    """Map quantization metadata."""
    quantization = {}

    if gguf_info.quantization_version:
        quantization["version"] = gguf_info.quantization_version
    if gguf_info.file_type:
        quantization["file_type"] = gguf_info.file_type

    return {"quantization": quantization} if quantization else {}


def _map_hyperparameters(gguf_info: GGUFModelInfo) -> Dict[str, Any]:
    """Map inference-shape hyperparameters."""
    hyperparams = {}

    if gguf_info.context_length:
        hyperparams["context_length"] = gguf_info.context_length
    if gguf_info.embedding_length:
        hyperparams["embedding_length"] = gguf_info.embedding_length
    if gguf_info.block_count:
        hyperparams["block_count"] = gguf_info.block_count
    if gguf_info.attention_head_count:
        hyperparams["attention_head_count"] = gguf_info.attention_head_count
    if gguf_info.attention_head_count_kv:
        hyperparams["attention_head_count_kv"] = gguf_info.attention_head_count_kv
    if gguf_info.feed_forward_length:
        hyperparams["feed_forward_length"] = gguf_info.feed_forward_length
    if gguf_info.rope_dimension_count:
        hyperparams["rope_dimension_count"] = gguf_info.rope_dimension_count

    return {"hyperparameter": hyperparams} if hyperparams else {}


def _build_security_status(template_hash: str, template_hash_structured: Optional[HashValue] = None) -> Dict[str, Any]:
    """Build the canonical security status structure."""
    subject = {
        "type": "chat_template",
        "hash": template_hash,
    }
    if template_hash_structured:
        subject["hash_structured"] = template_hash_structured.to_cyclonedx()

    return {
        "subject": subject,
        "status": "unscanned",
        "scanner_name": None,
        "scanner_version": None,
        "scan_timestamp": None,
        "report_uri": None,
        "findings": [],
    }


def _security_status_to_cdx_attestation(security_status: Dict[str, Any]) -> Dict[str, Any]:
    """Derive CycloneDX attestation from canonical security status."""
    return {
        "assessor": security_status["scanner_name"],
        "map": [
            {
                "requirement": "chat_template_integrity",
                "claims": [security_status["subject"]["hash"]],
                "status": security_status["status"],
            }
        ],
        "signature": None,
    }


def _map_chat_template_fields(
    gguf_info: GGUFModelInfo,
    model_id: str,
    include_template_content: bool,
    extraction_timestamp: str,
) -> Dict[str, Any]:
    """Map chat template and its provenance/security metadata."""
    if not gguf_info.chat_template or not gguf_info.chat_template.has_template:
        return {}

    ct = gguf_info.chat_template
    metadata = {}

    metadata["chat_template_hash"] = ct.template_hash
    if ct.template_hash_structured:
        metadata["chat_template_hash_structured"] = ct.template_hash_structured.to_cyclonedx()

    if include_template_content and ct.default_template:
        metadata["chat_template"] = ct.default_template

    metadata["extraction_provenance"] = {
        "source_file": gguf_info.filename,
        "source_repository": f"https://huggingface.co/{model_id}",
        "source_type": "gguf_embedded",
        "extraction_timestamp": extraction_timestamp,
        "extractor_tool": "aibom-generator",
    }

    metadata["model_lineage"] = {
        "inherited_from_base": False,
        "base_model": None,
        "derivation_method": None,
    }

    security_status = _build_security_status(ct.template_hash, ct.template_hash_structured)
    metadata["template_security_status"] = security_status
    metadata["cdx_attestation"] = _security_status_to_cdx_attestation(security_status)

    if ct.named_template_hashes:
        metadata["named_chat_templates"] = ct.named_template_hashes
    if ct.named_template_hashes_structured:
        metadata["named_chat_templates_structured"] = {
            name: h.to_cyclonedx() for name, h in ct.named_template_hashes_structured.items()
        }

    metadata["cdx_component_properties"] = [
        {"name": f"{AIBOM_NAMESPACE}:chat_template_hash", "value": ct.template_hash},
        {"name": f"{AIBOM_NAMESPACE}:template_source_type", "value": "gguf_embedded"},
        {"name": f"{AIBOM_NAMESPACE}:template_source_file", "value": gguf_info.filename},
    ]
    if include_template_content and ct.default_template:
        metadata["cdx_component_properties"].append(
            {"name": f"{AIBOM_NAMESPACE}:chat_template", "value": ct.default_template}
        )

    if ct.template_hash_structured:
        metadata["cdx_component_hashes"] = [ct.template_hash_structured.to_cyclonedx()]

    return metadata


def map_to_metadata(gguf_info: GGUFModelInfo) -> Dict[str, Any]:
    metadata = _map_core_fields(gguf_info)
    metadata |= _map_supplementary_fields(gguf_info)
    metadata |= _map_quantization(gguf_info)
    metadata |= _map_hyperparameters(gguf_info)
    # TODO: add chat template field mapping
    return metadata
