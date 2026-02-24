"""
Unit tests for GGUF metadata extraction module.

Tests the public interface of the gguf_metadata module.
"""

import hashlib
import struct
import unittest
from datetime import datetime, timezone

from src.models.gguf_metadata import (
    GGUF_MAGIC,
    GGUFModelInfo,
    BufferUnderrunError,
    InvalidMagicError,
    HashValue,
    parse_gguf_metadata,
    extract_chat_template_info,
    extract_model_info,
    build_huggingface_url,
    map_to_metadata,
    _map_chat_template_fields,
)

from tests.fixtures import (
    build_gguf_bytes,
    build_gguf_model_info,
    SAMPLE_CHAT_TEMPLATE,
    get_sample_chat_template_hash,
)


class TestParseGGUFMetadata(unittest.TestCase):

    def test_parse_minimal_valid_gguf(self):
        data = build_gguf_bytes(architecture="llama", model_name="test-model")

        result = parse_gguf_metadata(data, filename="test.gguf")

        self.assertEqual(result.version, 3)
        self.assertEqual(result.tensor_count, 0)
        self.assertEqual(result.kv_count, 2)
        self.assertEqual(result.metadata["general.architecture"], "llama")
        self.assertEqual(result.metadata["general.name"], "test-model")
        self.assertEqual(result.filename, "test.gguf")

    def test_parse_with_chat_template(self):
        data = build_gguf_bytes(chat_template=SAMPLE_CHAT_TEMPLATE)

        result = parse_gguf_metadata(data)

        self.assertIn("tokenizer.chat_template", result.metadata)
        self.assertEqual(result.metadata["tokenizer.chat_template"], SAMPLE_CHAT_TEMPLATE)

    def test_parse_with_various_metadata_types(self):
        data = build_gguf_bytes(
            extra_kv={
                "test.uint32": ("uint32", 42),
                "test.int32": ("int32", -100),
                "test.float32": ("float32", 3.14),
                "test.bool_true": ("bool", True),
                "test.bool_false": ("bool", False),
                "test.string": ("string", "hello world"),
            }
        )

        result = parse_gguf_metadata(data)

        self.assertEqual(result.metadata["test.uint32"], 42)
        self.assertEqual(result.metadata["test.int32"], -100)
        self.assertAlmostEqual(result.metadata["test.float32"], 3.14, places=2)
        self.assertTrue(result.metadata["test.bool_true"])
        self.assertFalse(result.metadata["test.bool_false"])
        self.assertEqual(result.metadata["test.string"], "hello world")

    def test_parse_invalid_magic_raises(self):
        data = b"NOT_GGUF_DATA_HERE"

        with self.assertRaises(InvalidMagicError) as context:
            parse_gguf_metadata(data)

        self.assertIn("invalid magic", str(context.exception).lower())

    def test_parse_truncated_header_raises(self):
        data = struct.pack("<I", GGUF_MAGIC)

        with self.assertRaises(BufferUnderrunError) as context:
            parse_gguf_metadata(data)

        self.assertIsNotNone(context.exception.required_bytes)
        self.assertGreater(context.exception.required_bytes, len(data))

    def test_parse_truncated_metadata_raises(self):
        data = build_gguf_bytes(chat_template="a" * 1000)
        truncated = data[:50]

        with self.assertRaises(BufferUnderrunError) as context:
            parse_gguf_metadata(truncated)

        self.assertGreater(context.exception.required_bytes, len(truncated))

    def test_buffer_underrun_carries_required_bytes(self):
        data = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3)

        with self.assertRaises(BufferUnderrunError) as context:
            parse_gguf_metadata(data)

        self.assertIsNotNone(context.exception.required_bytes)
        self.assertGreater(context.exception.required_bytes, 8)


class TestExtractChatTemplateInfo(unittest.TestCase):

    def test_extract_default_template(self):
        metadata = {"tokenizer.chat_template": SAMPLE_CHAT_TEMPLATE}

        result = extract_chat_template_info(metadata)

        self.assertTrue(result.has_template)
        self.assertEqual(result.default_template, SAMPLE_CHAT_TEMPLATE)
        self.assertEqual(result.template_hash, get_sample_chat_template_hash())
        self.assertEqual(result.named_templates, {})
        self.assertEqual(result.template_names, [])

    def test_extract_named_templates(self):
        metadata = {
            "tokenizer.chat_template": "{{ default }}",
            "tokenizer.chat_templates": ["chatml", "plain"],
            "tokenizer.chat_template.chatml": "{{ chatml_template }}",
            "tokenizer.chat_template.plain": "{{ plain_template }}",
        }

        result = extract_chat_template_info(metadata)

        self.assertTrue(result.has_template)
        self.assertEqual(result.default_template, "{{ default }}")
        self.assertIn("chatml", result.named_templates)
        self.assertIn("plain", result.named_templates)
        self.assertEqual(result.named_templates["chatml"], "{{ chatml_template }}")
        self.assertEqual(result.named_templates["plain"], "{{ plain_template }}")

    def test_extract_fallback_named_templates(self):
        metadata = {
            "tokenizer.chat_template": "{{ default }}",
            "tokenizer.chat_template.tool_use": "{{ tool_use }}",
            "tokenizer.chat_template.rag": "{{ rag }}",
        }

        result = extract_chat_template_info(metadata)

        self.assertIn("tool_use", result.named_templates)
        self.assertIn("rag", result.named_templates)

    def test_no_template(self):
        metadata = {"general.architecture": "bert", "general.name": "bert-base"}

        result = extract_chat_template_info(metadata)

        self.assertFalse(result.has_template)
        self.assertIsNone(result.default_template)
        self.assertIsNone(result.template_hash)
        self.assertEqual(result.named_templates, {})

    def test_different_templates_different_hashes(self):
        template1 = "{{ message1 }}"
        template2 = "{{ message2 }}"

        result1 = extract_chat_template_info({"tokenizer.chat_template": template1})
        result2 = extract_chat_template_info({"tokenizer.chat_template": template2})

        self.assertNotEqual(result1.template_hash, result2.template_hash)


class TestExtractModelInfo(unittest.TestCase):

    def test_extract_full_model_info(self):
        data = build_gguf_bytes(
            architecture="llama",
            model_name="Llama-2-7B-Chat",
            chat_template=SAMPLE_CHAT_TEMPLATE,
            tokenizer_model="gpt2",
            context_length=4096,
            embedding_length=4096,
            block_count=32,
            quantization_version=2,
            file_type=7,
        )

        parsed = parse_gguf_metadata(data, filename="llama.gguf")
        result = extract_model_info(parsed)

        self.assertEqual(result.filename, "llama.gguf")
        self.assertEqual(result.architecture, "llama")
        self.assertEqual(result.name, "Llama-2-7B-Chat")
        self.assertEqual(result.tokenizer_model, "gpt2")
        self.assertEqual(result.context_length, 4096)
        self.assertEqual(result.embedding_length, 4096)
        self.assertEqual(result.block_count, 32)
        self.assertEqual(result.quantization_version, 2)
        self.assertEqual(result.file_type, 7)
        self.assertIsNotNone(result.chat_template)
        self.assertTrue(result.chat_template.has_template)
        self.assertEqual(result.chat_template.default_template, SAMPLE_CHAT_TEMPLATE)

    def test_extract_minimal_model_info(self):
        data = build_gguf_bytes(architecture="gpt2", model_name="GPT-2")

        parsed = parse_gguf_metadata(data)
        result = extract_model_info(parsed)

        self.assertEqual(result.architecture, "gpt2")
        self.assertEqual(result.name, "GPT-2")
        self.assertIsNotNone(result.chat_template)
        self.assertFalse(result.chat_template.has_template)


class TestBuildHuggingfaceUrl(unittest.TestCase):

    def test_basic_url_construction(self):
        url = build_huggingface_url("meta-llama/Llama-2-7b-chat-hf", "model.gguf")
        self.assertEqual(url, "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/model.gguf")

    def test_custom_revision(self):
        url = build_huggingface_url("owner/repo", "file.gguf", revision="v1.0")
        self.assertIn("resolve/v1.0/", url)

    def test_nested_filename(self):
        url = build_huggingface_url("owner/repo", "models/quantized/model.gguf")
        self.assertIn("models/quantized/model.gguf", url)

    def test_invalid_repo_id_raises(self):
        with self.assertRaises(ValueError):
            build_huggingface_url("invalid", "file.gguf")

        with self.assertRaises(ValueError):
            build_huggingface_url("", "file.gguf")

    def test_special_characters_encoded(self):
        url = build_huggingface_url("owner/repo-name", "model file.gguf")
        self.assertIn("model%20file.gguf", url)


class TestMapToMetadata(unittest.TestCase):

    def test_core_fields_mapped(self):
        info = build_gguf_model_info(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
        )

        result = map_to_metadata(info)

        self.assertEqual(result["model_type"], "llama")
        self.assertEqual(result["gguf_filename"], "model.gguf")
        self.assertNotIn("chat_template_hash", result)
        self.assertNotIn("extraction_provenance", result)

    def test_map_without_chat_template(self):
        info = build_gguf_model_info(architecture="bert", name="BERT Base", chat_template=None)

        result = map_to_metadata(info)

        self.assertNotIn("chat_template", result)
        self.assertNotIn("chat_template_hash", result)
        self.assertEqual(result["model_type"], "bert")

    def test_supplementary_fields_mapped(self):
        info = GGUFModelInfo(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
            description="A test language model",
            license="MIT",
            author="Test Author",
            context_length=4096,
            embedding_length=4096,
            block_count=32,
            attention_head_count=32,
            attention_head_count_kv=8,
            quantization_version=2,
            file_type=7,
        )

        result = map_to_metadata(info)

        self.assertEqual(result["description"], "A test language model")
        self.assertEqual(result["gguf_license"], "MIT")
        self.assertEqual(result["suppliedBy"], "Test Author")

        self.assertIn("quantization", result)
        self.assertEqual(result["quantization"]["version"], 2)
        self.assertEqual(result["quantization"]["file_type"], 7)

        self.assertIn("hyperparameter", result)
        self.assertEqual(result["hyperparameter"]["context_length"], 4096)
        self.assertEqual(result["hyperparameter"]["embedding_length"], 4096)
        self.assertEqual(result["hyperparameter"]["block_count"], 32)
        self.assertEqual(result["hyperparameter"]["attention_head_count"], 32)
        self.assertEqual(result["hyperparameter"]["attention_head_count_kv"], 8)
        self.assertNotIn("vocab_size", result["hyperparameter"])


class TestChatTemplateMapping(unittest.TestCase):

    def test_chat_template_hash_produced(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ message }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        self.assertTrue(result["chat_template_hash"].startswith("sha256:"))

    def test_template_content_requires_explicit_opt_in(self):
        template = "{{ message }}"
        info = build_gguf_model_info(filename="model.gguf", chat_template=template)

        result_default = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")
        self.assertNotIn("chat_template", result_default)

        result_opted = _map_chat_template_fields(info, "owner/repo", True, "2025-01-01T00:00:00Z")
        self.assertEqual(result_opted["chat_template"], template)

    def test_extraction_provenance_tracks_source(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ msg }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-06-15T12:00:00Z")

        prov = result["extraction_provenance"]
        self.assertEqual(prov["source_file"], "model.gguf")
        self.assertEqual(prov["source_type"], "gguf_embedded")
        self.assertEqual(prov["source_repository"], "https://huggingface.co/owner/repo")
        self.assertEqual(prov["extraction_timestamp"], "2025-06-15T12:00:00Z")

    def test_security_status_defaults_to_unscanned(self):
        info = build_gguf_model_info(chat_template="{{ msg }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        status = result["template_security_status"]
        self.assertEqual(status["status"], "unscanned")
        self.assertEqual(status["subject"]["type"], "chat_template")

    def test_no_fields_without_chat_template(self):
        info = build_gguf_model_info(chat_template=None)

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        self.assertEqual(result, {})

    def test_cdx_properties_produced(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ message }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        props = result.get("cdx_component_properties", [])
        prop_dict = {p["name"]: p["value"] for p in props}
        self.assertIn("aibom:chat_template_hash", prop_dict)

    def test_attestation_derived_from_security_status(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ msg }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        status = result["template_security_status"]
        attestation = result["cdx_attestation"]
        self.assertEqual(attestation["map"][0]["status"], status["status"])


class TestHashValue(unittest.TestCase):

    def test_from_content_creates_sha256_hash(self):
        content = "test content"
        h = HashValue.from_content(content)

        self.assertEqual(h.algorithm, "SHA-256")
        self.assertEqual(len(h.value), 64)
        self.assertEqual(h.value, hashlib.sha256(content.encode("utf-8")).hexdigest())

    def test_to_cyclonedx_produces_structured_format(self):
        h = HashValue(algorithm="SHA-256", value="abc123")

        cdx = h.to_cyclonedx()

        self.assertEqual(cdx, {"alg": "SHA-256", "content": "abc123"})

    def test_to_prefixed_produces_string_format(self):
        h = HashValue(algorithm="SHA-256", value="abc123def456")

        prefixed = h.to_prefixed()

        self.assertEqual(prefixed, "sha256:abc123def456")

    def test_roundtrip_content_to_both_formats(self):
        content = "{% for m in messages %}{{ m.content }}{% endfor %}"
        h = HashValue.from_content(content)

        prefixed = h.to_prefixed()
        cdx = h.to_cyclonedx()

        self.assertTrue(prefixed.startswith("sha256:"))
        self.assertEqual(cdx["alg"], "SHA-256")
        self.assertEqual(cdx["content"], prefixed.split(":")[1])


class TestStructuredHashesInMetadata(unittest.TestCase):

    def test_structured_hash_included_in_chat_template_info(self):
        template = "{{ message }}"
        data = build_gguf_bytes(chat_template=template)

        parsed = parse_gguf_metadata(data)
        ct_info = extract_chat_template_info(parsed.metadata)

        self.assertIsNotNone(ct_info.template_hash_structured)
        self.assertEqual(ct_info.template_hash_structured.algorithm, "SHA-256")
        self.assertEqual(ct_info.template_hash, ct_info.template_hash_structured.to_prefixed())

    def test_structured_hash_in_chat_template_fields(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ msg }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        self.assertIn("chat_template_hash_structured", result)
        h = result["chat_template_hash_structured"]
        self.assertEqual(h["alg"], "SHA-256")

    def test_cdx_component_hashes_array(self):
        info = build_gguf_model_info(filename="model.gguf", chat_template="{{ msg }}")

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        self.assertIn("cdx_component_hashes", result)
        hashes = result["cdx_component_hashes"]
        self.assertEqual(len(hashes), 1)
        self.assertEqual(hashes[0]["alg"], "SHA-256")

    def test_named_templates_have_structured_hashes(self):
        info = build_gguf_model_info(
            filename="model.gguf",
            chat_template="{{ default }}",
            named_templates={"tool_use": "{{ tool }}", "rag": "{{ rag }}"}
        )

        result = _map_chat_template_fields(info, "owner/repo", False, "2025-01-01T00:00:00Z")

        self.assertIn("named_chat_templates_structured", result)
        structured = result["named_chat_templates_structured"]
        self.assertIn("tool_use", structured)
        self.assertIn("rag", structured)
        self.assertEqual(structured["tool_use"]["alg"], "SHA-256")


if __name__ == '__main__':
    unittest.main()
