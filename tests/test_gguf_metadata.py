"""
Unit tests for GGUF metadata extraction module.

Tests the public interface of the gguf_metadata module.
"""

import struct
import unittest

from src.models.gguf_metadata import (
    GGUF_MAGIC,
    GGUFModelInfo,
    BufferUnderrunError,
    InvalidMagicError,
    parse_gguf_metadata,
    extract_model_info,
    build_huggingface_url,
    map_to_metadata,
)

from tests.fixtures import (
    build_gguf_bytes,
    SAMPLE_CHAT_TEMPLATE,
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

    def test_extract_minimal_model_info(self):
        data = build_gguf_bytes(architecture="gpt2", model_name="GPT-2")

        parsed = parse_gguf_metadata(data)
        result = extract_model_info(parsed)

        self.assertEqual(result.architecture, "gpt2")
        self.assertEqual(result.name, "GPT-2")


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
        info = GGUFModelInfo(
            filename="model.gguf",
            architecture="llama",
            name="Test Model",
        )

        result = map_to_metadata(info)

        self.assertEqual(result["model_type"], "llama")
        self.assertEqual(result["gguf_filename"], "model.gguf")

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


if __name__ == '__main__':
    unittest.main()
