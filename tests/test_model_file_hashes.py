"""Tests for per-file model hashes (issue #138) and the consumer verification script."""

import hashlib
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import GatedRepoError

from src.models.model_file_hashes import (
    HASH_SOURCE_COMPUTED,
    HASH_SOURCE_LFS,
    HASH_SOURCE_UNKNOWN,
    collect_model_file_hashes,
    is_documentation_file,
)
from src.models.service import AIBOMService
from src.utils.validation import validate_aibom

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
CONFIG_BYTES = b'{"model_type": "qwen3"}'
TEMPLATE_BYTES = b"{{ messages }}"


def _sibling(path, size, lfs_sha=None, lfs=False):
    lfs_info = SimpleNamespace(sha256=lfs_sha, size=size) if (lfs or lfs_sha) else None
    return SimpleNamespace(rfilename=path, size=size, blob_id="0" * 40, lfs=lfs_info)


class FakeApi:
    def __init__(self, siblings=None, fail=False):
        self.siblings = siblings or []
        self.fail = fail
        self.calls = []
        self.token = None

    def model_info(self, repo_id, revision=None, files_metadata=False):
        self.calls.append({"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata})
        if self.fail:
            raise RuntimeError("network down")
        return SimpleNamespace(sha=revision, siblings=self.siblings)


def _gated_error():
    # GatedRepoError requires an HTTP response object whose type differs across
    # huggingface_hub versions; the code under test only checks the class.
    err = GatedRepoError.__new__(GatedRepoError)
    Exception.__init__(err, "401 gated")
    return err


class FakeDownloader:
    def __init__(self, tmp_path, contents, gated=False):
        self.tmp_path = tmp_path
        self.contents = contents
        self.gated = gated
        self.calls = []

    def __call__(self, repo_id, filename, revision=None, token=None):
        self.calls.append({"filename": filename, "revision": revision})
        if self.gated:
            raise _gated_error()
        if filename not in self.contents:
            raise RuntimeError("404")
        path = self.tmp_path / filename.replace("/", "_")
        path.write_bytes(self.contents[filename])
        return str(path)


def _standard_siblings():
    return [
        _sibling("README.md", 100),
        _sibling("LICENSE", 100),
        _sibling(".gitattributes", 100),
        _sibling("config.json", len(CONFIG_BYTES)),
        _sibling("chat_template.jinja", len(TEMPLATE_BYTES)),
        _sibling("model.safetensors", 1_000_000, lfs_sha=WEIGHTS_SHA),
    ]


class TestDocumentationFilter:
    @pytest.mark.parametrize("path", ["README.md", "LICENSE", "LICENSE.txt", ".gitattributes",
                                      "docs/figure.png", "NOTICE", "USE_POLICY.md"])
    def test_docs_are_excluded(self, path):
        assert is_documentation_file(path)

    @pytest.mark.parametrize("path", ["config.json", "tokenizer_config.json", "chat_template.jinja",
                                      "model.safetensors", "model-Q4_K_M.gguf", "modeling_custom.py",
                                      "tokenizer.model", "onnx/model.onnx"])
    def test_model_files_are_kept(self, path):
        assert not is_documentation_file(path)


class TestCollectModelFileHashes:
    def test_lfs_hash_from_api_and_non_lfs_computed_at_pinned_revision(self, tmp_path):
        api = FakeApi(_standard_siblings())
        download = FakeDownloader(tmp_path, {"config.json": CONFIG_BYTES, "chat_template.jinja": TEMPLATE_BYTES})

        files = collect_model_file_hashes("Qwen/Qwen3-0.6B", REVISION, hf_api=api, downloader=download)
        by_path = {f["path"]: f for f in files}

        assert set(by_path) == {"config.json", "chat_template.jinja", "model.safetensors"}
        assert by_path["model.safetensors"] == {
            "path": "model.safetensors", "size": 1_000_000, "sha256": WEIGHTS_SHA, "source": HASH_SOURCE_LFS,
        }
        assert by_path["config.json"]["sha256"] == hashlib.sha256(CONFIG_BYTES).hexdigest()
        assert by_path["config.json"]["source"] == HASH_SOURCE_COMPUTED
        assert by_path["chat_template.jinja"]["sha256"] == hashlib.sha256(TEMPLATE_BYTES).hexdigest()
        # The listing and every download use the pinned commit, never a branch.
        assert api.calls == [{"repo_id": "Qwen/Qwen3-0.6B", "revision": REVISION, "files_metadata": True}]
        assert {c["revision"] for c in download.calls} == {REVISION}
        # LFS files are never downloaded.
        assert "model.safetensors" not in {c["filename"] for c in download.calls}

    def test_no_revision_means_no_hashes(self, tmp_path):
        api = FakeApi(_standard_siblings())
        assert collect_model_file_hashes("o/m", None, hf_api=api, downloader=FakeDownloader(tmp_path, {})) == []
        assert api.calls == []

    def test_listing_failure_returns_empty(self, tmp_path):
        api = FakeApi(fail=True)
        assert collect_model_file_hashes("o/m", REVISION, hf_api=api, downloader=FakeDownloader(tmp_path, {})) == []

    def test_lfs_without_valid_sha_is_unknown(self, tmp_path):
        api = FakeApi([_sibling("model.safetensors", 10, lfs_sha="*" * 64)])
        files = collect_model_file_hashes("o/m", REVISION, hf_api=api, downloader=FakeDownloader(tmp_path, {}))
        assert files[0]["source"] == HASH_SOURCE_UNKNOWN
        assert files[0]["sha256"] is None
        assert "LFS" in files[0]["reason"]

    def test_failed_download_is_listed_as_unknown_not_omitted(self, tmp_path):
        api = FakeApi([_sibling("config.json", 10)])
        files = collect_model_file_hashes("o/m", REVISION, hf_api=api, downloader=FakeDownloader(tmp_path, {}))
        assert files == [{"path": "config.json", "size": 10, "sha256": None,
                          "source": HASH_SOURCE_UNKNOWN, "reason": "download failed"}]

    def test_oversized_non_lfs_file_is_not_downloaded(self, tmp_path):
        api = FakeApi([_sibling("big.bin", 2048)])
        download = FakeDownloader(tmp_path, {"big.bin": b"x"})
        files = collect_model_file_hashes("o/m", REVISION, hf_api=api, downloader=download, max_file_bytes=1024)
        assert files[0]["source"] == HASH_SOURCE_UNKNOWN
        assert download.calls == []

    def test_gated_repo_stops_further_downloads(self, tmp_path):
        api = FakeApi(_standard_siblings())
        download = FakeDownloader(tmp_path, {}, gated=True)
        files = collect_model_file_hashes("o/m", REVISION, hf_api=api, downloader=download)
        by_path = {f["path"]: f for f in files}
        assert len(download.calls) == 1
        assert by_path["config.json"]["source"] == HASH_SOURCE_UNKNOWN
        assert by_path["chat_template.jinja"]["source"] == HASH_SOURCE_UNKNOWN
        assert "gated" in by_path["chat_template.jinja"]["reason"]
        # LFS hashes still come from the API.
        assert by_path["model.safetensors"]["source"] == HASH_SOURCE_LFS

    def test_computed_file_cap(self, tmp_path):
        siblings = [_sibling(f"f{i}.json", 1) for i in range(3)]
        download = FakeDownloader(tmp_path, {f"f{i}.json": b"x" for i in range(3)})
        files = collect_model_file_hashes("o/m", REVISION, hf_api=FakeApi(siblings), downloader=download,
                                          max_computed_files=2)
        assert [f["source"] for f in files] == [HASH_SOURCE_COMPUTED, HASH_SOURCE_COMPUTED, HASH_SOURCE_UNKNOWN]


def _metadata_with_files():
    return {
        "commit": REVISION,
        "model_files": [
            {"path": "config.json", "size": 23, "sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
             "source": HASH_SOURCE_COMPUTED},
            {"path": "model.safetensors", "size": 1_000_000, "sha256": WEIGHTS_SHA, "source": HASH_SOURCE_LFS},
            {"path": "chat_template.jinja", "size": 14, "sha256": None, "source": HASH_SOURCE_UNKNOWN,
             "reason": "gated repository: no access with the token used"},
        ],
    }


class TestAibomFileComponents:
    def test_file_components_are_nested_under_the_model_and_schema_valid(self):
        service = AIBOMService(use_inference=False)
        aibom = service._create_aibom_structure("Qwen/Qwen3-0.6B", _metadata_with_files())

        valid, errors = validate_aibom(aibom)
        assert valid, errors

        model = aibom["components"][0]
        files = {c["name"]: c for c in model["components"]}
        assert set(files) == {"config.json", "model.safetensors", "chat_template.jinja"}

        weights = files["model.safetensors"]
        assert weights["type"] == "file"
        assert weights["bom-ref"] == "pkg:huggingface/Qwen/Qwen3-0.6B@c1899de2#model.safetensors"
        assert weights["hashes"] == [{"alg": "SHA-256", "content": WEIGHTS_SHA}]
        props = {p["name"]: p["value"] for p in weights["properties"]}
        assert props["owasp:aibom:file:hashSource"] == HASH_SOURCE_LFS
        assert props["owasp:aibom:file:size"] == "1000000"

    def test_unknown_hash_is_listed_with_reason(self):
        service = AIBOMService(use_inference=False)
        aibom = service._create_aibom_structure("Qwen/Qwen3-0.6B", _metadata_with_files())
        template = next(c for c in aibom["components"][0]["components"] if c["name"] == "chat_template.jinja")
        assert "hashes" not in template
        props = {p["name"]: p["value"] for p in template["properties"]}
        assert props["owasp:aibom:file:hashSource"] == HASH_SOURCE_UNKNOWN
        assert "gated" in props["owasp:aibom:file:hashUnavailableReason"]

    def test_no_file_components_without_commit(self):
        metadata = _metadata_with_files()
        metadata.pop("commit")
        aibom = AIBOMService(use_inference=False)._create_aibom_structure("o/m", metadata)
        assert "components" not in aibom["components"][0]

    def test_model_files_do_not_leak_into_model_card_properties(self):
        aibom = AIBOMService(use_inference=False)._create_aibom_structure("o/m", _metadata_with_files())
        card_props = aibom["components"][0].get("modelCard", {}).get("properties", [])
        assert all(p["name"] != "model_files" for p in card_props)


def _load_verify_script():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_model_files.py")
    spec = importlib.util.spec_from_file_location("verify_model_files", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConsumerVerification:
    @pytest.fixture
    def setup(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(CONFIG_BYTES)
        (model_dir / ".cache").mkdir()
        (model_dir / ".cache" / "ignored").write_bytes(b"x")
        aibom = AIBOMService(use_inference=False)._create_aibom_structure("o/m", _metadata_with_files())
        aibom_path = tmp_path / "aibom.json"
        aibom_path.write_text(json.dumps(aibom))
        return _load_verify_script(), aibom_path, model_dir

    def test_matching_files_pass(self, setup, capsys):
        verify, aibom_path, model_dir = setup
        assert verify.main([str(aibom_path), str(model_dir)]) == 0
        out = capsys.readouterr().out
        assert "MATCH       config.json" in out
        assert "declared but not present 2" in out

    def test_changed_file_fails(self, setup, capsys):
        verify, aibom_path, model_dir = setup
        (model_dir / "config.json").write_bytes(CONFIG_BYTES + b" ")
        assert verify.main([str(aibom_path), str(model_dir)]) == 1
        assert "MISMATCH    config.json" in capsys.readouterr().out

    def test_undeclared_file_fails(self, setup, capsys):
        verify, aibom_path, model_dir = setup
        (model_dir / "extra.jinja").write_bytes(b"{{ x }}")
        assert verify.main([str(aibom_path), str(model_dir)]) == 1
        assert "UNDECLARED  extra.jinja" in capsys.readouterr().out

    def test_file_without_recorded_hash_is_unverified(self, setup, capsys):
        verify, aibom_path, model_dir = setup
        (model_dir / "chat_template.jinja").write_bytes(TEMPLATE_BYTES)
        assert verify.main([str(aibom_path), str(model_dir)]) == 1
        assert "UNVERIFIED  chat_template.jinja" in capsys.readouterr().out

    def test_aibom_without_file_hashes(self, tmp_path, capsys):
        verify = _load_verify_script()
        aibom_path = tmp_path / "aibom.json"
        aibom_path.write_text(json.dumps({"components": [{"type": "machine-learning-model", "name": "m"}]}))
        assert verify.main([str(aibom_path), str(tmp_path)]) == 2


class TestExtractorWiring:
    def test_model_files_use_the_commit_from_model_info(self, monkeypatch, patch_extractor_io):
        from src.models import extractor as extractor_module
        from src.models.extractor import EnhancedExtractor

        seen = {}

        def fake_collect(model_id, revision, hf_api=None):
            seen["args"] = (model_id, revision)
            return [{"path": "config.json", "sha256": "0" * 64, "source": HASH_SOURCE_COMPUTED}]

        monkeypatch.setattr(extractor_module, "collect_model_file_hashes", fake_collect)
        monkeypatch.setattr(EnhancedExtractor, "_extract_distribution_metadata", lambda self, model_id: {})
        monkeypatch.setattr(EnhancedExtractor, "_extract_model_file_metadata", lambda self, model_id: {})

        info = SimpleNamespace(sha=REVISION, id="o/m", modelId="o/m", tags=[], pipeline_tag=None,
                               library_name=None, card_data=None, siblings=[])
        metadata = EnhancedExtractor(hf_api=None).extract_metadata("o/m", info, None)

        assert seen["args"] == ("o/m", REVISION)
        assert metadata["model_files"][0]["path"] == "config.json"
