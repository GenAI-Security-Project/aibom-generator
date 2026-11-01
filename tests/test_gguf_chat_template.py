"""Chat template integrity signal for GGUF models (#25), using synthetic GGUF bytes."""

import json

import pytest

from src.models import model_file_extractors
from src.models.gguf_metadata import (
    CHAT_TEMPLATE_KEY,
    CONSISTENCY_CONSISTENT,
    CONSISTENCY_INCONSISTENT,
    CONSISTENCY_NOT_COMPARABLE,
    GGUFModelInfo,
    MAX_EXPECTED_CHAT_TEMPLATE_CHARS,
    extract_model_info,
    fetch_gguf_metadata_from_url,
    hash_chat_template,
    map_to_metadata,
    parse_gguf_metadata,
    scan_chat_template,
    summarize_chat_template_consistency,
)
from src.models.model_file_extractors import (
    GGUFFileExtractor,
    MAX_GGUF_VARIANTS_EXAMINED,
    select_gguf_variants,
)
from src.models.scoring import calculate_completeness_score
from src.models.service import AIBOMService
from tests.fixtures import (
    SAMPLE_CHAT_TEMPLATE,
    build_gguf_bytes,
    get_sample_chat_template_hash,
)

OTHER_TEMPLATE = "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"


def _info(filename="model.gguf", **kwargs) -> GGUFModelInfo:
    data = build_gguf_bytes(architecture="llama", **kwargs)
    return extract_model_info(parse_gguf_metadata(data, filename=filename))


class TestChatTemplateExtraction:
    def test_template_and_hash_are_read_from_header_bytes(self):
        info = _info(chat_template=SAMPLE_CHAT_TEMPLATE)

        assert info.chat_template == SAMPLE_CHAT_TEMPLATE
        assert hash_chat_template(info.chat_template) == get_sample_chat_template_hash()

    def test_absent_or_blank_template_is_none(self):
        assert _info().chat_template is None
        assert _info(chat_template="   ").chat_template is None

    def test_named_templates_are_collected(self):
        info = _info(
            chat_template=SAMPLE_CHAT_TEMPLATE,
            extra_kv={f"{CHAT_TEMPLATE_KEY}.tool_use": ("string", OTHER_TEMPLATE)},
        )

        assert info.named_chat_templates == {"tool_use": OTHER_TEMPLATE}

    def test_identity_fields_are_still_extracted(self):
        info = _info(
            model_name="Llama-Chat",
            tokenizer_model="gpt2",
            quantization_version=2,
            chat_template=SAMPLE_CHAT_TEMPLATE,
        )

        assert (info.architecture, info.name) == ("llama", "Llama-Chat")
        assert (info.tokenizer_model, info.quantization_version) == ("gpt2", 2)


class TestChatTemplateMapping:
    def test_present_template_maps_hash_source_and_scan_result(self):
        metadata = map_to_metadata(
            _info(filename="m-q4.gguf", chat_template=SAMPLE_CHAT_TEMPLATE)
        )

        assert metadata["chat_template_present"] is True
        assert metadata["chat_template_hash"] == get_sample_chat_template_hash()
        assert metadata["chat_template_source"] == "gguf:m-q4.gguf#tokenizer.chat_template"
        assert metadata["chat_template_security_status"] == "no-indicators-found"
        assert "chat_template_security_indicators" not in metadata

    def test_missing_template_is_reported_as_absent(self):
        metadata = map_to_metadata(_info())

        assert metadata["chat_template_present"] is False
        for key in ("chat_template_hash", "chat_template_source", "chat_template_security_status"):
            assert key not in metadata

    def test_template_text_is_never_copied_into_metadata(self):
        metadata = map_to_metadata(_info(chat_template=SAMPLE_CHAT_TEMPLATE))

        assert SAMPLE_CHAT_TEMPLATE not in json.dumps(metadata, default=str)

    def test_suspicious_named_template_is_flagged(self):
        info = _info(
            chat_template=SAMPLE_CHAT_TEMPLATE,
            extra_kv={f"{CHAT_TEMPLATE_KEY}.tool_use": ("string", "{{ ''.__class__ }}")},
        )

        metadata = map_to_metadata(info)

        assert metadata["chat_template_security_status"] == "indicators-found"
        assert metadata["chat_template_security_indicators"] == "python-dunder-access"


class TestChatTemplateScan:
    def test_benign_template_has_no_indicators(self):
        assert scan_chat_template([SAMPLE_CHAT_TEMPLATE, OTHER_TEMPLATE]) == []

    @pytest.mark.parametrize(
        "template, indicator",
        [
            pytest.param("{{ ''.__class__.__mro__ }}", "python-dunder-access", id="dunder"),
            pytest.param("{{ lipsum.__globals__ }}", "jinja-introspection-global", id="lipsum"),
            pytest.param("Visit https://evil.example/payload first", "external-url", id="url"),
            pytest.param("note: " + "QUJD" * 30, "encoded-payload", id="base64"),
            pytest.param("hello\u200bworld", "hidden-characters", id="zero-width"),
            pytest.param("hello\u202eworld", "hidden-characters", id="bidi-override"),
            pytest.param(
                "x" * (MAX_EXPECTED_CHAT_TEMPLATE_CHARS + 1), "oversized-template", id="oversized"
            ),
        ],
    )
    def test_indicator_is_detected(self, template, indicator):
        assert indicator in scan_chat_template([template])

    def test_namespace_and_normal_whitespace_are_not_flagged(self):
        template = "{% set ns = namespace(first=true) %}\t{{ message['content'] }}\r\n"

        assert scan_chat_template([template]) == []

    def test_indicators_are_deduplicated_and_sorted(self):
        found = scan_chat_template(["http://a.example", "https://b.example __x__"])

        assert found == ["external-url", "python-dunder-access"]


class TestConsistencyAcrossVariants:
    def test_identical_templates_are_consistent(self):
        variants = [
            _info(filename="q4.gguf", chat_template=SAMPLE_CHAT_TEMPLATE),
            _info(filename="q8.gguf", chat_template=SAMPLE_CHAT_TEMPLATE),
        ]

        summary = summarize_chat_template_consistency(variants, total_variants=2)

        assert summary == {
            "chat_template_consistency": CONSISTENCY_CONSISTENT,
            "chat_template_variants_examined": 2,
            "chat_template_variants_total": 2,
        }

    def test_different_templates_are_inconsistent_and_hashes_are_listed(self):
        variants = [
            _info(filename="q4.gguf", chat_template=SAMPLE_CHAT_TEMPLATE),
            _info(filename="q8.gguf", chat_template=OTHER_TEMPLATE),
        ]

        summary = summarize_chat_template_consistency(variants)

        assert summary["chat_template_consistency"] == CONSISTENCY_INCONSISTENT
        assert json.loads(summary["chat_template_variant_hashes"]) == {
            "q4.gguf": hash_chat_template(SAMPLE_CHAT_TEMPLATE),
            "q8.gguf": hash_chat_template(OTHER_TEMPLATE),
        }

    def test_template_missing_from_one_variant_is_inconsistent(self):
        variants = [
            _info(filename="q4.gguf", chat_template=SAMPLE_CHAT_TEMPLATE),
            _info(filename="q8.gguf"),
        ]

        summary = summarize_chat_template_consistency(variants)

        assert summary["chat_template_consistency"] == CONSISTENCY_INCONSISTENT
        assert json.loads(summary["chat_template_variant_hashes"])["q8.gguf"] is None

    def test_no_template_in_any_variant_is_consistent(self):
        variants = [_info(filename="a.gguf"), _info(filename="b.gguf")]

        assert (
            summarize_chat_template_consistency(variants)["chat_template_consistency"]
            == CONSISTENCY_CONSISTENT
        )

    def test_single_variant_is_not_comparable_and_total_is_kept(self):
        summary = summarize_chat_template_consistency(
            [_info(chat_template=SAMPLE_CHAT_TEMPLATE)], total_variants=5
        )

        assert summary["chat_template_consistency"] == CONSISTENCY_NOT_COMPARABLE
        assert summary["chat_template_variants_examined"] == 1
        assert summary["chat_template_variants_total"] == 5
        assert "chat_template_variant_hashes" not in summary

    def test_no_variants_yields_nothing(self):
        assert summarize_chat_template_consistency([]) == {}


class TestSelectGgufVariants:
    def test_later_shards_are_dropped(self):
        files = [
            "m-Q4-00001-of-00003.gguf",
            "m-Q4-00002-of-00003.gguf",
            "m-Q4-00003-of-00003.gguf",
            "sub/m-Q8.gguf",
        ]

        assert select_gguf_variants(files) == ["m-Q4-00001-of-00003.gguf", "sub/m-Q8.gguf"]

    def test_falls_back_to_all_files_when_only_later_shards_exist(self):
        files = ["m-00002-of-00002.gguf"]

        assert select_gguf_variants(files) == files


class TestExtractorComparesVariants:
    def _patch(self, monkeypatch, files, templates, failing=()):
        monkeypatch.setattr(
            model_file_extractors, "list_repo_files", lambda model_id, **kw: ["README.md", *files]
        )
        fetched = []

        def fake_fetch(repo_id, filename, **kw):
            fetched.append(filename)
            if filename in failing:
                raise RuntimeError("boom")
            data = build_gguf_bytes(architecture="llama", chat_template=templates.get(filename))
            return extract_model_info(parse_gguf_metadata(data, filename=filename))

        monkeypatch.setattr(model_file_extractors, "fetch_gguf_metadata_from_repo", fake_fetch)
        return fetched

    def test_consistent_variants(self, monkeypatch):
        files = ["q4.gguf", "q8.gguf"]
        self._patch(monkeypatch, files, dict.fromkeys(files, SAMPLE_CHAT_TEMPLATE))

        metadata = GGUFFileExtractor().extract_metadata("owner/model")

        assert metadata["chat_template_consistency"] == "consistent"
        assert metadata["chat_template_hash"] == get_sample_chat_template_hash()
        assert metadata["chat_template_variants_examined"] == 2

    def test_tampered_variant_is_detected(self, monkeypatch):
        self._patch(
            monkeypatch,
            ["q4.gguf", "q8.gguf", "q2.gguf"],
            {"q4.gguf": SAMPLE_CHAT_TEMPLATE, "q8.gguf": SAMPLE_CHAT_TEMPLATE, "q2.gguf": OTHER_TEMPLATE},
        )

        metadata = GGUFFileExtractor().extract_metadata("owner/model")

        assert metadata["chat_template_consistency"] == "inconsistent"
        hashes = json.loads(metadata["chat_template_variant_hashes"])
        assert hashes["q2.gguf"] == hash_chat_template(OTHER_TEMPLATE)
        # primary metadata still comes from the first file
        assert metadata["chat_template_source"].startswith("gguf:q4.gguf#")

    def test_unreadable_variant_is_skipped_but_counted_in_total(self, monkeypatch):
        files = ["q4.gguf", "q8.gguf"]
        self._patch(monkeypatch, files, dict.fromkeys(files, SAMPLE_CHAT_TEMPLATE), failing={"q8.gguf"})

        metadata = GGUFFileExtractor().extract_metadata("owner/model")

        assert metadata["chat_template_variants_examined"] == 1
        assert metadata["chat_template_variants_total"] == 2
        assert metadata["chat_template_consistency"] == "not-comparable"

    def test_number_of_headers_fetched_is_bounded(self, monkeypatch):
        files = [f"m-{i}.gguf" for i in range(MAX_GGUF_VARIANTS_EXAMINED + 5)]
        fetched = self._patch(monkeypatch, files, {})

        metadata = GGUFFileExtractor().extract_metadata("owner/model")

        assert len(fetched) == MAX_GGUF_VARIANTS_EXAMINED
        assert metadata["chat_template_variants_examined"] == MAX_GGUF_VARIANTS_EXAMINED
        assert metadata["chat_template_variants_total"] == len(files)

    def test_unreadable_primary_still_returns_nothing(self, monkeypatch):
        self._patch(monkeypatch, ["q4.gguf"], {}, failing={"q4.gguf"})

        assert GGUFFileExtractor().extract_metadata("owner/model") == {}


class TestAibomOutput:
    def _properties(self, metadata):
        section = AIBOMService(hf_token="x")._create_model_card_section(metadata)
        return {p["name"]: p["value"] for p in section["properties"]}

    def test_signal_is_emitted_in_owasp_chat_template_namespace(self):
        metadata = map_to_metadata(_info(filename="q4.gguf", chat_template=SAMPLE_CHAT_TEMPLATE))
        metadata |= summarize_chat_template_consistency([_info(chat_template=SAMPLE_CHAT_TEMPLATE)], 3)

        props = self._properties(metadata)

        assert props["owasp:aibom:chatTemplate:present"] == "true"
        assert props["owasp:aibom:chatTemplate:hash"] == get_sample_chat_template_hash()
        assert props["owasp:aibom:chatTemplate:source"] == "gguf:q4.gguf#tokenizer.chat_template"
        assert props["owasp:aibom:chatTemplate:securityStatus"] == "no-indicators-found"
        assert props["owasp:aibom:chatTemplate:consistency"] == "not-comparable"
        assert props["owasp:aibom:chatTemplate:variantsExamined"] == "1"
        assert props["owasp:aibom:chatTemplate:variantsTotal"] == "3"
        assert not [name for name in props if name.startswith("chat_template")]

    def test_absent_template_is_emitted_as_false(self):
        props = self._properties(map_to_metadata(_info()))

        assert props["owasp:aibom:chatTemplate:present"] == "false"
        assert "owasp:aibom:chatTemplate:hash" not in props

    def test_non_gguf_model_has_no_chat_template_properties(self):
        section = AIBOMService(hf_token="x")._create_model_card_section({"name": "m"})

        assert not [p for p in section.get("properties", []) if "chatTemplate" in p["name"]]

    def test_registry_scores_emitted_fields_and_does_not_penalise_other_formats(self):
        service = AIBOMService(hf_token="x")
        gguf_meta = map_to_metadata(_info(chat_template=SAMPLE_CHAT_TEMPLATE))
        gguf_meta |= summarize_chat_template_consistency([_info(chat_template=SAMPLE_CHAT_TEMPLATE)] * 2)
        base = {"bomFormat": "CycloneDX", "components": [{"name": "m", "type": "machine-learning-model"}]}

        gguf_bom = {**base, "components": [
            {**base["components"][0], "modelCard": service._create_model_card_section(gguf_meta)}
        ]}
        plain_bom = {**base, "components": [
            {**base["components"][0], "modelCard": service._create_model_card_section({"name": "m"})}
        ]}
        gguf_result = calculate_completeness_score(gguf_bom, validate=False)
        plain_result = calculate_completeness_score(plain_bom, validate=False)

        for field in ("chat_template_hash", "chat_template_security_status", "chat_template_consistency"):
            assert gguf_result["field_checklist"][field].startswith("✔")
            assert plain_result["field_checklist"][field].startswith("✘")
            assert field not in plain_result["missing_fields"]["supplementary"]


class _Response:
    def __init__(self, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        return None


class TestHeaderOnlyRangeFetch:
    def test_only_the_header_bytes_are_requested_with_range(self, monkeypatch):
        import httpx

        data = build_gguf_bytes(architecture="llama", chat_template=SAMPLE_CHAT_TEMPLATE)
        # Pretend the file continues with (never requested) tensor data.
        remote = data + b"\0" * (4 * 1024 * 1024)
        ranges = []

        class StubClient:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def head(self, url, headers=None):
                return _Response()

            def get(self, url, headers=None):
                start, end = headers["Range"].removeprefix("bytes=").split("-")
                ranges.append((int(start), int(end)))
                return _Response(remote[int(start): int(end) + 1], 206)

        monkeypatch.setattr(httpx, "Client", StubClient)

        parsed = fetch_gguf_metadata_from_url(
            "https://huggingface.co/o/r/resolve/main/m.gguf", filename="m.gguf", initial_request_size=64
        )

        assert parsed.metadata[CHAT_TEMPLATE_KEY] == SAMPLE_CHAT_TEMPLATE
        assert parsed.version == 3 and parsed.tensor_count == 0
        assert ranges[0] == (0, 63)
        assert ranges[1][0] == 64
        assert max(end for _, end in ranges) < len(remote)
