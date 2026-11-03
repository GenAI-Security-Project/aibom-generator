import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

from src.models.extractor import EnhancedExtractor
from src.models.service import AIBOMService
from src.utils.validation import validate_aibom

REPO_ROOT = Path(__file__).resolve().parent.parent

README = "# Model\n\nLimitations: may hallucinate facts. Known issues: slow on CPU.\n"

ORDER_SNIPPET = """
import logging
logging.disable(logging.CRITICAL)
from src.models.extractor import EnhancedExtractor as E
e = E(hf_api=object())
r = 'Model type: bert\\nArchitecture: roberta\\nLimitations: slow on CPU. Known issues: drifts. Safety: low risk.'
print(e._try_text_pattern_extraction('model_type', {'readme_content': r}))
print(e._try_text_pattern_extraction('technicalLimitations', {'readme_content': r}))
"""


def _extract_limitations(readme):
    extractor = EnhancedExtractor(hf_api=object())
    context = {
        "model_id": "acme/foo",
        "model_info": None,
        "model_card": None,
        "readme_content": readme,
        "config_data": None,
        "tokenizer_config": None,
    }
    return extractor._extract_registry_field(
        "technicalLimitations", extractor.registry_fields["technicalLimitations"], context
    )


def _build_aibom(metadata):
    service = AIBOMService.__new__(AIBOMService)
    service.model_file_extractors = []
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return service._create_aibom_structure("acme/foo", {"commit": "a" * 40, **metadata})
    finally:
        logging.disable(previous)


class TestLimitationsFlattening(unittest.TestCase):
    def test_multiple_matches_yield_flat_schema_valid_list(self):
        value = _extract_limitations(README)
        self.assertEqual(value, ["may hallucinate facts", "slow on CPU"])

        aibom = _build_aibom({"technicalLimitations": value})
        considerations = aibom["components"][0]["modelCard"]["considerations"]
        self.assertEqual(
            considerations["technicalLimitations"], ["may hallucinate facts", "slow on CPU"]
        )
        is_valid, errors = validate_aibom(aibom)
        self.assertTrue(is_valid, errors)

    def test_single_string_is_still_wrapped(self):
        aibom = _build_aibom({"technicalLimitations": "slow on CPU"})
        considerations = aibom["components"][0]["modelCard"]["considerations"]
        self.assertEqual(considerations["technicalLimitations"], ["slow on CPU"])
        self.assertTrue(validate_aibom(aibom)[0])

    def test_list_values_are_flattened_for_other_considerations(self):
        aibom = _build_aibom({
            "intendedUse": ["chat", "summarisation"],
            "ethicalConsiderations": ["bias", "privacy"],
        })
        considerations = aibom["components"][0]["modelCard"]["considerations"]
        self.assertEqual(considerations["useCases"], ["chat", "summarisation"])
        self.assertEqual(
            considerations["ethicalConsiderations"], [{"name": "bias"}, {"name": "privacy"}]
        )
        is_valid, errors = validate_aibom(aibom)
        self.assertTrue(is_valid, errors)

    def test_empty_values_are_not_emitted(self):
        aibom = _build_aibom({"technicalLimitations": ["", None]})
        card = aibom["components"][0].get("modelCard", {})
        self.assertNotIn("technicalLimitations", card.get("considerations", {}))


class TestPatternMatchDeterminism(unittest.TestCase):
    def test_matches_follow_pattern_order(self):
        extractor = EnhancedExtractor(hf_api=object())
        readme = "Known issues: drifts. Limitations: slow on CPU."
        matches = extractor._find_pattern_matches(readme, extractor.PATTERNS["limitations"])
        self.assertEqual(matches, ["slow on CPU", "drifts"])

    def test_matches_are_deduplicated(self):
        extractor = EnhancedExtractor(hf_api=object())
        readme = "Limitations: same text. Known issues: same text."
        matches = extractor._find_pattern_matches(readme, extractor.PATTERNS["limitations"])
        self.assertEqual(matches, ["same text"])

    def test_results_do_not_depend_on_hash_seed(self):
        outputs = set()
        for seed in ("0", "1", "2", "3", "4", "5"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            result = subprocess.run(
                [sys.executable, "-c", ORDER_SNIPPET],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
            )
            outputs.add(result.stdout.strip())
        self.assertEqual(
            outputs, {"bert\n['slow on CPU', 'drifts']"}
        )


if __name__ == "__main__":
    unittest.main()
