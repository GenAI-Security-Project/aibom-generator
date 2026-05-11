import unittest
from src.utils.license_utils import (
    normalize_license_id,
    get_license_url,
    LICENSE_MAPPING,
)


class TestNormalizeLicenseId(unittest.TestCase):
    # --- falsy inputs ---
    def test_none_returns_none(self):
        self.assertIsNone(normalize_license_id(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(normalize_license_id(""))

    # --- exact SPDX IDs (fixed up via cdx_spdx.fixup_id) ---
    def test_correct_case_passthrough(self):
        self.assertEqual(normalize_license_id("Apache-2.0"), "Apache-2.0")

    def test_lowercase_spdx_id_normalized(self):
        self.assertEqual(normalize_license_id("apache-2.0"), "Apache-2.0")

    def test_mixed_case_spdx_id_normalized(self):
        self.assertEqual(normalize_license_id("MIT"), "MIT")
        self.assertEqual(normalize_license_id("mit"), "MIT")

    def test_cc_license_normalized(self):
        self.assertEqual(normalize_license_id("CC-BY-4.0"), "CC-BY-4.0")

    # --- LICENSE_MAPPING aliases ---
    def test_multi_word_alias_apache(self):
        self.assertEqual(normalize_license_id("apache license 2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("Apache License 2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("apache license version 2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("apache 2.0"), "Apache-2.0")
        self.assertEqual(normalize_license_id("Apache License, Version 2.0"), "Apache-2.0")

    def test_multi_word_alias_mit(self):
        self.assertEqual(normalize_license_id("MIT License"), "MIT")

    def test_multi_word_alias_bsd(self):
        self.assertEqual(normalize_license_id("bsd 3-clause"), "BSD-3-Clause")
        self.assertEqual(normalize_license_id("bsd 2-clause"), "BSD-2-Clause")

    def test_multi_word_alias_gpl(self):
        self.assertEqual(normalize_license_id("gnu general public license v3"), "GPL-3.0-only")
        self.assertEqual(normalize_license_id("gplv3"), "GPL-3.0-only")
        self.assertEqual(normalize_license_id("gpl-3.0"), "GPL-3.0-only")
        self.assertEqual(normalize_license_id("gnu general public license v2"), "GPL-2.0-only")
        self.assertEqual(normalize_license_id("gplv2"), "GPL-2.0-only")

    def test_multi_word_alias_nvidia(self):
        self.assertEqual(
            normalize_license_id("nvidia open model license agreement"),
            "nvidia-open-model-license",
        )

    # --- unknown tokens ---
    def test_unknown_simple_token_returned_as_is(self):
        result = normalize_license_id("custom-license-v1")
        self.assertEqual(result, "custom-license-v1")

    def test_unknown_long_multiword_returns_none(self):
        long_phrase = "a " * 30  # >50 chars, has spaces
        self.assertIsNone(normalize_license_id(long_phrase.strip()))


class TestGetLicenseUrl(unittest.TestCase):
    def test_known_license_returns_exact_url(self):
        self.assertEqual(
            get_license_url("Apache-2.0"),
            "https://www.apache.org/licenses/LICENSE-2.0.txt",
        )
        self.assertEqual(
            get_license_url("MIT"),
            "https://opensource.org/licenses/MIT",
        )

    def test_unknown_license_returns_spdx_fallback(self):
        url = get_license_url("unknown-license")
        self.assertIn("spdx.org", url)
        self.assertIn("unknown-license", url)

    def test_no_fallback_returns_none_for_unknown(self):
        self.assertIsNone(get_license_url("unknown-license", fallback=False))

    def test_no_fallback_returns_url_for_known(self):
        url = get_license_url("MIT", fallback=False)
        self.assertIsNotNone(url)

    def test_nvidia_custom_license_url(self):
        url = get_license_url("nvidia-open-model-license", fallback=False)
        self.assertIsNotNone(url)
        self.assertIn("nvidia.com", url)


class TestLicenseMappingCompleteness(unittest.TestCase):
    def test_all_values_are_valid_spdx_or_known_custom(self):
        """Every SPDX value in LICENSE_MAPPING should either pass is_supported_id
        or be a known custom ID (like nvidia-open-model-license)."""
        from cyclonedx import spdx as cdx_spdx

        known_custom = {"nvidia-open-model-license"}
        for alias, spdx_id in LICENSE_MAPPING.items():
            with self.subTest(alias=alias, spdx_id=spdx_id):
                self.assertTrue(
                    cdx_spdx.is_supported_id(spdx_id) or spdx_id in known_custom,
                    f"Value '{spdx_id}' for alias '{alias}' is neither a valid SPDX ID nor a known custom ID",
                )


if __name__ == "__main__":
    unittest.main()
