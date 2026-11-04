from unittest.mock import patch

from src.models.extractor import EnhancedExtractor


def _extract_from_readme(readme: str):
    extractor = EnhancedExtractor(hf_api=object())
    return extractor._try_text_pattern_extraction(
        "licenses",
        {"readme_content": readme},
    )


def test_license_value_does_not_consume_following_prose():
    readme = (
        "# Model\n\n"
        "License: MIT\n\n"
        "This model is a fine-tuned version of bert for sentiment.\n"
    )

    assert _extract_from_readme(readme) == "MIT"


def test_license_identifier_is_kept_when_same_line_prose_follows():
    assert _extract_from_readme("License: MIT This model is experimental.\n") == "MIT"


def test_prose_that_mentions_license_is_not_a_license_declaration():
    assert _extract_from_readme("Please read the license before use, thanks.\n") is None


def test_markdown_license_section_accepts_known_short_license_name():
    readme = "## License\n\nApache 2.0, see LICENSE file.\n"

    assert _extract_from_readme(readme) == "Apache-2.0"


def test_repository_license_file_precedes_readme_guess():
    extractor = EnhancedExtractor(hf_api=object())
    context = {
        "model_id": "acme/model",
        "model_info": None,
        "model_card": None,
        "config_data": None,
        "readme_content": "License: MIT\n",
    }

    with patch.object(
        extractor,
        "_detect_license_from_file",
        return_value="Apache-2.0",
    ):
        value = extractor._extract_registry_field(
            "licenses",
            extractor.registry_fields["licenses"],
            context,
        )

    assert value == "Apache-2.0"
    assert extractor.extraction_results["licenses"].extraction_method == "license_file"
