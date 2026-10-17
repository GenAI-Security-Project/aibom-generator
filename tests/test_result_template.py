import re
import unittest
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class ResultTemplateScoreTests(unittest.TestCase):
    def test_category_point_totals_come_from_scoring_payload(self):
        template = (
            Path(__file__).resolve().parents[1] / "src" / "templates" / "result.html"
        ).read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(r"/\d+ points", template),
            "category summaries must not contain hard-coded point totals",
        )
        self.assertIsNone(
            re.search(r"\bmax_score\b", template),
            "score report must not keep a separate hard-coded maximum",
        )
        self.assertRegex(
            template,
            re.compile(
                r"section_scores\[key\]\|round\(1\)\s*}}/{{\s*"
                r"completeness_score\.category_details\[key\]\.max_points\s*}}",
                re.DOTALL,
            ),
            "score report denominator must use the current category payload",
        )
        categories = (
            "required_fields",
            "metadata",
            "component_basic",
            "component_model_card",
            "external_references",
        )
        for category in categories:
            self.assertIn(
                f"category_details.{category}.max_points",
                template,
                f"{category} summary must use scoring payload max_points",
            )
            self.assertIn(
                f"('{self._display_name(category)}', '{category}')",
                template,
                f"{category} score-report tuple must not contain a numeric maximum",
            )

    def test_rendered_fields_and_totals_come_from_scoring_payload(self):
        template_root = Path(__file__).resolve().parents[1] / "src" / "templates"
        template = Environment(
            loader=FileSystemLoader(template_root),
            autoescape=True,
        ).get_template("result.html")
        categories = (
            "required_fields",
            "metadata",
            "component_basic",
            "component_model_card",
            "external_references",
        )
        max_scores = {category: 41 + index for index, category in enumerate(categories)}
        probes = {f"probe_{category}": "✘ missing" for category in categories}
        score = {
            "total_score": 50,
            "subtotal_score": 50,
            "completeness_profile": {"name": "Test", "description": "Test"},
            "field_checklist": probes,
            "field_types": {},
            "reference_urls": {},
            "missing_fields": {},
            "missing_counts": {},
            "max_scores": max_scores,
            "category_details": {
                category: {
                    "present_fields": 0,
                    "total_fields": 1,
                    "max_points": max_scores[category],
                    "percentage": 0,
                }
                for category in categories
            },
            "section_scores": dict.fromkeys(categories, 0),
            "category_fields_list": {
                category: [
                    {
                        "name": f"probe_{category}",
                        "tier": "Critical",
                        "path": f"probe.{category}",
                    }
                ]
                for category in categories
            },
            "penalty_applied": False,
            "penalty_reason": "",
            "recommendations": [],
        }
        aibom = {
            "metadata": {"timestamp": "2026-08-06T00:00:00Z"},
            "bomFormat": "CycloneDX",
            "serialNumber": "urn:uuid:test",
            "components": [],
        }

        rendered = template.render(
            model_id="test/model",
            completeness_score=score,
            aibom=aibom,
            result={},
            metadata={},
        )

        for category, max_points in max_scores.items():
            self.assertIn(f"probe_{category}", rendered)
            self.assertIn(f"/{max_points}", rendered)
        self.assertIn(f"/{sum(max_scores.values())}", rendered)

    @staticmethod
    def _display_name(category):
        return {
            "required_fields": "Required Fields",
            "metadata": "Metadata",
            "component_basic": "Component Basic",
            "component_model_card": "Model Card",
            "external_references": "External References",
        }[category]


if __name__ == "__main__":
    unittest.main()
