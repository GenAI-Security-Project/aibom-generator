import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates

from src.controllers import web_controller


def _sample_aibom() -> dict:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:12345678-1234-5678-1234-567812345678",
        "version": 1,
        "metadata": {
            "timestamp": "2026-06-10T00:00:00Z",
            "tools": {
                "components": [
                    {"name": "OWASP AIBOM Generator"}
                ]
            },
        },
        "components": [
            {
                "type": "machine-learning-model",
                "name": "local-web-model",
                "version": "1.0.0",
                "description": "Web controller UTF-8 regression path \u2713",
                "purl": "pkg:huggingface/owner/local-web-model",
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            }
        ],
        "externalReferences": [
            {"type": "website", "url": "https://huggingface.co/owner/local-web-model"}
        ],
    }


def _sample_score() -> dict:
    category_details = {
        "required_fields": {"present_fields": 4, "total_fields": 4, "percentage": 100, "max_points": 20},
        "metadata": {"present_fields": 2, "total_fields": 2, "percentage": 100, "max_points": 20},
        "component_basic": {"present_fields": 4, "total_fields": 4, "percentage": 100, "max_points": 20},
        "component_model_card": {"present_fields": 0, "total_fields": 0, "percentage": 0, "max_points": 30},
        "external_references": {"present_fields": 1, "total_fields": 1, "percentage": 100, "max_points": 10},
    }
    section_scores = {
        "required_fields": 20,
        "metadata": 20,
        "component_basic": 20,
        "component_model_card": 0,
        "external_references": 10,
    }
    return {
        "total_score": 70,
        "subtotal_score": 70,
        "completeness_profile": {"name": "Basic", "description": "Regression test profile"},
        "field_checklist": {
            "bomFormat": "\u2714 present",
            "specVersion": "\u2714 present",
            "serialNumber": "\u2714 present",
            "version": "\u2714 present",
        },
        "field_types": {},
        "reference_urls": {},
        "category_details": category_details,
        "category_fields_list": {
            "component_model_card": [],
            "external_references": [],
        },
        "section_scores": section_scores,
        "max_scores": {
            "required_fields": 20,
            "metadata": 20,
            "component_basic": 20,
            "component_model_card": 30,
            "external_references": 10,
        },
        "missing_counts": {"critical": 0, "important": 0, "supplementary": 0},
        "recommendations": [],
        "penalty_applied": False,
        "penalty_percentage": 0,
        "penalty_reason": "",
        "penalty_factor": 1,
        "validation": {"valid": True, "issues": []},
    }


class _FakeHfApi:
    def model_info(self, model_id):
        return {"modelId": model_id}


class _FakeService:
    def __init__(self, use_best_practices=True):
        self.use_best_practices = use_best_practices

    @staticmethod
    def _normalise_model_id(model_id):
        return model_id

    def generate_aibom(self, model_id, include_inference=False):
        return _sample_aibom()

    def get_enhancement_report(self):
        return {"final_score": _sample_score()}


class WebControllerTests(unittest.TestCase):
    def test_generate_route_writes_utf8_json_to_patched_output_dir(self):
        app = FastAPI()
        app.include_router(web_controller.router)

        repo_root = Path(__file__).resolve().parents[1]
        temp_root = repo_root / "sboms" / f"test-web-controller-{uuid.uuid4().hex}"
        output_dir = temp_root / "nested-output"
        try:
            self.assertFalse(output_dir.exists())

            with patch.object(web_controller, "OUTPUT_DIR", str(output_dir)), \
                 patch.object(web_controller, "HfApi", _FakeHfApi), \
                 patch.object(web_controller, "AIBOMService", _FakeService), \
                 patch.object(web_controller, "log_sbom_generation", lambda model_id: None), \
                 patch.object(web_controller, "get_sbom_count", lambda: "0"):
                self.assertIsInstance(web_controller.templates, Jinja2Templates)
                client = TestClient(app)
                response = client.post(
                    "/generate",
                    data={"model_id": "owner/local-web-model"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.template.name, "result.html")
            self.assertIn("AIBOM Summary", response.text)
            self.assertIn("CycloneDX 1.6", response.text)
            self.assertIn("CycloneDX 1.7", response.text)

            json_1_6 = output_dir / "owner_local-web-model_ai_sbom_1_6.json"
            json_1_7 = output_dir / "owner_local-web-model_ai_sbom_1_7.json"

            self.assertGreater(json_1_6.stat().st_size, 0)
            self.assertGreater(json_1_7.stat().st_size, 0)

            json.loads(json_1_6.read_text(encoding="utf-8"))
            json.loads(json_1_7.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    @patch("src.controllers.web_controller.aiofiles.open")
    def test_generate_uses_aiofiles_async(self, mock_aiofiles_open):
        from unittest.mock import AsyncMock
        mock_file = AsyncMock()
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file

        app = FastAPI()
        app.include_router(web_controller.router)

        with patch.object(web_controller, "OUTPUT_DIR", "/tmp/dummy"), \
             patch.object(web_controller, "HfApi", _FakeHfApi), \
             patch.object(web_controller, "AIBOMService", _FakeService), \
             patch.object(web_controller, "log_sbom_generation", lambda model_id: None), \
             patch.object(web_controller, "get_sbom_count", lambda: "0"):

            client = TestClient(app)
            response = client.post(
                "/generate",
                data={"model_id": "owner/local-web-model"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_aiofiles_open.call_count, 2)
        # Verify that await f.write(...) was called via the context manager
        self.assertEqual(mock_file.write.call_count, 2)

if __name__ == "__main__":
    unittest.main()
