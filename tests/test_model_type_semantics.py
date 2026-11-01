import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.controllers import web_controller
from src.models.registry import get_field_registry_manager
from src.models.scoring import calculate_completeness_score
from src.models.service import AIBOMService
from tests.test_web_controller import _FakeHfApi, _FakeService

MODEL_ID = "Nanbeige/Nanbeige4.1-3B"

# Fields whose registry jsonpath must resolve against the BOM we emit (#58).
EMITTED_FIELDS = (
    "primaryPurpose",
    "suppliedBy",
    "intendedUse",
    "ethicalConsiderations",
    "technicalLimitations",
    "datasets",
    "typeOfModel",
    "paper",
    "external_references",
)


def _metadata(**overrides):
    metadata = {
        "name": "Nanbeige4.1-3B",
        "author": "Nanbeige",
        "suppliedBy": "Nanbeige",
        "primaryPurpose": "text-generation",
        "typeOfModel": "llama",
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "datasets": ["acme/corpus"],
        "intendedUse": "Chat assistants",
        "technicalLimitations": "May hallucinate",
        "ethicalConsiderations": "May reflect training-data bias",
        "paper": "https://arxiv.org/abs/2401.00001",
        "license": "apache-2.0",
        "commit": "a" * 40,
    }
    metadata.update(overrides)
    return metadata


def _build_aibom(**overrides):
    service = AIBOMService(hf_token="fake_token")
    with patch.object(AIBOMService, "_verify_datasets_available", return_value=True):
        return service._create_aibom_structure(MODEL_ID, _metadata(**overrides))


class TestRegistryPathsMatchEmittedBom(unittest.TestCase):
    def test_no_registry_path_uses_singular_component_root(self):
        fields = get_field_registry_manager().get_field_definitions()
        for name, definition in fields.items():
            for path in (
                definition.get("jsonpath", ""),
                definition.get("aibom_generation", {}).get("location", ""),
            ):
                with self.subTest(field=name, path=path):
                    self.assertFalse(path.startswith("$.component."))
                    self.assertFalse(path.startswith("$.metadata.component.externalReferences"))

    def test_registry_jsonpath_resolves_for_every_emitted_field(self):
        manager = get_field_registry_manager()
        aibom = _build_aibom()
        for name in EMITTED_FIELDS:
            with self.subTest(field=name):
                present, value = manager.detect_field_presence(aibom, manager.get_field_jsonpath(name))
                self.assertTrue(present, f"{name} not found at {manager.get_field_jsonpath(name)}")
                self.assertTrue(value)

    def test_emitted_fields_are_checked_off_in_the_field_checklist(self):
        result = calculate_completeness_score(_build_aibom(), validate=False)
        for name in EMITTED_FIELDS:
            with self.subTest(field=name):
                self.assertTrue(result["field_checklist"][name].startswith("✔"))

    def test_checklist_location_matches_emitted_json_location(self):
        result = calculate_completeness_score(_build_aibom(), validate=False)
        paths = {
            f["name"]: f["path"]
            for fields in result["category_fields_list"].values()
            for f in fields
        }
        self.assertEqual(paths["primaryPurpose"], "modelCard.modelParameters.task")
        self.assertEqual(paths["typeOfModel"], "modelCard.modelParameters.modelArchitecture")
        self.assertEqual(paths["suppliedBy"], "supplier.name")


class TestModelTypeAndArchitectureMapping(unittest.TestCase):
    """The three concepts stay distinct in the emitted AIBOM (backward compatible)."""

    def test_component_type_family_and_implementation_are_separate(self):
        component = _build_aibom()["components"][0]

        self.assertEqual(component["type"], "machine-learning-model")
        self.assertEqual(
            component["modelCard"]["modelParameters"]["modelArchitecture"], "llama"
        )
        properties = {p["name"]: p["value"] for p in component["properties"]}
        self.assertEqual(properties["model_type"], "llama")
        self.assertEqual(properties["architectures"], "LlamaForCausalLM")

    def test_multiple_architectures_are_comma_joined(self):
        component = _build_aibom(architectures=["LlamaForCausalLM", "LlamaModel"])["components"][0]
        properties = {p["name"]: p["value"] for p in component["properties"]}
        self.assertEqual(properties["architectures"], "LlamaForCausalLM, LlamaModel")


class TestHumanViewTerminology(unittest.TestCase):
    def _render(self, aibom):
        class _Service(_FakeService):
            def generate_aibom(self, model_id, include_inference=False):
                return aibom

        app = FastAPI()
        app.include_router(web_controller.router)
        repo_root = Path(__file__).resolve().parents[1]
        temp_root = repo_root / "sboms" / f"test-model-semantics-{uuid.uuid4().hex}"
        try:
            with patch.object(web_controller, "OUTPUT_DIR", str(temp_root)), \
                 patch.object(web_controller, "HfApi", _FakeHfApi), \
                 patch.object(web_controller, "AIBOMService", _Service), \
                 patch.object(web_controller, "log_sbom_generation", lambda model_id: None), \
                 patch.object(web_controller, "get_sbom_count", lambda: "0"):
                response = TestClient(app).post("/generate", data={"model_id": MODEL_ID})
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
        self.assertEqual(response.status_code, 200)
        return response.text

    def test_labels_distinguish_component_type_family_and_implementation(self):
        html = self._render(_build_aibom())

        self.assertIn("Component type (CycloneDX):", html)
        self.assertIn("Model family (HF model_type):", html)
        self.assertIn("Model implementation (HF architectures):", html)
        self.assertIn("LlamaForCausalLM", html)
        self.assertNotIn('property-name">Type:', html)
        self.assertNotIn('property-name">Architecture:', html)

    def test_implementation_row_is_omitted_when_architectures_absent(self):
        for value in (None, []):
            with self.subTest(architectures=value):
                html = self._render(_build_aibom(architectures=value))

                self.assertIn("Model family (HF model_type):", html)
                self.assertNotIn("Model implementation (HF architectures):", html)

    def test_empty_model_type_and_architectures_do_not_emit_placeholder_properties(self):
        component = _build_aibom(architectures=None, model_type=None)["components"][0]
        names = {p["name"] for p in component.get("properties", [])}
        self.assertNotIn("architectures", names)
        self.assertNotIn("model_type", names)


if __name__ == "__main__":
    unittest.main()
