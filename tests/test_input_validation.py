import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.controllers import web_controller
from src.controllers.web_controller import is_valid_hf_input


class TestIsValidHfInput(unittest.TestCase):
    def test_accepts_owner_repo_and_hub_urls(self):
        for value in (
            "owner/model",
            "Owner-1/model_v1.2",
            "https://huggingface.co/owner/model",
            "https://huggingface.co/owner/model/tree/main",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_valid_hf_input(value))

    def test_rejects_markup_and_html_entities(self):
        for value in (
            "<script>org/model</script>",
            "org/<b>model</b>",
            "org/model&amp;x",
            "org/model\"onmouseover=\"x",
            "org/model'",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_valid_hf_input(value))

    def test_rejects_trailing_newline_and_control_characters(self):
        # `$` matches before a trailing "\n", so a regex using it accepts these.
        for value in (
            "org/model\n",
            "org/model\r\n",
            "org/model\t",
            "org/mo\x00del",
            "https://huggingface.co/org/model\n",
            " org/model",
        ):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_valid_hf_input(value))

    def test_rejects_dot_path_segments(self):
        for value in (
            "org/..",
            "../model",
            "./model",
            "org/.",
            "https://huggingface.co/org/..",
        ):
            with self.subTest(value=value):
                self.assertFalse(is_valid_hf_input(value))

    def test_rejects_malformed_ids(self):
        for value in (
            "",
            "model",
            "/model",
            "org/",
            "a/b/c",
            "x" * 201,
            "http://huggingface.co:80/org/model",
            "https://example.com/org/model",
            "https://huggingface.co/org",
        ):
            with self.subTest(value=value[:40]):
                self.assertFalse(is_valid_hf_input(value))


class TestGenerateValidatesBeforeSanitizing(unittest.TestCase):
    def _client(self):
        app = FastAPI()
        app.include_router(web_controller.router)
        return TestClient(app)

    def test_validator_receives_raw_input_not_html_escaped(self):
        raw = "org/model&x"
        with patch.object(web_controller, "is_valid_hf_input", return_value=False) as validator, \
             patch.object(web_controller, "get_sbom_count", lambda: "0"):
            self._client().post("/generate", data={"model_id": raw})

        validator.assert_called_once_with(raw)

    def test_invalid_input_is_rejected_before_any_hub_or_service_call(self):
        hf_api = MagicMock()
        service = MagicMock()
        with patch.object(web_controller, "HfApi", hf_api), \
             patch.object(web_controller, "AIBOMService", service), \
             patch.object(web_controller, "get_sbom_count", lambda: "0"):
            response = self._client().post(
                "/generate", data={"model_id": "<script>alert(1)</script>/x"}
            )

        self.assertEqual(response.template.name, "error.html")
        self.assertIn("Invalid model ID format", response.text)
        hf_api.assert_not_called()
        service.assert_not_called()

    def test_rejected_input_is_escaped_in_error_page(self):
        with patch.object(web_controller, "get_sbom_count", lambda: "0"):
            response = self._client().post(
                "/generate", data={"model_id": "<script>alert(1)</script>/x"}
            )

        self.assertNotIn("<script>alert(1)</script>", response.text)

    def test_trailing_newline_input_is_rejected(self):
        hf_api = MagicMock()
        with patch.object(web_controller, "HfApi", hf_api), \
             patch.object(web_controller, "get_sbom_count", lambda: "0"):
            response = self._client().post("/generate", data={"model_id": "org/model\n"})

        self.assertEqual(response.template.name, "error.html")
        hf_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
