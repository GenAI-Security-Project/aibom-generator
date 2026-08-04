"""
Tests for African regulatory compliance metadata support.

Covers:
  - _build_regulatory_properties() — property construction and namespace
  - Field registry registration — scoring, tiers, and recommendations
  - End-to-end injection at $.metadata.properties in generate_aibom()
  - GenerateRequest schema accepts regulatory_metadata
"""
import unittest
from unittest.mock import MagicMock, patch
from src.models.service import AIBOMService
from src.models.schemas import GenerateRequest
from src.models.scoring import calculate_completeness_score


# ---------------------------------------------------------------------------
# Regulatory property builder
# ---------------------------------------------------------------------------

class TestBuildRegulatoryProperties(unittest.TestCase):

    def test_single_jurisdiction_string(self):
        props = AIBOMService._build_regulatory_properties(
            {"african_deployment_jurisdictions": "NG"}
        )
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["name"], "owasp:aibom:regulatory:africanDeploymentJurisdictions")
        self.assertEqual(props[0]["value"], "NG")

    def test_multiple_jurisdictions_as_list_joined(self):
        props = AIBOMService._build_regulatory_properties(
            {"african_deployment_jurisdictions": ["NG", "ZA", "KE"]}
        )
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["value"], "NG, ZA, KE")

    def test_all_seven_fields_emitted(self):
        full_input = {
            "african_deployment_jurisdictions": ["NG", "ZA"],
            "ndpa_compliance":                  "registered; lawful_basis=legitimate_interests; dpia=completed",
            "nfiu_aml_compliance":              "cbn_licensed; aml_assessment=completed",
            "popia_compliance":                 "registered; information_officer=appointed",
            "kdpa_compliance":                  "registered; odpc_registration=active",
            "gdpa_compliance":                  "registered; dpc_registration=active",
            "regulatory_contact_point":         "dpo@example.com",
        }
        props = AIBOMService._build_regulatory_properties(full_input)
        self.assertEqual(len(props), 7)
        names = {p["name"] for p in props}
        self.assertIn("owasp:aibom:regulatory:africanDeploymentJurisdictions", names)
        self.assertIn("owasp:aibom:regulatory:ndpaCompliance", names)
        self.assertIn("owasp:aibom:regulatory:nfiuAmlCompliance", names)
        self.assertIn("owasp:aibom:regulatory:popiaCompliance", names)
        self.assertIn("owasp:aibom:regulatory:kdpaCompliance", names)
        self.assertIn("owasp:aibom:regulatory:gdpaCompliance", names)
        self.assertIn("owasp:aibom:regulatory:regulatoryContactPoint", names)

    def test_empty_dict_returns_no_props(self):
        self.assertEqual(AIBOMService._build_regulatory_properties({}), [])

    def test_none_values_skipped(self):
        props = AIBOMService._build_regulatory_properties(
            {"ndpa_compliance": None, "popia_compliance": "registered"}
        )
        self.assertEqual(len(props), 1)
        self.assertEqual(props[0]["name"], "owasp:aibom:regulatory:popiaCompliance")

    def test_unknown_keys_ignored(self):
        props = AIBOMService._build_regulatory_properties(
            {"unknown_key": "value", "ndpa_compliance": "registered"}
        )
        self.assertEqual(len(props), 1)

    def test_namespace_prefix_on_every_prop(self):
        props = AIBOMService._build_regulatory_properties(
            {"ndpa_compliance": "registered", "gdpa_compliance": "registered"}
        )
        for prop in props:
            self.assertTrue(
                prop["name"].startswith("owasp:aibom:regulatory:"),
                f"Property {prop['name']!r} does not use owasp:aibom:regulatory: namespace",
            )

    def test_non_string_scalar_coerced_to_string(self):
        props = AIBOMService._build_regulatory_properties(
            {"regulatory_contact_point": 42}
        )
        self.assertEqual(props[0]["value"], "42")


# ---------------------------------------------------------------------------
# Field registry — scoring and recommendations
# ---------------------------------------------------------------------------

class TestRegulatoryFieldRegistry(unittest.TestCase):
    """Verify the 7 new fields are registered and affect scoring correctly."""

    def _aibom_with_regulatory_props(self, props: list) -> dict:
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": "urn:uuid:test",
            "version": 1,
            "metadata": {
                "timestamp": "2026-01-01T00:00:00Z",
                "properties": props,
                "component": {
                    "name": "test-aibom",
                    "type": "application",
                    "version": "1.0",
                    "purl": "pkg:generic/test-aibom@1.0",
                },
            },
            "components": [
                {
                    "type": "machine-learning-model",
                    "name": "test-model",
                    "version": "1.0",
                    "bom-ref": "pkg:huggingface/org/test-model@abc12345",
                    "purl": "pkg:huggingface/org/test-model@abc12345",
                    "description": "A test model",
                }
            ],
        }

    def test_africanDeploymentJurisdictions_detected_when_present(self):
        aibom = self._aibom_with_regulatory_props([
            {
                "name": "owasp:aibom:regulatory:africanDeploymentJurisdictions",
                "value": "NG, ZA, KE",
            }
        ])
        score = calculate_completeness_score(aibom, validate=False)
        # Field is present — it should not appear in missing important fields
        missing_important = score.get("missing_fields", {}).get("important", [])
        self.assertNotIn("africanDeploymentJurisdictions", missing_important)

    def test_africanDeploymentJurisdictions_missing_reported_as_important(self):
        aibom = self._aibom_with_regulatory_props([])
        score = calculate_completeness_score(aibom, validate=False)
        missing_important = score.get("missing_fields", {}).get("important", [])
        # africanDeploymentJurisdictions is tier=important — must appear in missing
        self.assertIn("africanDeploymentJurisdictions", missing_important)

    def test_regulatory_fields_in_field_checklist(self):
        aibom = self._aibom_with_regulatory_props([])
        score = calculate_completeness_score(aibom, validate=False)
        checklist = score.get("field_checklist", {})
        for field in (
            "africanDeploymentJurisdictions",
            "ndpaCompliance",
            "nfiuAmlCompliance",
            "popiaCompliance",
            "kdpaCompliance",
            "gdpaCompliance",
            "regulatoryContactPoint",
        ):
            self.assertIn(field, checklist, f"{field} not found in field_checklist")

    def test_all_regulatory_supplementary_missing_reported(self):
        aibom = self._aibom_with_regulatory_props([])
        score = calculate_completeness_score(aibom, validate=False)
        missing_supp = score.get("missing_fields", {}).get("supplementary", [])
        for field in (
            "ndpaCompliance",
            "nfiuAmlCompliance",
            "popiaCompliance",
            "kdpaCompliance",
            "gdpaCompliance",
            "regulatoryContactPoint",
        ):
            self.assertIn(field, missing_supp, f"{field} not in missing supplementary fields")

    def test_populated_regulatory_fields_raise_score(self):
        """A fully documented regulatory section should raise total_score vs no props."""
        bare = self._aibom_with_regulatory_props([])
        full_props = [
            {"name": "owasp:aibom:regulatory:africanDeploymentJurisdictions", "value": "NG, ZA"},
            {"name": "owasp:aibom:regulatory:ndpaCompliance", "value": "registered"},
            {"name": "owasp:aibom:regulatory:nfiuAmlCompliance", "value": "aml_assessed"},
            {"name": "owasp:aibom:regulatory:popiaCompliance", "value": "registered"},
            {"name": "owasp:aibom:regulatory:kdpaCompliance", "value": "registered"},
            {"name": "owasp:aibom:regulatory:gdpaCompliance", "value": "registered"},
            {"name": "owasp:aibom:regulatory:regulatoryContactPoint", "value": "dpo@example.com"},
        ]
        populated = self._aibom_with_regulatory_props(full_props)
        bare_score = calculate_completeness_score(bare, validate=False)["total_score"]
        full_score = calculate_completeness_score(populated, validate=False)["total_score"]
        self.assertGreater(full_score, bare_score)


# ---------------------------------------------------------------------------
# End-to-end injection via generate_aibom
# ---------------------------------------------------------------------------

class TestRegulatoryMetadataInjection(unittest.TestCase):

    def _make_service(self) -> AIBOMService:
        svc = AIBOMService(hf_token="fake_token")
        svc.hf_api = MagicMock()
        return svc

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_regulatory_props_injected_at_metadata_properties(self, mock_extractor_cls, mock_score):
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model", "author": "org"}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 50}

        svc = self._make_service()
        svc.hf_api.model_info.return_value = MagicMock(sha="abc12345")
        svc.hf_api.model_card.return_value = MagicMock(data=MagicMock(to_dict=lambda: {}))

        aibom = svc.generate_aibom(
            "org/test-model",
            regulatory_metadata={
                "african_deployment_jurisdictions": ["NG", "ZA"],
                "ndpa_compliance": "registered; lawful_basis=legitimate_interests",
            },
        )

        meta_props = aibom.get("metadata", {}).get("properties", [])
        prop_map = {p["name"]: p["value"] for p in meta_props}

        self.assertIn("owasp:aibom:regulatory:africanDeploymentJurisdictions", prop_map)
        self.assertEqual(prop_map["owasp:aibom:regulatory:africanDeploymentJurisdictions"], "NG, ZA")
        self.assertIn("owasp:aibom:regulatory:ndpaCompliance", prop_map)
        self.assertEqual(
            prop_map["owasp:aibom:regulatory:ndpaCompliance"],
            "registered; lawful_basis=legitimate_interests",
        )

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_no_regulatory_metadata_no_injection(self, mock_extractor_cls, mock_score):
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {"name": "test-model"}
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 40}

        svc = self._make_service()
        svc.hf_api.model_info.return_value = MagicMock(sha="abc12345")
        svc.hf_api.model_card.return_value = MagicMock(data=MagicMock(to_dict=lambda: {}))

        aibom = svc.generate_aibom("org/test-model")

        meta_props = aibom.get("metadata", {}).get("properties", [])
        regulatory_props = [
            p for p in meta_props
            if p.get("name", "").startswith("owasp:aibom:regulatory:")
        ]
        self.assertEqual(regulatory_props, [])

    @patch("src.models.service.calculate_completeness_score")
    @patch("src.models.service.EnhancedExtractor")
    def test_full_nigerian_fintech_model_fixture(self, mock_extractor_cls, mock_score):
        """Simulate a Nigerian fintech AI model with complete compliance metadata."""
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_metadata.return_value = {
            "name": "fraud-detection-model",
            "author": "FirstBank Nigeria",
            "description": "NFIU-compliant fraud signal detection for CBN-licensed institutions",
        }
        mock_extractor.extraction_results = {}
        mock_score.return_value = {"total_score": 75}

        svc = self._make_service()
        svc.hf_api.model_info.return_value = MagicMock(sha="d3c34ef3")
        svc.hf_api.model_card.return_value = MagicMock(data=MagicMock(to_dict=lambda: {}))

        aibom = svc.generate_aibom(
            "firstbank-ng/fraud-detection-model",
            regulatory_metadata={
                "african_deployment_jurisdictions": ["NG"],
                "ndpa_compliance": (
                    "registered; nitda_reg=NITDA/REG/2025/001; "
                    "lawful_basis=legitimate_interests_ndpa_s25; dpia=completed"
                ),
                "nfiu_aml_compliance": (
                    "cbn_licensed; aml_assessed=true; "
                    "explainability_review=completed; nfiu_reg=2022"
                ),
                "regulatory_contact_point": "dpo@firstbanknigeria.com",
            },
        )

        meta_props = {p["name"]: p["value"] for p in aibom["metadata"].get("properties", [])}
        self.assertEqual(meta_props["owasp:aibom:regulatory:africanDeploymentJurisdictions"], "NG")
        self.assertIn("nitda_reg=NITDA/REG/2025/001", meta_props["owasp:aibom:regulatory:ndpaCompliance"])
        self.assertIn("nfiu_reg=2022", meta_props["owasp:aibom:regulatory:nfiuAmlCompliance"])
        self.assertEqual(meta_props["owasp:aibom:regulatory:regulatoryContactPoint"], "dpo@firstbanknigeria.com")
        # No other jurisdictions injected (only NG provided)
        self.assertNotIn("owasp:aibom:regulatory:popiaCompliance", meta_props)
        self.assertNotIn("owasp:aibom:regulatory:kdpaCompliance", meta_props)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestGenerateRequestSchema(unittest.TestCase):

    def test_regulatory_metadata_accepted(self):
        req = GenerateRequest(
            model_id="org/test-model",
            regulatory_metadata={
                "african_deployment_jurisdictions": ["NG", "KE"],
                "ndpa_compliance": "registered",
            },
        )
        self.assertIsNotNone(req.regulatory_metadata)
        self.assertEqual(req.regulatory_metadata["african_deployment_jurisdictions"], ["NG", "KE"])

    def test_regulatory_metadata_optional_defaults_none(self):
        req = GenerateRequest(model_id="org/test-model")
        self.assertIsNone(req.regulatory_metadata)

    def test_regulatory_metadata_empty_dict_accepted(self):
        req = GenerateRequest(model_id="org/test-model", regulatory_metadata={})
        self.assertIsNotNone(req.regulatory_metadata)
        self.assertEqual(req.regulatory_metadata, {})


if __name__ == "__main__":
    unittest.main()
