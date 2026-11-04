import json
import unittest

from src.utils.formatter import export_aibom
from src.utils.integrity import INTEGRITY_PROPERTY_PREFIX


def _sample_aibom() -> dict:
    return {
        "$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:12345678-1234-5678-1234-567812345678",
        "version": 1,
        "components": [
            {
                "type": "machine-learning-model",
                "name": "example-model",
                "modelCard": {
                    "modelParameters": {"task": "text-classification"},
                    "considerations": {
                        "ethicalConsiderations": [
                            {"name": "May reflect training-data bias."}
                        ]
                    },
                },
            }
        ],
        "dependencies": [
            {"ref": "metadata", "dependsOn": ["example-model"]}
        ],
    }


class TestFormatter(unittest.TestCase):
    def test_export_cyclonedx_1_6_pairs_schema_and_spec_version(self):
        exported = json.loads(export_aibom(_sample_aibom(), spec_version="1.6"))

        self.assertEqual(exported["bomFormat"], "CycloneDX")
        self.assertEqual(exported["specVersion"], "1.6")
        self.assertEqual(
            exported["$schema"],
            "http://cyclonedx.org/schema/bom-1.6.schema.json",
        )

    def test_export_cyclonedx_1_7_pairs_schema_and_spec_version(self):
        exported = json.loads(export_aibom(_sample_aibom(), spec_version="1.7"))

        self.assertEqual(exported["bomFormat"], "CycloneDX")
        self.assertEqual(exported["specVersion"], "1.7")
        self.assertEqual(
            exported["$schema"],
            "http://cyclonedx.org/schema/bom-1.7.schema.json",
        )

    def test_export_cyclonedx_1_7_does_not_change_body_mapping(self):
        exported_1_6 = json.loads(export_aibom(_sample_aibom(), spec_version="1.6"))
        exported_1_7 = json.loads(export_aibom(_sample_aibom(), spec_version="1.7"))

        for output in (exported_1_6, exported_1_7):
            output.pop("$schema")
            output.pop("specVersion")
            output.pop("bomFormat")
            output["properties"] = [
                prop for prop in output.get("properties", [])
                if not prop["name"].startswith(INTEGRITY_PROPERTY_PREFIX)
            ]

        self.assertEqual(exported_1_7, exported_1_6)


if __name__ == "__main__":
    unittest.main()
