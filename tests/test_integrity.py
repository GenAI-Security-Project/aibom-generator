import copy
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot6

from src.controllers.cli_controller import CLIController
from src.models.service import AIBOMService
from src.utils.formatter import export_aibom
from src.utils.integrity import (
    INTEGRITY_DIGEST_PROPERTY,
    add_integrity_attestation,
    verify_integrity_attestation,
)
from src.utils.validation import validate_aibom


def _cyclonedx_document() -> dict:
    bom = Bom()
    bom.serial_number = uuid.UUID("12345678-1234-5678-1234-567812345678")
    bom.components.add(
        Component(name="test-model", type=ComponentType.MACHINE_LEARNING_MODEL)
    )
    document = json.loads(JsonV1Dot6(bom).output_as_string())
    component = document["components"][0]
    component["hashes"] = [{"alg": "SHA-256", "content": "0" * 64}]
    component["licenses"] = [{"license": {"id": "MIT"}}]
    document["properties"] = [{"name": "dataset", "value": "owner/data"}]
    return document


def _spdx_document() -> dict:
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "test-document",
        "documentNamespace": "https://example.com/spdx/test-document",
        "creationInfo": {
            "created": "2026-01-01T00:00:00Z",
            "creators": ["Tool: test-suite"],
        },
    }


def _digest(document: dict) -> str:
    for prop in document["properties"]:
        if prop.get("name") == INTEGRITY_DIGEST_PROPERTY:
            return prop["value"]
    raise AssertionError("digest property missing")


def test_cyclonedx_attestation_is_schema_valid_and_verifiable():
    document = add_integrity_attestation(_cyclonedx_document())

    schema_valid, schema_errors = validate_aibom(document)
    result = verify_integrity_attestation(document)

    assert schema_valid, schema_errors
    assert result["status"] == "verified"
    assert result["valid"] is True


def test_service_generated_document_contains_valid_attestation():
    service = AIBOMService(hf_token="test-token")

    document = service._create_aibom_structure(
        "owner/test-model",
        {"name": "test-model", "commit": "a" * 40},
    )

    assert verify_integrity_attestation(document)["status"] == "verified"


def test_jcs_digest_is_stable_across_object_key_order():
    original = _cyclonedx_document()
    reordered = {key: original[key] for key in reversed(list(original))}

    assert _digest(add_integrity_attestation(original)) == _digest(
        add_integrity_attestation(reordered)
    )


def test_verification_detects_security_metadata_tampering():
    mutations = [
        lambda document: document["components"][0]["hashes"][0].update(
            {"content": "f" * 64}
        ),
        lambda document: document["components"][0]["licenses"][0]["license"].update(
            {"id": "Apache-2.0"}
        ),
        lambda document: document["properties"][0].update(
            {"value": "attacker/replacement"}
        ),
    ]

    for mutate in mutations:
        document = add_integrity_attestation(_cyclonedx_document())
        mutate(document)
        result = verify_integrity_attestation(document)
        assert result["status"] == "mismatch"
        assert result["valid"] is False


def test_legacy_unattested_document_is_explicitly_reported():
    result = AIBOMService.verify_aibom_document(_cyclonedx_document())

    assert result["valid"] is True
    assert result["integrity"]["status"] == "unattested"
    assert result["integrity"]["valid"] is None


def test_spdx_attestation_and_required_field_validation():
    document = add_integrity_attestation(_spdx_document())

    result = AIBOMService.verify_aibom_document(document)

    assert result["valid"] is True
    assert result["schema_validation"]["valid"] is True
    assert result["integrity"]["status"] == "verified"


def test_controller_spdx_schema_version_validation_is_implemented():
    controller = CLIController()

    valid, errors = controller._validate_spdx_schema_version(
        _spdx_document(),
        "2.3",
    )

    assert valid is True
    assert errors == []


def test_cyclonedx_1_7_export_verifies_against_declared_schema():
    document = json.loads(export_aibom(_cyclonedx_document(), spec_version="1.7"))

    result = AIBOMService.verify_aibom_document(document)

    assert result["valid"] is True
    assert result["schema_validation"]["valid"] is True
    assert result["integrity"]["status"] == "verified"


def test_command_line_verification_rejects_tampering():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "sboms" / f"test-integrity-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    output_file = output_dir / "document.json"
    try:
        document = add_integrity_attestation(_cyclonedx_document())
        output_file.write_text(json.dumps(document), encoding="utf-8")

        valid = subprocess.run(
            [sys.executable, "-m", "src.cli", "--verify", str(output_file)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert valid.returncode == 0, valid.stderr

        tampered = copy.deepcopy(document)
        tampered["components"][0]["licenses"][0]["license"]["id"] = "Apache-2.0"
        output_file.write_text(json.dumps(tampered), encoding="utf-8")
        invalid = subprocess.run(
            [sys.executable, "-m", "src.cli", "--verify", str(output_file)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert invalid.returncode == 1
        assert "digest mismatch" in invalid.stderr.lower()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
