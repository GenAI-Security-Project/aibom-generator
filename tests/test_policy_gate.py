"""Tests for the fail-closed CycloneDX AI-BOM policy gate."""

import json
from pathlib import Path

from src.models.policy_gate import evaluate_aibom_policy
from src.policy_gate import main


def _bom(**overrides):
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:12345678-1234-5678-1234-567812345678",
        "version": 1,
        "components": [
            {
                "type": "machine-learning-model",
                "name": "demo",
                "version": "1.0.0",
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "modelCard": {"modelParameters": {"task": "text-generation"}},
            }
        ],
    }
    document.update(overrides)
    return document


def test_empty_policy_fails_closed():
    decision = evaluate_aibom_policy(_bom(), {})
    assert decision.passed is False
    assert "non-empty" in decision.violations[0]


def test_unknown_policy_key_fails_closed():
    decision = evaluate_aibom_policy(_bom(), {"min_score": 70})
    assert decision.passed is False
    assert "unknown policy keys" in decision.violations[0]


def test_denied_license_and_expression_are_rejected():
    bom = _bom()
    bom["components"][0]["licenses"] = [{"expression": "MIT OR Apache-2.0"}]
    decision = evaluate_aibom_policy(bom, {"denied_licenses": ["mit"]})
    assert decision.passed is False
    assert "MIT" in decision.violations[0]


def test_allowed_task_and_spec_version_pass():
    decision = evaluate_aibom_policy(
        _bom(),
        {
            "spec_version": "1.6",
            "require_license": True,
            "allowed_tasks": ["text-generation"],
            "required_fields": ["bomFormat", "specVersion", "serialNumber"],
        },
    )
    assert decision.passed is True
    assert decision.violations == []


def test_disallowed_task_fails():
    decision = evaluate_aibom_policy(_bom(), {"allowed_tasks": ["fill-mask"]})
    assert decision.passed is False
    assert "text-generation" in decision.violations[0]


def test_caller_supplied_score_is_enforced_without_rescoring():
    decision = evaluate_aibom_policy(_bom(), {"minimum_score": 80}, score=79.5)
    assert decision.passed is False
    assert "79.5" in decision.violations[0]


def test_cli_exits_nonzero_on_violation(tmp_path: Path, capsys):
    bom_path = tmp_path / "bom.json"
    policy_path = tmp_path / "policy.yaml"
    bom_path.write_text(json.dumps(_bom()), encoding="utf-8")
    policy_path.write_text("spec_version: '1.7'\n", encoding="utf-8")

    code = main(["--bom", str(bom_path), "--policy", str(policy_path)])

    assert code == 1
    assert "1.6" in capsys.readouterr().out
