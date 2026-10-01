from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from panteon_v2.policy.experiment_registry import (
    ExperimentRegistryError,
    StrategyExperimentRegistry,
    validate_experiment_registry,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "configs" / "strategy_experiment_registry_v1.json"
TOOL_PATH = ROOT / "tools" / "check_strategy_experiment_registry.py"


def _payload() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "check_strategy_experiment_registry_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_project_registry_is_valid_and_has_no_operational_candidate():
    registry = StrategyExperimentRegistry.from_json(REGISTRY_PATH)

    assert registry.operational_candidate_id is None
    assert registry.family_is_terminal("carryflow_flow_features_v1")
    assert all(
        item["continuation_allowed"] is False
        for item in registry.payload["experiments"]
    )
    assert all(
        item["promotion_authority"] is False
        for item in registry.payload["experiments"]
    )


def test_terminal_experiment_cannot_become_operational_candidate():
    payload = _payload()
    payload["operational_candidate_id"] = payload["experiments"][0][
        "experiment_id"
    ]

    with pytest.raises(
        ExperimentRegistryError,
        match="terminal experiment cannot be an operational candidate",
    ):
        validate_experiment_registry(payload)


def test_terminal_experiment_cannot_allow_continuation():
    payload = _payload()
    payload["experiments"][0]["continuation_allowed"] = True

    with pytest.raises(ExperimentRegistryError, match="safety flags"):
        validate_experiment_registry(payload)


def _registry_with_test_artifacts(tmp_path):
    payload = _payload()
    for experiment_index, experiment in enumerate(payload["experiments"]):
        for artifact_index, artifact in enumerate(experiment["artifacts"]):
            relative = f"evidence/{experiment_index}_{artifact_index}.json"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"test evidence {experiment_index} {artifact_index}".encode()
            path.write_bytes(content)
            artifact["path"] = relative
            artifact["sha256"] = hashlib.sha256(content).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    return registry_path


def test_registry_tool_reports_terminal_state(tmp_path):
    tool = _load_tool()
    registry = StrategyExperimentRegistry.from_json(_registry_with_test_artifacts(tmp_path))

    artifact_integrity = tool.verify_registered_artifacts(registry, root=tmp_path)
    summary = tool.build_summary(
        registry,
        experiment_id="divergence_short_systemic_guard_v1_prospective",
        artifact_integrity=artifact_integrity,
    )

    assert summary["artifact_integrity"] == {"checked": 12, "passed": True}
    assert summary["operational_candidate_id"] is None
    assert summary["orders_enabled"] is False
    assert summary["promotion_authority"] is False
    assert summary["experiments"][0]["status"] == "terminal_rejected"


def test_registry_tool_rejects_artifact_hash_mismatch(tmp_path):
    tool = _load_tool()
    test_registry_path = _registry_with_test_artifacts(tmp_path)
    payload = json.loads(test_registry_path.read_text(encoding="utf-8"))
    payload["experiments"][0]["artifacts"][0]["sha256"] = "0" * 64
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = StrategyExperimentRegistry.from_json(registry_path)

    with pytest.raises(ExperimentRegistryError, match="SHA-256 mismatch"):
        tool.verify_registered_artifacts(registry, root=tmp_path)


def test_public_metadata_check_does_not_claim_evidence_integrity(capsys):
    tool = _load_tool()
    assert tool.main(["--metadata-only"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifact_integrity"]["passed"] is False
    assert summary["orders_enabled"] is False
