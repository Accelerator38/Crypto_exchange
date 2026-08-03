from __future__ import annotations

import importlib.util
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


def test_registry_tool_reports_terminal_state():
    tool = _load_tool()
    registry = StrategyExperimentRegistry.from_json(REGISTRY_PATH)

    artifact_integrity = tool.verify_registered_artifacts(registry, root=ROOT)
    summary = tool.build_summary(
        registry,
        experiment_id="divergence_short_systemic_guard_v1_prospective",
        artifact_integrity=artifact_integrity,
    )

    assert summary["artifact_integrity"] == {"checked": 15, "passed": True}
    assert summary["operational_candidate_id"] is None
    assert summary["orders_enabled"] is False
    assert summary["promotion_authority"] is False
    assert summary["experiments"][0]["status"] == "terminal_rejected"


def test_registry_tool_rejects_artifact_hash_mismatch(tmp_path):
    tool = _load_tool()
    payload = _payload()
    payload["experiments"][0]["artifacts"][0]["sha256"] = "0" * 64
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    registry = StrategyExperimentRegistry.from_json(registry_path)

    with pytest.raises(ExperimentRegistryError, match="SHA-256 mismatch"):
        tool.verify_registered_artifacts(registry, root=ROOT)
