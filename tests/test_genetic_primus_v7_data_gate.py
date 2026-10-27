from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/evaluate_genetic_primus_v7_data_source.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v7_data_gate_test", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v7_rejected_source_blocks_feasibility(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "V7_TERMINAL_REJECTED_DATA_SOURCE"}), encoding="utf-8")
    with pytest.raises(PermissionError, match="source status"):
        _module().assert_feasibility_authority(report)


def test_v7_explicit_source_acceptance_opens_feasibility(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = {
        "status": "V7_DATA_INTEGRITY_ACCEPTED_FEASIBILITY_PENDING",
        "feasibility_allowed": True,
        "feasibility_opened": False,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _module().assert_feasibility_authority(report) == payload
