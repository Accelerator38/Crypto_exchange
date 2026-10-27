from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/evaluate_genetic_primus_v6_data_source.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v6_data_gate_test", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_only_pending_pass_can_open_v6_feasibility(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = {
        "status": "V6_DATA_INTEGRITY_ACCEPTED_FEASIBILITY_PENDING",
        "feasibility_allowed": True,
        "feasibility_opened": False,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _module().assert_feasibility_authority(report) == payload
    payload["feasibility_opened"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="absent or consumed"):
        _module().assert_feasibility_authority(report)
