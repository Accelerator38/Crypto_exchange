from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/evaluate_genetic_primus_v5_data_source.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v5_data_gate_test", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_terminal_data_gate_blocks_evolution(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "V5_TERMINAL_REJECTED_DATA_SOURCE_INSUFFICIENT_COVERAGE",
                "evolution_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="evolution blocked"):
        _module().assert_evolution_authority(report)


def test_only_explicit_source_acceptance_can_open_evolution(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = {"status": "V5_DATA_SOURCE_ACCEPTED", "evolution_allowed": True}
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _module().assert_evolution_authority(report) == payload
