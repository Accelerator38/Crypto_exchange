from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_genetic_primus_v7.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v7_stage_gate_test", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_v7_manifest_blocks_development() -> None:
    with pytest.raises(PermissionError, match="event capacity"):
        _module().authorize_stage(
            "development",
            ROOT / "Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824/manifest.json",
        )


@pytest.mark.parametrize("stage", ["validation", "oos", "paper"])
def test_later_v7_stages_are_blocked(stage: str) -> None:
    with pytest.raises(PermissionError, match="frozen authority"):
        _module().authorize_stage(
            stage,
            ROOT / "Reports/Exia/Genetic_Primus/v7_event_feasibility_20260824/manifest.json",
        )
