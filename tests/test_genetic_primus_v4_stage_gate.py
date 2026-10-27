from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/evaluate_genetic_primus_v4_stage.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v4_stage_test", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rejected_development_cannot_open_validation(tmp_path: Path) -> None:
    (tmp_path / "freeze_manifest.json").write_text(
        json.dumps({"status": "DEVELOPMENT_V4_REJECTED_STRICT_GATES", "validation_allowed": False, "validation_opened": False}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="blocked by rejected development"):
        _module().assert_stage_authority(tmp_path, "validation")


def test_rejected_validation_cannot_open_oos(tmp_path: Path) -> None:
    (tmp_path / "validation_manifest.json").write_text(
        json.dumps({"status": "VALIDATION_V4_REJECTED_STRICT_GATES", "oos_allowed": False, "oos_opened": False}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="blocked by validation"):
        _module().assert_stage_authority(tmp_path, "oos")
