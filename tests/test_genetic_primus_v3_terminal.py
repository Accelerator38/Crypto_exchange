from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/finalize_genetic_primus_v3_rejection.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v3_terminal_test", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_terminal_manifest_hard_closes_all_authority(tmp_path: Path) -> None:
    freeze = {"status": "DEVELOPMENT_V3_PASSED_STRICT_GATES"}
    validation = {
        "status": "VALIDATION_V3_REJECTED",
        "oos_allowed": False,
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
        "champion_genome_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "metrics": {"lcb_bps": -1.0},
    }
    (tmp_path / "freeze_manifest.json").write_text(json.dumps(freeze), encoding="utf-8")
    (tmp_path / "validation_manifest.json").write_text(json.dumps(validation), encoding="utf-8")
    manifest = _module().build_terminal_manifest(tmp_path)
    assert manifest["status"] == "TERMINAL_REJECTED_VALIDATION_GENERALIZATION"
    for flag in ("retraining_allowed", "validation_reuse_allowed", "oos_allowed", "paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        assert manifest[flag] is False
