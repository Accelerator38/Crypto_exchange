from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/audit_genetic_primus_v3_generalization.py"


def _module():
    spec = importlib.util.spec_from_file_location("primus_v3_audit_test", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_population_stability_index_detects_shift() -> None:
    module = _module()
    reference = pd.Series(range(1000), dtype=float)
    same = pd.Series(range(1000), dtype=float)
    shifted = pd.Series(range(1000, 2000), dtype=float)
    assert module.population_stability_index(reference, same) < 1.0e-9
    assert module.population_stability_index(reference, shifted) > 0.25
