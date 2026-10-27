"""Direct regressions for qualification routing; no market data or search."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.analyze_genetic_primus_attribution_august_v2 import diagnostic_hac


def test_base_four_regressions():
    tests = runpy.run_path(str(Path(__file__).with_name("test_genetic_primus_attribution_august_v1.py")))
    functions = [v for k, v in tests.items() if k.startswith("test_")]
    assert len(functions) == 4
    for f in functions:
        f()


def test_hac_only_qualified_horizons():
    for n in (3, 91, 92):
        h = diagnostic_hac(np.zeros(n), np.zeros(n), 0)
        assert h["status"] == "descriptive_only_unqualified_horizon" and h["lcb"] is None
    for n in (30, 365):
        h = diagnostic_hac(np.zeros(n), np.zeros(n), 0)
        assert h["status"] == "insufficient_variation" and h["lcb"] is None
    try:
        diagnostic_hac(np.zeros(3), np.zeros(4), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched pairs accepted")
