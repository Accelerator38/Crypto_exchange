"""Small direct-call regression suite; no pytest or collector execution."""
import ast
from dataclasses import replace
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus_statistics.hac_adapter_v1 import (
    DAY_MS, assess_paired_daily, complete_daily_returns, diagnostic_gate, qualified_lcb_matrix,
)
from exia.genetic_primus.measurement_lab_v2 import block_bank, daily_returns


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def test_exact_qualified_formula_parity():
    reference = runpy.run_path(str(ROOT / "tools/run_genetic_primus_validation_ci_v1.py"))
    for days in (30, 365):
        paths = np.random.default_rng(9).normal(0, .01, (4, days))
        bank = block_bank(days, 32, int(np.ceil(np.sqrt(days))), 8)
        expected, _ = reference["ci_bounds"](paths, ["hac_small_sample_t"], bank, .025)
        np.testing.assert_array_equal(qualified_lcb_matrix(paths), expected["hac_small_sample_t"])


def test_paired_assessment_and_uninformative_abstention():
    times = np.arange(30, dtype=np.int64) * DAY_MS
    values = np.linspace(-.01, .02, 30)
    result = assess_paired_daily(values, np.zeros(30), times)
    assert result.status == "informative" and result.degrees_of_freedom == 2
    assert abs(result.mean_difference - values.mean()) < 1e-14
    assert len(result.observations_sha256) == 64
    equal = assess_paired_daily(values, values, times)
    assert equal.status == "insufficient_variation" and equal.lcb is None
    assert not equal.promotion_authority


def test_outside_horizon_bad_pairing_and_nonfinite_fail_closed():
    times = np.arange(30, dtype=np.int64) * DAY_MS
    rejects(lambda: assess_paired_daily(np.zeros(31), np.zeros(31), np.arange(31, dtype=np.int64) * DAY_MS))
    rejects(lambda: assess_paired_daily(np.zeros(30), np.zeros(29), times))
    rejects(lambda: assess_paired_daily(np.full(30, np.nan), np.zeros(30), times))
    bad = times.copy()
    bad[-1] += 1
    rejects(lambda: assess_paired_daily(np.zeros(30), np.zeros(30), bad))


def portfolio_fixture():
    dt = DAY_MS // 6
    start = 20 * 3_600_000
    end = start + 30 * DAY_MS
    times = np.arange(start + dt, end + dt, dt, dtype=np.int64)
    returns = np.linspace(-.001, .002, len(times))
    equity = np.cumprod(1.0 + returns)
    return times, returns, equity, start, end, dt


def test_complete_daily_compounding_matches_legacy():
    times, returns, equity, start, end, dt = portfolio_fixture()
    values, closes = complete_daily_returns(times, returns, equity, start=start, end=end, timeframe_ms=dt)
    old = daily_returns(SimpleNamespace(portfolio=pd.DataFrame({"timestamp": times, "net_return": returns})))
    np.testing.assert_allclose(values, old, rtol=0, atol=1e-14)
    assert len(values) == 30 and closes[0] == start + DAY_MS and closes[-1] == end
    np.testing.assert_allclose(np.prod(1.0 + values), equity[-1], rtol=0, atol=1e-12)


def test_missing_bar_bad_equity_and_shifted_close_are_rejected():
    times, returns, equity, start, end, dt = portfolio_fixture()
    kwargs = {"start": start, "end": end, "timeframe_ms": dt}
    rejects(lambda: complete_daily_returns(times[:-1], returns[:-1], equity[:-1], **kwargs))
    rejects(lambda: complete_daily_returns(times + 1, returns, equity, **kwargs))
    bad = equity.copy()
    bad[-1] += .001
    rejects(lambda: complete_daily_returns(times, returns, bad, **kwargs))


def test_gate_only_changes_statistics_not_authority():
    times = np.arange(30, dtype=np.int64) * DAY_MS
    strong = assess_paired_daily(np.tile([.01, .02], 15), np.zeros(30), times)
    weak = assess_paired_daily(np.zeros(30), np.zeros(30), times)
    positive = diagnostic_gate(strong, strong, .05)
    assert positive["passed"] and not positive["promotion_authority"] and not positive["runtime_authority"]
    assert not diagnostic_gate(strong, strong, .101)["passed"]
    assert not diagnostic_gate(strong, weak, .05)["passed"]
    assert not diagnostic_gate(replace(strong, lcb=float("nan")), strong, .05)["passed"]


def test_frozen_metric_comparison_and_no_fit_calls():
    path = ROOT / "tools/run_genetic_primus_frozen_hac_review_v1.py"
    runner = runpy.run_path(str(path))
    runner["assert_same"]({"x": .1, "safety": False}, {"x": .1, "safety": False}, 1e-12)
    rejects(lambda: runner["assert_same"]({"x": .2}, {"x": .1}, 1e-12))
    rejects(lambda: runner["assert_same"]({"safety": True}, {"safety": False}, 1e-12))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(fn, ast.Attribute) and fn.attr in {"fit", "train", "search"} for fn in calls)
    assert not any(isinstance(fn, ast.Name) and fn.id in {"fit", "train", "search"} for fn in calls)
    config, _ = runner["load_contract"]()
    assert config["training_runs"] == 0 and not any(config["safety"].values())
