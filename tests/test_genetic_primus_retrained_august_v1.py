"""Small deterministic isolation/budget regressions; callable without pytest."""
from copy import deepcopy
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "tools/run_genetic_primus_retrained_august_v1.py"))


def test_temporal_separation_and_window_lengths():
    c, _ = API["load_config"]()
    previous = None
    for i in range(10):
        b = API["bounds"](c, i)
        assert b["train_end"] - b["train_start"] == 365 * 86_400_000
        assert b["validation_end"] - b["validation_start"] == 30 * 86_400_000
        assert b["validation_start"] - b["train_end"] == 7 * 14_400_000
        assert b["test_start"] - b["validation_end"] == 7 * 14_400_000
        assert b["test_end"] - b["test_start"] == 3 * 86_400_000
        if previous:
            assert previous["test_end"] == b["test_start"]
        previous = b


def test_future_price_changes_do_not_change_training_features_or_scaler():
    dt = 14_400_000
    times = np.arange(400, dtype=np.int64) * dt
    price = 100 + np.arange(400) * 0.03 + np.sin(np.arange(400) / 7)
    panel = pd.DataFrame({"timestamp": times, "symbol": "BTC/USDT", "open": price,
                          "close": price + 0.01, "high": price + 1, "low": price - 1,
                          "volume": 100 + np.arange(400)})
    end = 300 * dt
    altered = panel.copy()
    altered.loc[altered.timestamp >= end, ["open", "close", "high", "low", "volume"]] = 1e12
    first = API["window"](panel, ("BTC/USDT",), 100 * dt, end, dt)
    second = API["window"](altered, ("BTC/USDT",), 100 * dt, end, dt)
    np.testing.assert_array_equal(first.values, second.values)
    assert API["LabScaler"].fit(first.values) == API["LabScaler"].fit(second.values)


def test_search_budget_unique_genomes_and_determinism():
    calls = []

    def fitness(genome):
        calls.append(genome)
        return -float(sum(abs(w) for w in genome[:4]))

    first = API["search"]("genetic_search", fitness, 24, 20260907, 6)
    assert len(calls) == len(set(calls)) == 24
    assert all(np.count_nonzero(g[:4]) <= 2 for g in calls)
    second = API["search"]("genetic_search", fitness, 24, 20260907, 6)
    assert first == second


def test_screen_is_strict_and_fails_closed_on_uninformative_lcb():
    c, _ = API["load_config"]()
    v = {cost: {"hac": {ref: {"lcb": 0.001} for ref in ("EMA", "NoTrade")},
                "metrics": {"max_drawdown": 0.02, "trades": 10}}
         for cost in ("normal", "stress")}
    assert API["screen"](v, c)["passed"]
    bad = deepcopy(v)
    bad["stress"]["hac"]["EMA"]["lcb"] = None
    assert not API["screen"](bad, c)["passed"]
    bad["stress"]["hac"]["EMA"]["lcb"] = 0.0
    assert not API["screen"](bad, c)["passed"]
    assert API["screen"](v, c)["promotion_authority"] is False
