"""Deterministic direct-call regressions. No collector, network or pytest needed."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from exia.genetic_primus.measurement_lab_v2 import (
    ALLOW_ALL, CATALOG, DAY_MS, FEATURES, NO_TRADE, LabScaler, PreparedWindow,
    array_sha, block_bank, build_lab_features, daily_returns, object_sha,
    paired_lcb, search, symmetric_rank, temporal_bounds, validation_gate,
)

DT = DAY_MS // 6


def fixture(kind: str, days: int = 12) -> PreparedWindow:
    count = days * 6
    price = 100.0
    rows = []
    for i in range(-1, count):
        good = (i // 6) % 2 == 0
        growth = 0.003 if kind == "positive" or (kind == "filter" and good) else 0.0
        if kind == "filter" and not good:
            growth = -0.003
        close = price * (1.0 + growth)
        # Feature at candle i is the state available for execution at i+1, not a future return.
        next_good = ((i + 1) // 6) % 2 == 0
        rows.append({"timestamp": i * DT, "symbol": "A", "open": price, "close": close,
                     "ema_gap_12_48": 0.1, "ema_slope_12_3": 0.1, "return_12": 0.1,
                     "efficiency_12": 0.9 if next_good else 0.35})
        price = close
    return PreparedWindow.from_frame(pd.DataFrame(rows), ("A",), 0, count * DT, DT)


def test_symmetric_rank_full8_ties_and_reflection():
    values = pd.Series(np.arange(8, dtype=float))
    rank = symmetric_rank(values)
    np.testing.assert_allclose(rank, [-.875, -.625, -.375, -.125, .125, .375, .625, .875])
    assert (rank > .5).sum() == (rank < -.5).sum() == 2
    np.testing.assert_allclose(symmetric_rank(-values), -rank)
    np.testing.assert_allclose(symmetric_rank(pd.Series([4., 4., 4.])), 0.0)
    assert symmetric_rank(pd.Series([4.])).iloc[0] == 0


def test_temporal_boundaries_and_train_only_scaler():
    bounds = temporal_bounds(300 * DAY_MS, 150, 30, 1, DT)
    assert bounds["train_end"] + DT == bounds["validation_start"]
    assert bounds["validation_end"] + DT == bounds["development_start"]
    assert bounds["development_end"] - bounds["development_start"] == 3 * DAY_MS
    values = np.arange(80, dtype=float).reshape(-1, 4)
    scaler = LabScaler.fit(values)
    before = object_sha(scaler.mapping())
    scaler.transform(np.full((3, 4), 1e12))
    assert object_sha(scaler.mapping()) == before
    assert scaler.fit_values_sha256 == array_sha(values)


def test_zero_pnl_has_no_evidence_and_does_not_force_trading():
    window = fixture("null")
    scaler = LabScaler.fit(window.values)
    empty = window.simulate(NO_TRADE, scaler, 16)
    bank = block_bank(len(daily_returns(empty)), 64, 3, 17)
    gate = validation_gate(empty, empty, bank)
    assert not gate["passed"] and "no_nonzero_return_evidence" in gate["reasons"]
    assert empty.metrics["trades"] == 0


def test_all_methods_equal_budget_null_and_positive_controls():
    for kind in ("null", "positive"):
        window = fixture(kind)
        scaler = LabScaler.fit(window.values)
        bank = block_bank(12, 64, 3, 21)
        cache = {}
        def evaluate(genome):
            if genome not in cache:
                returns = daily_returns(window.simulate(genome, scaler, 16))
                cache[genome] = paired_lcb(returns, np.zeros_like(returns), bank)
            return cache[genome]
        for method in ("sparse_grid", "random_search", "genetic_search"):
            result = search(method, evaluate, 24, 20260828)
            assert result["unique_evaluations"] == 24
            assert len({tuple(r["genome"]) for r in result["evaluations"]}) == 24
            if kind == "null":
                assert result["winner"] == NO_TRADE and result["score"] == 0
            else:
                assert result["score"] > 0
            assert result == search(method, evaluate, 24, 20260828)


def test_known_filter_signal_beats_unfiltered_proposal():
    window = fixture("filter")
    scaler = LabScaler.fit(window.values)
    filtered = window.simulate((0, 0, 0, 1, 0), scaler, 16)
    plain = window.simulate(ALLOW_ALL, scaler, 16)
    assert filtered.metrics["net_return"] > plain.metrics["net_return"]
    bank = block_bank(12, 64, 3, 35)
    values = daily_returns(filtered)
    assert paired_lcb(values, np.zeros_like(values), bank) > 0


def test_same_bank_pairing_and_catalog_complexity():
    bank = block_bank(30, 256, 5, 31)
    assert array_sha(bank) == array_sha(block_bank(30, 256, 5, 31))
    assert array_sha(bank) != array_sha(block_bank(30, 256, 5, 32))
    series = np.arange(30) * .001
    assert paired_lcb(series, series, bank) == 0
    assert len(CATALOG) == len(set(CATALOG)) == 98
    assert all(np.count_nonzero(g[:4]) <= 2 for g in CATALOG)


def test_daily_blocks_do_not_split_at_utc_midnight():
    closes = np.arange(12) * DT + 20 * 3_600_000
    simulation = SimpleNamespace(portfolio=pd.DataFrame({"timestamp": closes, "net_return": .001}))
    values = daily_returns(simulation)
    assert len(values) == 2
    np.testing.assert_allclose(values, np.full(2, 1.001 ** 6 - 1))


def test_decision_uses_previous_candle_and_missing_features_abstain():
    window = fixture("positive", 2)
    scaler = LabScaler.fit(window.values)
    result = window.simulate(ALLOW_ALL, scaler, 16, details=True)
    assert result.decisions.model_sha256.nunique() == 1
    assert result.decisions.model_sha256.iloc[0] == object_sha(window.model(ALLOW_ALL, scaler))
    assert (result.decisions.signal_timestamp <= result.decisions.execution_timestamp).all()
    before = result.decisions.iloc[0].copy()
    changed = window.values.copy()
    changed[-1, :] = np.nan
    after = replace(window, values=changed).simulate(ALLOW_ALL, scaler, 16, details=True)
    pd.testing.assert_series_equal(before, after.decisions.iloc[0])
    assert after.decisions.iloc[-1].desired_position == 0


def test_feature_builder_prefix_invariance():
    rows = []
    for symbol, shift in (("A", 0), ("B", 3)):
        for i in range(220):
            close = 100 + i * .1 + np.sin(i / 7 + shift)
            rows.append({"timestamp": i * DT, "symbol": symbol, "open": close * .999,
                         "high": close * 1.01, "low": close * .99, "close": close,
                         "volume": 1000 + i})
    panel = pd.DataFrame(rows)
    prefix = panel[panel.timestamp < 200 * DT].copy()
    full = build_lab_features(panel)
    past = build_lab_features(prefix)
    columns = ["timestamp", "symbol", *FEATURES, "relative_momentum_rank"]
    expected = full[full.timestamp < 200 * DT][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, past[columns].reset_index(drop=True))
