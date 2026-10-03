"""Synthetic future-screen regressions; no historical market data or collector."""

import copy
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.future_ema_evaluator_v1 import (
    DT, DAY, HORIZON, ema_schedule, evaluate_registered_panel,
    utc_ms, validate_market_grid, viability_gate,
)
from exia.genetic_primus.measurement_lab_v2 import block_bank
from tools.run_genetic_primus_future_ema_viability_v1 import commit_segment, evaluate_once


def market(start, end, symbols):
    times = np.arange(start, end, DT)
    index = np.arange(len(times))
    opening = 100 * np.exp((index - 0.25) * 0.0005)
    closing = 100 * np.exp(index * 0.0005)
    count = len(symbols)
    return pd.DataFrame({"timestamp": np.repeat(times, count),
                         "symbol": np.tile(symbols, len(times)),
                         "open": np.repeat(opening, count),
                         "high": np.repeat(closing * 1.001, count),
                         "low": np.repeat(opening * 0.999, count),
                         "close": np.repeat(closing, count), "volume": 100.0})


class FutureEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/genetic_primus_future_ema_viability_v1.json")
                                 .read_text(encoding="utf-8"))

    def test_grid_rejects_gaps_duplicates_and_impossible_ohlc(self):
        panel = market(100 * DT, 110 * DT, ("A", "B"))
        self.assertEqual(len(validate_market_grid(panel, ("A", "B"), 100 * DT, 110 * DT)), 20)
        impossible = panel.copy()
        impossible.loc[0, "high"] = impossible.loc[0, "low"] / 2
        for bad in (panel.iloc[:-1], pd.concat([panel.iloc[:-1], panel.iloc[:1]]), impossible):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_market_grid(bad, ("A", "B"), 100 * DT, 110 * DT)

    def test_future_price_mutation_does_not_change_earlier_signal(self):
        first, end = 120 * DT, 156 * DT
        panel = market(0, end, ("A", "B"))
        schedule, ledger = ema_schedule(panel, ("A", "B"), first, end, "fixed")
        mutated = panel.copy()
        changed = mutated.timestamp.ge(first + HORIZON)
        mutated.loc[changed, ["open", "high", "low", "close"]] *= 0.5
        replay, other = ema_schedule(mutated, ("A", "B"), first, end, "fixed")
        self.assertEqual(ledger, other)
        self.assertTrue(schedule.equals(replay))
        # Four positive symbol-decisions are only two distinct signal origins.
        self.assertEqual(sum(any(d["proposal_long"] for d in x["decisions"].values())
                             for x in ledger), 2)

    def test_final_analysis_and_commit_refuse_early_without_opening_files(self):
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        with patch("tools.run_genetic_primus_future_ema_viability_v1._read_segment") as read, \
             patch("tools.run_genetic_primus_future_ema_viability_v1._require_git_commit") as git:
            with self.assertRaises(ValueError):
                commit_segment(self.config, Path("nonexistent.parquet"), now)
            with self.assertRaises(ValueError):
                evaluate_once(self.config, now)
            read.assert_not_called()
            git.assert_not_called()

    def test_single_profitable_origin_is_not_a_viable_strategy(self):
        daily = np.zeros(180)
        daily[10] = 0.1
        metrics = {"daily_returns": daily.tolist(), "net_return": 0.1, "max_drawdown": 0.0}
        bank = block_bank(180, 64, 5, 123)
        gate = viability_gate({"normal": metrics, "execution_stress": metrics}, 20,
                              bank, bank, alpha=0.05, minimum_signal_origins=20,
                              drawdown_limit=0.1, concentration_limit=0.4)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["conditions"]["stress_origin_concentration"])
        self.assertFalse(gate["conditions"]["stress_four_positive_outer_slices"])
        self.assertEqual(gate["stress_positive_origin_concentration"], 1.0)

    def test_full_registered_synthetic_pipeline_and_signal_origin_gate(self):
        c = self.config
        join = utc_ms(c["known_revealed_end_exclusive_utc"])
        end = utc_ms(c["future_end_exclusive_utc"])
        panel = market(join - 100 * DT, end, tuple(c["universe"]))
        result = evaluate_registered_panel(c, panel.loc[panel.timestamp.lt(join)],
                                           panel.loc[panel.timestamp.ge(join)])
        self.assertEqual(len(result["origin_ledger"]), 60)
        self.assertEqual(result["gate"]["distinct_signal_origins"], 60)
        self.assertEqual(len(result["gate"]["outer_slice_returns"]["execution_stress"]), 6)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["scenarios"]["normal"]["closed_trades"], 8)
        self.assertEqual(result["scenarios"]["execution_stress"]["closed_trades"], 8)
        self.assertEqual(result["final_policy"], "NoTrade")
        self.assertFalse(result["promotion_authority"])
        # Even a profitable synthetic path cannot pass with only 19 independent origins.
        b = c["bootstrap"]
        primary = block_bank(180, 64, 5, b["primary_seed"])
        audit = block_bank(180, 64, 5, b["audit_seed"])
        gate = viability_gate(result["scenarios"], 19, primary, audit, alpha=0.05,
                              minimum_signal_origins=20, drawdown_limit=0.1,
                              concentration_limit=0.4)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["conditions"]["distinct_signal_origins"])


if __name__ == "__main__":
    unittest.main()
