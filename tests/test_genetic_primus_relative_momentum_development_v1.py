"""Causality and fail-closed unit checks; no snapshots, network or collector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.relative_momentum_development_v1 import (
    MODEL_SHA256, origin_schedule, select_at_origin,
)
from exia.genetic_primus.accounting_v2 import simulate_cash_targets

DT = 240 * 60_000
ORIGIN = 19 * DT
SYMBOLS = ("A", "B", "C")


def _panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": timestamp, "symbol": symbol, "close": price}
        for timestamp, prices in ((0, (100, 100, 100)),
                                  (18 * DT, (110, 110, 90)))
        for symbol, price in zip(SYMBOLS, prices)
    ])


class RelativeMomentumDevelopmentTests(unittest.TestCase):
    def test_lookback_ranking_and_tie_break_are_fixed(self) -> None:
        self.assertEqual(select_at_origin(_panel(), ORIGIN, SYMBOLS, DT), ("A", "B"))

    def test_future_candle_cannot_change_origin_selection(self) -> None:
        future = pd.concat([_panel(), pd.DataFrame([
            {"timestamp": ORIGIN, "symbol": "C", "close": 1000000},
        ])], ignore_index=True)
        self.assertEqual(select_at_origin(future, ORIGIN, SYMBOLS, DT), ("A", "B"))

    def test_missing_causal_bar_fails_closed(self) -> None:
        panel = _panel().loc[lambda frame: ~((frame.timestamp == 0) & (frame.symbol == "C"))]
        with self.assertRaisesRegex(ValueError, "causal ranking candle"):
            select_at_origin(panel, ORIGIN, SYMBOLS, DT)

    def test_schedule_is_three_days_causal_and_uses_one_model_sha(self) -> None:
        schedule = origin_schedule(ORIGIN, ORIGIN + 3 * 86_400_000, DT,
                                   SYMBOLS, ("A", "B"))
        self.assertEqual(len(schedule), 18 * len(SYMBOLS))
        self.assertEqual(schedule.groupby("symbol").target.sum().to_dict(),
                         {"A": 18, "B": 18, "C": 0})
        self.assertTrue((schedule.signal_timestamp <= schedule.execution_timestamp).all())
        self.assertEqual(schedule.model_sha256.unique().tolist(), [MODEL_SHA256])

    def test_same_selection_carries_position_across_origin(self) -> None:
        end = ORIGIN + 6 * 86_400_000
        bars = pd.DataFrame([
            {"timestamp": time, "symbol": symbol, "open": 100.0, "close": 100.0}
            for time in range(ORIGIN, end, DT) for symbol in SYMBOLS
        ])
        schedule = pd.concat([
            origin_schedule(ORIGIN, ORIGIN + 3 * 86_400_000, DT,
                            SYMBOLS, ("A", "B")),
            origin_schedule(ORIGIN + 3 * 86_400_000, end, DT,
                            SYMBOLS, ("A", "B")),
        ], ignore_index=True)
        simulation = simulate_cash_targets(
            bars, schedule, symbols=SYMBOLS, start_timestamp=ORIGIN,
            end_timestamp=end, timeframe_ms=DT,
            round_trip_cost_bps=16.0, record_details=False)
        self.assertEqual(simulation.metrics["trades"], 2)


if __name__ == "__main__":
    unittest.main()
