"""Small offline checks of the six predeclared counterfactual schedules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import RANDOM_SEEDS, targets_for_policy
from exia.genetic_primus.accounting_v2 import simulate_cash_targets

DT = 4 * 60 * 60 * 1000


class AblationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = pd.DataFrame({
            "execution_timestamp": [i * DT for i in range(4)],
            "signal_timestamp": [i * DT for i in range(4)],
            "symbol": ["BTC/USDT"] * 4,
            "origin_id": ["origin_01"] * 4,
            "proposal": [1, 1, 1, 0],
            "ga_action": [1, 0, 1, 0],
            "ema_model_sha256": ["ema"] * 4,
            "ga_model_sha256": ["ga"] * 4,
        })

    def test_entry_gate_holds_until_shared_ema_exit(self) -> None:
        gate = targets_for_policy(self.base, "GA_target_gate")
        entry = targets_for_policy(self.base, "GA_entry_only")
        self.assertEqual(gate["target"].tolist(), [1, 0, 1, 0])
        self.assertEqual(entry["target"].tolist(), [1, 1, 1, 0])
        bars = self.base[["execution_timestamp", "symbol"]].rename(
            columns={"execution_timestamp": "timestamp"})
        bars["open"] = 100.0
        bars["close"] = 100.0
        def replay(schedule: pd.DataFrame):
            return simulate_cash_targets(
                bars, schedule, symbols=("BTC/USDT",), start_timestamp=0,
                end_timestamp=4 * DT, timeframe_ms=DT,
                round_trip_cost_bps=16.0)
        self.assertEqual(replay(gate).metrics["fills"], 4)
        self.assertEqual(replay(entry).metrics["fills"], 2)

    def test_random_matching_is_exact_and_seeded(self) -> None:
        for seed in RANDOM_SEEDS:
            first = targets_for_policy(self.base, "EMA_long_random_matched", seed=seed)
            second = targets_for_policy(self.base, "EMA_long_random_matched", seed=seed)
            self.assertEqual(first["target"].tolist(), second["target"].tolist())
            self.assertEqual(int(first["target"].sum()), 2)
            self.assertEqual(int(first["target"].iloc[-1]), 0)


if __name__ == "__main__":
    unittest.main()
