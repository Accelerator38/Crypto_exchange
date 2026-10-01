"""Small selection gates; no historical data, collector, or network."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.selection_development_v1 import (
    absolute_score, choose_train_winner, fixed_candidates, validation_gate,
)
from exia.genetic_primus.measurement_lab_v2 import block_bank


class DevelopmentSelectionTests(unittest.TestCase):
    def test_budget_is_unique_and_deterministic(self) -> None:
        one = fixed_candidates(24, 20261001)
        self.assertEqual(one, fixed_candidates(24, 20261001))
        self.assertEqual(len(one), len(set(one)))
        self.assertEqual(len(one), 24)

    def test_negative_absolute_lcb_returns_no_trade(self) -> None:
        bank = block_bank(30, 128, 5, 8)
        score = absolute_score(np.full(30, -0.001), np.full(30, -0.002), bank)
        record = {"genome": [0, 0, 1, 1, -1], "training": score}
        self.assertIsNone(choose_train_winner([record]))

    def test_incremental_gain_cannot_override_absolute_loss(self) -> None:
        bank = block_bank(30, 128, 5, 8)
        values = np.full(30, -0.001)
        worse_baseline = np.full(30, -0.002)
        gate = validation_gate(
            values, values, worse_baseline, worse_baseline, bank,
            max_drawdown_normal=0.03, max_drawdown_stress=0.03,
            drawdown_limit=0.10)
        self.assertGreater(gate["normal_paired_lcb_vs_ema_long_only"], 0)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["normal_net_positive"])


if __name__ == "__main__":
    unittest.main()
