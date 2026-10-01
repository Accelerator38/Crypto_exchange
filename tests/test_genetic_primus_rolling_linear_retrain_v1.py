"""Causal label, purge and loss-weight checks without market or collector I/O."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.rolling_linear_retrain_v1 import (
    DT, HORIZON_MS, fit_at_origin, schedule_at_origin,
)

DEPLOY = 30 * HORIZON_MS
SYMBOLS = ("A", "B")


def examples() -> pd.DataFrame:
    rows = []
    for index in range(25):
        start = index * HORIZON_MS
        for symbol in SYMBOLS:
            rows.append({"origin_start": start,
                         "origin_end_exclusive": start + HORIZON_MS,
                         "symbol": symbol,
                         "return_12": 0.01 if symbol == "A" else -0.01,
                         "efficiency_12": 0.7,
                         "proposal_long": True,
                         "forward_gross_3d": -0.02 if symbol == "A" else 0.03})
    for symbol in SYMBOLS:
        rows.append({"origin_start": DEPLOY, "origin_end_exclusive": DEPLOY + HORIZON_MS,
                     "symbol": symbol, "return_12": 0.01, "efficiency_12": 0.7,
                     "proposal_long": True, "forward_gross_3d": 1.0})
    return pd.DataFrame(rows)


class RollingLinearRetrainTests(unittest.TestCase):
    def test_future_outcomes_cannot_change_model_and_losses_keep_weight(self) -> None:
        data = examples()
        before = fit_at_origin(data, DEPLOY, "half_life_90d_anchor_20pct",
                               minimum_opportunities=2, minimum_ess=1)
        altered = data.copy()
        altered.loc[altered.origin_start.eq(DEPLOY), "forward_gross_3d"] = -100
        after = fit_at_origin(altered, DEPLOY, "half_life_90d_anchor_20pct",
                              minimum_opportunities=2, minimum_ess=1)
        self.assertEqual(before["model_sha256"], after["model_sha256"])
        self.assertLessEqual(before["max_train_label_end_exclusive"], DEPLOY - 7 * DT)
        self.assertGreater(before["negative_outcome_weight"], 0)
        self.assertGreater(before["positive_outcome_weight"], 0)
        self.assertAlmostEqual(before["weight_sum"], 1)

    def test_purge_excludes_recent_finished_label(self) -> None:
        data = examples()
        first = fit_at_origin(data, DEPLOY, "uniform", minimum_opportunities=2,
                              minimum_ess=1)
        recent = pd.DataFrame([{"origin_start": DEPLOY - HORIZON_MS,
                                "origin_end_exclusive": DEPLOY - 6 * DT,
                                "symbol": "A", "return_12": 1.0,
                                "efficiency_12": 0.7, "proposal_long": True,
                                "forward_gross_3d": 1000.0}])
        second = fit_at_origin(pd.concat([data, recent], ignore_index=True), DEPLOY,
                               "uniform", minimum_opportunities=2, minimum_ess=1)
        self.assertEqual(first["model_sha256"], second["model_sha256"])

    def test_every_decision_carries_retrained_model_sha(self) -> None:
        data = examples()
        model = fit_at_origin(data, DEPLOY, "uniform", minimum_opportunities=2,
                              minimum_ess=1)
        schedule, _ = schedule_at_origin(data, SYMBOLS, DEPLOY, model)
        self.assertEqual(len(schedule), 18 * len(SYMBOLS))
        self.assertEqual(schedule.model_sha256.unique().tolist(), [model["model_sha256"]])
        self.assertTrue((schedule.signal_timestamp <= schedule.execution_timestamp).all())


if __name__ == "__main__":
    unittest.main()
