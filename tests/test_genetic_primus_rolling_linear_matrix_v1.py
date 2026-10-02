"""Pure offline regressions for the four-candidate linear matrix."""

import unittest

import numpy as np
import pandas as pd

from exia.genetic_primus.measurement_lab_v2 import object_sha
from exia.genetic_primus.rolling_linear_matrix_v1 import (
    FOUR_FEATURES, TWO_FEATURES, build_matrix_examples,
    build_matrix_schedule, fit_matrix_at_origin, schedule_matrix_origin,
)
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS, _iso


def _model(origin: int, gross: float, feature_set: str) -> dict:
    names = TWO_FEATURES if feature_set == "two" else FOUR_FEATURES
    model = {
        "schema": "exia.genetic_primus.rolling_linear_matrix_model/1",
        "feature_set": feature_set, "feature_names": names,
        "deploy_start_utc": _iso(origin),
        "coefficients": [gross] + [0.0] * len(names),
        "scaler": {"median": [0.0] * len(names), "iqr": [1.0] * len(names)},
        "decision_round_trip_cost_bps": 16.0,
    }
    model["model_sha256"] = object_sha(model)
    return model


class LinearMatrixTests(unittest.TestCase):
    def test_four_feature_fit_includes_loss_and_purges_future(self):
        h = HORIZON_MS
        examples = pd.DataFrame([{
            "origin_start": i * h, "origin_end_exclusive": (i + 1) * h,
            "symbol": "BTC", "proposal_long": True,
            "return_12": 0.0, "efficiency_12": 0.0,
            "ema_gap_over_atr14": 0.0, "atr_14_pct": 0.0,
            "forward_gross_3d": value,
        } for i, value in enumerate((-0.01, 0.02, 99.0))])
        model = fit_matrix_at_origin(
            examples, 3 * h, "four", "uniform",
            minimum_opportunities=2, minimum_ess=1)
        self.assertEqual(model["train_opportunities"], 2)
        self.assertEqual(model["max_train_label_end_exclusive"], 2 * h)
        self.assertEqual(model["negative_outcomes"], 1)
        self.assertAlmostEqual(model["negative_outcome_weight"], 0.5)
        self.assertAlmostEqual(model["coefficients"][0], 0.005)

    def test_position_dependent_threshold_in_both_feature_sets(self):
        for feature_set in ("two", "four"):
            origin = HORIZON_MS
            row = {"origin_start": origin, "symbol": "BTC",
                   "proposal_long": True, **{name: 0.0 for name in FOUR_FEATURES}}
            examples = pd.DataFrame([row])
            model = _model(origin, 0.001, feature_set)
            flat, _ = schedule_matrix_origin(
                examples, ("BTC",), origin, model, {"BTC": 0})
            held, decision = schedule_matrix_origin(
                examples, ("BTC",), origin, model, {"BTC": 1})
            self.assertEqual(set(flat.target), {0})
            self.assertEqual(set(held.target), {1})
            self.assertEqual(decision["decisions"]["BTC"]["position_before"], 1)

    def test_extra_features_use_only_previous_closed_bar(self):
        times = np.arange(0, 72 * DT, DT)
        opening = 100.0 + np.arange(72) * 0.1
        panel = pd.DataFrame({
            "timestamp": times, "symbol": ["BTC"] * 72,
            "open": opening, "high": opening + 0.2,
            "low": opening - 0.1, "close": opening + 0.05,
            "volume": 100.0,
        })
        original = build_matrix_examples(panel, ("BTC",), 0, 54 * DT, 72 * DT)
        changed = panel.copy()
        changed.loc[changed.timestamp.ge(54 * DT), ["close", "high"]] += 10.0
        future = build_matrix_examples(changed, ("BTC",), 0, 54 * DT, 72 * DT)
        selected = ["origin_start", "symbol", *FOUR_FEATURES, "proposal_long"]
        self.assertTrue(original[selected].equals(future[selected]))
        self.assertFalse(original.forward_gross_3d.equals(future.forward_gross_3d))

    def test_schedule_carries_position_and_ignores_future_labels(self):
        h = HORIZON_MS
        examples = pd.DataFrame([{
            "origin_start": i * h, "origin_end_exclusive": (i + 1) * h,
            "symbol": "BTC", "proposal_long": True,
            **{name: 0.0 for name in FOUR_FEATURES},
            "forward_gross_3d": 0.03,
        } for i in range(5)])
        options = {"minimum_opportunities": 2, "minimum_ess": 1}
        schedule, ledger = build_matrix_schedule(
            examples, ("BTC",), 3 * h, 5 * h, "two", "uniform", **options)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[1]["decision"]["decisions"]["BTC"]["position_before"], 1)
        self.assertEqual(set(schedule.target), {1})
        changed = examples.copy()
        changed.loc[changed.origin_start.ge(3 * h), "forward_gross_3d"] = -99.0
        replay, models = build_matrix_schedule(
            changed, ("BTC",), 3 * h, 5 * h, "two", "uniform", **options)
        self.assertTrue(schedule.equals(replay))
        self.assertEqual([x["model"]["model_sha256"] for x in ledger],
                         [x["model"]["model_sha256"] for x in models])


if __name__ == "__main__":
    unittest.main()
