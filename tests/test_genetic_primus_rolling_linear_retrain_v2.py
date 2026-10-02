"""Offline regressions for gross labels and position-dependent decisions."""

import unittest

import numpy as np
import pandas as pd

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.measurement_lab_v2 import object_sha
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, FEATURES, HORIZON_MS, _iso
from exia.genetic_primus.rolling_linear_retrain_v2 import (
    build_rolling_schedule, fit_gross_at_origin, schedule_with_position,
)


def _fixed_model(origin: int, gross: float) -> dict:
    model = {
        "schema": "exia.genetic_primus.rolling_linear_model/2",
        "deploy_start_utc": _iso(origin), "feature_names": FEATURES,
        "coefficients": [gross, 0.0, 0.0],
        "scaler": {"median": [0.0, 0.0], "iqr": [1.0, 1.0]},
        "decision_round_trip_cost_bps": 16.0,
    }
    model["model_sha256"] = object_sha(model)
    return model


def _example(origin: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "origin_start": origin, "symbol": "BTC", "proposal_long": True,
        "return_12": 0.0, "efficiency_12": 0.0,
    }])


class GrossDecisionTests(unittest.TestCase):
    def test_gross_fit_includes_losses_and_excludes_unavailable_label(self):
        h = HORIZON_MS
        examples = pd.DataFrame([
            {"origin_start": 0, "origin_end_exclusive": h, "symbol": "BTC",
             "proposal_long": True, "return_12": 0.0, "efficiency_12": 0.0,
             "forward_gross_3d": -0.01},
            {"origin_start": h, "origin_end_exclusive": 2 * h, "symbol": "BTC",
             "proposal_long": True, "return_12": 0.0, "efficiency_12": 0.0,
             "forward_gross_3d": 0.02},
            {"origin_start": 2 * h, "origin_end_exclusive": 3 * h, "symbol": "BTC",
             "proposal_long": True, "return_12": 0.0, "efficiency_12": 0.0,
             "forward_gross_3d": 0.50},
        ])
        model = fit_gross_at_origin(
            examples, 3 * h, "uniform", minimum_opportunities=2,
            minimum_ess=1.0)
        self.assertEqual(model["train_opportunities"], 2)
        self.assertEqual(model["max_train_label_end_exclusive"], 2 * h)
        self.assertEqual(model["negative_outcomes"], 1)
        self.assertAlmostEqual(model["negative_outcome_weight"], 0.5)
        self.assertAlmostEqual(model["coefficients"][0], 0.005)

    def test_entry_and_hold_have_different_economic_thresholds(self):
        origin = HORIZON_MS
        model = _fixed_model(origin, 0.001)
        flat, flat_decision = schedule_with_position(
            _example(origin), ("BTC",), origin, model, {"BTC": 0})
        held, held_decision = schedule_with_position(
            _example(origin), ("BTC",), origin, model, {"BTC": 1})
        self.assertEqual(set(flat.target), {0})
        self.assertEqual(set(held.target), {1})
        fee = 16.0 / 20_000.0
        self.assertAlmostEqual(flat_decision["decisions"]["BTC"]["incremental_threshold"],
                               (1.0 + fee) / (1.0 - fee) - 1.0)
        self.assertEqual(held_decision["decisions"]["BTC"]["incremental_threshold"], 0.0)
        self.assertEqual(flat.model_sha256.iloc[0], held.model_sha256.iloc[0])

    def test_carry_across_retrain_has_one_entry_and_one_exit(self):
        first, decision = schedule_with_position(
            _example(0), ("BTC",), 0, _fixed_model(0, 0.003), {"BTC": 0})
        second, final_decision = schedule_with_position(
            _example(HORIZON_MS), ("BTC",), HORIZON_MS,
            _fixed_model(HORIZON_MS, 0.001), decision["targets"])
        self.assertEqual(final_decision["targets"], {"BTC": 1})
        bars = pd.DataFrame({
            "timestamp": np.arange(0, 2 * HORIZON_MS, DT),
            "symbol": "BTC", "open": 100.0, "close": 100.0,
        })
        schedule = pd.concat([first, second], ignore_index=True)
        simulation = simulate_cash_targets(
            bars, schedule, symbols=("BTC",), start_timestamp=0,
            end_timestamp=2 * HORIZON_MS, timeframe_ms=DT,
            round_trip_cost_bps=16.0)
        self.assertEqual(simulation.metrics["trades"], 1)
        self.assertEqual(simulation.metrics["fills"], 2)
        self.assertEqual(schedule.model_sha256.nunique(), 2)
        self.assertAlmostEqual(simulation.metrics["net_return"],
                               -simulation.metrics["total_fees"])

    def test_orchestrator_refits_and_remembers_position(self):
        h = HORIZON_MS
        examples = pd.DataFrame([{
            "origin_start": i * h, "origin_end_exclusive": (i + 1) * h,
            "symbol": "BTC", "proposal_long": True,
            "return_12": 0.0, "efficiency_12": 0.0,
            "forward_gross_3d": 0.01,
        } for i in range(5)])
        schedule, ledger = build_rolling_schedule(
            examples, ("BTC",), 3 * h, 5 * h, "uniform",
            minimum_opportunities=2, minimum_ess=1.0)
        self.assertEqual(len(ledger), 2)
        self.assertEqual([row["model"]["train_opportunities"] for row in ledger], [2, 3])
        self.assertEqual(ledger[1]["decision"]["decisions"]["BTC"]["position_before"], 1)
        self.assertEqual(schedule.model_sha256.nunique(), 2)
        self.assertEqual(set(schedule.target), {1})
        changed = examples.copy()
        changed.loc[changed.origin_start.ge(3 * h), "forward_gross_3d"] = -99.0
        future_mutated, _ = build_rolling_schedule(
            changed, ("BTC",), 3 * h, 5 * h, "uniform",
            minimum_opportunities=2, minimum_ess=1.0)
        self.assertTrue(schedule.equals(future_mutated))


if __name__ == "__main__":
    unittest.main()
