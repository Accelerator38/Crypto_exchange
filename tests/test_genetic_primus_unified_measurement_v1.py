"""Deterministic synthetic measurement regressions; no pytest/collector needed."""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import unittest

from exia.genetic_primus.unified_measurement_v1 import (
    BAR_MS, HORIZON_MS, CostScenario, MeasurementSpec, build_ranking_key,
    as_primus_simulation, compare_policies, complete_origins, daily_measurements,
    identity, label_binding, make_decision, replay,
)


def fixture(weights=("0.5", "0.5"), *, symbols=("X",), tail_bars=0):
    terminal = len(weights) * HORIZON_MS + BAR_MS
    end = terminal + tail_bars * BAR_MS
    bars = [{"timestamp": i * BAR_MS, "prices": {s: "100" for s in symbols}}
            for i in range(terminal // BAR_MS + 1)]
    spec = MeasurementSpec(0, end, terminal, "100", symbols, identity(bars))
    decisions = []
    for i, weight in enumerate(weights):
        targets = weight if isinstance(weight, dict) else {s: weight for s in symbols}
        decisions.append(make_decision(
            origin_start_ms=i * HORIZON_MS, decision_asof_ms=i * HORIZON_MS,
            targets=targets, model_sha256=identity({"synthetic_model": i}),
            scaler_sha256=identity({"synthetic_scaler": i}),
            feature_prefix_sha256=identity({"synthetic_prefix": i}),
            training_identity_sha256=identity({"synthetic_training": i}),
        ))
    funding = {"complete": True, "coverage_start_ms": 0, "coverage_end_ms": end,
               "symbols": list(symbols), "provenance": "synthetic", "events": [],
               "evidence_sha256": identity({"synthetic_funding": "complete_empty"})}
    costs = CostScenario("zero", "0", "0", "0", "0", "synthetic", identity({"synthetic_costs": 0}))
    return spec, bars, decisions, funding, costs


def execute(data):
    spec, bars, decisions, funding, costs = data
    return replay(spec, price_bars=bars, decisions=decisions, funding=funding, costs=costs)


def register_prices(data):
    return (replace(data[0], market_inputs_sha256=identity(data[1])), *data[1:])


def d(value):
    return Decimal(str(value))


class UnifiedMeasurementTests(unittest.TestCase):
    def test_august_september_has_twenty_complete_origins(self):
        start = int(datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
        windows = complete_origins(start, end)
        self.assertEqual(len(windows), 20)
        self.assertEqual(windows[-1]["planned_exit_open_ms"], end - 20 * 3600000)
        self.assertTrue(all(w["planned_exit_open_ms"] - w["planned_entry_open_ms"] == HORIZON_MS for w in windows))
        self.assertTrue(all(a["planned_exit_open_ms"] == b["planned_entry_open_ms"] for a, b in zip(windows, windows[1:])))

    def test_execution_delay_boundary_keeps_eighteen_bars(self):
        a = complete_origins(0, 40 * BAR_MS, decision_offset_ms=BAR_MS - 1, execution_delay_ms=0)[0]
        b = complete_origins(0, 40 * BAR_MS, decision_offset_ms=BAR_MS - 1, execution_delay_ms=1)[0]
        self.assertEqual(b["planned_entry_open_ms"] - a["planned_entry_open_ms"], BAR_MS)
        self.assertEqual(b["planned_exit_open_ms"] - b["planned_entry_open_ms"], HORIZON_MS)

    def test_no_trade_keeps_capital_and_has_no_fills(self):
        result = execute(fixture(("0", "0")))
        self.assertEqual(d(result["metrics"]["final_capital_quote"]), 100)
        self.assertEqual(result["metrics"]["fills"], 0)
        self.assertEqual(result["metrics"]["trades"], 0)
        self.assertTrue(all(d(row["exposure_quote"]) == 0 for row in result["equity"]))
        self.assertTrue(all(value is False for value in result["safety"].values()))

    def test_unchanged_target_carries_quantity_across_model_change(self):
        result = execute(fixture())
        self.assertEqual(result["metrics"]["fills"], 2)
        self.assertEqual(result["metrics"]["trades"], 1)
        self.assertEqual(result["closed_trades"][0]["holding_bars"], 36)
        self.assertEqual(result["decisions"][0]["quantities_after"], result["decisions"][1]["quantities_after"])
        self.assertNotEqual(result["closed_trades"][0]["entry_model_sha256"], result["closed_trades"][0]["exit_model_sha256"])

    def test_transition_to_cash_closes_once(self):
        result = execute(fixture(("0.5", "0")))
        self.assertEqual(result["closed_trades"][0]["holding_bars"], 18)
        self.assertEqual(result["closed_trades"][0]["reason"], "target_transition")
        self.assertEqual(result["metrics"]["fills"], 2)

    def test_changed_positive_weight_has_explicit_exit_and_entry(self):
        result = execute(fixture(("0.5", "0.25")))
        self.assertEqual(result["metrics"]["fills"], 4)
        self.assertEqual(result["metrics"]["trades"], 2)

    def test_later_trade_uses_updated_shared_capital(self):
        data = fixture(("0.5", "0", "0.5"))
        for row in data[1]:
            if 19 * BAR_MS <= row["timestamp"] < 37 * BAR_MS or row["timestamp"] == 55 * BAR_MS:
                row["prices"]["X"] = "110"
        result = execute(register_prices(data))
        self.assertEqual(d(result["metrics"]["final_capital_quote"]), d("110.25"))
        buys = [row for row in result["ledger"] if row["kind"] == "buy"]
        self.assertEqual(d(buys[1]["allocated_quote"]), d("52.5"))

    def test_per_fill_costs_reconcile_independently(self):
        data = list(fixture())
        data[4] = CostScenario("normal", "6", "6", "2", "2", "synthetic", identity({"normal": 16}))
        result = execute(data)
        buys = [row for row in result["ledger"] if row["kind"] == "buy"]
        q = d(buys[0]["quantity"])
        self.assertLess(abs(q * d("100.02") * d("1.0006") - 50), d("1e-25"))
        metrics = result["metrics"]
        expected = 100 - d(metrics["fees_quote"]) - d(metrics["adverse_drag_quote"])
        self.assertLess(abs(d(metrics["final_capital_quote"]) - expected), d("1e-25"))
        self.assertGreater(d(metrics["max_drawdown"]), 0)

    def test_signed_funding_debits_and_credits_long(self):
        data = fixture()
        data[3]["events"] = [
            {"timestamp": 9 * BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"},
            {"timestamp": 10 * BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "-4"},
        ]
        result = execute(data)
        self.assertEqual(d(result["metrics"]["funding_quote"]), d("0.03"))
        self.assertEqual(d(result["metrics"]["final_capital_quote"]), d("99.97"))

    def test_funding_at_entry_excluded_and_exit_included(self):
        data = fixture(("0.5", "0"))
        data[3]["events"] = [
            {"timestamp": BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"},
            {"timestamp": 19 * BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"},
        ]
        result = execute(data)
        self.assertEqual(d(result["metrics"]["funding_quote"]), d("0.05"))
        at_exit = [row["kind"] for row in result["ledger"] if row["timestamp"] == 19 * BAR_MS]
        self.assertEqual(at_exit, ["funding", "sell"])

    def test_non_bar_funding_time_is_processed_before_next_mark(self):
        data = fixture()
        data[3]["events"] = [{"timestamp": BAR_MS + 1, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"}]
        result = execute(data)
        self.assertEqual(d(result["metrics"]["funding_quote"]), d("0.05"))

    def test_funding_cannot_invent_borrowing(self):
        data = fixture(("1", "1"))
        data[3]["events"] = [{"timestamp": 2 * BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"}]
        with self.assertRaisesRegex(ValueError, "cash reserve"):
            execute(data)

    def test_unknown_funding_is_not_zero_funding(self):
        data = fixture()
        data[3]["complete"] = False
        with self.assertRaisesRegex(ValueError, "funding evidence"):
            execute(data)

    def test_duplicate_funding_cannot_double_charge(self):
        data = fixture()
        event = {"timestamp": 2 * BAR_MS, "symbol": "X", "mark_price": "100", "signed_rate_bps": "10"}
        data[3]["events"] = [event, event.copy()]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            execute(data)

    def test_shared_universe_does_not_multiply_initial_capital(self):
        result = execute(fixture(({"X": "0.5", "Y": "0.5"},), symbols=("X", "Y")))
        self.assertEqual(d(result["metrics"]["final_capital_quote"]), 100)
        self.assertEqual(d(result["equity"][1]["exposure_quote"]), 100)

    def test_new_allocation_scales_to_cash_without_rebalancing_carry(self):
        data = fixture(({"X": "0.5", "Y": "0"}, {"X": "0.5", "Y": "0.5"}), symbols=("X", "Y"))
        for row in data[1]:
            if row["timestamp"] >= 19 * BAR_MS:
                row["prices"]["X"] = "200"
        result = execute(register_prices(data))
        self.assertEqual(result["decisions"][0]["quantities_after"]["X"], result["decisions"][1]["quantities_after"]["X"])
        self.assertLess(d(result["decisions"][1]["cash_scaling_factor"]), 1)
        self.assertEqual(d(result["metrics"]["final_capital_quote"]), 150)
        self.assertGreaterEqual(d(result["metrics"]["minimum_cash_quote"]), 0)

    def test_cash_tail_needs_no_future_market_prices(self):
        result = execute(fixture(tail_bars=3))
        self.assertEqual(len(result["equity"]), 41)
        self.assertTrue(all(d(row["exposure_quote"]) == 0 for row in result["equity"][-4:]))

    def test_contiguous_prices_are_required(self):
        data = fixture()
        data[1][4]["timestamp"] += 1
        with self.assertRaisesRegex(ValueError, "grid"):
            execute(data)

    def test_negative_or_leveraged_targets_are_rejected(self):
        data = fixture()
        for weight in ("-0.1", "1.1", "NaN"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                make_decision(origin_start_ms=0, decision_asof_ms=0, targets={"X": weight},
                              **{k: data[2][0][k] for k in ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256")})

    def test_mutated_horizon_and_skipped_origin_are_rejected(self):
        for change in ("horizon", "missing_origin"):
            data = fixture()
            if change == "horizon":
                data[2][1]["window"]["planned_exit_open_ms"] -= BAR_MS
            else:
                del data[2][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                execute(data)

    def test_varying_execution_delay_cannot_shorten_carry_period(self):
        data = list(fixture())
        original = data[2][1]
        data[2][1] = make_decision(origin_start_ms=HORIZON_MS, decision_asof_ms=HORIZON_MS,
                                 execution_delay_ms=1, targets=original["targets"],
                                 **{k: original[k] for k in ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256")})
        with self.assertRaisesRegex(ValueError, "economic horizon"):
            execute(data)

    def test_hashes_repeat_and_bind_model_identity(self):
        data = fixture()
        first = execute(data)
        self.assertEqual(first["output_sha256"], execute(data)["output_sha256"])
        changed = copy.deepcopy(data)
        changed[2][0]["model_sha256"] = identity({"different_model": True})
        second = execute(changed)
        self.assertNotEqual(first["input_sha256"], second["input_sha256"])
        self.assertNotEqual(first["output_sha256"], second["output_sha256"])
        self.assertEqual(first["metrics"], second["metrics"])

    def test_price_hash_prevents_silent_input_substitution(self):
        data = fixture()
        data[1][5]["prices"]["X"] = "101"
        with self.assertRaisesRegex(ValueError, "price input hash"):
            execute(data)

    def test_daily_returns_compound_to_shared_capital(self):
        data = fixture(tail_bars=5)
        for row in data[1]:
            row["prices"]["X"] = str(100 + row["timestamp"] // BAR_MS)
        result = execute(register_prices(data))
        wealth = d("100")
        for row in daily_measurements(result):
            wealth *= 1 + d(row["net_return"])
            self.assertLess(row["timestamp"], result["spec"]["end_ms"])
        self.assertLess(abs(wealth - d(result["metrics"]["final_capital_quote"])), d("1e-24"))

    def test_mutated_result_is_rejected_by_scoring_inputs(self):
        result = execute(fixture())
        result["equity"][1]["equity_quote"] = "999"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            daily_measurements(result)

    def test_cost_comparison_uses_same_capital_and_timestamps(self):
        spec, bars, decisions, funding, zero = fixture()
        flat = fixture(("0", "0"))[2]
        stress = CostScenario("stress", "8", "8", "4", "4", "synthetic", identity({"stress": 24}))
        results = compare_policies(spec, price_bars=iter(bars), policies={"carry": decisions, "NoTrade": flat},
                                   funding=funding, scenarios=[zero, stress])
        self.assertEqual(d(results["stress"]["NoTrade"]["metrics"]["final_capital_quote"]), 100)
        self.assertLess(d(results["stress"]["carry"]["metrics"]["final_capital_quote"]), 100)
        self.assertEqual(results["zero"]["carry"]["spec"], results["stress"]["NoTrade"]["spec"])

    def test_comparison_rejects_different_decision_clocks(self):
        spec, bars, decisions, funding, costs = fixture()
        with self.assertRaisesRegex(ValueError, "same origin"):
            compare_policies(spec, price_bars=bars, policies={"one": decisions, "two": []}, funding=funding, scenarios=[costs])

    def test_label_purge_uses_each_new_training_cutoff(self):
        sample = {"decision_asof_ms": 0, "symbol": "X", "entry_open_ms": BAR_MS,
                  "exit_open_ms": 19 * BAR_MS, "label_available_at_ms": 20 * BAR_MS,
                  "entry_observation": {"timestamp": BAR_MS, "symbol": "X", "open": "100"},
                  "exit_observation": {"timestamp": 19 * BAR_MS, "symbol": "X", "open": "110"},
                  "gross_open_return_decimal": "0.1",
                  "target_kind": "gross_open_open_diagnostic_not_net_training_target"}
        a = label_binding(sample, origin_start_ms=0, training_asof_ms=27 * BAR_MS - 1)
        b = label_binding(sample, origin_start_ms=0, training_asof_ms=27 * BAR_MS)
        self.assertFalse(a["timing_eligible"])
        self.assertTrue(b["timing_eligible"])
        self.assertFalse(b["fit_allowed"])
        sample["exit_observation"]["timestamp"] -= BAR_MS
        with self.assertRaisesRegex(ValueError, "horizon mismatch"):
            label_binding(sample, origin_start_ms=0, training_asof_ms=30 * BAR_MS)

    def test_legacy_last_close_target_cannot_enter_new_horizon(self):
        sample = {"decision_asof_ms": 0, "target_kind": "next_complete_3d_origin_gross_open_to_last_close"}
        with self.assertRaisesRegex(ValueError, "legacy"):
            label_binding(sample, origin_start_ms=0, training_asof_ms=30 * BAR_MS)

    def test_all_gates_and_penalized_fitness_control_ranking(self):
        valid = build_ranking_key(fitness=1, failed_gates=[])
        for failure in ("minimum_opportunities", "minimum_trade_days", "normalized_direction_concentration", "max_drawdown", "paired_incremental_lcb"):
            with self.subTest(failure=failure):
                self.assertGreater(valid, build_ranking_key(fitness=100, failed_gates=[failure]))
        self.assertGreater(build_ranking_key(fitness=2, failed_gates=[]), valid)
        self.assertEqual(build_ranking_key(fitness=1, failed_gates=["x", "x"]), build_ranking_key(fitness=1, failed_gates=["x"]))
        with self.assertRaises(ValueError):
            build_ranking_key(fitness=float("nan"), failed_gates=[])


class FitnessRankingIntegrationTests(unittest.TestCase):
    def test_scoring_adapter_retains_nav_units_and_empty_ledger_schema(self):
        for weights in (("0.5", "0.5"), ("0", "0")):
            result = execute(fixture(weights))
            simulation = as_primus_simulation(result)
            self.assertEqual(simulation.turnover_fills, result["metrics"]["fills"])
            self.assertAlmostEqual(simulation.portfolio.iloc[-1]["equity"], float(result["metrics"]["final_capital_quote"]) / 100)
            self.assertTrue({"entry_timestamp", "direction", "market_state", "symbol"} <= set(simulation.ledger.columns))
            self.assertEqual(simulation.action_counts["SHORT"], 0)
            self.assertEqual(simulation.diagnostics["capital_accounting"], "shared_cash_nav")

    def test_paired_constrained_overlay_preserves_evidence_and_concentration_gates(self):
        # Small real scoring path, one shared 4-resample bank; no fitter/simulator.
        import numpy as np
        import pandas as pd
        from exia.genetic_primus.fitness_v4 import score_simulation_v4
        from exia.genetic_primus.measurement import build_bootstrap_bank
        from exia.genetic_primus.simulator import PrimusSimulation
        day = 86400000
        timestamps = np.arange(13, dtype=np.int64) * day
        folds = [{"name": str(i), "start": pd.Timestamp((1 + 6 * i) * day, unit="ms", tz="UTC").isoformat(),
                  "end": pd.Timestamp((7 + 6 * i) * day, unit="ms", tz="UTC").isoformat()} for i in range(2)]
        bank = build_bootstrap_bank(pd.DataFrame({"timestamp": timestamps, "symbol": "X"}),
                                    start_timestamp=0, cutoff_timestamp=13 * day, folds=folds,
                                    bank_id="synthetic-ranking-regression", seed=7, samples=4, block_days=1)
        config = {"bootstrap_samples": 4, "bootstrap_alpha": 0.05, "bootstrap_block_days": 1,
                  "minimum_trades": 1, "max_drawdown": 1, "max_turnover_fills_per_day": 10,
                  "max_direction_share": 1, "max_state_share": 1, "max_symbol_share": 1,
                  "minimum_positive_fold_means": 1, "turnover_penalty_bps_per_fill_per_day": 0,
                  "abstention": {"minimum_allowed_fraction": 0.1, "maximum_allowed_fraction": 0.9, "coverage_penalty_bps": 0},
                  "abstention_evidence": {"minimum_opportunities": 50, "minimum_trade_days": 1},
                  "opportunity_normalized_concentration": {"maximum_allocation_lift": 2,
                      "minimum_bucket_opportunities": 1, "penalty_bps_per_excess_lift": 0},
                  "paired_incremental": {"references": ["NoTrade", "Momentum"], "absolute_reference": "NoTrade",
                      "baseline_reference": "Momentum", "minimum_lcb_bps": 0},
                  "adaptive_multiple_testing": {"max_hypotheses_per_origin": 2, "selection_bias_penalty_multiplier": 0}}
        reference = pd.DataFrame({"timestamp": timestamps[1:], "net_return": np.zeros(12)})

        def score(bps, insufficient=False, concentration=False):
            opportunities = 10 if insufficient else 100
            ledger = pd.DataFrame({"entry_timestamp": [day], "direction": ["LONG"], "market_state": ["trend"], "symbol": ["X"]})
            portfolio = reference.assign(net_return=bps / 10000, drawdown=0.0)
            diagnostics = {"overlay_opportunities": opportunities, "overlay_allowed": opportunities // 5,
                           "overlay_opportunities_by_direction": {"LONG": 10, "SHORT": 90},
                           "overlay_allowed_by_direction": {"LONG": 10, "SHORT": 0} if concentration else {"LONG": 1, "SHORT": 9}}
            simulation = PrimusSimulation(ledger, portfolio, {"LONG": 1, "FLAT": 11, "SHORT": 0}, 2, diagnostics=diagnostics)
            return score_simulation_v4(simulation, folds=folds, config=config, key="synthetic-only", bootstrap_bank=bank,
                                       paired_reference_portfolios={"NoTrade": reference, "Momentum": reference.copy()})

        valid = score(1)
        self.assertFalse(valid.failed_gates)
        for bad in (score(10, insufficient=True), score(10, concentration=True)):
            self.assertTrue(bad.failed_gates)
            self.assertGreater(bad.fitness, valid.fitness)
            self.assertGreater(valid.ranking_key, bad.ranking_key)
            self.assertEqual(bad.ranking_key[2], bad.fitness)
            self.assertEqual(bad.metrics["ranking_semantics"], "all_gates_then_penalized_fitness/1")


if __name__ == "__main__":
    unittest.main()
