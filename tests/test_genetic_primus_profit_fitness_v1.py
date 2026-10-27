"""Synthetic net-growth/risk/search/audit regressions; direct unittest only."""
from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from exia.genetic_primus.profit_fitness_v1 import (
    DAY_MS, ProfitObjective, check_profit_stability, growth_diagnostics,
    objectives_from_contract, research_objectives, score_net_growth,
)
from exia.genetic_primus.unified_measurement_v1 import (
    BAR_MS, CostScenario, as_primus_simulation, identity, replay,
)

_spec = importlib.util.spec_from_file_location(
    "profit_ledger_fixture", Path(__file__).with_name("test_genetic_primus_unified_measurement_v1.py")
)
_fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture_module)


def sample(*, minimum=95, final=115, weight="1", costs=None, smooth=False,
           funding_rate=None, tail_bars=5):
    spec, bars, decisions, funding, zero = _fixture_module.fixture((weight, weight), tail_bars=tail_bars)
    for i, row in enumerate(bars):
        if smooth:
            value = Decimal(100) + Decimal(i - 1 if i else 0) / 2
        elif i <= 1:
            value = Decimal(100)
        elif i <= 18:
            value = Decimal(100) + (Decimal(str(minimum)) - 100) * Decimal(i - 1) / 17
        elif i < 37:
            value = Decimal(str(minimum))
        else:
            value = Decimal(str(final))
        row["prices"]["X"] = str(value)
    spec = replace(spec, market_inputs_sha256=identity(bars))
    if funding_rate is not None:
        funding["events"] = [{"timestamp": 9 * BAR_MS, "symbol": "X", "mark_price": "100",
                              "signed_rate_bps": str(funding_rate)}]
        funding["evidence_sha256"] = identity({"synthetic_funding": funding["events"]})
    return replay(spec, price_bars=bars, decisions=decisions, funding=funding, costs=costs or zero)


def synthetic_costs(name="stress", *, fee="8", drag="4"):
    return CostScenario(name, fee, fee, drag, drag, "synthetic", identity({"synthetic": name, "fee": fee, "drag": drag}))


def bank_fixture(result, seed=7):
    import pandas as pd
    from exia.genetic_primus.measurement import build_bootstrap_bank
    folds = [{"name": str(i), "start": pd.Timestamp(start * DAY_MS, unit="ms", tz="UTC").isoformat(),
              "end": pd.Timestamp(end * DAY_MS, unit="ms", tz="UTC").isoformat()}
             for i, (start, end) in enumerate(((0, 3), (3, 7)))]
    frame = pd.DataFrame({"timestamp": [r["timestamp"] for r in result["equity"]], "symbol": "X"})
    bank = build_bootstrap_bank(frame, start_timestamp=0, cutoff_timestamp=7 * DAY_MS,
                                folds=folds, bank_id="synthetic-profit-" + str(seed), seed=seed,
                                samples=64, block_days=2)
    return folds, bank


def scoring_config(objective, *, compute=False):
    return {"search_objective": asdict(objective), "compute_stability_diagnostics": compute,
            "bootstrap_samples": 64, "bootstrap_block_days": 2, "bootstrap_alpha": 0.05}


class ProfitSearchTests(unittest.TestCase):
    def test_same_risk_feasibility_ranks_final_net_capital(self):
        a = score_net_growth(sample(final=115), objective=ProfitObjective(10))
        b = score_net_growth(sample(final=105), objective=ProfitObjective(10))
        self.assertGreater(a["ranking_key"], b["ranking_key"])
        self.assertFalse(a["failed_gates"])

    def test_fitness_is_net_log_growth_per_calendar_day(self):
        result = sample()
        score = score_net_growth(result, objective=ProfitObjective(10))
        expected = float((Decimal(115) / 100).ln() * 10000 / 7)
        self.assertAlmostEqual(score["fitness"], expected, places=10)
        self.assertEqual(score["metrics"]["calendar_days"], 7)

    def test_cash_days_are_in_the_denominator(self):
        a = score_net_growth(sample(tail_bars=5), objective=ProfitObjective(10))
        b = score_net_growth(sample(tail_bars=11), objective=ProfitObjective(10))
        self.assertEqual(a["metrics"]["final_capital_quote"], b["metrics"]["final_capital_quote"])
        self.assertGreater(a["fitness"], b["fitness"])

    def test_research_budget_changes_feasible_winner(self):
        paths = {"safe": sample(minimum=95, final=115), "medium": sample(minimum=85, final=125),
                 "high": sample(minimum=75, final=140), "NoTrade": sample(weight="0")}
        winners = []
        for objective in research_objectives():
            scores = {name: score_net_growth(result, objective=objective) for name, result in paths.items()}
            winner = max(scores, key=lambda name: scores[name]["ranking_key"])
            self.assertTrue(scores[winner]["metrics"]["search_risk_passed"])
            winners.append(winner)
        self.assertEqual(winners, ["safe", "medium", "high"])

    def test_exact_drawdown_boundary_is_allowed(self):
        at = score_net_growth(sample(minimum=90), objective=ProfitObjective(10))
        over = score_net_growth(sample(minimum="89.9999"), objective=ProfitObjective(10))
        self.assertFalse(at["failed_gates"])
        self.assertIn("normal_drawdown_budget", over["failed_gates"])

    def test_initial_peak_includes_entry_cost(self):
        result = sample(minimum=100, final=100, costs=synthetic_costs())
        score = score_net_growth(result, objective=ProfitObjective(10))
        self.assertGreater(score["metrics"]["max_drawdown"], 0)
        self.assertLess(score["fitness"], 0)

    def test_stress_drawdown_is_a_constraint_when_supplied(self):
        normal = sample()
        stress = sample(costs=synthetic_costs(fee="0", drag="1000"))
        score = score_net_growth(normal, objective=ProfitObjective(10), stress_result=stress)
        self.assertIn("stress_drawdown_budget", score["failed_gates"])
        self.assertTrue(score["metrics"]["stress_evaluated"])
        self.assertEqual(score["metrics"]["normal_max_drawdown"], 0.05)

    def test_no_trade_beats_net_losing_policy(self):
        flat = score_net_growth(sample(weight="0"), objective=ProfitObjective(10))
        losing = score_net_growth(sample(minimum=95, final=98), objective=ProfitObjective(10))
        self.assertEqual(flat["fitness"], 0)
        self.assertFalse(losing["failed_gates"])
        self.assertGreater(flat["ranking_key"], losing["ranking_key"])

    def test_gross_gain_does_not_hide_cost_loss(self):
        result = sample(minimum=100, final=105, costs=synthetic_costs(fee="1000", drag="0"))
        score = score_net_growth(result, objective=ProfitObjective(30))
        self.assertLess(score["metrics"]["net_return_pct"], 0)
        self.assertLess(score["fitness"], 0)

    def test_costs_and_funding_are_not_double_penalized(self):
        result = sample(weight="0.5", costs=synthetic_costs(), funding_rate=10)
        score = score_net_growth(result, objective=ProfitObjective(10))
        expected = float((Decimal(result["metrics"]["final_capital_quote"]) / 100).ln() * 10000 / 7)
        self.assertAlmostEqual(score["fitness"], expected, places=10)
        self.assertEqual(score["metrics"]["cost_double_counting_penalty_bps"], 0)

    def test_parsimony_is_inside_primary_score(self):
        objective = ProfitObjective(10, active_parameter_penalty_bps_per_100="10", active_feature_penalty_bps="0.5")
        plain = score_net_growth(sample(), objective=ProfitObjective(10))
        regularized = score_net_growth(sample(), objective=objective,
                                       complexity_metrics={"active_parameter_count": 3, "active_feature_count": 2})
        self.assertAlmostEqual(plain["fitness"] - regularized["fitness"], 1.3, places=10)
        self.assertLess(regularized["ranking_key"], plain["ranking_key"])

    def test_nonzero_penalty_requires_complexity_evidence(self):
        with self.assertRaisesRegex(ValueError, "complexity evidence"):
            score_net_growth(sample(), objective=ProfitObjective(10, active_feature_penalty_bps="1"))

    def test_operational_fill_cap_is_not_a_second_cost_penalty(self):
        score = score_net_growth(sample(), objective=ProfitObjective(10, max_fills_per_calendar_day="0"))
        self.assertIn("operational_fill_limit", score["failed_gates"])
        self.assertEqual(score["metrics"]["cost_double_counting_penalty_bps"], 0)

    def test_only_research_ten_twenty_thirty_are_registered(self):
        for value in (5, 15, True, 10.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ProfitObjective(value).validate()
        with self.assertRaises(ValueError):
            ProfitObjective(20, scope="live").validate()
        self.assertEqual(len({objective.sha256 for objective in research_objectives()}), 3)

    def test_independent_path_check_rejects_forged_summary(self):
        result = sample()
        result["metrics"]["max_drawdown"] = "0"
        result["output_sha256"] = identity({k: v for k, v in result.items() if k != "output_sha256"})
        with self.assertRaisesRegex(ValueError, "summary"):
            score_net_growth(result, objective=ProfitObjective(10))

    def test_stress_must_share_prices_clock_targets_and_funding(self):
        with self.assertRaisesRegex(ValueError, "comparison mismatch"):
            score_net_growth(sample(), objective=ProfitObjective(20), stress_result=sample(final=120))

    def test_search_never_claims_statistical_or_runtime_admission(self):
        score = score_net_growth(sample(), objective=ProfitObjective(10))
        self.assertEqual(score["metrics"]["validation_status"], "not_evaluated")
        self.assertFalse(score["metrics"]["promotion_authority"])
        self.assertFalse(score["metrics"]["temporal_gate_passed"])

    def test_machine_contract_yields_only_three_research_variants(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / "configs/genetic_primus_profit_fitness_v1.json").read_text(encoding="utf-8"))
        self.assertEqual([v.drawdown_budget_percent for v in objectives_from_contract(contract)], [10, 20, 30])
        contract["search_computes_lcb"] = True
        with self.assertRaises(ValueError):
            objectives_from_contract(contract)

    def test_objective_hash_normalizes_equivalent_decimal_parameters(self):
        self.assertEqual(ProfitObjective(10).sha256,
                         ProfitObjective(10, active_parameter_penalty_bps_per_100="0.00").sha256)


class ProfitAuditIntegrationTests(unittest.TestCase):
    def test_profitable_concentrated_path_can_fail_lcb_without_changing_search(self):
        normal = sample(minimum=75, final=140)
        stress = sample(minimum=75, final=140, costs=synthetic_costs())
        folds, bank = bank_fixture(normal, 7)
        _, audit_bank = bank_fixture(normal, 19)
        before = score_net_growth(normal, objective=ProfitObjective(30))
        audit = check_profit_stability(normal, stress=stress, folds=folds, selection_bank=bank, audit_bank=audit_bank)
        after = score_net_growth(normal, objective=ProfitObjective(30))
        self.assertGreater(before["fitness"], 0)
        self.assertEqual(audit["status"], "failed")
        self.assertLess(audit["banks"]["selection"]["normal"]["net_log_lcb_daily_bps"], 0)
        self.assertEqual(before, after)
        self.assertFalse(audit["affects_search_ranking"])

    def test_smooth_positive_path_passes_independent_cost_audit(self):
        normal, stress = sample(smooth=True), sample(smooth=True, costs=synthetic_costs())
        folds, bank = bank_fixture(normal)
        _, audit_bank = bank_fixture(normal, 19)
        audit = check_profit_stability(normal, stress=stress, folds=folds, selection_bank=bank, audit_bank=audit_bank)
        self.assertEqual(audit["status"], "passed")
        self.assertFalse(audit["promotion_authority"])
        self.assertEqual(audit["bootstrap_alpha"], 0.05)
        self.assertEqual(audit["normal_measurement_output_sha256"], normal["output_sha256"])

    def test_paired_audit_detects_inferior_profitable_challenger(self):
        normal = sample(smooth=True, weight="0.25")
        stress = sample(smooth=True, weight="0.25", costs=synthetic_costs())
        incumbent = sample(smooth=True, weight="0.5")
        stress_incumbent = sample(smooth=True, weight="0.5", costs=synthetic_costs())
        folds, bank = bank_fixture(normal)
        _, audit_bank = bank_fixture(normal, 19)
        audit = check_profit_stability(normal, stress=stress, folds=folds, selection_bank=bank, audit_bank=audit_bank,
                                        incumbent=incumbent, stress_incumbent=stress_incumbent)
        self.assertIn("selection_normal_paired_log_lcb", audit["failed_gates"])
        self.assertGreater(audit["banks"]["selection"]["normal"]["net_log_lcb_daily_bps"], 0)

    def test_audit_cannot_reuse_selection_seed(self):
        normal, stress = sample(), sample(costs=synthetic_costs())
        folds, bank = bank_fixture(normal)
        with self.assertRaisesRegex(ValueError, "independent audit"):
            check_profit_stability(normal, stress=stress, folds=folds, selection_bank=bank, audit_bank=bank)

    def test_mutated_resample_bank_is_rejected(self):
        normal = sample()
        folds, bank = bank_fixture(normal)
        bank.scope_indices["overall"][0, 0] = (int(bank.scope_indices["overall"][0, 0]) + 1) % 7
        with self.assertRaisesRegex(ValueError, "resample bank content"):
            growth_diagnostics(normal, folds=folds, bootstrap_bank=bank)

    def test_cutoff_and_fold_hashes_are_fixed(self):
        normal = sample()
        folds, bank = bank_fixture(normal)
        with self.assertRaisesRegex(ValueError, "cutoff/fold"):
            growth_diagnostics(normal, folds=folds, bootstrap_bank=replace(bank, cutoff_ms=8 * DAY_MS))

    def test_partial_days_are_not_mixed_into_day_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "complete calendar days"):
            growth_diagnostics(sample(tail_bars=0), folds=[])

    def test_invalid_alpha_cannot_produce_an_audit(self):
        for alpha in (0, 1, True, "NaN"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                growth_diagnostics(sample(), folds=[], alpha=alpha)

    def test_search_scorer_uses_profit_without_bootstrap_execution(self):
        from exia.genetic_primus.fitness_v4 import score_simulation_v4
        normal = sample()
        folds, bank = bank_fixture(normal)
        simulation = as_primus_simulation(normal)
        score = score_simulation_v4(simulation, config=scoring_config(ProfitObjective(10)),
                                    folds=folds, key="synthetic-profit", bootstrap_bank=bank)
        expected = score_net_growth(normal, objective=ProfitObjective(10))
        self.assertEqual(score.fitness, expected["fitness"])
        self.assertEqual(score.ranking_key, expected["ranking_key"])
        self.assertIsNone(score.metrics["lcb_daily_bps"])
        self.assertFalse(score.metrics["stability_bootstrap_evaluated"])

    def test_lcb_diagnostics_never_change_scorer_ranking(self):
        from exia.genetic_primus.fitness_v4 import score_simulation_v4
        normal = sample(minimum=75, final=140)
        folds, bank = bank_fixture(normal)
        simulation = as_primus_simulation(normal)
        scores = [score_simulation_v4(simulation, config=scoring_config(ProfitObjective(30), compute=flag),
                                      folds=folds, key="same", bootstrap_bank=bank) for flag in (False, True)]
        self.assertEqual(scores[0].fitness, scores[1].fitness)
        self.assertEqual(scores[0].ranking_key, scores[1].ranking_key)
        self.assertIn("selection_normal_log_lcb", scores[1].metrics["validation_failed_gates"])
        self.assertFalse(scores[1].failed_gates)

    def test_legacy_or_mutated_simulation_cannot_use_net_objective(self):
        from exia.genetic_primus.fitness_v4 import score_simulation_v4
        normal = sample()
        folds, bank = bank_fixture(normal)
        for mutate in ("legacy", "portfolio"):
            simulation = as_primus_simulation(normal)
            if mutate == "legacy":
                del simulation.diagnostics["measurement_result"]
            else:
                simulation.portfolio.loc[1, "net_return"] = 0.5
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                score_simulation_v4(simulation, config=scoring_config(ProfitObjective(10)),
                                    folds=folds, key="invalid", bootstrap_bank=bank)

    def test_generation_log_handles_uncomputed_lcb_and_records_objective(self):
        import numpy as np
        from exia.genetic_primus.evolution_v4 import _generation_row
        from exia.genetic_primus.fitness_v4 import score_simulation_v4
        normal = sample()
        folds, bank = bank_fixture(normal)
        score = score_simulation_v4(as_primus_simulation(normal), config=scoring_config(ProfitObjective(10)),
                                    folds=folds, key="logging-only", bootstrap_bank=bank)
        row = _generation_row(0, np.array([score.fitness]), SimpleNamespace(score=score))
        self.assertIsNone(row["median_fold_lcb_bps"])
        self.assertEqual(row["drawdown_budget_percent"], 10)
        self.assertIn("stability_not_evaluated", row["validation_failed_gates"])


if __name__ == "__main__":
    unittest.main()
