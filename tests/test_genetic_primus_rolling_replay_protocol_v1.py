"""Pure gate and receipt regressions, without model fitting or market replay."""
from __future__ import annotations

import unittest
from exia.genetic_primus import rolling_replay_protocol_v1 as p


def prerequisites():
    return dict.fromkeys(("policy_registered", "capital_and_allocation_registered", "normal_costs_registered",
                          "stress_costs_registered", "funding_complete_for_interval", "input_evidence_verified"), True)


def signal():
    return {"scope": "pre_august_causal_net_validation", "evidence_verified": True,
            "evaluation_end_ms": p.START_MS, "normal_lcb_bps_per_day": 1., "stress_lcb_bps_per_day": .5,
            "paired_lcb_bps_per_day": .4, "audit_paired_lcb_bps_per_day": .3,
            "failed_gates": [], "normal_ledger_sha256": "1"*64, "stress_ledger_sha256": "2"*64,
            "incumbent_ledger_sha256": "3"*64, "selection_bank_sha256": "4"*64,
            "audit_bank_sha256": "5"*64, "evidence_receipt_sha256": "6"*64}


def usage(cap=2):
    return {"origin_ms": p.START_MS, "unique_objective_calls": cap, "model_fits": cap*2,
            "inner_fold_evaluations": cap, "gradient_steps": 0, "candidate_sha256": [str(i+1)*64 for i in range(cap)],
            "candidate_space_sha256": "a"*64, "data_sha256": "b"*64,
            "fold_sha256": "c"*64, "resample_bank_sha256": "d"*64}


def fixtures():
    symbols = ["S"+str(i) for i in range(8)]
    results = []
    origins = [p.START_MS-(70-i)*p.HORIZON_MS for i in range(90)]
    available = [t+20*p.BAR_MS for t in origins]
    for i in range(20):
        origin = p.START_MS+i*p.HORIZON_MS
        scaler = p.identity({"origin": origin})
        train = {"train_end_ms": origin-p.BAR_MS, "cutoff_ms": origin-8*p.BAR_MS,
                 "inner_cutoff_ms": origin-400*p.BAR_MS, "train_samples_sha256": "a"*64,
                 "fold_sha256": "b"*64, "feature_definition_sha256": "c"*64}
        count = sum(a <= train["cutoff_ms"] for a in available)
        result = {"origin_ms": origin, "preparation_sha256": "f"*64, "symbols": symbols,
                  "entry_ms": origin+p.BAR_MS, "exit_ms": origin+19*p.BAR_MS,
                  "outcome_available_ms": origin+20*p.BAR_MS, "training_identity": train,
                  "train_origin_count": count, "train_sample_count": count*8,
                  "forecast_only": True, "trading_evaluations": 0, "arms": {}, "scaler_sha256": scaler,
                  "feature_prefix_sha256": "d"*64, "decision_feature_sha256": "e"*64}
        for arm in p.ARMS:
            linear = arm.startswith("linear6")
            members = []
            for seed in ([None] if linear else [11, 29, 47]):
                member = {"architecture": "linear6" if linear else "mlp8", "scaler_sha256": scaler,
                          "training_identity": train, "parameter_count": 7 if linear else 65,
                          "parameters": [0.]*(7 if linear else 65), "runtime_authority": False,
                          "promotion_authority": False, "seed": seed}
                member["sha256"] = p.identity(member)
                members.append(member)
            result["arms"][arm] = {"models": members, "prediction": [0.]*8, "weights_sha256": "0"*64}
        result["sha256"] = p.identity(result)
        results.append(result)
    return results, origins, available, "f"*64, symbols


def reseal(value):
    value["sha256"] = p.identity({k: v for k, v in value.items() if k != "sha256"})


class ProtocolTests(unittest.TestCase):
    def test_unknown_inputs_fail_closed(self):
        gate = p.stage_gates(prerequisites={}, signal=None)
        self.assertFalse(gate["reference_replay_ready"])
        self.assertFalse(gate["ga_search_design_ready"])

    def test_forecast_accuracy_is_not_net_signal(self):
        s = {"rmse_improvement": 100, "sign_accuracy": .99}
        gate = p.stage_gates(prerequisites=prerequisites(), signal=s, trial_registry_complete=True)
        self.assertFalse(gate["ga_search_design_ready"])

    def test_verified_net_signal_can_plan_not_launch(self):
        s = signal() | {"rmse_improvement": -100}
        gate = p.stage_gates(prerequisites=prerequisites(), signal=s, trial_registry_complete=True)
        self.assertTrue(gate["ga_search_design_ready"])
        self.assertFalse(gate["ga_executed"])
        self.assertFalse(gate["prospective_seal_allowed"])
        self.assertTrue(all(v is False for v in gate["safety"].values()))

    def test_missing_funding_blocks_even_positive_signal(self):
        req = prerequisites() | {"funding_complete_for_interval": False}
        self.assertFalse(p.stage_gates(prerequisites=req, signal=signal(), trial_registry_complete=True)["ga_search_design_ready"])

    def test_zero_or_negative_lcb_blocks(self):
        for value in (0, -1, float("nan"), True):
            with self.subTest(value=value):
                s = signal() | {"stress_lcb_bps_per_day": value}
                self.assertFalse(p.stage_gates(prerequisites=prerequisites(), signal=s, trial_registry_complete=True)["ga_search_design_ready"])

    def test_open_benchmark_cannot_select_signal(self):
        s = signal() | {"evaluation_end_ms": p.END_MS}
        self.assertFalse(p.stage_gates(prerequisites=prerequisites(), signal=s, trial_registry_complete=True)["ga_search_design_ready"])

    def test_nonindependent_or_missing_sha_blocks(self):
        for update in ({"audit_bank_sha256": "4"*64}, {"normal_ledger_sha256": "short"}):
            gate = p.stage_gates(prerequisites=prerequisites(), signal=signal() | update, trial_registry_complete=True)
            self.assertFalse(gate["ga_search_design_ready"])

    def test_trial_registry_and_risk_gates_required(self):
        self.assertFalse(p.stage_gates(prerequisites=prerequisites(), signal=signal())["ga_search_design_ready"])
        self.assertFalse(p.stage_gates(prerequisites=prerequisites(), signal=signal() | {"failed_gates": ["drawdown"]}, trial_registry_complete=True)["ga_search_design_ready"])

    def test_same_budget_no_execution(self):
        budget = p.equal_search_budget()
        self.assertEqual(budget["ga"], budget["random_search"])
        self.assertEqual(budget["current_objective_calls"], 0)

    def test_invalid_budget_rejected(self):
        for cap in (True, 0, 257, 1.5):
            with self.assertRaises(ValueError):
                p.equal_search_budget(cap)

    def test_measured_equal_budget(self):
        self.assertTrue(p.compare_search_usage(usage(), usage(), p.equal_search_budget(2))["fair_comparison"])

    def test_incomplete_or_unequal_actual_fits_not_equal_budget(self):
        other = usage() | {"model_fits": 3}
        self.assertFalse(p.compare_search_usage(usage(), other, p.equal_search_budget(2))["fair_comparison"])
        self.assertFalse(p.compare_search_usage(usage(1), usage(1), p.equal_search_budget(2))["fair_comparison"])

    def test_different_bank_not_fair(self):
        other = usage() | {"resample_bank_sha256": "e"*64}
        self.assertFalse(p.compare_search_usage(usage(), other, p.equal_search_budget(2))["fair_comparison"])

    def test_duplicates_and_excess_fit_budget_rejected(self):
        for update in ({"candidate_sha256": ["1"*64, "1"*64]}, {"model_fits": 5}):
            with self.assertRaises(ValueError):
                p.compare_search_usage(usage(), usage() | update, p.equal_search_budget(2))

    def test_checkpoint_ledger_clock_purge_ensemble(self):
        ledger = p.build_checkpoints(*fixtures())
        self.assertEqual(ledger["checkpoint_count"], 80)
        self.assertEqual(ledger["cash_tail_ms"], 5*p.BAR_MS)
        for row in ledger["rows"]:
            self.assertLessEqual(row["latest_training_label_available_ms"], row["cutoff_ms"])
            self.assertEqual(row["cutoff_ms"], row["origin_ms"]-8*p.BAR_MS)
            self.assertEqual(len(row["model_sha256"]), 64)
            if row["arm"].startswith("mlp8"):
                self.assertEqual(row["stored_coefficients"], 195)
                self.assertEqual(len(row["member_sha256"]), 3)

    def test_future_label_or_wrong_cutoff_rejected(self):
        for offset in (1, p.BAR_MS):
            args = fixtures()
            args[0][0]["training_identity"]["cutoff_ms"] += offset
            reseal(args[0][0])
            with self.assertRaises(ValueError):
                p.build_checkpoints(*args)

    def test_mutated_model_rejected(self):
        args = fixtures()
        args[0][0]["arms"]["linear6_uniform"]["models"][0]["parameters"][0] = 1
        reseal(args[0][0])
        with self.assertRaises(ValueError):
            p.build_checkpoints(*args)

    def test_stale_scaler_rejected(self):
        args = fixtures()
        stale = args[0][0]["scaler_sha256"]
        second = args[0][1]
        second["scaler_sha256"] = stale
        for arm in second["arms"].values():
            for member in arm["models"]:
                member["scaler_sha256"] = stale
                reseal(member)
        reseal(second)
        with self.assertRaises(ValueError):
            p.build_checkpoints(*args)

    def test_maturity_receipt_mismatch_rejected(self):
        args = fixtures()
        args[0][0]["train_origin_count"] += 1
        reseal(args[0][0])
        with self.assertRaises(ValueError):
            p.build_checkpoints(*args)


if __name__ == "__main__":
    unittest.main()
