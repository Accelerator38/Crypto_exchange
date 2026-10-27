"""Pure temporal checkpoints and fail-closed research-stage gates; no runners."""
from __future__ import annotations

import hashlib
import json
import math

BAR_MS = 14_400_000
HORIZON_MS = 18 * BAR_MS
START_MS = 1_785_542_400_000  # 2026-08-01 00:00 UTC
END_MS = 1_790_812_800_000  # 2026-10-01 00:00 UTC
ARMS = ("linear6_uniform", "linear6_half_life_90_anchor20",
        "mlp8_uniform", "mlp8_half_life_90_anchor20")
SAFETY = dict.fromkeys(("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
                       "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed"), False)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("full SHA256 required")
    return value


def _verified(value):
    if value.get("sha256") != identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("receipt SHA mismatch")


def build_checkpoints(results, sample_origins, sample_available, preparation_sha256, symbols):
    """Bind every forecast decision to its mature training prefix and model SHA.

    This is a forecast checkpoint ledger, NOT a position/fill/capital ledger.
    The caller first verifies the complete underlying evidence/source receipt.
    """
    digest(preparation_sha256)
    if len(results) != 20 or len(symbols) != 8 or len(set(symbols)) != 8:
        raise ValueError("fixed 20-origin full8 benchmark required")
    if len(sample_origins) != len(sample_available) or not sample_origins:
        raise ValueError("complete sample maturity index required")
    if any(type(t) is not int for t in sample_origins + sample_available):
        raise ValueError("integer sample times required")
    if any(b-a != HORIZON_MS for a, b in zip(sample_origins, sample_origins[1:])):
        raise ValueError("three-day sample grid required")
    if any(a != o + 20*BAR_MS for o, a in zip(sample_origins, sample_available)):
        raise ValueError("label availability must include completed exit bar")
    rows = []
    for i, result in enumerate(results):
        _verified(result)
        origin = START_MS + i*HORIZON_MS
        train = result["training_identity"]
        if (result["origin_ms"] != origin or result["preparation_sha256"] != preparation_sha256
                or result["symbols"] != symbols or result["entry_ms"] != origin+BAR_MS
                or result["exit_ms"] != origin+19*BAR_MS
                or result["outcome_available_ms"] != origin+20*BAR_MS
                or train["train_end_ms"] != origin-BAR_MS or train["cutoff_ms"] != origin-8*BAR_MS):
            raise ValueError("origin/execution/train cutoff mismatch")
        eligible = [a for a in sample_available if a <= train["cutoff_ms"]]
        if (not eligible or len(eligible) != result["train_origin_count"]
                or len(eligible)*8 != result["train_sample_count"]
                or train["inner_cutoff_ms"] > train["cutoff_ms"]
                or not result["forecast_only"] or result["trading_evaluations"] != 0):
            raise ValueError("maturity, inner cutoff or scope mismatch")
        if set(result["arms"]) != set(ARMS):
            raise ValueError("fixed arms required")
        for arm in ARMS:
            record = result["arms"][arm]
            models = record["models"]
            expected_count, architecture = (1, "linear6") if arm.startswith("linear6") else (3, "mlp8")
            if len(models) != expected_count:
                raise ValueError("fixed ensemble count required")
            for member in models:
                _verified(member)
                if (member["architecture"] != architecture or member["scaler_sha256"] != result["scaler_sha256"]
                        or member["training_identity"]["cutoff_ms"] != train["cutoff_ms"]
                        or member["training_identity"]["train_samples_sha256"] != train["train_samples_sha256"]
                        or member["parameter_count"] != (7 if architecture == "linear6" else 65)
                        or len(member["parameters"]) != member["parameter_count"]
                        or member["runtime_authority"] is not False or member["promotion_authority"] is not False):
                    raise ValueError("model/scaler/training identity mismatch")
            if expected_count == 3 and [m["seed"] for m in models] != [11, 29, 47]:
                raise ValueError("fixed ordered seeds required; best seed selection prohibited")
            member_shas = [digest(m["sha256"]) for m in models]
            model_sha = member_shas[0] if expected_count == 1 else identity({
                "schema": "fixed_mean_ensemble/1", "member_sha256": member_shas, "aggregation": "arithmetic_mean"})
            prediction = record["prediction"]
            if len(prediction) != 8 or any(type(v) not in (float, int) or not math.isfinite(v) for v in prediction):
                raise ValueError("finite full8 forecast required")
            rows.append({"kind": "forecast_checkpoint_not_order_or_fill", "arm": arm,
                         "origin_ms": origin, "decision_asof_ms": origin, "entry_ms": result["entry_ms"],
                         "planned_exit_ms": result["exit_ms"], "train_end_ms": train["train_end_ms"],
                         "cutoff_ms": train["cutoff_ms"], "purge_bars": 7,
                         "latest_training_label_available_ms": max(eligible),
                         "mature_training_origins": len(eligible),
                         "train_samples_sha256": digest(train["train_samples_sha256"]),
                         "inner_fold_sha256": digest(train["fold_sha256"]),
                         "feature_definition_sha256": digest(train["feature_definition_sha256"]),
                         "feature_prefix_sha256": digest(result["feature_prefix_sha256"]),
                         "decision_feature_sha256": digest(result["decision_feature_sha256"]),
                         "scaler_sha256": digest(result["scaler_sha256"]),
                         "model_sha256": model_sha, "member_sha256": member_shas,
                         "weights_sha256": digest(record["weights_sha256"]),
                         "parameters_per_member": models[0]["parameter_count"],
                         "stored_coefficients": sum(m["parameter_count"] for m in models),
                         "forecast_gross": dict(zip(symbols, prediction)),
                         "origin_receipt_sha256": result["sha256"]})
    if len({r["scaler_sha256"] for r in rows}) != 20:
        raise ValueError("distinct per-origin train-only scaler identities required")
    package = {"schema": "rolling_forecast_checkpoint_ledger/1", "rows": rows,
               "benchmark_start_ms": START_MS, "benchmark_end_exclusive_ms": END_MS,
               "complete_origins": 20, "checkpoint_count": 80,
               "final_terminal_ms": START_MS+19*HORIZON_MS+19*BAR_MS,
               "cash_tail_ms": END_MS-(START_MS+19*HORIZON_MS+19*BAR_MS),
               "preparation_sha256": preparation_sha256,
               "safety": SAFETY, "financial_ledger_computed": False}
    package["sha256"] = identity(package)
    return package


def equal_search_budget(cap=64):
    if type(cap) is not int or not 1 <= cap <= 256:
        raise ValueError("bounded integer unique-candidate cap required")
    # Both methods must reserve the same multiresource budget, not just generations.
    budget = {"unique_objective_calls_per_origin": cap, "model_fits_per_origin": 2*cap,
              "inner_fold_evaluations_per_origin": cap, "gradient_steps_per_origin": 0,
              "duplicate_candidates_charge_budget": False, "unlogged_candidate_calls_allowed": False,
              "model_family": "linear6_feature_mask_l2_only", "shared_candidate_space_required": True,
              "shared_folds_and_resample_banks_required": True, "incumbent_evaluations_charge_budget": True}
    return {"ga": budget.copy(), "random_search": budget.copy(), "current_objective_calls": 0,
            "registered_candidate_space_sha256": None, "search_stage_executed": False}


def compare_search_usage(ga, random_search, budget):
    """Check measured usage, not merely equality of configured generation caps."""
    names = {"unique_objective_calls": "unique_objective_calls_per_origin",
             "model_fits": "model_fits_per_origin", "inner_fold_evaluations": "inner_fold_evaluations_per_origin",
             "gradient_steps": "gradient_steps_per_origin"}
    reasons = []
    if budget["ga"] != budget["random_search"]:
        raise ValueError("different budget contracts")
    for method, usage in (("ga", ga), ("random_search", random_search)):
        for counter, maximum in names.items():
            value = usage.get(counter)
            if type(value) is not int or not 0 <= value <= budget[method][maximum]:
                raise ValueError("invalid/over-budget measured counter: " + counter)
        candidates = usage.get("candidate_sha256")
        if (type(candidates) is not list or len(candidates) != len(set(candidates))
                or len(candidates) != usage["unique_objective_calls"]):
            raise ValueError("unique candidate log/counter mismatch")
        for candidate in candidates:
            digest(candidate)
        if usage["unique_objective_calls"] != budget[method]["unique_objective_calls_per_origin"]:
            reasons.append(method + "_budget_not_completed")
    for name in ("origin_ms", "candidate_space_sha256", "data_sha256", "fold_sha256", "resample_bank_sha256"):
        if name.endswith("sha256"):
            digest(ga.get(name))
            digest(random_search.get(name))
        elif type(ga.get(name)) is not int or type(random_search.get(name)) is not int:
            raise ValueError("origin identity required")
        if ga[name] != random_search[name]:
            reasons.append(name + "_mismatch")
    for counter in names:
        if ga[counter] != random_search[counter]:
            reasons.append(counter + "_mismatch")
    return {"fair_comparison": not reasons, "failed_gates": reasons,
            "resource_counting": "actual_unique_evaluations_fits_folds_steps_not_generations"}


def stage_gates(*, prerequisites, signal, trial_registry_complete=False):
    """Do not treat forecast RMSE/sign accuracy as economic signal confirmation.

    Signal fields are validated evidence summaries supplied by a future net-ledger
    verifier, never an authorization to launch a runner. No code path performs GA.
    """
    required = ("policy_registered", "capital_and_allocation_registered", "normal_costs_registered",
                "stress_costs_registered", "funding_complete_for_interval", "input_evidence_verified")
    missing = [name for name in required if prerequisites.get(name) is not True]
    signal_missing = []
    if signal is None:
        signal_missing.append("simple_net_signal_not_evaluated")
    else:
        if signal.get("scope") != "pre_august_causal_net_validation" or signal.get("evidence_verified") is not True:
            signal_missing.append("verified_development_net_evidence_required")
        if type(signal.get("evaluation_end_ms")) is not int or signal["evaluation_end_ms"] > START_MS:
            signal_missing.append("opened_aug_sep_not_signal_selection_period")
        for name in ("normal_lcb_bps_per_day", "stress_lcb_bps_per_day",
                     "paired_lcb_bps_per_day", "audit_paired_lcb_bps_per_day"):
            value = signal.get(name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                signal_missing.append(name + "_not_positive")
        if signal.get("failed_gates") != []:
            signal_missing.append("economic_risk_or_stability_gates_not_passed")
        if signal.get("selection_bank_sha256") == signal.get("audit_bank_sha256"):
            signal_missing.append("audit_bank_not_independent")
        for name in ("normal_ledger_sha256", "stress_ledger_sha256", "incumbent_ledger_sha256",
                     "selection_bank_sha256", "audit_bank_sha256", "evidence_receipt_sha256"):
            try:
                digest(signal.get(name))
            except ValueError:
                signal_missing.append(name + "_missing")
    if trial_registry_complete is not True:
        signal_missing.append("global_trial_registry_incomplete")
    return {"reference_replay_ready": not missing, "replay_missing_prerequisites": missing,
            "ga_search_design_ready": not missing and not signal_missing,
            "ga_missing_prerequisites": missing + signal_missing,
            "ga_executed": False, "random_search_executed": False,
            "prospective_started": False, "prospective_seal_allowed": False,
            "prospective_requirements": ["net_replay_and_stability_completed", "final_policy_and_registry_frozen",
                                         "positive_economic_signal", "fixed_duration_and_final_analysis_only",
                                         "start_after_seal_and_all_consumed_data", "unseen_interval_verified"],
            "rmse_is_not_economic_signal_gate": True, "safety": SAFETY}
