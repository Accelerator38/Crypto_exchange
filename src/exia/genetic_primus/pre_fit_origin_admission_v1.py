"""Synthetic pre-fit plans with explicit denial; no scaler/model fitter exists."""
from __future__ import annotations

import hashlib
import json
import math

DT = 14_400_000
PROPOSAL_SHA = "a5abd987351f2ef60ad359fd396f6c2879e4c45ee166e08fde7ec2807b94e637"
VARIANTS = ("uniform", "half_life_90d_anchor_20pct")
ORIGIN_KEYS = ("schema", "evidence_class", "context", "training_inputs", "deploy_features",
               "deploy_source_sha256", "deploy_source_file_sha256", "independent_training_deploy_bindings",
               "shared_feature_definition_sha256", "minimum_opportunities_64_passed", "ess_gate_verified",
               "full_old_model_bundle_compatible", "model_ready", "entry_accounting_integrated",
               "real_execution_labels_verified", "market_data_provenance_verified", "actual_compute_latency_verified",
               "new_model_fits", "new_market_strategy_evaluations", "market_research_launch_ready", "real_money_ready",
               "safety", "payload_sha256")
TRAIN_INPUT_KEYS = ("schema", "training", "training_pins", "training_guard", "historical_feature_lineage",
                    "historical_feature_offset_ms", "training_guard_scaler_hash_scope", "scaler_candidate_input", "payload_sha256")
WEIGHT_ROW_KEYS = ("row_id", "symbol", "label_end_exclusive_ms", "label_available_at_ms", "gross_return_3d")
DEPLOY_KEYS = ("schema", "evidence_class", "decision_asof_ms", "history_start_ms", "feature_end_exclusive_ms",
               "prefix_sha256", "kernel_definition_sha256", "rows", "diagnostics", "feature_rows_sha256",
               "availability_is_declared_not_measured", "real_indicator_equivalence_verified", "market_data_provenance_verified",
               "new_model_fits", "new_market_strategy_evaluations", "market_research_launch_ready", "real_money_ready", "safety", "payload_sha256")


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def self_hash(value, guard):
    guard["digest"](value["payload_sha256"])
    if value["payload_sha256"] != sha({k: v for k, v in value.items() if k != "payload_sha256"}):
        raise ValueError("mutated input record")


def weight_bank(rows, variant, training_asof_ms, guard):
    """Only weighting metadata, never fitness/PnL or model execution."""
    if variant not in VARIANTS:
        raise ValueError("unregistered weight variant")
    guard["timestamp"](training_asof_ms)
    guard["exact_list"](rows)
    if not 1 <= len(rows) <= 512:
        raise ValueError("opportunity row budget")
    ages, identities = [], []
    for row in rows:
        guard["keys"](row, WEIGHT_ROW_KEYS, "projected weight row")
        if (type(row["row_id"]) is not str or not row["row_id"] or type(row["symbol"]) is not str
                or not row["symbol"].startswith("SYN_")):
            raise ValueError("synthetic opportunity identity required")
        for name in ("label_end_exclusive_ms", "label_available_at_ms"):
            guard["timestamp"](row[name])
        if not row["label_end_exclusive_ms"] <= row["label_available_at_ms"] <= training_asof_ms - 7 * DT:
            raise ValueError("future or unavailable weighted outcome")
        guard["number"](row["gross_return_3d"])
        if row["gross_return_3d"] <= -1:
            raise ValueError("invalid gross outcome")
        ages.append((training_asof_ms - row["label_end_exclusive_ms"]) / 86_400_000.0)
        identities.append(row["row_id"])
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate weighted opportunity")
    n = len(rows)
    if variant == "uniform":
        weights = [1.0 / n] * n
    else:
        # Subtract minimum age: normalized ratios unchanged, but at least one
        # exponent is zero even for extremely old declared synthetic history.
        minimum = min(ages)
        recent = [math.exp2(-(age - minimum) / 90.0) for age in ages]
        total = math.fsum(recent)
        weights = [0.20 / n + 0.80 * value / total for value in recent]
    if (any(not math.isfinite(w) or w <= 0 for w in weights)
            or abs(math.fsum(weights) - 1.0) > 1e-12):
        raise ValueError("invalid outcome weight bank")
    ess = 1.0 / math.fsum(w * w for w in weights)
    if ess > n + 1e-10:
        raise ValueError("Kish ESS exceeds opportunities")
    entries = [{**row, "weight": w} for row, w in zip(rows, weights)]
    result = {"variant": variant, "age_reference": "training_asof_minus_label_end_not_deployment_or_profit_sign",
              "training_asof_ms": training_asof_ms, "entries": entries, "opportunities": n,
              "weight_sum": math.fsum(weights), "kish_weight_ess": ess,
              "kish_ess_is_not_independent_market_observation_count": True,
              "negative_outcome_weight": math.fsum(w for row, w in zip(rows, weights) if row["gross_return_3d"] < 0),
              "positive_outcome_weight": math.fsum(w for row, w in zip(rows, weights) if row["gross_return_3d"] > 0),
              "zero_outcome_weight": math.fsum(w for row, w in zip(rows, weights) if row["gross_return_3d"] == 0)}
    result["weight_bank_sha256"] = sha(result)
    return result


def timing_gate(ctx, training_budget_seconds, execution_delay_seconds, guard):
    guard["context"](ctx)
    reasons = []
    window = ctx["decision_asof_ms"] - ctx["training_asof_ms"]
    if training_budget_seconds is None:
        reasons.append("training_latency_budget_unregistered")
    else:
        guard["number"](training_budget_seconds)
        if training_budget_seconds <= 0:
            raise ValueError("positive registered training budget required")
        if training_budget_seconds * 1_000 > window:
            reasons.append("training_budget_exceeds_scheduled_asof_window")
    if execution_delay_seconds is None:
        reasons.append("execution_delay_unregistered")
    else:
        guard["number"](execution_delay_seconds)
        if execution_delay_seconds < 0:
            raise ValueError("nonnegative declared execution delay required")
    return {"scheduled_asof_window_ms": window, "training_latency_budget_seconds": training_budget_seconds,
            "execution_delay_seconds": execution_delay_seconds, "declared_timing_gate_passed": not reasons,
            "actual_latency_or_execution_verified": False, "denial_reasons": reasons}


def validate_origin(origin, guard):
    guard["keys"](origin, ORIGIN_KEYS, "v2 origin inputs")
    if (origin["schema"] != "exia.genetic_primus.synthetic_causal_origin_inputs/2"
            or origin["evidence_class"] != "synthetic_origin_input_engineering_only"
            or origin["independent_training_deploy_bindings"] is not True):
        raise ValueError("independent synthetic v2 inputs required")
    if (set(origin["safety"]) != set(guard["SAFETY_KEYS"])
            or any(origin["safety"][k] is not False for k in guard["SAFETY_KEYS"])):
        raise ValueError("authority must remain literally false")
    for name in ("ess_gate_verified", "full_old_model_bundle_compatible", "model_ready", "entry_accounting_integrated",
                 "real_execution_labels_verified", "market_data_provenance_verified", "actual_compute_latency_verified",
                 "market_research_launch_ready", "real_money_ready"):
        if origin[name] is not False:
            raise ValueError("synthetic stage cannot earn actual admission")
    for name in ("new_model_fits", "new_market_strategy_evaluations"):
        if type(origin[name]) is not int or origin[name] != 0:
            raise ValueError("no fits/evaluations permitted")
    self_hash(origin, guard)
    inputs = origin["training_inputs"]
    guard["keys"](inputs, TRAIN_INPUT_KEYS, "training-only inputs")
    if inputs["schema"] != "exia.genetic_primus.synthetic_training_inputs/2":
        raise ValueError("training-only v2 required")
    self_hash(inputs, guard)
    training = inputs["training"]
    info = guard["validate_training"](training, inputs["training_pins"])
    if inputs["training_guard"] != info or training["context"] != origin["context"]:
        raise ValueError("training guard/context mismatch")
    if (inputs["training_guard_scaler_hash_scope"] != "all_training_rows_not_opportunity_filtered"
            or inputs["training_pins"]["deploy_feature_rows_sha256"] != sha([])):
        raise ValueError("training/deploy scope mismatch")
    selected = [r for r in training["rows"] if r["proposal_long"]]
    scaler_rows = [{k: row["features"][k] for k in guard["FEATURES"]} for row in selected]
    expected_candidate = {"selection": "all_proposal_long_outcomes_including_losses", "rows": scaler_rows,
                          "rows_sha256": sha(scaler_rows), "opportunities": len(selected),
                          "negative_opportunities": sum(r["gross_return_3d"] < 0 for r in selected),
                          "positive_opportunities": sum(r["gross_return_3d"] > 0 for r in selected),
                          "actual_scaler_fitted": False}
    candidate = inputs["scaler_candidate_input"]
    guard["keys"](candidate, expected_candidate, "scaler candidate")
    if candidate["actual_scaler_fitted"] is not False or any(type(candidate[k]) is not int for k in ("opportunities", "negative_opportunities", "positive_opportunities")):
        raise ValueError("typed unfitted scaler candidate required")
    guard["exact_list"](candidate["rows"])
    for row in candidate["rows"]:
        guard["keys"](row, guard["FEATURES"], "scaler feature row")
        for value in row.values():
            guard["number"](value)
    if candidate != expected_candidate:
        raise ValueError("wrong opportunity-filtered scaler input, including outcome selection")
    if (type(origin["minimum_opportunities_64_passed"]) is not bool
            or origin["minimum_opportunities_64_passed"] != (len(selected) >= 64)):
        raise ValueError("unearned opportunity gate")
    deploy = origin["deploy_features"]
    guard["keys"](deploy, DEPLOY_KEYS, "asof deploy feature candidate")
    if (deploy["schema"] != "exia.genetic_primus.synthetic_asof_feature_candidate/1"
            or deploy["evidence_class"] != "synthetic_candle_engineering_only"
            or deploy["availability_is_declared_not_measured"] is not True or deploy["safety"] != origin["safety"]
            or any(deploy["safety"][k] is not False for k in guard["SAFETY_KEYS"])
            or any(deploy[k] is not False for k in ("real_indicator_equivalence_verified", "market_data_provenance_verified", "market_research_launch_ready", "real_money_ready"))
            or any(type(deploy[k]) is not int or deploy[k] != 0 for k in ("new_model_fits", "new_market_strategy_evaluations"))):
        raise ValueError("deploy authority/scope changed")
    if (deploy["kernel_definition_sha256"] != training["binding"]["feature_definition_sha256"]
            or origin["shared_feature_definition_sha256"] != deploy["kernel_definition_sha256"]
            or deploy["decision_asof_ms"] != origin["context"]["decision_asof_ms"]):
        raise ValueError("different train/deploy feature contract")
    self_hash(deploy, guard)
    if deploy["feature_rows_sha256"] != sha(deploy["rows"]):
        raise ValueError("deploy feature hash")
    if [r["symbol"] for r in deploy["rows"]] != origin["context"]["universe"]:
        raise ValueError("deploy universe")
    for row in deploy["rows"]:
        guard["keys"](row, guard["DEPLOY_ROW_KEYS"], "label-free deploy row")
        guard["feature_row"](row, origin["context"], training["binding"], origin["context"]["decision_asof_ms"])
    return training, selected, scaler_rows


def admit(origin, proposal, guard):
    if sha(proposal) != PROPOSAL_SHA or proposal["status"] != "draft_not_executable":
        raise ValueError("pinned separate draft required")
    training, selected, scaler_rows = validate_origin(origin, guard)
    projected = [{k: row[k] for k in WEIGHT_ROW_KEYS} for row in selected]
    asof = origin["context"]["training_asof_ms"]
    timing = timing_gate(origin["context"], proposal["clock"]["training_latency_budget_seconds"],
                         proposal["clock"]["execution_delay_seconds"], guard)
    common = ["proposal_not_executable", "training_authority_disabled", "synthetic_only_not_market_admission",
              "actual_data_provenance_unverified", "actual_latency_unverified",
              "execution_aligned_label_definition_unregistered", "normal_stress_cost_contract_unregistered"] + timing["denial_reasons"]
    plans = []
    for variant in VARIANTS:
        bank = weight_bank(projected, variant, asof, guard) if projected else None
        n = len(projected)
        ess = bank["kish_weight_ess"] if bank else 0.0
        min_n, min_ess = proposal["models"]["minimum_completed_opportunities"], proposal["models"]["minimum_weight_ess"]
        reasons = list(common)
        if n < min_n:
            reasons.append("insufficient_completed_opportunities")
        if ess < min_ess:
            reasons.append("insufficient_kish_weight_ess")
        plans.append({"variant": variant, "opportunities": n, "minimum_opportunities": min_n,
                      "kish_weight_ess": ess, "minimum_weight_ess": min_ess,
                      "opportunity_gate_passed": n >= min_n, "weight_ess_gate_passed": ess >= min_ess,
                      "sample_gates_passed": n >= min_n and ess >= min_ess,
                      "weight_bank": bank, "scaler_input_scope": "all_proposal_long_outcomes_including_losses",
                      "scaler_candidate_rows_sha256": sha(scaler_rows), "fit_allowed": False,
                      "fallback": "NoTrade_no_fit_no_orders_not_a_trading_decision", "denial_reasons": reasons})
    result = {"schema": "exia.genetic_primus.synthetic_pre_fit_admission/1", "evidence_class": "synthetic_pre_fit_metadata_only",
              "proposal_semantic_sha256": PROPOSAL_SHA, "origin_id": origin["context"]["origin_id"],
              "training_input_sha256": origin["training_inputs"]["payload_sha256"],
              "training_payload_sha256": training["payload_sha256"],
              "timing": timing, "plans": plans, "minimum_trade_count_forcing": False,
              "scaler_actual_fit_verified": False, "fit_allowed": False,
              "new_model_fits": 0, "new_market_strategy_evaluations": 0,
              "market_research_launch_ready": False, "real_money_ready": False,
              "safety": {k: False for k in guard["SAFETY_KEYS"]}}
    result["payload_sha256"] = sha(result)
    return result
