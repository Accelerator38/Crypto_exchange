"""Pure registration/timing checks; no fitter, prices, market I/O or authority."""
from __future__ import annotations

import hashlib
import json
import math

BAR_MS = 14_400_000
HORIZON_MS = 18 * BAR_MS
PROPOSAL_SHA = "a5abd987351f2ef60ad359fd396f6c2879e4c45ee166e08fde7ec2807b94e637"
SAFETY = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
          "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")
POLICIES = {
    "scope": "research_ohlcv_sensitivity_not_release_execution",
    "clock": "UTC_integer_epoch_ms",
    "bar_ms": BAR_MS,
    "deployment_horizon_ms": HORIZON_MS,
    "retrain_cadence_ms": HORIZON_MS,
    "purge_bars": 7,
    "execution": "first_bar_open_strictly_after_decision_plus_registered_delay",
    "label": "gross_entry_open_to_next_scheduled_rebalance_open_same_delay",
    "label_availability": "max_entry_and_exit_open_observation_availability",
    "label_cutoff": "label_available_at_ms <= training_asof_ms - 7 * bar_ms",
    "decision_features": "closed_available_prefix_only_no_labels",
    "training_identity": "training_only_prefix_not_full_archive_or_deploy_hash",
    "cost_accounting": "continuous_cash_quantity_per_fill_costs_observed_funding",
    "unchanged_target": "carry_no_refill_no_retrain_forced_liquidation",
    "terminal": "single_registered_global_end_liquidation_with_all_costs",
    "outcome_weights": "all_completed_outcomes_including_losses",
    "missing_or_revised_data": "fail_closed_no_forward_fill",
    "fixed_delay_limitation": "planned_sensitivity_not_actual_latency_or_fill_evidence",
}
DATA_KEYS = ("manifest_sha256", "timestamp_spec_sha256", "availability_evidence_sha256",
             "revision_policy_sha256", "universe_sha256", "feature_definition_sha256",
             "training_prefix_manifest_sha256", "snapshot_cutoff_ms")
COST_KEYS = ("adapter_sha256", "normal_fee_bps_per_side", "normal_slippage_bps_per_side",
             "stress_fee_bps_per_side", "stress_slippage_bps_per_side",
             "calibration_evidence_sha256", "funding_evidence_sha256", "borrow_basis_sha256",
             "terminal_policy_sha256", "instrument_spec_sha256")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("exact schema required")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum or value > 2**53 - 1:
        raise ValueError("bounded integer required")
    return value


def digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA256 required")


def cost(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10_000:
        raise ValueError("finite nonnegative bps required")


def assess(contract):
    """Registration completeness only, never actual provenance or fit admission."""
    keys(contract, ("schema", "status", "proposal_semantic_sha256", "policies", "timing",
                    "data", "costs", "historical_results_requalified", "safety"))
    if contract["schema"] != "exia.genetic_primus.research_execution_contract/1":
        raise ValueError("schema mismatch")
    if contract["status"] not in ("draft_unregistered", "registered_research_only"):
        raise ValueError("research status required")
    if contract["proposal_semantic_sha256"] != PROPOSAL_SHA:
        raise ValueError("proposal mismatch")
    if canonical(contract["policies"]) != canonical(POLICIES):
        raise ValueError("policy mismatch")
    if contract["historical_results_requalified"] is not False:
        raise ValueError("historical results cannot be requalified")
    keys(contract["safety"], SAFETY)
    if any(contract["safety"][key] is not False for key in SAFETY):
        raise ValueError("authority must remain literal false")
    keys(contract["timing"], ("training_budget_ms", "execution_delay_ms", "budget_evidence_sha256"))
    keys(contract["data"], DATA_KEYS)
    keys(contract["costs"], COST_KEYS)
    missing = []
    for section in ("timing", "data", "costs"):
        for key, value in contract[section].items():
            if value is None:
                missing.append(section + "." + key)
            elif key.endswith("sha256"):
                digest(value)
            elif section == "costs":
                cost(value)
            else:
                integer(value, 1 if key == "training_budget_ms" else 0)
    delay = contract["timing"]["execution_delay_ms"]
    if delay is not None and delay >= BAR_MS:
        raise ValueError("this sensitivity requires registered delay shorter than one bar")
    for metric in ("fee", "slippage"):
        normal = contract["costs"]["normal_" + metric + "_bps_per_side"]
        stress = contract["costs"]["stress_" + metric + "_bps_per_side"]
        if normal is not None and stress is not None and stress < normal:
            raise ValueError("stress costs cannot be below normal costs")
    complete = not missing and contract["status"] == "registered_research_only"
    return {"contract_sha256": sha(contract), "missing_registration_fields": sorted(missing),
            "registration_complete": complete, "actual_data_or_execution_verified": False,
            "fit_allowed": False, "market_research_launch_ready": False, "real_money_ready": False,
            "safety": dict(contract["safety"])}


def execution_window(origin_start_ms, decision_asof_ms, execution_delay_ms):
    """Plan two opens, not fills. Shift both boundaries, never shorten 3 days."""
    for value in (origin_start_ms, decision_asof_ms, execution_delay_ms):
        integer(value)
    if origin_start_ms % BAR_MS or not origin_start_ms <= decision_asof_ms < origin_start_ms + BAR_MS:
        raise ValueError("aligned origin and decision in its first bar required")
    if execution_delay_ms >= BAR_MS:
        raise ValueError("delay exceeds this sensitivity scope")
    ready = integer(decision_asof_ms + execution_delay_ms)
    entry = integer((ready // BAR_MS + 1) * BAR_MS)
    exit_open = integer(entry + HORIZON_MS)
    return {"origin_start_ms": origin_start_ms, "decision_asof_ms": decision_asof_ms,
            "execution_delay_ms": execution_delay_ms, "ready_at_ms": ready,
            "planned_entry_open_ms": entry, "planned_exit_open_ms": exit_open,
            "next_scheduled_decision_asof_ms": integer(decision_asof_ms + HORIZON_MS),
            "gross_label_definition_sha256": sha({"label": POLICIES["label"],
                                                  "execution": POLICIES["execution"],
                                                  "horizon_ms": HORIZON_MS,
                                                  "execution_delay_ms": execution_delay_ms}),
            "actual_fill_verified": False}


def completed_label_gate(window, entry_open_available_at_ms, exit_open_available_at_ms, training_asof_ms):
    keys(window, execution_window(0, 0, 0).keys())
    expected = execution_window(window["origin_start_ms"], window["decision_asof_ms"],
                                window["execution_delay_ms"])
    if canonical(window) != canonical(expected):
        raise ValueError("execution window substitution")
    for value in (entry_open_available_at_ms, exit_open_available_at_ms, training_asof_ms):
        integer(value)
    if entry_open_available_at_ms < window["planned_entry_open_ms"] or exit_open_available_at_ms < window["planned_exit_open_ms"]:
        raise ValueError("observation cannot precede its open")
    cutoff = training_asof_ms - 7 * BAR_MS
    available = max(entry_open_available_at_ms, exit_open_available_at_ms)
    return {"label_available_at_ms": available, "training_cutoff_ms": cutoff,
            "completed_and_purged": available <= cutoff, "fit_allowed": False}


def origin_binding_gate(contract, evidence):
    """Check claimed metadata, NOT the existence/correctness of fits or market data."""
    registration = assess(contract)
    hash_keys = ("contract_sha256", "training_prefix_manifest_sha256", "training_rows_sha256",
                 "opportunity_feature_rows_sha256", "scaler_fit_rows_sha256", "scaler_payload_sha256",
                 "model_payload_sha256", "decision_model_sha256", "folds_sha256",
                 "resample_bank_sha256", "audit_bank_sha256")
    time_keys = ("origin_start_ms", "training_asof_ms", "training_cutoff_ms", "fit_started_at_ms",
                 "fit_completed_at_ms", "model_available_at_ms", "decision_asof_ms",
                 "submitted_at_ms", "planned_entry_open_ms", "planned_exit_open_ms")
    keys(evidence, hash_keys + time_keys)
    for key in hash_keys:
        digest(evidence[key])
    for key in time_keys:
        integer(evidence[key])
    if evidence["contract_sha256"] != registration["contract_sha256"]:
        raise ValueError("contract binding mismatch")
    prefix = contract["data"]["training_prefix_manifest_sha256"]
    if prefix is not None and prefix != evidence["training_prefix_manifest_sha256"]:
        raise ValueError("training prefix manifest substitution")
    if evidence["scaler_fit_rows_sha256"] != evidence["opportunity_feature_rows_sha256"]:
        raise ValueError("scaler must bind opportunity feature rows, not all labels")
    if evidence["model_payload_sha256"] != evidence["decision_model_sha256"]:
        raise ValueError("decision model substitution")
    if evidence["resample_bank_sha256"] == evidence["audit_bank_sha256"]:
        raise ValueError("audit bank must differ; digest inequality alone is not independence proof")
    if evidence["training_cutoff_ms"] != evidence["training_asof_ms"] - 7 * BAR_MS:
        raise ValueError("training cutoff substitution")
    clock = ("training_asof_ms", "fit_started_at_ms", "fit_completed_at_ms",
             "model_available_at_ms", "decision_asof_ms", "submitted_at_ms")
    if any(evidence[a] > evidence[b] for a, b in zip(clock, clock[1:])):
        raise ValueError("noncausal declared timing")
    budget = contract["timing"]["training_budget_ms"]
    if budget is not None and evidence["fit_completed_at_ms"] - evidence["fit_started_at_ms"] > budget:
        raise ValueError("declared fit exceeds registered budget")
    delay = contract["timing"]["execution_delay_ms"]
    reasons = [] if registration["registration_complete"] else ["contract_unregistered"]
    if delay is None:
        reasons.append("execution_delay_unregistered")
    else:
        planned = execution_window(evidence["origin_start_ms"], evidence["decision_asof_ms"], delay)
        if any(evidence[key] != planned[key] for key in ("planned_entry_open_ms", "planned_exit_open_ms")):
            raise ValueError("execution boundary substitution")
        if evidence["submitted_at_ms"] > planned["ready_at_ms"]:
            raise ValueError("submission exceeds declared decision-to-ready delay")
    return {"declared_bindings_consistent": not reasons, "denial_reasons": reasons,
            "claimed_metadata_only": True, "actual_fit_or_market_data_verified": False,
            "audit_seed_independence_verified": False, "fit_allowed": False}
