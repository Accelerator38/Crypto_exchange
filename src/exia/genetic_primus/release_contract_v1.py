"""Offline release identity and blockers. This module cannot authorize trading."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA = "exia.genetic_primus.release_candidate/1"
SAFETY_KEYS = (
    "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
    "promotion_authority", "runtime_authority", "network_allowed",
)
FEATURES = ["ema_gap_12_48", "ema_slope_12_3", "return_12", "efficiency_12"]
REQUIRED_EVIDENCE = (
    "instrument_rules", "account_fee_schedule", "book_mark_funding_dataset",
    "latency_slippage_calibration", "independent_statistical_audit",
    "global_objective_registry_audit", "shadow_fault_reconciliation_audit",
    "risk_owner_approval",
)
RISK_FIELDS = (
    "capital_usdt", "maximum_order_notional_usdt", "maximum_loss_usdt",
    "maximum_drawdown_fraction",
)
QUALIFICATION_GATES = {
    "comparison_alignment": "same_events_universe_capital_costs_and_nonoverlapping_3d_origins",
    "references": ["EMA_TrendConsensus", "NoTrade"],
    "normal_paired_incremental_lcb": "strictly_positive_against_both_references",
    "stress_paired_incremental_lcb": "strictly_positive_against_both_references",
    "minimum_positive_outer_fraction": "0.70",
    "confidence_protocol": "independently_calibrated_and_preregistered_dependent_returns",
    "independent_audit_bank_must_confirm": True,
    "drawdown": "absolute_owner_budget_and_no_worse_than_actively_trading_incumbent",
    "opportunities_ess_trades": "separate_preregistered_minima",
    "turnover_concentration_drift": "preregistered_execution_calibrated_thresholds",
    "multiple_testing": "global_immutable_objective_registry_and_DSR_PBO_audit",
    "sample_size": "preregistered_power_and_effective_sample_size_not_calendar_only",
}


def object_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def is_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def utc_ms(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
        raise ValueError("explicit UTC timestamp required")
    return int(dt.timestamp() * 1000)


def _positive_decimal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return Decimal(value).is_finite() and Decimal(value) > 0
    except InvalidOperation:
        return False


def validate_contract(contract: dict, *, root: Path | None = None) -> dict:
    """Check structure/source integrity separately from unavailable qualification.

    Evidence labels/hashes alone are NOT attestations. Even a fully populated
    contract requires an external review and a separate deployment authority.
    No reading credentials, venue APIs, model runtime, or evidence executables.
    """
    errors: list[str] = []
    blockers: list[str] = []
    if contract.get("schema_version") != SCHEMA:
        errors.append("unsupported_schema")
    safety = contract.get("safety", {})
    if not isinstance(safety, dict) or set(safety) != set(SAFETY_KEYS) or any(
            safety.get(k) is not False for k in SAFETY_KEYS):
        errors.append("all_safety_permissions_must_be_false")
    expected = {
        "venue": "bitget", "product": "USDT-FUTURES", "settlement": "USDT",
        "position_mode": "one_way", "instrument_type": "linear_perpetual",
        "adapter": "offline_snapshot_ioc_v1", "live_adapter": None,
    }
    if contract.get("execution_target") != expected:
        errors.append("unsupported_execution_target")
    if contract.get("policy") != {
        "kind": "continuous_binary_sparse_ema_overlay", "genome": [0, 0, 1, 1, -1],
        "features": FEATURES, "allow_rule": "finite_features_and_logit_strictly_positive",
        "missing_features": "abstain", "same_direction": "hold_filled_quantity",
        "zero_target": "reduce_only_exit", "reverse_target": "single_ioc_net_delta",
        "partial_fill": "keep_actual_position_no_synthetic_completion",
    }:
        errors.append("policy_not_frozen_continuous_overlay")
    if contract.get("temporal") != {
        "timeframe_ms": 14400000, "deployment_horizon_ms": 259200000,
        "retrain_cadence_ms": 259200000, "purge_bars": 7,
        "train_start_utc": "2022-01-01T00:00:00Z", "window": "expanding",
        "train_end_rule": "origin_start_minus_7_complete_bars_exclusive",
        "scaler": "median_iqr_clip_8_refit_each_origin", "genome_refit": False,
    }:
        errors.append("temporal_contract_mismatch")
    if contract.get("execution_semantics") != {
        "orders": "marketable_limit_IOC_taker_only", "clock": "integer_event_time_ms",
        "fill_rule": "first_fresh_snapshot_received_after_latency",
        "liquidity": "shared_per_snapshot_price_level_participation_cap",
        "funding": "explicit_settlement_before_same_time_fills",
        "marking": "observed_mark_only", "ohlcv_fill_fallback": False,
        "liquidation_simulated": False, "synthetic_evidence_can_qualify": False,
    }:
        errors.append("execution_semantics_mismatch")
    if contract.get("qualification_gates") != QUALIFICATION_GATES:
        errors.append("qualification_gate_contract_mismatch")
    universe = contract.get("symbol_map", {})
    expected_symbols = {s + "/USDT" for s in ("ADA", "BNB", "BTC", "DOGE", "ETH", "LINK", "SOL", "XRP")}
    if not isinstance(universe, dict) or set(universe) != expected_symbols or any(
            not isinstance(k, str) or v != k.replace("/", "") or not k.endswith("/USDT")
            for k, v in universe.items()):
        errors.append("invalid_full8_symbol_map")
    pins = contract.get("source_pins", {})
    required_pins = {
        "configs/genetic_primus_prospective_3d_v1.json",
        "src/exia/genetic_primus/measurement_lab_v2.py",
        "src/exia/genetic_primus/accounting_v2.py",
    }
    if not isinstance(pins, dict) or set(pins) != required_pins:
        errors.append("source_pin_set_mismatch")
        pins = {}
    for name, digest in pins.items():
        if not is_sha(digest):
            errors.append("invalid_source_sha:" + name)
        elif root is not None:
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append("source_missing:" + name)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append("source_changed:" + name)
    if root is None:
        blockers.append("source_pins_not_verified")
    if not is_sha(contract.get("research_model_sha256")):
        errors.append("invalid_research_model_sha")
    risk = contract.get("risk_budget", {})
    if not isinstance(risk, dict):
        errors.append("invalid_risk_budget")
        risk = {}
    if risk.get("gross_notional_to_equity_ceiling") != "1":
        errors.append("unlevered_ceiling_must_equal_one")
    for field in RISK_FIELDS:
        if risk.get(field) is None:
            blockers.append("risk_budget_missing:" + field)
        elif not _positive_decimal(risk[field]):
            errors.append("invalid_risk_budget:" + field)
    if _positive_decimal(risk.get("maximum_drawdown_fraction")) and Decimal(
            risk["maximum_drawdown_fraction"]) >= 1:
        errors.append("drawdown_fraction_must_be_below_one")
    evidence = contract.get("evidence", {})
    if not isinstance(evidence, dict) or set(evidence) != set(REQUIRED_EVIDENCE):
        errors.append("evidence_set_mismatch")
        evidence = {}
    for key in REQUIRED_EVIDENCE:
        item = evidence.get(key)
        if item is None:
            blockers.append("evidence_missing:" + key)
        elif not isinstance(item, dict) or not is_sha(item.get("sha256")):
            errors.append("invalid_evidence:" + key)
        elif item.get("synthetic") is not False:
            blockers.append("non_qualifying_evidence:" + key)
        else:
            blockers.append("external_evidence_review_required:" + key)
    blockers.extend([
        "venue_adapter_margin_liquidation_not_qualified",
        "separate_deployment_contract_and_explicit_approval_required",
    ])
    return {"schema_version": "exia.genetic_primus.release_check/1",
            "contract_sha256": object_sha(contract), "structurally_valid": not errors,
            "errors": errors, "qualification_blockers": blockers,
            "status": "INVALID" if errors else "OFFLINE_CANDIDATE_UNQUALIFIED",
            "performance_gate_evaluation": "NOT_RUN",
            "release_ready": False, "trading_authorized": False}


def bind_model(contract: dict, *, origin_start_ms: int, scaler: dict,
               data_sha256: str, train_rows_sha256: str, folds_sha256: str,
               resample_bank_sha256: str, audit_bank_sha256: str,
               implementation_sha256: str) -> dict:
    """Create a new deployment-origin identity, never reuse the research SHA.

    Caller must supply train-only scaler/provenance, not just a model label.
    This verifies bounds and identities, not truth of a claimed training dataset.
    """
    if not validate_contract(contract)["structurally_valid"]:
        raise ValueError("invalid release contract")
    tf = contract["temporal"]["timeframe_ms"]
    if type(origin_start_ms) is not int or origin_start_ms % tf:
        raise ValueError("origin must align to a complete 4h boundary")
    cutoff = origin_start_ms - 7 * tf
    train_start = utc_ms(contract["temporal"]["train_start_utc"])
    if cutoff <= train_start:
        raise ValueError("empty training history")
    if scaler.get("train_end_exclusive_ms") != cutoff or scaler.get("train_start_ms") != train_start:
        raise ValueError("scaler cutoff/window mismatch")
    if type(scaler.get("fitted_rows")) is not int or scaler["fitted_rows"] <= 0:
        raise ValueError("empty scaler fit")
    for key in ("median", "scale"):
        values = scaler.get(key, [])
        if len(values) != 4 or any(not isinstance(v, str) for v in values):
            raise ValueError("four decimal-string scaler values required")
        try:
            valid = all(Decimal(v).is_finite() and (key != "scale" or Decimal(v) > 0) for v in values)
        except InvalidOperation:
            valid = False
        if not valid:
            raise ValueError("invalid scaler")
    hashes = {"data_sha256": data_sha256, "train_rows_sha256": train_rows_sha256,
              "folds_sha256": folds_sha256, "resample_bank_sha256": resample_bank_sha256,
              "audit_bank_sha256": audit_bank_sha256,
              "implementation_sha256": implementation_sha256,
              "scaler_fit_values_sha256": scaler.get("fit_values_sha256")}
    if not all(is_sha(v) for v in hashes.values()):
        raise ValueError("all provenance hashes required")
    if resample_bank_sha256 == audit_bank_sha256:
        raise ValueError("audit resample bank must be independent")
    body = {"schema_version": "exia.genetic_primus.release_model/1",
            "contract_sha256": object_sha(contract), "origin_start_ms": origin_start_ms,
            "origin_end_exclusive_ms": origin_start_ms + 259200000,
            "train_end_exclusive_ms": cutoff, "scaler": scaler,
            "policy": contract["policy"], "provenance": hashes}
    # Detached body: caller mutation cannot alter the signed object in place.
    return {"model": json.loads(json.dumps(body)), "model_sha256": object_sha(body)}
