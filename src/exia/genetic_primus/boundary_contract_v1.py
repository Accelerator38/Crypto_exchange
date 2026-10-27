"""Pure engineering boundary rules; no model, data loader or simulator imports."""
from __future__ import annotations

import hashlib
import json
import math
import re

DAY_MS = 86_400_000
BAR_MS = 14_400_000
SHA = re.compile(r"^[0-9a-f]{64}$")
SAFETY = {
    "credentials_allowed": False, "orders_enabled": False,
    "paper": False, "live": False, "promotion_authority": False,
    "execution_allowed": False, "training_allowed": False,
    "financial_evaluation_allowed": False, "network_allowed": False,
}
REFERENCES = {
    "src/exia/genetic_primus/accounting_v2.py": "f9294a2e4b2220aa7552ebed7d2480e7e7647744fe5a62330888acba569af5bb",
    "src/exia/genetic_primus/measurement_lab_v2.py": "ff9ec7f12412190426ebffe57d817fca9ac0d0643e628c15646e72ff84b3f648",
    "src/exia/genetic_primus/ablation_v1.py": "429082fda31e371e71e2b2420e613c0681b08357d2495985b4ba8a8b8e645312",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def expected_contract():
    return {
        "schema": "exia.genetic_primus.forward_boundary_engineering_contract/1",
        "contract_id": "boundary_carry_engineering_v1_20261005",
        "scope": "synthetic_boundary_fixtures_only_not_research_objective",
        "supersedes_historical_or_prospective_contracts": False,
        "safety": dict(SAFETY),
        "time": {"unit": "utc_epoch_ms", "bar_ms": BAR_MS,
                 "interval": "half_open_start_inclusive_end_exclusive",
                 "retrain_cadence_ms": 3 * DAY_MS,
                 "deployment_horizon_ms": 3 * DAY_MS,
                 "partial_origins": "reject", "purge_bars": 7,
                 "train_end_exclusive": "origin_open",
                 "training_feature_open_strictly_before": "origin_open_minus_7_bars",
                 "training_label_end_no_later_than": "origin_open"},
        "capital": {"initial_equity": 1.0,
                    "sleeves": "equal_initial_fixed_universe_no_cross_rebalance",
                    "across_origins": "continuous_no_reset_no_rebase",
                    "aggregation": "compound_same_trajectory_returns_not_sum"},
        "boundary": {
            "retrain_event_causes_trade": False,
            "unchanged_target": "carry_quantity_cash_and_entry_model_identity",
            "changed_target": "exit_then_optional_enter_at_next_open",
            "interval_terminal": "liquidate_at_final_candle_close_include_fees",
            "force_flat_timestamps": [],
            "record_details": True,
            "force_flat_argument_semantics": "open_time_exit_not_previous_close",
            "origin_close_liquidation_supported": False,
            "terminal_after_final_decision": True,
        },
        "ledger": {"decision_model_sha256": "active_model_at_decision",
                   "position_entry_model_sha256": "entry_model_retained_until_exit",
                   "signal_time": "previous_candle_close_equal_next_open_boundary",
                   "same_timestamp_order": "previous_close_then_retrain_then_decision_then_open_fill",
                   "model_must_be_available_before_fill": True,
                   "training_latency": "unverified_no_execution_authority"},
        "costs": {"normal_round_trip_bps": 16.0, "stress_round_trip_bps": 24.0,
                  "status": "reference_synthetic_costs_not_exchange_calibration",
                  "fee_basis": "actual_fill_notional", "funding_borrow_verified": False},
        "launch_gate": {"real_evaluator_adapter_verified": False,
                        "runtime_dependency_graph_verified": False,
                        "environment_verified": False,
                        "data_and_model_payloads_verified": False,
                        "required": ["explicit_all_call_parameters", "transitive_dependency_byte_hashes",
                                     "python_and_distribution_versions", "data_folds_resample_cutoff_hashes",
                                     "model_scaler_payload_hashes", "before_after_input_identity",
                                     "actual_loaded_dependency_identity", "boundary_adapter_regressions",
                                     "separate_user_authorization_for_financial_run"]},
        "reviewed_source_references": dict(REFERENCES),
    }


def validate_contract(value):
    # Canonical bytes distinguish false from 0, and 1.0 from true.
    if canonical(value) != canonical(expected_contract()):
        raise ValueError("unregistered contract change; new version and review required")
    return value


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return value


def model_sha(value):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError("model identity must be a lowercase SHA256")
    return value


def boundary_plan(contract, start_ms, end_ms):
    validate_contract(contract)
    integer(start_ms, "start"); integer(end_ms, "end")
    cadence = contract["time"]["retrain_cadence_ms"]
    if start_ms % BAR_MS or end_ms % BAR_MS or end_ms <= start_ms:
        raise ValueError("invalid aligned half-open interval")
    if (end_ms - start_ms) % cadence:
        raise ValueError("partial origin rejected")
    count = (end_ms - start_ms) // cadence
    if count > 100:
        raise ValueError("engineering fixture capped at 100 origins")
    origins = [{"origin_open_ms": start_ms + i * cadence,
                "deployment_end_exclusive_ms": start_ms + (i + 1) * cadence,
                "train_end_exclusive_ms": start_ms + i * cadence,
                "training_feature_open_before_ms": start_ms + i * cadence - 7 * BAR_MS,
                "training_label_end_at_or_before_ms": start_ms + i * cadence,
                "capital_reset": False, "forced_trade": False} for i in range(count)]
    return {"contract_sha256": digest(contract), "origins": origins,
            "explicit_accounting_parameters": {
                "start_timestamp": start_ms, "end_timestamp": end_ms,
                "timeframe_ms": BAR_MS, "force_flat_timestamps": [],
                "record_details": True},
            "round_trip_cost_bps_by_scenario": {"normal": 16.0, "stress": 24.0},
            "terminal_close_ms": end_ms, "forced_origin_exits": [],
            "adapter_executed": False, "financial_evaluation_allowed": False}


def boundary_transition(contract, *, position, target, entry_model_sha256,
                        decision_model_sha256, origin_open_ms, signal_close_ms,
                        model_available_ms, interval_end_ms):
    """Decision identity specification, not a fill simulator or order command."""
    validate_contract(contract)
    for name, value in (("position", position), ("target", target)):
        if type(value) is not int or value not in (-1, 0, 1):
            raise ValueError(f"{name}: signed discrete direction required")
    for name, value in (("origin", origin_open_ms), ("signal", signal_close_ms),
                        ("available", model_available_ms), ("end", interval_end_ms)):
        integer(value, name)
    if origin_open_ms % BAR_MS or interval_end_ms % BAR_MS:
        raise ValueError("unaligned boundary")
    if signal_close_ms != origin_open_ms or model_available_ms > origin_open_ms:
        raise ValueError("boundary signal/model unavailable before fill")
    if origin_open_ms >= interval_end_ms:
        raise ValueError("terminal boundary must not create a new decision")
    model_sha(decision_model_sha256)
    if position:
        model_sha(entry_model_sha256)
    elif entry_model_sha256 is not None:
        raise ValueError("flat position cannot claim an entry model")
    if position == target:
        events = ["hold"] if position else ["abstain"]
        resulting_entry = entry_model_sha256
    else:
        events = (["exit"] if position else []) + (["enter"] if target else [])
        resulting_entry = decision_model_sha256 if target else None
    return {"semantic_events": events, "position_after": target,
            "decision_model_sha256": decision_model_sha256,
            "position_entry_model_sha256": resulting_entry,
            "capital_reset": False, "quantity_resized_for_retrain": False,
            "orders_emitted": False}


def long_cash_identity_fixture(capital, entry_price, exit_price, round_trip_bps):
    """One artificial long-entry/exit algebra identity, no bars/market input."""
    for value in (capital, entry_price, exit_price, round_trip_bps):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("finite fixture numbers required")
    if min(capital, entry_price, exit_price) <= 0 or not 0 <= round_trip_bps <= 100:
        raise ValueError("fixture outside bounded positive domain")
    fee_rate = round_trip_bps / 20000.0
    quantity = capital / (1.0 + fee_rate) / entry_price
    entry_fee = quantity * entry_price * fee_rate
    exit_fee = quantity * exit_price * fee_rate
    gross_pnl = quantity * (exit_price - entry_price)
    net_pnl = gross_pnl - entry_fee - exit_fee
    if not all(math.isfinite(value) for value in
               (quantity, entry_fee, exit_fee, gross_pnl, net_pnl, capital + net_pnl)):
        raise ValueError("nonfinite fixture arithmetic")
    return {"quantity": quantity, "gross_pnl": gross_pnl,
            "fees": entry_fee + exit_fee, "net_pnl": net_pnl,
            "ending_equity": capital + net_pnl, "synthetic_fixture_only": True}
