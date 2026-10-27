"""Frozen definitions + historical engineering plan, without data I/O or fit authority."""
from __future__ import annotations

from datetime import datetime, timezone

SOURCES = {
    "proposal": "configs/genetic_primus_forward_research_proposal_v1.json",
    "contract": "configs/genetic_primus_research_execution_contract_v1.json",
    "coverage": "Reports/Exia/Genetic_Primus/registration_coverage_v1_20261007/coverage.json",
    "manifest": "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json",
    "old_schedule": "configs/genetic_primus_prospective_3d_v1.json",
    "feature_definition": "Reports/Exia/Genetic_Primus/asof_features_v1_20261007/kernel_definition.json",
    "release": "configs/genetic_primus_release_candidate_v1.json",
    "linear_source": "src/exia/genetic_primus/rolling_linear_retrain_v1.py",
}


def utc_ms(value):
    if type(value) is not str or len(value) != 20 or not value.endswith("Z"):
        raise ValueError("second-resolution UTC ISO required")
    moment = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(moment.timestamp()) * 1000


def iso_ms(value):
    return datetime.fromtimestamp(value // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def select_engineering_origin(coverage_start_ms, coverage_end_exclusive_ms, anchor_ms, api):
    """Only coverage/calendar arithmetic, no outcomes or trading thresholds."""
    b, h = api["BAR_MS"], api["HORIZON_MS"]
    for value in (coverage_start_ms, coverage_end_exclusive_ms, anchor_ms):
        api["integer"](value)
        if value % b:
            raise ValueError("aligned coverage and anchor required")
    if coverage_start_ms >= coverage_end_exclusive_ms or anchor_ms < coverage_start_ms:
        raise ValueError("valid coverage/schedule required")
    # decision is in origin's first bar and 0 <= delay < bar, so entry shift is at most 2 bars.
    # exit OPEN must be strictly inside archive, not at its exclusive end.
    last = (coverage_end_exclusive_ms - h - 2 * b - 1 - anchor_ms) // h
    if last < 0:
        raise ValueError("no historical origin fits delayed label envelope")
    start = anchor_ms + last * h
    if start - 8 * b < coverage_start_ms + 48 * b:
        raise ValueError("insufficient planned training/warmup history")
    return {"schedule_anchor_ms": anchor_ms, "origin_start_ms": start,
            "nominal_deployment_end_exclusive_ms": start + h,
            "training_asof_ms": start - b, "training_cutoff_ms": start - 8 * b,
            "decision_window_start_ms": start, "decision_window_end_exclusive_ms": start + b,
            "maximum_planned_entry_shift_ms": 2 * b,
            "latest_required_exit_open_ms": start + h + 2 * b,
            "coverage_end_exclusive_ms": coverage_end_exclusive_ms,
            "origin_start_utc": iso_ms(start), "nominal_deployment_end_exclusive_utc": iso_ms(start + h),
            "latest_required_exit_open_utc": iso_ms(start + h + 2 * b),
            "selection_rule": "latest_old_schedule_origin_whose_worst_case_delayed_exit_open_is_inside_metadata_coverage",
            "selection_uses_outcomes": False, "actual_availability_verified": False,
            "training_asof_is_declared_planning_rule_not_measured": True,
            "proposed_only": True, "authorized_market_runs": 0}


def build(materials, identities, api):
    api["keys"](materials, SOURCES)
    api["keys"](identities, SOURCES)
    for role in SOURCES:
        api["keys"](identities[role], ("bytes", "sha256"))
        api["integer"](identities[role]["bytes"], 1)
        api["digest"](identities[role]["sha256"])
        if type(materials[role]) is not (str if role == "linear_source" else dict):
            raise ValueError("source type mismatch")
    proposal = materials["proposal"]
    if api["sha"](proposal) != api["PROPOSAL_SHA"]:
        raise ValueError("frozen proposal mismatch")
    assessment = api["assess"](materials["contract"])
    coverage = materials["coverage"]
    if coverage["contract_sha256"] != assessment["contract_sha256"]:
        raise ValueError("coverage contract mismatch")
    if coverage["registered_fields"] != 0 or coverage["remaining_registration_fields"] != 21:
        raise ValueError("baseline registration changed")
    if coverage["fit_allowed"] is not False or coverage["auto_registration_allowed"] is not False:
        raise ValueError("coverage authority mismatch")
    rows = coverage["rows"]
    if type(rows) is not list or len(rows) != 21 or len({r["field"] for r in rows}) != 21:
        raise ValueError("exact coverage bank required")
    if set(r["field"] for r in rows) != set(assessment["missing_registration_fields"]):
        raise ValueError("coverage field substitution")
    if any(r["registration_value"] is not None or r["auto_registration_allowed"] is not False for r in rows):
        raise ValueError("cannot upgrade coverage claims")
    for row in rows:
        for reference in row["references"]:
            if reference["source_role"] == "ohlcv_manifest" and reference["source_file_sha256"] != identities["manifest"]["sha256"]:
                raise ValueError("manifest identity mismatch")
    meta = materials["manifest"]["metadata"]
    if meta["timeframe_ms"] != api["BAR_MS"] or meta["columns"] != ["timestamp", "symbol", "open", "high", "low", "close", "volume"]:
        raise ValueError("legacy metadata basis changed")
    symbols = meta["symbols"]
    if type(symbols) is not list or len(symbols) != 8 or len(set(symbols)) != 8 or any(type(s) is not str for s in symbols):
        raise ValueError("legacy full8 reference universe required")
    outer = materials["old_schedule"]["outer"]
    if outer["deployment_horizon_days"] != 3 or outer["retrain_cadence_days"] != 3 or outer["overlap_allowed"] is not False:
        raise ValueError("old calendar changed")
    engineering = select_engineering_origin(meta["coverage_start"], meta["coverage_end_exclusive"],
                                            utc_ms(outer["start_utc"]), api)
    if engineering["nominal_deployment_end_exclusive_ms"] > utc_ms(outer["end_exclusive_utc"]):
        raise ValueError("engineering origin must remain in old calendar")
    source = materials["linear_source"]
    for anchor in ("median = np.median(x, axis=0)", "np.quantile(x, [0.25, 0.75]",
                   "q75 - q25 > 1e-12", "-8.0, 8.0", "[0.0, ridge_alpha, ridge_alpha]"):
        if anchor not in source:
            raise ValueError("scaler/ridge code anchor changed")
    feature = materials["feature_definition"]
    if feature["bar_ms"] != api["BAR_MS"] or feature["minimum_history_bars"] != 48:
        raise ValueError("feature definition changed")
    source_bindings = {role: {"path": SOURCES[role], **identities[role]} for role in SOURCES}
    models = proposal["models"]
    definitions = {
        "clock": {key: materials["contract"]["policies"][key] for key in
                  ("clock", "bar_ms", "deployment_horizon_ms", "retrain_cadence_ms", "purge_bars")},
        "model": {key: models[key] for key in ("family", "features", "coefficient_count", "hidden_layers", "ridge_alpha",
                                               "full_coefficients_and_scaler_refit_each_origin", "minimum_completed_opportunities",
                                               "minimum_weight_ess", "minimum_trades_forcing", "fallback", "ga_enabled")},
        "scaler": {"fit_scope": "all_selected_completed_training_opportunity_features_no_return_sign_filter",
                   "median_and_quantiles": "unweighted_per_feature_linear_interpolation_at_(n-1)*q",
                   "iqr_scale": "q75-q25_if_strictly_greater_than_1e-12_else_1",
                   "clip_absolute_value": 8, "refit_each_origin": True,
                   "scaler_fit_rows_sha_required": True, "source_code_sha256": identities["linear_source"]["sha256"]},
        "ridge": {"alpha": models["ridge_alpha"], "intercept_penalty": 0,
                  "feature_coefficient_penalty": models["ridge_alpha"],
                  "outcome_target": "planned_delayed_entry_open_to_next_rebalance_open_gross_return",
                  "target_is_not_net_profit": True},
        "features": {key: feature[key] for key in ("return_12", "efficiency_12", "flat_path", "minimum_history_bars", "history_start")},
        "feature_availability": "actual_closed_available_prefix_required_declared_asof_is_not_measurement",
        "opportunity_definition": feature["proposal"],
        "variants": list(models["variants"]), "recency": dict(models["recency"]),
        "recency_age": "training_asof_ms_minus_label_end_ms_in_days_not_return_sign_or_future_deploy",
        "training_identity": materials["contract"]["policies"]["training_identity"],
        "execution_label_policies": {key: materials["contract"]["policies"][key] for key in
                                      ("execution", "label", "label_availability", "label_cutoff", "fixed_delay_limitation")},
        "accounting_policies": dict(proposal["accounting"]),
        "primary_estimand": proposal["measurement"]["primary_estimand"],
        "economic_floor_bps_per_day": proposal["measurement"]["economic_floor_bps_per_day"],
        "user_allows_drawdowns": proposal["measurement"]["user_allows_drawdowns"],
        "no_loss_is_not_a_guarantee": True,
        "no_trade_reference": "cash_equity_constant_1_zero_positions_zero_fills_in_this_research_convention",
        "ema_reference": {"definition": feature["proposal"], "role": "fixed_descriptive_reference_not_additional_candidate_search",
                          "target": "long_if_current_available_opportunity_else_flat",
                          "no_refill_on_unchanged_target": True, "execution_and_costs_must_match_candidates": True,
                          "is_legacy_performance_equivalence_claim": False},
        "decision_policy": {"registered_executable_threshold_rule": None,
                            "reason": "legacy_composite_fee_thresholds_cannot_transfer_to_unregistered_cost_adapter",
                            "safe_administrative_fallback": "NoTrade_no_fit_no_orders_not_a_trading_decision"},
    }
    engineering.update({"candidate_source_manifest_path": SOURCES["manifest"],
                        "candidate_source_manifest_sha256": identities["manifest"]["sha256"],
                        "planned_reference_symbols": list(symbols),
                        "price_basis": "legacy_ohlcv_reference_venue_and_product_unverified_not_claimed_bitget_perpetual",
                        "planned_training_start_ms": meta["coverage_start"],
                        "planned_training_window": models["training_window"],
                        "planned_origin_count": 1, "planned_pipeline_count": 2,
                        "maximum_planned_engineering_pipeline_runs": 2,
                        "failed_runs_count_toward_budget": True, "outcome_class": "engineering_only_not_statistical_or_promotion_evidence"})
    unresolved = [{"field": row["field"], "value": None, "evidence_status": row["evidence_status"],
                   "required_action": row["required_action"]} for row in rows]
    for field, value in (
        ("decision.executable_threshold_rule", None),
        ("measurement.drawdown_budget", proposal["measurement"]["drawdown_budget"]),
        ("measurement.concentration_turnover_drift_gates", proposal["measurement"]["concentration_turnover_drift_gates"]),
        ("measurement.confidence_and_multiplicity_rule", proposal["measurement"]["confidence_and_multiplicity_rule"]),
        ("testing.historical_global_trial_count", proposal["testing_budget"]["historical_global_trial_count"]),
        ("testing.power_minimum_effect_and_block_design", proposal["testing_budget"]["power_minimum_effect_and_block_design"]),
        ("testing.actual_folds_resample_and_audit_banks", None),
        ("sealed.new_interval_start_utc", proposal["sealed"]["new_interval_start_utc"]),
        ("sealed.new_interval_end_exclusive_utc", proposal["sealed"]["new_interval_end_exclusive_utc"]),
    ):
        if value is not None:
            raise ValueError("baseline unknown was replaced")
        unresolved.append({"field": field, "value": None, "evidence_status": "unregistered_protocol_requirement",
                           "required_action": "register_before_corresponding_fit_inference_or_promotion_no_implicit_default"})
    body = {"schema": "exia.genetic_primus.forward_preregistration/1",
            "record_kind": "immutable_definitions_and_non_executable_historical_engineering_plan",
            "registered_on_date_utc": "2026-10-07",
            "status": "definitions_frozen_engineering_plan_proposed_launch_blocked",
            "registration_is_after_historical_engineering_period": True,
            "historical_plan_is_not_prospective_preregistration": True,
            "source_bindings": source_bindings, "definitions": definitions,
            "definitions_sha256": api["sha"](definitions), "engineering_plan": engineering,
            "engineering_plan_sha256": api["sha"](engineering), "unresolved": sorted(unresolved, key=lambda r: r["field"]),
            "evidence_classes": {"definitions": "normative_research_spec_not_actual_execution_proof",
                                 "planning": "declared_historical_engineering_assumptions_not_measured_availability",
                                 "source_metadata": "archived_claims_not_independent_raw_data_verification",
                                 "actual_fit_or_financial_results": "absent"},
            "testing_budget": {"candidate_pipelines": 2, "hyperparameter_search_trials": 0,
                               "authorized_market_runs": 0, "record_is_not_a_training_attempt": True,
                               "engineering_results_cannot_select_best_model_or_prove_edge": True,
                               "all_future_runs_and_failures_must_be_registered": True,
                               "no_post_outcome_changes_under_same_record_sha": True,
                               "no_additional_seeds_features_or_candidate_retries_under_this_record": True,
                               "common_search_resample_bank_required": True, "independent_audit_bank_required": True,
                               "historical_dsr_pbo_verified": False, "global_objective_registry_identity_verified": False},
            "release_bridge": {"previous_target_definition_only": materials["release"]["execution_target"],
                               "research_prices_are_not_bitget_fill_evidence": True,
                               "ohlcv_fill_fallback_for_release_allowed": False,
                               "funding_borrow_and_instrument_rules_unresolved_not_zero": True,
                               "paper_shadow_canary_authorized": False},
            "sealed": dict(proposal["sealed"]),
            "lineage_required": {"decision_model_sha256": True, "entry_model_sha256_for_carried_positions": True,
                                 "scaler_fit_rows_sha256": True, "cutoff_folds_resample_audit_sha256": True,
                                 "actual_fit_start_complete_model_available_timestamps": True},
            "next_action": "explicit_authorization_for_bounded_historical_offline_input_verification_not_more_mock_stages",
            "gate": {"fit_allowed": False, "market_data_read_authorized": False,
                     "financial_evaluation_allowed": False, "market_research_launch_ready": False, "real_money_ready": False,
                     "reason": "planning_freeze_is_not_data_cost_timing_or_execution_attestation"},
            "safety": dict(materials["contract"]["safety"])}
    return {"record": body, "record_semantic_sha256": api["sha"](body)}


def validate(record, materials, identities, api):
    api["keys"](record, ("record", "record_semantic_sha256"))
    api["digest"](record["record_semantic_sha256"])
    if api["sha"](record["record"]) != record["record_semantic_sha256"]:
        raise ValueError("record self hash mismatch")
    expected = build(materials, identities, api)
    if api["canonical"](record) != api["canonical"](expected):
        raise ValueError("record differs from its bound definitions and sources")
    return {"record_consistent": True, "definitions_frozen": True, "fit_allowed": False,
            "market_data_read_authorized": False, "real_money_ready": False}
