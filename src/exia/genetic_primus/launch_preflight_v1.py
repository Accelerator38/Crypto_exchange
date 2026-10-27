"""Metadata-only readiness classification, never launch or registration authority."""
from __future__ import annotations

PROMOTION_FIELDS = {
    "measurement.concentration_turnover_drift_gates",
    "measurement.confidence_and_multiplicity_rule", "measurement.drawdown_budget",
    "sealed.new_interval_end_exclusive_utc", "sealed.new_interval_start_utc",
    "testing.actual_folds_resample_and_audit_banks", "testing.historical_global_trial_count",
    "testing.power_minimum_effect_and_block_design",
}
BRIDGES = {
    "market_feature_label_matrix": "SYN_only_builders_do_not_attest_full8_market_matrix_under_new_semantics",
    "reference_price_to_cost_adapter": "OHLCV_open_not_attested_as_mid_or_fill_synthetic_adapter_rejects_it",
    "continuous_portfolio_ledger": "single_trade_arithmetic_not_continuous_cash_quantity_marks_or_terminal_policy",
    "causal_fitter_and_model_lineage": "input_manifests_not_fitted_scaler_model_or_per_decision_model_SHA",
}


def classify_field(field):
    if field in PROMOTION_FIELDS:
        return "prospective_or_promotion_not_smoke_prerequisite"
    if field == "decision.executable_threshold_rule" or field.startswith(("costs.", "data.", "timing.")):
        return "historical_smoke_protocol_requirement"
    raise ValueError("unclassified requirement")


def assess(research, record, availability, quality, origin, matrix, net):
    """Report evidence classes; filling a metadata value cannot authorize a fit."""
    if research["status"] != "draft_unregistered" or availability["status"] != "draft_unregistered":
        raise ValueError("unexpected frozen contract state")
    for item in (research, record, availability, quality, origin, matrix, net):
        if any(value is not False for value in item["safety"].values()) or len(item["safety"]) != 9:
            raise ValueError("authority drift")
    for item in (origin, matrix, net):
        if item["market_research_launch_ready"] is not False or item["real_money_ready"] is not False:
            raise ValueError("unsupported readiness claim")
    if (quality["files"] != 11 or quality["summary"]["rows"] != 83088
        or quality["summary"]["invalid_rows"] != 0 or quality["strict_quality_passed"] is not True):
        raise ValueError("historical quality evidence changed")
    if (matrix["baseline_manifest"]["assessment"]["net_objective_registered"] is not False
        or net["assessment"]["market_net_objective_registered"] is not False
        or net["assessment"]["costs_calibrated"] is not False):
        raise ValueError("synthetic evidence promoted to market evidence")
    core = {f"{section}.{field}": value for section in ("timing", "data", "costs")
            for field, value in research[section].items()}
    unresolved = record["unresolved"]
    if len(core) != 21 or len(unresolved) != 30:
        raise ValueError("registration inventory changed")
    fields = [row["field"] for row in unresolved]
    if len(set(fields)) != 30 or not set(core) <= set(fields) or not PROMOTION_FIELDS <= set(fields):
        raise ValueError("requirement bank mismatch")
    requirements = []
    for row in unresolved:
        field = row["field"]
        state = "unregistered" if field not in core or core[field] is None else "provided_not_independently_verified"
        requirements.append({"field": field, "scope": classify_field(field),
                             "registration_state": state, "original_evidence_status": row["evidence_status"]})
    budget = record["testing_budget"]
    if budget["authorized_market_runs"] != 0 or budget["candidate_pipelines"] != 2 or budget["hyperparameter_search_trials"] != 0:
        raise ValueError("frozen research budget changed")
    plan, definitions = record["engineering_plan"], record["definitions"]
    if (definitions["model"]["features"] != ["return_12", "efficiency_12"]
        or definitions["model"]["coefficient_count"] != 3 or definitions["model"]["ga_enabled"] is not False
        or definitions["variants"] != ["uniform", "half_life_90d_anchor_20pct"]
        or definitions["economic_floor_bps_per_day"] != 0.0):
        raise ValueError("candidate/objective drift")
    registration = [{"field": f"availability.{k}", "registration_state": "unregistered" if v is None else "provided_not_verified"}
                    for k, v in availability["availability"].items()]
    if len(registration) != 3:
        raise ValueError("availability registration inventory")
    return {
        "schema": "exia.genetic_primus.launch_preflight/1", "inspection_only": True,
        "historical_data_field_quality": {"approved_files": 11, "rows": 83088, "strict_quality_passed": True,
            "source_authenticity_verified": False, "original_arrival_verified": False},
        "requirements": requirements, "availability_requirements": registration,
        "original_requirement_count": 30, "smoke_protocol_requirement_count": 22,
        "prospective_promotion_requirement_count": 8,
        "audited_pipeline_bridges": [{"component": k, "state": "not_demonstrated_in_current_audited_pipeline",
                                       "reason": reason} for k, reason in BRIDGES.items()],
        "unit_evidence": {"origin_prefix_assertions": origin["direct_assertions"],
                          "label_matrix_assertions": matrix["direct_assertions"],
                          "accounting_assertions": net["direct_assertions"],
                          "scope": "synthetic_engineering_not_market_profitability"},
        "frozen_candidates": {"features": definitions["model"]["features"],
                              "coefficient_count": 3, "variants": definitions["variants"], "ga_enabled": False},
        "proposed_historical_scope": {"origin_start_utc": plan["origin_start_utc"],
            "nominal_end_exclusive_utc": plan["nominal_deployment_end_exclusive_utc"],
            "latest_required_exit_open_utc": plan["latest_required_exit_open_utc"],
            "maximum_planned_pipeline_runs_including_failures": plan["maximum_planned_engineering_pipeline_runs"],
            "authorized_market_runs_under_current_record": 0,
            "prospective_evidence": False, "global_prior_trial_count_verified": False},
        "go_no_go": {"metadata_preflight_complete": True, "historical_fit_allowed": False,
                      "market_financial_evaluation_allowed": False, "new_sealed_reads_allowed": False,
                      "paper_or_live_allowed": False, "promotion_allowed": False,
                      "reason": "registration_and_market_integration_not_demonstrated_no_fit_authority"},
        "scope_notes": ["21_core_fields_remain_unregistered_no_backfill_from_synthetic_receipts",
            "8_prospective_promotion_items_are_not_prerequisites_for_non_evidential_engineering_smoke",
            "declarations_can_bound_engineering_sensitivity_but_do_not_become_measured_evidence",
            "unit_evidence_does_not_prove_repository_wide_absence_of_other_implementations",
            "no_reported_model_rankings_or_edge_from_this_audit"],
        "next_work": ["declare_separate_immutable_engineering_protocol_without_editing_old_record",
            "adapt_existing_offline_market_path_to_closed_prefix_maturity_and_per_origin_lineage",
            "bind_reference_price_semantics_and_costs_to_continuous_portfolio_terminal_accounting",
            "only_after_protocol_registration_and_explicit_fit_authority_one_bounded_historical_smoke",
            "do_not_add_seeds_models_search_or_claim_prospective_edge"],
        "safety": research["safety"],
    }
