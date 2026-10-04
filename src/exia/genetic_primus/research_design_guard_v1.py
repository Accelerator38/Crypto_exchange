"""Pure checks of an unregistered research draft; never authorizes execution."""

from datetime import datetime
from math import isfinite
from typing import Any, Mapping


def assess_draft(
    draft: Mapping[str, Any], *, reserved_outer_end_utc: str
) -> dict[str, Any]:
    """Check design invariants without reading data, clocks, files or networks."""
    errors: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    def section(key: str) -> Mapping[str, Any]:
        value = draft.get(key)
        if not isinstance(value, Mapping):
            errors.append(f"missing_or_invalid_section:{key}")
            return {}
        return value

    def utc(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.endswith("Z"):
            return None
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            return None

    require(draft.get("status") == "DRAFT_NOT_REGISTERED", "must_remain_draft")
    require(type(draft.get("candidate_budget")) is int and draft["candidate_budget"] == 1,
            "one_candidate_only")
    require(draft.get("baseline") == "NoTrade", "NoTrade_reference_required")
    permissions = section("permissions")
    for key in ("training_allowed", "evaluation_allowed", "network_allowed",
                "credentials_allowed", "orders_enabled", "paper", "live",
                "promotion_authority"):
        require(permissions.get(key) is False, f"permission_must_be_false:{key}")
    model = section("model")
    require(model.get("learned_parameters") == 2, "two_learned_parameters_only")
    require(model.get("ga_allowed") is False, "GA_forbidden")
    require(model.get("weights") == "uniform_only", "no_weight_search")
    rolling = section("rolling")
    require(rolling.get("purge_bars") == 7, "purge_must_be_7_bars")
    require(rolling.get("deployment_horizon_days") == 3 and
            rolling.get("retrain_cadence_days") == 3, "three_day_horizon_and_cadence")
    evaluation = section("evaluation")
    require(evaluation.get("support_unit") == "distinct_signal_origins_not_assets_or_trades",
            "temporal_support_required")
    require(evaluation.get("paired_bootstrap") is True, "paired_bootstrap_required")
    require(evaluation.get("audit_seed_is_independent_market_data") is False,
            "audit_seed_is_not_new_market_data")
    costs = section("execution_assumptions")
    normal, stress = costs.get("normal"), costs.get("stress")
    if isinstance(normal, Mapping) and isinstance(stress, Mapping):
        n, s = normal.get("round_trip_cost_bps"), stress.get("round_trip_cost_bps")
        valid = all(type(x) in (int, float) and isfinite(x) and x > 0 for x in (n, s))
        require(valid and s >= n, "finite_positive_costs_with_stress_at_least_normal")
        require(normal.get("delay_bars") == 0 and stress.get("delay_bars") == 1,
                "registered_zero_and_one_bar_delays")
    else:
        errors.append("both_cost_scenarios_required")
    temporal = section("temporal_validation")
    reserved_end = utc(reserved_outer_end_utc)
    earliest = utc(temporal.get("outer_not_before_utc"))
    require(reserved_end is not None and earliest is not None and earliest >= reserved_end,
            "reserved_EMA_interval_must_not_be_reused")
    require(temporal.get("reserved_EMA_interval_forbidden") is True,
            "reserved_interval_exclusion_required")
    start_value, end_value = temporal.get("outer_start_utc"), temporal.get("outer_end_exclusive_utc")
    if start_value is not None or end_value is not None:
        start, end = utc(start_value), utc(end_value)
        require(start is not None and end is not None and earliest is not None and
                start >= earliest and end > start, "invalid_or_overlapping_outer_interval")
    require(temporal.get("no_interim_pnl_analysis") is True and
            temporal.get("single_final_analysis") is True, "single_final_analysis_required")
    provenance = section("provenance")
    require(provenance.get("contract_frozen") is False, "draft_not_frozen")
    return {
        "draft_consistent": not errors,
        "errors": errors,
        "execution_allowed": False,
        "blocking_requirements": [
            "power_and_minimum_economic_effect_design",
            "independent_outer_interval_and_support_rules",
            "data_folds_banks_cutoff_hashes_and_immutable_registration",
            "separate_authorization_for_any_training_or_market_data_access",
        ],
        "scope": "draft_consistency_only_not_registration_or_promotion_gate",
    }
