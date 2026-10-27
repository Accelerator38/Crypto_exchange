"""Pure timing scenarios for complete OHLCV rows; never observed-arrival proof."""
from __future__ import annotations

BAR_MS = 14_400_000
HORIZON_BARS = 18
PURGE_BARS = 7
MAX_INTEGER = 2**63 - 1
SAFETY = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper",
          "live", "promotion_authority", "training_allowed",
          "financial_evaluation_allowed", "network_allowed")
POLICIES = {
    "row_available_at": "bar_start + bar_ms + explicit_ingestion_lag_ms",
    "feature_available_at": "max_availability_of_all_contiguous_dependencies",
    "feature_cutoff": "feature_available_at <= decision_asof_ms",
    "entry_open": "first_grid_open_strictly_after_decision_plus_execution_delay",
    "exit_open": "entry_open + 18 * bar_ms",
    "label_available_at": "max_complete_row_availability_of_entry_and_exit",
    "training_cutoff": "label_available_at <= training_asof_ms - 7 * bar_ms",
    "coverage": "all_dependency_rows_inside_half_open_snapshot_interval",
    "revision": "immutable_snapshot_no_assumption_of_original_arrival",
}


def integer(value):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        raise ValueError("bounded nonnegative integer required")
    return value


def aligned(value):
    integer(value)
    if value % BAR_MS:
        raise ValueError("4h alignment required")
    return value


def coverage(start_ms, end_exclusive_ms):
    aligned(start_ms)
    aligned(end_exclusive_ms)
    if end_exclusive_ms <= start_ms:
        raise ValueError("nonempty half-open coverage required")


def row_available_at(bar_start_ms, ingestion_lag_ms):
    """Scenario, not measurement. Even open uses the complete-row boundary."""
    aligned(bar_start_ms)
    integer(ingestion_lag_ms)  # Deliberately no default and no null substitution.
    return integer(bar_start_ms + BAR_MS + ingestion_lag_ms)


def closed_prefix_plan(asof_ms, ingestion_lag_ms, coverage_start_ms, coverage_end_ms):
    """Calendar range only. Actual missing/duplicate rows still require input audit."""
    integer(asof_ms)
    integer(ingestion_lag_ms)
    coverage(coverage_start_ms, coverage_end_ms)
    last = min(((asof_ms - ingestion_lag_ms) // BAR_MS - 1) * BAR_MS,
               coverage_end_ms - BAR_MS)
    count = max(0, (last - coverage_start_ms) // BAR_MS + 1)
    return {"first_bar_start_ms": coverage_start_ms if count else None,
            "last_bar_start_ms": last if count else None, "bar_count": count,
            "observed_arrival_verified": False, "fit_allowed": False}


def feature_gate(dependencies, decision_asof_ms, ingestion_lag_ms,
                 coverage_start_ms, coverage_end_ms):
    """All input bars of this feature must be supplied, ordered and contiguous."""
    integer(decision_asof_ms)
    integer(ingestion_lag_ms)
    coverage(coverage_start_ms, coverage_end_ms)
    if type(dependencies) not in (tuple, list) or not 1 <= len(dependencies) <= 20000:
        raise ValueError("bounded explicit dependencies required")
    for start in dependencies:
        aligned(start)
    if any(b - a != BAR_MS for a, b in zip(dependencies, dependencies[1:])):
        raise ValueError("unordered, duplicate or missing dependency bars")
    available = row_available_at(dependencies[-1], ingestion_lag_ms)
    covered = coverage_start_ms <= dependencies[0] and dependencies[-1] < coverage_end_ms
    return {"feature_available_at_ms": available, "dependencies_in_coverage": covered,
            "available_by_decision": available <= decision_asof_ms,
            "scenario_eligible": covered and available <= decision_asof_ms,
            "observed_arrival_verified": False, "fit_allowed": False}


def label_gate(entry_open_ms, exit_open_ms, training_asof_ms, ingestion_lag_ms,
               coverage_start_ms, coverage_end_ms):
    """Label maturity of complete-row open observations, with seven-bar purge."""
    aligned(entry_open_ms)
    aligned(exit_open_ms)
    integer(training_asof_ms)
    integer(ingestion_lag_ms)
    coverage(coverage_start_ms, coverage_end_ms)
    if exit_open_ms - entry_open_ms != HORIZON_BARS * BAR_MS:
        raise ValueError("shifted entry/exit must preserve three days")
    available = max(row_available_at(entry_open_ms, ingestion_lag_ms),
                    row_available_at(exit_open_ms, ingestion_lag_ms))
    cutoff = training_asof_ms - PURGE_BARS * BAR_MS
    covered = coverage_start_ms <= entry_open_ms and exit_open_ms < coverage_end_ms
    return {"label_available_at_ms": available, "training_cutoff_ms": cutoff,
            "dependencies_in_coverage": covered,
            "completed_and_purged": available <= cutoff,
            "scenario_eligible": covered and available <= cutoff,
            "observed_arrival_verified": False, "fit_allowed": False}


def assess(contract):
    expected = {"schema", "status", "bar_ms", "label_horizon_bars", "purge_bars",
                "timestamp_semantics", "row_semantics", "policies", "availability",
                "historical_results_requalified", "safety"}
    if type(contract) is not dict or set(contract) != expected:
        raise ValueError("availability contract schema mismatch")
    if (contract["schema"] != "exia.genetic_primus.ohlcv_availability_contract/1"
        or contract["status"] != "draft_unregistered"
        or contract["timestamp_semantics"] != "bar_start"
        or contract["row_semantics"] != "complete_ohlcv_aggregate_not_independent_open_stream"
        or contract["policies"] != POLICIES
        or contract["historical_results_requalified"] is not False):
        raise ValueError("unapproved timing policy")
    for key, value in (("bar_ms", BAR_MS), ("label_horizon_bars", HORIZON_BARS),
                       ("purge_bars", PURGE_BARS)):
        if integer(contract[key]) != value:
            raise ValueError("timebase/horizon/purge mismatch")
    safety = contract["safety"]
    if type(safety) is not dict or set(safety) != set(SAFETY) or any(safety[k] is not False for k in SAFETY):
        raise ValueError("authority must remain false")
    availability = contract["availability"]
    if type(availability) is not dict or set(availability) != {
        "ingestion_lag_ms", "evidence_sha256", "late_revision_policy_sha256"}:
        raise ValueError("availability registration mismatch")
    missing = []
    for key, value in availability.items():
        if value is None:
            missing.append(key)
        elif key == "ingestion_lag_ms":
            integer(value)
        elif (type(value) is not str or len(value) != 64
              or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("invalid evidence identity")
    return {"missing_registration": sorted(missing), "status": "draft_unregistered",
            "observed_arrival_verified": False, "fit_allowed": False,
            "financial_evaluation_allowed": False, "promotion_authority": False}
