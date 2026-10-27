"""Price-free causal label scheduling; no labels, fits or execution authority."""
from __future__ import annotations

BAR_MS = 14_400_000
HORIZON_BARS = 18
PURGE_BARS = 7


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")


def opportunity(decision_asof_ms, training_asof_ms, ingestion_lag_ms,
                execution_delay_ms, coverage_start_ms, coverage_end_ms, kernel):
    """Strict next open after readiness; complete-row maturity plus purge."""
    for name, value in (("decision", decision_asof_ms), ("training", training_asof_ms),
                        ("lag", ingestion_lag_ms), ("delay", execution_delay_ms)):
        integer(value, name)
    if decision_asof_ms % BAR_MS:
        raise ValueError("decision must use the registered bar grid")
    ready = decision_asof_ms + execution_delay_ms
    entry = (ready // BAR_MS + 1) * BAR_MS
    exit_open = entry + HORIZON_BARS * BAR_MS
    gate = kernel["label_gate"](entry, exit_open, training_asof_ms, ingestion_lag_ms,
                                coverage_start_ms, coverage_end_ms)
    return {"decision_asof_ms": decision_asof_ms, "decision_ready_ms": ready,
            "entry_open_ms": entry, "exit_open_ms": exit_open,
            **gate, "execution_delay_ms": execution_delay_ms,
            "entry_is_reference_open_not_fill": True, "target_values_computed": False}


def build(events, clock, metadata, ingestion_lag_ms, execution_delay_ms, kernel):
    """Preserve existing opportunities; retain exclusions without reading prices."""
    origin, train = clock["origin_start_ms"], clock["training_asof_ms"]
    history = clock["history_start_ms"]
    for name, value in (("origin", origin), ("train", train), ("history", history),
                        ("lag", ingestion_lag_ms)):
        integer(value, name)
    if (origin % BAR_MS or train != origin - BAR_MS or history % BAR_MS
            or clock["bar_ms"] != BAR_MS or clock["horizon_bars"] != HORIZON_BARS
            or clock["retrain_bars"] != HORIZON_BARS or clock["purge_bars"] != PURGE_BARS
            or not history < train):
        raise ValueError("frozen clock mismatch")
    if metadata["timeframe_ms"] != BAR_MS or len(metadata["symbols"]) != 8:
        raise ValueError("market panel mismatch")
    if (not isinstance(events, list) or not events or len(events) > 20_000
            or [e["asof_ms"] for e in events] != sorted({e["asof_ms"] for e in events})):
        raise ValueError("event order/uniqueness/cap")
    roles = {"candidate_features_not_label_gated", "training_prefix_context_not_scaler_fit",
             "deployment_features_no_labels"}
    for event in events:
        if set(event) != {"asof_ms", "kind"} or event["kind"] not in roles:
            raise ValueError("event schema")
        integer(event["asof_ms"], "event")
        if event["asof_ms"] % BAR_MS:
            raise ValueError("event grid")
    candidates = [e["asof_ms"] for e in events if e["kind"] == "candidate_features_not_label_gated"]
    if (not candidates or any(not history <= t < train or (origin - t) % (18 * BAR_MS)
                              for t in candidates)
            or any(b - a != 18 * BAR_MS for a, b in zip(candidates, candidates[1:]))):
        raise ValueError("candidate calendar changed")
    contexts = [e for e in events if e["kind"] != "candidate_features_not_label_gated"]
    if contexts != [{"asof_ms": train, "kind": "training_prefix_context_not_scaler_fit"},
                    {"asof_ms": origin, "kind": "deployment_features_no_labels"}]:
        raise ValueError("context roles changed")
    common = {"schema": "exia.genetic_primus.causal_market_label_schedule/1",
              "candidate_events": len(candidates), "context_events_excluded": len(contexts),
              "ingestion_lag_ms": ingestion_lag_ms, "execution_delay_ms": execution_delay_ms,
              "labels_materialized": False, "fit_allowed": False,
              "net_training_objective_registered": False, "original_arrival_verified": False}
    if execution_delay_ms is None:
        return {**common, "scenario_selected": False, "label_value_read_allowed": False,
                "eligible": [], "excluded": []}
    integer(execution_delay_ms, "delay")
    if execution_delay_ms > BAR_MS:
        raise ValueError("engineering delay cap is one bar")
    rows = [opportunity(t, train, ingestion_lag_ms, execution_delay_ms,
                        metadata["coverage_start"], metadata["coverage_end_exclusive"], kernel)
            for t in candidates]
    eligible = [r for r in rows if r["scenario_eligible"]]
    excluded = [{**r, "reason": "outside_coverage" if not r["dependencies_in_coverage"]
                 else "label_not_completed_and_purged"} for r in rows if not r["scenario_eligible"]]
    return {**common, "scenario_selected": True,
            "label_value_read_allowed": False,  # a schedule is never read authorization
            "training_cutoff_ms": train - PURGE_BARS * BAR_MS,
            "eligible": eligible, "excluded": excluded}
