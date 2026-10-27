"""Pure job planning and feature-state checkpointing; no decoder or fitter."""
from __future__ import annotations

import copy
import math


def plan(protocol, metadata, audited_files, lag_bars, adapter):
    if lag_bars is None:
        return {"scenario_selected": False, "materialization_allowed": False,
                "jobs": [], "events": [], "fit_allowed": False}
    if type(lag_bars) is not int or lag_bars not in (0, 1):
        raise ValueError("explicit zero/one-bar scenario required")
    b = adapter["BAR_MS"]
    clock = protocol["clock"]
    lag = lag_bars * b
    cutoff = clock["origin_start_ms"] - b - lag
    ordered = sorted(audited_files, key=lambda f: f["first_timestamp_ms"])
    if len(ordered) != 11 or len({f["source"]["path"] for f in ordered}) != 11:
        raise ValueError("exact approved source bank required")
    next_start = metadata["coverage_start"]
    jobs = []
    for file in ordered:
        first, last = file["first_timestamp_ms"], file["last_timestamp_ms"]
        adapter["aligned"](first)
        adapter["aligned"](last)
        if first != next_start or last < first:
            raise ValueError("overlap/gap in historical source calendar")
        next_start = last + b
        if first <= cutoff:
            jobs.append({**file, "last_included_timestamp_ms": min(last, cutoff),
                         "expected_included_rows": ((min(last, cutoff) - first) // b + 1) * 8})
    if next_start != metadata["coverage_end_exclusive"] or not jobs:
        raise ValueError("source coverage mismatch")
    candidates = adapter["candidate_calendar"](clock["origin_start_ms"], clock["training_asof_ms"],
                                               clock["history_start_ms"], lag)
    events = [{"asof_ms": t, "kind": "candidate_features_not_label_gated"} for t in candidates]
    events += [{"asof_ms": clock["training_asof_ms"], "kind": "training_prefix_context_not_scaler_fit"},
               {"asof_ms": clock["origin_start_ms"], "kind": "deployment_features_no_labels"}]
    events.sort(key=lambda event: event["asof_ms"])
    if len({e["asof_ms"] for e in events}) != len(events):
        raise ValueError("ambiguous event roles")
    revision = {"policy": "final_pinned_snapshot_only_no_original_revision_or_arrival_claim",
                "source_manifest_sha256": protocol["source_manifest_byte_sha256"]}
    return {"scenario_selected": True, "materialization_allowed": True,
            "lag_bars": lag_bars, "ingestion_lag_ms": lag, "cutoff_bar_start_ms": cutoff,
            "decision_asof_ms": clock["origin_start_ms"], "jobs": jobs, "events": events,
            "revision_policy": revision, "revision_policy_sha256": adapter["sha"](revision),
            "original_arrival_verified": False, "fit_allowed": False,
            "label_execution_delay_registered": False, "labels_materialized": False}


def checkpoint(state, event_cursor):
    return {"schema": "exia.genetic_primus.market_feature_checkpoint/1", "history": state.history,
            "end": state.end, "lag": state.lag, "next_bar": state.next_bar, "count": state.count,
            "prefix_sha": state.prefix_sha, "states": copy.deepcopy(state.states), "event_cursor": event_cursor}


def restore(state, saved, event_count, adapter):
    if set(saved) != {"schema", "history", "end", "lag", "next_bar", "count", "prefix_sha", "states", "event_cursor"}:
        raise ValueError("checkpoint schema")
    if (saved["schema"] != "exia.genetic_primus.market_feature_checkpoint/1"
        or (saved["history"], saved["end"], saved["lag"]) != (state.history, state.end, state.lag)):
        raise ValueError("checkpoint context substitution")
    for key in ("count", "next_bar", "event_cursor"):
        adapter["integer"](saved[key])
    count = saved["count"]
    b = adapter["BAR_MS"]
    if count > 20000 or saved["next_bar"] != state.history + count * b or saved["next_bar"] > state.end:
        raise ValueError("checkpoint clock/count")
    if not 0 <= saved["event_cursor"] <= event_count:
        raise ValueError("checkpoint event cursor")
    adapter["digest"](saved["prefix_sha"])
    if count == 0:
        if saved["states"] or saved["prefix_sha"] != state.prefix_sha:
            raise ValueError("initial checkpoint substitution")
    else:
        if type(saved["states"]) is not dict or set(saved["states"]) != set(adapter["SYMBOLS"]):
            raise ValueError("checkpoint universe")
        for symbol, local in saved["states"].items():
            if set(local) != {"ema12", "ema48", "ema12_tail", "rows"}:
                raise ValueError("checkpoint feature state")
            if len(local["rows"]) != min(13, count) or len(local["ema12_tail"]) != min(4, count):
                raise ValueError("checkpoint tail count")
            for value in (local["ema12"], local["ema48"], *local["ema12_tail"]):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("checkpoint EMA")
            for i, row in enumerate(local["rows"]):
                if set(row) != adapter["ROW_KEYS"] or row["symbol"] != symbol or row["timestamp"] != saved["next_bar"] - len(local["rows"]) * b + i * b:
                    raise ValueError("checkpoint tail lineage")
    state.next_bar, state.count, state.prefix_sha = saved["next_bar"], count, saved["prefix_sha"]
    state.states = copy.deepcopy(saved["states"])
    return saved["event_cursor"]


def consume(bars, state, scenario, cursor, adapter):
    """Atomic file in caller memory; events are emitted at their own historical asof."""
    output = []
    events = scenario["events"]
    b = adapter["BAR_MS"]
    def emit_until(asof):
        nonlocal cursor
        while cursor < len(events) and events[cursor]["asof_ms"] <= asof:
            event = events[cursor]
            snapshot = state.snapshot(event["asof_ms"])
            for row in snapshot["rows"]:
                output.append({**row, "kind": event["kind"], "prefix_sha256": snapshot["prefix_sha256"],
                               "label_fields_present": False, "fit_allowed": False})
            cursor += 1
    included = 0
    for bar in bars:
        if bar[0]["timestamp"] > scenario["cutoff_bar_start_ms"]:
            break
        ready = bar[0]["timestamp"] + b + scenario["ingestion_lag_ms"]
        emit_until(ready - 1)
        if not state.feed_bar(bar, scenario["decision_asof_ms"]):
            raise ValueError("eligible bar was unavailable")
        included += len(bar)
        emit_until(ready)
    return output, cursor, included
