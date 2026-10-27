"""Synthetic origin binding and scaler input manifest; never fits a scaler/model."""
from __future__ import annotations

import hashlib
import json
import math
import re

REQUEST_KEYS = {"schema", "origin_start_ms", "training_asof_ms", "decision_asof_ms",
                "history_start_ms", "coverage_start_ms", "coverage_end_ms",
                "ingestion_lag_ms", "execution_delay_ms", "universe", "revision_id",
                "candidate_decisions_ms", "expected_snapshot_sha256", "prefix_pins"}
CANDLE_KEYS = {"symbol", "revision_id", "bar_open_ms", "bar_end_exclusive_ms", "available_at_ms",
               "open", "high", "low", "close", "volume"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("digest required")


def origin_context(request, timing, execution):
    if type(request) is not dict or set(request) != REQUEST_KEYS:
        raise ValueError("origin context schema")
    if request["schema"] != "exia.genetic_primus.synthetic_origin_context/1":
        raise ValueError("synthetic scope only")
    b = timing["BAR_MS"]
    for name in ("origin_start_ms", "history_start_ms", "coverage_start_ms", "coverage_end_ms"):
        timing["aligned"](request[name])
    for name in ("training_asof_ms", "decision_asof_ms", "ingestion_lag_ms", "execution_delay_ms"):
        timing["integer"](request[name])
    timing["coverage"](request["coverage_start_ms"], request["coverage_end_ms"])
    origin, train, decision = (request[k] for k in ("origin_start_ms", "training_asof_ms", "decision_asof_ms"))
    if train != origin - b or not origin <= decision < origin + b:
        raise ValueError("pinned train=origin-1bar and first-bar decision required")
    if not request["coverage_start_ms"] <= request["history_start_ms"] < origin:
        raise ValueError("fixed history outside coverage")
    universe = request["universe"]
    if (type(universe) is not list or not 1 <= len(universe) <= 4
        or any(type(s) is not str or re.fullmatch(r"SYN_[A-Z]{1,8}", s) is None for s in universe)
        or universe != sorted(set(universe))):
        raise ValueError("sorted synthetic universe required")
    if (type(request["revision_id"]) is not str
        or re.fullmatch(r"synthetic_revision_[0-9]+", request["revision_id"]) is None):
        raise ValueError("fixed revision required")
    candidates = request["candidate_decisions_ms"]
    if type(candidates) is not list or not 1 <= len(candidates) <= 128:
        raise ValueError("explicit bounded training candidates required")
    for candidate in candidates:
        timing["aligned"](candidate)
    if candidates != sorted(set(candidates)) or any(not request["history_start_ms"] <= c <= train for c in candidates):
        raise ValueError("candidate calendar outside training context")
    digest(request["expected_snapshot_sha256"])
    if type(request["prefix_pins"]) is not dict or len(request["prefix_pins"]) > 129:
        raise ValueError("bounded prefix bank required")
    for key, value in request["prefix_pins"].items():
        if type(key) is not str or not key.isascii() or not key.isdigit():
            raise ValueError("prefix key must be timestamp")
        digest(value)
    window = execution["execution_window"](origin, decision, request["execution_delay_ms"])
    if not (request["coverage_start_ms"] <= window["planned_entry_open_ms"]
            and window["planned_exit_open_ms"] < request["coverage_end_ms"]):
        raise ValueError("deployment entry/exit outside snapshot coverage")
    return {"origin_start_ms": origin, "training_asof_ms": train,
            "training_cutoff_ms": train - timing["PURGE_BARS"] * b,
            "decision_asof_ms": decision, "deployment_window": window,
            "scaler_fit_allowed": False, "model_fit_allowed": False,
            "observed_arrival_verified": False}


def index_candles(candles, request, timing):
    if type(candles) is not list or not 1 <= len(candles) <= 1024:
        raise ValueError("synthetic candle budget")
    if sha(candles) != request["expected_snapshot_sha256"]:
        raise ValueError("snapshot substitution")
    by_key = {}
    for row in candles:
        if type(row) is not dict or set(row) != CANDLE_KEYS:
            raise ValueError("candle envelope schema")
        start = timing["aligned"](row["bar_open_ms"])
        timing["integer"](row["available_at_ms"])
        timing["integer"](row["bar_end_exclusive_ms"])
        if row["bar_end_exclusive_ms"] != start + timing["BAR_MS"]:
            raise ValueError("bar boundary substitution")
        if row["symbol"] not in request["universe"] or row["revision_id"] != request["revision_id"]:
            raise ValueError("universe/revision substitution")
        if not request["coverage_start_ms"] <= start < request["coverage_end_ms"]:
            raise ValueError("candle outside snapshot coverage")
        if row["available_at_ms"] < timing["row_available_at"](start, request["ingestion_lag_ms"]):
            raise ValueError("arrival claim precedes complete row scenario")
        key = (start, row["symbol"])
        if key in by_key:
            raise ValueError("duplicate candle")
        by_key[key] = row
    return by_key


def visible_prefix(index, asof, request, timing):
    plan = timing["closed_prefix_plan"](asof, request["ingestion_lag_ms"],
                                         request["history_start_ms"], request["coverage_end_ms"])
    if not 48 <= plan["bar_count"] <= 256:
        raise ValueError("48..256 closed bars required")
    b = timing["BAR_MS"]
    rows = []
    for start in range(request["history_start_ms"], plan["last_bar_start_ms"] + b, b):
        for symbol in request["universe"]:
            row = index.get((start, symbol))
            # Availability gate precedes OHLCV validation/use. Late old rows are not backfilled.
            if row is None or row["available_at_ms"] > asof:
                raise ValueError("missing or unpublished dependency")
            for field in ("open", "high", "low", "close", "volume"):
                value = row[field]
                if type(value) not in (int, float):
                    raise ValueError("numeric visible OHLCV required")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite or (value < 0 if field == "volume" else value <= 0):
                    raise ValueError("invalid visible OHLCV")
            if row["high"] < max(row["open"], row["low"], row["close"]) or row["low"] > min(row["open"], row["high"], row["close"]):
                raise ValueError("visible OHLC inconsistent")
            rows.append(row)
    pin = request["prefix_pins"].get(str(asof))
    if pin is None or sha(rows) != pin:
        raise ValueError("visible prefix substitution")
    return rows


def feature_records(prefix, asof, request, timing, kernel):
    result = []
    for symbol in request["universe"]:
        rows = [row for row in prefix if row["symbol"] == symbol]
        starts = [row["bar_open_ms"] for row in rows]
        gate = timing["feature_gate"](starts, asof, request["ingestion_lag_ms"],
                                      request["coverage_start_ms"], request["coverage_end_ms"])
        if not gate["scenario_eligible"]:
            raise ValueError("feature dependency unavailable")
        close = [float(row["close"]) for row in rows]
        path = math.fsum(abs(b - a) for a, b in zip(close[-13:-1], close[-12:]))
        gross = close[-1] / close[-13] - 1.0
        efficiency = 0.0 if path == 0 else abs(close[-1] - close[-13]) / path
        e12, e48 = kernel["ema"](close, 12), kernel["ema"](close, 48)
        proposal = path != 0 and e12[-1] / e48[-1] - 1 > 0 and e12[-1] / e12[-4] - 1 > 0 and gross > 0 and efficiency >= 0.30
        values = {"return_12": gross, "efficiency_12": efficiency}
        for value in values.values():
            if not math.isfinite(value):
                raise ValueError("nonfinite synthetic feature")
        result.append({"symbol": symbol, "decision_asof_ms": asof,
            "feature_values_sha256": sha(values), "proposal_long": bool(proposal),
            "feature_available_at_ms": max(row["available_at_ms"] for row in rows),
            "value_dependencies": {"first_bar_ms": starts[-13], "last_bar_ms": starts[-1],
                                   "bars": 13, "rows_sha256": sha(rows[-13:])},
            "proposal_dependencies": {"first_bar_ms": starts[0], "last_bar_ms": starts[-1],
                                      "bars": len(rows), "rows_sha256": sha(rows)},
            "compute_availability_measured": False})
    return result


def prepare_origin(candles, request, timing, execution, kernel):
    context = origin_context(request, timing, execution)
    index = index_candles(candles, request, timing)
    train_rows, exclusions, used_pins = [], [], {str(request["decision_asof_ms"])}
    for candidate in request["candidate_decisions_ms"]:
        window = execution["execution_window"](candidate, candidate, request["execution_delay_ms"])
        entry, exit_open = window["planned_entry_open_ms"], window["planned_exit_open_ms"]
        gate = timing["label_gate"](entry, exit_open, request["training_asof_ms"], request["ingestion_lag_ms"],
                                    request["coverage_start_ms"], request["coverage_end_ms"])
        if not gate["scenario_eligible"]:
            exclusions.append({"decision_asof_ms": candidate, "reason": "label_unavailable_unpurged_or_outside_coverage"})
            continue
        label_rows = [index.get((start, symbol)) for start in (entry, exit_open) for symbol in request["universe"]]
        if any(row is None for row in label_rows):
            raise ValueError("missing label dependency")
        actual_label_available = max(row["available_at_ms"] for row in label_rows)
        if actual_label_available > context["training_cutoff_ms"]:
            exclusions.append({"decision_asof_ms": candidate, "reason": "label_rows_published_after_cutoff"})
            continue
        prefix = visible_prefix(index, candidate, request, timing)
        records = feature_records(prefix, candidate, request, timing, kernel)
        used_pins.add(str(candidate))
        for record in records:
            train_rows.append({**record, "entry_open_ms": entry, "exit_open_ms": exit_open,
                               "label_available_at_ms": actual_label_available})
    if set(request["prefix_pins"]) != used_pins:
        raise ValueError("unused or missing prefix bindings")
    decision_prefix = visible_prefix(index, request["decision_asof_ms"], request, timing)
    decisions = feature_records(decision_prefix, request["decision_asof_ms"], request, timing, kernel)
    row_ids = [(r["decision_asof_ms"], r["symbol"]) for r in train_rows]
    return {"schema": "exia.genetic_primus.synthetic_origin_prefix_plan/1", "context": context,
            "source_snapshot_sha256": request["expected_snapshot_sha256"],
            "decision_features": decisions, "scaler_train_rows": train_rows,
            "scaler_plan": {"row_count": len(train_rows), "row_ids_sha256": sha(row_ids),
                            "feature_rows_sha256": sha(train_rows), "training_asof_ms": request["training_asof_ms"],
                            "statistics_computed": False, "fit_allowed": False},
            "excluded_candidates": exclusions, "model_fit_count": 0,
            "financial_evaluation_count": 0, "observed_arrival_verified": False}
