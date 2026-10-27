"""Synthetic completed-label/as-of feature integration, without fitting.

APIs are caller-supplied trusted pure helpers bound by the publisher receipt.
Gross labels are engineering open-to-last-close references, not execution PnL.
"""
from __future__ import annotations

import hashlib
import json

DT = 14_400_000
HORIZON = 18 * DT
EVIDENCE = "synthetic_origin_input_engineering_only"
REQUEST_KEYS = ("schema", "evidence_class", "context", "history_start_ms", "origin_starts_ms",
                "revision_id", "historical_feature_offset_ms", "kernel_definition_sha256",
                "label_definition_sha256", "expected_training_source_sha256", "expected_deploy_source_sha256")
LABEL_DEFINITION = {
    "schema": "exia.genetic_primus.synthetic_completed_label_definition/1",
    "horizon_ms": HORIZON,
    "gross_label": "open_at_origin_start_to_close_of_last_origin_bar",
    "available_at": "max_availability_of_all_18_path_candles_per_symbol",
    "feature_asof": "origin_start_minus_one_bar_plus_pinned_synthetic_offset",
    "selection": "explicit_contiguous_completed_origins;no_filter_by_return_sign",
    "costs_fills_funding_borrow": "not_modelled_not_execution_labels",
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_request(request, features, guard):
    guard["keys"](request, REQUEST_KEYS, "synthetic origin request")
    if (request["schema"] != "exia.genetic_primus.synthetic_causal_origin_request/1"
            or request["evidence_class"] != EVIDENCE):
        raise ValueError("only synthetic origin integration supported")
    ctx = request["context"]
    guard["context"](ctx)
    guard["timestamp"](request["history_start_ms"])
    if request["history_start_ms"] % DT:
        raise ValueError("fixed aligned history seed required")
    offset = request["historical_feature_offset_ms"]
    if type(offset) is not int or not 0 <= offset < DT:
        raise ValueError("typed pinned synthetic feature offset")
    if ctx["decision_asof_ms"] != ctx["deploy_start_ms"] - DT + offset:
        raise ValueError("train/deploy feature deadline mismatch")
    if request["revision_id"] != "synthetic_revision_1":
        raise ValueError("pinned synthetic revision required")
    for name in ("kernel_definition_sha256", "label_definition_sha256", "expected_training_source_sha256", "expected_deploy_source_sha256"):
        guard["digest"](request[name])
    if request["kernel_definition_sha256"] != features["sha"](features["DEFINITION"]):
        raise ValueError("one feature kernel required for train and deploy")
    if request["label_definition_sha256"] != sha(LABEL_DEFINITION):
        raise ValueError("label definition changed")
    starts = request["origin_starts_ms"]
    guard["exact_list"](starts)
    if not 1 <= len(starts) <= 8:
        raise ValueError("completed origin budget")
    for start in starts:
        guard["timestamp"](start)
        if start % DT:
            raise ValueError("aligned label origin required")
    if (starts != sorted(starts) or len(set(starts)) != len(starts)
            or any(b - a != HORIZON for a, b in zip(starts, starts[1:]))):
        raise ValueError("explicit origins must be contiguous and unique")
    cutoff = ctx["training_asof_ms"] - ctx["purge_bars"] * DT
    if cutoff < 0 or starts[-1] + HORIZON > cutoff:
        raise ValueError("future label requested beyond training-asof purge")
    if starts[0] - DT - request["history_start_ms"] < features["MIN_HISTORY_BARS"] * DT:
        raise ValueError("insufficient fixed feature history")
    return cutoff


def training_window(candles, request, cutoff, features, guard):
    """Pin only the required training source, never the future deployment tail."""
    guard["exact_list"](candles)
    if not 1 <= len(candles) <= features["MAX_INPUT_ROWS"]:
        raise ValueError("source candle budget")
    begin = request["history_start_ms"]
    end = request["origin_starts_ms"][-1] + HORIZON
    universe = request["context"]["universe"]
    seen = {}
    for row in candles:
        guard["keys"](row, features["CANDLE_KEYS"], "synthetic source candle")
        for name in ("bar_open_ms", "bar_end_exclusive_ms", "available_at_ms"):
            guard["timestamp"](row[name])
        if (type(row["symbol"]) is not str or type(row["revision_id"]) is not str
                or row["bar_open_ms"] % DT or row["bar_end_exclusive_ms"] != row["bar_open_ms"] + DT):
            raise ValueError("invalid source candle envelope")
        if (row["bar_open_ms"] < begin or row["bar_end_exclusive_ms"] > end
                or row["available_at_ms"] > cutoff):
            continue
        if row["available_at_ms"] < row["bar_end_exclusive_ms"]:
            raise ValueError("source candle available before close")
        if row["revision_id"] != request["revision_id"] or row["symbol"] not in universe:
            raise ValueError("visible training revision/universe changed")
        for name in ("open", "high", "low", "close", "volume"):
            features["number"](row[name])
        if (min(row["open"], row["high"], row["low"], row["close"]) <= 0 or row["volume"] < 0
                or not row["low"] <= min(row["open"], row["close"])
                or not max(row["open"], row["close"]) <= row["high"]):
            raise ValueError("invalid training OHLCV domain")
        identity = (row["bar_open_ms"], row["symbol"])
        if identity in seen:
            raise ValueError("duplicate visible training candle")
        seen[identity] = dict(row)
    expected = [(t, s) for t in range(begin, end, DT) for s in universe]
    if len(expected) > features["MAX_PREFIX_ROWS"]:
        raise ValueError("training source window budget")
    if set(seen) != set(expected):
        raise ValueError("missing/unavailable training candle")
    source = [seen[identity] for identity in expected]
    if sha(source) != request["expected_training_source_sha256"]:
        raise ValueError("training source changed against pin")
    return source


def feature_request(request, asof, prefix_sha, features):
    return {"schema": "exia.genetic_primus.synthetic_asof_request/1", "evidence_class": features["EVIDENCE"],
            "decision_asof_ms": asof, "history_start_ms": request["history_start_ms"],
            "universe": list(request["context"]["universe"]), "revision_id": request["revision_id"],
            "expected_prefix_sha256": prefix_sha, "kernel_definition_sha256": request["kernel_definition_sha256"]}


def build_origin_inputs(candles, request, features, guard):
    cutoff = validate_request(request, features, guard)
    ctx = request["context"]
    source = training_window(candles, request, cutoff, features, guard)
    source_index = {(r["bar_open_ms"], r["symbol"]): r for r in source}
    train_rows, feature_lineage = [], []
    for start in request["origin_starts_ms"]:
        asof = start - DT + request["historical_feature_offset_ms"]
        end = start + HORIZON
        historic_prefix = [r for r in source if r["bar_end_exclusive_ms"] <= asof // DT * DT
                           and r["available_at_ms"] <= asof]
        # The outer immutable source pin supplies independent provenance for
        # these derived prefix pins; real-source authenticity is not attested.
        historic = features["build_asof_features"](source, feature_request(request, asof, sha(historic_prefix), features))
        feature_lineage.append({"label_start_ms": start, "feature_asof_ms": asof,
                                "prefix_sha256": historic["prefix_sha256"],
                                "feature_rows_sha256": historic["feature_rows_sha256"],
                                "kernel_definition_sha256": historic["kernel_definition_sha256"]})
        for row in historic["rows"]:
            symbol = row["symbol"]
            path = [source_index[(t, symbol)] for t in range(start, end, DT)]
            available = max(r["available_at_ms"] for r in path)
            if not end <= available <= cutoff:
                raise ValueError("training label availability exceeds cutoff")
            gross = path[-1]["close"] / path[0]["open"] - 1.0
            guard["number"](gross)
            train_rows.append({"row_id": str(start) + ":" + symbol, **row,
                               "label_start_ms": start, "label_end_exclusive_ms": end,
                               "label_available_at_ms": available, "gross_return_3d": gross})
    deploy_request = feature_request(request, ctx["decision_asof_ms"], request["expected_deploy_source_sha256"], features)
    deploy_source, _ = features["visible_prefix"](candles, deploy_request)
    deploy = features["build_asof_features"](candles, deploy_request)
    consumed = {"schema": "exia.genetic_primus.synthetic_consumed_origin_snapshot/1", "evidence_class": EVIDENCE,
                "training_source": source, "deploy_source": deploy_source,
                "timestamp_semantics": "bar_open_epoch_ms;per_row_declared_availability;fixed_revision",
                "actual_exchange_provenance_verified": False}
    binding = {"data_file_sha256": hashlib.sha256(canonical_bytes(consumed) + b"\n").hexdigest(),
               "data_content_sha256": sha(consumed), "complete_grid_sha256": sha([
                   [r["bar_open_ms"], r["symbol"]] for r in source]),
               "timestamp_semantics_sha256": sha(consumed["timestamp_semantics"]),
               "feature_definition_sha256": request["kernel_definition_sha256"],
               "label_definition_sha256": request["label_definition_sha256"],
               "revision_id": request["revision_id"], "universe_sha256": sha(ctx["universe"])}
    pins = {"context": dict(ctx), "binding": binding, "training_rows_sha256": sha(train_rows),
            "deploy_feature_rows_sha256": deploy["feature_rows_sha256"]}
    train_payload = {"schema": "exia.genetic_primus.completed_training_payload/1", "evidence_class": guard["EVIDENCE"],
                     "context": dict(ctx), "binding": binding, "train_cutoff_ms": cutoff,
                     "rows": train_rows, "rows_sha256": sha(train_rows)}
    train_payload["payload_sha256"] = sha(train_payload)
    train_info = guard["validate_training"](train_payload, pins)
    for row in deploy["rows"]:
        guard["keys"](row, guard["DEPLOY_ROW_KEYS"], "computed label-free deploy row")
        guard["feature_row"](row, ctx, binding, ctx["decision_asof_ms"])
    opportunities = [r for r in train_rows if r["proposal_long"]]
    scaler_rows = [{k: r["features"][k] for k in guard["FEATURES"]} for r in opportunities]
    result = {"schema": "exia.genetic_primus.synthetic_causal_origin_inputs/1", "evidence_class": EVIDENCE,
              "context": dict(ctx), "train_cutoff_ms": cutoff, "training": train_payload,
              "historical_feature_offset_ms": request["historical_feature_offset_ms"],
              "deploy_features": deploy, "trusted_fixture_pins": pins,
              "historical_feature_lineage": feature_lineage, "training_guard": train_info,
              "training_guard_scaler_hash_scope": "all_training_rows_not_opportunity_filtered",
              "scaler_candidate_input": {"selection": "all_proposal_long_outcomes_including_losses",
                                         "rows": scaler_rows, "rows_sha256": sha(scaler_rows),
                                         "opportunities": len(opportunities),
                                         "negative_opportunities": sum(r["gross_return_3d"] < 0 for r in opportunities),
                                         "positive_opportunities": sum(r["gross_return_3d"] > 0 for r in opportunities),
                                         "actual_scaler_fitted": False},
              "minimum_opportunities_64_passed": len(opportunities) >= 64, "ess_gate_verified": False,
              "shared_feature_definition_sha256": request["kernel_definition_sha256"],
              "label_definition_sha256": request["label_definition_sha256"],
              "model_ready": False, "full_old_model_bundle_compatible": False,
              "entry_accounting_integrated": False, "real_execution_labels_verified": False,
              "market_data_provenance_verified": False, "actual_compute_latency_verified": False,
              "new_model_fits": 0, "new_market_strategy_evaluations": 0,
              "market_research_launch_ready": False, "real_money_ready": False,
              "safety": {k: False for k in guard["SAFETY_KEYS"]}}
    result["payload_sha256"] = sha(result)
    return consumed, result
