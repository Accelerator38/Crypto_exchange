"""Independent synthetic training/deploy identities; supersedes shared v1 hash.

Trusted pure v1 helpers are used only for bounded validation and feature math.
Training construction does not require a deploy prefix or its source hash.
"""
from __future__ import annotations

import hashlib
import json


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def build_training_inputs(candles, request, labels, features, guard):
    fields = [k for k in labels["REQUEST_KEYS"] if k != "expected_deploy_source_sha256"]
    guard["keys"](request, fields, "training-only request: no deploy source pin")
    if request["schema"] != "exia.genetic_primus.synthetic_training_request/2":
        raise ValueError("separate training request required")
    compatible = dict(request)
    compatible["schema"] = "exia.genetic_primus.synthetic_causal_origin_request/1"
    # Validation shim only; never stored in training identity or read as data.
    compatible["expected_deploy_source_sha256"] = sha([])
    cutoff = labels["validate_request"](compatible, features, guard)
    source = labels["training_window"](candles, compatible, cutoff, features, guard)
    index = {(r["bar_open_ms"], r["symbol"]): r for r in source}
    rows, lineage = [], []
    for start in request["origin_starts_ms"]:
        asof = start - labels["DT"] + request["historical_feature_offset_ms"]
        prefix = [r for r in source if r["bar_end_exclusive_ms"] <= asof // labels["DT"] * labels["DT"]
                  and r["available_at_ms"] <= asof]
        historic = features["build_asof_features"](source, labels["feature_request"](compatible, asof, sha(prefix), features))
        lineage.append({"label_start_ms": start, "feature_asof_ms": asof, "prefix_sha256": historic["prefix_sha256"],
                        "feature_rows_sha256": historic["feature_rows_sha256"],
                        "kernel_definition_sha256": historic["kernel_definition_sha256"]})
        end = start + labels["HORIZON"]
        for feature in historic["rows"]:
            path = [index[(t, feature["symbol"])] for t in range(start, end, labels["DT"])]
            available = max(r["available_at_ms"] for r in path)
            if not end <= available <= cutoff:
                raise ValueError("label unavailable at train cutoff")
            gross = path[-1]["close"] / path[0]["open"] - 1.0
            guard["number"](gross)
            rows.append({"row_id": str(start) + ":" + feature["symbol"], **feature,
                         "label_start_ms": start, "label_end_exclusive_ms": end,
                         "label_available_at_ms": available, "gross_return_3d": gross})
    snapshot = {"schema": "exia.genetic_primus.synthetic_training_source_snapshot/2",
                "evidence_class": labels["EVIDENCE"], "source": source,
                "timestamp_semantics": "bar_open_epoch_ms;declared_availability;fixed_revision",
                "actual_exchange_provenance_verified": False}
    ctx = request["context"]
    binding = {"data_file_sha256": hashlib.sha256(canonical_bytes(snapshot) + b"\n").hexdigest(),
               "data_content_sha256": sha(snapshot), "complete_grid_sha256": sha([[r["bar_open_ms"], r["symbol"]] for r in source]),
               "timestamp_semantics_sha256": sha(snapshot["timestamp_semantics"]),
               "feature_definition_sha256": request["kernel_definition_sha256"],
               "label_definition_sha256": request["label_definition_sha256"], "revision_id": request["revision_id"],
               "universe_sha256": sha(ctx["universe"])}
    payload = {"schema": "exia.genetic_primus.completed_training_payload/1", "evidence_class": guard["EVIDENCE"],
               "context": dict(ctx), "binding": binding, "train_cutoff_ms": cutoff, "rows": rows, "rows_sha256": sha(rows)}
    payload["payload_sha256"] = sha(payload)
    pins = {"context": dict(ctx), "binding": binding, "training_rows_sha256": sha(rows),
            "deploy_feature_rows_sha256": sha([])}  # Empty compatibility slot, no deploy admission.
    train_info = guard["validate_training"](payload, pins)
    selected = [r for r in rows if r["proposal_long"]]
    scaler_rows = [{k: r["features"][k] for k in guard["FEATURES"]} for r in selected]
    record = {"schema": "exia.genetic_primus.synthetic_training_inputs/2", "training": payload,
              "training_pins": pins, "training_guard": train_info, "historical_feature_lineage": lineage,
              "historical_feature_offset_ms": request["historical_feature_offset_ms"],
              "training_guard_scaler_hash_scope": "all_training_rows_not_opportunity_filtered",
              "scaler_candidate_input": {"selection": "all_proposal_long_outcomes_including_losses", "rows": scaler_rows,
                                         "rows_sha256": sha(scaler_rows), "opportunities": len(selected),
                                         "negative_opportunities": sum(r["gross_return_3d"] < 0 for r in selected),
                                         "positive_opportunities": sum(r["gross_return_3d"] > 0 for r in selected),
                                         "actual_scaler_fitted": False}}
    record["payload_sha256"] = sha(record)
    return snapshot, record


def build_origin_inputs(candles, request, labels, features, guard):
    training_request = {k: v for k, v in request.items() if k != "expected_deploy_source_sha256"}
    training_request["schema"] = "exia.genetic_primus.synthetic_training_request/2"
    labels["validate_request"](request, features, guard)
    training_snapshot, training = build_training_inputs(candles, training_request, labels, features, guard)
    ctx = request["context"]
    deploy_request = labels["feature_request"](request, ctx["decision_asof_ms"], request["expected_deploy_source_sha256"], features)
    deploy_source, _ = features["visible_prefix"](candles, deploy_request)
    deploy = features["build_asof_features"](candles, deploy_request)
    deploy_snapshot = {"schema": "exia.genetic_primus.synthetic_deploy_source_snapshot/2",
                       "evidence_class": labels["EVIDENCE"], "source": deploy_source,
                       "actual_exchange_provenance_verified": False}
    for row in deploy["rows"]:
        guard["keys"](row, guard["DEPLOY_ROW_KEYS"], "label-free deploy row")
        guard["feature_row"](row, ctx, {"revision_id": request["revision_id"]}, ctx["decision_asof_ms"])
    if training["training"]["binding"]["feature_definition_sha256"] != deploy["kernel_definition_sha256"]:
        raise ValueError("different training/deploy feature definitions")
    result = {"schema": "exia.genetic_primus.synthetic_causal_origin_inputs/2", "evidence_class": labels["EVIDENCE"],
              "context": dict(ctx), "training_inputs": training, "deploy_features": deploy,
              "deploy_source_sha256": sha(deploy_snapshot),
              "deploy_source_file_sha256": hashlib.sha256(canonical_bytes(deploy_snapshot) + b"\n").hexdigest(),
              "independent_training_deploy_bindings": True,
              "shared_feature_definition_sha256": request["kernel_definition_sha256"],
              "minimum_opportunities_64_passed": training["scaler_candidate_input"]["opportunities"] >= 64,
              "ess_gate_verified": False, "full_old_model_bundle_compatible": False, "model_ready": False,
              "entry_accounting_integrated": False, "real_execution_labels_verified": False,
              "market_data_provenance_verified": False, "actual_compute_latency_verified": False,
              "new_model_fits": 0, "new_market_strategy_evaluations": 0,
              "market_research_launch_ready": False, "real_money_ready": False,
              "safety": {k: False for k in guard["SAFETY_KEYS"]}}
    result["payload_sha256"] = sha(result)
    return training_snapshot, deploy_snapshot, result
