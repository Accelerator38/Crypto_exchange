"""Strict synthetic causal payload contracts; no fitting, prediction or execution.

Hashes establish metadata coherence against caller-pinned fixture identities,
not authentic market data, indicator prefix invariance or measured fit latency.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

DT = 14_400_000
HORIZON = 3 * 86_400_000
FEATURES = ["return_12", "efficiency_12"]
EVIDENCE = "synthetic_metadata_only"
MAX_ROWS = 512
MAX_SYMBOLS = 24
SAFETY_KEYS = (
    "credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
    "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed",
)
BINDING_KEYS = (
    "data_file_sha256", "data_content_sha256", "complete_grid_sha256",
    "timestamp_semantics_sha256", "feature_definition_sha256", "label_definition_sha256",
    "revision_id", "universe_sha256",
)
CONTEXT_KEYS = (
    "origin_id", "deploy_start_ms", "deploy_end_exclusive_ms", "training_asof_ms",
    "decision_asof_ms", "bar_ms", "purge_bars", "universe",
)
TRAIN_ROW_KEYS = (
    "row_id", "symbol", "revision_id", "feature_bar_open_ms", "feature_bar_end_exclusive_ms",
    "feature_available_at_ms", "features", "proposal_long", "label_start_ms",
    "label_end_exclusive_ms", "label_available_at_ms", "gross_return_3d",
)
DEPLOY_ROW_KEYS = (
    "symbol", "revision_id", "feature_bar_open_ms", "feature_bar_end_exclusive_ms",
    "feature_available_at_ms", "features", "proposal_long",
)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("wrong exact schema: " + name)


def digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA-256 required")


def timestamp(value):
    if type(value) is not int or not 0 <= value < 2**53:
        raise ValueError("integer UTC epoch milliseconds required")


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite numeric scalar required, not bool")


def exact_list(value, length=None):
    if type(value) is not list or (length is not None and len(value) != length):
        raise ValueError("plain bounded list required")


def context(value):
    keys(value, CONTEXT_KEYS, "context")
    if (type(value["origin_id"]) is not str
            or re.fullmatch(r"synthetic_origin_[0-9]{2}", value["origin_id"]) is None):
        raise ValueError("synthetic origin identity required")
    for field in ("deploy_start_ms", "deploy_end_exclusive_ms", "training_asof_ms", "decision_asof_ms"):
        timestamp(value[field])
    if (type(value["bar_ms"]) is not int or value["bar_ms"] != DT
            or type(value["purge_bars"]) is not int or value["purge_bars"] != 7
            or value["deploy_end_exclusive_ms"] != value["deploy_start_ms"] + HORIZON
            or value["deploy_start_ms"] % DT
            or not value["training_asof_ms"] <= value["decision_asof_ms"] <= value["deploy_start_ms"]):
        raise ValueError("wrong chronological design")
    universe = value["universe"]
    exact_list(universe)
    if (not 1 <= len(universe) <= MAX_SYMBOLS or len(set(universe)) != len(universe)
            or universe != sorted(universe)
            or any(type(s) is not str or re.fullmatch(r"SYN_[A-Z]{1,8}", s) is None for s in universe)):
        raise ValueError("sorted unique synthetic universe required")


def binding(value, ctx):
    keys(value, BINDING_KEYS, "binding")
    for name in BINDING_KEYS:
        if name != "revision_id":
            digest(value[name])
    if (type(value["revision_id"]) is not str
            or re.fullmatch(r"synthetic_revision_[0-9]+", value["revision_id"]) is None
            or value["universe_sha256"] != sha(ctx["universe"])):
        raise ValueError("revision or universe binding")


def pin_check(payload, pins):
    keys(pins, ("context", "binding", "training_rows_sha256", "deploy_feature_rows_sha256"), "trusted fixture pins")
    context(pins["context"])
    binding(pins["binding"], pins["context"])
    digest(pins["training_rows_sha256"])
    digest(pins["deploy_feature_rows_sha256"])
    if payload["context"] != pins["context"] or payload["binding"] != pins["binding"]:
        raise ValueError("identity changed against independent fixture pins")


def feature_row(row, ctx, bind, available_by):
    if row["symbol"] not in ctx["universe"] or row["revision_id"] != bind["revision_id"]:
        raise ValueError("row revision or universe mismatch")
    for field in ("feature_bar_open_ms", "feature_bar_end_exclusive_ms", "feature_available_at_ms"):
        timestamp(row[field])
    start, end, available = (row[k] for k in
        ("feature_bar_open_ms", "feature_bar_end_exclusive_ms", "feature_available_at_ms"))
    if start % DT or end != start + DT or not end <= available <= available_by:
        raise ValueError("future or incomplete feature availability")
    keys(row["features"], FEATURES, "two deploy features")
    for name in FEATURES:
        number(row["features"][name])
    if not 0.0 <= row["features"]["efficiency_12"] <= 1.0:
        raise ValueError("efficiency outside indicator domain")
    if type(row["proposal_long"]) is not bool:
        raise ValueError("typed proposal required")


def verify_self_hash(payload, name):
    digest(payload[name])
    if payload[name] != sha({k: v for k, v in payload.items() if k != name}):
        raise ValueError("mutated payload")


def validate_training(payload, pins):
    keys(payload, ("schema", "evidence_class", "context", "binding", "train_cutoff_ms",
                   "rows", "rows_sha256", "payload_sha256"), "training")
    if payload["schema"] != "exia.genetic_primus.completed_training_payload/1" or payload["evidence_class"] != EVIDENCE:
        raise ValueError("only synthetic completed-label payloads supported")
    ctx, bind = payload["context"], payload["binding"]
    pin_check(payload, pins)
    timestamp(payload["train_cutoff_ms"])
    cutoff = ctx["training_asof_ms"] - 7 * DT
    if cutoff < 0 or payload["train_cutoff_ms"] != cutoff:
        raise ValueError("cutoff must be based on training_asof, not deployment")
    rows = payload["rows"]
    exact_list(rows)
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("training metadata size cap")
    order = []
    groups = {}
    for row in rows:
        keys(row, TRAIN_ROW_KEYS, "completed training row")
        for field in ("label_start_ms", "label_end_exclusive_ms", "label_available_at_ms"):
            timestamp(row[field])
        begin, end, available = (row[k] for k in
            ("label_start_ms", "label_end_exclusive_ms", "label_available_at_ms"))
        if begin % DT or end != begin + HORIZON or not end <= available <= cutoff:
            raise ValueError("future, incomplete or unavailable training label")
        feature_row(row, ctx, bind, min(begin, ctx["training_asof_ms"]))
        number(row["gross_return_3d"])
        if row["gross_return_3d"] <= -1:
            raise ValueError("nonpositive terminal price implied")
        if row["row_id"] != str(begin) + ":" + row["symbol"]:
            raise ValueError("row identity mismatch")
        order.append((begin, row["symbol"]))
        groups.setdefault(begin, []).append(row["symbol"])
    if order != sorted(order) or len(set(order)) != len(order):
        raise ValueError("duplicate or unordered training rows")
    if any(symbols != ctx["universe"] for symbols in groups.values()):
        raise ValueError("incomplete fixed-universe label group")
    starts = sorted(groups)
    if any(b - a != HORIZON for a, b in zip(starts, starts[1:])):
        raise ValueError("missing or overlapping three-day training origin")
    digest(payload["rows_sha256"])
    if payload["rows_sha256"] != sha(rows) or payload["rows_sha256"] != pins["training_rows_sha256"]:
        raise ValueError("training rows hash mismatch")
    verify_self_hash(payload, "payload_sha256")
    return {"train_payload_sha256": payload["payload_sha256"], "train_rows_sha256": sha(rows),
            "scaler_fit_rows_sha256": sha([{k: row["features"][k] for k in FEATURES} for row in rows]),
            "train_cutoff_ms": cutoff, "completed_rows": len(rows),
            "negative_outcomes": sum(row["gross_return_3d"] < 0 for row in rows),
            "positive_outcomes": sum(row["gross_return_3d"] > 0 for row in rows)}


def validate_deploy(payload, pins):
    keys(payload, ("schema", "evidence_class", "context", "binding", "rows", "rows_sha256",
                   "decision_model_sha256", "payload_sha256"), "deploy features only")
    if payload["schema"] != "exia.genetic_primus.deploy_features_payload/1" or payload["evidence_class"] != EVIDENCE:
        raise ValueError("only synthetic isolated deploy features supported")
    pin_check(payload, pins)
    rows, ctx = payload["rows"], payload["context"]
    exact_list(rows, len(ctx["universe"]))
    for row in rows:
        keys(row, DEPLOY_ROW_KEYS, "deploy row: outcomes/labels forbidden")
        feature_row(row, ctx, payload["binding"], ctx["decision_asof_ms"])
    if [r["symbol"] for r in rows] != ctx["universe"]:
        raise ValueError("missing, duplicate or unordered deploy symbol")
    digest(payload["rows_sha256"])
    digest(payload["decision_model_sha256"])
    if payload["rows_sha256"] != sha(rows) or payload["rows_sha256"] != pins["deploy_feature_rows_sha256"]:
        raise ValueError("deploy row hash mismatch")
    verify_self_hash(payload, "payload_sha256")
    return {"deploy_payload_sha256": payload["payload_sha256"], "feature_rows_sha256": sha(rows),
            "deploy_rows": len(rows), "decision_asof_ms": ctx["decision_asof_ms"]}


def validate_model(payload, pins, train_info):
    keys(payload, ("schema", "evidence_class", "fit_mode", "context", "binding", "features",
                   "train_payload_sha256", "train_rows_sha256", "scaler_fit_rows_sha256", "scaler",
                   "scaler_payload_sha256", "coefficients", "training_asof_ms", "fit_started_at_ms",
                   "fit_completed_at_ms", "model_available_at_ms", "model_sha256"), "manual model fixture")
    if (payload["schema"] != "exia.genetic_primus.synthetic_causal_model_payload/1"
            or payload["evidence_class"] != EVIDENCE or payload["fit_mode"] != "manual_fixture_not_fit"):
        raise ValueError("not an actual trained model admission contract")
    pin_check(payload, pins)
    ctx = payload["context"]
    if payload["features"] != FEATURES:
        raise ValueError("wrong feature order")
    for key in ("train_payload_sha256", "train_rows_sha256", "scaler_fit_rows_sha256"):
        digest(payload[key])
        if payload[key] != train_info[key]:
            raise ValueError("model training/scaler lineage mismatch")
    scaler = payload["scaler"]
    keys(scaler, ("feature_names", "median", "iqr", "fit_rows_sha256", "derivation"), "scaler")
    if (scaler["feature_names"] != FEATURES or scaler["derivation"] != "manual_fixture_not_fit"
            or scaler["fit_rows_sha256"] != train_info["scaler_fit_rows_sha256"]):
        raise ValueError("scaler input lineage mismatch")
    for name in ("median", "iqr"):
        exact_list(scaler[name], len(FEATURES))
        for value in scaler[name]:
            number(value)
    if any(v <= 0 for v in scaler["iqr"]):
        raise ValueError("nonpositive scaler denominator")
    digest(payload["scaler_payload_sha256"])
    if payload["scaler_payload_sha256"] != sha(scaler):
        raise ValueError("scaler payload hash mismatch")
    exact_list(payload["coefficients"], 3)
    for value in payload["coefficients"]:
        number(value)
    times = [payload[k] for k in
             ("training_asof_ms", "fit_started_at_ms", "fit_completed_at_ms", "model_available_at_ms")]
    for value in times:
        timestamp(value)
    if times[0] != ctx["training_asof_ms"] or times != sorted(times) or times[-1] > ctx["decision_asof_ms"]:
        raise ValueError("model not causally available at decision")
    verify_self_hash(payload, "model_sha256")
    return {"model_sha256": payload["model_sha256"], "scaler_payload_sha256": payload["scaler_payload_sha256"],
            "fit_mode": payload["fit_mode"], "model_available_at_ms": times[-1]}


def validate_bundle(bundle, pins):
    keys(bundle, ("training", "deploy", "model"), "isolated origin bundle")
    train = validate_training(bundle["training"], pins)
    deploy = validate_deploy(bundle["deploy"], pins)
    model = validate_model(bundle["model"], pins, train)
    if bundle["deploy"]["decision_model_sha256"] != model["model_sha256"]:
        raise ValueError("wrong active decision model")
    return {"schema": "exia.genetic_primus.causal_payload_guard_result/1", "evidence_class": EVIDENCE,
            "origin_id": pins["context"]["origin_id"], "training": train, "deploy": deploy, "model": model,
            "declared_metadata_coherent": True, "actual_feature_causality_verified": False,
            "actual_scaler_fitting_verified": False, "actual_training_latency_verified": False,
            "actual_revision_provenance_verified": False, "actual_grid_verified": False,
            "entry_model_sha_verified": False, "accounting_integration_verified": False,
            "new_model_fits": 0, "new_market_strategy_evaluations": 0,
            "market_research_launch_ready": False, "real_money_ready": False,
            "safety": {k: False for k in SAFETY_KEYS}}
