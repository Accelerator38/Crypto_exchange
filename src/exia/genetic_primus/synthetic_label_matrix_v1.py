"""Transient synthetic X/y + immutable manifest. No fit, costs or market I/O."""
from __future__ import annotations

import hashlib
import json
import math

FEATURES = ("return_12", "efficiency_12")
SAFETY = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
          "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")
FIXED = {
    "schema": "exia.genetic_primus.synthetic_label_matrix_contract/1",
    "status": "engineering_only_not_registered_market_objective",
    "purpose": "causal_label_and_matrix_checks_not_profitability_validation",
    "bar_ms": 14400000, "horizon_bars": 18, "purge_bars": 7,
    "features": list(FEATURES), "target": "exit_open / entry_open - 1",
    "target_kind": "gross_diagnostic_not_net_training_objective",
    "entry_exit": "both_shifted_by_first_grid_open_strictly_after_decision_plus_delay",
    "availability": "max_complete_entry_exit_row_publication",
    "cutoff": "label_available_at <= training_asof - 7 * bar_ms",
    "cost_model": None, "fill_evidence_verified": False,
    "drop_losses": False, "drop_flat_or_abstention_rows": False,
}


def sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def assess(contract):
    if type(contract) is not dict or set(contract) != set(FIXED) | {"safety"}:
        raise ValueError("synthetic matrix contract schema")
    # JSON comparison distinguishes false from 0, integer constants from booleans/floats.
    if sha({k: contract[k] for k in FIXED}) != sha(FIXED):
        raise ValueError("synthetic target policy substitution")
    safety = contract["safety"]
    if type(safety) is not dict or set(safety) != set(SAFETY) or any(safety[k] is not False for k in SAFETY):
        raise ValueError("authority must stay false")
    return {"fit_allowed": False, "net_objective_registered": False,
            "costs_registered": False, "fill_evidence_verified": False}


def positive(value):
    if type(value) not in (int, float):
        raise ValueError("positive numeric input required")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("positive finite input required")
    return value


def gross_open_target(entry_open, exit_open):
    """Synthetic diagnostic only: NOT realized return, fills or a net objective."""
    entry, exit_value = positive(entry_open), positive(exit_open)
    try:
        result = exit_value / entry - 1.0
    except OverflowError:
        raise ValueError("target overflow") from None
    if not math.isfinite(result):
        raise ValueError("nonfinite target")
    return result


def feature_values(prefix, symbol):
    rows = [r for r in prefix if r["symbol"] == symbol]
    if len(rows) < 48:
        raise ValueError("feature warmup required")
    closes = [float(positive(r["close"])) for r in rows[-13:]]
    path = math.fsum(abs(b - a) for a, b in zip(closes[:-1], closes[1:]))
    values = {"return_12": closes[-1] / closes[0] - 1.0,
              "efficiency_12": 0.0 if path == 0 else abs(closes[-1] - closes[0]) / path}
    if any(not math.isfinite(v) for v in values.values()):
        raise ValueError("nonfinite feature")
    return values


def materialize(candles, request, contract, origin, timing, execution, kernel):
    """Only origin's gated synthetic rows are materialized; raw values stay transient."""
    assessed = assess(contract)
    plan = origin["prepare_origin"](candles, request, timing, execution, kernel)
    index = origin["index_candles"](candles, request, timing)
    contexts = {}
    matrix, labels, lineage, row_ids = [], [], [], []
    cutoff = plan["context"]["training_cutoff_ms"]
    for row in plan["scaler_train_rows"]:
        decision, symbol = row["decision_asof_ms"], row["symbol"]
        entry = index[(row["entry_open_ms"], symbol)]
        exit_row = index[(row["exit_open_ms"], symbol)]
        available = max(entry["available_at_ms"], exit_row["available_at_ms"])
        panel_available = max(index[(start, member)]["available_at_ms"]
                              for start in (row["entry_open_ms"], row["exit_open_ms"])
                              for member in request["universe"])
        if panel_available > cutoff or panel_available != row["label_available_at_ms"]:
            raise ValueError("label maturity substitution")
        if decision not in contexts:
            contexts[decision] = origin["visible_prefix"](index, decision, request, timing)
        values = feature_values(contexts[decision], symbol)
        if sha(values) != row["feature_values_sha256"]:
            raise ValueError("feature/matrix mismatch")
        # Target OHLCV validation/use is after the complete-row availability/purge gate.
        for label_row in (entry, exit_row):
            for name in ("open", "high", "low", "close"):
                positive(label_row[name])
            if (label_row["high"] < max(label_row["open"], label_row["low"], label_row["close"])
                or label_row["low"] > min(label_row["open"], label_row["high"], label_row["close"])):
                raise ValueError("label OHLC inconsistent")
        target = gross_open_target(entry["open"], exit_row["open"])
        row_id = {"decision_asof_ms": decision, "symbol": symbol}
        matrix.append([values[f] for f in FEATURES])
        labels.append(target)
        row_ids.append(row_id)
        lineage.append({**row_id, "features_sha256": sha(values), "target_sha256": sha(target),
                        "entry_open_ms": row["entry_open_ms"], "exit_open_ms": row["exit_open_ms"],
                        "entry_row_sha256": sha(entry), "exit_row_sha256": sha(exit_row),
                        "label_available_at_ms": available, "panel_label_available_at_ms": panel_available,
                        "training_cutoff_ms": cutoff,
                        "feature_available_at_ms": row["feature_available_at_ms"],
                        "value_dependencies": row["value_dependencies"],
                        "proposal_dependencies": row["proposal_dependencies"]})
    if len({(r["decision_asof_ms"], r["symbol"]) for r in row_ids}) != len(row_ids):
        raise ValueError("duplicate matrix row")
    if [[r["decision_asof_ms"], r["symbol"]] for r in row_ids] != sorted(
        [r["decision_asof_ms"], r["symbol"]] for r in row_ids):
        raise ValueError("matrix row order")
    if origin["sha"]([(r["decision_asof_ms"], r["symbol"]) for r in row_ids]) != plan["scaler_plan"]["row_ids_sha256"]:
        raise ValueError("scaler/matrix row identity mismatch")
    deployment_prefix = origin["visible_prefix"](index, request["decision_asof_ms"], request, timing)
    deployment = []
    for row in plan["decision_features"]:
        values = feature_values(deployment_prefix, row["symbol"])
        if sha(values) != row["feature_values_sha256"]:
            raise ValueError("deployment feature substitution")
        # An explicit whitelist, never copy a training row/target into this payload.
        deployment.append({"symbol": row["symbol"], "decision_asof_ms": row["decision_asof_ms"],
                           "features": values, "proposal_long": row["proposal_long"],
                           "feature_available_at_ms": row["feature_available_at_ms"]})
    manifest = {"schema": "exia.genetic_primus.synthetic_training_matrix_manifest/1",
                "context": plan["context"], "contract_sha256": sha(contract),
                "source_snapshot_sha256": request["expected_snapshot_sha256"],
                "features": list(FEATURES), "shape": [len(matrix), 2],
                "row_ids_sha256": sha(row_ids), "x_sha256": sha(matrix), "y_sha256": sha(labels),
                "paired_xy_sha256": sha({"row_ids": row_ids, "x": matrix, "y": labels}),
                "training_rows": lineage, "scaler_plan": plan["scaler_plan"],
                "distinct_training_decisions": len({r["decision_asof_ms"] for r in row_ids}),
                "synthetic_target_counts": {"negative": sum(v < 0 for v in labels),
                                             "zero": sum(v == 0 for v in labels), "positive": sum(v > 0 for v in labels)},
                "deployment_features_sha256": sha(deployment), "excluded_candidates": plan["excluded_candidates"],
                "raw_prices_or_xy_persisted": False, "synthetic_only": True,
                "assessment": assessed, "scaler_fit_count": 0, "model_fit_count": 0,
                "market_financial_evaluation_count": 0, "observed_arrival_verified": False,
                "safety": contract["safety"]}
    # X/y and deployment values exist only for the caller's bounded synthetic assertions.
    return {"x": matrix, "y": labels, "row_ids": row_ids, "deployment": deployment, "manifest": manifest}
