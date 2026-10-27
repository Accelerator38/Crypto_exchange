"""Pure diagnostic open/open labels, never a net training objective."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math

BAR_MS = 14_400_000
FEATURES = ("return_12", "efficiency_12")
ROW_KEYS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "source_rows"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("invalid numeric observation")


def gross(entry, exit_open):
    number(entry, True)
    number(exit_open, True)
    with localcontext() as context:
        context.prec = 50
        return str(Decimal(str(exit_open)) / Decimal(str(entry)) - Decimal(1))


def observation(row, source, symbols):
    if set(row) != ROW_KEYS or type(row["timestamp"]) is not int or row["timestamp"] % BAR_MS:
        raise ValueError("observation schema/grid")
    if row["symbol"] not in symbols or type(row["source_rows"]) is not int or row["source_rows"] != 240:
        raise ValueError("symbol/source-row provenance")
    for key in ("open", "high", "low", "close"):
        number(row[key], True)
    number(row["volume"])
    if (row["volume"] < 0 or not row["low"] <= min(row["open"], row["close"])
            or not max(row["open"], row["close"]) <= row["high"]):
        raise ValueError("invalid OHLCV envelope")
    if (set(source) != {"path", "sha256", "size_bytes"}
            or type(source["path"]) is not str or not source["path"].endswith(".parquet")
            or type(source["sha256"]) is not str or len(source["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["sha256"])
            or type(source["size_bytes"]) is not int or source["size_bytes"] <= 0):
        raise ValueError("source identity")
    return {"timestamp": row["timestamp"], "symbol": row["symbol"], "open": row["open"],
            "canonical_full_row_sha256": sha(row), "source": dict(source)}


def sample(feature, schedule, entry, exit_open, feature_reference, lag_ms):
    symbol, asof = feature["symbol"], feature["decision_asof_ms"]
    if (feature["kind"] != "candidate_features_not_label_gated" or feature["fit_allowed"] is not False
            or feature["label_fields_present"] is not False or type(asof) is not int
            or asof != schedule["decision_asof_ms"] or not schedule["scenario_eligible"]
            or feature["feature_available_at_ms"] > asof):
        raise ValueError("feature timing/role")
    if set(feature["features"]) != set(FEATURES) or type(feature["proposal_long"]) is not bool:
        raise ValueError("feature layout")
    for name in FEATURES:
        number(feature["features"][name])
    if (schedule["execution_delay_ms"] != 0 or schedule["decision_ready_ms"] != asof
            or schedule["entry_open_ms"] != (asof // BAR_MS + 1) * BAR_MS
            or schedule["exit_open_ms"] - schedule["entry_open_ms"] != 18 * BAR_MS
            or schedule["label_available_at_ms"] != schedule["exit_open_ms"] + BAR_MS + lag_ms
            or schedule["label_available_at_ms"] > schedule["training_cutoff_ms"]):
        raise ValueError("registered label schedule changed")
    if (entry["symbol"] != symbol or exit_open["symbol"] != symbol
            or entry["timestamp"] != schedule["entry_open_ms"]
            or exit_open["timestamp"] != schedule["exit_open_ms"]):
        raise ValueError("label dependency key substitution")
    target = gross(entry["open"], exit_open["open"])
    return {"sample_id": sha({"symbol": symbol, "decision_asof_ms": asof}),
            "symbol": symbol, "decision_asof_ms": asof,
            "feature_order": list(FEATURES), "features": dict(feature["features"]),
            "feature_reference": feature_reference, "feature_row_sha256": sha(feature),
            "feature_prefix_sha256": feature["prefix_sha256"], "proposal_long": feature["proposal_long"],
            "entry_observation": entry, "exit_observation": exit_open,
            "entry_open_ms": schedule["entry_open_ms"], "exit_open_ms": schedule["exit_open_ms"],
            "label_available_at_ms": schedule["label_available_at_ms"],
            "training_cutoff_ms": schedule["training_cutoff_ms"],
            "gross_open_return_decimal": target,
            "target_kind": "gross_open_open_diagnostic_not_net_training_target",
            "entry_is_reference_open_not_fill": True, "outcome_filter_applied": False,
            "fit_allowed": False, "net_objective_allowed": False}
