"""Bounded synthetic as-of OHLCV feature builder; no models, labels or runtime.

The flat-path policy is an explicit engineering difference from the historical
pandas NaN/drop path. No claim of historical evaluator equivalence or real data
provenance is made. Caller pins the visible prefix independently of later bars.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

DT = 14_400_000
MAX_INPUT_ROWS = 1024
MAX_PREFIX_ROWS = 512
MIN_HISTORY_BARS = 48
EVIDENCE = "synthetic_candle_engineering_only"
SAFETY_KEYS = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
               "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")
CANDLE_KEYS = ("symbol", "revision_id", "bar_open_ms", "bar_end_exclusive_ms", "available_at_ms",
               "open", "high", "low", "close", "volume")
REQUEST_KEYS = ("schema", "evidence_class", "decision_asof_ms", "history_start_ms", "universe",
                "revision_id", "expected_prefix_sha256", "kernel_definition_sha256")
DEFINITION = {
    "schema": "exia.genetic_primus.synthetic_asof_feature_definition/1",
    "bar_ms": DT, "minimum_history_bars": MIN_HISTORY_BARS,
    "history_start": "fixed_caller_pin_not_moving_window",
    "return_12": "close[t]/close[t-12]-1",
    "efficiency_12": "abs(close[t]-close[t-12])/sum(abs(diff(close[t-12:t+1])))",
    "ema": "first_close_seed;alpha=2/(span+1);recursive_adjust_false;spans_12_48",
    "proposal": "ema12/ema48-1>0 and ema12/ema12[t-3]-1>0 and return12>0 and efficiency12>=0.30",
    "flat_path": "return12=0;efficiency12=0;proposal=false;not_legacy_nan_drop",
    "features_available_at": "declared_decision_asof_not_measured_compute_latency",
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("wrong exact schema: " + name)


def timestamp(value):
    if type(value) is not int or not 0 <= value < 2**53:
        raise ValueError("integer UTC epoch milliseconds required")


def digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA required")


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite OHLCV scalars required")


def validate_request(request):
    keys(request, REQUEST_KEYS, "asof request")
    if (request["schema"] != "exia.genetic_primus.synthetic_asof_request/1"
            or request["evidence_class"] != EVIDENCE):
        raise ValueError("only synthetic asof requests supported")
    for name in ("decision_asof_ms", "history_start_ms"):
        timestamp(request[name])
    if request["history_start_ms"] % DT:
        raise ValueError("aligned fixed history start required")
    end = request["decision_asof_ms"] // DT * DT
    count = (end - request["history_start_ms"]) // DT
    universe = request["universe"]
    if (type(universe) is not list or not 1 <= len(universe) <= 4
            or any(type(s) is not str or re.fullmatch(r"SYN_[A-Z]{1,8}", s) is None for s in universe)
            or len(set(universe)) != len(universe) or universe != sorted(universe)):
        raise ValueError("sorted unique synthetic universe required")
    if count < MIN_HISTORY_BARS or count * len(universe) > MAX_PREFIX_ROWS:
        raise ValueError("history warmup/prefix budget")
    if (type(request["revision_id"]) is not str
            or re.fullmatch(r"synthetic_revision_[0-9]+", request["revision_id"]) is None):
        raise ValueError("synthetic revision pin required")
    for name in ("expected_prefix_sha256", "kernel_definition_sha256"):
        digest(request[name])
    if request["kernel_definition_sha256"] != sha(DEFINITION):
        raise ValueError("kernel definition changed")
    return end, count


def visible_prefix(candles, request):
    """Validate visible data; unavailable/later OHLCV values never feed features."""
    end, count = validate_request(request)
    if type(candles) is not list or not 1 <= len(candles) <= MAX_INPUT_ROWS:
        raise ValueError("input row budget")
    seen = {}
    for row in candles:
        keys(row, CANDLE_KEYS, "synthetic OHLCV row")
        for name in ("bar_open_ms", "bar_end_exclusive_ms", "available_at_ms"):
            timestamp(row[name])
        if (type(row["symbol"]) is not str or type(row["revision_id"]) is not str
                or row["bar_open_ms"] % DT or row["bar_end_exclusive_ms"] != row["bar_open_ms"] + DT):
            raise ValueError("invalid candle envelope")
        # Gate before OHLCV values/revision checks. Later revisions to old bars
        # also stay invisible until their own availability timestamp.
        if (row["bar_end_exclusive_ms"] > end
                or row["available_at_ms"] > request["decision_asof_ms"]
                or row["bar_open_ms"] < request["history_start_ms"]):
            continue
        if row["available_at_ms"] < row["bar_end_exclusive_ms"]:
            raise ValueError("available before candle closed")
        if row["symbol"] not in request["universe"] or row["revision_id"] != request["revision_id"]:
            raise ValueError("visible revision/universe changed")
        for name in ("open", "high", "low", "close", "volume"):
            number(row[name])
        if (min(row["open"], row["high"], row["low"], row["close"]) <= 0
                or row["volume"] < 0 or not row["low"] <= min(row["open"], row["close"])
                or not max(row["open"], row["close"]) <= row["high"]):
            raise ValueError("invalid OHLCV domain")
        identity = (row["bar_open_ms"], row["symbol"])
        if identity in seen:
            raise ValueError("duplicate visible candle")
        seen[identity] = dict(row)
    expected = [(start, symbol) for start in range(request["history_start_ms"], end, DT)
                for symbol in request["universe"]]
    if len(expected) != count * len(request["universe"]) or set(seen) != set(expected):
        raise ValueError("missing/unavailable candle in exact asof grid")
    prefix = [seen[identity] for identity in expected]
    if sha(prefix) != request["expected_prefix_sha256"]:
        raise ValueError("visible prefix changed against pin")
    return prefix, end


def ema(values, span):
    result = [float(values[0])]
    alpha = 2.0 / (span + 1.0)
    for value in values[1:]:
        result.append((1.0 - alpha) * result[-1] + alpha * float(value))
    return result


def build_asof_features(candles, request):
    prefix, end = visible_prefix(candles, request)
    rows, diagnostics = [], []
    for symbol in request["universe"]:
        local = [r for r in prefix if r["symbol"] == symbol]
        closes = [r["close"] for r in local]
        path = math.fsum(abs(float(b) - float(a)) for a, b in zip(closes[-13:-1], closes[-12:]))
        gross = float(closes[-1]) / float(closes[-13]) - 1.0
        flat = path == 0.0
        efficiency = 0.0 if flat else abs(float(closes[-1]) - float(closes[-13])) / path
        e12, e48 = ema(closes, 12), ema(closes, 48)
        gap = e12[-1] / e48[-1] - 1.0
        slope = e12[-1] / e12[-4] - 1.0
        proposal = not flat and gap > 0 and slope > 0 and gross > 0 and efficiency >= 0.30
        for value in (gross, efficiency, gap, slope):
            number(value)
        if not 0.0 <= efficiency <= 1.0 + 1e-12:
            raise ValueError("invalid computed efficiency")
        # Clamp only floating roundoff to the mathematical [0, 1] domain.
        efficiency = min(efficiency, 1.0)
        rows.append({"symbol": symbol, "revision_id": request["revision_id"],
                     "feature_bar_open_ms": end - DT, "feature_bar_end_exclusive_ms": end,
                     "feature_available_at_ms": request["decision_asof_ms"],
                     "features": {"return_12": gross, "efficiency_12": efficiency},
                     "proposal_long": bool(proposal)})
        diagnostics.append({"symbol": symbol, "ema_gap_12_48": gap, "ema_slope_12_3": slope,
                            "flat_path_abstention": flat, "history_bars": len(local)})
    result = {"schema": "exia.genetic_primus.synthetic_asof_feature_candidate/1", "evidence_class": EVIDENCE,
              "decision_asof_ms": request["decision_asof_ms"], "history_start_ms": request["history_start_ms"],
              "feature_end_exclusive_ms": end, "prefix_sha256": sha(prefix),
              "kernel_definition_sha256": sha(DEFINITION), "rows": rows, "diagnostics": diagnostics,
              "feature_rows_sha256": sha(rows), "availability_is_declared_not_measured": True,
              "real_indicator_equivalence_verified": False, "market_data_provenance_verified": False,
              "new_model_fits": 0, "new_market_strategy_evaluations": 0,
              "market_research_launch_ready": False, "real_money_ready": False,
              "safety": {k: False for k in SAFETY_KEYS}}
    result["payload_sha256"] = sha(result)
    return result
