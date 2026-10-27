"""Bounded full8 row adapter; no decoder, filesystem, package import or fit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re

BAR_MS = 14_400_000
HORIZON_MS = 18 * BAR_MS
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
ROW_KEYS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "source_rows"}
SAFETY = {"credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
          "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed"}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def integer(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("bounded integer required")
    return value


def aligned(value):
    integer(value)
    if value % BAR_MS:
        raise ValueError("4h alignment required")
    return value


def digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("digest required")


def protocol_gate(protocol, manifest, manifest_byte_sha256):
    """Binding is metadata only; no raw bytes or original arrival are attested."""
    digest(manifest_byte_sha256)
    if (protocol["schema"] != "exia.genetic_primus.offline_market_engineering_protocol/1"
        or protocol["status"] != "draft_not_authorized"
        or protocol["source_manifest_path"] != "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
        or protocol["source_manifest_byte_sha256"] != manifest_byte_sha256
        or protocol["historical_scope_is_not_prospective"] is not True):
        raise ValueError("protocol binding/state")
    if set(protocol["safety"]) != SAFETY or any(v is not False for v in protocol["safety"].values()):
        raise ValueError("authority drift")
    if (set(protocol["authorization"]) != {"raw_parquet_read_for_this_protocol", "fit_for_this_protocol"}
        or any(v is not False for v in protocol["authorization"].values())):
        raise ValueError("no read/fit authorization in draft")
    clock, models, metadata = protocol["clock"], protocol["models"], manifest["metadata"]
    for key, expected in (("bar_ms", BAR_MS), ("horizon_bars", 18), ("retrain_bars", 18), ("purge_bars", 7)):
        if integer(clock[key]) != expected:
            raise ValueError("clock substitution")
    for key in ("history_start_ms", "origin_start_ms", "training_asof_ms", "nominal_end_exclusive_ms"):
        aligned(clock[key])
    if clock["training_asof_ms"] != clock["origin_start_ms"] - BAR_MS or clock["nominal_end_exclusive_ms"] != clock["origin_start_ms"] + HORIZON_MS:
        raise ValueError("origin/train/end substitution")
    if clock["candidate_calendar"] != "18bar_grid_anchored_to_origin_no_4h_opportunity_expansion":
        raise ValueError("candidate schedule substitution")
    expected_models = {"features": ["return_12", "efficiency_12"], "coefficient_count": 3,
        "hidden_layers": 0, "ridge_alpha": 0.5, "variants": ["uniform", "half_life_90d_anchor_20pct"],
        "ga_enabled": False, "maximum_planned_pipeline_runs_including_failures": 2}
    if sha(models) != sha(expected_models):
        raise ValueError("candidate/search substitution")
    if (metadata["timeframe_ms"] != BAR_MS or metadata["symbols"] != list(SYMBOLS)
        or metadata["coverage_start"] != clock["history_start_ms"]
        or not metadata["coverage_start"] < clock["origin_start_ms"] < clock["nominal_end_exclusive_ms"] < metadata["coverage_end_exclusive"]):
        raise ValueError("manifest clock/universe mismatch")
    files = manifest["snapshot_files"]
    if type(files) is not list or len(files) != 11:
        raise ValueError("approved eleven-file bank required")
    paths = []
    for source in files:
        if set(source) != {"path", "sha256", "size_bytes"}:
            raise ValueError("source identity schema")
        path = source["path"]
        if (type(path) is not str or not path.startswith("Retrodate/") or ":" in path or "\\" in path
            or ".." in PurePosixPath(path).parts or not path.endswith(".parquet")):
            raise ValueError("local approved source path required")
        digest(source["sha256"])
        if not 0 < integer(source["size_bytes"]) <= 8 * 1024 * 1024:
            raise ValueError("source size cap")
        paths.append(path)
    if len(set(paths)) != 11:
        raise ValueError("duplicate source")
    missing = []
    for section in ("availability", "timing", "accounting"):
        missing.extend(f"{section}.{key}" for key, value in protocol[section].items() if value is None)
    return {"metadata_binding_passed": True, "unresolved": sorted(missing),
            "approved_file_count": 11, "raw_file_identity_verified_by_this_step": False,
            "market_data_read_allowed": False, "fit_allowed": False,
            "original_arrival_verified": False}


def candidate_calendar(origin_start_ms, training_asof_ms, history_start_ms, ingestion_lag_ms):
    """Legacy 18-bar anchor, closed-prefix warmup; maturity/purge gate is separate."""
    for value in (origin_start_ms, training_asof_ms, history_start_ms):
        aligned(value)
    integer(ingestion_lag_ms)
    if not history_start_ms < training_asof_ms == origin_start_ms - BAR_MS:
        raise ValueError("fixed origin context required")
    if (origin_start_ms - history_start_ms) // BAR_MS > 20000:
        raise ValueError("calendar cap")
    first = origin_start_ms - ((origin_start_ms - history_start_ms) // HORIZON_MS) * HORIZON_MS
    return [start for start in range(first, training_asof_ms + 1, HORIZON_MS)
            if (start - ingestion_lag_ms - history_start_ms) // BAR_MS >= 48]


class MarketFeatureState:
    """Sequential complete-bar state, O(symbols*13) rows, anchored EMA history."""

    def __init__(self, metadata, manifest_sha256, history_start_ms, ingestion_lag_ms, revision_policy_sha256):
        digest(manifest_sha256)
        digest(revision_policy_sha256)
        aligned(history_start_ms)
        integer(ingestion_lag_ms)  # No implicit zero from draft null.
        aligned(metadata["coverage_start"])
        aligned(metadata["coverage_end_exclusive"])
        if (metadata["symbols"] != list(SYMBOLS) or metadata["timeframe_ms"] != BAR_MS
            or history_start_ms != metadata["coverage_start"]):
            raise ValueError("fixed full8 history/timebase required")
        if metadata["coverage_end_exclusive"] <= history_start_ms:
            raise ValueError("nonempty feature coverage required")
        self.history = history_start_ms
        self.end = metadata["coverage_end_exclusive"]
        self.next_bar = history_start_ms
        self.lag = ingestion_lag_ms
        self.count = 0
        self.states = {}
        self.prefix_sha = sha({"manifest_sha256": manifest_sha256, "history_start_ms": history_start_ms,
                              "revision_policy_sha256": revision_policy_sha256, "symbols": SYMBOLS})

    def feed_bar(self, rows, asof_ms):
        integer(asof_ms)
        if type(rows) is not list or len(rows) != 8:
            raise ValueError("one full8 bar required")
        for row in rows:
            if type(row) is not dict or set(row) != ROW_KEYS:
                raise ValueError("market row schema")
            aligned(row["timestamp"])
            if row["timestamp"] != self.next_bar or type(row["symbol"]) is not str:
                raise ValueError("nonsequential bar")
        if sorted(row["symbol"] for row in rows) != list(SYMBOLS):
            raise ValueError("missing/duplicate/non-full8 symbol")
        if self.next_bar >= self.end or self.count >= 20000:
            raise ValueError("coverage/prefix cap")
        available = integer(self.next_bar + BAR_MS + self.lag)
        if available > asof_ms:
            return False  # No price validation/hash/state update of an unpublished bar.
        ordered = sorted(rows, key=lambda row: row["symbol"])
        updates = {}
        for row in ordered:
            for name in ("open", "high", "low", "close", "volume"):
                value = row[name]
                if type(value) not in (int, float):
                    raise ValueError("numeric visible market OHLCV required")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite or (value < 0 if name == "volume" else value <= 0):
                    raise ValueError("invalid visible market value")
            if (row["high"] < max(row["open"], row["low"], row["close"])
                or row["low"] > min(row["open"], row["high"], row["close"])
                or type(row["source_rows"]) is not int or row["source_rows"] != 240):
                raise ValueError("OHLC/source rows violation")
            old = self.states.get(row["symbol"])
            close = float(row["close"])
            e12 = close if old is None else (1 - 2 / 13) * old["ema12"] + (2 / 13) * close
            e48 = close if old is None else (1 - 2 / 49) * old["ema48"] + (2 / 49) * close
            if not math.isfinite(e12) or not math.isfinite(e48) or min(e12, e48) <= 0:
                raise ValueError("EMA arithmetic overflow")
            updates[row["symbol"]] = {"ema12": e12, "ema48": e48,
                "ema12_tail": ((old["ema12_tail"] if old else []) + [e12])[-4:],
                "rows": ((old["rows"] if old else []) + [dict(row)])[-13:]}
        prefix_sha = sha({"previous": self.prefix_sha, "bar_rows": ordered})
        self.states = updates  # Commit atomically only after every symbol validates.
        self.prefix_sha = prefix_sha
        self.count += 1
        self.next_bar += BAR_MS
        return True

    def snapshot(self, decision_asof_ms):
        integer(decision_asof_ms)
        latest = min(((decision_asof_ms - self.lag) // BAR_MS - 1) * BAR_MS, self.end - BAR_MS)
        if self.count < 48 or self.next_bar - BAR_MS != latest:
            raise ValueError("incomplete/stale prefix or warmup")
        result = []
        for symbol in SYMBOLS:
            state = self.states[symbol]
            closes = [float(row["close"]) for row in state["rows"]]
            path = math.fsum(abs(b - a) for a, b in zip(closes[:-1], closes[1:]))
            values = {"return_12": closes[-1] / closes[0] - 1,
                      "efficiency_12": 0.0 if path == 0 else abs(closes[-1] - closes[0]) / path}
            gap = state["ema12"] / state["ema48"] - 1
            slope = state["ema12"] / state["ema12_tail"][0] - 1
            if any(not math.isfinite(v) for v in (*values.values(), gap, slope)):
                raise ValueError("nonfinite feature arithmetic")
            result.append({"symbol": symbol, "features": values,
                "proposal_long": path != 0 and gap > 0 and slope > 0 and values["return_12"] > 0 and values["efficiency_12"] >= 0.30,
                "decision_asof_ms": decision_asof_ms, "feature_available_at_ms": self.next_bar + self.lag,
                "value_dependencies": {"first_bar_ms": self.next_bar - 13 * BAR_MS, "last_bar_ms": self.next_bar - BAR_MS, "bars": 13},
                "proposal_dependencies": {"first_bar_ms": self.history, "last_bar_ms": self.next_bar - BAR_MS, "bars": self.count},
                "original_arrival_verified": False, "compute_completion_measured": False})
        return {"schema": "exia.genetic_primus.offline_market_feature_snapshot/1", "prefix_sha256": self.prefix_sha,
                "history_bars": self.count, "rows": result, "fit_allowed": False,
                "availability_kind": "explicit_engineering_scenario_not_observed_arrival"}
