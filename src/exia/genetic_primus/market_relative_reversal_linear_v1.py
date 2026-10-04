"""Pure research primitives. No I/O, runtime integration or trading authority."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isfinite
from typing import Iterable, Mapping

BAR_MS = 240 * 60_000
HORIZON_MS = 3 * 86_400_000
PURGE_BARS = 7
TRAIN_WINDOW_MS = 730 * 86_400_000
RESERVED_START_MS = 1791158400000
RESERVED_END_MS = 1806710400000
UNIVERSE = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT",
            "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
SCHEMA = "exia.genetic_primus.market_relative_reversal_linear/1"


@dataclass(frozen=True)
class Observation:
    origin_ms: int
    feature_bar_start_ms: int
    label_end_ms: int
    symbol: str
    relative_return: float
    gross_return_3d: float


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _finite(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("finite numeric value required")
    return float(value)


def _quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (1 - fraction) * sorted_values[lower] + fraction * sorted_values[upper]


def market_relative_features(returns_12: Mapping[str, float]) -> dict[str, float]:
    """Caller supplies returns from the same fully closed 4h bar for all assets."""
    if set(returns_12) != set(UNIVERSE):
        raise ValueError("exact full8 universe required")
    values = {symbol: _finite(returns_12[symbol]) for symbol in UNIVERSE}
    if any(value < -1 for value in values.values()):
        raise ValueError("invalid asset price return")
    center = _quantile(sorted(values.values()), 0.5)
    return {symbol: _finite(value - center) for symbol, value in values.items()}


def fit_origin(
    rows: Iterable[Observation], *, deploy_origin_ms: int,
    implementation_sha256: str,
) -> dict:
    """Fit supplied observations only; not a market-data loader or authorized run.

    Complete eligible origins are equally weighted. Ineligible label outcomes
    are not examined or hashed. Raw OHLCV authenticity requires a separate
    builder/manifest audit; a stamped observation alone cannot prove it.
    """
    if type(deploy_origin_ms) is not int or deploy_origin_ms < TRAIN_WINDOW_MS:
        raise ValueError("invalid deployment origin")
    if (not isinstance(implementation_sha256, str) or len(implementation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in implementation_sha256)):
        raise ValueError("implementation SHA256 required")
    cutoff = deploy_origin_ms - PURGE_BARS * BAR_MS
    window_start = deploy_origin_ms - TRAIN_WINDOW_MS
    eligible: dict[tuple[int, str], Observation] = {}
    for row in rows:
        if not isinstance(row, Observation):
            raise ValueError("Observation required")
        times = (row.origin_ms, row.feature_bar_start_ms, row.label_end_ms)
        if any(type(t) is not int for t in times):
            raise ValueError("integer timestamps required")
        if row.label_end_ms != row.origin_ms + HORIZON_MS:
            raise ValueError("complete three-day label required")
        if row.feature_bar_start_ms != row.origin_ms - BAR_MS:
            raise ValueError("prior closed bar required")
        if row.origin_ms < window_start or row.label_end_ms > cutoff:
            continue
        if row.origin_ms < RESERVED_END_MS and row.label_end_ms > RESERVED_START_MS:
            continue
        if (deploy_origin_ms - row.origin_ms) % HORIZON_MS:
            raise ValueError("nonoverlapping three-day training grid required")
        if row.symbol not in UNIVERSE:
            raise ValueError("unexpected symbol")
        _finite(row.relative_return)
        if _finite(row.gross_return_3d) < -1:
            raise ValueError("invalid gross price return")
        key = (row.origin_ms, row.symbol)
        if key in eligible:
            raise ValueError("duplicate origin/symbol")
        eligible[key] = row
    ordered = [eligible[key] for key in sorted(eligible)]
    counts = Counter(row.origin_ms for row in ordered)
    if any(count != len(UNIVERSE) for count in counts.values()):
        raise ValueError("incomplete training origin: NoTrade required")
    n = len(ordered)
    cluster_weights = [count / n for count in counts.values()] if n else []
    origin_ess = 1 / fsum(w * w for w in cluster_weights) if n else 0.0
    records = [[r.origin_ms, r.feature_bar_start_ms, r.label_end_ms, r.symbol,
                float(r.relative_return), float(r.gross_return_3d)] for r in ordered]
    model = {
        "schema": SCHEMA, "research_only": True, "execution_allowed": False,
        "implementation_sha256": implementation_sha256,
        "deploy_origin_ms": deploy_origin_ms, "train_cutoff_exclusive_ms": cutoff,
        "train_window_start_ms": window_start, "purge_bars": PURGE_BARS,
        "eligible_rows": n, "distinct_training_origins": len(counts),
        "origin_weight_ess": origin_ess, "row_weight_ess": float(n),
        "train_rows_sha256": _digest(records),
        "folds_sha256": _digest({"origins": sorted(counts), "cutoff_ms": cutoff,
                                 "reserved": [RESERVED_START_MS, RESERVED_END_MS]}),
        "weights_sha256": _digest([[r.origin_ms, r.symbol, 1 / n] for r in ordered]),
        "target": "next_complete_3d_gross_return", "ridge_alpha": 0.5,
        "negative_outcomes": sum(r.gross_return_3d < 0 for r in ordered),
        "scaler": None, "scaler_sha256": _digest(None),
        "intercept": None, "slope": None, "fallback": None,
    }
    if len(counts) < 60 or origin_ess < 40:
        model["fallback"] = "insufficient_temporal_support"
    else:
        values = sorted(float(r.relative_return) for r in ordered)
        median = _quantile(values, 0.5)
        iqr = _finite(_quantile(values, 0.75) - _quantile(values, 0.25))
        if iqr <= 0:
            model["fallback"] = "zero_IQR_NoTrade"
        else:
            scaler = {"median": median, "iqr": iqr, "clip": [-8.0, 8.0]}
            x = [max(-8.0, min(8.0, (r.relative_return - median) / iqr)) for r in ordered]
            y = [float(r.gross_return_3d) for r in ordered]
            x_mean, y_mean = fsum(x) / n, fsum(y) / n
            variance = fsum((v - x_mean) ** 2 for v in x) / n
            covariance = fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / n
            slope = min(0.0, covariance / (variance + 0.5))
            model.update(scaler=scaler, scaler_sha256=_digest(scaler),
                         slope=_finite(slope), intercept=_finite(y_mean - slope * x_mean))
    model["model_sha256"] = _digest(model)
    return model


def predict_gross(model: Mapping, relative_return: float, *, feature_bar_start_ms: int) -> float | None:
    """Pure forecast, not a target/order generator. Reject altered model payloads."""
    if model.get("schema") != SCHEMA or model.get("execution_allowed") is not False:
        raise ValueError("research model required")
    payload = {key: value for key, value in model.items() if key != "model_sha256"}
    if _digest(payload) != model.get("model_sha256"):
        raise ValueError("model SHA mismatch")
    if (type(feature_bar_start_ms) is not int
            or feature_bar_start_ms != model["deploy_origin_ms"] - BAR_MS):
        raise ValueError("deployment prior closed bar required")
    value = _finite(relative_return)
    if model["fallback"] is not None:
        return None
    scaler = model["scaler"]
    scaled = max(-8.0, min(8.0, (value - scaler["median"]) / scaler["iqr"]))
    return _finite(model["intercept"] + model["slope"] * scaled)
