"""Pure in-memory OHLCV builder. No loaders, clocks, runtime or network."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable

BAR_MS = 14_400_000
HORIZON_MS = 18 * BAR_MS
LOOKBACK_BARS = 12
RESERVED_START_MS = 1791158400000
RESERVED_END_MS = 1806710400000
UNIVERSE = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT",
            "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")


@dataclass(frozen=True)
class Candle:
    symbol: str
    open_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _number(value: object, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("finite numeric OHLCV required")
    result = float(value)
    if positive and result <= 0:
        raise ValueError("positive price required")
    return result


def _context(origin_ms: int, as_of_ms: int, implementation_sha256: str) -> None:
    if (type(origin_ms) is not int or origin_ms < 13 * BAR_MS
            or origin_ms % BAR_MS or type(as_of_ms) is not int or as_of_ms < origin_ms):
        raise ValueError("aligned historical origin and explicit as-of required")
    if (not isinstance(implementation_sha256, str) or len(implementation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in implementation_sha256)):
        raise ValueError("builder implementation SHA256 required")


def _index(candles: Iterable[Candle], start_ms: int, end_ms: int, as_of_ms: int) -> dict:
    result = {}
    for candle in candles:
        if not isinstance(candle, Candle) or type(candle.open_ms) is not int:
            raise ValueError("timestamped Candle required")
        # Do not examine price/outcome fields outside the requested causal window.
        if not start_ms <= candle.open_ms < end_ms:
            continue
        if candle.open_ms % BAR_MS or candle.open_ms + BAR_MS > as_of_ms:
            raise ValueError("aligned fully closed candle required")
        if candle.symbol not in UNIVERSE:
            raise ValueError("unexpected symbol inside requested window")
        key = (candle.open_ms, candle.symbol)
        if key in result:
            raise ValueError("duplicate candle inside requested window")
        o, h, low, close = [_number(v, positive=True) for v in
                            (candle.open, candle.high, candle.low, candle.close)]
        volume = _number(candle.volume)
        if low > min(o, close) or h < max(o, close) or low > h or volume < 0:
            raise ValueError("inconsistent OHLC or negative volume")
        result[key] = candle
    expected = {(t, symbol) for t in range(start_ms, end_ms, BAR_MS) for symbol in UNIVERSE}
    if set(result) != expected:
        raise ValueError("complete contiguous full8 window required: abstain")
    return result


def _records(index: dict, start_ms: int, end_ms: int) -> list:
    return [[t, symbol, float(c.open), float(c.high), float(c.low), float(c.close), float(c.volume)]
            for (t, symbol), c in sorted(index.items()) if start_ms <= t < end_ms]


def _features(index: dict, origin_ms: int, implementation_sha256: str) -> dict:
    first, last = origin_ms - 13 * BAR_MS, origin_ms - BAR_MS
    returns = {s: _number(index[(last, s)].close / index[(first, s)].close - 1)
               for s in UNIVERSE}
    ordered = sorted(returns.values())
    center = ordered[3] / 2 + ordered[4] / 2
    features = {s: _number(returns[s] - center) for s in UNIVERSE}
    payload = {
        "schema": "exia.genetic_primus.reversal_feature_bundle/1",
        "research_only": True, "execution_allowed": False,
        "implementation_sha256": implementation_sha256,
        "origin_ms": origin_ms, "feature_bar_start_ms": last,
        "first_feature_bar_start_ms": first, "latest_input_close_ms": origin_ms,
        "lookback_bars": LOOKBACK_BARS, "source_bars": 13 * len(UNIVERSE),
        "asset_returns_12": returns, "relative_features": features,
        "source_ohlcv_sha256": _digest(_records(index, first, origin_ms)),
    }
    payload["features_sha256"] = _digest(payload)
    return payload


def build_features(
    candles: Iterable[Candle], *, origin_ms: int, as_of_ms: int,
    implementation_sha256: str,
) -> dict:
    """Decision-time function: only thirteen pre-origin closed bars are inspected."""
    _context(origin_ms, as_of_ms, implementation_sha256)
    index = _index(candles, origin_ms - 13 * BAR_MS, origin_ms, as_of_ms)
    return _features(index, origin_ms, implementation_sha256)


def build_training_origin(
    candles: Iterable[Candle], *, origin_ms: int, as_of_ms: int,
    implementation_sha256: str,
) -> dict:
    """Historical labels only. Caller uses deployment-minus-purge as as_of_ms.

    No label is built before all eighteen forward bars close. Reserved EMA
    labels are forbidden even if data are supplied. Purge is a fitter/evaluator
    rule, not a justification for constructing unavailable labels here.
    """
    _context(origin_ms, as_of_ms, implementation_sha256)
    end = origin_ms + HORIZON_MS
    if end > as_of_ms:
        raise ValueError("three-day label not fully available at as-of")
    if origin_ms < RESERVED_END_MS and end > RESERVED_START_MS:
        raise ValueError("reserved EMA label interval forbidden")
    index = _index(candles, origin_ms - 13 * BAR_MS, end, as_of_ms)
    features = _features(index, origin_ms, implementation_sha256)
    rows = [{"origin_ms": origin_ms, "feature_bar_start_ms": origin_ms - BAR_MS,
             "label_end_ms": end, "symbol": symbol,
             "relative_return": features["relative_features"][symbol],
             "gross_return_3d": _number(index[(end-BAR_MS, symbol)].close /
                                        index[(origin_ms, symbol)].open - 1)}
            for symbol in UNIVERSE]
    payload = {
        "schema": "exia.genetic_primus.reversal_training_origin/1",
        "research_only": True, "execution_allowed": False,
        "implementation_sha256": implementation_sha256, "origin_ms": origin_ms,
        "label_end_ms": end, "as_of_ms": as_of_ms,
        "label_rule": "origin_open_to_last_3d_bar_close_not_close_to_close",
        "features": features, "observations": rows,
        "observations_sha256": _digest(rows),
        "label_ohlcv_sha256": _digest(_records(index, origin_ms, end)),
        "input_ohlcv_sha256": _digest(_records(index, origin_ms-13*BAR_MS, end)),
    }
    payload["training_origin_sha256"] = _digest(payload)
    return payload
