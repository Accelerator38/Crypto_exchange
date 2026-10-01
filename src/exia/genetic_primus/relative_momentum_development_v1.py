"""One fixed, causal 3-day cross-sectional long-only research baseline.

This is a prerequisite check for a new economic signal, not a genetic search
and not a deployable strategy. All tuning choices are fixed in the contract.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

MODEL_SPEC = {
    "schema": "exia.genetic_primus.relative_momentum_base/1",
    "lookback_bars": 18,
    "rank_top_k": 2,
    "minimum_prior_return": 0.0,
    "direction": "long_only",
    "tie_break": "symbol_ascending",
    "rebalance": "origin_3d_only",
}
MODEL_SHA256 = hashlib.sha256(
    json.dumps(MODEL_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def select_at_origin(panel: pd.DataFrame, origin_ms: int,
                     symbols: tuple[str, ...], timeframe_ms: int) -> tuple[str, ...]:
    """Use only two fully closed candles separated by exactly 18 bars."""
    if timeframe_ms <= 0 or not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("invalid fixed-universe origin")
    last = origin_ms - timeframe_ms
    first = last - MODEL_SPEC["lookback_bars"] * timeframe_ms
    rows = panel.loc[panel.timestamp.isin((first, last)) & panel.symbol.isin(symbols),
                     ["timestamp", "symbol", "close"]]
    if len(rows) != 2 * len(symbols) or rows.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("missing or duplicate causal ranking candle")
    close = rows.pivot(index="symbol", columns="timestamp", values="close")
    close = close.reindex(index=symbols, columns=[first, last])
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid causal ranking price")
    returns = values[:, 1] / values[:, 0] - 1.0
    ranked = sorted(zip(symbols, returns), key=lambda item: (-item[1], item[0]))
    return tuple(symbol for symbol, value in ranked[:MODEL_SPEC["rank_top_k"]]
                 if value > MODEL_SPEC["minimum_prior_return"])


def origin_schedule(start_ms: int, end_ms: int, timeframe_ms: int,
                    symbols: tuple[str, ...], selected: tuple[str, ...]) -> pd.DataFrame:
    if end_ms <= start_ms or (end_ms - start_ms) != 3 * 86_400_000:
        raise ValueError("one three-day origin required")
    if len(selected) > MODEL_SPEC["rank_top_k"] or len(set(selected)) != len(selected):
        raise ValueError("invalid ranked selection")
    if not set(selected).issubset(symbols):
        raise ValueError("selection is outside fixed universe")
    times = np.arange(start_ms, end_ms, timeframe_ms, dtype=np.int64)
    if not len(times) or times[-1] + timeframe_ms != end_ms:
        raise ValueError("whole candles required")
    schedule = pd.MultiIndex.from_product(
        [times, symbols], names=["execution_timestamp", "symbol"]
    ).to_frame(index=False)
    schedule["signal_timestamp"] = start_ms
    schedule["target"] = schedule.symbol.isin(selected).astype(int)
    schedule["model_sha256"] = MODEL_SHA256
    return schedule
