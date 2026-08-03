from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _events_to_target(
    *,
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: pd.Series,
    short_exit: pd.Series,
) -> pd.Series:
    state = 0
    values = np.zeros(len(long_entry), dtype=np.int8)
    for index in range(len(values)):
        if state == 1 and bool(long_exit.iloc[index]):
            state = 0
        elif state == -1 and bool(short_exit.iloc[index]):
            state = 0
        if bool(long_entry.iloc[index]):
            state = 1
        elif bool(short_entry.iloc[index]):
            state = -1
        values[index] = state
    return pd.Series(values, index=long_entry.index, dtype="int8")


def _ema_trend(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 48))
    threshold_bps = float(params.get("threshold_bps", 0.0))
    fast_line = frame["close"].ewm(span=fast, adjust=False).mean()
    slow_line = frame["close"].ewm(span=slow, adjust=False).mean()
    distance_bps = (fast_line / slow_line - 1.0) * 10_000.0
    return pd.Series(
        np.select(
            [distance_bps > threshold_bps, distance_bps < -threshold_bps],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )


def _donchian(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 20))
    high = frame["high"].shift(1).rolling(window, min_periods=window).max()
    low = frame["low"].shift(1).rolling(window, min_periods=window).min()
    mid = (high + low) / 2.0
    return _events_to_target(
        long_entry=frame["close"] > high,
        short_entry=frame["close"] < low,
        long_exit=frame["close"] < mid,
        short_exit=frame["close"] > mid,
    )


def _mean_reversion(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 24))
    entry_z = float(params.get("entry_z", 2.0))
    exit_z = float(params.get("exit_z", 0.25))
    mean = frame["close"].rolling(window, min_periods=window).mean()
    std = frame["close"].rolling(window, min_periods=window).std(ddof=0)
    zscore = (frame["close"] - mean) / std.replace(0, np.nan)
    return _events_to_target(
        long_entry=zscore < -entry_z,
        short_entry=zscore > entry_z,
        long_exit=zscore >= -exit_z,
        short_exit=zscore <= exit_z,
    )


def _vol_compression(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    threshold = float(params.get("compression_threshold", 0.65))
    breakout_window = int(params.get("breakout_window", 20))
    fast = frame["close"].ewm(span=12, adjust=False).mean()
    slow = frame["close"].ewm(span=48, adjust=False).mean()
    log_return = np.log(frame["close"]).diff()
    short_vol = log_return.rolling(24, min_periods=12).std(ddof=0)
    long_vol = log_return.rolling(96, min_periods=48).std(ddof=0)
    compressed = (short_vol / long_vol.replace(0, np.nan)).shift(1) < threshold
    high = (
        frame["high"].shift(1).rolling(breakout_window, min_periods=breakout_window).max()
    )
    low = frame["low"].shift(1).rolling(breakout_window, min_periods=breakout_window).min()
    return _events_to_target(
        long_entry=compressed & (frame["close"] > high),
        short_entry=compressed & (frame["close"] < low),
        long_exit=frame["close"] < fast,
        short_exit=frame["close"] > fast,
    ).where((slow > 0) & (fast > 0), 0)


def _regime_pullback(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    fast_span = int(params.get("fast", 12))
    slow_span = int(params.get("slow", 48))
    pullback_bps = float(params.get("pullback_bps", 15.0))
    fast = frame["close"].ewm(span=fast_span, adjust=False).mean()
    slow = frame["close"].ewm(span=slow_span, adjust=False).mean()
    distance_fast_bps = (frame["close"] / fast - 1.0) * 10_000.0
    return _events_to_target(
        long_entry=(fast > slow)
        & (frame["close"] > slow)
        & (distance_fast_bps < -pullback_bps),
        short_entry=(fast < slow)
        & (frame["close"] < slow)
        & (distance_fast_bps > pullback_bps),
        long_exit=(frame["close"] < slow) | (distance_fast_bps > pullback_bps),
        short_exit=(frame["close"] > slow) | (distance_fast_bps < -pullback_bps),
    )


def _candle_momentum(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    bars = int(params.get("bars", 3))
    threshold_bps = float(params.get("threshold_bps", 25.0))
    move_bps = frame["close"].pct_change(bars) * 10_000.0
    return pd.Series(
        np.select(
            [move_bps > threshold_bps, move_bps < -threshold_bps],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )


def _build_group_signal(
    frame: pd.DataFrame,
    *,
    kind: str,
    params: Mapping[str, object],
) -> pd.Series:
    if kind == "flat":
        return pd.Series(0, index=frame.index, dtype="int8")
    if kind == "long_only":
        return pd.Series(1, index=frame.index, dtype="int8")
    if kind == "ema_trend":
        return _ema_trend(frame, params)
    if kind == "donchian":
        return _donchian(frame, params)
    if kind == "mean_reversion":
        return _mean_reversion(frame, params)
    if kind == "vol_compression":
        return _vol_compression(frame, params)
    if kind == "regime_pullback":
        return _regime_pullback(frame, params)
    if kind == "candle_momentum":
        return _candle_momentum(frame, params)
    raise ValueError(f"unknown strategy kind: {kind}")


def build_signal(frame: pd.DataFrame, strategy: Mapping[str, object]) -> pd.Series:
    """Build close-time target positions without reading future bars."""
    kind = str(strategy["kind"])
    params = strategy.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("strategy params must be a mapping")
    signal = pd.Series(0, index=frame.index, dtype="int8")
    for _, group in frame.groupby("symbol", sort=True):
        ordered = group.sort_values("timestamp", kind="stable")
        signal.loc[ordered.index] = _build_group_signal(
            ordered,
            kind=kind,
            params=params,
        )
    values = set(int(value) for value in signal.dropna().unique())
    if not values.issubset({-1, 0, 1}):
        raise ValueError(f"invalid signal values: {sorted(values)}")
    return signal
