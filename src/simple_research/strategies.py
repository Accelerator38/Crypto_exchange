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
    fast_span = int(params.get("fast_span", 12))
    slow_span = int(params.get("slow_span", 48))
    short_vol_window = int(params.get("short_vol_window", 24))
    long_vol_window = int(params.get("long_vol_window", 96))
    short_vol_min_periods = int(params.get("short_vol_min_periods", 12))
    long_vol_min_periods = int(params.get("long_vol_min_periods", 48))
    fast = frame["close"].ewm(span=fast_span, adjust=False).mean()
    slow = frame["close"].ewm(span=slow_span, adjust=False).mean()
    log_return = np.log(frame["close"]).diff()
    short_vol = log_return.rolling(
        short_vol_window,
        min_periods=min(short_vol_min_periods, short_vol_window),
    ).std(ddof=0)
    long_vol = log_return.rolling(
        long_vol_window,
        min_periods=min(long_vol_min_periods, long_vol_window),
    ).std(ddof=0)
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


def build_market_regime(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    """Classify each completed panel bar as bullish, bearish, or neutral."""
    lookback = int(params.get("market_lookback", 12))
    threshold_bps = float(params.get("market_threshold_bps", 100.0))
    breadth_threshold = float(params.get("market_breadth", 0.625))
    minimum_symbols = int(params.get("market_minimum_symbols", 6))
    confirmation_bars = int(params.get("market_confirmation_bars", 2))
    if lookback <= 0 or confirmation_bars <= 0:
        raise ValueError("market regime windows must be positive")
    if not 0.5 < breadth_threshold <= 1.0:
        raise ValueError("market_breadth must be in (0.5, 1.0]")

    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    ordered["market_return_bps"] = (
        ordered.groupby("symbol", sort=True, observed=True)["close"]
        .pct_change(lookback)
        .mul(10_000.0)
    )
    ordered["market_positive"] = (ordered["market_return_bps"] > 0).where(
        ordered["market_return_bps"].notna()
    )
    snapshot = ordered.groupby("timestamp", sort=True, observed=True).agg(
        median_return_bps=("market_return_bps", "median"),
        positive_share=("market_positive", "mean"),
        samples=("market_return_bps", "count"),
    )
    valid = snapshot["samples"] >= minimum_symbols
    raw = pd.Series(0, index=snapshot.index, dtype="int8")
    raw.loc[
        valid
        & (snapshot["median_return_bps"] > threshold_bps)
        & (snapshot["positive_share"] >= breadth_threshold)
    ] = 1
    raw.loc[
        valid
        & (snapshot["median_return_bps"] < -threshold_bps)
        & (snapshot["positive_share"] <= 1.0 - breadth_threshold)
    ] = -1
    confirmed = raw.copy()
    for offset in range(1, confirmation_bars):
        confirmed = confirmed.where(raw.eq(raw.shift(offset)), 0)
    return confirmed.astype("int8")


def _apply_direction(signal: pd.Series, params: Mapping[str, object]) -> pd.Series:
    direction = str(params.get("direction", "both")).strip().lower()
    if direction == "both":
        return signal.astype("int8")
    if direction == "long":
        return signal.where(signal >= 0, 0).astype("int8")
    if direction == "short":
        return signal.where(signal <= 0, 0).astype("int8")
    raise ValueError(f"unknown direction: {direction}")


def _ema_market_quorum(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    market = build_market_regime(frame, params)
    signal = pd.Series(0, index=frame.index, dtype="int8")
    fast_span = int(params.get("fast", 12))
    slow_span = int(params.get("slow", 48))
    threshold_bps = float(params.get("threshold_bps", 20.0))
    slope_bars = int(params.get("slope_bars", 3))
    slope_threshold_bps = float(params.get("slope_threshold_bps", 10.0))
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        fast = ordered["close"].ewm(span=fast_span, adjust=False).mean()
        slow = ordered["close"].ewm(span=slow_span, adjust=False).mean()
        distance = (fast / slow - 1.0) * 10_000.0
        slope = (slow / slow.shift(slope_bars) - 1.0) * 10_000.0
        regime = ordered["timestamp"].map(market).fillna(0).astype("int8")
        values = pd.Series(
            np.select(
                [
                    (regime == 1)
                    & (distance > threshold_bps)
                    & (slope > slope_threshold_bps),
                    (regime == -1)
                    & (distance < -threshold_bps)
                    & (slope < -slope_threshold_bps),
                ],
                [1, -1],
                default=0,
            ),
            index=ordered.index,
            dtype="int8",
        )
        signal.loc[ordered.index] = _apply_direction(values, params)
    return signal


def _donchian_market_quorum(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    market = build_market_regime(frame, params)
    signal = pd.Series(0, index=frame.index, dtype="int8")
    window = int(params.get("window", 20))
    fast_span = int(params.get("fast", 12))
    slow_span = int(params.get("slow", 48))
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        local = _donchian(ordered, {"window": window})
        fast = ordered["close"].ewm(span=fast_span, adjust=False).mean()
        slow = ordered["close"].ewm(span=slow_span, adjust=False).mean()
        regime = ordered["timestamp"].map(market).fillna(0).astype("int8")
        gated = local.where(
            ((local == 1) & (fast > slow) & (regime == 1))
            | ((local == -1) & (fast < slow) & (regime == -1)),
            0,
        ).astype("int8")
        signal.loc[ordered.index] = _apply_direction(gated, params)
    return signal


def _relative_momentum_market_quorum(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    lookback = int(params.get("lookback", 12))
    rank_threshold = float(params.get("rank_threshold", 0.75))
    if not 0.5 < rank_threshold < 1.0:
        raise ValueError("rank_threshold must be in (0.5, 1.0)")
    market = build_market_regime(frame, params)
    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    ordered["momentum"] = ordered.groupby(
        "symbol", sort=True, observed=True
    )["close"].pct_change(lookback)
    ordered["momentum_rank"] = ordered.groupby(
        "timestamp", sort=True, observed=True
    )["momentum"].rank(method="average", pct=True)
    regime = ordered["timestamp"].map(market).fillna(0).astype("int8")
    values = pd.Series(
        np.select(
            [
                (regime == 1) & (ordered["momentum_rank"] >= rank_threshold),
                (regime == -1)
                & (ordered["momentum_rank"] <= 1.0 - rank_threshold),
            ],
            [1, -1],
            default=0,
        ),
        index=ordered.index,
        dtype="int8",
    )
    values = _apply_direction(values, params)
    return values.reindex(frame.index).fillna(0).astype("int8")


def _fresh_long_activation(frame: pd.DataFrame, state: pd.Series) -> pd.Series:
    events = pd.Series(0, index=frame.index, dtype="int8")
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        local = state.loc[ordered.index].astype("int8")
        events.loc[ordered.index] = ((local == 1) & (local.shift(1).fillna(0) != 1)).astype(
            "int8"
        )
    return events


def _ema_activation_event(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    """Emit one event when the sealed long EMA/quorum state becomes active."""
    state = _ema_market_quorum(frame, {**dict(params), "direction": "long"})
    return _fresh_long_activation(frame, state)


def _market_regime_transition_event(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    """Enter locally aligned symbols only on a fresh broad bullish transition."""
    market = build_market_regime(frame, params)
    transition = (market == 1) & (market.shift(1).fillna(0) != 1)
    signal = pd.Series(0, index=frame.index, dtype="int8")
    fast_span = int(params.get("fast", 12))
    slow_span = int(params.get("slow", 48))
    slope_bars = int(params.get("slope_bars", 3))
    slope_threshold_bps = float(params.get("slope_threshold_bps", 10.0))
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        fast = ordered["close"].ewm(span=fast_span, adjust=False).mean()
        slow = ordered["close"].ewm(span=slow_span, adjust=False).mean()
        slope = (slow / slow.shift(slope_bars) - 1.0) * 10_000.0
        local_transition = ordered["timestamp"].map(transition).fillna(False).astype(bool)
        signal.loc[ordered.index] = (
            local_transition
            & (ordered["close"] > fast)
            & (fast > slow)
            & (slope > slope_threshold_bps)
        ).astype("int8")
    return signal


def _cross_sectional_breakout_event(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    """Enter a fresh local breakout only in the strongest bullish full8 quartile."""
    market = build_market_regime(frame, params)
    window = int(params.get("breakout_window", 20))
    momentum_lookback = int(params.get("momentum_lookback", 12))
    rank_threshold = float(params.get("rank_threshold", 0.75))
    if window < 2 or momentum_lookback < 1:
        raise ValueError("breakout and momentum windows must be positive")
    if not 0.5 < rank_threshold < 1.0:
        raise ValueError("rank_threshold must be in (0.5, 1.0)")
    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    ordered["momentum"] = ordered.groupby(
        "symbol", sort=True, observed=True
    )["close"].pct_change(momentum_lookback)
    ordered["momentum_rank"] = ordered.groupby(
        "timestamp", sort=True, observed=True
    )["momentum"].rank(method="average", pct=True)
    ordered["prior_high"] = ordered.groupby(
        "symbol", sort=True, observed=True
    )["high"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).max()
    )
    bullish = ordered["timestamp"].map(market).fillna(0).eq(1)
    breakout = (
        bullish
        & (ordered["momentum_rank"] >= rank_threshold)
        & (ordered["close"] > ordered["prior_high"])
    )
    prior_breakout = (
        breakout.groupby(ordered["symbol"], sort=True).shift(1).fillna(False).astype(bool)
    )
    events = (breakout & ~prior_breakout).astype("int8")
    return events.reindex(frame.index).fillna(0).astype("int8")


def _trend_pullback_continuation_event(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    """Enter on a completed-bar fast-EMA reclaim inside a broad bullish trend."""
    market = build_market_regime(frame, params)
    signal = pd.Series(0, index=frame.index, dtype="int8")
    fast_span = int(params.get("fast", 12))
    slow_span = int(params.get("slow", 48))
    slope_bars = int(params.get("slope_bars", 3))
    slope_threshold_bps = float(params.get("slope_threshold_bps", 10.0))
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        fast = ordered["close"].ewm(span=fast_span, adjust=False).mean()
        slow = ordered["close"].ewm(span=slow_span, adjust=False).mean()
        slope = (slow / slow.shift(slope_bars) - 1.0) * 10_000.0
        bullish = ordered["timestamp"].map(market).fillna(0).eq(1)
        reclaim = (ordered["close"].shift(1) <= fast.shift(1)) & (
            ordered["close"] > fast
        )
        signal.loc[ordered.index] = (
            bullish & reclaim & (fast > slow) & (slope > slope_threshold_bps)
        ).astype("int8")
    return signal


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
    if kind == "ema_market_quorum":
        return _ema_market_quorum(frame, params)
    if kind == "donchian_market_quorum":
        return _donchian_market_quorum(frame, params)
    if kind == "relative_momentum_market_quorum":
        return _relative_momentum_market_quorum(frame, params)
    if kind == "ema_activation_event":
        return _ema_activation_event(frame, params)
    if kind == "market_regime_transition_event":
        return _market_regime_transition_event(frame, params)
    if kind == "cross_sectional_breakout_event":
        return _cross_sectional_breakout_event(frame, params)
    if kind == "trend_pullback_continuation_event":
        return _trend_pullback_continuation_event(frame, params)
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
