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


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _keltner_breakout(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    ema_span = int(params.get("ema_span", 20))
    atr_window = int(params.get("atr_window", 14))
    atr_multiplier = float(params.get("atr_multiplier", 1.8))
    center = frame["close"].ewm(span=ema_span, adjust=False).mean()
    atr = _true_range(frame).rolling(atr_window, min_periods=atr_window).mean()
    upper = center + atr_multiplier * atr
    lower = center - atr_multiplier * atr
    return _events_to_target(
        long_entry=frame["close"] > upper,
        short_entry=frame["close"] < lower,
        long_exit=frame["close"] < center,
        short_exit=frame["close"] > center,
    )


def _dual_thrust(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 12))
    coefficient = float(params.get("coefficient", 0.55))
    prior_high = frame["high"].shift(1).rolling(window, min_periods=window).max()
    prior_low = frame["low"].shift(1).rolling(window, min_periods=window).min()
    width = prior_high - prior_low
    upper = frame["open"] + coefficient * width
    lower = frame["open"] - coefficient * width
    midpoint = (prior_high + prior_low) / 2.0
    return _events_to_target(
        long_entry=frame["close"] > upper,
        short_entry=frame["close"] < lower,
        long_exit=frame["close"] < midpoint,
        short_exit=frame["close"] > midpoint,
    )


def _rsi_failure_swing(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 14))
    lower = float(params.get("lower", 30.0))
    upper = float(params.get("upper", 70.0))
    delta = frame["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / window, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    return _events_to_target(
        long_entry=(rsi.shift(1) <= lower) & (rsi > lower),
        short_entry=(rsi.shift(1) >= upper) & (rsi < upper),
        long_exit=rsi >= 50.0,
        short_exit=rsi <= 50.0,
    )


def _volume_price_divergence(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    lookback = int(params.get("lookback", 12))
    price_threshold_bps = float(params.get("price_threshold_bps", 80.0))
    signed_volume = np.sign(frame["close"].diff()).fillna(0.0) * frame["volume"]
    obv = signed_volume.cumsum()
    price_move = frame["close"].pct_change(lookback) * 10_000.0
    obv_move = obv.diff(lookback)
    center = frame["close"].ewm(span=max(4, lookback), adjust=False).mean()
    return _events_to_target(
        long_entry=(price_move < -price_threshold_bps) & (obv_move > 0),
        short_entry=(price_move > price_threshold_bps) & (obv_move < 0),
        long_exit=frame["close"] >= center,
        short_exit=frame["close"] <= center,
    )


def _efficiency_ratio_trend(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 16))
    minimum_efficiency = float(params.get("minimum_efficiency", 0.35))
    minimum_move_bps = float(params.get("minimum_move_bps", 50.0))
    change = frame["close"].diff(window)
    path = frame["close"].diff().abs().rolling(window, min_periods=window).sum()
    efficiency = change.abs() / path.replace(0, np.nan)
    move_bps = frame["close"].pct_change(window) * 10_000.0
    return pd.Series(
        np.select(
            [
                (efficiency >= minimum_efficiency) & (move_bps > minimum_move_bps),
                (efficiency >= minimum_efficiency) & (move_bps < -minimum_move_bps),
            ],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )


def _volatility_regime_switch(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    short_window = int(params.get("short_window", 6))
    long_window = int(params.get("long_window", 30))
    trend_lookback = int(params.get("trend_lookback", 6))
    z_window = int(params.get("z_window", 20))
    z_entry = float(params.get("z_entry", 1.5))
    returns = np.log(frame["close"]).diff()
    short_vol = returns.rolling(short_window, min_periods=short_window).std(ddof=0)
    long_vol = returns.rolling(long_window, min_periods=long_window).std(ddof=0)
    high_vol = short_vol > long_vol
    momentum = frame["close"].pct_change(trend_lookback)
    mean = frame["close"].rolling(z_window, min_periods=z_window).mean()
    std = frame["close"].rolling(z_window, min_periods=z_window).std(ddof=0)
    zscore = (frame["close"] - mean) / std.replace(0, np.nan)
    return pd.Series(
        np.select(
            [
                high_vol & (momentum > 0),
                high_vol & (momentum < 0),
                ~high_vol & (zscore < -z_entry),
                ~high_vol & (zscore > z_entry),
            ],
            [1, -1, 1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )


def _rolling_vwap_reversion(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    window = int(params.get("window", 24))
    entry_z = float(params.get("entry_z", 1.8))
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].replace(0, np.nan)
    vwap = (typical * volume).rolling(window, min_periods=window).sum() / volume.rolling(
        window, min_periods=window
    ).sum()
    deviation = frame["close"] - vwap
    scale = deviation.rolling(window, min_periods=window).std(ddof=0)
    zscore = deviation / scale.replace(0, np.nan)
    return _events_to_target(
        long_entry=zscore < -entry_z,
        short_entry=zscore > entry_z,
        long_exit=zscore >= 0,
        short_exit=zscore <= 0,
    )


def _candle_structure_breakout(frame: pd.DataFrame, params: Mapping[str, object]) -> pd.Series:
    volume_window = int(params.get("volume_window", 20))
    volume_multiplier = float(params.get("volume_multiplier", 1.25))
    body_fraction = float(params.get("body_fraction", 0.65))
    close_location = float(params.get("close_location", 0.80))
    span = (frame["high"] - frame["low"]).replace(0, np.nan)
    body = (frame["close"] - frame["open"]).abs() / span
    location = (frame["close"] - frame["low"]) / span
    volume_base = frame["volume"].shift(1).rolling(
        volume_window, min_periods=volume_window
    ).median()
    active_volume = frame["volume"] > volume_multiplier * volume_base
    center = frame["close"].ewm(span=12, adjust=False).mean()
    return _events_to_target(
        long_entry=active_volume & (body >= body_fraction) & (location >= close_location),
        short_entry=active_volume
        & (body >= body_fraction)
        & (location <= 1.0 - close_location),
        long_exit=frame["close"] < center,
        short_exit=frame["close"] > center,
    )


def _cross_sectional_residual_momentum(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    lookback = int(params.get("lookback", 12))
    rank_threshold = float(params.get("rank_threshold", 0.75))
    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    ordered["return"] = ordered.groupby("symbol", sort=True, observed=True)[
        "close"
    ].pct_change(lookback)
    market = ordered.groupby("timestamp", sort=True, observed=True)["return"].transform(
        "median"
    )
    residual = ordered["return"] - market
    rank = residual.groupby(ordered["timestamp"], sort=True).rank(method="average", pct=True)
    signal = pd.Series(
        np.select(
            [rank >= rank_threshold, rank <= 1.0 - rank_threshold],
            [1, -1],
            default=0,
        ),
        index=ordered.index,
        dtype="int8",
    )
    return signal.reindex(frame.index).fillna(0).astype("int8")


def _cross_sectional_short_term_reversal(
    frame: pd.DataFrame,
    params: Mapping[str, object],
) -> pd.Series:
    lookback = int(params.get("lookback", 2))
    rank_threshold = float(params.get("rank_threshold", 0.75))
    ordered = frame.sort_values(["symbol", "timestamp"], kind="stable").copy()
    move = ordered.groupby("symbol", sort=True, observed=True)["close"].pct_change(lookback)
    rank = move.groupby(ordered["timestamp"], sort=True).rank(method="average", pct=True)
    signal = pd.Series(
        np.select(
            [rank <= 1.0 - rank_threshold, rank >= rank_threshold],
            [1, -1],
            default=0,
        ),
        index=ordered.index,
        dtype="int8",
    )
    return signal.reindex(frame.index).fillna(0).astype("int8")


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
    if kind == "keltner_breakout":
        return _keltner_breakout(frame, params)
    if kind == "dual_thrust":
        return _dual_thrust(frame, params)
    if kind == "rsi_failure_swing":
        return _rsi_failure_swing(frame, params)
    if kind == "volume_price_divergence":
        return _volume_price_divergence(frame, params)
    if kind == "efficiency_ratio_trend":
        return _efficiency_ratio_trend(frame, params)
    if kind == "volatility_regime_switch":
        return _volatility_regime_switch(frame, params)
    if kind == "rolling_vwap_reversion":
        return _rolling_vwap_reversion(frame, params)
    if kind == "candle_structure_breakout":
        return _candle_structure_breakout(frame, params)
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
    if kind == "cross_sectional_residual_momentum":
        return _cross_sectional_residual_momentum(frame, params)
    if kind == "cross_sectional_short_term_reversal":
        return _cross_sectional_short_term_reversal(frame, params)
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
