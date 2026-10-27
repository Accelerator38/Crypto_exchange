from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ACTIVATION_LABELS = (
    "compression_breakout",
    "fake_breakout",
    "dead_range",
)


def _true_range(group: pd.DataFrame) -> pd.Series:
    previous_close = group["close"].shift(1)
    return pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - previous_close).abs(),
            (group["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def validate_neutral_subregime_params(params: dict[str, Any]) -> None:
    integer_fields = ("fast_atr_bars", "slow_atr_bars", "range_bars", "volume_bars")
    for field in integer_fields:
        if int(params.get(field, 0)) < 2:
            raise ValueError(f"{field} must be at least 2")
    if int(params["fast_atr_bars"]) >= int(params["slow_atr_bars"]):
        raise ValueError("fast ATR window must be shorter than slow ATR window")
    bounded = (
        "compression_ratio_max",
        "confirmation_volume_ratio_min",
        "confirmation_body_ratio_min",
        "confirmation_rejection_wick_max",
        "fake_body_ratio_max",
        "fake_rejection_wick_min",
        "dead_compression_ratio_max",
        "dead_volume_ratio_max",
    )
    for field in bounded:
        value = float(params.get(field, 0.0))
        if value <= 0:
            raise ValueError(f"{field} must be positive")
    if float(params["confirmation_rejection_wick_max"]) >= 1.0:
        raise ValueError("confirmation wick maximum must be below one")
    if float(params["confirmation_body_ratio_min"]) >= 1.0:
        raise ValueError("confirmation body minimum must be below one")
    if float(params["fake_body_ratio_max"]) >= 1.0:
        raise ValueError("fake body maximum must be below one")
    if float(params["fake_rejection_wick_min"]) >= 1.0:
        raise ValueError("fake wick minimum must be below one")


def classify_neutral_activations(
    frame: pd.DataFrame,
    source_target: pd.Series,
    market_regime: pd.Series,
    *,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Classify new source-target activations using completed-bar information only."""
    if len(frame) != len(source_target):
        raise ValueError("frame and source target lengths differ")
    validate_neutral_subregime_params(params)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if not required <= set(frame.columns):
        raise ValueError(f"frame columns missing: {sorted(required - set(frame.columns))}")

    result = pd.DataFrame(index=frame.index)
    result["activation_label"] = "no_activation"
    numeric_columns = (
        "compression_ratio",
        "volume_ratio",
        "body_ratio",
        "rejection_wick_ratio",
        "prior_range_width_pct",
    )
    for column in numeric_columns:
        result[column] = np.nan
    result["source_direction"] = 0
    result["decision_market_regime"] = "unknown"

    fast_bars = int(params["fast_atr_bars"])
    slow_bars = int(params["slow_atr_bars"])
    range_bars = int(params["range_bars"])
    volume_bars = int(params["volume_bars"])

    for _, raw_group in frame.groupby("symbol", sort=True):
        group = raw_group.sort_values("timestamp", kind="stable").copy()
        target = source_target.loc[group.index].astype("int8")
        previous_target = target.shift(1).fillna(0).astype("int8")
        activation = target.ne(0) & target.ne(previous_target)
        true_range = _true_range(group)
        fast_atr = true_range.shift(1).rolling(fast_bars, min_periods=fast_bars).mean()
        slow_atr = true_range.shift(1).rolling(slow_bars, min_periods=slow_bars).mean()
        compression_ratio = fast_atr / slow_atr.replace(0.0, np.nan)
        prior_high = group["high"].shift(1).rolling(range_bars, min_periods=range_bars).max()
        prior_low = group["low"].shift(1).rolling(range_bars, min_periods=range_bars).min()
        prior_mid = (prior_high + prior_low) / 2.0
        prior_range_width_pct = (prior_high - prior_low) / prior_mid.replace(0.0, np.nan)
        volume_baseline = (
            group["volume"].shift(1).rolling(volume_bars, min_periods=volume_bars).median()
        )
        volume_ratio = group["volume"] / volume_baseline.replace(0.0, np.nan)
        candle_range = (group["high"] - group["low"]).replace(0.0, np.nan)
        body_ratio = (group["close"] - group["open"]).abs() / candle_range
        long_direction = target.eq(1)
        short_direction = target.eq(-1)
        close_break = (long_direction & group["close"].gt(prior_high)) | (
            short_direction & group["close"].lt(prior_low)
        )
        breakout_attempt = (long_direction & group["high"].gt(prior_high)) | (
            short_direction & group["low"].lt(prior_low)
        )
        long_rejection = (group["high"] - group[["open", "close"]].max(axis=1)) / candle_range
        short_rejection = (group[["open", "close"]].min(axis=1) - group["low"]) / candle_range
        rejection_wick = pd.Series(
            np.where(long_direction, long_rejection, short_rejection),
            index=group.index,
            dtype="float64",
        )
        regime_code = group["timestamp"].map(market_regime).fillna(0).astype("int8")
        regime_label = regime_code.map({1: "bullish", -1: "bearish", 0: "neutral"})
        neutral_activation = activation & regime_code.eq(0)
        finite = (
            compression_ratio.notna()
            & prior_high.notna()
            & prior_low.notna()
            & volume_ratio.notna()
            & body_ratio.notna()
            & rejection_wick.notna()
        )

        confirmed = (
            neutral_activation
            & finite
            & compression_ratio.le(float(params["compression_ratio_max"]))
            & close_break
            & volume_ratio.ge(float(params["confirmation_volume_ratio_min"]))
            & body_ratio.ge(float(params["confirmation_body_ratio_min"]))
            & rejection_wick.le(float(params["confirmation_rejection_wick_max"]))
        )
        fake = (
            neutral_activation
            & finite
            & ~confirmed
            & breakout_attempt
            & (
                ~close_break
                | body_ratio.le(float(params["fake_body_ratio_max"]))
                | rejection_wick.ge(float(params["fake_rejection_wick_min"]))
                | volume_ratio.lt(float(params["confirmation_volume_ratio_min"]))
            )
        )
        dead = (
            neutral_activation
            & finite
            & ~confirmed
            & ~fake
            & ~close_break
            & compression_ratio.le(float(params["dead_compression_ratio_max"]))
            & volume_ratio.le(float(params["dead_volume_ratio_max"]))
        )
        other_neutral = neutral_activation & ~(confirmed | fake | dead)
        non_neutral = activation & ~regime_code.eq(0)
        insufficient = activation & regime_code.eq(0) & ~finite

        result.loc[group.index, "compression_ratio"] = compression_ratio.to_numpy()
        result.loc[group.index, "volume_ratio"] = volume_ratio.to_numpy()
        result.loc[group.index, "body_ratio"] = body_ratio.to_numpy()
        result.loc[group.index, "rejection_wick_ratio"] = rejection_wick.to_numpy()
        result.loc[group.index, "prior_range_width_pct"] = prior_range_width_pct.to_numpy()
        result.loc[group.index, "source_direction"] = target.to_numpy()
        result.loc[group.index, "decision_market_regime"] = regime_label.to_numpy()
        result.loc[confirmed.index[confirmed], "activation_label"] = "compression_breakout"
        result.loc[fake.index[fake], "activation_label"] = "fake_breakout"
        result.loc[dead.index[dead], "activation_label"] = "dead_range"
        result.loc[other_neutral.index[other_neutral], "activation_label"] = "other_neutral"
        result.loc[non_neutral.index[non_neutral], "activation_label"] = "non_neutral"
        result.loc[insufficient.index[insufficient], "activation_label"] = "insufficient_history"

    return result


def filter_target_by_activation_label(
    frame: pd.DataFrame,
    source_target: pd.Series,
    activation_labels: pd.Series,
    *,
    accepted_label: str,
) -> pd.Series:
    if accepted_label not in ACTIVATION_LABELS:
        raise ValueError(f"unsupported activation label: {accepted_label}")
    if len(frame) != len(source_target) or len(frame) != len(activation_labels):
        raise ValueError("frame, source target and activation label lengths differ")
    filtered = pd.Series(0, index=frame.index, dtype="int8")
    for _, raw_group in frame.groupby("symbol", sort=True):
        group = raw_group.sort_values("timestamp", kind="stable")
        active_target = 0
        previous_source = 0
        for index in group.index:
            current_source = int(source_target.loc[index])
            if current_source == 0:
                active_target = 0
            elif current_source != previous_source:
                active_target = (
                    current_source
                    if str(activation_labels.loc[index]) == accepted_label
                    else 0
                )
            filtered.loc[index] = active_target
            previous_source = current_source
    return filtered
