from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DIAGNOSTIC_COLUMNS = (
    "target",
    "state",
    "event",
    "rejection_reason",
    "setup_streak",
    "armed_age",
    "box_high",
    "box_low",
    "compression_ratio",
    "volume_ratio",
    "body_ratio",
    "rejection_wick_ratio",
    "decision_market_regime",
)


def validate_compression_transition_params(params: dict[str, Any]) -> None:
    integer_fields = (
        "fast_atr_bars",
        "slow_atr_bars",
        "setup_consecutive_bars",
        "box_bars",
        "max_armed_bars",
        "volume_bars",
        "cooldown_bars",
    )
    for field in integer_fields:
        if int(params.get(field, 0)) < 1:
            raise ValueError(f"{field} must be positive")
    if int(params["fast_atr_bars"]) >= int(params["slow_atr_bars"]):
        raise ValueError("fast ATR window must be shorter than slow ATR window")
    if int(params["box_bars"]) < int(params["setup_consecutive_bars"]):
        raise ValueError("box window cannot be shorter than the setup streak")
    for field in (
        "compression_ratio_max",
        "minimum_volume_ratio",
        "minimum_body_ratio",
        "maximum_rejection_wick_ratio",
    ):
        if float(params.get(field, 0.0)) <= 0.0:
            raise ValueError(f"{field} must be positive")
    if float(params["minimum_body_ratio"]) >= 1.0:
        raise ValueError("minimum body ratio must be below one")
    if float(params["maximum_rejection_wick_ratio"]) >= 1.0:
        raise ValueError("maximum rejection wick ratio must be below one")


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


def build_compression_transition_state(
    frame: pd.DataFrame,
    market_regime: pd.Series,
    *,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Build one causal neutral-compression state machine on completed bars."""
    validate_compression_transition_params(params)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if not required <= set(frame.columns):
        raise ValueError(f"frame columns missing: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("frame contains duplicate timestamp/symbol rows")

    result = pd.DataFrame(index=frame.index)
    result["target"] = pd.Series(0, index=frame.index, dtype="int8")
    result["state"] = pd.Series("idle", index=frame.index, dtype="string")
    result["event"] = pd.Series("none", index=frame.index, dtype="string")
    result["rejection_reason"] = pd.Series("", index=frame.index, dtype="string")
    result["setup_streak"] = pd.Series(0, index=frame.index, dtype="int16")
    result["armed_age"] = pd.Series(0, index=frame.index, dtype="int16")
    for column in (
        "box_high",
        "box_low",
        "compression_ratio",
        "volume_ratio",
        "body_ratio",
        "rejection_wick_ratio",
    ):
        result[column] = pd.Series(np.nan, index=frame.index, dtype="float64")
    result["decision_market_regime"] = pd.Series(
        "unknown", index=frame.index, dtype="string"
    )

    fast_bars = int(params["fast_atr_bars"])
    slow_bars = int(params["slow_atr_bars"])
    setup_required = int(params["setup_consecutive_bars"])
    box_bars = int(params["box_bars"])
    max_armed = int(params["max_armed_bars"])
    volume_bars = int(params["volume_bars"])
    cooldown_bars = int(params["cooldown_bars"])
    compression_max = float(params["compression_ratio_max"])
    volume_min = float(params["minimum_volume_ratio"])
    body_min = float(params["minimum_body_ratio"])
    wick_max = float(params["maximum_rejection_wick_ratio"])

    for _, raw_group in frame.groupby("symbol", sort=True):
        group = raw_group.sort_values("timestamp", kind="stable").copy()
        indexes = group.index.to_numpy()
        true_range = _true_range(group)
        fast_atr = true_range.rolling(fast_bars, min_periods=fast_bars).mean()
        slow_atr = true_range.rolling(slow_bars, min_periods=slow_bars).mean()
        compression = fast_atr / slow_atr.replace(0.0, np.nan)
        rolling_high = group["high"].rolling(box_bars, min_periods=box_bars).max()
        rolling_low = group["low"].rolling(box_bars, min_periods=box_bars).min()
        volume_baseline = (
            group["volume"].shift(1).rolling(volume_bars, min_periods=volume_bars).median()
        )
        volume_ratio = group["volume"] / volume_baseline.replace(0.0, np.nan)
        candle_range = (group["high"] - group["low"]).replace(0.0, np.nan)
        body_ratio = (group["close"] - group["open"]).abs() / candle_range
        regime_code = group["timestamp"].map(market_regime).fillna(0).astype("int8")
        regime_label = regime_code.map({1: "bullish", -1: "bearish", 0: "neutral"})

        open_values = group["open"].to_numpy(dtype="float64")
        high_values = group["high"].to_numpy(dtype="float64")
        low_values = group["low"].to_numpy(dtype="float64")
        close_values = group["close"].to_numpy(dtype="float64")
        compression_values = compression.to_numpy(dtype="float64")
        rolling_high_values = rolling_high.to_numpy(dtype="float64")
        rolling_low_values = rolling_low.to_numpy(dtype="float64")
        volume_ratio_values = volume_ratio.to_numpy(dtype="float64")
        body_ratio_values = body_ratio.to_numpy(dtype="float64")
        regime_values = regime_code.to_numpy(dtype="int8")
        regime_labels = regime_label.to_numpy(dtype=object)
        count = len(group)
        targets = np.zeros(count, dtype="int8")
        states = np.empty(count, dtype=object)
        events = np.full(count, "none", dtype=object)
        rejection_reasons = np.full(count, "", dtype=object)
        setup_streaks = np.zeros(count, dtype="int16")
        armed_ages = np.zeros(count, dtype="int16")
        box_highs = np.full(count, np.nan, dtype="float64")
        box_lows = np.full(count, np.nan, dtype="float64")
        rejection_wicks = np.full(count, np.nan, dtype="float64")

        machine_state = "idle"
        setup_streak = 0
        armed_age = 0
        cooldown_remaining = 0
        active_target = 0
        box_high = float("nan")
        box_low = float("nan")

        for position in range(count):
            open_price = float(open_values[position])
            high_price = float(high_values[position])
            low_price = float(low_values[position])
            close_price = float(close_values[position])
            current_compression = float(compression_values[position])
            current_volume_ratio = float(volume_ratio_values[position])
            current_body_ratio = float(body_ratio_values[position])
            current_regime = int(regime_values[position])
            finite_setup = bool(
                np.isfinite(current_compression)
                and np.isfinite(rolling_high_values[position])
                and np.isfinite(rolling_low_values[position])
            )
            setup_bar = bool(
                finite_setup
                and current_regime == 0
                and current_compression <= compression_max
            )
            event = "none"
            rejection_reason = ""
            rejection_wick = float("nan")

            if machine_state == "active":
                returned_inside = bool(
                    (active_target == 1 and close_price <= box_high)
                    or (active_target == -1 and close_price >= box_low)
                )
                if returned_inside:
                    active_target = 0
                    machine_state = "cooldown"
                    cooldown_remaining = cooldown_bars
                    event = "range_reentry_invalidation"
                targets[position] = active_target
            elif machine_state == "cooldown":
                targets[position] = 0
                cooldown_remaining -= 1
                if cooldown_remaining <= 0:
                    machine_state = "idle"
                    setup_streak = 0
                    event = "cooldown_complete"
            elif machine_state == "armed":
                targets[position] = 0
                armed_age += 1
                long_attempt = close_price > box_high
                short_attempt = close_price < box_low
                direction = 1 if long_attempt else -1 if short_attempt else 0
                if direction:
                    rejection_wick = (
                        (high_price - max(open_price, close_price)) / (high_price - low_price)
                        if direction == 1 and high_price > low_price
                        else (min(open_price, close_price) - low_price) / (high_price - low_price)
                        if direction == -1 and high_price > low_price
                        else float("nan")
                    )
                    reasons: list[str] = []
                    if not np.isfinite(current_volume_ratio) or current_volume_ratio < volume_min:
                        reasons.append("volume")
                    if not np.isfinite(current_body_ratio) or current_body_ratio < body_min:
                        reasons.append("body")
                    if not np.isfinite(rejection_wick) or rejection_wick > wick_max:
                        reasons.append("wick")
                    if (direction == 1 and current_regime == -1) or (
                        direction == -1 and current_regime == 1
                    ):
                        reasons.append("market_conflict")
                    if reasons:
                        machine_state = "cooldown"
                        cooldown_remaining = cooldown_bars
                        event = "breakout_rejected"
                        rejection_reason = ",".join(reasons)
                    else:
                        machine_state = "active"
                        active_target = direction
                        event = "breakout_entry"
                        targets[position] = direction
                elif armed_age >= max_armed:
                    machine_state = "cooldown"
                    cooldown_remaining = cooldown_bars
                    event = "armed_expired"
            else:
                targets[position] = 0
                setup_streak = setup_streak + 1 if setup_bar else 0
                if setup_streak >= setup_required:
                    box_high = float(rolling_high_values[position])
                    box_low = float(rolling_low_values[position])
                    machine_state = "armed"
                    armed_age = 0
                    event = "compression_armed"

            states[position] = machine_state
            events[position] = event
            rejection_reasons[position] = rejection_reason
            setup_streaks[position] = setup_streak
            armed_ages[position] = armed_age
            box_highs[position] = box_high
            box_lows[position] = box_low
            rejection_wicks[position] = rejection_wick

        result.loc[indexes, "target"] = targets
        result.loc[indexes, "state"] = states
        result.loc[indexes, "event"] = events
        result.loc[indexes, "rejection_reason"] = rejection_reasons
        result.loc[indexes, "setup_streak"] = setup_streaks
        result.loc[indexes, "armed_age"] = armed_ages
        result.loc[indexes, "box_high"] = box_highs
        result.loc[indexes, "box_low"] = box_lows
        result.loc[indexes, "compression_ratio"] = compression_values
        result.loc[indexes, "volume_ratio"] = volume_ratio_values
        result.loc[indexes, "body_ratio"] = body_ratio_values
        result.loc[indexes, "rejection_wick_ratio"] = rejection_wicks
        result.loc[indexes, "decision_market_regime"] = regime_labels

    return result.loc[:, DIAGNOSTIC_COLUMNS]
