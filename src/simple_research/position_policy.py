from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LEDGER_COLUMNS = (
    "symbol",
    "direction",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "holding_bars",
    "gross_bps",
    "fills",
    "exit_reason",
    "entry_atr",
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


def _optional_positive(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive when configured")
    return numeric


def validate_position_policy(params: dict[str, Any]) -> None:
    atr_period = int(params.get("atr_period", 0))
    if atr_period < 2:
        raise ValueError("atr_period must be at least 2")
    for name in (
        "hard_stop_atr",
        "breakeven_trigger_atr",
        "trailing_trigger_atr",
        "trailing_distance_atr",
        "no_progress_mfe_atr",
        "take_profit_atr",
    ):
        _optional_positive(params, name)
    for name in ("no_progress_bars", "max_holding_bars"):
        value = params.get(name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when configured")
    trailing_trigger = params.get("trailing_trigger_atr")
    trailing_distance = params.get("trailing_distance_atr")
    if (trailing_trigger is None) != (trailing_distance is None):
        raise ValueError("trailing trigger and distance must be configured together")
    no_progress_bars = params.get("no_progress_bars")
    no_progress_mfe = params.get("no_progress_mfe_atr")
    if (no_progress_bars is None) != (no_progress_mfe is None):
        raise ValueError("no-progress bars and MFE must be configured together")


def _exit_price_for_stop(
    *, direction: int, open_price: float, stop_price: float
) -> float:
    if direction == 1 and open_price <= stop_price:
        return open_price
    if direction == -1 and open_price >= stop_price:
        return open_price
    return stop_price


def _exit_price_for_target(
    *, direction: int, open_price: float, target_price: float
) -> float:
    if direction == 1 and open_price >= target_price:
        return open_price
    if direction == -1 and open_price <= target_price:
        return open_price
    return target_price


def simulate_ohlc_position_policy(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    start_timestamp: int,
    end_timestamp: int,
    policy: dict[str, Any],
    entry_mode: str = "target",
) -> pd.DataFrame:
    """Run one deterministic next-open policy with conservative OHLC fills."""
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    if start_timestamp >= end_timestamp:
        raise ValueError("start_timestamp must be before end_timestamp")
    if entry_mode not in {"target", "event"}:
        raise ValueError("entry_mode must be 'target' or 'event'")
    validate_position_policy(policy)
    source = frame.copy()
    source["signal"] = signal.astype("int8")
    rows: list[dict[str, Any]] = []
    atr_period = int(policy["atr_period"])
    hard_stop_atr = _optional_positive(policy, "hard_stop_atr")
    breakeven_trigger = _optional_positive(policy, "breakeven_trigger_atr")
    trailing_trigger = _optional_positive(policy, "trailing_trigger_atr")
    trailing_distance = _optional_positive(policy, "trailing_distance_atr")
    no_progress_bars = policy.get("no_progress_bars")
    no_progress_mfe = _optional_positive(policy, "no_progress_mfe_atr")
    max_holding_bars = policy.get("max_holding_bars")
    take_profit_atr = _optional_positive(policy, "take_profit_atr")
    needs_atr = any(
        value is not None
        for value in (
            hard_stop_atr,
            breakeven_trigger,
            trailing_trigger,
            no_progress_mfe,
            take_profit_atr,
        )
    )

    for symbol, raw_group in source.groupby("symbol", sort=True):
        group = raw_group.sort_values("timestamp", kind="stable").copy()
        group["desired"] = group["signal"].shift(1).fillna(0).astype("int8")
        group["entry_atr"] = (
            _true_range(group).rolling(atr_period, min_periods=atr_period).mean().shift(1)
        )
        group = group.loc[
            (group["timestamp"] >= start_timestamp)
            & (group["timestamp"] < end_timestamp)
        ].copy()
        if group.empty:
            continue

        position = 0
        entry_price = 0.0
        entry_timestamp = 0
        entry_atr = float("nan")
        bars_held = 0
        peak = float("nan")
        trough = float("nan")
        mfe_atr = 0.0

        def close_trade(timestamp: int, price: float, reason: str) -> None:
            nonlocal position, entry_price, entry_timestamp, entry_atr
            nonlocal bars_held, peak, trough, mfe_atr
            gross_bps = position * (price / entry_price - 1.0) * 10_000.0
            rows.append(
                {
                    "symbol": str(symbol),
                    "direction": position,
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": timestamp,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "holding_bars": bars_held,
                    "gross_bps": gross_bps,
                    "fills": 2,
                    "exit_reason": reason,
                    "entry_atr": entry_atr,
                }
            )
            position = 0
            entry_price = 0.0
            entry_timestamp = 0
            entry_atr = float("nan")
            bars_held = 0
            peak = float("nan")
            trough = float("nan")
            mfe_atr = 0.0

        for row in group.itertuples(index=False):
            timestamp = int(row.timestamp)
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            desired = int(row.desired)

            signal_requests_exit = entry_mode == "target" and desired != position
            opposite_event = entry_mode == "event" and desired == -position
            if position and (signal_requests_exit or opposite_event):
                close_trade(timestamp, open_price, "signal_exit")

            if position and max_holding_bars is not None and bars_held >= int(max_holding_bars):
                close_trade(timestamp, open_price, "max_holding")
            elif (
                position
                and no_progress_bars is not None
                and bars_held >= int(no_progress_bars)
                and mfe_atr < float(no_progress_mfe)
            ):
                close_trade(timestamp, open_price, "no_progress")

            candidate_atr = float(row.entry_atr)
            if position == 0 and desired in (-1, 1):
                if not needs_atr or (np.isfinite(candidate_atr) and candidate_atr > 0):
                    position = desired
                    entry_price = open_price
                    entry_timestamp = timestamp
                    entry_atr = candidate_atr
                    peak = open_price
                    trough = open_price
                    bars_held = 0
                    mfe_atr = 0.0

            if not position:
                continue

            stop_price: float | None = None
            if hard_stop_atr is not None:
                stop_price = entry_price - position * hard_stop_atr * entry_atr
            if breakeven_trigger is not None and mfe_atr >= breakeven_trigger:
                stop_price = (
                    max(stop_price, entry_price)
                    if position == 1 and stop_price is not None
                    else min(stop_price, entry_price)
                    if position == -1 and stop_price is not None
                    else entry_price
                )
            if trailing_trigger is not None and mfe_atr >= trailing_trigger:
                trailing_stop = (
                    peak - trailing_distance * entry_atr
                    if position == 1
                    else trough + trailing_distance * entry_atr
                )
                stop_price = (
                    max(stop_price, trailing_stop)
                    if position == 1 and stop_price is not None
                    else min(stop_price, trailing_stop)
                    if position == -1 and stop_price is not None
                    else trailing_stop
                )
            target_price = (
                entry_price + position * take_profit_atr * entry_atr
                if take_profit_atr is not None
                else None
            )
            stop_touched = bool(
                stop_price is not None
                and ((position == 1 and low_price <= stop_price) or (position == -1 and high_price >= stop_price))
            )
            target_touched = bool(
                target_price is not None
                and ((position == 1 and high_price >= target_price) or (position == -1 and low_price <= target_price))
            )
            if stop_touched:
                fill = _exit_price_for_stop(
                    direction=position, open_price=open_price, stop_price=float(stop_price)
                )
                close_trade(timestamp, fill, "stop")
                continue
            if target_touched:
                fill = _exit_price_for_target(
                    direction=position, open_price=open_price, target_price=float(target_price)
                )
                close_trade(timestamp, fill, "take_profit")
                continue

            if position == 1:
                peak = max(peak, high_price)
                mfe_atr = max(mfe_atr, (peak - entry_price) / entry_atr) if needs_atr else 0.0
            else:
                trough = min(trough, low_price)
                mfe_atr = max(mfe_atr, (entry_price - trough) / entry_atr) if needs_atr else 0.0
            bars_held += 1

        if position:
            final = group.iloc[-1]
            close_trade(int(final["timestamp"]), float(final["close"]), "split_end")

    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)
