from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMNS = (
    "symbol",
    "direction",
    "event_timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "horizon_bars",
    "entry_price",
    "exit_price",
    "gross_bps",
    "base_net_bps",
    "stress_net_bps",
    "mfe_bps",
    "mae_bps",
    "mfe_stress_surplus_bps",
    "capture_ratio",
)

DEGRADATION_POLICY_COLUMNS = (
    "symbol",
    "direction",
    "event_timestamp",
    "entry_timestamp",
    "horizon_bars",
    "continuation_exit_timestamp",
    "adaptive_exit_timestamp",
    "degradation_event_timestamp",
    "degradation_exit_triggered",
    "entry_price",
    "continuation_exit_price",
    "adaptive_exit_price",
    "continuation_gross_bps",
    "adaptive_gross_bps",
    "continuation_stress_net_bps",
    "adaptive_stress_net_bps",
    "adaptive_minus_continuation_bps",
)


def build_event_forward_labels(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    horizons_bars: Iterable[int],
    start_timestamp: int,
    end_timestamp: int,
    base_cost_bps: float,
    stress_cost_bps: float,
) -> pd.DataFrame:
    """Label completed-bar events using fixed next-open to future-open horizons."""
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    if start_timestamp >= end_timestamp:
        raise ValueError("start_timestamp must be before end_timestamp")
    horizons = sorted({int(value) for value in horizons_bars})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_bars must contain positive integers")
    if base_cost_bps <= 0 or stress_cost_bps < base_cost_bps:
        raise ValueError("cost floors must be positive and ordered")
    source = frame.copy()
    source["event"] = signal.astype("int8")
    rows: list[dict[str, object]] = []

    for symbol, raw_group in source.groupby("symbol", sort=True, observed=True):
        group = raw_group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        event_positions = np.flatnonzero(group["event"].to_numpy(dtype="int8") != 0)
        for event_position in event_positions:
            event_timestamp = int(group.iloc[event_position]["timestamp"])
            if not start_timestamp <= event_timestamp < end_timestamp:
                continue
            entry_position = event_position + 1
            if entry_position >= len(group):
                continue
            direction = int(group.iloc[event_position]["event"])
            entry_timestamp = int(group.iloc[entry_position]["timestamp"])
            if entry_timestamp < start_timestamp or entry_timestamp >= end_timestamp:
                continue
            entry_price = float(group.iloc[entry_position]["open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                raise ValueError(f"invalid entry price for {symbol}")
            for horizon in horizons:
                exit_position = entry_position + horizon
                if exit_position >= len(group):
                    continue
                exit_timestamp = int(group.iloc[exit_position]["timestamp"])
                if exit_timestamp >= end_timestamp:
                    continue
                exit_price = float(group.iloc[exit_position]["open"])
                path = group.iloc[entry_position:exit_position]
                if len(path) != horizon:
                    raise ValueError("forward path length differs from horizon")
                if direction == 1:
                    favorable = (path["high"].astype(float) / entry_price - 1.0) * 10_000.0
                    adverse = (1.0 - path["low"].astype(float) / entry_price) * 10_000.0
                elif direction == -1:
                    favorable = (1.0 - path["low"].astype(float) / entry_price) * 10_000.0
                    adverse = (path["high"].astype(float) / entry_price - 1.0) * 10_000.0
                else:
                    raise ValueError(f"invalid event direction: {direction}")
                gross_bps = direction * (exit_price / entry_price - 1.0) * 10_000.0
                mfe_bps = max(0.0, float(favorable.max()))
                mae_bps = max(0.0, float(adverse.max()))
                capture_ratio = gross_bps / mfe_bps if mfe_bps > 0 else None
                rows.append(
                    {
                        "symbol": str(symbol),
                        "direction": direction,
                        "event_timestamp": event_timestamp,
                        "entry_timestamp": entry_timestamp,
                        "exit_timestamp": exit_timestamp,
                        "horizon_bars": horizon,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_bps": gross_bps,
                        "base_net_bps": gross_bps - base_cost_bps,
                        "stress_net_bps": gross_bps - stress_cost_bps,
                        "mfe_bps": mfe_bps,
                        "mae_bps": mae_bps,
                        "mfe_stress_surplus_bps": mfe_bps - stress_cost_bps,
                        "capture_ratio": capture_ratio,
                    }
                )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def build_degradation_exit_labels(
    frame: pd.DataFrame,
    entry_signal: pd.Series,
    degradation_signal: pd.Series,
    *,
    horizons_bars: Iterable[int],
    start_timestamp: int,
    end_timestamp: int,
    stress_cost_bps: float,
) -> pd.DataFrame:
    """Compare fixed holding with a next-open exit after matching degradation."""

    if len(frame) != len(entry_signal) or len(frame) != len(degradation_signal):
        raise ValueError("frame and policy signal lengths differ")
    if start_timestamp >= end_timestamp:
        raise ValueError("start_timestamp must be before end_timestamp")
    horizons = sorted({int(value) for value in horizons_bars})
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons_bars must contain positive integers")
    if stress_cost_bps <= 0:
        raise ValueError("stress_cost_bps must be positive")
    source = frame.copy()
    source["entry_event"] = entry_signal.astype("int8")
    source["degradation_event"] = degradation_signal.astype("int8")
    rows: list[dict[str, object]] = []

    for symbol, raw_group in source.groupby("symbol", sort=True, observed=True):
        group = raw_group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        event_positions = np.flatnonzero(group["entry_event"].to_numpy(dtype="int8") != 0)
        for event_position in event_positions:
            event_timestamp = int(group.iloc[event_position]["timestamp"])
            if not start_timestamp <= event_timestamp < end_timestamp:
                continue
            entry_position = event_position + 1
            if entry_position >= len(group):
                continue
            entry_timestamp = int(group.iloc[entry_position]["timestamp"])
            if not start_timestamp <= entry_timestamp < end_timestamp:
                continue
            direction = int(group.iloc[event_position]["entry_event"])
            entry_price = float(group.iloc[entry_position]["open"])
            if direction not in (-1, 1) or not np.isfinite(entry_price) or entry_price <= 0:
                raise ValueError(f"invalid entry event for {symbol}")
            for horizon in horizons:
                continuation_exit_position = entry_position + horizon
                if continuation_exit_position >= len(group):
                    continue
                continuation_exit_timestamp = int(
                    group.iloc[continuation_exit_position]["timestamp"]
                )
                if continuation_exit_timestamp >= end_timestamp:
                    continue
                adaptive_exit_position = continuation_exit_position
                degradation_timestamp: int | None = None
                for decision_position in range(entry_position, continuation_exit_position):
                    if int(group.iloc[decision_position]["degradation_event"]) != direction:
                        continue
                    candidate_exit = decision_position + 1
                    if candidate_exit < continuation_exit_position:
                        adaptive_exit_position = candidate_exit
                        degradation_timestamp = int(group.iloc[decision_position]["timestamp"])
                    break
                continuation_exit_price = float(
                    group.iloc[continuation_exit_position]["open"]
                )
                adaptive_exit_price = float(group.iloc[adaptive_exit_position]["open"])
                continuation_gross = (
                    direction * (continuation_exit_price / entry_price - 1.0) * 10_000.0
                )
                adaptive_gross = (
                    direction * (adaptive_exit_price / entry_price - 1.0) * 10_000.0
                )
                rows.append(
                    {
                        "symbol": str(symbol),
                        "direction": direction,
                        "event_timestamp": event_timestamp,
                        "entry_timestamp": entry_timestamp,
                        "horizon_bars": horizon,
                        "continuation_exit_timestamp": continuation_exit_timestamp,
                        "adaptive_exit_timestamp": int(
                            group.iloc[adaptive_exit_position]["timestamp"]
                        ),
                        "degradation_event_timestamp": degradation_timestamp,
                        "degradation_exit_triggered": degradation_timestamp is not None,
                        "entry_price": entry_price,
                        "continuation_exit_price": continuation_exit_price,
                        "adaptive_exit_price": adaptive_exit_price,
                        "continuation_gross_bps": continuation_gross,
                        "adaptive_gross_bps": adaptive_gross,
                        "continuation_stress_net_bps": continuation_gross
                        - stress_cost_bps,
                        "adaptive_stress_net_bps": adaptive_gross - stress_cost_bps,
                        "adaptive_minus_continuation_bps": adaptive_gross
                        - continuation_gross,
                    }
                )
    return pd.DataFrame(rows, columns=DEGRADATION_POLICY_COLUMNS)
