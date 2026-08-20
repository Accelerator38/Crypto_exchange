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
