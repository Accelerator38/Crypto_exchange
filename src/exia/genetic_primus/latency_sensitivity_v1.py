"""Deterministic whole-bar delay for frozen long/flat OHLCV decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .measurement_lab_v2 import object_sha
from .rolling_linear_retrain_v1 import DT


def delay_target_schedule(schedule: pd.DataFrame, symbols: tuple[str, ...],
                          start_ms: int, end_exclusive_ms: int,
                          delay_bars: int) -> pd.DataFrame:
    """Delay target *changes* and their signal/model lineage by 0 or one 4h bar."""
    if (delay_bars not in (0, 1) or not symbols or len(set(symbols)) != len(symbols)
            or end_exclusive_ms <= start_ms or (end_exclusive_ms - start_ms) % DT):
        raise ValueError("invalid bounded delay experiment")
    required = {"execution_timestamp", "symbol", "signal_timestamp", "target", "model_sha256"}
    if not required.issubset(schedule.columns):
        raise ValueError("incomplete frozen target schedule")
    grid = pd.MultiIndex.from_product(
        [np.arange(start_ms, end_exclusive_ms, DT), symbols],
        names=["execution_timestamp", "symbol"]).to_frame(index=False)
    ordered = grid.merge(schedule[list(required)], on=["execution_timestamp", "symbol"],
                         how="left", validate="one_to_one", sort=False)
    if (len(ordered) != len(schedule) or ordered.isna().any().any()
            or not ordered.target.isin([0, 1]).all()
            or (ordered.signal_timestamp > ordered.execution_timestamp).any()
            or not ordered.model_sha256.map(lambda value: isinstance(value, str) and bool(value)).all()):
        raise ValueError("schedule is not a complete causal long/flat grid")
    ordered["target"] = ordered.target.astype(int)
    ordered["signal_timestamp"] = ordered.signal_timestamp.astype(np.int64)
    if delay_bars == 0:
        return ordered[["execution_timestamp", "symbol", "signal_timestamp",
                        "target", "model_sha256"]].copy()
    sentinel = object_sha({"policy": "NoTrade_before_first_delayed_signal",
                          "start_ms": start_ms, "delay_bars": 1})
    previous = ordered.groupby("symbol", sort=False)[
        ["target", "signal_timestamp", "model_sha256"]].shift(1)
    ordered["target"] = previous.target.fillna(0).astype(int)
    ordered["signal_timestamp"] = previous.signal_timestamp.fillna(start_ms).astype(np.int64)
    ordered["model_sha256"] = previous.model_sha256.fillna(sentinel).astype(str)
    if (ordered.signal_timestamp > ordered.execution_timestamp).any():
        raise AssertionError("delayed signal became future information")
    return ordered[["execution_timestamp", "symbol", "signal_timestamp",
                    "target", "model_sha256"]]
