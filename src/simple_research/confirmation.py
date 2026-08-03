from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd


HOUR_MS = 3_600_000
MINUTE_MS = 60_000


def aggregate_complete_minutes_to_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete UTC 1m groups to 1h bars and fail on partial hours."""
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"minute frame missing columns: {missing}")
    source = frame.loc[:, sorted(required)].copy()
    source["timestamp"] = pd.to_numeric(
        source["timestamp"], errors="raise"
    ).astype("int64")
    if (source["timestamp"] % MINUTE_MS != 0).any():
        raise ValueError("minute frame contains off-grid timestamps")
    if source.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("minute frame contains duplicate symbol/timestamp rows")
    for column in ("open", "high", "low", "close", "volume"):
        source[column] = pd.to_numeric(source[column], errors="raise").astype(
            "float64"
        )
    source = source.sort_values(["symbol", "timestamp"], kind="stable")
    source["hour_timestamp"] = source["timestamp"] // HOUR_MS * HOUR_MS
    counts = source.groupby(["symbol", "hour_timestamp"], sort=False).size()
    incomplete = counts.loc[counts != 60]
    if not incomplete.empty:
        first = incomplete.index[0]
        raise ValueError(
            "minute frame contains incomplete hour: "
            f"{first[0]}@{first[1]} has {int(incomplete.iloc[0])} rows"
        )
    hourly = (
        source.groupby(["symbol", "hour_timestamp"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"hour_timestamp": "timestamp"})
    )
    return hourly.loc[
        :, ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    ].sort_values(["symbol", "timestamp"], kind="stable", ignore_index=True)


def apply_regime_allowlist(
    frame: pd.DataFrame,
    signal: pd.Series,
    regime_lookup: dict[int, str],
    *,
    allowed_regimes: Collection[str],
) -> tuple[pd.Series, pd.Series]:
    """Mask close-time targets using the regime known at the same closed bar."""
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    regimes = frame["timestamp"].astype("int64").map(regime_lookup).fillna("unknown")
    allowed = regimes.isin(set(allowed_regimes))
    masked = signal.where(allowed, 0).astype("int8")
    return masked, regimes


def apply_regime_entry_gate(
    frame: pd.DataFrame,
    signal: pd.Series,
    regime_lookup: dict[int, str],
    *,
    allowed_entry_regimes: Collection[str],
) -> tuple[pd.Series, pd.Series]:
    """Gate new entries while letting admitted positions follow the base target."""
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    regimes = frame["timestamp"].astype("int64").map(regime_lookup).fillna("unknown")
    allowed = set(allowed_entry_regimes)
    gated = pd.Series(0, index=frame.index, dtype="int8")
    for _, group in frame.groupby("symbol", sort=True):
        state = 0
        for index in group.sort_values("timestamp", kind="stable").index:
            desired = int(signal.loc[index])
            if state == 0:
                if desired != 0 and regimes.loc[index] in allowed:
                    state = desired
            elif desired == state:
                pass
            elif desired == 0:
                state = 0
            else:
                state = desired if regimes.loc[index] in allowed else 0
            gated.loc[index] = state
    return gated, regimes


def moving_block_bootstrap(
    returns: pd.Series,
    *,
    block_size: int,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """Bootstrap an hourly portfolio series while retaining intra-day dependence."""
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype="float64")
    if not len(values):
        raise ValueError("bootstrap returns are empty")
    if block_size <= 0 or replicates <= 0:
        raise ValueError("block_size and replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    block_count = math.ceil(len(values) / block_size)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(replicates, block_count))
    offsets = np.arange(block_size, dtype="int64")
    indices = (starts[:, :, None] + offsets[None, None, :]) % len(values)
    sampled = values[indices].reshape(replicates, -1)[:, : len(values)]
    total_returns = np.prod(1.0 + sampled, axis=1) - 1.0
    mean_hourly_returns = sampled.mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "observations": int(len(values)),
        "block_size": int(block_size),
        "nominal_blocks": int(block_count),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "observed_total_return": float(np.prod(1.0 + values) - 1.0),
        "total_return_lcb": float(np.quantile(total_returns, alpha)),
        "observed_mean_hourly_return": float(values.mean()),
        "mean_hourly_return_lcb": float(
            np.quantile(mean_hourly_returns, alpha)
        ),
    }
