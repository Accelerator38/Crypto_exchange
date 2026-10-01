"""One prespecified, long-only entry hypothesis for disclosed offline data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .measurement_lab_v2 import object_sha

POLICY_ID = "trend_strength_efficiency_entry_v1"
STRENGTH_MIN = 1.0
EFFICIENCY_MIN = 0.50
LOOKBACK_BARS = 12
VOL_BARS = 6


def causal_entry_features(panel: pd.DataFrame, timeframe_ms: int) -> pd.DataFrame:
    """Features at candle close, shifted to the next candle's execution open."""
    if timeframe_ms <= 0 or panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("positive timeframe and unique market bars required")
    parts = []
    for symbol, group in panel.groupby("symbol", sort=True, observed=True):
        ordered = group.sort_values("timestamp", kind="stable")
        close = ordered["close"].astype(float)
        previous = close.shift(LOOKBACK_BARS)
        movement = close.diff().abs().rolling(LOOKBACK_BARS, min_periods=LOOKBACK_BARS).sum()
        bar_return = close.pct_change(fill_method=None)
        volatility = bar_return.rolling(VOL_BARS, min_periods=VOL_BARS).std(ddof=0)
        frame = pd.DataFrame({
            "execution_timestamp": ordered["timestamp"].to_numpy(np.int64) + timeframe_ms,
            "symbol": str(symbol),
            "return_12": (close / previous - 1.0).to_numpy(float),
            "volatility_6": volatility.to_numpy(float),
            "efficiency_12": ((close - previous).abs()
                              / movement.replace(0.0, np.nan)).to_numpy(float),
        })
        parts.append(frame)
    if not parts:
        raise ValueError("empty market panel")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["execution_timestamp", "symbol"], kind="stable").reset_index(drop=True)


def entry_schedule(base: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold an admitted long until the common EMA proposal exits or reverses."""
    required = ("execution_timestamp", "signal_timestamp", "symbol", "origin_id",
                "proposal", "ema_model_sha256")
    if any(column not in base for column in required):
        raise ValueError("archived EMA decisions are incomplete")
    rows = base.sort_values(["execution_timestamp", "symbol"], kind="stable").reset_index(drop=True)
    joined = rows.merge(features, on=["execution_timestamp", "symbol"],
                        how="left", validate="one_to_one", indicator=True)
    if joined["_merge"].ne("both").any():
        raise ValueError("feature row is missing on a registered decision")
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = (joined["proposal"].to_numpy(float)
                    * joined["return_12"].to_numpy(float)
                    / (np.sqrt(LOOKBACK_BARS)
                       * joined["volatility_6"].to_numpy(float)))
    can_enter = ((joined["proposal"].to_numpy(int) == 1)
                 & np.isfinite(strength) & (strength >= STRENGTH_MIN)
                 & np.isfinite(joined["efficiency_12"].to_numpy(float))
                 & (joined["efficiency_12"].to_numpy(float) >= EFFICIENCY_MIN))
    target = np.zeros(len(joined), dtype=np.int8)
    held: dict[str, bool] = {}
    for index, row in joined.iterrows():
        symbol = str(row["symbol"])
        active = held.get(symbol, False)
        if int(row["proposal"]) != 1:
            active = False
        elif not active and can_enter[index]:
            active = True
        target[index] = int(active)
        held[symbol] = active
    schedule = joined[["execution_timestamp", "signal_timestamp", "symbol"]].copy()
    schedule["target"] = target
    schedule["model_sha256"] = [
        object_sha({"policy": POLICY_ID, "ema_source_model_sha256": source,
                    "strength_min": STRENGTH_MIN, "efficiency_min": EFFICIENCY_MIN,
                    "lookback_bars": LOOKBACK_BARS, "volatility_bars": VOL_BARS,
                    "exit": "EMA_proposal_changes"})
        for source in joined["ema_model_sha256"]]
    diagnostics = joined[["execution_timestamp", "symbol", "origin_id", "proposal"]].copy()
    diagnostics["strength"] = strength
    diagnostics["efficiency_12"] = joined["efficiency_12"].to_numpy(float)
    diagnostics["can_enter"] = can_enter
    diagnostics["target"] = target
    return schedule, diagnostics
