from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_IDS = (
    "CONFIRMED_RANGE_REJECTION",
    "BREAKOUT_HOLD",
    "PULLBACK_RECLAIM",
    "TREND_DEGRADATION",
)


@dataclass(frozen=True)
class MarketStructureEventConfig:
    range_window: int = 20
    ema_fast: int = 12
    ema_slow: int = 48
    slope_bars: int = 3
    minimum_slope_bps: float = 10.0

    def __post_init__(self) -> None:
        if min(self.range_window, self.ema_fast, self.ema_slow, self.slope_bars) <= 0:
            raise ValueError("event windows must be positive")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be below ema_slow")
        if self.minimum_slope_bps < 0:
            raise ValueError("minimum_slope_bps cannot be negative")


def _direction(long_condition: pd.Series, short_condition: pd.Series) -> pd.Series:
    collision = long_condition & short_condition
    return pd.Series(
        np.select(
            [long_condition & ~collision, short_condition & ~collision],
            [1, -1],
            default=0,
        ),
        index=long_condition.index,
        dtype="int8",
    )


def build_market_structure_events(
    frame: pd.DataFrame,
    config: MarketStructureEventConfig | None = None,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Build four causal, completed-bar market-structure event streams."""

    cfg = config or MarketStructureEventConfig()
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"event frame is missing columns: {sorted(missing)}")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("event frame contains duplicate timestamp/symbol rows")

    event_parts: dict[str, list[pd.Series]] = {event_id: [] for event_id in EVENT_IDS}
    diagnostic_parts: list[pd.DataFrame] = []
    for _, raw_group in frame.groupby("symbol", sort=True, observed=True):
        group = raw_group.sort_values("timestamp", kind="stable").copy()
        close = group["close"].astype("float64")
        open_ = group["open"].astype("float64")
        high = group["high"].astype("float64")
        low = group["low"].astype("float64")
        fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        slope_bps = (slow / slow.shift(cfg.slope_bars) - 1.0) * 10_000.0

        prior_high = high.shift(1).rolling(
            cfg.range_window, min_periods=cfg.range_window
        ).max()
        prior_low = low.shift(1).rolling(
            cfg.range_window, min_periods=cfg.range_window
        ).min()
        rejected_upper = (high > prior_high) & (close < prior_high) & (close < open_)
        rejected_lower = (low < prior_low) & (close > prior_low) & (close > open_)
        rejection = _direction(rejected_lower, rejected_upper)

        pre_breakout_high = high.shift(2).rolling(
            cfg.range_window, min_periods=cfg.range_window
        ).max()
        pre_breakout_low = low.shift(2).rolling(
            cfg.range_window, min_periods=cfg.range_window
        ).min()
        first_upper_break = (
            (close.shift(1) > pre_breakout_high)
            & (close.shift(2) <= pre_breakout_high)
        )
        first_lower_break = (
            (close.shift(1) < pre_breakout_low)
            & (close.shift(2) >= pre_breakout_low)
        )
        breakout_hold = _direction(
            first_upper_break & (close > pre_breakout_high),
            first_lower_break & (close < pre_breakout_low),
        )

        long_trend = (fast > slow) & (slope_bps >= cfg.minimum_slope_bps)
        short_trend = (fast < slow) & (slope_bps <= -cfg.minimum_slope_bps)
        pullback_reclaim = _direction(
            long_trend
            & (close.shift(1) <= fast.shift(1))
            & (close.shift(1) > slow.shift(1))
            & (close > fast)
            & (close > open_),
            short_trend
            & (close.shift(1) >= fast.shift(1))
            & (close.shift(1) < slow.shift(1))
            & (close < fast)
            & (close < open_),
        )

        trend_degradation = _direction(
            (fast.shift(1) > slow.shift(1))
            & (slope_bps.shift(1) >= cfg.minimum_slope_bps)
            & (close.shift(1) >= fast.shift(1))
            & (close < fast),
            (fast.shift(1) < slow.shift(1))
            & (slope_bps.shift(1) <= -cfg.minimum_slope_bps)
            & (close.shift(1) <= fast.shift(1))
            & (close > fast),
        )

        local_events = {
            "CONFIRMED_RANGE_REJECTION": rejection,
            "BREAKOUT_HOLD": breakout_hold,
            "PULLBACK_RECLAIM": pullback_reclaim,
            "TREND_DEGRADATION": trend_degradation,
        }
        for event_id, signal in local_events.items():
            event_parts[event_id].append(signal)
        diagnostic_parts.append(
            pd.DataFrame(
                {
                    "timestamp": group["timestamp"],
                    "symbol": group["symbol"],
                    "prior_high": prior_high,
                    "prior_low": prior_low,
                    "pre_breakout_high": pre_breakout_high,
                    "pre_breakout_low": pre_breakout_low,
                    "ema_fast": fast,
                    "ema_slow": slow,
                    "ema_slope_bps": slope_bps,
                    **{event_id: signal for event_id, signal in local_events.items()},
                },
                index=group.index,
            )
        )

    events = {
        event_id: pd.concat(parts).reindex(frame.index).fillna(0).astype("int8")
        for event_id, parts in event_parts.items()
    }
    diagnostics = pd.concat(diagnostic_parts).reindex(frame.index)
    return events, diagnostics


__all__ = [
    "EVENT_IDS",
    "MarketStructureEventConfig",
    "build_market_structure_events",
]
