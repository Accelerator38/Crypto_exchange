from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas import DataFrame


class MarketMode(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True)
class MarketModeContract:
    timeframe: str = "1h"
    timeframe_seconds: int = 3600
    ema_fast_hours: int = 24
    ema_slow_hours: int = 96
    momentum_hours: int = 24
    realized_volatility_hours: int = 24
    compression_slow_hours: int = 96
    volatility_lookback_hours: int = 720
    volatility_min_periods: int = 720
    unsafe_volatility_percentile: float = 0.95
    runtime_startup_bars: int = 999
    state_age_cap_bars: int = 168

    @property
    def minimum_history_bars(self) -> int:
        return self.realized_volatility_hours + self.volatility_min_periods


DEFAULT_MARKET_MODE_CONTRACT = MarketModeContract()
MARKET_MODE_ORDER = tuple(mode.value for mode in MarketMode)
REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def _validate_input(dataframe: DataFrame) -> DataFrame:
    missing = [name for name in REQUIRED_COLUMNS if name not in dataframe]
    if missing:
        raise ValueError(f"market mode input missing columns: {missing}")
    frame = dataframe.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("market mode timestamps must be unique and increasing")
    return frame


def _contiguous_bars(frame: DataFrame, *, seconds: int) -> pd.Series:
    expected = pd.Timedelta(seconds=seconds)
    valid_ohlcv = (
        np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    cadence_ok = frame["date"].diff().eq(expected)
    starts_new_run = (~valid_ohlcv) | (~cadence_ok)
    if len(starts_new_run):
        starts_new_run.iloc[0] = True
    run = starts_new_run.cumsum()
    age = frame.groupby(run, sort=False).cumcount() + 1
    return age.where(valid_ohlcv, 0).astype("int64")


def append_market_mode_columns(
    dataframe: DataFrame,
    *,
    contract: MarketModeContract = DEFAULT_MARKET_MODE_CONTRACT,
) -> DataFrame:
    """Append restart-stable causal Exia market state and diagnostics."""
    frame = _validate_input(dataframe)
    close = pd.to_numeric(frame["close"], errors="coerce")
    log_return = np.log(close).diff()
    ema_fast = close.ewm(span=contract.ema_fast_hours, adjust=False).mean()
    ema_slow = close.ewm(span=contract.ema_slow_hours, adjust=False).mean()
    momentum = close.pct_change(contract.momentum_hours, fill_method=None)
    realized_volatility = log_return.rolling(
        contract.realized_volatility_hours,
        min_periods=contract.realized_volatility_hours,
    ).std(ddof=0)
    slow_volatility = log_return.rolling(
        contract.compression_slow_hours,
        min_periods=contract.compression_slow_hours,
    ).std(ddof=0)
    shifted_volatility = realized_volatility.shift(1)
    volatility_threshold = shifted_volatility.rolling(
        contract.volatility_lookback_hours,
        min_periods=contract.volatility_min_periods,
    ).quantile(contract.unsafe_volatility_percentile)
    contiguous_bars_raw = _contiguous_bars(
        frame,
        seconds=contract.timeframe_seconds,
    )
    warmup_complete = (
        (contiguous_bars_raw >= contract.minimum_history_bars)
        & ema_fast.notna()
        & ema_slow.notna()
        & momentum.notna()
        & realized_volatility.notna()
        & volatility_threshold.notna()
    )
    volatility_shock = warmup_complete & (
        realized_volatility > volatility_threshold
    )
    trend_up = (
        warmup_complete
        & ~volatility_shock
        & (close > ema_fast)
        & (ema_fast > ema_slow)
        & (momentum > 0)
    )
    trend_down = (
        warmup_complete
        & ~volatility_shock
        & (close < ema_fast)
        & (ema_fast < ema_slow)
        & (momentum < 0)
    )

    mode = pd.Series(MarketMode.RANGE.value, index=frame.index, dtype="object")
    mode.loc[trend_up] = MarketMode.TREND_UP.value
    mode.loc[trend_down] = MarketMode.TREND_DOWN.value
    mode.loc[~warmup_complete | volatility_shock] = MarketMode.UNSAFE.value
    previous_mode = mode.shift(1).fillna(MarketMode.UNSAFE.value)
    state_run = mode.ne(mode.shift(1)).cumsum()
    state_age = (mode.groupby(state_run, sort=False).cumcount() + 1).clip(
        upper=contract.state_age_cap_bars
    )

    frame["exia_ema_fast"] = ema_fast
    frame["exia_ema_slow"] = ema_slow
    frame["exia_momentum_24h"] = momentum
    frame["exia_realized_volatility_24h"] = realized_volatility
    frame["exia_volatility_threshold"] = volatility_threshold
    frame["exia_volatility_percentile"] = shifted_volatility.rolling(
        contract.volatility_lookback_hours,
        min_periods=contract.volatility_min_periods,
    ).rank(pct=True)
    frame["exia_compression_ratio"] = realized_volatility / slow_volatility.replace(
        0.0, np.nan
    )
    frame["exia_trend_score_bps"] = (ema_fast / ema_slow - 1.0) * 10_000.0
    frame["exia_contiguous_bars"] = contiguous_bars_raw.clip(
        upper=contract.minimum_history_bars
    )
    frame["exia_warmup_complete"] = warmup_complete
    frame["exia_volatility_shock"] = volatility_shock
    frame["exia_market_mode"] = mode
    frame["exia_previous_mode"] = previous_mode
    frame["exia_state_age_bars"] = state_age.astype("int64")
    return frame
