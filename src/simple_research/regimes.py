from __future__ import annotations

from typing import Any

import pandas as pd


REGIME_ORDER = (
    "bullish",
    "bearish",
    "volatile_mixed",
    "range_low_vol",
    "neutral",
    "unknown",
)

REGIME_CONTRACT: dict[str, Any] = {
    "source": "BTC/USDT",
    "timeframe": "1h",
    "attribution_time": "last_closed_hour_before_trade_entry",
    "trend_priority": True,
    "ema_fast_hours": 72,
    "ema_slow_hours": 336,
    "momentum_hours": 168,
    "atr_hours": 14,
    "volatility_percentile_lookback_hours": 4320,
    "volatility_percentile_min_periods": 720,
    "low_vol_percentile": 0.25,
    "high_vol_percentile": 0.75,
    "used_by_strategy": False,
}


def build_market_regime_lookup(
    frame: pd.DataFrame,
    *,
    symbol: str = "BTC/USDT",
) -> dict[int, str]:
    """Classify each closed BTC hour using only current and earlier candles."""
    source = (
        frame.loc[frame["symbol"] == symbol].copy()
        if "symbol" in frame
        else frame.copy()
    )
    if source.empty:
        raise ValueError(f"regime source is empty for {symbol}")
    source = (
        source.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    timestamps = pd.to_numeric(source["timestamp"], errors="raise").astype("int64")
    if len(source) > 1 and not (timestamps.diff().dropna() == 3_600_000).all():
        raise ValueError("regime source must have continuous 1h cadence")

    close = source["close"]
    ema_fast = close.ewm(span=72, adjust=False).mean()
    ema_slow = close.ewm(span=336, adjust=False).mean()
    momentum_7d = close.pct_change(168)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            source["high"] - source["low"],
            (source["high"] - previous_close).abs(),
            (source["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = true_range.rolling(14, min_periods=14).mean() / close
    trailing_atr = atr_pct.shift(1).rolling(4320, min_periods=720)
    low_vol_threshold = trailing_atr.quantile(0.25)
    high_vol_threshold = trailing_atr.quantile(0.75)

    bullish = (close > ema_fast) & (ema_fast > ema_slow) & (momentum_7d > 0)
    bearish = (close < ema_fast) & (ema_fast < ema_slow) & (momentum_7d < 0)
    regimes = pd.Series("neutral", index=source.index, dtype="object")
    regimes.loc[atr_pct <= low_vol_threshold] = "range_low_vol"
    regimes.loc[atr_pct >= high_vol_threshold] = "volatile_mixed"
    regimes.loc[bullish] = "bullish"
    regimes.loc[bearish] = "bearish"
    return {
        int(timestamp): str(regime)
        for timestamp, regime in zip(timestamps, regimes)
    }


def attribute_trade_regimes(
    ledger: pd.DataFrame,
    regime_lookup: dict[int, str],
) -> pd.DataFrame:
    result = ledger.copy()
    if result.empty:
        result["market_regime"] = pd.Series(dtype="object")
        return result
    decision_timestamp = result["entry_timestamp"].astype("int64") - 3_600_000
    result["market_regime"] = decision_timestamp.map(regime_lookup).fillna("unknown")
    return result
