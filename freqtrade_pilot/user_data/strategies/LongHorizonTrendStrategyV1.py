from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class LongHorizonTrendStrategyV1(IStrategy):
    """Fixed low-turnover trend candidate for retrospective evaluation only."""

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = 700
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.08
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    ema_fast_hours = 72
    ema_slow_hours = 336
    breakout_hours = 168
    momentum_hours = 672
    min_abs_momentum = 0.05
    max_holding_hours = 336

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_price_type": "mark",
    }

    def bot_start(self, **kwargs: Any) -> None:
        if self.config.get("dry_run") is not True:
            raise RuntimeError("long horizon candidate is restricted to dry_run=true")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        dataframe["ema_fast"] = close.ewm(
            span=self.ema_fast_hours, adjust=False
        ).mean()
        dataframe["ema_slow"] = close.ewm(
            span=self.ema_slow_hours, adjust=False
        ).mean()
        dataframe["breakout_high"] = (
            dataframe["high"]
            .shift(1)
            .rolling(self.breakout_hours, min_periods=self.breakout_hours)
            .max()
        )
        dataframe["breakout_low"] = (
            dataframe["low"]
            .shift(1)
            .rolling(self.breakout_hours, min_periods=self.breakout_hours)
            .min()
        )
        dataframe["momentum"] = close.pct_change(self.momentum_hours)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prior_close = dataframe["close"].shift(1)
        prior_high = dataframe["breakout_high"].shift(1)
        prior_low = dataframe["breakout_low"].shift(1)

        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] > dataframe["breakout_high"])
            & (prior_close <= prior_high)
            & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["momentum"] >= self.min_abs_momentum),
            ["enter_long", "enter_tag"],
        ] = (1, "long_horizon_breakout")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] < dataframe["breakout_low"])
            & (prior_close >= prior_low)
            & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["momentum"] <= -self.min_abs_momentum),
            ["enter_short", "enter_tag"],
        ] = (1, "short_horizon_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] < dataframe["ema_fast"]),
            ["exit_long", "exit_tag"],
        ] = (1, "long_trend_invalidation")
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] > dataframe["ema_fast"]),
            ["exit_short", "exit_tag"],
        ] = (1, "short_trend_invalidation")
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        if current_time - trade.open_date_utc >= timedelta(
            hours=self.max_holding_hours
        ):
            return "max_holding_14d"
        return None
