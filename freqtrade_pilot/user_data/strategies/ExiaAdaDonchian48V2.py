from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

from exia_market_mode_v2 import (
    DEFAULT_MARKET_MODE_CONTRACT,
    MarketMode,
    append_market_mode_columns,
)


class ExiaAdaDonchian48V2(IStrategy):
    """Fixed ADA-only 48-hour breakout candidate on restart-stable taxonomy v2."""

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = DEFAULT_MARKET_MODE_CONTRACT.runtime_startup_bars
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.05
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

    allowed_pair = "ADA/USDT:USDT"
    breakout_hours = 48
    max_holding_hours = 168

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
            raise RuntimeError("Exia candidate research is restricted to dry_run=true")

    def leverage(
        self,
        pair: str,
        current_time: object,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = append_market_mode_columns(dataframe)
        dataframe["exia_breakout_high_48"] = (
            dataframe["high"]
            .shift(1)
            .rolling(self.breakout_hours, min_periods=self.breakout_hours)
            .max()
        )
        dataframe["exia_breakout_low_48"] = (
            dataframe["low"]
            .shift(1)
            .rolling(self.breakout_hours, min_periods=self.breakout_hours)
            .min()
        )
        return dataframe

    @staticmethod
    def _empty_entries(dataframe: DataFrame) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._empty_entries(dataframe)
        if metadata.get("pair") != self.allowed_pair:
            return dataframe

        prior_close = dataframe["close"].shift(1)
        prior_high = dataframe["exia_breakout_high_48"].shift(1)
        prior_low = dataframe["exia_breakout_low_48"].shift(1)
        volume_ok = dataframe["volume"] > 0
        dataframe.loc[
            volume_ok
            & (dataframe["exia_market_mode"] == MarketMode.TREND_UP.value)
            & (dataframe["close"] > dataframe["exia_breakout_high_48"])
            & (prior_close <= prior_high),
            ["enter_long", "enter_tag"],
        ] = (1, "ada_donchian48_long")
        dataframe.loc[
            volume_ok
            & (dataframe["exia_market_mode"] == MarketMode.TREND_DOWN.value)
            & (dataframe["close"] < dataframe["exia_breakout_low_48"])
            & (prior_close >= prior_low),
            ["enter_short", "enter_tag"],
        ] = (1, "ada_donchian48_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        long_invalid = dataframe["exia_market_mode"].isin(
            [MarketMode.TREND_DOWN.value, MarketMode.UNSAFE.value]
        ) | (dataframe["close"] < dataframe["exia_ema_slow"])
        short_invalid = dataframe["exia_market_mode"].isin(
            [MarketMode.TREND_UP.value, MarketMode.UNSAFE.value]
        ) | (dataframe["close"] > dataframe["exia_ema_slow"])
        dataframe.loc[long_invalid, ["exit_long", "exit_tag"]] = (
            1,
            "trend_long_invalid",
        )
        dataframe.loc[short_invalid, ["exit_short", "exit_tag"]] = (
            1,
            "trend_short_invalid",
        )
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
            return "max_holding_7d"
        return None
