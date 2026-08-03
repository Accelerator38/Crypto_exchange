from __future__ import annotations

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class NoTradeStrategy(IStrategy):
    """Engineering baseline that can never emit an entry or exit signal."""

    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short = True
    startup_candle_count = 1
    process_only_new_candles = True
    minimal_roi = {"0": 0.0}
    stoploss = -0.01
    trailing_stop = False
    use_exit_signal = True
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }

    def version(self) -> str:
        return "reset-v1"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
