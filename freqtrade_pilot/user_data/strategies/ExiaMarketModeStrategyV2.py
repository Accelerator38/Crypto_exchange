from __future__ import annotations

from typing import Any

from freqtrade.strategy import IStrategy
from pandas import DataFrame

from exia_market_mode_v2 import (
    DEFAULT_MARKET_MODE_CONTRACT,
    append_market_mode_columns,
)


class ExiaMarketModeStrategyV2(IStrategy):
    """Restart-stable fail-closed Exia foundation with no entries."""

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = DEFAULT_MARKET_MODE_CONTRACT.runtime_startup_bars
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.01
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False

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
            raise RuntimeError("Exia foundation is restricted to dry_run=true")

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
        return append_market_mode_columns(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe
