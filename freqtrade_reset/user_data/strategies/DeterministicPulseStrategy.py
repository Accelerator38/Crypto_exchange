from __future__ import annotations

from pandas import DataFrame

from freqtrade.exceptions import OperationalException
from freqtrade.strategy import IStrategy


class DeterministicPulseStrategy(IStrategy):
    """Dry-run-only pulse used to verify signal and simulated-order plumbing."""

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    startup_candle_count = 2
    process_only_new_candles = True
    minimal_roi = {"0": 10.0}
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

    def bot_start(self, **kwargs) -> None:
        if self.config.get("dry_run") is not True:
            raise OperationalException(
                "DeterministicPulseStrategy is engineering-only and requires dry_run=true"
            )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        date = dataframe["date"].dt
        pulse = (date.hour % 6 == 0) & (date.minute == 0) & (dataframe["volume"] > 0)
        dataframe.loc[pulse, ["enter_long", "enter_tag"]] = (1, "engineering_pulse")
        dataframe["enter_long"] = dataframe["enter_long"].fillna(0).astype(int)
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        date = dataframe["date"].dt
        dataframe.loc[date.minute == 15, ["exit_long", "exit_tag"]] = (
            1,
            "engineering_pulse_exit",
        )
        dataframe["exit_long"] = dataframe["exit_long"].fillna(0).astype(int)
        dataframe["exit_short"] = 0
        return dataframe
