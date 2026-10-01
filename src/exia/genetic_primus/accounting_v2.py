"""Offline fixed-quantity cash ledger. No exchange/runtime dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class InsolventSleeveError(ValueError):
    pass


@dataclass
class CashSleeve:
    initial_capital: float
    fee_rate: float
    cash: float = field(init=False)
    quantity: float = 0.0
    entry_price: float = 0.0
    entry_fee: float = 0.0
    entry_timestamp: int = 0
    entry_signal_timestamp: int = 0
    entry_model_sha256: str = ""
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    fills: int = 0
    closed_trades: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial capital must be finite and positive")
        if not np.isfinite(self.fee_rate) or not 0 <= self.fee_rate < 1:
            raise ValueError("fee rate must be finite in [0, 1)")
        self.cash = float(self.initial_capital)

    @staticmethod
    def _price(price: float) -> float:
        if not np.isfinite(price) or price <= 0:
            raise ValueError("mark/fill price must be finite and positive")
        return float(price)

    @property
    def direction(self) -> int:
        return int(np.sign(self.quantity))

    def equity(self, price: float) -> float:
        return float(self.cash + self.quantity * self._price(price))

    def unrealized_pnl(self, price: float) -> float:
        return (
            float(self.quantity * (self._price(price) - self.entry_price) - self.entry_fee)
            if self.quantity else 0.0
        )

    def reconciliation_error(self, price: float) -> float:
        return float(
            self.equity(price) - self.initial_capital
            - self.realized_pnl - self.unrealized_pnl(price)
        )

    def enter(self, direction: int, price: float, timestamp: int,
              signal_timestamp: int, model_sha256: str) -> None:
        price = self._price(price)
        if direction not in (-1, 1) or self.quantity:
            raise ValueError("entry requires a flat sleeve and direction -1/+1")
        if signal_timestamp > timestamp or not model_sha256:
            raise ValueError("entry requires causal signal and model identity")
        if self.cash <= 0:
            raise InsolventSleeveError("cannot enter with nonpositive equity")
        notional = self.cash / (1.0 + self.fee_rate)
        self.quantity = float(direction * notional / price)
        self.entry_fee = notional * self.fee_rate
        self.cash -= self.quantity * price + self.entry_fee
        self.entry_price = price
        self.entry_timestamp = int(timestamp)
        self.entry_signal_timestamp = int(signal_timestamp)
        self.entry_model_sha256 = str(model_sha256)
        self.total_fees += self.entry_fee
        self.fills += 1

    def close(self, price: float, timestamp: int, reason: str) -> dict[str, Any] | None:
        if not self.quantity:
            return None
        price = self._price(price)
        if timestamp < self.entry_timestamp:
            raise ValueError("exit cannot precede entry")
        exit_fee = abs(self.quantity) * price * self.fee_rate
        gross = self.quantity * (price - self.entry_price)
        net = gross - self.entry_fee - exit_fee
        trade = {
            "direction": self.direction,
            "quantity": self.quantity,
            "signal_timestamp": self.entry_signal_timestamp,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": int(timestamp),
            "entry_price": self.entry_price,
            "exit_price": price,
            "entry_fee": self.entry_fee,
            "exit_fee": exit_fee,
            "gross_pnl": gross,
            "net_pnl": net,
            "entry_model_sha256": self.entry_model_sha256,
            "exit_reason": reason,
        }
        self.cash += self.quantity * price - exit_fee
        self.realized_pnl += net
        self.total_fees += exit_fee
        self.fills += 1
        self.closed_trades += 1
        self.quantity = 0.0
        self.entry_fee = 0.0
        return trade


@dataclass(frozen=True)
class CashSimulation:
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    events: pd.DataFrame
    metrics: dict[str, Any]


def simulate_cash_targets(
    bars: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    symbols: tuple[str, ...],
    start_timestamp: int,
    end_timestamp: int,
    timeframe_ms: int,
    round_trip_cost_bps: float,
    force_flat_timestamps: tuple[int, ...] = (),
    record_details: bool = True,
) -> CashSimulation:
    """Execute already causal schedules; hold quantity until an explicit exit.

    Each schedule row is an order decision at a candle open. Its signal must be
    available by that time. Full-universe bars/schedules are mandatory. Events
    sharing a timestamp retain their order; their returns are never summed.
    """
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("fixed universe must be nonempty and unique")
    if timeframe_ms <= 0 or end_timestamp <= start_timestamp:
        raise ValueError("invalid simulation interval")
    if (end_timestamp - start_timestamp) % timeframe_ms:
        raise ValueError("simulation must contain whole candles")
    if not np.isfinite(round_trip_cost_bps) or round_trip_cost_bps < 0:
        raise ValueError("invalid cost")
    times = np.arange(start_timestamp, end_timestamp, timeframe_ms, dtype=np.int64)
    grid = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    market = bars.loc[
        bars.timestamp.ge(start_timestamp) & bars.timestamp.lt(end_timestamp)
        & bars.symbol.isin(symbols), ["timestamp", "symbol", "open", "close"]
    ]
    if market.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate market candle")
    market = market.set_index(["timestamp", "symbol"]).reindex(grid)
    if market.isna().any().any() or not np.isfinite(market.to_numpy(float)).all():
        raise ValueError("missing/nonfinite fixed-universe candle")
    if (market.to_numpy(float) <= 0).any():
        raise ValueError("nonpositive market price")
    required = ["execution_timestamp", "symbol", "target", "signal_timestamp", "model_sha256"]
    schedule = targets.loc[
        targets.execution_timestamp.ge(start_timestamp)
        & targets.execution_timestamp.lt(end_timestamp), required
    ].rename(columns={"execution_timestamp": "timestamp"})
    if schedule.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate target")
    if not schedule.symbol.isin(symbols).all() or len(schedule) != len(grid):
        raise ValueError("incomplete/extraneous fixed-universe target schedule")
    schedule = schedule.set_index(["timestamp", "symbol"]).reindex(grid)
    if schedule.isna().any().any() or not schedule.target.isin((-1, 0, 1)).all():
        raise ValueError("missing/invalid target")
    execution_times = schedule.index.get_level_values("timestamp").to_numpy(np.int64)
    if (schedule.signal_timestamp.to_numpy(np.int64) > execution_times).any():
        raise ValueError("future signal in target schedule")
    if not schedule.model_sha256.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("every decision requires a model SHA")

    shape = (len(times), len(symbols))
    opens = market.open.to_numpy(float).reshape(shape)
    closes = market.close.to_numpy(float).reshape(shape)
    desired = schedule.target.to_numpy(int).reshape(shape)
    signals = schedule.signal_timestamp.to_numpy(np.int64).reshape(shape)
    models = schedule.model_sha256.to_numpy(str).reshape(shape)
    sleeves = [CashSleeve(1.0 / len(symbols), round_trip_cost_bps / 20_000.0) for _ in symbols]
    reset_times = set(force_flat_timestamps)
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    portfolio: list[dict[str, Any]] = []
    peak = 1.0
    max_drawdown = 0.0
    maximum_error = 0.0
    event_index = 0

    def snapshot(timestamp: int, phase: str, prices: np.ndarray) -> float:
        nonlocal peak, max_drawdown, maximum_error, event_index
        equities = [sleeve.equity(float(price)) for sleeve, price in zip(sleeves, prices)]
        if any(value <= 0 for value in equities):
            raise InsolventSleeveError("nonpositive marked sleeve equity; no clipping or hidden leverage")
        equity = float(sum(equities))
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        error = max(abs(sleeve.reconciliation_error(float(price))) for sleeve, price in zip(sleeves, prices))
        maximum_error = max(maximum_error, error)
        if error > 1e-9:
            raise ArithmeticError("cash/ledger/equity reconciliation failed")
        if record_details:
            events.append({"event_index": event_index, "timestamp": int(timestamp),
                           "phase": phase, "equity": equity, "drawdown": equity / peak - 1.0,
                           "cash": float(sum(sleeve.cash for sleeve in sleeves)),
                           "realized_pnl": float(sum(sleeve.realized_pnl for sleeve in sleeves)),
                           "unrealized_pnl": float(sum(sleeve.unrealized_pnl(float(price))
                                                      for sleeve, price in zip(sleeves, prices)))})
        event_index += 1
        return equity

    def exit_sleeve(index: int, price: float, timestamp: int, reason: str) -> None:
        trade = sleeves[index].close(price, timestamp, reason)
        if trade is not None and record_details:
            trades.append({"symbol": symbols[index], **trade})

    previous_equity = snapshot(start_timestamp, "initial", opens[0])
    for row, timestamp in enumerate(times):
        snapshot(int(timestamp), "open_mark", opens[row])
        for column, sleeve in enumerate(sleeves):
            target = int(desired[row, column])
            before = sleeve.direction
            reset = int(timestamp) in reset_times
            if sleeve.quantity and (reset or before != target):
                exit_sleeve(column, float(opens[row, column]), int(timestamp),
                            "retrain_forced_exit" if reset else "target_change")
            if not sleeve.quantity and target:
                sleeve.enter(target, float(opens[row, column]), int(timestamp),
                             int(signals[row, column]), str(models[row, column]))
            if record_details:
                decisions.append({"symbol": symbols[column], "execution_timestamp": int(timestamp),
                                  "signal_timestamp": int(signals[row, column]),
                                  "model_sha256": str(models[row, column]),
                                  "position_before": before, "desired_position": target,
                                  "quantity_after": sleeve.quantity, "forced_reset": reset})
        snapshot(int(timestamp), "after_fills", opens[row])
        close_time = int(timestamp) + timeframe_ms
        equity = snapshot(close_time, "close_mark", closes[row])
        if close_time == end_timestamp:
            for column in range(len(sleeves)):
                exit_sleeve(column, float(closes[row, column]), close_time, "interval_end")
            equity = snapshot(close_time, "terminal_fills", closes[row])
        portfolio.append({"timestamp": close_time, "net_return": equity / previous_equity - 1.0,
                          "equity": equity})
        previous_equity = equity
    net_realized = float(sum(sleeve.realized_pnl for sleeve in sleeves))
    if abs((previous_equity - 1.0) - net_realized) > 1e-9:
        raise ArithmeticError("final realized ledger does not match portfolio PnL")
    return CashSimulation(
        pd.DataFrame(portfolio), pd.DataFrame(trades), pd.DataFrame(decisions), pd.DataFrame(events),
        {"accounting_version": 2, "final_equity": previous_equity,
         "net_return": previous_equity - 1.0, "max_drawdown": max_drawdown,
         "net_realized_pnl": net_realized, "total_fees": float(sum(x.total_fees for x in sleeves)),
         "trades": sum(x.closed_trades for x in sleeves), "fills": sum(x.fills for x in sleeves),
         "decision_count": len(grid), "maximum_reconciliation_error": maximum_error,
         "market_readiness": False, "orders_enabled": False, "promotion_authority": False},
    )
