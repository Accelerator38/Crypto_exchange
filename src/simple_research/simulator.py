from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    fee_bps_per_fill: float = 4.0
    slippage_bps_per_fill: float = 2.0

    def __post_init__(self) -> None:
        if self.fee_bps_per_fill <= 0:
            raise ValueError("fee_bps_per_fill must be positive")
        if self.slippage_bps_per_fill < 0:
            raise ValueError("slippage_bps_per_fill must be non-negative")

    @property
    def per_fill_bps(self) -> float:
        return self.fee_bps_per_fill + self.slippage_bps_per_fill

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_fill_bps


@dataclass(frozen=True)
class SimulationResult:
    ledger: pd.DataFrame
    portfolio: pd.DataFrame
    metrics: dict[str, object]


def _trade_metrics(
    ledger: pd.DataFrame,
    portfolio: pd.DataFrame,
    *,
    trial_count: int,
) -> dict[str, object]:
    closed = int(len(ledger))
    fills = int(ledger["fills"].sum()) if closed else 0
    net = ledger["net_bps"] if closed else pd.Series(dtype=float)
    gross = ledger["gross_bps"] if closed else pd.Series(dtype=float)
    mean_net = float(net.mean()) if closed else 0.0
    std_net = float(net.std(ddof=1)) if closed > 1 else 0.0
    standard_error = std_net / np.sqrt(closed) if closed > 1 else float("inf")
    lcb = mean_net - 1.6448536269514722 * standard_error
    adjusted_z = NormalDist().inv_cdf(1.0 - 0.05 / max(1, trial_count))
    adjusted_lcb = mean_net - adjusted_z * standard_error
    lcb_value = float(lcb) if np.isfinite(lcb) else None
    adjusted_lcb_value = (
        float(adjusted_lcb) if np.isfinite(adjusted_lcb) else None
    )
    max_drawdown = (
        float(portfolio["drawdown"].min()) * -1.0 if not portfolio.empty else 0.0
    )
    direction = {}
    symbol_counts = {}
    if closed:
        for value, label in ((1, "long"), (-1, "short")):
            subset = ledger.loc[ledger["direction"] == value, "net_bps"]
            direction[label] = {
                "trades": int(len(subset)),
                "mean_net_bps": float(subset.mean()) if len(subset) else None,
            }
        symbol_counts = {
            str(symbol): int(count)
            for symbol, count in ledger["symbol"].value_counts().sort_index().items()
        }
    max_symbol_share = (
        max(symbol_counts.values()) / closed if closed and symbol_counts else 0.0
    )
    return {
        "closed_trades": closed,
        "fills": fills,
        "mean_gross_bps": float(gross.mean()) if closed else 0.0,
        "mean_net_bps": mean_net,
        "lcb_95_net_bps": lcb_value,
        "multiple_testing_lcb_95_net_bps": adjusted_lcb_value,
        "win_rate": float((net > 0).mean()) if closed else 0.0,
        "total_net_return": (
            float(portfolio["equity"].iloc[-1] - 1.0) if not portfolio.empty else 0.0
        ),
        "max_drawdown": max_drawdown,
        "direction": direction,
        "symbol_trade_counts": symbol_counts,
        "max_symbol_trade_share": max_symbol_share,
    }


def simulate_targets(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    start_timestamp: int,
    end_timestamp: int,
    costs: CostModel,
    trial_count: int = 1,
) -> SimulationResult:
    """Execute close-time targets at the next bar open with one shared cost model."""
    if len(frame) != len(signal):
        raise ValueError("frame and signal lengths differ")
    if start_timestamp >= end_timestamp:
        raise ValueError("start_timestamp must be before end_timestamp")
    source = frame.copy()
    source["signal"] = signal.astype("int8")
    ledgers: list[dict[str, object]] = []
    returns_by_symbol: list[pd.DataFrame] = []
    for symbol, raw_group in source.groupby("symbol", sort=True):
        group = raw_group.sort_values("timestamp", kind="stable").copy()
        group["target"] = group["signal"].shift(1).fillna(0).astype("int8")
        group = group.loc[
            (group["timestamp"] >= start_timestamp)
            & (group["timestamp"] < end_timestamp)
        ].copy()
        if group.empty:
            continue
        group["previous_target"] = group["target"].shift(1).fillna(0).astype("int8")
        group["turnover"] = (group["target"] - group["previous_target"]).abs()
        group["next_open"] = group["open"].shift(-1)
        group["gross_return"] = group["target"] * (
            group["next_open"] / group["open"] - 1.0
        )
        group["cost_return"] = (
            group["turnover"] * costs.per_fill_bps / 10_000.0
        )
        group["net_return"] = group["gross_return"].fillna(0.0) - group[
            "cost_return"
        ]

        position = 0
        entry_price = 0.0
        entry_timestamp = 0
        entry_bar = 0
        for bar_number, row in enumerate(group.itertuples(index=False)):
            desired = int(row.target)
            price = float(row.open)
            if desired == position:
                continue
            if position:
                gross_bps = position * (price / entry_price - 1.0) * 10_000.0
                ledgers.append(
                    {
                        "symbol": str(symbol),
                        "direction": position,
                        "entry_timestamp": entry_timestamp,
                        "exit_timestamp": int(row.timestamp),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "holding_bars": bar_number - entry_bar,
                        "gross_bps": gross_bps,
                        "cost_bps": costs.round_trip_bps,
                        "net_bps": gross_bps - costs.round_trip_bps,
                        "fills": 2,
                        "exit_reason": "target_change",
                    }
                )
            position = desired
            if position:
                entry_price = price
                entry_timestamp = int(row.timestamp)
                entry_bar = bar_number
        if position:
            final = group.iloc[-1]
            exit_price = float(final["close"])
            gross_bps = position * (exit_price / entry_price - 1.0) * 10_000.0
            ledgers.append(
                {
                    "symbol": str(symbol),
                    "direction": position,
                    "entry_timestamp": entry_timestamp,
                    "exit_timestamp": int(final["timestamp"]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "holding_bars": len(group) - 1 - entry_bar,
                    "gross_bps": gross_bps,
                    "cost_bps": costs.round_trip_bps,
                    "net_bps": gross_bps - costs.round_trip_bps,
                    "fills": 2,
                    "exit_reason": "split_end",
                }
            )
            final_index = group.index[-1]
            group.loc[final_index, "net_return"] -= costs.per_fill_bps / 10_000.0
        returns_by_symbol.append(
            group.loc[:, ["timestamp", "net_return"]].assign(symbol=str(symbol))
        )

    ledger_columns = (
        "symbol",
        "direction",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "holding_bars",
        "gross_bps",
        "cost_bps",
        "net_bps",
        "fills",
        "exit_reason",
    )
    ledger = pd.DataFrame(ledgers, columns=ledger_columns)
    if returns_by_symbol:
        combined = pd.concat(returns_by_symbol, ignore_index=True)
        portfolio = (
            combined.pivot(index="timestamp", columns="symbol", values="net_return")
            .fillna(0.0)
            .sort_index()
        )
        net_return = portfolio.mean(axis=1)
        portfolio_frame = pd.DataFrame(
            {"timestamp": net_return.index.astype("int64"), "net_return": net_return.values}
        )
        portfolio_frame["equity"] = (1.0 + portfolio_frame["net_return"]).cumprod()
        portfolio_frame["drawdown"] = (
            portfolio_frame["equity"]
            / portfolio_frame["equity"].cummax().replace(0, np.nan)
            - 1.0
        )
    else:
        portfolio_frame = pd.DataFrame(
            columns=["timestamp", "net_return", "equity", "drawdown"]
        )
    metrics = _trade_metrics(ledger, portfolio_frame, trial_count=trial_count)
    metrics["cost_model"] = {
        "fee_bps_per_fill": costs.fee_bps_per_fill,
        "slippage_bps_per_fill": costs.slippage_bps_per_fill,
        "round_trip_bps": costs.round_trip_bps,
    }
    return SimulationResult(ledger=ledger, portfolio=portfolio_frame, metrics=metrics)
