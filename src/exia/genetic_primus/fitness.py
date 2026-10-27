from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from exia.scoring import moving_block_lcb

from .contracts import MARKET_STATES
from .simulator import PrimusSimulation


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    selection_score: float
    constraint_violation: float
    metrics: Mapping[str, Any]
    periods: pd.DataFrame
    slices: pd.DataFrame
    failed_gates: tuple[str, ...]


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _summary(ledger: pd.DataFrame, *, key: str, samples: int, alpha: float) -> dict[str, Any]:
    if ledger.empty:
        return {"trades": 0, "mean_bps": None, "median_bps": None, "lcb_bps": None}
    values = ledger["net_bps"].astype("float64")
    return {
        "trades": int(len(ledger)),
        "mean_bps": float(values.mean()),
        "median_bps": float(values.median()),
        "lcb_bps": moving_block_lcb(
            values,
            ledger["signal_timestamp"],
            samples=int(samples),
            alpha=float(alpha),
            seed=_seed(key),
        ),
    }


def score_simulation(
    simulation: PrimusSimulation,
    *,
    config: Mapping[str, Any],
    key: str,
    final_gate: bool = False,
) -> FitnessResult:
    ledger = simulation.ledger.copy()
    samples = int(config["bootstrap_samples"])
    alpha = float(config["bootstrap_alpha"])
    minimum_period = int(config["minimum_period_trades"])
    weights = config["weights"]
    if not ledger.empty:
        ledger["period"] = pd.to_datetime(
            ledger["entry_timestamp"], unit="ms", utc=True
        ).dt.strftime("%Y-%m")
    periods: list[dict[str, Any]] = []
    if not simulation.portfolio.empty:
        portfolio_dates = pd.to_datetime(
            simulation.portfolio["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)
        period_keys = pd.period_range(
            portfolio_dates.min().to_period("M"),
            portfolio_dates.max().to_period("M"),
            freq="M",
        ).astype(str)
    else:
        period_keys = []
    for period in period_keys:
        subset = ledger.loc[ledger["period"] == period] if not ledger.empty else ledger
        summary = _summary(subset, key=f"{key}:period:{period}", samples=samples, alpha=alpha)
        supported = summary["trades"] >= minimum_period and summary["lcb_bps"] is not None
        period_score = None
        if supported:
            period_score = (
                float(weights["mean"]) * float(summary["mean_bps"])
                + float(weights["median"]) * float(summary["median_bps"])
                + float(weights["lcb"]) * float(summary["lcb_bps"])
            )
        periods.append({"period": period, **summary, "supported": supported, "period_score": period_score})
    period_frame = pd.DataFrame(
        periods,
        columns=["period", "trades", "mean_bps", "median_bps", "lcb_bps", "supported", "period_score"],
    )
    supported = period_frame.loc[period_frame["supported"]] if not period_frame.empty else period_frame
    fitness = float(supported["period_score"].median()) if len(supported) else -1.0e9
    overall = _summary(ledger, key=f"{key}:overall", samples=samples, alpha=alpha)
    max_drawdown = (
        float(-simulation.portfolio["drawdown"].min()) if not simulation.portfolio.empty else 0.0
    )
    direction_counts = ledger["direction"].value_counts() if not ledger.empty else pd.Series(dtype=int)
    state_counts = ledger["market_state"].value_counts() if not ledger.empty else pd.Series(dtype=int)
    symbol_counts = ledger["symbol"].value_counts() if not ledger.empty else pd.Series(dtype=int)
    max_direction_share = float(direction_counts.max() / len(ledger)) if len(ledger) else 0.0
    max_state_share = float(state_counts.max() / len(ledger)) if len(ledger) else 0.0
    max_symbol_share = float(symbol_counts.max() / len(ledger)) if len(ledger) else 0.0
    supported_rate = float(len(supported) / len(period_frame)) if len(period_frame) else 0.0
    failures: list[str] = []
    if len(ledger) < int(config["minimum_split_trades"]):
        failures.append("minimum_split_trades")
    if supported_rate < float(config["minimum_supported_period_rate"]):
        failures.append("supported_period_rate")
    if max_drawdown > float(config["max_drawdown"]):
        failures.append("max_drawdown")
    if max_direction_share > float(config["max_direction_share"]):
        failures.append("direction_collapse")
    if max_state_share > float(config["max_state_share"]):
        failures.append("regime_collapse")
    if max_symbol_share > float(config["max_symbol_share"]):
        failures.append("symbol_collapse")
    slice_rows: list[dict[str, Any]] = []
    for scope, values in (
        ("market_state", MARKET_STATES),
        ("direction", ("LONG", "SHORT")),
        ("symbol", tuple(sorted(ledger["symbol"].unique())) if not ledger.empty else ()),
    ):
        for value in values:
            subset = ledger.loc[ledger[scope] == value] if not ledger.empty else ledger
            slice_rows.append(
                {"scope": scope, "value": value, **_summary(
                    subset, key=f"{key}:{scope}:{value}", samples=samples, alpha=alpha
                )}
            )
    slices = pd.DataFrame(slice_rows, columns=["scope", "value", "trades", "mean_bps", "median_bps", "lcb_bps"])
    if final_gate:
        if overall["mean_bps"] is None or float(overall["mean_bps"]) <= 0.0:
            failures.append("nonpositive_mean")
        if overall["lcb_bps"] is None or float(overall["lcb_bps"]) <= 0.0:
            failures.append("nonpositive_lcb")
        minimum_slice = int(config["minimum_slice_trades"])
        for row in slices.itertuples(index=False):
            if int(row.trades) < minimum_slice:
                continue
            if row.lcb_bps is None or float(row.lcb_bps) <= 0.0:
                failures.append(f"nonpositive_{row.scope}_lcb:{row.value}")
    minimum_split = float(config["minimum_split_trades"])
    constraint_violation = (
        max(0.0, (minimum_split - len(ledger)) / max(1.0, minimum_split))
        + max(
            0.0,
            float(config["minimum_supported_period_rate"]) - supported_rate,
        )
        + max(0.0, max_drawdown / float(config["max_drawdown"]) - 1.0)
        + max(0.0, max_direction_share / float(config["max_direction_share"]) - 1.0)
        + max(0.0, max_state_share / float(config["max_state_share"]) - 1.0)
        + max(0.0, max_symbol_share / float(config["max_symbol_share"]) - 1.0)
    )
    if constraint_violation > 0.0:
        bounded_fitness = float(np.clip(fitness, -100_000.0, 100_000.0))
        selection_score = -1_000_000.0 - constraint_violation + bounded_fitness * 1.0e-6
    else:
        selection_score = fitness
    metrics = {
        **overall,
        "fitness": fitness,
        "selection_score": selection_score,
        "constraint_violation": constraint_violation,
        "max_drawdown": max_drawdown,
        "turnover_fills": simulation.turnover_fills,
        "supported_periods": int(len(supported)),
        "total_periods": int(len(period_frame)),
        "supported_period_rate": supported_rate,
        "max_direction_share": max_direction_share,
        "max_state_share": max_state_share,
        "max_symbol_share": max_symbol_share,
        "action_counts": dict(simulation.action_counts),
        "failed_gates": failures,
    }
    return FitnessResult(
        fitness=fitness,
        selection_score=selection_score,
        constraint_violation=constraint_violation,
        metrics=metrics,
        periods=period_frame,
        slices=slices,
        failed_gates=tuple(failures),
    )
