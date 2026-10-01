"""Fixed, offline counterfactual policies for a disclosed historical interval.

These comparisons describe September 2026. They cannot produce prospective
evidence or authorize a model change.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .accounting_v2 import CashSimulation
from .measurement_lab_v2 import object_sha

POLICIES = (
    "NoTrade", "EMA_TrendConsensus", "EMA_long_only", "GA_target_gate",
    "GA_entry_only", "EMA_long_random_matched",
)
RANDOM_SEEDS = (11, 23, 47, 71, 101, 137, 191, 269)


def targets_for_policy(base: pd.DataFrame, policy: str, *, seed: int | None = None) -> pd.DataFrame:
    """Use archived, already causal EMA proposals and GA target decisions."""
    if policy not in POLICIES:
        raise ValueError("unregistered ablation policy")
    required = ("execution_timestamp", "signal_timestamp", "symbol", "origin_id",
                "proposal", "ga_action", "ema_model_sha256", "ga_model_sha256")
    if any(column not in base for column in required):
        raise ValueError("archived decision columns missing")
    rows = base.sort_values(["execution_timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if rows.duplicated(["execution_timestamp", "symbol"]).any():
        raise ValueError("duplicate archived decision")
    proposal = rows["proposal"].to_numpy(int)
    ga = rows["ga_action"].to_numpy(int)
    if (not np.isin(proposal, (-1, 0, 1)).all()
            or not np.isin(ga, (-1, 0, 1)).all()
            or np.any((ga != 0) & (ga != proposal))):
        raise ValueError("GA target is not a gate on the registered proposal")
    if policy == "NoTrade":
        target = np.zeros(len(rows), dtype=np.int8)
    elif policy == "EMA_TrendConsensus":
        target = proposal.astype(np.int8)
    elif policy == "EMA_long_only":
        target = np.maximum(proposal, 0).astype(np.int8)
    elif policy == "GA_target_gate":
        target = ga.astype(np.int8)
    elif policy == "GA_entry_only":
        target = np.zeros(len(rows), dtype=np.int8)
        held: dict[str, int] = {}
        for index, row in rows.iterrows():
            symbol = str(row["symbol"])
            previous = held.get(symbol, 0)
            proposed = int(row["proposal"])
            if previous != proposed:
                previous = 0
                if proposed and int(row["ga_action"]) == proposed:
                    previous = proposed
            target[index] = previous
            held[symbol] = previous
    else:
        if type(seed) is not int or seed not in RANDOM_SEEDS:
            raise ValueError("random baseline requires a registered seed")
        target = np.zeros(len(rows), dtype=np.int8)
        rng = np.random.default_rng(seed)
        for symbol, group in rows.groupby("symbol", sort=True, observed=True):
            eligible = group.index[group["proposal"] == 1].to_numpy(dtype=int)
            matched_count = int((group["ga_action"] == 1).sum())
            if matched_count > len(eligible):
                raise ValueError(f"GA long coverage exceeds EMA opportunities for {symbol}")
            chosen = rng.choice(eligible, size=matched_count, replace=False)
            target[chosen] = 1
    result = rows[["execution_timestamp", "signal_timestamp", "symbol"]].copy()
    result["target"] = target
    source = "ga_model_sha256" if policy in ("GA_target_gate", "GA_entry_only") else "ema_model_sha256"
    result["model_sha256"] = [
        object_sha({"ablation_policy": policy, "source_model_sha256": original,
                    "seed": seed}) for original in rows[source]]
    return result


def summarize_simulation(
    simulation: CashSimulation, schedule: pd.DataFrame, base: pd.DataFrame,
    origin_ends: tuple[int, ...], cost_bps: float,
) -> dict[str, Any]:
    """Portfolio-level arithmetic PnL and exposure diagnostics on unit capital."""
    trades = simulation.trades
    gross = float(trades["gross_pnl"].sum()) if len(trades) else 0.0
    net = float(simulation.metrics["net_return"])
    fees = float(simulation.metrics["total_fees"])
    if abs(gross - fees - net) > 1e-9:
        raise ArithmeticError("gross/fees/net reconciliation failed")
    by_symbol = Counter()
    by_origin = Counter()
    if len(trades):
        for trade in trades.to_dict("records"):
            by_symbol[str(trade["symbol"])] += abs(float(trade["quantity"])
                                                     * float(trade["entry_price"]))
            entry = int(trade["entry_timestamp"])
            by_origin[next(index for index, end in enumerate(origin_ends) if entry < end)] += 1
    total_entry = sum(by_symbol.values())
    decisions = simulation.decisions
    active = decisions.assign(active=decisions["quantity_after"].ne(0)).groupby(
        "execution_timestamp", sort=True)["active"].sum()
    target = schedule["target"].to_numpy(int)
    opportunities = int(base["proposal"].ne(0).sum())
    return {
        "gross_return_pct": 100.0 * gross,
        "fees_pct": 100.0 * fees,
        "net_return_pct": 100.0 * net,
        "stress_cost_round_trip_bps": float(cost_bps),
        "max_drawdown_pct": 100.0 * float(simulation.metrics["max_drawdown"]),
        "trades": int(simulation.metrics["trades"]),
        "fills": int(simulation.metrics["fills"]),
        "long_trades": int((trades["direction"] == 1).sum()) if len(trades) else 0,
        "short_trades": int((trades["direction"] == -1).sum()) if len(trades) else 0,
        "entry_notional_turnover": total_entry,
        "target_opportunity_coverage": float(np.count_nonzero(target) / opportunities),
        "target_nonzero_decisions": int(np.count_nonzero(target)),
        "mean_simultaneous_positions": float(active.mean()),
        "max_simultaneous_positions": int(active.max()),
        "fraction_times_with_four_plus_positions": float((active >= 4).mean()),
        "max_symbol_entry_notional_share": max(by_symbol.values(), default=0.0)
        / total_entry if total_entry else 0.0,
        "max_origin_entry_count_share": max(by_origin.values(), default=0)
        / len(trades) if len(trades) else 0.0,
        "daily_returns": (
            simulation.portfolio.assign(day=lambda frame: (
                (frame["timestamp"] - int(frame["timestamp"].iloc[0])) // 86_400_000))
            .groupby("day", sort=True)["net_return"]
            .apply(lambda values: float(np.prod(1.0 + values) - 1.0)).tolist()),
    }
