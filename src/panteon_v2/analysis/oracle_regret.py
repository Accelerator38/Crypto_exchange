"""Oracle/regret helpers for Panteon allocation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LeaderPnL:
    label: str
    pnl_usd: float = 0.0
    bars: int = 0
    trades: int = 0


@dataclass(frozen=True)
class RegretReport:
    selected_label: str
    selected_pnl_usd: float
    best_label: str
    best_pnl_usd: float
    regret_usd: float
    no_trade_share_pct: float
    profitable_leader_share_pct: float
    leader_count: int


def compute_leader_regret(
    rows: Sequence[LeaderPnL],
    *,
    selected_label: str = "Panteon",
) -> RegretReport:
    """Compare selected allocator PnL against the best available leader row."""

    clean_rows = [row for row in rows if row.label]
    if not clean_rows:
        return RegretReport(
            selected_label=selected_label,
            selected_pnl_usd=0.0,
            best_label="",
            best_pnl_usd=0.0,
            regret_usd=0.0,
            no_trade_share_pct=0.0,
            profitable_leader_share_pct=0.0,
            leader_count=0,
        )

    selected = next((row for row in clean_rows if row.label == selected_label), None)
    selected_pnl = float(selected.pnl_usd) if selected is not None else 0.0
    best = max(clean_rows, key=lambda row: (float(row.pnl_usd), int(row.trades), row.label))
    total_bars = sum(max(0, int(row.bars)) for row in clean_rows)
    no_trade_bars = sum(
        max(0, int(row.bars))
        for row in clean_rows
        if row.label == "NoTrade"
    )
    profitable = sum(1 for row in clean_rows if float(row.pnl_usd) > 0.0)
    leader_count = len(clean_rows)

    return RegretReport(
        selected_label=selected_label,
        selected_pnl_usd=selected_pnl,
        best_label=best.label,
        best_pnl_usd=float(best.pnl_usd),
        regret_usd=max(0.0, float(best.pnl_usd) - selected_pnl),
        no_trade_share_pct=(
            100.0 * no_trade_bars / total_bars if total_bars > 0 else 0.0
        ),
        profitable_leader_share_pct=100.0 * profitable / leader_count,
        leader_count=leader_count,
    )
