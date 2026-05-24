from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Deque


@dataclass(frozen=True)
class _Event:
    bar: int
    pnl_usd: float
    closed_trades: int
    winning_trades: int


@dataclass(frozen=True)
class SoftShadowStats:
    score: float
    closed_trades: int
    winning_trades: int
    recent_downside_usd: float
    pnl_per_trade_mean_usd: float = 0.0
    pnl_per_trade_std_usd: float = 0.0

    @property
    def losing_trades(self) -> int:
        return max(0, int(self.closed_trades) - int(self.winning_trades))

    @property
    def win_rate_pct(self) -> float:
        if self.closed_trades <= 0:
            return 0.0
        return float(self.winning_trades) / float(self.closed_trades) * 100.0

    @property
    def pnl_per_trade_lcb_usd(self) -> float:
        closed = max(0, int(self.closed_trades))
        if closed <= 0:
            return 0.0
        return float(self.pnl_per_trade_mean_usd) - (
            float(self.pnl_per_trade_std_usd) / math.sqrt(float(closed))
        )

    def as_confirmation_payload(self) -> dict:
        return {
            "score": float(self.score),
            "closed_trades": int(self.closed_trades),
            "winning_trades": int(self.winning_trades),
            "losing_trades": int(self.losing_trades),
            "win_rate_pct": float(self.win_rate_pct),
            "recent_downside_usd": float(self.recent_downside_usd),
            "pnl_per_trade_mean_usd": float(self.pnl_per_trade_mean_usd),
            "pnl_per_trade_std_usd": float(self.pnl_per_trade_std_usd),
            "pnl_per_trade_lcb_usd": float(self.pnl_per_trade_lcb_usd),
        }


class SoftShadowScoreState:
    def __init__(self, *, window_bars: int = 24, min_closed_trades: int = 50) -> None:
        self.window_bars = max(1, int(window_bars))
        self.min_closed_trades = max(0, int(min_closed_trades))
        self._events: dict[tuple[str, str, str, str], Deque[_Event]] = defaultdict(deque)

    def update(
        self,
        *,
        bar: int,
        label: str,
        regime: str,
        pnl_usd: float,
        closed_trades: int,
        winning_trades: int = 0,
        symbol: str = "",
        action: str = "",
    ) -> None:
        if not label:
            return
        key = (
            str(label),
            _regime_key(regime),
            _symbol_key(symbol),
            _action_key(action),
        )
        self._events[key].append(
            _Event(
                bar=int(bar),
                pnl_usd=float(pnl_usd),
                closed_trades=max(0, int(closed_trades)),
                winning_trades=max(0, int(winning_trades)),
            )
        )

    def score(
        self,
        *,
        label: str,
        regime: str,
        symbol: str = "",
        action: str = "",
        current_bar: int,
    ) -> float:
        score, trades = self.score_with_trade_count(
            label=label,
            regime=regime,
            symbol=symbol,
            action=action,
            current_bar=current_bar,
        )
        if trades < self.min_closed_trades:
            return 0.0
        return score

    def score_with_trade_count(
        self,
        *,
        label: str,
        regime: str,
        symbol: str = "",
        action: str = "",
        current_bar: int,
    ) -> tuple[float, int]:
        stats = self.stats(
            label=label,
            regime=regime,
            symbol=symbol,
            action=action,
            current_bar=current_bar,
        )
        return float(stats.score), int(stats.closed_trades)

    def stats(
        self,
        *,
        label: str,
        regime: str,
        symbol: str = "",
        action: str = "",
        current_bar: int,
    ) -> SoftShadowStats:
        key = (
            str(label),
            _regime_key(regime),
            _symbol_key(symbol),
            _action_key(action),
        )
        events = self._events.get(key)
        if not events:
            return SoftShadowStats(
                score=0.0,
                closed_trades=0,
                winning_trades=0,
                recent_downside_usd=0.0,
            )
        cutoff = int(current_bar) - self.window_bars
        previous_events = [event for event in events if event.bar < int(current_bar)]
        trades = sum(event.closed_trades for event in previous_events)
        wins = sum(event.winning_trades for event in previous_events)
        recent_pnl = sum(
            event.pnl_usd
            for event in previous_events
            if event.bar > cutoff
        )
        recent_downside = sum(
            abs(event.pnl_usd)
            for event in previous_events
            if event.bar > cutoff and event.pnl_usd < 0.0
        )
        pnl_mean, pnl_std = _weighted_pnl_per_trade_stats(previous_events)
        return SoftShadowStats(
            score=float(recent_pnl),
            closed_trades=int(trades),
            winning_trades=int(min(wins, trades)),
            recent_downside_usd=float(recent_downside),
            pnl_per_trade_mean_usd=float(pnl_mean),
            pnl_per_trade_std_usd=float(pnl_std),
        )


def _weighted_pnl_per_trade_stats(events: list[_Event]) -> tuple[float, float]:
    total_trades = sum(max(0, int(event.closed_trades)) for event in events)
    if total_trades <= 0:
        return 0.0, 0.0
    mean = (
        sum(
            (float(event.pnl_usd) / float(event.closed_trades)) * int(event.closed_trades)
            for event in events
            if int(event.closed_trades) > 0
        )
        / float(total_trades)
    )
    if total_trades < 2:
        return float(mean), 0.0
    variance = (
        sum(
            int(event.closed_trades)
            * ((float(event.pnl_usd) / float(event.closed_trades)) - mean) ** 2
            for event in events
            if int(event.closed_trades) > 0
        )
        / float(total_trades - 1)
    )
    return float(mean), math.sqrt(max(0.0, variance))


def _regime_key(regime: object) -> str:
    return str(regime or "all").lower()


def _symbol_key(symbol: object) -> str:
    return str(symbol or "all").upper()


def _action_key(action: object) -> str:
    return str(action or "all").lower()
