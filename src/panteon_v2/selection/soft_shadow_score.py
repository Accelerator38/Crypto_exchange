from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class _Event:
    bar: int
    pnl_usd: float
    closed_trades: int


class SoftShadowScoreState:
    def __init__(self, *, window_bars: int = 24, min_closed_trades: int = 50) -> None:
        self.window_bars = max(1, int(window_bars))
        self.min_closed_trades = max(0, int(min_closed_trades))
        self._events: dict[tuple[str, str], Deque[_Event]] = defaultdict(deque)

    def update(
        self,
        *,
        bar: int,
        label: str,
        regime: str,
        pnl_usd: float,
        closed_trades: int,
    ) -> None:
        if not label:
            return
        key = (str(label), _regime_key(regime))
        self._events[key].append(
            _Event(
                bar=int(bar),
                pnl_usd=float(pnl_usd),
                closed_trades=max(0, int(closed_trades)),
            )
        )

    def score(self, *, label: str, regime: str, current_bar: int) -> float:
        score, trades = self.score_with_trade_count(
            label=label,
            regime=regime,
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
        current_bar: int,
    ) -> tuple[float, int]:
        key = (str(label), _regime_key(regime))
        events = self._events.get(key)
        if not events:
            return 0.0, 0
        cutoff = int(current_bar) - self.window_bars
        previous_events = [event for event in events if event.bar < int(current_bar)]
        trades = sum(event.closed_trades for event in previous_events)
        recent_pnl = sum(
            event.pnl_usd
            for event in previous_events
            if event.bar > cutoff
        )
        return float(recent_pnl), int(trades)


def _regime_key(regime: object) -> str:
    return str(regime or "all").lower()
