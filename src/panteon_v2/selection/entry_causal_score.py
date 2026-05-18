from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class EntryCausalStats:
    score: float
    recent_pnl_usd: float
    closed_trades: int
    recent_signals: int
    recent_filled: int
    recent_bars: int
    recent_actionable_bars: int
    actionable_share: float
    has_data: bool


@dataclass(frozen=True)
class _Event:
    bar: int
    pnl_usd: float
    closed_trades: int
    signals: int
    filled: int


class EntryCausalScoreState:
    """Score shadow PnL only when it came from a recently actionable player."""

    def __init__(
        self,
        *,
        window_bars: int = 24,
        min_closed_trades: int = 20,
        min_filled: int = 3,
        actionability_weight: float = 1.0,
    ) -> None:
        self.window_bars = max(1, int(window_bars))
        self.min_closed_trades = max(0, int(min_closed_trades))
        self.min_filled = max(0, int(min_filled))
        self.actionability_weight = max(0.0, float(actionability_weight))
        self._events: dict[tuple[str, str], Deque[_Event]] = defaultdict(deque)

    def update(
        self,
        *,
        bar: int,
        label: str,
        regime: str,
        pnl_usd: float,
        closed_trades: int,
        signals: int,
        filled: int,
    ) -> None:
        if not label:
            return
        self._events[(str(label), _regime_key(regime))].append(
            _Event(
                bar=int(bar),
                pnl_usd=float(pnl_usd),
                closed_trades=max(0, int(closed_trades)),
                signals=max(0, int(signals)),
                filled=max(0, int(filled)),
            )
        )

    def score_with_stats(
        self,
        *,
        label: str,
        regime: str,
        current_bar: int,
    ) -> EntryCausalStats:
        key = (str(label), _regime_key(regime))
        events = self._events.get(key)
        if not events:
            return EntryCausalStats(
                score=0.0,
                recent_pnl_usd=0.0,
                closed_trades=0,
                recent_signals=0,
                recent_filled=0,
                recent_bars=0,
                recent_actionable_bars=0,
                actionable_share=0.0,
                has_data=False,
            )
        previous_events = [event for event in events if event.bar < int(current_bar)]
        cutoff = int(current_bar) - self.window_bars
        recent_events = [event for event in previous_events if event.bar > cutoff]
        closed_trades = sum(event.closed_trades for event in previous_events)
        recent_pnl = sum(event.pnl_usd for event in recent_events)
        recent_signals = sum(event.signals for event in recent_events)
        recent_filled = sum(event.filled for event in recent_events)
        actionable_bars = sum(
            1 for event in recent_events if event.signals > 0 or event.filled > 0
        )
        recent_bars = len(recent_events)
        actionable_share = actionable_bars / recent_bars if recent_bars else 0.0
        has_data = (
            closed_trades >= self.min_closed_trades
            and recent_filled >= self.min_filled
        )
        if not has_data:
            score = 0.0
        else:
            fill_confidence = (
                1.0
                if self.min_filled <= 0
                else min(1.0, recent_filled / max(1, self.min_filled))
            )
            actionability_factor = (
                actionable_share ** self.actionability_weight
                if self.actionability_weight > 0.0
                else 1.0
            )
            score = recent_pnl * fill_confidence * actionability_factor
        return EntryCausalStats(
            score=float(score),
            recent_pnl_usd=float(recent_pnl),
            closed_trades=int(closed_trades),
            recent_signals=int(recent_signals),
            recent_filled=int(recent_filled),
            recent_bars=int(recent_bars),
            recent_actionable_bars=int(actionable_bars),
            actionable_share=float(actionable_share),
            has_data=bool(has_data),
        )


def _regime_key(regime: object) -> str:
    return str(regime or "all").lower()
