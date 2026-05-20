from __future__ import annotations

from bisect import bisect_left, bisect_right
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


@dataclass
class _Series:
    bars: list[int]
    pnl_prefix: list[float]
    closed_prefix: list[int]
    signals_prefix: list[int]
    filled_prefix: list[int]
    actionable_prefix: list[int]

    @classmethod
    def empty(cls) -> "_Series":
        return cls(
            bars=[],
            pnl_prefix=[0.0],
            closed_prefix=[0],
            signals_prefix=[0],
            filled_prefix=[0],
            actionable_prefix=[0],
        )

    def append(self, event: _Event) -> None:
        self.bars.append(event.bar)
        self.pnl_prefix.append(self.pnl_prefix[-1] + event.pnl_usd)
        self.closed_prefix.append(self.closed_prefix[-1] + event.closed_trades)
        self.signals_prefix.append(self.signals_prefix[-1] + event.signals)
        self.filled_prefix.append(self.filled_prefix[-1] + event.filled)
        actionable = 1 if event.signals > 0 or event.filled > 0 else 0
        self.actionable_prefix.append(self.actionable_prefix[-1] + actionable)

    def range_float(self, prefix: list[float], start: int, end: int) -> float:
        return float(prefix[end] - prefix[start])

    def range_int(self, prefix: list[int], start: int, end: int) -> int:
        return int(prefix[end] - prefix[start])


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
        self._series: dict[tuple[str, str], _Series] = defaultdict(_Series.empty)

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
        event = _Event(
            bar=int(bar),
            pnl_usd=float(pnl_usd),
            closed_trades=max(0, int(closed_trades)),
            signals=max(0, int(signals)),
            filled=max(0, int(filled)),
        )
        key = (str(label), _regime_key(regime))
        self._events[key].append(event)
        self._series[key].append(event)

    def score_with_stats(
        self,
        *,
        label: str,
        regime: str,
        current_bar: int,
    ) -> EntryCausalStats:
        key = (str(label), _regime_key(regime))
        series = self._series.get(key)
        if series is None or not series.bars:
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
        end = bisect_left(series.bars, int(current_bar))
        cutoff = int(current_bar) - self.window_bars
        start = bisect_right(series.bars, cutoff, hi=end)
        closed_trades = series.closed_prefix[end]
        recent_pnl = series.range_float(series.pnl_prefix, start, end)
        recent_signals = series.range_int(series.signals_prefix, start, end)
        recent_filled = series.range_int(series.filled_prefix, start, end)
        actionable_bars = series.range_int(series.actionable_prefix, start, end)
        recent_bars = end - start
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
