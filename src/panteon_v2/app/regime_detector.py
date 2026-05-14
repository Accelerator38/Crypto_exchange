"""Market-regime detection for live v2 feeds.

This component is the single owner for converting raw market prices into the
canonical v2 ``Regime``. It deliberately does not read agent internals: agents
consume market regime, they do not define it.
"""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from ..domain.types import Regime


class PriceRegimeDetector:
    """Deterministic multi-symbol regime detector with anchors and confidence."""

    def __init__(
        self,
        *,
        lookback: int = 3,
        long_lookback: int = 12,
        bullish_return: float = 0.010,
        bearish_return: float = -0.010,
        crash_return: float = -0.060,
        anchor_symbols: Tuple[str, ...] = ("BTC", "ETH"),
        anchor_weight: float = 3.0,
        min_crash_breadth: float = 0.50,
        hysteresis_bars: int = 1,
        max_history: int = 128,
    ) -> None:
        self.lookback = max(2, int(lookback or 2))
        self.long_lookback = max(self.lookback, int(long_lookback or self.lookback))
        self.bullish_return = float(bullish_return)
        self.bearish_return = float(bearish_return)
        self.crash_return = float(crash_return)
        self.anchor_symbols = tuple(str(sym).upper() for sym in anchor_symbols)
        self.anchor_weight = max(1.0, float(anchor_weight or 1.0))
        self.min_crash_breadth = max(0.0, min(1.0, float(min_crash_breadth)))
        self.hysteresis_bars = max(1, int(hysteresis_bars or 1))
        self._history: Deque[Dict[str, float]] = deque(
            maxlen=max(self.long_lookback, int(max_history))
        )
        self._current = Regime.NEUTRAL
        self._confidence = 0.0
        self._pending: Optional[Regime] = None
        self._pending_count = 0

    @property
    def current(self) -> Regime:
        return self._current

    @property
    def confidence(self) -> float:
        return self._confidence

    def update(self, prices: Mapping[str, float]) -> Regime:
        clean = self._clean_prices(prices)
        if not clean:
            self._confidence = 0.0
            return self._current

        self._history.append(clean)
        short_returns = self._returns(self.lookback)
        if not short_returns:
            self._current = Regime.NEUTRAL
            self._confidence = 0.0
            return self._current

        long_returns = self._returns(self.long_lookback)
        proposed, confidence = self._classify(short_returns, long_returns)
        self._confidence = confidence
        self._current = self._apply_hysteresis(proposed)
        return self._current

    def _classify(
        self,
        short_returns: Mapping[str, float],
        long_returns: Mapping[str, float],
    ) -> tuple[Regime, float]:
        short_values = list(short_returns.values())
        if not short_values:
            return Regime.NEUTRAL, 0.0

        center = self._weighted_median(short_returns)
        long_center = self._weighted_median(long_returns) if long_returns else center
        trend = (center * 0.70) + (long_center * 0.30)
        worst_anchor = min(
            (short_returns[sym] for sym in self.anchor_symbols if sym in short_returns),
            default=1.0,
        )
        crash_breadth = (
            sum(1 for value in short_values if value <= self.crash_return)
            / max(len(short_values), 1)
        )
        if worst_anchor <= self.crash_return or crash_breadth >= self.min_crash_breadth:
            crash_pressure = max(abs(min(short_values)), abs(min(worst_anchor, 0.0)))
            confidence = max(
                min(1.0, crash_pressure / max(abs(self.crash_return), 1e-12)),
                crash_breadth,
            )
            return Regime.CRASH, confidence
        eps = 1e-12
        if trend >= self.bullish_return - eps:
            return Regime.BULLISH, self._bounded_confidence(abs(trend), abs(self.bullish_return))
        if trend <= self.bearish_return + eps:
            return Regime.BEARISH, self._bounded_confidence(abs(trend), abs(self.bearish_return))
        band = max(abs(self.bullish_return), abs(self.bearish_return), 1e-12)
        return Regime.NEUTRAL, max(0.1, 1.0 - min(1.0, abs(trend) / band))

    def _apply_hysteresis(self, proposed: Regime) -> Regime:
        if proposed == self._current:
            self._pending = None
            self._pending_count = 0
            return self._current
        if proposed == Regime.CRASH:
            self._pending = None
            self._pending_count = 0
            return proposed
        if self._pending == proposed:
            self._pending_count += 1
        else:
            self._pending = proposed
            self._pending_count = 1
        if self._pending_count >= self.hysteresis_bars:
            self._pending = None
            self._pending_count = 0
            return proposed
        return self._current

    def seed(self, rows: Iterable[Mapping[str, float]], *, limit: Optional[int] = None) -> Regime:
        selected = list(rows)
        if limit is not None and int(limit) > 0:
            selected = selected[-int(limit):]
        regime = self._current
        for row in selected:
            regime = self.update(row)
        return regime

    def _returns(self, lookback: int) -> Dict[str, float]:
        if len(self._history) < 2:
            return {}
        rows = list(self._history)
        lookback_rows = rows[-max(2, min(int(lookback), len(rows))):]
        first = lookback_rows[0]
        last = lookback_rows[-1]
        out: Dict[str, float] = {}
        for sym, last_price in last.items():
            first_price = first.get(sym)
            if first_price and first_price > 0 and last_price > 0:
                out[sym] = (last_price - first_price) / first_price
        return out

    def _weighted_median(self, returns: Mapping[str, float]) -> float:
        if not returns:
            return 0.0
        expanded: List[float] = []
        for sym, value in returns.items():
            weight = int(round(self.anchor_weight)) if sym in self.anchor_symbols else 1
            expanded.extend([float(value)] * max(1, weight))
        return median(expanded)

    @staticmethod
    def _clean_prices(prices: Mapping[str, float]) -> Dict[str, float]:
        clean: Dict[str, float] = {}
        for sym, price in (prices or {}).items():
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                clean[str(sym).upper()] = value
        return clean

    @staticmethod
    def _bounded_confidence(value: float, threshold: float) -> float:
        if threshold <= 0:
            return 1.0
        return max(0.1, min(1.0, value / (threshold * 3.0)))
