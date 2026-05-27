from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from panteon_v2.domain.types import TechnicalIndicators


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.rsi_period <= 1:
            raise ValueError("rsi_period must be > 1")
        if self.macd_fast_period <= 1:
            raise ValueError("macd_fast_period must be > 1")
        if self.macd_slow_period <= self.macd_fast_period:
            raise ValueError("macd_slow_period must be greater than macd_fast_period")
        if self.macd_signal_period <= 1:
            raise ValueError("macd_signal_period must be > 1")
        if self.atr_period <= 1:
            raise ValueError("atr_period must be > 1")


@dataclass
class _SymbolState:
    closes: Deque[float]
    true_ranges: Deque[float]
    prev_close: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    macd_signal: Optional[float] = None


@dataclass
class TechnicalIndicatorState:
    config: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    _symbols: dict[str, _SymbolState] = field(default_factory=dict)

    def update_symbol(
        self,
        symbol: str,
        *,
        high: float,
        low: float,
        close: float,
    ) -> TechnicalIndicators:
        clean = str(symbol or "").upper()
        if not clean:
            raise ValueError("symbol must be non-empty")
        high_f = _finite_positive(high, "high")
        low_f = _finite_positive(low, "low")
        close_f = _finite_positive(close, "close")
        if high_f < low_f:
            raise ValueError("high must be >= low")

        maxlen = max(
            self.config.rsi_period + 1,
            self.config.atr_period,
            self.config.macd_slow_period + self.config.macd_signal_period + 1,
        )
        state = self._symbols.get(clean)
        if state is None:
            state = _SymbolState(
                closes=deque(maxlen=maxlen),
                true_ranges=deque(maxlen=self.config.atr_period),
            )
            self._symbols[clean] = state

        tr = _true_range(high_f, low_f, state.prev_close)
        state.true_ranges.append(tr)
        state.closes.append(close_f)
        state.ema_fast = _ema_update(
            state.ema_fast,
            close_f,
            self.config.macd_fast_period,
        )
        state.ema_slow = _ema_update(
            state.ema_slow,
            close_f,
            self.config.macd_slow_period,
        )

        macd_line_pct = None
        macd_signal_pct = None
        macd_histogram_pct = None
        if state.ema_fast is not None and state.ema_slow is not None:
            macd_line = state.ema_fast - state.ema_slow
            state.macd_signal = _ema_update(
                state.macd_signal,
                macd_line,
                self.config.macd_signal_period,
            )
            macd_line_pct = macd_line / close_f * 100.0
            if state.macd_signal is not None:
                macd_signal_pct = state.macd_signal / close_f * 100.0
                macd_histogram_pct = (
                    (macd_line - state.macd_signal) / close_f * 100.0
                )

        state.prev_close = close_f
        return TechnicalIndicators(
            rsi_14=_rsi(list(state.closes), self.config.rsi_period),
            macd_line_pct=macd_line_pct,
            macd_signal_pct=macd_signal_pct,
            macd_histogram_pct=macd_histogram_pct,
            atr_14_pct=_atr_pct(state.true_ranges, close_f, self.config.atr_period),
        )


def _finite_positive(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _ema_update(previous: Optional[float], value: float, period: int) -> float:
    if previous is None:
        return float(value)
    alpha = 2.0 / (float(period) + 1.0)
    return float(previous) + alpha * (float(value) - float(previous))


def _rsi(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1) :]
    gains = []
    losses = []
    for previous, current in zip(window, window[1:]):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / float(period)
    avg_loss = sum(losses) / float(period)
    if avg_loss <= 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _atr_pct(
    true_ranges: Deque[float],
    close: float,
    period: int,
) -> Optional[float]:
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges) / float(period)
    return atr / close * 100.0
