"""Core immutable domain types для Panteon v2.

Принципы:
- Все типы — frozen dataclasses или Enum (immutable).
- Действия — IntEnum с явными свойствами (`is_open`, `side`), а не
  магические числа `(1, 2, 4, 5)`.
- Signal и Trade связаны через обязательное поле `signal_id` в Trade.
- Metrics — read-only view, win_rate / pnl_per_trade — computed properties.

Это база для всех остальных компонентов v2. Любая v1-несовместимость
решается на уровне адаптеров (см. Phase 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

# ────────────────────────────────────────────────────────────────────
# Action — типизированные торговые действия
# ────────────────────────────────────────────────────────────────────


class Action(IntEnum):
    """Торговое действие.

    Числа сохраняются совместимыми с v1, чтобы адаптерам было проще,
    но используются как Enum: ``action.is_open``, ``action.side``.
    Никаких ``if action in (1, 2, 4, 5)`` в коде v2.
    """

    HOLD            = 0
    SPOT_BUY_HALF   = 1
    SPOT_BUY_FULL   = 2
    SPOT_SELL_ALL   = 3
    FUT_LONG_HALF   = 4
    FUT_LONG_FULL   = 5
    FUT_SHORT_HALF  = 6
    FUT_SHORT_FULL  = 7
    FUT_CLOSE_ALL   = 8

    # ── Семантические свойства ──────────────────────────────────────

    @property
    def is_hold(self) -> bool:
        return self == Action.HOLD

    @property
    def is_open(self) -> bool:
        """Действие открывает новую позицию (любого направления)."""
        return self in _OPENS

    @property
    def is_close(self) -> bool:
        """Действие закрывает существующую позицию."""
        return self in _CLOSES

    @property
    def is_long_open(self) -> bool:
        return self in _LONG_OPENS

    @property
    def is_short_open(self) -> bool:
        return self in _SHORT_OPENS

    @property
    def side(self) -> str:
        """Сторона позиции, которую инициирует действие.

        Возвращает пустую строку для HOLD/CLOSE — у них нет своей стороны
        (close не привязан к стороне без контекста позиции).
        """
        if self in _LONG_OPENS:
            return "long"
        if self in _SHORT_OPENS:
            return "short"
        return ""

    @property
    def is_fraction_full(self) -> bool:
        """True для full-size действий (vs half)."""
        return self in (
            Action.SPOT_BUY_FULL,
            Action.FUT_LONG_FULL,
            Action.FUT_SHORT_FULL,
        )

    @property
    def fraction(self) -> float:
        """Доля от max-position size (0.5 для half, 1.0 для full, 0 иначе)."""
        if self.is_fraction_full:
            return 1.0
        if self in (Action.SPOT_BUY_HALF, Action.FUT_LONG_HALF, Action.FUT_SHORT_HALF):
            return 0.5
        return 0.0


# Frozen sets для быстрых проверок без копий. Build один раз на module load.
_LONG_OPENS:   frozenset["Action"] = frozenset({
    Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL,
    Action.FUT_LONG_HALF, Action.FUT_LONG_FULL,
})
_SHORT_OPENS:  frozenset["Action"] = frozenset({
    Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL,
})
_OPENS:   frozenset["Action"] = _LONG_OPENS | _SHORT_OPENS
_CLOSES:  frozenset["Action"] = frozenset({
    Action.SPOT_SELL_ALL, Action.FUT_CLOSE_ALL,
})


# ────────────────────────────────────────────────────────────────────
# Regime — состояние рынка
# ────────────────────────────────────────────────────────────────────


class Regime(IntEnum):
    """Канонические режимы рынка.

    Все детекторы возвращают одно из этих значений.
    Никаких "unknown", "sideways", "bull" - единая канонизация.
    """

    BULLISH = 1
    BEARISH = 2
    NEUTRAL = 3
    CRASH   = 4
    RANGE_LOW_VOL = 5
    CHOPPY_DOWN = 6
    CHOPPY_UP = 7
    MIXED_ROTATIONAL = 8

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def from_string(cls, value: str) -> "Regime":
        """Канонизация любых строковых вариантов в один из известных режимов."""
        v = (value or "").strip().lower()
        # Прямой матч
        for regime in cls:
            if v == regime.label:
                return regime
        # Алиасы из v1
        if v in ("bull", "up", "uptrend"):
            return cls.BULLISH
        if v in ("bear", "down", "downtrend"):
            return cls.BEARISH
        if v in ("flat", "sideways", "range", "unknown", ""):
            return cls.NEUTRAL
        if v in ("crash", "panic", "flash_crash"):
            return cls.CRASH
        normalized = v.replace("-", "_").replace(" ", "_").replace("/", "_")
        if normalized in ("range_low_vol", "low_vol_range", "low_vol"):
            return cls.RANGE_LOW_VOL
        if normalized in ("choppy_down", "chop_down", "volatile_down"):
            return cls.CHOPPY_DOWN
        if normalized in ("choppy_up", "chop_up", "volatile_up"):
            return cls.CHOPPY_UP
        if normalized in ("mixed_rotational", "mixed_rotation", "rotational", "mixed"):
            return cls.MIXED_ROTATIONAL
        return cls.NEUTRAL


# ────────────────────────────────────────────────────────────────────
# MarketSnapshot — что видит игрок на одном баре
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TechnicalIndicators:
    """Per-symbol normalized technical context for a closed market bar."""

    rsi_14: Optional[float] = None
    macd_line_pct: Optional[float] = None
    macd_signal_pct: Optional[float] = None
    macd_histogram_pct: Optional[float] = None
    atr_14_pct: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "rsi_14",
            "macd_line_pct",
            "macd_signal_pct",
            "macd_histogram_pct",
            "atr_14_pct",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"TechnicalIndicators.{name} must be finite")
            object.__setattr__(self, name, parsed)
        if self.rsi_14 is not None and not 0.0 <= self.rsi_14 <= 100.0:
            raise ValueError("TechnicalIndicators.rsi_14 must be in [0, 100]")
        if self.atr_14_pct is not None and self.atr_14_pct < 0.0:
            raise ValueError("TechnicalIndicators.atr_14_pct must be >= 0")


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable снимок состояния рынка для одного bar-а.

    Передаётся в `Player.vote(market)` — игрок не должен мутировать
    его. Передача snapshot-ов вместо ссылок на shared state — одна
    из ключевых гарантий v2.
    """

    bar:        int
    timestamp:  datetime
    regime:     Regime
    prices:     Dict[str, float]                     # sym → spot/futures price
    volumes:    Dict[str, float]                     # sym → 1m volume
    regime_confidence: float = 1.0
    funding:    Dict[str, float] = field(default_factory=dict)  # sym → funding rate
    month:      Optional[int] = None                # для seasonality агентов

    fees_bps_by_symbol: Dict[str, float] = field(default_factory=dict)
    lookback_returns_pct: Dict[str, Dict[int, float]] = field(default_factory=dict)
    lookback_volatility_pct: Dict[str, Dict[int, float]] = field(default_factory=dict)
    technicals_by_symbol: Dict[str, TechnicalIndicators] = field(default_factory=dict)
    regimes_by_symbol: Dict[str, Regime] = field(default_factory=dict)
    regime_features_by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        regime = (
            self.regime
            if isinstance(self.regime, Regime)
            else Regime.from_string(str(self.regime or ""))
        )
        object.__setattr__(self, "regime", regime)
        if not 0.0 <= float(self.regime_confidence) <= 1.0:
            raise ValueError("MarketSnapshot.regime_confidence must be in [0, 1]")
        price_symbols = {str(item).upper() for item in self.prices}
        for symbol in self.technicals_by_symbol:
            if str(symbol).upper() not in price_symbols:
                raise ValueError(
                    "MarketSnapshot.technicals_by_symbol contains unknown symbol "
                    f"{symbol!r}"
                )
        clean_regimes: Dict[str, Regime] = {}
        for symbol, raw_regime in (self.regimes_by_symbol or {}).items():
            clean_symbol = str(symbol or "").strip().upper()
            if not clean_symbol:
                continue
            if clean_symbol not in price_symbols:
                raise ValueError(
                    "MarketSnapshot.regimes_by_symbol contains unknown symbol "
                    f"{symbol!r}"
                )
            clean_regimes[clean_symbol] = (
                raw_regime
                if isinstance(raw_regime, Regime)
                else Regime.from_string(str(raw_regime or ""))
            )
        object.__setattr__(self, "regimes_by_symbol", clean_regimes)
        clean_features: Dict[str, Dict[str, Any]] = {}
        for symbol, raw_features in (self.regime_features_by_symbol or {}).items():
            clean_symbol = str(symbol or "").strip().upper()
            if not clean_symbol:
                continue
            if clean_symbol not in price_symbols:
                raise ValueError(
                    "MarketSnapshot.regime_features_by_symbol contains unknown "
                    f"symbol {symbol!r}"
                )
            if isinstance(raw_features, dict):
                clean_features[clean_symbol] = dict(raw_features)
        object.__setattr__(self, "regime_features_by_symbol", clean_features)

    def has_price(self, sym: str) -> bool:
        return sym in self.prices and self.prices[sym] > 0

    def regime_for_symbol(self, sym: str) -> Regime:
        return self.regimes_by_symbol.get(str(sym or "").strip().upper(), self.regime)

    def regime_features_for_symbol(self, sym: str) -> Dict[str, Any]:
        return dict(self.regime_features_by_symbol.get(str(sym or "").strip().upper(), {}))

    def with_regime_for_symbol(self, sym: str) -> "MarketSnapshot":
        regime = self.regime_for_symbol(sym)
        if regime == self.regime:
            return self
        return replace(self, regime=regime)


# ────────────────────────────────────────────────────────────────────
# Signal — выходное решение игрока
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Signal:
    """Сигнал на торговое действие.

    `id` — уникальный ID сигнала (нужен для связи с Trade'ами).
    `by_player` и `by_agent` — кто принял решение и какой агент инициировал.
    """

    id:         int
    bar:        int
    sym:        str
    action:     Action
    price:      float                                # ожидаемая цена входа/выхода
    regime:     Regime
    by_player:  str                                  # имя текущего лидера
    by_agent:   str = ""                             # имя агента-инициатора (если ensemble)
    position_scope: str = ""                         # namespace для real/shadow позиций
    vote_weights: Dict[str, float] = field(default_factory=dict)
    vote_actions: Dict[str, Action] = field(default_factory=dict)
    risk_mult:  float = 1.0
    close_fraction: float = 1.0
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError(f"Signal.id must be non-negative, got {self.id}")
        if not self.sym:
            raise ValueError("Signal.sym must be non-empty")
        if self.price < 0:
            raise ValueError(f"Signal.price must be non-negative, got {self.price}")
        if not self.by_player:
            raise ValueError("Signal.by_player must be non-empty")
        if "|" in self.position_scope:
            raise ValueError("Signal.position_scope must not contain '|'")
        if not 0.0 < float(self.close_fraction) <= 1.0:
            raise ValueError("Signal.close_fraction must be in (0, 1]")


# ────────────────────────────────────────────────────────────────────
# Trade — фактическое исполнение
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    """Исполнение сделки на бирже.

    Поле `signal_id` ОБЯЗАТЕЛЬНОЕ — это ключевая гарантия v2:
    каждая trade жёстко связана с сигналом, который её инициировал.
    Это устраняет проблему v1 P16: signals и trades разрозненны.
    """

    signal_id:        int
    bar:              int
    sym:              str
    side:             str                            # "long" | "short"
    qty:              float
    fill_price:       float
    fee:              float
    funding:          float = 0.0
    exchange_order_id: Optional[str] = None
    timestamp:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.signal_id < 0:
            raise ValueError(
                f"Trade.signal_id must be non-negative (no orphan trades), "
                f"got {self.signal_id}"
            )
        if self.side not in ("long", "short"):
            raise ValueError(f"Trade.side must be 'long' or 'short', got {self.side!r}")
        if self.qty <= 0:
            raise ValueError(f"Trade.qty must be positive, got {self.qty}")
        if self.fill_price <= 0:
            raise ValueError(f"Trade.fill_price must be positive, got {self.fill_price}")

    @property
    def notional(self) -> float:
        return self.qty * self.fill_price


# ────────────────────────────────────────────────────────────────────
# Metrics — read-only метрики агента/игрока
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Metrics:
    """Производительность агента/игрока за период (общий или per-regime).

    Все числа — calculated, никаких "магических" обновлений на месте.
    Создаётся в PerformanceMemory.get(...). Для модификации — replace.
    """

    pnl_pct:        float = 0.0                      # совокупный % PnL за период
    closed_trades:  int   = 0
    entries:        int   = 0
    signals:        int   = 0
    wins:           int   = 0
    losses:         int   = 0
    sharpe:         float = 0.0
    max_dd_pct:     float = 0.0                      # максимальная просадка в %
    blocked_signals: int = 0
    rejected_signals: int = 0
    pending_signals: int = 0
    execution_failures: int = 0
    pnl_gross_pct: float = 0.0
    fee_pct: float = 0.0
    funding_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.closed_trades < 0 or self.entries < 0 or self.signals < 0:
            raise ValueError(f"Metrics counts must be non-negative: {self}")
        if self.wins < 0 or self.losses < 0:
            raise ValueError(f"Metrics wins/losses must be non-negative: {self}")
        if self.wins + self.losses > self.closed_trades:
            raise ValueError(
                f"wins + losses ({self.wins + self.losses}) > "
                f"closed_trades ({self.closed_trades}): {self}"
            )
        if (
            self.blocked_signals < 0
            or self.rejected_signals < 0
            or self.pending_signals < 0
            or self.execution_failures < 0
        ):
            raise ValueError(f"execution counters must be non-negative: {self}")

    # ── Computed properties (никакой мутации) ──────────────────────

    @property
    def win_rate(self) -> float:
        """Процент выигрышных сделок (0..100). Если нет закрытых — 0."""
        if self.closed_trades <= 0:
            return 0.0
        return self.wins / self.closed_trades * 100.0

    @property
    def pnl_per_trade(self) -> float:
        """Средний % PnL на одну закрытую сделку."""
        if self.closed_trades <= 0:
            return 0.0
        return self.pnl_pct / self.closed_trades

    @property
    def pnl_net_pct(self) -> float:
        """Net PnL percentage; pnl_pct is already net after fees and funding."""
        return self.pnl_pct

    @property
    def trading_cost_pct(self) -> float:
        """Explicit trading cost percentage; negative funding is a rebate."""
        return self.fee_pct + self.funding_pct

    @property
    def has_data(self) -> bool:
        """True если есть хоть какая-то торговая активность."""
        return (
            self.signals > 0
            or self.entries > 0
            or self.closed_trades > 0
            or self.execution_failures > 0
        )

    # ── Удобный конструктор для тестов и заглушек ──────────────────

    @classmethod
    def empty(cls) -> "Metrics":
        return cls()


# ────────────────────────────────────────────────────────────────────
# Тип-алиас для удобства
# ────────────────────────────────────────────────────────────────────


ScoredAgent = Tuple[str, float]  # (label, score) — упрощённое представление
