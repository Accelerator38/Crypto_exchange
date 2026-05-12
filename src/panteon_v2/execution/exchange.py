"""Exchange Protocol — абстракция над биржевыми коннекторами.

В v1 BITGET и MEXC connector-ы были тесно связаны с runtime (в `Panteon_Trade`
жил bridge, в exchange_api_runtime — API логика). Это создавало hidden
coupling.

В v2 Exchange — это **Protocol**, а адаптеры существующих v1-connector-ов
будут жить за пределами этого пакета (или в отдельных модулях phase 8).
TradeExecutor работает только через интерфейс Exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Protocol, runtime_checkable

from ..domain.types import Signal, Trade


# ────────────────────────────────────────────────────────────────────
# Order result types
# ────────────────────────────────────────────────────────────────────


class OrderStatus(Enum):
    """Статус ордера со стороны биржи (после отправки)."""

    FILLED   = "filled"      # биржа исполнила, есть fill_price
    PENDING  = "pending"     # принято, но ещё не исполнено (limit/timeout)
    REJECTED = "rejected"    # биржа отвергла (мин. размер, недостаток маржи и т.п.)


@dataclass(frozen=True)
class OrderResult:
    """Что вернула биржа после send_order.

    Если status=FILLED — есть trade с заполненной ценой.
    Если status=PENDING — exchange_order_id может быть, но trade=None
                          (TradeExecutor должен пытаться poll или таймаутить).
    Если status=REJECTED — есть message с причиной.
    """

    status:            OrderStatus
    signal_id:         int
    sym:               str
    exchange_order_id: str = ""
    trade:             Optional[Trade] = None
    message:           str = ""

    def __post_init__(self) -> None:
        if self.status == OrderStatus.FILLED and self.trade is None:
            raise ValueError("OrderResult.FILLED requires non-None trade")
        if self.trade is not None and self.trade.signal_id != self.signal_id:
            raise ValueError(
                f"trade.signal_id ({self.trade.signal_id}) != "
                f"signal_id ({self.signal_id})"
            )


# ────────────────────────────────────────────────────────────────────
# Position read-model
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExchangePosition:
    """Позиция как её видит биржа (для синхронизации)."""

    sym:        str
    side:       str       # "long" | "short"
    qty:        float
    entry:      float
    leverage:   int = 1
    unrealized_pnl: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Exchange Protocol
# ────────────────────────────────────────────────────────────────────


@runtime_checkable
class Exchange(Protocol):
    """Структурный интерфейс биржевого коннектора.

    Реализуется адаптерами над v1-connector-ами (BITGET, MEXC).
    Для тестов есть FakeExchange (см. ниже).
    """

    name: str  # "BITGET" | "MEXC" | "FAKE" и т.п.

    def send_order(self, signal: Signal, *, qty: float) -> OrderResult:
        """Отправить ордер на биржу. qty — фактический объём в монетах
        (TradeExecutor рассчитывает его из risk_limits и signal.action.fraction)."""
        ...

    def get_position(self, sym: str) -> Optional[ExchangePosition]:
        """Текущая позиция по символу (None если нет)."""
        ...

    def get_all_positions(self) -> Dict[str, ExchangePosition]:
        """Все открытые позиции."""
        ...

    def get_min_notional(self, sym: str) -> float:
        """Минимальный размер позиции в USDT для биржи (для pre-flight)."""
        ...


# ────────────────────────────────────────────────────────────────────
# FakeExchange — для тестов
# ────────────────────────────────────────────────────────────────────


class FakeExchange:
    """Идеальная биржа для unit-тестов.

    По умолчанию каждый ордер исполняется по signal.price. Можно
    переопределить через configure_*:
      • set_reject(sym) — следующий ордер по этому sym будет REJECTED
      • set_pending(sym, count) — следующие N ордеров будут PENDING
      • set_min_notional(sym, value) — для pre-flight тестов
      • set_slippage_pct(value) — fill_price = signal.price * (1 + slip)
    """

    def __init__(self, *, name: str = "FAKE"):
        self.name = name
        self._positions: Dict[str, ExchangePosition] = {}
        self._next_orders: Dict[str, List[OrderStatus]] = {}  # forced statuses
        self._min_notional: Dict[str, float] = {}
        self._slippage: float = 0.0
        self._fee_rate: float = 0.0006
        # Idempotency / orders log (для тестов)
        self._orders_log: List[OrderResult] = []

    # ── Configurations for tests ─────────────────────────────────────

    def configure_reject(self, sym: str, count: int = 1) -> None:
        self._next_orders.setdefault(sym, []).extend(
            [OrderStatus.REJECTED] * count
        )

    def configure_pending(self, sym: str, count: int = 1) -> None:
        self._next_orders.setdefault(sym, []).extend(
            [OrderStatus.PENDING] * count
        )

    def set_min_notional(self, sym: str, value: float) -> None:
        self._min_notional[sym] = float(value)

    def set_slippage_pct(self, value: float) -> None:
        self._slippage = float(value)

    def set_fee_rate(self, rate: float) -> None:
        self._fee_rate = float(rate)

    @property
    def orders_log(self) -> List[OrderResult]:
        return list(self._orders_log)

    # ── Exchange Protocol ───────────────────────────────────────────

    def send_order(self, signal: Signal, *, qty: float) -> OrderResult:
        # Forced status?
        forced_queue = self._next_orders.get(signal.sym)
        forced = forced_queue.pop(0) if forced_queue else None

        if forced == OrderStatus.REJECTED:
            result = OrderResult(
                status=OrderStatus.REJECTED,
                signal_id=signal.id,
                sym=signal.sym,
                message="forced reject (test)",
            )
            self._orders_log.append(result)
            return result
        if forced == OrderStatus.PENDING:
            result = OrderResult(
                status=OrderStatus.PENDING,
                signal_id=signal.id,
                sym=signal.sym,
                exchange_order_id=f"FAKE-PENDING-{signal.id}",
                message="forced pending (test)",
            )
            self._orders_log.append(result)
            return result

        # Normal fill
        side = signal.action.side
        if not side:  # close action — определим из существующей позиции
            existing = self._positions.get(signal.sym)
            if existing is None and signal.action.is_close:
                # Closing inexistent position
                result = OrderResult(
                    status=OrderStatus.REJECTED,
                    signal_id=signal.id,
                    sym=signal.sym,
                    message="no position to close",
                )
                self._orders_log.append(result)
                return result
            if existing is not None:
                side = existing.side
            else:
                side = "long"  # для HOLD не должно вызываться

        fill_price = signal.price * (
            1.0 + self._slippage if signal.action.is_long_open
            else 1.0 - self._slippage if signal.action.is_short_open
            else 1.0
        )
        notional = qty * fill_price
        fee = notional * self._fee_rate

        trade = Trade(
            signal_id=signal.id,
            bar=signal.bar,
            sym=signal.sym,
            side=side,
            qty=qty,
            fill_price=fill_price,
            fee=fee,
            funding=0.0,
            exchange_order_id=f"FAKE-{signal.id}",
            timestamp=datetime.now(timezone.utc),
        )

        # Обновляем internal positions
        if signal.action.is_open:
            self._positions[signal.sym] = ExchangePosition(
                sym=signal.sym, side=side, qty=qty, entry=fill_price,
            )
        elif signal.action.is_close:
            self._positions.pop(signal.sym, None)

        result = OrderResult(
            status=OrderStatus.FILLED,
            signal_id=signal.id,
            sym=signal.sym,
            exchange_order_id=f"FAKE-{signal.id}",
            trade=trade,
        )
        self._orders_log.append(result)
        return result

    def get_position(self, sym: str) -> Optional[ExchangePosition]:
        return self._positions.get(sym)

    def get_all_positions(self) -> Dict[str, ExchangePosition]:
        return dict(self._positions)

    def get_min_notional(self, sym: str) -> float:
        return self._min_notional.get(sym, 0.0)
