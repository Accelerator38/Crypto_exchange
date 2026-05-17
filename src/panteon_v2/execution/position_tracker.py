"""PositionTracker — учёт открытых позиций для парного close-расчёта.

Хранит open positions in-memory по `sym`. На close-trade автоматически
парит с открывающим signal_id. Эмиттит соответствующие events:
  • PositionOpened  — на open trade
  • PositionClosed  — на close trade (с realized_pnl)

Это нужно AttributionLedger в Phase 5 — он строит атрибуцию из этих
событий, не из stats.signals/trades разрозненно.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..attribution.events import PositionClosed, PositionOpened
from ..domain.types import Action, Signal, Trade


@dataclass(frozen=True)
class TrackedPosition:
    """Открытая позиция, отслеживаемая трекером."""

    open_signal_id: int
    sym:            str
    side:           str       # "long" | "short"
    entry_price:    float
    qty:            float
    fee_open:       float
    by_player:      str
    by_agent:       str
    opened_at:      datetime
    opened_bar:     int = 0


class PositionTracker:
    """Mutable owner state открытых позиций.

    Использование:
        tracker = PositionTracker()
        # На fill open:
        events = tracker.on_open(signal=open_sig, trade=open_trade)
        # На fill close:
        events = tracker.on_close(signal=close_sig, trade=close_trade)
        # events — список Event-ов, которые EventLog должен записать
    """

    def __init__(self) -> None:
        self._positions: Dict[str, TrackedPosition] = {}

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def has(self, sym: str) -> bool:
        return sym in self._positions

    def get(self, sym: str) -> Optional[TrackedPosition]:
        return self._positions.get(sym)

    def all_open(self) -> Dict[str, TrackedPosition]:
        return dict(self._positions)

    # ── Mutation ────────────────────────────────────────────────────

    def on_open(
        self,
        *,
        signal: Signal,
        trade:  Trade,
    ) -> List[object]:
        """Регистрирует открытие позиции и возвращает event(ы) для лога."""
        if not signal.action.is_open:
            return []
        if signal.action.side != trade.side:
            # Защита от рассинхрона
            pass  # доверяем trade.side как actual
        sym = trade.sym
        if sym in self._positions:
            # Существующая позиция остаётся (новый open signal на занятом sym
            # должен был быть отфильтрован RiskLimits). Игнорируем.
            return []
        pos = TrackedPosition(
            open_signal_id=signal.id,
            sym=sym,
            side=trade.side,
            entry_price=trade.fill_price,
            qty=trade.qty,
            fee_open=trade.fee,
            by_player=signal.by_player,
            by_agent=signal.by_agent,
            opened_at=trade.timestamp,
            opened_bar=signal.bar,
        )
        self._positions[sym] = pos
        return [PositionOpened(
            bar=signal.bar,
            timestamp=trade.timestamp,
            trace_id=f"{sym}-{signal.id}",
            signal_id=signal.id,
            sym=sym,
            side=trade.side,
            entry=trade.fill_price,
            qty=trade.qty,
        )]

    def on_close(
        self,
        *,
        signal: Signal,
        trade:  Trade,
    ) -> List[object]:
        """Регистрирует закрытие позиции и возвращает event(ы) с realized_pnl."""
        if not signal.action.is_close:
            return []
        sym = trade.sym
        opened = self._positions.pop(sym, None)
        if opened is None:
            # Закрытие несуществующей позиции — без PnL события
            return []

        # Realized PnL
        if opened.side == "long":
            return_abs = (trade.fill_price - opened.entry_price) * opened.qty
        else:
            return_abs = (opened.entry_price - trade.fill_price) * opened.qty
        net_pnl = return_abs - opened.fee_open - trade.fee

        return [PositionClosed(
            bar=signal.bar,
            timestamp=trade.timestamp,
            trace_id=f"{sym}-{signal.id}",
            open_signal_id=opened.open_signal_id,
            close_signal_id=signal.id,
            sym=sym,
            side=opened.side,
            entry=opened.entry_price,
            exit=trade.fill_price,
            qty=opened.qty,
            realized_pnl=net_pnl,
            by_player=opened.by_player,    # атрибуция тому, кто ОТКРЫЛ
            by_agent=opened.by_agent,
        )]

    # ── Reconcile (для startup и периодической сверки с биржей) ────

    def force_set(self, position: TrackedPosition) -> None:
        """Принудительно установить позицию (например, при инициализации
        системы из реальных данных биржи). Используется в Phase 8 при
        recovery после рестарта."""
        self._positions[position.sym] = position

    def force_remove(self, sym: str) -> Optional[TrackedPosition]:
        return self._positions.pop(sym, None)

    def clear(self) -> None:
        self._positions.clear()
