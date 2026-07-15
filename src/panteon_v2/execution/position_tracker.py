"""PositionTracker — учёт открытых позиций для парного close-расчёта.

Хранит open positions in-memory по `sym`. На close-trade автоматически
парит с открывающим signal_id. Эмиттит соответствующие events:
  • PositionOpened  — на open trade
  • PositionClosed  — на close trade (с realized_pnl)

Это нужно AttributionLedger в Phase 5 — он строит атрибуцию из этих
событий, не из stats.signals/trades разрозненно.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    open_action:    str = ""
    open_regime:    str = ""
    funding_open:   float = 0.0
    partial_profit_locked: bool = False
    stop_price:     float = 0.0
    stop_loss_pct:  float = 0.0


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

    def __init__(self, *, strict_close_action_family: bool = False) -> None:
        self._positions: Dict[str, TrackedPosition] = {}
        self._strict_close_action_family = bool(strict_close_action_family)

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def has(self, sym: str) -> bool:
        return sym in self._positions

    def get(self, sym: str) -> Optional[TrackedPosition]:
        return self._positions.get(sym)

    def all_open(self) -> Dict[str, TrackedPosition]:
        return dict(self._positions)

    def snapshot(self) -> Dict[str, dict]:
        """Return a JSON-safe snapshot of open tracked positions."""
        out: Dict[str, dict] = {}
        for sym, position in self._positions.items():
            opened_at = position.opened_at
            if isinstance(opened_at, datetime):
                opened_at_value = opened_at.isoformat()
            else:
                opened_at_value = str(opened_at or "")
            out[str(sym).upper()] = {
                "open_signal_id": int(position.open_signal_id),
                "symbol": str(position.sym).upper(),
                "sym": str(position.sym).upper(),
                "side": str(position.side),
                "qty": float(position.qty),
                "entry": float(position.entry_price),
                "entry_price": float(position.entry_price),
                "fee_open": float(position.fee_open),
                "by_player": str(position.by_player or ""),
                "by_agent": str(position.by_agent or ""),
                "opened_bar": int(position.opened_bar),
                "opened_at": opened_at_value,
                "open_action": str(position.open_action or ""),
                "open_regime": str(position.open_regime or ""),
                "funding_open": float(position.funding_open),
                "partial_profit_locked": bool(position.partial_profit_locked),
                "stop_price": float(position.stop_price),
                "stop_loss_pct": float(position.stop_loss_pct),
            }
        return out

    def restore(self, snapshot: dict) -> None:
        """Restore open positions from a snapshot produced by snapshot()."""
        self.clear()
        if not isinstance(snapshot, dict):
            return
        for raw_sym, raw_payload in snapshot.items():
            if not isinstance(raw_payload, dict):
                continue
            sym = str(
                raw_payload.get("sym")
                or raw_payload.get("symbol")
                or raw_sym
                or ""
            ).upper()
            if not sym:
                continue
            try:
                qty = float(raw_payload.get("qty") or 0.0)
                entry = float(
                    raw_payload.get("entry_price")
                    if raw_payload.get("entry_price") is not None
                    else raw_payload.get("entry")
                )
            except (TypeError, ValueError):
                continue
            if qty <= 0.0 or entry <= 0.0:
                continue
            side = str(raw_payload.get("side") or "long").lower()
            if side not in ("long", "short"):
                side = "long"
            self.force_set(TrackedPosition(
                open_signal_id=_safe_int(raw_payload.get("open_signal_id")),
                sym=sym,
                side=side,
                entry_price=entry,
                qty=qty,
                fee_open=_safe_float(raw_payload.get("fee_open")),
                by_player=str(raw_payload.get("by_player") or ""),
                by_agent=str(raw_payload.get("by_agent") or ""),
                opened_at=_parse_snapshot_datetime(raw_payload.get("opened_at")),
                opened_bar=_safe_int(raw_payload.get("opened_bar")),
                open_action=str(raw_payload.get("open_action") or ""),
                open_regime=str(raw_payload.get("open_regime") or ""),
                funding_open=_safe_float(raw_payload.get("funding_open")),
                partial_profit_locked=bool(raw_payload.get("partial_profit_locked", False)),
                stop_price=max(0.0, _safe_float(raw_payload.get("stop_price"))),
                stop_loss_pct=max(0.0, _safe_float(raw_payload.get("stop_loss_pct"))),
            ))

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
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
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
            open_action=signal.action.name,
            open_regime=signal.regime.label,
            funding_open=trade.funding,
            stop_price=max(0.0, _safe_float(metadata.get("stop_price"))),
            stop_loss_pct=max(0.0, _safe_float(metadata.get("stop_loss_pct"))),
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
            open_regime=signal.regime.label,
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
        opened = self._positions.get(sym)
        if opened is None:
            # Закрытие несуществующей позиции — без PnL события
            return []

        if (
            self._strict_close_action_family
            and not _close_matches_open_action(signal.action, opened.open_action)
        ):
            return []

        quantity_limited_close = _is_quantity_limited_close(signal)
        close_qty = (
            min(float(trade.qty), float(opened.qty))
            if quantity_limited_close
            else float(opened.qty)
        )
        if close_qty <= 0.0:
            return []
        open_fraction = close_qty / float(opened.qty)
        trade_fraction = close_qty / float(trade.qty) if quantity_limited_close else 1.0
        fee_open = opened.fee_open * open_fraction
        funding_open = opened.funding_open * open_fraction
        fee_close = trade.fee * trade_fraction
        funding_close = trade.funding * trade_fraction

        # Realized PnL
        if opened.side == "long":
            return_abs = (trade.fill_price - opened.entry_price) * close_qty
        else:
            return_abs = (opened.entry_price - trade.fill_price) * close_qty
        net_pnl = (
            return_abs
            - fee_open
            - fee_close
            - funding_open
            - funding_close
        )

        remaining_qty = float(opened.qty) - close_qty
        if not quantity_limited_close or remaining_qty <= 1e-12:
            self._positions.pop(sym, None)
        else:
            self._positions[sym] = replace(
                opened,
                qty=remaining_qty,
                fee_open=opened.fee_open - fee_open,
                funding_open=opened.funding_open - funding_open,
                partial_profit_locked=True,
            )

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
            qty=close_qty,
            realized_pnl=net_pnl,
            by_player=opened.by_player,    # атрибуция тому, кто ОТКРЫЛ
            by_agent=opened.by_agent,
            open_action=opened.open_action,
            open_regime=opened.open_regime,
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


def _is_intentional_partial_close(signal: Signal) -> bool:
    if not signal.action.is_close:
        return False
    try:
        return 0.0 < float(getattr(signal, "close_fraction", 1.0) or 1.0) < 1.0
    except (TypeError, ValueError):
        return False


def _is_quantity_limited_close(signal: Signal) -> bool:
    return signal.action == Action.SPOT_SELL_ALL or _is_intentional_partial_close(signal)


_SPOT_OPEN_ACTION_NAMES = frozenset({
    Action.SPOT_BUY_HALF.name,
    Action.SPOT_BUY_FULL.name,
})
_FUTURE_OPEN_ACTION_NAMES = frozenset({
    Action.FUT_LONG_HALF.name,
    Action.FUT_LONG_FULL.name,
    Action.FUT_SHORT_HALF.name,
    Action.FUT_SHORT_FULL.name,
})


def _close_matches_open_action(close_action: Action, open_action: str) -> bool:
    open_action_name = str(open_action or "")
    if not open_action_name:
        return True
    if close_action == Action.SPOT_SELL_ALL:
        return open_action_name in _SPOT_OPEN_ACTION_NAMES
    if close_action == Action.FUT_CLOSE_ALL:
        return open_action_name in _FUTURE_OPEN_ACTION_NAMES
    return True


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_snapshot_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raw = str(value or "").strip()
    if raw:
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)
