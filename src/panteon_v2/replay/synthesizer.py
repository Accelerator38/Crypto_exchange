"""Synthesizer — конвертация V1 данных в state v2 компонентов.

Выход: tuple (event_log, perf, qm, ledger), готовый для validation/render.

Стратегия:
  • Из V1Signal-ов строим Signal+Trade+PositionOpened/Closed events.
  • Из leaderboard-статусов восстанавливаем seed для QuarantineManager.
  • PerformanceMemory заполняется через тот же update_from_trade.
  • AttributionLedger.replay_from_event_log в конце даёт реальную картину.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..attribution import (
    AttributionLedger,
    EventLog,
    OrderFilled,
    OrderSent,
    PositionClosed,
    PositionOpened,
)
from ..domain.types import Action, Regime, Signal, Trade
from ..memory import PerformanceMemory, QuarantineManager
from .v1_parser import V1Session, V1Signal


@dataclass(frozen=True)
class SynthState:
    """Полное состояние v2 после синтеза из v1-сессии."""

    event_log:  EventLog
    perf:       PerformanceMemory
    qm:         QuarantineManager
    ledger:     AttributionLedger
    n_open:     int
    n_close:    int
    n_orphan_close: int   # close без матчующего open (данные неполные)


# ────────────────────────────────────────────────────────────────────


def _signal_from_v1(v1sig: V1Signal, sid: int, default_regime: str) -> Optional[Signal]:
    """V1Signal → v2 Signal. None если action не торговое."""
    if v1sig.action_code == 0 or v1sig.price <= 0 or not v1sig.sym:
        return None
    try:
        action = Action(v1sig.action_code)
    except ValueError:
        return None
    regime = Regime.from_string(v1sig.regime or default_regime)
    by_player = (v1sig.selected_player or "").replace("V_", "") or "Unknown"
    by_agent = (v1sig.selected_agents or "").replace("V_", "") or by_player
    return Signal(
        id=sid,
        bar=v1sig.bar,
        sym=v1sig.sym,
        action=action,
        price=v1sig.price,
        regime=regime,
        by_player=by_player,
        by_agent=by_agent,
        risk_mult=v1sig.risk_multiplier,
        timestamp=v1sig.timestamp or datetime.now(timezone.utc),
    )


def _synth_trade_from_signal(
    signal: Signal,
    *,
    qty: float = 1.0,
    fee_rate: float = 0.0006,
) -> Trade:
    """Синтетический Trade из Signal (фактическая цена = price из сигнала).

    Используем qty=1.0 — это нормализация. AttributionLedger всё равно
    корректно посчитает realized_pnl × qty. Если v1 предоставил реальный
    qty — можно усложнить, но для replay-валидации этого достаточно.
    """
    side = signal.action.side
    if not side:
        # close — определим из контекста (long по умолчанию для replay)
        side = "long"
    notional = qty * signal.fill_price_or(signal.price) if hasattr(signal, "fill_price_or") else qty * signal.price
    fee = notional * fee_rate
    return Trade(
        signal_id=signal.id,
        bar=signal.bar,
        sym=signal.sym,
        side=side,
        qty=qty,
        fill_price=signal.price,
        fee=fee,
        funding=0.0,
        exchange_order_id=f"v1-replay-{signal.id}",
        timestamp=signal.timestamp,
    )


def synthesize_v2_state(session: V1Session) -> SynthState:
    """Главный entry-point. Перегоняет V1Session в state v2.

    Возвращает SynthState — все ключевые компоненты заполнены, готовы
    к validation/rendering.
    """
    event_log = EventLog()
    perf = PerformanceMemory(trade_fraction=1.0)

    # Стартовое состояние QM — карантинные из v1-leaderboard.
    # Это не «настоящий seed v2», но даёт точку отсчёта для replay.
    qm = QuarantineManager(seed=set(session.quarantined))

    # Открытые позиции по sym (для парного close-расчёта)
    open_pos: Dict[str, Tuple[Signal, Trade]] = {}
    n_open = 0
    n_close = 0
    n_orphan_close = 0

    sid_counter = 1
    for v1sig in session.real_signals:
        v2sig = _signal_from_v1(v1sig, sid_counter, session.current_regime)
        sid_counter += 1
        if v2sig is None:
            continue

        # Синтетический trade
        v2trade = Trade(
            signal_id=v2sig.id,
            bar=v2sig.bar,
            sym=v2sig.sym,
            side=v2sig.action.side or (
                open_pos[v2sig.sym][1].side if v2sig.sym in open_pos else "long"
            ),
            qty=1.0,
            fill_price=v2sig.price,
            fee=v2sig.price * 1.0 * 0.0006,
            funding=0.0,
            exchange_order_id=f"v1-replay-{v2sig.id}",
            timestamp=v2sig.timestamp,
        )

        # Open vs close events
        if v2sig.action.is_open:
            if v2sig.sym in open_pos:
                # дубль open на занятый sym — игнорируем
                continue
            open_pos[v2sig.sym] = (v2sig, v2trade)
            event_log.emit(PositionOpened(
                bar=v2sig.bar, timestamp=v2sig.timestamp,
                trace_id=f"replay-{v2sig.id}",
                signal_id=v2sig.id,
                sym=v2sig.sym, side=v2trade.side,
                entry=v2trade.fill_price, qty=v2trade.qty,
            ))
            event_log.emit(OrderFilled(
                bar=v2sig.bar, timestamp=v2sig.timestamp,
                trace_id=f"replay-{v2sig.id}",
                trade=v2trade,
            ))
            perf.update_from_trade(v2trade, v2sig)
            n_open += 1

        elif v2sig.action.is_close:
            opened = open_pos.pop(v2sig.sym, None)
            if opened is None:
                n_orphan_close += 1
                continue
            open_sig, open_trade = opened
            # Realized PnL
            if open_trade.side == "long":
                pnl_abs = (v2trade.fill_price - open_trade.fill_price) * open_trade.qty
            else:
                pnl_abs = (open_trade.fill_price - v2trade.fill_price) * open_trade.qty
            net_pnl = pnl_abs - open_trade.fee - v2trade.fee
            event_log.emit(PositionClosed(
                bar=v2sig.bar, timestamp=v2sig.timestamp,
                trace_id=f"replay-{v2sig.id}",
                open_signal_id=open_sig.id,
                close_signal_id=v2sig.id,
                sym=v2sig.sym, side=open_trade.side,
                entry=open_trade.fill_price, exit=v2trade.fill_price,
                qty=open_trade.qty, realized_pnl=net_pnl,
                by_player=open_sig.by_player,
                by_agent=open_sig.by_agent,
            ))
            event_log.emit(OrderFilled(
                bar=v2sig.bar, timestamp=v2sig.timestamp,
                trace_id=f"replay-{v2sig.id}",
                trade=v2trade,
            ))
            perf.update_from_trade(v2trade, v2sig)
            n_close += 1

    # Replay через AttributionLedger
    ledger = AttributionLedger()
    ledger.replay_from_event_log(event_log)

    return SynthState(
        event_log=event_log,
        perf=perf,
        qm=qm,
        ledger=ledger,
        n_open=n_open,
        n_close=n_close,
        n_orphan_close=n_orphan_close,
    )
