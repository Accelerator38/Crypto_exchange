"""Event types для EventLog.

Все события — frozen dataclasses, наследуются от базового `Event`.
Каждое событие имеет `bar`, `timestamp`, `trace_id`. Это позволяет
прогнать сессию через `EventLog.query(trace_id=...)` и увидеть все
решения для конкретного бара.

События — единственный канал коммуникации между компонентами в части
"что произошло". Никаких глобальных stats-объектов с ad-hoc полями
(как `stats.signals.append(...)` в v1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import FrozenSet, Optional, Tuple

from ..domain.types import Action, Regime, Signal, Trade


# ────────────────────────────────────────────────────────────────────
# Базовый класс события
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    """Базовый event. Все наследники — frozen dataclasses.

    `trace_id` объединяет несколько событий одного бара / решения;
    позволяет запрашивать "всё, что произошло в этом trace".
    """

    bar:       int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id:  str = ""    # обычно "{exchange}-{bar}" или uuid

    @property
    def event_type(self) -> str:
        return type(self).__name__


# ────────────────────────────────────────────────────────────────────
# Lifecycle events
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BarStarted(Event):
    pass


@dataclass(frozen=True)
class BarEnded(Event):
    pass


# ────────────────────────────────────────────────────────────────────
# Decision events
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegimeDetected(Event):
    """Регим определён (или подтверждён) на этом баре."""

    regime:     Regime = Regime.NEUTRAL
    from_regime: Regime = Regime.NEUTRAL  # предыдущий устойчивый регим
    confidence: float  = 1.0
    is_change:  bool   = False


@dataclass(frozen=True)
class QuarantineRecomputed(Event):
    """QuarantineManager пересчитал blocklist."""

    added:   FrozenSet[str] = frozenset()
    removed: FrozenSet[str] = frozenset()
    current: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class LeaderSelected(Event):
    """Strategist выбрал текущего лидера."""

    player_label: str = ""
    previous_label: str = ""  # пустой если первый раз
    score:         float = 0.0
    margin:        float = 0.0   # на сколько лучше предыдущего
    is_urgent:     bool  = False
    reason:        str   = ""
    decision_id:   str   = ""


@dataclass(frozen=True)
class DecisionStarted(Event):
    """Start of a leader-selection decision for forensic reconstruction."""

    decision_id: str = ""
    exchange:    str = ""
    symbol:      str = ""
    timeframe:   str = ""
    mode:        str = ""
    run_id:      str = ""
    session_id:  str = ""
    candidate_labels: Tuple[str, ...] = ()
    shadow_total_signals: int = 0
    shadow_total_filled: int = 0
    shadow_total_rejected: int = 0
    shadow_total_blocked: int = 0


@dataclass(frozen=True)
class CandidateScored(Event):
    """One candidate score row from the full selector candidate list."""

    decision_id: str = ""
    player_label: str = ""
    rank: int = 0
    score: float = 0.0
    score_source: str = ""
    selected_by_pantheon: bool = False
    has_data: bool = False
    closed_trades: int = 0
    signals: int = 0
    execution_failures: int = 0
    uncertainty_penalty: float = 0.0
    memory_keys_read: Tuple[str, ...] = ()
    agent_labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRejected(Event):
    """Candidate filtered out before scoring."""

    decision_id: str = ""
    player_label: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ShadowActorUpdated(Event):
    """Current-bar shadow trading update that fed virtual performance memory."""

    actor_label: str = ""
    actor_type: str = ""
    signals: int = 0
    filled: int = 0
    rejected: int = 0
    blocked: int = 0


@dataclass(frozen=True)
class AgentVoteFailed(Event):
    """Agent failed while producing a vote/action for a bar."""

    player_label: str = ""
    agent_label:  str = ""
    reason:       str = ""


@dataclass(frozen=True)
class PlayerVoteFailed(Event):
    """Player failed while producing real or shadow signals for a bar."""

    player_label: str = ""
    reason:       str = ""


# ────────────────────────────────────────────────────────────────────
# Signal/Order/Trade events
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalEmitted(Event):
    """Игрок сгенерировал сигнал."""

    signal: Optional[Signal] = None  # Optional только для дефолтного значения dataclass


@dataclass(frozen=True)
class OrderSent(Event):
    """Ордер отправлен на биржу."""

    signal_id: int = -1
    sym:       str = ""
    action:    Action = Action.HOLD
    exchange_order_id: str = ""


@dataclass(frozen=True)
class OrderFilled(Event):
    """Биржа подтвердила исполнение."""

    trade: Optional[Trade] = None


@dataclass(frozen=True)
class OrderRejected(Event):
    """Биржа отклонила ордер или таймаут pending."""

    signal_id: int  = -1
    sym:       str  = ""
    reason:    str  = ""


# ────────────────────────────────────────────────────────────────────
# Position events — для AttributionLedger
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryUpdateFailed(Event):
    """Filled trade was recorded, but PerformanceMemory update failed."""

    signal_id: int = -1
    sym:       str = ""
    reason:    str = ""


@dataclass(frozen=True)
class PositionOpened(Event):
    """Открыта новая позиция (после OrderFilled)."""

    signal_id: int   = -1
    sym:       str   = ""
    side:      str   = "long"     # "long" | "short"
    entry:     float = 0.0
    qty:       float = 0.0


@dataclass(frozen=True)
class PositionClosed(Event):
    """Позиция закрыта. Содержит realized PnL для AttributionLedger."""

    open_signal_id:  int   = -1
    close_signal_id: int   = -1
    sym:             str   = ""
    side:            str   = "long"
    entry:           float = 0.0
    exit:            float = 0.0
    qty:             float = 0.0
    realized_pnl:    float = 0.0   # net of fees
    by_player:       str   = ""    # с открывающего сигнала
    by_agent:        str   = ""


# ────────────────────────────────────────────────────────────────────
# Health events
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolBlocked(Event):
    """SymbolHealthMonitor заблокировал символ из-за повторных failures."""

    sym:               str = ""
    failures_in_window: int = 0
    blocked_until_ts:  float = 0.0
    reason:            str = "pending_failures_threshold"
