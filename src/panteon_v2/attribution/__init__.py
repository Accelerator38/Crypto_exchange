"""Attribution layer — append-only журнал решений и проекция в PnL.

EventLog хранит все решения системы (пуш). AttributionLedger строит
из него детерминированную проекцию (пулл) — кто что заработал.

Это даёт:
  • REAL attribution на 100% (Trade.signal_id обязателен)
  • Replay через EventLog.replay()
  • Decision log с trace_id для любого "почему?"
"""

from .events import (
    Event,
    BarStarted,
    BarEnded,
    RegimeDetected,
    QuarantineRecomputed,
    LeaderSelected,
    DecisionStarted,
    CandidateScored,
    CandidateRejected,
    ShadowActorUpdated,
    AgentVoteFailed,
    PlayerVoteFailed,
    SignalEmitted,
    OrderSent,
    OrderFilled,
    OrderRejected,
    MemoryUpdateFailed,
    PositionOpened,
    PositionClosed,
    SymbolBlocked,
)
from .event_log import EventLog, EventLogPersistenceError
from .ledger import Attribution, AttributionLedger

__all__ = [
    # Events
    "Event",
    "BarStarted",
    "BarEnded",
    "RegimeDetected",
    "QuarantineRecomputed",
    "LeaderSelected",
    "DecisionStarted",
    "CandidateScored",
    "CandidateRejected",
    "ShadowActorUpdated",
    "AgentVoteFailed",
    "PlayerVoteFailed",
    "SignalEmitted",
    "OrderSent",
    "OrderFilled",
    "OrderRejected",
    "MemoryUpdateFailed",
    "PositionOpened",
    "PositionClosed",
    "SymbolBlocked",
    # Storage
    "EventLog",
    "EventLogPersistenceError",
    # Ledger
    "Attribution",
    "AttributionLedger",
]
