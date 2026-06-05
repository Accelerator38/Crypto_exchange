"""Order lifecycle ledger for real exchange execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from ..domain.types import Action, Regime, Signal, Trade
from .exchange import OrderResult, OrderStatus


class OrderStage(Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OrderRecord:
    signal_id: int
    sym: str
    action: str
    stage: OrderStage
    exchange_order_id: str = ""
    message: str = ""
    signal: Optional[Signal] = None
    trade: Optional[Trade] = None
    execution_key: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderLedger:
    """Append-friendly in-memory order state with snapshot/restore support."""

    def __init__(self) -> None:
        self._records: Dict[str, OrderRecord] = {}
        self._execution_keys: Dict[str, str] = {}

    def record_submitted(self, signal: Signal, *, execution_key: str = "") -> OrderRecord:
        key = self._signal_key(signal)
        record = OrderRecord(
            signal_id=signal.id,
            sym=signal.sym,
            action=signal.action.name,
            stage=OrderStage.SUBMITTED,
            signal=signal,
            execution_key=str(execution_key or ""),
        )
        self._records[key] = record
        self._remember_execution_key(record, key)
        return record

    def record_result(
        self,
        signal: Signal,
        result: OrderResult,
        *,
        execution_key: str = "",
    ) -> OrderRecord:
        stage = _stage_from_status(result.status)
        key = result.exchange_order_id or self._signal_key(signal)
        previous = self._records.get(key) or self._records.get(self._signal_key(signal))
        created_at = previous.created_at if previous is not None else datetime.now(timezone.utc)
        resolved_execution_key = (
            str(execution_key or "")
            or (previous.execution_key if previous is not None else "")
        )
        record = OrderRecord(
            signal_id=signal.id,
            sym=result.sym or signal.sym,
            action=signal.action.name,
            stage=stage,
            exchange_order_id=result.exchange_order_id,
            message=result.message,
            signal=(previous.signal if previous is not None and previous.signal is not None else signal),
            trade=result.trade,
            execution_key=resolved_execution_key,
            created_at=created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._records[key] = record
        if key != self._signal_key(signal):
            self._records.pop(self._signal_key(signal), None)
        self._remember_execution_key(record, key)
        return record

    def get(self, key: str) -> Optional[OrderRecord]:
        return self._records.get(str(key))

    def has_execution_key(self, execution_key: str) -> bool:
        return str(execution_key or "") in self._execution_keys

    def pending_records(self) -> list[OrderRecord]:
        return [
            record
            for record in self._records.values()
            if record.stage in {
                OrderStage.SUBMITTED,
                OrderStage.ACCEPTED,
                OrderStage.PARTIALLY_FILLED,
            }
        ]

    def expire_pending(self, *, max_age_sec: float, now: Optional[datetime] = None) -> list[OrderRecord]:
        current = now or datetime.now(timezone.utc)
        expired: list[OrderRecord] = []
        for record in list(self.pending_records()):
            age_sec = (current - record.created_at).total_seconds()
            if age_sec < float(max_age_sec):
                continue
            expired.append(self.mark_timed_out(record))
        return expired

    def mark_timed_out(
        self,
        record: OrderRecord,
        *,
        message: str = "pending timeout",
    ) -> OrderRecord:
        key = record.exchange_order_id or f"signal:{record.signal_id}"
        updated = OrderRecord(
            signal_id=record.signal_id,
            sym=record.sym,
            action=record.action,
            stage=OrderStage.TIMED_OUT,
            exchange_order_id=record.exchange_order_id,
            message=message,
            signal=record.signal,
            trade=record.trade,
            execution_key=record.execution_key,
            created_at=record.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self._records[key] = updated
        self._remember_execution_key(updated, key)
        return updated

    def snapshot(self) -> dict:
        return {
            key: {
                "signal_id": record.signal_id,
                "sym": record.sym,
                "action": record.action,
                "stage": record.stage.value,
                "exchange_order_id": record.exchange_order_id,
                "message": record.message,
                "execution_key": record.execution_key,
                "signal": _signal_to_dict(record.signal),
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
            }
            for key, record in self._records.items()
        }

    def restore(self, snapshot: dict) -> None:
        self._records.clear()
        self._execution_keys.clear()
        for key, payload in (snapshot or {}).items():
            try:
                stage = OrderStage(str(payload.get("stage")))
            except Exception:
                stage = OrderStage.SUBMITTED
            self._records[str(key)] = OrderRecord(
                signal_id=int(payload.get("signal_id", 0) or 0),
                sym=str(payload.get("sym", "") or ""),
                action=str(payload.get("action", "") or ""),
                stage=stage,
                exchange_order_id=str(payload.get("exchange_order_id", "") or ""),
                message=str(payload.get("message", "") or ""),
                signal=_signal_from_dict(payload.get("signal")),
                execution_key=str(payload.get("execution_key", "") or ""),
                created_at=_parse_ts(payload.get("created_at")),
                updated_at=_parse_ts(payload.get("updated_at")),
            )
            self._remember_execution_key(self._records[str(key)], str(key))

    @staticmethod
    def _signal_key(signal: Signal) -> str:
        return f"signal:{signal.id}"

    def _remember_execution_key(self, record: OrderRecord, record_key: str) -> None:
        execution_key = str(record.execution_key or "")
        if execution_key:
            self._execution_keys[execution_key] = str(record_key)


def _stage_from_status(status: OrderStatus) -> OrderStage:
    if status == OrderStatus.FILLED:
        return OrderStage.FILLED
    if status == OrderStatus.PENDING:
        return OrderStage.ACCEPTED
    return OrderStage.REJECTED


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.now(timezone.utc)


def _signal_to_dict(signal: Optional[Signal]) -> Optional[dict]:
    if signal is None:
        return None
    return {
        "id": signal.id,
        "bar": signal.bar,
        "sym": signal.sym,
        "action": signal.action.name,
        "price": signal.price,
        "regime": signal.regime.label,
        "by_player": signal.by_player,
        "by_agent": signal.by_agent,
        "position_scope": signal.position_scope,
        "risk_mult": signal.risk_mult,
        "close_fraction": signal.close_fraction,
        "timestamp": signal.timestamp.isoformat(),
        "metadata": dict(getattr(signal, "metadata", {}) or {}),
    }


def _signal_from_dict(payload) -> Optional[Signal]:
    if not isinstance(payload, dict):
        return None
    try:
        action_raw = payload.get("action", Action.HOLD.name)
        if isinstance(action_raw, str) and action_raw in Action.__members__:
            action = Action[action_raw]
        else:
            action = Action(int(action_raw))
        return Signal(
            id=int(payload.get("id", 0) or 0),
            bar=int(payload.get("bar", 0) or 0),
            sym=str(payload.get("sym", "") or ""),
            action=action,
            price=float(payload.get("price", 0.0) or 0.0),
            regime=Regime.from_string(str(payload.get("regime", "neutral") or "neutral")),
            by_player=str(payload.get("by_player", "") or "RecoveredOrder"),
            by_agent=str(payload.get("by_agent", "") or ""),
            position_scope=str(payload.get("position_scope", "") or ""),
            risk_mult=float(payload.get("risk_mult", 1.0) or 1.0),
            close_fraction=float(payload.get("close_fraction", 1.0) or 1.0),
            timestamp=_parse_ts(payload.get("timestamp")),
            metadata=dict(payload.get("metadata") or {}),
        )
    except Exception:
        return None
