"""EventLog — append-only журнал всех событий системы.

Текущая реализация: in-memory + опциональный JSONL flush. В продакшене
заменяется на SQLite (см. PANTEON_V2_ARCHITECTURE.md §10.1).

Гарантии:
- append-only (нет API для удаления/модификации)
- thread-safe append (через lock)
- query() — детерминирована для одинаковых параметров
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import is_dataclass, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Type, Union

from .events import Event


class EventLogPersistenceError(RuntimeError):
    """Raised when an EventLog JSONL append cannot be persisted."""


class EventLog:
    """Append-only список событий с фильтрацией.

    Использование:
        log = EventLog()
        log.emit(BarStarted(bar=N, trace_id="MEXC-N"))
        log.emit(SignalEmitted(bar=N, trace_id="MEXC-N", signal=sig))
        for ev in log.query(trace_id="MEXC-N"):
            ...
    """

    def __init__(self, *, jsonl_path: Optional[Union[str, Path]] = None):
        self._events: List[Event] = []
        self._lock = threading.Lock()
        self._jsonl_path: Optional[Path] = (
            Path(jsonl_path) if jsonl_path else None
        )
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # touch создаст файл, если его нет
            self._jsonl_path.touch(exist_ok=True)

    # ── Основной API ───────────────────────────────────────────────

    def emit(self, event: Event) -> None:
        """Добавить событие в журнал. Thread-safe."""
        if not isinstance(event, Event):
            raise TypeError(
                f"EventLog.emit expects Event subclass, got {type(event).__name__}"
            )
        with self._lock:
            self._events.append(event)
            if self._jsonl_path is not None:
                self._append_jsonl(event)

    def emit_many(self, events: Iterable[Event]) -> None:
        """Удобство — пачкой."""
        with self._lock:
            for ev in events:
                if not isinstance(ev, Event):
                    raise TypeError(
                        f"EventLog.emit_many: non-Event in batch ({type(ev).__name__})"
                    )
                self._events.append(ev)
                if self._jsonl_path is not None:
                    self._append_jsonl(ev)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ── Запросы ────────────────────────────────────────────────────

    def query(
        self,
        *,
        trace_id:     Optional[str] = None,
        event_types:  Optional[Iterable[Type[Event]]] = None,
        after_bar:    Optional[int] = None,
        before_bar:   Optional[int] = None,
    ) -> Iterator[Event]:
        """Фильтруем events по любой комбинации параметров.

        Все параметры опциональны — если ничего не указано, возвращает
        ВСЕ события в порядке вставки.
        """
        with self._lock:
            snapshot = list(self._events)  # копия — чтобы итерация не блокировалась

        types_tuple = tuple(event_types) if event_types else None
        for ev in snapshot:
            if trace_id is not None and ev.trace_id != trace_id:
                continue
            if types_tuple is not None and not isinstance(ev, types_tuple):
                continue
            if after_bar is not None and ev.bar < after_bar:
                continue
            if before_bar is not None and ev.bar >= before_bar:
                continue
            yield ev

    def all(self) -> List[Event]:
        """Снимок всех событий (для тестов)."""
        with self._lock:
            return list(self._events)

    # ── Persistence ────────────────────────────────────────────────

    def _append_jsonl(self, event: Event) -> None:
        """Запись одного события в JSONL (вызывается под self._lock)."""
        try:
            payload = self._event_to_dict(event)
            path = Path(self._jsonl_path)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=_json_default) + "\n")
        except Exception as exc:
            raise EventLogPersistenceError(
                f"failed to append {type(event).__name__} to {self._jsonl_path}: {exc}"
            ) from exc

    @staticmethod
    def _event_to_dict(event: Event) -> dict:
        """Конвертирует event в dict для сериализации."""
        out = {"_type": type(event).__name__}
        if is_dataclass(event):
            for f in fields(event):
                val = getattr(event, f.name)
                out[f.name] = val
        return out

    def flush(self) -> None:
        """Принудительный sync на диск (no-op для line-buffered JSONL)."""
        # JSONL пишется построчно; sync OS-кэша на наше усмотрение, не делаем.
        pass


# ────────────────────────────────────────────────────────────────────
# JSON helpers
# ────────────────────────────────────────────────────────────────────


def _json_default(obj):
    """Default-сериализация для типов, не известных json."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value if hasattr(obj, "value") else str(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    if is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)
