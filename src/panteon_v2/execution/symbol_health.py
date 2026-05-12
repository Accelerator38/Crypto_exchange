"""SymbolHealthMonitor — единый owner per-symbol health-state.

Заменяет в v1 разрозненные `pending_failures`, `blocked_until`,
`PENDING_FAILURE_LIMIT` в `PositionSyncHealth`. В v2 это полноценный
компонент в execution/, не догвет внутри health-объекта.

Гарантии:
  • Если за окно времени для символа произошло N подряд reject/pending →
    символ блокируется на duration. После duration снова открыт.
  • Любой успешный fill → счётчик failures для этого symbol сбрасывается.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional


@dataclass(frozen=True)
class SymbolHealthConfig:
    """Параметры health monitor."""

    failure_threshold:    int   = 3       # сколько подряд failures для блока
    failure_window_sec:   float = 3600.0  # окно подсчёта failures (rolling)
    block_duration_sec:   float = 3600.0  # длительность блока

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be ≥ 1")
        if self.failure_window_sec <= 0:
            raise ValueError("failure_window_sec must be > 0")
        if self.block_duration_sec <= 0:
            raise ValueError("block_duration_sec must be > 0")


DEFAULT_HEALTH_CONFIG = SymbolHealthConfig()


@dataclass(frozen=True)
class SymbolStatus:
    """Снимок состояния символа для дашбордов / диагностики."""

    sym:                str
    is_blocked:         bool
    failures_in_window: int
    blocked_until_ts:   Optional[float]   # unix timestamp когда снимется блок


class SymbolHealthMonitor:
    """Единый owner per-symbol health.

    Использование:
        health = SymbolHealthMonitor()
        # После каждой попытки отправить ордер:
        if not health.is_blocked(sym):
            result = exchange.send_order(...)
            if result.is_failure:
                blocked_now = health.record_pending_failure(sym)
                if blocked_now:
                    # эмитим SymbolBlocked event
            else:
                health.record_success(sym)
    """

    def __init__(
        self,
        *,
        config:  SymbolHealthConfig = DEFAULT_HEALTH_CONFIG,
        now_fn:  Optional[Callable[[], float]] = None,
    ):
        self._config = config
        import time
        self._now_fn: Callable[[], float] = now_fn or time.time
        # sym → deque[ts] of recent failures
        self._failures: Dict[str, Deque[float]] = {}
        # sym → unix ts when block expires (no entry = not blocked)
        self._blocked_until: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> SymbolHealthConfig:
        return self._config

    def now(self) -> float:
        return float(self._now_fn())

    # ── Mutation API ────────────────────────────────────────────────

    def record_pending_failure(self, sym: str) -> bool:
        """Отметить, что для sym произошёл reject/pending-timeout.

        Возвращает True, если этот вызов вызвал переход в "blocked".
        Используется callsite-ом для логирования / event эмиссии.
        """
        sym = self._normalize(sym)
        if not sym:
            return False
        ts = self.now()
        with self._lock:
            dq = self._failures.setdefault(sym, deque(maxlen=self._config.failure_threshold * 4))
            dq.append(ts)
            cutoff = ts - self._config.failure_window_sec
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._config.failure_threshold:
                already_blocked = (
                    sym in self._blocked_until
                    and self._blocked_until[sym] > ts
                )
                self._blocked_until[sym] = ts + self._config.block_duration_sec
                return not already_blocked
            return False

    def record_success(self, sym: str) -> None:
        """Sym успешно выполнил ордер → сбрасываем failures и блок."""
        sym = self._normalize(sym)
        if not sym:
            return
        with self._lock:
            self._failures.pop(sym, None)
            self._blocked_until.pop(sym, None)

    # ── Read API ────────────────────────────────────────────────────

    def is_blocked(self, sym: str) -> bool:
        sym = self._normalize(sym)
        if not sym:
            return False
        ts = self.now()
        with self._lock:
            until = self._blocked_until.get(sym)
            if until is None:
                return False
            if ts >= until:
                self._blocked_until.pop(sym, None)
                return False
            return True

    def status(self, sym: str) -> SymbolStatus:
        sym = self._normalize(sym)
        with self._lock:
            dq = self._failures.get(sym)
            failures = len(dq) if dq else 0
            until = self._blocked_until.get(sym)
        is_blocked = self.is_blocked(sym)  # auto-clears expired
        return SymbolStatus(
            sym=sym,
            is_blocked=is_blocked,
            failures_in_window=failures,
            blocked_until_ts=until if is_blocked else None,
        )

    def all_blocked(self) -> Dict[str, float]:
        """Список всех заблокированных символов (sym → seconds_remaining).

        Автоматически чистит истёкшие.
        """
        ts = self.now()
        out: Dict[str, float] = {}
        with self._lock:
            expired = []
            for sym, until in self._blocked_until.items():
                if ts >= until:
                    expired.append(sym)
                else:
                    out[sym] = round(until - ts, 1)
            for sym in expired:
                self._blocked_until.pop(sym, None)
        return out

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(sym: str) -> str:
        return str(sym or "").upper().strip()
