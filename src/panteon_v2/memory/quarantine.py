"""QuarantineManager — единый источник о карантине.

Заменяет в v1:
- `Panteon.LIVE_AGENT_BLOCKLIST` (frozenset на классе + копии у наследников)
- `Panteon.AGENT_QUARANTINE`
- `Panteon._SEED_AGENT_QUARANTINE`
- propagation на shadow-варианты через ad-hoc loops

→ ОДИН owner. Любой компонент v2 спрашивает `is_quarantined(label)`.
Изменения публикуются через subscribe/notify.

Гарантия Q1 (carantine consistency):
    На любом баре N, если QM.is_quarantined("X") = True, то ни один
    Player.agents не содержит X. Доказательство: Selector.select() —
    единственный конструктор Player.agents, и он фильтрует через QM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional, Set

from ..scoring import (
    DEFAULT_SCORING,
    ScoringConfig,
    is_hopeless_in_all_regimes,
    is_locally_proven,
)
from .performance import PerformanceMemory


# ────────────────────────────────────────────────────────────────────
# Результат пересчёта — для логирования и дебага
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecomputeResult:
    """Что произошло при `QuarantineManager.recompute(...)`."""

    added:    FrozenSet[str]
    removed:  FrozenSet[str]
    current:  FrozenSet[str]

    @property
    def is_no_op(self) -> bool:
        return len(self.added) == 0 and len(self.removed) == 0


# Тип callback-а для observers
QuarantineObserver = Callable[[FrozenSet[str], RecomputeResult], None]


# ────────────────────────────────────────────────────────────────────
# QuarantineManager
# ────────────────────────────────────────────────────────────────────


class QuarantineManager:
    """Один owner blocklist-а карантина.

    Жизненный цикл:
        qm = QuarantineManager(seed={"FundingArb", "RichardDennis"})
        qm.subscribe(my_callback)
        # ... со временем ...
        result = qm.recompute(perf)   # PerformanceMemory
        # observers получили result + новый current
    """

    def __init__(
        self,
        seed: Iterable[str] = (),
        *,
        config: ScoringConfig = DEFAULT_SCORING,
    ):
        self._seed: FrozenSet[str] = frozenset(seed)
        self._dynamic: Set[str] = set(self._seed)
        self._observers: List[QuarantineObserver] = []
        self._config: ScoringConfig = config
        self._lock = threading.Lock()

    # ── Чтение ───────────────────────────────────────────────────────

    def is_quarantined(self, label: str) -> bool:
        if not label:
            return False
        with self._lock:
            return label in self._dynamic

    def all_quarantined(self) -> FrozenSet[str]:
        """Immutable снимок текущего набора."""
        with self._lock:
            return frozenset(self._dynamic)

    @property
    def seed(self) -> FrozenSet[str]:
        return self._seed

    # ── Recompute ────────────────────────────────────────────────────

    def recompute(
        self,
        perf: PerformanceMemory,
        *,
        config: Optional[ScoringConfig] = None,
    ) -> RecomputeResult:
        """Пересчитать карантин на основе PerformanceMemory.

        Логика (порядок имеет значение):
          1. Каждый label из `seed` стартово в карантине, кроме тех, кто
             накопил positive regime experience (`is_locally_proven`).
          2. Каждый label из perf проверяется на `is_hopeless_in_all_regimes`.
             Если да — добавляем в карантин.
          3. Каждый label, ранее в карантине динамически (не из seed),
             проверяется на `is_locally_proven` — если да, выпускаем.

        Возвращает RecomputeResult (added/removed/current). Если изменения
        непустые — оповещает observers.
        """
        cfg = config or self._config

        with self._lock:
            previous = set(self._dynamic)
            new_set: Set[str] = set()

            # 1. Seed: для каждого seed-label решаем оставить или выпустить
            all_labels = set(self._seed) | set(perf.all_labels())
            for label in all_labels:
                per_regime = perf.per_regime_for_label(label)
                proven = is_locally_proven(per_regime, config=cfg)
                hopeless = is_hopeless_in_all_regimes(per_regime, config=cfg)

                if label in self._seed:
                    # Стартово в карантине; выходит только если proven
                    if proven:
                        # выпускаем
                        continue
                    new_set.add(label)
                else:
                    # Не в seed — добавляется только если hopeless
                    # (даже без явного proven, но без явных доказательств
                    # карантина не трогаем).
                    if hopeless:
                        new_set.add(label)

            self._dynamic = new_set
            added = frozenset(new_set - previous)
            removed = frozenset(previous - new_set)
            result = RecomputeResult(
                added=added,
                removed=removed,
                current=frozenset(new_set),
            )

        # Notify observers ВНЕ lock-а (callback может звать обратно).
        if not result.is_no_op:
            self._notify(result)
        return result

    # ── Observers ────────────────────────────────────────────────────

    def subscribe(self, callback: QuarantineObserver) -> None:
        """Подписать callback, который вызывается при каждом изменении.

        callback(current_set, recompute_result) — current_set immutable,
        recompute_result содержит added/removed.
        """
        with self._lock:
            self._observers.append(callback)

    def unsubscribe(self, callback: QuarantineObserver) -> bool:
        with self._lock:
            try:
                self._observers.remove(callback)
                return True
            except ValueError:
                return False

    def _notify(self, result: RecomputeResult) -> None:
        # Snapshot observers под lock-ом, вызываем без lock-а
        with self._lock:
            observers = list(self._observers)
            current_snapshot = frozenset(self._dynamic)
        for cb in observers:
            try:
                cb(current_snapshot, result)
            except Exception:
                # observer не должен ломать рантайм
                pass

    # ── Manual ops (для тестов и initialization) ─────────────────────

    def force_quarantine(self, label: str) -> None:
        """Принудительно добавить в карантин (минуя recompute).

        Используется только для bootstrap или тестов.
        """
        with self._lock:
            if label not in self._dynamic:
                self._dynamic.add(label)
                result = RecomputeResult(
                    added=frozenset({label}),
                    removed=frozenset(),
                    current=frozenset(self._dynamic),
                )
            else:
                return
        self._notify(result)

    def force_release(self, label: str) -> None:
        """Принудительно снять с карантина (минуя recompute)."""
        with self._lock:
            if label in self._dynamic:
                self._dynamic.discard(label)
                result = RecomputeResult(
                    added=frozenset(),
                    removed=frozenset({label}),
                    current=frozenset(self._dynamic),
                )
            else:
                return
        self._notify(result)
