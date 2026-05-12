"""AgentSelector — выбирает top-k агентов для текущего режима.

Гарантия Q1 (carantine consistency):
  АgentSelector НИКОГДА не возвращает карантинного агента в результирующем
  списке. Фильтрация делается ДО скоринга — невозможно случайно выбрать
  карантинного. Это устраняет все корни проблем v1 P1-P4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from ..domain.types import Metrics, Regime
from ..memory import PerformanceMemory, QuarantineManager
from ..scoring import DEFAULT_SCORING, ScoringConfig, regime_score
from .agent import Agent, AgentRegistry


# ────────────────────────────────────────────────────────────────────
# ScoredAgent — результат селекции
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredAgent:
    """Агент + его текущий score + метрики.

    Используется в Player.agents (для исполнения) и в Strategist
    (для verifying что агенты не в карантине).
    """

    agent:    Agent
    score:    float
    metrics:  Metrics

    @property
    def label(self) -> str:
        return self.agent.label


# ────────────────────────────────────────────────────────────────────
# Scorer-функция: тип для зависимости в Selector
# ────────────────────────────────────────────────────────────────────


# Callable signature: (Metrics, Regime) -> float
ScorerFn = Callable[[Metrics, Regime], float]


def _default_scorer_factory(config: ScoringConfig) -> ScorerFn:
    def scorer(metrics: Metrics, regime: Regime) -> float:
        return regime_score(metrics, regime, config=config)
    return scorer


# ────────────────────────────────────────────────────────────────────
# AgentSelector
# ────────────────────────────────────────────────────────────────────


class AgentSelector:
    """Выбирает top-k агентов для конкретного режима рынка.

    Использование:
        sel = AgentSelector(registry, perf, qm)
        top = sel.select(Regime.BULLISH, k=5)
        # top = [ScoredAgent, ...] — все НЕ в карантине, отсортированы по score↓

    Параметры:
      • registry: AgentRegistry — каталог зарегистрированных агентов
      • perf:     PerformanceMemory — источник метрик
      • qm:       QuarantineManager — источник карантина
      • config:   ScoringConfig — параметры скоринга
      • scorer:   опциональная замена скоринговой функции (для тестов)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        perf:     PerformanceMemory,
        qm:       QuarantineManager,
        *,
        config:   ScoringConfig = DEFAULT_SCORING,
        scorer:   Optional[ScorerFn] = None,
    ):
        self._registry = registry
        self._perf = perf
        self._qm = qm
        self._config = config
        self._scorer: ScorerFn = scorer or _default_scorer_factory(config)

    def select(
        self,
        regime: Regime,
        k: int,
        *,
        min_score: Optional[float] = None,
    ) -> List[ScoredAgent]:
        """Возвращает top-k агентов с самым высоким score для регима.

        Гарантии:
          • ни один результат не находится в карантине (проверка ДО scoring)
          • отсортированы по score↓
          • возвращены только агенты с score > min_score
            (по умолчанию config.min_eligible_score)

        Если k > eligible — возвращает столько, сколько есть.
        """
        if k <= 0:
            return []

        threshold = (
            float(min_score)
            if min_score is not None
            else float(self._config.min_eligible_score)
        )

        scored: List[ScoredAgent] = []
        for label in self._registry.all_labels():
            # SYNCHRONIZATION POINT: фильтруем карантин ДО скоринга.
            if self._qm.is_quarantined(label):
                continue
            agent = self._registry.get(label)
            if agent is None:  # parano
                continue
            metrics = self._perf.get(label, regime=regime)
            score = float(self._scorer(metrics, regime))
            if score > threshold:
                scored.append(ScoredAgent(agent=agent, score=score, metrics=metrics))

        scored.sort(key=lambda sa: -sa.score)
        return scored[:k]

    def select_with_fallback(
        self,
        regime: Regime,
        k: int,
        *,
        fallback_threshold: float = -10.0,
    ) -> List[ScoredAgent]:
        """Как select, но если ни один агент не прошёл базовый порог,
        мы всё равно возвращаем top-k (с пониженным cutoff) чтобы система
        не оставалась без сигналов.

        Используется в Strategist для cold-start ситуации.
        """
        result = self.select(regime, k)
        if result:
            return result
        return self.select(regime, k, min_score=fallback_threshold)
