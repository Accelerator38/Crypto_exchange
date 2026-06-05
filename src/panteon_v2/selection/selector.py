"""AgentSelector — выбирает top-k агентов для текущего режима.

Гарантия Q1 (carantine consistency):
  АgentSelector НИКОГДА не возвращает карантинного агента в результирующем
  списке. Фильтрация делается ДО скоринга — невозможно случайно выбрать
  карантинного. Это устраняет все корни проблем v1 P1-P4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class SessionOverlayConfig:
    """Optional current-session overlay for selector scores.

    The baseline is captured lazily from restored memory on the first select()
    call, so persisted all-time memory does not masquerade as current-session
    performance.
    """

    enabled: bool = False
    overlay_weight: float = 1.00
    stale_penalty: float = 0.15
    underperformance_weight: float = 0.75
    session_pnl_cap_pct: float = 3.0
    min_session_activity: int = 1

    def __post_init__(self) -> None:
        if self.min_session_activity < 0:
            raise ValueError("min_session_activity must be >= 0")
        if self.session_pnl_cap_pct < 0:
            raise ValueError("session_pnl_cap_pct must be >= 0")


def _metrics_delta(current: Metrics, baseline: Metrics) -> Metrics:
    closed_trades = max(0, current.closed_trades - baseline.closed_trades)
    raw_wins = max(0, current.wins - baseline.wins)
    raw_losses = max(0, current.losses - baseline.losses)
    losses = min(raw_losses, closed_trades)
    wins = min(raw_wins, max(0, closed_trades - losses))
    return Metrics(
        pnl_pct=current.pnl_pct - baseline.pnl_pct,
        closed_trades=closed_trades,
        entries=max(0, current.entries - baseline.entries),
        signals=max(0, current.signals - baseline.signals),
        wins=wins,
        losses=losses,
        max_dd_pct=max(0.0, current.max_dd_pct - baseline.max_dd_pct),
        blocked_signals=max(0, current.blocked_signals - baseline.blocked_signals),
        rejected_signals=max(0, current.rejected_signals - baseline.rejected_signals),
        pending_signals=max(0, current.pending_signals - baseline.pending_signals),
        execution_failures=max(0, current.execution_failures - baseline.execution_failures),
    )


def _session_activity(metrics: Metrics) -> int:
    return (
        int(metrics.closed_trades)
        + int(metrics.entries)
        + int(metrics.signals)
        + int(metrics.execution_failures)
    )


def _cap_abs(value: float, cap: float) -> float:
    if cap <= 0:
        return float(value)
    return max(-cap, min(cap, float(value)))


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
        session_overlay: SessionOverlayConfig = SessionOverlayConfig(),
    ):
        self._registry = registry
        self._perf = perf
        self._qm = qm
        self._config = config
        self._scorer: ScorerFn = scorer or _default_scorer_factory(config)
        self._session_overlay = session_overlay
        self._session_baseline: Dict[Tuple[str, Regime], Metrics] = {}

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
        self._ensure_session_baseline(regime)
        for label in self._registry.all_labels():
            # SYNCHRONIZATION POINT: фильтруем карантин ДО скоринга.
            row = self.score_registered(
                label,
                regime,
                include_quarantined=False,
                ensure_baseline=False,
            )
            if row is None:
                continue
            if row.score > threshold:
                scored.append(row)

        scored.sort(
            key=lambda sa: (
                -sa.score,
                -sa.metrics.closed_trades,
                -sa.metrics.signals,
                sa.label,
            )
        )
        return scored[:k]

    def select_with_fallback(
        self,
        regime: Regime,
        k: int,
        *,
        fallback_threshold: float = -10.0,
        min_count: int = 1,
    ) -> List[ScoredAgent]:
        """Как select, но если базовый порог дал меньше нужного числа агентов,
        мы всё равно возвращаем top-k (с пониженным cutoff) чтобы система
        не оставалась без сигналов.

        Используется в Strategist для cold-start ситуации.
        """
        result = self.select(regime, k)
        if len(result) >= max(1, int(min_count)):
            return result
        return self.select(regime, k, min_score=fallback_threshold)

    def score_registered(
        self,
        label: str,
        regime: Regime,
        *,
        include_quarantined: bool = False,
        ensure_baseline: bool = True,
    ) -> Optional[ScoredAgent]:
        """Score one registered agent label."""
        if not include_quarantined and self._qm.is_quarantined(label):
            return None
        agent = self._registry.get(label)
        if agent is None:
            return None
        if ensure_baseline:
            self._ensure_session_baseline(regime)
        metrics = self._perf.get(label, regime=regime)
        score = float(self._scorer(metrics, regime))
        score = self._apply_session_overlay(label, regime, metrics, score)
        return ScoredAgent(agent=agent, score=score, metrics=metrics)

    def capture_session_baseline(self, regime: Optional[Regime] = None) -> None:
        """Capture current memory as the start-of-session baseline."""
        regimes = [regime] if regime is not None else list(Regime)
        for reg in regimes:
            for label in self._registry.all_labels():
                self._session_baseline[(label, reg)] = self._perf.get(label, regime=reg)

    def _ensure_session_baseline(self, regime: Regime) -> None:
        if not self._session_overlay.enabled:
            return
        for label in self._registry.all_labels():
            key = (label, regime)
            if key not in self._session_baseline:
                self._session_baseline[key] = self._perf.get(label, regime=regime)

    def _apply_session_overlay(
        self,
        label: str,
        regime: Regime,
        metrics: Metrics,
        base_score: float,
    ) -> float:
        overlay = self._session_overlay
        if not overlay.enabled:
            return float(base_score)

        baseline = self._session_baseline.get((label, regime), Metrics.empty())
        session = _metrics_delta(metrics, baseline)
        session_pnl = _cap_abs(session.pnl_pct, overlay.session_pnl_cap_pct)
        activity = _session_activity(session)

        adjusted = float(base_score)
        if activity >= overlay.min_session_activity or abs(session_pnl) > 1e-12:
            adjusted += float(overlay.overlay_weight) * session_pnl
            adjusted -= max(0.0, -session_pnl) * float(overlay.underperformance_weight)
        else:
            adjusted -= float(overlay.stale_penalty)
        return adjusted
