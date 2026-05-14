"""PlayerComposer — фабрика игроков из конфигурационных профилей.

В v1 у нас были 7 классов-наследников Panteon (PanteonResearch,
PanteonTrendResearch, PanteonMeanRevResearch, PanteonDefensiveResearch,
PanteonConsensusResearch, PanteonNextResearch, PanteonResilient).
Каждый имел свой `BOOTSTRAP_WEIGHTS`, свой `_score_shadow_candidate`,
свой state.

В v2 — 0 классов. Только конфигурационные `PlayerProfile`-ы +
EnsemblePlayer, который собирается через PlayerComposer.compose_*.

Гарантия Q1 (carantine consistency):
  PlayerComposer вызывает AgentSelector. AgentSelector фильтрует карантин
  ДО скоринга → результирующий EnsemblePlayer гарантированно не имеет
  карантинных агентов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..domain.types import Regime
from .player import EnsemblePlayer
from .selector import AgentSelector, ScoredAgent
from .voting import (
    StrongConsensus,
    ThresholdProfile,
    VotingPolicy,
    WeightedConsensus,
)


# ────────────────────────────────────────────────────────────────────
# PlayerProfile — конфигурация игрока
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlayerProfile:
    """Конфигурация EnsemblePlayer'а.

    В v1 это были классы-наследники (`PanteonTrendResearch`,
    `PanteonMeanRevResearch`...). В v2 — простые dataclass-ы.

    Replacement v1 → v2:
      class PanteonTrendResearch(_PanteonShadowVariant):
        BOOTSTRAP_WEIGHTS = (...)
        SCORE_BONUS_LABELS = frozenset({...})
        OPEN_SINGLE_THRESHOLD = 0.32
        ...
    →
      PROFILE_TREND_RESEARCH = PlayerProfile(
        label="TrendResearch",
        affinity=Regime.BULLISH,
        voting=WeightedConsensus(),
        thresholds=ThresholdProfile(open_single=0.32, ...),
        max_agents=5,
      )
    """

    label:        str
    voting:       VotingPolicy
    thresholds:   ThresholdProfile
    affinity:     Optional[Regime] = None
    max_agents:   int = 5
    min_agents:   int = 2

    # Bias-weights — добавочные веса к агентам с заданными labels.
    # Это мягкий аналог BOOTSTRAP_WEIGHTS из v1 — но не блокирует
    # выбор Selector-а, а только корректирует веса post-selection.
    bias:         Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("PlayerProfile.label must be non-empty")
        if self.max_agents < self.min_agents:
            raise ValueError(
                f"max_agents ({self.max_agents}) < min_agents ({self.min_agents})"
            )
        if self.max_agents <= 0:
            raise ValueError("max_agents must be positive")


# ────────────────────────────────────────────────────────────────────
# Pre-defined profiles — заменяют v1-классы
# ────────────────────────────────────────────────────────────────────


PROFILE_DEFAULT_ENSEMBLE = PlayerProfile(
    label="DefaultEnsemble",
    voting=WeightedConsensus(),
    thresholds=ThresholdProfile(),
)

PROFILE_TREND_RESEARCH = PlayerProfile(
    label="TrendResearch",
    voting=WeightedConsensus(),
    thresholds=ThresholdProfile(
        open_single=0.32, open_multi=0.26, open_floor=0.22,
        close_single=0.28, close_multi=0.23,
    ),
    affinity=Regime.BULLISH,
    max_agents=4,
    min_agents=2,
    bias={
        "LiveTrendFollow":     0.15,
        "LiveAfterShock":      0.10,
        "LiveCrashHunter":     0.10,
        "LiveOIBreakout":      0.10,
    },
)

PROFILE_MEAN_REV_RESEARCH = PlayerProfile(
    label="MeanRevResearch",
    voting=WeightedConsensus(),
    thresholds=ThresholdProfile(
        open_single=0.31, open_multi=0.25, open_floor=0.22,
        close_single=0.27, close_multi=0.22,
    ),
    affinity=Regime.NEUTRAL,
    max_agents=5,
    min_agents=3,
    bias={
        "LiveMeanRev":         0.20,
        "LiveVolCompress":     0.15,
        "LiveRegimePullback":  0.10,
        "NeutralRangeScalper": 0.05,
    },
)

PROFILE_NEUTRAL_EDGE_RESEARCH = PlayerProfile(
    label="NeutralEdgeResearch",
    voting=WeightedConsensus(),
    thresholds=ThresholdProfile(
        open_single=0.30, open_multi=0.23, open_floor=0.18,
        close_single=0.26, close_multi=0.20,
    ),
    affinity=Regime.NEUTRAL,
    max_agents=6,
    min_agents=2,
    bias={
        "ResearchValidatorAgent": 0.28,
        "LiveTrendFollow": 0.20,
        "BullRotationAgent": 0.18,
        "BearReliefFadeAgent": 0.16,
        "LiveRegimePullback": 0.14,
        "LiveOIBreakout": 0.10,
        "LiveCrashHunter": 0.08,
        "NeutralLiquiditySweep": 0.22,
        "AnchorFlowMomentum": 0.18,
    },
)

PROFILE_DEFENSIVE_RESEARCH = PlayerProfile(
    label="DefensiveResearch",
    voting=WeightedConsensus(),
    thresholds=ThresholdProfile(
        open_single=0.34, open_multi=0.28, open_floor=0.24,
        close_single=0.25, close_multi=0.20,
    ),
    affinity=Regime.BEARISH,  # имеет смысл для крах/просадки
    max_agents=4,
    min_agents=2,
    bias={
        "LiveCrashHunter":     0.20,
        "LiveRegimePullback":  0.12,
        "BearReliefFadeAgent": 0.10,
        "CarryFlowAgentV2":    0.05,
    },
)

PROFILE_BOMBERMAN_STRONG = PlayerProfile(
    label="BombermanStrong",
    voting=StrongConsensus(),
    thresholds=ThresholdProfile(open_single=0.40, open_floor=0.35),
    affinity=None,
    max_agents=3,
    min_agents=2,
)


# ────────────────────────────────────────────────────────────────────
# PlayerComposer
# ────────────────────────────────────────────────────────────────────


class PlayerComposer:
    """Фабрика EnsemblePlayer-ов.

    Гарантии:
      • compose_from_profile(profile, regime) даёт EnsemblePlayer, у которого
        agents отфильтрованы Selector-ом — никаких карантинных.
      • Если меньше min_agents проходят — возвращает None (caller
        решает, что делать).
      • Если bias задан — повышает веса соответствующих агентов
        post-selection (не влияет на сам отбор).
    """

    def __init__(self, selector: AgentSelector):
        self._selector = selector

    def compose_from_profile(
        self,
        profile: PlayerProfile,
        regime: Regime,
    ) -> Optional[EnsemblePlayer]:
        """Создаёт EnsemblePlayer для текущего регима по профилю."""
        scored = self._selector.select(regime, k=profile.max_agents)
        if len(scored) < profile.min_agents:
            return None
        return self._build(profile, scored)

    def compose_from_profile_with_fallback(
        self,
        profile: PlayerProfile,
        regime: Regime,
    ) -> Optional[EnsemblePlayer]:
        """Как compose_from_profile, но при недостатке агентов
        использует AgentSelector.select_with_fallback (понижает порог
        scoring).
        """
        scored = self._selector.select_with_fallback(regime, k=profile.max_agents)
        if len(scored) < profile.min_agents:
            return None
        return self._build(profile, scored)

    def _build(
        self,
        profile: PlayerProfile,
        scored: Sequence[ScoredAgent],
    ) -> EnsemblePlayer:
        # Веса = score, но с biases добавляем
        raw: Dict[str, float] = {}
        for sa in scored:
            base = max(sa.score, 0.0001)  # защита от делений на 0
            bonus = float(profile.bias.get(sa.label, 0.0))
            raw[sa.label] = base + bonus
        total = sum(raw.values())
        if total <= 0:
            # фолбэк: равные веса
            equal = 1.0 / len(scored)
            weights = {sa.label: equal for sa in scored}
        else:
            weights = {lbl: w / total for lbl, w in raw.items()}
        agents = [sa.agent for sa in scored]
        return EnsemblePlayer(
            label=profile.label,
            agents=agents,
            weights=weights,
            voting=profile.voting,
            thresholds=profile.thresholds,
            affinity=profile.affinity,
        )
