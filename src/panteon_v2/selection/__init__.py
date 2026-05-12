"""Selection layer (Phase 3): Agent, Player, Selector, Composer, Strategist.

Layered exports:
  agent      — Agent Protocol, AgentRegistry
  voting     — VotingPolicy + 3 implementations + ThresholdProfile
  player     — Player Protocol, EnsemblePlayer
  selector   — AgentSelector, ScoredAgent
  composer   — PlayerComposer, PlayerProfile + 5 pre-defined profiles
  strategist — Strategist, StrategistConfig, SwitchDecision
"""

from .agent import Agent, AgentRegistry
from .voting import (
    AgentVotes,
    RiskParity,
    StrongConsensus,
    ThresholdProfile,
    VotingPolicy,
    WeightedConsensus,
)
from .player import EnsemblePlayer, Player
from .selector import AgentSelector, ScoredAgent
from .composer import (
    PROFILE_BOMBERMAN_STRONG,
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PlayerComposer,
    PlayerProfile,
)
from .strategist import (
    DEFAULT_STRATEGIST,
    Strategist,
    StrategistConfig,
    SwitchDecision,
)

__all__ = [
    # agent
    "Agent",
    "AgentRegistry",
    # voting
    "AgentVotes",
    "RiskParity",
    "StrongConsensus",
    "ThresholdProfile",
    "VotingPolicy",
    "WeightedConsensus",
    # player
    "EnsemblePlayer",
    "Player",
    # selector
    "AgentSelector",
    "ScoredAgent",
    # composer
    "PROFILE_BOMBERMAN_STRONG",
    "PROFILE_DEFAULT_ENSEMBLE",
    "PROFILE_DEFENSIVE_RESEARCH",
    "PROFILE_MEAN_REV_RESEARCH",
    "PROFILE_TREND_RESEARCH",
    "PlayerComposer",
    "PlayerProfile",
    # strategist
    "DEFAULT_STRATEGIST",
    "Strategist",
    "StrategistConfig",
    "SwitchDecision",
]
