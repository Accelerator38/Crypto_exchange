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
from .player import EnsemblePlayer, NoTradePlayer, Player, RotatingAgentPlayer
from .selector import AgentSelector, ScoredAgent, SessionOverlayConfig
from .flash_allocator import (
    FLASH_EXPERIMENTAL_FLAGS,
    FLASH_PRESET_AGGRESSIVE,
    FLASH_PRESET_DEFAULT,
    FLASH_PRESET_SAFE,
    FLASH_PRESET_SAFE_PINNED_VALUES,
    FlashAllocator,
    FlashAllocatorConfig,
    FlashCandidateAudit,
    FlashDecision,
)
from .promotion_manifest import (
    PromotionManifest,
    PromotionManifestConfig,
    PromotionRejection,
    build_promotion_manifest,
)
from .composer import (
    PROFILE_BOMBERMAN_STRONG,
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_GENETICS_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_NEUTRAL_EDGE_RESEARCH,
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
    "NoTradePlayer",
    "Player",
    "RotatingAgentPlayer",
    # selector
    "AgentSelector",
    "ScoredAgent",
    "SessionOverlayConfig",
    # flash allocator
    "FLASH_EXPERIMENTAL_FLAGS",
    "FLASH_PRESET_AGGRESSIVE",
    "FLASH_PRESET_DEFAULT",
    "FLASH_PRESET_SAFE",
    "FLASH_PRESET_SAFE_PINNED_VALUES",
    "FlashAllocator",
    "FlashAllocatorConfig",
    "FlashCandidateAudit",
    "FlashDecision",
    # promotion manifest
    "PromotionManifest",
    "PromotionManifestConfig",
    "PromotionRejection",
    "build_promotion_manifest",
    # composer
    "PROFILE_BOMBERMAN_STRONG",
    "PROFILE_DEFAULT_ENSEMBLE",
    "PROFILE_DEFENSIVE_RESEARCH",
    "PROFILE_GENETICS_RESEARCH",
    "PROFILE_MEAN_REV_RESEARCH",
    "PROFILE_NEUTRAL_EDGE_RESEARCH",
    "PROFILE_TREND_RESEARCH",
    "PlayerComposer",
    "PlayerProfile",
    # strategist
    "DEFAULT_STRATEGIST",
    "Strategist",
    "StrategistConfig",
    "SwitchDecision",
]
