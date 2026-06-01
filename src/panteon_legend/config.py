from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from panteon_v2.domain.types import Regime
from panteon_v2.selection import PlayerProfile
from panteon_v2.selection.voting import ThresholdProfile, WeightedConsensus


LEGEND_YEARS: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
LEGACY_TRADE_FRACTION = 0.10
LEGACY_LEVERAGE = 3
LEGACY_MAX_TRADES_PER_PERIOD = 1000
LEGACY_LIQUIDITY_MIN_ADV = 5000
LEGEND_SOFT_SELECTOR_POLICY = "soft_regime_top1_24"
LEGEND_SOFT_SELECTOR_WINDOW_BARS = 24
LEGEND_SOFT_SELECTOR_MIN_CLOSED_TRADES = 20
LEGEND_SOFT_SELECTOR_MIN_SCORE_TO_TRADE = 1e-9
LEGEND_EXECUTION_CANDIDATE_ALLOW_LABELS: tuple[str, ...] = (
    "Legend_Defensive",
    "Legend_RegimeSwitch",
    "Legend_BearDefense",
    "Solo_MomentumScalper",
    "Solo_ResearchValidatorAgent",
)


LEGEND_CORE_ACTORS: tuple[str, ...] = (
    "LiveOIBreakout",
    "LiveCrashHunter",
    "MomentumScalper",
    "VolBreakoutHunter",
    "ResearchValidatorAgent",
    "BullRotationAgent",
    "FundingArb",
    "NeutralLiquiditySweep",
    "NeutralRangeScalper",
    "AnchorFlowMomentum",
    "LiveAfterShock",
    "LiveRegimePullback",
    "LiveMeanRev",
    "LiveVolCompress",
    "LiveTrendFollow",
    "BearReliefFadeAgent",
    "CrashPanicShortAgent",
    "RichardDennis",
    "GeneticsCore",
)


LEGEND_FIXED_AGENT_PLAYER_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Legend_LegacyCore",
        (
            "MomentumScalper",
            "VolBreakoutHunter",
            "ResearchValidatorAgent",
            "LiveCrashHunter",
            "FundingArb",
            "LiveOIBreakout",
            "BullRotationAgent",
        ),
    ),
    (
        "Legend_BearDefense",
        (
            "LiveCrashHunter",
            "FundingArb",
            "BearReliefFadeAgent",
            "CrashPanicShortAgent",
            "LiveRegimePullback",
        ),
    ),
    (
        "Legend_BullBreakout",
        (
            "VolBreakoutHunter",
            "MomentumScalper",
            "RichardDennis",
            "LiveOIBreakout",
            "LiveAfterShock",
            "BullRotationAgent",
        ),
    ),
    (
        "Legend_NeutralValidator",
        (
            "ResearchValidatorAgent",
            "NeutralLiquiditySweep",
            "NeutralRangeScalper",
            "AnchorFlowMomentum",
            "LiveMeanRev",
            "LiveVolCompress",
        ),
    ),
    (
        "Legend_RotationFlow",
        (
            "BullRotationAgent",
            "BearReliefFadeAgent",
            "AnchorFlowMomentum",
            "LiveTrendFollow",
            "LiveRegimePullback",
        ),
    ),
)


LEGEND_ROTATING_AGENT_PLAYER_SETS: tuple[
    tuple[str, dict[str, tuple[str, ...]], tuple[str, ...]],
    ...
] = (
    (
        "Legend_CurrentActorPool",
        {
            "bullish": (
                "LiveOIBreakout",
                "VolBreakoutHunter",
                "MomentumScalper",
                "BullRotationAgent",
                "LiveAfterShock",
                "RichardDennis",
                "LiveTrendFollow",
            ),
            "bearish": (
                "LiveCrashHunter",
                "FundingArb",
                "MomentumScalper",
                "BearReliefFadeAgent",
                "LiveRegimePullback",
                "CrashPanicShortAgent",
            ),
            "neutral": (
                "ResearchValidatorAgent",
                "NeutralLiquiditySweep",
                "NeutralRangeScalper",
                "AnchorFlowMomentum",
                "LiveMeanRev",
                "LiveVolCompress",
                "LiveOIBreakout",
            ),
            "crash": (
                "LiveCrashHunter",
                "CrashPanicShortAgent",
                "FundingArb",
                "MomentumScalper",
                "LiveRegimePullback",
                "LiveOIBreakout",
            ),
        },
        (
            "ResearchValidatorAgent",
            "LiveCrashHunter",
            "MomentumScalper",
            "LiveOIBreakout",
            "FundingArb",
        ),
    ),
)


LEGEND_REGIME_SWITCH_PLAYER_SETS: tuple[tuple[str, dict[str, tuple[str, ...]]], ...] = (
    (
        "Legend_RegimeSwitch",
        {
            "bullish": ("LiveOIBreakout", "VolBreakoutHunter", "MomentumScalper"),
            "bearish": ("LiveCrashHunter", "FundingArb", "LiveRegimePullback"),
            "neutral": (
                "ResearchValidatorAgent",
                "NeutralLiquiditySweep",
                "AnchorFlowMomentum",
            ),
            "crash": ("LiveCrashHunter", "CrashPanicShortAgent", "FundingArb"),
        },
    ),
)


@dataclass(frozen=True)
class LegendProfile:
    name: str = "panteon_legend_v2"
    flash_enabled: bool = False
    trade_fraction: float = LEGACY_TRADE_FRACTION
    leverage: int = LEGACY_LEVERAGE
    apply_leverage_to_notional: bool = False
    max_trades_per_period: int = LEGACY_MAX_TRADES_PER_PERIOD
    liquidity_min_adv: int = LEGACY_LIQUIDITY_MIN_ADV
    max_open_positions: int = 8
    max_new_opens_per_bar: int = 3
    solo_agent_candidate_limit: int = 12
    include_optional_agents: bool = False
    years: tuple[int, ...] = LEGEND_YEARS
    soft_selector_policy: str = LEGEND_SOFT_SELECTOR_POLICY
    soft_selector_window_bars: int = LEGEND_SOFT_SELECTOR_WINDOW_BARS
    soft_selector_min_closed_trades: int = LEGEND_SOFT_SELECTOR_MIN_CLOSED_TRADES
    soft_selector_min_score_to_trade: float = LEGEND_SOFT_SELECTOR_MIN_SCORE_TO_TRADE
    current_actionable_gate_enabled: bool = False
    solo_current_actionable_gate_enabled: bool = True
    execution_candidate_allow_labels: tuple[str, ...] = (
        LEGEND_EXECUTION_CANDIDATE_ALLOW_LABELS
    )


def build_legend_profile() -> LegendProfile:
    return LegendProfile()


def legend_actor_labels() -> tuple[str, ...]:
    labels: list[str] = []
    _extend_unique(labels, LEGEND_CORE_ACTORS)
    for _, actors in LEGEND_FIXED_AGENT_PLAYER_SETS:
        _extend_unique(labels, actors)
    for _, regime_map, fallback in LEGEND_ROTATING_AGENT_PLAYER_SETS:
        for actors in regime_map.values():
            _extend_unique(labels, actors)
        _extend_unique(labels, fallback)
    for _, regime_map in LEGEND_REGIME_SWITCH_PLAYER_SETS:
        for actors in regime_map.values():
            _extend_unique(labels, actors)
    return tuple(labels)


def legend_player_profiles() -> tuple[PlayerProfile, ...]:
    """Low-friction v2 players kept for observability and broad agent scoring."""
    return (
        PlayerProfile(
            label="Legend_Consensus",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(
                open_single=0.22,
                open_multi=0.18,
                open_floor=0.14,
                close_single=0.22,
                close_multi=0.18,
            ),
            affinity=None,
            max_agents=12,
            min_agents=1,
            allowed_labels=legend_actor_labels(),
            bias={
                "LiveOIBreakout": 0.18,
                "LiveCrashHunter": 0.18,
                "MomentumScalper": 0.16,
                "ResearchValidatorAgent": 0.14,
                "FundingArb": 0.12,
            },
        ),
        PlayerProfile(
            label="Legend_Bullish",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(open_single=0.22, open_multi=0.18, open_floor=0.14),
            affinity=Regime.BULLISH,
            max_agents=8,
            min_agents=1,
            allowed_labels=(
                "LiveOIBreakout",
                "VolBreakoutHunter",
                "MomentumScalper",
                "BullRotationAgent",
                "LiveAfterShock",
                "RichardDennis",
            ),
        ),
        PlayerProfile(
            label="Legend_Defensive",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(open_single=0.24, open_multi=0.19, open_floor=0.15),
            affinity=Regime.BEARISH,
            max_agents=8,
            min_agents=1,
            allowed_labels=(
                "LiveCrashHunter",
                "FundingArb",
                "BearReliefFadeAgent",
                "CrashPanicShortAgent",
                "LiveRegimePullback",
            ),
        ),
    )


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)


def csv_arg(values: Sequence[str]) -> str:
    return ",".join(str(value).strip() for value in values if str(value).strip())
