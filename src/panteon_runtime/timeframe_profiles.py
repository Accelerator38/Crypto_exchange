from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FOUR_HOUR_PROFILE_ID = "exia_4h_v1"
FOUR_HOUR_TIMEFRAME_MINUTES = 240
FOUR_HOUR_TIMEFRAME_ALIASES = frozenset(
    {"4h", "4hour", "4hours", "four_hour", "four_hours", "240m"}
)


# Every temporal value below is expressed in completed 4h bars. Percentage
# thresholds are explicit because the volatility of a 4h bar is materially
# different from the legacy one-minute runtime.
FOUR_HOUR_CLASS_OVERRIDES: dict[str, dict[str, Any]] = {
    "Bomberman": {
        "BOP_PERIOD": 12,
        "MRC_PERIOD": 36,
        "DONCHIAN_PERIOD": 20,
        "CHECK_INT": 1,
        "STOP_PCT": 0.025,
        "TARGET_PCT": 0.055,
        "VOL_CONFIRM": 1.15,
    },
    "MomentumScalper": {
        "EMA_F": 6,
        "EMA_M": 18,
        "EMA_S": 42,
        "VOL_WIN": 18,
        "CHECK_INT": 1,
        "STOP": 0.025,
        "TARGET": 0.060,
        "MOM_MIN": 0.004,
    },
    "LiveAfterShock": {
        "LOOKBACK": 12,
        "RSI_N": 14,
        "DROP": 0.018,
        "CHECK_INT": 1,
        "HOLD": 6,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.025,
        "TARGET": 0.055,
    },
    "LiveCrashHunter": {
        "RSI_N": 14,
        "MOM_LB": 12,
        "CHECK_INT": 1,
        "HOLD": 6,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.025,
        "TARGET": 0.055,
    },
    "LiveTrendFollow": {
        "EMA_F": 6,
        "EMA_M": 18,
        "EMA_S": 42,
        "CHECK_INT": 1,
        "STOP": 0.030,
        "TARGET": 0.080,
        "MIN_MOM": 0.005,
    },
    "RichardDennisTurtle": {
        "FAST_ENTRY": 20,
        "SLOW_ENTRY": 55,
        "FAST_EXIT": 10,
        "SLOW_EXIT": 20,
        "ATR_LB": 14,
        "CHECK_INT": 1,
        "MIN_BREAKOUT": 0.001,
    },
    "LiveRegimePullback": {
        "EMA_FAST": 6,
        "EMA_MID": 18,
        "EMA_SLOW": 42,
        "PULLBACK_LB": 12,
        "TURN_LB": 3,
        "TREND_LB": 12,
        "CHECK_INT": 1,
        "HOLD": 12,
        "STOP_MIN": 0.020,
        "STOP_MAX": 0.050,
        "PULLBACK_DEPTH": 0.008,
        "BOUNCE_MIN": 0.003,
        "TURN_MOM_MIN": 0.002,
        "TREND_MOM_MIN": 0.010,
    },
    "LiveMeanRev": {
        "BB_PERIOD": 20,
        "RSI_N": 14,
        "ENTRY_Z": 2.0,
        "EXIT_Z": 0.30,
        "CHECK_INT": 1,
        "HOLD": 6,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.030,
    },
    "LiveVolCompress": {
        "BB_PERIOD": 20,
        "MOM_N": 3,
        "BBW_THRESH": 0.050,
        "MIN_BBW": 0.006,
        "MOM_MIN": 0.008,
        "CHECK_INT": 1,
        "HOLD": 12,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.025,
        "TARGET": 0.060,
    },
    "FundingArb": {
        "EMA_F": 12,
        "EMA_S": 42,
        "CHECK_INT": 1,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.025,
        "TARGET": 0.055,
    },
    "LiveOIBreakout": {
        "CHECK_INT": 1,
        "MOM_N": 12,
        "MOM_MIN": 0.010,
        "VOL_MULT": 1.30,
        "HOLD": 12,
        "STOP": 0.030,
        "TARGET": 0.080,
    },
    "VolBreakoutHunter": {
        "CHECK_INT": 1,
        "LOOKBACK": 20,
        "ATR_WINDOW": 14,
        "VOL_WINDOW": 24,
        "HOLD": 12,
        "COOLDOWN_AFTER_STOP": 6,
        "VOL_MULT": 1.30,
        "ATR_EXPANSION_MIN": 1.15,
    },
    "BullRotationAgent": {
        "CHECK_INT": 1,
        "MARKET_LB": 42,
        "REL_LB": 42,
        "FAST_EMA": 6,
        "SLOW_EMA": 24,
        "TURN_LB": 3,
        "PULLBACK_LB": 12,
        "HOLD": 12,
        "STOP": 0.025,
        "TARGET": 0.060,
        "MIN_REL_STRENGTH": 0.012,
        "MIN_PULLBACK": 0.006,
        "MIN_TURN": 0.002,
    },
    "BearReliefFadeAgent": {
        "CHECK_INT": 1,
        "MARKET_LB": 42,
        "FAST_EMA": 6,
        "SLOW_EMA": 24,
        "BOUNCE_LB": 12,
        "TURN_LB": 3,
        "RSI_N": 14,
        "HOLD": 9,
        "STOP": 0.025,
        "TARGET": 0.055,
        "MIN_BOUNCE": 0.015,
        "MIN_WEAKNESS": 0.008,
    },
    "NeutralRangeScalper": {
        "CHECK_INT": 1,
        "RANGE_LB": 24,
        "MARKET_LB": 42,
        "TURN_LB": 3,
        "HOLD": 6,
        "MIN_WIDTH": 0.020,
        "MAX_WIDTH": 0.200,
        "MAX_MARKET_MOM": 0.060,
        "STOP": 0.018,
        "TARGET": 0.040,
    },
    "NeutralLiquiditySweepAgent": {
        "CHECK_INT": 1,
        "LOOKBACK": 24,
        "MARKET_LB": 42,
        "VOL_WINDOW": 24,
        "HOLD": 6,
        "MIN_WIDTH": 0.020,
        "MAX_WIDTH": 0.200,
        "SWEEP_MARGIN": 0.003,
        "REENTRY_MARGIN": 0.001,
        "MAX_ANCHOR_MOM": 0.060,
        "STOP": 0.020,
        "TARGET": 0.045,
    },
    "AnchorFlowMomentumAgent": {
        "CHECK_INT": 1,
        "FAST_LB": 6,
        "SLOW_LB": 24,
        "MARKET_LB": 42,
        "VOL_WINDOW": 24,
        "HOLD": 12,
        "MIN_ANCHOR_MOM": 0.015,
        "MIN_REL_STRENGTH": 0.005,
        "STOP": 0.025,
        "TARGET": 0.060,
    },
    "CrashPanicShortAgent": {
        "CHECK_INT": 1,
        "SHORT_LB": 12,
        "MID_LB": 42,
        "BREAKDOWN_LB": 12,
        "FAST_EMA": 6,
        "SLOW_EMA": 24,
        "HOLD": 6,
        "MIN_SYMBOL_DROP": -0.030,
        "MIN_VOL_RATIO": 1.20,
        "STOP": 0.025,
        "TARGET": 0.065,
    },
    "CarryFlowAgentV2": {
        "CHECK_INT": 1,
        "HOLD": 12,
        "EMA_FAST": 6,
        "EMA_SLOW": 24,
        "ENTRY_COOLDOWN": 3,
        "STOP": 0.025,
        "TARGET": 0.055,
    },
    "ResearchValidatorAgent": {
        "CHECK_INT": 1,
        "EMA_FAST": 6,
        "EMA_MID": 18,
        "EMA_SLOW": 42,
        "BREAKOUT": 20,
        "VOL_WIN": 24,
        "NOISE_WIN": 42,
        "HOLD": 12,
        "STOP": 0.030,
        "TARGET": 0.080,
        "MAX_NOISE": 0.030,
    },
    "CandlePatternAgent": {
        "AGG_BARS": 3,
        "HISTORY_MAX": 120,
        "CHECK_INT": 3,
        "ENTRY_COOLDOWN": 3,
        "HOLD": 6,
        "STOP": 0.030,
        "TARGET": 0.060,
    },
    "MeanRevConfirmedAgent": {
        "BB_PERIOD": 24,
        "VOL_LOOKBACK": 24,
        "ENTRY_Z": 2.2,
        "EXIT_Z": 0.35,
        "CHECK_INT": 1,
        "HOLD_BARS": 6,
        "ENTRY_COOLDOWN": 3,
        "STOP_PCT": 0.030,
    },
    "AdaptiveVolTrendAgent": {
        "EMA_FAST": 6,
        "EMA_SLOW": 24,
        "ATR_N": 14,
        "CHECK_INT": 1,
        "HOLD_MAX": 18,
        "ATR_K_TRAIL": 2.5,
        "ATR_K_STOP": 2.0,
    },
    "Panteon": {
        "STATE_INT": 6,
        "PLAYER_ROTATION_INT": 6,
        "PLAYER_SWITCH_COOLDOWN_BARS": 6,
        "REGIME_HYSTERESIS_BARS": 3,
        "ENSEMBLE_PURGATORY_BARS": 42,
        "LIVE_GATE_GRACE_BARS": 12,
        "STALE_BARS": 42,
        "ROTATION_INT": 6,
        "CONTEXT_REFRESH_INT": 6,
        "RECENT_SIGNAL_WINDOW": 42,
        "SL_PCT": 0.040,
        "TP_PCT": 0.090,
        "TRAIL_PCT": 0.035,
    },
    "PanteonResearch": {"ROTATION_INT": 6},
    "PanteonTrendResearch": {"ROTATION_INT": 6},
    "PanteonMeanRevResearch": {"ROTATION_INT": 6},
    "PanteonDefensiveResearch": {"ROTATION_INT": 6},
    "PanteonConsensusResearch": {"ROTATION_INT": 6},
    "PanteonResilient": {"ROTATION_INT": 6},
    "PanteonNextResearch": {"ROTATION_INT": 6},
}


def get_timeframe_profile(profile_id: str) -> Mapping[str, Mapping[str, Any]]:
    if str(profile_id) != FOUR_HOUR_PROFILE_ID:
        raise ValueError(f"unknown timeframe profile: {profile_id}")
    return FOUR_HOUR_CLASS_OVERRIDES


def require_profile_timeframe(profile_id: str, timeframe: object) -> None:
    get_timeframe_profile(profile_id)
    normalized = str(timeframe or "").strip().lower()
    if normalized not in FOUR_HOUR_TIMEFRAME_ALIASES:
        raise ValueError(
            f"{profile_id} requires completed 4h bars, got timeframe={timeframe!r}"
        )


def build_profiled_component(
    factory: Any,
    *,
    profile_id: str,
    timeframe: object,
) -> object:
    """Construct one runtime component under an explicit timeframe contract."""
    require_profile_timeframe(profile_id, timeframe)
    component = factory()
    report = apply_timeframe_profile(component, profile_id)
    setattr(component, "_timeframe_profile_report", report)
    return component


def _object_profile_overrides(
    value: object,
    profile: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    effective: dict[str, Any] = {}
    for cls in reversed(type(value).__mro__):
        effective.update(profile.get(cls.__name__, {}))
    return effective


def apply_timeframe_profile(root: object, profile_id: str) -> dict[str, Any]:
    """Apply a sealed bar-based profile to a component and all nested agents."""
    profile = get_timeframe_profile(profile_id)
    visited: set[int] = set()
    stack: list[object] = [root]
    applied: list[dict[str, Any]] = []
    bomberman_players: list[object] = []

    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)

        overrides = _object_profile_overrides(value, profile)
        if overrides:
            for field, setting in overrides.items():
                setattr(value, field, setting)
            applied.append(
                {
                    "class_name": type(value).__name__,
                    "fields": dict(sorted(overrides.items())),
                }
            )
        if any(cls.__name__ == "PlayerBomberman" for cls in type(value).__mro__):
            bomberman_players.append(value)

        namespace = getattr(value, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for child in namespace.values():
            if isinstance(child, dict):
                stack.extend(item for item in child.values() if hasattr(item, "__dict__"))
            elif isinstance(child, (list, tuple, set)):
                stack.extend(item for item in child if hasattr(item, "__dict__"))
            elif hasattr(child, "__dict__"):
                module = str(type(child).__module__)
                if module.startswith(("panteon_", "agent_", "player_", "agents_v2")):
                    stack.append(child)

    # Preserve the player's two-speed confirmation instead of turning both
    # nested Bomberman instances into exact duplicates.
    for player in bomberman_players:
        cautious = getattr(player, "_b2", None)
        if cautious is not None:
            cautious.BOP_THRESH = 0.70
            cautious.DONCHIAN_PERIOD = 30
            cautious.MRC_PERIOD = 48

    return {
        "profile_id": FOUR_HOUR_PROFILE_ID,
        "timeframe_minutes": FOUR_HOUR_TIMEFRAME_MINUTES,
        "objects_updated": len(applied),
        "applied": applied,
    }
