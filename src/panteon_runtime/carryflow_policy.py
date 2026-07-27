"""Versioned CarryFlow policies with no independently editable actor knobs."""

from __future__ import annotations

from dataclasses import dataclass


ALL_REGIMES = (
    "neutral",
    "bullish",
    "bearish",
    "choppy_up",
    "choppy_down",
    "mixed_rotational",
    "range_low_vol",
    "crash",
)
NON_RANGE_REGIMES = tuple(
    regime for regime in ALL_REGIMES if regime != "range_low_vol"
)
SHORT_SYSTEMIC_GUARD_REGIMES = tuple(
    regime for regime in ALL_REGIMES if regime != "bullish"
)


@dataclass(frozen=True)
class CarryFlowPolicyProfile:
    profile_id: str
    flow_model: str
    allowed_regimes: tuple[str, ...]
    hold_bars: int
    stop_pct: float
    target_pct: float
    oi_lookback_bars: int
    price_lookback_bars: int
    oi_expansion: float
    min_price_return: float
    max_price_return: float
    funding_entry: float = 0.00008
    funding_exit: float = 0.00002
    crowd_ratio: float = 0.58
    basis_floor: float = -0.0015
    required_history: int = 4
    ema_fast: int = 3
    ema_slow: int = 6
    rsi_n: int = 3
    rsi_overbought: float = 58.0
    rsi_oversold: float = 42.0
    extreme_extension: float = 0.0
    entry_cooldown: int = 6
    exit_on_normalization: bool = False
    min_hold_before_normalization: int = 0
    max_data_age_seconds: int = 1200
    max_positions: int = 1
    signal_feature: str = "diagnostic.edge"
    signal_slope_bps_per_unit: float = 100.0
    signal_lcb_haircut_bps: float = 4.0
    signal_min_feature: float = 0.20
    signal_max_expected_move_bps: float = 250.0
    min_regime_confidence: float = 0.50
    max_spread_bps: float = 5.0
    max_slippage_bps: float = 5.0
    risk_mult: float = 1.0
    max_signal_age_seconds: int = 120

    def actor_values(self) -> dict[str, bool | int | float | str]:
        return {
            "PROFILE_ID": self.profile_id,
            "FLOW_MODEL": self.flow_model,
            "CHECK_INT": 1,
            "HOLD": self.hold_bars,
            "STOP": self.stop_pct,
            "TARGET": self.target_pct,
            "FUNDING_ENTRY": self.funding_entry,
            "FUNDING_EXIT": self.funding_exit,
            "OI_SPIKE": self.oi_expansion,
            "CROWD_RATIO": self.crowd_ratio,
            "BASIS_ENTRY": 0.0004,
            "SHORT_BASIS_FLOOR": self.basis_floor,
            "EXTREME_EXT": self.extreme_extension,
            "EMA_FAST": self.ema_fast,
            "EMA_SLOW": self.ema_slow,
            "RSI_N": self.rsi_n,
            "RSI_OB": self.rsi_overbought,
            "RSI_OS": self.rsi_oversold,
            "MAX_DATA_AGE_SEC": self.max_data_age_seconds,
            "MAX_POS": self.max_positions,
            "ENTRY_COOLDOWN": self.entry_cooldown,
            "ALLOW_LONG": False,
            "ALLOW_SHORT": True,
            "EXIT_ON_NORMALIZATION": self.exit_on_normalization,
            "MIN_HOLD_BEFORE_NORMALIZATION": (
                self.min_hold_before_normalization
            ),
            "OI_LOOKBACK_BARS": self.oi_lookback_bars,
            "PRICE_LOOKBACK_BARS": self.price_lookback_bars,
            "MIN_PRICE_RETURN": self.min_price_return,
            "MAX_PRICE_RETURN": self.max_price_return,
            "REQUIRED_HISTORY": self.required_history,
        }


_PROFILES = {
    "screened_short_v1": CarryFlowPolicyProfile(
        profile_id="screened_short_v1",
        flow_model="legacy_momentum",
        allowed_regimes=ALL_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=1,
        price_lookback_bars=1,
        oi_expansion=0.02,
        min_price_return=-1.0,
        max_price_return=1.0,
        required_history=53,
        ema_fast=12,
        ema_slow=48,
        rsi_n=14,
        extreme_extension=0.004,
        entry_cooldown=24,
        exit_on_normalization=True,
    ),
    "divergence_short_v1": CarryFlowPolicyProfile(
        profile_id="divergence_short_v1",
        flow_model="oi_price_divergence",
        allowed_regimes=ALL_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=3,
        price_lookback_bars=3,
        oi_expansion=0.015,
        min_price_return=-1.0,
        max_price_return=0.005,
    ),
    "divergence_short_systemic_guard_v1": CarryFlowPolicyProfile(
        profile_id="divergence_short_systemic_guard_v1",
        flow_model="oi_price_divergence_systemic_guard",
        allowed_regimes=SHORT_SYSTEMIC_GUARD_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=3,
        price_lookback_bars=3,
        oi_expansion=0.015,
        min_price_return=-1.0,
        max_price_return=0.005,
    ),
    "divergence_short_non_range_v1": CarryFlowPolicyProfile(
        profile_id="divergence_short_non_range_v1",
        flow_model="oi_price_divergence",
        allowed_regimes=NON_RANGE_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=3,
        price_lookback_bars=3,
        oi_expansion=0.015,
        min_price_return=-1.0,
        max_price_return=0.005,
    ),
    "divergence_short_strict_v1": CarryFlowPolicyProfile(
        profile_id="divergence_short_strict_v1",
        flow_model="oi_price_divergence",
        allowed_regimes=NON_RANGE_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=3,
        price_lookback_bars=3,
        oi_expansion=0.02,
        min_price_return=-1.0,
        max_price_return=0.0,
    ),
    "crowded_overextension_short_v1": CarryFlowPolicyProfile(
        profile_id="crowded_overextension_short_v1",
        flow_model="crowded_price_overextension",
        allowed_regimes=ALL_REGIMES,
        hold_bars=6,
        stop_pct=0.012,
        target_pct=0.024,
        oi_lookback_bars=1,
        price_lookback_bars=3,
        oi_expansion=0.0,
        min_price_return=0.005,
        max_price_return=1.0,
        funding_entry=0.00005,
        crowd_ratio=0.68,
        basis_floor=-0.0015,
    ),
}


def carryflow_profile_ids() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_carryflow_profile(profile_id: str) -> CarryFlowPolicyProfile:
    key = str(profile_id or "").strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"unknown CarryFlow policy profile: {profile_id}") from exc


def apply_carryflow_profile(runtime: object, profile_id: str) -> CarryFlowPolicyProfile:
    profile = get_carryflow_profile(profile_id)
    for name, value in profile.actor_values().items():
        setattr(runtime, name, value)
    return profile
