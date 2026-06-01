from __future__ import annotations

from typing import Sequence

from panteon_v2.app.bootstrap import LiveExecutionConfig, build_production_pipeline
from panteon_v2.execution import Exchange, RiskLimitsConfig
from panteon_v2.selection import AgentRegistry, StrategistConfig

from .config import (
    LEGEND_FIXED_AGENT_PLAYER_SETS,
    LEGEND_REGIME_SWITCH_PLAYER_SETS,
    LEGEND_ROTATING_AGENT_PLAYER_SETS,
    LegendProfile,
    build_legend_profile,
    legend_player_profiles,
)


def legend_risk_config(profile: LegendProfile | None = None) -> RiskLimitsConfig:
    profile = profile or build_legend_profile()
    return RiskLimitsConfig(
        max_open_positions=profile.max_open_positions,
        max_leverage=profile.leverage,
        apply_leverage_to_notional=profile.apply_leverage_to_notional,
        capital_fraction=profile.trade_fraction,
        floor_to_exchange_min_notional=True,
    )


def legend_live_execution_config(
    profile: LegendProfile | None = None,
) -> LiveExecutionConfig:
    profile = profile or build_legend_profile()
    return LiveExecutionConfig(
        max_new_opens_per_bar=profile.max_new_opens_per_bar,
        max_daily_loss_pct=5.0,
        max_equity_peak_drawdown_pct=10.0,
        max_consecutive_failed_orders=5,
        max_exchange_desync_events=5,
        max_stale_feed_polls=12,
        max_slippage_pct=0.75,
        max_api_error_streak=5,
        adopt_existing_positions_enabled=True,
        adopt_existing_position_symbols=("*",),
        adopt_existing_position_player="PanteonLegendAdopted",
        adopt_existing_position_agent="AdoptedExchangePosition",
    )


def legend_strategist_config(profile: LegendProfile | None = None) -> StrategistConfig:
    profile = profile or build_legend_profile()
    return StrategistConfig(
        cooldown_bars=0,
        switch_margin=0.0,
        min_score_to_switch=-10.0,
        streak_needed=1,
        hard_negative=-10.0,
        urgent_gap=0.0,
        min_live_closed_trades=0,
        min_live_score=-1_000_000.0,
        max_live_drawdown_pct=100.0,
        hard_policy_enabled=False,
        real_promotion_gate_enabled=False,
        no_trade_when_all_rejected=False,
        use_v3_rolling_score=True,
        use_v3_entry_causal_score=False,
        use_v3_shadow_rolling_score=False,
        use_v3_soft_shadow_score=True,
        v3_current_actionable_gate_enabled=profile.current_actionable_gate_enabled,
        v3_solo_current_actionable_gate_enabled=(
            profile.solo_current_actionable_gate_enabled
        ),
        v3_candidate_allow_labels=profile.execution_candidate_allow_labels,
        v3_shadow_position_gate_enabled=False,
        v3_shadow_rolling_window_bars=profile.soft_selector_window_bars,
        v3_shadow_rolling_min_closed_trades=profile.soft_selector_min_closed_trades,
        v3_min_score_to_trade=profile.soft_selector_min_score_to_trade,
        v3_real_loss_kill_min_closed_trades=0,
        candidate_ttl_bars=0,
    )


def configure_legend_pipeline(
    pipeline: object,
    profile: LegendProfile | None = None,
) -> None:
    profile = profile or build_legend_profile()
    setattr(pipeline, "flash_enabled", False)
    setattr(pipeline, "legend_profile", profile.name)
    setattr(pipeline, "legend_soft_selector_policy", profile.soft_selector_policy)
    setattr(pipeline, "solo_agent_candidate_limit", profile.solo_agent_candidate_limit)
    setattr(pipeline, "fixed_agent_player_sets", LEGEND_FIXED_AGENT_PLAYER_SETS)
    setattr(pipeline, "rotating_agent_player_sets", LEGEND_ROTATING_AGENT_PLAYER_SETS)
    setattr(pipeline, "regime_switch_player_sets", LEGEND_REGIME_SWITCH_PLAYER_SETS)
    setattr(pipeline, "actionable_fallback_enabled", False)
    setattr(
        pipeline,
        "current_actionable_candidate_layer_enabled",
        profile.current_actionable_gate_enabled,
    )


def build_legend_pipeline(
    *,
    registry: AgentRegistry,
    exchange: Exchange,
    initial_capital: float,
    seed_quarantine: Sequence[str] = (),
    profile: LegendProfile | None = None,
    jsonl_event_log: str | None = None,
):
    profile = profile or build_legend_profile()
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=initial_capital,
        seed_quarantine=seed_quarantine,
        profiles=legend_player_profiles(),
        strategist_config=legend_strategist_config(profile),
        risk_config=legend_risk_config(profile),
        live_execution_config=legend_live_execution_config(profile),
        flash_enabled=False,
        perf_trade_fraction=profile.trade_fraction,
        jsonl_event_log=jsonl_event_log,
    )
    configure_legend_pipeline(pipeline, profile)
    return pipeline
