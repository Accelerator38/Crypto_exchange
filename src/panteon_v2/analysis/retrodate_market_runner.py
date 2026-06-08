"""Run Panteon v2 live-like benchmarks on Retrodate OHLCV CSV data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping, Optional, Sequence

from ..app.agent_bootstrap import (
    experimental_flash_agent_labels,
    register_all_v1_agents,
    register_experimental_flash_agents,
)
from ..app.bootstrap import LiveExecutionConfig, build_production_pipeline
from ..app.main_loop import StepResult, main_loop
from ..app.output_writer import OutputWriter, OutputWriterConfig
from ..app.shadow_tournament import ProductionShadowTournament
from ..attribution import (
    CandidateRejected,
    CandidateScored,
    EventLog,
    ExecutionAttributed,
    OrderFilled,
    PositionClosed,
    PositionOpened,
    RegimeDetected,
    ShadowActorUpdated,
)
from ..domain.types import MarketSnapshot, Regime
from ..execution import FakeExchange, RiskLimitsConfig
from ..scoring import DEFAULT_SCORING, ScoringConfig
from ..selection import AgentRegistry, FlashAllocatorConfig, StrategistConfig
from ..selection.promotion_manifest import (
    PromotionManifestConfig,
    Z_95_ONE_SIDED,
    build_promotion_manifest,
)
from ..shadow.feed import ReplayFeed
from .soft_allocator import (
    OnlineSoftAllocator,
    ShadowPnLEvent,
    SoftAllocatorPolicy,
    shadow_pnl_events_from_shadow_updates,
    simulate_perfect_monthly_panteon,
    simulate_soft_allocator_policies,
    write_shadow_pnl_events,
    write_perfect_panteon_report,
    write_soft_allocator_report,
)
from .retrodate_validator import (
    RetrodateDirReport,
    RetrodateFileReport,
    RetrodateValidationError,
    validate_retrodate_dir,
)
from .allocation_diagnostics import analyze_trading_log, write_allocation_diagnostics
from .technical_indicators import TechnicalIndicatorState
from .walk_forward import write_walk_forward_report_from_events


DEFAULT_YEARS = (2022, 2023, 2024, 2025, 2026)
DEFAULT_FIXED_AGENT_PLAYER_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Fixed_BearDefense",
        ("LiveCrashHunter", "FundingArb", "BearReliefFadeAgent", "CrashPanicShortAgent"),
    ),
    (
        "Fixed_BullBreakout",
        ("VolBreakoutHunter", "MomentumScalper", "RichardDennis", "LiveOIBreakout"),
    ),
    (
        "Fixed_NeutralValidator",
        (
            "ResearchValidatorAgent",
            "NeutralLiquiditySweep",
            "NeutralRangeScalper",
            "AnchorFlowMomentum",
        ),
    ),
    (
        "Fixed_RegimePullback",
        ("LiveRegimePullback", "LiveMeanRev", "LiveVolCompress"),
    ),
    (
        "Fixed_RotationFlow",
        ("BullRotationAgent", "BearReliefFadeAgent", "AnchorFlowMomentum", "LiveTrendFollow"),
    ),
    (
        "Fixed_CoreDiversified",
        (
            "LiveCrashHunter",
            "VolBreakoutHunter",
            "ResearchValidatorAgent",
            "NeutralLiquiditySweep",
            "AnchorFlowMomentum",
        ),
    ),
    (
        "Fixed_TrendRecovery",
        ("LiveTrendFollow", "RichardDennis", "LiveRegimePullback"),
    ),
)
DEFAULT_PROBATION_LOSS_LABEL_PREFIXES: tuple[str, ...] = (
    "Solo_",
    "Fixed_",
    "Antonius_",
    "Optimal_",
)
DEFAULT_RETRO_HARD_POLICY_DENY_LABELS: tuple[str, ...] = (
    "Solo_PlayerFunding",
    "Solo_LiveAfterShock",
    "Solo_CarryFlowAgentV2",
    "Solo_LiveRegimePullback",
    "Solo_LiveTrendFollow",
    "DefaultEnsemble",
    "DefensiveResearch",
    "TrendResearch",
    "NeutralEdgeResearch",
    "MeanRevResearch",
    "Fixed_BearDefense",
    "Fixed_BullBreakout",
    "Fixed_CoreDiversified",
    "Fixed_NeutralValidator",
    "Fixed_RegimePullback",
    "Fixed_RotationFlow",
    "Fixed_TrendRecovery",
    "Perfect_CrashSwitch",
    "Perfect_GeneticsBearCrash",
    "Perfect_MeanRev",
    "Perfect_NeutralValidator",
    "Perfect_OIBreakout",
    "Solo_BullRotationAgent",
    "Solo_GeneticsCore",
    "Solo_LiveMeanRev",
    "Solo_LiveOIBreakout",
    "Solo_RichardDennis",
)
FLASH_INACTIVE_REJECTION_REASONS: frozenset[str] = frozenset({
    "inactive",
    "quarantined",
})


def _experimental_flash_fixed_agent_player_sets() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "Fixed_ProfitTriad",
            (
                "MomentumScalperShortOnly",
                "FundingArb",
                "CrashPanicShortAgent",
            ),
        ),
        (
            "Fixed_SpotQualityPair",
            (
                "MomentumScalperSpotQuality",
                "VolBreakoutSpotOnly",
            ),
        ),
    )


def _genetics_probation_fixed_agent_player_sets(
    config: RetrodateMarketConfig,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not bool(config.genetics_probation_execution_enabled):
        return ()
    fixed_sets: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for raw_label in config.genetics_probation_labels:
        label = str(raw_label or "").strip()
        if not label or label in seen or not label.startswith("Genetics"):
            continue
        seen.add(label)
        fixed_sets.append((label, (label,)))
    return tuple(fixed_sets)


def _experimental_flash_rotating_agent_player_sets() -> tuple[
    tuple[str, dict[str, tuple[str, ...]], tuple[str, ...]],
    ...
]:
    return (
        (
            "Experimental_FlashEdgeRotator",
            {
                "bullish": (
                    "MomentumScalperSpotQuality",
                    "MomentumScalperShortOnly",
                    "FundingArb",
                ),
                "bearish": (
                    "MomentumScalperShortOnly",
                    "FundingArb",
                    "CrashPanicShortAgent",
                ),
                "neutral": (
                    "MomentumScalperSpotQuality",
                    "VolBreakoutSpotOnly",
                    "FundingArb",
                ),
                "crash": (
                    "CrashPanicShortAgent",
                    "MomentumScalperShortOnly",
                    "FundingArb",
                ),
            },
            (
                "MomentumScalperShortOnly",
                "MomentumScalperSpotQuality",
                "FundingArb",
                "CrashPanicShortAgent",
            ),
        ),
    )


def _clone_registry_with_experimental_flash_agents(
    registry: AgentRegistry,
) -> AgentRegistry:
    cloned = AgentRegistry()
    for agent in registry.all_agents():
        agent_clone = _clone_agent_for_shadow_registry(agent)
        if agent_clone is None:
            continue
        cloned.register(agent_clone, replace=True)
    register_experimental_flash_agents(cloned)
    return cloned


def _clone_agent_for_shadow_registry(agent: Any) -> Optional[Any]:
    clone = getattr(agent, "clone_for_shadow", None)
    if callable(clone):
        try:
            return clone()
        except Exception:
            return None
    try:
        return copy.deepcopy(agent)
    except Exception:
        return None


@dataclass(frozen=True)
class RetrodateMarketConfig:
    """Configuration for a Retrodate market replay benchmark."""

    data_dir: Path | str = Path("Retrodate")
    results_root: Path | str = Path("Results") / "RetrodateMarket"
    years: tuple[int, ...] = DEFAULT_YEARS
    timeframe: str = "1m"
    stride_minutes: int = 60
    initial_capital: float = 1000.0
    risk_capital_fraction: float = 0.10
    # Phase 4 / B7: реалистичный slippage в ретро-FakeExchange. 0.0 = старое
    # поведение (идеальные филлы по close). На лайве есть спред/проскальзывание,
    # поэтому нулевой slippage завышает PnL и сильнее раздувает частых акторов,
    # чем редкий ансамбль. Задавайте напр. 0.0005 (5 б.п.) для сопоставимости.
    slippage_pct: float = 0.0
    # Phase 2/3 opt-in флаги для валидационных прогонов (default = как в лайве по
    # умолчанию, т.е. выключено).
    voting_directional: bool = False
    flash_gate_pnl_per_trade_enabled: bool = False
    perf_max_returns_history: int = 0
    use_per_trade_pnl_score: bool = False
    per_trade_pnl_scale: float = 1.0
    min_eligible_score: float = 0.0
    # Consistency-tilted scoring: сместить вес с PnL (реверсит) на win-rate/sharpe
    # (стабильные предикторы). Defaults = текущие значения ScoringConfig.
    scoring_pnl_weight: float = 0.62
    scoring_sharpe_weight: float = 0.34
    scoring_win_bonus_divisor: float = 18.0
    flash_global_health_gate_enabled: bool = False
    flash_global_health_min_cum_pnl_pct: float = -5.0
    flash_global_health_min_closed_trades: int = 50
    include_optional_agents: bool = False
    optional_agent_labels: tuple[str, ...] = ()
    invalid_policy: str = "exclude"
    write_every_bars: int = 2000
    full_snapshot_every: int = 2000
    recompute_quarantine_every: int = 24
    progress_every_bars: int = 1000
    max_bars: Optional[int] = None
    flash_audit_events_enabled: bool = True
    shadow_audit_events_enabled: bool = True
    step_result_retention_enabled: bool = True
    compact_causal_entry_selected_only: bool = False
    shadow_agent_include_labels: tuple[str, ...] = ()
    shadow_player_include_labels: tuple[str, ...] = ()
    shadow_parallel_workers: int = 1
    shadow_position_diagnostic_bars: tuple[int, ...] = ()
    shadow_position_diagnostic_labels: tuple[str, ...] = ()
    solo_agent_candidate_limit: int = 3
    fixed_agent_players_enabled: bool = False
    fixed_agent_player_sets: tuple[tuple[str, tuple[str, ...]], ...] = ()
    player_profiles: tuple[Any, ...] = ()
    regime_switch_player_sets: tuple[Any, ...] = ()
    rotating_agent_player_sets: tuple[Any, ...] = ()
    flash_enabled: bool = False
    flash_min_score_to_trade: float = 0.0
    flash_actionable_bonus: float = 0.25
    flash_no_data_score: float = 0.0
    flash_min_closed_trades_to_trade: int = 3
    flash_min_pnl_pct_to_trade: float = 0.0
    flash_shadow_confirmation_enabled: bool = False
    flash_symbol_shadow_confirmation_enabled: bool = False
    flash_shadow_actor_fallback_confirmation_enabled: bool = False
    flash_shadow_base_fallback_confirmation_enabled: bool = False
    flash_shadow_signal_handoff_enabled: bool = False
    flash_genetics_confirmation_overlay_enabled: bool = False
    flash_genetics_confirmation_labels: tuple[str, ...] = ()
    flash_genetics_confirmation_allowed_signal_keys: tuple[str, ...] = ()
    flash_genetics_confirmation_contra_signal_keys: tuple[str, ...] = ()
    flash_genetics_contra_validation_manifest_path: Optional[Path | str] = None
    flash_genetics_confirmation_contra_side_match_enabled: bool = False
    flash_genetics_confirmation_contra_static_enabled: bool = False
    flash_genetics_confirmation_contra_no_backfill_enabled: bool = False
    flash_genetics_confirmation_contra_risk_sizing_enabled: bool = False
    flash_genetics_confirmation_contra_risk_mult: float = 1.0
    flash_genetics_confirmation_quality_gate_enabled: bool = False
    flash_genetics_confirmation_min_closed_trades: int = 0
    flash_genetics_confirmation_min_pnl_per_trade_pct: float = 0.0
    flash_genetics_confirmation_score_bonus: float = 0.0
    flash_genetics_confirmation_score_penalty: float = 0.0
    flash_genetics_confirmation_contra_score_penalty: float = 0.0
    flash_shadow_actor_fallback_min_base_score: float = 0.0
    flash_shadow_position_replay_actor_fallback_min_base_score: float = 0.0
    flash_shadow_position_replay_actor_fallback_min_shadow_score: float = 0.0
    flash_shadow_base_fallback_actor_keys: tuple[str, ...] = ()
    flash_actor_switch_margin: float = 0.0
    flash_anchor_actor_keys: tuple[str, ...] = ()
    flash_portfolio_actor_keys: tuple[str, ...] = ()
    flash_portfolio_shadow_bootstrap_min_closed_enabled: bool = False
    flash_anchor_min_score_to_trade: Optional[float] = None
    flash_anchor_shadow_min_score: Optional[float] = None
    flash_anchor_min_score_advantage: float = 0.0
    flash_prefer_solo_player_wrappers_enabled: bool = False
    flash_prefer_proven_solo_player_wrappers_enabled: bool = False
    flash_proven_solo_min_score_advantage: float = 0.0
    flash_shadow_min_score: float = 0.0
    flash_shadow_min_closed_trades: int = 50
    flash_shadow_min_full_open_closed_trades: int = 0
    flash_shadow_quality_confirmation_enabled: bool = False
    flash_shadow_min_win_rate_pct: float = 0.0
    flash_shadow_max_recent_downside_usd: float = 0.0
    flash_shadow_min_pnl_per_trade_lcb_usd: Optional[float] = None
    flash_shadow_pnl_per_trade_lcb_z: float = 1.0
    flash_shadow_pnl_per_trade_lcb_penalty_floor_usd: float = 0.0
    flash_shadow_pnl_per_trade_lcb_penalty_weight: float = 0.0
    flash_shadow_pnl_lcb_risk_sizing_enabled: bool = False
    flash_shadow_pnl_lcb_risk_min_mult: float = 0.25
    flash_shadow_pnl_lcb_risk_floor_usd: float = 0.0
    flash_shadow_pnl_lcb_risk_scale_usd: float = 1.0
    flash_shadow_symbol_health_enabled: bool = False
    flash_shadow_symbol_health_min_closed_trades: int = 0
    flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd: Optional[float] = None
    flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd: float = 0.0
    flash_shadow_symbol_health_pnl_lcb_penalty_weight: float = 0.0
    flash_actor_risk_sizing_enabled: bool = False
    flash_actor_risk_min_mult: float = 0.25
    flash_actor_risk_max_mult: float = 1.0
    flash_actor_risk_edge_scale_pct: float = 0.50
    flash_funding_score_weight: float = 0.0
    flash_funding_risk_mult_weight: float = 0.0
    flash_funding_risk_mult_cap: float = 0.25
    flash_no_trade_fee_saving_score_enabled: bool = False
    flash_no_trade_default_fee_bps: float = 0.0
    flash_volatility_risk_sizing_enabled: bool = False
    flash_volatility_risk_target_pct: float = 2.0
    flash_volatility_risk_min_volatility_pct: float = 0.5
    flash_volatility_risk_max_mult: float = 2.0
    flash_mixed_rotational_symbol_local_shadow_gate_enabled: bool = False
    flash_mixed_rotational_symbol_local_shadow_min_score: float = 4.0
    flash_mixed_rotational_symbol_local_shadow_min_closed_trades: int = 5
    flash_genetics_probation_bypass_min_closed_enabled: bool = False
    flash_genetics_probation_bypass_trend_gate_enabled: bool = False
    flash_technical_overlay_enabled: bool = False
    flash_technical_hard_gate_enabled: bool = False
    flash_technical_score_bonus: float = 0.10
    flash_technical_score_penalty: float = 0.25
    flash_technical_rsi_long_min: float = 45.0
    flash_technical_rsi_long_max: float = 72.0
    flash_technical_rsi_short_min: float = 28.0
    flash_technical_rsi_short_max: float = 55.0
    flash_technical_macd_histogram_min_abs_pct: float = 0.0
    flash_technical_atr_risk_sizing_enabled: bool = False
    flash_technical_atr_target_pct: float = 2.0
    flash_technical_atr_min_pct: float = 0.25
    flash_technical_atr_max_mult: float = 1.5
    flash_selected_subset_score_boosts: tuple[str, ...] = ()
    flash_selected_subset_context_score_boosts: tuple[str, ...] = ()
    flash_selected_subset_do_not_demote_signal_keys: tuple[str, ...] = ()
    flash_selected_subset_risk_mult_overrides: tuple[str, ...] = ()
    flash_selected_subset_context_risk_mult_overrides: tuple[str, ...] = ()
    flash_selected_subset_risk_min_mult: float = 0.75
    flash_selected_subset_risk_max_mult: float = 1.15
    flash_max_signals_per_actor: int = 0
    flash_overextension_guard_enabled: bool = False
    flash_overextension_lookback_bars: int = 12
    flash_short_overextension_return_floor_pct: float = -8.0
    flash_long_overextension_return_ceiling_pct: float = 8.0
    flash_overextension_volatility_normalized_enabled: bool = False
    flash_short_overextension_z_floor: float = -2.5
    flash_long_overextension_z_ceiling: float = 2.5
    flash_overextension_min_volatility_pct: float = 0.1
    flash_denied_signal_keys: tuple[str, ...] = ()
    flash_terminal_denied_signal_keys: tuple[str, ...] = ()
    flash_terminal_denied_context_signal_keys: tuple[str, ...] = ()
    flash_denied_open_symbols: tuple[str, ...] = ()
    flash_denied_open_regimes: tuple[str, ...] = ()
    flash_degradation_guard_enabled: bool = False
    flash_degradation_actor_guard_enabled: bool = False
    flash_degradation_actor_scope: str = "actor"
    flash_degradation_signal_cooldown_bars: int = 0
    flash_degradation_actor_cooldown_bars: int = 0
    flash_degradation_symbol_guard_enabled: bool = False
    flash_degradation_symbol_cooldown_bars: int = 0
    flash_degradation_symbol_lookback_bars: int = 0
    flash_degradation_symbol_window_closed_trades: int = 0
    flash_degradation_symbol_min_closed_trades: int = 0
    flash_degradation_symbol_max_recent_pnl_usd: Optional[float] = None
    flash_degradation_window_closed_trades: int = 3
    flash_degradation_min_closed_trades: int = 3
    flash_degradation_max_recent_pnl_usd: float = -25.0
    flash_degradation_signal_min_pnl_per_trade_lcb_usd: Optional[float] = None
    flash_degradation_pnl_per_trade_lcb_z: float = 1.0
    flash_degradation_signal_risk_sizing_enabled: bool = False
    flash_degradation_signal_risk_mult: float = 0.20
    flash_degradation_reserve_actor_cap: bool = False
    flash_degradation_recovery_enabled: bool = False
    flash_degradation_recovery_min_closed_trades: int = 3
    flash_degradation_recovery_min_recent_pnl_usd: float = 0.0
    flash_stale_position_exit_enabled: bool = False
    flash_stale_position_exit_max_age_bars: int = 168
    flash_stale_position_exit_require_nonpositive_unrealized: bool = True
    flash_partial_profit_lock_enabled: bool = False
    flash_partial_profit_lock_trigger_pnl_pct: float = 1.5
    flash_partial_profit_lock_close_fraction: float = 0.5
    flash_partial_profit_lock_min_age_bars: int = 2
    flash_partial_profit_lock_skip_protected_signal_keys: bool = True
    flash_partial_profit_lock_skip_signal_keys: tuple[str, ...] = ()
    flash_promotion_manifest_enabled: bool = False
    flash_promotion_manifest_path: Optional[Path | str] = None
    flash_promoted_signal_keys: tuple[str, ...] = ()
    flash_promotion_min_full_closed_trades: int = 50
    flash_promotion_min_latest_closed_trades: int = 10
    flash_promotion_min_full_pnl_pct: float = 0.0
    flash_promotion_min_latest_pnl_pct: float = 0.0
    flash_promotion_min_full_pnl_per_trade_lcb_pct: float = 0.0
    flash_promotion_min_latest_pnl_per_trade_lcb_pct: float = 0.0
    flash_promotion_pnl_per_trade_lcb_z: float = Z_95_ONE_SIDED
    flash_promotion_max_drawdown_pct: float = 25.0
    flash_promotion_min_win_rate_pct: float = 52.0
    flash_promotion_max_recent_downside_usd: float = 5.0
    flash_earned_cap_overrides_enabled: bool = False
    flash_promoted_actor_cap_overrides: tuple[str, ...] = ()
    experimental_flash_actors_enabled: bool = False
    experimental_flash_real_actors_enabled: bool = False
    actionable_fallback_enabled: bool = False
    actionable_fallback_min_score: Optional[float] = None
    actionable_fallback_require_has_data: bool = False
    soft_allocator_execution_enabled: bool = False
    soft_allocator_execution_soft_only: bool = False
    soft_allocator_execution_allow_labels: tuple[str, ...] = ()
    soft_allocator_execution_deny_actor_symbols: tuple[str, ...] = ()
    soft_allocator_realized_gate_enabled: bool = False
    soft_allocator_realized_gate_lookback_bars: int = 720
    soft_allocator_realized_gate_min_closed_trades: int = 12
    soft_allocator_realized_gate_max_recent_pnl_usd: float = -25.0
    soft_allocator_execution_policy: Optional[SoftAllocatorPolicy] = None
    current_actionable_candidate_layer_enabled: bool = False
    real_promotion_gate_enabled: bool = False
    real_promotion_min_closed_trades: int = 20
    real_promotion_min_pnl_pct: float = 0.0
    real_promotion_max_drawdown_pct: float = 25.0
    real_promotion_loss_budget_pct: float = -1.0
    real_promotion_probation_min_score: float = 0.0
    use_v3_rolling_score: bool = False
    use_v3_shadow_rolling_score: bool = False
    use_v3_soft_shadow_score: bool = False
    use_v3_executable_soft_top1_score: bool = False
    use_v3_executable_soft_confirmed_score: bool = False
    use_v3_entry_causal_score: bool = False
    v3_candidate_allow_labels: tuple[str, ...] = ()
    v3_shadow_position_gate_enabled: bool = True
    v3_shadow_flat_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_max_age_bars: int = 1
    v3_shadow_fresh_handoff_require_positive_unrealized: bool = True
    v3_shadow_rolling_window_bars: int = 24
    v3_shadow_rolling_min_closed_trades: int = 50
    v3_entry_causal_min_filled: int = 3
    v3_entry_causal_actionability_weight: float = 1.0
    v3_real_loss_rescue_enabled: bool = False
    v3_real_loss_rescue_min_virtual_pnl_pct: float = 10.0
    v3_real_loss_rescue_max_virtual_dd_pct: float = 50.0
    v3_real_loss_rescue_min_actionable_share: float = 0.05
    v3_real_loss_rescue_min_recent_filled: int = 1
    v3_real_loss_rescue_max_real_loss_pct: float = -3.0
    v3_real_loss_rescue_allow_genetics: bool = False
    v3_probation_shadow_rescue_enabled: bool = False
    v3_probation_shadow_rescue_min_virtual_pnl_pct: float = 10.0
    v3_probation_shadow_rescue_max_virtual_dd_pct: float = 50.0
    v3_probation_shadow_rescue_min_actionable_share: float = 0.05
    v3_probation_shadow_rescue_min_recent_filled: int = 1
    v3_probation_shadow_rescue_min_recent_pnl_usd: float = 0.0
    v3_probation_shadow_rescue_allow_genetics: bool = False
    v3_persistent_loss_kill_min_closed_trades: int = 0
    v3_persistent_loss_kill_pnl_pct: float = -2.0
    v3_persistent_loss_kill_win_rate_pct: float = 0.0
    v3_persistent_loss_requires_virtual_weakness: bool = True
    v3_persistent_loss_virtual_max_pnl_pct: float = 0.0
    v3_persistent_loss_virtual_min_dd_pct: float = 25.0
    v3_probation_loss_kill_min_closed_trades: int = 0
    v3_probation_loss_kill_pnl_pct: float = -0.15
    v3_probation_loss_kill_win_rate_pct: float = 50.0
    v3_probation_loss_kill_label_prefixes: tuple[str, ...] = (
        DEFAULT_PROBATION_LOSS_LABEL_PREFIXES
    )
    max_new_opens_per_bar: int = 1
    risk_max_open_positions: int = 8
    risk_max_leverage: int = 5
    apply_risk_leverage_to_notional: bool = False
    genetics_probation_execution_enabled: bool = False
    genetics_probation_labels: tuple[str, ...] = ("GeneticsResearch",)
    genetics_probation_allowed_regimes: tuple[str, ...] = ("bearish", "crash")
    genetics_probation_allowed_signal_keys: tuple[str, ...] = ()
    genetics_probation_risk_mult: float = 0.25
    genetics_probation_min_regime_confidence: float = 0.0
    genetics_probation_max_real_trades: int = 20
    genetics_probation_max_daily_trades: int = 0
    genetics_probation_require_shadow_confirmation: bool = True
    v3_realized_profit_lock_min_closed_trades: int = 0
    v3_realized_profit_lock_min_peak_pnl_pct: float = 0.75
    v3_realized_profit_lock_max_giveback_pct: float = 0.55
    v3_realized_profit_lock_floor_pnl_pct: float = 0.25
    v3_panteon_equity_guard_enabled: bool = False
    v3_panteon_equity_guard_min_peak_pnl_pct: float = 2.0
    v3_panteon_equity_guard_max_giveback_pct: float = 1.0
    v3_panteon_equity_guard_floor_pnl_pct: float = 2.0
    v3_panteon_equity_guard_cooldown_bars: int = 720
    hard_policy_deny_labels: tuple[str, ...] = DEFAULT_RETRO_HARD_POLICY_DENY_LABELS
    hard_policy_enabled: bool = True

    def __post_init__(self) -> None:
        if self.stride_minutes <= 0:
            raise ValueError("stride_minutes must be > 0")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not 0.0 < self.risk_capital_fraction <= 1.0:
            raise ValueError("risk_capital_fraction must be in (0, 1]")
        if self.slippage_pct < 0.0:
            raise ValueError("slippage_pct must be >= 0")
        if self.solo_agent_candidate_limit < 1:
            raise ValueError("solo_agent_candidate_limit must be >= 1")
        object.__setattr__(
            self,
            "shadow_position_diagnostic_bars",
            tuple(
                dict.fromkeys(
                    int(bar)
                    for bar in self.shadow_position_diagnostic_bars
                    if int(bar) > 0
                )
            ),
        )
        object.__setattr__(
            self,
            "shadow_position_diagnostic_labels",
            tuple(
                dict.fromkeys(
                    str(label).strip()
                    for label in self.shadow_position_diagnostic_labels
                    if str(label).strip()
                )
            ),
        )
        if self.invalid_policy not in {"exclude", "fail"}:
            raise ValueError("invalid_policy must be 'exclude' or 'fail'")
        if self.flash_overextension_lookback_bars <= 0:
            raise ValueError("flash_overextension_lookback_bars must be > 0")
        if self.flash_overextension_min_volatility_pct <= 0:
            raise ValueError("flash_overextension_min_volatility_pct must be > 0")
        if self.flash_short_overextension_z_floor >= 0:
            raise ValueError("flash_short_overextension_z_floor must be < 0")
        if self.flash_long_overextension_z_ceiling <= 0:
            raise ValueError("flash_long_overextension_z_ceiling must be > 0")
        if self.flash_promotion_min_full_closed_trades < 0:
            raise ValueError("flash_promotion_min_full_closed_trades must be >= 0")
        if self.flash_promotion_min_latest_closed_trades < 0:
            raise ValueError("flash_promotion_min_latest_closed_trades must be >= 0")
        if self.flash_promotion_max_drawdown_pct < 0:
            raise ValueError("flash_promotion_max_drawdown_pct must be >= 0")
        if self.flash_promotion_min_win_rate_pct < 0:
            raise ValueError("flash_promotion_min_win_rate_pct must be >= 0")
        if self.flash_promotion_pnl_per_trade_lcb_z < 0:
            raise ValueError("flash_promotion_pnl_per_trade_lcb_z must be >= 0")
        if self.flash_shadow_pnl_per_trade_lcb_z < 0:
            raise ValueError("flash_shadow_pnl_per_trade_lcb_z must be >= 0")
        if self.flash_shadow_pnl_per_trade_lcb_penalty_weight < 0:
            raise ValueError(
                "flash_shadow_pnl_per_trade_lcb_penalty_weight must be >= 0"
            )
        if self.flash_shadow_pnl_lcb_risk_min_mult < 0:
            raise ValueError("flash_shadow_pnl_lcb_risk_min_mult must be >= 0")
        if self.flash_shadow_pnl_lcb_risk_min_mult > 1.0:
            raise ValueError("flash_shadow_pnl_lcb_risk_min_mult must be <= 1")
        if self.flash_shadow_pnl_lcb_risk_scale_usd <= 0:
            raise ValueError("flash_shadow_pnl_lcb_risk_scale_usd must be > 0")
        if self.flash_shadow_symbol_health_min_closed_trades < 0:
            raise ValueError("flash_shadow_symbol_health_min_closed_trades must be >= 0")
        if self.flash_shadow_symbol_health_pnl_lcb_penalty_weight < 0:
            raise ValueError(
                "flash_shadow_symbol_health_pnl_lcb_penalty_weight must be >= 0"
            )
        if self.flash_shadow_position_replay_actor_fallback_min_base_score < 0:
            raise ValueError(
                "flash_shadow_position_replay_actor_fallback_min_base_score must be >= 0"
            )
        if self.flash_shadow_position_replay_actor_fallback_min_shadow_score < 0:
            raise ValueError(
                "flash_shadow_position_replay_actor_fallback_min_shadow_score must be >= 0"
            )
        if self.flash_actor_risk_min_mult < 0:
            raise ValueError("flash_actor_risk_min_mult must be >= 0")
        if self.flash_actor_risk_max_mult <= 0:
            raise ValueError("flash_actor_risk_max_mult must be > 0")
        if self.flash_actor_risk_min_mult > self.flash_actor_risk_max_mult:
            raise ValueError("flash_actor_risk_min_mult must be <= flash_actor_risk_max_mult")
        if self.flash_actor_risk_edge_scale_pct <= 0:
            raise ValueError("flash_actor_risk_edge_scale_pct must be > 0")
        if self.flash_funding_score_weight < 0:
            raise ValueError("flash_funding_score_weight must be >= 0")
        if self.flash_funding_risk_mult_weight < 0:
            raise ValueError("flash_funding_risk_mult_weight must be >= 0")
        if self.flash_funding_risk_mult_cap < 0:
            raise ValueError("flash_funding_risk_mult_cap must be >= 0")
        if self.flash_no_trade_default_fee_bps < 0:
            raise ValueError("flash_no_trade_default_fee_bps must be >= 0")
        if self.flash_volatility_risk_target_pct <= 0:
            raise ValueError("flash_volatility_risk_target_pct must be > 0")
        if self.flash_volatility_risk_min_volatility_pct <= 0:
            raise ValueError("flash_volatility_risk_min_volatility_pct must be > 0")
        if self.flash_volatility_risk_max_mult <= 0:
            raise ValueError("flash_volatility_risk_max_mult must be > 0")
        if self.flash_mixed_rotational_symbol_local_shadow_min_score < 0:
            raise ValueError(
                "flash_mixed_rotational_symbol_local_shadow_min_score must be >= 0"
            )
        if self.flash_mixed_rotational_symbol_local_shadow_min_closed_trades < 0:
            raise ValueError(
                "flash_mixed_rotational_symbol_local_shadow_min_closed_trades must be >= 0"
            )
        if self.flash_technical_score_bonus < 0:
            raise ValueError("flash_technical_score_bonus must be >= 0")
        if self.flash_technical_score_penalty < 0:
            raise ValueError("flash_technical_score_penalty must be >= 0")
        if not (
            0.0
            <= self.flash_technical_rsi_long_min
            <= self.flash_technical_rsi_long_max
            <= 100.0
        ):
            raise ValueError(
                "flash technical RSI long bounds must be ordered within [0, 100]"
            )
        if not (
            0.0
            <= self.flash_technical_rsi_short_min
            <= self.flash_technical_rsi_short_max
            <= 100.0
        ):
            raise ValueError(
                "flash technical RSI short bounds must be ordered within [0, 100]"
            )
        if self.flash_technical_macd_histogram_min_abs_pct < 0:
            raise ValueError(
                "flash_technical_macd_histogram_min_abs_pct must be >= 0"
            )
        if self.flash_technical_atr_target_pct <= 0:
            raise ValueError("flash_technical_atr_target_pct must be > 0")
        if self.flash_technical_atr_min_pct <= 0:
            raise ValueError("flash_technical_atr_min_pct must be > 0")
        if self.flash_technical_atr_max_mult <= 0:
            raise ValueError("flash_technical_atr_max_mult must be > 0")
        if self.flash_selected_subset_risk_min_mult < 0:
            raise ValueError("flash_selected_subset_risk_min_mult must be >= 0")
        if self.flash_selected_subset_risk_max_mult <= 0:
            raise ValueError("flash_selected_subset_risk_max_mult must be > 0")
        if (
            self.flash_selected_subset_risk_min_mult
            > self.flash_selected_subset_risk_max_mult
        ):
            raise ValueError(
                "flash_selected_subset_risk_min_mult must be <= flash_selected_subset_risk_max_mult"
            )
        if self.flash_partial_profit_lock_trigger_pnl_pct < 0:
            raise ValueError("flash_partial_profit_lock_trigger_pnl_pct must be >= 0")
        if not 0.0 < self.flash_partial_profit_lock_close_fraction <= 1.0:
            raise ValueError(
                "flash_partial_profit_lock_close_fraction must be in (0, 1]"
            )
        if self.flash_partial_profit_lock_min_age_bars < 0:
            raise ValueError("flash_partial_profit_lock_min_age_bars must be >= 0")
        object.__setattr__(
            self,
            "flash_partial_profit_lock_skip_signal_keys",
            tuple(
                str(item).strip()
                for item in self.flash_partial_profit_lock_skip_signal_keys
                if str(item or "").strip()
            ),
        )
        if self.flash_promotion_max_recent_downside_usd < 0:
            raise ValueError("flash_promotion_max_recent_downside_usd must be >= 0")
        object.__setattr__(
            self,
            "optional_agent_labels",
            tuple(str(label) for label in self.optional_agent_labels),
        )
        object.__setattr__(
            self,
            "shadow_agent_include_labels",
            _normalize_label_tuple(self.shadow_agent_include_labels),
        )
        object.__setattr__(
            self,
            "shadow_player_include_labels",
            _normalize_label_tuple(self.shadow_player_include_labels),
        )
        raw_genetics_confirmation_labels = self.flash_genetics_confirmation_labels
        genetics_confirmation_labels = (
            raw_genetics_confirmation_labels.split(",")
            if isinstance(raw_genetics_confirmation_labels, str)
            else raw_genetics_confirmation_labels
        )
        object.__setattr__(
            self,
            "flash_genetics_confirmation_labels",
            tuple(
                str(label).strip()
                for label in (genetics_confirmation_labels or ())
                if str(label).strip()
            ),
        )
        contra_validation_manifest_path = (
            None
            if self.flash_genetics_contra_validation_manifest_path in (None, "")
            else Path(self.flash_genetics_contra_validation_manifest_path)
        )
        object.__setattr__(
            self,
            "flash_genetics_contra_validation_manifest_path",
            contra_validation_manifest_path,
        )
        object.__setattr__(
            self,
            "flash_genetics_confirmation_contra_signal_keys",
            _load_flash_validated_contra_signal_keys(
                contra_validation_manifest_path,
                explicit_keys=self.flash_genetics_confirmation_contra_signal_keys,
            ),
        )
        if self.flash_genetics_confirmation_score_bonus < 0:
            raise ValueError("flash_genetics_confirmation_score_bonus must be >= 0")
        if self.flash_genetics_confirmation_score_penalty < 0:
            raise ValueError("flash_genetics_confirmation_score_penalty must be >= 0")
        if self.flash_genetics_confirmation_contra_score_penalty < 0:
            raise ValueError(
                "flash_genetics_confirmation_contra_score_penalty must be >= 0"
            )
        if self.flash_genetics_confirmation_contra_risk_mult < 0:
            raise ValueError(
                "flash_genetics_confirmation_contra_risk_mult must be >= 0"
            )
        if self.flash_genetics_confirmation_min_closed_trades < 0:
            raise ValueError(
                "flash_genetics_confirmation_min_closed_trades must be >= 0"
            )
        object.__setattr__(
            self,
            "hard_policy_deny_labels",
            tuple(str(label) for label in self.hard_policy_deny_labels),
        )
        fixed_sets = _normalize_fixed_agent_player_sets(self.fixed_agent_player_sets)
        if self.fixed_agent_players_enabled and not fixed_sets:
            fixed_sets = DEFAULT_FIXED_AGENT_PLAYER_SETS
        object.__setattr__(self, "fixed_agent_player_sets", fixed_sets)
        if self.real_promotion_min_closed_trades < 0:
            raise ValueError("real_promotion_min_closed_trades must be >= 0")
        if self.real_promotion_max_drawdown_pct < 0:
            raise ValueError("real_promotion_max_drawdown_pct must be >= 0")
        if self.max_new_opens_per_bar < 0:
            raise ValueError("max_new_opens_per_bar must be >= 0")
        if self.shadow_parallel_workers < 1:
            raise ValueError("shadow_parallel_workers must be >= 1")
        if self.risk_max_open_positions < 0:
            raise ValueError("risk_max_open_positions must be >= 0")
        if self.risk_max_leverage < 1:
            raise ValueError("risk_max_leverage must be >= 1")
        if self.genetics_probation_max_daily_trades < 0:
            raise ValueError("genetics_probation_max_daily_trades must be >= 0")
        if self.flash_degradation_window_closed_trades <= 0:
            raise ValueError("flash_degradation_window_closed_trades must be > 0")
        if self.flash_degradation_actor_scope not in {"actor", "actor_regime"}:
            raise ValueError(
                "flash_degradation_actor_scope must be 'actor' or 'actor_regime'"
            )
        if self.flash_degradation_signal_cooldown_bars < 0:
            raise ValueError("flash_degradation_signal_cooldown_bars must be >= 0")
        if self.flash_degradation_actor_cooldown_bars < 0:
            raise ValueError("flash_degradation_actor_cooldown_bars must be >= 0")
        if self.flash_degradation_symbol_cooldown_bars < 0:
            raise ValueError("flash_degradation_symbol_cooldown_bars must be >= 0")
        if self.flash_degradation_symbol_lookback_bars < 0:
            raise ValueError("flash_degradation_symbol_lookback_bars must be >= 0")
        if self.flash_degradation_symbol_window_closed_trades < 0:
            raise ValueError("flash_degradation_symbol_window_closed_trades must be >= 0")
        if self.flash_degradation_symbol_min_closed_trades < 0:
            raise ValueError("flash_degradation_symbol_min_closed_trades must be >= 0")
        if (
            self.flash_degradation_symbol_window_closed_trades > 0
            and self.flash_degradation_symbol_min_closed_trades
            > self.flash_degradation_symbol_window_closed_trades
        ):
            raise ValueError(
                "flash_degradation_symbol_min_closed_trades must be "
                "<= flash_degradation_symbol_window_closed_trades"
            )
        if self.flash_degradation_min_closed_trades <= 0:
            raise ValueError("flash_degradation_min_closed_trades must be > 0")
        if (
            self.flash_degradation_min_closed_trades
            > self.flash_degradation_window_closed_trades
        ):
            raise ValueError(
                "flash_degradation_min_closed_trades must be "
                "<= flash_degradation_window_closed_trades"
            )
        if self.flash_degradation_pnl_per_trade_lcb_z < 0:
            raise ValueError("flash_degradation_pnl_per_trade_lcb_z must be >= 0")
        if self.flash_degradation_signal_risk_mult < 0:
            raise ValueError("flash_degradation_signal_risk_mult must be >= 0")
        if self.flash_degradation_signal_risk_mult > 1.0:
            raise ValueError("flash_degradation_signal_risk_mult must be <= 1")
        if self.flash_degradation_recovery_enabled:
            if self.flash_degradation_recovery_min_closed_trades <= 0:
                raise ValueError(
                    "flash_degradation_recovery_min_closed_trades must be > 0"
                )
            if (
                self.flash_degradation_recovery_min_closed_trades
                > self.flash_degradation_window_closed_trades
            ):
                raise ValueError(
                    "flash_degradation_recovery_min_closed_trades must be "
                    "<= flash_degradation_window_closed_trades"
                )
        if self.flash_stale_position_exit_max_age_bars < 0:
            raise ValueError("flash_stale_position_exit_max_age_bars must be >= 0")
        manifest_path = (
            None
            if self.flash_promotion_manifest_path in (None, "")
            else Path(self.flash_promotion_manifest_path)
        )
        object.__setattr__(self, "flash_promotion_manifest_path", manifest_path)
        object.__setattr__(
            self,
            "flash_promoted_signal_keys",
            _load_flash_promoted_signal_keys(
                manifest_path,
                explicit_keys=self.flash_promoted_signal_keys,
            ),
        )
        object.__setattr__(
            self,
            "flash_promoted_actor_cap_overrides",
            tuple(
                str(item).strip()
                for item in self.flash_promoted_actor_cap_overrides
                if str(item or "").strip()
            ),
        )
        if (
            self.actionable_fallback_min_score is not None
            and not isinstance(self.actionable_fallback_min_score, (int, float))
        ):
            raise ValueError("actionable_fallback_min_score must be numeric or None")
        if self.v3_persistent_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_persistent_loss_kill_min_closed_trades must be >= 0")
        if self.v3_probation_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_probation_loss_kill_min_closed_trades must be >= 0")
        if not (0.0 < self.genetics_probation_risk_mult <= 1.0):
            raise ValueError("genetics_probation_risk_mult must be in (0, 1]")
        if not (0.0 <= self.genetics_probation_min_regime_confidence <= 1.0):
            raise ValueError("genetics_probation_min_regime_confidence must be in [0, 1]")
        if self.genetics_probation_max_real_trades < 0:
            raise ValueError("genetics_probation_max_real_trades must be >= 0")
        for name in (
            "genetics_probation_labels",
            "genetics_probation_allowed_regimes",
            "genetics_probation_allowed_signal_keys",
        ):
            raw_value = getattr(self, name)
            values = (raw_value,) if isinstance(raw_value, str) else raw_value
            object.__setattr__(
                self,
                name,
                tuple(
                    str(item).strip()
                    for item in (values or ())
                    if str(item).strip()
                ),
            )
        if self.v3_realized_profit_lock_min_closed_trades < 0:
            raise ValueError("v3_realized_profit_lock_min_closed_trades must be >= 0")
        if self.v3_shadow_rolling_window_bars < 1:
            raise ValueError("v3_shadow_rolling_window_bars must be >= 1")
        if self.v3_shadow_rolling_min_closed_trades < 0:
            raise ValueError("v3_shadow_rolling_min_closed_trades must be >= 0")
        if self.v3_entry_causal_min_filled < 0:
            raise ValueError("v3_entry_causal_min_filled must be >= 0")
        if self.v3_entry_causal_actionability_weight < 0:
            raise ValueError("v3_entry_causal_actionability_weight must be >= 0")
        if self.v3_shadow_fresh_handoff_max_age_bars < 0:
            raise ValueError("v3_shadow_fresh_handoff_max_age_bars must be >= 0")
        if not 0.0 <= self.v3_real_loss_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_real_loss_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_real_loss_rescue_min_recent_filled < 0:
            raise ValueError("v3_real_loss_rescue_min_recent_filled must be >= 0")
        if self.v3_real_loss_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_real_loss_rescue_max_virtual_dd_pct must be >= 0")
        if not 0.0 <= self.v3_probation_shadow_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_probation_shadow_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_probation_shadow_rescue_min_recent_filled < 0:
            raise ValueError("v3_probation_shadow_rescue_min_recent_filled must be >= 0")
        if self.v3_probation_shadow_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_probation_shadow_rescue_max_virtual_dd_pct must be >= 0")
        if not 0.0 <= self.v3_persistent_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_persistent_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_persistent_loss_virtual_min_dd_pct < 0:
            raise ValueError("v3_persistent_loss_virtual_min_dd_pct must be >= 0")
        if not 0.0 <= self.v3_probation_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_probation_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_realized_profit_lock_min_peak_pnl_pct < 0:
            raise ValueError("v3_realized_profit_lock_min_peak_pnl_pct must be >= 0")
        if self.v3_realized_profit_lock_max_giveback_pct < 0:
            raise ValueError("v3_realized_profit_lock_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_min_peak_pnl_pct < 0:
            raise ValueError("v3_panteon_equity_guard_min_peak_pnl_pct must be >= 0")
        if self.v3_panteon_equity_guard_max_giveback_pct < 0:
            raise ValueError("v3_panteon_equity_guard_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_cooldown_bars < 0:
            raise ValueError("v3_panteon_equity_guard_cooldown_bars must be >= 0")
        object.__setattr__(
            self,
            "v3_probation_loss_kill_label_prefixes",
            tuple(
                str(prefix)
                for prefix in self.v3_probation_loss_kill_label_prefixes
                if str(prefix)
            ),
        )


@dataclass(frozen=True)
class RetrodateFileSelection:
    """Validated Retrodate files selected for a benchmark run."""

    report: RetrodateDirReport
    valid_reports: tuple[RetrodateFileReport, ...]
    excluded_files: tuple[RetrodateFileReport, ...]
    missing_years: tuple[int, ...]

    @property
    def valid_files(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.valid_reports)

    @property
    def executed_years(self) -> tuple[int, ...]:
        return tuple(
            int(item.expected_year)
            for item in self.valid_reports
            if item.expected_year is not None
        )


@dataclass
class RetrodateSnapshotState:
    """Mutable state preserved while loading multiple yearly files."""

    next_bar: int = 1
    btc_closes: Deque[float] = field(default_factory=lambda: deque(maxlen=24))
    symbol_closes: dict[str, Deque[float]] = field(default_factory=dict)
    technicals: TechnicalIndicatorState = field(default_factory=TechnicalIndicatorState)


@dataclass(frozen=True)
class RetrodateRunSummary:
    """Summary returned after a benchmark run finishes."""

    output_dir: Path
    report_path: Path
    summary_path: Path
    requested_years: tuple[int, ...]
    executed_years: tuple[int, ...]
    excluded_files: tuple[str, ...]
    missing_years: tuple[int, ...]
    bars_processed: int
    registered_agents: tuple[str, ...]
    player_profile_count: int
    first_timestamp: str
    last_timestamp: str
    stride_minutes: int
    max_bars: Optional[int]
    step_errors: tuple[str, ...]


def select_retrodate_files(config: RetrodateMarketConfig) -> RetrodateFileSelection:
    """Validate and select requested Retrodate CSV files."""
    root = Path(config.data_dir)
    report = validate_retrodate_dir(root)
    if report.issues:
        raise RetrodateValidationError(_format_directory_issues(report))

    requested = tuple(int(year) for year in config.years)
    selected = [
        item
        for item in report.files
        if item.expected_year in requested and item.timeframe == config.timeframe
    ]
    excluded = tuple(item for item in selected if not item.is_valid)
    if excluded and config.invalid_policy == "fail":
        raise RetrodateValidationError(_format_selection_errors(excluded))

    valid_reports = tuple(
        sorted(
            (item for item in selected if item.is_valid),
            key=lambda item: (item.expected_year or 0, item.path.name),
        )
    )
    seen_years = {item.expected_year for item in valid_reports}
    missing_years = tuple(year for year in requested if year not in seen_years)
    if not valid_reports:
        raise RetrodateValidationError(
            "No valid Retrodate files selected for requested years: "
            + ", ".join(str(year) for year in requested)
        )
    return RetrodateFileSelection(
        report=report,
        valid_reports=valid_reports,
        excluded_files=excluded,
        missing_years=missing_years,
    )


def load_retrodate_year_snapshots(
    csv_path: str | Path,
    *,
    stride_minutes: int,
    state: RetrodateSnapshotState,
    max_snapshots: Optional[int] = None,
) -> list[MarketSnapshot]:
    """Load one yearly CSV into chronological multi-symbol market snapshots."""
    if stride_minutes <= 0:
        raise ValueError("stride_minutes must be > 0")
    stride_ms = int(stride_minutes) * 60 * 1000
    snapshot_limit = (
        max(0, int(max_snapshots))
        if max_snapshots is not None
        else None
    )
    if snapshot_limit == 0:
        return []
    prices_by_ts: dict[int, dict[str, float]] = {}
    candles_by_ts: dict[int, dict[str, tuple[float, float, float]]] = {}
    volumes_by_ts: dict[int, dict[str, float]] = {}

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = int(str(row.get("timestamp") or "").strip())
            except ValueError:
                continue
            if timestamp % stride_ms != 0:
                continue
            if (
                snapshot_limit is not None
                and len(prices_by_ts) >= snapshot_limit
                and timestamp not in prices_by_ts
            ):
                break
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            close = _coerce_float(row.get("close"))
            if close <= 0:
                continue
            high = _coerce_float(row.get("high"))
            low = _coerce_float(row.get("low"))
            if high <= 0 or low <= 0 or high < low:
                high = close
                low = close
            prices_by_ts.setdefault(timestamp, {})[symbol] = close
            candles_by_ts.setdefault(timestamp, {})[symbol] = (high, low, close)
            volumes_by_ts.setdefault(timestamp, {})[symbol] = max(0.0, _coerce_float(row.get("volume")))

    snapshots: list[MarketSnapshot] = []
    for timestamp in sorted(prices_by_ts):
        prices = dict(sorted(prices_by_ts[timestamp].items()))
        volumes = {
            symbol: volumes_by_ts.get(timestamp, {}).get(symbol, 0.0)
            for symbol in prices
        }
        lookback_returns_pct = _snapshot_lookback_returns_pct(prices, state)
        lookback_volatility_pct = _snapshot_lookback_volatility_pct(prices, state)
        technicals_by_symbol = {
            symbol: state.technicals.update_symbol(
                symbol,
                high=candles_by_ts[timestamp][symbol][0],
                low=candles_by_ts[timestamp][symbol][1],
                close=candles_by_ts[timestamp][symbol][2],
            )
            for symbol in prices
        }
        regime, confidence = _classify_regime(prices, state)
        regimes_by_symbol = _classify_symbol_regimes(
            prices,
            state,
            fallback=regime,
        )
        dt = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc)
        snapshots.append(
            MarketSnapshot(
                bar=state.next_bar,
                timestamp=dt,
                regime=regime,
                regime_confidence=confidence,
                prices=prices,
                volumes=volumes,
                month=dt.month,
                lookback_returns_pct=lookback_returns_pct,
                lookback_volatility_pct=lookback_volatility_pct,
                technicals_by_symbol=technicals_by_symbol,
                regimes_by_symbol=regimes_by_symbol,
            )
        )
        _update_symbol_close_history(prices, state)
        state.next_bar += 1
    return snapshots


def _snapshot_lookback_returns_pct(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
) -> dict[str, dict[int, float]]:
    returns: dict[str, dict[int, float]] = {}
    for symbol, price in prices.items():
        history = state.symbol_closes.get(symbol)
        if not history:
            continue
        for lookback in (12, 24):
            if len(history) < lookback:
                continue
            old_price = history[-lookback]
            if old_price <= 0:
                continue
            returns.setdefault(symbol, {})[lookback] = (float(price) / float(old_price) - 1.0) * 100.0
    return returns


def _snapshot_lookback_volatility_pct(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
) -> dict[str, dict[int, float]]:
    volatilities: dict[str, dict[int, float]] = {}
    for symbol, price in prices.items():
        history = state.symbol_closes.get(symbol)
        if not history:
            continue
        for lookback in (12, 24):
            if len(history) < lookback:
                continue
            closes = [float(value) for value in list(history)[-lookback:]]
            closes.append(float(price))
            returns_pct: list[float] = []
            for previous, current in zip(closes, closes[1:]):
                if previous <= 0:
                    continue
                returns_pct.append((current / previous - 1.0) * 100.0)
            if not returns_pct:
                continue
            rms = math.sqrt(sum(value * value for value in returns_pct) / len(returns_pct))
            volatilities.setdefault(symbol, {})[lookback] = rms
    return volatilities


def _update_symbol_close_history(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
) -> None:
    for symbol, price in prices.items():
        state.symbol_closes.setdefault(symbol, deque(maxlen=24)).append(float(price))


def run_retrodate_market_benchmark(config: RetrodateMarketConfig) -> RetrodateRunSummary:
    """Run the selected Retrodate files through the live-like Panteon pipeline."""
    selection = select_retrodate_files(config)
    output_dir = _new_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=False)

    registry = AgentRegistry()
    registered_agent_labels = list(
        register_all_v1_agents(
            registry,
            include_optional=config.include_optional_agents,
            optional_agent_labels=(
                config.optional_agent_labels
                if config.optional_agent_labels
                else None
            ),
            skip_on_error=True,
        )
    )
    shadow_registry = registry
    if (
        config.experimental_flash_actors_enabled
        and config.experimental_flash_real_actors_enabled
    ):
        registered_agent_labels.extend(register_experimental_flash_agents(registry))
    elif config.experimental_flash_actors_enabled:
        shadow_registry = _clone_registry_with_experimental_flash_agents(registry)
        registered_agent_labels.extend(
            label
            for label in experimental_flash_agent_labels()
            if shadow_registry.has(label)
        )
    registered_agents = tuple(registered_agent_labels)
    exchange = FakeExchange(name="RETRODATE_MARKET")
    # Phase 4 / B7: применяем реалистичный slippage (0.0 = идеальные филлы).
    if config.slippage_pct > 0.0:
        exchange.set_slippage_pct(config.slippage_pct)
    strategist_config = _build_strategist_config(config)
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=config.initial_capital,
        profiles=(config.player_profiles or None),
        strategist_config=strategist_config,
        live_execution_config=_build_live_execution_config(config),
        risk_config=_build_risk_config(config),
        flash_enabled=config.flash_enabled,
        flash_allocator_config=_build_flash_allocator_config(config),
        scoring_config=_build_scoring_config(config),
        perf_max_returns_history=int(config.perf_max_returns_history),
        voting_directional=bool(config.voting_directional),
    )
    pipeline.mode = "retrodate_market"
    pipeline.timeframe = f"{config.stride_minutes}m-from-{config.timeframe}"
    pipeline.session_id = output_dir.name
    pipeline.run_id = output_dir.name
    pipeline.shadow_agent_labels = tuple(shadow_registry.all_labels())
    pipeline._flash_promoted_signal_keys = tuple(config.flash_promoted_signal_keys)
    pipeline.actionable_fallback_enabled = bool(config.actionable_fallback_enabled)
    pipeline.actionable_fallback_min_score = config.actionable_fallback_min_score
    pipeline.actionable_fallback_require_has_data = bool(
        config.actionable_fallback_require_has_data
    )
    if (
        config.soft_allocator_execution_enabled
        and config.soft_allocator_execution_policy is not None
    ):
        pipeline.soft_allocator_execution_enabled = True
        pipeline.soft_allocator_execution_soft_only = bool(
            config.soft_allocator_execution_soft_only
        )
        pipeline.soft_allocator_execution_allow_labels = tuple(
            str(label or "").strip()
            for label in config.soft_allocator_execution_allow_labels
            if str(label or "").strip()
        )
        pipeline.soft_allocator_execution_deny_actor_symbols = tuple(
            str(item or "").strip()
            for item in config.soft_allocator_execution_deny_actor_symbols
            if str(item or "").strip()
        )
        pipeline.soft_allocator_realized_gate_enabled = bool(
            config.soft_allocator_realized_gate_enabled
        )
        pipeline.soft_allocator_realized_gate_lookback_bars = int(
            config.soft_allocator_realized_gate_lookback_bars
        )
        pipeline.soft_allocator_realized_gate_min_closed_trades = int(
            config.soft_allocator_realized_gate_min_closed_trades
        )
        pipeline.soft_allocator_realized_gate_max_recent_pnl_usd = float(
            config.soft_allocator_realized_gate_max_recent_pnl_usd
        )
        pipeline.soft_allocator_execution = OnlineSoftAllocator(
            policy=config.soft_allocator_execution_policy,
            initial_capital=config.initial_capital,
        )
    pipeline.current_actionable_candidate_layer_enabled = bool(
        config.current_actionable_candidate_layer_enabled
        or config.use_v3_executable_soft_confirmed_score
    )
    if config.regime_switch_player_sets:
        pipeline.regime_switch_player_sets = tuple(config.regime_switch_player_sets)
    if config.rotating_agent_player_sets:
        pipeline.rotating_agent_player_sets = tuple(config.rotating_agent_player_sets)
    pipeline.flash_stale_position_exit_enabled = bool(
        config.flash_stale_position_exit_enabled
    )
    pipeline.flash_stale_position_exit_max_age_bars = int(
        config.flash_stale_position_exit_max_age_bars
    )
    pipeline.flash_stale_position_exit_require_nonpositive_unrealized = bool(
        config.flash_stale_position_exit_require_nonpositive_unrealized
    )
    pipeline.flash_partial_profit_lock_enabled = bool(
        config.flash_partial_profit_lock_enabled
    )
    pipeline.flash_partial_profit_lock_trigger_pnl_pct = float(
        config.flash_partial_profit_lock_trigger_pnl_pct
    )
    pipeline.flash_partial_profit_lock_close_fraction = float(
        config.flash_partial_profit_lock_close_fraction
    )
    pipeline.flash_partial_profit_lock_min_age_bars = int(
        config.flash_partial_profit_lock_min_age_bars
    )
    pipeline.flash_partial_profit_lock_skip_protected_signal_keys = bool(
        config.flash_partial_profit_lock_skip_protected_signal_keys
    )
    pipeline.flash_partial_profit_lock_skip_signal_keys = tuple(
        config.flash_partial_profit_lock_skip_signal_keys
    )
    pipeline.flash_selected_subset_do_not_demote_signal_keys = tuple(
        config.flash_selected_subset_do_not_demote_signal_keys
    )
    pipeline.flash_audit_events_enabled = bool(config.flash_audit_events_enabled)
    pipeline.shadow_position_diagnostic_bars = tuple(
        config.shadow_position_diagnostic_bars
    )
    pipeline.shadow_position_diagnostic_labels = tuple(
        config.shadow_position_diagnostic_labels
    )
    pipeline.solo_agent_candidate_limit = int(config.solo_agent_candidate_limit)
    fixed_agent_player_sets = (
        tuple(config.fixed_agent_player_sets)
        if config.fixed_agent_players_enabled
        else ()
    )
    fixed_agent_player_sets += _genetics_probation_fixed_agent_player_sets(config)
    if (
        config.experimental_flash_actors_enabled
        and config.experimental_flash_real_actors_enabled
    ):
        fixed_agent_player_sets += _experimental_flash_fixed_agent_player_sets()
        pipeline.rotating_agent_player_sets = (
            tuple(pipeline.rotating_agent_player_sets)
            + _experimental_flash_rotating_agent_player_sets()
        )
    pipeline.fixed_agent_player_sets = fixed_agent_player_sets
    shadow_event_log = EventLog() if config.shadow_audit_events_enabled else None
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=shadow_registry,
        perf=pipeline.virtual_perf,
        risk_config=pipeline.risk_config,
        event_log=shadow_event_log,
        runtime_event_logs_enabled=bool(config.shadow_audit_events_enabled),
        agent_include_labels=config.shadow_agent_include_labels,
        player_include_labels=config.shadow_player_include_labels,
        parallel_workers=config.shadow_parallel_workers,
    )

    writer = OutputWriter(
        pipeline,
        OutputWriterConfig(
            output_dir=str(output_dir),
            write_every_bars=max(1, int(config.write_every_bars)),
            full_snapshot_every=max(1, int(config.full_snapshot_every)),
            compact_causal_entry_decisions=True,
            compact_causal_entry_selected_only=bool(
                config.compact_causal_entry_selected_only
            ),
            latest_dir=str(Path(config.results_root).resolve()),
        ),
    )

    state = RetrodateSnapshotState()
    bars_processed = 0
    first_timestamp = ""
    last_timestamp = ""
    step_errors: list[str] = []
    processed_years: list[int] = []

    try:
        for file_report in selection.valid_reports:
            remaining = None
            if config.max_bars is not None:
                remaining = max(0, int(config.max_bars) - bars_processed)
                if remaining <= 0:
                    break
            snapshots = load_retrodate_year_snapshots(
                file_report.path,
                stride_minutes=config.stride_minutes,
                state=state,
                max_snapshots=remaining,
            )
            if remaining is not None:
                snapshots = snapshots[:remaining]
            if not snapshots:
                continue
            if not first_timestamp:
                first_timestamp = snapshots[0].timestamp.isoformat()
            last_timestamp = snapshots[-1].timestamp.isoformat()
            processed_years.append(int(file_report.expected_year or 0))
            feed = ReplayFeed(snapshots=snapshots)
            processed_counter = [0]
            steps = main_loop(
                pipeline,
                feed,
                recompute_quarantine_every=max(1, int(config.recompute_quarantine_every)),
                on_step=_make_on_step(
                    writer,
                    config,
                    step_errors=(
                        None
                        if config.step_result_retention_enabled
                        else step_errors
                    ),
                    processed_counter=processed_counter,
                ),
                collect_results=bool(config.step_result_retention_enabled),
            )
            bars_processed += (
                len(steps)
                if config.step_result_retention_enabled
                else int(processed_counter[0])
            )
            if config.step_result_retention_enabled:
                step_errors.extend(str(step.error) for step in steps if step.error)
    finally:
        writer.close()

    try:
        trading_log = output_dir / "trading.log"
        if trading_log.exists():
            write_allocation_diagnostics(
                output_dir,
                analyze_trading_log(trading_log),
            )
    except Exception as exc:
        step_errors.append(
            f"allocation_diagnostics_failed: {type(exc).__name__}: {exc}"
        )

    try:
        write_candidate_diagnostics(
            output_dir,
            candidate_events=tuple(
                pipeline.event_log.query(event_types=[CandidateScored])
            ),
            rejection_events=tuple(
                pipeline.event_log.query(event_types=[CandidateRejected])
            ),
        )
    except Exception as exc:
        step_errors.append(
            f"candidate_diagnostics_failed: {type(exc).__name__}: {exc}"
        )

    try:
        write_flash_attribution_summary(
            output_dir,
            execution_events=tuple(
                pipeline.event_log.query(event_types=[ExecutionAttributed])
            ),
            position_closed_events=tuple(
                pipeline.event_log.query(event_types=[PositionClosed])
            ),
        )
    except Exception as exc:
        step_errors.append(
            f"flash_attribution_summary_failed: {type(exc).__name__}: {exc}"
        )

    try:
        _write_walk_forward_report_from_event_log(output_dir, pipeline.event_log)
    except Exception as exc:
        step_errors.append(
            f"walk_forward_report_failed: {type(exc).__name__}: {exc}"
        )

    try:
        shadow_updates = (
            tuple(shadow_event_log.query(event_types=[ShadowActorUpdated]))
            if shadow_event_log is not None
            else ()
        )
        shadow_player_pnl_events = shadow_pnl_events_from_shadow_updates(shadow_updates)
        shadow_pnl_events = shadow_player_pnl_events
        shadow_agent_pnl_events = shadow_pnl_events_from_shadow_updates(
            shadow_updates,
            actor_type="agent",
        )
        soft_report = simulate_soft_allocator_policies(
            shadow_pnl_events,
            initial_capital=config.initial_capital,
        )
        perfect_report = simulate_perfect_monthly_panteon(
            shadow_pnl_events,
            initial_capital=config.initial_capital,
        )
        write_soft_allocator_report(output_dir, soft_report)
        write_perfect_panteon_report(output_dir, perfect_report)
        write_shadow_pnl_events(output_dir, shadow_pnl_events)
        write_shadow_pnl_events(
            output_dir,
            shadow_agent_pnl_events,
            filename="shadow_agent_pnl_events.jsonl",
        )
        _write_retrodate_final_artifact_reports(
            output_dir,
            config=config,
            shadow_updates=shadow_updates,
            shadow_agent_pnl_events=shadow_agent_pnl_events,
            shadow_player_pnl_events=shadow_player_pnl_events,
            candidate_rejections=tuple(
                pipeline.event_log.query(event_types=[CandidateRejected])
            ),
            step_errors=step_errors,
        )
    except Exception as exc:
        step_errors.append(f"soft_allocator_report_failed: {type(exc).__name__}: {exc}")

    report_path = output_dir / "analysis_report.md"
    summary_path = output_dir / "run_summary.json"
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=report_path,
        summary_path=summary_path,
        requested_years=tuple(config.years),
        executed_years=tuple(year for year in processed_years if year),
        excluded_files=tuple(item.path.name for item in selection.excluded_files),
        missing_years=selection.missing_years,
        bars_processed=bars_processed,
        registered_agents=registered_agents,
        player_profile_count=len(pipeline.profiles),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        stride_minutes=config.stride_minutes,
        max_bars=config.max_bars,
        step_errors=tuple(step_errors[:50]),
    )
    _write_run_summary(summary_path, summary, selection, config)
    _write_analysis_report(report_path, summary, selection)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = _parse_cli_config(argv)
    summary = run_retrodate_market_benchmark(config)
    print(f"output_dir={summary.output_dir}", flush=True)
    print(f"analysis_report={summary.report_path}", flush=True)
    print(f"run_summary={summary.summary_path}", flush=True)
    return 0


def _parse_cli_config(argv: Optional[Sequence[str]] = None) -> RetrodateMarketConfig:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    return RetrodateMarketConfig(
        data_dir=Path(args.data_dir),
        results_root=Path(args.results_root),
        years=_parse_years(args.years),
        timeframe=args.timeframe,
        stride_minutes=args.stride_minutes,
        initial_capital=args.initial_capital,
        risk_capital_fraction=args.risk_capital_fraction,
        slippage_pct=args.slippage_pct,
        voting_directional=args.voting_directional,
        flash_gate_pnl_per_trade_enabled=args.flash_gate_pnl_per_trade,
        perf_max_returns_history=args.perf_max_returns_history,
        use_per_trade_pnl_score=args.use_per_trade_pnl_score,
        per_trade_pnl_scale=args.per_trade_pnl_scale,
        min_eligible_score=args.min_eligible_score,
        scoring_pnl_weight=args.scoring_pnl_weight,
        scoring_sharpe_weight=args.scoring_sharpe_weight,
        scoring_win_bonus_divisor=args.scoring_win_bonus_divisor,
        flash_global_health_gate_enabled=args.flash_global_health_gate,
        flash_global_health_min_cum_pnl_pct=args.flash_global_health_min_cum_pnl_pct,
        flash_global_health_min_closed_trades=args.flash_global_health_min_closed_trades,
        risk_max_leverage=args.risk_max_leverage,
        apply_risk_leverage_to_notional=args.apply_risk_leverage_to_notional,
        include_optional_agents=args.include_optional_agents,
        optional_agent_labels=_parse_optional_agent_labels(args.optional_agent_labels),
        invalid_policy=args.invalid_policy,
        write_every_bars=args.write_every_bars,
        full_snapshot_every=args.full_snapshot_every,
        recompute_quarantine_every=args.recompute_quarantine_every,
        progress_every_bars=args.progress_every_bars,
        max_bars=args.max_bars,
        flash_audit_events_enabled=not args.disable_flash_audit_events,
        shadow_audit_events_enabled=not args.disable_shadow_audit_events,
        step_result_retention_enabled=not args.disable_step_result_retention,
        compact_causal_entry_selected_only=args.compact_causal_entry_selected_only,
        shadow_agent_include_labels=_normalize_label_tuple(
            args.shadow_agent_include_label
        ),
        shadow_player_include_labels=_normalize_label_tuple(
            args.shadow_player_include_label
        ),
        shadow_parallel_workers=args.shadow_parallel_workers,
        shadow_position_diagnostic_bars=tuple(
            int(bar)
            for bar in args.shadow_position_diagnostic_bar
            if int(bar) > 0
        ),
        shadow_position_diagnostic_labels=tuple(
            str(label).strip()
            for label in args.shadow_position_diagnostic_label
            if str(label).strip()
        ),
        solo_agent_candidate_limit=args.solo_agent_candidate_limit,
        fixed_agent_players_enabled=(
            args.enable_fixed_agent_players or bool(args.fixed_agent_player_set)
        ),
        fixed_agent_player_sets=_parse_fixed_agent_player_sets(args.fixed_agent_player_set),
        flash_enabled=args.enable_flash,
        flash_min_score_to_trade=args.flash_min_score_to_trade,
        flash_actionable_bonus=args.flash_actionable_bonus,
        flash_no_data_score=args.flash_no_data_score,
        flash_min_closed_trades_to_trade=args.flash_min_closed_trades_to_trade,
        flash_min_pnl_pct_to_trade=args.flash_min_pnl_pct_to_trade,
        flash_shadow_confirmation_enabled=args.enable_flash_shadow_confirmation,
        flash_symbol_shadow_confirmation_enabled=(
            args.enable_flash_symbol_shadow_confirmation
        ),
        flash_shadow_actor_fallback_confirmation_enabled=(
            args.enable_flash_shadow_actor_fallback_confirmation
        ),
        flash_shadow_base_fallback_confirmation_enabled=(
            args.enable_flash_shadow_base_fallback_confirmation
        ),
        flash_shadow_signal_handoff_enabled=(
            args.enable_flash_shadow_signal_handoff
        ),
        flash_genetics_confirmation_overlay_enabled=(
            args.enable_flash_genetics_confirmation_overlay
        ),
        flash_genetics_confirmation_labels=tuple(
            str(label).strip()
            for label in args.flash_genetics_confirmation_label
            if str(label).strip()
        ),
        flash_genetics_confirmation_allowed_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_genetics_confirmation_allowed_signal_key
            if str(key).strip()
        ),
        flash_genetics_confirmation_contra_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_genetics_confirmation_contra_signal_key
            if str(key).strip()
        ),
        flash_genetics_contra_validation_manifest_path=(
            Path(args.flash_genetics_contra_validation_manifest_path)
            if args.flash_genetics_contra_validation_manifest_path
            else None
        ),
        flash_genetics_confirmation_contra_side_match_enabled=(
            args.enable_flash_genetics_confirmation_contra_side_match
        ),
        flash_genetics_confirmation_contra_static_enabled=(
            args.enable_flash_genetics_confirmation_contra_static
        ),
        flash_genetics_confirmation_contra_no_backfill_enabled=(
            args.enable_flash_genetics_confirmation_contra_no_backfill
        ),
        flash_genetics_confirmation_contra_risk_sizing_enabled=(
            args.enable_flash_genetics_confirmation_contra_risk_sizing
        ),
        flash_genetics_confirmation_contra_risk_mult=(
            args.flash_genetics_confirmation_contra_risk_mult
        ),
        flash_genetics_confirmation_quality_gate_enabled=(
            args.enable_flash_genetics_confirmation_quality_gate
        ),
        flash_genetics_confirmation_min_closed_trades=(
            args.flash_genetics_confirmation_min_closed_trades
        ),
        flash_genetics_confirmation_min_pnl_per_trade_pct=(
            args.flash_genetics_confirmation_min_pnl_per_trade_pct
        ),
        flash_genetics_confirmation_score_bonus=(
            args.flash_genetics_confirmation_score_bonus
        ),
        flash_genetics_confirmation_score_penalty=(
            args.flash_genetics_confirmation_score_penalty
        ),
        flash_genetics_confirmation_contra_score_penalty=(
            args.flash_genetics_confirmation_contra_score_penalty
        ),
        flash_shadow_actor_fallback_min_base_score=(
            args.flash_shadow_actor_fallback_min_base_score
        ),
        flash_shadow_position_replay_actor_fallback_min_base_score=(
            args.flash_shadow_position_replay_actor_fallback_min_base_score
        ),
        flash_shadow_position_replay_actor_fallback_min_shadow_score=(
            args.flash_shadow_position_replay_actor_fallback_min_shadow_score
        ),
        flash_shadow_base_fallback_actor_keys=tuple(
            str(key).strip()
            for key in args.flash_shadow_base_fallback_actor_key
            if str(key).strip()
        ),
        flash_actor_switch_margin=args.flash_actor_switch_margin,
        flash_anchor_actor_keys=tuple(
            str(key).strip()
            for key in args.flash_anchor_actor_key
            if str(key).strip()
        ),
        flash_portfolio_actor_keys=tuple(
            str(key).strip()
            for key in args.flash_portfolio_actor_key
            if str(key).strip()
        ),
        flash_portfolio_shadow_bootstrap_min_closed_enabled=(
            args.enable_flash_portfolio_shadow_bootstrap_min_closed
        ),
        flash_anchor_min_score_to_trade=args.flash_anchor_min_score_to_trade,
        flash_anchor_shadow_min_score=args.flash_anchor_shadow_min_score,
        flash_anchor_min_score_advantage=args.flash_anchor_min_score_advantage,
        flash_prefer_solo_player_wrappers_enabled=(
            args.enable_flash_prefer_solo_player_wrappers
        ),
        flash_prefer_proven_solo_player_wrappers_enabled=(
            args.enable_flash_prefer_proven_solo_player_wrappers
        ),
        flash_proven_solo_min_score_advantage=(
            args.flash_proven_solo_min_score_advantage
        ),
        flash_shadow_min_score=args.flash_shadow_min_score,
        flash_shadow_min_closed_trades=args.flash_shadow_min_closed_trades,
        flash_shadow_min_full_open_closed_trades=(
            args.flash_shadow_min_full_open_closed_trades
        ),
        flash_shadow_quality_confirmation_enabled=(
            args.enable_flash_shadow_quality_confirmation
        ),
        flash_shadow_min_win_rate_pct=args.flash_shadow_min_win_rate_pct,
        flash_shadow_max_recent_downside_usd=(
            args.flash_shadow_max_recent_downside_usd
        ),
        flash_shadow_min_pnl_per_trade_lcb_usd=(
            args.flash_shadow_min_pnl_per_trade_lcb_usd
        ),
        flash_shadow_pnl_per_trade_lcb_z=(
            args.flash_shadow_pnl_per_trade_lcb_z
        ),
        flash_shadow_pnl_per_trade_lcb_penalty_floor_usd=(
            args.flash_shadow_pnl_per_trade_lcb_penalty_floor_usd
        ),
        flash_shadow_pnl_per_trade_lcb_penalty_weight=(
            args.flash_shadow_pnl_per_trade_lcb_penalty_weight
        ),
        flash_shadow_pnl_lcb_risk_sizing_enabled=(
            args.enable_flash_shadow_pnl_lcb_risk_sizing
        ),
        flash_shadow_pnl_lcb_risk_min_mult=(
            args.flash_shadow_pnl_lcb_risk_min_mult
        ),
        flash_shadow_pnl_lcb_risk_floor_usd=(
            args.flash_shadow_pnl_lcb_risk_floor_usd
        ),
        flash_shadow_pnl_lcb_risk_scale_usd=(
            args.flash_shadow_pnl_lcb_risk_scale_usd
        ),
        flash_shadow_symbol_health_enabled=(
            args.enable_flash_shadow_symbol_health
        ),
        flash_shadow_symbol_health_min_closed_trades=(
            args.flash_shadow_symbol_health_min_closed_trades
        ),
        flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd=(
            args.flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd
        ),
        flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd=(
            args.flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd
        ),
        flash_shadow_symbol_health_pnl_lcb_penalty_weight=(
            args.flash_shadow_symbol_health_pnl_lcb_penalty_weight
        ),
        flash_actor_risk_sizing_enabled=args.enable_flash_actor_risk_sizing,
        flash_actor_risk_min_mult=args.flash_actor_risk_min_mult,
        flash_actor_risk_max_mult=args.flash_actor_risk_max_mult,
        flash_actor_risk_edge_scale_pct=args.flash_actor_risk_edge_scale_pct,
        flash_funding_score_weight=args.flash_funding_score_weight,
        flash_funding_risk_mult_weight=args.flash_funding_risk_mult_weight,
        flash_funding_risk_mult_cap=args.flash_funding_risk_mult_cap,
        flash_no_trade_fee_saving_score_enabled=(
            args.enable_flash_no_trade_fee_saving_score
        ),
        flash_no_trade_default_fee_bps=args.flash_no_trade_default_fee_bps,
        flash_volatility_risk_sizing_enabled=(
            args.enable_flash_volatility_risk_sizing
        ),
        flash_volatility_risk_target_pct=args.flash_volatility_risk_target_pct,
        flash_volatility_risk_min_volatility_pct=(
            args.flash_volatility_risk_min_volatility_pct
        ),
        flash_volatility_risk_max_mult=args.flash_volatility_risk_max_mult,
        flash_mixed_rotational_symbol_local_shadow_gate_enabled=(
            args.enable_flash_mixed_rotational_symbol_local_shadow_gate
        ),
        flash_mixed_rotational_symbol_local_shadow_min_score=(
            args.flash_mixed_rotational_symbol_local_shadow_min_score
        ),
        flash_mixed_rotational_symbol_local_shadow_min_closed_trades=(
            args.flash_mixed_rotational_symbol_local_shadow_min_closed_trades
        ),
        flash_genetics_probation_bypass_min_closed_enabled=(
            args.enable_flash_genetics_probation_bypass_min_closed
        ),
        flash_genetics_probation_bypass_trend_gate_enabled=(
            args.enable_flash_genetics_probation_bypass_trend_gate
        ),
        flash_technical_overlay_enabled=args.enable_flash_technical_overlay,
        flash_technical_hard_gate_enabled=args.enable_flash_technical_hard_gate,
        flash_technical_score_bonus=args.flash_technical_score_bonus,
        flash_technical_score_penalty=args.flash_technical_score_penalty,
        flash_technical_rsi_long_min=args.flash_technical_rsi_long_min,
        flash_technical_rsi_long_max=args.flash_technical_rsi_long_max,
        flash_technical_rsi_short_min=args.flash_technical_rsi_short_min,
        flash_technical_rsi_short_max=args.flash_technical_rsi_short_max,
        flash_technical_macd_histogram_min_abs_pct=(
            args.flash_technical_macd_histogram_min_abs_pct
        ),
        flash_technical_atr_risk_sizing_enabled=(
            args.enable_flash_technical_atr_risk_sizing
        ),
        flash_technical_atr_target_pct=args.flash_technical_atr_target_pct,
        flash_technical_atr_min_pct=args.flash_technical_atr_min_pct,
        flash_technical_atr_max_mult=args.flash_technical_atr_max_mult,
        flash_selected_subset_score_boosts=tuple(
            str(item).strip()
            for item in args.flash_selected_subset_score_boost
            if str(item).strip()
        ),
        flash_selected_subset_context_score_boosts=tuple(
            str(item).strip()
            for item in args.flash_selected_subset_context_score_boost
            if str(item).strip()
        ),
        flash_selected_subset_do_not_demote_signal_keys=tuple(
            str(item).strip()
            for item in args.flash_selected_subset_do_not_demote_signal_key
            if str(item).strip()
        ),
        flash_selected_subset_risk_mult_overrides=tuple(
            str(item).strip()
            for item in args.flash_selected_subset_risk_mult
            if str(item).strip()
        ),
        flash_selected_subset_context_risk_mult_overrides=tuple(
            str(item).strip()
            for item in args.flash_selected_subset_context_risk_mult
            if str(item).strip()
        ),
        flash_selected_subset_risk_min_mult=(
            args.flash_selected_subset_risk_min_mult
        ),
        flash_selected_subset_risk_max_mult=(
            args.flash_selected_subset_risk_max_mult
        ),
        flash_partial_profit_lock_enabled=args.enable_flash_partial_profit_lock,
        flash_partial_profit_lock_trigger_pnl_pct=(
            args.flash_partial_profit_lock_trigger_pnl_pct
        ),
        flash_partial_profit_lock_close_fraction=(
            args.flash_partial_profit_lock_close_fraction
        ),
        flash_partial_profit_lock_min_age_bars=(
            args.flash_partial_profit_lock_min_age_bars
        ),
        flash_partial_profit_lock_skip_protected_signal_keys=(
            not args.disable_flash_partial_profit_lock_skip_protected_signal_keys
        ),
        flash_partial_profit_lock_skip_signal_keys=tuple(
            str(item).strip()
            for item in args.flash_partial_profit_lock_skip_signal_key
            if str(item or "").strip()
        ),
        flash_max_signals_per_actor=args.flash_max_signals_per_actor,
        flash_overextension_guard_enabled=args.enable_flash_overextension_guard,
        flash_overextension_lookback_bars=args.flash_overextension_lookback_bars,
        flash_short_overextension_return_floor_pct=(
            args.flash_short_overextension_return_floor_pct
        ),
        flash_long_overextension_return_ceiling_pct=(
            args.flash_long_overextension_return_ceiling_pct
        ),
        flash_overextension_volatility_normalized_enabled=(
            args.enable_flash_overextension_volatility_normalized
        ),
        flash_short_overextension_z_floor=args.flash_short_overextension_z_floor,
        flash_long_overextension_z_ceiling=args.flash_long_overextension_z_ceiling,
        flash_overextension_min_volatility_pct=(
            args.flash_overextension_min_volatility_pct
        ),
        flash_denied_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_deny_signal_key
            if str(key).strip()
        ),
        flash_terminal_denied_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_terminal_deny_signal_key
            if str(key).strip()
        ),
        flash_terminal_denied_context_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_terminal_deny_context_signal_key
            if str(key).strip()
        ),
        flash_denied_open_symbols=tuple(
            str(symbol).strip()
            for symbol in args.flash_denied_open_symbol
            if str(symbol).strip()
        ),
        flash_denied_open_regimes=tuple(
            str(regime).strip()
            for regime in args.flash_denied_open_regime
            if str(regime).strip()
        ),
        flash_degradation_guard_enabled=args.enable_flash_degradation_guard,
        flash_degradation_actor_guard_enabled=(
            args.enable_flash_degradation_actor_guard
        ),
        flash_degradation_actor_scope=args.flash_degradation_actor_scope,
        flash_degradation_signal_cooldown_bars=(
            args.flash_degradation_signal_cooldown_bars
        ),
        flash_degradation_actor_cooldown_bars=(
            args.flash_degradation_actor_cooldown_bars
        ),
        flash_degradation_symbol_guard_enabled=(
            args.enable_flash_degradation_symbol_guard
        ),
        flash_degradation_symbol_cooldown_bars=(
            args.flash_degradation_symbol_cooldown_bars
        ),
        flash_degradation_symbol_lookback_bars=(
            args.flash_degradation_symbol_lookback_bars
        ),
        flash_degradation_symbol_window_closed_trades=(
            args.flash_degradation_symbol_window_closed_trades
        ),
        flash_degradation_symbol_min_closed_trades=(
            args.flash_degradation_symbol_min_closed_trades
        ),
        flash_degradation_symbol_max_recent_pnl_usd=(
            args.flash_degradation_symbol_max_recent_pnl_usd
        ),
        flash_degradation_window_closed_trades=(
            args.flash_degradation_window_closed_trades
        ),
        flash_degradation_min_closed_trades=(
            args.flash_degradation_min_closed_trades
        ),
        flash_degradation_max_recent_pnl_usd=(
            args.flash_degradation_max_recent_pnl_usd
        ),
        flash_degradation_signal_min_pnl_per_trade_lcb_usd=(
            args.flash_degradation_signal_min_pnl_per_trade_lcb_usd
        ),
        flash_degradation_pnl_per_trade_lcb_z=(
            args.flash_degradation_pnl_per_trade_lcb_z
        ),
        flash_degradation_signal_risk_sizing_enabled=(
            args.enable_flash_degradation_signal_risk_sizing
        ),
        flash_degradation_signal_risk_mult=(
            args.flash_degradation_signal_risk_mult
        ),
        flash_degradation_reserve_actor_cap=(
            args.flash_degradation_reserve_actor_cap
        ),
        flash_degradation_recovery_enabled=(
            args.enable_flash_degradation_recovery
        ),
        flash_degradation_recovery_min_closed_trades=(
            args.flash_degradation_recovery_min_closed_trades
        ),
        flash_degradation_recovery_min_recent_pnl_usd=(
            args.flash_degradation_recovery_min_recent_pnl_usd
        ),
        flash_stale_position_exit_enabled=(
            args.enable_flash_stale_position_exit
        ),
        flash_stale_position_exit_max_age_bars=(
            args.flash_stale_position_exit_max_age_bars
        ),
        flash_stale_position_exit_require_nonpositive_unrealized=(
            not args.flash_stale_position_exit_allow_profitable
        ),
        flash_promotion_manifest_enabled=args.enable_flash_promotion_manifest,
        flash_promotion_manifest_path=(
            Path(args.flash_promotion_manifest_path)
            if args.flash_promotion_manifest_path
            else None
        ),
        flash_promoted_signal_keys=tuple(
            str(key).strip()
            for key in args.flash_promoted_signal_key
            if str(key).strip()
        ),
        flash_promotion_min_full_closed_trades=(
            args.flash_promotion_min_full_closed_trades
        ),
        flash_promotion_min_latest_closed_trades=(
            args.flash_promotion_min_latest_closed_trades
        ),
        flash_promotion_min_full_pnl_pct=args.flash_promotion_min_full_pnl_pct,
        flash_promotion_min_latest_pnl_pct=args.flash_promotion_min_latest_pnl_pct,
        flash_promotion_min_full_pnl_per_trade_lcb_pct=(
            args.flash_promotion_min_full_pnl_per_trade_lcb_pct
        ),
        flash_promotion_min_latest_pnl_per_trade_lcb_pct=(
            args.flash_promotion_min_latest_pnl_per_trade_lcb_pct
        ),
        flash_promotion_pnl_per_trade_lcb_z=(
            args.flash_promotion_pnl_per_trade_lcb_z
        ),
        flash_promotion_max_drawdown_pct=args.flash_promotion_max_drawdown_pct,
        flash_promotion_min_win_rate_pct=args.flash_promotion_min_win_rate_pct,
        flash_promotion_max_recent_downside_usd=(
            args.flash_promotion_max_recent_downside_usd
        ),
        flash_earned_cap_overrides_enabled=args.enable_flash_earned_cap_overrides,
        flash_promoted_actor_cap_overrides=tuple(
            str(item).strip()
            for item in args.flash_promoted_actor_cap_override
            if str(item).strip()
        ),
        experimental_flash_actors_enabled=(
            args.enable_experimental_flash_actors
            or args.allow_experimental_flash_real_actors
        ),
        experimental_flash_real_actors_enabled=(
            args.allow_experimental_flash_real_actors
        ),
        hard_policy_deny_labels=_merge_hard_policy_deny_labels(
            args.hard_policy_deny_label
        ),
        hard_policy_enabled=not args.disable_hard_policy,
        actionable_fallback_enabled=args.enable_actionable_fallback,
        actionable_fallback_min_score=args.actionable_fallback_min_score,
        actionable_fallback_require_has_data=args.actionable_fallback_require_has_data,
        current_actionable_candidate_layer_enabled=(
            args.enable_current_actionable_candidate_layer
        ),
        real_promotion_gate_enabled=args.real_promotion_gate,
        real_promotion_min_closed_trades=args.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=args.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=args.real_promotion_max_dd_pct,
        real_promotion_loss_budget_pct=args.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=args.real_promotion_probation_min_score,
        use_v3_rolling_score=args.use_v3_rolling_score,
        use_v3_shadow_rolling_score=args.use_v3_shadow_rolling_score,
        use_v3_soft_shadow_score=args.use_v3_soft_shadow_score,
        use_v3_executable_soft_top1_score=(
            args.use_v3_executable_soft_top1_score
        ),
        use_v3_executable_soft_confirmed_score=(
            args.use_v3_executable_soft_confirmed_score
        ),
        use_v3_entry_causal_score=args.use_v3_entry_causal_score,
        v3_shadow_position_gate_enabled=not args.disable_v3_shadow_position_gate,
        v3_shadow_flat_handoff_enabled=args.enable_v3_shadow_flat_handoff,
        v3_shadow_fresh_handoff_enabled=args.enable_v3_shadow_fresh_handoff,
        v3_shadow_fresh_handoff_max_age_bars=args.v3_shadow_fresh_handoff_max_age_bars,
        v3_shadow_fresh_handoff_require_positive_unrealized=(
            not args.v3_shadow_fresh_handoff_allow_nonpositive_unrealized
        ),
        v3_shadow_rolling_window_bars=args.v3_shadow_rolling_window_bars,
        v3_shadow_rolling_min_closed_trades=args.v3_shadow_rolling_min_closed_trades,
        v3_entry_causal_min_filled=args.v3_entry_causal_min_filled,
        v3_entry_causal_actionability_weight=args.v3_entry_causal_actionability_weight,
        v3_real_loss_rescue_enabled=args.enable_v3_real_loss_rescue,
        v3_real_loss_rescue_min_virtual_pnl_pct=(
            args.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        v3_real_loss_rescue_max_virtual_dd_pct=(
            args.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        v3_real_loss_rescue_min_actionable_share=(
            args.v3_real_loss_rescue_min_actionable_share
        ),
        v3_real_loss_rescue_min_recent_filled=(
            args.v3_real_loss_rescue_min_recent_filled
        ),
        v3_real_loss_rescue_max_real_loss_pct=(
            args.v3_real_loss_rescue_max_real_loss_pct
        ),
        v3_real_loss_rescue_allow_genetics=args.allow_genetics_real_loss_rescue,
        v3_probation_shadow_rescue_enabled=args.enable_v3_probation_shadow_rescue,
        v3_probation_shadow_rescue_min_virtual_pnl_pct=(
            args.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        v3_probation_shadow_rescue_max_virtual_dd_pct=(
            args.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        v3_probation_shadow_rescue_min_actionable_share=(
            args.v3_probation_shadow_rescue_min_actionable_share
        ),
        v3_probation_shadow_rescue_min_recent_filled=(
            args.v3_probation_shadow_rescue_min_recent_filled
        ),
        v3_probation_shadow_rescue_min_recent_pnl_usd=(
            args.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        v3_probation_shadow_rescue_allow_genetics=(
            args.allow_genetics_probation_shadow_rescue
        ),
        v3_persistent_loss_kill_min_closed_trades=(
            args.v3_persistent_loss_kill_min_closed_trades
        ),
        v3_persistent_loss_kill_pnl_pct=args.v3_persistent_loss_kill_pnl_pct,
        v3_persistent_loss_kill_win_rate_pct=args.v3_persistent_loss_kill_win_rate_pct,
        v3_persistent_loss_requires_virtual_weakness=(
            not args.v3_persistent_loss_ignore_virtual_quality
        ),
        v3_persistent_loss_virtual_max_pnl_pct=(
            args.v3_persistent_loss_virtual_max_pnl_pct
        ),
        v3_persistent_loss_virtual_min_dd_pct=(
            args.v3_persistent_loss_virtual_min_dd_pct
        ),
        v3_probation_loss_kill_min_closed_trades=(
            args.v3_probation_loss_kill_min_closed_trades
        ),
        v3_probation_loss_kill_pnl_pct=args.v3_probation_loss_kill_pnl_pct,
        v3_probation_loss_kill_win_rate_pct=(
            args.v3_probation_loss_kill_win_rate_pct
        ),
        v3_probation_loss_kill_label_prefixes=(
            _parse_probation_loss_label_prefixes(args)
        ),
        max_new_opens_per_bar=args.max_new_opens_per_bar,
        risk_max_open_positions=args.risk_max_open_positions,
        genetics_probation_execution_enabled=(
            args.enable_genetics_probation_execution
        ),
        genetics_probation_labels=tuple(
            str(label).strip()
            for label in (args.genetics_probation_label or ("GeneticsResearch",))
            if str(label).strip()
        ),
        genetics_probation_allowed_regimes=tuple(
            str(regime).strip()
            for regime in (
                args.genetics_probation_allowed_regime or ("bearish", "crash")
            )
            if str(regime).strip()
        ),
        genetics_probation_allowed_signal_keys=tuple(
            str(key).strip()
            for key in args.genetics_probation_allowed_signal_key
            if str(key).strip()
        ),
        genetics_probation_risk_mult=args.genetics_probation_risk_mult,
        genetics_probation_min_regime_confidence=(
            args.genetics_probation_min_regime_confidence
        ),
        genetics_probation_max_real_trades=args.genetics_probation_max_real_trades,
        genetics_probation_max_daily_trades=args.genetics_probation_max_daily_trades,
        genetics_probation_require_shadow_confirmation=(
            not args.disable_genetics_probation_shadow_confirmation
        ),
        v3_realized_profit_lock_min_closed_trades=(
            args.v3_realized_profit_lock_min_closed_trades
        ),
        v3_realized_profit_lock_min_peak_pnl_pct=(
            args.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        v3_realized_profit_lock_max_giveback_pct=(
            args.v3_realized_profit_lock_max_giveback_pct
        ),
        v3_realized_profit_lock_floor_pnl_pct=(
            args.v3_realized_profit_lock_floor_pnl_pct
        ),
        v3_panteon_equity_guard_enabled=args.enable_v3_panteon_equity_guard,
        v3_panteon_equity_guard_min_peak_pnl_pct=(
            args.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        v3_panteon_equity_guard_max_giveback_pct=(
            args.v3_panteon_equity_guard_max_giveback_pct
        ),
        v3_panteon_equity_guard_floor_pnl_pct=(
            args.v3_panteon_equity_guard_floor_pnl_pct
        ),
        v3_panteon_equity_guard_cooldown_bars=(
            args.v3_panteon_equity_guard_cooldown_bars
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="Retrodate")
    parser.add_argument("--results-root", default=str(Path("Results") / "RetrodateMarket"))
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--risk-capital-fraction", type=float, default=0.10)
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=0.0,
        help="Per-fill slippage fraction for the retro FakeExchange (e.g. 0.0005 = 5bps). 0 = ideal fills.",
    )
    parser.add_argument(
        "--voting-directional",
        action="store_true",
        help="Phase 2/A1: directional WeightedConsensus for ensembles.",
    )
    parser.add_argument(
        "--flash-gate-pnl-per-trade",
        action="store_true",
        help="Phase 2/A3: gate on per-trade pnl instead of cumulative pnl_pct.",
    )
    parser.add_argument(
        "--perf-max-returns-history",
        type=int,
        default=0,
        help="Phase 3/C6: cap per-(label,regime) returns ring buffer (0 = unlimited).",
    )
    parser.add_argument(
        "--use-per-trade-pnl-score",
        action="store_true",
        help="Live no-trade fix: regime_score uses per-trade pnl instead of cumulative.",
    )
    parser.add_argument(
        "--per-trade-pnl-scale",
        type=float,
        default=1.0,
        help="Scale factor for the per-trade pnl score component.",
    )
    parser.add_argument(
        "--min-eligible-score",
        type=float,
        default=0.0,
        help="Selector min eligible score (recalibrate when using per-trade scoring).",
    )
    parser.add_argument("--scoring-pnl-weight", type=float, default=0.62)
    parser.add_argument("--scoring-sharpe-weight", type=float, default=0.34)
    parser.add_argument("--scoring-win-bonus-divisor", type=float, default=18.0)
    parser.add_argument(
        "--flash-global-health-gate",
        action="store_true",
        help="Selection fix: veto actors whose AGGREGATE cumulative pnl is catastrophic.",
    )
    parser.add_argument(
        "--flash-global-health-min-cum-pnl-pct",
        type=float,
        default=-5.0,
        help="Aggregate cumulative pnl%% below this vetoes the actor (with --flash-global-health-gate).",
    )
    parser.add_argument(
        "--flash-global-health-min-closed-trades",
        type=int,
        default=50,
        help="Min aggregate closed trades before the global-health veto applies.",
    )
    parser.add_argument("--risk-max-leverage", type=int, default=5)
    parser.add_argument("--apply-risk-leverage-to-notional", action="store_true")
    parser.add_argument("--include-optional-agents", action="store_true")
    parser.add_argument("--optional-agent-labels", default="")
    parser.add_argument("--invalid-policy", choices=("exclude", "fail"), default="exclude")
    parser.add_argument("--write-every-bars", type=int, default=2000)
    parser.add_argument("--full-snapshot-every", type=int, default=2000)
    parser.add_argument("--recompute-quarantine-every", type=int, default=24)
    parser.add_argument("--progress-every-bars", type=int, default=1000)
    parser.add_argument("--max-bars", type=int)
    parser.add_argument(
        "--disable-flash-audit-events",
        action="store_true",
        help=(
            "Do not retain per-candidate Flash audit events in memory. "
            "Useful for long retro runs with many actors; causal JSONL and "
            "execution attribution are still written."
        ),
    )
    parser.add_argument(
        "--disable-shadow-audit-events",
        action="store_true",
        help=(
            "Do not retain per-bar shadow actor audit events in memory. "
            "Shadow execution and performance memory still run, but shadow "
            "diagnostic reports that require those events will be empty."
        ),
    )
    parser.add_argument(
        "--disable-step-result-retention",
        action="store_true",
        help=(
            "Do not retain every StepResult in memory. OutputWriter still "
            "streams trading.log, status snapshots, and causal JSONL."
        ),
    )
    parser.add_argument(
        "--compact-causal-entry-selected-only",
        action="store_true",
        help=(
            "When compact causal JSONL is enabled, write only selected Flash "
            "decisions with real signals. This keeps attribution/context rows "
            "available for gate runs while avoiding huge NoTrade payloads."
        ),
    )
    parser.add_argument(
        "--shadow-agent-include-label",
        action="append",
        default=[],
        help=(
            "Limit shadow solo-agent execution to this label. Can be repeated. "
            "Empty default keeps all agents."
        ),
    )
    parser.add_argument(
        "--shadow-player-include-label",
        action="append",
        default=[],
        help=(
            "Limit shadow player execution to this label. Can be repeated. "
            "Empty default keeps all players."
        ),
    )
    parser.add_argument(
        "--shadow-parallel-workers",
        type=int,
        default=1,
        help=(
            "Experimental offline-only thread parallelism inside the shadow "
            "tournament. Default 1 preserves deterministic sequential behavior."
        ),
    )
    parser.add_argument(
        "--shadow-position-diagnostic-bar",
        action="append",
        type=int,
        default=[],
        help=(
            "Capture flash causal shadow-position diagnostics for a specific "
            "1-based replay bar. Can be repeated."
        ),
    )
    parser.add_argument(
        "--shadow-position-diagnostic-label",
        action="append",
        default=[],
        help=(
            "Limit shadow-position diagnostics to a player label. "
            "Can be repeated."
        ),
    )
    parser.add_argument("--solo-agent-candidate-limit", type=int, default=3)
    parser.add_argument("--enable-fixed-agent-players", action="store_true")
    parser.add_argument(
        "--fixed-agent-player-set",
        action="append",
        default=[],
        help="Add fixed player as Label=AgentA,AgentB. Can be repeated.",
    )
    parser.add_argument("--enable-flash", action="store_true")
    parser.add_argument("--flash-min-score-to-trade", type=float, default=0.0)
    parser.add_argument("--flash-actionable-bonus", type=float, default=0.25)
    parser.add_argument("--flash-no-data-score", type=float, default=0.0)
    parser.add_argument("--flash-min-closed-trades-to-trade", type=int, default=3)
    parser.add_argument("--flash-min-pnl-pct-to-trade", type=float, default=0.0)
    parser.add_argument("--enable-flash-shadow-confirmation", action="store_true")
    parser.add_argument("--enable-flash-symbol-shadow-confirmation", action="store_true")
    parser.add_argument("--enable-flash-shadow-actor-fallback-confirmation", action="store_true")
    parser.add_argument("--enable-flash-shadow-base-fallback-confirmation", action="store_true")
    parser.add_argument("--enable-flash-shadow-signal-handoff", action="store_true")
    parser.add_argument("--enable-flash-genetics-confirmation-overlay", action="store_true")
    parser.add_argument("--flash-genetics-confirmation-label", action="append", default=[])
    parser.add_argument(
        "--flash-genetics-confirmation-allowed-signal-key",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--flash-genetics-confirmation-contra-signal-key",
        action="append",
        default=[],
    )
    parser.add_argument("--flash-genetics-contra-validation-manifest-path")
    parser.add_argument(
        "--enable-flash-genetics-confirmation-contra-side-match",
        action="store_true",
    )
    parser.add_argument(
        "--enable-flash-genetics-confirmation-contra-static",
        action="store_true",
    )
    parser.add_argument(
        "--enable-flash-genetics-confirmation-contra-no-backfill",
        action="store_true",
    )
    parser.add_argument(
        "--enable-flash-genetics-confirmation-contra-risk-sizing",
        action="store_true",
    )
    parser.add_argument(
        "--flash-genetics-confirmation-contra-risk-mult",
        type=float,
        default=1.0,
    )
    parser.add_argument("--enable-flash-genetics-confirmation-quality-gate", action="store_true")
    parser.add_argument("--flash-genetics-confirmation-min-closed-trades", type=int, default=0)
    parser.add_argument("--flash-genetics-confirmation-min-pnl-per-trade-pct", type=float, default=0.0)
    parser.add_argument("--flash-genetics-confirmation-score-bonus", type=float, default=0.0)
    parser.add_argument("--flash-genetics-confirmation-score-penalty", type=float, default=0.0)
    parser.add_argument("--flash-genetics-confirmation-contra-score-penalty", type=float, default=0.0)
    parser.add_argument("--flash-shadow-actor-fallback-min-base-score", type=float, default=0.0)
    parser.add_argument(
        "--flash-shadow-position-replay-actor-fallback-min-base-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flash-shadow-position-replay-actor-fallback-min-shadow-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flash-shadow-base-fallback-actor-key",
        action="append",
        default=[],
        help=(
            "Allow one Flash actor key to use production metrics as shadow base "
            "fallback. Can be repeated."
        ),
    )
    parser.add_argument("--flash-actor-switch-margin", type=float, default=0.0)
    parser.add_argument("--flash-anchor-actor-key", action="append", default=[])
    parser.add_argument("--flash-portfolio-actor-key", action="append", default=[])
    parser.add_argument(
        "--enable-flash-portfolio-shadow-bootstrap-min-closed",
        action="store_true",
    )
    parser.add_argument("--flash-anchor-min-score-to-trade", type=float, default=None)
    parser.add_argument("--flash-anchor-shadow-min-score", type=float, default=None)
    parser.add_argument("--flash-anchor-min-score-advantage", type=float, default=0.0)
    parser.add_argument("--enable-flash-prefer-solo-player-wrappers", action="store_true")
    parser.add_argument("--enable-flash-prefer-proven-solo-player-wrappers", action="store_true")
    parser.add_argument("--flash-proven-solo-min-score-advantage", type=float, default=0.0)
    parser.add_argument("--flash-shadow-min-score", type=float, default=0.0)
    parser.add_argument("--flash-shadow-min-closed-trades", type=int, default=50)
    parser.add_argument("--flash-shadow-min-full-open-closed-trades", type=int, default=0)
    parser.add_argument("--enable-flash-shadow-quality-confirmation", action="store_true")
    parser.add_argument("--flash-shadow-min-win-rate-pct", type=float, default=0.0)
    parser.add_argument("--flash-shadow-max-recent-downside-usd", type=float, default=0.0)
    parser.add_argument("--flash-shadow-min-pnl-per-trade-lcb-usd", type=float, default=None)
    parser.add_argument("--flash-shadow-pnl-per-trade-lcb-z", type=float, default=1.0)
    parser.add_argument(
        "--flash-shadow-pnl-per-trade-lcb-penalty-floor-usd",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flash-shadow-pnl-per-trade-lcb-penalty-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--enable-flash-shadow-pnl-lcb-risk-sizing", action="store_true")
    parser.add_argument("--flash-shadow-pnl-lcb-risk-min-mult", type=float, default=0.25)
    parser.add_argument("--flash-shadow-pnl-lcb-risk-floor-usd", type=float, default=0.0)
    parser.add_argument("--flash-shadow-pnl-lcb-risk-scale-usd", type=float, default=1.0)
    parser.add_argument("--enable-flash-shadow-symbol-health", action="store_true")
    parser.add_argument("--flash-shadow-symbol-health-min-closed-trades", type=int, default=0)
    parser.add_argument("--flash-shadow-symbol-health-min-pnl-per-trade-lcb-usd", type=float, default=None)
    parser.add_argument("--flash-shadow-symbol-health-pnl-lcb-penalty-floor-usd", type=float, default=0.0)
    parser.add_argument("--flash-shadow-symbol-health-pnl-lcb-penalty-weight", type=float, default=0.0)
    parser.add_argument("--enable-flash-actor-risk-sizing", action="store_true")
    parser.add_argument("--flash-actor-risk-min-mult", type=float, default=0.25)
    parser.add_argument("--flash-actor-risk-max-mult", type=float, default=1.0)
    parser.add_argument("--flash-actor-risk-edge-scale-pct", type=float, default=0.50)
    parser.add_argument("--flash-funding-score-weight", type=float, default=0.0)
    parser.add_argument("--flash-funding-risk-mult-weight", type=float, default=0.0)
    parser.add_argument("--flash-funding-risk-mult-cap", type=float, default=0.25)
    parser.add_argument("--enable-flash-no-trade-fee-saving-score", action="store_true")
    parser.add_argument("--flash-no-trade-default-fee-bps", type=float, default=0.0)
    parser.add_argument("--enable-flash-volatility-risk-sizing", action="store_true")
    parser.add_argument("--flash-volatility-risk-target-pct", type=float, default=2.0)
    parser.add_argument("--flash-volatility-risk-min-volatility-pct", type=float, default=0.5)
    parser.add_argument("--flash-volatility-risk-max-mult", type=float, default=2.0)
    parser.add_argument(
        "--enable-flash-mixed-rotational-symbol-local-shadow-gate",
        action="store_true",
    )
    parser.add_argument(
        "--flash-mixed-rotational-symbol-local-shadow-min-score",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--flash-mixed-rotational-symbol-local-shadow-min-closed-trades",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--enable-flash-genetics-probation-bypass-min-closed",
        action="store_true",
    )
    parser.add_argument(
        "--enable-flash-genetics-probation-bypass-trend-gate",
        action="store_true",
    )
    parser.add_argument("--enable-flash-technical-overlay", action="store_true")
    parser.add_argument("--enable-flash-technical-hard-gate", action="store_true")
    parser.add_argument("--flash-technical-score-bonus", type=float, default=0.10)
    parser.add_argument("--flash-technical-score-penalty", type=float, default=0.25)
    parser.add_argument("--flash-technical-rsi-long-min", type=float, default=45.0)
    parser.add_argument("--flash-technical-rsi-long-max", type=float, default=72.0)
    parser.add_argument("--flash-technical-rsi-short-min", type=float, default=28.0)
    parser.add_argument("--flash-technical-rsi-short-max", type=float, default=55.0)
    parser.add_argument(
        "--flash-technical-macd-histogram-min-abs-pct",
        type=float,
        default=0.0,
    )
    parser.add_argument("--enable-flash-technical-atr-risk-sizing", action="store_true")
    parser.add_argument("--flash-technical-atr-target-pct", type=float, default=2.0)
    parser.add_argument("--flash-technical-atr-min-pct", type=float, default=0.25)
    parser.add_argument("--flash-technical-atr-max-mult", type=float, default=1.5)
    parser.add_argument(
        "--flash-selected-subset-score-boost",
        action="append",
        default=[],
        help=(
            "Add a positive Flash selected-subset score boost as "
            "actor_key|SYMBOL|ACTION=boost. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-selected-subset-context-score-boost",
        action="append",
        default=[],
        help=(
            "Add a regime-specific selected-subset score boost as "
            "actor_key|SYMBOL|ACTION|regime=boost. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-selected-subset-do-not-demote-signal-key",
        action="append",
        default=[],
        help=(
            "Protect one sparse positive selected-subset cell from LCB demotion. "
            "Format: actor_key|SYMBOL|ACTION. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-selected-subset-risk-mult",
        action="append",
        default=[],
        help=(
            "Override bounded risk multiplier for one selected-subset cell as "
            "actor_key|SYMBOL|ACTION=mult. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-selected-subset-context-risk-mult",
        action="append",
        default=[],
        help=(
            "Override bounded regime-specific selected-subset risk multiplier as "
            "actor_key|SYMBOL|ACTION|regime=mult. Can be repeated."
        ),
    )
    parser.add_argument("--flash-selected-subset-risk-min-mult", type=float, default=0.75)
    parser.add_argument("--flash-selected-subset-risk-max-mult", type=float, default=1.15)
    parser.add_argument("--enable-flash-partial-profit-lock", action="store_true")
    parser.add_argument("--flash-partial-profit-lock-trigger-pnl-pct", type=float, default=1.5)
    parser.add_argument("--flash-partial-profit-lock-close-fraction", type=float, default=0.5)
    parser.add_argument("--flash-partial-profit-lock-min-age-bars", type=int, default=2)
    parser.add_argument(
        "--disable-flash-partial-profit-lock-skip-protected-signal-keys",
        action="store_true",
        help=(
            "Allow partial profit-lock to cut selected-subset protected "
            "do-not-demote signal keys."
        ),
    )
    parser.add_argument(
        "--flash-partial-profit-lock-skip-signal-key",
        action="append",
        default=[],
        help=(
            "Signal key Actor|Symbol|Action to exclude from partial "
            "profit-lock. Can be repeated."
        ),
    )
    parser.add_argument("--flash-max-signals-per-actor", type=int, default=0)
    parser.add_argument("--enable-flash-overextension-guard", action="store_true")
    parser.add_argument("--flash-overextension-lookback-bars", type=int, default=12)
    parser.add_argument("--flash-short-overextension-return-floor-pct", type=float, default=-8.0)
    parser.add_argument("--flash-long-overextension-return-ceiling-pct", type=float, default=8.0)
    parser.add_argument(
        "--enable-flash-overextension-volatility-normalized",
        action="store_true",
    )
    parser.add_argument("--flash-short-overextension-z-floor", type=float, default=-2.5)
    parser.add_argument("--flash-long-overextension-z-ceiling", type=float, default=2.5)
    parser.add_argument("--flash-overextension-min-volatility-pct", type=float, default=0.1)
    parser.add_argument("--enable-flash-degradation-guard", action="store_true")
    parser.add_argument("--enable-flash-degradation-actor-guard", action="store_true")
    parser.add_argument(
        "--flash-degradation-actor-scope",
        choices=("actor", "actor_regime"),
        default="actor",
    )
    parser.add_argument("--flash-degradation-signal-cooldown-bars", type=int, default=0)
    parser.add_argument("--flash-degradation-actor-cooldown-bars", type=int, default=0)
    parser.add_argument("--enable-flash-degradation-symbol-guard", action="store_true")
    parser.add_argument("--flash-degradation-symbol-cooldown-bars", type=int, default=0)
    parser.add_argument("--flash-degradation-symbol-lookback-bars", type=int, default=0)
    parser.add_argument("--flash-degradation-symbol-window-closed-trades", type=int, default=0)
    parser.add_argument("--flash-degradation-symbol-min-closed-trades", type=int, default=0)
    parser.add_argument("--flash-degradation-symbol-max-recent-pnl-usd", type=float, default=None)
    parser.add_argument("--flash-degradation-window-closed-trades", type=int, default=3)
    parser.add_argument("--flash-degradation-min-closed-trades", type=int, default=3)
    parser.add_argument("--flash-degradation-max-recent-pnl-usd", type=float, default=-25.0)
    parser.add_argument("--flash-degradation-signal-min-pnl-per-trade-lcb-usd", type=float, default=None)
    parser.add_argument("--flash-degradation-pnl-per-trade-lcb-z", type=float, default=1.0)
    parser.add_argument("--enable-flash-degradation-signal-risk-sizing", action="store_true")
    parser.add_argument("--flash-degradation-signal-risk-mult", type=float, default=0.20)
    parser.add_argument("--flash-degradation-reserve-actor-cap", action="store_true")
    parser.add_argument("--enable-flash-degradation-recovery", action="store_true")
    parser.add_argument("--flash-degradation-recovery-min-closed-trades", type=int, default=3)
    parser.add_argument("--flash-degradation-recovery-min-recent-pnl-usd", type=float, default=0.0)
    parser.add_argument("--enable-flash-stale-position-exit", action="store_true")
    parser.add_argument("--flash-stale-position-exit-max-age-bars", type=int, default=168)
    parser.add_argument(
        "--flash-stale-position-exit-allow-profitable",
        action="store_true",
        help="Close stale positions by age even when current unrealized PnL is positive.",
    )
    parser.add_argument("--enable-flash-promotion-manifest", action="store_true")
    parser.add_argument("--flash-promotion-manifest-path", default="")
    parser.add_argument("--flash-promotion-min-full-closed-trades", type=int, default=50)
    parser.add_argument("--flash-promotion-min-latest-closed-trades", type=int, default=10)
    parser.add_argument("--flash-promotion-min-full-pnl-pct", type=float, default=0.0)
    parser.add_argument("--flash-promotion-min-latest-pnl-pct", type=float, default=0.0)
    parser.add_argument(
        "--flash-promotion-min-full-pnl-per-trade-lcb-pct",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flash-promotion-min-latest-pnl-per-trade-lcb-pct",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--flash-promotion-pnl-per-trade-lcb-z",
        type=float,
        default=Z_95_ONE_SIDED,
    )
    parser.add_argument("--flash-promotion-max-drawdown-pct", type=float, default=25.0)
    parser.add_argument("--flash-promotion-min-win-rate-pct", type=float, default=52.0)
    parser.add_argument("--flash-promotion-max-recent-downside-usd", type=float, default=5.0)
    parser.add_argument(
        "--flash-promoted-signal-key",
        action="append",
        default=[],
        help=(
            "Allow one exact promoted Flash actor/symbol/action key. Format: "
            "actor_key|SYMBOL|ACTION. Can be repeated."
        ),
    )
    parser.add_argument("--enable-flash-earned-cap-overrides", action="store_true")
    parser.add_argument(
        "--flash-promoted-actor-cap-override",
        action="append",
        default=[],
        help="Override promoted cap as actor_key=cap. Can be repeated.",
    )
    parser.add_argument("--enable-experimental-flash-actors", action="store_true")
    parser.add_argument("--allow-experimental-flash-real-actors", action="store_true")
    parser.add_argument(
        "--flash-deny-signal-key",
        action="append",
        default=[],
        help=(
            "Reject one exact Flash actor/symbol/action key. Format: "
            "actor_key|SYMBOL|ACTION. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-terminal-deny-signal-key",
        action="append",
        default=[],
        help=(
            "Reject one exact Flash actor/symbol/action key and keep NoTrade "
            "when that rejected key would have been the top candidate. Format: "
            "actor_key|SYMBOL|ACTION. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-terminal-deny-context-signal-key",
        action="append",
        default=[],
        help=(
            "Reject one exact Flash actor/symbol/action/regime key and keep "
            "NoTrade when that rejected key would have been the top candidate. "
            "Format: actor_key|SYMBOL|ACTION|REGIME. Can be repeated."
        ),
    )
    parser.add_argument(
        "--flash-denied-open-regime",
        action="append",
        default=[],
        help="Reject Flash open signals in this market regime. Can be repeated.",
    )
    parser.add_argument(
        "--flash-denied-open-symbol",
        action="append",
        default=[],
        help=(
            "Reject Flash open signals for this symbol while allowing closes. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--hard-policy-deny-label",
        action="append",
        default=[],
        help=(
            "Add a candidate label to the retro hard-policy denylist. "
            "Can be repeated; defaults are preserved."
        ),
    )
    parser.add_argument("--enable-actionable-fallback", action="store_true")
    parser.add_argument("--actionable-fallback-min-score", type=float)
    parser.add_argument("--actionable-fallback-require-has-data", action="store_true")
    parser.add_argument("--enable-current-actionable-candidate-layer", action="store_true")
    parser.add_argument("--real-promotion-gate", action="store_true")
    parser.add_argument("--real-promotion-min-closed-trades", type=int, default=20)
    parser.add_argument("--real-promotion-min-pnl-pct", type=float, default=0.0)
    parser.add_argument("--real-promotion-max-dd-pct", type=float, default=25.0)
    parser.add_argument("--real-promotion-loss-budget-pct", type=float, default=-1.0)
    parser.add_argument("--real-promotion-probation-min-score", type=float, default=0.0)
    parser.add_argument("--use-v3-rolling-score", action="store_true")
    parser.add_argument("--use-v3-shadow-rolling-score", action="store_true")
    parser.add_argument("--use-v3-soft-shadow-score", action="store_true")
    parser.add_argument("--use-v3-executable-soft-top1-score", action="store_true")
    parser.add_argument("--use-v3-executable-soft-confirmed-score", action="store_true")
    parser.add_argument("--use-v3-entry-causal-score", action="store_true")
    parser.add_argument("--disable-v3-shadow-position-gate", action="store_true")
    parser.add_argument("--enable-v3-shadow-flat-handoff", action="store_true")
    parser.add_argument("--enable-v3-shadow-fresh-handoff", action="store_true")
    parser.add_argument("--v3-shadow-fresh-handoff-max-age-bars", type=int, default=1)
    parser.add_argument(
        "--v3-shadow-fresh-handoff-allow-nonpositive-unrealized",
        action="store_true",
    )
    parser.add_argument("--v3-shadow-rolling-window-bars", type=int, default=24)
    parser.add_argument("--v3-shadow-rolling-min-closed-trades", type=int, default=50)
    parser.add_argument("--v3-entry-causal-min-filled", type=int, default=3)
    parser.add_argument("--v3-entry-causal-actionability-weight", type=float, default=1.0)
    parser.add_argument("--enable-v3-real-loss-rescue", action="store_true")
    parser.add_argument("--v3-real-loss-rescue-min-virtual-pnl-pct", type=float, default=10.0)
    parser.add_argument("--v3-real-loss-rescue-max-virtual-dd-pct", type=float, default=50.0)
    parser.add_argument("--v3-real-loss-rescue-min-actionable-share", type=float, default=0.05)
    parser.add_argument("--v3-real-loss-rescue-min-recent-filled", type=int, default=1)
    parser.add_argument("--v3-real-loss-rescue-max-real-loss-pct", type=float, default=-3.0)
    parser.add_argument("--allow-genetics-real-loss-rescue", action="store_true")
    parser.add_argument("--enable-v3-probation-shadow-rescue", action="store_true")
    parser.add_argument("--v3-probation-shadow-rescue-min-virtual-pnl-pct", type=float, default=10.0)
    parser.add_argument("--v3-probation-shadow-rescue-max-virtual-dd-pct", type=float, default=50.0)
    parser.add_argument("--v3-probation-shadow-rescue-min-actionable-share", type=float, default=0.05)
    parser.add_argument("--v3-probation-shadow-rescue-min-recent-filled", type=int, default=1)
    parser.add_argument("--v3-probation-shadow-rescue-min-recent-pnl-usd", type=float, default=0.0)
    parser.add_argument("--allow-genetics-probation-shadow-rescue", action="store_true")
    parser.add_argument("--v3-persistent-loss-kill-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-persistent-loss-kill-pnl-pct", type=float, default=-2.0)
    parser.add_argument("--v3-persistent-loss-kill-win-rate-pct", type=float, default=0.0)
    parser.add_argument("--v3-persistent-loss-ignore-virtual-quality", action="store_true")
    parser.add_argument("--v3-persistent-loss-virtual-max-pnl-pct", type=float, default=0.0)
    parser.add_argument("--v3-persistent-loss-virtual-min-dd-pct", type=float, default=25.0)
    parser.add_argument("--v3-probation-loss-kill-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-probation-loss-kill-pnl-pct", type=float, default=-0.15)
    parser.add_argument("--v3-probation-loss-kill-win-rate-pct", type=float, default=50.0)
    parser.add_argument(
        "--v3-probation-loss-kill-label-prefix",
        action="append",
        default=None,
        help=(
            "Restrict probation loss-kill to labels with this prefix. "
            "Can be repeated. Defaults to Solo_, Fixed_, Antonius_, and Optimal_."
        ),
    )
    parser.add_argument(
        "--v3-probation-loss-kill-all-labels",
        action="store_true",
        help="Apply probation loss-kill to every candidate label.",
    )
    parser.add_argument("--max-new-opens-per-bar", type=int, default=1)
    parser.add_argument("--risk-max-open-positions", type=int, default=8)
    parser.add_argument("--enable-genetics-probation-execution", action="store_true")
    parser.add_argument(
        "--genetics-probation-label",
        action="append",
        default=[],
        help="Allow this label through genetics probation execution. Repeatable.",
    )
    parser.add_argument(
        "--genetics-probation-allowed-regime",
        action="append",
        default=[],
        help="Allow genetics probation opens in this regime. Repeatable.",
    )
    parser.add_argument(
        "--genetics-probation-allowed-signal-key",
        action="append",
        default=[],
        help=(
            "Allow only this exact genetics probation signal key. "
            "When omitted, all otherwise eligible genetics probation signals are allowed."
        ),
    )
    parser.add_argument("--genetics-probation-risk-mult", type=float, default=0.25)
    parser.add_argument(
        "--genetics-probation-min-regime-confidence",
        type=float,
        default=0.0,
    )
    parser.add_argument("--genetics-probation-max-real-trades", type=int, default=20)
    parser.add_argument("--genetics-probation-max-daily-trades", type=int, default=0)
    parser.add_argument(
        "--disable-genetics-probation-shadow-confirmation",
        action="store_true",
    )
    parser.add_argument("--v3-realized-profit-lock-min-closed-trades", type=int, default=0)
    parser.add_argument("--v3-realized-profit-lock-min-peak-pnl-pct", type=float, default=0.75)
    parser.add_argument("--v3-realized-profit-lock-max-giveback-pct", type=float, default=0.55)
    parser.add_argument("--v3-realized-profit-lock-floor-pnl-pct", type=float, default=0.25)
    parser.add_argument("--enable-v3-panteon-equity-guard", action="store_true")
    parser.add_argument("--v3-panteon-equity-guard-min-peak-pnl-pct", type=float, default=2.0)
    parser.add_argument("--v3-panteon-equity-guard-max-giveback-pct", type=float, default=1.0)
    parser.add_argument("--v3-panteon-equity-guard-floor-pnl-pct", type=float, default=2.0)
    parser.add_argument("--v3-panteon-equity-guard-cooldown-bars", type=int, default=720)
    parser.add_argument(
        "--disable-hard-policy",
        action="store_true",
        help="Disable non-Flash strategist hard-policy vetoes for legacy-like runs.",
    )
    return parser


def _build_scoring_config(config: RetrodateMarketConfig) -> ScoringConfig:
    # Если ничего не переопределено — отдаём дефолтный синглтон.
    if (
        not config.use_per_trade_pnl_score
        and config.min_eligible_score == DEFAULT_SCORING.min_eligible_score
        and config.scoring_pnl_weight == DEFAULT_SCORING.pnl_weight
        and config.scoring_sharpe_weight == DEFAULT_SCORING.sharpe_weight
        and config.scoring_win_bonus_divisor == DEFAULT_SCORING.win_bonus_divisor
    ):
        return DEFAULT_SCORING
    return ScoringConfig(
        pnl_weight=config.scoring_pnl_weight,
        sharpe_weight=config.scoring_sharpe_weight,
        win_bonus_divisor=config.scoring_win_bonus_divisor,
        use_per_trade_pnl=config.use_per_trade_pnl_score,
        per_trade_pnl_scale=config.per_trade_pnl_scale,
        min_eligible_score=config.min_eligible_score,
    )


def _build_flash_allocator_config(config: RetrodateMarketConfig) -> FlashAllocatorConfig:
    return FlashAllocatorConfig(
        min_score_to_trade=config.flash_min_score_to_trade,
        actionable_bonus=config.flash_actionable_bonus,
        no_data_score=config.flash_no_data_score,
        min_closed_trades_to_trade=config.flash_min_closed_trades_to_trade,
        min_pnl_pct_to_trade=config.flash_min_pnl_pct_to_trade,
        gate_pnl_per_trade_enabled=config.flash_gate_pnl_per_trade_enabled,
        global_health_gate_enabled=config.flash_global_health_gate_enabled,
        global_health_min_cum_pnl_pct=config.flash_global_health_min_cum_pnl_pct,
        global_health_min_closed_trades=config.flash_global_health_min_closed_trades,
        shadow_confirmation_enabled=config.flash_shadow_confirmation_enabled,
        shadow_symbol_confirmation_enabled=(
            config.flash_symbol_shadow_confirmation_enabled
        ),
        shadow_actor_fallback_confirmation_enabled=(
            config.flash_shadow_actor_fallback_confirmation_enabled
        ),
        shadow_base_fallback_confirmation_enabled=(
            config.flash_shadow_base_fallback_confirmation_enabled
        ),
        shadow_signal_handoff_enabled=config.flash_shadow_signal_handoff_enabled,
        genetics_confirmation_overlay_enabled=(
            config.flash_genetics_confirmation_overlay_enabled
        ),
        genetics_confirmation_labels=config.flash_genetics_confirmation_labels,
        genetics_confirmation_allowed_signal_keys=(
            config.flash_genetics_confirmation_allowed_signal_keys
        ),
        genetics_confirmation_contra_signal_keys=(
            config.flash_genetics_confirmation_contra_signal_keys
        ),
        genetics_confirmation_contra_side_match_enabled=(
            config.flash_genetics_confirmation_contra_side_match_enabled
        ),
        genetics_confirmation_contra_static_enabled=(
            config.flash_genetics_confirmation_contra_static_enabled
        ),
        genetics_confirmation_contra_no_backfill_enabled=(
            config.flash_genetics_confirmation_contra_no_backfill_enabled
        ),
        genetics_confirmation_contra_risk_sizing_enabled=(
            config.flash_genetics_confirmation_contra_risk_sizing_enabled
        ),
        genetics_confirmation_contra_risk_mult=(
            config.flash_genetics_confirmation_contra_risk_mult
        ),
        genetics_confirmation_quality_gate_enabled=(
            config.flash_genetics_confirmation_quality_gate_enabled
        ),
        genetics_confirmation_min_closed_trades=(
            config.flash_genetics_confirmation_min_closed_trades
        ),
        genetics_confirmation_min_pnl_per_trade_pct=(
            config.flash_genetics_confirmation_min_pnl_per_trade_pct
        ),
        genetics_confirmation_score_bonus=(
            config.flash_genetics_confirmation_score_bonus
        ),
        genetics_confirmation_score_penalty=(
            config.flash_genetics_confirmation_score_penalty
        ),
        genetics_confirmation_contra_score_penalty=(
            config.flash_genetics_confirmation_contra_score_penalty
        ),
        shadow_actor_fallback_min_base_score=(
            config.flash_shadow_actor_fallback_min_base_score
        ),
        shadow_position_replay_actor_fallback_min_base_score=(
            config.flash_shadow_position_replay_actor_fallback_min_base_score
        ),
        shadow_position_replay_actor_fallback_min_shadow_score=(
            config.flash_shadow_position_replay_actor_fallback_min_shadow_score
        ),
        shadow_base_fallback_actor_keys=config.flash_shadow_base_fallback_actor_keys,
        actor_switch_margin=config.flash_actor_switch_margin,
        anchor_actor_keys=config.flash_anchor_actor_keys,
        portfolio_actor_keys=config.flash_portfolio_actor_keys,
        portfolio_shadow_bootstrap_min_closed_enabled=(
            config.flash_portfolio_shadow_bootstrap_min_closed_enabled
        ),
        anchor_min_score_to_trade=config.flash_anchor_min_score_to_trade,
        anchor_shadow_min_score=config.flash_anchor_shadow_min_score,
        anchor_min_score_advantage=config.flash_anchor_min_score_advantage,
        prefer_solo_player_wrappers_enabled=(
            config.flash_prefer_solo_player_wrappers_enabled
        ),
        prefer_proven_solo_player_wrappers_enabled=(
            config.flash_prefer_proven_solo_player_wrappers_enabled
        ),
        proven_solo_min_score_advantage=(
            config.flash_proven_solo_min_score_advantage
        ),
        shadow_confirmation_min_score=config.flash_shadow_min_score,
        shadow_confirmation_min_closed_trades=config.flash_shadow_min_closed_trades,
        shadow_confirmation_min_full_open_closed_trades=(
            config.flash_shadow_min_full_open_closed_trades
        ),
        shadow_quality_confirmation_enabled=(
            config.flash_shadow_quality_confirmation_enabled
        ),
        shadow_confirmation_min_win_rate_pct=config.flash_shadow_min_win_rate_pct,
        shadow_confirmation_max_recent_downside_usd=(
            config.flash_shadow_max_recent_downside_usd
        ),
        shadow_confirmation_min_pnl_per_trade_lcb_usd=(
            config.flash_shadow_min_pnl_per_trade_lcb_usd
        ),
        shadow_confirmation_pnl_per_trade_lcb_z=(
            config.flash_shadow_pnl_per_trade_lcb_z
        ),
        shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd=(
            config.flash_shadow_pnl_per_trade_lcb_penalty_floor_usd
        ),
        shadow_confirmation_pnl_per_trade_lcb_penalty_weight=(
            config.flash_shadow_pnl_per_trade_lcb_penalty_weight
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled=(
            config.flash_shadow_pnl_lcb_risk_sizing_enabled
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_min_mult=(
            config.flash_shadow_pnl_lcb_risk_min_mult
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd=(
            config.flash_shadow_pnl_lcb_risk_floor_usd
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd=(
            config.flash_shadow_pnl_lcb_risk_scale_usd
        ),
        shadow_symbol_health_enabled=config.flash_shadow_symbol_health_enabled,
        shadow_symbol_health_min_closed_trades=(
            config.flash_shadow_symbol_health_min_closed_trades
        ),
        shadow_symbol_health_min_pnl_per_trade_lcb_usd=(
            config.flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd
        ),
        shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd=(
            config.flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd
        ),
        shadow_symbol_health_pnl_per_trade_lcb_penalty_weight=(
            config.flash_shadow_symbol_health_pnl_lcb_penalty_weight
        ),
        actor_risk_sizing_enabled=config.flash_actor_risk_sizing_enabled,
        actor_risk_min_mult=config.flash_actor_risk_min_mult,
        actor_risk_max_mult=config.flash_actor_risk_max_mult,
        actor_risk_edge_scale_pct=config.flash_actor_risk_edge_scale_pct,
        funding_score_weight=config.flash_funding_score_weight,
        funding_risk_mult_weight=config.flash_funding_risk_mult_weight,
        funding_risk_mult_cap=config.flash_funding_risk_mult_cap,
        no_trade_fee_saving_score_enabled=(
            config.flash_no_trade_fee_saving_score_enabled
        ),
        no_trade_default_fee_bps=config.flash_no_trade_default_fee_bps,
        volatility_risk_sizing_enabled=config.flash_volatility_risk_sizing_enabled,
        volatility_risk_target_pct=config.flash_volatility_risk_target_pct,
        volatility_risk_min_volatility_pct=(
            config.flash_volatility_risk_min_volatility_pct
        ),
        volatility_risk_max_mult=config.flash_volatility_risk_max_mult,
        mixed_rotational_symbol_local_shadow_gate_enabled=(
            config.flash_mixed_rotational_symbol_local_shadow_gate_enabled
        ),
        mixed_rotational_symbol_local_shadow_min_score=(
            config.flash_mixed_rotational_symbol_local_shadow_min_score
        ),
        mixed_rotational_symbol_local_shadow_min_closed_trades=(
            config.flash_mixed_rotational_symbol_local_shadow_min_closed_trades
        ),
        genetics_probation_bypass_min_closed_enabled=(
            config.flash_genetics_probation_bypass_min_closed_enabled
        ),
        genetics_probation_bypass_trend_gate_enabled=(
            config.flash_genetics_probation_bypass_trend_gate_enabled
        ),
        technical_overlay_enabled=config.flash_technical_overlay_enabled,
        technical_hard_gate_enabled=config.flash_technical_hard_gate_enabled,
        technical_score_bonus=config.flash_technical_score_bonus,
        technical_score_penalty=config.flash_technical_score_penalty,
        technical_rsi_long_min=config.flash_technical_rsi_long_min,
        technical_rsi_long_max=config.flash_technical_rsi_long_max,
        technical_rsi_short_min=config.flash_technical_rsi_short_min,
        technical_rsi_short_max=config.flash_technical_rsi_short_max,
        technical_macd_histogram_min_abs_pct=(
            config.flash_technical_macd_histogram_min_abs_pct
        ),
        technical_atr_risk_sizing_enabled=(
            config.flash_technical_atr_risk_sizing_enabled
        ),
        technical_atr_target_pct=config.flash_technical_atr_target_pct,
        technical_atr_min_pct=config.flash_technical_atr_min_pct,
        technical_atr_max_mult=config.flash_technical_atr_max_mult,
        selected_subset_score_boosts=config.flash_selected_subset_score_boosts,
        selected_subset_context_score_boosts=(
            config.flash_selected_subset_context_score_boosts
        ),
        selected_subset_do_not_demote_signal_keys=(
            config.flash_selected_subset_do_not_demote_signal_keys
        ),
        selected_subset_risk_mult_overrides=(
            config.flash_selected_subset_risk_mult_overrides
        ),
        selected_subset_context_risk_mult_overrides=(
            config.flash_selected_subset_context_risk_mult_overrides
        ),
        selected_subset_risk_min_mult=config.flash_selected_subset_risk_min_mult,
        selected_subset_risk_max_mult=config.flash_selected_subset_risk_max_mult,
        max_signals_per_actor=config.flash_max_signals_per_actor,
        open_overextension_guard_enabled=config.flash_overextension_guard_enabled,
        overextension_lookback_bars=config.flash_overextension_lookback_bars,
        short_overextension_return_floor_pct=(
            config.flash_short_overextension_return_floor_pct
        ),
        long_overextension_return_ceiling_pct=(
            config.flash_long_overextension_return_ceiling_pct
        ),
        overextension_volatility_normalized_enabled=(
            config.flash_overextension_volatility_normalized_enabled
        ),
        short_overextension_z_floor=config.flash_short_overextension_z_floor,
        long_overextension_z_ceiling=config.flash_long_overextension_z_ceiling,
        overextension_min_volatility_pct=(
            config.flash_overextension_min_volatility_pct
        ),
        denied_signal_keys=config.flash_denied_signal_keys,
        terminal_denied_signal_keys=config.flash_terminal_denied_signal_keys,
        terminal_denied_context_signal_keys=(
            config.flash_terminal_denied_context_signal_keys
        ),
        denied_open_symbols=config.flash_denied_open_symbols,
        denied_open_regimes=config.flash_denied_open_regimes,
        degradation_guard_enabled=config.flash_degradation_guard_enabled,
        degradation_actor_guard_enabled=config.flash_degradation_actor_guard_enabled,
        degradation_actor_scope=config.flash_degradation_actor_scope,
        degradation_signal_cooldown_bars=config.flash_degradation_signal_cooldown_bars,
        degradation_actor_cooldown_bars=config.flash_degradation_actor_cooldown_bars,
        degradation_symbol_guard_enabled=(
            config.flash_degradation_symbol_guard_enabled
        ),
        degradation_symbol_cooldown_bars=(
            config.flash_degradation_symbol_cooldown_bars
        ),
        degradation_symbol_lookback_bars=(
            config.flash_degradation_symbol_lookback_bars
        ),
        degradation_symbol_window_closed_trades=(
            config.flash_degradation_symbol_window_closed_trades
        ),
        degradation_symbol_min_closed_trades=(
            config.flash_degradation_symbol_min_closed_trades
        ),
        degradation_symbol_max_recent_pnl_usd=(
            config.flash_degradation_symbol_max_recent_pnl_usd
        ),
        degradation_window_closed_trades=(
            config.flash_degradation_window_closed_trades
        ),
        degradation_min_closed_trades=config.flash_degradation_min_closed_trades,
        degradation_max_recent_pnl_usd=(
            config.flash_degradation_max_recent_pnl_usd
        ),
        degradation_signal_min_pnl_per_trade_lcb_usd=(
            config.flash_degradation_signal_min_pnl_per_trade_lcb_usd
        ),
        degradation_pnl_per_trade_lcb_z=(
            config.flash_degradation_pnl_per_trade_lcb_z
        ),
        degradation_signal_risk_sizing_enabled=(
            config.flash_degradation_signal_risk_sizing_enabled
        ),
        degradation_signal_risk_mult=config.flash_degradation_signal_risk_mult,
        degradation_reserve_actor_cap=config.flash_degradation_reserve_actor_cap,
        degradation_recovery_enabled=config.flash_degradation_recovery_enabled,
        degradation_recovery_min_closed_trades=(
            config.flash_degradation_recovery_min_closed_trades
        ),
        degradation_recovery_min_recent_pnl_usd=(
            config.flash_degradation_recovery_min_recent_pnl_usd
        ),
        promotion_manifest_enabled=config.flash_promotion_manifest_enabled,
        promoted_actor_cap_overrides=_flash_promoted_actor_cap_overrides(config),
    )


def _load_flash_promoted_signal_keys(
    manifest_path: Optional[Path],
    *,
    explicit_keys: Sequence[str],
) -> tuple[str, ...]:
    keys: set[str] = {
        str(key).strip()
        for key in explicit_keys
        if str(key or "").strip()
    }
    if manifest_path is not None and manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        allowed = data.get("allowed_signal_keys") if isinstance(data, dict) else None
        if isinstance(allowed, list):
            keys.update(str(key).strip() for key in allowed if str(key or "").strip())
    return tuple(sorted(keys))


def _load_flash_validated_contra_signal_keys(
    manifest_path: Optional[Path],
    *,
    explicit_keys: Sequence[str],
) -> tuple[str, ...]:
    explicit = tuple(
        dict.fromkeys(
            str(key).strip()
            for key in explicit_keys
            if str(key or "").strip()
        )
    )
    if manifest_path is None:
        return explicit
    if not manifest_path.exists():
        return ()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(data, dict):
        return ()
    summary = data.get("summary")
    eligible = bool(summary.get("eligible")) if isinstance(summary, dict) else False
    if not eligible:
        return ()
    raw_allowed = data.get("allowed_contra_signal_keys")
    if not isinstance(raw_allowed, list):
        return ()
    allowed = tuple(
        dict.fromkeys(
            str(key).strip()
            for key in raw_allowed
            if str(key or "").strip()
        )
    )
    if not explicit:
        return allowed
    explicit_set = set(explicit)
    return tuple(key for key in allowed if key in explicit_set)


def _flash_promoted_actor_cap_overrides(
    config: RetrodateMarketConfig,
) -> tuple[str, ...]:
    explicit = tuple(config.flash_promoted_actor_cap_overrides)
    if not config.flash_earned_cap_overrides_enabled:
        return explicit

    explicit_actors = {
        item.split("=", 1)[0].strip()
        for item in explicit
        if item.split("=", 1)[0].strip()
    }
    keys_by_actor: dict[str, set[str]] = {}
    for signal_key in config.flash_promoted_signal_keys:
        actor_key = str(signal_key).split("|", 1)[0].strip()
        if not actor_key:
            continue
        keys_by_actor.setdefault(actor_key, set()).add(str(signal_key).strip())

    auto = tuple(
        f"{actor_key}=2"
        for actor_key in sorted(keys_by_actor)
        if actor_key not in explicit_actors and len(keys_by_actor[actor_key]) >= 2
    )
    return explicit + auto


def _build_live_execution_config(config: RetrodateMarketConfig) -> LiveExecutionConfig:
    return LiveExecutionConfig(
        max_new_opens_per_bar=config.max_new_opens_per_bar,
        genetics_probation_execution_enabled=(
            config.genetics_probation_execution_enabled
        ),
        genetics_probation_labels=config.genetics_probation_labels,
        genetics_probation_allowed_regimes=(
            config.genetics_probation_allowed_regimes
        ),
        genetics_probation_allowed_signal_keys=(
            config.genetics_probation_allowed_signal_keys
        ),
        genetics_probation_risk_mult=config.genetics_probation_risk_mult,
        genetics_probation_min_regime_confidence=(
            config.genetics_probation_min_regime_confidence
        ),
        genetics_probation_max_real_trades=config.genetics_probation_max_real_trades,
        genetics_probation_max_daily_trades=(
            config.genetics_probation_max_daily_trades
        ),
        genetics_probation_require_shadow_confirmation=(
            config.genetics_probation_require_shadow_confirmation
        ),
    )


def _build_risk_config(config: RetrodateMarketConfig) -> RiskLimitsConfig:
    return RiskLimitsConfig(
        max_open_positions=config.risk_max_open_positions,
        max_leverage=config.risk_max_leverage,
        apply_leverage_to_notional=config.apply_risk_leverage_to_notional,
        capital_fraction=config.risk_capital_fraction,
    )


def _build_strategist_config(config: RetrodateMarketConfig) -> StrategistConfig:
    executable_soft_top1 = bool(config.use_v3_executable_soft_top1_score)
    executable_soft_confirmed = bool(config.use_v3_executable_soft_confirmed_score)
    executable_soft = executable_soft_top1 or executable_soft_confirmed
    rolling_window = (
        24 if executable_soft else config.v3_shadow_rolling_window_bars
    )
    min_closed_trades = (
        20 if executable_soft_top1
        else max(50, int(config.v3_shadow_rolling_min_closed_trades))
        if executable_soft_confirmed
        else config.v3_shadow_rolling_min_closed_trades
    )
    hard_policy_deny_labels = tuple(config.hard_policy_deny_labels)
    if (
        config.experimental_flash_actors_enabled
        and not config.experimental_flash_real_actors_enabled
    ):
        hard_policy_deny_labels = tuple(
            dict.fromkeys(
                hard_policy_deny_labels + tuple(experimental_flash_agent_labels())
            )
        )
    return StrategistConfig(
        hard_policy_deny_labels=hard_policy_deny_labels,
        hard_policy_enabled=config.hard_policy_enabled,
        real_promotion_gate_enabled=config.real_promotion_gate_enabled,
        real_promotion_min_closed_trades=config.real_promotion_min_closed_trades,
        real_promotion_min_pnl_pct=config.real_promotion_min_pnl_pct,
        real_promotion_max_drawdown_pct=config.real_promotion_max_drawdown_pct,
        real_promotion_loss_budget_pct=config.real_promotion_loss_budget_pct,
        real_promotion_probation_min_score=config.real_promotion_probation_min_score,
        use_v3_rolling_score=config.use_v3_rolling_score or executable_soft,
        use_v3_entry_causal_score=config.use_v3_entry_causal_score,
        use_v3_shadow_rolling_score=config.use_v3_shadow_rolling_score,
        use_v3_soft_shadow_score=config.use_v3_soft_shadow_score or executable_soft,
        v3_current_actionable_gate_enabled=(
            config.current_actionable_candidate_layer_enabled
            or executable_soft_confirmed
        ),
        v3_solo_current_actionable_gate_enabled=executable_soft_top1,
        v3_candidate_allow_labels=config.v3_candidate_allow_labels,
        v3_shadow_position_gate_enabled=config.v3_shadow_position_gate_enabled,
        v3_shadow_flat_handoff_enabled=config.v3_shadow_flat_handoff_enabled,
        v3_shadow_fresh_handoff_enabled=config.v3_shadow_fresh_handoff_enabled,
        v3_shadow_fresh_handoff_max_age_bars=config.v3_shadow_fresh_handoff_max_age_bars,
        v3_shadow_fresh_handoff_require_positive_unrealized=(
            config.v3_shadow_fresh_handoff_require_positive_unrealized
        ),
        v3_shadow_rolling_window_bars=rolling_window,
        v3_shadow_rolling_min_closed_trades=min_closed_trades,
        v3_entry_causal_min_filled=config.v3_entry_causal_min_filled,
        v3_entry_causal_actionability_weight=config.v3_entry_causal_actionability_weight,
        v3_min_score_to_trade=(1e-9 if executable_soft_top1 else 0.0),
        v3_real_loss_kill_min_closed_trades=(0 if executable_soft_top1 else 3),
        v3_real_loss_rescue_enabled=config.v3_real_loss_rescue_enabled,
        v3_real_loss_rescue_min_virtual_pnl_pct=(
            config.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        v3_real_loss_rescue_max_virtual_dd_pct=(
            config.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        v3_real_loss_rescue_min_actionable_share=(
            config.v3_real_loss_rescue_min_actionable_share
        ),
        v3_real_loss_rescue_min_recent_filled=(
            config.v3_real_loss_rescue_min_recent_filled
        ),
        v3_real_loss_rescue_max_real_loss_pct=(
            config.v3_real_loss_rescue_max_real_loss_pct
        ),
        v3_real_loss_rescue_allow_genetics=(
            config.v3_real_loss_rescue_allow_genetics
        ),
        v3_probation_shadow_rescue_enabled=(
            config.v3_probation_shadow_rescue_enabled
        ),
        v3_probation_shadow_rescue_min_virtual_pnl_pct=(
            config.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        v3_probation_shadow_rescue_max_virtual_dd_pct=(
            config.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        v3_probation_shadow_rescue_min_actionable_share=(
            config.v3_probation_shadow_rescue_min_actionable_share
        ),
        v3_probation_shadow_rescue_min_recent_filled=(
            config.v3_probation_shadow_rescue_min_recent_filled
        ),
        v3_probation_shadow_rescue_min_recent_pnl_usd=(
            config.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        v3_probation_shadow_rescue_allow_genetics=(
            config.v3_probation_shadow_rescue_allow_genetics
        ),
        v3_persistent_loss_kill_min_closed_trades=(
            config.v3_persistent_loss_kill_min_closed_trades
        ),
        v3_persistent_loss_kill_pnl_pct=config.v3_persistent_loss_kill_pnl_pct,
        v3_persistent_loss_kill_win_rate_pct=(
            config.v3_persistent_loss_kill_win_rate_pct
        ),
        v3_persistent_loss_requires_virtual_weakness=(
            config.v3_persistent_loss_requires_virtual_weakness
        ),
        v3_persistent_loss_virtual_max_pnl_pct=(
            config.v3_persistent_loss_virtual_max_pnl_pct
        ),
        v3_persistent_loss_virtual_min_dd_pct=(
            config.v3_persistent_loss_virtual_min_dd_pct
        ),
        v3_probation_loss_kill_min_closed_trades=(
            config.v3_probation_loss_kill_min_closed_trades
        ),
        v3_probation_loss_kill_pnl_pct=config.v3_probation_loss_kill_pnl_pct,
        v3_probation_loss_kill_win_rate_pct=(
            config.v3_probation_loss_kill_win_rate_pct
        ),
        v3_probation_loss_kill_label_prefixes=(
            config.v3_probation_loss_kill_label_prefixes
        ),
        v3_realized_profit_lock_min_closed_trades=(
            config.v3_realized_profit_lock_min_closed_trades
        ),
        v3_realized_profit_lock_min_peak_pnl_pct=(
            config.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        v3_realized_profit_lock_max_giveback_pct=(
            config.v3_realized_profit_lock_max_giveback_pct
        ),
        v3_realized_profit_lock_floor_pnl_pct=(
            config.v3_realized_profit_lock_floor_pnl_pct
        ),
        v3_panteon_equity_guard_enabled=config.v3_panteon_equity_guard_enabled,
        v3_panteon_equity_guard_min_peak_pnl_pct=(
            config.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        v3_panteon_equity_guard_max_giveback_pct=(
            config.v3_panteon_equity_guard_max_giveback_pct
        ),
        v3_panteon_equity_guard_floor_pnl_pct=(
            config.v3_panteon_equity_guard_floor_pnl_pct
        ),
        v3_panteon_equity_guard_cooldown_bars=(
            config.v3_panteon_equity_guard_cooldown_bars
        ),
        hard_policy_experimental_min_bar=0,
    )


def _make_on_step(
    configured_writer: OutputWriter,
    config: RetrodateMarketConfig,
    *,
    step_errors: Optional[list[str]] = None,
    processed_counter: Optional[list[int]] = None,
):
    def _on_step(step: StepResult) -> None:
        if processed_counter is not None:
            processed_counter[0] = int(processed_counter[0]) + 1
        if step_errors is not None and step.error:
            step_errors.append(str(step.error))
        configured_writer.write(step)
        progress_every = int(config.progress_every_bars or 0)
        if progress_every > 0 and step.bar % progress_every == 0:
            print(
                "retrodate progress "
                f"bar={step.bar} regime={step.regime.label} "
                f"leader={step.leader or '-'} "
                f"shadow_filled={step.n_shadow_filled}",
                flush=True,
            )

    return _on_step


def _classify_regime(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
) -> tuple[Regime, float]:
    btc = prices.get("BTC/USDT") or prices.get("BTCUSDT")
    if btc is None:
        return Regime.NEUTRAL, 0.50
    if len(state.btc_closes) < state.btc_closes.maxlen:
        state.btc_closes.append(float(btc))
        return Regime.NEUTRAL, 0.55

    anchor = float(state.btc_closes[0])
    ret = (float(btc) / anchor - 1.0) if anchor > 0 else 0.0
    state.btc_closes.append(float(btc))
    confidence = min(0.95, 0.55 + abs(ret) * 4.0)
    if ret <= -0.08:
        return Regime.CRASH, confidence
    if ret <= -0.025:
        return Regime.BEARISH, confidence
    if ret >= 0.025:
        return Regime.BULLISH, confidence
    return Regime.NEUTRAL, max(0.55, 0.70 - abs(ret) * 3.0)


def _classify_symbol_regimes(
    prices: dict[str, float],
    state: RetrodateSnapshotState,
    *,
    fallback: Regime,
) -> dict[str, Regime]:
    regimes: dict[str, Regime] = {}
    for symbol, price in prices.items():
        history = state.symbol_closes.get(symbol)
        if not history:
            regimes[symbol] = fallback
            continue
        anchor = float(history[0])
        ret = (float(price) / anchor - 1.0) if anchor > 0 else 0.0
        if ret <= -0.08:
            regimes[symbol] = Regime.CRASH
        elif ret <= -0.025:
            regimes[symbol] = Regime.BEARISH
        elif ret >= 0.025:
            regimes[symbol] = Regime.BULLISH
        else:
            regimes[symbol] = Regime.NEUTRAL
    return regimes


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _new_output_dir(config: RetrodateMarketConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path(config.results_root).resolve()
        / "RETRODATE_MARKET"
        / f"{ts}_retrodate_market_v2"
    )


def _parse_years(raw: str) -> tuple[int, ...]:
    years = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not years:
        raise ValueError("years must not be empty")
    return years


def _parse_optional_agent_labels(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def _normalize_label_tuple(raw: object) -> tuple[str, ...]:
    parts: list[str] = []
    if isinstance(raw, str):
        values = raw.replace(";", ",").split(",")
    else:
        try:
            values = list(raw or ())
        except TypeError:
            values = [raw]
    for value in values:
        for part in str(value or "").replace(";", ",").split(","):
            label = part.strip()
            if label and label not in parts:
                parts.append(label)
    return tuple(parts)


def _merge_hard_policy_deny_labels(raw_labels: Sequence[str]) -> tuple[str, ...]:
    labels: list[str] = []
    for label in tuple(DEFAULT_RETRO_HARD_POLICY_DENY_LABELS) + tuple(raw_labels or ()):
        clean = str(label or "").strip()
        if clean and clean not in labels:
            labels.append(clean)
    return tuple(labels)


def _parse_probation_loss_label_prefixes(args: argparse.Namespace) -> tuple[str, ...]:
    if bool(getattr(args, "v3_probation_loss_kill_all_labels", False)):
        return ()
    raw_prefixes = getattr(args, "v3_probation_loss_kill_label_prefix", None)
    if raw_prefixes is None:
        return DEFAULT_PROBATION_LOSS_LABEL_PREFIXES
    return tuple(str(prefix).strip() for prefix in raw_prefixes if str(prefix).strip())


def write_candidate_diagnostics(
    output_dir: Path | str,
    *,
    candidate_events: Sequence[object],
    rejection_events: Sequence[object] = (),
    filename: str = "candidate_diagnostics.json",
) -> Path:
    buckets: dict[str, dict[str, Any]] = {}
    selected_rows = 0
    for event in candidate_events or ():
        label = str(_event_value(event, "player_label") or "").strip()
        if not label:
            continue
        row = buckets.setdefault(label, _new_candidate_diagnostic_bucket(label))
        row["bars_seen"] += 1
        selected = bool(_event_value(event, "selected_by_pantheon") or False)
        row["selected_bars"] += int(selected)
        selected_rows += int(selected)
        row["score_sum"] += _safe_float(_event_value(event, "score"))
        row["has_data_rows"] += int(bool(_event_value(event, "has_data") or False))
        row["closed_trades_sum"] += _safe_float(_event_value(event, "closed_trades"))
        row["signals_sum"] += _safe_float(_event_value(event, "signals"))
        row["recent_bars_sum"] += _safe_float(_event_value(event, "recent_bars"))
        row["recent_actionable_bars_sum"] += _safe_float(
            _event_value(event, "recent_actionable_bars")
        )
        row["actionable_share_sum"] += _safe_float(
            _event_value(event, "actionable_share")
        )
        row["recent_filled_sum"] += _safe_float(_event_value(event, "recent_filled"))
        row["recent_pnl_usd_sum"] += _safe_float(_event_value(event, "recent_pnl_usd"))

    rejections: dict[str, dict[str, Any]] = {}
    rejection_reason_counts: dict[str, int] = {}
    flash_reason_counts: dict[str, int] = {}
    flash_active_reason_counts: dict[str, int] = {}
    for event in rejection_events or ():
        label = str(_event_value(event, "player_label") or "").strip()
        if not label:
            continue
        reason = str(_event_value(event, "reason") or "")
        _count_str_key(rejection_reason_counts, reason)
        flash_reason = _flash_rejection_reason(reason)
        if flash_reason:
            _count_str_key(flash_reason_counts, flash_reason)
            if flash_reason not in FLASH_INACTIVE_REJECTION_REASONS:
                _count_str_key(flash_active_reason_counts, flash_reason)
        row = rejections.setdefault(label, {"count": 0, "last_reason": ""})
        row["count"] += 1
        row["last_reason"] = reason

    candidate_payload = {
        label: _candidate_diagnostic_payload(row)
        for label, row in sorted(
            buckets.items(),
            key=lambda item: (-item[1]["selected_bars"], item[0]),
        )
    }
    data = {
        "candidate_rows": sum(row["bars_seen"] for row in buckets.values()),
        "selected_rows": selected_rows,
        "rejection_rows": sum(row["count"] for row in rejections.values()),
        "rejection_summary": {
            "reason_counts": dict(sorted(rejection_reason_counts.items())),
            "flash_reason_counts": dict(sorted(flash_reason_counts.items())),
            "flash_active_reason_counts": dict(
                sorted(flash_active_reason_counts.items())
            ),
        },
        "group_summary": _candidate_group_summary(candidate_payload),
        "candidates": candidate_payload,
        "rejections": dict(sorted(rejections.items())),
    }
    path = Path(output_dir) / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _count_str_key(counts: dict[str, int], key: str) -> None:
    clean = str(key or "").strip()
    if not clean:
        return
    counts[clean] = int(counts.get(clean, 0) or 0) + 1


def _flash_rejection_reason(reason: str) -> str:
    clean = str(reason or "").strip()
    if not clean.startswith("flash:"):
        return ""
    return clean.split(":", 1)[1].strip()


def _new_candidate_diagnostic_bucket(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "group_type": _candidate_group_type(label),
        "bars_seen": 0,
        "selected_bars": 0,
        "score_sum": 0.0,
        "has_data_rows": 0,
        "closed_trades_sum": 0.0,
        "signals_sum": 0.0,
        "recent_bars_sum": 0.0,
        "recent_actionable_bars_sum": 0.0,
        "actionable_share_sum": 0.0,
        "recent_filled_sum": 0.0,
        "recent_pnl_usd_sum": 0.0,
    }


def _candidate_diagnostic_payload(row: dict[str, Any]) -> dict[str, Any]:
    bars = max(1, int(row["bars_seen"]))
    return {
        "group_type": row["group_type"],
        "bars_seen": int(row["bars_seen"]),
        "selected_bars": int(row["selected_bars"]),
        "selected_share_pct": _share(row["selected_bars"], row["bars_seen"]),
        "avg_score": float(row["score_sum"]) / bars,
        "has_data_share_pct": _share(row["has_data_rows"], row["bars_seen"]),
        "avg_closed_trades": float(row["closed_trades_sum"]) / bars,
        "avg_signals": float(row["signals_sum"]) / bars,
        "avg_recent_bars": float(row["recent_bars_sum"]) / bars,
        "avg_recent_actionable_bars": (
            float(row["recent_actionable_bars_sum"]) / bars
        ),
        "avg_actionable_share": float(row["actionable_share_sum"]) / bars,
        "avg_recent_filled": float(row["recent_filled_sum"]) / bars,
        "avg_recent_pnl_usd": float(row["recent_pnl_usd_sum"]) / bars,
    }


def _candidate_group_summary(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for row in candidates.values():
        group = str(row.get("group_type") or "profile")
        bucket = summary.setdefault(group, {
            "candidate_count": 0,
            "bars_seen": 0,
            "selected_bars": 0,
        })
        bucket["candidate_count"] += 1
        bucket["bars_seen"] += int(row.get("bars_seen", 0) or 0)
        bucket["selected_bars"] += int(row.get("selected_bars", 0) or 0)
    for bucket in summary.values():
        bucket["selected_share_pct"] = _share(
            bucket["selected_bars"],
            bucket["bars_seen"],
        )
    return dict(sorted(summary.items()))


def _candidate_group_type(label: str) -> str:
    if label == "NoTrade":
        return "notrade"
    if label.startswith("Solo_"):
        return "solo"
    if label.startswith("Fixed_"):
        return "fixed"
    if label.startswith("Antonius_") or label.startswith("Perfect_"):
        return "regime_switch"
    if label.endswith("StaticRotator") or label.startswith("Optimal_"):
        return "rotating"
    return "profile"


def write_flash_attribution_summary(
    output_dir: str | Path,
    *,
    execution_events: Sequence[object],
    position_closed_events: Sequence[object] = (),
    filename: str = "flash_attribution_summary.json",
) -> Path:
    output = Path(output_dir)
    causal_path = output / "causal_entry_decisions.jsonl"
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    context_buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    signal_index: dict[int, tuple[str, str, str]] = {}
    signal_context_index: dict[int, tuple[str, str, str, str]] = {}
    summary: dict[str, Any] = {
        "flash_decisions": 0,
        "selected_decisions": 0,
        "no_trade_decisions": 0,
        "selected_signals": 0,
        "selected_without_signal": 0,
        "executable_selected_signals": 0,
        "selected_filtered_before_execution": 0,
        "soft_allocator_signals": 0,
        "filled_signals": 0,
        "blocked_signals": 0,
        "rejected_signals": 0,
        "pending_signals": 0,
        "missing_execution_signals": 0,
        "unattributed_execution_events": 0,
        "unattributed_closed_trades": 0,
        "closed_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "realized_pnl_usd": 0.0,
        "filter_detail_counts": {},
    }
    executable_selected_signal_ids: set[int] = set()

    for row in _iter_jsonl(causal_path):
        executable_payloads = row.get("executable_signals")
        executable_ids = _signal_ids_from_payloads(executable_payloads)
        executable_known = isinstance(executable_payloads, list)
        filter_detail_counts = _flash_filter_detail_counts(
            row.get("signal_filter_details")
        )
        _merge_int_counts(summary["filter_detail_counts"], filter_detail_counts)
        for decision in row.get("flash_decisions") or ():
            if not isinstance(decision, dict):
                continue
            summary["flash_decisions"] += 1
            selected_actor = str(decision.get("selected_actor") or "")
            if selected_actor == "NoTrade":
                summary["no_trade_decisions"] += 1
                continue
            summary["selected_decisions"] += 1
            signal = decision.get("signal")
            if not isinstance(signal, dict):
                summary["selected_without_signal"] += 1
                continue
            signal_id = _safe_int(signal.get("id"), default=-1)
            if signal_id < 0:
                summary["selected_without_signal"] += 1
                continue
            actor_type = str(decision.get("actor_type") or "")
            actor_key = _flash_decision_actor_key(decision)
            symbol = str(decision.get("symbol") or signal.get("sym") or "").upper()
            action = str(decision.get("action") or signal.get("action") or "")
            regime = _flash_context_regime(row.get("regime") or signal.get("regime"))
            key = (actor_key, symbol, action)
            bucket = buckets.setdefault(
                key,
                _new_flash_attribution_bucket(
                    actor_key=actor_key,
                    actor_label=selected_actor,
                    actor_type=actor_type,
                    symbol=symbol,
                    action=action,
                ),
            )
            context_key = (actor_key, symbol, action, regime)
            context_bucket = context_buckets.setdefault(
                context_key,
                _new_flash_attribution_bucket(
                    actor_key=actor_key,
                    actor_label=selected_actor,
                    actor_type=actor_type,
                    symbol=symbol,
                    action=action,
                    regime=regime,
                ),
            )
            selected_candidate = _flash_selected_candidate(decision)
            _accumulate_flash_selected_bucket(bucket, decision, selected_candidate)
            _accumulate_flash_selected_bucket(context_bucket, decision, selected_candidate)
            signal_index[signal_id] = key
            signal_context_index[signal_id] = context_key
            summary["selected_signals"] += 1
            if executable_known and signal_id not in executable_ids:
                bucket["selected_filtered_before_execution"] += 1
                context_bucket["selected_filtered_before_execution"] += 1
                summary["selected_filtered_before_execution"] += 1
                _merge_int_counts(
                    bucket["filter_detail_counts"],
                    filter_detail_counts,
                )
                _merge_int_counts(
                    context_bucket["filter_detail_counts"],
                    filter_detail_counts,
                )
            else:
                bucket["executable_selected_signals"] += 1
                context_bucket["executable_selected_signals"] += 1
                summary["executable_selected_signals"] += 1
                executable_selected_signal_ids.add(signal_id)
        if isinstance(row.get("soft_allocator"), dict):
            for signal in (
                executable_payloads if isinstance(executable_payloads, list) else ()
            ):
                if not isinstance(signal, dict):
                    continue
                signal_id = _safe_int(signal.get("id"), default=-1)
                if signal_id < 0 or signal_id in signal_index:
                    continue
                actor_label = str(
                    signal.get("by_player")
                    or row.get("executed_leader")
                    or row.get("selected_leader")
                    or "PanteonSoft"
                )
                actor_key = f"soft:{actor_label}"
                actor_type = "soft_allocator"
                symbol = str(signal.get("sym") or "").upper()
                action = str(signal.get("action") or "")
                regime = _flash_context_regime(row.get("regime") or signal.get("regime"))
                key = (actor_key, symbol, action)
                bucket = buckets.setdefault(
                    key,
                    _new_flash_attribution_bucket(
                        actor_key=actor_key,
                        actor_label=actor_label,
                        actor_type=actor_type,
                        symbol=symbol,
                        action=action,
                    ),
                )
                context_key = (actor_key, symbol, action, regime)
                context_bucket = context_buckets.setdefault(
                    context_key,
                    _new_flash_attribution_bucket(
                        actor_key=actor_key,
                        actor_label=actor_label,
                        actor_type=actor_type,
                        symbol=symbol,
                        action=action,
                        regime=regime,
                    ),
                )
                _accumulate_soft_allocator_selected_bucket(bucket, signal)
                _accumulate_soft_allocator_selected_bucket(context_bucket, signal)
                signal_index[signal_id] = key
                signal_context_index[signal_id] = context_key
                summary["selected_signals"] += 1
                summary["executable_selected_signals"] += 1
                summary["soft_allocator_signals"] += 1
                executable_selected_signal_ids.add(signal_id)

    executed_signal_ids: set[int] = set()
    for event in execution_events or ():
        signal_id = _safe_int(_event_value(event, "signal_id"), default=-1)
        key = signal_index.get(signal_id)
        if key is None:
            summary["unattributed_execution_events"] += 1
            continue
        executed_signal_ids.add(signal_id)
        status = str(_event_value(event, "status") or "").lower()
        bucket = buckets[key]
        context_key = signal_context_index.get(signal_id)
        context_bucket = context_buckets.get(context_key) if context_key else None
        if status == "filled":
            bucket["filled_signals"] += 1
            if context_bucket is not None:
                context_bucket["filled_signals"] += 1
            summary["filled_signals"] += 1
        elif status == "blocked":
            bucket["blocked_signals"] += 1
            if context_bucket is not None:
                context_bucket["blocked_signals"] += 1
            summary["blocked_signals"] += 1
            _increment_flash_failure(bucket, event)
            if context_bucket is not None:
                _increment_flash_failure(context_bucket, event)
        elif status == "rejected":
            bucket["rejected_signals"] += 1
            if context_bucket is not None:
                context_bucket["rejected_signals"] += 1
            summary["rejected_signals"] += 1
            _increment_flash_failure(bucket, event)
            if context_bucket is not None:
                _increment_flash_failure(context_bucket, event)
        elif status == "pending":
            bucket["pending_signals"] += 1
            if context_bucket is not None:
                context_bucket["pending_signals"] += 1
            summary["pending_signals"] += 1
            _increment_flash_failure(bucket, event)
            if context_bucket is not None:
                _increment_flash_failure(context_bucket, event)

    for signal_id in executable_selected_signal_ids:
        if signal_id not in executed_signal_ids:
            summary["missing_execution_signals"] += 1

    for event in position_closed_events or ():
        signal_id = _safe_int(_event_value(event, "open_signal_id"), default=-1)
        key = signal_index.get(signal_id)
        if key is None:
            summary["unattributed_closed_trades"] += 1
            continue
        pnl = _safe_float(_event_value(event, "realized_pnl"))
        bucket = buckets[key]
        context_key = signal_context_index.get(signal_id)
        context_bucket = context_buckets.get(context_key) if context_key else None
        bucket["closed_trades"] += 1
        bucket["realized_pnl_usd"] += pnl
        bucket["winning_trades"] += int(pnl > 0.0)
        bucket["losing_trades"] += int(pnl <= 0.0)
        if context_bucket is not None:
            context_bucket["closed_trades"] += 1
            context_bucket["realized_pnl_usd"] += pnl
            context_bucket["winning_trades"] += int(pnl > 0.0)
            context_bucket["losing_trades"] += int(pnl <= 0.0)
        summary["closed_trades"] += 1
        summary["winning_trades"] += int(pnl > 0.0)
        summary["losing_trades"] += int(pnl <= 0.0)
        summary["realized_pnl_usd"] += pnl

    rows = [_flash_attribution_payload(row) for row in buckets.values()]
    context_rows = [_flash_attribution_payload(row) for row in context_buckets.values()]
    rows.sort(
        key=lambda row: (
            -float(row.get("realized_pnl_usd", 0.0) or 0.0),
            -int(row.get("filled_signals", 0) or 0),
            str(row.get("actor_key") or ""),
            str(row.get("symbol") or ""),
            str(row.get("action") or ""),
        )
    )
    context_rows.sort(
        key=lambda row: (
            -float(row.get("realized_pnl_usd", 0.0) or 0.0),
            -int(row.get("filled_signals", 0) or 0),
            str(row.get("actor_key") or ""),
            str(row.get("symbol") or ""),
            str(row.get("action") or ""),
            str(row.get("regime") or ""),
        )
    )
    data = {
        "summary": summary,
        "actor_summary": _flash_actor_summary(rows),
        "rows": rows,
        "context_rows": context_rows,
    }
    path = output / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def write_component_benchmark_report(
    output: Path,
    *,
    panteon_pnl_usd: float,
    initial_capital: float,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    excluded_component_labels: Sequence[str] = (),
    min_alpha_pct: float = 2.0,
    filename: str = "component_benchmark_report.json",
) -> Path:
    output_dir = Path(output)
    capital = _validate_component_benchmark_capital(initial_capital)
    excluded_labels = _component_benchmark_label_set(excluded_component_labels)
    components = _component_benchmark_rows(
        initial_capital=capital,
        shadow_agent_pnl_events=shadow_agent_pnl_events,
        shadow_player_pnl_events=shadow_player_pnl_events,
        excluded_component_labels=excluded_labels,
    )
    best = components[0] if components else {}
    panteon_pnl = float(panteon_pnl_usd or 0.0)
    panteon_pct = panteon_pnl / capital * 100.0
    best_pct = float(best.get("pnl_pct", 0.0) or 0.0)
    alpha_pct = panteon_pct - best_pct
    data = {
        "summary": {
            "panteon_pnl_usd": panteon_pnl,
            "panteon_pnl_pct": panteon_pct,
            "best_component_label": str(best.get("label") or ""),
            "best_component_type": str(best.get("actor_type") or ""),
            "best_component_pnl_usd": float(best.get("pnl_usd", 0.0) or 0.0),
            "best_component_pnl_pct": best_pct,
            "panteon_alpha_pct": alpha_pct,
            "min_alpha_pct": float(min_alpha_pct),
            "panteon_beats_best_component": alpha_pct >= float(min_alpha_pct),
            "excluded_component_labels": sorted(excluded_labels),
        },
        "components": components,
    }
    path = output_dir / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path = output_dir / Path(filename).with_suffix(".md").name
    md_path.write_text(
        "\n".join(_component_benchmark_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path


def write_standalone_vs_flash_selected_report(
    output_dir: str | Path,
    *,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    flash_attribution: Optional[dict[str, Any]] = None,
    initial_capital: float = 1000.0,
    target_labels: Sequence[str] = ("Solo_MomentumScalper", "LiveOIBreakout"),
    filename: str = "standalone_vs_flash_selected_report.json",
) -> Path:
    output = Path(output_dir)
    capital = _validate_component_benchmark_capital(initial_capital)
    target_set = tuple(
        dict.fromkeys(
            str(label).strip()
            for label in target_labels or ()
            if str(label or "").strip()
        )
    )
    components = _component_benchmark_rows(
        initial_capital=capital,
        shadow_agent_pnl_events=shadow_agent_pnl_events,
        shadow_player_pnl_events=shadow_player_pnl_events,
    )
    component_by_label: dict[str, dict[str, Any]] = {}
    for row in components:
        label = str(row.get("label") or "")
        if label and label not in component_by_label:
            component_by_label[label] = row

    attribution = flash_attribution if isinstance(flash_attribution, dict) else {}
    actor_summary = attribution.get("actor_summary")
    if not isinstance(actor_summary, dict):
        actor_summary = {}
    attribution_rows = attribution.get("rows")
    if not isinstance(attribution_rows, list):
        attribution_rows = []

    actor_rows: list[dict[str, Any]] = []
    for label in target_set:
        standalone = component_by_label.get(label, {})
        standalone_actor_type = str(standalone.get("actor_type") or "")
        flash_actor_key, flash_actor = _standalone_vs_flash_actor_summary(
            label,
            standalone_actor_type=standalone_actor_type,
            actor_summary=actor_summary,
        )
        standalone_pnl_usd = _safe_float(standalone.get("pnl_usd"))
        flash_pnl_usd = _safe_float(flash_actor.get("realized_pnl_usd"))
        signal_rows = _standalone_vs_flash_signal_rows(
            attribution_rows,
            label=label,
            actor_key=flash_actor_key,
            initial_capital=capital,
        )
        row = {
            "label": label,
            "standalone_actor_type": standalone_actor_type,
            "flash_actor_key": flash_actor_key,
            "flash_actor_type": str(flash_actor.get("actor_type") or ""),
            "standalone_pnl_usd": standalone_pnl_usd,
            "standalone_pnl_pct": standalone_pnl_usd / capital * 100.0,
            "standalone_closed_trades": int(standalone.get("closed_trades", 0) or 0),
            "standalone_wins": int(standalone.get("wins", 0) or 0),
            "standalone_win_rate_pct": _safe_float(standalone.get("win_rate_pct")),
            "flash_selected_pnl_usd": flash_pnl_usd,
            "flash_selected_pnl_pct": flash_pnl_usd / capital * 100.0,
            "flash_selected_signals": int(
                flash_actor.get("selected_signals", 0) or 0
            ),
            "flash_executable_selected_signals": int(
                flash_actor.get("executable_selected_signals", 0) or 0
            ),
            "flash_selected_filtered_before_execution": int(
                flash_actor.get("selected_filtered_before_execution", 0) or 0
            ),
            "flash_filled_signals": int(flash_actor.get("filled_signals", 0) or 0),
            "flash_closed_trades": int(flash_actor.get("closed_trades", 0) or 0),
            "selection_alpha_usd": flash_pnl_usd - standalone_pnl_usd,
            "selection_alpha_pct": (
                (flash_pnl_usd - standalone_pnl_usd) / capital * 100.0
            ),
            "flash_capture_ratio_pct": _share(flash_pnl_usd, standalone_pnl_usd),
            "best_signal_keys": signal_rows["best"],
            "worst_signal_keys": signal_rows["worst"],
        }
        actor_rows.append(row)

    actor_rows.sort(
        key=lambda row: (
            float(row["selection_alpha_pct"]),
            str(row["label"]),
        )
    )
    total_standalone = sum(float(row["standalone_pnl_usd"]) for row in actor_rows)
    total_flash = sum(float(row["flash_selected_pnl_usd"]) for row in actor_rows)
    data = {
        "summary": {
            "target_count": len(actor_rows),
            "targets_with_flash_selection": sum(
                1 for row in actor_rows if int(row["flash_selected_signals"]) > 0
            ),
            "total_standalone_pnl_usd": total_standalone,
            "total_standalone_pnl_pct": total_standalone / capital * 100.0,
            "total_flash_selected_pnl_usd": total_flash,
            "total_flash_selected_pnl_pct": total_flash / capital * 100.0,
            "total_selection_alpha_usd": total_flash - total_standalone,
            "total_selection_alpha_pct": (
                (total_flash - total_standalone) / capital * 100.0
            ),
        },
        "actors": actor_rows,
    }
    path = output / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    md_path = output / Path(filename).with_suffix(".md").name
    md_path.write_text(
        "\n".join(_standalone_vs_flash_selected_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path


def _write_walk_forward_report_from_event_log(
    output_dir: str | Path,
    event_log: EventLog,
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = [
        EventLog._event_to_dict(event)
        for event in event_log.query(
            event_types=[RegimeDetected, OrderFilled, PositionOpened, PositionClosed]
        )
    ]
    return write_walk_forward_report_from_events(
        results_root=str(output),
        events=events,
        output_path=str(output / "walk_forward_report.json"),
    )


def _standalone_vs_flash_actor_summary(
    label: str,
    *,
    standalone_actor_type: str,
    actor_summary: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    preferred_actor_type = "ensemble" if standalone_actor_type == "player" else "agent"
    preferred_key = f"{preferred_actor_type}:{label}"
    actor = actor_summary.get(preferred_key)
    if isinstance(actor, Mapping):
        return preferred_key, actor
    for key, value in actor_summary.items():
        if not isinstance(value, Mapping):
            continue
        if str(value.get("actor_label") or "") == label:
            return str(key), value
    return preferred_key, {}


def _standalone_vs_flash_signal_rows(
    rows: Sequence[object],
    *,
    label: str,
    actor_key: str,
    initial_capital: float,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    matching: list[dict[str, Any]] = []
    for raw in rows or ():
        if not isinstance(raw, Mapping):
            continue
        if (
            str(raw.get("actor_key") or "") != actor_key
            and str(raw.get("actor_label") or "") != label
        ):
            continue
        symbol = str(raw.get("symbol") or "")
        action = str(raw.get("action") or "")
        key = str(raw.get("actor_key") or actor_key)
        signal_key = f"{key}|{symbol}|{action}" if symbol or action else key
        pnl_usd = _safe_float(raw.get("realized_pnl_usd"))
        matching.append({
            "signal_key": signal_key,
            "actor_key": key,
            "symbol": symbol,
            "action": action,
            "selected_signals": int(raw.get("selected_signals", 0) or 0),
            "filled_signals": int(raw.get("filled_signals", 0) or 0),
            "closed_trades": int(raw.get("closed_trades", 0) or 0),
            "realized_pnl_usd": pnl_usd,
            "realized_pnl_pct": pnl_usd / initial_capital * 100.0,
        })
    return {
        "best": sorted(
            matching,
            key=lambda row: (-float(row["realized_pnl_usd"]), str(row["signal_key"])),
        )[:limit],
        "worst": sorted(
            matching,
            key=lambda row: (float(row["realized_pnl_usd"]), str(row["signal_key"])),
        )[:limit],
    }


def _component_benchmark_rows(
    *,
    initial_capital: float,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    excluded_component_labels: Sequence[str] = (),
) -> list[dict[str, Any]]:
    capital = _validate_component_benchmark_capital(initial_capital)
    excluded_labels = _component_benchmark_label_set(excluded_component_labels)
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for actor_type, events in (
        ("agent", shadow_agent_pnl_events),
        ("player", shadow_player_pnl_events),
    ):
        for event in events or ():
            label = str(_event_value(event, "label") or "").strip()
            if not label:
                continue
            if label in excluded_labels:
                continue
            bucket = buckets.setdefault(
                (actor_type, label),
                {
                    "actor_type": actor_type,
                    "label": label,
                    "pnl_usd": 0.0,
                    "closed_trades": 0,
                    "wins": 0,
                },
            )
            bucket["pnl_usd"] += _safe_float(_event_value(event, "pnl_usd"))
            bucket["closed_trades"] += _safe_int(_event_value(event, "closed_trades"))
            bucket["wins"] += _safe_int(_event_value(event, "wins"))

    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        closed = int(bucket["closed_trades"])
        wins = int(bucket["wins"])
        pnl_usd = float(bucket["pnl_usd"])
        rows.append({
            "actor_type": str(bucket["actor_type"]),
            "label": str(bucket["label"]),
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_usd / capital * 100.0,
            "closed_trades": closed,
            "wins": wins,
            "win_rate_pct": _share(wins, closed),
        })
    rows.sort(
        key=lambda row: (
            -float(row["pnl_pct"]),
            str(row["actor_type"]),
            str(row["label"]),
        )
    )
    return rows


def _component_benchmark_label_set(labels: Sequence[str]) -> set[str]:
    return {
        str(label).strip()
        for label in labels or ()
        if str(label or "").strip()
    }


def _validate_component_benchmark_capital(initial_capital: float) -> float:
    try:
        capital = float(initial_capital)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_capital must be > 0") from exc
    if not (0.0 < capital < float("inf")):
        raise ValueError("initial_capital must be > 0")
    return capital


def _live_session_panteon_pnl_usd(status: object) -> float | None:
    if not isinstance(status, dict):
        return None
    live_session = status.get("live_session")
    if not isinstance(live_session, dict):
        return None
    for key in (
        "panteon_owned_total_pnl_usd",
        "panteon_owned_realized_pnl_usd",
    ):
        if key not in live_session:
            continue
        try:
            pnl = float(live_session[key])
        except (TypeError, ValueError):
            return None
        if not (float("-inf") < pnl < float("inf")):
            return None
        return pnl
    return None


def _write_component_benchmark_report_from_status(
    output_dir: str | Path,
    status: object,
    *,
    initial_capital: float,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    excluded_component_labels: Sequence[str] = (),
    step_errors: list[str],
) -> Path | None:
    panteon_pnl_usd = _live_session_panteon_pnl_usd(status)
    if panteon_pnl_usd is None:
        # Phase 4 / C13: пустой прогон (Panteon не торговал) — это валидный
        # результат, а не сбой отчёта. Пишем бенчмарк с 0.0 и явной пометкой,
        # чтобы genuine empty runs были отличимы от ошибок и видны в отчёте.
        step_errors.append("component_benchmark_note: panteon_did_not_trade")
        panteon_pnl_usd = 0.0
    return write_component_benchmark_report(
        output_dir,
        panteon_pnl_usd=panteon_pnl_usd,
        initial_capital=initial_capital,
        shadow_agent_pnl_events=shadow_agent_pnl_events,
        shadow_player_pnl_events=shadow_player_pnl_events,
        excluded_component_labels=excluded_component_labels,
    )


def _write_retrodate_final_artifact_reports(
    output_dir: str | Path,
    *,
    config: RetrodateMarketConfig,
    shadow_updates: Sequence[object],
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    candidate_rejections: Sequence[object],
    step_errors: list[str],
) -> None:
    output = Path(output_dir)
    try:
        write_flash_signal_key_shadow_report(
            output,
            shadow_updates=shadow_updates,
            initial_capital=config.initial_capital,
            promotion_manifest_config=_flash_promotion_manifest_config(config),
        )
    except Exception as exc:
        step_errors.append(
            f"flash_signal_key_shadow_report_failed: {type(exc).__name__}: {exc}"
        )
    try:
        write_flash_genetics_intersection_report(output)
    except Exception as exc:
        step_errors.append(
            f"flash_genetics_intersection_report_failed: {type(exc).__name__}: {exc}"
        )
    _write_component_benchmark_report_from_status(
        output,
        _load_json(output / "status.json"),
        initial_capital=config.initial_capital,
        shadow_agent_pnl_events=shadow_agent_pnl_events,
        shadow_player_pnl_events=shadow_player_pnl_events,
        excluded_component_labels=_component_benchmark_excluded_labels(config),
        step_errors=step_errors,
    )
    try:
        write_standalone_vs_flash_selected_report(
            output,
            shadow_agent_pnl_events=shadow_agent_pnl_events,
            shadow_player_pnl_events=shadow_player_pnl_events,
            flash_attribution=_load_json(output / "flash_attribution_summary.json"),
            initial_capital=config.initial_capital,
        )
    except Exception as exc:
        step_errors.append(
            f"standalone_vs_flash_selected_report_failed: {type(exc).__name__}: {exc}"
        )
    if config.experimental_flash_actors_enabled:
        try:
            write_experimental_flash_shadow_report(
                output,
                shadow_agent_pnl_events=shadow_agent_pnl_events,
                experimental_labels=experimental_flash_agent_labels(),
                initial_capital=config.initial_capital,
                flash_attribution=_load_json(output / "flash_attribution_summary.json"),
                min_closed_trades=50,
                min_pnl_pct=0.0,
                max_drawdown_pct=25.0,
            )
        except Exception as exc:
            step_errors.append(
                "experimental_flash_shadow_report_failed: "
                f"{type(exc).__name__}: {exc}"
            )
    try:
        write_oracle_mismatch_report(
            output,
            output / "trading.log",
            shadow_updates=shadow_updates,
            candidate_rejections=candidate_rejections,
        )
    except Exception as exc:
        step_errors.append(
            f"oracle_mismatch_report_failed: {type(exc).__name__}: {exc}"
        )


def _component_benchmark_excluded_labels(
    config: RetrodateMarketConfig,
) -> tuple[str, ...]:
    labels = list(config.hard_policy_deny_labels)
    if (
        config.experimental_flash_actors_enabled
        and not config.experimental_flash_real_actors_enabled
    ):
        labels.extend(experimental_flash_agent_labels())
    return tuple(dict.fromkeys(str(label) for label in labels if str(label)))


def _component_benchmark_report_lines(data: dict[str, Any]) -> list[str]:
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    components = data.get("components", []) if isinstance(data, dict) else []
    best_label = summary.get("best_component_label") or "-"
    best_type = summary.get("best_component_type") or "-"
    excluded_labels = summary.get("excluded_component_labels") or []
    lines = [
        "# Component Benchmark",
        "",
        f"- Panteon PnL: {_fmt_pct(summary.get('panteon_pnl_pct'))}",
        f"- Best component: `{best_type}:{best_label}` at {_fmt_pct(summary.get('best_component_pnl_pct'))}",
        f"- Panteon alpha: {_fmt_pct(summary.get('panteon_alpha_pct'))}",
        f"- Required alpha: {_fmt_pct(summary.get('min_alpha_pct'))}",
        f"- Beats best component: {bool(summary.get('panteon_beats_best_component'))}",
        f"- Excluded non-deployable labels: {len(excluded_labels) if isinstance(excluded_labels, list) else 0}",
        "",
        "## Top Components",
    ]
    if not isinstance(components, list) or not components:
        lines.append("- none")
        return lines
    for row in components[:20]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('actor_type')}:{row.get('label')}`: "
            f"{_fmt_pct(row.get('pnl_pct'))}, "
            f"closed {int(row.get('closed_trades', 0) or 0)}, "
            f"win rate {_fmt_pct(row.get('win_rate_pct'))}"
        )
    return lines


def _standalone_vs_flash_selected_report_lines(data: dict[str, Any]) -> list[str]:
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    actors = data.get("actors", []) if isinstance(data, dict) else []
    lines = [
        "# Standalone vs Flash Selected",
        "",
        f"- Targets: {int(summary.get('target_count', 0) or 0)}",
        f"- Targets with Flash selection: {int(summary.get('targets_with_flash_selection', 0) or 0)}",
        f"- Standalone total PnL: {_fmt_pct(summary.get('total_standalone_pnl_pct'))}",
        f"- Flash-selected total PnL: {_fmt_pct(summary.get('total_flash_selected_pnl_pct'))}",
        f"- Selection alpha: {_fmt_pct(summary.get('total_selection_alpha_pct'))}",
        "",
        "## Actors",
    ]
    if not isinstance(actors, list) or not actors:
        lines.append("- none")
        return lines
    for row in actors:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('label')}`: standalone "
            f"{_fmt_pct(row.get('standalone_pnl_pct'))}, Flash-selected "
            f"{_fmt_pct(row.get('flash_selected_pnl_pct'))}, alpha "
            f"{_fmt_pct(row.get('selection_alpha_pct'))}, selected "
            f"{int(row.get('flash_selected_signals', 0) or 0)}"
        )
        worst = row.get("worst_signal_keys")
        if isinstance(worst, list) and worst:
            top_worst = worst[0]
            if isinstance(top_worst, dict):
                lines.append(
                    f"  - worst `{top_worst.get('signal_key')}`: "
                    f"{_fmt_pct(top_worst.get('realized_pnl_pct'))}"
                )
    return lines


def write_experimental_flash_shadow_report(
    output_dir: str | Path,
    *,
    shadow_agent_pnl_events: Sequence[object],
    experimental_labels: Sequence[str],
    initial_capital: float = 1000.0,
    flash_attribution: Optional[dict[str, Any]] = None,
    min_closed_trades: int = 50,
    min_pnl_pct: float = 0.0,
    max_drawdown_pct: float = 25.0,
    filename: str = "experimental_flash_shadow_report.json",
) -> Path:
    output = Path(output_dir)
    capital = max(1e-9, float(initial_capital or 0.0))
    labels = tuple(
        dict.fromkeys(str(label) for label in experimental_labels if str(label))
    )
    states: dict[str, dict[str, Any]] = {
        label: {
            "label": label,
            "shadow_pnl_usd": 0.0,
            "closed_trades": 0,
            "wins": 0,
            "bars": 0,
            "equity": capital,
            "peak": capital,
            "max_drawdown_pct": 0.0,
        }
        for label in labels
    }
    period_states: dict[tuple[str, str], dict[str, Any]] = {}

    ordered_events = sorted(
        (
            event
            for event in shadow_agent_pnl_events or ()
            if str(_event_value(event, "label") or "") in states
        ),
        key=lambda event: (
            _safe_int(_event_value(event, "bar"), default=0),
            str(_event_value(event, "label") or ""),
        ),
    )
    for event in ordered_events:
        label = str(_event_value(event, "label") or "")
        state = states[label]
        pnl = _safe_float(_event_value(event, "pnl_usd"))
        state["shadow_pnl_usd"] += pnl
        state["closed_trades"] += _safe_int(_event_value(event, "closed_trades"))
        state["wins"] += _safe_int(_event_value(event, "wins"))
        state["bars"] += 1
        state["equity"] += pnl
        state["peak"] = max(float(state["peak"]), float(state["equity"]))
        if float(state["peak"]) > 0:
            drawdown = (
                (float(state["peak"]) - float(state["equity"]))
                / float(state["peak"])
                * 100.0
            )
            state["max_drawdown_pct"] = max(
                float(state["max_drawdown_pct"]),
                drawdown,
            )
        period = _shadow_event_half_year_period(event)
        period_state = period_states.setdefault(
            (label, period),
            {
                "label": label,
                "period": period,
                "pnl_usd": 0.0,
                "closed_trades": 0,
                "wins": 0,
                "bars": 0,
                "equity": capital,
                "peak": capital,
                "max_drawdown_pct": 0.0,
            },
        )
        _update_experimental_shadow_state(period_state, event, capital)

    real_summary = (
        flash_attribution.get("actor_summary", {})
        if isinstance(flash_attribution, dict)
        else {}
    )
    rows: list[dict[str, Any]] = []
    for label, state in states.items():
        real = _experimental_real_actor_summary(real_summary, label)
        closed = int(state["closed_trades"])
        wins = int(state["wins"])
        pnl_usd = float(state["shadow_pnl_usd"])
        pnl_pct = pnl_usd / capital * 100.0
        max_dd = float(state["max_drawdown_pct"])
        shadow_gate_passed = (
            closed >= int(min_closed_trades)
            and pnl_pct >= float(min_pnl_pct)
            and max_dd <= float(max_drawdown_pct)
        )
        real_selected = int(real["selected_signals"])
        promote_to_real = bool(shadow_gate_passed and real_selected == 0)
        reasons = []
        if closed < int(min_closed_trades):
            reasons.append("not_enough_closed_trades")
        if pnl_pct < float(min_pnl_pct):
            reasons.append("shadow_pnl_below_gate")
        if max_dd > float(max_drawdown_pct):
            reasons.append("shadow_drawdown_above_gate")
        if real_selected:
            reasons.append("already_traded_real")
        rows.append({
            "label": label,
            "shadow_pnl_usd": pnl_usd,
            "shadow_pnl_pct": pnl_pct,
            "closed_trades": closed,
            "wins": wins,
            "win_rate_pct": _share(wins, closed),
            "bars_observed": int(state["bars"]),
            "max_drawdown_pct": max_dd,
            "real_selected_signals": real_selected,
            "real_closed_trades": int(real["closed_trades"]),
            "realized_pnl_usd": float(real["realized_pnl_usd"]),
            "shadow_gate_passed": shadow_gate_passed,
            "promote_to_real": promote_to_real,
            "gate_reasons": reasons,
        })
    rows.sort(
        key=lambda row: (
            not bool(row["promote_to_real"]),
            -float(row["shadow_pnl_pct"]),
            str(row["label"]),
        )
    )
    period_rows = [
        _experimental_shadow_period_payload(
            state,
            initial_capital=capital,
            min_closed_trades=min_closed_trades,
            min_pnl_pct=min_pnl_pct,
            max_drawdown_pct=max_drawdown_pct,
        )
        for state in period_states.values()
    ]
    period_rows.sort(key=lambda row: (str(row["period"]), str(row["label"])))
    latest_period = _latest_known_period(period_rows)
    latest_period_gate_passed = sum(
        1
        for row in period_rows
        if row["period"] == latest_period and row["gate_passed"]
    )
    summary = {
        "label_count": len(rows),
        "shadow_gate_passed": sum(1 for row in rows if row["shadow_gate_passed"]),
        "promote_to_real": sum(1 for row in rows if row["promote_to_real"]),
        "real_selected_signals": sum(int(row["real_selected_signals"]) for row in rows),
        "latest_period": latest_period,
        "latest_period_gate_passed": latest_period_gate_passed,
        "min_closed_trades": int(min_closed_trades),
        "min_pnl_pct": float(min_pnl_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "initial_capital": capital,
    }
    data = {"summary": summary, "labels": rows, "periods": period_rows}
    path = output / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    md_path = output / "experimental_flash_shadow_report.md"
    md_path.write_text(
        "\n".join(_experimental_flash_shadow_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path


def _experimental_real_actor_summary(
    actor_summary: object,
    label: str,
) -> dict[str, float]:
    out = {
        "selected_signals": 0.0,
        "closed_trades": 0.0,
        "realized_pnl_usd": 0.0,
    }
    if not isinstance(actor_summary, dict):
        return out
    for key in (f"agent:{label}", f"ensemble:{label}", label):
        row = actor_summary.get(key)
        if not isinstance(row, dict):
            continue
        out["selected_signals"] += _safe_float(row.get("selected_signals"))
        out["closed_trades"] += _safe_float(row.get("closed_trades"))
        out["realized_pnl_usd"] += _safe_float(row.get("realized_pnl_usd"))
    return out


def _update_experimental_shadow_state(
    state: dict[str, Any],
    event: object,
    initial_capital: float,
) -> None:
    pnl = _safe_float(_event_value(event, "pnl_usd"))
    state["pnl_usd"] += pnl
    state["closed_trades"] += _safe_int(_event_value(event, "closed_trades"))
    state["wins"] += _safe_int(_event_value(event, "wins"))
    state["bars"] += 1
    state["equity"] += pnl
    state["peak"] = max(float(state["peak"]), float(state["equity"]))
    if float(state["peak"]) > 0:
        drawdown = (
            (float(state["peak"]) - float(state["equity"]))
            / float(state["peak"])
            * 100.0
        )
        state["max_drawdown_pct"] = max(
            float(state["max_drawdown_pct"]),
            drawdown,
        )


def _experimental_shadow_period_payload(
    state: dict[str, Any],
    *,
    initial_capital: float,
    min_closed_trades: int,
    min_pnl_pct: float,
    max_drawdown_pct: float,
) -> dict[str, Any]:
    closed = int(state.get("closed_trades", 0) or 0)
    wins = int(state.get("wins", 0) or 0)
    pnl_usd = float(state.get("pnl_usd", 0.0) or 0.0)
    pnl_pct = pnl_usd / max(1e-9, float(initial_capital or 0.0)) * 100.0
    max_dd = float(state.get("max_drawdown_pct", 0.0) or 0.0)
    gate_passed = (
        closed >= int(min_closed_trades)
        and pnl_pct >= float(min_pnl_pct)
        and max_dd <= float(max_drawdown_pct)
    )
    return {
        "label": str(state.get("label") or ""),
        "period": str(state.get("period") or "unknown"),
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "closed_trades": closed,
        "wins": wins,
        "win_rate_pct": _share(wins, closed),
        "bars_observed": int(state.get("bars", 0) or 0),
        "max_drawdown_pct": max_dd,
        "gate_passed": gate_passed,
    }


def _shadow_event_half_year_period(event: object) -> str:
    text = str(_event_value(event, "timestamp") or "").strip()
    if len(text) >= 7:
        try:
            year = int(text[:4])
            month = int(text[5:7])
        except ValueError:
            return "unknown"
        half = "H1" if month <= 6 else "H2"
        return f"{year:04d}-{half}"
    return "unknown"


def _latest_known_period(rows: Sequence[dict[str, Any]]) -> str:
    periods = sorted(
        str(row.get("period") or "")
        for row in rows
        if str(row.get("period") or "") and str(row.get("period") or "") != "unknown"
    )
    return periods[-1] if periods else ""


def _experimental_flash_shadow_report_lines(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    rows = report.get("labels", []) if isinstance(report, dict) else []
    lines = [
        "# Experimental Flash Shadow Gate",
        "",
        f"- Labels: {int(summary.get('label_count', 0) or 0)}",
        f"- Shadow gate passed: {int(summary.get('shadow_gate_passed', 0) or 0)}",
        f"- Promote-to-real labels: {int(summary.get('promote_to_real', 0) or 0)}",
        f"- Real selected signals: {int(summary.get('real_selected_signals', 0) or 0)}",
        f"- Latest period: `{summary.get('latest_period') or '-'}`",
        f"- Latest period gate passed: {int(summary.get('latest_period_gate_passed', 0) or 0)}",
        (
            "- Gate: "
            f"closed >= {int(summary.get('min_closed_trades', 0) or 0)}, "
            f"PnL >= {_fmt_pct(summary.get('min_pnl_pct'))}, "
            f"DD <= {_fmt_pct(summary.get('max_drawdown_pct'))}"
        ),
        "",
        "## Labels",
    ]
    if not isinstance(rows, list) or not rows:
        lines.append("- none")
        return lines
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        gate = "promote" if row.get("promote_to_real") else "hold"
        lines.append(
            f"- `{row.get('label')}`: shadow PnL "
            f"{_fmt_pct(row.get('shadow_pnl_pct'))}, "
            f"DD {_fmt_pct(row.get('max_drawdown_pct'))}, "
            f"closed {int(row.get('closed_trades', 0) or 0)}, "
            f"real selected {int(row.get('real_selected_signals', 0) or 0)}, "
            f"{gate}"
        )
    return lines


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _new_flash_attribution_bucket(
    *,
    actor_key: str,
    actor_label: str,
    actor_type: str,
    symbol: str,
    action: str,
    regime: str = "",
) -> dict[str, Any]:
    return {
        "actor_key": actor_key,
        "actor_label": actor_label,
        "actor_type": actor_type,
        "symbol": symbol,
        "action": action,
        "regime": regime,
        "selected_signals": 0,
        "executable_selected_signals": 0,
        "selected_filtered_before_execution": 0,
        "filled_signals": 0,
        "blocked_signals": 0,
        "rejected_signals": 0,
        "pending_signals": 0,
        "closed_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "realized_pnl_usd": 0.0,
        "score_sum": 0.0,
        "shadow_score_sum": 0.0,
        "shadow_closed_trades_sum": 0.0,
        "shadow_pnl_per_trade_lcb_usd_sum": 0.0,
        "shadow_symbol_health_lcb_usd_sum": 0.0,
        "shadow_symbol_health_penalty_sum": 0.0,
        "selected_subset_score_boost_sum": 0.0,
        "selected_subset_protected_count": 0,
        "selected_subset_risk_mult_sum": 0.0,
        "risk_mult_sum": 0.0,
        "failure_reasons": {},
        "filter_detail_counts": {},
    }


def _accumulate_flash_selected_bucket(
    bucket: dict[str, Any],
    decision: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> None:
    bucket["selected_signals"] += 1
    bucket["score_sum"] += _safe_float(decision.get("score"))
    if selected_candidate:
        bucket["shadow_score_sum"] += _safe_float(selected_candidate.get("shadow_score"))
        bucket["shadow_closed_trades_sum"] += _safe_float(
            selected_candidate.get("shadow_closed_trades")
        )
        bucket["shadow_pnl_per_trade_lcb_usd_sum"] += _safe_float(
            selected_candidate.get("shadow_pnl_per_trade_lcb_usd")
        )
        bucket["shadow_symbol_health_lcb_usd_sum"] += _safe_float(
            selected_candidate.get("shadow_symbol_health_pnl_per_trade_lcb_usd")
        )
        bucket["shadow_symbol_health_penalty_sum"] += _safe_float(
            selected_candidate.get("shadow_symbol_health_penalty")
        )
        bucket["selected_subset_score_boost_sum"] += _safe_float(
            selected_candidate.get("selected_subset_score_boost")
        )
        bucket["selected_subset_protected_count"] += int(
            bool(selected_candidate.get("selected_subset_protected"))
        )
        bucket["selected_subset_risk_mult_sum"] += _safe_float(
            selected_candidate.get("selected_subset_risk_mult", 1.0)
        )
        risk_mult = _finite_float_or_none(selected_candidate.get("risk_mult"))
        bucket["risk_mult_sum"] += risk_mult if risk_mult is not None else 1.0
    else:
        bucket["selected_subset_risk_mult_sum"] += 1.0
        bucket["risk_mult_sum"] += 1.0


def _accumulate_soft_allocator_selected_bucket(
    bucket: dict[str, Any],
    signal: Mapping[str, Any],
) -> None:
    bucket["selected_signals"] += 1
    bucket["executable_selected_signals"] += 1
    risk_mult = _finite_float_or_none(signal.get("risk_mult"))
    if risk_mult is None:
        risk_mult = 1.0
    bucket["risk_mult_sum"] += risk_mult
    bucket["selected_subset_risk_mult_sum"] += risk_mult


def _flash_context_regime(raw: object) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "bull": "bullish",
        "bear": "bearish",
        "flat": "neutral",
        "range": "neutral",
        "sideways": "neutral",
        "panic": "crash",
        "flash_crash": "crash",
    }
    return aliases.get(value, value) or "unknown"


def _flash_attribution_payload(row: dict[str, Any]) -> dict[str, Any]:
    selected = max(1, int(row.get("selected_signals", 0) or 0))
    failure_reasons = row.get("failure_reasons", {})
    if not isinstance(failure_reasons, dict):
        failure_reasons = {}
    filter_detail_counts = row.get("filter_detail_counts", {})
    if not isinstance(filter_detail_counts, dict):
        filter_detail_counts = {}
    payload = {
        "actor_key": row["actor_key"],
        "actor_label": row["actor_label"],
        "actor_type": row["actor_type"],
        "symbol": row["symbol"],
        "action": row["action"],
        "selected_signals": int(row.get("selected_signals", 0) or 0),
        "executable_selected_signals": int(
            row.get("executable_selected_signals", 0) or 0
        ),
        "selected_filtered_before_execution": int(
            row.get("selected_filtered_before_execution", 0) or 0
        ),
        "filled_signals": int(row.get("filled_signals", 0) or 0),
        "blocked_signals": int(row.get("blocked_signals", 0) or 0),
        "rejected_signals": int(row.get("rejected_signals", 0) or 0),
        "pending_signals": int(row.get("pending_signals", 0) or 0),
        "closed_trades": int(row.get("closed_trades", 0) or 0),
        "winning_trades": int(row.get("winning_trades", 0) or 0),
        "losing_trades": int(row.get("losing_trades", 0) or 0),
        "realized_pnl_usd": float(row.get("realized_pnl_usd", 0.0) or 0.0),
        "avg_score": float(row.get("score_sum", 0.0) or 0.0) / selected,
        "avg_shadow_score": float(row.get("shadow_score_sum", 0.0) or 0.0) / selected,
        "avg_shadow_closed_trades": (
            float(row.get("shadow_closed_trades_sum", 0.0) or 0.0) / selected
        ),
        "avg_shadow_pnl_per_trade_lcb_usd": (
            float(row.get("shadow_pnl_per_trade_lcb_usd_sum", 0.0) or 0.0)
            / selected
        ),
        "avg_shadow_symbol_health_lcb_usd": (
            float(row.get("shadow_symbol_health_lcb_usd_sum", 0.0) or 0.0)
            / selected
        ),
        "avg_shadow_symbol_health_penalty": (
            float(row.get("shadow_symbol_health_penalty_sum", 0.0) or 0.0)
            / selected
        ),
        "avg_selected_subset_score_boost": (
            float(row.get("selected_subset_score_boost_sum", 0.0) or 0.0)
            / selected
        ),
        "selected_subset_protected_count": int(
            row.get("selected_subset_protected_count", 0) or 0
        ),
        "avg_selected_subset_risk_mult": (
            float(row.get("selected_subset_risk_mult_sum", 0.0) or 0.0)
            / selected
        ),
        "avg_risk_mult": float(row.get("risk_mult_sum", 0.0) or 0.0) / selected,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "filter_detail_counts": dict(sorted(filter_detail_counts.items())),
    }
    if row.get("regime"):
        payload["regime"] = str(row.get("regime") or "")
        payload["context_key"] = (
            f"{payload['actor_key']}|{payload['symbol']}|"
            f"{payload['action']}|{payload['regime']}"
        )
    return payload


def _flash_actor_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        actor_key = str(row.get("actor_key") or "")
        if not actor_key:
            continue
        bucket = summary.setdefault(actor_key, {
            "actor_label": str(row.get("actor_label") or ""),
            "actor_type": str(row.get("actor_type") or ""),
            "selected_signals": 0,
            "executable_selected_signals": 0,
            "selected_filtered_before_execution": 0,
            "filled_signals": 0,
            "closed_trades": 0,
            "realized_pnl_usd": 0.0,
        })
        bucket["selected_signals"] += int(row.get("selected_signals", 0) or 0)
        bucket["executable_selected_signals"] += int(
            row.get("executable_selected_signals", 0) or 0
        )
        bucket["selected_filtered_before_execution"] += int(
            row.get("selected_filtered_before_execution", 0) or 0
        )
        bucket["filled_signals"] += int(row.get("filled_signals", 0) or 0)
        bucket["closed_trades"] += int(row.get("closed_trades", 0) or 0)
        bucket["realized_pnl_usd"] += float(row.get("realized_pnl_usd", 0.0) or 0.0)
    return dict(sorted(
        summary.items(),
        key=lambda item: (-float(item[1]["realized_pnl_usd"]), item[0]),
    ))


def write_flash_signal_key_shadow_report(
    output: Path,
    *,
    shadow_updates: Sequence[object],
    initial_capital: float,
    promotion_manifest_config: PromotionManifestConfig = PromotionManifestConfig(),
    filename: str = "flash_signal_key_shadow_report.json",
) -> Path:
    output_dir = Path(output)
    capital = _validate_component_benchmark_capital(initial_capital)
    events = _flash_signal_key_shadow_events(shadow_updates)
    latest_period = _latest_flash_signal_key_period(events)
    buckets: dict[str, dict[str, Any]] = {}

    for event in events:
        signal_key = str(event["signal_key"])
        bucket = buckets.setdefault(
            signal_key,
            {
                "signal_key": signal_key,
                "actor_key": event["actor_key"],
                "actor_label": event["actor_label"],
                "actor_type": event["actor_type"],
                "source_actor_type": event.get("source_actor_type", event["actor_type"]),
                "symbol": event["symbol"],
                "action": event["action"],
                "full_pnl_usd": 0.0,
                "full_closed_trades": 0,
                "wins": 0,
                "latest_pnl_usd": 0.0,
                "latest_closed_trades": 0,
                "recent_downside_usd": 0.0,
                "cumulative_pnl_usd": 0.0,
                "peak_cumulative_pnl_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "full_trade_pnl_pct_sum": 0.0,
                "full_trade_pnl_pct_sumsq": 0.0,
                "latest_trade_pnl_pct_sum": 0.0,
                "latest_trade_pnl_pct_sumsq": 0.0,
            },
        )
        pnl = float(event["pnl_usd"])
        closed = int(event["closed_trades"])
        wins = int(event["wins"])
        bucket["full_pnl_usd"] += pnl
        bucket["full_closed_trades"] += closed
        bucket["wins"] += wins
        bucket["cumulative_pnl_usd"] += pnl
        bucket["peak_cumulative_pnl_usd"] = max(
            float(bucket["peak_cumulative_pnl_usd"]),
            float(bucket["cumulative_pnl_usd"]),
        )
        drawdown_usd = (
            float(bucket["peak_cumulative_pnl_usd"])
            - float(bucket["cumulative_pnl_usd"])
        )
        bucket["max_drawdown_pct"] = max(
            float(bucket["max_drawdown_pct"]),
            drawdown_usd / capital * 100.0,
        )
        if closed > 0:
            pnl_per_trade_pct = (pnl / capital * 100.0) / float(closed)
            bucket["full_trade_pnl_pct_sum"] += pnl_per_trade_pct * closed
            bucket["full_trade_pnl_pct_sumsq"] += (
                pnl_per_trade_pct * pnl_per_trade_pct * closed
            )
        if latest_period == "all" or event["period"] == latest_period:
            bucket["latest_pnl_usd"] += pnl
            bucket["latest_closed_trades"] += closed
            if closed > 0:
                bucket["latest_trade_pnl_pct_sum"] += pnl_per_trade_pct * closed
                bucket["latest_trade_pnl_pct_sumsq"] += (
                    pnl_per_trade_pct * pnl_per_trade_pct * closed
                )
            if pnl < 0.0:
                bucket["recent_downside_usd"] += abs(pnl)

    rows = [
        _flash_signal_key_shadow_payload(
            bucket,
            initial_capital=capital,
            pnl_per_trade_lcb_z=promotion_manifest_config.pnl_per_trade_lcb_z,
        )
        for bucket in buckets.values()
    ]
    rows.sort(key=lambda row: (-float(row["full_pnl_pct"]), str(row["signal_key"])))
    data = {
        "summary": {
            "signal_key_count": len(rows),
            "positive_signal_keys": sum(
                1 for row in rows if float(row["full_pnl_usd"]) > 0.0
            ),
            "latest_period": latest_period,
        },
        "rows": rows,
    }
    path = output_dir / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / Path(filename).with_suffix(".md").name).write_text(
        "\n".join(_flash_signal_key_shadow_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    _write_flash_promotion_manifest(
        output_dir,
        rows,
        config=promotion_manifest_config,
    )
    return path


def write_flash_genetics_intersection_report(
    output: str | Path,
    *,
    filename: str = "flash_genetics_intersection_report.json",
) -> Path:
    output_dir = Path(output)
    shadow_report = _load_json(output_dir / "flash_signal_key_shadow_report.json")
    attribution = _load_json(output_dir / "flash_attribution_summary.json")
    attribution_by_key = _flash_attribution_rows_by_signal_key(attribution)
    genetics_by_symbol = _genetics_shadow_rows_by_symbol(shadow_report)
    rows: list[dict[str, Any]] = []
    selected_open_signals = 0

    for causal_row in _iter_jsonl(output_dir / "causal_entry_decisions.jsonl"):
        bar = _safe_int(causal_row.get("bar"), default=0)
        regime = str(causal_row.get("regime") or "")
        for decision in causal_row.get("flash_decisions") or ():
            if not isinstance(decision, dict):
                continue
            if str(decision.get("selected_actor") or "") == "NoTrade":
                continue
            signal = decision.get("signal")
            if not isinstance(signal, dict):
                continue
            symbol = str(decision.get("symbol") or signal.get("sym") or "").upper()
            action = _normalize_flash_signal_action(
                decision.get("action") or signal.get("action")
            )
            selected_side = _flash_open_side(action)
            if not symbol or not selected_side:
                continue
            selected_open_signals += 1
            actor_key = _flash_decision_actor_key(decision)
            selected_signal_key = f"{actor_key}|{symbol}|{action}"
            selected_attr = attribution_by_key.get(selected_signal_key, {})

            for genetics in genetics_by_symbol.get(symbol, ()):
                genetics_action = str(genetics.get("action") or "").upper()
                genetics_side = _flash_open_side(genetics_action)
                if not genetics_side:
                    continue
                rows.append({
                    "bar": bar,
                    "regime": regime,
                    "symbol": symbol,
                    "selected_signal_key": selected_signal_key,
                    "selected_actor_key": actor_key,
                    "selected_actor_label": str(decision.get("selected_actor") or ""),
                    "selected_actor_type": str(decision.get("actor_type") or ""),
                    "selected_action": action,
                    "selected_side": selected_side,
                    "selected_score": _safe_float(decision.get("score")),
                    "selected_realized_pnl_usd": _safe_float(
                        selected_attr.get("realized_pnl_usd")
                    ),
                    "selected_closed_trades": _safe_int(
                        selected_attr.get("closed_trades")
                    ),
                    "relationship": _flash_action_relationship(
                        selected_action=action,
                        selected_side=selected_side,
                        genetics_action=genetics_action,
                        genetics_side=genetics_side,
                    ),
                    "genetics_signal_key": str(genetics.get("signal_key") or ""),
                    "genetics_actor_key": str(genetics.get("actor_key") or ""),
                    "genetics_actor_label": str(genetics.get("actor_label") or ""),
                    "genetics_action": genetics_action,
                    "genetics_side": genetics_side,
                    "genetics_full_closed_trades": _safe_int(
                        genetics.get("full_closed_trades")
                    ),
                    "genetics_full_pnl_pct": _safe_float(
                        genetics.get("full_pnl_pct")
                    ),
                    "genetics_full_pnl_per_trade_lcb_pct": _safe_float(
                        genetics.get("full_pnl_per_trade_lcb_pct")
                    ),
                    "genetics_latest_closed_trades": _safe_int(
                        genetics.get("latest_closed_trades")
                    ),
                    "genetics_latest_pnl_pct": _safe_float(
                        genetics.get("latest_pnl_pct")
                    ),
                    "genetics_latest_pnl_per_trade_lcb_pct": _safe_float(
                        genetics.get("latest_pnl_per_trade_lcb_pct")
                    ),
                    "genetics_max_drawdown_pct": _safe_float(
                        genetics.get("max_drawdown_pct")
                    ),
                    "genetics_recent_downside_usd": _safe_float(
                        genetics.get("recent_downside_usd")
                    ),
                })

    rows.sort(key=_flash_genetics_intersection_sort_key)
    direct_genetics = _flash_direct_genetics_selection_summary(attribution)
    summary = _flash_genetics_intersection_summary(
        rows,
        selected_open_signals=selected_open_signals,
    )
    summary.update(direct_genetics["summary"])
    data = {
        "summary": summary,
        "direct_genetics_actors": direct_genetics["actors"],
        "rows": rows,
    }
    path = output_dir / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / Path(filename).with_suffix(".md").name).write_text(
        "\n".join(_flash_genetics_intersection_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path


def _flash_attribution_rows_by_signal_key(
    attribution: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    rows = attribution.get("rows") if isinstance(attribution, Mapping) else ()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        actor_key = str(row.get("actor_key") or "")
        symbol = str(row.get("symbol") or "").upper()
        action = _normalize_flash_signal_action(row.get("action"))
        if not actor_key or not symbol or not action:
            continue
        out[f"{actor_key}|{symbol}|{action}"] = row
    return out


def _genetics_shadow_rows_by_symbol(
    shadow_report: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    rows = shadow_report.get("rows") if isinstance(shadow_report, Mapping) else ()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        actor_key = str(row.get("actor_key") or "")
        actor_label = str(row.get("actor_label") or "")
        if not _is_genetics_shadow_actor(actor_key, actor_label):
            continue
        symbol = str(row.get("symbol") or "").upper()
        action = _normalize_flash_signal_action(row.get("action"))
        if not symbol or not _flash_open_side(action):
            continue
        enriched = dict(row)
        enriched["symbol"] = symbol
        enriched["action"] = action
        out.setdefault(symbol, []).append(enriched)
    return out


def _is_genetics_shadow_actor(actor_key: str, actor_label: str) -> bool:
    return (
        actor_key.startswith("agent:Genetics")
        or actor_key.startswith("ensemble:Genetics")
        or actor_label.startswith("Genetics")
        or actor_label.startswith("Solo_Genetics")
    )


def _flash_open_side(action: str) -> str:
    clean = str(action or "").upper()
    if clean.startswith("FUT_LONG") or clean.startswith("SPOT_BUY"):
        return "long"
    if clean.startswith("FUT_SHORT"):
        return "short"
    return ""


def _flash_action_relationship(
    *,
    selected_action: str,
    selected_side: str,
    genetics_action: str,
    genetics_side: str,
) -> str:
    if genetics_action == selected_action:
        return "same_action"
    if genetics_side == selected_side:
        return "same_side"
    return "opposite_side"


def _flash_genetics_intersection_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_open_signals: int,
) -> dict[str, Any]:
    same_action = [
        row for row in rows if str(row.get("relationship") or "") == "same_action"
    ]
    return {
        "selected_open_signals": int(selected_open_signals),
        "intersection_rows": len(rows),
        "same_action_rows": len(same_action),
        "same_side_rows": sum(
            1 for row in rows if str(row.get("relationship") or "") == "same_side"
        ),
        "opposite_side_rows": sum(
            1
            for row in rows
            if str(row.get("relationship") or "") == "opposite_side"
        ),
        "negative_same_action_rows": sum(
            1
            for row in same_action
            if _safe_float(row.get("genetics_full_pnl_per_trade_lcb_pct")) < 0.0
        ),
        "positive_same_action_rows": sum(
            1
            for row in same_action
            if _safe_float(row.get("genetics_full_pnl_per_trade_lcb_pct")) > 0.0
        ),
    }


def _flash_direct_genetics_selection_summary(
    attribution: Mapping[str, Any],
) -> dict[str, Any]:
    actor_summary = attribution.get("actor_summary") if isinstance(attribution, Mapping) else {}
    if not isinstance(actor_summary, Mapping):
        actor_summary = {}

    actors: list[dict[str, Any]] = []
    for actor_key, raw_actor in sorted(actor_summary.items()):
        if not isinstance(raw_actor, Mapping):
            continue
        actor_label = str(raw_actor.get("actor_label") or "")
        if not _is_genetics_shadow_actor(str(actor_key), actor_label):
            continue
        actors.append({
            "actor_key": str(actor_key),
            "actor_label": actor_label,
            "actor_type": str(raw_actor.get("actor_type") or ""),
            "selected_signals": _safe_int(raw_actor.get("selected_signals")),
            "executable_selected_signals": _safe_int(
                raw_actor.get("executable_selected_signals")
            ),
            "filled_signals": _safe_int(raw_actor.get("filled_signals")),
            "closed_trades": _safe_int(raw_actor.get("closed_trades")),
            "realized_pnl_usd": _safe_float(raw_actor.get("realized_pnl_usd")),
        })

    selected = sum(int(row["selected_signals"]) for row in actors)
    executable = sum(int(row["executable_selected_signals"]) for row in actors)
    filled = sum(int(row["filled_signals"]) for row in actors)
    closed = sum(int(row["closed_trades"]) for row in actors)
    pnl = sum(float(row["realized_pnl_usd"]) for row in actors)
    return {
        "summary": {
            "direct_genetics_actor_count": len(actors),
            "direct_genetics_selected_signals": selected,
            "direct_genetics_executable_selected_signals": executable,
            "direct_genetics_filled_signals": filled,
            "direct_genetics_closed_trades": closed,
            "direct_genetics_realized_pnl_usd": pnl,
            "direct_genetics_selection_gate_passed": selected > 0 and closed > 0,
        },
        "actors": actors,
    }


def _flash_genetics_intersection_sort_key(row: Mapping[str, Any]) -> tuple:
    relationship = str(row.get("relationship") or "")
    priority = {"same_action": 0, "same_side": 1, "opposite_side": 2}.get(
        relationship,
        9,
    )
    return (
        priority,
        _safe_float(row.get("genetics_full_pnl_per_trade_lcb_pct")),
        _safe_int(row.get("bar")),
        str(row.get("selected_signal_key") or ""),
        str(row.get("genetics_signal_key") or ""),
    )


def _flash_genetics_intersection_report_lines(data: Mapping[str, Any]) -> list[str]:
    summary = data.get("summary", {}) if isinstance(data, Mapping) else {}
    rows = data.get("rows", []) if isinstance(data, Mapping) else []
    lines = [
        "# Flash Genetics Intersection Report",
        "",
        f"- Selected open signals: {int(summary.get('selected_open_signals', 0) or 0)}",
        f"- Intersection rows: {int(summary.get('intersection_rows', 0) or 0)}",
        f"- Same-action rows: {int(summary.get('same_action_rows', 0) or 0)}",
        f"- Negative same-action rows: {int(summary.get('negative_same_action_rows', 0) or 0)}",
        f"- Direct genetics selected signals: {int(summary.get('direct_genetics_selected_signals', 0) or 0)}",
        f"- Direct genetics closed trades: {int(summary.get('direct_genetics_closed_trades', 0) or 0)}",
        f"- Direct genetics gate passed: {bool(summary.get('direct_genetics_selection_gate_passed', False))}",
        "",
        "| selected | genetics | relation | selected pnl | genetics LCB |",
        "|---|---|---:|---:|---:|",
    ]
    for row in list(rows)[:25]:
        lines.append(
            "| "
            f"`{row.get('selected_signal_key', '')}` | "
            f"`{row.get('genetics_signal_key', '')}` | "
            f"{row.get('relationship', '')} | "
            f"{_fmt_money(row.get('selected_realized_pnl_usd'))} | "
            f"{_fmt_pct(row.get('genetics_full_pnl_per_trade_lcb_pct'))} |"
        )
    return lines


def _flash_signal_key_shadow_events(
    shadow_updates: Sequence[object],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for update in sorted(
        shadow_updates or (),
        key=lambda event: (
            _safe_int(_event_value(event, "bar"), default=0),
            str(_event_value(event, "timestamp") or ""),
            str(_event_value(event, "actor_type") or ""),
            str(_event_value(event, "actor_label") or ""),
        ),
    ):
        actor_type = str(_event_value(update, "actor_type") or "").strip()
        actor_label = str(_event_value(update, "actor_label") or "").strip()
        if not actor_type or not actor_label:
            continue
        source_actor_type = actor_type
        allocator_actor_type = "ensemble" if actor_type == "player" else actor_type
        actor_key = f"{allocator_actor_type}:{actor_label}"
        period = _shadow_event_half_year_period(update)
        outcomes = _event_value(update, "symbol_action_outcomes")
        if not isinstance(outcomes, (list, tuple)):
            continue
        for outcome in outcomes:
            if not isinstance(outcome, (list, tuple)) or len(outcome) < 5:
                continue
            symbol = str(outcome[0] or "").strip()
            if not symbol:
                continue
            action = _normalize_flash_signal_action(outcome[1])
            pnl_usd = _finite_float_or_none(outcome[2])
            if pnl_usd is None:
                continue
            signal_key = f"{actor_key}|{symbol}|{action}"
            events.append({
                "signal_key": signal_key,
                "actor_key": actor_key,
                "actor_label": actor_label,
                "actor_type": allocator_actor_type,
                "source_actor_type": source_actor_type,
                "symbol": symbol,
                "action": action,
                "pnl_usd": pnl_usd,
                "closed_trades": max(0, _safe_int(outcome[3])),
                "wins": max(0, _safe_int(outcome[4])),
                "period": period,
            })
    return events


def _normalize_flash_signal_action(value: object) -> str:
    action = str(value or "").strip().upper()
    if action == "LONG":
        return "FUT_LONG_FULL"
    if action == "SHORT":
        return "FUT_SHORT_FULL"
    return action


def _latest_flash_signal_key_period(events: Sequence[dict[str, Any]]) -> str:
    periods = sorted(
        str(event.get("period") or "")
        for event in events
        if str(event.get("period") or "") and str(event.get("period") or "") != "unknown"
    )
    return periods[-1] if periods else "all"


def _flash_signal_key_shadow_payload(
    bucket: dict[str, Any],
    *,
    initial_capital: float,
    pnl_per_trade_lcb_z: float = Z_95_ONE_SIDED,
) -> dict[str, Any]:
    full_pnl_usd = float(bucket.get("full_pnl_usd", 0.0) or 0.0)
    latest_pnl_usd = float(bucket.get("latest_pnl_usd", 0.0) or 0.0)
    full_closed = int(bucket.get("full_closed_trades", 0) or 0)
    latest_closed = int(bucket.get("latest_closed_trades", 0) or 0)
    wins = int(bucket.get("wins", 0) or 0)
    full_pnl_per_trade = _pnl_per_trade_lcb_payload(
        count=full_closed,
        total=float(bucket.get("full_trade_pnl_pct_sum", 0.0) or 0.0),
        sumsq=float(bucket.get("full_trade_pnl_pct_sumsq", 0.0) or 0.0),
        z=pnl_per_trade_lcb_z,
    )
    latest_pnl_per_trade = _pnl_per_trade_lcb_payload(
        count=latest_closed,
        total=float(bucket.get("latest_trade_pnl_pct_sum", 0.0) or 0.0),
        sumsq=float(bucket.get("latest_trade_pnl_pct_sumsq", 0.0) or 0.0),
        z=pnl_per_trade_lcb_z,
    )
    return {
        "signal_key": str(bucket.get("signal_key") or ""),
        "actor_key": str(bucket.get("actor_key") or ""),
        "actor_label": str(bucket.get("actor_label") or ""),
        "actor_type": str(bucket.get("actor_type") or ""),
        "source_actor_type": str(bucket.get("source_actor_type") or ""),
        "symbol": str(bucket.get("symbol") or ""),
        "action": str(bucket.get("action") or ""),
        "full_pnl_usd": full_pnl_usd,
        "full_pnl_pct": full_pnl_usd / initial_capital * 100.0,
        "full_closed_trades": full_closed,
        "wins": wins,
        "win_rate_pct": _share(wins, full_closed),
        "full_pnl_per_trade_mean_pct": full_pnl_per_trade["mean_pct"],
        "full_pnl_per_trade_lcb_pct": full_pnl_per_trade["lcb_pct"],
        "latest_pnl_usd": latest_pnl_usd,
        "latest_pnl_pct": latest_pnl_usd / initial_capital * 100.0,
        "latest_closed_trades": latest_closed,
        "latest_pnl_per_trade_mean_pct": latest_pnl_per_trade["mean_pct"],
        "latest_pnl_per_trade_lcb_pct": latest_pnl_per_trade["lcb_pct"],
        "max_drawdown_pct": float(bucket.get("max_drawdown_pct", 0.0) or 0.0),
        "recent_downside_usd": float(bucket.get("recent_downside_usd", 0.0) or 0.0),
    }


def _pnl_per_trade_lcb_payload(
    *,
    count: int,
    total: float,
    sumsq: float,
    z: float,
) -> dict[str, float]:
    if count <= 0:
        return {"mean_pct": 0.0, "lcb_pct": 0.0}
    n = float(count)
    mean = float(total) / n
    if count <= 1:
        return {"mean_pct": mean, "lcb_pct": mean}
    variance = max(0.0, (float(sumsq) - (float(total) * float(total) / n)) / (n - 1.0))
    std = math.sqrt(variance)
    z_value = max(0.0, float(z))
    return {
        "mean_pct": mean,
        "lcb_pct": mean - z_value * std / math.sqrt(n),
    }


def _flash_signal_key_shadow_report_lines(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    rows = report.get("rows", []) if isinstance(report, dict) else []
    lines = [
        "# Flash Signal-Key Shadow Report",
        "",
        f"- Signal keys: {int(summary.get('signal_key_count', 0) or 0)}",
        f"- Positive signal keys: {int(summary.get('positive_signal_keys', 0) or 0)}",
        f"- Latest period: `{summary.get('latest_period') or '-'}`",
        "",
        "## Top Signal Keys",
    ]
    if not isinstance(rows, list) or not rows:
        lines.append("- none")
        return lines
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('signal_key')}`: "
            f"{_fmt_pct(row.get('full_pnl_pct'))}, "
            f"latest {_fmt_pct(row.get('latest_pnl_pct'))}, "
            f"trade LCB {_fmt_pct(row.get('full_pnl_per_trade_lcb_pct'))}, "
            f"closed {int(row.get('full_closed_trades', 0) or 0)}, "
            f"win rate {_fmt_pct(row.get('win_rate_pct'))}"
        )
    return lines


def _write_flash_promotion_manifest(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    *,
    config: PromotionManifestConfig = PromotionManifestConfig(),
) -> Path:
    manifest = build_promotion_manifest(rows, config)
    data = manifest.as_dict()
    path = output_dir / "flash_promotion_manifest.json"
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "flash_promotion_manifest.md").write_text(
        "\n".join(_flash_promotion_manifest_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path


def _flash_promotion_manifest_config(
    config: RetrodateMarketConfig,
) -> PromotionManifestConfig:
    return PromotionManifestConfig(
        min_full_closed_trades=config.flash_promotion_min_full_closed_trades,
        min_latest_closed_trades=config.flash_promotion_min_latest_closed_trades,
        min_full_pnl_pct=config.flash_promotion_min_full_pnl_pct,
        min_latest_pnl_pct=config.flash_promotion_min_latest_pnl_pct,
        min_full_pnl_per_trade_lcb_pct=(
            config.flash_promotion_min_full_pnl_per_trade_lcb_pct
        ),
        min_latest_pnl_per_trade_lcb_pct=(
            config.flash_promotion_min_latest_pnl_per_trade_lcb_pct
        ),
        pnl_per_trade_lcb_z=config.flash_promotion_pnl_per_trade_lcb_z,
        max_drawdown_pct=config.flash_promotion_max_drawdown_pct,
        min_win_rate_pct=config.flash_promotion_min_win_rate_pct,
        max_recent_downside_usd=config.flash_promotion_max_recent_downside_usd,
    )


def _flash_promotion_manifest_lines(data: dict[str, Any]) -> list[str]:
    allowed = data.get("allowed_signal_keys", [])
    rejected = data.get("rejected", [])
    lines = [
        "# Flash Promotion Manifest",
        "",
        f"- Allowed signal keys: {len(allowed) if isinstance(allowed, list) else 0}",
        f"- Rejected signal keys: {len(rejected) if isinstance(rejected, list) else 0}",
        "",
        "## Allowed",
    ]
    if isinstance(allowed, list) and allowed:
        lines.extend(f"- `{key}`" for key in allowed[:50])
    else:
        lines.append("- none")
    lines.extend(["", "## Rejected"])
    if isinstance(rejected, list) and rejected:
        for row in rejected[:50]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('signal_key')}`: {row.get('reason') or '-'}"
            )
    else:
        lines.append("- none")
    return lines


def _flash_decision_actor_key(decision: Mapping[str, Any]) -> str:
    selected_actor = str(decision.get("selected_actor") or "")
    actor_type = str(decision.get("actor_type") or "unknown") or "unknown"
    candidates = decision.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if (
                str(candidate.get("label") or "") == selected_actor
                and str(candidate.get("actor_type") or "") == actor_type
            ):
                actor_key = str(candidate.get("actor_key") or "")
                if actor_key:
                    return actor_key
    if selected_actor == "NoTrade":
        return "NoTrade"
    return f"{actor_type}:{selected_actor}"


def _flash_selected_candidate(decision: Mapping[str, Any]) -> dict[str, Any]:
    selected_actor = str(decision.get("selected_actor") or "")
    actor_type = str(decision.get("actor_type") or "")
    candidates = decision.get("candidates")
    if not isinstance(candidates, list):
        return {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if (
            str(candidate.get("label") or "") == selected_actor
            and str(candidate.get("actor_type") or "") == actor_type
        ):
            return candidate
    return {}


def _signal_ids_from_payloads(payloads: object) -> set[int]:
    if not isinstance(payloads, list):
        return set()
    out: set[int] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        signal_id = _safe_int(payload.get("id"), default=-1)
        if signal_id >= 0:
            out.add(signal_id)
    return out


def _flash_filter_detail_counts(details: object) -> dict[str, int]:
    if not isinstance(details, (list, tuple)):
        return {}
    counts: dict[str, int] = {}
    for item in details:
        reason = str(item or "").split(":", 1)[0].strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _merge_int_counts(target: object, updates: Mapping[str, int]) -> None:
    if not isinstance(target, dict):
        return
    for key, value in updates.items():
        clean_key = str(key or "").strip()
        count = int(value or 0)
        if not clean_key or count <= 0:
            continue
        target[clean_key] = int(target.get(clean_key, 0) or 0) + count


def _increment_flash_failure(bucket: dict[str, Any], event: object) -> None:
    reason = str(_event_value(event, "reason") or "").strip()
    if not reason:
        return
    reasons = bucket.setdefault("failure_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
        bucket["failure_reasons"] = reasons
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1


def write_oracle_mismatch_report(
    output_dir: str | Path,
    trading_log: str | Path,
    *,
    shadow_updates: Sequence[object],
    candidate_rejections: Sequence[object],
    filename: str = "oracle_mismatch_report.json",
) -> Path:
    rows = _parse_trading_log_rows(Path(trading_log))
    player_updates = [
        event for event in shadow_updates or ()
        if str(_event_value(event, "actor_type") or "") == "player"
    ]
    current_shadow = _shadow_update_index(player_updates)
    perfect_by_bar = _perfect_monthly_leader_by_bar(player_updates)
    soft_by_bar = _soft_confirmed_leader_by_bar(player_updates)
    rejections = _rejection_index(candidate_rejections)
    out_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for row in rows:
        bar = int(row.get("bar", 0) or 0)
        if bar <= 0:
            continue
        real_leader = str(
            row.get("executed_leader")
            or row.get("selected_leader")
            or row.get("leader")
            or ""
        )
        perfect_leader = perfect_by_bar.get(bar, "")
        soft_leader = soft_by_bar.get(bar, "")
        target = perfect_leader if perfect_leader and perfect_leader != "CASH" else soft_leader
        if not target or target == real_leader:
            continue
        reason = _oracle_mismatch_reason(
            bar=bar,
            target_label=target,
            trading_row=row,
            current_shadow=current_shadow,
            rejections=rejections,
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        out_rows.append({
            "bar": bar,
            "regime": row.get("regime", ""),
            "real_leader": real_leader,
            "selected_leader": row.get("selected_leader", ""),
            "executed_leader": row.get("executed_leader", ""),
            "perfect_leader": perfect_leader,
            "soft_leader": soft_leader,
            "reason": reason,
            "raw_signals": int(row.get("raw_signals", 0) or 0),
            "signals": int(row.get("signals", 0) or 0),
            "filled": int(row.get("filled", 0) or 0),
        })
    data = {
        "summary": {
            "trading_rows": len(rows),
            "mismatch_rows": len(out_rows),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "rows": out_rows,
    }
    path = Path(output_dir) / filename
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return path


def _parse_trading_log_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields: dict[str, Any] = {}
        for part in line.split("  "):
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {
                "bar",
                "raw_signals",
                "signals",
                "filled",
                "rejected",
                "blocked",
            }:
                try:
                    fields[key] = int(value)
                except ValueError:
                    fields[key] = 0
            else:
                fields[key] = value
        if fields:
            rows.append(fields)
    return rows


def _shadow_update_index(
    updates: Sequence[object],
) -> dict[tuple[int, str], dict[str, float]]:
    index: dict[tuple[int, str], dict[str, float]] = {}
    for event in updates or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(_event_value(event, "actor_label") or "").strip()
        if bar <= 0 or not label:
            continue
        row = index.setdefault((bar, label), {
            "signals": 0.0,
            "filled": 0.0,
            "pnl_usd": 0.0,
            "closed_trades": 0.0,
        })
        row["signals"] += _safe_float(_event_value(event, "signals"))
        row["filled"] += _safe_float(_event_value(event, "filled"))
        row["pnl_usd"] += _safe_float(_event_value(event, "realized_pnl_usd"))
        row["closed_trades"] += _safe_float(_event_value(event, "closed_trades"))
    return index


def _perfect_monthly_leader_by_bar(updates: Sequence[object]) -> dict[int, str]:
    month_label_pnl: dict[str, dict[str, float]] = {}
    month_by_bar: dict[int, str] = {}
    for event in updates or ():
        month = _event_month_from_update(event)
        label = str(_event_value(event, "actor_label") or "").strip()
        bar = int(_event_value(event, "bar") or 0)
        if not month or not label or bar <= 0:
            continue
        month_by_bar[bar] = month
        bucket = month_label_pnl.setdefault(month, {})
        bucket[label] = bucket.get(label, 0.0) + _safe_float(
            _event_value(event, "realized_pnl_usd")
        )
    best_by_month: dict[str, str] = {}
    for month, pnl_by_label in month_label_pnl.items():
        if not pnl_by_label:
            continue
        best_label = max(pnl_by_label, key=lambda label: (pnl_by_label[label], label))
        best_by_month[month] = best_label if pnl_by_label[best_label] > 0.0 else "CASH"
    return {
        bar: best_by_month.get(month, "")
        for bar, month in month_by_bar.items()
    }


def _soft_confirmed_leader_by_bar(updates: Sequence[object]) -> dict[int, str]:
    grouped: dict[int, list[object]] = {}
    for event in updates or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(_event_value(event, "actor_label") or "").strip()
        if bar > 0 and label:
            grouped.setdefault(bar, []).append(event)
    state: dict[tuple[str, str], dict[str, Any]] = {}
    out: dict[int, str] = {}
    for bar in sorted(grouped):
        bar_events = grouped[bar]
        regime = str(_event_value(bar_events[0], "regime") or "all").strip().lower() or "all"
        out[bar] = _soft_confirmed_leader_for_regime(state, regime, current_bar=bar)
        for event in bar_events:
            label = str(_event_value(event, "actor_label") or "").strip()
            event_regime = str(_event_value(event, "regime") or "all").strip().lower() or "all"
            row = state.setdefault(
                (event_regime, label),
                {"trades": 0, "bars": [], "prefix": [0.0]},
            )
            row["trades"] += max(0, int(_event_value(event, "closed_trades") or 0))
            row["bars"].append(bar)
            row["prefix"].append(
                float(row["prefix"][-1])
                + _safe_float(_event_value(event, "realized_pnl_usd"))
            )
    return out


def _soft_confirmed_leader_for_regime(
    state: dict[tuple[str, str], dict[str, Any]],
    regime: str,
    *,
    current_bar: int,
) -> str:
    scored: list[tuple[str, float]] = []
    cutoff = int(current_bar) - 24
    for (item_regime, label), row in state.items():
        if item_regime != regime:
            continue
        if int(row.get("trades", 0) or 0) < 50:
            continue
        bars = row.get("bars", []) or []
        prefix = row.get("prefix", [0.0]) or [0.0]
        first_index = bisect_right(bars, cutoff)
        score = float(prefix[-1]) - float(prefix[first_index])
        if score > 0.0:
            scored.append((label, score))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[0][0]


def _rejection_index(
    rejections: Sequence[object],
) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for event in rejections or ():
        bar = int(_event_value(event, "bar") or 0)
        label = str(
            _event_value(event, "player_label")
            or _event_value(event, "label")
            or ""
        ).strip()
        reason = str(_event_value(event, "reason") or "")
        if bar > 0 and label:
            out[(bar, label)] = reason
    return out


def _oracle_mismatch_reason(
    *,
    bar: int,
    target_label: str,
    trading_row: dict[str, Any],
    current_shadow: dict[tuple[int, str], dict[str, float]],
    rejections: dict[tuple[int, str], str],
) -> str:
    rejection = str(rejections.get((bar, target_label), "") or "").lower()
    if "kill" in rejection:
        return "kill"
    if rejection:
        return "rejected by policy"
    shadow = current_shadow.get((bar, target_label), {})
    if _safe_float(shadow.get("signals", 0.0)) <= 0.0 and _safe_float(
        shadow.get("filled", 0.0)
    ) <= 0.0:
        return "no current signal"
    reason_text = " ".join(
        str(trading_row.get(key, "") or "").lower()
        for key in ("reason", "fallback_reason", "error")
    )
    if "cooldown" in reason_text:
        return "cooldown"
    if str(trading_row.get("executed_leader") or trading_row.get("leader") or "") in {
        "",
        "-",
        "NoTrade",
    }:
        return "no candidate"
    return "low score"


def _event_month_from_update(event: object) -> str:
    value = _event_value(event, "timestamp")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    text = str(value or "").strip()
    if len(text) >= 7:
        return text[:7]
    return ""


def _event_value(event: object, key: str) -> object:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _safe_float(value: object) -> float:
    parsed = _finite_float_or_none(value)
    return parsed if parsed is not None else 0.0


def _finite_float_or_none(value: object) -> float | None:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _share(part: object, total: object) -> float:
    denom = _safe_float(total)
    if denom <= 0:
        return 0.0
    return _safe_float(part) / denom * 100.0


def _parse_fixed_agent_player_sets(
    raw_items: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for raw in raw_items or ():
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("fixed agent player set must use Label=AgentA,AgentB")
        label, raw_agents = text.split("=", 1)
        agents = tuple(
            part.strip() for part in raw_agents.split(",") if part.strip()
        )
        parsed.append((label.strip(), agents))
    return _normalize_fixed_agent_player_sets(parsed)


def _normalize_fixed_agent_player_sets(
    raw_sets: Sequence[tuple[str, Sequence[str]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen_labels = set()
    for raw_label, raw_agents in raw_sets or ():
        label = str(raw_label or "").strip()
        if not label:
            raise ValueError("fixed agent player label must not be empty")
        if label in seen_labels:
            raise ValueError(f"duplicate fixed agent player label: {label}")
        seen_labels.add(label)
        if isinstance(raw_agents, str):
            agents = tuple(
                part.strip() for part in raw_agents.split(",") if part.strip()
            )
        else:
            agents = tuple(
                str(agent or "").strip()
                for agent in raw_agents
                if str(agent or "").strip()
            )
        if not agents:
            raise ValueError(f"fixed agent player {label} must include at least one agent")
        if len(set(agents)) != len(agents):
            raise ValueError(f"fixed agent player {label} has duplicate agents")
        normalized.append((label, agents))
    return tuple(normalized)


def _format_directory_issues(report: RetrodateDirReport) -> str:
    lines = [f"Retrodate directory is not usable: {report.path}"]
    for issue in report.issues:
        lines.append(f"directory:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _format_selection_errors(reports: Iterable[RetrodateFileReport]) -> str:
    lines = ["Requested Retrodate files failed validation"]
    for report in reports:
        for issue in report.issues:
            lines.append(f"{report.path.name}:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _write_run_summary(
    path: Path,
    summary: RetrodateRunSummary,
    selection: RetrodateFileSelection,
    config: RetrodateMarketConfig,
) -> None:
    effective_strategist = _build_strategist_config(config)
    data = {
        "output_dir": str(summary.output_dir),
        "analysis_report": str(summary.report_path),
        "requested_years": summary.requested_years,
        "executed_years": summary.executed_years,
        "excluded_files": [_file_report_payload(item) for item in selection.excluded_files],
        "missing_years": summary.missing_years,
        "bars_processed": summary.bars_processed,
        "first_timestamp": summary.first_timestamp,
        "last_timestamp": summary.last_timestamp,
        "stride_minutes": summary.stride_minutes,
        "timeframe": config.timeframe,
        "initial_capital": config.initial_capital,
        "risk_capital_fraction": config.risk_capital_fraction,
        "risk_max_leverage": config.risk_max_leverage,
        "apply_risk_leverage_to_notional": config.apply_risk_leverage_to_notional,
        "include_optional_agents": config.include_optional_agents,
        "optional_agent_labels": list(config.optional_agent_labels),
        "flash_audit_events_enabled": bool(config.flash_audit_events_enabled),
        "shadow_audit_events_enabled": bool(config.shadow_audit_events_enabled),
        "step_result_retention_enabled": bool(
            config.step_result_retention_enabled
        ),
        "compact_causal_entry_selected_only": bool(
            config.compact_causal_entry_selected_only
        ),
        "shadow_agent_include_labels": list(config.shadow_agent_include_labels),
        "shadow_player_include_labels": list(config.shadow_player_include_labels),
        "shadow_parallel_workers": int(config.shadow_parallel_workers),
        "shadow_position_diagnostic_bars": list(
            config.shadow_position_diagnostic_bars
        ),
        "shadow_position_diagnostic_labels": list(
            config.shadow_position_diagnostic_labels
        ),
        "solo_agent_candidate_limit": config.solo_agent_candidate_limit,
        "fixed_agent_players_enabled": config.fixed_agent_players_enabled,
        "fixed_agent_player_count": (
            len(config.fixed_agent_player_sets)
            if config.fixed_agent_players_enabled
            else 0
        ),
        "fixed_agent_player_sets": [
            {"label": label, "agent_labels": list(agent_labels)}
            for label, agent_labels in (
                config.fixed_agent_player_sets
                if config.fixed_agent_players_enabled
                else ()
            )
        ],
        "player_profile_labels": [
            str(getattr(profile, "label", ""))
            for profile in config.player_profiles
        ],
        "regime_switch_player_count": len(config.regime_switch_player_sets),
        "rotating_agent_player_count": len(config.rotating_agent_player_sets),
        "flash_enabled": config.flash_enabled,
        "hard_policy_enabled": config.hard_policy_enabled,
        "flash_min_score_to_trade": config.flash_min_score_to_trade,
        "flash_actionable_bonus": config.flash_actionable_bonus,
        "flash_no_data_score": config.flash_no_data_score,
        "flash_min_closed_trades_to_trade": config.flash_min_closed_trades_to_trade,
        "flash_min_pnl_pct_to_trade": config.flash_min_pnl_pct_to_trade,
        "flash_shadow_confirmation_enabled": config.flash_shadow_confirmation_enabled,
        "flash_symbol_shadow_confirmation_enabled": (
            config.flash_symbol_shadow_confirmation_enabled
        ),
        "flash_shadow_actor_fallback_confirmation_enabled": (
            config.flash_shadow_actor_fallback_confirmation_enabled
        ),
        "flash_shadow_base_fallback_confirmation_enabled": (
            config.flash_shadow_base_fallback_confirmation_enabled
        ),
        "flash_shadow_signal_handoff_enabled": (
            config.flash_shadow_signal_handoff_enabled
        ),
        "flash_genetics_confirmation_overlay_enabled": (
            config.flash_genetics_confirmation_overlay_enabled
        ),
        "flash_genetics_confirmation_labels": list(
            config.flash_genetics_confirmation_labels
        ),
        "flash_genetics_confirmation_allowed_signal_keys": list(
            config.flash_genetics_confirmation_allowed_signal_keys
        ),
        "flash_genetics_confirmation_contra_signal_keys": list(
            config.flash_genetics_confirmation_contra_signal_keys
        ),
        "flash_genetics_contra_validation_manifest_path": (
            str(config.flash_genetics_contra_validation_manifest_path)
            if config.flash_genetics_contra_validation_manifest_path is not None
            else None
        ),
        "flash_genetics_confirmation_contra_side_match_enabled": (
            config.flash_genetics_confirmation_contra_side_match_enabled
        ),
        "flash_genetics_confirmation_contra_static_enabled": (
            config.flash_genetics_confirmation_contra_static_enabled
        ),
        "flash_genetics_confirmation_contra_no_backfill_enabled": (
            config.flash_genetics_confirmation_contra_no_backfill_enabled
        ),
        "flash_genetics_confirmation_contra_risk_sizing_enabled": (
            config.flash_genetics_confirmation_contra_risk_sizing_enabled
        ),
        "flash_genetics_confirmation_contra_risk_mult": (
            config.flash_genetics_confirmation_contra_risk_mult
        ),
        "flash_genetics_confirmation_quality_gate_enabled": (
            config.flash_genetics_confirmation_quality_gate_enabled
        ),
        "flash_genetics_confirmation_min_closed_trades": (
            config.flash_genetics_confirmation_min_closed_trades
        ),
        "flash_genetics_confirmation_min_pnl_per_trade_pct": (
            config.flash_genetics_confirmation_min_pnl_per_trade_pct
        ),
        "flash_genetics_confirmation_score_bonus": (
            config.flash_genetics_confirmation_score_bonus
        ),
        "flash_genetics_confirmation_score_penalty": (
            config.flash_genetics_confirmation_score_penalty
        ),
        "flash_genetics_confirmation_contra_score_penalty": (
            config.flash_genetics_confirmation_contra_score_penalty
        ),
        "flash_shadow_actor_fallback_min_base_score": (
            config.flash_shadow_actor_fallback_min_base_score
        ),
        "flash_shadow_position_replay_actor_fallback_min_base_score": (
            config.flash_shadow_position_replay_actor_fallback_min_base_score
        ),
        "flash_shadow_position_replay_actor_fallback_min_shadow_score": (
            config.flash_shadow_position_replay_actor_fallback_min_shadow_score
        ),
        "flash_shadow_base_fallback_actor_keys": list(
            config.flash_shadow_base_fallback_actor_keys
        ),
        "flash_actor_switch_margin": config.flash_actor_switch_margin,
        "flash_anchor_actor_keys": list(config.flash_anchor_actor_keys),
        "flash_portfolio_actor_keys": list(config.flash_portfolio_actor_keys),
        "flash_portfolio_shadow_bootstrap_min_closed_enabled": (
            config.flash_portfolio_shadow_bootstrap_min_closed_enabled
        ),
        "flash_anchor_min_score_to_trade": config.flash_anchor_min_score_to_trade,
        "flash_anchor_shadow_min_score": config.flash_anchor_shadow_min_score,
        "flash_anchor_min_score_advantage": config.flash_anchor_min_score_advantage,
        "flash_prefer_solo_player_wrappers_enabled": (
            config.flash_prefer_solo_player_wrappers_enabled
        ),
        "flash_prefer_proven_solo_player_wrappers_enabled": (
            config.flash_prefer_proven_solo_player_wrappers_enabled
        ),
        "flash_proven_solo_min_score_advantage": (
            config.flash_proven_solo_min_score_advantage
        ),
        "flash_shadow_min_score": config.flash_shadow_min_score,
        "flash_shadow_min_closed_trades": config.flash_shadow_min_closed_trades,
        "flash_shadow_min_full_open_closed_trades": (
            config.flash_shadow_min_full_open_closed_trades
        ),
        "flash_shadow_quality_confirmation_enabled": (
            config.flash_shadow_quality_confirmation_enabled
        ),
        "flash_shadow_min_win_rate_pct": config.flash_shadow_min_win_rate_pct,
        "flash_shadow_max_recent_downside_usd": (
            config.flash_shadow_max_recent_downside_usd
        ),
        "flash_shadow_min_pnl_per_trade_lcb_usd": (
            config.flash_shadow_min_pnl_per_trade_lcb_usd
        ),
        "flash_shadow_pnl_per_trade_lcb_z": (
            config.flash_shadow_pnl_per_trade_lcb_z
        ),
        "flash_shadow_pnl_per_trade_lcb_penalty_floor_usd": (
            config.flash_shadow_pnl_per_trade_lcb_penalty_floor_usd
        ),
        "flash_shadow_pnl_per_trade_lcb_penalty_weight": (
            config.flash_shadow_pnl_per_trade_lcb_penalty_weight
        ),
        "flash_shadow_pnl_lcb_risk_sizing_enabled": (
            config.flash_shadow_pnl_lcb_risk_sizing_enabled
        ),
        "flash_shadow_pnl_lcb_risk_min_mult": (
            config.flash_shadow_pnl_lcb_risk_min_mult
        ),
        "flash_shadow_pnl_lcb_risk_floor_usd": (
            config.flash_shadow_pnl_lcb_risk_floor_usd
        ),
        "flash_shadow_pnl_lcb_risk_scale_usd": (
            config.flash_shadow_pnl_lcb_risk_scale_usd
        ),
        "flash_shadow_symbol_health_enabled": (
            config.flash_shadow_symbol_health_enabled
        ),
        "flash_shadow_symbol_health_min_closed_trades": (
            config.flash_shadow_symbol_health_min_closed_trades
        ),
        "flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd": (
            config.flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd
        ),
        "flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd": (
            config.flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd
        ),
        "flash_shadow_symbol_health_pnl_lcb_penalty_weight": (
            config.flash_shadow_symbol_health_pnl_lcb_penalty_weight
        ),
        "flash_actor_risk_sizing_enabled": config.flash_actor_risk_sizing_enabled,
        "flash_actor_risk_min_mult": config.flash_actor_risk_min_mult,
        "flash_actor_risk_max_mult": config.flash_actor_risk_max_mult,
        "flash_actor_risk_edge_scale_pct": config.flash_actor_risk_edge_scale_pct,
        "flash_funding_score_weight": config.flash_funding_score_weight,
        "flash_funding_risk_mult_weight": config.flash_funding_risk_mult_weight,
        "flash_funding_risk_mult_cap": config.flash_funding_risk_mult_cap,
        "flash_no_trade_fee_saving_score_enabled": (
            config.flash_no_trade_fee_saving_score_enabled
        ),
        "flash_no_trade_default_fee_bps": config.flash_no_trade_default_fee_bps,
        "flash_volatility_risk_sizing_enabled": (
            config.flash_volatility_risk_sizing_enabled
        ),
        "flash_volatility_risk_target_pct": config.flash_volatility_risk_target_pct,
        "flash_volatility_risk_min_volatility_pct": (
            config.flash_volatility_risk_min_volatility_pct
        ),
        "flash_volatility_risk_max_mult": config.flash_volatility_risk_max_mult,
        "flash_mixed_rotational_symbol_local_shadow_gate_enabled": (
            config.flash_mixed_rotational_symbol_local_shadow_gate_enabled
        ),
        "flash_mixed_rotational_symbol_local_shadow_min_score": (
            config.flash_mixed_rotational_symbol_local_shadow_min_score
        ),
        "flash_mixed_rotational_symbol_local_shadow_min_closed_trades": (
            config.flash_mixed_rotational_symbol_local_shadow_min_closed_trades
        ),
        "flash_genetics_probation_bypass_min_closed_enabled": (
            config.flash_genetics_probation_bypass_min_closed_enabled
        ),
        "flash_genetics_probation_bypass_trend_gate_enabled": (
            config.flash_genetics_probation_bypass_trend_gate_enabled
        ),
        "flash_technical_overlay_enabled": config.flash_technical_overlay_enabled,
        "flash_technical_hard_gate_enabled": config.flash_technical_hard_gate_enabled,
        "flash_technical_score_bonus": config.flash_technical_score_bonus,
        "flash_technical_score_penalty": config.flash_technical_score_penalty,
        "flash_technical_rsi_long_min": config.flash_technical_rsi_long_min,
        "flash_technical_rsi_long_max": config.flash_technical_rsi_long_max,
        "flash_technical_rsi_short_min": config.flash_technical_rsi_short_min,
        "flash_technical_rsi_short_max": config.flash_technical_rsi_short_max,
        "flash_technical_macd_histogram_min_abs_pct": (
            config.flash_technical_macd_histogram_min_abs_pct
        ),
        "flash_technical_atr_risk_sizing_enabled": (
            config.flash_technical_atr_risk_sizing_enabled
        ),
        "flash_technical_atr_target_pct": config.flash_technical_atr_target_pct,
        "flash_technical_atr_min_pct": config.flash_technical_atr_min_pct,
        "flash_technical_atr_max_mult": config.flash_technical_atr_max_mult,
        "flash_selected_subset_score_boosts": list(
            config.flash_selected_subset_score_boosts
        ),
        "flash_selected_subset_context_score_boosts": list(
            config.flash_selected_subset_context_score_boosts
        ),
        "flash_selected_subset_do_not_demote_signal_keys": list(
            config.flash_selected_subset_do_not_demote_signal_keys
        ),
        "flash_selected_subset_risk_mult_overrides": list(
            config.flash_selected_subset_risk_mult_overrides
        ),
        "flash_selected_subset_context_risk_mult_overrides": list(
            config.flash_selected_subset_context_risk_mult_overrides
        ),
        "flash_selected_subset_risk_min_mult": (
            config.flash_selected_subset_risk_min_mult
        ),
        "flash_selected_subset_risk_max_mult": (
            config.flash_selected_subset_risk_max_mult
        ),
        "flash_max_signals_per_actor": config.flash_max_signals_per_actor,
        "flash_overextension_guard_enabled": config.flash_overextension_guard_enabled,
        "flash_overextension_lookback_bars": config.flash_overextension_lookback_bars,
        "flash_short_overextension_return_floor_pct": (
            config.flash_short_overextension_return_floor_pct
        ),
        "flash_long_overextension_return_ceiling_pct": (
            config.flash_long_overextension_return_ceiling_pct
        ),
        "flash_overextension_volatility_normalized_enabled": (
            config.flash_overextension_volatility_normalized_enabled
        ),
        "flash_short_overextension_z_floor": (
            config.flash_short_overextension_z_floor
        ),
        "flash_long_overextension_z_ceiling": (
            config.flash_long_overextension_z_ceiling
        ),
        "flash_overextension_min_volatility_pct": (
            config.flash_overextension_min_volatility_pct
        ),
        "flash_denied_signal_keys": list(config.flash_denied_signal_keys),
        "flash_terminal_denied_signal_keys": list(
            config.flash_terminal_denied_signal_keys
        ),
        "flash_terminal_denied_context_signal_keys": list(
            config.flash_terminal_denied_context_signal_keys
        ),
        "flash_denied_open_symbols": list(config.flash_denied_open_symbols),
        "flash_denied_open_regimes": list(config.flash_denied_open_regimes),
        "flash_degradation_guard_enabled": config.flash_degradation_guard_enabled,
        "flash_degradation_actor_guard_enabled": (
            config.flash_degradation_actor_guard_enabled
        ),
        "flash_degradation_actor_scope": config.flash_degradation_actor_scope,
        "flash_degradation_signal_cooldown_bars": (
            config.flash_degradation_signal_cooldown_bars
        ),
        "flash_degradation_actor_cooldown_bars": (
            config.flash_degradation_actor_cooldown_bars
        ),
        "flash_degradation_symbol_guard_enabled": (
            config.flash_degradation_symbol_guard_enabled
        ),
        "flash_degradation_symbol_cooldown_bars": (
            config.flash_degradation_symbol_cooldown_bars
        ),
        "flash_degradation_symbol_lookback_bars": (
            config.flash_degradation_symbol_lookback_bars
        ),
        "flash_degradation_symbol_window_closed_trades": (
            config.flash_degradation_symbol_window_closed_trades
        ),
        "flash_degradation_symbol_min_closed_trades": (
            config.flash_degradation_symbol_min_closed_trades
        ),
        "flash_degradation_symbol_max_recent_pnl_usd": (
            config.flash_degradation_symbol_max_recent_pnl_usd
        ),
        "flash_degradation_window_closed_trades": (
            config.flash_degradation_window_closed_trades
        ),
        "flash_degradation_min_closed_trades": (
            config.flash_degradation_min_closed_trades
        ),
        "flash_degradation_max_recent_pnl_usd": (
            config.flash_degradation_max_recent_pnl_usd
        ),
        "flash_degradation_signal_min_pnl_per_trade_lcb_usd": (
            config.flash_degradation_signal_min_pnl_per_trade_lcb_usd
        ),
        "flash_degradation_pnl_per_trade_lcb_z": (
            config.flash_degradation_pnl_per_trade_lcb_z
        ),
        "flash_degradation_signal_risk_sizing_enabled": (
            config.flash_degradation_signal_risk_sizing_enabled
        ),
        "flash_degradation_signal_risk_mult": (
            config.flash_degradation_signal_risk_mult
        ),
        "flash_degradation_reserve_actor_cap": (
            config.flash_degradation_reserve_actor_cap
        ),
        "flash_degradation_recovery_enabled": (
            config.flash_degradation_recovery_enabled
        ),
        "flash_degradation_recovery_min_closed_trades": (
            config.flash_degradation_recovery_min_closed_trades
        ),
        "flash_degradation_recovery_min_recent_pnl_usd": (
            config.flash_degradation_recovery_min_recent_pnl_usd
        ),
        "flash_stale_position_exit_enabled": (
            config.flash_stale_position_exit_enabled
        ),
        "flash_stale_position_exit_max_age_bars": (
            config.flash_stale_position_exit_max_age_bars
        ),
        "flash_stale_position_exit_require_nonpositive_unrealized": (
            config.flash_stale_position_exit_require_nonpositive_unrealized
        ),
        "flash_partial_profit_lock_enabled": (
            config.flash_partial_profit_lock_enabled
        ),
        "flash_partial_profit_lock_trigger_pnl_pct": (
            config.flash_partial_profit_lock_trigger_pnl_pct
        ),
        "flash_partial_profit_lock_close_fraction": (
            config.flash_partial_profit_lock_close_fraction
        ),
        "flash_partial_profit_lock_min_age_bars": (
            config.flash_partial_profit_lock_min_age_bars
        ),
        "flash_partial_profit_lock_skip_protected_signal_keys": (
            config.flash_partial_profit_lock_skip_protected_signal_keys
        ),
        "flash_partial_profit_lock_skip_signal_keys": list(
            config.flash_partial_profit_lock_skip_signal_keys
        ),
        "flash_promotion_manifest_enabled": (
            config.flash_promotion_manifest_enabled
        ),
        "flash_promotion_manifest_path": (
            str(config.flash_promotion_manifest_path)
            if config.flash_promotion_manifest_path is not None
            else None
        ),
        "flash_promoted_signal_keys": list(config.flash_promoted_signal_keys),
        "flash_promotion_min_full_closed_trades": (
            config.flash_promotion_min_full_closed_trades
        ),
        "flash_promotion_min_latest_closed_trades": (
            config.flash_promotion_min_latest_closed_trades
        ),
        "flash_promotion_min_full_pnl_pct": (
            config.flash_promotion_min_full_pnl_pct
        ),
        "flash_promotion_min_latest_pnl_pct": (
            config.flash_promotion_min_latest_pnl_pct
        ),
        "flash_promotion_min_full_pnl_per_trade_lcb_pct": (
            config.flash_promotion_min_full_pnl_per_trade_lcb_pct
        ),
        "flash_promotion_min_latest_pnl_per_trade_lcb_pct": (
            config.flash_promotion_min_latest_pnl_per_trade_lcb_pct
        ),
        "flash_promotion_pnl_per_trade_lcb_z": (
            config.flash_promotion_pnl_per_trade_lcb_z
        ),
        "flash_promotion_max_drawdown_pct": (
            config.flash_promotion_max_drawdown_pct
        ),
        "flash_promotion_min_win_rate_pct": (
            config.flash_promotion_min_win_rate_pct
        ),
        "flash_promotion_max_recent_downside_usd": (
            config.flash_promotion_max_recent_downside_usd
        ),
        "flash_earned_cap_overrides_enabled": (
            config.flash_earned_cap_overrides_enabled
        ),
        "flash_promoted_actor_cap_overrides": list(
            config.flash_promoted_actor_cap_overrides
        ),
        "experimental_flash_actors_enabled": (
            config.experimental_flash_actors_enabled
        ),
        "experimental_flash_real_actors_enabled": (
            config.experimental_flash_real_actors_enabled
        ),
        "experimental_flash_agent_labels": (
            list(experimental_flash_agent_labels())
            if config.experimental_flash_actors_enabled
            else []
        ),
        "experimental_flash_fixed_agent_player_sets": [
            {"label": label, "agent_labels": list(agent_labels)}
            for label, agent_labels in (
                _experimental_flash_fixed_agent_player_sets()
                if (
                    config.experimental_flash_actors_enabled
                    and config.experimental_flash_real_actors_enabled
                )
                else ()
            )
        ],
        "experimental_flash_rotating_agent_player_sets": [
            {
                "label": label,
                "regime_agent_labels": {
                    regime: list(agent_labels)
                    for regime, agent_labels in mapping.items()
                },
                "fallback_agent_labels": list(fallback),
            }
            for label, mapping, fallback in (
                _experimental_flash_rotating_agent_player_sets()
                if (
                    config.experimental_flash_actors_enabled
                    and config.experimental_flash_real_actors_enabled
                )
                else ()
            )
        ],
        "actionable_fallback_enabled": config.actionable_fallback_enabled,
        "actionable_fallback_min_score": config.actionable_fallback_min_score,
        "actionable_fallback_require_has_data": (
            config.actionable_fallback_require_has_data
        ),
        "soft_allocator_execution_enabled": config.soft_allocator_execution_enabled,
        "soft_allocator_execution_soft_only": config.soft_allocator_execution_soft_only,
        "soft_allocator_execution_allow_labels": list(
            config.soft_allocator_execution_allow_labels
        ),
        "soft_allocator_execution_deny_actor_symbols": list(
            config.soft_allocator_execution_deny_actor_symbols
        ),
        "soft_allocator_realized_gate_enabled": (
            config.soft_allocator_realized_gate_enabled
        ),
        "soft_allocator_realized_gate_lookback_bars": (
            config.soft_allocator_realized_gate_lookback_bars
        ),
        "soft_allocator_realized_gate_min_closed_trades": (
            config.soft_allocator_realized_gate_min_closed_trades
        ),
        "soft_allocator_realized_gate_max_recent_pnl_usd": (
            config.soft_allocator_realized_gate_max_recent_pnl_usd
        ),
        "soft_allocator_execution_policy": (
            getattr(config.soft_allocator_execution_policy, "name", "")
            if config.soft_allocator_execution_policy is not None
            else ""
        ),
        "current_actionable_candidate_layer_enabled": (
            config.current_actionable_candidate_layer_enabled
            or config.use_v3_executable_soft_confirmed_score
        ),
        "real_promotion_gate_enabled": config.real_promotion_gate_enabled,
        "real_promotion_min_closed_trades": config.real_promotion_min_closed_trades,
        "real_promotion_min_pnl_pct": config.real_promotion_min_pnl_pct,
        "real_promotion_max_drawdown_pct": config.real_promotion_max_drawdown_pct,
        "real_promotion_loss_budget_pct": config.real_promotion_loss_budget_pct,
        "real_promotion_probation_min_score": config.real_promotion_probation_min_score,
        "use_v3_rolling_score": config.use_v3_rolling_score,
        "use_v3_entry_causal_score": config.use_v3_entry_causal_score,
        "use_v3_shadow_rolling_score": config.use_v3_shadow_rolling_score,
        "use_v3_soft_shadow_score": config.use_v3_soft_shadow_score,
        "use_v3_executable_soft_top1_score": (
            config.use_v3_executable_soft_top1_score
        ),
        "use_v3_executable_soft_confirmed_score": (
            config.use_v3_executable_soft_confirmed_score
        ),
        "v3_candidate_allow_labels": list(config.v3_candidate_allow_labels),
        "v3_shadow_position_gate_enabled": config.v3_shadow_position_gate_enabled,
        "v3_shadow_flat_handoff_enabled": config.v3_shadow_flat_handoff_enabled,
        "v3_shadow_fresh_handoff_enabled": config.v3_shadow_fresh_handoff_enabled,
        "v3_shadow_fresh_handoff_max_age_bars": config.v3_shadow_fresh_handoff_max_age_bars,
        "v3_shadow_fresh_handoff_require_positive_unrealized": (
            config.v3_shadow_fresh_handoff_require_positive_unrealized
        ),
        "v3_shadow_rolling_window_bars": config.v3_shadow_rolling_window_bars,
        "v3_shadow_rolling_min_closed_trades": config.v3_shadow_rolling_min_closed_trades,
        "effective_v3_current_actionable_gate_enabled": (
            effective_strategist.v3_current_actionable_gate_enabled
        ),
        "effective_v3_solo_current_actionable_gate_enabled": (
            effective_strategist.v3_solo_current_actionable_gate_enabled
        ),
        "effective_v3_shadow_rolling_window_bars": (
            effective_strategist.v3_shadow_rolling_window_bars
        ),
        "effective_v3_shadow_rolling_min_closed_trades": (
            effective_strategist.v3_shadow_rolling_min_closed_trades
        ),
        "v3_entry_causal_min_filled": config.v3_entry_causal_min_filled,
        "v3_entry_causal_actionability_weight": (
            config.v3_entry_causal_actionability_weight
        ),
        "v3_real_loss_rescue_enabled": config.v3_real_loss_rescue_enabled,
        "v3_real_loss_rescue_min_virtual_pnl_pct": (
            config.v3_real_loss_rescue_min_virtual_pnl_pct
        ),
        "v3_real_loss_rescue_max_virtual_dd_pct": (
            config.v3_real_loss_rescue_max_virtual_dd_pct
        ),
        "v3_real_loss_rescue_min_actionable_share": (
            config.v3_real_loss_rescue_min_actionable_share
        ),
        "v3_real_loss_rescue_min_recent_filled": (
            config.v3_real_loss_rescue_min_recent_filled
        ),
        "v3_real_loss_rescue_max_real_loss_pct": (
            config.v3_real_loss_rescue_max_real_loss_pct
        ),
        "v3_real_loss_rescue_allow_genetics": (
            config.v3_real_loss_rescue_allow_genetics
        ),
        "v3_probation_shadow_rescue_enabled": (
            config.v3_probation_shadow_rescue_enabled
        ),
        "v3_probation_shadow_rescue_min_virtual_pnl_pct": (
            config.v3_probation_shadow_rescue_min_virtual_pnl_pct
        ),
        "v3_probation_shadow_rescue_max_virtual_dd_pct": (
            config.v3_probation_shadow_rescue_max_virtual_dd_pct
        ),
        "v3_probation_shadow_rescue_min_actionable_share": (
            config.v3_probation_shadow_rescue_min_actionable_share
        ),
        "v3_probation_shadow_rescue_min_recent_filled": (
            config.v3_probation_shadow_rescue_min_recent_filled
        ),
        "v3_probation_shadow_rescue_min_recent_pnl_usd": (
            config.v3_probation_shadow_rescue_min_recent_pnl_usd
        ),
        "v3_probation_shadow_rescue_allow_genetics": (
            config.v3_probation_shadow_rescue_allow_genetics
        ),
        "v3_persistent_loss_kill_min_closed_trades": (
            config.v3_persistent_loss_kill_min_closed_trades
        ),
        "v3_persistent_loss_kill_pnl_pct": config.v3_persistent_loss_kill_pnl_pct,
        "v3_persistent_loss_kill_win_rate_pct": (
            config.v3_persistent_loss_kill_win_rate_pct
        ),
        "v3_persistent_loss_requires_virtual_weakness": (
            config.v3_persistent_loss_requires_virtual_weakness
        ),
        "v3_persistent_loss_virtual_max_pnl_pct": (
            config.v3_persistent_loss_virtual_max_pnl_pct
        ),
        "v3_persistent_loss_virtual_min_dd_pct": (
            config.v3_persistent_loss_virtual_min_dd_pct
        ),
        "v3_probation_loss_kill_min_closed_trades": (
            config.v3_probation_loss_kill_min_closed_trades
        ),
        "v3_probation_loss_kill_pnl_pct": config.v3_probation_loss_kill_pnl_pct,
        "v3_probation_loss_kill_win_rate_pct": (
            config.v3_probation_loss_kill_win_rate_pct
        ),
        "v3_probation_loss_kill_label_prefixes": list(
            config.v3_probation_loss_kill_label_prefixes
        ),
        "hard_policy_deny_labels": list(config.hard_policy_deny_labels),
        "max_new_opens_per_bar": config.max_new_opens_per_bar,
        "risk_max_open_positions": config.risk_max_open_positions,
        "risk_max_leverage": config.risk_max_leverage,
        "apply_risk_leverage_to_notional": config.apply_risk_leverage_to_notional,
        "genetics_probation_execution_enabled": (
            config.genetics_probation_execution_enabled
        ),
        "genetics_probation_labels": list(config.genetics_probation_labels),
        "genetics_probation_allowed_regimes": list(
            config.genetics_probation_allowed_regimes
        ),
        "genetics_probation_allowed_signal_keys": list(
            config.genetics_probation_allowed_signal_keys
        ),
        "genetics_probation_risk_mult": config.genetics_probation_risk_mult,
        "genetics_probation_min_regime_confidence": (
            config.genetics_probation_min_regime_confidence
        ),
        "genetics_probation_max_real_trades": (
            config.genetics_probation_max_real_trades
        ),
        "genetics_probation_max_daily_trades": (
            config.genetics_probation_max_daily_trades
        ),
        "genetics_probation_require_shadow_confirmation": (
            config.genetics_probation_require_shadow_confirmation
        ),
        "v3_realized_profit_lock_min_closed_trades": (
            config.v3_realized_profit_lock_min_closed_trades
        ),
        "v3_realized_profit_lock_min_peak_pnl_pct": (
            config.v3_realized_profit_lock_min_peak_pnl_pct
        ),
        "v3_realized_profit_lock_max_giveback_pct": (
            config.v3_realized_profit_lock_max_giveback_pct
        ),
        "v3_realized_profit_lock_floor_pnl_pct": (
            config.v3_realized_profit_lock_floor_pnl_pct
        ),
        "v3_panteon_equity_guard_enabled": (
            config.v3_panteon_equity_guard_enabled
        ),
        "v3_panteon_equity_guard_min_peak_pnl_pct": (
            config.v3_panteon_equity_guard_min_peak_pnl_pct
        ),
        "v3_panteon_equity_guard_max_giveback_pct": (
            config.v3_panteon_equity_guard_max_giveback_pct
        ),
        "v3_panteon_equity_guard_floor_pnl_pct": (
            config.v3_panteon_equity_guard_floor_pnl_pct
        ),
        "v3_panteon_equity_guard_cooldown_bars": (
            config.v3_panteon_equity_guard_cooldown_bars
        ),
        "registered_agents": list(summary.registered_agents),
        "player_profile_count": summary.player_profile_count,
        "max_bars": summary.max_bars,
        "step_errors": list(summary.step_errors),
        "validation_files": [_file_report_payload(item) for item in selection.report.files],
    }
    soft_allocator = _load_json(summary.output_dir / "soft_allocator_report.json")
    if soft_allocator:
        data["soft_allocator_report"] = {
            "best_single_label": soft_allocator.get("best_single_label", ""),
            "best_single_pnl_usd": soft_allocator.get("best_single_pnl_usd", 0.0),
            "best_policy_name": soft_allocator.get("best_policy_name", ""),
            "best_policy_pnl_usd": soft_allocator.get("best_policy_pnl_usd", 0.0),
            "best_policy_max_drawdown_pct": soft_allocator.get(
                "best_policy_max_drawdown_pct",
                0.0,
            ),
            "regret_vs_best_single_usd": soft_allocator.get(
                "regret_vs_best_single_usd",
                0.0,
            ),
            "beats_best_single": soft_allocator.get("beats_best_single", False),
        }
    perfect_panteon = _load_json(summary.output_dir / "perfect_panteon_report.json")
    if perfect_panteon:
        data["perfect_panteon_report"] = {
            "pnl_usd": perfect_panteon.get("pnl_usd", 0.0),
            "pnl_pct": perfect_panteon.get("pnl_pct", 0.0),
            "max_drawdown_pct": perfect_panteon.get("max_drawdown_pct", 0.0),
            "month_count": perfect_panteon.get("month_count", 0),
            "profitable_months": perfect_panteon.get("profitable_months", 0),
            "cash_months": perfect_panteon.get("cash_months", 0),
            "closed_trades": perfect_panteon.get("closed_trades", 0.0),
        }
    allocation_diagnostics = _load_json(
        summary.output_dir / "allocation_diagnostics.json"
    )
    if allocation_diagnostics:
        data["allocation_diagnostics"] = {
            "bars": allocation_diagnostics.get("bars", 0),
            "no_trade_bars": allocation_diagnostics.get("no_trade_bars", 0),
            "no_trade_share_pct": allocation_diagnostics.get("no_trade_share_pct", 0.0),
            "raw_zero_bars": allocation_diagnostics.get("raw_zero_bars", 0),
            "raw_zero_share_pct": allocation_diagnostics.get("raw_zero_share_pct", 0.0),
            "filled_zero_bars": allocation_diagnostics.get("filled_zero_bars", 0),
            "filled_zero_share_pct": allocation_diagnostics.get(
                "filled_zero_share_pct",
                0.0,
            ),
        }
    candidate_diagnostics = _load_json(
        summary.output_dir / "candidate_diagnostics.json"
    )
    if candidate_diagnostics:
        data["candidate_diagnostics"] = {
            "candidate_rows": candidate_diagnostics.get("candidate_rows", 0),
            "selected_rows": candidate_diagnostics.get("selected_rows", 0),
            "rejection_rows": candidate_diagnostics.get("rejection_rows", 0),
            "group_summary": candidate_diagnostics.get("group_summary", {}),
        }
    flash_attribution = _load_json(
        summary.output_dir / "flash_attribution_summary.json"
    )
    if flash_attribution:
        data["flash_attribution_summary"] = dict(
            flash_attribution.get("summary", {}) or {}
        )
    component_benchmark = _load_json(
        summary.output_dir / "component_benchmark_report.json"
    )
    if component_benchmark:
        data["component_benchmark_report"] = dict(
            component_benchmark.get("summary", {}) or {}
        )
    standalone_vs_flash = _load_json(
        summary.output_dir / "standalone_vs_flash_selected_report.json"
    )
    if standalone_vs_flash:
        data["standalone_vs_flash_selected_report"] = dict(
            standalone_vs_flash.get("summary", {}) or {}
        )
    experimental_shadow = _load_json(
        summary.output_dir / "experimental_flash_shadow_report.json"
    )
    if experimental_shadow:
        data["experimental_flash_shadow_report"] = dict(
            experimental_shadow.get("summary", {}) or {}
        )
    signal_key_shadow = _load_json(
        summary.output_dir / "flash_signal_key_shadow_report.json"
    )
    if signal_key_shadow:
        data["flash_signal_key_shadow_report"] = dict(
            signal_key_shadow.get("summary", {}) or {}
        )
    promotion_manifest = _load_json(summary.output_dir / "flash_promotion_manifest.json")
    if promotion_manifest:
        data["flash_promotion_manifest"] = {
            "allowed_signal_keys": len(
                promotion_manifest.get("allowed_signal_keys", []) or []
            ),
            "rejected_signal_keys": len(promotion_manifest.get("rejected", []) or []),
        }
    oracle_mismatch = _load_json(summary.output_dir / "oracle_mismatch_report.json")
    if oracle_mismatch:
        data["oracle_mismatch_report"] = oracle_mismatch.get("summary", {})
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_analysis_report(
    path: Path,
    summary: RetrodateRunSummary,
    selection: RetrodateFileSelection,
) -> None:
    status = _load_json(summary.output_dir / "status.json")
    agents_payload = _load_json(summary.output_dir / "leaderboard_agents.json")
    players_payload = _load_json(summary.output_dir / "leaderboard_players.json")
    agents = agents_payload.get("agents", {}) if isinstance(agents_payload, dict) else {}
    players = players_payload.get("players", {}) if isinstance(players_payload, dict) else {}
    live_session = status.get("live_session", {}) if isinstance(status, dict) else {}
    shadow = status.get("shadow", {}) if isinstance(status, dict) else {}
    soft_allocator = _load_json(summary.output_dir / "soft_allocator_report.json")
    perfect_panteon = _load_json(summary.output_dir / "perfect_panteon_report.json")
    allocation_diagnostics = _load_json(
        summary.output_dir / "allocation_diagnostics.json"
    )
    candidate_diagnostics = _load_json(
        summary.output_dir / "candidate_diagnostics.json"
    )
    flash_attribution = _load_json(
        summary.output_dir / "flash_attribution_summary.json"
    )
    component_benchmark = _load_json(
        summary.output_dir / "component_benchmark_report.json"
    )
    standalone_vs_flash = _load_json(
        summary.output_dir / "standalone_vs_flash_selected_report.json"
    )
    experimental_shadow = _load_json(
        summary.output_dir / "experimental_flash_shadow_report.json"
    )
    signal_key_shadow = _load_json(
        summary.output_dir / "flash_signal_key_shadow_report.json"
    )
    promotion_manifest = _load_json(summary.output_dir / "flash_promotion_manifest.json")
    oracle_mismatch = _load_json(summary.output_dir / "oracle_mismatch_report.json")

    excluded_lines = []
    for item in selection.excluded_files:
        issues = "; ".join(f"{issue.code}: {issue.message}" for issue in item.issues)
        excluded_lines.append(f"- `{item.path.name}`: {issues}")
    if not excluded_lines:
        excluded_lines.append("- none")

    dashboard_files = [
        "dashboard.html",
        "dashboard.txt",
        "dashboard_latest.png",
        "shadow_dashboard.png",
        "regime_dashboard.png",
        "memory_dashboard.png",
        "status.json",
        "leaderboard_agents.json",
        "leaderboard_players.json",
        "soft_allocator_report.md",
        "soft_allocator_report.json",
        "perfect_panteon_report.md",
        "perfect_panteon_report.json",
        "allocation_diagnostics.json",
        "candidate_diagnostics.json",
        "flash_attribution_summary.json",
        "component_benchmark_report.md",
        "component_benchmark_report.json",
        "standalone_vs_flash_selected_report.md",
        "standalone_vs_flash_selected_report.json",
        "walk_forward_report.json",
        "events.jsonl",
        "experimental_flash_shadow_report.md",
        "experimental_flash_shadow_report.json",
        "flash_signal_key_shadow_report.md",
        "flash_signal_key_shadow_report.json",
        "flash_promotion_manifest.md",
        "flash_promotion_manifest.json",
        "oracle_mismatch_report.json",
        "causal_entry_decisions.jsonl",
        "shadow_player_pnl_events.jsonl",
        "shadow_agent_pnl_events.jsonl",
    ]
    dashboard_lines = [
        f"- `{name}`"
        for name in dashboard_files
        if (summary.output_dir / name).exists()
    ]
    note_lines = [
        "- This run uses the existing live-like Panteon v2 pipeline, OutputWriter, leaderboards, and dashboard renderers.",
        "- Shadow tournament evaluates registered agents and composed player profiles on every replayed bar.",
        "- The default run is an hourly-stride benchmark; a full 1m replay is available with --stride-minutes 1 but is much heavier.",
    ]
    if any(
        int(report.expected_year or 0) == 2026 and not report.is_valid
        for report in selection.excluded_files
    ):
        note_lines.insert(
            2,
            "- The requested 2026 file was excluded by validation; check Data Exclusions above.",
        )

    lines = [
        "# Retrodate Market Benchmark",
        "",
        "## Scope",
        f"- Requested years: {', '.join(str(year) for year in summary.requested_years)}",
        f"- Executed years: {', '.join(str(year) for year in summary.executed_years) or 'none'}",
        f"- Period: {summary.first_timestamp or 'n/a'} to {summary.last_timestamp or 'n/a'}",
        f"- Bars processed: {summary.bars_processed}",
        f"- Replay stride: {summary.stride_minutes} minutes from 1m OHLCV data",
        f"- Registered agents: {len(summary.registered_agents)}",
        f"- Configured player profiles: {summary.player_profile_count}",
        f"- Player leaderboard rows: {len(players)}",
        "",
        "## Data Exclusions",
        *excluded_lines,
        "",
        "## Panteon Result",
        f"- Current leader: `{status.get('current_leader') or '-'}`",
        f"- Panteon owned PnL: {_fmt_pct(live_session.get('panteon_owned_pnl_pct'))}",
        f"- Panteon realized PnL USD: {_fmt_money(live_session.get('panteon_owned_realized_pnl_usd'))}",
        f"- Panteon max drawdown: {_fmt_pct(status.get('panteon_max_drawdown_pct'))}",
        f"- Panteon realized max drawdown: {_fmt_pct(status.get('panteon_realized_max_drawdown_pct'))}",
        f"- Real closed trades: {live_session.get('real_closed_trades', 0)}",
        f"- Open Panteon positions: {live_session.get('panteon_owned_positions_count', 0)}",
        f"- Last shadow actors: {shadow.get('actors', 0)}",
        "",
        *(
            _allocation_diagnostics_report_lines(allocation_diagnostics) + [""]
            if allocation_diagnostics else []
        ),
        *(
            _candidate_diagnostics_report_lines(candidate_diagnostics) + [""]
            if candidate_diagnostics else []
        ),
        *(
            _flash_attribution_report_lines(flash_attribution) + [""]
            if flash_attribution else []
        ),
        *(
            _component_benchmark_analysis_report_lines(component_benchmark) + [""]
            if component_benchmark else []
        ),
        *(
            _standalone_vs_flash_selected_analysis_report_lines(standalone_vs_flash)
            + [""]
            if standalone_vs_flash else []
        ),
        *(
            _experimental_flash_shadow_gate_report_lines(experimental_shadow) + [""]
            if experimental_shadow else []
        ),
        *(
            _flash_signal_key_shadow_analysis_report_lines(signal_key_shadow) + [""]
            if signal_key_shadow else []
        ),
        *(
            _flash_promotion_manifest_analysis_report_lines(promotion_manifest) + [""]
            if promotion_manifest else []
        ),
        *(
            _oracle_mismatch_report_lines(oracle_mismatch) + [""]
            if oracle_mismatch else []
        ),
        *(_soft_allocator_report_lines(soft_allocator) + [""] if soft_allocator else []),
        *(_perfect_panteon_report_lines(perfect_panteon) + [""] if perfect_panteon else []),
        "## Top Agents By Virtual PnL",
        _markdown_table(_top_rows(agents, limit=10)),
        "",
        "## Top Players By Virtual PnL",
        _markdown_table(_top_rows(players, limit=10)),
        "",
        "## Notes",
        *note_lines,
        "",
        "## Dashboard Artifacts",
        *(dashboard_lines or ["- none"]),
    ]
    if summary.step_errors:
        lines.extend([
            "",
            "## Step Errors",
            *[f"- {error}" for error in summary.step_errors[:20]],
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _allocation_diagnostics_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    top = []
    leaders = report.get("leaders", {})
    if isinstance(leaders, dict):
        for label, payload in list(leaders.items())[:5]:
            if isinstance(payload, dict):
                top.append(
                    f"- `{label}`: bars {int(payload.get('bars', 0) or 0)}, "
                    f"raw-zero {_fmt_pct(payload.get('raw_zero_share_pct'))}, "
                    f"filled-zero {_fmt_pct(payload.get('filled_zero_share_pct'))}"
                )
    return [
        "## Allocation Diagnostics",
        f"- Bars analyzed: {int(report.get('bars', 0) or 0)}",
        f"- NoTrade share: {_fmt_pct(report.get('no_trade_share_pct'))}",
        f"- Raw-zero share: {_fmt_pct(report.get('raw_zero_share_pct'))}",
        f"- Filled-zero share: {_fmt_pct(report.get('filled_zero_share_pct'))}",
        f"- Details: `allocation_diagnostics.json`",
        *(["- Top leaders by bar share:"] + top if top else []),
    ]


def _candidate_diagnostics_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    groups = report.get("group_summary", {})
    rejection_summary = (
        report.get("rejection_summary", {}) if isinstance(report, dict) else {}
    )
    active_reasons = (
        rejection_summary.get("flash_active_reason_counts", {})
        if isinstance(rejection_summary, dict)
        else {}
    )
    group_lines = []
    if isinstance(groups, dict):
        for group, payload in groups.items():
            if isinstance(payload, dict):
                group_lines.append(
                    f"- `{group}`: candidates {int(payload.get('candidate_count', 0) or 0)}, "
                    f"selected bars {int(payload.get('selected_bars', 0) or 0)}, "
                    f"selected share {_fmt_pct(payload.get('selected_share_pct'))}"
                )
    active_reason_lines = _top_count_lines(active_reasons, limit=5)
    return [
        "## Candidate Diagnostics",
        f"- Candidate score rows: {int(report.get('candidate_rows', 0) or 0)}",
        f"- Selected rows: {int(report.get('selected_rows', 0) or 0)}",
        f"- Rejection rows: {int(report.get('rejection_rows', 0) or 0)}",
        f"- Details: `candidate_diagnostics.json`",
        *(["- Group summary:"] + group_lines if group_lines else []),
        *(
            ["- Top active Flash rejection reasons:"] + active_reason_lines
            if active_reason_lines else []
        ),
    ]


def _top_count_lines(counts: object, *, limit: int = 5) -> list[str]:
    if not isinstance(counts, dict):
        return []
    rows = [
        (str(key), int(value or 0))
        for key, value in counts.items()
        if str(key) and int(value or 0) > 0
    ]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return [f"- `{key}`: {count}" for key, count in rows[:limit]]


def _flash_attribution_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    rows = report.get("rows", []) if isinstance(report, dict) else []
    top_lines: list[str] = []
    if isinstance(rows, list):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            top_lines.append(
                f"- `{row.get('actor_key') or '-'}` / "
                f"`{row.get('symbol') or '-'}` / `{row.get('action') or '-'}`: "
                f"selected {int(row.get('selected_signals', 0) or 0)}, "
                f"filled {int(row.get('filled_signals', 0) or 0)}, "
                f"closed {int(row.get('closed_trades', 0) or 0)}, "
                f"PnL {_fmt_money(row.get('realized_pnl_usd'))}"
            )
    return [
        "## Flash Attribution",
        f"- Selected signals: {int(summary.get('selected_signals', 0) or 0)}",
        f"- Executable selected signals: {int(summary.get('executable_selected_signals', 0) or 0)}",
        f"- Guard-filtered selected signals: {int(summary.get('selected_filtered_before_execution', 0) or 0)}",
        f"- Missing execution events: {int(summary.get('missing_execution_signals', 0) or 0)}",
        f"- Filled signals: {int(summary.get('filled_signals', 0) or 0)}",
        f"- Blocked signals: {int(summary.get('blocked_signals', 0) or 0)}",
        f"- Rejected signals: {int(summary.get('rejected_signals', 0) or 0)}",
        f"- Pending signals: {int(summary.get('pending_signals', 0) or 0)}",
        f"- Closed trades: {int(summary.get('closed_trades', 0) or 0)}",
        f"- Realized PnL: {_fmt_money(summary.get('realized_pnl_usd'))}",
        f"- Details: `flash_attribution_summary.json`",
        *(["- Top actor/symbol/action rows:"] + top_lines if top_lines else []),
    ]


def _component_benchmark_analysis_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    components = report.get("components", []) if isinstance(report, dict) else []
    top_lines: list[str] = []
    if isinstance(components, list):
        for row in components[:5]:
            if not isinstance(row, dict):
                continue
            top_lines.append(
                f"- `{row.get('actor_type') or '-'}:{row.get('label') or '-'}`: "
                f"{_fmt_pct(row.get('pnl_pct'))}, "
                f"closed {int(row.get('closed_trades', 0) or 0)}, "
                f"win rate {_fmt_pct(row.get('win_rate_pct'))}"
            )
    return [
        "## Component Benchmark",
        f"- Panteon PnL: {_fmt_pct(summary.get('panteon_pnl_pct'))}",
        (
            f"- Best component: `{summary.get('best_component_type') or '-'}:"
            f"{summary.get('best_component_label') or '-'}` at "
            f"{_fmt_pct(summary.get('best_component_pnl_pct'))}"
        ),
        f"- Panteon alpha: {_fmt_pct(summary.get('panteon_alpha_pct'))}",
        f"- Required alpha: {_fmt_pct(summary.get('min_alpha_pct'))}",
        f"- Beats best component: {bool(summary.get('panteon_beats_best_component'))}",
        f"- Details: `component_benchmark_report.json`",
        *(["- Top components:"] + top_lines if top_lines else []),
    ]


def _standalone_vs_flash_selected_analysis_report_lines(
    report: dict[str, Any],
) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    actors = report.get("actors", []) if isinstance(report, dict) else []
    actor_lines: list[str] = []
    if isinstance(actors, list):
        for row in actors[:5]:
            if not isinstance(row, dict):
                continue
            actor_lines.append(
                f"- `{row.get('label') or '-'}`: standalone "
                f"{_fmt_pct(row.get('standalone_pnl_pct'))}, Flash-selected "
                f"{_fmt_pct(row.get('flash_selected_pnl_pct'))}, alpha "
                f"{_fmt_pct(row.get('selection_alpha_pct'))}, selected "
                f"{int(row.get('flash_selected_signals', 0) or 0)}"
            )
    return [
        "## Standalone vs Flash Selected",
        f"- Targets: {int(summary.get('target_count', 0) or 0)}",
        f"- Targets with Flash selection: {int(summary.get('targets_with_flash_selection', 0) or 0)}",
        f"- Standalone total PnL: {_fmt_pct(summary.get('total_standalone_pnl_pct'))}",
        f"- Flash-selected total PnL: {_fmt_pct(summary.get('total_flash_selected_pnl_pct'))}",
        f"- Selection alpha: {_fmt_pct(summary.get('total_selection_alpha_pct'))}",
        f"- Details: `standalone_vs_flash_selected_report.json`",
        *(["- Target actors:"] + actor_lines if actor_lines else []),
    ]


def _experimental_flash_shadow_gate_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    rows = report.get("labels", []) if isinstance(report, dict) else []
    label_lines: list[str] = []
    if isinstance(rows, list):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            label_lines.append(
                f"- `{row.get('label') or '-'}`: shadow PnL "
                f"{_fmt_pct(row.get('shadow_pnl_pct'))}, "
                f"DD {_fmt_pct(row.get('max_drawdown_pct'))}, "
                f"closed {int(row.get('closed_trades', 0) or 0)}, "
                f"real selected {int(row.get('real_selected_signals', 0) or 0)}, "
                f"promote {bool(row.get('promote_to_real'))}"
            )
    return [
        "## Experimental Flash Shadow Gate",
        f"- Labels: {int(summary.get('label_count', 0) or 0)}",
        f"- Shadow gate passed: {int(summary.get('shadow_gate_passed', 0) or 0)}",
        f"- Promote-to-real labels: {int(summary.get('promote_to_real', 0) or 0)}",
        f"- Real selected signals: {int(summary.get('real_selected_signals', 0) or 0)}",
        f"- Latest period: `{summary.get('latest_period') or '-'}`",
        f"- Latest period gate passed: {int(summary.get('latest_period_gate_passed', 0) or 0)}",
        f"- Details: `experimental_flash_shadow_report.json`",
        *(["- Label summary:"] + label_lines if label_lines else []),
    ]


def _flash_signal_key_shadow_analysis_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    rows = report.get("rows", []) if isinstance(report, dict) else []
    top_lines: list[str] = []
    if isinstance(rows, list):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            top_lines.append(
                f"- `{row.get('signal_key') or '-'}`: "
                f"{_fmt_pct(row.get('full_pnl_pct'))}, "
                f"latest {_fmt_pct(row.get('latest_pnl_pct'))}, "
                f"closed {int(row.get('full_closed_trades', 0) or 0)}, "
                f"win rate {_fmt_pct(row.get('win_rate_pct'))}"
            )
    return [
        "## Flash Signal-Key Shadow",
        f"- Signal keys: {int(summary.get('signal_key_count', 0) or 0)}",
        f"- Positive signal keys: {int(summary.get('positive_signal_keys', 0) or 0)}",
        f"- Latest period: `{summary.get('latest_period') or '-'}`",
        f"- Details: `flash_signal_key_shadow_report.json`",
        *(["- Top signal keys:"] + top_lines if top_lines else []),
    ]


def _flash_promotion_manifest_analysis_report_lines(data: dict[str, Any]) -> list[str]:
    if not data:
        return []
    allowed = data.get("allowed_signal_keys", [])
    rejected = data.get("rejected", [])
    allowed_count = len(allowed) if isinstance(allowed, list) else 0
    rejected_count = len(rejected) if isinstance(rejected, list) else 0
    allowed_lines = [
        f"- `{key}`"
        for key in (allowed[:5] if isinstance(allowed, list) else [])
    ]
    return [
        "## Flash Promotion Manifest",
        f"- Allowed signal keys: {allowed_count}",
        f"- Rejected signal keys: {rejected_count}",
        f"- Details: `flash_promotion_manifest.json`",
        *(["- Allowed keys:"] + allowed_lines if allowed_lines else []),
    ]


def _oracle_mismatch_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    reason_counts = summary.get("reason_counts", {}) if isinstance(summary, dict) else {}
    reason_lines = []
    if isinstance(reason_counts, dict):
        reason_lines = [
            f"- `{reason}`: {int(count or 0)}"
            for reason, count in sorted(reason_counts.items())
        ]
    return [
        "## Oracle Mismatch",
        f"- Trading rows: {int(summary.get('trading_rows', 0) or 0)}",
        f"- Mismatch rows: {int(summary.get('mismatch_rows', 0) or 0)}",
        f"- Details: `oracle_mismatch_report.json`",
        *(["- Reason counts:"] + reason_lines if reason_lines else []),
    ]


def _soft_allocator_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    return [
        "## Soft Allocator Simulation",
        f"- Best single shadow player: `{report.get('best_single_label') or '-'}` "
        f"({_fmt_money(report.get('best_single_pnl_usd'))}, "
        f"{_fmt_pct(report.get('best_single_pnl_pct'))}, "
        f"DD {_fmt_pct(report.get('best_single_max_drawdown_pct'))})",
        f"- Best soft policy: `{report.get('best_policy_name') or '-'}` "
        f"({_fmt_money(report.get('best_policy_pnl_usd'))}, "
        f"{_fmt_pct(report.get('best_policy_pnl_pct'))}, "
        f"DD {_fmt_pct(report.get('best_policy_max_drawdown_pct'))})",
        f"- Regret vs best single: {_fmt_money(report.get('regret_vs_best_single_usd'))}",
        f"- Beats best single: {'yes' if report.get('beats_best_single') else 'no'}",
        f"- Details: `soft_allocator_report.md`, `soft_allocator_report.json`",
    ]


def _perfect_panteon_report_lines(report: dict[str, Any]) -> list[str]:
    if not report:
        return []
    return [
        "## Perfect Panteon Monthly Oracle",
        f"- Perfect monthly PnL: {_fmt_money(report.get('pnl_usd'))}, "
        f"{_fmt_pct(report.get('pnl_pct'))}",
        f"- Max drawdown: {_fmt_pct(report.get('max_drawdown_pct'))}",
        f"- Months observed: {int(report.get('month_count', 0) or 0)}",
        f"- Profitable months: {int(report.get('profitable_months', 0) or 0)}",
        f"- Cash months: {int(report.get('cash_months', 0) or 0)}",
        f"- Closed trades: {float(report.get('closed_trades', 0.0) or 0.0):.2f}",
        f"- Details: `perfect_panteon_report.md`, `perfect_panteon_report.json`",
    ]


def _file_report_payload(report: RetrodateFileReport) -> dict[str, Any]:
    return {
        "path": str(report.path),
        "expected_year": report.expected_year,
        "timeframe": report.timeframe,
        "rows": report.rows,
        "symbols": report.symbols,
        "first_datetime": report.first_datetime,
        "last_datetime": report.last_datetime,
        "found_years": report.found_years,
        "is_valid": report.is_valid,
        "issues": [
            {"code": issue.code, "message": issue.message}
            for issue in report.issues
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _top_rows(payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "label": str(label),
                "pnl_pct": _coerce_float(metrics.get("pnl_pct")),
                "closed_trades": int(_coerce_float(metrics.get("closed_trades"))),
                "win_rate": _coerce_float(metrics.get("win_rate")),
                "signals": int(_coerce_float(metrics.get("signals"))),
                "max_drawdown_pct": _coerce_float(metrics.get("max_drawdown_pct")),
            }
        )
    return sorted(rows, key=lambda item: item["pnl_pct"], reverse=True)[:limit]


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No data._"
    lines = [
        "| Rank | Label | PnL % | Trades | Win rate % | Signals | Max DD % |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['label']}` | "
            f"{row['pnl_pct']:.2f} | {row['closed_trades']} | "
            f"{row['win_rate']:.2f} | {row['signals']} | "
            f"{row['max_drawdown_pct']:.2f} |"
        )
    return "\n".join(lines)


def _fmt_pct(value: object) -> str:
    return f"{_coerce_float(value):.2f}%"


def _fmt_money(value: object) -> str:
    return f"${_coerce_float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
