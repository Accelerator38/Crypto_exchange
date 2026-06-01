from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from panteon_v2.analysis.retrodate_market_runner import (
    RetrodateMarketConfig,
    RetrodateRunSummary,
    run_retrodate_market_benchmark,
)

from .config import (
    LEGEND_FIXED_AGENT_PLAYER_SETS,
    LEGEND_REGIME_SWITCH_PLAYER_SETS,
    LEGEND_ROTATING_AGENT_PLAYER_SETS,
    LEGEND_YEARS,
    build_legend_profile,
    legend_player_profiles,
)


def build_legend_retrodate_config(
    *,
    results_root: str | Path = Path("Results") / "PanteonLegend",
    data_dir: str | Path = "Retrodate",
    years: Sequence[int] = LEGEND_YEARS,
    max_bars: Optional[int] = None,
    include_optional_agents: bool = False,
    optional_agent_labels: Sequence[str] = (),
) -> RetrodateMarketConfig:
    profile = build_legend_profile()
    return RetrodateMarketConfig(
        data_dir=Path(data_dir),
        results_root=Path(results_root),
        years=tuple(int(year) for year in years),
        timeframe="1m",
        stride_minutes=60,
        risk_capital_fraction=profile.trade_fraction,
        risk_max_leverage=profile.leverage,
        apply_risk_leverage_to_notional=profile.apply_leverage_to_notional,
        include_optional_agents=include_optional_agents,
        optional_agent_labels=tuple(
            str(label).strip() for label in optional_agent_labels if str(label).strip()
        ),
        max_bars=max_bars,
        flash_enabled=False,
        hard_policy_enabled=False,
        flash_audit_events_enabled=False,
        shadow_audit_events_enabled=True,
        step_result_retention_enabled=False,
        compact_causal_entry_selected_only=True,
        fixed_agent_players_enabled=True,
        fixed_agent_player_sets=LEGEND_FIXED_AGENT_PLAYER_SETS,
        player_profiles=legend_player_profiles(),
        regime_switch_player_sets=LEGEND_REGIME_SWITCH_PLAYER_SETS,
        rotating_agent_player_sets=LEGEND_ROTATING_AGENT_PLAYER_SETS,
        solo_agent_candidate_limit=profile.solo_agent_candidate_limit,
        max_new_opens_per_bar=profile.max_new_opens_per_bar,
        risk_max_open_positions=profile.max_open_positions,
        real_promotion_gate_enabled=False,
        current_actionable_candidate_layer_enabled=profile.current_actionable_gate_enabled,
        use_v3_rolling_score=True,
        use_v3_shadow_rolling_score=False,
        use_v3_soft_shadow_score=True,
        use_v3_executable_soft_top1_score=True,
        v3_candidate_allow_labels=profile.execution_candidate_allow_labels,
        v3_shadow_position_gate_enabled=False,
        v3_shadow_rolling_window_bars=profile.soft_selector_window_bars,
        v3_shadow_rolling_min_closed_trades=profile.soft_selector_min_closed_trades,
        hard_policy_deny_labels=(),
    )


def run_legend_retrodate_benchmark(
    *,
    results_root: str | Path = Path("Results") / "PanteonLegend",
    data_dir: str | Path = "Retrodate",
    years: Sequence[int] = LEGEND_YEARS,
    max_bars: Optional[int] = None,
    include_optional_agents: bool = False,
    optional_agent_labels: Sequence[str] = (),
) -> RetrodateRunSummary:
    config = build_legend_retrodate_config(
        results_root=results_root,
        data_dir=data_dir,
        years=years,
        max_bars=max_bars,
        include_optional_agents=include_optional_agents,
        optional_agent_labels=optional_agent_labels,
    )
    return run_retrodate_market_benchmark(config)
