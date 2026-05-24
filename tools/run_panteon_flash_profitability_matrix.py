from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_REPORTS_DIR = ROOT / "Reports"
DEFAULT_RESULTS_ROOT = ROOT / "Results" / "PanteonFlashProfitabilityMatrix"
MATRIX_FILENAME = "panteon_flash_profitability_matrix.json"
REGIME_FLOOR_MIN_CLOSED_TRADES = 30
CHURN_BUDGET_MAX_ACTOR_SWITCHES_PER_DAY = 24.0

BASELINE_FLASH_ARGS = [
    "--enable-flash",
    "--flash-min-score-to-trade",
    "4.0",
    "--enable-flash-shadow-confirmation",
    "--enable-flash-symbol-shadow-confirmation",
    "--enable-flash-shadow-actor-fallback-confirmation",
    "--enable-flash-shadow-signal-handoff",
    "--flash-shadow-actor-fallback-min-base-score",
    "3.0",
    "--enable-flash-shadow-quality-confirmation",
    "--flash-shadow-min-closed-trades",
    "10",
    "--flash-max-signals-per-actor",
    "3",
    "--enable-flash-degradation-guard",
    "--flash-degradation-reserve-actor-cap",
    "--flash-degradation-window-closed-trades",
    "1",
    "--flash-degradation-min-closed-trades",
    "1",
    "--flash-degradation-max-recent-pnl-usd",
    "-1.0",
    "--flash-promotion-min-full-closed-trades",
    "2",
    "--flash-promotion-min-latest-closed-trades",
    "1",
    "--flash-promotion-min-win-rate-pct",
    "0.0",
]


def _with_arg_value(args: Sequence[str], flag: str, value: str) -> list[str]:
    out = list(args)
    try:
        index = out.index(flag)
    except ValueError:
        out.extend([flag, value])
    else:
        if index + 1 >= len(out):
            out.append(value)
        else:
            out[index + 1] = value
    return out


def _without_flag(args: Sequence[str], flag: str) -> list[str]:
    return [arg for arg in args if arg != flag]


PROVEN_SOLO_POSITION_HANDOFF_AGE24_SHADOW5_ARGS = _with_arg_value(
    BASELINE_FLASH_ARGS,
    "--flash-shadow-min-closed-trades",
    "5",
) + [
    "--enable-flash-prefer-proven-solo-player-wrappers",
    "--flash-proven-solo-min-score-advantage",
    "0.0",
    "--enable-v3-shadow-fresh-handoff",
    "--v3-shadow-fresh-handoff-max-age-bars",
    "24",
]

PROVEN_SOLO_STALE_POSITION_EXIT_AGE168_SHADOW5_ARGS = (
    PROVEN_SOLO_POSITION_HANDOFF_AGE24_SHADOW5_ARGS
    + [
        "--enable-flash-stale-position-exit",
        "--flash-stale-position-exit-max-age-bars",
        "168",
    ]
)

PROVEN_SOLO_STALE_EXIT_QUALITY60_AGE168_SHADOW5_ARGS = _with_arg_value(
    PROVEN_SOLO_STALE_POSITION_EXIT_AGE168_SHADOW5_ARGS,
    "--flash-shadow-min-win-rate-pct",
    "60.0",
)

PROVEN_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS = (
    PROVEN_SOLO_STALE_POSITION_EXIT_AGE168_SHADOW5_ARGS
    + [
        "--v3-shadow-fresh-handoff-allow-nonpositive-unrealized",
    ]
)

PROVEN_SOLO_PORTFOLIO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS = (
    PROVEN_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    + [
        "--flash-portfolio-actor-key",
        "ensemble:Solo_MomentumScalper",
        "--flash-portfolio-actor-key",
        "ensemble:Solo_LiveCrashHunter",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS = (
    PROVEN_SOLO_PORTFOLIO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    + [
        "--enable-flash-prefer-solo-player-wrappers",
        "--flash-portfolio-actor-key",
        "agent:LiveOIBreakout",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    + [
        "--flash-promoted-actor-cap-override",
        "ensemble:Solo_MomentumScalper=10",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--flash-shadow-min-pnl-per-trade-lcb-usd",
        "0.0",
        "--flash-shadow-pnl-per-trade-lcb-z",
        "1.0",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_PENALTY_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--flash-shadow-pnl-per-trade-lcb-penalty-floor-usd",
        "0.0",
        "--flash-shadow-pnl-per-trade-lcb-penalty-weight",
        "20.0",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_MANIFEST_PNL_LCB_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--enable-flash-promotion-manifest",
        "--flash-promotion-min-full-pnl-per-trade-lcb-pct",
        "0.0",
        "--flash-promotion-min-latest-pnl-per-trade-lcb-pct",
        "0.0",
        "--flash-promotion-pnl-per-trade-lcb-z",
        "1.0",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|TRX/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ICP/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ADA/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ATOM/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|APT/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|DOT/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|AVAX/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|BNB/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|XLM/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|SOL/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Antonius_conservative|ADA/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Antonius_conservative|BTC/USDT|FUT_LONG_FULL",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_V2_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_ARGS
    + [
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|LINK/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|NEAR/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ATOM/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|XLM/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|SOL/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_ResearchValidatorAgent|ADA/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "agent:PlayerFunding|ETH/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|DOT/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|LINK/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|XRP/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|ETH/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|BNB/USDT|SPOT_BUY_FULL",
        "--flash-deny-signal-key",
        "ensemble:Optimal_StaticRotator|APT/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "agent:FundingArb|ETC/USDT|FUT_SHORT_HALF",
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|TRX/USDT|FUT_SHORT_FULL",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_NO_NEUTRAL_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--flash-denied-open-regime",
        "neutral",
    ]
)

PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_WIDER_EXECUTION_ARGS = (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    + [
        "--max-new-opens-per-bar",
        "3",
        "--risk-max-open-positions",
        "16",
    ]
)

SOLO_HARD_ACTOR_GUARD_AGE24_SHADOW5_ARGS = (
    PROVEN_SOLO_POSITION_HANDOFF_AGE24_SHADOW5_ARGS
    + [
        "--enable-flash-prefer-solo-player-wrappers",
        "--enable-flash-degradation-actor-guard",
    ]
)

SYMBOL_ONLY_HARD_SOLO_ACTOR_GUARD_AGE24_SHADOW5_ARGS = _without_flag(
    SOLO_HARD_ACTOR_GUARD_AGE24_SHADOW5_ARGS,
    "--enable-flash-shadow-actor-fallback-confirmation",
)

SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS = (
    SYMBOL_ONLY_HARD_SOLO_ACTOR_GUARD_AGE24_SHADOW5_ARGS
    + [
        "--enable-flash-shadow-base-fallback-confirmation",
        "--flash-shadow-base-fallback-actor-key",
        "ensemble:Solo_MomentumScalper",
    ]
)

SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_MIN1_AGE24_SHADOW5_ARGS = _with_arg_value(
    SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS,
    "--flash-shadow-actor-fallback-min-base-score",
    "1.0",
)

SYMBOL_ONLY_ANCHOR_MIN_SCORE_2_5_AGE24_SHADOW5_ARGS = _with_arg_value(
    SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS,
    "--flash-min-score-to-trade",
    "2.5",
)

SYMBOL_ONLY_ANCHOR_DOMINANCE_AGE24_SHADOW5_ARGS = (
    SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS
    + [
        "--flash-anchor-actor-key",
        "ensemble:Solo_MomentumScalper",
        "--flash-anchor-min-score-to-trade",
        "1.0",
        "--flash-anchor-min-score-advantage",
        "1.0",
    ]
)

SYMBOL_ONLY_ANCHOR_DOMINANCE_NO_ACTOR_GUARD_AGE24_SHADOW5_ARGS = _without_flag(
    SYMBOL_ONLY_ANCHOR_DOMINANCE_AGE24_SHADOW5_ARGS,
    "--enable-flash-degradation-actor-guard",
)

SYMBOL_ONLY_ANCHOR_SHADOW_FLOOR_AGE24_SHADOW5_ARGS = (
    SYMBOL_ONLY_ANCHOR_DOMINANCE_AGE24_SHADOW5_ARGS
    + [
        "--flash-anchor-shadow-min-score",
        "-0.1",
    ]
)

SYMBOL_ONLY_ANCHOR_ACTOR_COOLDOWN_AGE24_SHADOW5_ARGS = (
    SYMBOL_ONLY_ANCHOR_SHADOW_FLOOR_AGE24_SHADOW5_ARGS
    + [
        "--flash-degradation-actor-cooldown-bars",
        "72",
    ]
)

SYMBOL_ONLY_ANCHOR_ACTOR_REGIME_GUARD_AGE24_SHADOW5_ARGS = (
    SYMBOL_ONLY_ANCHOR_SHADOW_FLOOR_AGE24_SHADOW5_ARGS
    + [
        "--flash-degradation-actor-scope",
        "actor_regime",
    ]
)


EXPERIMENTS = [
    {
        "name": "baseline_reserve_cap",
        "args": BASELINE_FLASH_ARGS,
    },
    {
        "name": "proven_solo_position_handoff_age24_shadow5",
        "args": PROVEN_SOLO_POSITION_HANDOFF_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_stale_position_exit_age168_shadow5",
        "args": PROVEN_SOLO_STALE_POSITION_EXIT_AGE168_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_stale_exit_quality60_age168_shadow5",
        "args": PROVEN_SOLO_STALE_EXIT_QUALITY60_AGE168_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_nonpositive_handoff_stale_exit_age168_shadow5",
        "args": PROVEN_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_portfolio_nonpositive_handoff_stale_exit_age168_shadow5",
        "args": PROVEN_SOLO_PORTFOLIO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_nonpositive_handoff_stale_exit_age168_shadow5",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_shadow_pnl_lcb",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_shadow_pnl_lcb_penalty",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_PENALTY_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_manifest_pnl_lcb",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_MANIFEST_PNL_LCB_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_selected_deny_probe",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_selected_deny_probe_v2",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_V2_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
        "selected_deny_source_experiment": "proven_solo_portfolio_hard_solo_momentum_cap10",
        "selected_deny_max_realized_pnl_usd": -0.1,
        "selected_deny_min_closed_trades": 1,
        "selected_deny_max_keys": 50,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
        "selected_deny_source_experiments": [
            "proven_solo_portfolio_hard_solo_momentum_cap10",
            "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
        ],
        "selected_deny_max_realized_pnl_usd": -0.1,
        "selected_deny_min_closed_trades": 1,
        "selected_deny_max_keys": 50,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_oos2025",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
        "selected_deny_source_experiments": [
            "proven_solo_portfolio_hard_solo_momentum_cap10",
            "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
        ],
        "selected_deny_source_tier": "2025",
        "selected_deny_max_realized_pnl_usd": -0.1,
        "selected_deny_min_closed_trades": 1,
        "selected_deny_max_keys": 50,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_terminal_atom",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
        + [
            "--flash-terminal-deny-signal-key",
            "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
        ],
        "selected_deny_source_experiments": [
            "proven_solo_portfolio_hard_solo_momentum_cap10",
            "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
        ],
        "selected_deny_source_tier": "2025",
        "selected_deny_max_realized_pnl_usd": -0.1,
        "selected_deny_min_closed_trades": 1,
        "selected_deny_max_keys": 50,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_round2_terminal_atom_signal_cooldown720",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
        + [
            "--flash-terminal-deny-signal-key",
            "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
            "--flash-degradation-signal-cooldown-bars",
            "720",
        ],
        "selected_deny_source_experiments": [
            "proven_solo_portfolio_hard_solo_momentum_cap10",
            "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
        ],
        "selected_deny_source_tier": "2025",
        "selected_deny_max_realized_pnl_usd": -0.1,
        "selected_deny_min_closed_trades": 1,
        "selected_deny_max_keys": 50,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_no_neutral_open",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_NO_NEUTRAL_ARGS,
    },
    {
        "name": "proven_solo_portfolio_hard_solo_momentum_cap10_wider_execution",
        "args": PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_WIDER_EXECUTION_ARGS,
    },
    {
        "name": "solo_hard_actor_guard_age24_shadow5",
        "args": SOLO_HARD_ACTOR_GUARD_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_hard_solo_actor_guard_age24_shadow5",
        "args": SYMBOL_ONLY_HARD_SOLO_ACTOR_GUARD_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_base_fallback_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_base_fallback_min1_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_MIN1_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_min_score_2_5_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_MIN_SCORE_2_5_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_dominance_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_DOMINANCE_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_dominance_no_actor_guard_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_DOMINANCE_NO_ACTOR_GUARD_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_shadow_floor_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_SHADOW_FLOOR_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_actor_cooldown_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_ACTOR_COOLDOWN_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "symbol_only_anchor_actor_regime_guard_age24_shadow5",
        "args": SYMBOL_ONLY_ANCHOR_ACTOR_REGIME_GUARD_AGE24_SHADOW5_ARGS,
    },
    {
        "name": "manifest_cap1",
        "args": [
            "--enable-flash",
            "--flash-min-score-to-trade",
            "4.0",
            "--enable-flash-shadow-confirmation",
            "--enable-flash-symbol-shadow-confirmation",
            "--enable-flash-shadow-actor-fallback-confirmation",
            "--enable-flash-shadow-signal-handoff",
            "--flash-shadow-actor-fallback-min-base-score",
            "3.0",
            "--enable-flash-shadow-quality-confirmation",
            "--flash-shadow-min-closed-trades",
            "10",
            "--enable-flash-promotion-manifest",
            "--flash-max-signals-per-actor",
            "3",
            "--enable-flash-degradation-guard",
            "--flash-degradation-reserve-actor-cap",
            "--flash-degradation-window-closed-trades",
            "1",
            "--flash-degradation-min-closed-trades",
            "1",
            "--flash-degradation-max-recent-pnl-usd",
            "-1.0",
        ],
    },
    {
        "name": "manifest_earned_cap2",
        "args": [
            "--enable-flash",
            "--flash-min-score-to-trade",
            "4.0",
            "--enable-flash-shadow-confirmation",
            "--enable-flash-symbol-shadow-confirmation",
            "--enable-flash-shadow-actor-fallback-confirmation",
            "--enable-flash-shadow-signal-handoff",
            "--flash-shadow-actor-fallback-min-base-score",
            "3.0",
            "--enable-flash-shadow-quality-confirmation",
            "--flash-shadow-min-closed-trades",
            "10",
            "--enable-flash-promotion-manifest",
            "--flash-max-signals-per-actor",
            "3",
            "--enable-flash-degradation-guard",
            "--flash-degradation-reserve-actor-cap",
            "--flash-degradation-window-closed-trades",
            "1",
            "--flash-degradation-min-closed-trades",
            "1",
            "--flash-degradation-max-recent-pnl-usd",
            "-1.0",
            "--enable-flash-earned-cap-overrides",
        ],
    },
]


def _overextension_args(
    *,
    normalized: bool,
    short_floor: str = "-8.0",
    long_ceiling: str = "8.0",
    z: str = "2.5",
) -> list[str]:
    args = [
        "--enable-flash-overextension-guard",
        "--flash-short-overextension-return-floor-pct",
        short_floor,
        "--flash-long-overextension-return-ceiling-pct",
        long_ceiling,
    ]
    if normalized:
        args.extend([
            "--enable-flash-overextension-volatility-normalized",
            "--flash-short-overextension-z-floor",
            f"-{z}",
            "--flash-long-overextension-z-ceiling",
            z,
        ])
    return args


OVEREXTENSION_CALIBRATION_EXPERIMENTS = [
    {
        "name": "baseline_no_overextension",
        "args": BASELINE_FLASH_ARGS,
    },
    {
        "name": "raw_overextension_8pct",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=False),
    },
    {
        "name": "vol_z_2_0",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="2.0"),
    },
    {
        "name": "vol_z_2_5",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="2.5"),
    },
    {
        "name": "vol_z_3_0",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="3.0"),
    },
    {
        "name": "vol_z_4_0",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="4.0"),
    },
    {
        "name": "vol_z_5_0",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="5.0"),
    },
    {
        "name": "vol_z_6_0",
        "args": BASELINE_FLASH_ARGS + _overextension_args(normalized=True, z="6.0"),
    },
]


TIERS = [
    ("2026_h1", ["--years", "2026", "--max-bars", "3600"]),
    ("2025", ["--years", "2025"]),
    ("full_2022_2026", ["--years", "2022", "2023", "2024", "2025", "2026"]),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_runner_args(args: Sequence[str]) -> list[str]:
    """Convert plan-style multi-token --years into retro runner comma syntax."""
    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        item = args[idx]
        if item != "--years":
            normalized.append(item)
            idx += 1
            continue

        years: list[str] = []
        idx += 1
        while idx < len(args) and not str(args[idx]).startswith("--"):
            years.append(str(args[idx]))
            idx += 1
        normalized.extend(["--years", ",".join(years)])
    return normalized


def _runner_command(
    *,
    python: str,
    runner: str,
    results_root: Path,
    experiment_name: str,
    experiment_args: Sequence[str],
    tier_args: Sequence[str],
) -> list[str]:
    runner_target = Path(runner)
    if runner_target.suffix == ".py" or runner_target.exists():
        command = [python, str(runner_target)]
    else:
        command = [python, "-m", runner]

    output_root = results_root / experiment_name
    command.extend(["--results-root", str(output_root)])
    command.extend(_normalize_runner_args(tier_args))
    command.extend(_normalize_runner_args(experiment_args))
    return command


def _find_declared_output_dir(stdout: str, stderr: str) -> Path | None:
    combined = "\n".join([stdout or "", stderr or ""])
    for pattern in (r"^output_dir=(.+)$", r'"output_dir"\s*:\s*"([^"]+)"'):
        match = re.search(pattern, combined, flags=re.MULTILINE)
        if match:
            path = Path(match.group(1).strip())
            return path if path.is_absolute() else ROOT / path
    return None


def _newest_output_dir(results_root: Path, started_at: float) -> Path | None:
    candidates: list[Path] = []
    if not results_root.exists():
        return None
    for summary_path in results_root.rglob("run_summary.json"):
        try:
            if summary_path.stat().st_mtime + 1.0 >= started_at:
                candidates.append(summary_path.parent)
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_metrics(output_dir: Path | None) -> dict[str, Any]:
    status = _load_json(output_dir / "status.json") if output_dir else {}
    benchmark = (
        _load_json(output_dir / "component_benchmark_report.json")
        if output_dir
        else {}
    )
    allocation = (
        _load_json(output_dir / "allocation_diagnostics.json")
        if output_dir
        else {}
    )
    flash_attribution = (
        _load_json(output_dir / "flash_attribution_summary.json")
        if output_dir
        else {}
    )
    standalone_vs_flash = (
        _load_json(output_dir / "standalone_vs_flash_selected_report.json")
        if output_dir
        else {}
    )
    walk_forward = (
        _load_json(output_dir / "walk_forward_report.json")
        if output_dir
        else {}
    )
    run_summary = _load_json(output_dir / "run_summary.json") if output_dir else {}

    live = status.get("live_session", {})
    if not isinstance(live, dict):
        live = {}
    benchmark_summary = benchmark.get("summary", {})
    if not isinstance(benchmark_summary, dict):
        benchmark_summary = {}
    benchmark_components = benchmark.get("components", [])
    if not isinstance(benchmark_components, list):
        benchmark_components = []
    flash_summary = flash_attribution.get("summary", {})
    if not isinstance(flash_summary, dict):
        flash_summary = {}
    standalone_vs_flash_summary = standalone_vs_flash.get("summary", {})
    if not isinstance(standalone_vs_flash_summary, dict):
        standalone_vs_flash_summary = {}

    pnl_pct = _float_or_zero(
        live.get("panteon_owned_pnl_pct", status.get("panteon_owned_pnl_pct"))
    )
    max_drawdown_pct = _float_or_zero(
        status.get(
            "panteon_max_drawdown_pct",
            live.get("panteon_max_drawdown_pct", status.get("max_drawdown_pct")),
        )
    )
    alpha_pct = _float_or_zero(benchmark_summary.get("panteon_alpha_pct"))
    beats = bool(benchmark_summary.get("panteon_beats_best_component", False))
    best_component_pct = _float_or_zero(
        benchmark_summary.get("best_component_pnl_pct")
    )
    flash_decisions = int(flash_summary.get("flash_decisions", 0) or 0)
    no_trade_decisions = int(flash_summary.get("no_trade_decisions", 0) or 0)
    flash_selected_signals = _int_or_zero(flash_summary.get("selected_signals"))
    flash_executable_selected_signals = _int_or_zero(
        flash_summary.get("executable_selected_signals")
    )
    flash_filtered_before_execution = _int_or_zero(
        flash_summary.get("selected_filtered_before_execution")
    )
    flash_filled_signals = _int_or_zero(flash_summary.get("filled_signals"))
    flash_filter_detail_counts = flash_summary.get("filter_detail_counts", {})
    if not isinstance(flash_filter_detail_counts, dict):
        flash_filter_detail_counts = {}
    switch_metrics = _actor_switch_metrics(output_dir)
    regime_floor = _regime_floor_metrics(walk_forward)
    ensemble_lift = _ensemble_lift_metrics(benchmark_components)
    churn_budget = _churn_budget_metrics(switch_metrics)

    return {
        "pnl_pct": pnl_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "panteon_alpha_pct": alpha_pct,
        "beats_best_component": beats,
        "best_component_pnl_pct": best_component_pct,
        "panteon_advantage": _panteon_advantage(alpha_pct, max_drawdown_pct),
        "dominance_equity_ratio": _dominance_equity_ratio(
            pnl_pct,
            best_component_pct,
        ),
        "panteon_lcb_dominance": _panteon_lcb_dominance(
            pnl_pct,
            best_component_pct,
        ),
        "no_trade_decision_share_pct": _share_pct(
            no_trade_decisions,
            flash_decisions,
        ),
        "flash_selected_signals": flash_selected_signals,
        "flash_executable_selected_signals": flash_executable_selected_signals,
        "flash_selected_filtered_before_execution": flash_filtered_before_execution,
        "flash_missing_execution_signals": _int_or_zero(
            flash_summary.get("missing_execution_signals")
        ),
        "flash_filtered_before_execution_share_pct": _share_pct(
            flash_filtered_before_execution,
            flash_selected_signals,
        ),
        "flash_executable_fill_rate_pct": _share_pct(
            flash_filled_signals,
            flash_executable_selected_signals,
        ),
        "flash_filter_detail_counts": dict(sorted(flash_filter_detail_counts.items())),
        "allocation_no_trade_share_pct": _float_or_zero(
            allocation.get("no_trade_share_pct")
        ),
        "standalone_vs_flash_selection_alpha_pct": _float_or_zero(
            standalone_vs_flash_summary.get("total_selection_alpha_pct")
        ),
        "standalone_vs_flash_standalone_pnl_pct": _float_or_zero(
            standalone_vs_flash_summary.get("total_standalone_pnl_pct")
        ),
        "standalone_vs_flash_selected_pnl_pct": _float_or_zero(
            standalone_vs_flash_summary.get("total_flash_selected_pnl_pct")
        ),
        **switch_metrics,
        **churn_budget,
        **regime_floor,
        **ensemble_lift,
        "output_dir": _display_path(output_dir) if output_dir else "",
        "bars_processed": int(run_summary.get("bars_processed", 0) or 0),
    }


def _should_stop_early(tier_name: str, row: dict[str, Any]) -> str:
    if row.get("beats_best_component") is False:
        return f"panteon_under_best_component_on_{tier_name}"
    if row.get("panteon_lcb_dominance") is False:
        return f"panteon_lcb_dominance_failed_on_{tier_name}"
    if row.get("ensemble_lift_pass") is False:
        return f"ensemble_lift_failed_on_{tier_name}"
    if row.get("regime_floor_pass") is False:
        return f"regime_floor_failed_on_{tier_name}"
    if row.get("churn_budget_pass") is False:
        return f"churn_budget_failed_on_{tier_name}"
    if (
        _float_or_zero(row.get("pnl_pct")) < 0.0
        and _float_or_zero(row.get("max_drawdown_pct")) > 10.0
    ):
        return "negative_pnl_and_drawdown_above_10pct"
    return ""


def _panteon_advantage(alpha_pct: float, max_drawdown_pct: float) -> float:
    dd = abs(float(max_drawdown_pct or 0.0))
    if dd <= 0.0:
        return 0.0
    return float(alpha_pct or 0.0) / dd


def _dominance_equity_ratio(panteon_pct: float, best_pct: float) -> float:
    best_equity = 100.0 + float(best_pct or 0.0)
    if best_equity <= 0.0:
        return 0.0
    return (100.0 + float(panteon_pct or 0.0)) / best_equity


def _panteon_lcb_dominance(panteon_pct: float, best_pct: float) -> bool:
    best_equity = 100.0 + float(best_pct or 0.0)
    panteon_equity = 100.0 + float(panteon_pct or 0.0)
    if best_equity <= 0.0:
        return panteon_equity >= best_equity
    return panteon_equity >= 0.95 * best_equity


def _share_pct(part: int, total: int) -> float:
    return (float(part) / float(total) * 100.0) if total > 0 else 0.0


def _actor_switch_metrics(output_dir: Path | None) -> dict[str, Any]:
    path = output_dir / "causal_entry_decisions.jsonl" if output_dir else None
    previous_by_symbol: dict[str, str] = {}
    switches = 0
    bars = 0
    if path is None or not path.exists():
        return {
            "actor_switches": 0,
            "actor_switches_per_day": 0.0,
        }
    for row in _iter_jsonl(path):
        selected = row.get("flash_selected_actors_by_symbol")
        if not isinstance(selected, dict):
            continue
        bars += 1
        for raw_symbol, raw_actor in selected.items():
            symbol = str(raw_symbol or "").upper()
            actor = str(raw_actor or "")
            if not symbol or not actor:
                continue
            previous = previous_by_symbol.get(symbol)
            if previous is not None and previous != actor:
                switches += 1
            previous_by_symbol[symbol] = actor
    days = max(float(bars) / 24.0, 1.0 / 24.0)
    return {
        "actor_switches": switches,
        "actor_switches_per_day": switches / days,
    }


def _churn_budget_metrics(switch_metrics: Mapping[str, Any]) -> dict[str, Any]:
    switches_per_day = _float_or_zero(switch_metrics.get("actor_switches_per_day"))
    return {
        "churn_budget_max_actor_switches_per_day": (
            CHURN_BUDGET_MAX_ACTOR_SWITCHES_PER_DAY
        ),
        "churn_budget_pass": (
            switches_per_day <= CHURN_BUDGET_MAX_ACTOR_SWITCHES_PER_DAY
        ),
    }


def _regime_floor_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    by_regime = report.get("by_regime") if isinstance(report, Mapping) else {}
    if not isinstance(by_regime, dict) or not by_regime:
        return {
            "regime_floor_pass": None,
            "regime_floor_failed_regimes": [],
            "regime_floor_insufficient_regimes": [],
            "regime_floor_min_closed_trades": REGIME_FLOOR_MIN_CLOSED_TRADES,
        }
    failed: list[str] = []
    insufficient: list[str] = []
    checked = 0
    for regime, payload in by_regime.items():
        if not isinstance(payload, dict):
            continue
        closed = int(payload.get("closed_trades", 0) or 0)
        if closed <= 0:
            continue
        if closed < REGIME_FLOOR_MIN_CLOSED_TRADES:
            insufficient.append(str(regime))
            continue
        checked += 1
        if _float_or_zero(payload.get("net_pnl")) < 0.0:
            failed.append(str(regime))
    return {
        "regime_floor_pass": (not failed) if checked > 0 else None,
        "regime_floor_failed_regimes": sorted(failed),
        "regime_floor_insufficient_regimes": sorted(insufficient),
        "regime_floor_min_closed_trades": REGIME_FLOOR_MIN_CLOSED_TRADES,
    }


def _ensemble_lift_metrics(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_agent = None
    best_player = None
    for row in components:
        if not isinstance(row, Mapping):
            continue
        actor_type = str(row.get("actor_type") or "")
        pnl_pct = _float_or_zero(row.get("pnl_pct"))
        if actor_type == "agent":
            best_agent = pnl_pct if best_agent is None else max(best_agent, pnl_pct)
        elif actor_type == "player":
            best_player = pnl_pct if best_player is None else max(best_player, pnl_pct)
    if best_agent is None or best_player is None:
        return {
            "ensemble_lift_pct": 0.0,
            "ensemble_lift_pass": None,
        }
    lift = float(best_player) - float(best_agent)
    return {
        "ensemble_lift_pct": lift,
        "ensemble_lift_pass": lift >= 0.0,
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _flash_selected_deny_key_args_from_report(
    report: Mapping[str, Any],
    *,
    max_realized_pnl_usd: float = -0.1,
    min_closed_trades: int = 1,
    max_keys: int = 50,
) -> list[str]:
    rows = report.get("rows") if isinstance(report, Mapping) else None
    if not isinstance(rows, list) or max_keys <= 0:
        return []

    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        actor_key = str(row.get("actor_key") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        action = str(row.get("action") or "").strip().upper()
        if not actor_key or not symbol or not action:
            continue
        closed_trades = _int_or_zero(row.get("closed_trades"))
        if closed_trades < min_closed_trades:
            continue
        realized_pnl_usd = _float_or_zero(row.get("realized_pnl_usd"))
        if (
            not math.isfinite(realized_pnl_usd)
            or realized_pnl_usd > max_realized_pnl_usd
        ):
            continue
        signal_key = f"{actor_key}|{symbol}|{action}"
        if signal_key in seen:
            continue
        seen.add(signal_key)
        candidates.append((realized_pnl_usd, signal_key))

    out: list[str] = []
    for _pnl, signal_key in sorted(candidates)[:max_keys]:
        out.extend(["--flash-deny-signal-key", signal_key])
    return out


def _signal_key_from_shadow_row(row: Mapping[str, Any]) -> str:
    signal_key = str(row.get("signal_key") or "").strip()
    if signal_key:
        parts = [part.strip() for part in signal_key.split("|")]
        if len(parts) == 3 and all(parts):
            return f"{parts[0]}|{parts[1].upper()}|{parts[2].upper()}"
    return _signal_key_from_attribution_row(row)


def _actor_key_from_signal_key(signal_key: str) -> str:
    return signal_key.split("|", 1)[0] if "|" in signal_key else ""


def _flash_shadow_deny_key_args_from_report(
    report: Mapping[str, Any],
    *,
    max_full_pnl_usd: float = -0.1,
    min_full_closed_trades: int = 2,
    max_keys: int = 50,
    allowed_actor_keys: Sequence[str] = (),
    allowed_signal_keys: Sequence[str] = (),
) -> list[str]:
    rows = report.get("rows") if isinstance(report, Mapping) else None
    if not isinstance(rows, list) or max_keys <= 0:
        return []

    allowed = {
        str(actor_key).strip()
        for actor_key in allowed_actor_keys
        if str(actor_key).strip()
    }
    allowed_signals = {
        str(signal_key).strip()
        for signal_key in allowed_signal_keys
        if str(signal_key).strip()
    }
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        signal_key = _signal_key_from_shadow_row(row)
        if not signal_key:
            continue
        if allowed_signals and signal_key not in allowed_signals:
            continue
        actor_key = str(row.get("actor_key") or "").strip()
        if not actor_key:
            actor_key = _actor_key_from_signal_key(signal_key)
        if allowed and actor_key not in allowed:
            continue
        closed_trades = _int_or_zero(row.get("full_closed_trades"))
        if closed_trades < min_full_closed_trades:
            continue
        full_pnl_usd = _float_or_zero(row.get("full_pnl_usd"))
        if not math.isfinite(full_pnl_usd) or full_pnl_usd > max_full_pnl_usd:
            continue
        if signal_key in seen:
            continue
        seen.add(signal_key)
        candidates.append((full_pnl_usd, signal_key))

    out: list[str] = []
    for _pnl, signal_key in sorted(candidates)[:max_keys]:
        out.extend(["--flash-deny-signal-key", signal_key])
    return out


def _signal_key_from_attribution_row(row: Mapping[str, Any]) -> str:
    actor_key = str(row.get("actor_key") or "").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    action = str(row.get("action") or "").strip().upper()
    if not actor_key or not symbol or not action:
        return ""
    return f"{actor_key}|{symbol}|{action}"


def _signal_key_stats_from_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for report in reports:
        rows = report.get("rows") if isinstance(report, Mapping) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            signal_key = _signal_key_from_attribution_row(row)
            if not signal_key:
                continue
            actor_key, symbol, action = signal_key.split("|", 2)
            item = stats.setdefault(
                signal_key,
                {
                    "signal_key": signal_key,
                    "actor_key": actor_key,
                    "symbol": symbol,
                    "action": action,
                    "selected_signals": 0,
                    "closed_trades": 0,
                    "realized_pnl_usd": 0.0,
                },
            )
            item["selected_signals"] += _int_or_zero(row.get("selected_signals"))
            item["closed_trades"] += _int_or_zero(row.get("closed_trades"))
            item["realized_pnl_usd"] += _float_or_zero(row.get("realized_pnl_usd"))
    return stats


def _signal_key_train_eval_rows(
    train_reports: Sequence[Mapping[str, Any]],
    eval_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train_stats = _signal_key_stats_from_reports(train_reports)
    eval_stats = _signal_key_stats_from_reports([eval_report])
    rows: list[dict[str, Any]] = []
    for signal_key, eval_item in eval_stats.items():
        train_item = train_stats.get(signal_key, {})
        train_closed = _int_or_zero(train_item.get("closed_trades"))
        eval_closed = _int_or_zero(eval_item.get("closed_trades"))
        train_pnl = _float_or_zero(train_item.get("realized_pnl_usd"))
        eval_pnl = _float_or_zero(eval_item.get("realized_pnl_usd"))
        rows.append(
            {
                "signal_key": signal_key,
                "actor_key": str(eval_item.get("actor_key") or ""),
                "symbol": str(eval_item.get("symbol") or ""),
                "action": str(eval_item.get("action") or ""),
                "train_selected_signals": _int_or_zero(
                    train_item.get("selected_signals")
                ),
                "train_closed_trades": train_closed,
                "train_realized_pnl_usd": train_pnl,
                "train_pnl_per_trade_usd": (
                    train_pnl / train_closed if train_closed > 0 else 0.0
                ),
                "eval_selected_signals": _int_or_zero(
                    eval_item.get("selected_signals")
                ),
                "eval_closed_trades": eval_closed,
                "eval_realized_pnl_usd": eval_pnl,
                "eval_pnl_per_trade_usd": (
                    eval_pnl / eval_closed if eval_closed > 0 else 0.0
                ),
                "pnl_delta_usd": eval_pnl - train_pnl,
                "supported_by_train": train_closed > 0,
            }
        )
    return sorted(rows, key=lambda row: (_float_or_zero(row["eval_realized_pnl_usd"]), row["signal_key"]))


def _generated_selected_deny_args(
    experiment: Mapping[str, Any],
    *,
    tier_name: str,
    output_dirs_by_experiment_tier: Mapping[tuple[str, str], Path],
) -> tuple[list[str], str]:
    raw_sources = experiment.get("selected_deny_source_experiments")
    source_experiments: list[str] = []
    if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, str):
        source_experiments.extend(str(source) for source in raw_sources if source)
    source_experiment = str(experiment.get("selected_deny_source_experiment") or "")
    if source_experiment:
        source_experiments.insert(0, source_experiment)
    source_experiments = list(dict.fromkeys(source_experiments))
    if not source_experiments:
        return [], ""

    source_tier = str(experiment.get("selected_deny_source_tier") or tier_name)
    warnings: list[str] = []
    seen: set[str] = set()
    deny_args: list[str] = []
    loaded_sources = 0
    for source in source_experiments:
        source_output_dir = output_dirs_by_experiment_tier.get((source, source_tier))
        if source_output_dir is None:
            warnings.append(f"selected_deny_source_missing:{source}:{source_tier}")
            continue

        loaded_sources += 1
        source_attribution_report = _load_json(
            source_output_dir / "flash_attribution_summary.json"
        )
        source_deny_args = _flash_selected_deny_key_args_from_report(
            source_attribution_report,
            max_realized_pnl_usd=_float_or_zero(
                experiment.get("selected_deny_max_realized_pnl_usd", -0.1)
            ),
            min_closed_trades=max(
                0,
                _int_or_zero(experiment.get("selected_deny_min_closed_trades", 1)),
            ),
            max_keys=max(0, _int_or_zero(experiment.get("selected_deny_max_keys", 50))),
        )
        for signal_key in source_deny_args[1::2]:
            if signal_key in seen:
                continue
            seen.add(signal_key)
            deny_args.extend(["--flash-deny-signal-key", signal_key])
        if bool(experiment.get("selected_deny_include_shadow_report")):
            raw_allowed_actor_keys = experiment.get("selected_deny_shadow_actor_keys", ())
            allowed_actor_keys: list[str] = []
            if isinstance(raw_allowed_actor_keys, Sequence) and not isinstance(
                raw_allowed_actor_keys,
                str,
            ):
                allowed_actor_keys = [
                    str(actor_key)
                    for actor_key in raw_allowed_actor_keys
                    if str(actor_key).strip()
                ]
            allowed_signal_keys: list[str] = []
            if bool(experiment.get("selected_deny_shadow_require_selected_history")):
                source_rows = (
                    source_attribution_report.get("rows")
                    if isinstance(source_attribution_report, Mapping)
                    else None
                )
                if isinstance(source_rows, list):
                    allowed_signal_keys = [
                        signal_key
                        for signal_key in (
                            _signal_key_from_attribution_row(row)
                            for row in source_rows
                            if isinstance(row, Mapping)
                        )
                        if signal_key
                    ]
            shadow_path = source_output_dir / "flash_signal_key_shadow_report.json"
            if not shadow_path.exists():
                warnings.append(
                    f"selected_deny_shadow_report_missing:{source}:{source_tier}"
                )
            else:
                source_shadow_deny_args = _flash_shadow_deny_key_args_from_report(
                    _load_json(shadow_path),
                    max_full_pnl_usd=_float_or_zero(
                        experiment.get("selected_deny_shadow_max_full_pnl_usd", -0.1)
                    ),
                    min_full_closed_trades=max(
                        0,
                        _int_or_zero(
                            experiment.get(
                                "selected_deny_shadow_min_full_closed_trades",
                                2,
                            )
                        ),
                    ),
                    max_keys=max(
                        0,
                        _int_or_zero(
                            experiment.get("selected_deny_shadow_max_keys", 50)
                        ),
                    ),
                    allowed_actor_keys=allowed_actor_keys,
                    allowed_signal_keys=allowed_signal_keys,
                )
                for signal_key in source_shadow_deny_args[1::2]:
                    if signal_key in seen:
                        continue
                    seen.add(signal_key)
                    deny_args.extend(["--flash-deny-signal-key", signal_key])
    if loaded_sources > 0 and not deny_args:
        warnings.append(
            f"selected_deny_keys_empty:{','.join(source_experiments)}:{source_tier}"
        )
    if warnings:
        return deny_args, ";".join(warnings)
    return deny_args, ""


def _manifest_experiment_needs_baseline(experiment_name: str) -> bool:
    return bool(_manifest_baseline_experiment(experiment_name))


def _manifest_baseline_experiment(experiment_name: str) -> str:
    if experiment_name in {
        "manifest_cap1",
        "manifest_earned_cap2",
    }:
        return "baseline_reserve_cap"
    if experiment_name == "proven_solo_portfolio_hard_solo_momentum_cap10_manifest_pnl_lcb":
        return "proven_solo_portfolio_hard_solo_momentum_cap10"
    return ""


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_matrix(reports_dir: Path, rows: list[dict[str, Any]]) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / MATRIX_FILENAME
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _smoke_rows(results_root: Path) -> list[dict[str, Any]]:
    experiment = EXPERIMENTS[0]
    tier_name, _tier_args = TIERS[0]
    return [
        {
            "name": experiment["name"],
            "tier": tier_name,
            "pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "panteon_alpha_pct": 0.0,
            "beats_best_component": False,
            "output_dir": _display_path(results_root / "SMOKE"),
            "smoke": True,
        }
    ]


def _dry_run_rows(
    results_root: Path,
    *,
    experiments: Sequence[dict[str, Any]] = EXPERIMENTS,
    tiers: Sequence[tuple[str, list[str]]] = TIERS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for tier_name, _tier_args in tiers:
            rows.append(
                {
                    "name": str(experiment["name"]),
                    "tier": tier_name,
                    "pnl_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "panteon_alpha_pct": 0.0,
                    "beats_best_component": False,
                    "output_dir": _display_path(results_root / str(experiment["name"])),
                    "dry_run": True,
                }
            )
    return rows


def _run_matrix(
    args: argparse.Namespace,
    *,
    experiments: Sequence[dict[str, Any]] = EXPERIMENTS,
    tiers: Sequence[tuple[str, list[str]]] = TIERS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results_root = Path(args.results_root)
    runner = str(args.runner)
    python = str(args.python)
    output_dirs_by_experiment_tier: dict[tuple[str, str], Path] = {}

    for experiment in experiments:
        stop_reason = ""
        for tier_name, tier_args in tiers:
            if stop_reason:
                break
            experiment_name = str(experiment["name"])
            experiment_args = list(experiment["args"])
            warnings: list[str] = []
            baseline_experiment = _manifest_baseline_experiment(experiment_name)
            if baseline_experiment:
                manifest_path = (
                    output_dirs_by_experiment_tier.get((baseline_experiment, tier_name))
                    / "flash_promotion_manifest.json"
                    if (baseline_experiment, tier_name) in output_dirs_by_experiment_tier
                    else None
                )
                if manifest_path is not None and manifest_path.exists():
                    experiment_args.extend([
                        "--flash-promotion-manifest-path",
                        str(manifest_path),
                    ])
                else:
                    warnings.append("baseline_flash_promotion_manifest_missing")
            generated_deny_args, generated_deny_warning = _generated_selected_deny_args(
                experiment,
                tier_name=tier_name,
                output_dirs_by_experiment_tier=output_dirs_by_experiment_tier,
            )
            if generated_deny_args:
                experiment_args.extend(generated_deny_args)
            if generated_deny_warning:
                warnings.append(generated_deny_warning)
            command = _runner_command(
                python=python,
                runner=runner,
                results_root=results_root,
                experiment_name=experiment_name,
                experiment_args=experiment_args,
                tier_args=tier_args,
            )
            started_at = datetime.now().timestamp()
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(SRC)
                if not env.get("PYTHONPATH")
                else str(SRC) + os.pathsep + str(env["PYTHONPATH"])
            )
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            output_dir = _find_declared_output_dir(proc.stdout, proc.stderr)
            if output_dir is None:
                output_dir = _newest_output_dir(results_root / experiment_name, started_at)
            row = {
                "name": experiment_name,
                "tier": tier_name,
                **_extract_metrics(output_dir),
                "returncode": proc.returncode,
            }
            if generated_deny_args:
                row["generated_selected_deny_keys"] = (
                    len(generated_deny_args) // 2
                )
            if warnings:
                row["warning"] = ";".join(warnings)
            if proc.returncode != 0:
                row["error"] = "runner_failed"
                row["stderr_tail"] = "\n".join((proc.stderr or "").splitlines()[-20:])
                rows.append(row)
                break
            if output_dir is not None:
                output_dirs_by_experiment_tier[(experiment_name, tier_name)] = output_dir
            stop_reason = _should_stop_early(tier_name, row)
            if stop_reason:
                row["stopped_early"] = True
                row["stop_reason"] = stop_reason
                row["walk_forward_gate"] = "failed"
                row["walk_forward_gate_tier"] = tier_name
            rows.append(row)
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run candidate Panteon Flash configs through profitability tiers."
    )
    parser.add_argument("--smoke", action="store_true", help="Write one fake row quickly.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned rows without invoking retrodata.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--runner",
        default="panteon_v2.analysis.retrodate_market_runner",
        help="Python module or .py file for the retrodate runner.",
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--overextension-calibration",
        action="store_true",
        help="Run the short z-threshold sweep for Flash overextension guard.",
    )
    parser.add_argument("--calibration-year", default="2026")
    parser.add_argument("--calibration-max-bars", type=int, default=3600)
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Run only matching experiment name; can be repeated.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        help="Run only matching tier name; can be repeated.",
    )
    return parser.parse_args(argv)


def _selected_experiments(args: argparse.Namespace) -> Sequence[dict[str, Any]]:
    if args.overextension_calibration:
        experiments = OVEREXTENSION_CALIBRATION_EXPERIMENTS
    else:
        experiments = EXPERIMENTS
    selected = {str(name) for name in getattr(args, "experiment", ()) or ()}
    if not selected:
        return experiments
    return [
        experiment
        for experiment in experiments
        if str(experiment.get("name") or "") in selected
    ]


def _selected_tiers(args: argparse.Namespace) -> Sequence[tuple[str, list[str]]]:
    if args.overextension_calibration:
        tiers = [
            (
                "2026_calibration",
                [
                    "--years",
                    str(args.calibration_year),
                    "--max-bars",
                    str(args.calibration_max_bars),
                ],
            )
        ]
    else:
        tiers = TIERS
    selected = {str(name) for name in getattr(args, "tier", ()) or ()}
    if not selected:
        return tiers
    return [tier for tier in tiers if tier[0] in selected]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reports_dir = Path(args.reports_dir)
    results_root = Path(args.results_root)
    experiments = _selected_experiments(args)
    tiers = _selected_tiers(args)

    if args.smoke:
        path = _write_matrix(reports_dir, _smoke_rows(results_root))
    elif args.dry_run:
        path = _write_matrix(
            reports_dir,
            _dry_run_rows(results_root, experiments=experiments, tiers=tiers),
        )
    else:
        path = _write_matrix(
            reports_dir,
            _run_matrix(args, experiments=experiments, tiers=tiers),
        )

    print(f"matrix={_display_path(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
