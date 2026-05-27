from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_panteon_flash_profitability_matrix as matrix  # noqa: E402
from panteon_v2.analysis import retrodate_market_runner as runner  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--risk-capital-fraction", type=float, default=0.10)
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--max-new-opens-per-bar", type=int, default=1)
    parser.add_argument("--risk-max-open-positions", type=int, default=8)
    parser.add_argument("--stale-exit-age-bars", type=int, default=168)
    parser.add_argument("--enable-partial-profit-lock", action="store_true")
    parser.add_argument("--partial-profit-lock-trigger-pnl-pct", type=float, default=1.5)
    parser.add_argument("--partial-profit-lock-close-fraction", type=float, default=0.5)
    parser.add_argument("--partial-profit-lock-min-age-bars", type=int, default=2)
    parser.add_argument(
        "--disable-partial-profit-lock-skip-protected-signal-keys",
        action="store_true",
    )
    parser.add_argument("--partial-profit-lock-skip-signal-key", action="append", default=[])
    parser.add_argument("--context-score-boost", action="append", default=[])
    parser.add_argument("--signal-risk-mult", action="append", default=[])
    parser.add_argument("--context-risk-mult", action="append", default=[])
    parser.add_argument("--experimental-flash-real-actors", action="store_true")
    parser.add_argument("--disable-flash-audit-events", action="store_true")
    parser.add_argument("--disable-shadow-audit-events", action="store_true")
    parser.add_argument("--disable-step-result-retention", action="store_true")
    parser.add_argument("--compact-causal-entry-selected-only", action="store_true")
    parser.add_argument("--shadow-position-diagnostic-bar", action="append", type=int, default=[])
    parser.add_argument("--shadow-position-diagnostic-label", action="append", default=[])
    parser.add_argument(
        "--flash-shadow-actor-fallback-min-base-score",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-shadow-position-replay-actor-fallback-min-base-score",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-shadow-position-replay-actor-fallback-min-shadow-score",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-shadow-min-pnl-per-trade-lcb-usd",
        type=float,
        default=None,
    )
    parser.add_argument("--enable-flash-shadow-pnl-lcb-risk-sizing", action="store_true")
    parser.add_argument(
        "--flash-shadow-pnl-lcb-risk-min-mult",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-shadow-pnl-lcb-risk-floor-usd",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-shadow-pnl-lcb-risk-scale-usd",
        type=float,
        default=None,
    )
    parser.add_argument("--enable-flash-actor-risk-sizing", action="store_true")
    parser.add_argument("--flash-actor-risk-min-mult", type=float, default=None)
    parser.add_argument("--flash-actor-risk-max-mult", type=float, default=None)
    parser.add_argument("--flash-actor-risk-edge-scale-pct", type=float, default=None)
    parser.add_argument("--flash-funding-risk-mult-weight", type=float, default=None)
    parser.add_argument("--flash-funding-risk-mult-cap", type=float, default=None)
    parser.add_argument("--enable-flash-volatility-risk-sizing", action="store_true")
    parser.add_argument("--flash-volatility-risk-target-pct", type=float, default=None)
    parser.add_argument(
        "--flash-volatility-risk-min-volatility-pct",
        type=float,
        default=None,
    )
    parser.add_argument("--flash-volatility-risk-max-mult", type=float, default=None)
    parser.add_argument("--enable-flash-overextension-guard", action="store_true")
    parser.add_argument("--flash-overextension-lookback-bars", type=int, default=None)
    parser.add_argument(
        "--flash-short-overextension-return-floor-pct",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flash-long-overextension-return-ceiling-pct",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--enable-flash-overextension-volatility-normalized",
        action="store_true",
    )
    parser.add_argument("--flash-short-overextension-z-floor", type=float, default=None)
    parser.add_argument("--flash-long-overextension-z-ceiling", type=float, default=None)
    parser.add_argument("--flash-overextension-min-volatility-pct", type=float, default=None)
    parser.add_argument("--enable-flash-degradation-symbol-guard", action="store_true")
    parser.add_argument("--flash-degradation-symbol-cooldown-bars", type=int, default=None)
    parser.add_argument("--flash-degradation-symbol-lookback-bars", type=int, default=None)
    parser.add_argument(
        "--flash-degradation-symbol-window-closed-trades",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flash-degradation-symbol-min-closed-trades",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flash-degradation-symbol-max-recent-pnl-usd",
        type=float,
        default=None,
    )
    parser.add_argument("--enable-flash-degradation-recovery", action="store_true")
    parser.add_argument(
        "--flash-degradation-recovery-min-closed-trades",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flash-degradation-recovery-min-recent-pnl-usd",
        type=float,
        default=None,
    )
    parser.add_argument("--enable-flash-earned-cap-overrides", action="store_true")
    parser.add_argument(
        "--flash-terminal-deny-signal-key",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--flash-terminal-deny-context-signal-key",
        action="append",
        default=[],
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runner_args = [
        "--results-root",
        str(args.results_root),
        "--years",
        str(args.years),
        "--risk-capital-fraction",
        f"{float(args.risk_capital_fraction):.4g}",
        "--stride-minutes",
        str(int(args.stride_minutes)),
        "--max-new-opens-per-bar",
        str(int(args.max_new_opens_per_bar)),
        "--risk-max-open-positions",
        str(int(args.risk_max_open_positions)),
    ]
    if args.max_bars is not None:
        runner_args += ["--max-bars", str(int(args.max_bars))]
    if args.disable_flash_audit_events:
        runner_args += ["--disable-flash-audit-events"]
    if args.disable_shadow_audit_events:
        runner_args += ["--disable-shadow-audit-events"]
    if args.disable_step_result_retention:
        runner_args += ["--disable-step-result-retention"]
    if args.compact_causal_entry_selected_only:
        runner_args += ["--compact-causal-entry-selected-only"]
    for bar in args.shadow_position_diagnostic_bar:
        runner_args += ["--shadow-position-diagnostic-bar", str(int(bar))]
    for label in args.shadow_position_diagnostic_label:
        text = str(label or "").strip()
        if text:
            runner_args += ["--shadow-position-diagnostic-label", text]

    profile_args = list(matrix.ROUND3_DENY8_ENTRY_REGIME_ARGS)
    profile_args = matrix._with_arg_value(
        profile_args,
        "--flash-stale-position-exit-max-age-bars",
        str(int(args.stale_exit_age_bars)),
    )
    if args.experimental_flash_real_actors:
        profile_args += [
            "--enable-experimental-flash-actors",
            "--allow-experimental-flash-real-actors",
        ]
    if args.flash_shadow_actor_fallback_min_base_score is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-actor-fallback-min-base-score",
            f"{float(args.flash_shadow_actor_fallback_min_base_score):.4g}",
        )
    if args.flash_shadow_position_replay_actor_fallback_min_base_score is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-position-replay-actor-fallback-min-base-score",
            f"{float(args.flash_shadow_position_replay_actor_fallback_min_base_score):.4g}",
        )
    if args.flash_shadow_position_replay_actor_fallback_min_shadow_score is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-position-replay-actor-fallback-min-shadow-score",
            f"{float(args.flash_shadow_position_replay_actor_fallback_min_shadow_score):.4g}",
        )
    if args.flash_shadow_min_pnl_per_trade_lcb_usd is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-min-pnl-per-trade-lcb-usd",
            f"{float(args.flash_shadow_min_pnl_per_trade_lcb_usd):.4g}",
        )
    if args.enable_flash_shadow_pnl_lcb_risk_sizing and (
        "--enable-flash-shadow-pnl-lcb-risk-sizing" not in profile_args
    ):
        profile_args += ["--enable-flash-shadow-pnl-lcb-risk-sizing"]
    if args.flash_shadow_pnl_lcb_risk_min_mult is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-pnl-lcb-risk-min-mult",
            f"{float(args.flash_shadow_pnl_lcb_risk_min_mult):.4g}",
        )
    if args.flash_shadow_pnl_lcb_risk_floor_usd is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-pnl-lcb-risk-floor-usd",
            f"{float(args.flash_shadow_pnl_lcb_risk_floor_usd):.4g}",
        )
    if args.flash_shadow_pnl_lcb_risk_scale_usd is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-shadow-pnl-lcb-risk-scale-usd",
            f"{float(args.flash_shadow_pnl_lcb_risk_scale_usd):.4g}",
        )
    if args.enable_flash_actor_risk_sizing and (
        "--enable-flash-actor-risk-sizing" not in profile_args
    ):
        profile_args += ["--enable-flash-actor-risk-sizing"]
    if args.flash_actor_risk_min_mult is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-actor-risk-min-mult",
            f"{float(args.flash_actor_risk_min_mult):.4g}",
        )
    if args.flash_actor_risk_max_mult is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-actor-risk-max-mult",
            f"{float(args.flash_actor_risk_max_mult):.4g}",
        )
    if args.flash_actor_risk_edge_scale_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-actor-risk-edge-scale-pct",
            f"{float(args.flash_actor_risk_edge_scale_pct):.4g}",
        )
    if args.flash_funding_risk_mult_weight is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-funding-risk-mult-weight",
            f"{float(args.flash_funding_risk_mult_weight):.4g}",
        )
    if args.flash_funding_risk_mult_cap is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-funding-risk-mult-cap",
            f"{float(args.flash_funding_risk_mult_cap):.4g}",
        )
    if args.enable_flash_volatility_risk_sizing and (
        "--enable-flash-volatility-risk-sizing" not in profile_args
    ):
        profile_args += ["--enable-flash-volatility-risk-sizing"]
    if args.flash_volatility_risk_target_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-volatility-risk-target-pct",
            f"{float(args.flash_volatility_risk_target_pct):.4g}",
        )
    if args.flash_volatility_risk_min_volatility_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-volatility-risk-min-volatility-pct",
            f"{float(args.flash_volatility_risk_min_volatility_pct):.4g}",
        )
    if args.flash_volatility_risk_max_mult is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-volatility-risk-max-mult",
            f"{float(args.flash_volatility_risk_max_mult):.4g}",
        )
    if args.enable_flash_overextension_guard and (
        "--enable-flash-overextension-guard" not in profile_args
    ):
        profile_args += ["--enable-flash-overextension-guard"]
    if args.flash_overextension_lookback_bars is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-overextension-lookback-bars",
            str(int(args.flash_overextension_lookback_bars)),
        )
    if args.flash_short_overextension_return_floor_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-short-overextension-return-floor-pct",
            f"{float(args.flash_short_overextension_return_floor_pct):.4g}",
        )
    if args.flash_long_overextension_return_ceiling_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-long-overextension-return-ceiling-pct",
            f"{float(args.flash_long_overextension_return_ceiling_pct):.4g}",
        )
    if args.enable_flash_overextension_volatility_normalized and (
        "--enable-flash-overextension-volatility-normalized" not in profile_args
    ):
        profile_args += ["--enable-flash-overextension-volatility-normalized"]
    if args.flash_short_overextension_z_floor is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-short-overextension-z-floor",
            f"{float(args.flash_short_overextension_z_floor):.4g}",
        )
    if args.flash_long_overextension_z_ceiling is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-long-overextension-z-ceiling",
            f"{float(args.flash_long_overextension_z_ceiling):.4g}",
        )
    if args.flash_overextension_min_volatility_pct is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-overextension-min-volatility-pct",
            f"{float(args.flash_overextension_min_volatility_pct):.4g}",
        )
    if args.enable_flash_degradation_symbol_guard and (
        "--enable-flash-degradation-symbol-guard" not in profile_args
    ):
        profile_args += ["--enable-flash-degradation-symbol-guard"]
    if args.flash_degradation_symbol_cooldown_bars is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-symbol-cooldown-bars",
            str(int(args.flash_degradation_symbol_cooldown_bars)),
        )
    if args.flash_degradation_symbol_lookback_bars is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-symbol-lookback-bars",
            str(int(args.flash_degradation_symbol_lookback_bars)),
        )
    if args.flash_degradation_symbol_window_closed_trades is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-symbol-window-closed-trades",
            str(int(args.flash_degradation_symbol_window_closed_trades)),
        )
    if args.flash_degradation_symbol_min_closed_trades is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-symbol-min-closed-trades",
            str(int(args.flash_degradation_symbol_min_closed_trades)),
        )
    if args.flash_degradation_symbol_max_recent_pnl_usd is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-symbol-max-recent-pnl-usd",
            f"{float(args.flash_degradation_symbol_max_recent_pnl_usd):.4g}",
        )
    if args.enable_flash_degradation_recovery and (
        "--enable-flash-degradation-recovery" not in profile_args
    ):
        profile_args += ["--enable-flash-degradation-recovery"]
    if args.flash_degradation_recovery_min_closed_trades is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-recovery-min-closed-trades",
            str(int(args.flash_degradation_recovery_min_closed_trades)),
        )
    if args.flash_degradation_recovery_min_recent_pnl_usd is not None:
        profile_args = matrix._with_arg_value(
            profile_args,
            "--flash-degradation-recovery-min-recent-pnl-usd",
            f"{float(args.flash_degradation_recovery_min_recent_pnl_usd):.4g}",
        )
    if args.enable_flash_earned_cap_overrides and (
        "--enable-flash-earned-cap-overrides" not in profile_args
    ):
        profile_args += ["--enable-flash-earned-cap-overrides"]
    for item in args.flash_terminal_deny_signal_key:
        text = str(item or "").strip()
        if text:
            profile_args += ["--flash-terminal-deny-signal-key", text]
    for item in manifest.get("context_terminal_deny_signal_keys", []):
        text = str(item or "").strip()
        if text:
            profile_args += ["--flash-terminal-deny-context-signal-key", text]
    for item in args.flash_terminal_deny_context_signal_key:
        text = str(item or "").strip()
        if text:
            profile_args += ["--flash-terminal-deny-context-signal-key", text]
    runner_args += profile_args

    for item in manifest.get("score_boosts", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or "").strip()
        boost = item.get("boost")
        if key and boost is not None:
            runner_args += ["--flash-selected-subset-score-boost", f"{key}={boost}"]
    for item in manifest.get("context_score_boosts", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or item.get("context_key") or "").strip()
        boost = item.get("boost")
        if key and boost is not None:
            runner_args += [
                "--flash-selected-subset-context-score-boost",
                f"{key}={boost}",
            ]
    for item in args.context_score_boost:
        text = str(item or "").strip()
        if text:
            runner_args += ["--flash-selected-subset-context-score-boost", text]
    for key in manifest.get("do_not_demote_signal_keys", []):
        text = str(key or "").strip()
        if text:
            runner_args += ["--flash-selected-subset-do-not-demote-signal-key", text]
    for item in manifest.get("risk_mult_overrides", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or "").strip()
        risk_mult = item.get("risk_mult")
        if key and risk_mult is not None:
            runner_args += ["--flash-selected-subset-risk-mult", f"{key}={risk_mult}"]
    for item in args.signal_risk_mult:
        text = str(item or "").strip()
        if text:
            runner_args += ["--flash-selected-subset-risk-mult", text]
    for item in manifest.get("context_risk_mult_overrides", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or item.get("context_key") or "").strip()
        risk_mult = item.get("risk_mult")
        if key and risk_mult is not None:
            runner_args += [
                "--flash-selected-subset-context-risk-mult",
                f"{key}={risk_mult}",
            ]
    for item in args.context_risk_mult:
        text = str(item or "").strip()
        if text:
            runner_args += ["--flash-selected-subset-context-risk-mult", text]
    runner_args += [
        "--flash-selected-subset-risk-min-mult",
        "0.75",
        "--flash-selected-subset-risk-max-mult",
        "1.15",
    ]
    if args.enable_partial_profit_lock:
        runner_args += [
            "--enable-flash-partial-profit-lock",
            "--flash-partial-profit-lock-trigger-pnl-pct",
            f"{float(args.partial_profit_lock_trigger_pnl_pct):.4g}",
            "--flash-partial-profit-lock-close-fraction",
            f"{float(args.partial_profit_lock_close_fraction):.4g}",
            "--flash-partial-profit-lock-min-age-bars",
            str(int(args.partial_profit_lock_min_age_bars)),
        ]
        if args.disable_partial_profit_lock_skip_protected_signal_keys:
            runner_args += [
                "--disable-flash-partial-profit-lock-skip-protected-signal-keys"
            ]
        for item in args.partial_profit_lock_skip_signal_key:
            text = str(item or "").strip()
            if text:
                runner_args += [
                    "--flash-partial-profit-lock-skip-signal-key",
                    text,
                ]
    return runner.main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
