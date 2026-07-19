"""Run the production player-only selector on Retrodate data.

The command intentionally exposes only the parameters that belong to the new
Pantheon -> players contract.  It writes a compact promotion artifact consumed
by the live preflight in addition to the normal replay artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panteon_v2.analysis.retrodate_market_runner import (  # noqa: E402
    DEFAULT_YEARS,
    RetrodateMarketConfig,
    run_retrodate_market_benchmark,
)


def _default_data_dir() -> Path:
    root = PROJECT_ROOT / "Retrodate"
    preferred = sorted(
        (
            path
            for pattern in ("bitget_futures_current_*", "bitget_futures_only_*")
            for path in root.glob(pattern)
            if any(path.glob("crypto_1m_*_all_symbols.csv"))
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]
    for name in ("mexc_bitget_futures", "mexc_genetic_core_walk_forward"):
        candidate = root / name
        if any(candidate.glob("crypto_1m_*_all_symbols.csv")):
            return candidate
    return root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal player-by-regime Retrodate efficiency test."
    )
    parser.add_argument("--data-dir", default=str(_default_data_dir()))
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "Results" / "PlayerEfficiency"),
    )
    parser.add_argument(
        "--summarize-run",
        default="",
        help=(
            "Build the compact promotion artifact from an existing completed "
            "run directory or run_summary.json without replaying market data."
        ),
    )
    parser.add_argument(
        "--years",
        default=",".join(str(year) for year in DEFAULT_YEARS),
        help="Comma-separated years.",
    )
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--skip-bars", type=int, default=0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0006)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--mode", choices=("multi", "singleton"), default="multi")
    parser.add_argument("--fixed-player", default="")
    parser.add_argument(
        "--shadow-scope",
        choices=("all", "fixed"),
        default="all",
        help="fixed runs only the singleton player in shadow (baseline speed-up).",
    )
    parser.add_argument("--recency-decay", type=float, default=0.94)
    parser.add_argument("--return-window", type=int, default=100)
    parser.add_argument("--min-player-trades", type=int, default=5)
    parser.add_argument("--min-global-player-trades", type=int, default=20)
    parser.add_argument("--global-score-weight", type=float, default=0.25)
    parser.add_argument("--downside-penalty", type=float, default=0.25)
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--cooldown-bars", type=int, default=3)
    parser.add_argument("--shadow-workers", type=int, default=1)
    parser.add_argument("--min-bars-for-promotion", type=int, default=5000)
    parser.add_argument("--min-real-closed", type=int, default=30)
    parser.add_argument("--min-shadow-closed", type=int, default=100)
    parser.add_argument("--max-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--min-positive-window-ratio", type=float, default=0.80)
    parser.add_argument(
        "--require-latest-window-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--required-extra-cost-bps",
        type=int,
        choices=(0, 5, 10),
        default=5,
        help="Require positive PnL after this additional turnover cost stress.",
    )
    parser.add_argument(
        "--keep-verbose-artifacts",
        action="store_true",
        help="Keep large causal/shadow JSONL files and trading.log.",
    )
    return parser


def _years(raw: str) -> tuple[int, ...]:
    parsed = tuple(
        int(part.strip())
        for part in str(raw or "").replace(";", ",").split(",")
        if part.strip()
    )
    if not parsed:
        raise ValueError("at least one year is required")
    return parsed


def build_config(args: argparse.Namespace) -> RetrodateMarketConfig:
    if args.mode == "singleton" and not str(args.fixed_player or "").strip():
        raise ValueError("--fixed-player is required for singleton mode")
    if args.shadow_scope == "fixed" and args.mode != "singleton":
        raise ValueError("--shadow-scope fixed is valid only in singleton mode")
    if not 0.0 <= float(args.min_positive_window_ratio) <= 1.0:
        raise ValueError("--min-positive-window-ratio must be in [0, 1]")
    fixed_player = str(args.fixed_player or "").strip()
    return RetrodateMarketConfig(
        data_dir=Path(args.data_dir),
        results_root=Path(args.results_root),
        years=_years(args.years),
        stride_minutes=int(args.stride_minutes),
        max_bars=args.max_bars,
        skip_bars=int(args.skip_bars),
        initial_capital=float(args.initial_capital),
        fee_rate=float(args.fee_rate),
        slippage_pct=float(args.slippage_pct),
        use_live_regime_detector=True,
        player_only_runtime=True,
        trade_mode=str(args.mode),
        fixed_player_label=fixed_player,
        player_recency_decay=float(args.recency_decay),
        player_return_window=int(args.return_window),
        player_min_closed_trades=int(args.min_player_trades),
        player_min_global_closed_trades=int(args.min_global_player_trades),
        player_global_score_weight=float(args.global_score_weight),
        player_downside_penalty=float(args.downside_penalty),
        player_switch_margin=float(args.switch_margin),
        player_cooldown_bars=int(args.cooldown_bars),
        perf_max_returns_history=int(args.return_window),
        flash_enabled=False,
        shadow_parallel_workers=int(args.shadow_workers),
        shadow_player_include_labels=(
            (fixed_player,) if args.shadow_scope == "fixed" else ()
        ),
        write_every_bars=1000,
        full_snapshot_every=5000,
        progress_every_bars=1000,
        compact_causal_entry_decisions=False,
        flash_audit_events_enabled=False,
        shadow_audit_events_enabled=False,
        step_result_retention_enabled=False,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _number(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _existing_summary(raw_path: str):
    requested = Path(raw_path).expanduser().resolve()
    summary_path = requested / "run_summary.json" if requested.is_dir() else requested
    payload = _load_json(summary_path)
    if not payload:
        raise ValueError(f"run summary is missing or invalid: {summary_path}")
    output_raw = str(payload.get("output_dir") or "").strip()
    output_dir = Path(output_raw).expanduser().resolve() if output_raw else summary_path.parent
    required = ("status.json", "leaderboard_players.json", "walk_forward_report.json")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(
            "completed run artifacts are missing: " + ", ".join(missing)
        )
    return SimpleNamespace(
        output_dir=output_dir,
        summary_path=summary_path,
        bars_processed=int(_number(payload, "bars_processed")),
        step_errors=tuple(payload.get("step_errors") or ()),
    )


def _promotion_summary(summary, args: argparse.Namespace) -> dict[str, Any]:
    status = _load_json(summary.output_dir / "status.json")
    leaderboard = _load_json(summary.output_dir / "leaderboard_players.json")
    players_raw = leaderboard.get("players")
    players = players_raw if isinstance(players_raw, Mapping) else {}
    real_trades_raw = status.get("real_trades")
    real_trades = real_trades_raw if isinstance(real_trades_raw, Mapping) else {}
    real_closed = int(_number(real_trades, "closed"))
    realized_pnl_usd = _number(status, "pnl_usd")
    expectancy = realized_pnl_usd / real_closed if real_closed else 0.0
    max_drawdown = _number(status, "panteon_realized_max_drawdown_pct")
    shadow_closed = sum(
        int(_number(row, "closed_trades"))
        for row in players.values()
        if isinstance(row, Mapping)
    )
    shadow_raw = status.get("shadow")
    shadow_status = shadow_raw if isinstance(shadow_raw, Mapping) else {}
    shadow_player_count = int(_number(shadow_status, "actors"))
    walk_forward = _load_json(summary.output_dir / "walk_forward_report.json")
    temporal_raw = walk_forward.get("temporal_windows")
    temporal = temporal_raw if isinstance(temporal_raw, Mapping) else {}
    windows_raw = temporal.get("windows")
    windows = windows_raw if isinstance(windows_raw, list) else []
    temporal_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(windows, start=1):
        if not isinstance(raw, Mapping):
            continue
        stats_raw = raw.get("stats")
        stats = stats_raw if isinstance(stats_raw, Mapping) else {}
        net_pnl = _number(stats, "net_pnl")
        window_expectancy = _number(stats, "expectancy")
        temporal_rows.append({
            "window": str(raw.get("window") or f"window_{index:02d}"),
            "bar_start": int(_number(raw, "bar_start")),
            "bar_end": int(_number(raw, "bar_end")),
            "closed_trades": int(_number(stats, "closed_trades")),
            "net_pnl_usd": net_pnl,
            "expectancy_usd": window_expectancy,
            "positive_after_costs": net_pnl > 0.0 and window_expectancy > 0.0,
        })
    positive_windows = sum(
        1 for row in temporal_rows if row["positive_after_costs"]
    )
    positive_window_ratio = (
        positive_windows / len(temporal_rows) if temporal_rows else 0.0
    )
    latest_window = temporal_rows[-1] if temporal_rows else {}
    totals_raw = walk_forward.get("totals")
    totals = totals_raw if isinstance(totals_raw, Mapping) else {}
    extra_cost_bps = int(args.required_extra_cost_bps)
    stress_key = (
        "net_pnl"
        if extra_cost_bps == 0
        else f"net_pnl_minus_{extra_cost_bps}bps"
    )
    stressed_pnl = _number(totals, stress_key)
    fail_reasons: list[str] = []
    if summary.bars_processed < int(args.min_bars_for_promotion):
        fail_reasons.append(
            f"bars<{int(args.min_bars_for_promotion)}"
        )
    if real_closed < int(args.min_real_closed):
        fail_reasons.append(f"real_closed<{int(args.min_real_closed)}")
    if shadow_closed < int(args.min_shadow_closed):
        fail_reasons.append(f"shadow_closed<{int(args.min_shadow_closed)}")
    if realized_pnl_usd <= 0.0:
        fail_reasons.append("realized_pnl_usd<=0")
    if expectancy <= 0.0:
        fail_reasons.append("expectancy_after_costs<=0")
    if max_drawdown > float(args.max_drawdown_pct):
        fail_reasons.append(f"max_drawdown>{float(args.max_drawdown_pct):g}")
    if summary.step_errors:
        fail_reasons.append("step_errors")
    if not temporal_rows:
        fail_reasons.append("temporal_windows_missing")
    elif positive_window_ratio < float(args.min_positive_window_ratio):
        fail_reasons.append(
            "positive_window_ratio"
            f"<{float(args.min_positive_window_ratio):g}"
        )
    if (
        bool(args.require_latest_window_positive)
        and temporal_rows
        and not bool(latest_window.get("positive_after_costs"))
    ):
        fail_reasons.append("latest_window_not_positive")
    if not totals:
        fail_reasons.append("walk_forward_totals_missing")
    elif stressed_pnl <= 0.0:
        fail_reasons.append(f"pnl_after_extra_{extra_cost_bps}bps<=0")

    ranked_players = []
    for label, raw in players.items():
        if not isinstance(raw, Mapping):
            continue
        ranked_players.append({
            "label": str(label)[2:] if str(label).startswith("V_") else str(label),
            "pnl_pct": _number(raw, "pnl_pct"),
            "closed_trades": int(_number(raw, "closed_trades")),
            "win_rate": _number(raw, "win_rate"),
            "max_drawdown_pct": _number(raw, "max_drawdown_pct"),
            "per_regime": raw.get("per_regime", {}),
        })
    ranked_players.sort(
        key=lambda row: (-row["pnl_pct"], -row["closed_trades"], row["label"])
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_contract": "pantheon_players_v1",
        "player_only_runtime": True,
        "trade_mode": args.mode,
        "fixed_player_label": str(args.fixed_player or "").strip(),
        "shadow_scope": str(args.shadow_scope),
        "causal_selection": True,
        "memory_scope": "player_regime",
        "rating_scope": "player_regime_plus_global",
        "recency_weighted": True,
        "player_rating": {
            "min_regime_closed_trades": int(args.min_player_trades),
            "min_global_closed_trades": int(args.min_global_player_trades),
            "global_score_weight": float(args.global_score_weight),
            "recency_decay": float(args.recency_decay),
            "return_window": int(args.return_window),
        },
        "max_real_players": 1,
        "shadow_player_count": shadow_player_count,
        "bars_processed": summary.bars_processed,
        "real_closed_trades": real_closed,
        "shadow_closed_trades": shadow_closed,
        "realized_pnl_usd": realized_pnl_usd,
        "expectancy_after_costs": expectancy,
        "max_drawdown_pct": max_drawdown,
        "temporal_stability": {
            "window_count": len(temporal_rows),
            "positive_windows": positive_windows,
            "positive_window_ratio": positive_window_ratio,
            "latest_window_positive": bool(
                latest_window.get("positive_after_costs", False)
            ),
            "windows": temporal_rows,
        },
        "cost_stress": {
            "additional_cost_bps": extra_cost_bps,
            "net_pnl_usd": stressed_pnl,
            "source_metric": stress_key,
        },
        "step_errors": list(summary.step_errors),
        "promotion_verdict": {
            "passed": not fail_reasons,
            "fail_reasons": fail_reasons,
        },
        "gates": {
            "min_bars": int(args.min_bars_for_promotion),
            "min_real_closed": int(args.min_real_closed),
            "min_shadow_closed": int(args.min_shadow_closed),
            "max_drawdown_pct": float(args.max_drawdown_pct),
            "positive_after_costs_required": True,
            "min_positive_window_ratio": float(args.min_positive_window_ratio),
            "require_latest_window_positive": bool(
                args.require_latest_window_positive
            ),
            "required_extra_cost_bps": extra_cost_bps,
        },
        "players": ranked_players,
        "source_run_summary": str(summary.summary_path),
        "source_status": str(summary.output_dir / "status.json"),
    }


def _write_compact_report(output_dir: Path, payload: Mapping[str, Any]) -> None:
    path = output_dir / "player_efficiency_report.md"
    verdict = payload.get("promotion_verdict", {})
    players = payload.get("players", [])
    temporal = payload.get("temporal_stability", {})
    stress = payload.get("cost_stress", {})
    lines = [
        "# Player efficiency retro report",
        "",
        f"Promotion gate: **{'PASS' if verdict.get('passed') else 'FAIL'}**",
        "",
        f"Bars: {payload.get('bars_processed', 0)}; real closed: "
        f"{payload.get('real_closed_trades', 0)}; shadow closed: "
        f"{payload.get('shadow_closed_trades', 0)}.",
        f"Realized PnL: {payload.get('realized_pnl_usd', 0.0):+.4f} USD; "
        f"expectancy after costs: {payload.get('expectancy_after_costs', 0.0):+.6f} USD/trade; "
        f"max drawdown: {payload.get('max_drawdown_pct', 0.0):.3f}%.",
        f"Positive temporal windows: {temporal.get('positive_windows', 0)}/"
        f"{temporal.get('window_count', 0)}; latest positive: "
        f"{bool(temporal.get('latest_window_positive', False))}.",
        f"PnL after +{stress.get('additional_cost_bps', 0)} bps stress: "
        f"{stress.get('net_pnl_usd', 0.0):+.4f} USD.",
        "",
    ]
    fail_reasons = verdict.get("fail_reasons", [])
    if fail_reasons:
        lines.extend(["Fail reasons: " + ", ".join(str(item) for item in fail_reasons), ""])
    lines.extend([
        "| Player | PnL % | Closed | Win rate % | Max DD % |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in list(players)[:20]:
        lines.append(
            f"| {row['label']} | {row['pnl_pct']:+.4f} | {row['closed_trades']} | "
            f"{row['win_rate']:.2f} | {row['max_drawdown_pct']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prune_verbose_artifacts(output_dir: Path) -> list[str]:
    verbose_names = (
        "causal_entry_decisions.jsonl",
        "shadow_player_pnl_events.jsonl",
        "shadow_agent_pnl_events.jsonl",
        "events.jsonl",
        "trading.log",
        "oracle_mismatch_report.json",
    )
    removed: list[str] = []
    for name in verbose_names:
        path = output_dir / name
        try:
            if path.is_file():
                path.unlink()
                removed.append(name)
        except OSError:
            continue
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.summarize_run or "").strip():
        try:
            summary = _existing_summary(str(args.summarize_run))
        except ValueError as exc:
            _parser().error(str(exc))
    else:
        try:
            config = build_config(args)
        except ValueError as exc:
            _parser().error(str(exc))
        summary = run_retrodate_market_benchmark(config)
    payload = _promotion_summary(summary, args)
    if not args.keep_verbose_artifacts:
        payload["pruned_verbose_artifacts"] = _prune_verbose_artifacts(
            summary.output_dir
        )
    artifact = summary.output_dir / "player_efficiency_summary.json"
    artifact.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_compact_report(summary.output_dir, payload)
    print(f"player_efficiency_summary={artifact}", flush=True)
    print(f"promotion_passed={payload['promotion_verdict']['passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
