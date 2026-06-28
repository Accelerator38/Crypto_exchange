from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_RESULTS_ROOT = ROOT / "Results" / "Panteon3PreLiveMatrix"
DEFAULT_REPORTS_DIR = ROOT / "Reports" / "Panteon3PreLiveMatrix"
DEFAULT_FUTURES_REPLAY_DATA_DIR = ROOT / "Retrodate" / "mexc_bitget_futures"
PLAN_FILENAME = "panteon3_pre_live_matrix_plan.json"
RESULTS_FILENAME = "panteon3_pre_live_matrix_results.json"
SUMMARY_FILENAME = "panteon3_pre_live_matrix_summary.json"
SUMMARY_MARKDOWN_FILENAME = "panteon3_pre_live_matrix_summary.md"

FORBIDDEN_LIVE_ARGS = frozenset({
    "--live",
    "--exchange-live",
    "--real-orders",
    "--enable-real-orders",
    "--place-orders",
    "--allow-live-orders",
})

CONTROLLED_EXPLORATION_REASONS = (
    "expected_edge_below_cost",
    "range_low_vol_actor_not_allowed",
    "shadow_unconfirmed",
    "insufficient_closed_trades",
)

OHLCV_ONLY_PROMOTION_DERIVED_ACTOR_LABELS = (
    "LiveOIBreakout",
    "MomentumScalper",
    "LiveCrashHunter",
    "LiveVolCompress",
    "FundingArb",
)

DERIVATIVES_CONTEXT_ACTOR_LABELS = (
    "CarryFlowAgentV2",
)

PROMOTION_DERIVED_ACTOR_LABELS = (
    *DERIVATIVES_CONTEXT_ACTOR_LABELS,
    "MomentumScalper",
    "LiveCrashHunter",
    "LiveVolCompress",
    "FundingArb",
)

DIAGNOSTIC_COMPONENT_LABELS = tuple(dict.fromkeys((
    *PROMOTION_DERIVED_ACTOR_LABELS,
    "AnchorFlowMomentum",
)))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_utc")


def _str_path(path: Path | str) -> str:
    return str(Path(path))


def _base_runner_args(
    *,
    data_dir: Path,
    results_dir: Path,
    years: str,
    max_bars: int | None,
    skip_bars: int,
    stride_minutes: int,
    initial_capital: float,
    risk_capital_fraction: float,
    extra_runner_args: Sequence[str],
) -> list[str]:
    args = [
        "-m",
        "panteon_v2.analysis.retrodate_market_runner",
        "--data-dir",
        _str_path(data_dir),
        "--results-root",
        _str_path(results_dir),
        "--years",
        str(years),
        "--stride-minutes",
        str(int(stride_minutes)),
        "--initial-capital",
        str(float(initial_capital)),
        "--risk-capital-fraction",
        str(float(risk_capital_fraction)),
        "--enable-flash",
    ]
    if max_bars is not None:
        args.extend(["--max-bars", str(int(max_bars))])
    if int(skip_bars) > 0:
        args.extend(["--skip-bars", str(int(skip_bars))])
    for label in DIAGNOSTIC_COMPONENT_LABELS:
        args.extend(["--compact-causal-entry-include-label", label])
    args.extend(str(arg) for arg in extra_runner_args)
    return args


def _controlled_exploration_args(
    *,
    initial_capital: float,
    risk_capital_fraction: float,
    risk_mult: float = 0.05,
    min_notional_max_risk_mult: float = 0.10,
    default_min_notional_usd: float = 5.0,
) -> list[str]:
    args = [
        "--enable-flash-controlled-exploration",
        "--flash-controlled-exploration-risk-mult",
        f"{risk_mult:.2f}",
        "--flash-controlled-exploration-min-shadow-score",
        "0.0",
        "--flash-controlled-exploration-min-shadow-closed",
        "0",
        "--flash-controlled-exploration-max-daily-trades",
        "6",
        "--flash-controlled-exploration-max-open-positions",
        "2",
        "--flash-controlled-exploration-min-rolling-expectancy",
        "0.0",
        "--enable-flash-controlled-exploration-min-notional-sizing",
        "--flash-controlled-exploration-account-equity-usd",
        str(float(initial_capital)),
        "--flash-controlled-exploration-capital-fraction",
        str(float(risk_capital_fraction)),
        "--flash-controlled-exploration-min-notional-max-risk-mult",
        f"{min_notional_max_risk_mult:.2f}",
        "--flash-controlled-exploration-default-min-notional-usd",
        str(float(default_min_notional_usd)),
    ]
    for reason in CONTROLLED_EXPLORATION_REASONS:
        args.extend(["--flash-controlled-exploration-allowed-reason", reason])
    return args


def _promotion_derived_args(
    *,
    actor_labels: Sequence[str],
    initial_capital: float = 1000.0,
    risk_capital_fraction: float = 0.10,
    min_notional_max_risk_mult: float = 0.10,
    default_min_notional_usd: float = 5.0,
) -> list[str]:
    args = [
        "--enable-flash-promotion-derived-router",
    ]
    for label in actor_labels:
        args.extend(["--flash-promotion-derived-actor-label", label])
    args.extend([
        "--flash-promotion-derived-min-closed-trades",
        "5",
        "--flash-promotion-derived-dynamic-best",
        "--flash-promotion-derived-min-expectancy",
        "0.0",
        "--flash-promotion-derived-risk-mult",
        "0.03",
        "--enable-flash-promotion-derived-min-notional-sizing",
        "--flash-promotion-derived-account-equity-usd",
        str(float(initial_capital)),
        "--flash-promotion-derived-capital-fraction",
        str(float(risk_capital_fraction)),
        "--flash-promotion-derived-min-notional-max-risk-mult",
        f"{min_notional_max_risk_mult:.2f}",
        "--flash-promotion-derived-default-min-notional-usd",
        str(float(default_min_notional_usd)),
    ])
    return args


def _legacy_flash_real_agent_args() -> list[str]:
    args: list[str] = []
    for label in OHLCV_ONLY_PROMOTION_DERIVED_ACTOR_LABELS:
        args.extend(["--flash-legacy-real-agent-label", label])
    return args


def _promotion_actor_labels(
    *,
    include_derivatives_context_actors: bool,
) -> tuple[str, ...]:
    labels = list(OHLCV_ONLY_PROMOTION_DERIVED_ACTOR_LABELS)
    if include_derivatives_context_actors:
        labels = [*DERIVATIVES_CONTEXT_ACTOR_LABELS, *labels]
    return tuple(dict.fromkeys(labels))


def _legacy_flash_real_agent_args_for(
    actor_labels: Sequence[str],
) -> list[str]:
    args: list[str] = []
    for label in actor_labels:
        args.extend(["--flash-legacy-real-agent-label", label])
    return args


def _futures_replay_signal_fix_args() -> list[str]:
    return ["--enable-futures-replay-signal-fixes"]


def _causal_router_args() -> list[str]:
    return [
        "--enable-flash-causal-actor-router",
        "--flash-causal-actor-router-min-closed-trades",
        "5",
        "--flash-causal-actor-router-min-expectancy",
        "0.0",
        "--enable-flash-causal-actor-router-exploration",
        "--flash-causal-actor-router-exploration-risk-mult",
        "0.05",
    ]


def _assert_safe_replay_command(command: Sequence[str]) -> None:
    forbidden = sorted(FORBIDDEN_LIVE_ARGS.intersection(str(arg) for arg in command))
    if forbidden:
        raise ValueError(f"live/order flags are not allowed in pre-live matrix: {forbidden}")


def _normalize_window_skip_bars(raw: Sequence[int] | None) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw or (0,):
        skip_bars = int(item)
        if skip_bars < 0:
            raise ValueError("window skip bars must be >= 0")
        values.append(skip_bars)
    if not values:
        values.append(0)
    return tuple(dict.fromkeys(values))


def _parse_window_skip_bars(raw: Sequence[str]) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw or ():
        for part in str(item or "").split(","):
            clean = part.strip()
            if not clean:
                continue
            values.append(int(clean))
    return _normalize_window_skip_bars(values or (0,))


def build_matrix_plan(
    *,
    python_executable: str,
    data_dir: Path | str = DEFAULT_FUTURES_REPLAY_DATA_DIR,
    results_root: Path | str,
    years: str,
    max_bars: int | None,
    stride_minutes: int,
    initial_capital: float,
    risk_capital_fraction: float,
    extra_runner_args: Sequence[str] = (),
    window_skip_bars: Sequence[int] = (0,),
    include_derivatives_context_actors: bool = False,
) -> list[dict[str, Any]]:
    results_root = Path(results_root)
    data_dir = Path(data_dir)
    promotion_actor_labels = _promotion_actor_labels(
        include_derivatives_context_actors=include_derivatives_context_actors,
    )
    base_variants: list[tuple[str, list[str]]] = [
        ("baseline", []),
        (
            "controlled_exploration",
            _controlled_exploration_args(
                initial_capital=initial_capital,
                risk_capital_fraction=risk_capital_fraction,
            ),
        ),
        ("causal_router", _causal_router_args()),
        (
            "panteon3_candidate",
            _controlled_exploration_args(
                initial_capital=initial_capital,
                risk_capital_fraction=risk_capital_fraction,
            )
            + _futures_replay_signal_fix_args()
            + _causal_router_args()
            + _legacy_flash_real_agent_args_for(promotion_actor_labels)
            + _promotion_derived_args(
                actor_labels=promotion_actor_labels,
                initial_capital=initial_capital,
                risk_capital_fraction=risk_capital_fraction,
            ),
        ),
    ]

    plan: list[dict[str, Any]] = []
    windows = _normalize_window_skip_bars(window_skip_bars)
    use_window_prefix = len(windows) > 1 or any(skip > 0 for skip in windows)
    for skip_bars in windows:
        for base_variant, variant_args in base_variants:
            variant = (
                f"skip_{int(skip_bars)}__{base_variant}"
                if use_window_prefix
                else base_variant
            )
            results_dir = results_root / variant
            command = [
                python_executable,
                *_base_runner_args(
                    results_dir=results_dir,
                    data_dir=data_dir,
                    years=years,
                    max_bars=max_bars,
                    skip_bars=int(skip_bars),
                    stride_minutes=stride_minutes,
                    initial_capital=initial_capital,
                    risk_capital_fraction=risk_capital_fraction,
                    extra_runner_args=extra_runner_args,
                ),
                *variant_args,
            ]
            _assert_safe_replay_command(command)
            plan.append({
                "variant": variant,
                "base_variant": base_variant,
                "window_skip_bars": int(skip_bars),
                "data_dir": _str_path(data_dir),
                "results_dir": _str_path(results_dir),
                "command": command,
            })
    return plan


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latest_run_dir(variant_dir: Path) -> Path | None:
    candidates = [
        path.parent
        for path in variant_dir.rglob("flash_attribution_summary.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss < 0.0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0.0:
        return 9999.0
    return 0.0


def _first_actor_stats(walk_forward: dict[str, Any]) -> dict[str, Any]:
    by_actor = walk_forward.get("by_actor")
    if not isinstance(by_actor, dict) or not by_actor:
        return {}
    best_label, best_stats = max(
        by_actor.items(),
        key=lambda item: _safe_float((item[1] or {}).get("net_pnl"), 0.0)
        if isinstance(item[1], dict)
        else 0.0,
    )
    if not isinstance(best_stats, dict):
        return {"label": str(best_label)}
    return {"label": str(best_label), **best_stats}


def _variant_summary_row(variant: str, run_dir: Path) -> dict[str, Any]:
    flash = _load_json(run_dir / "flash_attribution_summary.json")
    allocation = _load_json(run_dir / "allocation_diagnostics.json")
    walk_forward_path = run_dir / "walk_forward_report.json"
    component_path = run_dir / "component_benchmark_report.json"
    candidate_path = run_dir / "candidate_diagnostics.json"
    walk_forward = _load_json(walk_forward_path) if walk_forward_path.exists() else {}
    component = _load_json(component_path) if component_path.exists() else {}
    candidate = _load_json(candidate_path) if candidate_path.exists() else {}

    flash_summary = flash.get("summary", {})
    actor_stats = _first_actor_stats(walk_forward)
    closed_trades = _safe_int(flash_summary.get("closed_trades"))
    pnl_usd = _safe_float(flash_summary.get("realized_pnl_usd"))
    gross_profit = _safe_float(actor_stats.get("gross_profit"))
    gross_loss = _safe_float(actor_stats.get("gross_loss"))
    windows = (
        ((walk_forward.get("temporal_windows") or {}).get("windows") or [])
        if isinstance(walk_forward.get("temporal_windows"), dict)
        else []
    )
    regimes = walk_forward.get("by_regime") or {}
    if not isinstance(regimes, dict):
        regimes = {}

    profitable_windows = sum(
        1
        for window in windows
        if _safe_float(((window or {}).get("stats") or {}).get("net_pnl")) > 0.0
    )
    positive_regimes = sum(
        1
        for stats in regimes.values()
        if isinstance(stats, dict) and _safe_float(stats.get("net_pnl")) > 0.0
    )
    component_summary = component.get("summary", {})
    active_rejections = (
        ((candidate.get("rejection_summary") or {}).get("flash_active_reason_counts") or {})
        if isinstance(candidate.get("rejection_summary"), dict)
        else {}
    )

    return {
        "variant": variant,
        "base_variant": _base_variant_name(variant),
        "run_dir": _str_path(run_dir),
        "selected_signals": _safe_int(flash_summary.get("selected_signals")),
        "executable_selected_signals": _safe_int(
            flash_summary.get("executable_selected_signals")
        ),
        "filled_signals": _safe_int(flash_summary.get("filled_signals")),
        "blocked_signals": _safe_int(flash_summary.get("blocked_signals")),
        "rejected_signals": _safe_int(flash_summary.get("rejected_signals")),
        "pending_signals": _safe_int(flash_summary.get("pending_signals")),
        "closed_trades": closed_trades,
        "winning_trades": _safe_int(flash_summary.get("winning_trades")),
        "losing_trades": _safe_int(flash_summary.get("losing_trades")),
        "realized_pnl_usd": pnl_usd,
        "expectancy_usd": pnl_usd / closed_trades if closed_trades else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": round(
            _safe_float(
                actor_stats.get("profit_factor"),
                _profit_factor(gross_profit, gross_loss),
            ),
            10,
        ),
        "no_trade_share_pct": _safe_float(allocation.get("no_trade_share_pct")),
        "raw_zero_share_pct": _safe_float(allocation.get("raw_zero_share_pct")),
        "filled_zero_share_pct": _safe_float(allocation.get("filled_zero_share_pct")),
        "profitable_windows": profitable_windows,
        "window_count": len(windows),
        "positive_regimes": positive_regimes,
        "regime_count": len(regimes),
        "best_actor": actor_stats.get("label", ""),
        "best_component_label": component_summary.get("best_component_label", ""),
        "best_component_pnl_usd": _safe_float(
            component_summary.get("best_component_pnl_usd")
        ),
        "panteon_beats_best_component": bool(
            component_summary.get("panteon_beats_best_component", False)
        ),
        "active_rejection_reasons": dict(active_rejections),
    }


def _base_variant_name(variant: str) -> str:
    text = str(variant or "")
    if text.startswith("skip_") and "__" in text:
        return text.split("__", 1)[1]
    return text


def _aggregate_reason_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = row.get("active_rejection_reasons") or {}
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            counts[str(reason)] = counts.get(str(reason), 0) + _safe_int(count)
    return counts


def _aggregate_variant_rows(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any] | None:
    selected = [row for row in rows if row.get("base_variant") == variant]
    if not selected:
        return None
    int_fields = (
        "selected_signals",
        "executable_selected_signals",
        "filled_signals",
        "blocked_signals",
        "rejected_signals",
        "pending_signals",
        "closed_trades",
        "winning_trades",
        "losing_trades",
        "profitable_windows",
        "window_count",
        "positive_regimes",
        "regime_count",
    )
    float_sum_fields = ("realized_pnl_usd", "gross_profit", "gross_loss")
    float_avg_fields = (
        "no_trade_share_pct",
        "raw_zero_share_pct",
        "filled_zero_share_pct",
    )
    out: dict[str, Any] = {
        "variant": variant,
        "base_variant": variant,
        "run_dir": "",
        "source_variants": [str(row.get("variant") or "") for row in selected],
    }
    for field in int_fields:
        out[field] = sum(_safe_int(row.get(field)) for row in selected)
    for field in float_sum_fields:
        out[field] = sum(_safe_float(row.get(field)) for row in selected)
    for field in float_avg_fields:
        out[field] = (
            sum(_safe_float(row.get(field)) for row in selected) / len(selected)
        )
    closed_trades = _safe_int(out.get("closed_trades"))
    pnl_usd = _safe_float(out.get("realized_pnl_usd"))
    gross_profit = _safe_float(out.get("gross_profit"))
    gross_loss = _safe_float(out.get("gross_loss"))
    out["expectancy_usd"] = pnl_usd / closed_trades if closed_trades else 0.0
    out["profit_factor"] = round(_profit_factor(gross_profit, gross_loss), 10)
    best_component = max(
        selected,
        key=lambda row: _safe_float(row.get("best_component_pnl_usd")),
    )
    out["best_actor"] = ""
    out["best_component_label"] = best_component.get("best_component_label", "")
    out["best_component_pnl_usd"] = _safe_float(
        best_component.get("best_component_pnl_usd")
    )
    out["panteon_beats_best_component"] = all(
        bool(row.get("panteon_beats_best_component", False)) for row in selected
    )
    out["active_rejection_reasons"] = _aggregate_reason_counts(selected)
    return out


def _promotion_fail_reasons(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    min_filled: int,
    min_closed_trades: int,
    max_blocked: int,
    max_rejected: int,
    require_positive_expectancy: bool,
    min_profitable_windows: int,
    min_positive_regimes: int,
    require_beats_baseline: bool,
) -> list[str]:
    reasons: list[str] = []
    if _safe_int(candidate.get("filled_signals")) < min_filled:
        reasons.append(
            f"filled_signals {candidate.get('filled_signals')} < min_filled {min_filled}"
        )
    if _safe_int(candidate.get("closed_trades")) < min_closed_trades:
        reasons.append(
            "closed_trades "
            f"{candidate.get('closed_trades')} < min_closed_trades {min_closed_trades}"
        )
    if _safe_int(candidate.get("blocked_signals")) > max_blocked:
        reasons.append(
            f"blocked_signals {candidate.get('blocked_signals')} > max_blocked {max_blocked}"
        )
    if _safe_int(candidate.get("rejected_signals")) > max_rejected:
        reasons.append(
            f"rejected_signals {candidate.get('rejected_signals')} > max_rejected {max_rejected}"
        )
    if require_positive_expectancy and _safe_float(candidate.get("expectancy_usd")) <= 0.0:
        reasons.append(f"expectancy_usd {candidate.get('expectancy_usd')} <= 0")
    if _safe_int(candidate.get("profitable_windows")) < min_profitable_windows:
        reasons.append(
            "profitable_windows "
            f"{candidate.get('profitable_windows')} < min_profitable_windows {min_profitable_windows}"
        )
    if _safe_int(candidate.get("positive_regimes")) < min_positive_regimes:
        reasons.append(
            "positive_regimes "
            f"{candidate.get('positive_regimes')} < min_positive_regimes {min_positive_regimes}"
        )
    if require_beats_baseline and baseline is not None:
        candidate_pnl = _safe_float(candidate.get("realized_pnl_usd"))
        baseline_pnl = _safe_float(baseline.get("realized_pnl_usd"))
        if candidate_pnl <= baseline_pnl:
            reasons.append(
                f"candidate_pnl_usd {candidate_pnl} <= baseline_pnl_usd {baseline_pnl}"
            )
    return reasons


def collect_matrix_summary(
    results_root: Path | str,
    *,
    candidate_variant: str = "panteon3_candidate",
    min_filled: int = 20,
    min_closed_trades: int = 10,
    max_blocked: int = 0,
    max_rejected: int = 0,
    require_positive_expectancy: bool = True,
    min_profitable_windows: int = 1,
    min_positive_regimes: int = 1,
    require_beats_baseline: bool = True,
) -> dict[str, Any]:
    results_root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for variant_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        run_dir = _latest_run_dir(variant_dir)
        if run_dir is None:
            continue
        rows.append(_variant_summary_row(variant_dir.name, run_dir))
    rows_by_variant = {row["variant"]: row for row in rows}
    candidate = rows_by_variant.get(candidate_variant) or _aggregate_variant_rows(
        rows,
        variant=candidate_variant,
    )
    if candidate is None:
        raise FileNotFoundError(f"candidate variant not found: {candidate_variant}")
    baseline = rows_by_variant.get("baseline") or _aggregate_variant_rows(
        rows,
        variant="baseline",
    )
    fail_reasons = _promotion_fail_reasons(
        candidate=candidate,
        baseline=baseline,
        min_filled=min_filled,
        min_closed_trades=min_closed_trades,
        max_blocked=max_blocked,
        max_rejected=max_rejected,
        require_positive_expectancy=require_positive_expectancy,
        min_profitable_windows=min_profitable_windows,
        min_positive_regimes=min_positive_regimes,
        require_beats_baseline=require_beats_baseline,
    )
    return {
        "results_root": _str_path(results_root),
        "rows": rows,
        "candidate": candidate,
        "baseline": baseline,
        "promotion_gates": {
            "candidate_variant": candidate_variant,
            "min_filled": min_filled,
            "min_closed_trades": min_closed_trades,
            "max_blocked": max_blocked,
            "max_rejected": max_rejected,
            "require_positive_expectancy": require_positive_expectancy,
            "min_profitable_windows": min_profitable_windows,
            "min_positive_regimes": min_positive_regimes,
            "require_beats_baseline": require_beats_baseline,
        },
        "promotion_verdict": {
            "passed": not fail_reasons,
            "fail_reasons": fail_reasons,
        },
    }


def format_matrix_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Panteon 3.0 Pre-Live Matrix Summary",
        "",
        f"Results root: `{summary.get('results_root', '')}`",
        "",
        "| Variant | Filled | Blocked | Rejected | Closed | PnL USD | Expectancy | Profit factor | Windows + | Regimes + |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {variant} | {filled} | {blocked} | {rejected} | {closed} | "
            "{pnl:.4f} | {expectancy:.4f} | {profit_factor:.4f} | "
            "{windows} | {regimes} |".format(
                variant=row.get("variant", ""),
                filled=_safe_int(row.get("filled_signals")),
                blocked=_safe_int(row.get("blocked_signals")),
                rejected=_safe_int(row.get("rejected_signals")),
                closed=_safe_int(row.get("closed_trades")),
                pnl=_safe_float(row.get("realized_pnl_usd")),
                expectancy=_safe_float(row.get("expectancy_usd")),
                profit_factor=_safe_float(row.get("profit_factor")),
                windows=_safe_int(row.get("profitable_windows")),
                regimes=_safe_int(row.get("positive_regimes")),
            )
        )
    verdict = summary.get("promotion_verdict", {})
    lines.extend([
        "",
        f"Promotion passed: `{bool(verdict.get('passed'))}`",
        "",
        "Fail reasons:",
    ])
    fail_reasons = verdict.get("fail_reasons") or []
    if fail_reasons:
        lines.extend(f"- `{reason}`" for reason in fail_reasons)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_matrix_plan(plan: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"

    rows: list[dict[str, Any]] = []
    for entry in plan:
        command = [str(part) for part in entry["command"]]
        _assert_safe_replay_command(command)
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        elapsed = time.perf_counter() - start
        rows.append({
            "variant": entry["variant"],
            "results_dir": entry["results_dir"],
            "returncode": int(completed.returncode),
            "elapsed_seconds": round(elapsed, 3),
            "command": command,
        })
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a safe replay-only Panteon 3.0 pre-live comparison matrix.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_FUTURES_REPLAY_DATA_DIR),
        help="Retrodate-compatible MEXC/BITGET futures data directory.",
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT / _timestamp()))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--years", default="2026")
    parser.add_argument("--max-bars", type=int, default=240)
    parser.add_argument(
        "--window-skip-bars",
        action="append",
        default=[],
        help=(
            "Replay window start offsets in bars. Can be repeated or comma-separated; "
            "0 means start from the first selected bar."
        ),
    )
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--risk-capital-fraction", type=float, default=0.10)
    parser.add_argument("--extra-runner-arg", action="append", default=[])
    parser.add_argument(
        "--include-derivatives-context-actors",
        action="store_true",
        help=(
            "Include actors that require funding/OI/crowding context, such as "
            "CarryFlowAgentV2, in the candidate universe. Keep disabled for "
            "OHLCV-only replay data."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--candidate-variant", default="panteon3_candidate")
    parser.add_argument("--min-filled", type=int, default=20)
    parser.add_argument("--min-closed-trades", type=int, default=10)
    parser.add_argument("--max-blocked", type=int, default=0)
    parser.add_argument("--max-rejected", type=int, default=0)
    parser.add_argument("--min-profitable-windows", type=int, default=1)
    parser.add_argument("--min-positive-regimes", type=int, default=1)
    parser.add_argument("--allow-nonpositive-expectancy", action="store_true")
    parser.add_argument("--no-require-beats-baseline", action="store_true")
    parser.add_argument("--fail-on-promotion-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reports_dir = Path(args.reports_dir)
    if args.summary_only:
        summary = collect_matrix_summary(
            Path(args.results_root),
            candidate_variant=args.candidate_variant,
            min_filled=args.min_filled,
            min_closed_trades=args.min_closed_trades,
            max_blocked=args.max_blocked,
            max_rejected=args.max_rejected,
            require_positive_expectancy=not args.allow_nonpositive_expectancy,
            min_profitable_windows=args.min_profitable_windows,
            min_positive_regimes=args.min_positive_regimes,
            require_beats_baseline=not args.no_require_beats_baseline,
        )
        write_json(reports_dir / SUMMARY_FILENAME, summary)
        write_markdown(
            reports_dir / SUMMARY_MARKDOWN_FILENAME,
            format_matrix_summary_markdown(summary),
        )
        print(json.dumps(summary["promotion_verdict"], ensure_ascii=False, indent=2))
        if args.fail_on_promotion_failure and not summary["promotion_verdict"]["passed"]:
            return 2
        return 0

    plan = build_matrix_plan(
        python_executable=args.python,
        data_dir=Path(args.data_dir),
        results_root=Path(args.results_root),
        years=args.years,
        max_bars=args.max_bars,
        window_skip_bars=_parse_window_skip_bars(args.window_skip_bars),
        stride_minutes=args.stride_minutes,
        initial_capital=args.initial_capital,
        risk_capital_fraction=args.risk_capital_fraction,
        extra_runner_args=args.extra_runner_arg,
        include_derivatives_context_actors=args.include_derivatives_context_actors,
    )
    write_json(reports_dir / PLAN_FILENAME, plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    rows = run_matrix_plan(plan)
    write_json(reports_dir / RESULTS_FILENAME, rows)
    failed = [row for row in rows if row["returncode"] != 0]
    if failed and args.stop_on_failure:
        return int(failed[0]["returncode"])
    if not failed:
        summary = collect_matrix_summary(
            Path(args.results_root),
            candidate_variant=args.candidate_variant,
            min_filled=args.min_filled,
            min_closed_trades=args.min_closed_trades,
            max_blocked=args.max_blocked,
            max_rejected=args.max_rejected,
            require_positive_expectancy=not args.allow_nonpositive_expectancy,
            min_profitable_windows=args.min_profitable_windows,
            min_positive_regimes=args.min_positive_regimes,
            require_beats_baseline=not args.no_require_beats_baseline,
        )
        write_json(reports_dir / SUMMARY_FILENAME, summary)
        write_markdown(
            reports_dir / SUMMARY_MARKDOWN_FILENAME,
            format_matrix_summary_markdown(summary),
        )
        if args.fail_on_promotion_failure and not summary["promotion_verdict"]["passed"]:
            return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
