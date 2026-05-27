"""Promotion gates for Flash RSI/MACD/ATR technical overlay candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MODE_BASELINE = "baseline"
MODE_DIAGNOSTIC = "diagnostic"
MODE_ATR_SIZING = "atr_sizing"
MODE_SOFT_OVERLAY = "soft_overlay"
MODE_HARD_GATE = "hard_gate"
KNOWN_MODES = frozenset({
    MODE_DIAGNOSTIC,
    MODE_ATR_SIZING,
    MODE_SOFT_OVERLAY,
    MODE_HARD_GATE,
})


@dataclass(frozen=True)
class TechnicalValidationGates:
    diagnostic_pnl_tolerance_pct: float = 0.05
    diagnostic_drawdown_tolerance_pct: float = 0.05
    diagnostic_closed_trade_tolerance: int = 1
    atr_max_pnl_giveback_pct: float = 1.0
    atr_min_closed_trades: int = 100
    soft_min_closed_trades: int = 100
    hard_min_closed_trades: int = 100
    metric_tolerance_pct: float = 0.005
    window_min_count: int = 4
    window_min_pnl_positive_share: float = 0.70
    window_min_drawdown_not_worse_share: float = 0.70

    def as_dict(self) -> dict[str, float | int]:
        return {
            "diagnostic_pnl_tolerance_pct": self.diagnostic_pnl_tolerance_pct,
            "diagnostic_drawdown_tolerance_pct": (
                self.diagnostic_drawdown_tolerance_pct
            ),
            "diagnostic_closed_trade_tolerance": (
                self.diagnostic_closed_trade_tolerance
            ),
            "atr_max_pnl_giveback_pct": self.atr_max_pnl_giveback_pct,
            "atr_min_closed_trades": self.atr_min_closed_trades,
            "soft_min_closed_trades": self.soft_min_closed_trades,
            "hard_min_closed_trades": self.hard_min_closed_trades,
            "metric_tolerance_pct": self.metric_tolerance_pct,
            "window_min_count": self.window_min_count,
            "window_min_pnl_positive_share": self.window_min_pnl_positive_share,
            "window_min_drawdown_not_worse_share": (
                self.window_min_drawdown_not_worse_share
            ),
        }


DEFAULT_TECHNICAL_VALIDATION_GATES = TechnicalValidationGates()


def load_technical_run_metrics(
    path: str | Path,
    *,
    name: str = "",
    mode: str = "",
) -> dict[str, Any]:
    """Load the minimal metrics needed to validate a technical overlay run."""

    output_dir = _resolve_run_dir(path)
    status = _load_json(output_dir / "status.json")
    flash_attribution = _load_json(output_dir / "flash_attribution_summary.json")
    run_summary = _load_json(output_dir / "run_summary.json")
    walk_forward = _load_json(output_dir / "walk_forward_report.json")

    live = _mapping(status.get("live_session"))
    real_trades = _mapping(status.get("real_trades"))
    flash_summary = _mapping(flash_attribution.get("summary"))

    initial_capital = _first_number(status.get("initial_capital"), 1000.0)
    pnl_usd = _first_number(
        live.get("clean_panteon_total_pnl_usd"),
        live.get("clean_panteon_realized_pnl_usd"),
        live.get("panteon_owned_total_pnl_usd"),
        live.get("panteon_owned_realized_pnl_usd"),
        status.get("pnl_usd"),
    )
    pnl_pct = _first_number(
        live.get("clean_panteon_pnl_pct"),
        live.get("panteon_owned_pnl_pct"),
        live.get("panteon_pnl_pct"),
        (pnl_usd / initial_capital * 100.0) if initial_capital > 0.0 else 0.0,
    )
    max_drawdown_pct = _first_number(
        live.get("clean_panteon_max_drawdown_pct"),
        status.get("clean_panteon_max_drawdown_pct"),
        live.get("panteon_max_drawdown_pct"),
        status.get("panteon_max_drawdown_pct"),
        status.get("max_drawdown_pct"),
    )
    realized_max_drawdown_pct = _first_number(
        live.get("clean_panteon_realized_max_drawdown_pct"),
        status.get("clean_panteon_realized_max_drawdown_pct"),
        live.get("panteon_realized_max_drawdown_pct"),
        status.get("panteon_realized_max_drawdown_pct"),
        max_drawdown_pct,
    )
    closed_trades = int(_first_number(
        live.get("clean_real_closed_trades"),
        live.get("real_closed_trades"),
        real_trades.get("clean_closed"),
        real_trades.get("closed"),
        flash_summary.get("closed_trades"),
    ))
    winning_trades = int(_first_number(
        live.get("clean_real_successful_trades"),
        live.get("real_successful_trades"),
        real_trades.get("clean_successful"),
        real_trades.get("successful"),
        flash_summary.get("winning_trades"),
    ))
    losing_trades = int(_first_number(
        live.get("clean_real_unsuccessful_trades"),
        live.get("real_unsuccessful_trades"),
        real_trades.get("clean_unsuccessful"),
        real_trades.get("unsuccessful"),
        flash_summary.get("losing_trades"),
    ))
    equity_curve = _float_list(
        status.get("clean_panteon_equity_curve")
        or status.get("panteon_equity_curve")
        or status.get("equity_curve")
    )
    trade_windows = _trade_windows_from_walk_forward_report(walk_forward)
    win_rate_pct = (
        winning_trades / float(closed_trades) * 100.0
        if closed_trades > 0
        else 0.0
    )

    return {
        "name": str(name or ""),
        "mode": str(mode or ""),
        "output_dir": str(output_dir),
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "realized_max_drawdown_pct": realized_max_drawdown_pct,
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate_pct,
        "selected_signals": int(_first_number(flash_summary.get("selected_signals"))),
        "filled_signals": int(_first_number(flash_summary.get("filled_signals"))),
        "bars_processed": int(_first_number(run_summary.get("bars_processed"))),
        "equity_curve": equity_curve,
        "equity_points": len(equity_curve),
        "trade_windows": trade_windows,
        "trade_window_count": len(trade_windows),
    }


def evaluate_technical_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gates: TechnicalValidationGates = DEFAULT_TECHNICAL_VALIDATION_GATES,
) -> dict[str, Any]:
    """Evaluate one RSI/MACD/ATR candidate against baseline metrics."""

    mode = str(candidate.get("mode") or MODE_HARD_GATE)
    deltas = _deltas(baseline, candidate)
    checks: list[dict[str, Any]] = []

    if mode not in KNOWN_MODES:
        checks.append(_check(
            gate="mode_supported",
            value=mode,
            threshold=",".join(sorted(KNOWN_MODES)),
            status="failed",
        ))
    elif mode == MODE_DIAGNOSTIC:
        checks.extend(_diagnostic_checks(baseline, candidate, gates))
    elif mode == MODE_ATR_SIZING:
        checks.extend(_atr_sizing_checks(candidate, deltas, gates))
    elif mode == MODE_SOFT_OVERLAY:
        checks.extend(_soft_overlay_checks(candidate, deltas, gates))
    elif mode == MODE_HARD_GATE:
        checks.extend(_hard_gate_checks(candidate, deltas, gates))

    window_gate = evaluate_temporal_trade_window_gate(
        baseline,
        candidate,
        gates=gates,
    )
    gate_name = "temporal_trade_window_gate"
    if window_gate is None:
        window_gate = evaluate_equity_window_gate(
            baseline,
            candidate,
            gates=gates,
        )
        gate_name = "equity_window_gate"
    if window_gate is not None:
        checks.append(_check(
            gate=gate_name,
            value=window_gate["summary"],
            threshold={
                "min_windows": gates.window_min_count,
                "min_pnl_positive_share": gates.window_min_pnl_positive_share,
                "min_drawdown_not_worse_share": (
                    gates.window_min_drawdown_not_worse_share
                ),
            },
            status="passed" if window_gate["promotion_eligible"] else "failed",
        ))

    failures = [
        str(check["gate"])
        for check in checks
        if check.get("status") != "passed"
    ]
    return {
        "name": str(candidate.get("name") or ""),
        "mode": mode,
        "promotion_eligible": not failures,
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "deltas": deltas,
        "checks": checks,
        "failures": failures,
        **({"window_gate": window_gate} if window_gate is not None else {}),
    }


def evaluate_temporal_trade_window_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gates: TechnicalValidationGates = DEFAULT_TECHNICAL_VALIDATION_GATES,
) -> dict[str, Any] | None:
    baseline_windows = _trade_windows_from_row(baseline.get("trade_windows"))
    candidate_windows = _trade_windows_from_row(candidate.get("trade_windows"))
    if not baseline_windows and not candidate_windows:
        return None

    windows: list[dict[str, Any]] = []
    for baseline_window, candidate_window in zip(baseline_windows, candidate_windows):
        pnl_delta = candidate_window["net_pnl"] - baseline_window["net_pnl"]
        drawdown_delta = (
            candidate_window["max_drawdown_pnl"]
            - baseline_window["max_drawdown_pnl"]
        )
        windows.append({
            "window": candidate_window["window"],
            "baseline": baseline_window,
            "candidate": candidate_window,
            "pnl_delta": pnl_delta,
            "max_drawdown_delta": drawdown_delta,
            "pnl_delta_positive": pnl_delta > gates.metric_tolerance_pct,
            "drawdown_not_worse": drawdown_delta <= gates.metric_tolerance_pct,
        })

    return _window_gate_result(
        source="walk_forward_temporal_trade_windows",
        windows=windows,
        baseline_window_count=len(baseline_windows),
        candidate_window_count=len(candidate_windows),
        gates=gates,
    )


def evaluate_equity_window_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gates: TechnicalValidationGates = DEFAULT_TECHNICAL_VALIDATION_GATES,
    window_count: int | None = None,
) -> dict[str, Any] | None:
    baseline_windows = equity_windows_from_curve(
        baseline.get("equity_curve"),
        window_count=window_count,
    )
    candidate_windows = equity_windows_from_curve(
        candidate.get("equity_curve"),
        window_count=window_count,
    )
    if not baseline_windows and not candidate_windows:
        return None

    windows: list[dict[str, Any]] = []
    for baseline_window, candidate_window in zip(baseline_windows, candidate_windows):
        pnl_delta = candidate_window["pnl_pct"] - baseline_window["pnl_pct"]
        drawdown_delta = (
            candidate_window["max_drawdown_pct"]
            - baseline_window["max_drawdown_pct"]
        )
        windows.append({
            "window": candidate_window["window"],
            "baseline": baseline_window,
            "candidate": candidate_window,
            "pnl_delta_pct": pnl_delta,
            "max_drawdown_delta_pct": drawdown_delta,
            "pnl_delta_positive": pnl_delta > gates.metric_tolerance_pct,
            "drawdown_not_worse": drawdown_delta <= gates.metric_tolerance_pct,
        })

    return _window_gate_result(
        source="sampled_equity_curve",
        windows=windows,
        baseline_window_count=len(baseline_windows),
        candidate_window_count=len(candidate_windows),
        gates=gates,
    )


def _window_gate_result(
    *,
    source: str,
    windows: Sequence[Mapping[str, Any]],
    baseline_window_count: int,
    candidate_window_count: int,
    gates: TechnicalValidationGates,
) -> dict[str, Any]:
    window_count = len(windows)
    pnl_positive = sum(1 for window in windows if window["pnl_delta_positive"])
    drawdown_not_worse = sum(1 for window in windows if window["drawdown_not_worse"])
    pnl_share = pnl_positive / float(window_count) if window_count else 0.0
    drawdown_share = (
        drawdown_not_worse / float(window_count) if window_count else 0.0
    )
    checks = [
        _check(
            gate="equity_window_count_match",
            value={
                "baseline": baseline_window_count,
                "candidate": candidate_window_count,
            },
            threshold="equal",
            status=(
                "passed"
                if baseline_window_count == candidate_window_count
                else "failed"
            ),
        ),
        _check(
            gate=f"equity_windows>={gates.window_min_count}",
            value=window_count,
            threshold=gates.window_min_count,
            status=(
                "passed" if window_count >= gates.window_min_count else "failed"
            ),
        ),
        _check(
            gate=(
                "window_pnl_positive_share>="
                f"{gates.window_min_pnl_positive_share:.2f}"
            ),
            value=pnl_share,
            threshold=gates.window_min_pnl_positive_share,
            status=(
                "passed"
                if pnl_share >= gates.window_min_pnl_positive_share
                else "failed"
            ),
        ),
        _check(
            gate=(
                "window_drawdown_not_worse_share>="
                f"{gates.window_min_drawdown_not_worse_share:.2f}"
            ),
            value=drawdown_share,
            threshold=gates.window_min_drawdown_not_worse_share,
            status=(
                "passed"
                if drawdown_share >= gates.window_min_drawdown_not_worse_share
                else "failed"
            ),
        ),
    ]
    failures = [
        str(check["gate"])
        for check in checks
        if check.get("status") != "passed"
    ]
    return {
        "source": source,
        "promotion_eligible": not failures,
        "checks": checks,
        "failures": failures,
        "summary": {
            "windows": window_count,
            "baseline_windows": baseline_window_count,
            "candidate_windows": candidate_window_count,
            "pnl_delta_positive_windows": pnl_positive,
            "pnl_delta_positive_share": pnl_share,
            "drawdown_not_worse_windows": drawdown_not_worse,
            "drawdown_not_worse_share": drawdown_share,
        },
        "windows": windows,
    }


def equity_windows_from_curve(
    curve: object,
    *,
    window_count: int | None = None,
) -> list[dict[str, Any]]:
    values = _float_list(curve)
    if len(values) < 2:
        return []

    max_windows = len(values) - 1
    if window_count is None:
        windows_count = max_windows
    else:
        windows_count = max(1, min(int(window_count), max_windows))

    windows: list[dict[str, Any]] = []
    for index in range(windows_count):
        start_index = round(index * max_windows / windows_count)
        end_index = round((index + 1) * max_windows / windows_count)
        if end_index <= start_index:
            end_index = min(start_index + 1, max_windows)
        window_values = values[start_index:end_index + 1]
        start = window_values[0]
        end = window_values[-1]
        pnl_pct = ((end - start) / start * 100.0) if start > 0.0 else 0.0
        windows.append({
            "window": f"window_{index + 1:02d}",
            "start_index": start_index,
            "end_index": end_index,
            "start_equity": start,
            "end_equity": end,
            "pnl_pct": pnl_pct,
            "max_drawdown_pct": _equity_curve_drawdown_pct(window_values),
        })
    return windows


def build_technical_validation_report(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    gates: TechnicalValidationGates = DEFAULT_TECHNICAL_VALIDATION_GATES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    for candidate in candidates:
        result = evaluate_technical_candidate(baseline, candidate, gates=gates)
        rows.append(result)
        eligible += int(result["promotion_eligible"])
    return {
        "schema": "panteon_flash_technical_validation_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates.as_dict(),
        "baseline": dict(baseline),
        "summary": {
            "candidates": len(rows),
            "promotion_eligible": eligible,
            "promotion_rejected": len(rows) - eligible,
        },
        "candidates": rows,
    }


def write_technical_validation_report(
    *,
    output_json: str | Path,
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    gates: TechnicalValidationGates = DEFAULT_TECHNICAL_VALIDATION_GATES,
    output_md: str | Path | None = None,
) -> tuple[Path, Path | None]:
    report = build_technical_validation_report(
        baseline,
        candidates,
        gates=gates,
    )
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path: Path | None = None
    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def _diagnostic_checks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gates: TechnicalValidationGates,
) -> list[dict[str, Any]]:
    closed_delta = int(_metric(candidate, "closed_trades") or 0) - int(
        _metric(baseline, "closed_trades") or 0
    )
    deltas = _deltas(baseline, candidate)
    return [
        _check(
            gate=f"abs(pnl_pct_delta)<={gates.diagnostic_pnl_tolerance_pct:.2f}",
            value=abs(float(deltas.get("pnl_pct", 0.0))),
            threshold=gates.diagnostic_pnl_tolerance_pct,
            status=(
                "passed"
                if abs(float(deltas.get("pnl_pct", 0.0)))
                <= gates.diagnostic_pnl_tolerance_pct
                else "failed"
            ),
        ),
        _check(
            gate=(
                "abs(max_drawdown_pct_delta)<="
                f"{gates.diagnostic_drawdown_tolerance_pct:.2f}"
            ),
            value=abs(float(deltas.get("max_drawdown_pct", 0.0))),
            threshold=gates.diagnostic_drawdown_tolerance_pct,
            status=(
                "passed"
                if abs(float(deltas.get("max_drawdown_pct", 0.0)))
                <= gates.diagnostic_drawdown_tolerance_pct
                else "failed"
            ),
        ),
        _check(
            gate=(
                "abs(closed_trades_delta)<="
                f"{gates.diagnostic_closed_trade_tolerance}"
            ),
            value=abs(closed_delta),
            threshold=gates.diagnostic_closed_trade_tolerance,
            status=(
                "passed"
                if abs(closed_delta) <= gates.diagnostic_closed_trade_tolerance
                else "failed"
            ),
        ),
    ]


def _atr_sizing_checks(
    candidate: Mapping[str, Any],
    deltas: Mapping[str, float],
    gates: TechnicalValidationGates,
) -> list[dict[str, Any]]:
    pnl_delta = float(deltas.get("pnl_pct", 0.0))
    max_dd_delta = float(deltas.get("max_drawdown_pct", 0.0))
    realized_dd_delta = float(deltas.get("realized_max_drawdown_pct", 0.0))
    closed = int(_metric(candidate, "closed_trades") or 0)
    drawdown_improved = (
        max_dd_delta < -gates.metric_tolerance_pct
        or realized_dd_delta < -gates.metric_tolerance_pct
    )
    return [
        _check(
            gate=f"pnl_pct_delta>=-{gates.atr_max_pnl_giveback_pct:.2f}",
            value=pnl_delta,
            threshold=-gates.atr_max_pnl_giveback_pct,
            status=(
                "passed"
                if pnl_delta >= -gates.atr_max_pnl_giveback_pct
                else "failed"
            ),
        ),
        _check(
            gate="max_or_realized_drawdown_delta<0",
            value=min(max_dd_delta, realized_dd_delta),
            threshold=0.0,
            status="passed" if drawdown_improved else "failed",
        ),
        _check(
            gate=f"closed_trades>={gates.atr_min_closed_trades}",
            value=closed,
            threshold=gates.atr_min_closed_trades,
            status=(
                "passed"
                if closed >= gates.atr_min_closed_trades
                else "failed"
            ),
        ),
    ]


def _soft_overlay_checks(
    candidate: Mapping[str, Any],
    deltas: Mapping[str, float],
    gates: TechnicalValidationGates,
) -> list[dict[str, Any]]:
    pnl_delta = float(deltas.get("pnl_pct", 0.0))
    max_dd_delta = float(deltas.get("max_drawdown_pct", 0.0))
    closed = int(_metric(candidate, "closed_trades") or 0)
    return [
        _check(
            gate="pnl_pct_delta>0",
            value=pnl_delta,
            threshold=0.0,
            status="passed" if pnl_delta > gates.metric_tolerance_pct else "failed",
        ),
        _check(
            gate="max_drawdown_pct_delta<=0",
            value=max_dd_delta,
            threshold=0.0,
            status=(
                "passed"
                if max_dd_delta <= gates.metric_tolerance_pct
                else "failed"
            ),
        ),
        _check(
            gate=f"closed_trades>={gates.soft_min_closed_trades}",
            value=closed,
            threshold=gates.soft_min_closed_trades,
            status=(
                "passed"
                if closed >= gates.soft_min_closed_trades
                else "failed"
            ),
        ),
    ]


def _hard_gate_checks(
    candidate: Mapping[str, Any],
    deltas: Mapping[str, float],
    gates: TechnicalValidationGates,
) -> list[dict[str, Any]]:
    pnl_delta = float(deltas.get("pnl_pct", 0.0))
    max_dd_delta = float(deltas.get("max_drawdown_pct", 0.0))
    realized_dd_delta = float(deltas.get("realized_max_drawdown_pct", 0.0))
    win_rate_delta = float(deltas.get("win_rate_pct", 0.0))
    closed = int(_metric(candidate, "closed_trades") or 0)
    return [
        _check(
            gate="pnl_pct_delta>0",
            value=pnl_delta,
            threshold=0.0,
            status="passed" if pnl_delta > gates.metric_tolerance_pct else "failed",
        ),
        _check(
            gate="max_drawdown_pct_delta<0",
            value=max_dd_delta,
            threshold=0.0,
            status=(
                "passed"
                if max_dd_delta < -gates.metric_tolerance_pct
                else "failed"
            ),
        ),
        _check(
            gate="realized_max_drawdown_pct_delta<0",
            value=realized_dd_delta,
            threshold=0.0,
            status=(
                "passed"
                if realized_dd_delta < -gates.metric_tolerance_pct
                else "failed"
            ),
        ),
        _check(
            gate=f"closed_trades>={gates.hard_min_closed_trades}",
            value=closed,
            threshold=gates.hard_min_closed_trades,
            status=(
                "passed"
                if closed >= gates.hard_min_closed_trades
                else "failed"
            ),
        ),
        _check(
            gate="win_rate_delta>=0",
            value=win_rate_delta,
            threshold=0.0,
            status="passed" if win_rate_delta >= 0.0 else "failed",
        ),
    ]


def _deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    keys = (
        "pnl_pct",
        "pnl_usd",
        "max_drawdown_pct",
        "realized_max_drawdown_pct",
        "closed_trades",
        "win_rate_pct",
        "selected_signals",
        "filled_signals",
    )
    deltas: dict[str, float] = {}
    for key in keys:
        base = _metric(baseline, key)
        cand = _metric(candidate, key)
        if base is None or cand is None:
            continue
        deltas[key] = cand - base
    return deltas


def _metric(row: Mapping[str, Any], key: str) -> float | None:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def _check(
    *,
    gate: str,
    value: object,
    threshold: object,
    status: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "value": value,
        "threshold": threshold,
        "status": status,
    }


def _resolve_run_dir(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source.parent
    if (source / "status.json").exists():
        return source
    candidates = sorted(
        source.rglob("status.json") if source.exists() else (),
        key=lambda item: item.stat().st_mtime,
    )
    if candidates:
        return candidates[-1].parent
    return source


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _trade_windows_from_walk_forward_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    temporal = _mapping(report.get("temporal_windows"))
    return _trade_windows_from_row(temporal.get("windows"))


def _trade_windows_from_row(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    windows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw)
        if not row:
            continue
        stats = _mapping(row.get("stats"))
        windows.append({
            "window": str(row.get("window") or f"window_{index + 1:02d}"),
            "bar_start": int(_first_number(row.get("bar_start"))),
            "bar_end": int(_first_number(row.get("bar_end"))),
            "closed_trades": int(_first_number(
                row.get("closed_trades"),
                stats.get("closed_trades"),
            )),
            "net_pnl": _first_number(row.get("net_pnl"), stats.get("net_pnl")),
            "max_drawdown_pnl": _first_number(
                row.get("max_drawdown_pnl"),
                stats.get("max_drawdown_pnl"),
            ),
            "winrate_pct": _first_number(
                row.get("winrate_pct"),
                stats.get("winrate_pct"),
            ),
        })
    return windows


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_number(*values: object) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _float_list(values: object) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _equity_curve_drawdown_pct(values: Sequence[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = value
        if peak <= 0.0:
            continue
        drawdown = (peak - value) / peak * 100.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_count_share(value: object, total: object) -> str:
    try:
        count_value = int(float(value))
        total_value = int(float(total))
    except (TypeError, ValueError):
        return "-"
    return f"{count_value}/{total_value}" if total_value > 0 else "-"


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Panteon Flash Technical Validation Report",
        "",
        "| Candidate | Mode | Promotion | Window gate | Window PnL | Window DD | PnL | PnL delta | Max DD | DD delta | Closed | Win rate delta | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report.get("candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        raw = _mapping(candidate.get("candidate"))
        deltas = _mapping(candidate.get("deltas"))
        window_gate = _mapping(candidate.get("window_gate"))
        window_summary = _mapping(window_gate.get("summary"))
        failures = ", ".join(str(item) for item in candidate.get("failures", []) or [])
        lines.append(
            "| {name} | {mode} | {promotion} | {window_gate} | {window_pnl} | {window_dd} | {pnl} | {pnl_delta} | {dd} | {dd_delta} | {closed} | {win_delta} | {failures} |".format(
                name=str(candidate.get("name") or ""),
                mode=str(candidate.get("mode") or ""),
                promotion="PASS" if candidate.get("promotion_eligible") else "FAIL",
                window_gate=(
                    "PASS"
                    if window_gate.get("promotion_eligible") is True
                    else "FAIL"
                    if window_gate
                    else "-"
                ),
                window_pnl=_format_count_share(
                    window_summary.get("pnl_delta_positive_windows"),
                    window_summary.get("windows"),
                ),
                window_dd=_format_count_share(
                    window_summary.get("drawdown_not_worse_windows"),
                    window_summary.get("windows"),
                ),
                pnl=_format_pct(raw.get("pnl_pct")),
                pnl_delta=_format_pct(deltas.get("pnl_pct")),
                dd=_format_pct(raw.get("max_drawdown_pct")),
                dd_delta=_format_pct(deltas.get("max_drawdown_pct")),
                closed=int(_first_number(raw.get("closed_trades"))),
                win_delta=_format_pct(deltas.get("win_rate_pct")),
                failures=failures or "-",
            )
        )
    return "\n".join(lines) + "\n"


def _parse_candidate_spec(raw: str) -> tuple[str, str, Path]:
    parts = [part.strip() for part in str(raw or "").split("|", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError("--candidate must use 'name|mode|path'")
    name, mode, path = parts
    if mode not in KNOWN_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(KNOWN_MODES))}")
    return name, mode, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline run path")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as 'name|mode|path'. Repeat for each candidate.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)

    baseline = load_technical_run_metrics(
        args.baseline,
        name=MODE_BASELINE,
        mode=MODE_BASELINE,
    )
    candidates = [
        load_technical_run_metrics(path, name=name, mode=mode)
        for name, mode, path in (_parse_candidate_spec(raw) for raw in args.candidate)
    ]
    json_path, md_path = write_technical_validation_report(
        output_json=args.output_json,
        output_md=args.output_md,
        baseline=baseline,
        candidates=tuple(candidates),
    )
    print(json_path)
    if md_path is not None:
        print(md_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
