from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ACTOR_LABEL = "LiveOIBreakout"
DEFAULT_MOM_MIN_GRID = (0.0005, 0.001, 0.0015, 0.002)
WAIT_REASON = "check_interval_wait"
SIGNAL_REASONS = {"candidate_long", "candidate_short"}
TRANSITION_TARGET_REGIMES = {"bullish", "bearish", "volatile", "crash"}


def resolve_causal_paths(inputs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("causal_entry_decisions.jsonl")))
        elif path.is_file():
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def analyze_paths(
    paths: Sequence[str | Path],
    *,
    actor_label: str = DEFAULT_ACTOR_LABEL,
    mom_min_grid: Sequence[float] = DEFAULT_MOM_MIN_GRID,
    min_real_checks_per_symbol: int = 3,
) -> dict[str, Any]:
    causal_paths = [Path(path) for path in paths]
    reason_counts: Counter[str] = Counter()
    real_reason_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    effective_checks_by_symbol: Counter[str] = Counter()
    waits_by_symbol: Counter[str] = Counter()
    symbols: set[str] = set()
    momentums: list[float] = []
    thresholds: list[float] = []
    diagnostics = 0
    real_checks = 0
    wait_checks = 0
    candidate_signals = 0
    transition_breakouts = 0
    confirmed_transition_breakouts = 0
    transition_breakouts_by_symbol: Counter[str] = Counter()
    bad_rows = 0
    rows = 0
    files_detail: dict[str, dict[str, Any]] = {}
    expected = max(0, int(min_real_checks_per_symbol))
    file_min_checks: list[int] = []
    file_sufficiency: list[bool] = []
    grid = {
        _grid_key(value): {
            "mom_min": float(value),
            "would_pass_breakout_and_momentum": 0,
            "long_count": 0,
            "short_count": 0,
            "by_symbol": {},
        }
        for value in mom_min_grid
    }
    grid_by_symbol: dict[str, Counter[str]] = {
        key: Counter() for key in grid
    }

    for path in causal_paths:
        detail: dict[str, Any] = {
            "rows": 0,
            "bad_rows": 0,
            "candidate_diagnostics": 0,
            "real_checks": 0,
            "wait_checks": 0,
        }
        detail_checks_by_symbol: Counter[str] = Counter()
        detail_waits_by_symbol: Counter[str] = Counter()
        for row in _iter_jsonl(path):
            if row is None:
                bad_rows += 1
                detail["bad_rows"] += 1
                continue
            rows += 1
            detail["rows"] += 1
            for item in _iter_actor_diagnostics(row, actor_label=actor_label):
                diagnostics += 1
                detail["candidate_diagnostics"] += 1
                symbol = item["symbol"]
                if symbol:
                    symbols.add(symbol)
                diag = item["diagnostics"]
                reason = _reason(diag)
                reason_counts[reason] += 1
                _collect_regime(row, symbol, regime_counts)

                if reason == WAIT_REASON:
                    wait_checks += 1
                    detail["wait_checks"] += 1
                    if symbol:
                        waits_by_symbol[symbol] += 1
                        detail_waits_by_symbol[symbol] += 1
                    continue

                real_checks += 1
                detail["real_checks"] += 1
                real_reason_counts[reason] += 1
                if symbol:
                    effective_checks_by_symbol[symbol] += 1
                    detail_checks_by_symbol[symbol] += 1
                if reason in SIGNAL_REASONS or _is_signal_action(diag.get("action")):
                    candidate_signals += 1
                if _is_transition_breakout_context(item["row"], diag):
                    transition_breakouts += 1
                    if _is_confirmed_transition_breakout(item["row"], diag):
                        confirmed_transition_breakouts += 1
                        if symbol:
                            transition_breakouts_by_symbol[symbol] += 1

                momentum = _float_value(diag.get("momentum"))
                if momentum is None:
                    continue
                momentums.append(momentum)
                threshold = _float_value(diag.get("momentum_threshold"))
                if threshold is not None:
                    thresholds.append(threshold)
                if not _has_breakout(diag):
                    continue
                for key, entry in grid.items():
                    mom_min = float(entry["mom_min"])
                    if momentum > mom_min:
                        entry["would_pass_breakout_and_momentum"] += 1
                        entry["long_count"] += 1
                        if symbol:
                            grid_by_symbol[key][symbol] += 1
                    elif momentum < -mom_min:
                        entry["would_pass_breakout_and_momentum"] += 1
                        entry["short_count"] += 1
                        if symbol:
                            grid_by_symbol[key][symbol] += 1
        detail_checks = dict(sorted((symbol, int(count)) for symbol, count in detail_checks_by_symbol.items()))
        detail_min_checks = min(detail_checks.values()) if detail_checks else 0
        detail_sufficient = bool(detail_checks) and all(count >= expected for count in detail_checks.values())
        detail["effective_checks_by_symbol"] = detail_checks
        detail["wait_checks_by_symbol"] = dict(
            sorted((symbol, int(count)) for symbol, count in detail_waits_by_symbol.items())
        )
        detail["min_effective_checks_per_symbol"] = detail_min_checks
        detail["sufficient_real_checks"] = detail_sufficient
        files_detail[str(path)] = detail
        file_min_checks.append(detail_min_checks)
        file_sufficiency.append(detail_sufficient)

    for key, counts in grid_by_symbol.items():
        grid[key]["by_symbol"] = _sorted_counter(counts)

    checks = dict(sorted((symbol, int(count)) for symbol, count in effective_checks_by_symbol.items()))
    min_checks = min(checks.values()) if checks else 0
    max_checks = max(checks.values()) if checks else 0
    sufficient = bool(checks) and all(count >= expected for count in checks.values())
    min_file_checks = min(file_min_checks) if file_min_checks else 0
    sufficient_per_file = bool(file_sufficiency) and all(file_sufficiency)
    activation_blockers = _activation_blockers(
        files=causal_paths,
        diagnostics=diagnostics,
        real_checks=real_checks,
        candidate_signals=candidate_signals,
        sufficient_real_checks=sufficient,
        sufficient_real_checks_per_file=sufficient_per_file,
        real_reason_counts=real_reason_counts,
    )
    diagnostic_warnings = _diagnostic_warnings(real_reason_counts)
    report: dict[str, Any] = {
        "actor_label": actor_label,
        "files": [str(path) for path in causal_paths],
        "files_detail": files_detail,
        "rows": rows,
        "bad_rows": bad_rows,
        "candidate_diagnostics": diagnostics,
        "real_check_count": real_checks,
        "wait_check_count": wait_checks,
        "candidate_signal_count": candidate_signals,
        "symbols": sorted(symbols),
        "reason_counts": _sorted_counter(reason_counts),
        "real_reason_counts": _sorted_counter(real_reason_counts),
        "regime_counts": _sorted_counter(regime_counts),
        "effective_checks_by_symbol": checks,
        "wait_checks_by_symbol": dict(sorted((symbol, int(count)) for symbol, count in waits_by_symbol.items())),
        "min_effective_checks_per_symbol": min_checks,
        "max_effective_checks_per_symbol": max_checks,
        "min_effective_checks_per_file_symbol": min_file_checks,
        "expected_min_real_checks_per_symbol": expected,
        "sufficient_real_checks": sufficient,
        "sufficient_real_checks_per_file": sufficient_per_file,
        "momentum": _momentum_summary(momentums, thresholds),
        "near_miss_grid": grid,
        "transition_breakouts": {
            "total": int(transition_breakouts),
            "confirmed": int(confirmed_transition_breakouts),
            "by_symbol": _sorted_counter(transition_breakouts_by_symbol),
        },
        "activation_blockers": activation_blockers,
        "diagnostic_warnings": diagnostic_warnings,
        "hard_blocked": bool(activation_blockers),
        "notes": [
            "Wait rows do not contain fresh LiveOIBreakout momentum/volume calculations.",
            "Use this report for activation diagnosis only; it is not a live trading approval.",
        ],
    }
    return report


def write_reports(report: Mapping[str, Any], out: str | Path) -> tuple[Path, Path]:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = out_path.with_suffix(".md")
    md_path.write_text(format_markdown_report(report), encoding="utf-8")
    return out_path, md_path


def format_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report.get('actor_label', DEFAULT_ACTOR_LABEL)} Activation Report",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "rows",
        "bad_rows",
        "candidate_diagnostics",
        "real_check_count",
        "wait_check_count",
        "candidate_signal_count",
        "min_effective_checks_per_symbol",
        "min_effective_checks_per_file_symbol",
        "expected_min_real_checks_per_symbol",
    ):
        lines.append(f"| {key} | {report.get(key, 0)} |")
    lines.append(f"| hard_blocked | {str(bool(report.get('hard_blocked'))).lower()} |")

    lines.extend(["", "## Activation Blockers", "", "| blocker |", "|---|"])
    blockers = report.get("activation_blockers")
    if isinstance(blockers, list) and blockers:
        for blocker in blockers:
            lines.append(f"| {blocker} |")
    else:
        lines.append("| none |")

    lines.extend(["", "## Diagnostic Warnings", "", "| warning |", "|---|"])
    warnings = report.get("diagnostic_warnings")
    if isinstance(warnings, list) and warnings:
        for warning in warnings:
            lines.append(f"| {warning} |")
    else:
        lines.append("| none |")

    _append_mapping_table(lines, "Reasons", report.get("reason_counts", {}), "reason")
    _append_mapping_table(lines, "Real Check Reasons", report.get("real_reason_counts", {}), "reason")
    _append_mapping_table(lines, "Effective Checks By Symbol", report.get("effective_checks_by_symbol", {}), "symbol")

    lines.extend(["", "## Momentum", "", "| metric | value |", "|---|---:|"])
    momentum = report.get("momentum", {})
    if isinstance(momentum, Mapping):
        for key in ("count", "abs_max", "abs_p50", "abs_p90", "observed_threshold_min"):
            lines.append(f"| {key} | {momentum.get(key, 0)} |")

    transition = report.get("transition_breakouts", {})
    lines.extend(["", "## Transition Breakouts", "", "| metric | value |", "|---|---:|"])
    if isinstance(transition, Mapping):
        for key in ("total", "confirmed"):
            lines.append(f"| {key} | {transition.get(key, 0)} |")
    else:
        lines.append("| total | 0 |")
        lines.append("| confirmed | 0 |")

    lines.extend([
        "",
        "## Near Miss Grid",
        "",
        "| mom_min | pass_count | long | short |",
        "|---:|---:|---:|---:|",
    ])
    grid = report.get("near_miss_grid", {})
    if isinstance(grid, Mapping) and grid:
        for key, item in grid.items():
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| "
                f"{key} | "
                f"{item.get('would_pass_breakout_and_momentum', 0)} | "
                f"{item.get('long_count', 0)} | "
                f"{item.get('short_count', 0)} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 |")

    notes = report.get("notes")
    if isinstance(notes, list) and notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze LiveOIBreakout activation diagnostics from causal_entry_decisions.jsonl."
    )
    parser.add_argument("paths", nargs="+", help="causal_entry_decisions.jsonl files or directories.")
    parser.add_argument("--actor-label", default=DEFAULT_ACTOR_LABEL)
    parser.add_argument(
        "--mom-min-grid",
        default=",".join(str(value) for value in DEFAULT_MOM_MIN_GRID),
        help="Comma-separated momentum thresholds to evaluate against observed breakout rows.",
    )
    parser.add_argument("--min-real-checks-per-symbol", type=int, default=3)
    parser.add_argument(
        "--out",
        default="Reports/Panteon3Canary/live_oi_breakout_activation_report.json",
        help="Output JSON path. A sibling .md report is written too.",
    )
    args = parser.parse_args(argv)

    paths = resolve_causal_paths(args.paths)
    report = analyze_paths(
        paths,
        actor_label=str(args.actor_label),
        mom_min_grid=_parse_float_list(args.mom_min_grid),
        min_real_checks_per_symbol=int(args.min_real_checks_per_symbol),
    )
    out_path, md_path = write_reports(report, args.out)
    print(json.dumps({"json": str(out_path), "markdown": str(md_path), "hard_blocked": report["hard_blocked"]}, ensure_ascii=False))
    return 1 if report["hard_blocked"] else 0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any] | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    yield None
                    continue
                yield row if isinstance(row, dict) else None
    except OSError:
        yield None


def _iter_actor_diagnostics(row: Mapping[str, Any], *, actor_label: str) -> Iterable[dict[str, Any]]:
    if _candidate_matches(row, actor_label):
        diagnostics = row.get("agent_diagnostics") or row.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            merged = dict(diagnostics)
            if "action" not in merged and row.get("action") is not None:
                merged["action"] = row.get("action")
            symbol = str(row.get("symbol") or "").strip()
            yield {"symbol": symbol, "diagnostics": merged, "row": row}

    decisions = row.get("flash_decisions")
    if not isinstance(decisions, list):
        return
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        symbol = str(decision.get("symbol") or "").strip()
        for candidate in _candidate_items(decision):
            if not _candidate_matches(candidate, actor_label):
                continue
            diagnostics = candidate.get("agent_diagnostics")
            if not isinstance(diagnostics, Mapping):
                continue
            merged = dict(diagnostics)
            if "action" not in merged and candidate.get("action") is not None:
                merged["action"] = candidate.get("action")
            if "reason" not in merged and candidate.get("reason") is not None:
                merged["reason"] = candidate.get("reason")
            yield {"symbol": symbol, "diagnostics": merged, "row": row}


def _candidate_items(decision: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("top_rejected_candidates", "candidates"):
        items = decision.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, Mapping):
                yield item


def _candidate_matches(candidate: Mapping[str, Any], actor_label: str) -> bool:
    wanted = _clean_actor_label(actor_label)
    for key in ("label", "actor_label", "actor", "actor_key"):
        value = _clean_actor_label(candidate.get(key))
        if value == wanted:
            return True
    return False


def _clean_actor_label(value: Any) -> str:
    text = str(value or "").strip()
    if "|" in text:
        text = text.split("|", 1)[0]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    for prefix in ("Solo_", "V_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def _collect_regime(row: Mapping[str, Any], symbol: str, regime_counts: Counter[str]) -> None:
    regime = ""
    regimes_by_symbol = row.get("regimes_by_symbol")
    if symbol and isinstance(regimes_by_symbol, Mapping):
        regime = str(regimes_by_symbol.get(symbol) or "").strip()
    if not regime:
        regime = str(row.get("regime") or "").strip()
    if regime:
        regime_counts[regime] += 1


def _context_value(
    row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    key: str,
) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    return diagnostics.get(key)


def _is_transition_breakout_context(
    row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    regime = str(_context_value(row, diagnostics, "regime") or "").lower()
    next_regime = str(_context_value(row, diagnostics, "next_regime") or "").lower()
    return regime == "range_low_vol" and next_regime in TRANSITION_TARGET_REGIMES


def _is_confirmed_transition_breakout(
    row: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> bool:
    diag = diagnostics
    if not isinstance(diag, Mapping):
        raw_diag = row.get("agent_diagnostics") or row.get("diagnostics") or {}
        diag = raw_diag if isinstance(raw_diag, Mapping) else {}
    if not _is_transition_breakout_context(row, diag):
        return False
    return all(
        bool(diag.get(key))
        for key in (
            "volume_spike",
            "oi_expansion",
            "close_outside_range",
            "spread_ok",
            "slippage_ok",
        )
    )


def _activation_blockers(
    *,
    files: Sequence[Path],
    diagnostics: int,
    real_checks: int,
    candidate_signals: int,
    sufficient_real_checks: bool,
    sufficient_real_checks_per_file: bool,
    real_reason_counts: Counter[str],
) -> list[str]:
    blockers: list[str] = []
    if not files:
        blockers.append("no_causal_entry_decisions")
    if diagnostics <= 0:
        blockers.append("missing_actor_diagnostics")
    if real_checks <= 0:
        blockers.append("zero_real_checks")
    if not sufficient_real_checks:
        blockers.append("insufficient_real_checks")
    if not sufficient_real_checks_per_file:
        blockers.append("insufficient_real_checks_per_file")
    if candidate_signals <= 0:
        blockers.append("zero_candidate_signals")
    if real_reason_counts:
        top_reason, _ = real_reason_counts.most_common(1)[0]
        if candidate_signals <= 0 and top_reason not in SIGNAL_REASONS:
            blockers.append(f"dominant_filter:{top_reason}")
    return blockers


def _diagnostic_warnings(real_reason_counts: Counter[str]) -> list[str]:
    if not real_reason_counts:
        return []
    top_reason, _ = real_reason_counts.most_common(1)[0]
    if top_reason in SIGNAL_REASONS:
        return []
    return [f"dominant_filter:{top_reason}"]


def _momentum_summary(momentums: Sequence[float], thresholds: Sequence[float]) -> dict[str, Any]:
    abs_values = sorted(abs(value) for value in momentums if math.isfinite(value))
    if not abs_values:
        return {
            "count": 0,
            "abs_max": 0.0,
            "abs_p50": 0.0,
            "abs_p90": 0.0,
            "observed_threshold_min": 0.0,
            "observed_threshold_max": 0.0,
        }
    clean_thresholds = [value for value in thresholds if math.isfinite(value)]
    return {
        "count": len(abs_values),
        "abs_max": max(abs_values),
        "abs_p50": _percentile(abs_values, 50),
        "abs_p90": _percentile(abs_values, 90),
        "observed_threshold_min": min(clean_thresholds) if clean_thresholds else 0.0,
        "observed_threshold_max": max(clean_thresholds) if clean_thresholds else 0.0,
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (float(percentile) / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[int(rank)])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _has_breakout(diagnostics: Mapping[str, Any]) -> bool:
    return bool(diagnostics.get("volume_spike")) or bool(diagnostics.get("oi_expansion"))


def _is_signal_action(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().upper()
        return text not in {"", "0", "HOLD", "NONE", "NOOP"}
    try:
        return int(value or 0) != 0
    except (TypeError, ValueError):
        return False


def _float_value(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _reason(diagnostics: Mapping[str, Any]) -> str:
    return str(diagnostics.get("reason") or "unknown").strip() or "unknown"


def _parse_float_list(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = raw
    values: list[float] = []
    for part in parts:
        value = _float_value(str(part).strip())
        if value is not None:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _grid_key(value: float) -> str:
    return f"{float(value):.6f}"


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    }


def _append_mapping_table(
    lines: list[str],
    title: str,
    mapping: Any,
    first_col: str,
) -> None:
    lines.extend(["", f"## {title}", "", f"| {first_col} | count |", "|---|---:|"])
    if isinstance(mapping, Mapping) and mapping:
        for key, value in mapping.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("| none | 0 |")


if __name__ == "__main__":
    raise SystemExit(main())
