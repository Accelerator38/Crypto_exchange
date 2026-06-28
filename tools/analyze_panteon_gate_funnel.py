from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def analyze_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "files": [str(Path(path)) for path in paths],
        "bars": 0,
        "bad_rows": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "raw_nonzero_bars": 0,
        "executable_nonzero_bars": 0,
        "real_signal_bars": 0,
        "filled_bars": 0,
        "rejected_bars": 0,
        "blocked_bars": 0,
        "decisions": 0,
        "symbols": [],
        "selected_actors": {},
        "decision_reasons": {},
        "candidate_totals": {
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "rejected_candidate_count": 0,
        },
        "candidate_rejection_reasons": {},
        "gate_funnel_top_blockers": {},
        "real_universe_reasons": {},
    }
    selected_actors: Counter[str] = Counter()
    decision_reasons: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    gate_top_blockers: Counter[str] = Counter()
    real_universe_reasons: Counter[str] = Counter()
    symbols: set[str] = set()
    candidate_totals = {
        "candidate_count": 0,
        "eligible_candidate_count": 0,
        "rejected_candidate_count": 0,
    }

    for path in paths:
        for row in _iter_jsonl(Path(path)):
            if row is None:
                report["bad_rows"] += 1
                continue
            report["bars"] += 1
            timestamp = str(row.get("timestamp") or "")
            if timestamp:
                if not report["first_timestamp"]:
                    report["first_timestamp"] = timestamp
                report["last_timestamp"] = timestamp
            if _positive_count(row, "raw_signal_count") or _nonempty_list(row, "raw_signals"):
                report["raw_nonzero_bars"] += 1
            if _positive_count(row, "executable_signal_count") or _nonempty_list(row, "executable_signals"):
                report["executable_nonzero_bars"] += 1
            if _has_real_signal_activity(row):
                report["real_signal_bars"] += 1
            if _positive_count(row, "n_filled"):
                report["filled_bars"] += 1
            if _positive_count(row, "n_rejected"):
                report["rejected_bars"] += 1
            if _positive_count(row, "n_blocked"):
                report["blocked_bars"] += 1

            _collect_real_universe_reasons(row, real_universe_reasons)
            _collect_gate_funnel(row, gate_top_blockers, candidate_totals, symbols)
            report["decisions"] += _collect_flash_decisions(
                row,
                selected_actors=selected_actors,
                decision_reasons=decision_reasons,
                candidate_reasons=candidate_reasons,
                candidate_totals=candidate_totals,
                symbols=symbols,
            )

    report["symbols"] = sorted(symbols)
    report["selected_actors"] = _sorted_counter(selected_actors)
    report["decision_reasons"] = _sorted_counter(decision_reasons)
    report["candidate_totals"] = dict(candidate_totals)
    report["candidate_rejection_reasons"] = _sorted_counter(candidate_reasons)
    report["gate_funnel_top_blockers"] = _sorted_counter(gate_top_blockers)
    report["real_universe_reasons"] = _sorted_counter(real_universe_reasons)
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
        "# Panteon Gate Funnel Report",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "bars",
        "real_signal_bars",
        "raw_nonzero_bars",
        "executable_nonzero_bars",
        "filled_bars",
        "rejected_bars",
        "blocked_bars",
        "decisions",
        "bad_rows",
    ):
        lines.append(f"| {key} | {report.get(key, 0)} |")

    sections = (
        ("Selected Actors", report.get("selected_actors", {})),
        ("Decision Reasons", report.get("decision_reasons", {})),
        ("Candidate Rejection Reasons", report.get("candidate_rejection_reasons", {})),
        ("Gate Funnel Top Blockers", report.get("gate_funnel_top_blockers", {})),
        ("Real Universe Reasons", report.get("real_universe_reasons", {})),
    )
    for title, mapping in sections:
        lines.extend(["", f"## {title}", "", "| reason | count |", "|---|---:|"])
        if isinstance(mapping, Mapping) and mapping:
            for reason, count in list(mapping.items())[:20]:
                lines.append(f"| {reason} | {count} |")
        else:
            lines.append("| none | 0 |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Panteon causal_entry_decisions.jsonl gate funnel diagnostics."
    )
    parser.add_argument("paths", nargs="+", help="One or more causal_entry_decisions.jsonl files.")
    parser.add_argument(
        "--out",
        default="gate_funnel_report.json",
        help="Output JSON path. A sibling .md report is written too.",
    )
    args = parser.parse_args(argv)

    report = analyze_files([Path(path) for path in args.paths])
    out_path, md_path = write_reports(report, args.out)
    print(json.dumps({"json": str(out_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any] | None]:
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


def _collect_flash_decisions(
    row: Mapping[str, Any],
    *,
    selected_actors: Counter[str],
    decision_reasons: Counter[str],
    candidate_reasons: Counter[str],
    candidate_totals: dict[str, int],
    symbols: set[str],
) -> int:
    decisions = row.get("flash_decisions")
    if not isinstance(decisions, list):
        return 0
    decision_count = 0
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        decision_count += 1
        symbol = str(decision.get("symbol") or "").strip()
        if symbol:
            symbols.add(symbol)
        selected = str(decision.get("selected_actor") or "").strip() or "unknown"
        selected_actors[selected] += 1
        reason = str(decision.get("reason") or "").strip() or "unknown"
        decision_reasons[reason] += 1
        candidate_count = _int_value(decision.get("candidate_count"))
        rejected_count = _int_value(decision.get("rejected_candidate_count"))
        eligible_count = _int_value(decision.get("eligible_candidate_count"))
        candidates = decision.get("candidates")
        if isinstance(candidates, list) and candidate_count <= 0:
            candidate_count = len(candidates)
            rejected_count = sum(
                1 for candidate in candidates
                if isinstance(candidate, Mapping) and bool(candidate.get("rejected"))
            )
            eligible_count = max(0, candidate_count - rejected_count)
        candidate_totals["candidate_count"] += candidate_count
        candidate_totals["rejected_candidate_count"] += rejected_count
        candidate_totals["eligible_candidate_count"] += eligible_count
        rejection_counts = decision.get("candidate_rejection_counts")
        if isinstance(rejection_counts, Mapping):
            for key, value in rejection_counts.items():
                reason_key = _normalize_reason(key)
                if reason_key:
                    candidate_reasons[reason_key] += _int_value(value)
        elif isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, Mapping) or not bool(candidate.get("rejected")):
                    continue
                reason_key = _normalize_reason(candidate.get("reason"))
                if reason_key:
                    candidate_reasons[reason_key] += 1
    return decision_count


def _collect_gate_funnel(
    row: Mapping[str, Any],
    gate_top_blockers: Counter[str],
    candidate_totals: dict[str, int],
    symbols: set[str],
) -> None:
    gate_funnel = row.get("flash_gate_funnel_by_symbol")
    if not isinstance(gate_funnel, Mapping):
        return
    for symbol, item in gate_funnel.items():
        clean_symbol = str(symbol or "").strip()
        if clean_symbol:
            symbols.add(clean_symbol)
        if not isinstance(item, Mapping):
            continue
        blocker = _normalize_reason(item.get("top_blocker"))
        if blocker:
            gate_top_blockers[blocker] += 1


def _collect_real_universe_reasons(
    row: Mapping[str, Any],
    real_universe_reasons: Counter[str],
) -> None:
    reasons = row.get("real_universe_symbol_reject_reasons")
    if not isinstance(reasons, Mapping):
        return
    for reason in reasons.values():
        reason_key = _normalize_reason(reason)
        if reason_key:
            real_universe_reasons[reason_key] += 1


def _has_real_signal_activity(row: Mapping[str, Any]) -> bool:
    return (
        _positive_count(row, "n_signals")
        or _positive_count(row, "raw_signal_count")
        or _positive_count(row, "executable_signal_count")
        or _nonempty_list(row, "raw_signals")
        or _nonempty_list(row, "executable_signals")
        or _nonempty_list(row, "guarded_signals")
    )


def _normalize_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if "min_executable_notional" in lower:
        return "min_executable_notional"
    if "expected_edge_below_cost" in lower:
        return "expected_edge_below_cost"
    if "flash_symbol_degraded" in lower:
        return "flash_symbol_degraded"
    return text.split(":", 1)[0].strip()


def _positive_count(row: Mapping[str, Any], key: str) -> bool:
    return _int_value(row.get(key)) > 0


def _nonempty_list(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    return isinstance(value, list) and len(value) > 0


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    }


if __name__ == "__main__":
    raise SystemExit(main())
