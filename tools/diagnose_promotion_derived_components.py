from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ACTOR_LABELS = (
    "CarryFlowAgentV2",
    "MomentumScalper",
    "LiveCrashHunter",
)


def build_promotion_derived_diagnostics(
    paths: Sequence[str | Path],
    *,
    actor_labels: Sequence[str] = DEFAULT_ACTOR_LABELS,
    min_closed_trades: int = 5,
    min_expectancy: float = 0.0,
) -> dict[str, Any]:
    labels = tuple(_normalize_label(label) for label in actor_labels if _normalize_label(label))
    reason_counts: Counter[str] = Counter()
    actor_reason_counts: dict[str, Counter[str]] = {
        label: Counter() for label in labels
    }
    symbol_reason_counts: dict[str, Counter[str]] = {}
    details: list[dict[str, Any]] = []
    bars_read = 0
    bad_rows = 0
    decisions_read = 0
    first_timestamp = ""
    last_timestamp = ""

    for path in paths:
        for payload in _iter_jsonl(Path(path)):
            if payload is None:
                bad_rows += 1
                continue
            bars_read += 1
            timestamp = _timestamp(payload)
            if timestamp:
                if not first_timestamp:
                    first_timestamp = timestamp
                last_timestamp = timestamp
            decisions = payload.get("flash_decisions")
            if not isinstance(decisions, list):
                continue
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    continue
                decisions_read += 1
                for label in labels:
                    row = _diagnose_actor_for_decision(
                        payload,
                        decision,
                        actor_label=label,
                        min_closed_trades=int(min_closed_trades),
                        min_expectancy=float(min_expectancy),
                    )
                    reason = str(row.get("why_not_promoted") or "unknown")
                    reason_counts[reason] += 1
                    actor_reason_counts[label][reason] += 1
                    symbol = str(row.get("symbol") or "")
                    if symbol:
                        symbol_reason_counts.setdefault(symbol, Counter())[reason] += 1
                    details.append(row)

    promoted_rows = sum(1 for row in details if bool(row.get("promoted")))
    missing_detail_rows = sum(
        1
        for row in details
        if str(row.get("why_not_promoted") or "") in {
            "missing_candidate",
            "not_observed_in_compact_audit",
        }
    )
    return {
        "summary": {
            "files": [str(Path(path)) for path in paths],
            "actor_labels": list(labels),
            "min_closed_trades": int(min_closed_trades),
            "min_expectancy": float(min_expectancy),
            "bars_read": int(bars_read),
            "bad_rows": int(bad_rows),
            "flash_decisions": int(decisions_read),
            "detail_rows": int(len(details)),
            "promoted_rows": int(promoted_rows),
            "missing_detail_rows": int(missing_detail_rows),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        },
        "reason_counts": _sorted_counter(reason_counts),
        "actor_reason_counts": {
            label: _sorted_counter(counter)
            for label, counter in actor_reason_counts.items()
        },
        "symbol_reason_counts": {
            symbol: _sorted_counter(counter)
            for symbol, counter in sorted(symbol_reason_counts.items())
        },
        "details": details,
    }


def write_reports(report: Mapping[str, Any], out: str | Path) -> tuple[Path, Path, Path]:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    details_path = out_path.with_suffix(".details.jsonl")
    with details_path.open("w", encoding="utf-8") as handle:
        for row in report.get("details", []) or []:
            if isinstance(row, Mapping):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(format_markdown_report(report), encoding="utf-8")
    return out_path, details_path, md_path


def format_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Promotion-Derived Components: Why Not Promoted",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "bars_read",
        "bad_rows",
        "flash_decisions",
        "detail_rows",
        "promoted_rows",
        "missing_detail_rows",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines.extend([
        f"| min_closed_trades | {summary.get('min_closed_trades', 0)} |",
        f"| min_expectancy | {summary.get('min_expectancy', 0.0)} |",
        "",
        "## Reason Counts",
        "",
        "| why_not_promoted | count |",
        "|---|---:|",
    ])
    reason_counts = report.get("reason_counts", {})
    if isinstance(reason_counts, Mapping) and reason_counts:
        for reason, count in reason_counts.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## By Actor", ""])
    actor_counts = report.get("actor_reason_counts", {})
    if isinstance(actor_counts, Mapping):
        for actor, counts in actor_counts.items():
            lines.extend([f"### {actor}", "", "| why_not_promoted | count |", "|---|---:|"])
            if isinstance(counts, Mapping) and counts:
                for reason, count in counts.items():
                    lines.append(f"| {reason} | {count} |")
            else:
                lines.append("| none | 0 |")
            lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why promotion-derived allowlisted components were not "
            "routed by Panteon Flash."
        )
    )
    parser.add_argument("paths", nargs="+", help="causal_entry_decisions.jsonl path(s).")
    parser.add_argument(
        "--out",
        default="promotion_derived_diagnostics.json",
        help="Output JSON path. Sibling .details.jsonl and .md files are also written.",
    )
    parser.add_argument(
        "--actor-label",
        action="append",
        default=[],
        help="Allowlisted actor label. Can be repeated or comma-separated.",
    )
    parser.add_argument("--min-closed-trades", type=int, default=5)
    parser.add_argument("--min-expectancy", type=float, default=0.0)
    args = parser.parse_args(argv)

    labels = _parse_labels(args.actor_label) or DEFAULT_ACTOR_LABELS
    report = build_promotion_derived_diagnostics(
        [Path(path) for path in args.paths],
        actor_labels=labels,
        min_closed_trades=args.min_closed_trades,
        min_expectancy=args.min_expectancy,
    )
    json_path, details_path, md_path = write_reports(report, args.out)
    print(json.dumps({
        "json": str(json_path),
        "details": str(details_path),
        "markdown": str(md_path),
    }, ensure_ascii=False))
    return 0


def _diagnose_actor_for_decision(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    actor_label: str,
    min_closed_trades: int,
    min_expectancy: float,
) -> dict[str, Any]:
    candidate, source, observed_count = _find_candidate(decision, actor_label)
    symbol = str(decision.get("symbol") or "")
    selected_actor = str(decision.get("selected_actor") or "")
    selected_reasons = decision.get("selected_reasons")
    promoted = (
        _labels_match(selected_actor, actor_label)
        and (
            str(decision.get("reason") or "") == "promotion_derived_router"
            or (
                isinstance(selected_reasons, list)
                and "promotion_derived_router" in {str(item) for item in selected_reasons}
            )
        )
    )
    reason = "promoted" if promoted else ""
    action = ""
    candidate_reason = ""
    closed_trades: int | None = None
    pnl_net_pct: float | None = None
    rolling_expectancy: float | None = None
    score: float | None = None
    rank: int | None = None
    rejected: bool | None = None

    if candidate is None:
        candidate_count = _int_or_none(decision.get("candidate_count")) or 0
        reason = (
            "not_observed_in_compact_audit"
            if candidate_count > observed_count
            else "missing_candidate"
        )
    else:
        action = str(candidate.get("action") or "")
        candidate_reason = str(candidate.get("reason") or "")
        closed_trades = _int_or_none(candidate.get("closed_trades"))
        pnl_net_pct = _float_or_none(candidate.get("pnl_net_pct"))
        score = _float_or_none(candidate.get("score"))
        rank = _int_or_none(candidate.get("rank"))
        rejected = bool(candidate.get("rejected")) if "rejected" in candidate else None
        if closed_trades and closed_trades > 0 and pnl_net_pct is not None:
            rolling_expectancy = pnl_net_pct / float(closed_trades)
        if not reason:
            reason = _why_not_promoted(
                action=action,
                candidate_reason=candidate_reason,
                closed_trades=closed_trades,
                rolling_expectancy=rolling_expectancy,
                min_closed_trades=min_closed_trades,
                min_expectancy=min_expectancy,
                selected_actor=selected_actor,
                actor_label=actor_label,
            )

    return {
        "bar": _int_or_none(payload.get("bar")) or 0,
        "timestamp": _timestamp(payload),
        "symbol": symbol,
        "regime": str(payload.get("regime") or ""),
        "actor": actor_label,
        "selected_actor": selected_actor,
        "decision_reason": str(decision.get("reason") or ""),
        "promoted": bool(promoted),
        "why_not_promoted": reason,
        "audit_source": source,
        "action": action,
        "candidate_reason": candidate_reason,
        "closed_trades": closed_trades,
        "pnl_net_pct": pnl_net_pct,
        "rolling_expectancy": rolling_expectancy,
        "score": score,
        "rank": rank,
        "rejected": rejected,
    }


def _why_not_promoted(
    *,
    action: str,
    candidate_reason: str,
    closed_trades: int | None,
    rolling_expectancy: float | None,
    min_closed_trades: int,
    min_expectancy: float,
    selected_actor: str,
    actor_label: str,
) -> str:
    normalized_reason = _normalize_reason(candidate_reason)
    if normalized_reason == "foreign_position_owner":
        return "foreign owner"
    if normalized_reason == "inactive":
        return "inactive"
    if not _is_open_action(action):
        return "non-open action"
    if normalized_reason and normalized_reason not in {"eligible", "promotion_derived_router"}:
        return normalized_reason
    if closed_trades is None:
        return "missing closed_trades"
    if closed_trades < min_closed_trades:
        return "closed<trades"
    if rolling_expectancy is None:
        return "missing expectancy"
    if rolling_expectancy <= min_expectancy:
        return "expectancy<=0" if min_expectancy == 0.0 else "expectancy<=threshold"
    if _labels_match(selected_actor, actor_label):
        return "selected_without_promotion"
    return "not_selected"


def _find_candidate(
    decision: Mapping[str, Any],
    actor_label: str,
) -> tuple[Mapping[str, Any] | None, str, int]:
    candidates = decision.get("candidates")
    top_rejected = decision.get("top_rejected_candidates")
    observed: list[tuple[Mapping[str, Any], str]] = []
    if isinstance(candidates, list):
        observed.extend((candidate, "candidates") for candidate in candidates if isinstance(candidate, Mapping))
    if isinstance(top_rejected, list):
        observed.extend(
            (candidate, "top_rejected_candidates")
            for candidate in top_rejected
            if isinstance(candidate, Mapping)
        )
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[Mapping[str, Any], str]] = []
    for candidate, source in observed:
        key = (
            str(candidate.get("label") or ""),
            str(candidate.get("actor_key") or ""),
            str(candidate.get("action") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((candidate, source))
    matches = [
        (candidate, source)
        for candidate, source in unique
        if _candidate_matches(candidate, actor_label)
    ]
    if matches:
        candidate, source = max(matches, key=lambda item: _candidate_match_rank(item[0]))
        return candidate, source, len(unique)
    return None, "", len(unique)


def _candidate_matches(candidate: Mapping[str, Any], actor_label: str) -> bool:
    values = set(_actor_label_family_aliases(candidate.get("label")))
    values.update(_actor_label_family_aliases(candidate.get("actor_key")))
    actor_key = str(candidate.get("actor_key") or "")
    if ":" in actor_key:
        values.update(_actor_label_family_aliases(actor_key.split(":", 1)[1]))
    return bool(values & _actor_label_family_aliases(actor_label))


def _candidate_match_rank(candidate: Mapping[str, Any]) -> tuple[int, int, int]:
    action = str(candidate.get("action") or "")
    reason = _normalize_reason(candidate.get("reason"))
    hard_reason = bool(
        reason
        and reason
        not in {
            "eligible",
            "promotion_derived_router",
            "inactive",
        }
    )
    rank = _int_or_none(candidate.get("rank"))
    return (
        2 if _is_open_action(action) else 1 if action and action.upper() != "HOLD" else 0,
        1 if hard_reason else 0,
        -int(rank if rank is not None else 1_000_000),
    )


def _labels_match(value: Any, actor_label: str) -> bool:
    return bool(_actor_label_family_aliases(value) & _actor_label_family_aliases(actor_label))


def _is_open_action(action: str) -> bool:
    upper = str(action or "").upper()
    return (
        upper.startswith("FUT_LONG")
        or upper.startswith("FUT_SHORT")
        or upper.startswith("SPOT_BUY")
    )


def _normalize_reason(value: Any) -> str:
    return str(value or "").strip().split(":", 1)[0]


def _normalize_label(value: Any) -> str:
    return str(value or "").strip()


def _actor_label_family_aliases(value: Any) -> set[str]:
    text = _normalize_label(value)
    if not text:
        return set()
    label = text.split(":", 1)[-1].strip()
    labels = {text, label}
    if label.startswith("V_"):
        labels.add(label[2:])
    if label.startswith("Solo_"):
        labels.add(label[len("Solo_") :])
    for item in tuple(labels):
        if item.startswith("V_"):
            labels.add(item[2:])
        if item.startswith("Solo_"):
            labels.add(item[len("Solo_") :])

    aliases: set[str] = set()
    for item in labels:
        clean = _normalize_label(item)
        if not clean:
            continue
        aliases.add(clean.lower())
        aliases.add(f"V_{clean}".lower())
        aliases.add(f"Solo_{clean}".lower())
        aliases.add(f"agent:{clean}".lower())
        aliases.add(f"ensemble:{clean}".lower())
        aliases.add(f"ensemble:Solo_{clean}".lower())
    return aliases


def _parse_labels(raw_values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        label.strip()
        for raw in raw_values
        for label in str(raw or "").split(",")
        if label.strip()
    )


def _timestamp(payload: Mapping[str, Any]) -> str:
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, Mapping):
        return str(timestamp.get("utc") or timestamp.get("timestamp_utc") or "")
    return str(timestamp or "")


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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-int(item[1]), item[0]))
    }


if __name__ == "__main__":
    raise SystemExit(main())
