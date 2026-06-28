from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ACTOR_LABELS = (
    "LiveTrendFollow",
    "FundingArb",
    "LiveCrashHunter",
    "CarryFlowAgentV2",
    "MomentumScalper",
)


def build_benchmark_flash_parity_report(
    paths: Sequence[str | Path],
    *,
    actor_labels: Sequence[str] = DEFAULT_ACTOR_LABELS,
    component_benchmark_paths: Sequence[str | Path] = (),
    standalone_event_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    labels = tuple(
        dict.fromkeys(
            _normalize_label(label)
            for label in actor_labels
            if _normalize_label(label)
        )
    )
    benchmark_components = _load_component_benchmark_components(
        paths,
        component_benchmark_paths=component_benchmark_paths,
    )
    status_counts: Counter[str] = Counter()
    actor_status_counts: dict[str, Counter[str]] = {
        label: Counter() for label in labels
    }
    symbol_status_counts: dict[str, Counter[str]] = {}
    actor_action_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    actor_regime_status_counts: dict[str, Counter[str]] = {
        label: Counter() for label in labels
    }
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
                    )
                    status = str(row.get("flash_status") or "unknown")
                    status_counts[status] += 1
                    actor_status_counts[label][status] += 1
                    symbol = str(row.get("symbol") or "")
                    if symbol:
                        symbol_status_counts.setdefault(symbol, Counter())[status] += 1
                    action = str(row.get("action") or "")
                    if action:
                        actor_action_counts[label][action] += 1
                    regime = str(row.get("regime") or "")
                    if regime:
                        actor_regime_status_counts[label][
                            f"{regime}|{status}"
                        ] += 1
                    details.append(row)

    standalone_trade_parity = _build_standalone_trade_parity(
        _load_standalone_trade_events(
            paths,
            actor_labels=labels,
            standalone_event_paths=standalone_event_paths,
        ),
        details,
    )
    standalone_trade_status_counts = Counter(
        str(row.get("flash_status") or "unknown")
        for row in standalone_trade_parity
    )
    actor_standalone_trade_status_counts: dict[str, Counter[str]] = {
        label: Counter() for label in labels
    }
    for row in standalone_trade_parity:
        actor = str(row.get("actor") or "")
        if actor in actor_standalone_trade_status_counts:
            actor_standalone_trade_status_counts[actor][
                str(row.get("flash_status") or "unknown")
            ] += 1

    selected_rows = sum(1 for row in details if row.get("flash_status") == "selected")
    missing_rows = sum(
        1 for row in details if row.get("flash_status") == "missing_candidate"
    )
    return {
        "summary": {
            "files": [str(Path(path)) for path in paths],
            "actor_labels": list(labels),
            "component_benchmark_files": [
                str(Path(path))
                for path in _component_benchmark_paths(
                    paths,
                    component_benchmark_paths=component_benchmark_paths,
                )
            ],
            "standalone_event_files": [
                str(Path(path))
                for path in _standalone_event_paths(
                    paths,
                    standalone_event_paths=standalone_event_paths,
                )
            ],
            "standalone_trade_rows_available": bool(standalone_trade_parity),
            "standalone_trade_rows": int(len(standalone_trade_parity)),
            "bars_read": int(bars_read),
            "bad_rows": int(bad_rows),
            "flash_decisions": int(decisions_read),
            "detail_rows": int(len(details)),
            "selected_rows": int(selected_rows),
            "missing_candidate_rows": int(missing_rows),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        },
        "status_counts": _sorted_counter(status_counts),
        "actor_status_counts": {
            label: _sorted_counter(counter)
            for label, counter in actor_status_counts.items()
        },
        "symbol_status_counts": {
            symbol: _sorted_counter(counter)
            for symbol, counter in sorted(symbol_status_counts.items())
        },
        "actor_action_counts": {
            label: _sorted_counter(counter)
            for label, counter in actor_action_counts.items()
        },
        "actor_regime_status_counts": {
            label: _sorted_counter(counter)
            for label, counter in actor_regime_status_counts.items()
        },
        "benchmark_components": {
            label: benchmark_components.get(label, {}) for label in labels
        },
        "standalone_trade_status_counts": _sorted_counter(
            standalone_trade_status_counts
        ),
        "actor_standalone_trade_status_counts": {
            label: _sorted_counter(counter)
            for label, counter in actor_standalone_trade_status_counts.items()
        },
        "standalone_trade_parity": standalone_trade_parity,
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
        "# Benchmark-to-Flash Parity Diagnostics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "bars_read",
        "bad_rows",
        "flash_decisions",
        "detail_rows",
        "selected_rows",
        "missing_candidate_rows",
        "standalone_trade_rows",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")
    lines.extend([
        (
            "| standalone_trade_rows_available | "
            f"{summary.get('standalone_trade_rows_available', False)} |"
        ),
        "",
        "## Status Counts",
        "",
        "| flash_status | count |",
        "|---|---:|",
    ])
    status_counts = report.get("status_counts", {})
    if isinstance(status_counts, Mapping) and status_counts:
        for status, count in status_counts.items():
            lines.append(f"| {status} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend([
        "",
        "## Standalone Trade Parity",
        "",
        "| flash_status | closed standalone events |",
        "|---|---:|",
    ])
    standalone_counts = report.get("standalone_trade_status_counts", {})
    if isinstance(standalone_counts, Mapping) and standalone_counts:
        for status, count in standalone_counts.items():
            lines.append(f"| {status} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## By Actor", ""])
    actor_counts = report.get("actor_status_counts", {})
    benchmark = report.get("benchmark_components", {})
    if isinstance(actor_counts, Mapping):
        for actor, counts in actor_counts.items():
            lines.extend([f"### {actor}", ""])
            component = benchmark.get(actor, {}) if isinstance(benchmark, Mapping) else {}
            if isinstance(component, Mapping) and component:
                lines.extend([
                    "| standalone_metric | value |",
                    "|---|---:|",
                    f"| pnl_usd | {component.get('pnl_usd', 0.0)} |",
                    f"| closed_trades | {component.get('closed_trades', 0)} |",
                    f"| win_rate_pct | {component.get('win_rate_pct', 0.0)} |",
                    "",
                ])
            lines.extend(["| flash_status | count |", "|---|---:|"])
            if isinstance(counts, Mapping) and counts:
                for status, count in counts.items():
                    lines.append(f"| {status} | {count} |")
            else:
                lines.append("| none | 0 |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standalone benchmark component labels with Flash per-bar "
            "candidate/action audit rows."
        )
    )
    parser.add_argument("paths", nargs="+", help="causal_entry_decisions.jsonl path(s).")
    parser.add_argument(
        "--out",
        default="benchmark_flash_parity.json",
        help="Output JSON path. Sibling .details.jsonl and .md files are also written.",
    )
    parser.add_argument(
        "--actor-label",
        action="append",
        default=[],
        help="Actor label to diagnose. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--component-benchmark",
        action="append",
        default=[],
        help=(
            "Optional component_benchmark_report.json path. If omitted, sibling "
            "reports next to each causal JSONL are loaded when present."
        ),
    )
    parser.add_argument(
        "--standalone-event",
        action="append",
        default=[],
        help=(
            "Optional shadow_agent_pnl_events.jsonl/shadow_player_pnl_events.jsonl "
            "path. If omitted, sibling shadow event files are loaded when present."
        ),
    )
    args = parser.parse_args(argv)

    labels = _parse_labels(args.actor_label) or DEFAULT_ACTOR_LABELS
    report = build_benchmark_flash_parity_report(
        [Path(path) for path in args.paths],
        actor_labels=labels,
        component_benchmark_paths=[Path(path) for path in args.component_benchmark],
        standalone_event_paths=[Path(path) for path in args.standalone_event],
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
) -> dict[str, Any]:
    candidate, source, observed_count = _find_candidate(decision, actor_label)
    selected_actor = str(decision.get("selected_actor") or "")
    selected_action = str(decision.get("action") or "")
    action = ""
    candidate_reason = ""
    closed_trades: int | None = None
    pnl_net_pct: float | None = None
    rolling_expectancy: float | None = None
    score: float | None = None
    base_score: float | None = None
    effective_score: float | None = None
    gate_score: float | None = None
    rank: int | None = None
    rejected: bool | None = None
    risk_mult: float | None = None

    if candidate is None:
        status = "missing_candidate"
    else:
        action = str(candidate.get("action") or "")
        candidate_reason = str(candidate.get("reason") or "")
        closed_trades = _int_or_none(candidate.get("closed_trades"))
        pnl_net_pct = _float_or_none(candidate.get("pnl_net_pct"))
        if closed_trades and closed_trades > 0 and pnl_net_pct is not None:
            rolling_expectancy = pnl_net_pct / float(closed_trades)
        score = _float_or_none(candidate.get("score"))
        base_score = _float_or_none(candidate.get("base_score"))
        effective_score = _float_or_none(candidate.get("effective_score"))
        gate_score = _float_or_none(candidate.get("gate_score"))
        rank = _int_or_none(candidate.get("rank"))
        rejected = bool(candidate.get("rejected")) if "rejected" in candidate else None
        risk_mult = _float_or_none(candidate.get("risk_mult"))
        status = _flash_status(
            actor_label=actor_label,
            selected_actor=selected_actor,
            action=action,
            candidate_reason=candidate_reason,
            rejected=rejected,
        )

    return {
        "bar": _int_or_none(payload.get("bar")) or 0,
        "timestamp": _timestamp(payload),
        "symbol": str(decision.get("symbol") or ""),
        "regime": str(payload.get("regime") or ""),
        "actor": actor_label,
        "selected_actor": selected_actor,
        "selected_action": selected_action,
        "decision_reason": str(decision.get("reason") or ""),
        "flash_status": status,
        "audit_source": source,
        "candidate_count": _int_or_none(decision.get("candidate_count")),
        "observed_candidate_rows": observed_count,
        "action": action,
        "candidate_reason": candidate_reason,
        "closed_trades": closed_trades,
        "pnl_net_pct": pnl_net_pct,
        "rolling_expectancy": rolling_expectancy,
        "score": score,
        "base_score": base_score,
        "effective_score": effective_score,
        "gate_score": gate_score,
        "rank": rank,
        "rejected": rejected,
        "risk_mult": risk_mult,
    }


def _flash_status(
    *,
    actor_label: str,
    selected_actor: str,
    action: str,
    candidate_reason: str,
    rejected: bool | None,
) -> str:
    reason = _normalize_reason(candidate_reason)
    if _labels_match(selected_actor, actor_label):
        return "selected"
    upper_action = str(action or "").upper()
    if not upper_action or upper_action == "HOLD":
        return f"hold_in_flash:{reason}" if reason else "hold_in_flash"
    if bool(rejected):
        return f"rejected:{reason}" if reason else "rejected"
    if not _is_open_action(upper_action):
        return f"non_open_action:{upper_action}"
    return "eligible_not_selected"


def _build_standalone_trade_parity(
    standalone_events: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    details_by_bar_actor: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for detail in details:
        bar = _int_or_none(detail.get("bar")) or 0
        actor = str(detail.get("actor") or "")
        if not bar or not actor:
            continue
        details_by_bar_actor.setdefault((bar, actor), []).append(detail)

    rows: list[dict[str, Any]] = []
    for event in standalone_events:
        bar = _int_or_none(event.get("bar")) or 0
        actor = str(event.get("actor") or "")
        matching = details_by_bar_actor.get((bar, actor), [])
        detail = _best_trade_parity_detail(matching)
        if detail is None:
            rows.append({
                "bar": bar,
                "timestamp": str(event.get("timestamp") or ""),
                "regime": str(event.get("regime") or ""),
                "actor": actor,
                "standalone_label": str(event.get("label") or actor),
                "standalone_pnl_usd": _float_or_none(event.get("pnl_usd")),
                "standalone_closed_trades": _int_or_none(event.get("closed_trades")),
                "standalone_wins": _int_or_none(event.get("wins")),
                "matched_flash_rows": 0,
                "flash_status": "no_flash_decision_for_bar",
                "symbol": "",
                "selected_actor": "",
                "selected_action": "",
                "flash_action": "",
                "candidate_reason": "",
                "flash_score": None,
                "flash_rank": None,
            })
            continue
        rows.append({
            "bar": bar,
            "timestamp": str(event.get("timestamp") or detail.get("timestamp") or ""),
            "regime": str(event.get("regime") or detail.get("regime") or ""),
            "actor": actor,
            "standalone_label": str(event.get("label") or actor),
            "standalone_pnl_usd": _float_or_none(event.get("pnl_usd")),
            "standalone_closed_trades": _int_or_none(event.get("closed_trades")),
            "standalone_wins": _int_or_none(event.get("wins")),
            "matched_flash_rows": len(matching),
            "flash_status": str(detail.get("flash_status") or ""),
            "symbol": str(detail.get("symbol") or ""),
            "selected_actor": str(detail.get("selected_actor") or ""),
            "selected_action": str(detail.get("selected_action") or ""),
            "flash_action": str(detail.get("action") or ""),
            "candidate_reason": str(detail.get("candidate_reason") or ""),
            "flash_score": _float_or_none(detail.get("score")),
            "flash_rank": _int_or_none(detail.get("rank")),
        })
    rows.sort(key=lambda row: (int(row.get("bar", 0) or 0), str(row.get("actor") or "")))
    return rows


def _best_trade_parity_detail(
    details: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not details:
        return None
    return max(details, key=_trade_parity_detail_rank)


def _trade_parity_detail_rank(detail: Mapping[str, Any]) -> tuple[int, float, int]:
    status = str(detail.get("flash_status") or "")
    if status == "selected":
        priority = 5
    elif status == "eligible_not_selected":
        priority = 4
    elif status.startswith("rejected:"):
        priority = 3
    elif status.startswith("hold_in_flash"):
        priority = 2
    elif status == "missing_candidate":
        priority = 1
    else:
        priority = 0
    score = _float_or_none(detail.get("score"))
    rank = _int_or_none(detail.get("rank"))
    return (
        priority,
        float(score if score is not None else 0.0),
        -int(rank if rank is not None else 1_000_000),
    )


def _find_candidate(
    decision: Mapping[str, Any],
    actor_label: str,
) -> tuple[Mapping[str, Any] | None, str, int]:
    observed: list[tuple[Mapping[str, Any], str]] = []
    candidates = decision.get("candidates")
    top_rejected = decision.get("top_rejected_candidates")
    if isinstance(candidates, list):
        observed.extend(
            (candidate, "candidates")
            for candidate in candidates
            if isinstance(candidate, Mapping)
        )
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
    if not matches:
        return None, "", len(unique)
    candidate, source = max(matches, key=lambda item: _candidate_match_rank(item[0]))
    return candidate, source, len(unique)


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
            "selected",
            "inactive",
        }
    )
    rank = _int_or_none(candidate.get("rank"))
    return (
        2 if _is_open_action(action) else 1 if action and action.upper() != "HOLD" else 0,
        1 if hard_reason else 0,
        -int(rank if rank is not None else 1_000_000),
    )


def _load_standalone_trade_events(
    paths: Sequence[str | Path],
    *,
    actor_labels: Sequence[str],
    standalone_event_paths: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in _standalone_event_paths(
        paths,
        standalone_event_paths=standalone_event_paths,
    ):
        for row in _iter_jsonl(Path(path)):
            if not isinstance(row, Mapping):
                continue
            label = str(row.get("label") or "")
            actor = _canonical_actor_label(label, actor_labels)
            if not actor:
                continue
            closed_trades = _int_or_none(row.get("closed_trades")) or 0
            pnl_usd = _float_or_none(row.get("pnl_usd")) or 0.0
            if closed_trades <= 0 and pnl_usd == 0.0:
                continue
            events.append({
                "bar": _int_or_none(row.get("bar")) or 0,
                "label": label,
                "actor": actor,
                "timestamp": _timestamp(row),
                "regime": str(row.get("regime") or ""),
                "pnl_usd": pnl_usd,
                "closed_trades": closed_trades,
                "wins": _int_or_none(row.get("wins")) or 0,
            })
    events.sort(key=lambda row: (int(row["bar"]), str(row["actor"])))
    return events


def _standalone_event_paths(
    paths: Sequence[str | Path],
    *,
    standalone_event_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    explicit = tuple(Path(path) for path in standalone_event_paths or ())
    if explicit:
        return explicit
    inferred: list[Path] = []
    for path in paths:
        parent = Path(path).parent
        for filename in (
            "shadow_agent_pnl_events.jsonl",
            "shadow_player_pnl_events.jsonl",
        ):
            candidate = parent / filename
            if candidate.exists():
                inferred.append(candidate)
    return tuple(dict.fromkeys(inferred))


def _canonical_actor_label(
    label: str,
    actor_labels: Sequence[str],
) -> str:
    for actor_label in actor_labels:
        if _labels_match(label, actor_label):
            return str(actor_label)
    return ""


def _load_component_benchmark_components(
    paths: Sequence[str | Path],
    *,
    component_benchmark_paths: Sequence[str | Path] = (),
) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for path in _component_benchmark_paths(
        paths,
        component_benchmark_paths=component_benchmark_paths,
    ):
        payload = _load_json(Path(path))
        components = payload.get("components") if isinstance(payload, Mapping) else []
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, Mapping):
                continue
            label = str(component.get("label") or "")
            if label and label not in out:
                out[label] = component
    return out


def _component_benchmark_paths(
    paths: Sequence[str | Path],
    *,
    component_benchmark_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    explicit = tuple(Path(path) for path in component_benchmark_paths or ())
    if explicit:
        return explicit
    inferred: list[Path] = []
    for path in paths:
        candidate = Path(path).parent / "component_benchmark_report.json"
        if candidate.exists():
            inferred.append(candidate)
    return tuple(dict.fromkeys(inferred))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_open_action(action: str) -> bool:
    upper = str(action or "").upper()
    return (
        upper.startswith("FUT_LONG")
        or upper.startswith("FUT_SHORT")
        or upper.startswith("SPOT_BUY")
    )


def _labels_match(value: Any, actor_label: str) -> bool:
    return bool(_actor_label_family_aliases(value) & _actor_label_family_aliases(actor_label))


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
