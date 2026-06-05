"""Live Flash admission attribution from causal_entry_decisions.jsonl."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TARGET_ACTORS = (
    "GeneticsCore",
    "V_GeneticsCore",
    "GeneticsRiskTight",
    "V_GeneticsRiskTight",
)


def build_flash_live_admission_attribution(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_actors: Sequence[str] = DEFAULT_TARGET_ACTORS,
    top_n: int = 50,
) -> dict[str, Any]:
    target_set = {_normalize_actor_name(actor) for actor in target_actors}
    selected_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    candidate_rejection_counts: Counter[str] = Counter()
    actor_summaries: dict[str, dict[str, Any]] = {}
    denial_groups: dict[tuple[str, str, str], dict[str, Any]] = {}

    rows_read = 0
    flash_decisions = 0
    raw_signals = 0
    executable_signals = 0
    filled = 0
    denied_original_actor_decisions = 0
    target_denied_decisions = 0
    candidate_detail_available = False

    for row in rows:
        rows_read += 1
        raw_signals += _as_int(row.get("raw_signal_count"))
        executable_signals += _as_int(
            row.get("executable_signal_count", row.get("n_signals"))
        )
        filled += _as_int(row.get("n_filled"))
        bar = _as_int(row.get("bar"))
        for decision in _iter_flash_decisions(row.get("flash_decisions")):
            flash_decisions += 1
            selected_actor = str(decision.get("selected_actor") or "").strip()
            original_actor = str(
                decision.get("original_selected_actor") or selected_actor
            ).strip()
            reason = str(decision.get("reason") or "unknown").strip() or "unknown"
            symbol = str(decision.get("symbol") or "").strip().upper()

            selected_counts[selected_actor or ""] += 1
            original_counts[original_actor or ""] += 1
            reason_counts[reason] += 1
            _merge_counter(
                candidate_rejection_counts,
                decision.get("candidate_rejection_counts"),
            )
            if decision.get("candidates") or decision.get("top_rejected_candidates"):
                candidate_detail_available = True

            if selected_actor != "NoTrade" or not original_actor or original_actor == "NoTrade":
                continue
            denied_original_actor_decisions += 1
            normalized_actor = _normalize_actor_name(original_actor)
            if target_set and normalized_actor not in target_set:
                continue
            target_denied_decisions += 1
            _record_actor_summary(
                actor_summaries,
                actor=original_actor,
                symbol=symbol,
                reason=reason,
                rejection_counts=decision.get("candidate_rejection_counts"),
            )
            _record_denial_group(
                denial_groups,
                actor=original_actor,
                symbol=symbol,
                reason=reason,
                bar=bar,
                rejection_counts=decision.get("candidate_rejection_counts"),
            )

    groups = sorted(
        denial_groups.values(),
        key=lambda item: (
            -_as_int(item.get("denied_decisions")),
            str(item.get("actor") or ""),
            str(item.get("symbol") or ""),
            str(item.get("reason") or ""),
        ),
    )[: max(0, int(top_n or 0))]
    return {
        "summary": {
            "rows_read": rows_read,
            "flash_decisions": flash_decisions,
            "raw_signals": raw_signals,
            "executable_signals": executable_signals,
            "filled": filled,
            "denied_original_actor_decisions": denied_original_actor_decisions,
            "target_denied_decisions": target_denied_decisions,
            "target_actors": list(target_actors),
            "candidate_detail_available": candidate_detail_available,
        },
        "selected_counts": _counter_dict(selected_counts),
        "original_counts": _counter_dict(original_counts),
        "reason_counts": _counter_dict(reason_counts),
        "candidate_rejection_counts": _counter_dict(candidate_rejection_counts),
        "actor_summaries": _finalize_actor_summaries(actor_summaries),
        "denial_groups": groups,
    }


def write_flash_live_admission_attribution(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    filename: str = "flash_live_admission_attribution.json",
    markdown_filename: str = "flash_live_admission_attribution.md",
    target_actors: Sequence[str] = DEFAULT_TARGET_ACTORS,
    top_n: int = 50,
) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    rows = tuple(_iter_jsonl(run_path / "causal_entry_decisions.jsonl"))
    report = build_flash_live_admission_attribution(
        rows,
        target_actors=target_actors,
        top_n=top_n,
    )
    out_dir = Path(output_dir) if output_dir is not None else run_path
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / filename
    md_path = out_dir / markdown_filename
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(
        _markdown_report(report, run_path=run_path, top_n=top_n),
        encoding="utf-8",
    )
    return json_path, md_path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload


def _iter_flash_decisions(raw: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        iterable = raw.values()
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = ()
    for item in iterable:
        if isinstance(item, Mapping):
            yield item


def _record_actor_summary(
    actor_summaries: dict[str, dict[str, Any]],
    *,
    actor: str,
    symbol: str,
    reason: str,
    rejection_counts: Any,
) -> None:
    summary = actor_summaries.setdefault(
        actor,
        {
            "denied_decisions": 0,
            "symbol_counts": Counter(),
            "reason_counts": Counter(),
            "candidate_rejection_counts": Counter(),
        },
    )
    summary["denied_decisions"] += 1
    summary["symbol_counts"][symbol] += 1
    summary["reason_counts"][reason] += 1
    _merge_counter(summary["candidate_rejection_counts"], rejection_counts)


def _record_denial_group(
    groups: dict[tuple[str, str, str], dict[str, Any]],
    *,
    actor: str,
    symbol: str,
    reason: str,
    bar: int,
    rejection_counts: Any,
) -> None:
    key = (actor, symbol, reason)
    group = groups.setdefault(
        key,
        {
            "actor": actor,
            "symbol": symbol,
            "reason": reason,
            "denied_decisions": 0,
            "first_bar": bar,
            "last_bar": bar,
            "candidate_rejection_counts": Counter(),
        },
    )
    group["denied_decisions"] += 1
    group["first_bar"] = min(_as_int(group.get("first_bar")), bar)
    group["last_bar"] = max(_as_int(group.get("last_bar")), bar)
    _merge_counter(group["candidate_rejection_counts"], rejection_counts)


def _finalize_actor_summaries(
    actor_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for actor, summary in sorted(
        actor_summaries.items(),
        key=lambda item: (-_as_int(item[1].get("denied_decisions")), item[0]),
    ):
        out[actor] = {
            "denied_decisions": _as_int(summary.get("denied_decisions")),
            "symbol_counts": _counter_dict(summary.get("symbol_counts")),
            "reason_counts": _counter_dict(summary.get("reason_counts")),
            "candidate_rejection_counts": _counter_dict(
                summary.get("candidate_rejection_counts")
            ),
        }
    return out


def _merge_counter(counter: Counter[str], raw: Any) -> None:
    if not isinstance(raw, Mapping):
        return
    for key, value in raw.items():
        counter[str(key)] += _as_int(value)


def _counter_dict(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Counter):
        raw = Counter(raw or {})
    return {
        str(key): int(value)
        for key, value in sorted(raw.items(), key=lambda item: (-item[1], str(item[0])))
    }


def _markdown_report(report: Mapping[str, Any], *, run_path: Path, top_n: int) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Flash live admission attribution",
        "",
        f"Run: `{run_path}`",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "rows_read",
        "flash_decisions",
        "raw_signals",
        "executable_signals",
        "filled",
        "denied_original_actor_decisions",
        "target_denied_decisions",
        "candidate_detail_available",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Actor summaries", ""])
    actor_summaries = report.get("actor_summaries") or {}
    if not actor_summaries:
        lines.append("_No target actor denials._")
    else:
        lines.append("| Actor | Denied | Top reasons | Top symbols |")
        lines.append("|---|---:|---|---|")
        for actor, row in actor_summaries.items():
            lines.append(
                "| {actor} | {denied} | {reasons} | {symbols} |".format(
                    actor=actor,
                    denied=_as_int(row.get("denied_decisions")),
                    reasons=_compact_counts(row.get("reason_counts")),
                    symbols=_compact_counts(row.get("symbol_counts")),
                )
            )
    lines.extend(["", f"## Top {top_n} denial groups", ""])
    groups = list(report.get("denial_groups") or [])
    if not groups:
        lines.append("_No target denial groups._")
    else:
        lines.append(
            "| Actor | Symbol | Reason | Denied | First bar | Last bar | Rejections |"
        )
        lines.append("|---|---|---|---:|---:|---:|---|")
        for row in groups:
            lines.append(
                "| {actor} | {symbol} | {reason} | {denied} | {first_bar} | {last_bar} | {rejections} |".format(
                    actor=row.get("actor", ""),
                    symbol=row.get("symbol", ""),
                    reason=row.get("reason", ""),
                    denied=_as_int(row.get("denied_decisions")),
                    first_bar=_as_int(row.get("first_bar")),
                    last_bar=_as_int(row.get("last_bar")),
                    rejections=_compact_counts(row.get("candidate_rejection_counts")),
                )
            )
    lines.extend(
        [
            "",
            "Note: when `candidate_detail_available` is false, compact live logs",
            "do not contain per-candidate payloads. The report can still attribute",
            "denials by original actor, symbol and reason, but cannot rank hidden",
            "rejected candidates by score.",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact_counts(raw: Any, *, limit: int = 3) -> str:
    counts = _counter_dict(raw)
    if not counts:
        return ""
    return ", ".join(f"{key}:{value}" for key, value in list(counts.items())[:limit])


def _normalize_actor_name(actor: object) -> str:
    value = str(actor or "").strip()
    return value[2:] if value.startswith("V_") else value


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
