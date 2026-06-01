from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from panteon_v2.analysis.walk_forward import build_walk_forward_report_from_events


def iter_jsonl_events(
    paths: Iterable[str | Path],
    *,
    tail_bytes: Optional[int] = None,
    max_events: Optional[int] = None,
) -> Iterator[dict]:
    emitted = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        with path.open("rb") as handle:
            if tail_bytes is not None and tail_bytes > 0:
                size = path.stat().st_size
                offset = max(0, size - int(tail_bytes))
                handle.seek(offset)
                if offset > 0:
                    handle.readline()
            for raw_line in handle:
                try:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                payload["_source"] = str(path)
                yield payload
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    return


def build_stitched_log_report(
    log_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    tail_bytes: Optional[int] = None,
    max_events: Optional[int] = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = list(
        iter_jsonl_events(
            log_paths,
            tail_bytes=tail_bytes,
            max_events=max_events,
        )
    )
    event_type_counts: Counter[str] = Counter(_event_type(event) for event in events)
    reject_reasons: Counter[str] = Counter(
        _reject_reason(event)
        for event in events
        if _event_type(event) == "CandidateRejected"
    )
    if "" in reject_reasons:
        del reject_reasons[""]
    walk_forward = build_walk_forward_report_from_events(
        results_root=str(output),
        events=events,
    )
    report = {
        "log_paths": [str(Path(path)) for path in log_paths],
        "tail_bytes": tail_bytes,
        "max_events": max_events,
        "total_events": len(events),
        "event_type_counts": dict(event_type_counts),
        "reject_reasons": dict(reject_reasons),
        "walk_forward": walk_forward,
    }
    (output / "stitched_live_log_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "stitched_live_log_report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    return report


def _event_type(event: Mapping) -> str:
    return str(event.get("_type") or event.get("event_type") or event.get("type") or "")


def _reject_reason(event: Mapping) -> str:
    return str(
        event.get("reason")
        or event.get("rejection_reason")
        or event.get("reject_reason")
        or ""
    )


def _markdown_report(report: Mapping) -> str:
    totals = report.get("walk_forward", {}).get("totals", {})  # type: ignore[union-attr]
    lines = [
        "# Panteon Legend Stitched Live Log Retrotest",
        "",
        "## Scope",
        f"- Logs: {len(report.get('log_paths', []))}",
        f"- Tail bytes: {report.get('tail_bytes') or 'full'}",
        f"- Events parsed: {report.get('total_events', 0)}",
        "",
        "## Event Types",
        _counter_table(report.get("event_type_counts", {})),
        "",
        "## Candidate Reject Reasons",
        _counter_table(report.get("reject_reasons", {})),
        "",
        "## Walk-Forward From Live Events",
        f"- Closed trades: {int(totals.get('closed_trades', 0) or 0)}",
        f"- Net PnL: {float(totals.get('net_pnl', 0.0) or 0.0):.6f}",
        f"- Win rate: {float(totals.get('winrate_pct', 0.0) or 0.0):.2f}%",
        f"- Profit factor: {float(totals.get('profit_factor', 0.0) or 0.0):.4f}",
        "",
        "## Note",
        (
            "This is a stitched event-log retrotest/audit. It replays execution "
            "and attribution events already present in live logs; it does not "
            "reconstruct missing OHLCV candles that were not recorded in JSONL."
        ),
        "",
    ]
    return "\n".join(lines)


def _counter_table(counter: object, *, limit: int = 20) -> str:
    if not isinstance(counter, Mapping) or not counter:
        return "_No data._"
    rows = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    lines = ["| Item | Count |", "| --- | ---: |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {int(value)} |")
    return "\n".join(lines)
