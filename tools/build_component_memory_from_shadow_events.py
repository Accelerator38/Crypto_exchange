from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _parse_csv_values(values: Sequence[str]) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in values or ():
        for part in str(value or "").split(","):
            clean = part.strip()
            if clean:
                parsed.append(clean)
    return tuple(dict.fromkeys(parsed))


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        return ()
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


def build_component_memory_rows(
    input_paths: Sequence[str | Path],
    *,
    actor_labels: Sequence[str] = (),
    min_closed_trades: int = 1,
    min_expectancy: float | None = None,
    symbol: str = "*",
    action: str = "*",
    include_cumulative_rollups: bool = False,
    seed_prior_bar: int | None = None,
) -> list[dict[str, Any]]:
    allowed = {label.strip() for label in actor_labels if str(label).strip()}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int, str]] = set()
    cumulative: dict[tuple[str, str], dict[str, float]] = {}
    cumulative_symbol_action: dict[tuple[str, str, str, str], dict[str, float]] = {}

    def append_row(
        *,
        label: str,
        bar: int,
        regime: str,
        closed_trades: int,
        pnl_usd: float,
        source_file: str,
        memory_scope: str,
        row_symbol: str | None = None,
        row_action: str | None = None,
    ) -> None:
        if closed_trades < int(min_closed_trades):
            return
        expectancy = pnl_usd / float(closed_trades) if closed_trades else 0.0
        if min_expectancy is not None and expectancy < float(min_expectancy):
            return
        effective_symbol = str(symbol if row_symbol is None else row_symbol)
        effective_action = str(action if row_action is None else row_action)
        effective_bar = int(seed_prior_bar) if seed_prior_bar is not None else int(bar)
        key = (label, effective_symbol, regime, effective_action, int(bar), memory_scope)
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "actor_label": label,
            "symbol": effective_symbol,
            "regime": regime,
            "action": effective_action,
            "bar": effective_bar,
            "closed_trades": closed_trades,
            "expectancy": expectancy,
            "pnl_lcb": expectancy,
            "source_file": source_file,
            "memory_scope": memory_scope,
        })

    for raw_path in input_paths:
        path = Path(raw_path)
        for event in _read_jsonl(path):
            label = str(event.get("label", event.get("actor_label", "")) or "").strip()
            if not label or (allowed and label not in allowed):
                continue
            closed_trades = _safe_int(event.get("closed_trades"))
            if closed_trades <= 0:
                continue
            pnl_usd = _safe_float(event.get("pnl_usd", event.get("realized_pnl_usd")))
            bar = _safe_int(event.get("bar"))
            if bar <= 0:
                continue
            regime = str(event.get("regime", "all") or "all").strip() or "all"
            append_row(
                label=label,
                bar=bar,
                regime=regime,
                closed_trades=closed_trades,
                pnl_usd=pnl_usd,
                source_file=str(path),
                memory_scope="event",
            )
            for outcome in event.get("symbol_action_outcomes") or ():
                if not isinstance(outcome, Iterable) or isinstance(outcome, (str, bytes)):
                    continue
                parts = tuple(outcome)
                if len(parts) < 5:
                    continue
                outcome_symbol = str(parts[0] or "").strip().upper()
                outcome_action = str(parts[1] or "").strip().upper()
                outcome_closed = _safe_int(parts[3])
                if not outcome_symbol or not outcome_action or outcome_closed <= 0:
                    continue
                append_row(
                    label=label,
                    bar=bar,
                    regime=regime,
                    closed_trades=outcome_closed,
                    pnl_usd=_safe_float(parts[2]),
                    source_file=str(path),
                    memory_scope="symbol_action",
                    row_symbol=outcome_symbol,
                    row_action=outcome_action,
                )
                if include_cumulative_rollups:
                    outcome_pnl = _safe_float(parts[2])
                    for scope, rollup_regime in (
                        ("cumulative_symbol_action_regime", regime),
                        ("cumulative_symbol_action", "*"),
                    ):
                        bucket = cumulative_symbol_action.setdefault(
                            (
                                label,
                                outcome_symbol,
                                outcome_action,
                                rollup_regime,
                            ),
                            {"closed_trades": 0.0, "pnl_usd": 0.0},
                        )
                        bucket["closed_trades"] += float(outcome_closed)
                        bucket["pnl_usd"] += float(outcome_pnl)
                        append_row(
                            label=label,
                            bar=bar,
                            regime=rollup_regime,
                            closed_trades=int(bucket["closed_trades"]),
                            pnl_usd=float(bucket["pnl_usd"]),
                            source_file=str(path),
                            memory_scope=scope,
                            row_symbol=outcome_symbol,
                            row_action=outcome_action,
                        )
            if not include_cumulative_rollups:
                continue
            for scope, rollup_regime in (
                ("cumulative_regime", regime),
                ("cumulative_global", "*"),
            ):
                bucket = cumulative.setdefault(
                    (label, rollup_regime),
                    {"closed_trades": 0.0, "pnl_usd": 0.0},
                )
                bucket["closed_trades"] += float(closed_trades)
                bucket["pnl_usd"] += float(pnl_usd)
                append_row(
                    label=label,
                    bar=bar,
                    regime=rollup_regime,
                    closed_trades=int(bucket["closed_trades"]),
                    pnl_usd=float(bucket["pnl_usd"]),
                    source_file=str(path),
                    memory_scope=scope,
                )
    rows.sort(
        key=lambda row: (
            str(row.get("actor_label") or ""),
            _safe_int(row.get("bar")),
            str(row.get("regime") or ""),
        )
    )
    return rows


def write_component_memory_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def write_report(path: str | Path, *, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({str(row.get("actor_label") or "") for row in rows})
    payload = {
        "generated_at": _timestamp(),
        "rows": len(rows),
        "labels": labels,
        "bars": {
            "min": min((_safe_int(row.get("bar")) for row in rows), default=0),
            "max": max((_safe_int(row.get("bar")) for row in rows), default=0),
        },
        "positive_expectancy_rows": sum(
            1 for row in rows if _safe_float(row.get("expectancy")) > 0.0
        ),
        "nonpositive_expectancy_rows": sum(
            1 for row in rows if _safe_float(row.get("expectancy")) <= 0.0
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build prior-bar Flash component memory from shadow PnL events."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="shadow_agent_pnl_events.jsonl or shadow_player_pnl_events.jsonl path.",
    )
    parser.add_argument("--out", required=True, help="Output component-memory JSONL path.")
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--actor-label",
        action="append",
        default=[],
        help="Allowed actor label. Can be repeated or comma-separated.",
    )
    parser.add_argument("--min-closed-trades", type=int, default=1)
    parser.add_argument("--min-expectancy", type=float, default=None)
    parser.add_argument("--symbol", default="*")
    parser.add_argument("--action", default="*")
    parser.add_argument(
        "--include-cumulative-rollups",
        action="store_true",
        help="Emit cumulative per-regime and global actor rollups for sparse components.",
    )
    parser.add_argument(
        "--seed-prior-bar",
        type=int,
        default=None,
        help=(
            "Override emitted memory bar, useful when offline replay stats are intentionally "
            "seeded as prior evidence for a fresh paper/live session."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = build_component_memory_rows(
        args.input,
        actor_labels=_parse_csv_values(args.actor_label),
        min_closed_trades=args.min_closed_trades,
        min_expectancy=args.min_expectancy,
        symbol=args.symbol,
        action=args.action,
        include_cumulative_rollups=args.include_cumulative_rollups,
        seed_prior_bar=args.seed_prior_bar,
    )
    out_path = write_component_memory_jsonl(args.out, rows)
    report_path = Path(args.report) if str(args.report or "").strip() else out_path.with_suffix(".summary.json")
    write_report(report_path, rows=rows)
    print(f"rows={len(rows)}", flush=True)
    print(f"output={out_path}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
