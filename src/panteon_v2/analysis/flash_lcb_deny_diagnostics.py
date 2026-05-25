"""Diagnostics for Flash actor/symbol/action LCB deny candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_flash_lcb_diagnostics(
    *,
    shadow_rows: Sequence[Mapping[str, Any]],
    attribution_rows: Sequence[Mapping[str, Any]],
    min_shadow_closed: int = 10,
    reliable_negative_lcb_pct: float = 0.0,
) -> dict[str, Any]:
    """Join Flash shadow signal-key stats with selected attribution rows.

    The output deliberately proposes deny candidates only as candidates requiring
    retro validation. Static shadow LCB is not enough to safely mutate live
    selection, because a deny key can change later open/close and rate-limit
    sequencing.
    """

    min_closed = max(0, int(min_shadow_closed or 0))
    lcb_floor = float(reliable_negative_lcb_pct or 0.0)
    shadow_by_key = {
        str(row.get("signal_key") or "").strip(): dict(row)
        for row in shadow_rows
        if str(row.get("signal_key") or "").strip()
    }
    selected_without_shadow = 0
    selected_joined: list[dict[str, Any]] = []
    sparse_alpha_selected_winners: list[dict[str, Any]] = []
    reliable_negative_selected_losers: list[dict[str, Any]] = []
    deny_candidates: list[dict[str, Any]] = []

    for row in attribution_rows:
        signal_key = _selected_signal_key(row)
        if not signal_key:
            continue
        shadow = shadow_by_key.get(signal_key)
        joined = _joined_selected_row(row, shadow, min_closed=min_closed)
        selected_joined.append(joined)
        if shadow is None:
            selected_without_shadow += 1

        selected_pnl_usd = _as_float(row.get("realized_pnl_usd"))
        shadow_closed = _as_int(joined.get("shadow_full_closed_trades"))
        shadow_lcb = _optional_float(joined.get("shadow_full_pnl_per_trade_lcb_pct"))
        weak_or_sparse_shadow = (
            shadow is None
            or shadow_closed < min_closed
            or shadow_lcb is None
            or shadow_lcb < lcb_floor
        )
        if selected_pnl_usd > 0.0 and weak_or_sparse_shadow:
            sparse_alpha_selected_winners.append(joined)
        if (
            selected_pnl_usd < 0.0
            and shadow is not None
            and shadow_closed >= min_closed
            and shadow_lcb is not None
            and shadow_lcb < lcb_floor
            and _as_float(shadow.get("full_pnl_usd")) < 0.0
            and _as_int(row.get("closed_trades")) > 0
        ):
            reliable_negative_selected_losers.append(joined)
            candidate = dict(joined)
            candidate["requires_retro_validation"] = True
            deny_candidates.append(candidate)

    reliable_negative_shadow_cells = [
        _shadow_cell(row)
        for row in shadow_rows
        if _as_int(row.get("full_closed_trades")) >= min_closed
        and (_optional_float(row.get("full_pnl_per_trade_lcb_pct")) or 0.0)
        < lcb_floor
    ]
    reliable_positive_shadow_cells = [
        _shadow_cell(row)
        for row in shadow_rows
        if _as_int(row.get("full_closed_trades")) >= min_closed
        and (_optional_float(row.get("full_pnl_per_trade_lcb_pct")) or 0.0)
        > lcb_floor
    ]

    sparse_alpha_selected_winners.sort(
        key=lambda item: _as_float(item.get("selected_realized_pnl_usd")),
        reverse=True,
    )
    reliable_negative_selected_losers.sort(
        key=lambda item: _as_float(item.get("selected_realized_pnl_usd")),
    )
    deny_candidates.sort(key=lambda item: _as_float(item.get("selected_realized_pnl_usd")))
    reliable_negative_shadow_cells.sort(key=lambda item: _as_float(item.get("full_pnl_usd")))
    reliable_positive_shadow_cells.sort(
        key=lambda item: _as_float(item.get("full_pnl_per_trade_lcb_pct")),
        reverse=True,
    )

    return {
        "summary": {
            "shadow_rows": len(shadow_rows),
            "selected_rows": len(attribution_rows),
            "selected_joined_rows": len(selected_joined),
            "selected_rows_without_shadow": selected_without_shadow,
            "min_shadow_closed": min_closed,
            "reliable_negative_lcb_pct": lcb_floor,
            "sparse_alpha_selected_winners": len(sparse_alpha_selected_winners),
            "reliable_negative_selected_losers": len(
                reliable_negative_selected_losers
            ),
            "deny_candidates": len(deny_candidates),
            "reliable_negative_shadow_cells": len(reliable_negative_shadow_cells),
            "reliable_positive_shadow_cells": len(reliable_positive_shadow_cells),
        },
        "sparse_alpha_selected_winners": sparse_alpha_selected_winners,
        "reliable_negative_selected_losers": reliable_negative_selected_losers,
        "deny_candidates": deny_candidates,
        "reliable_negative_shadow_cells": reliable_negative_shadow_cells,
        "reliable_positive_shadow_cells": reliable_positive_shadow_cells,
    }


def write_flash_lcb_diagnostics(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    min_shadow_closed: int = 10,
    reliable_negative_lcb_pct: float = 0.0,
    top_n: int = 25,
) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    shadow_path = run_path / "flash_signal_key_shadow_report.json"
    attribution_path = run_path / "flash_attribution_summary.json"
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    report = build_flash_lcb_diagnostics(
        shadow_rows=tuple(shadow.get("rows", ())),
        attribution_rows=tuple(attribution.get("rows", ())),
        min_shadow_closed=min_shadow_closed,
        reliable_negative_lcb_pct=reliable_negative_lcb_pct,
    )

    out_dir = Path(output_dir) if output_dir is not None else run_path
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "flash_lcb_deny_diagnostics.json"
    md_path = out_dir / "flash_lcb_deny_diagnostics.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(report, run_path=run_path, top_n=top_n), encoding="utf-8")
    return json_path, md_path


def _selected_signal_key(row: Mapping[str, Any]) -> str:
    actor_key = str(row.get("actor_key") or "").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    action = str(row.get("action") or "").strip().upper()
    if not actor_key or not symbol or not action:
        return ""
    return f"{actor_key}|{symbol}|{action}"


def _joined_selected_row(
    row: Mapping[str, Any],
    shadow: Mapping[str, Any] | None,
    *,
    min_closed: int,
) -> dict[str, Any]:
    signal_key = _selected_signal_key(row)
    shadow_closed = 0 if shadow is None else _as_int(shadow.get("full_closed_trades"))
    shadow_lcb = None if shadow is None else _optional_float(
        shadow.get("full_pnl_per_trade_lcb_pct")
    )
    if shadow is None:
        shadow_status = "missing"
    elif shadow_closed < min_closed:
        shadow_status = "sparse"
    elif shadow_lcb is None:
        shadow_status = "missing_lcb"
    elif shadow_lcb < 0.0:
        shadow_status = "reliable_negative"
    else:
        shadow_status = "reliable_nonnegative"
    return {
        "signal_key": signal_key,
        "actor_key": str(row.get("actor_key") or ""),
        "actor_label": str(row.get("actor_label") or ""),
        "actor_type": str(row.get("actor_type") or ""),
        "symbol": str(row.get("symbol") or ""),
        "action": str(row.get("action") or ""),
        "selected_signals": _as_int(row.get("selected_signals")),
        "selected_closed_trades": _as_int(row.get("closed_trades")),
        "selected_realized_pnl_usd": _as_float(row.get("realized_pnl_usd")),
        "shadow_status": shadow_status,
        "shadow_full_closed_trades": shadow_closed,
        "shadow_full_pnl_usd": 0.0
        if shadow is None
        else _as_float(shadow.get("full_pnl_usd")),
        "shadow_full_pnl_per_trade_lcb_pct": shadow_lcb,
        "shadow_win_rate_pct": None
        if shadow is None
        else _optional_float(shadow.get("win_rate_pct")),
    }


def _shadow_cell(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "signal_key": str(row.get("signal_key") or ""),
        "actor_key": str(row.get("actor_key") or ""),
        "actor_label": str(row.get("actor_label") or ""),
        "actor_type": str(row.get("actor_type") or ""),
        "symbol": str(row.get("symbol") or ""),
        "action": str(row.get("action") or ""),
        "full_pnl_usd": _as_float(row.get("full_pnl_usd")),
        "full_closed_trades": _as_int(row.get("full_closed_trades")),
        "full_pnl_per_trade_lcb_pct": _optional_float(
            row.get("full_pnl_per_trade_lcb_pct")
        ),
        "win_rate_pct": _optional_float(row.get("win_rate_pct")),
    }


def _markdown_report(report: Mapping[str, Any], *, run_path: Path, top_n: int) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Flash LCB deny diagnostics",
        "",
        f"Run: `{run_path}`",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "shadow_rows",
        "selected_rows",
        "selected_rows_without_shadow",
        "min_shadow_closed",
        "sparse_alpha_selected_winners",
        "reliable_negative_selected_losers",
        "deny_candidates",
        "reliable_negative_shadow_cells",
        "reliable_positive_shadow_cells",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(
        [
            "",
            "Deny candidates are proposals only. Each candidate requires a retro run",
            "before it can be added to a live or pre-live profile.",
            "",
        ]
    )
    _append_table(
        lines,
        "Sparse/weak-shadow selected winners",
        report.get("sparse_alpha_selected_winners", ()),
        top_n=top_n,
        selected=True,
    )
    _append_table(
        lines,
        "Reliable negative selected losers / deny candidates",
        report.get("deny_candidates", ()),
        top_n=top_n,
        selected=True,
    )
    _append_table(
        lines,
        "Reliable negative shadow cells",
        report.get("reliable_negative_shadow_cells", ()),
        top_n=top_n,
        selected=False,
    )
    return "\n".join(lines) + "\n"


def _append_table(
    lines: list[str],
    title: str,
    rows: Any,
    *,
    top_n: int,
    selected: bool,
) -> None:
    lines.extend(["", f"## {title}", ""])
    limited = list(rows or ())[: max(0, int(top_n or 0))]
    if not limited:
        lines.append("_No rows._")
        return
    if selected:
        lines.append(
            "| Signal key | Selected PnL USD | Shadow closed | Shadow LCB % | Status |"
        )
        lines.append("|---|---:|---:|---:|---|")
        for row in limited:
            lines.append(
                "| {signal_key} | {pnl:.2f} | {closed} | {lcb} | {status} |".format(
                    signal_key=row.get("signal_key", ""),
                    pnl=_as_float(row.get("selected_realized_pnl_usd")),
                    closed=_as_int(row.get("shadow_full_closed_trades")),
                    lcb=_fmt_optional(row.get("shadow_full_pnl_per_trade_lcb_pct")),
                    status=row.get("shadow_status", ""),
                )
            )
        return
    lines.append("| Signal key | Full PnL USD | Closed | LCB % | Win rate % |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in limited:
        lines.append(
            "| {signal_key} | {pnl:.2f} | {closed} | {lcb} | {win_rate} |".format(
                signal_key=row.get("signal_key", ""),
                pnl=_as_float(row.get("full_pnl_usd")),
                closed=_as_int(row.get("full_closed_trades")),
                lcb=_fmt_optional(row.get("full_pnl_per_trade_lcb_pct")),
                win_rate=_fmt_optional(row.get("win_rate_pct")),
            )
        )


def _fmt_optional(value: Any) -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4f}"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-shadow-closed", type=int, default=10)
    parser.add_argument("--reliable-negative-lcb-pct", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args(argv)
    json_path, md_path = write_flash_lcb_diagnostics(
        args.run_dir,
        output_dir=args.output_dir,
        min_shadow_closed=args.min_shadow_closed,
        reliable_negative_lcb_pct=args.reliable_negative_lcb_pct,
        top_n=args.top_n,
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
