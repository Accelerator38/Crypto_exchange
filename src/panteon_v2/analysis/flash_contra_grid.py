"""Build advisory Flash/genetics contra experiment grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONTRA_RELATIONSHIPS = ("same_action", "same_side")
DEFAULT_CONTRA_MODE = "static_no_backfill"


def build_flash_contra_grid(
    *,
    intersection_rows: Sequence[Mapping[str, Any]],
    min_genetics_closed: int = 10,
    reliable_negative_lcb_pct: float = 0.0,
    allowed_relationships: Sequence[str] = DEFAULT_CONTRA_RELATIONSHIPS,
    min_selected_loss_usd: float = 0.0,
    support_period_bars: int = 720,
    min_support_periods: int = 1,
    min_regimes: int = 1,
) -> dict[str, Any]:
    """Return validation-required contra candidates from intersection rows.

    The grid is deliberately advisory. A candidate only means "worth validating
    as no-backfill contra"; it must not be promoted or enabled by this report.
    """

    min_closed = max(0, int(min_genetics_closed or 0))
    lcb_floor = float(reliable_negative_lcb_pct or 0.0)
    selected_loss_floor = abs(float(min_selected_loss_usd or 0.0))
    period_bars = max(1, int(support_period_bars or 1))
    min_periods = max(1, int(min_support_periods or 1))
    min_regime_count = max(1, int(min_regimes or 1))
    relationships = {
        str(relationship or "").strip()
        for relationship in allowed_relationships
        if str(relationship or "").strip()
    }

    groups: dict[str, dict[str, Any]] = {}
    seen_selected_by_key: dict[str, set[str]] = {}
    rejected_relationship = 0
    rejected_selected_not_loss = 0
    rejected_sparse_genetics = 0
    rejected_nonnegative_genetics_lcb = 0
    rejected_unstable_support_periods = 0
    rejected_unstable_regimes = 0
    eligible_rows = 0

    for row in intersection_rows:
        relationship = str(row.get("relationship") or "").strip()
        if relationship not in relationships:
            rejected_relationship += 1
            continue

        key = str(row.get("genetics_signal_key") or "").strip()
        if not key:
            continue
        selected_pnl_usd = _as_float(row.get("selected_realized_pnl_usd"))
        selected_closed = _as_int(row.get("selected_closed_trades"))
        if selected_closed <= 0 or selected_pnl_usd >= -selected_loss_floor:
            rejected_selected_not_loss += 1
            continue

        genetics_closed = _as_int(row.get("genetics_full_closed_trades"))
        if genetics_closed < min_closed:
            rejected_sparse_genetics += 1
            continue

        genetics_lcb = _optional_float(row.get("genetics_full_pnl_per_trade_lcb_pct"))
        if genetics_lcb is None or genetics_lcb >= lcb_floor:
            rejected_nonnegative_genetics_lcb += 1
            continue

        eligible_rows += 1
        group = groups.setdefault(
            key,
            _new_candidate(
                row,
                contra_signal_key=key,
                min_genetics_closed=min_closed,
                reliable_negative_lcb_pct=lcb_floor,
                min_selected_loss_usd=selected_loss_floor,
                support_period_bars=period_bars,
                min_support_periods=min_periods,
                min_regimes=min_regime_count,
            ),
        )
        _merge_candidate_row(
            group,
            row,
            relationship=relationship,
            selected_pnl_usd=selected_pnl_usd,
            selected_closed=selected_closed,
            genetics_closed=genetics_closed,
            genetics_lcb=genetics_lcb,
            seen_selected_keys=seen_selected_by_key.setdefault(key, set()),
            support_period=_support_period(row, period_bars=period_bars),
        )

    stable_groups: list[dict[str, Any]] = []
    for candidate in groups.values():
        if len(candidate.get("support_periods") or ()) < min_periods:
            rejected_unstable_support_periods += 1
            continue
        if len(candidate.get("regimes") or ()) < min_regime_count:
            rejected_unstable_regimes += 1
            continue
        stable_groups.append(candidate)

    candidates = [_finalize_candidate(candidate) for candidate in stable_groups]
    candidates.sort(
        key=lambda item: (
            _as_float(item.get("selected_loss_usd")),
            _as_float(item.get("genetics_min_lcb_pct")),
            str(item.get("contra_signal_key") or ""),
        )
    )

    return {
        "summary": {
            "intersection_rows": len(intersection_rows),
            "eligible_rows": eligible_rows,
            "allowed_relationships": sorted(relationships),
            "min_genetics_closed": min_closed,
            "reliable_negative_lcb_pct": lcb_floor,
            "min_selected_loss_usd": selected_loss_floor,
            "support_period_bars": period_bars,
            "min_support_periods": min_periods,
            "min_regimes": min_regime_count,
            "rejected_relationship": rejected_relationship,
            "rejected_selected_not_loss": rejected_selected_not_loss,
            "rejected_sparse_genetics": rejected_sparse_genetics,
            "rejected_nonnegative_genetics_lcb": rejected_nonnegative_genetics_lcb,
            "rejected_unstable_support_periods": rejected_unstable_support_periods,
            "rejected_unstable_regimes": rejected_unstable_regimes,
            "contra_candidates": len(candidates),
        },
        "contra_candidates": candidates,
        "experiment_specs": [
            _experiment_spec(candidate, index=index)
            for index, candidate in enumerate(candidates, start=1)
        ],
    }


def write_flash_contra_grid(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    min_genetics_closed: int = 10,
    reliable_negative_lcb_pct: float = 0.0,
    allowed_relationships: Sequence[str] = DEFAULT_CONTRA_RELATIONSHIPS,
    min_selected_loss_usd: float = 0.0,
    support_period_bars: int = 720,
    min_support_periods: int = 1,
    min_regimes: int = 1,
    top_n: int = 25,
) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    report_path = run_path / "flash_genetics_intersection_report.json"
    intersection = json.loads(report_path.read_text(encoding="utf-8"))
    report = build_flash_contra_grid(
        intersection_rows=tuple(intersection.get("rows", ())),
        min_genetics_closed=min_genetics_closed,
        reliable_negative_lcb_pct=reliable_negative_lcb_pct,
        allowed_relationships=allowed_relationships,
        min_selected_loss_usd=min_selected_loss_usd,
        support_period_bars=support_period_bars,
        min_support_periods=min_support_periods,
        min_regimes=min_regimes,
    )

    out_dir = Path(output_dir) if output_dir is not None else run_path
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "flash_genetics_contra_grid.json"
    md_path = out_dir / "flash_genetics_contra_grid.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(
        _markdown_report(report, run_path=run_path, top_n=top_n),
        encoding="utf-8",
    )
    return json_path, md_path


def _new_candidate(
    row: Mapping[str, Any],
    *,
    contra_signal_key: str,
    min_genetics_closed: int,
    reliable_negative_lcb_pct: float,
    min_selected_loss_usd: float,
    support_period_bars: int,
    min_support_periods: int,
    min_regimes: int,
) -> dict[str, Any]:
    return {
        "contra_signal_key": contra_signal_key,
        "mode": DEFAULT_CONTRA_MODE,
        "requires_validation": True,
        "reason": "negative_genetics_lcb_and_selected_loss",
        "genetics_actor_key": str(row.get("genetics_actor_key") or ""),
        "genetics_actor_label": str(row.get("genetics_actor_label") or ""),
        "symbols": set(),
        "actions": set(),
        "regimes": set(),
        "support_periods": set(),
        "relationship_counts": {},
        "selected_signal_keys": set(),
        "selected_loss_usd": 0.0,
        "selected_closed_trades": 0,
        "genetics_min_lcb_pct": None,
        "genetics_max_closed_trades": 0,
        "genetics_max_drawdown_pct": None,
        "support_rows": 0,
        "filters": {
            "min_genetics_closed": min_genetics_closed,
            "reliable_negative_lcb_pct": reliable_negative_lcb_pct,
            "min_selected_loss_usd": min_selected_loss_usd,
            "support_period_bars": support_period_bars,
            "min_support_periods": min_support_periods,
            "min_regimes": min_regimes,
        },
        "recommended_flags": [
            "--enable-flash-genetics-confirmation-overlay",
            "--enable-flash-genetics-confirmation-contra-static",
            "--enable-flash-genetics-confirmation-contra-no-backfill",
        ],
    }


def _merge_candidate_row(
    candidate: dict[str, Any],
    row: Mapping[str, Any],
    *,
    relationship: str,
    selected_pnl_usd: float,
    selected_closed: int,
    genetics_closed: int,
    genetics_lcb: float,
    seen_selected_keys: set[str],
    support_period: int,
) -> None:
    candidate["support_rows"] = _as_int(candidate.get("support_rows")) + 1
    _add_if_present(candidate["symbols"], row.get("symbol"))
    _add_if_present(candidate["actions"], row.get("genetics_action") or row.get("selected_action"))
    _add_if_present(candidate["regimes"], row.get("regime"))
    candidate["support_periods"].add(support_period)
    relationship_counts = candidate["relationship_counts"]
    relationship_counts[relationship] = _as_int(relationship_counts.get(relationship)) + 1

    selected_key = str(row.get("selected_signal_key") or "").strip()
    unique_selected_key = selected_key or f"row:{candidate['support_rows']}"
    if unique_selected_key not in seen_selected_keys:
        seen_selected_keys.add(unique_selected_key)
        candidate["selected_signal_keys"].add(unique_selected_key)
        candidate["selected_loss_usd"] = _as_float(
            candidate.get("selected_loss_usd")
        ) + selected_pnl_usd
        candidate["selected_closed_trades"] = _as_int(
            candidate.get("selected_closed_trades")
        ) + selected_closed

    current_lcb = _optional_float(candidate.get("genetics_min_lcb_pct"))
    candidate["genetics_min_lcb_pct"] = (
        genetics_lcb if current_lcb is None else min(current_lcb, genetics_lcb)
    )
    candidate["genetics_max_closed_trades"] = max(
        _as_int(candidate.get("genetics_max_closed_trades")),
        genetics_closed,
    )
    drawdown = _optional_float(row.get("genetics_max_drawdown_pct"))
    if drawdown is not None:
        current_drawdown = _optional_float(candidate.get("genetics_max_drawdown_pct"))
        candidate["genetics_max_drawdown_pct"] = (
            drawdown if current_drawdown is None else max(current_drawdown, drawdown)
        )


def _finalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    result["symbols"] = sorted(str(item) for item in result.get("symbols", ()) if item)
    result["actions"] = sorted(str(item) for item in result.get("actions", ()) if item)
    result["regimes"] = sorted(str(item) for item in result.get("regimes", ()) if item)
    support_periods = sorted(
        _as_int(item) for item in result.get("support_periods", ()) if item is not None
    )
    result["support_periods"] = support_periods
    result["support_period_count"] = len(support_periods)
    result["selected_signal_keys"] = sorted(
        str(item) for item in result.get("selected_signal_keys", ()) if item
    )
    result["relationship_counts"] = {
        str(key): _as_int(value)
        for key, value in sorted((result.get("relationship_counts") or {}).items())
    }
    result["genetics_closed_trades"] = _as_int(result.pop("genetics_max_closed_trades", 0))
    return result


def _support_period(row: Mapping[str, Any], *, period_bars: int) -> int:
    return max(0, _as_int(row.get("bar"))) // max(1, int(period_bars or 1))


def _experiment_spec(candidate: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    signal_key = str(candidate.get("contra_signal_key") or "")
    safe_key = (
        signal_key.replace(":", "_")
        .replace("|", "_")
        .replace("/", "")
        .replace(" ", "_")
    )
    return {
        "candidate_id": f"contra_grid_{index:03d}_{safe_key}",
        "contra_signal_keys": [signal_key],
        "mode": candidate.get("mode"),
        "requires_validation": True,
        "requires_validation_manifest": True,
        "promotion_allowed": False,
        "recommended_flags": list(candidate.get("recommended_flags") or ()),
        "selection_basis": {
            "reason": candidate.get("reason"),
            "selected_loss_usd": candidate.get("selected_loss_usd"),
            "genetics_min_lcb_pct": candidate.get("genetics_min_lcb_pct"),
            "support_rows": candidate.get("support_rows"),
            "relationship_counts": candidate.get("relationship_counts"),
        },
    }


def _markdown_report(report: Mapping[str, Any], *, run_path: Path, top_n: int) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Flash genetics contra grid",
        "",
        f"Run: `{run_path}`",
        "",
        "This report is advisory only: every contra candidate requires validation",
        "and must not be promoted from this grid alone.",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "intersection_rows",
        "eligible_rows",
        "allowed_relationships",
        "min_genetics_closed",
        "reliable_negative_lcb_pct",
        "min_selected_loss_usd",
        "support_period_bars",
        "min_support_periods",
        "min_regimes",
        "rejected_relationship",
        "rejected_selected_not_loss",
        "rejected_sparse_genetics",
        "rejected_nonnegative_genetics_lcb",
        "rejected_unstable_support_periods",
        "rejected_unstable_regimes",
        "contra_candidates",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")

    lines.extend(["", "## Contra candidates", ""])
    candidates = list(report.get("contra_candidates") or ())[: max(0, int(top_n or 0))]
    if not candidates:
        lines.append("_No rows._")
        return "\n".join(lines) + "\n"

    lines.append(
        "| Contra key | Mode | Selected loss USD | Genetics LCB % | Closed | Support | Periods | Relationships |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for candidate in candidates:
        lines.append(
            "| {key} | {mode} | {loss:.2f} | {lcb} | {closed} | {support} | {periods} | {relationships} |".format(
                key=candidate.get("contra_signal_key", ""),
                mode=candidate.get("mode", ""),
                loss=_as_float(candidate.get("selected_loss_usd")),
                lcb=_fmt_optional(candidate.get("genetics_min_lcb_pct")),
                closed=_as_int(candidate.get("genetics_closed_trades")),
                support=_as_int(candidate.get("support_rows")),
                periods=_as_int(candidate.get("support_period_count")),
                relationships=json.dumps(
                    candidate.get("relationship_counts") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
    return "\n".join(lines) + "\n"


def _add_if_present(target: set[Any], value: Any) -> None:
    text = str(value or "").strip()
    if text:
        target.add(text)


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


def _split_csv(values: str | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_CONTRA_RELATIONSHIPS
    return tuple(item.strip() for item in values.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-genetics-closed", type=int, default=10)
    parser.add_argument("--reliable-negative-lcb-pct", type=float, default=0.0)
    parser.add_argument("--allowed-relationships", default=",".join(DEFAULT_CONTRA_RELATIONSHIPS))
    parser.add_argument("--min-selected-loss-usd", type=float, default=0.0)
    parser.add_argument("--support-period-bars", type=int, default=720)
    parser.add_argument("--min-support-periods", type=int, default=1)
    parser.add_argument("--min-regimes", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args(argv)
    json_path, md_path = write_flash_contra_grid(
        args.run_dir,
        output_dir=args.output_dir,
        min_genetics_closed=args.min_genetics_closed,
        reliable_negative_lcb_pct=args.reliable_negative_lcb_pct,
        allowed_relationships=_split_csv(args.allowed_relationships),
        min_selected_loss_usd=args.min_selected_loss_usd,
        support_period_bars=args.support_period_bars,
        min_support_periods=args.min_support_periods,
        min_regimes=args.min_regimes,
        top_n=args.top_n,
    )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
