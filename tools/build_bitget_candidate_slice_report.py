from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _slice_score(row: Mapping[str, Any]) -> float:
    return (
        _as_float(row.get("lcb_usd"))
        + _as_float(row.get("expectancy_usd")) * 0.5
        - abs(_as_float(row.get("gross_loss"))) * 0.25
        - _as_float(row.get("max_drawdown_usd")) * 0.5
    )


def _has_transition_confirmation(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("transition_confirmed")
        or row.get("breakout_transition_confirmed")
        or row.get("confirmed_transition_breakout")
    )


def _range_low_vol_allowed(row: Mapping[str, Any]) -> bool:
    if str(row.get("regime") or "").lower() != "range_low_vol":
        return True
    return _has_transition_confirmation(row)


def _slice_fail_reasons(
    row: Mapping[str, Any],
    *,
    min_closed_trades: int,
    min_filled_signals: int,
    max_drawdown_usd: float | None,
) -> list[str]:
    reasons = [
        str(reason)
        for reason in (row.get("fail_reasons") or [])
        if str(reason).strip()
    ]
    if not bool(row.get("promotion_eligible", False)):
        reasons.append("matrix_slice_not_promotion_eligible")
    if _as_int(row.get("filled_signals")) < int(min_filled_signals):
        reasons.append("min_filled_signals")
    if _as_int(row.get("closed_trades")) < int(min_closed_trades):
        reasons.append("min_closed_trades")
    if _as_float(row.get("expectancy_usd")) <= 0.0:
        reasons.append("nonpositive_expectancy")
    if _as_float(row.get("lcb_usd")) <= 0.0:
        reasons.append("nonpositive_lcb")
    if max_drawdown_usd is not None and _as_float(row.get("max_drawdown_usd")) > float(max_drawdown_usd):
        reasons.append("max_drawdown_usd")
    if not _range_low_vol_allowed(row):
        reasons.append("range_low_vol_requires_transition_confirmation")
    return sorted(set(reasons))


def _iter_slice_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_slices = summary.get("candidate_slices") or {}
    if isinstance(raw_slices, Mapping):
        return [
            {"key": str(key), **dict(value)}
            for key, value in raw_slices.items()
            if isinstance(value, Mapping)
        ]
    if isinstance(raw_slices, Sequence) and not isinstance(raw_slices, (str, bytes)):
        return [
            dict(value)
            for value in raw_slices
            if isinstance(value, Mapping)
        ]
    return []


def build_slice_report(
    matrix_summary: Mapping[str, Any],
    *,
    min_closed_trades: int = 10,
    min_filled_signals: int = 10,
    max_drawdown_usd: float | None = 0.2,
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for row in _iter_slice_rows(matrix_summary):
        fail_reasons = _slice_fail_reasons(
            row,
            min_closed_trades=min_closed_trades,
            min_filled_signals=min_filled_signals,
            max_drawdown_usd=max_drawdown_usd,
        )
        scored = {
            **row,
            "score": round(_slice_score(row), 12),
            "eligible_for_canary": not fail_reasons,
            "fail_reasons": fail_reasons,
        }
        ranked.append(scored)
    ranked.sort(
        key=lambda item: (
            bool(item.get("eligible_for_canary")),
            _as_float(item.get("score")),
            _as_float(item.get("lcb_usd")),
            _as_int(item.get("closed_trades")),
        ),
        reverse=True,
    )
    recommended = [row for row in ranked if bool(row.get("eligible_for_canary"))]
    return {
        "source_results_root": matrix_summary.get("results_root", ""),
        "source_candidate_variant": (
            (matrix_summary.get("promotion_gates") or {}).get("candidate_variant", "")
            if isinstance(matrix_summary.get("promotion_gates"), Mapping)
            else ""
        ),
        "gates": {
            "min_closed_trades": int(min_closed_trades),
            "min_filled_signals": int(min_filled_signals),
            "max_drawdown_usd": max_drawdown_usd,
        },
        "ranked_slices": ranked,
        "recommended_slices": recommended,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a ranked Bitget candidate slice report.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-closed-trades", type=int, default=10)
    parser.add_argument("--min-filled-signals", type=int, default=10)
    parser.add_argument("--max-drawdown-usd", type=float, default=0.2)
    args = parser.parse_args(argv)

    matrix_summary = json.loads(Path(args.matrix_summary).read_text(encoding="utf-8"))
    report = build_slice_report(
        matrix_summary,
        min_closed_trades=args.min_closed_trades,
        min_filled_signals=args.min_filled_signals,
        max_drawdown_usd=args.max_drawdown_usd,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "recommended_slices": len(report["recommended_slices"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
