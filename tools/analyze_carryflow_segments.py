"""Analyze independent CarryFlow evidence roots without joining their timelines."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.carryflow_segment_screening import (  # noqa: E402
    CarryFlowScreeningConfig,
    screen_carryflow_segments,
)
from panteon_v2.policy import CarryFlowEvidenceTape  # noqa: E402


DEFAULT_ROOT = ROOT / "Retrodate" / "bitget_carryflow_tape"
DEFAULT_OUTPUT = ROOT / "Reports" / "CarryFlow" / "segment_screening_latest.json"


def _resolve_tapes(run_dirs: Sequence[str]) -> list[Path]:
    if run_dirs:
        roots = [Path(item) for item in run_dirs]
    else:
        roots = sorted(DEFAULT_ROOT.glob("hourly_root_*"))
    paths: list[Path] = []
    for root in roots:
        candidate = (
            root
            if root.name.endswith(".jsonl")
            else root / "carryflow_evidence_tape.jsonl"
        )
        if not candidate.is_file():
            raise FileNotFoundError(f"evidence tape not found: {candidate}")
        resolved = candidate.resolve()
        if resolved not in paths:
            paths.append(resolved)
    if not paths:
        raise ValueError("no CarryFlow evidence tapes selected")
    return paths


def run(args: argparse.Namespace) -> dict:
    paths = _resolve_tapes(args.run_dir)
    tapes = [CarryFlowEvidenceTape.from_jsonl(path) for path in paths]
    config = CarryFlowScreeningConfig(
        oi_spike=args.oi_spike,
        funding_entry=args.funding_entry,
        crowd_ratio=args.crowd_ratio,
        current_short_basis_floor=args.current_short_basis_floor,
        research_short_basis_floor=args.research_short_basis_floor,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        hold_bars=args.hold_bars,
        round_trip_cost_bps=args.round_trip_cost_bps,
    )
    report = screen_carryflow_segments(tapes, config=config)
    grid_rows = []
    for oi_spike in (0.015, 0.02, 0.025, 0.03):
        for hold_bars in (6, 12, 18, 24):
            variant = screen_carryflow_segments(
                tapes,
                config=CarryFlowScreeningConfig(
                    oi_spike=oi_spike,
                    funding_entry=args.funding_entry,
                    crowd_ratio=args.crowd_ratio,
                    current_short_basis_floor=args.current_short_basis_floor,
                    research_short_basis_floor=args.research_short_basis_floor,
                    stop_pct=args.stop_pct,
                    target_pct=args.target_pct,
                    hold_bars=hold_bars,
                    round_trip_cost_bps=args.round_trip_cost_bps,
                ),
            )
            metrics = variant["research_candidate"]["trade_metrics"]
            grid_rows.append(
                {
                    "oi_spike": oi_spike,
                    "hold_bars": hold_bars,
                    "filled_orders": metrics["filled_orders"],
                    "closed_trades": metrics["closed_trades"],
                    "mean_net_bps": metrics["mean_net_bps"],
                    "expectancy_lcb_95_bps": metrics[
                        "expectancy_lcb_95_bps"
                    ],
                    "max_drawdown_bps": metrics["max_drawdown_bps"],
                    "max_symbol_share": metrics["max_symbol_share"],
                    "max_segment_share": metrics["max_segment_share"],
                    "failures": metrics["failures"],
                }
            )
    eligible = [
        row
        for row in grid_rows
        if row["closed_trades"] >= 10 and row["mean_net_bps"] > 0.0
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["expectancy_lcb_95_bps"]
            if row["expectancy_lcb_95_bps"] is not None
            else float("-inf"),
            row["mean_net_bps"],
        ),
        default=None,
    )
    report["hypothesis_grid"] = {
        "selection_data_is_in_sample": True,
        "multiple_testing_warning": True,
        "variants": grid_rows,
        "selected_for_fresh_prospective_validation": selected,
        "selection_is_promotion_evidence": False,
    }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["input_paths"] = [str(path) for path in paths]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["output"] = str(output.resolve())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and screen independent CarryFlow tapes. Results are "
            "research-only and never concatenate segment chronology."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Tape run directory or JSONL path; repeat for multiple roots.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--oi-spike", type=float, default=0.02)
    parser.add_argument("--funding-entry", type=float, default=0.00008)
    parser.add_argument("--crowd-ratio", type=float, default=0.58)
    parser.add_argument("--current-short-basis-floor", type=float, default=0.0004)
    parser.add_argument("--research-short-basis-floor", type=float, default=-0.0015)
    parser.add_argument("--stop-pct", type=float, default=0.012)
    parser.add_argument("--target-pct", type=float, default=0.024)
    parser.add_argument("--hold-bars", type=int, default=6)
    parser.add_argument("--round-trip-cost-bps", type=float, default=12.0)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
