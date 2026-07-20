"""Evaluate immutable CarryFlow profiles on independent seeded tape roots."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carryflow_policy import carryflow_profile_ids  # noqa: E402

REPLAY_TOOL = ROOT / "tools" / "run_policy_replay_v1.py"
DEFAULT_OUTPUT = (
    ROOT / "Reports" / "CarryFlow" / "exact_profile_evaluation_latest.json"
)
PROFILE_IDS = carryflow_profile_ids()
SCREENING_MAX_DRAWDOWN_USD = 0.20
EVIDENCE_EXTENSION_ONLY_FAILURES = frozenset({"nonpositive_lcb"})


def _load_replay_tool():
    spec = importlib.util.spec_from_file_location(
        "run_policy_replay_v1_for_exact_sweep",
        REPLAY_TOOL,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load policy replay tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_segment(value: str) -> tuple[Path, Path]:
    parts = str(value).split("::", 1)
    if len(parts) != 2:
        raise ValueError("segment must use TAPE::WARMUP_SEED format")
    tape = Path(parts[0]).resolve()
    seed = Path(parts[1]).resolve()
    if not tape.is_file() or not seed.is_file():
        raise FileNotFoundError(f"segment input missing: {tape} :: {seed}")
    return tape, seed


def _run_namespace(
    *,
    tape: Path,
    seed: Path,
    out_dir: Path,
    profile_id: str,
) -> Namespace:
    return Namespace(
        data="unused.csv",
        evidence_tape=str(tape),
        warmup_seed=str(seed),
        derivatives_context_csv=None,
        allow_legacy_split_input=False,
        out_dir=str(out_dir),
        policy_id=f"exact-profile-{profile_id}-{out_dir.name}",
        profile=profile_id,
        max_snapshots=None,
        oos_fraction=0.35,
        use_live_regime_detector=False,
        initial_capital_usd=1000.0,
        capital_fraction=0.01,
        max_notional_usd=10.0,
        max_daily_loss_usd=5.0,
        exchange_min_notional_usd=5.0,
        round_trip_fee_bps=8.0,
        slippage_bps=4.0,
        safety_buffer_bps=4.0,
        assumed_spread_bps=2.0,
        no_flatten_end=True,
    )


def _load_trade_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _aggregate(
    profile_id: str,
    segment_results: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [float(row["net_pnl_usd"]) for row in trades]
    closed = len(values)
    mean = statistics.fmean(values) if values else 0.0
    lcb = (
        mean - 1.6448536269514722 * statistics.stdev(values) / math.sqrt(closed)
        if closed >= 2
        else None
    )
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    segment_trade_values = {
        int(row["segment"]): [
            float(trade["net_pnl_usd"])
            for trade in trades
            if int(trade.get("_segment", -1)) == int(row["segment"])
        ]
        for row in segment_results
    }
    active_roots = sum(bool(root_values) for root_values in segment_trade_values.values())
    root_collapses = [
        segment
        for segment, root_values in segment_trade_values.items()
        if root_values and statistics.fmean(root_values) <= 0.0
    ]
    leave_one_root_out = []
    for segment, held_out_values in segment_trade_values.items():
        train_values = [
            value
            for other_segment, root_values in segment_trade_values.items()
            if other_segment != segment
            for value in root_values
        ]
        leave_one_root_out.append(
            {
                "held_out_segment": segment,
                "train_closed_trades": len(train_values),
                "train_expectancy_usd": (
                    statistics.fmean(train_values) if train_values else None
                ),
                "held_out_closed_trades": len(held_out_values),
                "held_out_expectancy_usd": (
                    statistics.fmean(held_out_values)
                    if held_out_values
                    else None
                ),
                "held_out_positive": bool(
                    held_out_values
                    and statistics.fmean(held_out_values) > 0.0
                ),
            }
        )
    failures = []
    if closed * 2 < 20:
        failures.append("fills_below_20")
    if closed < 10:
        failures.append("closed_trades_below_10")
    if mean <= 0.0:
        failures.append("nonpositive_costed_expectancy")
    if lcb is None or lcb <= 0.0:
        failures.append("nonpositive_lcb")
    if drawdown > SCREENING_MAX_DRAWDOWN_USD:
        failures.append("drawdown_above_limit")
    if active_roots < 2:
        failures.append("active_roots_below_2")
    failures.extend(
        f"root_expectancy_collapse:segment_{segment}"
        for segment in root_collapses
    )
    evidence_extension_eligible = bool(
        failures
        and set(failures) == EVIDENCE_EXTENSION_ONLY_FAILURES
    )
    return {
        "profile_id": profile_id,
        "candidate_signals": sum(
            int(row["candidate_signals"]) for row in segment_results
        ),
        "filled_orders": closed * 2,
        "closed_trades": closed,
        "gross_pnl_usd": sum(float(row["gross_pnl_usd"]) for row in trades),
        "total_costs_usd": sum(float(row["costs_usd"]) for row in trades),
        "net_pnl_usd": sum(values),
        "mean_net_pnl_usd": mean,
        "expectancy_lcb_95_usd": lcb,
        "positive_rate": (
            sum(value > 0.0 for value in values) / closed if closed else 0.0
        ),
        "exit_reasons": dict(
            Counter(str(row.get("close_reason", "unknown")) for row in trades)
        ),
        "max_drawdown_usd": drawdown,
        "max_drawdown_limit_usd": SCREENING_MAX_DRAWDOWN_USD,
        "mean_cost_bps": (
            statistics.fmean(
                float(row["costs_usd"])
                / float(row["notional_usd"])
                * 10_000.0
                for row in trades
                if float(row["notional_usd"]) > 0.0
            )
            if trades
            else 0.0
        ),
        "active_roots": active_roots,
        "root_expectancy_collapses": root_collapses,
        "leave_one_root_out": leave_one_root_out,
        "segment_results": list(segment_results),
        "screening_passed": not failures,
        "evidence_extension_eligible": evidence_extension_eligible,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    segments = [_parse_segment(value) for value in args.segment]
    if not segments:
        raise ValueError("at least one --segment is required")
    replay_tool = _load_replay_tool()
    output = Path(args.output).resolve()
    run_root = output.parent / f"exact_profiles_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    rows = []
    for profile_id in PROFILE_IDS:
        segment_summaries = []
        trade_rows = []
        for index, (tape, seed) in enumerate(segments, start=1):
            segment_dir = run_root / profile_id / f"segment_{index}"
            summary = replay_tool.run(
                _run_namespace(
                    tape=tape,
                    seed=seed,
                    out_dir=segment_dir,
                    profile_id=profile_id,
                )
            )
            segment_summaries.append(
                {
                    "segment": index,
                    "tape": str(tape),
                    "seed": str(seed),
                    "seed_prospective": bool(
                        summary["warmup_seed"]["prospective_for_tape"]
                    ),
                    "manifest_sha256": summary["manifest_sha256"],
                    "candidate_signals": summary["candidate_signals"],
                    "filled_orders": summary["filled_orders"],
                    "closed_trades": summary["closed_trades"],
                    "remaining_open_positions": summary[
                        "remaining_open_positions"
                    ],
                    "end_positions_censored": summary[
                        "end_positions_censored"
                    ],
                    "net_pnl_usd": summary["net_pnl_usd"],
                    "expectancy_after_costs_usd": summary[
                        "expectancy_after_costs_usd"
                    ],
                    "expectancy_lcb_usd": summary["expectancy_lcb_usd"],
                }
            )
            for trade in _load_trade_rows(segment_dir / "closed_trades.jsonl"):
                trade["_segment"] = index
                trade_rows.append(trade)
        rows.append(_aggregate(profile_id, segment_summaries, trade_rows))
    ranked = sorted(
        rows,
        key=lambda row: (
            row["expectancy_lcb_95_usd"]
            if row["expectancy_lcb_95_usd"] is not None
            else float("-inf"),
            row["mean_net_pnl_usd"],
            row["closed_trades"],
        ),
        reverse=True,
    )
    selectable = [row for row in ranked if row["screening_passed"]]
    extension_candidates = [
        row for row in ranked if row["evidence_extension_eligible"]
    ]
    report = {
        "schema_version": "panteon.carryflow_exact_profile_evaluation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "promotion_authority": False,
        "orders_enabled": False,
        "segments_concatenated": False,
        "profile_selection_only": True,
        "independent_actor_knobs": 1,
        "profile_count": len(PROFILE_IDS),
        "screening_max_drawdown_usd": SCREENING_MAX_DRAWDOWN_USD,
        "endpoint_policy": "right_censor_open_positions",
        "root_robustness_required": True,
        "fresh_prospective_validation_required": True,
        "evidence_extension_orders_enabled": False,
        "evidence_extension_is_promotion": False,
        "segments": [
            {"tape": str(tape), "warmup_seed": str(seed)}
            for tape, seed in segments
        ],
        "ranked_profiles": ranked,
        "best_diagnostic_only": ranked[0] if ranked else None,
        "selected_for_prospective_validation": (
            selectable[0] if selectable else None
        ),
        "selected_for_evidence_extension": (
            extension_candidates[0] if extension_candidates else None
        ),
        "output_root": str(run_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["output"] = str(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exact immutable CarryFlow profile evaluation across independent "
            "seeded roots."
        )
    )
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Repeat TAPE::WARMUP_SEED for each independent root.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
