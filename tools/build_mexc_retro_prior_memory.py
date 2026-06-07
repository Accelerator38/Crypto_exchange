"""Build a conservative MEXC retro-prior PerformanceMemory snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.memory.retro_prior import (  # noqa: E402
    build_retro_prior_snapshot_from_breakdown_files,
    write_retro_prior_snapshot,
)


DEFAULT_OUTPUT = ROOT / "panteon_v2_state" / "mexc_retro_prior_memory.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small MEXC virtual-memory prior from Panteon retrotest "
            "agent/player regime breakdown CSV files."
        )
    )
    parser.add_argument(
        "--retro-dir",
        type=Path,
        help="Retrotest output directory containing agent/player regime breakdown CSVs.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Explicit breakdown CSV path. Can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output prior snapshot path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--source-run-id", default="", help="Audit label stored in _retro_prior metadata.")
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--max-prior-closed-trades", type=int, default=24)
    parser.add_argument("--max-abs-prior-pnl-pct", type=float, default=8.0)
    parser.add_argument("--min-closed-trades", type=int, default=5)
    return parser.parse_args(argv)


def _inputs_from_args(args: argparse.Namespace) -> list[Path]:
    inputs = list(args.input or [])
    if args.retro_dir:
        for name in ("agent_regime_breakdown.csv", "player_regime_breakdown.csv"):
            candidate = args.retro_dir / name
            if candidate.exists():
                inputs.append(candidate)
    seen: set[Path] = set()
    out: list[Path] = []
    for item in inputs:
        path = item.resolve()
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = _inputs_from_args(args)
    if not inputs:
        print("No breakdown inputs found. Use --retro-dir or --input.", file=sys.stderr)
        return 2
    snapshot = build_retro_prior_snapshot_from_breakdown_files(
        inputs,
        exchange="MEXC",
        source_run_id=args.source_run_id or (str(args.retro_dir) if args.retro_dir else ""),
        prior_weight=args.prior_weight,
        max_prior_closed_trades=args.max_prior_closed_trades,
        max_abs_prior_pnl_pct=args.max_abs_prior_pnl_pct,
        min_closed_trades=args.min_closed_trades,
    )
    write_retro_prior_snapshot(snapshot, args.output)
    meta = snapshot.get("_retro_prior") or {}
    print(
        "Built MEXC retro prior:",
        f"pairs={len(snapshot.get('state') or {})}",
        f"rows_read={meta.get('rows_read')}",
        f"rows_used={meta.get('rows_used')}",
        f"output={args.output}",
    )
    warnings = meta.get("warnings") or []
    for warning in warnings[:5]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

