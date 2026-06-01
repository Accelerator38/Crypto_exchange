from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_legend.stitched_logs import build_stitched_log_report  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="JSONL live event log path. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Reports" / "PanteonLegend_20260531" / "stitched_live_logs",
    )
    parser.add_argument("--tail-mb", type=float, default=80.0)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args(argv)

    logs = [Path(path) for path in args.log] or [
        ROOT / "logs" / "v2_bitget_events.jsonl",
        ROOT / "logs" / "v2_mexc_events.jsonl",
    ]
    tail_bytes = None
    if args.tail_mb and args.tail_mb > 0:
        tail_bytes = int(float(args.tail_mb) * 1024 * 1024)

    report = build_stitched_log_report(
        logs,
        output_dir=args.output_dir,
        tail_bytes=tail_bytes,
        max_events=args.max_events,
    )
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"events={report['total_events']}", flush=True)
    print(f"report={args.output_dir / 'stitched_live_log_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
