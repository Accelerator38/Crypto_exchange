from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.flash_offline_context_attribution import (  # noqa: E402
    write_offline_context_attribution_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a retrodate run's causal_entry_decisions.jsonl and write a "
            "flash attribution summary with context_rows."
        )
    )
    parser.add_argument("output_dir")
    parser.add_argument(
        "--filename",
        default="flash_attribution_summary_offline_context.json",
    )
    parser.add_argument("--initial-capital", type=float, default=None)
    parser.add_argument("--risk-capital-fraction", type=float, default=None)
    parser.add_argument("--risk-max-open-positions", type=int, default=None)
    args = parser.parse_args(argv)

    result = write_offline_context_attribution_summary(
        args.output_dir,
        filename=args.filename,
        initial_capital=args.initial_capital,
        risk_capital_fraction=args.risk_capital_fraction,
        risk_max_open_positions=args.risk_max_open_positions,
    )
    print(json.dumps({
        "path": str(result.path),
        "rows_read": result.rows_read,
        "replayed_signals": result.replayed_signals,
        "status_counts": result.status_counts,
        "execution_event_count": result.execution_event_count,
        "position_closed_event_count": result.position_closed_event_count,
        "realized_pnl_usd": result.realized_pnl_usd,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
