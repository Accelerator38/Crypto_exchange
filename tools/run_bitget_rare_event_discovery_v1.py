"""Run the fixed Bitget rare-event discovery campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (ROOT, SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.analysis.rare_event_discovery import (  # noqa: E402
    evaluate_rare_event_discovery,
    render_rare_event_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run five fixed low-turnover Bitget event hypotheses with anchored "
            "validation, OOS and cost stress."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    report = evaluate_rare_event_discovery(args.dataset_dir)
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_rare_event_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "passed_candidates": report["passed_candidates"],
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "raw_signal_outcomes": row["raw_signal_outcomes"],
                        "aggregate_oos": row["aggregate_oos"],
                        "failures": row["failures"],
                    }
                    for row in report["candidates"]
                ],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "runtime_profile_created": False,
                "orders_enabled": False,
                "promotion_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
