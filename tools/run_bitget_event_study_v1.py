"""Run the fixed costed Bitget triple-barrier event study."""

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

from panteon_v2.data.bitget import (  # noqa: E402
    evaluate_event_study,
    render_event_study_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed research-only costed triple-barrier study on a "
            "sealed Bitget microstructure dataset."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    report = evaluate_event_study(args.dataset_dir)
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(
        render_event_study_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "campaign_failures": report["campaign_failures"],
                "outcomes": report["outcomes"],
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "aggregate": row["aggregate"],
                        "oos": row["splits"]["oos"],
                        "statistical_pass": row["statistical_pass"],
                        "failures": row["failures"],
                    }
                    for row in report["candidates"]
                ],
                "output_json": str(output_json),
                "output_md": str(output_md),
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
