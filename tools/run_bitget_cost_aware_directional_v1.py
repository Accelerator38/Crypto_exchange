"""Run the fixed Bitget anchored cost-aware directional classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from panteon_v2.analysis.cost_aware_directional import (  # noqa: E402
    evaluate_cost_aware_directional,
    render_directional_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed anchored walk-forward cost-aware directional discovery "
            "on a validated Bitget full8 1m history dataset."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    report = evaluate_cost_aware_directional(args.dataset_dir)
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_directional_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "aggregate_oos": report["aggregate_oos"],
                "baseline_oos": report["baseline_oos"],
                "failures": report["failures"],
                "folds": [
                    {
                        "fold": row["fold"],
                        "selected_threshold": row["selected_threshold"],
                        "validation_gate_passed": row[
                            "validation_gate_passed"
                        ],
                        "validation_metrics": row["validation_metrics"],
                        "oos_metrics": row["oos_metrics"],
                    }
                    for row in report["folds"]
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
