from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.analysis.strategy_lab import (  # noqa: E402
    evaluate_registered_candidates,
    write_strategy_lab_report,
)
from panteon_v2.policy.strategy_candidate_registry import (  # noqa: E402
    StrategyCandidateRegistry,
)


DEFAULT_REGISTRY = ROOT / "configs" / "strategy_candidates_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "StrategyLab" / "p2_historical_oos_20260727"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed-profile Bitget Strategy Lab V1 evaluation."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    registry = StrategyCandidateRegistry.from_json(args.registry)
    report = evaluate_registered_candidates(registry, repository_root=ROOT)
    json_path, markdown_path = write_strategy_lab_report(
        report, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "verdict": row["verdict"],
                        "passed": row["passed"],
                    }
                    for row in report["candidates"]
                ],
                "orders_enabled": False,
                "promotion_authority": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
