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

from panteon_v2.policy.strategy_candidate_registry import (  # noqa: E402
    StrategyCandidateRegistry,
)


DEFAULT_REGISTRY = ROOT / "configs" / "strategy_candidates_v1.json"


def build_summary(registry: StrategyCandidateRegistry) -> dict[str, object]:
    return {
        "schema_version": registry.payload["schema_version"],
        "dataset_id": registry.payload["dataset"]["dataset_id"],
        "operational_candidate_id": registry.payload["operational_candidate_id"],
        "ready_for_historical_oos": list(registry.ready_candidate_ids),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "status": item["status"],
                "profile_sha256": item["profile_sha256"],
                "missing_fields": item["data_requirements"]["missing_fields"],
                "historical_evaluation_allowed": item[
                    "historical_evaluation_allowed"
                ],
                "paper_allowed": item["paper_allowed"],
                "live_allowed": item["live_allowed"],
            }
            for item in registry.payload["candidates"]
        ],
        "orders_enabled": False,
        "promotion_authority": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate immutable P2 strategy candidate contracts."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    registry = StrategyCandidateRegistry.from_json(args.registry)
    print(
        json.dumps(
            build_summary(registry),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
