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
from panteon_v2.policy.experiment_registry import (  # noqa: E402
    StrategyExperimentRegistry,
)


DEFAULT_REGISTRY = ROOT / "configs" / "strategy_candidates_v1.json"
DEFAULT_EXPERIMENT_REGISTRY = (
    ROOT / "configs" / "strategy_experiment_registry_v1.json"
)


def build_summary(
    registry: StrategyCandidateRegistry,
    *,
    experiment_registry: StrategyExperimentRegistry | None = None,
) -> dict[str, object]:
    terminal_ids = (
        {
            str(item["profile_id"])
            for item in experiment_registry.payload["experiments"]
            if item["status"] == "terminal_rejected" and item["profile_id"]
        }
        if experiment_registry is not None
        else set()
    )
    ready_ids = [
        candidate_id
        for candidate_id in registry.ready_candidate_ids
        if candidate_id not in terminal_ids
    ]
    return {
        "schema_version": registry.payload["schema_version"],
        "dataset_id": registry.payload["dataset"]["dataset_id"],
        "operational_candidate_id": registry.payload["operational_candidate_id"],
        "ready_for_historical_oos": ready_ids,
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "contract_status": item["status"],
                "research_status": (
                    "terminal_rejected"
                    if item["candidate_id"] in terminal_ids
                    else item["status"]
                ),
                "profile_sha256": item["profile_sha256"],
                "missing_fields": item["data_requirements"]["missing_fields"],
                "historical_evaluation_allowed": (
                    item["historical_evaluation_allowed"]
                    and item["candidate_id"] not in terminal_ids
                ),
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
        description="Validate P2 contracts and overlay terminal research state."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--experiment-registry",
        default=str(DEFAULT_EXPERIMENT_REGISTRY),
    )
    args = parser.parse_args(argv)
    registry = StrategyCandidateRegistry.from_json(args.registry)
    experiment_registry = StrategyExperimentRegistry.from_json(
        args.experiment_registry
    )
    print(
        json.dumps(
            build_summary(
                registry,
                experiment_registry=experiment_registry,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
