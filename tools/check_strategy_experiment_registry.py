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

from panteon_v2.policy.experiment_registry import (  # noqa: E402
    ExperimentRegistryError,
    StrategyExperimentRegistry,
    sha256_file,
)


DEFAULT_REGISTRY = ROOT / "configs" / "strategy_experiment_registry_v1.json"


def build_summary(
    registry: StrategyExperimentRegistry,
    *,
    experiment_id: str = "",
    artifact_integrity: dict[str, object] | None = None,
) -> dict[str, object]:
    experiments = registry.payload["experiments"]
    if experiment_id:
        experiments = [registry.experiment(experiment_id)]
    summary: dict[str, object] = {
        "schema_version": registry.payload["schema_version"],
        "exchange": registry.payload["exchange"],
        "operational_candidate_id": registry.operational_candidate_id,
        "family_verdicts": [
            {
                "family_id": item["family_id"],
                "status": item["status"],
                "retry_policy": item["retry_policy"],
            }
            for item in registry.payload["family_verdicts"]
        ],
        "experiments": [
            {
                "experiment_id": item["experiment_id"],
                "family_id": item["family_id"],
                "status": item["status"],
                "retry_policy": item["retry_policy"],
                "continuation_allowed": item["continuation_allowed"],
            }
            for item in experiments
        ],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    if artifact_integrity is not None:
        summary["artifact_integrity"] = artifact_integrity
    return summary


def verify_registered_artifacts(
    registry: StrategyExperimentRegistry,
    *,
    root: Path = ROOT,
) -> dict[str, object]:
    checked = 0
    for experiment in registry.payload["experiments"]:
        for artifact in experiment["artifacts"]:
            path = root / artifact["path"]
            if not path.is_file():
                raise ExperimentRegistryError(
                    f"registered artifact is missing: {artifact['path']}"
                )
            actual = sha256_file(path)
            if actual != artifact["sha256"]:
                raise ExperimentRegistryError(
                    f"registered artifact SHA-256 mismatch: {artifact['path']}"
                )
            checked += 1
    return {"checked": checked, "passed": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize terminal strategy experiments."
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--experiment-id", default="")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate registry schema and terminal flags without checking unpublished evidence files.",
    )
    args = parser.parse_args(argv)

    registry = StrategyExperimentRegistry.from_json(args.registry)
    artifact_integrity = (
        {"checked": 0, "passed": False, "reason": "metadata-only; evidence not verified"}
        if args.metadata_only
        else verify_registered_artifacts(registry)
    )
    print(
        json.dumps(
            build_summary(
                registry,
                experiment_id=args.experiment_id,
                artifact_integrity=artifact_integrity,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
