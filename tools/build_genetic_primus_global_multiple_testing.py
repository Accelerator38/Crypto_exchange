from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.global_multiple_testing import (  # noqa: E402
    build_global_multiple_testing_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline Genetic Primus DSR/PBO evidence")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_global_multiple_testing_evidence(
        args.catalog if args.catalog.is_absolute() else ROOT / args.catalog,
        workspace=ROOT,
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    canonical_promotion_path = (
        ROOT / "Reports/Exia/Genetic_Primus/global_multiple_testing_v2.json"
    ).resolve()
    if output.resolve() == canonical_promotion_path and not evidence["promotion_eligible"]:
        raise PermissionError("diagnostic evidence cannot occupy the promotion evidence path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "promotion_eligible": evidence["promotion_eligible"],
                "materialized_objectives": evidence["materialized_objective_count"],
                "observations": evidence["observation_count"],
                "failed_readiness_gates": evidence["failed_readiness_gates"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
