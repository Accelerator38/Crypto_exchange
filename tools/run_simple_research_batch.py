from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simple_research.batch import run_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the preregistered reset batch.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "simple_research"
            / "preregistered_strategies_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Reports" / "SimpleResearch" / "reset_v1",
    )
    args = parser.parse_args()
    report = run_batch(
        registry_path=args.registry.resolve(),
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"survivors": report["survivors"], "strategy_count": report["strategy_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
