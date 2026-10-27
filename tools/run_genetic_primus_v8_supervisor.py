from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus_v8_clean.supervisor import run_bounded_supervisor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded fail-closed Genetic Primus V8 supervisor")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--root-duration-seconds", type=float, default=30.0)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=ROOT / "Retrodate/genetic_primus_v8_clean_supervised",
    )
    args = parser.parse_args()
    result = asyncio.run(
        run_bounded_supervisor(
            root=ROOT,
            contract_path=ROOT / "configs/genetic_primus_v8_clean.json",
            base_dir=args.base_dir,
            cycles=args.cycles,
            root_duration_seconds=args.root_duration_seconds,
            pause_seconds=args.pause_seconds,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["result"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
