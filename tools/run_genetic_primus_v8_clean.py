from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus_v8_clean.collector import collect_public_liquidations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded read-only Genetic Primus V8 clean collector")
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "Retrodate/genetic_primus_v8_clean")
    args = parser.parse_args()
    root_id = f"root_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    result = asyncio.run(
        collect_public_liquidations(
            contract_path=ROOT / "configs/genetic_primus_v8_clean.json",
            run_dir=args.base_dir / root_id,
            duration_seconds=args.duration_seconds,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
