"""Validate a sealed Bitget microstructure research dataset."""

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

from panteon_v2.data.bitget import validate_research_dataset  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Bitget microstructure_research_v1."
    )
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            validate_research_dataset(args.dataset_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
