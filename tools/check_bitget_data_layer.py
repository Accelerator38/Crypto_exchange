"""Validate immutable Bitget data segments without reading trading state."""

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

from panteon_v2.data.bitget import validate_data_session  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a sealed Bitget public-data session."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_data_session(args.session_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["segments_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
