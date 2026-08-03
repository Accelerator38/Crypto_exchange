"""Collect and process one fixed 12-hour Bitget precursor evidence session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (ROOT, SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.data.bitget.evidence_session import (  # noqa: E402
    run_precursor_evidence_session,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed 12-hour read-only Bitget collection, materialize its "
            "sealed data and update the frozen precursor campaign."
        )
    )
    parser.parse_args(argv)
    result = run_precursor_evidence_session(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
