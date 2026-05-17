"""
Retrostart MEXC.

Runs a retrospective replay for every real MEXC trading session in
Results/MEXC/<timestamp>/ where live trading lasted at least 3 hours.

For each session it reads trading.log and all_signals.csv, replays the real
signals over that period, compares replay PnL with the saved live report and
shadow Panteon metrics, then writes Results/MEXC/_retrostart_report.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
RUNTIME_DIR = PROJECT_ROOT / "src" / "panteon_runtime"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from panteon_v2.analysis.retro_whatif import run_retrodate_whatif_for_exchange  # noqa: E402
from retrostart_analyzer import run_retrostart  # noqa: E402


def _technical_legacy_code(code: int) -> int:
    """Legacy code 1 means diagnostic findings, not a failed retro run."""
    return 0 if code == 1 else code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline MEXC retrostart and selector what-if")
    parser.add_argument("--skip-legacy", action="store_true", help="skip old retrostart analyzer")
    parser.add_argument("--no-png", action="store_true", help="skip PNG dashboard rendering")
    parser.add_argument("--max-decisions", type=int, default=None, help="limit analyzed decisions")
    args = parser.parse_args(argv)

    legacy_code = 0
    if not args.skip_legacy:
        legacy_code = _technical_legacy_code(
            run_retrostart("MEXC", project_root=PROJECT_ROOT)
        )
    whatif_code = run_retrodate_whatif_for_exchange(
        "MEXC",
        project_root=PROJECT_ROOT,
        include_png=not args.no_png,
        max_decisions=args.max_decisions,
    )
    return whatif_code if whatif_code else legacy_code


if __name__ == "__main__":
    sys.exit(main())
