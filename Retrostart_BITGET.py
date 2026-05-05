"""
Retrostart BITGET.

Runs a retrospective replay for every real BITGET trading session in
Results/BITGET/<timestamp>/ where live trading lasted at least 3 hours.

For each session it reads trading.log and all_signals.csv, replays the real
signals over that period, compares replay PnL with the saved live report and
shadow Panteon metrics, then writes Results/BITGET/_retrostart_report.json.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "src" / "panteon_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from retrostart_analyzer import run_retrostart  # noqa: E402


if __name__ == "__main__":
    exit_code = run_retrostart("BITGET", project_root=PROJECT_ROOT)
    sys.exit(exit_code)
