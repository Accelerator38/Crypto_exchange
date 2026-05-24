from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.retrodate_market_runner import main as retrodate_main
from tools.run_panteon_flash_profitability_matrix import (
    PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
)


PREVIOUS_BEST_RUN_SUMMARY = (
    ROOT
    / "Results"
    / "PanteonFlashRound3Deny8EntryRegime_Full2022_2026_20260524"
    / "RETRODATE_MARKET"
    / "2026-05-23_21-15-29_retrodate_market_v2"
    / "run_summary.json"
)
RESULTS_ROOT = ROOT / "Results" / "PanteonFlashPreLiveSymbolGuard_Full2022_2026_20260524"


def _load_previous_keys(name: str) -> list[str]:
    data = json.loads(PREVIOUS_BEST_RUN_SUMMARY.read_text(encoding="utf-8"))
    values = data.get(name, [])
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value or "").strip()]


def build_args() -> list[str]:
    args = [
        "--years",
        "2022,2023,2024,2025,2026",
        "--stride-minutes",
        "60",
        "--results-root",
        str(RESULTS_ROOT),
        "--progress-every-bars",
        "1000",
        *PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS,
        "--enable-flash-degradation-actor-guard",
        "--flash-degradation-actor-scope",
        "actor_regime",
        "--flash-degradation-signal-cooldown-bars",
        "720",
        "--flash-degradation-actor-cooldown-bars",
        "72",
        "--enable-flash-degradation-symbol-guard",
        "--flash-degradation-symbol-cooldown-bars",
        "720",
        "--flash-degradation-symbol-window-closed-trades",
        "3",
        "--flash-degradation-symbol-min-closed-trades",
        "3",
        "--flash-degradation-symbol-max-recent-pnl-usd",
        "-25.0",
    ]
    for key in _load_previous_keys("flash_denied_signal_keys"):
        args.extend(["--flash-deny-signal-key", key])
    for key in _load_previous_keys("flash_terminal_denied_signal_keys"):
        args.extend(["--flash-terminal-deny-signal-key", key])
    return args


def main() -> int:
    return int(retrodate_main(build_args()) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
