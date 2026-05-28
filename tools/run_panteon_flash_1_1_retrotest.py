from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_flash_selected_subset_candidate as candidate  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT
    / "Reports"
    / "PanteonFlashSelectedSubsetPredeploy_20260525"
    / "flash_selected_subset_manifest_no_weak_cut.json"
)
DEFAULT_RESULTS_BASE = ROOT / "Results" / "PanteonFlash_1_1_20260528"
DEFAULT_YEARS = "2022,2023,2024,2025,2026"

FLASH_1_1_BEST_COMPATIBLE_ARGS = (
    "--risk-capital-fraction",
    "0.12",
    "--max-new-opens-per-bar",
    "1",
    "--risk-max-open-positions",
    "8",
    "--flash-min-score-to-trade",
    "4.0",
)

FLASH_1_1_REPORT_CORE_ARGS = (
    *FLASH_1_1_BEST_COMPATIBLE_ARGS,
    "--v3-shadow-fresh-handoff-max-age-bars",
    "12",
    "--flash-degradation-window-closed-trades",
    "5",
    "--flash-degradation-min-closed-trades",
    "3",
    "--flash-degradation-max-recent-pnl-usd",
    "-50.0",
    "--flash-degradation-signal-cooldown-bars",
    "240",
    "--flash-degradation-actor-cooldown-bars",
    "36",
    "--enable-flash-degradation-symbol-guard",
    "--flash-degradation-symbol-cooldown-bars",
    "48",
    "--flash-degradation-symbol-lookback-bars",
    "504",
    "--flash-degradation-symbol-window-closed-trades",
    "5",
    "--flash-degradation-symbol-min-closed-trades",
    "3",
    "--flash-degradation-symbol-max-recent-pnl-usd",
    "-50.0",
    "--enable-flash-degradation-recovery",
    "--flash-degradation-recovery-min-closed-trades",
    "3",
    "--flash-degradation-recovery-min-recent-pnl-usd",
    "0.0",
)

FLASH_1_1_ABLATION_ARGS = {
    "handoff-age12": (
        "--v3-shadow-fresh-handoff-max-age-bars",
        "12",
    ),
    "symbol-lookback504": (
        "--flash-degradation-symbol-lookback-bars",
        "504",
    ),
    "signal-cooldown480": (
        "--flash-degradation-signal-cooldown-bars",
        "480",
    ),
}

FLASH_1_1_EXPERIMENTAL_PROFIT_ARGS = (
    "--max-new-opens-per-bar",
    "3",
    "--flash-min-score-to-trade",
    "3.5",
    "--enable-partial-profit-lock",
    "--partial-profit-lock-trigger-pnl-pct",
    "3.0",
    "--partial-profit-lock-close-fraction",
    "0.33",
    "--partial-profit-lock-min-age-bars",
    "6",
    "--disable-partial-profit-lock-skip-protected-signal-keys",
    "--enable-flash-volatility-risk-sizing",
    "--flash-volatility-risk-target-pct",
    "2.0",
    "--flash-volatility-risk-min-volatility-pct",
    "0.5",
    "--flash-volatility-risk-max-mult",
    "1.5",
    "--enable-flash-actor-risk-sizing",
    "--flash-actor-risk-min-mult",
    "0.5",
    "--flash-actor-risk-max-mult",
    "1.5",
    "--flash-actor-risk-edge-scale-pct",
    "1.0",
    "--flash-anchor-actor-key",
    "ensemble:Optimal_StaticRotator",
    "--flash-anchor-actor-key",
    "ensemble:Antonius_conservative",
    "--flash-portfolio-actor-key",
    "ensemble:Optimal_StaticRotator",
    "--flash-portfolio-actor-key",
    "ensemble:Antonius_conservative",
    "--flash-anchor-min-score-advantage",
    "0.3",
)


def build_args(parsed: argparse.Namespace) -> list[str]:
    results_root = (
        parsed.results_root
        if parsed.results_root is not None
        else DEFAULT_RESULTS_BASE / f"{parsed.profile}_full2022_2026"
    )
    args = [
        "--manifest",
        str(parsed.manifest),
        "--results-root",
        str(results_root),
        "--years",
        str(parsed.years),
    ]
    if parsed.max_bars is not None:
        args += ["--max-bars", str(int(parsed.max_bars))]
    if not parsed.keep_audit_events:
        args += ["--disable-flash-audit-events", "--disable-shadow-audit-events"]
    if not parsed.keep_step_results:
        args += ["--disable-step-result-retention"]
    if not parsed.full_causal_entry_log:
        args += ["--compact-causal-entry-selected-only"]
    if parsed.profile == "best-compatible":
        args.extend(FLASH_1_1_BEST_COMPATIBLE_ARGS)
    elif parsed.profile in FLASH_1_1_ABLATION_ARGS:
        args.extend(FLASH_1_1_BEST_COMPATIBLE_ARGS)
        args.extend(FLASH_1_1_ABLATION_ARGS[parsed.profile])
    else:
        args.extend(FLASH_1_1_REPORT_CORE_ARGS)
    if parsed.profile == "experimental":
        args.extend(FLASH_1_1_EXPERIMENTAL_PROFIT_ARGS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--years", default=DEFAULT_YEARS)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument(
        "--profile",
        choices=(
            "best-compatible",
            "handoff-age12",
            "symbol-lookback504",
            "signal-cooldown480",
            "report-core",
            "experimental",
        ),
        default="best-compatible",
        help=(
            "best-compatible keeps the proven full-window risk_12 Flash profile; "
            "handoff-age12, symbol-lookback504, and signal-cooldown480 run "
            "single-parameter ablations; "
            "report-core applies the report's degradation/handoff changes; "
            "experimental also applies unproven profit-expansion ideas."
        ),
    )
    parser.add_argument("--keep-audit-events", action="store_true")
    parser.add_argument("--keep-step-results", action="store_true")
    parser.add_argument("--full-causal-entry-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(argv)

    args = build_args(parsed)
    if parsed.dry_run:
        print(json.dumps(args, indent=2))
        return 0
    return candidate.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
