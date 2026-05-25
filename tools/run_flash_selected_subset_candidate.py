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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_panteon_flash_profitability_matrix as matrix  # noqa: E402
from panteon_v2.analysis import retrodate_market_runner as runner  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--risk-capital-fraction", type=float, default=0.10)
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--max-new-opens-per-bar", type=int, default=1)
    parser.add_argument("--risk-max-open-positions", type=int, default=8)
    parser.add_argument("--stale-exit-age-bars", type=int, default=168)
    parser.add_argument("--experimental-flash-real-actors", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runner_args = [
        "--results-root",
        str(args.results_root),
        "--years",
        str(args.years),
        "--risk-capital-fraction",
        f"{float(args.risk_capital_fraction):.4g}",
        "--stride-minutes",
        str(int(args.stride_minutes)),
        "--max-new-opens-per-bar",
        str(int(args.max_new_opens_per_bar)),
        "--risk-max-open-positions",
        str(int(args.risk_max_open_positions)),
    ]
    if args.max_bars is not None:
        runner_args += ["--max-bars", str(int(args.max_bars))]

    profile_args = list(matrix.ROUND3_DENY8_ENTRY_REGIME_ARGS)
    profile_args = matrix._with_arg_value(
        profile_args,
        "--flash-stale-position-exit-max-age-bars",
        str(int(args.stale_exit_age_bars)),
    )
    if args.experimental_flash_real_actors:
        profile_args += [
            "--enable-experimental-flash-actors",
            "--allow-experimental-flash-real-actors",
        ]
    runner_args += profile_args

    for item in manifest.get("score_boosts", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or "").strip()
        boost = item.get("boost")
        if key and boost is not None:
            runner_args += ["--flash-selected-subset-score-boost", f"{key}={boost}"]
    for key in manifest.get("do_not_demote_signal_keys", []):
        text = str(key or "").strip()
        if text:
            runner_args += ["--flash-selected-subset-do-not-demote-signal-key", text]
    for item in manifest.get("risk_mult_overrides", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("signal_key") or "").strip()
        risk_mult = item.get("risk_mult")
        if key and risk_mult is not None:
            runner_args += ["--flash-selected-subset-risk-mult", f"{key}={risk_mult}"]
    runner_args += [
        "--flash-selected-subset-risk-min-mult",
        "0.75",
        "--flash-selected-subset-risk-max-mult",
        "1.15",
    ]
    return runner.main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
