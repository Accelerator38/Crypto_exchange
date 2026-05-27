from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_flash_dynamic_risk_sizing_sweep import (  # noqa: E402
    _metrics_from_output_dir,
    _parse_output_dir,
    _read_rows,
    _write_rows,
)


PRETRADE_RISK_VARIANTS = (
    {
        "name": "baseline_control",
        "args": (),
    },
    {
        "name": "funding_w1_cap025",
        "args": (
            "--flash-funding-risk-mult-weight",
            "1.0",
            "--flash-funding-risk-mult-cap",
            "0.25",
        ),
    },
    {
        "name": "funding_w2_cap025",
        "args": (
            "--flash-funding-risk-mult-weight",
            "2.0",
            "--flash-funding-risk-mult-cap",
            "0.25",
        ),
    },
    {
        "name": "vol_downonly_t2_max100",
        "args": (
            "--enable-flash-volatility-risk-sizing",
            "--flash-volatility-risk-target-pct",
            "2.0",
            "--flash-volatility-risk-min-volatility-pct",
            "0.5",
            "--flash-volatility-risk-max-mult",
            "1.0",
        ),
    },
    {
        "name": "vol_downonly_t3_max100",
        "args": (
            "--enable-flash-volatility-risk-sizing",
            "--flash-volatility-risk-target-pct",
            "3.0",
            "--flash-volatility-risk-min-volatility-pct",
            "0.5",
            "--flash-volatility-risk-max-mult",
            "1.0",
        ),
    },
    {
        "name": "vol_balanced_t2_max125",
        "args": (
            "--enable-flash-volatility-risk-sizing",
            "--flash-volatility-risk-target-pct",
            "2.0",
            "--flash-volatility-risk-min-volatility-pct",
            "0.5",
            "--flash-volatility-risk-max-mult",
            "1.25",
        ),
    },
    {
        "name": "funding_w1_vol_downonly_t2",
        "args": (
            "--flash-funding-risk-mult-weight",
            "1.0",
            "--flash-funding-risk-mult-cap",
            "0.25",
            "--enable-flash-volatility-risk-sizing",
            "--flash-volatility-risk-target-pct",
            "2.0",
            "--flash-volatility-risk-min-volatility-pct",
            "0.5",
            "--flash-volatility-risk-max-mult",
            "1.0",
        ),
    },
    {
        "name": "overextension_z2",
        "args": (
            "--enable-flash-overextension-guard",
            "--enable-flash-overextension-volatility-normalized",
            "--flash-short-overextension-z-floor",
            "-2.0",
            "--flash-long-overextension-z-ceiling",
            "2.0",
        ),
    },
    {
        "name": "overextension_z3",
        "args": (
            "--enable-flash-overextension-guard",
            "--enable-flash-overextension-volatility-normalized",
            "--flash-short-overextension-z-floor",
            "-3.0",
            "--flash-long-overextension-z-ceiling",
            "3.0",
        ),
    },
    {
        "name": "funding_w1_overextension_z3",
        "args": (
            "--flash-funding-risk-mult-weight",
            "1.0",
            "--flash-funding-risk-mult-cap",
            "0.25",
            "--enable-flash-overextension-guard",
            "--enable-flash-overextension-volatility-normalized",
            "--flash-short-overextension-z-floor",
            "-3.0",
            "--flash-long-overextension-z-ceiling",
            "3.0",
        ),
    },
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--years", default="2026")
    parser.add_argument("--max-bars", type=int, default=0)
    parser.add_argument("--risk-capital-fraction", type=float, default=0.10)
    parser.add_argument("--risk-max-open-positions", type=int, default=8)
    parser.add_argument("--stale-exit-age-bars", type=int, default=168)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only-variant", action="append", default=[])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)

    selected_names = {str(name) for name in args.only_variant}
    variants = [
        variant
        for variant in PRETRADE_RISK_VARIANTS
        if not selected_names or str(variant["name"]) in selected_names
    ]
    if selected_names and len(variants) != len(selected_names):
        known = ", ".join(str(variant["name"]) for variant in PRETRADE_RISK_VARIANTS)
        raise SystemExit(f"unknown --only-variant value; known variants: {known}")

    report_csv = Path(args.report_csv)
    rows = _read_rows(report_csv) if args.append and report_csv.exists() else []
    for variant in variants:
        name = str(variant["name"])
        variant_root = Path(args.results_root) / name
        command = [
            args.python,
            str(ROOT / "tools" / "run_flash_selected_subset_candidate.py"),
            "--manifest",
            str(args.manifest),
            "--results-root",
            str(variant_root),
            "--years",
            str(args.years),
            "--risk-capital-fraction",
            f"{float(args.risk_capital_fraction):.4g}",
            "--stride-minutes",
            "60",
            "--max-new-opens-per-bar",
            "1",
            "--risk-max-open-positions",
            str(int(args.risk_max_open_positions)),
            "--stale-exit-age-bars",
            str(int(args.stale_exit_age_bars)),
            "--disable-flash-audit-events",
            "--disable-shadow-audit-events",
            "--disable-step-result-retention",
            "--compact-causal-entry-selected-only",
            *variant["args"],
        ]
        if int(args.max_bars) > 0:
            insert_at = command.index("--risk-capital-fraction")
            command[insert_at:insert_at] = ["--max-bars", str(int(args.max_bars))]
        print(f"running {name}", flush=True)
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        variant_root.mkdir(parents=True, exist_ok=True)
        (variant_root / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (variant_root / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        output_dir = _parse_output_dir(proc.stdout)
        metrics = _metrics_from_output_dir(output_dir) if output_dir else {}
        row = {
            "name": name,
            "risk": float(args.risk_capital_fraction),
            "returncode": proc.returncode,
            "output_dir": output_dir,
            "extra_args": " ".join(variant["args"]),
            **metrics,
        }
        if proc.returncode != 0:
            row["stderr_tail"] = "\n".join((proc.stderr or "").splitlines()[-8:])
        rows.append(row)
        _write_rows(report_csv, rows)
        print(f"finished {name} returncode={proc.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
