from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.flash_partial_profit_lock_sweep import (  # noqa: E402
    build_partial_profit_lock_ab_variants,
    build_partial_profit_lock_command,
    write_partial_profit_lock_ab_plan,
)
from panteon_v2.analysis.flash_profitability_gates import load_flash_run_metrics  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--years", default="2026")
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-flash-audit-events", action="store_true")
    parser.add_argument("--disable-shadow-audit-events", action="store_true")
    parser.add_argument("--disable-step-result-retention", action="store_true")
    args = parser.parse_args(argv)

    candidate_runner = ROOT / "tools" / "run_flash_selected_subset_candidate.py"
    variants = build_partial_profit_lock_ab_variants()
    if args.dry_run:
        path = write_partial_profit_lock_ab_plan(
            output_csv=args.report_csv,
            python=args.python,
            candidate_runner=candidate_runner,
            manifest=args.manifest,
            results_root=args.results_root,
            years=args.years,
            max_bars=args.max_bars,
            variants=variants,
        )
        print(path)
        return 0

    rows: list[dict[str, Any]] = []
    for variant in variants:
        command = build_partial_profit_lock_command(
            python=args.python,
            candidate_runner=candidate_runner,
            manifest=args.manifest,
            results_root=args.results_root,
            years=args.years,
            max_bars=args.max_bars,
            variant=variant,
            disable_flash_audit_events=args.disable_flash_audit_events,
            disable_shadow_audit_events=args.disable_shadow_audit_events,
            disable_step_result_retention=args.disable_step_result_retention,
        )
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        output_dir = _parse_output_dir(proc.stdout, proc.stderr)
        metrics = load_flash_run_metrics(output_dir) if output_dir else {}
        row = {
            **dict(variant),
            "returncode": proc.returncode,
            "output_dir": str(output_dir or ""),
            **metrics,
        }
        if proc.returncode != 0:
            row["stderr_tail"] = "\n".join((proc.stderr or "").splitlines()[-5:])
        rows.append(row)

    out = Path(args.report_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


def _parse_output_dir(stdout: str, stderr: str) -> str:
    combined = "\n".join([stdout or "", stderr or ""])
    for pattern in (r"^output_dir=(.+)$", r'"output_dir"\s*:\s*"([^"]+)"'):
        match = re.search(pattern, combined, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
