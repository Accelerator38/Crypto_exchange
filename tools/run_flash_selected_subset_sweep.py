from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


SWEEP_VARIANTS = (
    {"name": "risk_8", "risk": 0.08, "max_positions": 8, "stale": 168},
    {"name": "risk_10", "risk": 0.10, "max_positions": 8, "stale": 168},
    {"name": "risk_12", "risk": 0.12, "max_positions": 8, "stale": 168},
    {"name": "maxpos_6", "risk": 0.10, "max_positions": 6, "stale": 168},
    {"name": "maxpos_10", "risk": 0.10, "max_positions": 10, "stale": 168},
    {"name": "stale_120", "risk": 0.10, "max_positions": 8, "stale": 120},
    {"name": "stale_240", "risk": 0.10, "max_positions": 8, "stale": 240},
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--years", default="2026")
    parser.add_argument("--max-bars", type=int, default=3600)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    rows: list[dict[str, object]] = []
    for variant in SWEEP_VARIANTS:
        variant_root = Path(args.results_root) / str(variant["name"])
        command = [
            args.python,
            str(ROOT / "tools" / "run_flash_selected_subset_candidate.py"),
            "--manifest",
            str(args.manifest),
            "--results-root",
            str(variant_root),
            "--years",
            str(args.years),
            "--max-bars",
            str(int(args.max_bars)),
            "--risk-capital-fraction",
            f"{float(variant['risk']):.4g}",
            "--stride-minutes",
            "60",
            "--max-new-opens-per-bar",
            "1",
            "--risk-max-open-positions",
            str(int(variant["max_positions"])),
            "--stale-exit-age-bars",
            str(int(variant["stale"])),
        ]
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        output_dir = _parse_output_dir(proc.stdout)
        metrics = _metrics_from_output_dir(output_dir) if output_dir else {}
        row = {
            "name": variant["name"],
            "risk": variant["risk"],
            "max_positions": variant["max_positions"],
            "stale": variant["stale"],
            "returncode": proc.returncode,
            "output_dir": output_dir,
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


def _parse_output_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        match = re.match(r"output_dir=(.+)", line.strip())
        if match:
            return match.group(1)
    return ""


def _metrics_from_output_dir(raw: str) -> dict[str, object]:
    output_dir = Path(raw)
    attribution = output_dir / "flash_attribution_summary.json"
    if not attribution.exists():
        return {}
    data = json.loads(attribution.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    return {
        "realized_pnl_usd": float(summary.get("realized_pnl_usd", 0.0) or 0.0),
        "closed_trades": int(summary.get("closed_trades", 0) or 0),
        "selected_signals": int(summary.get("selected_signals", 0) or 0),
        "filled_signals": int(summary.get("filled_signals", 0) or 0),
        "winning_trades": int(summary.get("winning_trades", 0) or 0),
        "losing_trades": int(summary.get("losing_trades", 0) or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
