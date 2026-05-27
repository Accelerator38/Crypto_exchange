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


DYNAMIC_RISK_VARIANTS = (
    {
        "name": "actor_025_115_scale05",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "0.25",
            "--flash-actor-risk-max-mult",
            "1.15",
            "--flash-actor-risk-edge-scale-pct",
            "0.5",
        ),
    },
    {
        "name": "actor_050_125_scale10",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "0.50",
            "--flash-actor-risk-max-mult",
            "1.25",
            "--flash-actor-risk-edge-scale-pct",
            "1.0",
        ),
    },
    {
        "name": "actor_075_130_scale10",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "0.75",
            "--flash-actor-risk-max-mult",
            "1.30",
            "--flash-actor-risk-edge-scale-pct",
            "1.0",
        ),
    },
    {
        "name": "actor_up_only_100_125_scale05",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "1.00",
            "--flash-actor-risk-max-mult",
            "1.25",
            "--flash-actor-risk-edge-scale-pct",
            "0.5",
        ),
    },
    {
        "name": "actor_up_only_100_130_scale10",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "1.00",
            "--flash-actor-risk-max-mult",
            "1.30",
            "--flash-actor-risk-edge-scale-pct",
            "1.0",
        ),
    },
    {
        "name": "shadow_lcb_050_scale2",
        "args": (
            "--enable-flash-shadow-pnl-lcb-risk-sizing",
            "--flash-shadow-pnl-lcb-risk-min-mult",
            "0.50",
            "--flash-shadow-pnl-lcb-risk-floor-usd",
            "0.0",
            "--flash-shadow-pnl-lcb-risk-scale-usd",
            "2.0",
        ),
    },
    {
        "name": "shadow_lcb_025_scale5",
        "args": (
            "--enable-flash-shadow-pnl-lcb-risk-sizing",
            "--flash-shadow-pnl-lcb-risk-min-mult",
            "0.25",
            "--flash-shadow-pnl-lcb-risk-floor-usd",
            "0.0",
            "--flash-shadow-pnl-lcb-risk-scale-usd",
            "5.0",
        ),
    },
    {
        "name": "combo_actor050_125_shadow050_scale2",
        "args": (
            "--enable-flash-actor-risk-sizing",
            "--flash-actor-risk-min-mult",
            "0.50",
            "--flash-actor-risk-max-mult",
            "1.25",
            "--flash-actor-risk-edge-scale-pct",
            "1.0",
            "--enable-flash-shadow-pnl-lcb-risk-sizing",
            "--flash-shadow-pnl-lcb-risk-min-mult",
            "0.50",
            "--flash-shadow-pnl-lcb-risk-floor-usd",
            "0.0",
            "--flash-shadow-pnl-lcb-risk-scale-usd",
            "2.0",
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
        for variant in DYNAMIC_RISK_VARIANTS
        if not selected_names or str(variant["name"]) in selected_names
    ]
    if selected_names and len(variants) != len(selected_names):
        known = ", ".join(str(variant["name"]) for variant in DYNAMIC_RISK_VARIANTS)
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
            command[command.index("--risk-capital-fraction"):command.index("--risk-capital-fraction")] = [
                "--max-bars",
                str(int(args.max_bars)),
            ]
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


def _parse_output_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        match = re.match(r"output_dir=(.+)", line.strip())
        if match:
            return match.group(1)
    return ""


def _metrics_from_output_dir(raw: str) -> dict[str, object]:
    output_dir = Path(raw)
    attribution_path = output_dir / "flash_attribution_summary.json"
    final_report_path = output_dir / "final_session_report.json"
    metrics: dict[str, object] = {}
    if attribution_path.exists():
        attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
        summary = attribution.get("summary", {}) if isinstance(attribution, dict) else {}
        metrics.update({
            "pnl_usd": round(float(summary.get("realized_pnl_usd", 0.0) or 0.0), 2),
            "closed": int(summary.get("closed_trades", 0) or 0),
            "filled": int(summary.get("filled_signals", 0) or 0),
            "selected": int(summary.get("selected_signals", 0) or 0),
            "wins": int(summary.get("winning_trades", 0) or 0),
            "losses": int(summary.get("losing_trades", 0) or 0),
        })
    if final_report_path.exists():
        final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        pnl_usd = float(final_report.get("pnl_usd", metrics.get("pnl_usd", 0.0)) or 0.0)
        metrics["pnl_pct"] = round(pnl_usd / 1000.0 * 100.0, 2)
        metrics["max_dd_pct"] = round(
            float(final_report.get("panteon_max_drawdown_pct", 0.0) or 0.0),
            2,
        )
    return metrics


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
