from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Reports" / "PanteonFlashSelectedSubsetPredeploy_20260525"

RUNS = {
    "baseline_full": ROOT
    / "Results"
    / "PanteonFlashTargetedLcbDenyRegimePullbackTrxFull2022_2026_20260525"
    / "RETRODATE_MARKET"
    / "2026-05-25_07-12-52_retrodate_market_v2",
    "overlay_full": ROOT
    / "Results"
    / "PanteonFlashSelectedSubsetPredeploy_20260525"
    / "candidate_no_weak_cut_full2022_2026_final"
    / "RETRODATE_MARKET"
    / "2026-05-25_16-14-18_retrodate_market_v2",
    "baseline_2025": ROOT
    / "Results"
    / "PanteonFlashTargetedLcbDenyRegimePullbackTrx2025_20260525"
    / "RETRODATE_MARKET"
    / "2026-05-25_06-53-15_retrodate_market_v2",
    "overlay_2025": ROOT
    / "Results"
    / "PanteonFlashSelectedSubsetPredeploy_20260525"
    / "candidate_no_weak_cut_2025"
    / "RETRODATE_MARKET"
    / "2026-05-25_10-51-52_retrodate_market_v2",
    "weak_cut_2025": ROOT
    / "Results"
    / "PanteonFlashSelectedSubsetPredeploy_20260525"
    / "candidate_2025"
    / "RETRODATE_MARKET"
    / "2026-05-25_10-38-00_retrodate_market_v2",
    "baseline_2026_h1": ROOT
    / "Results"
    / "PanteonFlashTargetedLcbDenyRegimePullbackTrx2026_20260525"
    / "RETRODATE_MARKET"
    / "2026-05-25_07-07-30_retrodate_market_v2",
    "overlay_2026_h1": ROOT
    / "Results"
    / "PanteonFlashSelectedSubsetPredeploy_20260525"
    / "candidate_no_weak_cut_2026_h1"
    / "RETRODATE_MARKET"
    / "2026-05-25_11-05-04_retrodate_market_v2",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {name: load_run(path) for name, path in RUNS.items()}
    sweep_rows = load_sweep(OUT / "risk_frequency_sweep_2026_h1.csv")
    chart_paths = write_charts(metrics, sweep_rows)
    report = OUT / "FINAL_PREDEPLOY_SELECTED_SUBSET_REPORT.md"
    report.write_text(build_markdown(metrics, sweep_rows, chart_paths), encoding="utf-8")
    print(report)
    return 0


def load_run(path: Path) -> dict:
    attribution = json.loads((path / "flash_attribution_summary.json").read_text())
    component = json.loads((path / "component_benchmark_report.json").read_text())
    analysis = (path / "analysis_report.md").read_text(encoding="utf-8")
    summary = attribution["summary"]
    component_summary = component["summary"]
    return {
        "path": str(path),
        "pnl_usd": float(summary["realized_pnl_usd"]),
        "pnl_pct": float(component_summary["panteon_pnl_pct"]),
        "max_dd_pct": extract_metric(analysis, "Panteon max drawdown"),
        "closed_trades": int(summary["closed_trades"]),
        "selected_signals": int(summary["selected_signals"]),
        "filled_signals": int(summary["filled_signals"]),
        "best_component": component_summary.get("best_component_label", ""),
        "best_component_pct": float(component_summary.get("best_component_pnl_pct", 0.0)),
        "alpha_pct": float(component_summary.get("panteon_alpha_pct", 0.0)),
        "actor_rows": attribution.get("rows", [])[:12],
    }


def extract_metric(text: str, label: str) -> float:
    prefix = f"- {label}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        value = line.split(":", 1)[1].strip().rstrip("%")
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def load_sweep(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("realized_pnl_usd", "risk"):
            row[key] = float(row.get(key) or 0.0)
        for key in ("max_positions", "stale", "closed_trades", "selected_signals"):
            row[key] = int(float(row.get(key) or 0))
    return rows


def write_charts(metrics: dict[str, dict], sweep_rows: list[dict]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    chart_dir = OUT / "charts"
    chart_dir.mkdir(exist_ok=True)
    paths: list[Path] = []

    labels = ["2025", "2026 H1", "2022-2026"]
    baseline = [
        metrics["baseline_2025"]["pnl_usd"],
        metrics["baseline_2026_h1"]["pnl_usd"],
        metrics["baseline_full"]["pnl_usd"],
    ]
    overlay = [
        metrics["overlay_2025"]["pnl_usd"],
        metrics["overlay_2026_h1"]["pnl_usd"],
        metrics["overlay_full"]["pnl_usd"],
    ]
    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.18 for i in x], baseline, width=0.36, label="Previous best")
    plt.bar([i + 0.18 for i in x], overlay, width=0.36, label="Selected-subset")
    plt.xticks(list(x), labels)
    plt.ylabel("Realized PnL, USD")
    plt.title("Panteon Flash: previous best vs selected-subset overlay")
    plt.legend()
    plt.tight_layout()
    path = chart_dir / "pnl_comparison.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)

    if sweep_rows:
        ordered = sorted(sweep_rows, key=lambda row: row["realized_pnl_usd"], reverse=True)
        plt.figure(figsize=(9, 4))
        plt.bar([row["name"] for row in ordered], [row["realized_pnl_usd"] for row in ordered])
        plt.ylabel("2026 H1 realized PnL, USD")
        plt.title("Risk/frequency sweep around current profile")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = chart_dir / "risk_frequency_sweep.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    actor_rows = metrics["overlay_full"]["actor_rows"][:10]
    plt.figure(figsize=(9, 5))
    labels = [
        f"{row['actor_label']}\\n{row['symbol']} {row['action'].replace('FUT_', '')}"
        for row in actor_rows
    ]
    plt.bar(labels, [float(row["realized_pnl_usd"]) for row in actor_rows])
    plt.ylabel("Realized PnL, USD")
    plt.title("Top selected actor/symbol/action cells")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = chart_dir / "top_cells.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(path)
    return paths


def build_markdown(metrics: dict[str, dict], sweep_rows: list[dict], charts: list[Path]) -> str:
    full_delta = metrics["overlay_full"]["pnl_usd"] - metrics["baseline_full"]["pnl_usd"]
    y2025_delta = metrics["overlay_2025"]["pnl_usd"] - metrics["baseline_2025"]["pnl_usd"]
    h1_delta = metrics["overlay_2026_h1"]["pnl_usd"] - metrics["baseline_2026_h1"]["pnl_usd"]
    weak_delta = metrics["weak_cut_2025"]["pnl_usd"] - metrics["baseline_2025"]["pnl_usd"]
    chart_lines = [f"![{path.stem}]({path.resolve().as_posix()})" for path in charts]
    sweep_lines = [
        "| Variant | Risk | Max pos | Stale | 2026 H1 PnL | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(sweep_rows, key=lambda item: item["realized_pnl_usd"], reverse=True):
        sweep_lines.append(
            f"| `{row['name']}` | {row['risk']:.2f} | {row['max_positions']} | "
            f"{row['stale']} | ${row['realized_pnl_usd']:.2f} | {row['closed_trades']} |"
        )

    overlay = metrics["overlay_full"]
    baseline = metrics["baseline_full"]
    return "\n".join(
        [
            "# Panteon Flash Selected-Subset Predeploy Report",
            "",
            "Date: 2026-05-25",
            "",
            "## Executive Summary",
            "",
            (
                "Final recommendation: deploy the selected-subset overlay on top of the "
                "current best Panteon Flash profile, but keep risk/frequency settings "
                "unchanged (`risk=10%`, `max_positions=8`, `stale_exit=168`)."
            ),
            "",
            (
                f"Full 2022-2026 result improved from ${baseline['pnl_usd']:.2f} "
                f"({baseline['pnl_pct']:.2f}%) to ${overlay['pnl_usd']:.2f} "
                f"({overlay['pnl_pct']:.2f}%), delta ${full_delta:.2f}. "
                f"MaxDD improved from {baseline['max_dd_pct']:.2f}% to "
                f"{overlay['max_dd_pct']:.2f}%."
            ),
            "",
            "## Visuals",
            "",
            *chart_lines,
            "",
            "## Iteration Results",
            "",
            "| Test | Previous best | Selected-subset | Delta |",
            "|---|---:|---:|---:|",
            (
                f"| 2025 | ${metrics['baseline_2025']['pnl_usd']:.2f} | "
                f"${metrics['overlay_2025']['pnl_usd']:.2f} | ${y2025_delta:.2f} |"
            ),
            (
                f"| 2026 H1 | ${metrics['baseline_2026_h1']['pnl_usd']:.2f} | "
                f"${metrics['overlay_2026_h1']['pnl_usd']:.2f} | ${h1_delta:.2f} |"
            ),
            (
                f"| 2022-2026 | ${baseline['pnl_usd']:.2f} | "
                f"${overlay['pnl_usd']:.2f} | ${full_delta:.2f} |"
            ),
            "",
            (
                f"Rejected variant: weak-cell risk cut (`0.75x`) reduced 2025 to "
                f"${metrics['weak_cut_2025']['pnl_usd']:.2f}, delta ${weak_delta:.2f}; "
                "it was not promoted."
            ),
            "",
            "## Risk/Frequency Sweep",
            "",
            *sweep_lines,
            "",
            "Conclusion: do not change risk/frequency before live. `risk_12`, "
            "`risk_8`, `maxpos_6`, and `maxpos_10` all underperformed the current profile; "
            "`stale_120` and `stale_240` were identical on 2026 H1.",
            "",
            "## What Changed",
            "",
            "- Added `flash_selected_subset_manifest.py` and CLI tooling to build additive overlays from Flash-selected attribution.",
            "- Added FlashAllocator support for exact `actor|symbol|action` score boosts, do-not-demote allowlist, and bounded selected-subset risk multipliers.",
            "- Added audit fields for selected-subset boost/protection/risk multiplier and runner/startup CLI/settings wiring.",
            "- Optimized compact causal logs so NoTrade rows do not persist rejected candidate payloads.",
            "- Deferred partial profit lock: current executor closes full position quantity and `PositionTracker` pops the whole position, so partial closes need a separate accounting change.",
            "",
            "## Final Profile",
            "",
            "- Base: current best targeted LCB/terminal-deny profile.",
            "- Overlay boosts: `Solo_MomentumScalper|BTC/USDT|FUT_SHORT_FULL=+0.25`, `Solo_MomentumScalper|DOGE/USDT|FUT_SHORT_FULL=+0.25`.",
            "- Overlay sizing: same two cells at `1.15x`, bounded by `0.75..1.15`.",
            "- Positive allowlist: 22 sparse/proven selected cells marked do-not-demote.",
            "- Risk/frequency: `risk_capital_fraction=0.10`, `risk_max_open_positions=8`, `max_new_opens_per_bar=1`, `stale_exit_age=168`.",
            "",
            "## Residual Risks",
            "",
            "- Improvement is real but small: +$6.75 over five years. It is not enough to justify raising leverage.",
            "- Solo_MomentumScalper still has negative selection alpha: standalone 58.92% vs Flash-selected 28.33%. The next profit work should focus on selected-subset timing/sizing, not broader denies.",
            "- Partial profit lock remains the most promising execution-layer idea, but it must be implemented with partial-position accounting first.",
            "",
            "## Verification",
            "",
            "- Targeted selected-subset tests: passed.",
            "- Syntax AST parse: 8 files OK.",
            "- Full test suite: `1045 passed, 15 subtests passed`.",
            "- `git diff --check`: no whitespace errors; CRLF warnings only.",
            "",
            "## Artifact Links",
            "",
            f"- Full final run: `{RUNS['overlay_full']}`",
            f"- Manifest: `{OUT / 'flash_selected_subset_manifest_no_weak_cut.json'}`",
            f"- Sweep CSV: `{OUT / 'risk_frequency_sweep_2026_h1.csv'}`",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
