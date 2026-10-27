from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREEN = ROOT / "Reports/Exia/Genetic_Primus/v4_architecture_screen_20260824"
DEFAULT_FINAL = ROOT / "Reports/Exia/Genetic_Primus/v4_final_20260824"


def _architecture_chart(screen_dir: Path) -> Path:
    summary = pd.read_parquet(screen_dir / "architecture_summary.parquet").sort_values("parameter_count")
    labels = summary["architecture_id"].tolist()
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(x - width, summary["minimum_worst_fold_mean_bps"], width, label="Worst seed/fold mean")
    axis.bar(x, summary["median_fold_lcb_bps"], width, label="Median fold LCB")
    axis.bar(x + width, summary["median_overall_lcb_bps"], width, label="Median overall LCB")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("bps per portfolio day")
    axis.set_title("Genetic_Primus V4 architecture screen")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    output = screen_dir / "architecture_comparison.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _fold_chart(final_dir: Path) -> Path:
    base = pd.read_parquet(final_dir / "strict_base16_folds.parquet")
    stress = pd.read_parquet(final_dir / "strict_stress24_folds.parquet")
    labels = base["fold"].tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(x, base["mean_daily_bps"], marker="o", label="Mean, 16 bps")
    axes[0].plot(x, base["lcb_daily_bps"], marker="o", label="LCB, 16 bps")
    axes[0].plot(x, stress["mean_daily_bps"], marker="s", label="Mean, 24 bps")
    axes[0].plot(x, stress["lcb_daily_bps"], marker="s", label="LCB, 24 bps")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("bps per portfolio day")
    axes[0].set_title("Genetic_Primus V4 temporal robustness")
    axes[0].legend(ncol=2)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - 0.18, base["trades"], 0.36, label="Entries, 16 bps")
    axes[1].bar(x + 0.18, stress["trades"], 0.36, label="Entries, 24 bps")
    axes[1].set_ylabel("new trades")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    output = final_dir / "fold_robustness.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def run(screen_dir: Path, final_dir: Path) -> dict[str, object]:
    architecture_chart = _architecture_chart(screen_dir)
    fold_chart = _fold_chart(final_dir)
    base = pd.read_parquet(final_dir / "strict_base16_folds.parquet")
    stress = pd.read_parquet(final_dir / "strict_stress24_folds.parquet")
    zero_base = base.loc[base["trades"] <= 0, "fold"].astype(str).tolist()
    zero_stress = stress.loc[stress["trades"] <= 0, "fold"].astype(str).tolist()
    passed = not zero_base and not zero_stress
    payload: dict[str, object] = {
        "schema_version": "exia.genetic_primus_post_freeze_audit.v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "FOLD_INCIDENCE_PASSED" if passed else "ADDITIONAL_REJECTION_ZERO_ENTRY_FOLDS",
        "minimum_required_entries_per_fold": 1,
        "zero_entry_folds_base16": zero_base,
        "zero_entry_folds_stress24": zero_stress,
        "validation_allowed": False,
        "oos_allowed": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "architecture_chart": architecture_chart.name,
        "fold_chart": fold_chart.name,
    }
    (final_dir / "post_freeze_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Genetic_Primus V4 post-freeze audit",
        "",
        f"Status: `{payload['status']}`.",
        "",
        "The strict run was already rejected by return, LCB and drawdown gates.",
        "This audit adds an independent incidence failure: 2024_H1 and 2024_H2",
        "contain mark-to-market returns from positions entered earlier, but no new entries.",
        "Therefore those folds are not independent evidence of entry generalization.",
        "",
        "Validation 2025, OOS 2026, paper and live remain closed.",
    ]
    (final_dir / "post_freeze_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Genetic_Primus V4 visual and fold-incidence audit")
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    args = parser.parse_args()
    print(json.dumps(run(args.screen_dir.resolve(), args.final_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
