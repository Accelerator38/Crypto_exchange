from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_gate_progress(generations: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    generation = generations["generation"]
    axes[0].step(
        generation,
        generations["slice_failure_count"],
        where="mid",
        label="Failed supported slices",
        color="#b42318",
    )
    axes[0].step(
        generation,
        generations["overall_failure_count"],
        where="mid",
        label="Failed overall gates",
        color="#d97706",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("gate count")
    axes[0].set_title("V2 lexicographic gate progress")
    axes[0].legend()

    axes[1].plot(
        generation,
        generations["worst_supported_slice_lcb_bps"],
        label="Worst supported slice LCB",
        color="#b42318",
    )
    axes[1].plot(
        generation,
        generations["champion_lcb_bps"],
        label="Overall LCB",
        color="#147d64",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("costed bps per closed trade")
    axes[1].set_title("Overall versus weakest-slice evidence")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Generation")
    figure.savefig(path, dpi=170)
    plt.close(figure)
