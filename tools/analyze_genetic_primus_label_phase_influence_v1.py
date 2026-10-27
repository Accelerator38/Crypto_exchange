"""Post-replay exploratory diagnostics of frozen fits; zero refits or selection."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_genetic_primus_label_phase_v1 as r


def tail_share(values, count):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(values < 0) or not np.isfinite(values).all() or count < 1:
        raise ValueError("nonnegative finite diagnostic masses required")
    return float(np.sort(values)[-count:].sum() / values.sum()) if values.sum() else 0.


def utc(t):
    return datetime.fromtimestamp(int(t) / 1000, timezone.utc).strftime("%Y-%m-%d")


def analyze():
    reg, _, _, samples, _, _ = r.load()
    source_pin = r.parent.pin(Path(__file__))
    r.write("influence_diagnostic_provenance.json", {
        "registration_sha256": reg["sha256"], "source_pin": source_pin,
        "source_path": r.parent.relative(Path(__file__)),
        "registered_after_financial_results": True, "exploratory_not_candidate_selection": True,
        "additional_model_fits": 0, "additional_replays": 0, "safety": r.c.SAFETY})
    rows = []
    for period, settings in reg["periods"].items():
        grid = r.c.Clocks(3, 3, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        for j, t in enumerate(grid):
            for policy in r.POLICIES:
                bank = samples[policy]
                train = r.c.old.split(bank, t)["train"]
                x, y = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
                model = r.model_item(period, policy, j)[0]["model"]
                scaler = model["scaler"]
                z = np.clip((x - scaler["mean"]) / scaler["scale"], -scaler["clip"], scaler["clip"])
                target = (y - scaler["target_mean"]) / scaler["target_scale"]
                design = np.column_stack((z, np.ones(len(z))))
                gram = design.T @ design / len(z) + np.diag([.01] * 6 + [0.])
                rhs = design.T @ target / len(z)
                parameters = np.asarray(model["parameters"])
                residual = float(np.max(abs(gram @ parameters - rhs)))
                if residual > 1e-10:
                    raise ValueError("stored fit does not satisfy frozen ridge normal equations")
                # A linear decomposition of stored coefficients with scaler and Gram held fixed.
                # This is not leave-one-out fitting or an estimate after deleting observations.
                parts = (design @ np.linalg.inv(gram).T) * (target / len(z))[:, None]
                reconstruction_error = float(np.max(abs(parts.sum(axis=0) - parameters)))
                if reconstruction_error > 1e-10:
                    raise ValueError("stored coefficient decomposition mismatch")
                grouped_parts = parts.reshape(len(train), len(reg["symbols"]), 7).sum(axis=1)
                masses = np.linalg.norm(grouped_parts, axis=1)
                square = (y.reshape(len(train), -1) ** 2).sum(axis=1)
                ordered = np.argsort(square)[-5:][::-1]
                raw_labels = y.reshape(len(train), -1)
                flat_max = int(np.argmax(abs(raw_labels)))
                di, si = np.unravel_index(flat_max, raw_labels.shape)
                rows.append({"period": period, "policy": policy, "fit_ms": t,
                    "regularized_design_condition": float(np.linalg.cond(gram)),
                    "normal_equation_residual": residual, "coefficient_reconstruction_error": reconstruction_error,
                    "feature_target_moment_bps_per_train_std": (z.T @ y / len(z) * 10000).tolist(),
                    "feature_correlation": np.corrcoef(z, rowvar=False).tolist(),
                    "training_clipped_cells": int(np.sum(abs((x - scaler["mean"]) / scaler["scale"]) > scaler["clip"])),
                    "label_square_top_origin_share": {str(n): tail_share(square, n) for n in (1, 3, 10)},
                    "coefficient_contribution_norm_top_origin_share": {str(n): tail_share(masses, n) for n in (1, 3, 10)},
                    "largest_absolute_label": {"origin_UTC": utc(bank["origins"][train[di]]),
                        "symbol": reg["symbols"][si], "return_bps": float(raw_labels[di, si] * 10000)},
                    "top_label_square_origins": [{"origin_UTC": utc(bank["origins"][train[i]]),
                         "share": float(square[i] / square.sum())} for i in ordered]})
    r.write("training_influence.json", {"rows": rows, "stored_normal_equations_verified": len(rows),
        "coefficient_decompositions_verified": len(rows), "additional_model_fits": 0, "additional_replays": 0,
        "fixed_scaler_gram_decomposition_not_deletion_refit": True, "source_pin": source_pin, "safety": r.c.SAFETY})
    print(json.dumps({"stored_fits_analyzed": len(rows), "additional_fits": 0, "additional_replays": 0}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    provenance = r.sealed(r.OUT / "influence_diagnostic_provenance.json")
    if r.parent.pin(Path(__file__)) != provenance["source_pin"]:
        raise ValueError("exploratory diagnostic source changed")
    influence = r.sealed(r.OUT / "training_influence.json")["rows"]
    diagnostic = r.sealed(r.OUT / "diagnostics.json")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    colors = ("#1768ac", "#e67e22", "#27ae60", "#a93226")
    x = np.arange(6)
    forecasts = diagnostic["predictions"]
    shared_values = [v["prediction_bps"] for v in forecasts if v["period"] == "report"]
    bins = np.linspace(min(shared_values), max(shared_values), 31)
    for i, (policy, color) in enumerate(zip(r.POLICIES, colors)):
        fit_rows = [v for v in influence if v["period"] == "report" and v["policy"] == policy]
        moments = np.mean([v["feature_target_moment_bps_per_train_std"] for v in fit_rows], axis=0)
        coeff = diagnostic["period_summary"]["report"][policy]["coefficient_mean_bps_per_train_std"]
        ax[0, 0].bar(x + (i - 1.5) * .19, moments, width=.19, color=color, label=policy)
        ax[0, 1].bar(x + (i - 1.5) * .19, coeff, width=.19, color=color)
        pred = [v for v in forecasts if v["period"] == "report" and v["policy"] == policy]
        ax[1, 0].hist([v["prediction_bps"] for v in pred], bins=bins, histtype="step", color=color, label=policy, lw=1.5)
        dates = sorted(set(v["origin_ms"] for v in pred))
        fractions = [np.mean([v["target_long"] for v in pred if v["origin_ms"] == t]) * 100 for t in dates]
        ax[1, 1].plot([datetime.fromtimestamp(t / 1000, timezone.utc) for t in dates], fractions, color=color, label=policy)
    names = ["EMA dist", "ret18", "eff18", "relative ret", "vol48", "breadth"]
    for panel in ax[0]:
        panel.set_xticks(x, names, rotation=20)
        panel.axhline(0, color="gray", lw=.8)
        panel.set_ylabel("bps per train standard deviation")
    ax[0, 0].set_title("Feature-target moment: average across report origins")
    ax[0, 1].set_title("Ridge coefficients: average across report origins")
    ax[1, 0].axvline(20, color="gray", linestyle="--", label="20 bps threshold")
    ax[1, 0].set_title("Forecasts on matched report decision/asset grid")
    ax[1, 0].set_xlabel("Forecast, bps")
    ax[1, 0].set_ylabel("Decision/asset observations (not independent)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_title("Long targets at each decision (not capital exposure)")
    ax[1, 1].set_ylabel("Long targets, % of 8 assets")
    ax[1, 1].set_xlabel("UTC decision origin")
    ax[1, 1].tick_params(axis="x", rotation=25)
    for panel in ax.flat:
        panel.grid(alpha=.2)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4,
               title="Exploratory frozen-model diagnostics | P0=A, P1/P2=label phases, B=daily labels")
    fig.tight_layout(rect=(0, 0, 1, .92))
    target = r.OUT / "diagnostic_structure.png"
    if target.exists():
        raise ValueError("diagnostic plot already exists")
    fig.savefig(target, dpi=130)
    plt.close(fig)
    print(json.dumps({"plot": target.name, "bytes": target.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("analyze", "plot"))
    args = parser.parse_args()
    globals()[args.step]()
