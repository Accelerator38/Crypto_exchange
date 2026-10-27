from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.experiment_matrix_v4 import (  # noqa: E402
    derive_rolling_protocol,
    load_experiment_matrix,
    select_top_two_weekly,
    weekly_cells,
)
from exia.genetic_primus.rolling import build_rolling_origins, load_rolling_protocol  # noqa: E402


MATRIX_PATH = ROOT / "configs/genetic_primus_experiment_matrix_v1.json"
ROLLING_PATH = ROOT / "configs/genetic_primus_rolling_3d_v1.json"
SAFETY_KEYS = (
    "credentials_allowed",
    "paper_allowed",
    "live_allowed",
    "orders_enabled",
    "promotion_authority",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed offline Stage 1 cells.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    matrix = load_experiment_matrix(MATRIX_PATH)
    base = load_rolling_protocol(ROLLING_PATH)
    cells = {cell.cell_id: cell for cell in weekly_cells(matrix)}
    summaries: list[dict[str, object]] = []
    benchmark_frames: list[pd.DataFrame] = []
    ensemble_frames: list[pd.DataFrame] = []
    promotion_rows: list[dict[str, object]] = []
    completed_by_cell: dict[str, int] = {}

    for cell_dir in args.cell_dir:
        manifest_path = cell_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cell_id = str(manifest.get("experiment_cell_id"))
        if cell_id not in cells:
            raise ValueError(f"unexpected Stage 1 cell: {cell_id}")
        if any(manifest.get(key) is not False for key in SAFETY_KEYS):
            raise PermissionError(f"unsafe manifest flags in {cell_dir}")
        for name, expected_sha in manifest["artifact_sha256"].items():
            if _sha256(cell_dir / f"{name}.parquet") != expected_sha:
                raise ValueError(f"artifact hash mismatch: {cell_dir}/{name}.parquet")
        summary = json.loads((cell_dir / "cell_summary.json").read_text(encoding="utf-8"))
        summary["source_manifest_sha256"] = _sha256(manifest_path)
        summaries.append(summary)
        completed_by_cell[cell_id] = int(manifest["completed_origins"])

        benchmarks = pd.read_parquet(cell_dir / "benchmarks.parquet")
        benchmarks.insert(0, "variant_id", cells[cell_id].variant_id)
        benchmark_frames.append(benchmarks)
        for origin_dir in sorted(cell_dir.glob("origin_*")):
            origin_index = int(origin_dir.name.split("_")[-1])
            ensemble = pd.read_parquet(origin_dir / "ensemble_evaluation.parquet")
            ensemble.insert(0, "origin_index", origin_index)
            ensemble.insert(0, "variant_id", cells[cell_id].variant_id)
            ensemble_frames.append(ensemble)
            gate_path = origin_dir / "promotion_gate.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                metrics = dict(gate.get("metrics", {}))
                metrics.update({"variant_id": cells[cell_id].variant_id, "origin_index": origin_index})
                promotion_rows.append(metrics)

    if set(completed_by_cell) != set(cells):
        raise ValueError("summary requires exactly the five registered Stage 1 cells")
    weekly = pd.DataFrame(summaries)
    top_two = select_top_two_weekly(matrix, weekly)
    weekly = weekly.sort_values(
        ["audit_paired_lcb_bps", "positive_outer_slice_fraction", "stress_cost_lcb_bps", "variant_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    benchmarks = pd.concat(benchmark_frames, ignore_index=True)
    baseline = (
        benchmarks.groupby(["variant_id", "agent_id"], observed=True)
        .agg(
            origins=("origin_index", "nunique"),
            total_net_bps=("net_bps", "sum"),
            mean_net_bps=("net_bps", "mean"),
            worst_origin_net_bps=("net_bps", "min"),
            worst_max_drawdown=("max_drawdown", "max"),
            total_trades=("trades", "sum"),
            total_turnover_fills=("turnover_fills", "sum"),
        )
        .reset_index()
    )
    ensemble = pd.concat(ensemble_frames, ignore_index=True)
    ensemble_summary = (
        ensemble.groupby(["variant_id", "ensemble_size"], observed=True)
        .agg(
            origins=("origin_index", "nunique"),
            selected_fraction=("selected_by_selection_bank", "mean"),
            mean_selection_fitness=("selection_fitness", "mean"),
            worst_selection_fitness=("selection_fitness", "min"),
            mean_audit_fitness=("audit_fitness", "mean"),
            worst_audit_fitness=("audit_fitness", "min"),
        )
        .reset_index()
    )
    expected_full = {
        cell.cell_id: len(build_rolling_origins(derive_rolling_protocol(base, matrix, cell)))
        for cell in cells.values()
    }
    full_stage1 = all(completed_by_cell[key] == expected_full[key] for key in cells)
    payload = {
        "schema_version": "exia.genetic_primus.stage1_summary/1",
        "matrix_sha256": matrix.sha256,
        "rolling_protocol_sha256": base.sha256,
        "completed_origins_by_cell": completed_by_cell,
        "expected_full_origins_by_cell": expected_full,
        "outer6_pilot_completed": all(value >= 6 for value in completed_by_cell.values()),
        "full_stage1_completed": full_stage1,
        "eligible_for_stage2": full_stage1,
        "provisional_top_two": list(top_two),
        "all_promotion_pass_fraction_zero": bool((weekly["promotion_pass_fraction"] == 0.0).all()),
        "all_audit_lcb_nonpositive": bool((weekly["audit_paired_lcb_bps"] <= 0.0).all()),
        "safety": {key: False for key in SAFETY_KEYS},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    weekly.to_parquet(args.output_dir / "weekly_results.parquet", index=False)
    baseline.to_parquet(args.output_dir / "baseline_results.parquet", index=False)
    ensemble_summary.to_parquet(args.output_dir / "ensemble_results.parquet", index=False)
    pd.DataFrame(promotion_rows).to_parquet(args.output_dir / "promotion_gates.parquet", index=False)
    artifact_names = (
        "weekly_results.parquet",
        "baseline_results.parquet",
        "ensemble_results.parquet",
        "promotion_gates.parquet",
    )
    payload["artifact_sha256"] = {
        name: _sha256(args.output_dir / name) for name in artifact_names
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
