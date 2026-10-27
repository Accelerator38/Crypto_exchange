from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.experiment_matrix_v4 import (  # noqa: E402
    cadence_cells,
    derive_rolling_protocol,
    load_experiment_matrix,
    summarize_experiment_cell,
    weekly_cells,
)
from exia.genetic_primus.rolling import load_rolling_protocol  # noqa: E402


DEFAULT_MATRIX = ROOT / "configs/genetic_primus_experiment_matrix_v1.json"
DEFAULT_ROLLING = ROOT / "configs/genetic_primus_rolling_3d_v1.json"
DEFAULT_V4_CONFIG = ROOT / "configs/genetic_primus_v4.json"
DEFAULT_MEASUREMENT = ROOT / "configs/genetic_primus_measurement_3d_v1.json"
DEFAULT_PROMOTION = ROOT / "configs/genetic_primus_promotion_gate_v1.json"


def _rolling_runner():
    path = ROOT / "tools/run_genetic_primus_rolling_3d.py"
    spec = importlib.util.spec_from_file_location("genetic_primus_rolling_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed rolling runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one sealed offline Genetic Primus experiment cell.")
    parser.add_argument("--stage", choices=("weekly", "cadence"), required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-origins", type=int, required=True)
    parser.add_argument("--weekly-results", type=Path)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()

    matrix = load_experiment_matrix(DEFAULT_MATRIX)
    if args.stage == "weekly":
        cells = weekly_cells(matrix)
    else:
        if args.weekly_results is None:
            parser.error("--weekly-results is required for cadence stage")
        cells = cadence_cells(matrix, pd.read_parquet(args.weekly_results))
    matches = [cell for cell in cells if cell.cell_id == args.cell_id]
    if len(matches) != 1:
        parser.error("cell-id is not registered for the selected stage")
    cell = matches[0]
    if cell.reuse_weekly_result:
        parser.error("cadence 7d must reuse the completed weekly result")

    protocol = derive_rolling_protocol(load_rolling_protocol(DEFAULT_ROLLING), matrix, cell)
    manifest = _rolling_runner()(
        v4_config_path=DEFAULT_V4_CONFIG,
        measurement_path=DEFAULT_MEASUREMENT,
        rolling_path=DEFAULT_ROLLING,
        promotion_path=DEFAULT_PROMOTION,
        output_dir=args.output_dir,
        max_origins=args.max_origins,
        rolling_override=protocol,
        resume_partial=args.resume_partial,
    )
    summary = summarize_experiment_cell(cell, args.output_dir)
    summary["source_manifest_sha256"] = hashlib.sha256(
        (args.output_dir / "manifest.json").read_bytes()
    ).hexdigest()
    (args.output_dir / "cell_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
