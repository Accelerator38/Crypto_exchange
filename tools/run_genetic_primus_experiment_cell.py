from __future__ import annotations

import argparse
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
    weekly_cells,
)
from exia.genetic_primus.rolling import load_rolling_protocol  # noqa: E402
from run_genetic_primus_rolling_3d import run as run_rolling  # noqa: E402


DEFAULT_MATRIX = ROOT / "configs/genetic_primus_experiment_matrix_v1.json"
DEFAULT_V4 = ROOT / "configs/genetic_primus_v4.json"
DEFAULT_MEASUREMENT = ROOT / "configs/genetic_primus_measurement_3d_v1.json"
DEFAULT_ROLLING = ROOT / "configs/genetic_primus_rolling_3d_v1.json"
DEFAULT_PROMOTION = ROOT / "configs/genetic_primus_promotion_gate_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one preregistered Genetic Primus experiment cell")
    parser.add_argument("--stage", choices=("weekly", "cadence"), required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--weekly-results", type=Path)
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--v4-config", type=Path, default=DEFAULT_V4)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--rolling-config", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--promotion-config", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-origins", type=int, default=1)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()

    matrix = load_experiment_matrix(args.matrix_config)
    if args.stage == "weekly":
        cells = weekly_cells(matrix)
    else:
        if args.weekly_results is None:
            raise ValueError("cadence stage requires the completed five-row weekly results parquet")
        weekly_results = pd.read_parquet(args.weekly_results)
        cells = cadence_cells(matrix, weekly_results)
    matches = [cell for cell in cells if cell.cell_id == args.cell_id]
    if len(matches) != 1:
        raise ValueError("cell id is not eligible in the requested experiment stage")
    cell = matches[0]
    if cell.reuse_weekly_result:
        raise ValueError("cadence 7d must reuse the corresponding weekly result; rerun is forbidden")
    base = load_rolling_protocol(args.rolling_config)
    derived = derive_rolling_protocol(base, matrix, cell)
    result = run_rolling(
        v4_config_path=args.v4_config.resolve(),
        measurement_path=args.measurement_config.resolve(),
        rolling_path=args.rolling_config.resolve(),
        promotion_path=args.promotion_config.resolve(),
        output_dir=args.output_dir.resolve(),
        max_origins=args.max_origins,
        rolling_override=derived,
        resume_partial=args.resume_partial,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_cell_id": cell.cell_id,
                "completed_origins": result["completed_origins"],
                "orders_enabled": result["orders_enabled"],
            }
        )
    )


if __name__ == "__main__":
    main()
