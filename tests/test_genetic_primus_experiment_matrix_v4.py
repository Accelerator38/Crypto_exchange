from __future__ import annotations

from pathlib import Path

import pandas as pd

from exia.genetic_primus.experiment_matrix_v4 import (
    cadence_cells,
    derive_rolling_protocol,
    load_experiment_matrix,
    select_top_two_weekly,
    weekly_cells,
)
from exia.genetic_primus.rolling import build_rolling_origins, load_rolling_protocol


ROOT = Path(__file__).resolve().parents[1]


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": [
                "expanding_uniform",
                "rolling_730_uniform",
                "rolling_730_hl180",
                "rolling_365_hl90",
                "rolling_180_hl30",
            ],
            "audit_paired_lcb_bps": [0.1, 0.2, 0.5, 0.4, -0.1],
            "positive_outer_slice_fraction": [0.6, 0.7, 0.65, 0.8, 0.9],
            "stress_cost_lcb_bps": [0.0, 0.1, 0.3, 0.4, 0.5],
        }
    )


def test_weekly_matrix_is_exactly_the_five_preregistered_cells() -> None:
    matrix = load_experiment_matrix(ROOT / "configs/genetic_primus_experiment_matrix_v1.json")
    cells = weekly_cells(matrix)
    assert len(cells) == 5
    assert {cell.retrain_cadence_days for cell in cells} == {7}
    assert [cell.variant_id for cell in cells] == _results()["variant_id"].tolist()


def test_only_top_two_weekly_variants_expand_to_cadence_stage() -> None:
    matrix = load_experiment_matrix(ROOT / "configs/genetic_primus_experiment_matrix_v1.json")
    assert select_top_two_weekly(matrix, _results()) == (
        "rolling_730_hl180",
        "rolling_365_hl90",
    )
    cells = cadence_cells(matrix, _results())
    assert len(cells) == 8
    assert {cell.variant_id for cell in cells} == {
        "rolling_730_hl180",
        "rolling_365_hl90",
    }
    assert {cell.retrain_cadence_days for cell in cells} == {1, 3, 7, 30}
    assert sum(cell.reuse_weekly_result for cell in cells) == 2


def test_derived_protocol_applies_window_weighting_and_contiguous_cadence() -> None:
    matrix = load_experiment_matrix(ROOT / "configs/genetic_primus_experiment_matrix_v1.json")
    base = load_rolling_protocol(ROOT / "configs/genetic_primus_rolling_3d_v1.json")
    weekly = next(cell for cell in weekly_cells(matrix) if cell.variant_id == "rolling_180_hl30")
    derived = derive_rolling_protocol(base, matrix, weekly)
    origins = build_rolling_origins(derived)
    assert len(origins) == 52
    assert all(origin.train_end - origin.train_start == 180 * 86_400_000 for origin in origins)
    assert all(origin.deployment_end - origin.deployment_start == 7 * 86_400_000 for origin in origins)
    assert derived.payload["recency_fitness"]["half_life_days"] == [30]
    uniform = derive_rolling_protocol(base, matrix, weekly_cells(matrix)[0])
    assert uniform.payload["recency_fitness"]["uniform_anchor"] == 1.0
