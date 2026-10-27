from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np

from .rolling import RollingProtocol


EXPECTED_VARIANTS = (
    ("expanding_uniform", "expanding", None, "uniform", None),
    ("rolling_730_uniform", "rolling", 730, "uniform", None),
    ("rolling_730_hl180", "rolling", 730, "half_life", 180),
    ("rolling_365_hl90", "rolling", 365, "half_life", 90),
    ("rolling_180_hl30", "rolling", 180, "half_life", 30),
)


@dataclass(frozen=True)
class ExperimentMatrix:
    source_path: Path
    payload: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True)
class ExperimentCell:
    stage: str
    variant_id: str
    training_mode: str
    training_window_days: int | None
    weighting: str
    half_life_days: int | None
    retrain_cadence_days: int
    reuse_weekly_result: bool = False

    @property
    def cell_id(self) -> str:
        return f"{self.stage}__{self.variant_id}__cadence_{self.retrain_cadence_days}d"


def load_experiment_matrix(path: Path) -> ExperimentMatrix:
    source = path.resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != "exia.genetic_primus.experiment_matrix/1":
        raise ValueError("unexpected Genetic Primus experiment matrix schema")
    if payload.get("matrix_id") != "genetic_primus_limited_matrix_v1":
        raise ValueError("unexpected Genetic Primus experiment matrix id")
    stage_1 = payload.get("stage_1", {})
    variants = tuple(
        (
            item.get("id"),
            item.get("training_mode"),
            item.get("training_window_days"),
            item.get("weighting"),
            item.get("half_life_days"),
        )
        for item in stage_1.get("variants", [])
    )
    if int(stage_1.get("retrain_cadence_days", 0)) != 7 or variants != EXPECTED_VARIANTS:
        raise ValueError("weekly experiment stage is not sealed")
    stage_2 = payload.get("stage_2", {})
    if stage_2 != {
        "eligible_variants": 2,
        "retrain_cadence_days": [1, 3, 7, 30],
        "reuse_stage_1_cadence": True,
        "cell_summary": {
            "audit_paired_lcb_bps": "minimum_across_promotable_origins",
            "positive_outer_slice_fraction": "mean_across_promotable_origins",
            "stress_cost_lcb_bps": "minimum_across_promotable_origins",
        },
        "ranking": [
            "audit_paired_lcb_bps",
            "positive_outer_slice_fraction",
            "stress_cost_lcb_bps",
            "variant_id",
        ],
    }:
        raise ValueError("cadence experiment stage is not sealed")
    safety = payload.get("safety", {})
    expected_safety = {
        "credentials_allowed",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    }
    if set(safety) != expected_safety or any(safety[key] is not False for key in expected_safety):
        raise PermissionError("experiment matrix must remain fail-closed")
    return ExperimentMatrix(source, payload, hashlib.sha256(raw).hexdigest())


def weekly_cells(matrix: ExperimentMatrix) -> tuple[ExperimentCell, ...]:
    cadence = int(matrix.payload["stage_1"]["retrain_cadence_days"])
    return tuple(
        ExperimentCell(
            stage="weekly",
            variant_id=item["id"],
            training_mode=item["training_mode"],
            training_window_days=item["training_window_days"],
            weighting=item["weighting"],
            half_life_days=item["half_life_days"],
            retrain_cadence_days=cadence,
        )
        for item in matrix.payload["stage_1"]["variants"]
    )


def select_top_two_weekly(matrix: ExperimentMatrix, results: pd.DataFrame) -> tuple[str, str]:
    required = {
        "variant_id",
        "audit_paired_lcb_bps",
        "positive_outer_slice_fraction",
        "stress_cost_lcb_bps",
    }
    if not required <= set(results.columns):
        raise ValueError("weekly results do not contain preregistered ranking columns")
    expected_ids = {item[0] for item in EXPECTED_VARIANTS}
    if len(results) != len(expected_ids) or set(results["variant_id"]) != expected_ids:
        raise ValueError("top-two selection requires exactly five completed weekly variants")
    if results.loc[:, list(required - {"variant_id"})].isna().any().any():
        raise ValueError("weekly ranking metrics cannot be missing")
    ranked = results.sort_values(
        [
            "audit_paired_lcb_bps",
            "positive_outer_slice_fraction",
            "stress_cost_lcb_bps",
            "variant_id",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return tuple(ranked["variant_id"].iloc[:2])  # type: ignore[return-value]


def cadence_cells(
    matrix: ExperimentMatrix,
    weekly_results: pd.DataFrame,
) -> tuple[ExperimentCell, ...]:
    selected = select_top_two_weekly(matrix, weekly_results)
    variants = {cell.variant_id: cell for cell in weekly_cells(matrix)}
    return tuple(
        ExperimentCell(
            stage="cadence",
            variant_id=variant_id,
            training_mode=variants[variant_id].training_mode,
            training_window_days=variants[variant_id].training_window_days,
            weighting=variants[variant_id].weighting,
            half_life_days=variants[variant_id].half_life_days,
            retrain_cadence_days=int(cadence),
            reuse_weekly_result=int(cadence) == 7,
        )
        for variant_id in selected
        for cadence in matrix.payload["stage_2"]["retrain_cadence_days"]
    )


def summarize_experiment_cell(cell: ExperimentCell, output_dir: Path) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for path in sorted(output_dir.glob("origin_*/promotion_gate.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        if "paired_audit_lcb_bps" in metrics:
            rows.append(metrics)
    if not rows:
        raise ValueError("experiment cell has no incumbent-comparable promotion origins")
    return {
        "cell_id": cell.cell_id,
        "variant_id": cell.variant_id,
        "retrain_cadence_days": cell.retrain_cadence_days,
        "comparable_origins": len(rows),
        "audit_paired_lcb_bps": float(min(row["paired_audit_lcb_bps"] for row in rows)),
        "positive_outer_slice_fraction": float(
            np.mean([row["positive_outer_slice_fraction"] for row in rows])
        ),
        "stress_cost_lcb_bps": float(min(row["stress_cost_lcb_bps"] for row in rows)),
        "promotion_pass_fraction": float(
            np.mean([bool(row["promotion_gate_passed"]) for row in rows])
        ),
    }


def derive_rolling_protocol(
    base: RollingProtocol,
    matrix: ExperimentMatrix,
    cell: ExperimentCell,
) -> RollingProtocol:
    payload = copy.deepcopy(dict(base.payload))
    payload["protocol_id"] = f"{base.payload['protocol_id']}::{cell.cell_id}"
    payload["experiment_matrix_sha256"] = matrix.sha256
    payload["experiment_cell_id"] = cell.cell_id
    payload["training_window_days"] = cell.training_window_days
    payload["retrain_cadence_days"] = cell.retrain_cadence_days
    payload["evaluation_interval_days"] = cell.retrain_cadence_days
    recency = dict(payload["recency_fitness"])
    if cell.weighting == "uniform":
        recency["half_life_days"] = [180]
        recency["uniform_anchor"] = 1.0
        recency["minimum_ess_days"] = {"180": 180.0}
    else:
        assert cell.half_life_days is not None
        half_life = int(cell.half_life_days)
        recency["half_life_days"] = [half_life]
        recency["uniform_anchor"] = 0.2
        recency["minimum_ess_days"] = {
            str(half_life): float(base.payload["recency_fitness"]["minimum_ess_days"][str(half_life)])
        }
    payload["recency_fitness"] = recency
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return RollingProtocol(matrix.source_path, payload, hashlib.sha256(encoded).hexdigest())
