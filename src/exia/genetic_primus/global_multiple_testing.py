from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .objective_registry import (
    deflated_sharpe_probability,
    load_immutable_objective_registry,
    probability_of_backtest_overfitting,
)


def build_global_multiple_testing_evidence(
    catalog_path: Path, *, workspace: Path
) -> dict[str, Any]:
    """Build hash-locked DSR/PBO evidence from exactly aligned offline outcomes."""
    root = workspace.resolve()
    catalog_source = catalog_path.resolve()
    catalog_raw = catalog_source.read_bytes()
    catalog = json.loads(catalog_raw)
    if catalog.get("schema_version") != "exia.genetic_primus.outcome_catalog/1":
        raise ValueError("unexpected outcome catalog schema")
    registry = load_immutable_objective_registry(
        root / str(catalog["objective_registry_path"]), workspace=root
    )
    if catalog.get("objective_registry_sha256") != registry["registry_sha256"]:
        raise ValueError("outcome catalog registry SHA mismatch")
    entries = catalog.get("outcomes", [])
    objective_ids = [str(entry["objective_id"]) for entry in entries]
    if len(objective_ids) < 2 or len(objective_ids) != len(set(objective_ids)):
        raise ValueError("outcome catalog requires at least two unique objectives")
    baseline_ids: set[str] = set()
    for baseline in catalog.get("pre_registered_baselines", []):
        source = (root / str(baseline["source_contract_path"])).resolve()
        source_raw = source.read_bytes()
        if hashlib.sha256(source_raw).hexdigest() != str(
            baseline["source_contract_sha256"]
        ):
            raise ValueError("pre-registered baseline contract SHA mismatch")
        value: Any = json.loads(source_raw)
        for key in str(baseline["contract_reference"]).split("."):
            value = value[key]
        if str(baseline["contract_value"]) not in value:
            raise ValueError("baseline is absent from its frozen contract reference")
        baseline_ids.add(str(baseline["objective_id"]))
    unknown = set(objective_ids).difference(
        set(registry["objective_ids"]).union(baseline_ids)
    )
    if unknown:
        raise ValueError(f"outcome catalog has unregistered objectives: {sorted(unknown)}")
    series: list[pd.Series] = []
    artifact_hashes: dict[str, str] = {}
    timestamp_sets: list[tuple[int, ...]] = []
    for entry in entries:
        artifact = (root / str(entry["artifact_path"])).resolve()
        if root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"invalid outcome artifact path: {artifact}")
        actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha != str(entry["artifact_sha256"]):
            raise ValueError(f"outcome artifact hash mismatch: {entry['objective_id']}")
        frame = pd.read_parquet(artifact)
        for column, value in entry.get("filters", {}).items():
            if column not in frame:
                raise ValueError(f"outcome filter column is absent: {column}")
            frame = frame.loc[frame[column].eq(value)]
        timestamp_column = str(catalog["timestamp_column"])
        return_column = str(catalog["return_column"])
        if timestamp_column not in frame or return_column not in frame:
            raise ValueError("outcome artifact is missing timestamp or return column")
        local = frame.loc[:, [timestamp_column, return_column]].copy()
        if local[timestamp_column].duplicated().any() or local.empty:
            raise ValueError(f"outcomes must have unique timestamps: {entry['objective_id']}")
        local = local.sort_values(timestamp_column, kind="stable")
        if not np.isfinite(local[return_column].to_numpy(float)).all():
            raise ValueError(f"outcomes contain non-finite returns: {entry['objective_id']}")
        timestamps = tuple(local[timestamp_column].astype("int64"))
        timestamp_sets.append(timestamps)
        series.append(
            pd.Series(
                local[return_column].to_numpy(float),
                index=np.asarray(timestamps, dtype=np.int64),
                name=str(entry["objective_id"]),
            )
        )
        artifact_hashes[str(entry["objective_id"])] = actual_sha
    if len(set(timestamp_sets)) != 1:
        raise ValueError("outcome objectives do not have identical timestamp sets")
    aligned = pd.concat(series, axis=1, join="inner")
    trial_means = aligned.mean(axis=0)
    trial_standard_deviations = aligned.std(axis=0, ddof=1)
    trial_sharpes = trial_means / trial_standard_deviations.replace(0.0, np.nan)
    zero_baseline = trial_standard_deviations.eq(0.0) & trial_means.eq(0.0)
    trial_sharpes.loc[zero_baseline] = 0.0
    if trial_sharpes.isna().any() or not np.isfinite(trial_sharpes.to_numpy(float)).all():
        raise ValueError("outcome objective has constant returns")
    candidate_id = str(catalog["candidate_objective_id"])
    if candidate_id not in aligned:
        raise ValueError("candidate objective is absent from outcome catalog")
    dsr = deflated_sharpe_probability(
        aligned[candidate_id].to_numpy(float),
        registered_trial_sharpes=trial_sharpes.to_numpy(float),
        registered_trial_count=int(registry["objective_count"]),
    )
    pbo = probability_of_backtest_overfitting(
        aligned.to_numpy(float), slice_count=int(catalog["cscv_slice_count"])
    )
    registered_ids = set(registry["objective_ids"])
    missing = sorted(registered_ids.difference(objective_ids))
    required_pbo_ids = {str(value) for value in catalog["required_pbo_objective_ids"]}
    allowed_pbo_ids = registered_ids.union(baseline_ids)
    if not required_pbo_ids or not required_pbo_ids.issubset(allowed_pbo_ids):
        raise ValueError("required PBO comparison set is empty or unregistered")
    complete_pbo_set = required_pbo_ids.issubset(objective_ids)
    sufficient_observations = len(aligned) >= int(catalog["minimum_observations"])
    sealed_candidate = candidate_id == str(catalog["required_candidate_objective_id"])
    sealed_role = catalog.get("evidence_role") == "sealed_outer"
    promotion_eligible = bool(
        complete_pbo_set and sufficient_observations and sealed_candidate and sealed_role
    )
    return {
        "schema_version": "exia.genetic_primus.global_multiple_testing/1",
        "status": "PROMOTION_EVIDENCE" if promotion_eligible else "DIAGNOSTIC_ONLY",
        "promotion_eligible": promotion_eligible,
        "objective_registry_sha256": registry["registry_sha256"],
        "outcome_catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "registered_objective_count": int(registry["objective_count"]),
        "materialized_objective_count": int(len(objective_ids)),
        "missing_objective_ids": missing,
        "required_pbo_objective_ids": sorted(required_pbo_ids),
        "complete_pbo_comparison_set": complete_pbo_set,
        "observation_count": int(len(aligned)),
        "minimum_observations": int(catalog["minimum_observations"]),
        "candidate_objective_id": candidate_id,
        "artifact_sha256": artifact_hashes,
        "trial_sharpes": {key: float(value) for key, value in trial_sharpes.items()},
        **dsr,
        **pbo,
        "failed_readiness_gates": [
            name
            for name, passed in (
                ("complete_pbo_comparison_set", complete_pbo_set),
                ("minimum_observations", sufficient_observations),
                ("sealed_candidate", sealed_candidate),
                ("sealed_outer_role", sealed_role),
            )
            if not passed
        ],
        "credentials_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
