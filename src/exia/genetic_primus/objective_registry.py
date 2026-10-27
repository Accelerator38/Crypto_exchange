from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np


def load_immutable_objective_registry(path: Path, *, workspace: Path) -> dict[str, Any]:
    """Validate a versioned, hash-locked objective registry and its local contracts."""
    raw = path.resolve().read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != "exia.genetic_primus.objective_registry/1":
        raise ValueError("unexpected objective registry schema")
    if payload.get("mutation_policy") != "append_via_new_registry_version_only":
        raise ValueError("objective registry is not immutable")
    entries = payload.get("objectives", [])
    ids = [str(entry["objective_id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate objective id")
    root = workspace.resolve()
    previous_ids: tuple[str, ...] = ()
    previous_count = 0
    previous_path = payload.get("previous_registry_path")
    if previous_path is not None:
        previous_source = (root / str(previous_path)).resolve()
        previous_raw = previous_source.read_bytes()
        if hashlib.sha256(previous_raw).hexdigest() != payload.get(
            "previous_registry_sha256"
        ):
            raise ValueError("previous objective registry hash mismatch")
        previous = load_immutable_objective_registry(previous_source, workspace=root)
        previous_ids = tuple(previous["objective_ids"])
        previous_count = int(previous["objective_count"])
        if set(ids).intersection(previous_ids):
            raise ValueError("objective id already exists in previous registry")
    for entry in entries:
        source = (root / str(entry["config_path"])).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError(f"invalid objective config path: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != str(entry["config_sha256"]):
            raise ValueError(f"objective config hash mismatch: {entry['objective_id']}")
    return {
        "payload": payload,
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "objective_count": previous_count + len(entries),
        "objective_ids": (*previous_ids, *ids),
    }


def deflated_sharpe_probability(
    returns: Sequence[float],
    *,
    registered_trial_sharpes: Sequence[float],
    registered_trial_count: int | None = None,
) -> dict[str, float]:
    """Bailey-Lopez de Prado DSR using the registered trial family."""
    values = np.asarray(returns, dtype=np.float64)
    trials = np.asarray(registered_trial_sharpes, dtype=np.float64)
    if len(values) < 3 or len(trials) < 2 or not np.isfinite(values).all():
        raise ValueError("DSR requires finite returns and at least two registered trials")
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation <= 0.0:
        raise ValueError("DSR requires non-constant returns")
    sharpe = float(values.mean() / standard_deviation)
    centered = values - values.mean()
    moment2 = float(np.mean(centered**2))
    skew = float(np.mean(centered**3) / moment2**1.5)
    kurtosis = float(np.mean(centered**4) / moment2**2)
    trial_scale = float(trials.std(ddof=1))
    if trial_scale <= 0.0:
        trial_scale = float(1.0 / np.sqrt(len(values)))
    count = len(trials) if registered_trial_count is None else int(registered_trial_count)
    if count < len(trials):
        raise ValueError("registered trial count cannot be smaller than observed trials")
    gamma = 0.5772156649015329
    normal = NormalDist()
    expected_maximum = trial_scale * (
        (1.0 - gamma) * normal.inv_cdf(1.0 - 1.0 / count)
        + gamma * normal.inv_cdf(1.0 - np.exp(-1.0) / count)
    )
    denominator = np.sqrt(
        max(1.0e-12, 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2)
    )
    statistic = float((sharpe - expected_maximum) * np.sqrt(len(values) - 1) / denominator)
    return {
        "observed_sharpe": sharpe,
        "expected_maximum_sharpe": float(expected_maximum),
        "deflated_sharpe_probability": float(normal.cdf(statistic)),
        "registered_trial_count": float(count),
    }


def probability_of_backtest_overfitting(
    returns_by_objective: np.ndarray, *, slice_count: int = 8
) -> dict[str, float]:
    """Deterministic CSCV estimate; columns must be registered objectives."""
    matrix = np.asarray(returns_by_objective, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or slice_count < 4 or slice_count % 2:
        raise ValueError("PBO requires >=2 objectives and an even slice_count >=4")
    if matrix.shape[0] < slice_count or not np.isfinite(matrix).all():
        raise ValueError("PBO requires enough finite observations")
    slices = np.array_split(np.arange(matrix.shape[0]), slice_count)
    logits: list[float] = []
    all_slices = set(range(slice_count))
    for training_ids in itertools.combinations(range(slice_count), slice_count // 2):
        testing_ids = sorted(all_slices.difference(training_ids))
        train = matrix[np.concatenate([slices[index] for index in training_ids])]
        test = matrix[np.concatenate([slices[index] for index in testing_ids])]
        train_score = train.mean(axis=0) / np.maximum(train.std(axis=0, ddof=1), 1.0e-12)
        selected = int(np.argmax(train_score))
        test_score = test.mean(axis=0) / np.maximum(test.std(axis=0, ddof=1), 1.0e-12)
        percentile = float((np.sum(test_score < test_score[selected]) + 1) / (matrix.shape[1] + 1))
        logits.append(float(np.log(percentile / (1.0 - percentile))))
    values = np.asarray(logits, dtype=np.float64)
    return {
        "pbo": float(np.mean(values <= 0.0)),
        "median_oos_rank_logit": float(np.median(values)),
        "cscv_combinations": float(len(values)),
        "registered_objective_count": float(matrix.shape[1]),
    }
