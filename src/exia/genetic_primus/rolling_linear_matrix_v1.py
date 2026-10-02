"""Bounded two-/four-feature causal ridge policies for disclosed development."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .measurement_lab_v2 import build_lab_features, object_sha
from .rolling_linear_retrain_v1 import (
    DT, FEATURES, HORIZON_MS, VARIANTS, _iso, build_origin_examples,
)

TWO_FEATURES = FEATURES
FOUR_FEATURES = (*FEATURES, "ema_gap_over_atr14", "atr_14_pct")
FEATURE_SETS = {"two": TWO_FEATURES, "four": FOUR_FEATURES}


def build_matrix_examples(panel: pd.DataFrame, symbols: tuple[str, ...],
                          history_start_ms: int, development_start_ms: int,
                          end_ms: int) -> pd.DataFrame:
    """Join two extra features from the last fully closed 4h candle only."""
    examples = build_origin_examples(
        panel, symbols, history_start_ms, development_start_ms, end_ms)
    features = build_lab_features(panel.loc[panel.timestamp.lt(end_ms)])[
        ["timestamp", "symbol", "ema_gap_12_48", "atr_14_pct"]].copy()
    if features.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate feature candle")
    features = features.rename(columns={"timestamp": "feature_timestamp"})
    examples["feature_timestamp"] = examples.origin_start - DT
    examples = examples.merge(features, on=["feature_timestamp", "symbol"],
                              how="left", validate="one_to_one")
    volatility = examples.atr_14_pct.to_numpy(float)
    gap = examples.ema_gap_12_48.to_numpy(float)
    examples["ema_gap_over_atr14"] = gap / np.where(volatility > 0, volatility, np.nan)
    finite = np.isfinite(examples[list(FOUR_FEATURES)].to_numpy(float)).all(axis=1)
    if (~finite & examples.origin_start.ge(development_start_ms)).any():
        raise ValueError("nonfinite feature on development origin")
    return (examples.loc[finite].drop(columns=["feature_timestamp", "ema_gap_12_48"])
            .sort_values(["origin_start", "symbol"], kind="stable").reset_index(drop=True))


def fit_matrix_at_origin(examples: pd.DataFrame, deploy_start_ms: int,
                         feature_set: str, variant: str, *, purge_bars: int = 7,
                         ridge_alpha: float = 0.5,
                         round_trip_cost_bps: float = 16.0,
                         minimum_opportunities: int = 64,
                         minimum_ess: float = 40.0) -> dict:
    if (feature_set not in FEATURE_SETS or variant not in VARIANTS
            or purge_bars < 1 or not np.isfinite(ridge_alpha) or ridge_alpha <= 0
            or not np.isfinite(round_trip_cost_bps)
            or not 0 <= round_trip_cost_bps < 20_000):
        raise ValueError("unregistered model or invalid fixed parameter")
    names = FEATURE_SETS[feature_set]
    cutoff = deploy_start_ms - purge_bars * DT
    train = examples.loc[examples.origin_end_exclusive.le(cutoff)
                         & examples.proposal_long].copy()
    train = train.sort_values(["origin_start", "symbol"], kind="stable")
    values = train[[*names, "forward_gross_3d"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite completed training row")
    n = len(train)
    target = train.forward_gross_3d.to_numpy(float)
    if n:
        age_days = (deploy_start_ms - train.origin_end_exclusive.to_numpy(np.int64)) / 86_400_000.0
        if (age_days < 0).any():
            raise AssertionError("future label entered training")
        if variant == "uniform":
            weights = np.full(n, 1.0 / n)
        else:
            recent = np.exp2(-age_days / 90.0)
            weights = 0.20 / n + 0.80 * recent / recent.sum()
        ess = float(1.0 / np.square(weights).sum())
    else:
        weights = np.empty(0, float)
        ess = 0.0
    lineage = {
        "schema": "exia.genetic_primus.rolling_linear_matrix_model/1",
        "feature_set": feature_set, "feature_names": names, "variant": variant,
        "deploy_start_utc": _iso(deploy_start_ms),
        "train_cutoff_exclusive_utc": _iso(cutoff), "purge_bars": purge_bars,
        "ridge_alpha": ridge_alpha,
        "prediction_target": "next_complete_3d_origin_gross_open_to_last_close",
        "decision_round_trip_cost_bps": float(round_trip_cost_bps),
        "train_opportunities": n,
        "train_rows_sha256": object_sha(train[["origin_start", "origin_end_exclusive",
                                               "symbol", *names, "forward_gross_3d"]]
                                       .to_dict("records")),
        "max_train_label_end_exclusive": int(train.origin_end_exclusive.max()) if n else None,
        "weight_sha256": object_sha(weights.tolist()),
        "weight_ess": ess, "negative_outcomes": int((target <= 0).sum()),
        "negative_outcome_weight": float(weights[target <= 0].sum()),
        "positive_outcomes": int((target > 0).sum()),
        "positive_outcome_weight": float(weights[target > 0].sum()),
        "fallback": None,
    }
    if n < minimum_opportunities or ess < minimum_ess:
        lineage["fallback"] = "insufficient_completed_opportunities_or_ESS"
        lineage["scaler"] = None
        lineage["coefficients"] = None
    else:
        x = train[list(names)].to_numpy(float)
        median = np.median(x, axis=0)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
        scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
        normalized = np.clip((x - median) / scale, -8.0, 8.0)
        design = np.column_stack([np.ones(n), normalized])
        penalty = np.diag([0.0, *([ridge_alpha] * len(names))])
        beta = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                               design.T @ (weights * target))
        lineage["scaler"] = {"median": median.tolist(), "iqr": scale.tolist(),
                             "fitted_rows": n, "fit_values_sha256": object_sha(x.tolist())}
        lineage["coefficients"] = beta.tolist()
    lineage["model_sha256"] = object_sha(lineage)
    return lineage


def schedule_matrix_origin(examples: pd.DataFrame, symbols: tuple[str, ...],
                           deploy_start_ms: int, model: dict,
                           previous_targets: dict[str, int]) -> tuple[pd.DataFrame, dict]:
    if (not symbols or len(set(symbols)) != len(symbols)
            or set(previous_targets) != set(symbols)
            or any(type(value) is not int or value not in (0, 1)
                   for value in previous_targets.values())):
        raise ValueError("complete prior long/flat state required")
    names = FEATURE_SETS.get(model.get("feature_set"))
    if (names is None or model.get("schema") != "exia.genetic_primus.rolling_linear_matrix_model/1"
            or model.get("feature_names") != names
            or model.get("deploy_start_utc") != _iso(deploy_start_ms)
            or model.get("model_sha256") != object_sha({
                key: value for key, value in model.items() if key != "model_sha256"})):
        raise ValueError("wrong, mutated or stale model")
    deploy = examples.loc[examples.origin_start.eq(deploy_start_ms)].set_index("symbol")
    if set(deploy.index) != set(symbols) or len(deploy) != len(symbols):
        raise ValueError("incomplete fixed-universe deployment features")
    fee = float(model["decision_round_trip_cost_bps"]) / 20_000.0
    if not 0 <= fee < 1:
        raise ValueError("invalid execution fee")
    targets = {}
    decisions = {}
    for symbol in symbols:
        row = deploy.loc[symbol]
        gross = None
        if model["coefficients"] is not None and bool(row.proposal_long):
            values = row[list(names)].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite deployment feature")
            normalized = np.clip(
                (values - np.asarray(model["scaler"]["median"], float))
                / np.asarray(model["scaler"]["iqr"], float), -8.0, 8.0)
            beta = np.asarray(model["coefficients"], float)
            gross = float(beta[0] + normalized @ beta[1:])
            if not np.isfinite(gross):
                raise ValueError("nonfinite forecast")
        prior = previous_targets[symbol]
        threshold = 0.0 if prior else (1.0 + fee) / (1.0 - fee) - 1.0
        target = int(gross is not None and gross > threshold)
        targets[symbol] = target
        decisions[symbol] = {"position_before": prior, "target": target,
                             "predicted_gross_3d": gross,
                             "incremental_threshold": threshold}
    schedule = pd.MultiIndex.from_product(
        [np.arange(deploy_start_ms, deploy_start_ms + HORIZON_MS, DT), symbols],
        names=["execution_timestamp", "symbol"]).to_frame(index=False)
    schedule["signal_timestamp"] = deploy_start_ms
    schedule["target"] = schedule.symbol.map(targets).astype(int)
    schedule["model_sha256"] = model["model_sha256"]
    return schedule, {"targets": targets, "decisions": decisions,
                      "model_sha256": model["model_sha256"]}


def build_matrix_schedule(examples: pd.DataFrame, symbols: tuple[str, ...],
                          first_origin_ms: int, end_exclusive_ms: int,
                          feature_set: str, variant: str, **fit_options) -> tuple[pd.DataFrame, list[dict]]:
    if end_exclusive_ms <= first_origin_ms or (end_exclusive_ms - first_origin_ms) % HORIZON_MS:
        raise ValueError("complete nonempty three-day origins required")
    if examples.duplicated(["origin_start", "symbol"]).any():
        raise ValueError("duplicate origin example")
    previous = {symbol: 0 for symbol in symbols}
    schedules = []
    ledger = []
    for origin in range(first_origin_ms, end_exclusive_ms, HORIZON_MS):
        model = fit_matrix_at_origin(examples, origin, feature_set, variant, **fit_options)
        schedule, decision = schedule_matrix_origin(examples, symbols, origin, model, previous)
        ledger.append({"origin_start": origin, "model": model, "decision": decision})
        schedules.append(schedule)
        previous = decision["targets"]
    return pd.concat(schedules, ignore_index=True), ledger
