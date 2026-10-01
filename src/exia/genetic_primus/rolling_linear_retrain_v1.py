"""Causal three-day label preparation and per-origin linear overlay retraining."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .measurement_lab_v2 import build_lab_features, object_sha

DT = 14_400_000
HORIZON_MS = 3 * 86_400_000
FEATURES = ("return_12", "efficiency_12")
VARIANTS = ("uniform", "half_life_90d_anchor_20pct")


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def build_origin_examples(panel: pd.DataFrame, symbols: tuple[str, ...],
                          history_start_ms: int, development_start_ms: int,
                          end_ms: int) -> pd.DataFrame:
    """At each origin use the previous closed bar; label only its next 3 days."""
    if not symbols or len(set(symbols)) != len(symbols) or end_ms <= development_start_ms:
        raise ValueError("invalid fixed-universe chronological interval")
    if (end_ms - development_start_ms) % HORIZON_MS:
        raise ValueError("whole three-day origins required")
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate market candle")
    relevant = panel.loc[panel.timestamp.lt(end_ms)].copy()
    if relevant.empty:
        raise ValueError("empty market history")
    features = build_lab_features(relevant).set_index(["timestamp", "symbol"])
    prices = relevant.set_index(["timestamp", "symbol"])
    count_back = (development_start_ms - history_start_ms) // HORIZON_MS
    first = development_start_ms - count_back * HORIZON_MS
    rows = []
    for start in range(first, end_ms, HORIZON_MS):
        previous = start - DT
        last = start + HORIZON_MS - DT
        if previous < int(relevant.timestamp.min()) or last > int(relevant.timestamp.max()):
            continue
        prior_index = pd.MultiIndex.from_product([[previous], symbols], names=["timestamp", "symbol"])
        open_index = pd.MultiIndex.from_product([[start], symbols], names=["timestamp", "symbol"])
        close_index = pd.MultiIndex.from_product([[last], symbols], names=["timestamp", "symbol"])
        prior = features.reindex(prior_index)
        opens = prices.reindex(open_index).open.to_numpy(float)
        closes = prices.reindex(close_index).close.to_numpy(float)
        if (not np.isfinite(opens).all() or not np.isfinite(closes).all()
                or (opens <= 0).any() or (closes <= 0).any()):
            raise ValueError("incomplete or invalid three-day label prices")
        full = prior[["ema_gap_12_48", "ema_slope_12_3", "return_12", "efficiency_12"]].to_numpy(float)
        finite = np.isfinite(full).all(axis=1)
        proposal = finite & (full[:, 0] > 0) & (full[:, 1] > 0) & (full[:, 2] > 0) & (full[:, 3] >= 0.30)
        returns = closes / opens - 1.0
        for index, symbol in enumerate(symbols):
            if not finite[index]:
                continue
            rows.append({"origin_start": start, "origin_end_exclusive": start + HORIZON_MS,
                         "symbol": symbol, "return_12": float(full[index, 2]),
                         "efficiency_12": float(full[index, 3]),
                         "proposal_long": bool(proposal[index]),
                         "forward_gross_3d": float(returns[index])})
    if not rows:
        raise ValueError("no finite causal origin examples")
    return pd.DataFrame(rows).sort_values(["origin_start", "symbol"], kind="stable").reset_index(drop=True)


def fit_at_origin(examples: pd.DataFrame, deploy_start_ms: int, variant: str, *,
                  purge_bars: int = 7, ridge_alpha: float = 0.5,
                  normal_round_trip_bps: float = 16.0,
                  minimum_opportunities: int = 64,
                  minimum_ess: float = 40.0) -> dict:
    """Fit from completed labels only; recency weights cover every outcome."""
    if variant not in VARIANTS or purge_bars < 1 or ridge_alpha <= 0:
        raise ValueError("unregistered retraining variant")
    cutoff = deploy_start_ms - purge_bars * DT
    train = examples.loc[examples.origin_end_exclusive.le(cutoff)
                         & examples.proposal_long].copy()
    train = train.sort_values(["origin_start", "symbol"], kind="stable")
    values = train[list(FEATURES) + ["forward_gross_3d"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite training feature or outcome")
    n = len(train)
    if n:
        target = train.forward_gross_3d.to_numpy(float) - normal_round_trip_bps / 10_000.0
        age_days = (deploy_start_ms - train.origin_end_exclusive.to_numpy(np.int64)) / 86_400_000.0
        if (age_days < 0).any():
            raise AssertionError("future training label")
        if variant == "uniform":
            weights = np.full(n, 1.0 / n)
        else:
            recent = np.exp2(-age_days / 90.0)
            weights = 0.20 / n + 0.80 * recent / recent.sum()
        ess = float(1.0 / np.square(weights).sum())
    else:
        target = np.empty(0)
        weights = np.empty(0)
        ess = 0.0
    lineage = {
        "schema": "exia.genetic_primus.rolling_linear_model/1",
        "variant": variant, "deploy_start_utc": _iso(deploy_start_ms),
        "train_cutoff_exclusive_utc": _iso(cutoff),
        "purge_bars": purge_bars, "feature_names": FEATURES,
        "ridge_alpha": ridge_alpha, "train_opportunities": n,
        "train_rows_sha256": object_sha(train[["origin_start", "origin_end_exclusive", "symbol",
                                                  *FEATURES, "forward_gross_3d"]].to_dict("records")),
        "max_train_label_end_exclusive": int(train.origin_end_exclusive.max()) if n else None,
        "weight_ess": ess, "weight_sum": float(weights.sum()),
        "negative_outcomes": int((target <= 0).sum()),
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
        x = train[list(FEATURES)].to_numpy(float)
        median = np.median(x, axis=0)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
        scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
        normalized = np.clip((x - median) / scale, -8.0, 8.0)
        design = np.column_stack([np.ones(n), normalized])
        penalty = np.diag([0.0, ridge_alpha, ridge_alpha])
        coefficient = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                      design.T @ (weights * target))
        lineage["scaler"] = {"median": median.tolist(), "iqr": scale.tolist(),
                             "fitted_rows": n,
                             "fit_values_sha256": object_sha(x.tolist())}
        lineage["coefficients"] = coefficient.tolist()
    lineage["model_sha256"] = object_sha(lineage)
    return lineage


def schedule_at_origin(examples: pd.DataFrame, symbols: tuple[str, ...],
                       deploy_start_ms: int, model: dict) -> tuple[pd.DataFrame, dict]:
    deploy = examples.loc[examples.origin_start.eq(deploy_start_ms)].set_index("symbol")
    if set(deploy.index) != set(symbols) or len(deploy) != len(symbols):
        raise ValueError("incomplete deploy feature universe")
    deploy = deploy.reindex(symbols)
    if model["deploy_start_utc"] != _iso(deploy_start_ms):
        raise ValueError("model belongs to another origin")
    selected = []
    predicted = {}
    for symbol, row in deploy.iterrows():
        prediction = None
        if model["coefficients"] is not None and bool(row.proposal_long):
            values = row[list(FEATURES)].to_numpy(float)
            median = np.asarray(model["scaler"]["median"])
            scale = np.asarray(model["scaler"]["iqr"])
            normalized = np.clip((values - median) / scale, -8.0, 8.0)
            beta = np.asarray(model["coefficients"])
            prediction = float(beta[0] + normalized @ beta[1:])
            if prediction > 0.0:
                selected.append(str(symbol))
        predicted[str(symbol)] = prediction
    times = np.arange(deploy_start_ms, deploy_start_ms + HORIZON_MS, DT)
    schedule = pd.MultiIndex.from_product(
        [times, symbols], names=["execution_timestamp", "symbol"]).to_frame(index=False)
    schedule["signal_timestamp"] = deploy_start_ms
    schedule["target"] = schedule.symbol.isin(selected).astype(int)
    schedule["model_sha256"] = model["model_sha256"]
    return schedule, {"selected_symbols": selected, "predicted_net_3d": predicted}
