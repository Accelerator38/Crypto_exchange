"""Causal six-feature panel and bounded rolling forecast fits; no I/O/runtime."""
from __future__ import annotations

import numpy as np

from . import compact_forecast_v1 as model

BAR_MS = 14_400_000
HORIZON_MS = 18 * BAR_MS
FEATURE_DEFINITION = {
    "schema": "compact_six_features/1", "features": list(model.FEATURES),
    "distance_ema48": "close / ema48 - 1; adjust_false; seed first observed close",
    "return_18": "close / close_18_bars_ago - 1",
    "efficiency_18": "abs(close-close18) / sum(abs(diff(close)),18); flat=0",
    "relative_return_18": "return18 - full_universe_mean(return18)",
    "volatility_48": "std(last48 log close returns, ddof=0)",
    "breadth_ema48": "full_universe_mean(close > ema48)",
    "min_history_bars": 49, "missing_symbol_policy": "reject", "lag_bars": 0,
    "feature_availability": "bar_open + BAR_MS; completed bars only",
}
FEATURE_SHA256 = model.identity(FEATURE_DEFINITION)


def feature_panel(times, closes):
    timestamps = np.asarray(times)
    close = np.asarray(closes, dtype=float)
    if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu" or len(timestamps) < 49:
        raise ValueError("at least 49 integer timestamp rows required")
    if np.any(timestamps < 0) or np.any(timestamps % BAR_MS) or np.any(np.diff(timestamps) != BAR_MS):
        raise ValueError("complete regular four-hour grid required")
    if close.ndim != 2 or close.shape[0] != len(timestamps) or close.shape[1] < 1:
        raise ValueError("full-universe close panel required")
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("strictly positive finite closes required")
    ema = np.empty_like(close)
    ema[0] = close[0]
    alpha = 2 / 49
    for i in range(1, len(close)):
        # Difference form preserves an exactly flat market (including breadth).
        ema[i] = ema[i - 1] + alpha * (close[i] - ema[i - 1])
    returns = np.full_like(close, np.nan)
    returns[18:] = close[18:] / close[:-18] - 1
    changes = np.abs(np.diff(close, axis=0))
    denominator = np.lib.stride_tricks.sliding_window_view(changes, 18, axis=0).sum(axis=-1)
    efficiency = np.full_like(close, np.nan)
    numerator = np.abs(close[18:] - close[:-18])
    efficiency[18:] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    volatility = np.full_like(close, np.nan)
    log_returns = np.diff(np.log(close), axis=0)
    volatility[48:] = np.lib.stride_tricks.sliding_window_view(log_returns, 48, axis=0).std(axis=-1)
    relative = returns - returns.mean(axis=1, keepdims=True)
    breadth = np.repeat((close > ema).mean(axis=1, keepdims=True), close.shape[1], axis=1)
    result = np.stack((close / ema - 1, returns, efficiency, relative, volatility, breadth), axis=-1)
    result[:48] = np.nan
    if not np.isfinite(result[48:]).all():
        raise ValueError("nonfinite post-warmup features")
    return result


def make_samples(times, opens, closes, anchor_ms):
    timestamps = np.asarray(times, dtype=np.int64)
    opening = np.asarray(opens, dtype=float)
    closing = np.asarray(closes, dtype=float)
    features = feature_panel(timestamps, closing)
    if opening.shape != closing.shape or not np.isfinite(opening).all() or np.any(opening <= 0):
        raise ValueError("positive aligned open panel required")
    if type(anchor_ms) is not int or anchor_ms % BAR_MS:
        raise ValueError("registered four-hour origin anchor required")
    # The last feature candle opens at origin-B and closes exactly at origin.
    earliest = int(timestamps[48]) + BAR_MS
    first = anchor_ms + -((anchor_ms - earliest) // HORIZON_MS) * HORIZON_MS
    # Outcome uses the next open after decision-asof, then exactly 18 bars.
    last = int(timestamps[-1]) - 19 * BAR_MS
    origins = np.arange(first, last + 1, HORIZON_MS, dtype=np.int64)
    if not len(origins):
        raise ValueError("no complete outcomes")
    feature_idx = np.searchsorted(timestamps, origins - BAR_MS)
    entry_idx = np.searchsorted(timestamps, origins + BAR_MS)
    exit_idx = np.searchsorted(timestamps, origins + 19 * BAR_MS)
    if not np.array_equal(timestamps[feature_idx], origins - BAR_MS):
        raise ValueError("missing feature origin")
    return {"origins": origins, "available": origins + 20 * BAR_MS,
            "entry": origins + BAR_MS, "exit": origins + 19 * BAR_MS,
            "x": features[feature_idx], "y": opening[exit_idx] / opening[entry_idx] - 1,
            "feature_definition_sha256": FEATURE_SHA256}


def eligible_split(samples, origin_ms, *, purge_bars=7, inner_days=60):
    if type(origin_ms) is not int or origin_ms % BAR_MS or purge_bars != 7 or inner_days != 60:
        raise ValueError("unregistered origin/purge/inner split")
    # Training ends one bar before decision; labels then have an additional 7-bar purge.
    train_end = origin_ms - BAR_MS
    cutoff = train_end - purge_bars * BAR_MS
    train = np.flatnonzero(samples["available"] <= cutoff)
    if len(train) < 50:
        raise ValueError("insufficient temporal training support")
    inner_start = int(samples["origins"][train[-1]]) - inner_days * model.DAY_MS
    validation = train[samples["origins"][train] >= inner_start]
    inner_cutoff = int(samples["origins"][validation[0]]) - BAR_MS - purge_bars * BAR_MS
    inner_train = train[samples["available"][train] <= inner_cutoff]
    if len(inner_train) < 25 or len(validation) < 10:
        raise ValueError("insufficient purged chronological inner support")
    decision = np.flatnonzero(samples["origins"] == origin_ms)
    if len(decision) != 1:
        raise ValueError("decision needs one complete revealed benchmark outcome")
    return {"train": train, "inner_train": inner_train, "inner_validation": validation,
            "decision": int(decision[0]), "train_end_ms": train_end, "cutoff_ms": cutoff,
            "inner_cutoff_ms": inner_cutoff,
            "inner_start_ms": int(samples["origins"][validation[0]])}


def _flatten(samples, indices):
    x = samples["x"][indices].reshape(-1, 6)
    y = samples["y"][indices].reshape(-1)
    return x, y


def fit_origin(samples, origin_ms):
    """Four fixed arms; no seed selection or benchmark-dependent tuning."""
    split = eligible_split(samples, origin_ms)
    train, inner, validation = split["train"], split["inner_train"], split["inner_validation"]
    tx, ty = _flatten(samples, train)
    ix, iy = _flatten(samples, inner)
    vx, vy = _flatten(samples, validation)
    decision_x = samples["x"][split["decision"]]
    scaler, inner_scaler = model.fit_scaler(tx, ty), model.fit_scaler(ix, iy)
    train_sha = model.array_identity(samples["origins"][train], samples["available"][train], tx, ty)
    fold_sha = model.array_identity(samples["origins"][inner], samples["origins"][validation])
    training_identity = {"train_samples_sha256": train_sha, "fold_sha256": fold_sha,
                         "train_end_ms": split["train_end_ms"], "cutoff_ms": split["cutoff_ms"],
                         "feature_definition_sha256": FEATURE_SHA256,
                         "l2": 0.01, "learning_rate": 0.001, "max_epochs": 120,
                         "patience": 12, "inner_cutoff_ms": split["inner_cutoff_ms"]}
    result = {"schema": "compact_origin_forecast/1", "origin_ms": origin_ms,
              "entry_ms": origin_ms + BAR_MS, "exit_ms": origin_ms + 19 * BAR_MS,
              "outcome_available_ms": origin_ms + 20 * BAR_MS,
              "training_identity": training_identity,
              "feature_prefix_sha256": model.array_identity(samples["origins"][:split["decision"] + 1],
                                                             samples["x"][:split["decision"] + 1]),
              "decision_feature_sha256": model.array_identity(decision_x),
              "train_origin_count": len(train), "train_sample_count": len(ty),
              "inner_train_origin_count": len(inner), "inner_validation_origin_count": len(validation),
              "target_positive_count": int(np.count_nonzero(ty > 0)),
              "target_negative_count": int(np.count_nonzero(ty < 0)),
              "target_zero_count": int(np.count_nonzero(ty == 0)),
              "scaler_sha256": scaler.sha256, "inner_scaler_sha256": inner_scaler.sha256,
              "actual": samples["y"][split["decision"]].tolist(), "arms": {},
              "baseline_train_mean": np.full(len(decision_x), ty.mean()).tolist(),
              "forecast_only": True, "trading_evaluations": 0,
              "model_fit_count": 0, "gradient_steps": 0, "scaler_fit_count": 2}
    symbols = samples["x"].shape[1]
    for weighting in model.WEIGHTINGS:
        ow = model.temporal_weights(samples["available"][train], split["cutoff_ms"], weighting)
        iw = model.temporal_weights(samples["available"][inner], split["inner_cutoff_ms"], weighting)
        weights, inner_weights = np.repeat(ow / symbols, symbols), np.repeat(iw / symbols, symbols)
        weighting_info = {"origin_ess": model.origin_ess(ow), "weights_sha256": model.array_identity(ow),
                          "positive_weight_mass": float(weights[ty > 0].sum()),
                          "negative_weight_mass": float(weights[ty < 0].sum()),
                          "zero_weight_mass": float(weights[ty == 0].sum()),
                          "minimum_sample_weight": float(weights.min()),
                          "maximum_sample_weight": float(weights.max())}
        params = model.fit_linear(scaler.transform(tx), scaler.target(ty), weights)
        linear = model.model_payload("linear6", params, scaler, weighting=weighting,
                                     seed=None, epochs=None, training_identity=training_identity | weighting_info)
        result["model_fit_count"] += 1
        result["arms"]["linear6_" + weighting] = weighting_info | {
            "models": [linear], "prediction": model.predict(linear, decision_x).tolist(),
            "parameter_count_per_model": 7, "stored_parameter_count": 7}
        models, predictions, selection = [], [], []
        for seed in model.SEEDS:
            _, trace = model.fit_mlp(inner_scaler.transform(ix), inner_scaler.target(iy), inner_weights,
                                     seed=seed, validation=(inner_scaler.transform(vx), inner_scaler.target(vy)))
            epochs = trace["selected_epoch"]
            params, refit = model.fit_mlp(scaler.transform(tx), scaler.target(ty), weights,
                                         seed=seed, epochs=epochs)
            payload = model.model_payload("mlp8", params, scaler, weighting=weighting,
                                          seed=seed, epochs=epochs, training_identity=training_identity | weighting_info)
            models.append(payload)
            predictions.append(model.predict(payload, decision_x))
            selection.append({"seed": seed, "selected_epoch": epochs,
                              "inner_epochs_run": trace["epochs_run"],
                              "inner_best_mse": trace["history"][epochs - 1]["validation_mse"],
                              "refit_epochs_run": refit["epochs_run"]})
            result["model_fit_count"] += 2
            result["gradient_steps"] += trace["gradient_steps"] + refit["gradient_steps"]
        result["arms"]["mlp8_" + weighting] = weighting_info | {
            "models": models, "prediction": np.mean(predictions, axis=0).tolist(),
            "seed_predictions": np.asarray(predictions).tolist(), "epoch_selection": selection,
            "parameter_count_per_model": 65, "stored_parameter_count": 195,
            "seed_aggregation": "fixed_mean_no_best_seed_selection"}
    result["sha256"] = model.identity(result)
    return result
