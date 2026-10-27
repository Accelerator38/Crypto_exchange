"""Small causal forecasting hypotheses. No files, event loops, runtime or network."""
from __future__ import annotations
import warnings
import numpy as np
from . import direction_selective_v1 as d


def feature_rows(bank, indices, kind, minutes):
    ix = np.asarray(indices, int)
    base = np.asarray(bank["features"][ix], float)
    if kind == "base12": return base
    if kind == "base6": return base[:, :, :6]
    if kind == "asset19":
        ids = np.broadcast_to(np.eye(8)[:, 1:], (len(ix), 8, 7))
        return np.concatenate([base, ids], axis=2)
    if kind == "lag24":
        if ix.min() < 6: raise ValueError("six complete previous bars required")
        lags = [bank["features"][ix-lag][:, :, [6, 7, 11]] for lag in [1, 2, 3, 6]]
        return np.concatenate([base, *lags], axis=2)
    if kind != "context20": raise ValueError("registered feature kind required")
    closes = bank["closes"]
    hour, day, week = [v // minutes for v in (60, 1440, 10080)]
    if ix.min() < week: raise ValueError("complete causal seven-day context required")
    c = np.asarray(closes[ix])
    rh, rd, rw = [c / closes[ix-lag] - 1 for lag in [hour, day, week]]
    lr = np.diff(np.log(np.asarray(closes)), axis=0)
    prefix = np.vstack([np.zeros(8), np.cumsum(lr, axis=0)])
    square = np.vstack([np.zeros(8), np.cumsum(lr*lr, axis=0)])
    mean = (prefix[ix]-prefix[ix-day])/day
    vol = np.sqrt(np.maximum(0, (square[ix]-square[ix-day])/day-mean*mean))
    market = np.broadcast_to(rd.mean(axis=1, keepdims=True), rd.shape)
    phase = (bank["times"][ix] % d.DAY) / d.DAY * 2*np.pi
    sin, cos = [np.broadcast_to(v[:, None], rd.shape) for v in (np.sin(phase), np.cos(phase))]
    extra = np.stack([rh, rd, rw, vol, rd-market, market, sin, cos], axis=2)
    return np.concatenate([base, extra], axis=2)


def larger_train(times, original, origin, horizon_ms, cap, seed):
    rows = np.arange(48, len(times)-1)
    boundary = origin - 7*d.DAY
    eligible = rows[(times[rows] >= origin-30*d.DAY) & (times[rows+1] <= boundary-7*horizon_ms)]
    remaining = np.setdiff1d(eligible, original, assume_unique=True)
    count = min(max(0, cap-len(original)), len(remaining))
    supplement = np.random.default_rng(seed+90000).choice(remaining, count, replace=False)
    result = np.sort(np.concatenate([original, supplement]))
    if not np.isin(original, result).all(): raise ValueError("original training sample must be retained")
    return result


def fit_binary(kind, x, y, seed, weights=None, robust=False, minimum_leaf=128):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    x, y = np.asarray(x, float), np.asarray(y)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("finite causal features and binary fitting labels required")
    if len(x) < 64 or len(np.unique(y)) != 2: raise ValueError("adequate mature two-class train required")
    if weights is not None:
        weights = np.asarray(weights, float)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("positive weights for both outcome classes required")
        weights = weights / weights.mean()
    lower, upper = np.quantile(x, [.005, .995], axis=0) if robust else (None, None)
    z = np.clip(x, lower, upper) if robust else x
    mean, scale = z.mean(axis=0), z.std(axis=0)
    scale[scale < 1e-12] = 1
    if kind == "logistic":
        estimator = LogisticRegression(C=.1, max_iter=120, tol=1e-6, random_state=seed)
    elif kind == "boosting":
        estimator = HistGradientBoostingClassifier(max_iter=32, max_leaf_nodes=7, learning_rate=.08,
            l2_regularization=10, min_samples_leaf=minimum_leaf, early_stopping=False, random_state=seed)
    elif kind == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=10,
                                  max_iter=80, max_fun=160, tol=1e-6, random_state=seed)
    else:
        raise ValueError("registered estimator required")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit((z-mean)/scale, y, sample_weight=weights)
    info = {"rows": len(y), "features": x.shape[1], "kind": kind,
            "warnings": [str(w.message)[:180] for w in caught], "robust_train_only": robust,
            "weighted_positive_mass": float(weights[y == 1].sum()) if weights is not None else int((y == 1).sum()),
            "weighted_negative_mass": float(weights[y == 0].sum()) if weights is not None else int((y == 0).sum()),
            "kish_weight_ess_not_independent_observations": float(weights.sum()**2 / np.sum(weights**2)) if weights is not None else len(y)}
    if kind == "logistic": info["parameter_count"] = x.shape[1]+1
    if kind == "mlp": info["parameter_count"] = sum(a.size for a in estimator.coefs_+estimator.intercepts_)
    return {"estimator": estimator, "mean": mean, "scale": scale, "lower": lower, "upper": upper, "info": info}


def predict(model, x):
    x = np.asarray(x, float)
    if model["lower"] is not None: x = np.clip(x, model["lower"], model["upper"])
    probability = model["estimator"].predict_proba((x-model["mean"])/model["scale"])[:, 1]
    d.confidence(probability)
    return probability


def recency_weights(times, origin, half_life_days):
    times = np.asarray(times)
    if np.any(times > origin) or half_life_days <= 0: raise ValueError("past rows and positive half-life required")
    return np.exp2(-(origin-times)/(d.DAY*half_life_days))


def score_threshold(score, times, symbol, coverage, seed):
    score = np.asarray(score)
    if np.any((score < 0) | (score > 1)): raise ValueError("score in [0,1] required")
    return d.calibrate((score+1)/2, times, symbol, coverage, seed)


def score_mask(score, times, symbol, threshold, seed):
    return d.emit((np.asarray(score)+1)/2, times, symbol, threshold, seed)


def direct_checks():
    n = 20000
    c = np.arange(n*8, dtype=float).reshape(n, 8)*.001+100
    x = np.repeat(c[:, :, None], 12, axis=2)
    bank = {"times": np.arange(n)*60000, "closes": c, "features": x}
    ix = np.array([12000, 12001, 12002])
    for kind, dimensions in [("lag24", 24), ("context20", 20), ("asset19", 19), ("base6", 6)]:
        before = feature_rows(bank, ix, kind, 1)
        assert before.shape == (3, 8, dimensions)
        changed = {"times": bank["times"], "closes": c.copy(), "features": x.copy()}
        changed["closes"][12003:] *= 3; changed["features"][12003:] *= 4
        assert np.array_equal(before, feature_rows(changed, ix, kind, 1))
    original, _ = d.split(np.arange(100*1440)*60000, 90*d.DAY, 60000, seed=1)
    expanded = larger_train(np.arange(100*1440)*60000, original, 90*d.DAY, 60000, 16000, 1)
    assert len(expanded) == 16000 and np.isin(original, expanded).all()
    weights = recency_weights([0, 90*d.DAY], 90*d.DAY, 90)
    assert np.array_equal(weights, [.5, 1])
    threshold = score_threshold([.1, .2, .3, .4], [1, 2, 3, 4], "BTC", .5, 1)
    assert score_mask([.1, .2, .3, .4], [1, 2, 3, 4], "BTC", threshold, 1).tolist() == [False, False, True, True]
    return {"future_suffix_causality": True, "large_train_nested": True, "recency_all_signs": True, "score_threshold": True}
