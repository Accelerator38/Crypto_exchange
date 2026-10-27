"""Causal direction/abstention experiment. Pure arrays, no market or runtime IO."""
from __future__ import annotations

import hashlib
import math
import warnings
import numpy as np

DAY = 86_400_000


def signs(current, future):
    current, future = np.asarray(current, float), np.asarray(future, float)
    if current.shape != future.shape or not np.isfinite(current).all() or not np.isfinite(future).all():
        raise ValueError("matching finite prices required")
    if np.any(current <= 0) or np.any(future <= 0):
        raise ValueError("positive prices required")
    return np.sign(future - current).astype(np.int8)


def tie_keys(times, symbol, seed):
    # Stable integer comparison, avoiding floating-point jitter in probabilities.
    return np.array([int.from_bytes(hashlib.sha256(f"{seed}|{symbol}|{int(t)}".encode()).digest()[:8], "big")
                     for t in times], dtype=np.uint64)


def confidence(probability):
    p = np.asarray(probability, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite one-dimensional probabilities in [0,1] required")
    return np.abs(2 * p - 1)


def calibrate(probability, times, symbol, coverage, seed):
    c = confidence(probability)
    times = np.asarray(times, np.int64)
    if len(c) != len(times) or not len(c) or not 0 < coverage <= 1 or np.any(np.diff(times) <= 0):
        raise ValueError("nonempty chronological calibration and valid coverage required")
    if coverage == 1:
        return {"coverage_target": 1.0, "confidence": -1.0, "tie_key": 0, "calibration_rows": len(c)}
    keys = tie_keys(times, symbol, seed)
    chosen = np.lexsort((keys, c))[-math.ceil(coverage * len(c))]
    return {"coverage_target": coverage, "confidence": float(c[chosen]),
            "tie_key": int(keys[chosen]), "calibration_rows": len(c)}


def emit(probability, times, symbol, threshold, seed):
    c = confidence(probability)
    if len(c) != len(times):
        raise ValueError("matching times and predictions required")
    keys = tie_keys(times, symbol, seed)
    return (c > threshold["confidence"]) | ((c == threshold["confidence"]) & (keys >= threshold["tie_key"]))


def split(times, origin, horizon_ms, train_days=30, calibration_days=7, purge=7, cap=4000, seed=0):
    times = np.asarray(times, np.int64)
    if len(times) < 3 or np.any(np.diff(times) != horizon_ms):
        raise ValueError("complete chronological horizon grid required")
    rows = np.arange(48, len(times) - 1)
    boundary = origin - calibration_days * DAY
    train = rows[(times[rows] >= origin - train_days * DAY) & (times[rows + 1] <= boundary - purge * horizon_ms)]
    calibration = rows[(times[rows] >= boundary) & (times[rows + 1] <= origin - purge * horizon_ms)]
    if not len(train) or not len(calibration):
        raise ValueError("adequate preceding training/calibration required")
    if len(train) > cap:
        train = np.sort(np.random.default_rng(seed).choice(train, cap, replace=False))
    if times[train[-1] + 1] > times[calibration[0]] - purge * horizon_ms:
        raise ValueError("train/calibration label overlap")
    return train, calibration


def fit(kind, features, labels, seed, minimum_leaf=128):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    x, y = np.asarray(features, float), np.asarray(labels)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("finite features and true directional labels required")
    valid = y != 0
    x, y = x[valid], (y[valid] > 0).astype(np.int8)
    if len(x) < 64 or len(np.unique(y)) != 2:
        raise ValueError("adequate mature binary classes required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    if kind == "logistic":
        estimator = LogisticRegression(C=0.1, max_iter=120, tol=1e-6, random_state=seed)
    elif kind == "boosting":
        estimator = HistGradientBoostingClassifier(max_iter=32, max_leaf_nodes=7, learning_rate=0.08,
            l2_regularization=10, min_samples_leaf=minimum_leaf, early_stopping=False, random_state=seed)
    else:
        raise ValueError("registered method required")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit((x - mean) / scale, y)
    return {"estimator": estimator, "mean": mean, "scale": scale, "rows": len(y),
            "train_majority": bool(y.mean() >= .5), "warnings": [str(w.message)[:160] for w in caught]}


def predict(model, features):
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("matching finite features required")
    probability = model["estimator"].predict_proba((x - model["mean"]) / model["scale"])[:, 1]
    confidence(probability)
    return probability


def evaluate(predicted_up, labels, mask=None):
    p, y = np.asarray(predicted_up, bool), np.asarray(labels)
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    if y.ndim != 1 or p.shape != y.shape or mask.shape != y.shape or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("matching binary decisions and ternary outcomes required")
    correct = ((p & (y == 1)) | (~p & (y == -1))) & mask
    n, nf = int(mask.sum()), int((mask & (y != 0)).sum())
    recalls = [float(correct[y == sign].sum() / (mask & (y == sign)).sum())
               if (mask & (y == sign)).any() else None for sign in [1, -1]]
    return {"n_eligible": len(y), "n_emitted": n, "n_nonflat_emitted": nf,
            "correct": int(correct.sum()), "coverage_actual": n / len(y),
            "accuracy_all": float(correct.sum() / n) if n else None,
            "accuracy_nonflat": float(correct.sum() / nf) if nf else None,
            "flat_emitted": int((mask & (y == 0)).sum()), "flat_eligible": int((y == 0).sum()),
            "recall_up": recalls[0], "recall_down": recalls[1],
            "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None,
            "confusion": {"up_up": int((mask & p & (y == 1)).sum()),
                          "up_down": int((mask & ~p & (y == 1)).sum()),
                          "down_up": int((mask & p & (y == -1)).sum()),
                          "down_down": int((mask & ~p & (y == -1)).sum()),
                          "flat_up": int((mask & p & (y == 0)).sum()),
                          "flat_down": int((mask & ~p & (y == 0)).sum())}}


def daily_counts(times, labels, predicted_up, mask, day_start, days):
    y, p, mask = np.asarray(labels), np.asarray(predicted_up, bool), np.asarray(mask, bool)
    day = (np.asarray(times) - day_start) // DAY
    if np.any((day < 0) | (day >= days)):
        raise ValueError("registered day range required")
    correct = ((p & (y == 1)) | (~p & (y == -1))) & mask
    return np.stack([np.bincount(day, weights=a, minlength=days)
                     for a in [correct, mask, mask & (y != 0)]], axis=1)


def block_bank(days, repetitions, block_days, seed):
    if days < block_days or min(repetitions, block_days) < 1:
        raise ValueError("adequate calendar blocks required")
    starts = np.random.default_rng(seed).integers(0, days, (repetitions, math.ceil(days / block_days)))
    return ((starts[:, :, None] + np.arange(block_days)) % days).reshape(repetitions, -1)[:, :days]


def resample_statistics(daily, bank):
    a = np.asarray(daily, float)
    total = a.sum(axis=0)
    sampled = a[bank].sum(axis=1)
    accuracy = np.divide(sampled[:, 0], sampled[:, 1], out=np.full(len(bank), np.nan), where=sampled[:, 1] > 0)
    finite = accuracy[np.isfinite(accuracy)]
    return {"accuracy": float(total[0] / total[1]) if total[1] else None,
            "ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "emitted_days": int((a[:, 1] > 0).sum())}, accuracy


def simultaneous_lcb(observed, replicates):
    """Exploratory one-sided max-t family bound; shared calendar draws."""
    obs, rep = np.asarray(observed), np.asarray(replicates)
    if rep.ndim != 2 or rep.shape[1] != len(obs) or not np.isfinite(rep).all():
        raise ValueError("finite shared paired bootstrap family required")
    se = rep.std(axis=0, ddof=1)
    active = se > 1e-12
    if not active.any():
        return obs.copy(), 0.0
    maximum = np.max((obs[None, active] - rep[:, active]) / se[None, active], axis=1)
    critical = max(0.0, float(np.quantile(maximum, .95)))
    return obs - critical * se, critical


def validate_onchain(times, closes, features, available_ms, feature_names):
    """Fail closed on late features. Availability evidence must come from provider."""
    t, x, a = np.asarray(times), np.asarray(features, float), np.asarray(available_ms)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) != DAY) or np.any(t % DAY):
        raise ValueError("complete UTC daily decision grid required")
    if x.ndim != 2 or len(x) != len(t) or a.shape != x.shape or len(feature_names) != x.shape[1]:
        raise ValueError("daily feature values and per-value publication times required")
    if np.asarray(closes).shape != t.shape:
        raise ValueError("one close per daily decision required")
    if len(set(feature_names)) != len(feature_names) or not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("unique named finite point-in-time features required")
    if np.any(a > t[:, None]):
        raise ValueError("on-chain feature unavailable at decision")
    return signs(np.asarray(closes)[:-1], np.asarray(closes)[1:])


def direct_checks():
    # Hand-computed accuracy, including a flat; no collector or event loop.
    y = signs([10, 10, 10, 10], [11, 9, 10, 11])
    a = evaluate([True, False, True, False], y)
    assert a["accuracy_all"] == .5 and a["accuracy_nonflat"] == 2 / 3 and a["flat_emitted"] == 1
    assert evaluate([True] * 4, y, [False] * 4)["accuracy_all"] is None
    times = np.arange(20) * 60000
    p = np.full(20, .5)
    for coverage in [1, .5, .2, .1]:
        threshold = calibrate(p, times, "BTC", coverage, 5)
        assert int(emit(p, times, "BTC", threshold, 5).sum()) == math.ceil(coverage * 20)
        before = dict(threshold)
        emit([.99, .01, .51], [9000000, 9060000, 9120000], "BTC", threshold, 5)
        assert threshold == before
    t = np.arange(100 * 1440) * 60000
    train, cal = split(t, 90 * DAY, 60000, cap=1000, seed=4)
    assert t[train[-1] + 1] <= t[cal[0]] - 7 * 60000
    assert t[cal[-1] + 1] <= 90 * DAY - 7 * 60000
    daily = daily_counts([0, DAY, 2 * DAY, 3 * DAY], y, [True, False, True, False], [True] * 4, 0, 4)
    assert np.array_equal(daily.sum(axis=0), [2, 4, 3])
    bank = block_bank(4, 20, 2, 3)
    assert np.array_equal(bank, block_bank(4, 20, 2, 3))
    validate_onchain([0, DAY], [10, 11], [[1], [2]], [[0], [DAY]], ["activity"])
    try:
        validate_onchain([0, DAY], [10, 11], [[1], [2]], [[1], [DAY]], ["activity"])
    except ValueError:
        pass
    else:
        raise AssertionError("late on-chain value admitted")
    return {"target_and_flat": True, "train_only_threshold": True, "purge": True,
            "calendar_bootstrap": True, "late_onchain_rejected": True}
