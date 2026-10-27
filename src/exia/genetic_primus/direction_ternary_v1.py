"""Three-state direction semantics and a compact, causal offline classifier."""
from __future__ import annotations
import warnings
import numpy as np

STATES = np.array([-1, 0, 1], dtype=np.int8)
DAY = 86_400_000


def labels(return_pct, band_pct):
    r, band = np.asarray(return_pct, float), np.asarray(band_pct, float)
    if not np.isfinite(r).all() or not np.isfinite(band).all() or np.any(band < 0):
        raise ValueError("finite percentage returns and nonnegative bands required")
    return np.where(r > band, 1, np.where(r < -band, -1, 0)).astype(np.int8)


def train_bands(train_return_pct):
    r = np.asarray(train_return_pct, float)
    if r.ndim != 2 or not len(r) or not np.isfinite(r).all():
        raise ValueError("finite time-by-asset training returns required")
    return np.median(np.abs(r), axis=0)


def counts_per_asset(y):
    y = np.asarray(y)
    return np.stack([(y == state).sum(axis=0) for state in STATES], axis=-1)


def fit(x, y, config, seed):
    from sklearn.linear_model import LogisticRegression
    x, y = np.asarray(x, float), np.asarray(y, np.int8)
    if x.ndim != 3 or y.shape != x.shape[:2] or not np.isfinite(x).all():
        raise ValueError("finite time/asset/features and matching labels required")
    if not np.isin(y, STATES).all(): raise ValueError("unknown state")
    per_asset = counts_per_asset(y)
    pooled = per_asset.sum(axis=0)
    priors = (per_asset+1)/(len(y)+3)  # Train-only Laplace smoothing.
    z, target = x.reshape(-1, x.shape[-1]), y.ravel()
    mean, scale = z.mean(axis=0), z.std(axis=0)
    scale[scale < 1e-12] = 1
    model = {"mean": mean, "scale": scale, "train_counts": per_asset, "train_priors": priors,
             "kind": "prior_only_sparse_classes", "estimator": None, "warnings": [],
             "nominal_coefficient_count": 0, "train_rows_including_neutral": len(target)}
    if np.min(pooled) < config["minimum_pooled_train_rows_per_state"]:
        return model
    estimator = LogisticRegression(C=config["C"], class_weight=config["class_weight"],
        max_iter=config["max_iter"], tol=config["tol"], random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit((z-mean)/scale, target)
    if not np.array_equal(estimator.classes_, STATES): raise ValueError("class order mismatch")
    model.update(kind="three_class_balanced_logit", estimator=estimator,
                 warnings=[str(w.message)[:240] for w in caught],
                 nominal_coefficient_count=estimator.coef_.size+estimator.intercept_.size)
    return model


def probability(model, x):
    x = np.asarray(x, float)
    if x.ndim != 3 or not np.isfinite(x).all(): raise ValueError("finite feature cube required")
    if model["estimator"] is None:
        p = np.broadcast_to(model["train_priors"], (*x.shape[:2], 3)).copy()
    else:
        z = (x.reshape(-1, x.shape[-1])-model["mean"])/model["scale"]
        p = model["estimator"].predict_proba(z).reshape(*x.shape[:2], 3)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=-1), 1, atol=1e-10):
        raise ValueError("invalid three-state probability")
    return p


def predict(p):
    return STATES[np.argmax(p, axis=-1)]  # Tie policy fixed: down, then neutral, then up.


def confusion(truth, predicted):
    y, p = np.asarray(truth).ravel(), np.asarray(predicted).ravel()
    if y.shape != p.shape or not np.isin(y, STATES).all() or not np.isin(p, STATES).all():
        raise ValueError("matching three-state labels required")
    return np.bincount((y.astype(int)+1)*3+p.astype(int)+1, minlength=9).reshape(3, 3)


def from_confusion(matrix):
    cm = np.asarray(matrix, float)
    if cm.shape != (3, 3) or np.any(cm < 0): raise ValueError("3x3 nonnegative confusion required")
    actual, emitted, correct = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    n = cm.sum()
    recalls = [float(correct[k]/actual[k]) if actual[k] else None for k in range(3)]
    precision = [float(correct[k]/emitted[k]) if emitted[k] else None for k in range(3)]
    f1 = np.divide(2*correct, actual+emitted, out=np.zeros(3), where=actual+emitted > 0)
    denominator = np.sqrt(max(0, n*n-np.dot(actual, actual))*max(0, n*n-np.dot(emitted, emitted)))
    mcc = float((np.trace(cm)*n-np.dot(actual, emitted))/denominator) if denominator else None
    directional_calls, actual_moves = emitted[0]+emitted[2], actual[0]+actual[2]
    direction_correct = correct[0]+correct[2]
    return {"n": int(n), "accuracy": float(correct.sum()/n) if n else None,
            "macro_f1": float(f1.mean()), "macro_recall_all_three": float(np.mean(recalls)) if all(v is not None for v in recalls) else None,
            "recall_down_neutral_up": recalls, "precision_down_neutral_up": precision,
            "f1_down_neutral_up": f1.tolist(), "mcc": mcc,
            "actual_down_neutral_up": actual.astype(int).tolist(), "predicted_down_neutral_up": emitted.astype(int).tolist(),
            "actual_neutral_fraction": float(actual[1]/n) if n else None,
            "directional_call_fraction": float(directional_calls/n) if n else None,
            "directional_precision_including_false_moves": float(direction_correct/directional_calls) if directional_calls else None,
            "correct_direction_fraction_of_actual_moves": float(direction_correct/actual_moves) if actual_moves else None,
            "all_three_true_states_present": bool(np.all(actual > 0)), "confusion_down_neutral_up": cm.astype(int).tolist()}


def metrics(truth, predicted):
    return from_confusion(confusion(truth, predicted))


def daily_confusion(times, truth, predicted, start, days):
    day = (np.asarray(times)-start)//DAY
    if np.any(day < 0) or np.any(day >= days): raise ValueError("decision outside registered days")
    code = day*9+(np.asarray(truth, int)+1)*3+np.asarray(predicted, int)+1
    return np.bincount(code, minlength=days*9).reshape(days, 3, 3)


def calendar_weights(bank, days):
    bank = np.asarray(bank, int)
    return np.stack([np.bincount(row, minlength=days) for row in bank])


def bootstrap_metrics(daily, weights):
    cm = (weights @ np.asarray(daily).reshape(len(daily), 9)).reshape(-1, 3, 3)
    diag = np.diagonal(cm, axis1=1, axis2=2)
    actual, emitted = cm.sum(axis=2), cm.sum(axis=1)
    f1 = np.divide(2*diag, actual+emitted, out=np.zeros_like(diag, dtype=float), where=actual+emitted > 0)
    return {"accuracy": diag.sum(axis=1)/cm.sum(axis=(1, 2)), "macro_f1": f1.mean(axis=1)}


def direct_checks():
    checks = []
    r = np.array([-5.1, -5., -.5, 0., .5, 5., 5.1])
    assert labels(r, 5.).tolist() == [-1, 0, 0, 0, 0, 0, 1]
    assert labels(r, .5).tolist() == [-1, -1, 0, 0, 0, 1, 1]
    checks.append("percentage_units_and_inclusive_band_edges")
    assert labels(np.array([-.0001, 0, .0001]), 0).tolist() == [-1, 0, 1]
    checks.append("exact_zero_preserves_direction")
    train = np.array([[.01, -.2], [-.03, .4], [.05, .6]])
    assert np.allclose(train_bands(train), [.03, .4])
    changed = np.vstack([train, [[100., 100.]]])
    assert np.array_equal(train_bands(train), train_bands(changed[:3]))
    checks.append("per_asset_train_only_band")
    y, p = np.array([-1, 0, 1, 1]), np.array([-1, 0, 0, -1])
    m = metrics(y, p)
    assert m["accuracy"] == .5 and m["actual_down_neutral_up"] == [1, 1, 2]
    assert np.isclose(m["macro_f1"], (2/3+2/3)/3)
    checks.append("hand_recomputed_three_class_metrics")
    constant = metrics(np.zeros(20, int), np.zeros(20, int))
    assert constant["accuracy"] == 1 and constant["mcc"] is None
    assert constant["macro_recall_all_three"] is None and np.isclose(constant["macro_f1"], 1/3)
    checks.append("all_neutral_not_misreported_as_perfect_three_state_skill")
    t = np.array([1, 2, DAY+1, DAY+2])
    daily = daily_confusion(t, y, p, 0, 2)
    assert np.array_equal(daily.sum(axis=0), confusion(y, p))
    weights = calendar_weights(np.array([[0, 1], [1, 0]]), 2)
    boot = bootstrap_metrics(daily, weights)
    assert np.array_equal(boot["accuracy"], [.5, .5])
    checks.append("shared_calendar_bootstrap_matches_hand_accuracy")
    return checks
