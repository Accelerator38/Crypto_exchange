"""Bounded causal direction benchmark. Pure features, estimators and scoring."""
from __future__ import annotations

import warnings
import numpy as np
from scipy.special import ndtr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from . import compact_forecast_v1 as m

BAR = 14_400_000
DAY = 86_400_000
EXTRA_FEATURES = ["return_1", "candle_body", "candle_range", "upper_wick", "lower_wick", "log_volume_minus_previous_42_mean"]
FEATURES = list(m.FEATURES) + EXTRA_FEATURES
MODELS = ["ridge6", "logistic6", "logistic12", "gaussian_nb12", "lda12", "qda12", "tree12", "forest12", "extra_trees12", "hist_boost12", "sgd_logistic12", "mlp12"]


def feature_panel(closes, old_features, ohlcv):
    c = np.asarray(closes, float)
    a = np.asarray(ohlcv, float)
    if a.shape != (*c.shape, 5) or not np.array_equal(a[:, :, 3], c):
        raise ValueError("actual OHLCV and frozen closes must align exactly")
    o, h, lo, _, v = np.moveaxis(a, 2, 0)
    if (not np.isfinite(a).all() or np.any(a[:, :, :4] <= 0) or np.any(v < 0)
            or np.any(h < np.maximum(o, c)) or np.any(lo > np.minimum(o, c)) or np.any(h < lo)):
        raise ValueError("invalid OHLCV")
    previous_log_volume = np.zeros_like(c)
    lv = np.log1p(v)
    cumulative = np.concatenate([np.zeros((1, c.shape[1])), np.cumsum(lv, axis=0)])
    for i in range(42, len(c)):
        previous_log_volume[i] = (cumulative[i] - cumulative[i - 42]) / 42
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1
    extra = np.stack([ret, c / o - 1, (h - lo) / c, (h - np.maximum(o, c)) / c,
                      (np.minimum(o, c) - lo) / c, lv - previous_log_volume], axis=2)
    result = np.concatenate([old_features, extra], axis=2)
    if not np.isfinite(result[48:]).all():
        raise ValueError("finite completed features required")
    return result


def parameters(name, seed=20261012):
    common = {"random_state": seed}
    if name in ("logistic6", "logistic12"):
        return LogisticRegression(C=0.1, max_iter=120, tol=1e-6, solver="lbfgs", **common)
    if name == "gaussian_nb12":
        return GaussianNB(var_smoothing=1e-5)
    if name == "lda12":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "qda12":
        return QuadraticDiscriminantAnalysis(reg_param=0.2, tol=1e-8)
    if name == "tree12":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=128, **common)
    if name in ("forest12", "extra_trees12"):
        cls = RandomForestClassifier if name == "forest12" else ExtraTreesClassifier
        return cls(n_estimators=24, max_depth=4, min_samples_leaf=128, max_features=0.75, n_jobs=1, **common)
    if name == "hist_boost12":
        return HistGradientBoostingClassifier(max_iter=32, max_leaf_nodes=7, min_samples_leaf=128,
                                              learning_rate=0.08, l2_regularization=10, early_stopping=False, **common)
    if name == "sgd_logistic12":
        return SGDClassifier(loss="log_loss", alpha=0.001, max_iter=150, tol=1e-4, shuffle=False, **common)
    if name == "mlp12":
        return MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, batch_size=512, learning_rate_init=0.001,
                             max_iter=40, n_iter_no_change=40, tol=0.0, early_stopping=False, **common)
    raise ValueError("unregistered estimator")


def fit(name, x, returns):
    n = 6 if name.endswith("6") else 12
    x = np.asarray(x, float)[:, :n]
    returns = np.asarray(returns, float)
    nonflat = returns != 0
    x, returns = x[nonflat], returns[nonflat]
    y = (returns > 0).astype(np.int8)
    if len(x) < 1000 or len(np.unique(y)) != 2:
        raise ValueError("two classes and adequate mature labels required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    info = {"name": name, "features": FEATURES[:n], "mean": mean.tolist(), "scale": scale.tolist(),
            "rows": len(x), "up_fraction": float(y.mean()), "training_sha256": m.array_identity(x, returns)}
    if name == "ridge6":
        ym, ys = float(returns.mean()), float(returns.std())
        if ys < 1e-12:
            raise ValueError("nondegenerate returns required")
        coeff = m.fit_linear(z, (returns - ym) / ys, np.ones(len(z)) / len(z), l2=0.01)
        info.update({"coefficients": np.asarray(coeff).tolist(), "y_mean": ym, "y_scale": ys,
                     "parameter_count": 7, "probability_mapping": "Gaussian residual CDF, train-only residual scale"})
        pred = np.column_stack([z, np.ones(len(z))]) @ np.asarray(coeff)
        # compact fit_linear uses intercept last (asserted by direct checks).
        info["residual_scale"] = float(np.std(returns - (pred * ys + ym)))
        estimator = None
    else:
        estimator = parameters(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(z, y)
        info["warnings"] = [type(w.message).__name__ + ": " + str(w.message)[:220] for w in caught]
        info["hyperparameters"] = estimator.get_params()
        if name.startswith("logistic") or name.startswith("sgd"):
            info["parameter_count"] = n + 1
        elif name == "mlp12":
            info["parameter_count"] = sum(a.size for a in estimator.coefs_ + estimator.intercepts_)
            info["iterations"] = int(estimator.n_iter_)
        elif hasattr(estimator, "estimators_"):
            info["tree_nodes"] = sum(e.tree_.node_count for e in estimator.estimators_)
        elif hasattr(estimator, "tree_"):
            info["tree_nodes"] = int(estimator.tree_.node_count)
        else:
            info["parameterization"] = "class means/covariances or bounded boosting; not a neural parameter count"
    return {"info": info, "estimator": estimator}


def predict(fitted, x):
    info = fitted["info"]
    n = len(info["features"])
    z = (np.asarray(x, float)[:, :n] - info["mean"]) / info["scale"]
    if info["name"] == "ridge6":
        raw = np.column_stack([z, np.ones(len(z))]) @ np.asarray(info["coefficients"])
        expected = raw * info["y_scale"] + info["y_mean"]
        probability = ndtr(expected / max(info["residual_scale"], 1e-12))
    else:
        probability = fitted["estimator"].predict_proba(z)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("nonfinite prediction")
    return np.clip(probability, 1e-8, 1 - 1e-8)


def accuracy_summary(probability, returns):
    from sklearn.metrics import roc_auc_score
    p, r = np.asarray(probability, float), np.asarray(returns, float)
    valid = r != 0
    y, p = r[valid] > 0, p[valid]
    sign = p >= 0.5
    correct = sign == y
    up_recall, down_recall = float(sign[y].mean()), float((~sign[~y]).mean())
    return {"n": len(p), "flat": int((~valid).sum()), "coverage_nonflat": 1.0,
            "accuracy": float(correct.mean()), "balanced_accuracy": (up_recall + down_recall) / 2,
            "up_recall": up_recall, "down_recall": down_recall,
            "up_fraction": float(y.mean()), "predicted_up_fraction": float(sign.mean()),
            "auc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-y.astype(float) * np.log(p) - (~y).astype(float) * np.log1p(-p))),
            "signed_gross_return_bps": float(np.mean(np.where(sign, 1, -1) * r[valid]) * 10000),
            "confusion": {"true_up_pred_up": int((y & sign).sum()), "true_up_pred_down": int((y & ~sign).sum()),
                          "true_down_pred_up": int((~y & sign).sum()), "true_down_pred_down": int((~y & ~sign).sum())}}


def direct_checks():
    rng = np.random.default_rng(19)
    c = np.exp(np.cumsum(rng.normal(0, 0.01, (100, 2)), axis=0)) * 100
    o = c / 1.001
    a = np.stack([o, c * 1.002, o * 0.998, c, np.full_like(c, 100)], axis=2)
    old = rng.normal(size=(100, 2, 6))
    first = feature_panel(c, old, a)
    later_a, later_c, later_old = a.copy(), c.copy(), old.copy()
    later_a[70:] *= 2
    later_c[70:] *= 2
    later_old[70:] *= 2
    assert np.array_equal(first[:70], feature_panel(later_c, later_old, later_a)[:70])
    assert np.allclose(first[48:, :, 6], c[48:] / c[47:-1] - 1)
    x = rng.normal(size=(1100, 12))
    r = 0.001 * x[:, 0] + rng.normal(0, 0.01, 1100)
    model = fit("ridge6", x, r)
    assert np.shape(model["info"]["coefficients"]) == (7,)
    z = (x[:, :6] - model["info"]["mean"]) / model["info"]["scale"]
    expected = m.predict(m.model_payload("linear6", np.asarray(model["info"]["coefficients"]),
        m.fit_scaler(x[:, :6], r), weighting="uniform", seed=None, epochs=None, training_identity="synthetic"), x[:, :6])
    manual = (np.column_stack([z, np.ones(len(z))]) @ model["info"]["coefficients"]) * r.std() + r.mean()
    assert np.allclose(expected, manual)
    summary = accuracy_summary(np.array([0.9, 0.1, 0.6, 0.5]), np.array([1., -1., -1., 0.]))
    assert summary["n"] == 3 and summary["flat"] == 1 and summary["accuracy"] == 2 / 3
    return {"passed": 5, "collector_executed": False, "network": False}
