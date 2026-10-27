"""Bounded three-state forecasting hypotheses; no I/O, network or event loops."""
from __future__ import annotations
import warnings
import numpy as np
from . import direction_hypotheses_v1 as old
from . import direction_ternary_v1 as tri


def feature_rows(bank, indices, kind, minutes):
    if kind != "context20" or minutes != 240:
        return old.feature_rows(bank, indices, kind, minutes)
    ix = np.asarray(indices, int)
    if ix.min() < 42:
        raise ValueError("seven days of causal context required")
    closes = bank["closes"]
    c = np.asarray(closes[ix], float)
    rh, rd, rw = [c / closes[ix-lag] - 1 for lag in (1, 6, 42)]
    lr = np.diff(np.log(np.asarray(closes)), axis=0)
    prefix = np.vstack([np.zeros(8), np.cumsum(lr, axis=0)])
    square = np.vstack([np.zeros(8), np.cumsum(lr*lr, axis=0)])
    mean = (prefix[ix]-prefix[ix-6])/6
    vol = np.sqrt(np.maximum(0, (square[ix]-square[ix-6])/6-mean*mean))
    market = np.broadcast_to(rd.mean(axis=1, keepdims=True), rd.shape)
    phase = (bank["times"][ix] % tri.DAY) / tri.DAY * 2*np.pi
    sin, cos = [np.broadcast_to(v[:, None], rd.shape) for v in (np.sin(phase), np.cos(phase))]
    return np.concatenate([bank["features"][ix], np.stack([rh, rd, rw, vol, rd-market, market, sin, cos], axis=2)], axis=2)


def combine_hierarchy(move, up):
    move, up = np.asarray(move), np.asarray(up)
    return np.stack([move*(1-up), 1-move, move*up], axis=-1)


def fit_matrix(x, y, kind, config, seed, weights=None, quantiles=None, classes=(-1, 0, 1)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    x, y, classes = np.asarray(x, float), np.asarray(y), np.asarray(classes)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(y, classes).all():
        raise ValueError("finite causal features and declared training classes required")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if w.shape != y.shape or np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("positive finite weights for every outcome required")
    w = w / w.mean()
    lower, upper = np.quantile(x, quantiles, axis=0) if quantiles else (None, None)
    z = np.clip(x, lower, upper) if quantiles else x
    mean, scale = z.mean(axis=0), z.std(axis=0)
    scale[scale < 1e-12] = 1
    counts = np.array([(y == state).sum() for state in classes])
    prior = (counts+1)/(len(y)+len(classes))
    info = {"rows": len(y), "features": x.shape[1], "kind": kind, "classes": classes.tolist(),
            "train_counts": counts.tolist(), "weight_mass": [float(w[y == s].sum()) for s in classes],
            "kish_weight_ess_not_independent_observations": float(w.sum()**2/(w*w).sum()),
            "warnings": [], "new_fits": 0, "fallback": bool(counts.min() < config["minimum_train_rows_per_class"])}
    estimator = None
    if not info["fallback"]:
        if kind == "logistic":
            estimator = LogisticRegression(**config["logit"], random_state=seed)
        elif kind == "boosting":
            estimator = HistGradientBoostingClassifier(**config["boost"], random_state=seed)
        elif kind == "mlp":
            args = {k: v for k, v in config["mlp"].items() if k != "nominal_parameters"}
            args["hidden_layer_sizes"] = tuple(args["hidden_layer_sizes"])
            estimator = MLPClassifier(**args, random_state=seed)
        else:
            raise ValueError("registered estimator required")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit((z-mean)/scale, y, sample_weight=w)
        if not np.array_equal(estimator.classes_, classes):
            raise ValueError("class order changed")
        info.update(new_fits=1, warnings=[str(v.message)[:240] for v in caught],
                    fitted_iterations=np.asarray(estimator.n_iter_).tolist())
        if kind == "logistic":
            info["parameter_count"] = int(estimator.coef_.size+estimator.intercept_.size)
        if kind == "mlp":
            info["parameter_count"] = sum(a.size for a in estimator.coefs_+estimator.intercepts_)
    return dict(estimator=estimator, mean=mean, scale=scale, lower=lower, upper=upper, prior=prior, info=info)


def probability_matrix(model, x):
    x = np.asarray(x, float)
    if model["estimator"] is None:
        return np.broadcast_to(model["prior"], (len(x), len(model["prior"]))).copy()
    if model["lower"] is not None:
        x = np.clip(x, model["lower"], model["upper"])
    return model["estimator"].predict_proba((x-model["mean"])/model["scale"])


def fit_variant(x, y, variant, config, seed, times, origin):
    flat, truth = np.asarray(x).reshape(len(x)*8, -1), np.asarray(y).reshape(-1)
    if variant["estimator"] == "hierarchy":
        moves = truth != 0
        move = fit_matrix(flat, moves.astype(int), "logistic", config, seed, classes=(0, 1))
        direction = fit_matrix(flat[moves], (truth[moves] > 0).astype(int), "logistic", config, seed+1, classes=(0, 1))
        return dict(kind="hierarchy", move=move, direction=direction,
                    info={"new_fits": move["info"]["new_fits"]+direction["info"]["new_fits"],
                          "stages": [move["info"], direction["info"]],
                          "warnings": move["info"]["warnings"]+direction["info"]["warnings"],
                          "parameter_count": sum(m["info"].get("parameter_count", 0) for m in (move, direction))})
    weights = np.repeat(old.recency_weights(times, origin, variant["half_life_days"]), 8) if "half_life_days" in variant else None
    payload = fit_matrix(flat, truth, variant["estimator"], config, seed, weights, variant.get("winsor_quantiles"))
    return dict(kind="single", model=payload, info=payload["info"])


def probability(payload, x):
    flat = np.asarray(x).reshape(len(x)*8, -1)
    if payload["kind"] == "hierarchy":
        p = combine_hierarchy(probability_matrix(payload["move"], flat)[:, 1], probability_matrix(payload["direction"], flat)[:, 1])
    else:
        p = probability_matrix(payload["model"], flat)
    p = p.reshape(len(x), 8, 3)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=-1), 1, atol=1e-12):
        raise ValueError("three-state probability simplex required")
    return p


def direct_checks():
    checks = tri.direct_checks()
    assert np.allclose(combine_hierarchy(np.array([0, 1, .6]), np.array([.5, 1, .25])), [[0, 1, 0], [0, 0, 1], [.45, .4, .15]])
    checks.append("hierarchy_hand_probabilities_no_abstention")
    rng = np.random.default_rng(14)
    bank = {"times": np.arange(100)*240*60000, "closes": np.exp(rng.normal(0, .01, (100, 8)).cumsum(axis=0)), "features": rng.normal(size=(100, 8, 12))}
    ix = np.array([50, 51, 52]); before = feature_rows(bank, ix, "context20", 240)
    bank["closes"][53:] *= 20; bank["features"][53:] += 100
    assert np.array_equal(before, feature_rows(bank, ix, "context20", 240))
    for kind, count in [("base6", 6), ("base12", 12), ("lag24", 24), ("asset19", 19), ("context20", 20)]:
        assert feature_rows(bank, ix, kind, 240).shape == (3, 8, count)
    checks.append("feature_dimensions_and_240m_future_suffix_invariance")
    w = old.recency_weights(np.array([0, tri.DAY, 2*tri.DAY]), 3*tri.DAY, 90)
    assert np.all(w > 0) and np.all(np.diff(w) > 0)
    checks.append("recency_positive_for_down_neutral_up")
    return checks
