"""Fixed horizon-dependent deadbands, true 3-class logit/boost, no execution path."""
from __future__ import annotations
import warnings
import numpy as np
from . import direction_ternary_v1 as ternary
from . import direction_selective_v1 as causal

SAFETY = ("credentials_allowed", "orders_enabled", "paper", "live", "runtime_authority", "network_allowed", "promotion_authority")


def band(config, minutes):
    if any(config[k] is not False for k in SAFETY): raise ValueError("unsafe permissions")
    expected = {1: .020, 10: .064, 30: .109, 240: .320}
    if minutes not in expected or config["neutral_bands_pct"][str(minutes)] != expected[minutes]:
        raise ValueError("unregistered horizon band")
    return expected[minutes]


def fold(times, origin, minutes, config, seed):
    width = minutes*60000
    train, cal = causal.split(times, origin, width, calibration_days=config["calibration_days"],
                              purge=config["purge_bars"], cap=config["train_time_cap"], seed=seed)
    train_cutoff = origin-config["calibration_days"]*ternary.DAY-config["purge_bars"]*width
    cal_cutoff = origin-config["purge_bars"]*width
    train = train[times[train+1] < train_cutoff]
    cal = cal[times[cal+1] < cal_cutoff]
    if not len(train) or not len(cal): raise ValueError("empty causal fold")
    return train, cal


def fit(x, labels, candidate, config, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if candidate not in config["candidates"]: raise ValueError("unregistered candidate")
    x, labels = np.asarray(x, float), np.asarray(labels, np.int8)
    if x.ndim != 3 or labels.shape != x.shape[:2] or not np.isfinite(x).all():
        raise ValueError("finite matching time/asset/features and labels required")
    if not np.isin(labels, ternary.STATES).all(): raise ValueError("unknown state")
    z, y = x.reshape(-1, x.shape[-1]), labels.ravel()
    mean, scale = z.mean(axis=0), z.std(axis=0); scale[scale < 1e-12] = 1
    counts = ternary.counts_per_asset(labels); pooled = counts.sum(axis=0)
    priors = (counts+1)/(len(labels)+3)
    model = {"candidate": candidate, "kind": "prior_only_sparse_classes", "estimator": None,
             "mean": mean, "scale": scale, "train_counts": counts, "train_priors": priors,
             "warnings": [], "coefficient_count": 0, "fitted_iterations": 0,
             "train_rows_including_neutral": len(y), "class_weight_mass": pooled.astype(float)}
    if pooled.min() < config["minimum_pooled_train_rows_per_state"]: return model
    balanced = candidate.endswith("_balanced")
    weight = len(y)/(3*pooled[y+1]) if balanced else np.ones(len(y))
    weight /= weight.mean()
    model["class_weight_mass"] = np.array([weight[y == state].sum() for state in ternary.STATES])
    if candidate.startswith("logit_"):
        estimator = LogisticRegression(C=config["logit"]["C"], max_iter=config["logit"]["max_iter"],
            tol=config["logit"]["tol"], random_state=seed)
    else:
        estimator = HistGradientBoostingClassifier(**{k: v for k, v in config["boost"].items() if k != "maximum_leaf_values"}, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit((z-mean)/scale, y, sample_weight=weight)
    if not np.array_equal(estimator.classes_, ternary.STATES): raise ValueError("class order mismatch")
    model.update(kind="three_class_"+candidate, estimator=estimator,
                 warnings=[str(w.message)[:240] for w in caught])
    if candidate.startswith("logit_"):
        model["coefficient_count"] = estimator.coef_.size+estimator.intercept_.size
        model["fitted_iterations"] = int(estimator.n_iter_.max())
    else:
        model["fitted_iterations"] = int(estimator.n_iter_)
        model["maximum_leaf_values"] = config["boost"]["maximum_leaf_values"]
    return model


def probabilities(model, x):
    return ternary.probability(model, x)


def predictions(ledger):
    """Candidates, five causal controls, then four explicitly legacy diagnostics."""
    p, u, prior = ledger["probability"], ledger["uniform"], ledger["train_priors"]
    cdf = prior.cumsum(axis=-1)
    shape = u.shape
    candidates = [ternary.predict(value) for value in p]
    controls = [np.zeros(shape, np.int8), np.broadcast_to(ternary.predict(prior), shape),
                ternary.STATES[np.minimum((u*3).astype(int), 2)],
                np.where(u < cdf[:, 0], -1, np.where(u < cdf[:, 1], 0, 1)).astype(np.int8),
                ternary.labels(ledger["past_return_pct"], float(ledger["band_pct"]))]
    return [*candidates, *controls, *ledger["legacy_predictions"]]


def direct_checks(config):
    checks = ternary.direct_checks()
    for minutes, threshold in [(1, .020), (10, .064), (30, .109), (240, .320)]:
        assert band(config, minutes) == threshold
        assert ternary.labels(np.array([-threshold*1.01, -threshold, 0, threshold, threshold*1.01]), threshold).tolist() == [-1, 0, 0, 0, 1]
    checks.append("all_four_exact_user_bands_and_inclusive_edges")
    times = np.arange(60*1440, dtype=np.int64)*60000
    for minutes in [1, 10, 30, 240]:
        t = times[::minutes]
        a, b = fold(t, 30*ternary.DAY, minutes, config, 17)
        assert np.all(t[a+1] < 23*ternary.DAY-7*minutes*60000)
        assert np.all(t[b+1] < 30*ternary.DAY-7*minutes*60000)
    checks.append("strict_cutoffs_for_all_four_horizons")
    return checks
