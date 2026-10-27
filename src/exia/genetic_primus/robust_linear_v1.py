"""Fixed, bounded, symmetric Huber linear forecast; no runtime authority."""
from __future__ import annotations
import numpy as np
from . import cadence_matrix_v1 as c


def huber_values(residual, delta):
    values = np.asarray(residual, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(delta) or delta <= 0:
        raise ValueError("finite residuals and positive Huber delta required")
    absolute = np.abs(values)
    return np.where(absolute <= delta, .5 * values ** 2, delta * (absolute - .5 * delta))


def objective(design, target, parameters, delta, l2):
    residual = target - design @ parameters
    return float(huber_values(residual, delta).mean() + .5 * l2 * (parameters[:6] @ parameters[:6]))


def residual_multipliers(residual, delta):
    absolute = np.abs(residual)
    result = np.ones_like(absolute)
    outside = absolute > delta
    result[outside] = delta / absolute[outside]
    return result


def fit_huber(x, y, initial, *, delta=1.345, l2=.01, max_iterations=100,
              parameter_tolerance=1e-10, gradient_tolerance=1e-8):
    """Minimize mean Huber + lambda/2 ||slopes||^2, intercept unpenalized.

    IRLS weights are uniform base 1/N times min(1, delta/abs(residual));
    do not renormalize effective weights, which would change regularization.
    """
    x, y, parameters = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(initial, dtype=float).copy()
    if (x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),) or len(x) < 8
            or parameters.shape != (7,) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(parameters).all() or not np.isfinite(l2) or l2 <= 0
            or type(max_iterations) is not int or not 1 <= max_iterations <= 100
            or not 0 < parameter_tolerance <= 1e-6 or not 0 < gradient_tolerance <= 1e-6):
        raise ValueError("bounded finite seven-parameter regression required")
    huber_values(y, delta)
    design = np.column_stack((x, np.ones(len(x))))
    penalty = np.diag([l2] * 6 + [0.])
    initial_objective = objective(design, y, parameters, delta, l2)
    history = [initial_objective]
    converged = False
    for iteration in range(1, max_iterations + 1):
        residual = y - design @ parameters
        robust = residual_multipliers(residual, delta)
        effective = robust / len(y)
        # This solve is one IRLS update, not a new candidate or training origin.
        updated = np.linalg.solve(design.T @ (effective[:, None] * design) + penalty,
                                  design.T @ (effective * y))
        value = objective(design, y, updated, delta, l2)
        if not np.isfinite(updated).all() or value > history[-1] + 1e-12 * max(1., abs(history[-1])):
            raise ValueError("Huber solver nonfinite or objective increased")
        step = float(np.max(abs(updated - parameters)))
        parameters = updated
        history.append(value)
        residual = y - design @ parameters
        gradient = -design.T @ np.clip(residual, -delta, delta) / len(y) + penalty @ parameters
        gradient_norm = float(np.max(abs(gradient)))
        if step <= parameter_tolerance and gradient_norm <= gradient_tolerance:
            converged = True
            break
    if not converged:
        raise ValueError("bounded Huber solver did not converge; no model published")
    robust = residual_multipliers(residual, delta)
    if np.any(robust <= 0) or np.any(robust > 1):
        raise ValueError("outcomes must retain positive weights")
    return {"parameters": parameters, "robust_weights": robust, "solver": {
        "iterations": iteration, "converged": True, "initial_objective": initial_objective,
        "final_objective": history[-1], "objective_history": history,
        "last_parameter_step_inf": step, "gradient_inf": gradient_norm,
        "effective_weight_sum": float(robust.mean()), "weights_renormalized": False}}


def train_origin(bank, fit_ms, control, *, feature_prefix_sha):
    temporal = c.old.split(bank, fit_ms)
    train, inner, validation = (temporal[k] for k in ("train", "inner", "validation"))
    tx, ty = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
    if control["fit_ms"] != fit_ms:
        raise ValueError("same-origin frozen control required")
    scaler = c.m.fit_scaler(tx, ty)
    if scaler.payload() != control["model"]["scaler"] or scaler.sha256 != control["scaler_sha256"]:
        raise ValueError("new origin scaler must exactly match the control")
    identity = dict(control["training_identity"])
    if (identity["cutoff_ms"] != temporal["cutoff_ms"]
            or identity["train_samples_sha256"] != c.m.array_identity(bank["origins"][train], bank["available"][train], tx, ty)
            or identity["folds_sha256"] != c.m.array_identity(bank["origins"][inner], bank["origins"][validation])):
        raise ValueError("same labels, cutoff and folds required")
    result = fit_huber(scaler.transform(tx), scaler.target(ty), control["model"]["parameters"])
    robust = result["robust_weights"]
    identity.update(loss="mean_Huber_plus_half_l2_slopes", huber_delta_train_target_std=1.345,
                    robust_weights_sha256=c.m.array_identity(robust), effective_weights_renormalized=False,
                    initialization_model_sha256=control["model"]["sha256"])
    model = c.m.model_payload("linear6", result["parameters"], scaler, weighting="uniform", seed=None,
                              epochs=None, training_identity=identity)
    model["target_kind"] = control["model"]["target_kind"]
    model["loss"] = {"name": "Huber", "delta_train_target_std": 1.345, "symmetric": True,
                     "base_weights": "uniform_all_outcomes", "effective_weights_renormalized": False}
    model["sha256"] = c.m.identity({k: v for k, v in model.items() if k != "sha256"})
    normalized_for_diagnostics = robust / robust.sum()
    origin_weights = robust.reshape(len(train), bank["x"].shape[1]).sum(axis=1)
    return {"schema": "robust_linear_fit/1", "fit_ms": fit_ms,
        "assumed_model_ready_us": fit_ms * 1000 + 100000, "model": model, "scaler_sha256": scaler.sha256,
        "training_identity": identity, "feature_prefix_sha256": feature_prefix_sha,
        "train_origins": len(train), "train_asset_rows": len(ty),
        "latest_mature_label_ms": int(bank["available"][train[-1]]),
        "solver": result["solver"], "robust_weight_sha256": c.m.array_identity(robust),
        "weight_diagnostics": {"min_multiplier": float(robust.min()), "max_multiplier": float(robust.max()),
            "downweighted_fraction": float(np.mean(robust < 1)), "mean_multiplier": float(robust.mean()),
            "positive_outcome_normalized_mass": float(normalized_for_diagnostics[ty > 0].sum()),
            "negative_outcome_normalized_mass": float(normalized_for_diagnostics[ty < 0].sum()),
            "origin_kish_ess_not_overlap_adjusted": float(origin_weights.sum() ** 2 / (origin_weights @ origin_weights)),
            "normalization_for_diagnostics_only": True},
        "huber_threshold_raw_bps": 1.345 * scaler.target_scale * 10000,
        "training_fits": 1, "scaler_fits": 1, "runtime_authority": False, "safety": c.SAFETY}


def direct_checks():
    labels = []
    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        labels.append(label)
    check(np.array_equal(huber_values([-2, 0, 2], 1), [1.5, 0, 1.5]), "Huber piecewise values")
    check(np.array_equal(huber_values([-3, -1, 1, 3], 1), huber_values([3, 1, -1, -3], 1)), "symmetric loss")
    rng = np.random.default_rng(8127)
    x = rng.normal(size=(160, 6))
    y = x @ np.array([.1, -.2, .05, 0, .1, 0]) + .02
    ridge = c.m.fit_linear(x, y, np.ones(len(y)), l2=.01)
    result = fit_huber(x, y, ridge, delta=100)
    check(np.allclose(result["parameters"], ridge, rtol=1e-10, atol=1e-10), "Huber quadratic region equals ridge")
    check(np.all(result["robust_weights"] == 1), "quadratic weights all one")
    outliers = y.copy()
    outliers[0], outliers[1] = 20, -20
    start = c.m.fit_linear(x, outliers, np.ones(len(y)), l2=.01)
    robust = fit_huber(x, outliers, start)
    check(robust["solver"]["converged"], "bounded deterministic convergence")
    check(robust["solver"]["gradient_inf"] <= 1e-8, "Huber stationarity")
    check(np.all(np.diff(robust["solver"]["objective_history"]) <= 1e-12), "monotone objective")
    check(np.all(robust["robust_weights"] > 0), "no outcome excluded")
    check(np.all(robust["robust_weights"][:2] < 1), "both positive and negative outliers downweighted")
    check(np.linalg.norm(robust["parameters"] - ridge) < np.linalg.norm(start - ridge), "toy outlier influence reduced")
    duplicate = fit_huber(np.repeat(x, 2, axis=0), np.repeat(outliers, 2), start)
    check(np.allclose(duplicate["parameters"], robust["parameters"], rtol=1e-10, atol=1e-10), "duplicate rows do not change objective scale")
    reverse = fit_huber(x[::-1], outliers[::-1], start)
    check(np.allclose(reverse["parameters"], robust["parameters"], rtol=1e-10, atol=1e-10), "row order invariance")
    zero_start = fit_huber(x, outliers, np.zeros(7))
    check(np.allclose(zero_start["parameters"], robust["parameters"], rtol=1e-8, atol=1e-9), "initialization not separate search")
    # Verify fixed effective weights, without accidental renormalization, satisfy the stored solution.
    design = np.column_stack((x, np.ones(len(y))))
    w = robust["robust_weights"] / len(y)
    gradient = design.T @ (w * (design @ robust["parameters"] - outliers)) + np.r_[.01 * robust["parameters"][:6], 0]
    check(np.max(abs(gradient)) <= 1e-8, "fixed-base IRLS normal equation")
    check(robust["solver"]["effective_weight_sum"] < 1, "effective weights not renormalized")
    try:
        fit_huber(x, outliers, start, max_iterations=1)
    except ValueError:
        check(True, "fail closed on nonconvergence")
    else:
        raise AssertionError("nonconverged model accepted")
    try:
        huber_values([1], 0)
    except ValueError:
        check(True, "reject zero delta")
    else:
        raise AssertionError("bad delta accepted")
    return {"checks_passed": len(labels), "labels": labels, "real_data_fits": 0,
            "synthetic_only": True, "network_used": False, "collector_used": False, "safety": c.SAFETY}
