"""Deterministic CPU-only predictors. No execution, networking or fit on import.

These forecast gross three-day outcomes, not net trading profits. Calling code
must own temporal eligibility, provenance and evaluation; runtime is out of scope.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

DAY_MS = 86_400_000
FEATURES = ("distance_ema48", "return_18", "efficiency_18",
            "relative_return_18", "volatility_48", "breadth_ema48")
SEEDS = (11, 29, 47)
PARAMETERS = {"linear6": 7, "mlp8": 65}
WEIGHTINGS = ("uniform", "half_life_90_anchor20")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_identity(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def temporal_weights(available_ms, cutoff_ms, mode):
    """One weight per origin; independent of outcome sign. Sum is one."""
    dates = np.asarray(available_ms)
    if dates.ndim != 1 or dates.size == 0 or dates.dtype.kind not in "iu":
        raise ValueError("nonempty integer origin availability required")
    if type(cutoff_ms) is not int or cutoff_ms < 0 or np.any(dates < 0):
        raise ValueError("nonnegative integer cutoff/times required")
    if np.any(dates > cutoff_ms) or len(np.unique(dates)) != len(dates):
        raise ValueError("unique mature origins required; future outcome rejected")
    if mode not in WEIGHTINGS:
        raise ValueError("unregistered weighting")
    ages = (cutoff_ms - dates.astype(np.float64)) / DAY_MS
    if mode == "uniform":
        return np.full(len(dates), 1.0 / len(dates))
    recency = np.exp2(-ages / 90.0)
    # Normalize each component BEFORE mixing. A raw 0.2 floor does not
    # preserve a 20% uniform mixture on histories much longer than 90 days.
    return 0.2 / len(dates) + 0.8 * recency / recency.sum()


def origin_ess(weights):
    value = np.asarray(weights, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("positive finite origin weights required")
    return float(value.sum() ** 2 / (value @ value))


def _x(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 6 or not len(result) or not np.isfinite(result).all():
        raise ValueError("finite nonempty six-feature matrix required")
    return result


def _target_weights(y, weights, size):
    target, w = np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    if target.shape != (size,) or w.shape != (size,):
        raise ValueError("target/weight shape mismatch")
    if not np.isfinite(target).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("finite targets and strictly positive weights required")
    return target, w / w.sum()


@dataclass(frozen=True)
class Scaler:
    mean: tuple
    scale: tuple
    target_mean: float
    target_scale: float
    training_sha256: str
    clip: float = 8.0

    def transform(self, x):
        return np.clip((_x(x) - np.asarray(self.mean)) / np.asarray(self.scale), -self.clip, self.clip)

    def target(self, y):
        return (np.asarray(y, dtype=float) - self.target_mean) / self.target_scale

    def payload(self):
        return {"schema": "compact_forecast_scaler/1", "mean": list(self.mean),
                "scale": list(self.scale), "target_mean": self.target_mean,
                "target_scale": self.target_scale, "clip": self.clip,
                "training_sha256": self.training_sha256,
                "fitting_rule": "unweighted_train_only_ddof0"}

    @property
    def sha256(self):
        return identity(self.payload())


def fit_scaler(x, y):
    values = _x(x)
    target, _ = _target_weights(y, np.ones(len(values)), len(values))
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    target_scale = float(target.std())
    return Scaler(tuple(values.mean(axis=0)), tuple(scale), float(target.mean()),
                  target_scale if target_scale >= 1e-12 else 1.0,
                  array_identity(values, target))


def initialize_mlp(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.normal(0, 0.15, 48), np.zeros(8),
                           rng.normal(0, 0.15, 8), np.zeros(1)))


def _forward(parameters, x, architecture):
    p = np.asarray(parameters, dtype=float)
    if architecture not in PARAMETERS or p.shape != (PARAMETERS[architecture],) or not np.isfinite(p).all():
        raise ValueError("architecture/parameter count mismatch")
    if architecture == "linear6":
        return x @ p[:6] + p[6], None
    hidden = np.tanh(x @ p[:48].reshape(6, 8) + p[48:56])
    return hidden @ p[56:64] + p[64], hidden


def loss_gradient(parameters, x, y, weights, l2, architecture):
    """Weighted mean squared error; L2 on coefficients, never on biases."""
    values = _x(x)
    target, w = _target_weights(y, weights, len(values))
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("finite nonnegative regularization required")
    p = np.asarray(parameters, dtype=float)
    prediction, hidden = _forward(p, values, architecture)
    error = prediction - target
    delta = 2 * w * error
    mask = np.ones(len(p))
    if architecture == "linear6":
        mask[6] = 0
        gradient = np.concatenate((values.T @ delta, [delta.sum()]))
    else:
        mask[48:56], mask[64] = 0, 0
        back = delta[:, None] * p[56:64] * (1 - hidden ** 2)
        gradient = np.concatenate(((values.T @ back).ravel(), back.sum(axis=0),
                                   hidden.T @ delta, [delta.sum()]))
    loss = float(w @ error ** 2 + l2 * ((mask * p) @ p))
    return loss, gradient + 2 * l2 * mask * p


def fit_linear(x, y, weights, *, l2=0.01):
    values = _x(x)
    target, w = _target_weights(y, weights, len(values))
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("finite nonnegative L2 required")
    design = np.column_stack((values, np.ones(len(values))))
    penalty = np.diag([l2] * 6 + [0.0])
    # lstsq also handles constant/collinear features without dropping samples.
    parameters = np.linalg.lstsq(design.T @ (w[:, None] * design) + penalty,
                                 design.T @ (w * target), rcond=None)[0]
    return parameters


def fit_mlp(x, y, weights, *, seed, epochs=120, l2=0.01, learning_rate=0.001,
            validation=None, patience=12):
    """Bounded full-batch Adam; no waits, minibatch RNG, workers or wall clock."""
    values = _x(x)
    target, w = _target_weights(y, weights, len(values))
    if type(epochs) is not int or not 1 <= epochs <= 120 or type(patience) is not int or patience < 1:
        raise ValueError("bounded epoch/patience required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive learning rate required")
    p = initialize_mlp(seed)
    moment, velocity = np.zeros(65), np.zeros(65)
    best, best_epoch, stale, best_loss = p.copy(), 1, 0, math.inf
    history = []
    if validation is not None:
        vx = _x(validation[0])
        vy, vw = _target_weights(validation[1], np.ones(len(vx)), len(vx))
    for epoch in range(1, epochs + 1):
        loss, grad = loss_gradient(p, values, target, w, l2, "mlp8")
        if not math.isfinite(loss) or not np.isfinite(grad).all():
            raise ValueError("nonfinite optimization; fail closed")
        moment = 0.9 * moment + 0.1 * grad
        velocity = 0.999 * velocity + 0.001 * grad ** 2
        p -= learning_rate * (moment / (1 - 0.9 ** epoch)) / (np.sqrt(velocity / (1 - 0.999 ** epoch)) + 1e-8)
        metric = float(vw @ (_forward(p, vx, "mlp8")[0] - vy) ** 2) if validation is not None else loss
        history.append({"epoch": epoch, "train_loss_before_update": loss, "validation_mse": metric if validation is not None else None})
        if metric < best_loss - 1e-12:
            best, best_epoch, best_loss, stale = p.copy(), epoch, metric, 0
        else:
            stale += 1
        if validation is not None and stale >= patience:
            break
    # Full-prefix refit is for exactly the chosen epochs, not in-sample best loss.
    chosen = best if validation is not None else p
    if not np.isfinite(chosen).all():
        raise ValueError("nonfinite fitted model")
    return chosen, {"epochs_run": epoch, "selected_epoch": best_epoch if validation is not None else epoch,
                    "history": history, "gradient_steps": epoch}


def model_payload(architecture, parameters, scaler, *, weighting, seed, epochs, training_identity):
    p = np.asarray(parameters, dtype=float)
    _forward(p, np.zeros((1, 6)), architecture)
    if weighting not in WEIGHTINGS:
        raise ValueError("unregistered weighting")
    model = {"schema": "compact_forecast_model/1", "architecture": architecture,
             "features": list(FEATURES), "parameter_count": PARAMETERS[architecture],
             "parameters": p.tolist(), "scaler": scaler.payload(), "scaler_sha256": scaler.sha256,
             "weighting": weighting, "seed": seed, "epochs": epochs,
             "training_identity": training_identity,
             "target_kind": "gross_open_open_forecast_not_net_profit",
             "runtime_authority": False, "promotion_authority": False}
    model["sha256"] = identity(model)
    return model


def predict(model, x):
    if model.get("sha256") != identity({k: v for k, v in model.items() if k != "sha256"}):
        raise ValueError("model SHA mismatch")
    if model.get("features") != list(FEATURES) or model.get("parameter_count") != PARAMETERS.get(model.get("architecture")):
        raise ValueError("feature/architecture mismatch")
    data = model["scaler"]
    if identity(data) != model["scaler_sha256"]:
        raise ValueError("scaler SHA mismatch")
    scaler = Scaler(tuple(data["mean"]), tuple(data["scale"]), data["target_mean"],
                    data["target_scale"], data["training_sha256"], data["clip"])
    return _forward(model["parameters"], scaler.transform(x), model["architecture"])[0] * scaler.target_scale + scaler.target_mean
