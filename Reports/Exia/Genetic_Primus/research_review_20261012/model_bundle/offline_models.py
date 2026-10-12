"""Research-only numerical inference. No orders, I/O, networking or fitting.

Inputs must be supplied explicitly. This is not an adapter to the trading bot.
Timestamps are UTC close/availability boundaries, never candle open times.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

DAY_MS = 86_400_000
BAR_MS = 14_400_000
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT",
           "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
FEATURES = ("distance_ema48", "return_18", "efficiency_18",
            "relative_return_18", "volatility_48", "breadth_ema48")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def verify_model(model):
    if model.get("sha256") != identity({k: v for k, v in model.items() if k != "sha256"}):
        raise ValueError("model SHA mismatch")
    if (model.get("architecture") != "linear6" or model.get("parameter_count") != 7
            or model.get("features") != list(FEATURES)
            or model.get("runtime_authority") is not False
            or model.get("promotion_authority") is not False):
        raise ValueError("only frozen research linear models are accepted")
    if identity(model["scaler"]) != model["scaler_sha256"]:
        raise ValueError("scaler SHA mismatch")
    p = np.asarray(model["parameters"], float)
    s = model["scaler"]
    mean, scale = np.asarray(s["mean"], float), np.asarray(s["scale"], float)
    if (p.shape != (7,) or mean.shape != (6,) or scale.shape != (6,)
            or not np.isfinite(np.r_[p, mean, scale, s["target_mean"], s["target_scale"], s["clip"]]).all()
            or np.any(scale <= 0) or s["target_scale"] <= 0 or s["clip"] <= 0):
        raise ValueError("invalid parameters or scaler")


def predict(model, features):
    """Gross three-day forecast, NOT expected net PnL or an order."""
    verify_model(model)
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError("finite rows by six features required")
    s = model["scaler"]
    x = np.clip((x - np.asarray(s["mean"])) / np.asarray(s["scale"]), -s["clip"], s["clip"])
    p = np.asarray(model["parameters"])
    return (x @ p[:6] + p[6]) * s["target_scale"] + s["target_mean"]


def latest_features(close_times_ms, closes, *, asof_ms, history_anchor_ms, symbols):
    """Full frozen-history EMA seed is required; truncated history is rejected.

    Only the latest completed row is returned. Caller owns source qualification
    and truthful availability timestamps. Checks cannot prove input authenticity.
    """
    t, p = np.asarray(close_times_ms), np.asarray(closes, float)
    if (tuple(symbols) != SYMBOLS or t.ndim != 1 or t.dtype.kind not in "iu"
            or len(t) < 49 or p.shape != (len(t), 8)
            or t[0] != history_anchor_ms or t[-1] != asof_ms
            or np.any(t < 0) or np.any(t % BAR_MS) or np.any(np.diff(t) != BAR_MS)
            or not np.isfinite(p).all() or np.any(p <= 0)):
        raise ValueError("complete positive as-of full8 history with registered anchor required")
    ema = p[0].copy()
    for row in p[1:]:
        ema = ema + (2 / 49) * (row - ema)
    r = p[-1] / p[-19] - 1
    den = np.abs(np.diff(p[-19:], axis=0)).sum(axis=0)
    efficiency = np.divide(abs(p[-1] - p[-19]), den, out=np.zeros(8), where=den > 0)
    vol = np.diff(np.log(p[-49:]), axis=0).std(axis=0)
    return np.stack((p[-1] / ema - 1, r, efficiency, r - r.mean(), vol,
                     np.repeat((p[-1] > ema).mean(), 8)), axis=-1)


def lagged_covariance(daily_times_ms, daily_closes, *, decision_ms):
    """Exactly 61 daily closes, ending one full day before the decision."""
    t, p = np.asarray(daily_times_ms), np.asarray(daily_closes, float)
    expected = np.arange(decision_ms - 61 * DAY_MS, decision_ms, DAY_MS)
    if (t.dtype.kind not in "iu" or not np.array_equal(t, expected)
            or p.shape != (61, 8) or not np.isfinite(p).all() or np.any(p <= 0)):
        raise ValueError("complete strictly lagged daily history required")
    sample = np.cov(np.diff(np.log(p), axis=0), rowvar=False, ddof=1)
    covariance = .8 * sample + .2 * np.diag(np.diag(sample))
    covariance += np.diag(np.maximum(.002 ** 2 - np.diag(covariance), 0))
    return covariance


def research_targets(policy, features, covariance, *, model=None):
    """Diagnostic target weights only; no position transitions or execution.

    EMA_V and Passive_V are the independent baselines. Ridge_C_V is retained
    as a rejected-alpha reference. Huber_H3 is a separate 3-day policy without
    this controller and must not silently be converted into a new strategy.
    """
    x, cov = np.asarray(features, float), np.asarray(covariance, float)
    if (x.shape != (8, 6) or cov.shape != (8, 8)
            or not np.isfinite(x).all() or not np.isfinite(cov).all()
            or np.max(abs(cov - cov.T)) > 1e-10 or np.linalg.eigvalsh(cov).min() < -1e-10):
        raise ValueError("full8 features and finite PSD covariance required")
    if policy == "EMA_V":
        mask = x[:, 0] > 0
    elif policy == "Passive_V":
        mask = np.ones(8, bool)
    elif policy == "Ridge_C_V":
        mask = predict(model, x) > .002
    else:
        raise ValueError("unregistered research policy")
    weights = mask.astype(float) * .1
    risk = float(np.sqrt(max(0, weights @ cov @ weights)))
    if risk > .005:
        weights *= .005 / risk
    return weights
