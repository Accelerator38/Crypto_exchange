"""Direction-only causal bars, features and independently auditable statistics."""
from __future__ import annotations
import numpy as np
from . import direction_benchmark_v1 as models

MINUTE = 60_000
DAY = 86_400_000


def aggregate(open_times, minute_ohlcv, minutes):
    times, a = np.asarray(open_times), np.asarray(minute_ohlcv)
    width = minutes * MINUTE
    if (minutes not in (1, 10, 30, 240) or len(times) % minutes or times[0] % width
            or np.any(np.diff(times) != MINUTE) or a.shape != (len(times), 8, 5)):
        raise ValueError("complete UTC-anchored real minute OHLCV required")
    blocks = a.reshape(-1, minutes, 8, 5)
    result = np.stack([blocks[:, 0, :, 0], blocks[:, :, :, 1].max(axis=1),
                       blocks[:, :, :, 2].min(axis=1), blocks[:, -1, :, 3], blocks[:, :, :, 4].sum(axis=1)], axis=2)
    return times[::minutes] + width, result


def rolling_sum(a, length):
    prefix = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
    return prefix[length:] - prefix[:-length]


def features(closes, ohlcv):
    c = np.asarray(closes, float)
    ema = np.empty_like(c); ema[0] = c[0]
    for i in range(1, len(c)):
        ema[i] = ema[i - 1] + (2 / 49) * (c[i] - ema[i - 1])
    ret = np.full_like(c, np.nan); ret[18:] = c[18:] / c[:-18] - 1
    eff = np.full_like(c, np.nan)
    denominator = rolling_sum(np.abs(np.diff(c, axis=0)), 18)
    eff[18:] = np.divide(np.abs(c[18:] - c[:-18]), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    log_return = np.diff(np.log(c), axis=0)
    volatility = np.full_like(c, np.nan)
    mu = rolling_sum(log_return, 48) / 48
    variance = rolling_sum(log_return ** 2, 48) / 48 - mu ** 2
    volatility[48:] = np.sqrt(np.maximum(0, variance))
    relative = ret - ret.mean(axis=1, keepdims=True)
    breadth = np.broadcast_to((c > ema).mean(axis=1, keepdims=True), c.shape)
    base = np.stack([c / ema - 1, ret, eff, relative, volatility, breadth], axis=2)
    result = models.feature_panel(c, base, ohlcv)
    result[:48] = np.nan
    if not np.isfinite(result[48:]).all():
        raise ValueError("finite causal post-warmup features required")
    return result


def metrics(predicted_up, current_price, next_price):
    p = np.asarray(predicted_up, bool)
    difference = np.asarray(next_price) - np.asarray(current_price)
    up, down, flat = difference > 0, difference < 0, difference == 0
    correct = (p & up) | (~p & down)
    n, n_nonflat = len(p), int((~flat).sum())
    tp, fn, tn, fp = int((p & up).sum()), int((~p & up).sum()), int((~p & down).sum()), int((p & down).sum())
    up_recall = tp / int(up.sum()) if up.any() else None
    down_recall = tn / int(down.sum()) if down.any() else None
    return {"n_all": n, "n_nonflat": n_nonflat, "n_up": int(up.sum()), "n_down": int(down.sum()), "n_flat": int(flat.sum()),
            "n_correct": int(correct.sum()), "accuracy_all": float(correct.mean()),
            "accuracy_nonflat": float(correct.sum() / n_nonflat) if n_nonflat else None,
            "balanced_accuracy": (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None,
            "up_recall": up_recall, "down_recall": down_recall, "predicted_up_fraction": float(p.mean()),
            "flat_fraction": float(flat.mean()), "confusion": {"up_up": tp, "up_down": fn, "down_down": tn, "down_up": fp,
                "flat_up": int((flat & p).sum()), "flat_down": int((flat & ~p).sum())}, "coverage": 1.0}


def sample_bank(days, repetitions, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(repetitions, int(np.ceil(days / block))))
    return ((starts[:, :, None] + np.arange(block)) % days).reshape(repetitions, -1)[:, :days]


def direct_checks():
    rng = np.random.default_rng(81)
    c = np.exp(np.cumsum(rng.normal(0, 0.001, (480, 8)), axis=0)) * 100
    o = c / 1.001
    a = np.stack([o, c * 1.002, o * 0.998, c, np.ones_like(c)], axis=2)
    t = np.arange(480, dtype=np.int64) * MINUTE
    for width in (1, 10, 30, 240):
        clock, bars = aggregate(t, a, width)
        assert np.array_equal(clock, np.arange(1, len(clock) + 1) * width * MINUTE)
        assert np.array_equal(bars[:, :, 3], c[width - 1::width])
    x = features(c, a)
    changed_a, changed_c = a.copy(), c.copy()
    changed_a[300:] *= 2; changed_c[300:] *= 2
    assert np.array_equal(x[48:300], features(changed_c, changed_a)[48:300])
    reference_std = np.lib.stride_tricks.sliding_window_view(np.diff(np.log(c), axis=0), 48, axis=0).std(axis=-1)
    assert np.allclose(x[48:, :, 4], reference_std, atol=1e-12, rtol=1e-9)
    first, last = np.array([1., 2., 3., 4., 5.]), np.array([2., 1., 3., 5., 4.])
    perfect = metrics(np.array([1, 0, 1, 1, 0]), first, last)
    inverse = metrics(np.array([0, 1, 0, 0, 1]), first, last)
    assert perfect["accuracy_all"] == 0.8 and perfect["accuracy_nonflat"] == 1
    assert inverse["accuracy_all"] == 0 and inverse["accuracy_nonflat"] == 0
    assert perfect["n_flat"] == 1 and perfect["confusion"]["flat_up"] == 1
    coins = np.random.default_rng(50521).integers(0, 2, 10000)
    assert 0.48 < coins.mean() < 0.52
    return {"assertions": 14, "fits": 0, "financial_replays": 0, "network": False}
