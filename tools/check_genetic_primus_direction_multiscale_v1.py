"""Independent forecast-only verification. No fitting or trading modules."""
from __future__ import annotations
import run_genetic_primus_direction_multiscale_v1 as b
import argparse
import numpy as np
import joblib
from pathlib import Path
from scipy.special import expit, ndtr


def check(minutes):
    reg, bank, path = b.registration(minutes), b.load_bank(minutes), b.tf_path(minutes)
    prep = b.sealed(path / "preparation.json")
    counts = {"models": 0, "scaler_checks": 0, "prediction_cells": 0, "fold_checks": 0,
              "independent_metric_rows": 0, "confusion_checks": 0, "independent_linear_prediction_cells": 0}
    warnings = {}
    for i, fold in enumerate(prep["folds"]):
        with np.load(path / f"fold_{i:02d}.npz", allow_pickle=False) as stream:
            train, test = stream["train"], stream["test"]
        candidates = np.flatnonzero((bank["times"][:-1] >= fold["origin_ms"] - 30 * b.d.DAY)
            & (bank["times"][:-1] >= bank["times"][48]) & (bank["times"][1:] <= fold["cutoff_ms"]))
        expected = np.sort(np.random.default_rng(np.random.SeedSequence([20261012, minutes, i])).choice(
            candidates, min(4000, len(candidates)), replace=False))
        assert np.array_equal(train, expected)
        assert bank["times"][train[-1] + 1] <= fold["origin_ms"] - 7 * minutes * b.d.MINUTE
        assert len(test) == fold["test_origins"]
        counts["fold_checks"] += 3
        x, r = np.asarray(bank["features"][train]), np.asarray(bank["returns"][train])
        assert b.m.array_identity(x, r) == fold["train_data_sha256"]
        x, r = x.reshape(-1, 12), r.reshape(-1)
        xx = np.asarray(bank["features"][test]).reshape(-1, 12)
        receipt = b.sealed(path / f"origin_{i:02d}_receipt.json")
        assert receipt["registration_sha256"] == reg["sha256"]
        for name in b.d.models.MODELS:
            meta = b.sealed(path / f"origin_{i:02d}_{name}.json")
            model_path, pred_path = path / f"origin_{i:02d}_{name}.joblib", path / f"origin_{i:02d}_{name}.npz"
            assert b.pin(model_path) == meta["model_pin"] and b.pin(pred_path) == meta["prediction_pin"]
            assert receipt["fits"][name]["sha256"] == meta["sha256"]
            assert meta["fold_sha256"] == b.m.identity(fold) and meta["cutoff_ms"] == fold["cutoff_ms"]
            fitted = joblib.load(model_path)
            assert b.plain(fitted["info"]) == meta["info"]
            info = fitted["info"]; n = len(info["features"])
            selected = x[r != 0, :n]
            scale = selected.std(axis=0); scale[scale < 1e-12] = 1
            assert np.array_equal(selected.mean(axis=0), info["mean"])
            assert np.array_equal(scale, info["scale"])
            assert b.m.array_identity(selected, r[r != 0]) == info["training_sha256"]
            counts["scaler_checks"] += 3
            with np.load(pred_path, allow_pickle=False) as stream:
                saved = stream["probability"]
                assert np.array_equal(stream["times"], bank["times"][test])
                assert np.array_equal(saved, b.d.models.predict(fitted, xx).reshape(len(test), 8))
                if name in ("ridge6", "logistic6", "logistic12", "sgd_logistic12"):
                    z = (xx[:, :n] - info["mean"]) / info["scale"]
                    if name == "ridge6":
                        coef = np.array(info["coefficients"])
                        expected_p = ndtr(((z @ coef[:-1] + coef[-1]) * info["y_scale"] + info["y_mean"]) / info["residual_scale"])
                    else:
                        estimator = fitted["estimator"]
                        assert np.array_equal(estimator.classes_, np.array([0, 1]))
                        expected_p = expit(z @ estimator.coef_[0] + estimator.intercept_[0])
                    assert np.allclose(saved.reshape(-1), np.clip(expected_p, 1e-8, 1 - 1e-8), atol=1e-12, rtol=1e-12)
                    counts["independent_linear_prediction_cells"] += saved.size
                counts["prediction_cells"] += saved.size
            for warning in info.get("warnings", []):
                key = name + ":" + warning.split(":")[0]; warnings[key] = warnings.get(key, 0) + 1
            counts["models"] += 1
    ix = np.flatnonzero((bank["times"][:-1] >= b.START) & (bank["times"][:-1] < b.END))
    expected_times = np.arange(b.START, b.END, minutes * b.d.MINUTE)
    assert np.array_equal(bank["times"][ix], expected_times)
    current, future = np.asarray(bank["closes"][ix]), np.asarray(bank["closes"][ix + 1])
    sign = np.sign(future - current).astype(np.int8)
    scores = b.sealed(path / "scores.json")
    by_key = {(row["model"], row["symbol"]): row for row in scores["rows"]}
    for name in b.d.models.MODELS + b.CONTROLS:
        clocks, p = b.predictions(minutes, name)
        predicted = np.where(p >= 0.5, 1, -1)
        equal = predicted == sign  # Zero can never equal either binary prediction.
        for a, symbol in enumerate(b.SYMBOLS):
            row = by_key[name, symbol]
            count = int(np.count_nonzero(equal[:, a])); nonflat = int(np.count_nonzero(sign[:, a]))
            assert count == row["n_correct"] and nonflat == row["n_nonflat"]
            assert row["n_all"] == len(clocks) and row["n_flat"] == len(clocks) - nonflat
            assert row["accuracy_all"] == count / len(clocks) and row["accuracy_nonflat"] == count / nonflat
            cm = np.bincount(3 * (predicted[:, a] == 1).astype(int) + sign[:, a] + 1, minlength=6).reshape(2, 3)
            actual = row["confusion"]
            assert [actual[k] for k in ("down_down", "flat_down", "up_down", "down_up", "flat_up", "up_up")] == cm.reshape(-1).tolist()
            counts["confusion_checks"] += 6
            for month, mask in (("2026-08", clocks < b.START + 31 * b.d.DAY), ("2026-09", clocks >= b.START + 31 * b.d.DAY)):
                saved = row["by_month"][month]
                assert saved["n_correct"] == int(equal[mask, a].sum())
                assert saved["n_nonflat"] == int(np.count_nonzero(sign[mask, a]))
            counts["independent_metric_rows"] += 1
    coins = np.load(path / "random_coins.npy", allow_pickle=False)
    for a in range(8):
        expected = np.random.default_rng(np.random.SeedSequence([50521, minutes, a])).integers(0, 2, len(ix))
        assert np.array_equal(coins[:, a], expected)
    oracle_ceiling = {symbol: float(np.count_nonzero(sign[:, a]) / len(ix)) for a, symbol in enumerate(b.SYMBOLS)}
    result = b.write(path / "independent_checks.json", {"registration_sha256": reg["sha256"], "checker_pin": b.pin(Path(__file__)),
        "timeframe_minutes": minutes, "checks": counts, "warnings": warnings, "binary_all_interval_oracle_ceiling": oracle_ceiling,
        "random_seed_reproduced_per_asset": True, "price_sign_checked_without_metric_function": True,
        "fits": 0, "financial_replays": 0, "network": False})
    print({"tf_min": minutes, "models_verified": counts["models"], "prediction_cells": counts["prediction_cells"],
           "metric_rows": counts["independent_metric_rows"], "refits": 0, "passed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--minutes", type=int, required=True)
    check(parser.parse_args().minutes)
