"""Independent recomputation of the revealed-day engineering ledger; no refitting."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_book_multiday_v1 as run

import joblib
import numpy as np


def main():
    c, reg = run.contract(), run.registered()
    entry = run.read(run.OUT / "available_days.json")["entries"][0]
    bank, audit = run.core.load_day(entry["bank_path"], c)
    directory = run.OUT / "known_day_engineering"
    receipt = run.read(directory / "receipt.json")
    results = run.read(run.OUT / "known_day_results.json")
    ledger_path = directory / "forecast_ledger.npz"
    assert run.core.file_sha(ledger_path) == receipt["ledger_sha256"]
    with np.load(ledger_path, allow_pickle=False) as archive:
        ledger = {k: archive[k].copy() for k in archive.files}
    t, start, eligible = bank["times"], audit["utc_start_ms"], audit["eligible"]
    train_end, origin = start+int(.6*run.core.DAY), start+int(.8*run.core.DAY)
    # Recompute eligibility/labels independently of fit_panel's saved ledger.
    valid = np.r_[eligible[:-1] & eligible[1:], False]
    train = np.flatnonzero(valid & (t+run.core.MINUTE < train_end-7*run.core.MINUTE))
    cal = np.flatnonzero(valid & (t >= train_end) & (t+run.core.MINUTE < origin-7*run.core.MINUTE))
    test = np.flatnonzero(valid & (t >= origin))
    assert np.array_equal(ledger["times"], t[test])
    assert np.all(t[train]+run.core.MINUTE < receipt["fold"]["train_label_cutoff_exclusive_ms"])
    assert np.all(t[cal]+run.core.MINUTE < receipt["fold"]["calibration_label_cutoff_exclusive_ms"])
    checks, model_receipts = [], []
    check_count = 0
    for j, (target, model_name) in enumerate(zip(ledger["targets"], ledger["models"])):
        target_no = c["targets"].index(str(target))
        prices = bank[str(target)]
        labels = np.zeros(len(t), dtype=np.int8)
        labels[:-1] = np.where(prices[1:] > prices[:-1], 1,
                              np.where(prices[1:] < prices[:-1], -1, 0))
        assert np.array_equal(ledger["labels"][target_no], labels[test])
        path = directory / (str(target)+"__"+str(model_name)+".joblib")
        expected_sha = str(ledger["model_sha256"][j])
        assert run.core.file_sha(path) == expected_sha == receipt["models"][j]["model_sha256"]
        model = joblib.load(path)
        selected = train[labels[train] != 0]
        features = bank["price" if str(model_name).startswith("price_") else "book"]
        expected_mean, expected_scale = features[selected].mean(axis=0), features[selected].std(axis=0)
        expected_scale[expected_scale < 1e-12] = 1
        assert np.allclose(model["mean"], expected_mean, rtol=1e-12, atol=1e-12)
        assert np.allclose(model["scale"], expected_scale, rtol=1e-12, atol=1e-12)
        probability = run.estimators.predict(model, features[test])
        assert np.array_equal(probability, ledger["probabilities"][j])
        assert not model["info"]["warnings"]
        for coverage in c["coverage_targets"]:
            threshold = run.selective.calibrate(run.estimators.predict(model, features[cal]), t[cal],
                                                c["asset"], coverage, c["selection_seed"])
            assert threshold == receipt["thresholds"][str(target)+"__"+str(model_name)][str(coverage)]
            mask = run.selective.emit(probability, t[test], c["asset"], threshold, c["selection_seed"])
            predicted = np.where(probability >= .5, 1, -1)
            correct = int(np.count_nonzero((predicted == labels[test]) & mask))
            row = next(r for r in results["rows"] if r["target"] == target and r["model"] == model_name
                       and r["coverage_target"] == coverage)
            assert row["correct"] == correct and row["n_emitted"] == int(mask.sum())
            assert row["flat_emitted"] == int(np.count_nonzero((labels[test] == 0) & mask))
            check_count += 1
        model_receipts.append({"target": str(target), "model": str(model_name), "sha256": expected_sha,
                               "recomputed_label_scaler_predictions_and_thresholds": True})
    checks.extend(["hand_recomputed_direction_and_flat_denominator", "strict_maturity_and_purge",
                   "all_8_model_hashes", "train_only_scalers", "forecast_reload_parity",
                   "all_32_calibration_thresholds_and_metrics", "no_estimator_warnings"])
    for target in c["targets"]:
        models = [joblib.load(directory / (target+"__"+name+".joblib")) for name in c["models"]]
        assert all(m["actual_train_times_index_sha256"] == models[0]["actual_train_times_index_sha256"] for m in models)
        assert np.array_equal(models[0]["mean"], models[2]["mean"][:6])
        assert np.array_equal(models[0]["scale"], models[2]["scale"][:6])
    checks.append("matched_feature_sets_share_training_times_and_price_scaler")
    assert not (run.OUT / "data_freeze.json").exists()
    try:
        run.freeze(run.OUT / "available_days.json")
        raise AssertionError("incomplete multiday dataset admitted")
    except ValueError as exc:
        assert "DATA_INSUFFICIENT" in str(exc)
    assert not (run.OUT / "data_freeze.json").exists()
    checks.append("one_day_cannot_freeze_multiday_test")
    assert all(c[k] is False for k in run.core.SAFETY)
    checks.append("all_seven_permissions_false")
    record = {"checks": checks, "metric_cells_recomputed": check_count, "models": model_receipts,
              "historical_refits": 0, "multiday_fits": 0, "status": "DATA_INSUFFICIENT",
              "known_days": 1, "required_days": 60,
              "checker_code_sha256": run.core.file_sha(Path(__file__)),
              "registration_sha256": run.core.file_sha(run.OUT / "registration.json"),
              "forecast_ledger_sha256": run.core.file_sha(ledger_path),
              "independent_validation": False, **{k: False for k in run.core.SAFETY}}
    run.write(run.OUT / "engineering_audit.json", record)
    print(json.dumps({"audit": "PASS", "models": 8, "metric_cells": check_count,
                      "checks": len(checks), "multiday_status": "DATA_INSUFFICIENT"}))


if __name__ == "__main__":
    main()
