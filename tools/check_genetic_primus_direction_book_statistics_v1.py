"""Synthetic-only end-to-end check of shared-calendar statistical aggregation."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_book_multiday_v1 as run
import numpy as np


def main():
    real_out, real_registered = run.OUT, run.registered
    reg = real_registered()
    c = run.contract()
    fixture = real_out / "synthetic_statistics_fixture"
    fixture.mkdir()  # Never overwrite/reuse an earlier statistical test.
    with (fixture / "README_SYNTHETIC_ONLY.md").open("x", encoding="utf-8") as stream:
        stream.write("# SYNTHETIC FIXTURE ONLY\n\nNo market data and no model fitting. "
                     "The scorer's `fits: 80` field is a protocol expectation, NOT "
                     "an actual fit count for this mocked fixture. All .joblib files "
                     "are hash-only text stubs and must never be deserialized.\n")
    run.OUT = fixture
    # No market-data registration exists in this isolated, labelled fixture.
    # Qualification is bypassed only here; production CLI has no bypass option.
    run.registered = lambda: reg
    start = 30*run.core.DAY
    run.write(fixture / "data_freeze.json", {"test_start_ms": start,
              "test_end_exclusive_ms": start+30*run.core.DAY, "code_pins": run.pins(),
              "synthetic_fixture_only": True})
    true_correct, true_total = 0, 0
    for origin_index in range(10):
        directory = fixture / "origin_{:02d}".format(origin_index)
        directory.mkdir()
        times = start+origin_index*3*run.core.DAY+np.arange(4320, dtype=np.int64)*run.core.MINUTE
        step = np.arange(4320)+origin_index*4320
        probability = .5+.4*np.sin(step/17)
        labels = np.where(np.sin(step/13) >= 0, 1, -1).astype(np.int8)
        labels[step % 19 == 0] = 0
        mid_labels = np.roll(labels, 3)
        arrays, targets, model_names, hashes, model_meta, thresholds = [], [], [], [], [], {}
        for target in c["targets"]:
            for model in c["models"]:
                name = target+"__"+model
                path = directory / (name+".joblib")
                # Hash-only fixture: score() never deserializes model files.
                with path.open("xb") as stream: stream.write(("SYNTHETIC_NOT_A_MODEL:"+name).encode())
                digest = run.core.file_sha(path)
                targets.append(target); model_names.append(model); hashes.append(digest)
                arrays.append(probability)
                model_meta.append({"name": name, "model_sha256": digest})
                cal_times = times[0]-np.arange(1001, 1, -1, dtype=np.int64)*run.core.MINUTE
                cal_probability = .5+.4*np.sin(np.arange(1000)/17)
                thresholds[name] = {str(coverage): run.selective.calibrate(cal_probability, cal_times,
                                    c["asset"], coverage, c["selection_seed"])
                                   for coverage in c["coverage_targets"]}
        ledger = {"times": times, "models": np.array(model_names), "targets": np.array(targets),
                  "model_sha256": np.array(hashes), "probabilities": np.array(arrays),
                  "labels": np.array([labels, mid_labels]),
                  "random": np.array([step % 2 == 0, step % 2 == 1]),
                  "majority_train_only": np.ones((2, 4320), bool),
                  "momentum_1m": np.array([np.roll(labels, 1) >= 0, np.roll(mid_labels, 1) >= 0])}
        path = directory / "forecast_ledger.npz"
        with path.open("xb") as stream: np.savez_compressed(stream, **ledger)
        run.write(directory / "receipt.json", {"ledger_sha256": run.core.file_sha(path),
                  "data_freeze_sha256": run.core.file_sha(fixture / "data_freeze.json"),
                  "models": model_meta, "thresholds": thresholds, "code_pins": run.pins(),
                  "synthetic_fixture_only": True})
        true_correct += int(np.count_nonzero(np.where(probability >= .5, 1, -1) == labels))
        true_total += len(times)
    run.score()
    result = run.read(fixture / "multiday_results.json")
    assert len(result["cells"]) == 32 and len(result["comparisons"]) == 128
    assert result["supported_family_cells"] == 128
    reference = next(r for r in result["cells"] if r["key"] == "last_trade__book_logit__1.0")
    assert reference["correct"] == true_correct and reference["n_emitted"] == true_total
    assert reference["accuracy_all"] == true_correct/true_total
    paired = [r for r in result["comparisons"] if r["control"] == "matched_price_model"]
    assert len(paired) == 32
    assert all(r["accuracy_difference"] == 0 for r in paired)
    assert all(r["paired_lcb95_by_bank"] == [0.0, 0.0] for r in paired)
    assert all(r["simultaneous_lcb95_by_bank"] == [0.0, 0.0] for r in paired)
    for seed, path in zip(c["bootstrap"]["seeds"], result["bootstrap_pins"]):
        # JSON writer sorts keys: locate by seed instead of assuming iteration order.
        expected_path = fixture / ("bootstrap_bank_"+str(seed)+".npz")
        with np.load(expected_path, allow_pickle=False) as archive:
            expected = run.selective.block_bank(30, 1000, 3, seed)
            assert np.array_equal(archive["indices"], expected)
    try:
        run.score()
        raise AssertionError("repeated final analysis admitted")
    except ValueError as exc:
        assert "already exists" in str(exc)
    run.OUT, run.registered = real_out, real_registered
    run.write(real_out / "statistics_engineering_audit.json", {
        "synthetic_fixture_only": True, "historical_fits": 0, "multiday_market_data_used": False,
        "actual_synthetic_model_training": 0, "fixture_protocol_fit_count_field": 80,
        "synthetic_calendar_days": 30, "synthetic_origins": 10, "metric_cells": 32,
        "paired_comparisons": 128, "bootstrap_banks": 2, "draws_per_bank": 1000,
        "checks": ["manual_accuracy_matches_aggregation", "128_finite_shared_family_comparisons",
                   "identical_candidate_parent_difference_and_both_LCBs_zero",
                   "bootstrap_banks_match_registered_seeds", "second_final_analysis_rejected"],
        "fixture_results_sha256": run.core.file_sha(fixture / "multiday_results.json"),
        "checker_code_sha256": run.core.file_sha(Path(__file__)),
        "independent_validation": False, **{k: False for k in run.core.SAFETY}})
    print(json.dumps({"statistics_audit": "PASS", "synthetic_only": True,
                      "checks": 5, "historical_fits": 0}))


if __name__ == "__main__":
    main()
