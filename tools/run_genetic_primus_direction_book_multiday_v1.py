"""Bounded offline stages. One day qualification or one origin per invocation."""
from __future__ import annotations
import os
for thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import direction_book_multiday_v1 as core
from exia.genetic_primus import direction_selective_v1 as selective
from exia.genetic_primus import direction_hypotheses_v1 as estimators

CONFIG = ROOT / "configs/genetic_primus_direction_book_multiday_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_book_multiday_v1_20261013"
OLD = ROOT / "Reports/Exia/Genetic_Primus/direction_micro_hypotheses_v1_20261013"
PREVIOUS_REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v17.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v18.json"
DOWNLOADS = Path("C:/Users/anton/Downloads")
TRADES = DOWNLOADS / "bitget-futures_trades_2024-12-01_BTCUSDT.csv.gz"
BOOK = DOWNLOADS / "bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz"
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_book_multiday_v1.py",
        ROOT / "src/exia/genetic_primus/direction_selective_v1.py",
        ROOT / "src/exia/genetic_primus/direction_hypotheses_v1.py"]


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(path).is_relative_to(OUT) and "sha256" in value:
        payload = {k: v for k, v in value.items() if k != "sha256"}
        if core.fingerprint(payload) != value["sha256"]:
            raise ValueError("artifact self-hash mismatch: " + str(path))
    return value


def numpy_json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=numpy_json)
    payload = json.loads(encoded)
    payload["sha256"] = core.fingerprint(payload)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def pins():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): core.file_sha(p) for p in CODE}


def contract():
    c = read(CONFIG)
    core.check_contract(c)
    return c


def registered():
    reg = read(OUT / "registration.json")
    if reg["code_pins"] != pins():
        raise ValueError("registered experiment code/config changed; create a new version")
    return reg


def prepare():
    """Convert the already-revealed day; never parse the 23M book rows again."""
    import pandas as pd
    c = contract()
    OUT.mkdir(exist_ok=True)
    with np.load(OLD / "feature_bank.npz", allow_pickle=False) as archive:
        old = {k: archive[k].copy() for k in ("times", "price", "book", "closes", "mid")}
    with np.load(OLD / "quotes.npz", allow_pickle=False) as archive:
        quotes, stamps = archive["quotes"].copy(), archive["stamps"].copy()
    frame = pd.read_csv(TRADES, usecols=["local_timestamp", "price"], nrows=300001)
    if len(frame) > 300000:
        raise ValueError("engineering trade row budget exceeded")
    receipt, prices = frame["local_timestamp"].to_numpy(np.int64), frame["price"].to_numpy(float)
    if np.any(np.diff(receipt) < 0):
        raise ValueError("trade receipts out of order")
    bounds = old["times"] * 1000
    rows = np.searchsorted(receipt, bounds, side="left")-1
    if np.any(rows < 0) or not np.allclose(prices[rows], old["closes"], atol=1e-9, rtol=0):
        raise ValueError("receipt-clock last trade does not match previous bank")
    # Decimal milliseconds may round to x.999999 microseconds in float64.
    # Restore the integer clock by rounding, not truncating one microsecond.
    if not np.array_equal(stamps, bounds-np.rint(quotes[:, 5]*1000).astype(np.int64)):
        raise ValueError("quote age/receipt mismatch")
    day = {"times": old["times"], "price": old["price"], "book": old["book"],
           "last_trade": old["closes"], "midprice": old["mid"],
           "bid": quotes[:, 0], "ask": quotes[:, 1],
           "last_trade_receipt_us": receipt[rows], "book_receipt_us": stamps,
           "feature_max_receipt_us": np.maximum(receipt[rows], stamps)}
    report = core.validate_day(day, c); report.pop("eligible")
    bank_path = OUT / "known_day_bank.npz"
    with bank_path.open("xb") as stream:
        np.savez_compressed(stream, **day)
    entry = {"asset": c["asset"], "exchange": c["exchange"],
             "bank_path": str(bank_path), "bank_sha256": core.file_sha(bank_path),
             "sources": {"trades": {"path": str(TRADES), "sha256": core.file_sha(TRADES)},
                         "book": {"path": str(BOOK), "sha256": core.file_sha(BOOK)}},
             "book_mode": "atomic_full_snapshots", "sequence_verified": False,
             "clock": "local_receipt_strict_before_boundary", "outcomes_already_examined": True,
             "builder_code_pins": {str(ROOT / k): v for k, v in pins().items()},
             "derived_from": {str(OLD / name): core.file_sha(OLD / name)
                              for name in ("feature_bank.npz", "quotes.npz", "data_audit.json")}}
    write(OUT / "available_days.json", {"schema": "exia.book_minute_manifest/1", "entries": [entry]})
    write(OUT / "known_day_preparation.json", dict(report, code_pins=pins(), trade_rows=len(frame),
          bank_sha256=entry["bank_sha256"], independent_validation=False,
          book_source_rows=23346510, all_source_book_rows_are_snapshots=True,
          no_sequence_completeness_claim=True, **{k: False for k in core.SAFETY}))
    print(json.dumps({"stage": "prepare", "days": 1, "eligible_minutes": report["eligible_minutes"]}))


def checks():
    c = contract()
    passed = core.direct_checks(c)
    times = np.arange(1, 60*1440+1, dtype=np.int64)*core.MINUTE
    prices = 100 + np.sin(np.arange(len(times))/9)
    labels, valid = core.direction_labels(prices, times, np.ones(len(times), bool))
    features = np.column_stack([np.sin(np.arange(len(times))/23), times/core.DAY])
    first = 30*core.DAY
    a, b, test = core.split_origin(times, valid, first, c)
    m1 = core.fit_target("logistic", features, labels, a, c, c["selection_seed"])
    changed = features.copy(); changed[times >= first] += 10000
    m2 = core.fit_target("logistic", changed, labels, a, c, c["selection_seed"])
    assert np.array_equal(m1["mean"], m2["mean"])
    assert np.array_equal(m1["scale"], m2["scale"])
    assert np.array_equal(estimators.predict(m1, features[b]), estimators.predict(m2, features[b]))
    passed.append("fitted_scaler_and_predictions_ignore_future_mutation")
    a2, _, _ = core.split_origin(times, valid, first+3*core.DAY, c)
    m3 = core.fit_target("logistic", features, labels, a2, c, c["selection_seed"])
    assert not np.array_equal(m1["mean"], m3["mean"])
    assert np.allclose(m3["mean"], features[a2[labels[a2] != 0]].mean(axis=0))
    passed.append("each_origin_refits_scaler_from_its_own_training_rows")
    for coverage in c["coverage_targets"]:
        probability = estimators.predict(m1, features[b])
        threshold = selective.calibrate(probability, times[b], "BTCUSDT", coverage, c["selection_seed"])
        fixed = json.dumps(threshold, sort_keys=True)
        selective.emit(np.zeros(len(test)), times[test], "BTCUSDT", threshold, c["selection_seed"])
        selective.emit(np.ones(len(test)), times[test], "BTCUSDT", threshold, c["selection_seed"])
        assert json.dumps(threshold, sort_keys=True) == fixed
    passed.append("calibration_threshold_is_not_a_test_quantile")
    OUT.mkdir(exist_ok=True)
    receipt = {"checks": passed, "synthetic_estimator_fits": 3, "historical_fits": 0,
               "code_pins": pins(), **{k: False for k in core.SAFETY}}
    destination = OUT / ("checks_" + core.fingerprint(pins())[:12] + ".json")
    write(destination, receipt)
    print(json.dumps({"stage": "checks", "passed": len(passed), "synthetic_fits": 3}))


def init():
    c = contract()
    check_path = OUT / ("checks_" + core.fingerprint(pins())[:12] + ".json")
    evidence = read(check_path)
    if evidence["code_pins"] != pins(): raise ValueError("engineering checks are stale")
    synthetic_fit_count = sum(read(p)["synthetic_estimator_fits"] for p in OUT.glob("checks_*.json"))
    manifest = read(OUT / "available_days.json")
    reg = write(OUT / "registration.json", {"experiment_id": c["experiment_id"],
                "created_utc": datetime.now(timezone.utc).isoformat(), "contract": c,
                "code_pins": pins(), "engineering_manifest_sha256": core.file_sha(OUT / "available_days.json"),
                "engineering_checks_sha256": core.file_sha(check_path),
                "multiday_data_registered": False, "multiday_fit_budget": 80,
                "known_day_fit_budget": 8, "synthetic_fit_disclosure": synthetic_fit_count,
                "independent_validation": False, **{k: False for k in core.SAFETY}})
    previous = read(PREVIOUS_REGISTRY)
    previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, registration_date_utc=reg["created_utc"],
                    previous_registry_path=str(PREVIOUS_REGISTRY.relative_to(ROOT)).replace("\\", "/"),
                    previous_registry_sha256=core.file_sha(PREVIOUS_REGISTRY))
    previous["objectives"].append({"objective_id": c["experiment_id"],
        "family": "matched_price_book_models_separate_trade_and_mid_targets",
        "registration_path": str((OUT / "registration.json").relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256": core.file_sha(OUT / "registration.json"),
        "historical_fit_budget": 8, "conditional_multiday_fit_budget": 80,
        "multiday_dates_registered": False, "synthetic_estimator_fits": synthetic_fit_count,
        "financial_replays": 0, "independent_validation": False,
        "historical_adaptive_trial_count_complete": False, "promotion_authority": False})
    write(REGISTRY, previous)
    print(json.dumps({"stage": "init", "registry": "v18", "available_days": len(manifest["entries"])}))


def qualify(manifest_path, index):
    registered()
    entries = read(manifest_path)["entries"]
    if not 0 <= index < len(entries): raise ValueError("day index out of range")
    report = core.qualify_entries([entries[index]], contract())
    report.update(manifest_path=str(Path(manifest_path).resolve()),
                  manifest_sha256=core.file_sha(manifest_path), entry_index=index,
                  entry_sha256=core.fingerprint(entries[index]), code_pins=pins())
    name = "qualification_{}_day_{:02d}.json".format(core.file_sha(manifest_path)[:12], index)
    destination = OUT / name
    if destination.exists():
        old = read(destination)
        if old["entry_sha256"] != report["entry_sha256"] or old["code_pins"] != pins():
            raise ValueError("existing day qualification is stale")
    else:
        write(destination, report)
    print(json.dumps({"stage": "qualify", "day_index": index, "days": len(entries),
                      "required_days": 60, "status": "DAY_VALIDATED_NOT_MULTIDAY_VALIDATION"}))


def freeze(manifest_path):
    registered()
    c, manifest = contract(), read(manifest_path)
    entries = manifest["entries"]
    if len(entries) != 60: raise ValueError("DATA_INSUFFICIENT: exactly 30 context + 30 test days required")
    reports, starts, audit_pins = [], [], {}
    manifest_sha = core.file_sha(manifest_path)
    for i, entry in enumerate(entries):
        qpath = OUT / "qualification_{}_day_{:02d}.json".format(manifest_sha[:12], i)
        audit = read(qpath)
        if audit["entry_sha256"] != core.fingerprint(entry) or audit["code_pins"] != pins():
            raise ValueError("missing/stale individual day qualification")
        _, current = core.load_day(entry["bank_path"], c)
        if core.file_sha(entry["bank_path"]) != entry["bank_sha256"]:
            raise ValueError("bank changed since qualification")
        current.pop("eligible"); reports.append(current); starts.append(current["utc_start_ms"])
        audit_pins[str(qpath)] = core.file_sha(qpath)
    if starts != sorted(set(starts)) or np.any(np.diff(starts) != core.DAY):
        raise ValueError("60 consecutive complete source days required")
    if any(e.get("outcomes_already_examined", False) for e in entries[30:]):
        raise ValueError("known-day engineering cannot become new test evidence")
    origins = [starts[30]+i*3*core.DAY for i in range(10)]
    write(OUT / "data_freeze.json", {"manifest_path": str(Path(manifest_path).resolve()),
          "manifest_sha256": manifest_sha, "entries": entries, "per_day": reports,
          "qualification_pins": audit_pins, "origins_ms": origins,
          "test_start_ms": starts[30], "test_end_exclusive_ms": starts[0]+60*core.DAY,
          "code_pins": pins(), "test_interval_is_historical_not_prospective": True,
          "exchange_sequence_completeness_attested": False, **{k: False for k in core.SAFETY}})
    print(json.dumps({"stage": "freeze", "days": 60, "origins": 10, "fit_budget": 80}))


def fit_panel(bank, eligible, train, cal, test, origin, tag):
    c = contract()
    directory = OUT / tag
    directory.mkdir(exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("origin output is nonempty; no silent refit or overwrite")
    models, targets, probabilities, model_shas, details, thresholds = [], [], [], [], [], {}
    labels_by_target, control_by_target = [], []
    fold = core.fold_receipt(bank["times"], train, cal, test, origin, c)
    if tag == "known_day_engineering":
        day_start = int(bank["times"][0])-core.MINUTE
        fold["train_label_cutoff_exclusive_ms"] = day_start+int(.6*core.DAY)-7*core.MINUTE
        fold["retrain_days"] = None
        fold["split_mode"] = "known_day_60_20_20_engineering_not_multiday_protocol"
    for target_no, target in enumerate(c["targets"]):
        labels, valid = core.direction_labels(bank[target], bank["times"], eligible)
        if not valid[np.r_[train, cal, test]].all(): raise ValueError("invalid label in fold")
        labels_by_target.append(labels[test])
        control_by_target.append(core.controls(bank["times"], bank[target], labels, train, c["selection_seed"]))
        for model_name in c["models"]:
            name = target + "__" + model_name
            feature_kind = "price" if model_name.startswith("price_") else "book"
            kind = "logistic" if model_name.endswith("logit") else "boosting"
            seed = (c["selection_seed"] + int(origin)//core.MINUTE + target_no) % (2**32-1)
            model = core.fit_target(kind, bank[feature_kind], labels, train, c, seed)
            selected = train[labels[train] != 0]
            assert np.allclose(model["mean"], bank[feature_kind][selected].mean(axis=0))
            model_path = directory / (name + ".joblib")
            with model_path.open("xb") as stream: joblib.dump(model, stream, compress=3)
            probability = estimators.predict(model, bank[feature_kind][test])
            # Reload only our just-written, pinned model, never an external pickle.
            restored = joblib.load(model_path)
            assert np.array_equal(probability, estimators.predict(restored, bank[feature_kind][test]))
            cal_probability = estimators.predict(model, bank[feature_kind][cal])
            thresholds[name] = {str(coverage): selective.calibrate(cal_probability, bank["times"][cal],
                                 c["asset"], coverage, c["selection_seed"])
                                for coverage in c["coverage_targets"]}
            models.append(model_name); targets.append(target); probabilities.append(probability)
            model_shas.append(core.file_sha(model_path))
            details.append({"name": name, "model_sha256": model_shas[-1], "feature_count": bank[feature_kind].shape[1],
                            "actual_training_index_sha256": model["actual_train_times_index_sha256"],
                            "training_label_sha256": core.sha256(labels[selected].tobytes()).hexdigest(),
                            "flat_train_rows_excluded": model["flat_train_rows_excluded"],
                            "train_only_scaler_sha256": core.sha256(model["mean"].tobytes()+model["scale"].tobytes()).hexdigest(),
                            "info": model["info"]})
    ledger = {"times": bank["times"][test], "models": np.array(models), "targets": np.array(targets),
              "probabilities": np.array(probabilities), "model_sha256": np.array(model_shas),
              "labels": np.array(labels_by_target),
              "random": np.array([v["random"][test] for v in control_by_target]),
              "majority_train_only": np.array([v["majority_train_only"][test] for v in control_by_target]),
              "momentum_1m": np.array([v["momentum_1m"][test] for v in control_by_target])}
    ledger_path = directory / "forecast_ledger.npz"
    with ledger_path.open("xb") as stream: np.savez_compressed(stream, **ledger)
    receipt = write(directory / "receipt.json", {"fold": fold, "fold_sha256": core.fingerprint(fold),
          "ledger_sha256": core.file_sha(ledger_path), "thresholds": thresholds,
          "models": details, "model_sha_lookup": "model_sha256[model_index] applies to every time in that forecast row",
          "fitted_models": len(models), "code_pins": pins(),
          "data_freeze_sha256": core.file_sha(OUT / "data_freeze.json") if tag != "known_day_engineering" else None,
          "previously_unexamined_test_by_manifest": tag != "known_day_engineering",
          "independent_validation": False, **{k: False for k in core.SAFETY}})
    return ledger, receipt


def known_fit():
    reg, c = registered(), contract()
    if core.file_sha(OUT / "available_days.json") != reg["engineering_manifest_sha256"]:
        raise ValueError("engineering manifest changed")
    entry = read(OUT / "available_days.json")["entries"][0]
    if core.file_sha(entry["bank_path"]) != entry["bank_sha256"]: raise ValueError("engineering bank changed")
    bank, audit = core.load_day(entry["bank_path"], c)
    eligible = audit["eligible"]
    _, valid = core.direction_labels(bank["last_trade"], bank["times"], eligible)
    start = audit["utc_start_ms"]
    train_end, origin = start+int(.6*core.DAY), start+int(.8*core.DAY)
    t = bank["times"]
    train = np.flatnonzero(valid & (t+core.MINUTE < train_end-7*core.MINUTE))
    cal = np.flatnonzero(valid & (t >= train_end) & (t+core.MINUTE < origin-7*core.MINUTE))
    test = np.flatnonzero(valid & (t >= origin))
    ledger, receipt = fit_panel(bank, eligible, train, cal, test, origin, "known_day_engineering")
    # This exceptional one-day split is explicitly not the multiday 30d/7d protocol.
    rows = cell_metrics(ledger, receipt)
    write(OUT / "known_day_results.json", {"rows": rows, "fitted_models": 8,
          "split": {"train_end_ms": train_end, "calibration_start_ms": train_end,
                    "test_start_ms": origin, "purge_minutes": 7},
          "engineering_only": True, "independent_validation": False,
          "inferential_intervals_computed": False, "eligible_minutes": int(eligible.sum()),
          "common_test_rows": len(test), **{k: False for k in core.SAFETY}})
    print(json.dumps({"stage": "known_fit", "fits": 8, "cells": len(rows), "test_rows": len(test),
                      "independent_validation": False}))


def cell_metrics(ledger, receipt):
    c, rows = contract(), []
    for j, (target, model) in enumerate(zip(ledger["targets"], ledger["models"])):
        label = ledger["labels"][c["targets"].index(str(target))]
        for coverage in c["coverage_targets"]:
            threshold = receipt["thresholds"][str(target)+"__"+str(model)][str(coverage)]
            mask = selective.emit(ledger["probabilities"][j], ledger["times"], c["asset"], threshold, c["selection_seed"])
            metrics = selective.evaluate(ledger["probabilities"][j] >= .5, label, mask)
            rows.append(dict(target=str(target), model=str(model), coverage_target=coverage, **metrics))
    return rows


def load_frozen():
    registered()
    data, c = read(OUT / "data_freeze.json"), contract()
    if data["code_pins"] != pins(): raise ValueError("data freeze code mismatch")
    if core.file_sha(data["manifest_path"]) != data["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    pieces, masks = [], []
    for entry in data["entries"]:
        if core.file_sha(entry["bank_path"]) != entry["bank_sha256"]: raise ValueError("frozen bank changed")
        bank, audit = core.load_day(entry["bank_path"], c)
        pieces.append(bank); masks.append(audit["eligible"])
    return {key: np.concatenate([p[key] for p in pieces]) for key in core.REQUIRED_ARRAYS}, np.concatenate(masks), data


def fit_origin(index):
    bank, eligible, data = load_frozen()
    if not 0 <= index < 10: raise ValueError("origin index out of registered budget")
    origin = data["origins_ms"][index]
    _, valid = core.direction_labels(bank["last_trade"], bank["times"], eligible)
    train, cal, test = core.split_origin(bank["times"], valid, origin, contract())
    _, receipt = fit_panel(bank, eligible, train, cal, test, origin, "origin_{:02d}".format(index))
    print(json.dumps({"stage": "fit_origin", "origin": index, "fits": receipt["fitted_models"]}))


def score():
    """Single final calendar-paired analysis, only after every registered origin."""
    registered()
    if (OUT / "multiday_results.json").exists():
        raise ValueError("final analysis already exists; no silent repeated analysis")
    c, data = contract(), read(OUT / "data_freeze.json")
    ledgers, receipts, input_pins = [], [], {}
    for index in range(10):
        directory = OUT / "origin_{:02d}".format(index)
        receipt = read(directory / "receipt.json")
        path = directory / "forecast_ledger.npz"
        if receipt["code_pins"] != pins() or core.file_sha(path) != receipt["ledger_sha256"]:
            raise ValueError("origin code or forecast ledger changed")
        if receipt["data_freeze_sha256"] != core.file_sha(OUT / "data_freeze.json"):
            raise ValueError("origin belongs to a different data freeze")
        for model in receipt["models"]:
            if core.file_sha(directory / (model["name"]+".joblib")) != model["model_sha256"]:
                raise ValueError("model SHA no longer matches decisions")
        with np.load(path, allow_pickle=False) as archive:
            ledgers.append({key: archive[key].copy() for key in archive.files})
        receipts.append(receipt)
        input_pins[str(path)] = core.file_sha(path)
        input_pins[str(directory / "receipt.json")] = core.file_sha(directory / "receipt.json")
    times = np.concatenate([p["times"] for p in ledgers])
    if np.any(np.diff(times) <= 0): raise ValueError("duplicate/noncausal decision times")
    if times.min() < data["test_start_ms"] or times.max() >= data["test_end_exclusive_ms"]:
        raise ValueError("forecasts outside frozen test interval")
    start, days = data["test_start_ms"], c["test_days"]
    labels = np.concatenate([p["labels"] for p in ledgers], axis=1)
    probabilities = np.concatenate([p["probabilities"] for p in ledgers], axis=1)
    models, targets = ledgers[0]["models"], ledgers[0]["targets"]
    cells, masks, daily, metadata = [], {}, {}, {}
    for j, (target, model) in enumerate(zip(targets, models)):
        target_no = c["targets"].index(str(target))
        for coverage in c["coverage_targets"]:
            pieces = []
            for ledger, receipt in zip(ledgers, receipts):
                threshold = receipt["thresholds"][str(target)+"__"+str(model)][str(coverage)]
                pieces.append(selective.emit(ledger["probabilities"][j], ledger["times"], c["asset"],
                                             threshold, c["selection_seed"]))
            mask = np.concatenate(pieces)
            key = "{}__{}__{}".format(target, model, coverage)
            masks[key] = mask
            daily[key] = selective.daily_counts(times, labels[target_no], probabilities[j] >= .5,
                                                mask, start, days)
            metadata[key] = (target_no, j, coverage)
            cells.append(dict(key=key, target=str(target), model=str(model), coverage_target=coverage,
                              **selective.evaluate(probabilities[j] >= .5, labels[target_no], mask)))
    comparisons, replicate_banks, bootstrap_pins = [], [], {}
    for seed in c["bootstrap"]["seeds"]:
        bank = selective.block_bank(days, c["bootstrap"]["repetitions"],
                                    c["bootstrap"]["calendar_block_days"], seed)
        bank_path = OUT / ("bootstrap_bank_"+str(seed)+".npz")
        with bank_path.open("xb") as stream: np.savez_compressed(stream, indices=bank)
        bootstrap_pins[str(bank_path)] = core.file_sha(bank_path)
        replicate_banks.append(bank)
    family_observed, family_replicates = [], [[], []]
    supported_indices = []
    for key, (target_no, j, coverage) in metadata.items():
        if not str(models[j]).startswith("book_"): continue
        parent_model = str(models[j]).replace("book_", "price_")
        parent_j = next(k for k, (t, m) in enumerate(zip(targets, models))
                        if t == targets[j] and m == parent_model)
        parent_key = "{}__{}__{}".format(targets[j], parent_model, coverage)
        for control_name in c["controls"]:
            prediction = (probabilities[parent_j] >= .5) if control_name == "matched_price_model" else np.concatenate(
                [p[control_name][target_no] for p in ledgers])
            for mode in ("matched_candidate_mask", "own_policy_mask"):
                mask = masks[key] if mode == "matched_candidate_mask" else masks[parent_key] if control_name == "matched_price_model" else np.ones(len(times), bool)
                control_daily = selective.daily_counts(times, labels[target_no], prediction, mask, start, days)
                candidate_total, control_total = daily[key].sum(axis=0), control_daily.sum(axis=0)
                supported = candidate_total[1] > 0 and control_total[1] > 0
                observed = float(candidate_total[0]/candidate_total[1]-control_total[0]/control_total[1]) if supported else None
                rep = []
                for bank in replicate_banks:
                    _, candidate_draw = selective.resample_statistics(daily[key], bank)
                    _, control_draw = selective.resample_statistics(control_daily, bank)
                    rep.append(candidate_draw-control_draw)
                supported = supported and all(np.isfinite(v).all() for v in rep)
                record = {"candidate_key": key, "control": control_name, "comparison": mode,
                          "accuracy_difference": observed, "bootstrap_supported": bool(supported),
                          "paired_lcb95_by_bank": [float(np.quantile(v, .05)) for v in rep] if supported else None,
                          "simultaneous_lcb95_by_bank": None}
                comparisons.append(record)
                if supported:
                    supported_indices.append(len(comparisons)-1)
                    family_observed.append(observed)
                    for k in range(2): family_replicates[k].append(rep[k])
    critical_values = []
    if supported_indices:
        bounds = []
        for k in range(2):
            lower, critical = selective.simultaneous_lcb(family_observed, np.array(family_replicates[k]).T)
            bounds.append(lower); critical_values.append(critical)
        for position, comparison_index in enumerate(supported_indices):
            comparisons[comparison_index]["simultaneous_lcb95_by_bank"] = [float(v[position]) for v in bounds]
    for cell in cells:
        estimates = [selective.resample_statistics(daily[cell["key"]], bank)[0] for bank in replicate_banks]
        cell["calendar_block_ci95_by_bank"] = [e["ci95"] for e in estimates]
        cell["emitted_days"] = estimates[0]["emitted_days"]
    write(OUT / "multiday_results.json", {"cells": cells, "comparisons": comparisons,
          "test_start_ms": start, "test_end_exclusive_ms": data["test_end_exclusive_ms"],
          "origins": 10, "fits": 80, "bootstrap_pins": bootstrap_pins,
          "bootstrap_family_cells": len(comparisons), "supported_family_cells": len(supported_indices),
          "max_t_critical_values": critical_values, "input_pins": input_pins,
          "source_sequence_completeness_attested": False,
          "improvement_not_automatically_financial_profit": True,
          "independent_validation": False, **{k: False for k in core.SAFETY}})
    print(json.dumps({"stage": "score", "forecast_cells": len(cells), "paired_comparisons": len(comparisons),
                      "supported_comparisons": len(supported_indices)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "checks", "init", "qualify", "freeze", "known_fit", "fit_origin", "score"])
    parser.add_argument("--manifest", type=Path, default=OUT / "available_days.json")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "qualify": qualify(args.manifest, args.index)
    elif args.stage == "freeze": freeze(args.manifest)
    elif args.stage == "fit_origin": fit_origin(args.index)
    else: globals()[args.stage]()


if __name__ == "__main__":
    main()
