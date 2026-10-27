"""Bounded per-asset direction fits and audits; no trading, network or collector."""
from __future__ import annotations
import run_genetic_primus_direction_multiscale_v1 as b
import argparse
import time
from pathlib import Path
import numpy as np

OUT = b.ROOT / "Reports/Exia/Genetic_Primus/direction_specialization_v1_20261012"
CONFIG = b.ROOT / "configs/genetic_primus_direction_specialization_v1.json"
REGISTRY = b.ROOT / "configs/genetic_primus_objective_registry_v13.json"
PREVIOUS_REGISTRY = b.ROOT / "configs/genetic_primus_objective_registry_v12.json"
MODELS = ("logistic12", "forest12")
TF = 10


def paired_draws(a, c, valid, day, banks):
    """Paired calendar block estimates; all rows retained, including flat."""
    a, c, valid, day = np.asarray(a), np.asarray(c), np.asarray(valid), np.asarray(day)
    if a.ndim != 1 or not (a.shape == c.shape == valid.shape == day.shape):
        raise ValueError("aligned one-dimensional arrays required")
    if day.min() != 0 or day.max() != 60 or len(np.unique(day)) != 61:
        raise ValueError("complete 61-day grid required")
    delta = np.bincount(day, weights=a.astype(float) - c.astype(float), minlength=61)
    total = np.bincount(day, minlength=61)
    nonflat = np.bincount(day, weights=valid.astype(int), minlength=61)
    result = {}
    for key, bank in banks.items():
        numerator = delta[bank].sum(axis=1)
        denominator = nonflat[bank].sum(axis=1)
        if np.any(denominator <= 0): raise ValueError("empty resampled nonflat set")
        result[key] = {"all": numerator / total[bank].sum(axis=1), "nonflat": numerator / denominator}
    return result


def checks():
    p = np.array([True, False, True, False])
    metric = b.d.metrics(p, np.ones(4), np.array([2., 0., 1., 1.]))
    assert metric["accuracy_all"] == 0.5 and metric["accuracy_nonflat"] == 1
    days = np.repeat(np.arange(61), 2)
    bank = {"test": b.d.sample_bank(61, 20, 3, 123)}
    draws = paired_draws(np.ones(122, bool), np.zeros(122, bool), np.ones(122, bool), days, bank)
    assert np.array_equal(draws["test"]["all"], np.ones(20))
    negative = paired_draws(np.zeros(122, bool), np.ones(122, bool), np.ones(122, bool), days, bank)
    assert np.array_equal(negative["test"]["all"], -np.ones(20))
    assert b.read(CONFIG)["models"] == list(MODELS)
    assert not any(b.read(CONFIG)["safety"].values())
    print({"pure_checks": 6, "fits": 0, "network": False})


def parent_files():
    files = [b.OUT / "registration.json", b.OUT / "resample_bank.npz", b.tf_path(TF) / "preparation.json"]
    for i in range(21):
        files += [b.tf_path(TF) / f"fold_{i:02d}.npz", b.tf_path(TF) / f"origin_{i:02d}_receipt.json",
                  b.tf_path(TF) / f"origin_{i:02d}_controls.npz"]
        for name in MODELS:
            files += [b.tf_path(TF) / f"origin_{i:02d}_{name}{suffix}" for suffix in (".json", ".joblib", ".npz")]
    return files


def register():
    cfg, parent = b.read(CONFIG), b.registration(TF)
    if cfg["models"] != list(MODELS) or cfg["symbols"] != b.SYMBOLS or cfg["timeframe_minutes"] != TF:
        raise ValueError("fixed protocol mismatch")
    if any(cfg["safety"].values()) or cfg["new_fit_budget"] != 336:
        raise ValueError("offline budget violation")
    if OUT.exists() or REGISTRY.exists(): raise ValueError("registration must be new")
    previous = b.sealed(PREVIOUS_REGISTRY)
    prep = b.sealed(b.tf_path(TF) / "preparation.json")
    if len(prep["folds"]) != 21: raise ValueError("21 origins required")
    for fold in prep["folds"]:
        if fold["latest_label_ms"] > fold["cutoff_ms"]: raise ValueError("mature/purge failure")
    parents = {b.relative(p): b.pin(p) for p in parent_files()}
    sources = {b.relative(p): b.pin(p) for p in [Path(__file__), CONFIG, PREVIOUS_REGISTRY] + b.CODE}
    OUT.mkdir(parents=True)
    reg = b.write(OUT / "registration.json", {
        "schema_version": "direction_specialization_registration/1", "config": cfg,
        "parent_registration_sha256": parent["sha256"], "parent_pins": parents, "source_pins": sources,
        "hyperparameters": {name: b.plain(b.d.models.parameters(name).get_params()) for name in MODELS},
        "time_indices": "unchanged parent folds", "historical_new_fit_budget": 336,
        "independent_validation": False, "safety": cfg["safety"]})
    updated = {k: v for k, v in previous.items() if k != "sha256"}
    updated.update({"registry_id": "genetic_primus_objective_registry_v13", "registration_date_utc": "2026-10-12",
                    "previous_registry_path": b.relative(PREVIOUS_REGISTRY),
                    "previous_registry_sha256": b.pin(PREVIOUS_REGISTRY)["sha256"],
                    "predecessor": {"path": b.relative(PREVIOUS_REGISTRY), "sha256": previous["sha256"],
                                    "objectives_preserved": True}})
    updated["objectives"] = previous["objectives"] + [{
        "objective_id": cfg["objective_id"], "family": "direction_only_asset_specialization",
        "config_path": b.relative(CONFIG), "config_sha256": b.pin(CONFIG)["sha256"],
        "registration_path": b.relative(OUT / "registration.json"), "registration_sha256": reg["sha256"],
        "historical_fit_budget": 336, "reused_pooled_fit_count": 42, "financial_replays": 0,
        "historical_adaptive_trial_count_complete": False, "independent_validation": False, "promotion_authority": False}]
    b.write(REGISTRY, updated)
    print({"registered": cfg["objective_id"], "new_fit_budget": 336, "reused_pooled": 42, "registry": 13})


def registration():
    reg = b.sealed(OUT / "registration.json")
    if any(reg["safety"].values()): raise ValueError("offline-only")
    for name, expected in reg["source_pins"].items():
        if b.pin(b.ROOT / name) != expected: raise ValueError("registered source changed: " + name)
    b.registration(TF)
    return reg


def fold_data(i):
    if not 0 <= i < 21: raise ValueError("bounded origin required")
    bank = b.load_bank(TF)
    fold = b.sealed(b.tf_path(TF) / "preparation.json")["folds"][i]
    with np.load(b.tf_path(TF) / f"fold_{i:02d}.npz", allow_pickle=False) as data:
        train, test = data["train"], data["test"]
    x, r = np.asarray(bank["features"][train]), np.asarray(bank["returns"][train])
    if b.m.array_identity(x, r) != fold["train_data_sha256"]: raise ValueError("train bank mismatch")
    if fold["latest_label_ms"] > fold["cutoff_ms"]: raise ValueError("purge failed")
    direction = np.sign(np.asarray(bank["closes"][train + 1]) - np.asarray(bank["closes"][train]))
    if not np.array_equal(direction, np.sign(r)): raise ValueError("training label/price mismatch")
    return bank, fold, train, test, x, r


def prefix(i, asset, name):
    return OUT / f"origin_{i:02d}_{asset:02d}_{name}"


def validate_parent(i, name, reg):
    base = b.tf_path(TF) / f"origin_{i:02d}_{name}"
    for suffix in (".json", ".joblib", ".npz"):
        p = base.with_suffix(suffix)
        if b.pin(p) != reg["parent_pins"][b.relative(p)]: raise ValueError("parent pooled artifact changed")
    meta = b.sealed(base.with_suffix(".json"))
    if b.pin(base.with_suffix(".joblib")) != meta["model_pin"]: raise ValueError("parent model pin mismatch")
    if b.pin(base.with_suffix(".npz")) != meta["prediction_pin"]: raise ValueError("parent prediction pin mismatch")
    return base


def fit_origin(i):
    import joblib
    reg = registration()
    bank, fold, train, test, x, r = fold_data(i)
    entries, fresh = {}, 0
    for asset, symbol in enumerate(b.SYMBOLS):
        for name in MODELS:
            validate_parent(i, name, reg)
            base = prefix(i, asset, name)
            meta_path, model_path, pred_path = (base.with_suffix(s) for s in (".json", ".joblib", ".npz"))
            if meta_path.exists():
                meta = b.sealed(meta_path)
                if b.pin(model_path) != meta["model_pin"] or b.pin(pred_path) != meta["prediction_pin"]:
                    raise ValueError("partial fit artifacts changed")
            else:
                if model_path.exists() or pred_path.exists(): raise ValueError("incomplete fit; no silent overwrite/refit")
                started = time.perf_counter()
                fitted = b.d.models.fit(name, x[:, asset], r[:, asset])
                probability = b.d.models.predict(fitted, np.asarray(bank["features"][test, asset]))
                with model_path.open("xb") as stream: joblib.dump(fitted, stream, compress=3)
                model_pin = b.pin(model_path)
                with pred_path.open("xb") as stream:
                    np.savez_compressed(stream, times=np.asarray(bank["times"][test]), probability=probability,
                                        model_sha256=np.array(model_pin["sha256"]))
                meta = b.write(meta_path, {"registration_sha256": reg["sha256"], "origin": i, "symbol": symbol,
                    "model": name, "info": fitted["info"], "fold": fold, "model_pin": model_pin,
                    "prediction_pin": b.pin(pred_path), "elapsed_seconds": time.perf_counter() - started,
                    "source_train_indices_sha256": b.m.array_identity(train), "n_test": len(test)})
                fresh += 1
            entries[f"{asset}:{name}"] = {"metadata": meta_path.name, "sha256": meta["sha256"]}
    receipt = OUT / f"origin_{i:02d}_receipt.json"
    body = {"registration_sha256": reg["sha256"], "origin": i, "fits": entries, "fit_count": 16}
    if receipt.exists():
        if b.sealed(receipt) != {**body, "sha256": b.m.identity(body)}: raise ValueError("receipt mismatch")
    else: b.write(receipt, body)
    print({"origin": i, "completed_fits": 16, "new_fits": fresh, "financial_replays": 0})


def audit_origin(i):
    import joblib
    reg = registration()
    bank, fold, train, test, x, r = fold_data(i)
    entries, cells = {}, 0
    for asset, symbol in enumerate(b.SYMBOLS):
        for name in MODELS:
            base = prefix(i, asset, name)
            meta = b.sealed(base.with_suffix(".json"))
            if meta["registration_sha256"] != reg["sha256"] or meta["symbol"] != symbol or meta["origin"] != i:
                raise ValueError("fit provenance mismatch")
            if b.pin(base.with_suffix(".joblib")) != meta["model_pin"] or b.pin(base.with_suffix(".npz")) != meta["prediction_pin"]:
                raise ValueError("fit pins changed")
            fitted = joblib.load(base.with_suffix(".joblib"))
            mask = r[:, asset] != 0
            selected, ret = x[mask, asset], r[mask, asset]
            mean, scale = selected.mean(axis=0), selected.std(axis=0)
            scale[scale < 1e-12] = 1
            if not np.array_equal(mean, fitted["info"]["mean"]) or not np.array_equal(scale, fitted["info"]["scale"]):
                raise ValueError("train-only scaler mismatch")
            if b.m.array_identity(selected, ret) != fitted["info"]["training_sha256"]:
                raise ValueError("training SHA mismatch")
            xx = np.asarray(bank["features"][test, asset])
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
                probability, times = data["probability"], data["times"]
                if str(data["model_sha256"]) != meta["model_pin"]["sha256"]: raise ValueError("prediction model SHA missing")
            if not np.array_equal(times, bank["times"][test]): raise ValueError("prediction grid mismatch")
            if not np.array_equal(probability, b.d.models.predict(fitted, xx)): raise ValueError("model restoration mismatch")
            z = (xx - mean) / scale
            if name == "logistic12":
                raw = z @ fitted["estimator"].coef_[0] + fitted["estimator"].intercept_[0]
                independent = np.clip(1 / (1 + np.exp(-raw)), 1e-8, 1 - 1e-8)
            else:
                independent = np.clip(fitted["estimator"].predict_proba(z)[:, 1], 1e-8, 1 - 1e-8)
            if not np.allclose(probability, independent, rtol=0, atol=1e-14): raise ValueError("independent prediction mismatch")
            pooled = validate_parent(i, name, reg)
            parent_fit = joblib.load(pooled.with_suffix(".joblib"))
            with np.load(pooled.with_suffix(".npz"), allow_pickle=False) as data:
                parent_p = data["probability"][:, asset]
                if not np.array_equal(data["times"], times): raise ValueError("paired grid mismatch")
            restored_parent = b.d.models.predict(parent_fit, xx)
            if not np.allclose(parent_p, restored_parent, rtol=0, atol=1e-14):
                raise ValueError("parent prediction replay mismatch")
            if not np.array_equal(parent_p >= 0.5, restored_parent >= 0.5):
                raise ValueError("parent decision replay mismatch")
            cells += len(probability) * 2
            entries[f"{asset}:{name}"] = {"metadata_sha256": meta["sha256"], "training_rows": len(selected)}
    path = OUT / f"origin_{i:02d}_audit.json"
    if path.exists(): raise ValueError("audit already recorded")
    b.write(path, {"origin": i, "registration_sha256": reg["sha256"], "models": 16,
                   "new_and_pooled_probability_cells": cells, "entries": entries, "fits": 0})
    print({"audited_origin": i, "asset_models": 16, "new_and_pooled_probability_cells": cells, "fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["checks", "register", "fit", "audit"])
    parser.add_argument("--origin", type=int)
    args = parser.parse_args()
    if args.stage == "checks": checks()
    elif args.stage == "register": register()
    elif args.stage == "fit": fit_origin(args.origin)
    else: audit_origin(args.origin)
