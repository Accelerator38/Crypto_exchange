"""Fixed-band offline recalculation: one origin or one horizon per command."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import joblib
import numpy as np
import sklearn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import run_genetic_primus_direction_selective_v1 as parent
from exia.genetic_primus import direction_ternary_fixed_v1 as core
from exia.genetic_primus import direction_ternary_v1 as tri
from exia.genetic_primus import direction_selective_v1 as causal

CONFIG = ROOT / "configs/genetic_primus_direction_ternary_fixed_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_fixed_v1_20261013"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v22.json"
ADAPTIVE = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_v2_20261013"
OLD01 = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_01pct_v1_20261013"
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_ternary_fixed_v1.py",
        ROOT / "src/exia/genetic_primus/direction_ternary_v1.py",
        ROOT / "src/exia/genetic_primus/direction_selective_v1.py", Path(parent.__file__)]


def read(path):
    value = parent.read(path)
    if Path(path).is_relative_to(OUT) and "sha256" in value:
        if parent.identity({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
            raise ValueError("artifact self-hash mismatch")
    return value


def config():
    c = read(CONFIG)
    for minutes in c["timeframes_minutes"]: core.band(c, minutes)
    return c


def pins():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): parent.digest(p) for p in CODE}


def versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}


def methods(c):
    return [*c["candidates"], *c["primary_controls"], *c["legacy_diagnostics"]]


def registered():
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only the registered workspace venv interpreter is allowed")
    r = read(OUT / "registration.json")
    if r["code_pins"] != pins() or r["versions"] != versions(): raise ValueError("registered code/environment changed")
    return r


def checks():
    c, passed = config(), core.direct_checks(config())
    rng = np.random.default_rng(22)
    x = rng.normal(size=(160, 8, 12)); labels = tri.STATES[np.arange(160*8).reshape(160, 8) % 3]
    tested = []
    for candidate in c["candidates"]:
        model = core.fit(x[:120], labels[:120], candidate, c, 17)
        probability = core.probabilities(model, x[120:])
        assert probability.shape == (40, 8, 3) and np.allclose(probability.sum(axis=-1), 1)
        assert np.array_equal(model["mean"], x[:120].reshape(-1, 12).mean(axis=0))
        if candidate.endswith("_balanced"): assert np.allclose(model["class_weight_mass"], [320, 320, 320])
        tested.append(model)
    passed.append("all_four_true_three_class_candidates_and_weights")
    changed = x.copy(); changed[120:] += 10000
    repeat = core.fit(changed[:120], labels[:120], c["candidates"][0], c, 17)
    assert np.array_equal(core.probabilities(repeat, x[120:]), core.probabilities(tested[0], x[120:]))
    passed.append("future_mutation_does_not_change_training_or_forecasts")
    shifted = core.fit(x[10:130], labels[10:130], c["candidates"][0], c, 18)
    assert np.array_equal(shifted["mean"], x[10:130].reshape(-1, 12).mean(axis=0))
    assert not np.array_equal(shifted["mean"], tested[0]["mean"])
    passed.append("new_origin_refits_its_own_scaler")
    sparse = core.fit(x[:120], np.zeros((120, 8), np.int8), c["candidates"][0], c, 17)
    assert sparse["estimator"] is None and np.all(tri.predict(core.probabilities(sparse, x[120:])) == 0)
    passed.append("sparse_state_prior_fallback_is_explicit")
    OUT.mkdir(exist_ok=True)
    parent.write(OUT / "checks.json", {"checks": passed, "synthetic_estimator_fits": 6, "code_pins": pins(),
                 "independent_validation": False, **{k: False for k in core.SAFETY}})
    print(parent.json.dumps({"checks": len(passed), "synthetic_fits": 6, "status": "PASS"}))


def legacy_paths(minutes, origin):
    return [parent.OUT / "tf_{}m_origin_{:02d}_{}.npz".format(minutes, origin, kind) for kind in ["logistic", "boosting"]]+[
        base / "origin_{:02d}".format(origin) / "tf_{}m.npz".format(minutes) for base in [ADAPTIVE, OLD01]]


def init():
    c, old = config(), parent.registration()
    if read(OUT / "checks.json")["code_pins"] != pins(): raise ValueError("checks are stale")
    data_pins = dict(old["data_pins"])
    for minutes in c["timeframes_minutes"]:
        for origin in range(21):
            for path in legacy_paths(minutes, origin):
                data_pins[str(path.relative_to(ROOT)).replace("\\", "/")] = parent.digest(path)
    for p, sha in data_pins.items():
        if parent.digest(ROOT / p) != sha: raise ValueError("parent data changed")
    parent.write(OUT / "registration.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "contract": c,
        "code_pins": pins(), "versions": versions(), "data_pins": data_pins,
        "parent_registration_sha256": parent.digest(ROOT / c["parent_registration"]),
        "adaptive_parent_registration_sha256": parent.digest(ADAPTIVE / "registration.json"),
        "checks_sha256": parent.digest(OUT / "checks.json"), "historical_fit_budget": 336,
        "synthetic_estimator_fits": 6, "threshold_selection_authority": "explicit_user_request",
        "independent_validation": False, **{k: False for k in core.SAFETY}})
    previous_path = ROOT / c["previous_registry"]
    previous = read(previous_path); previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, registration_date_utc=datetime.now(timezone.utc).isoformat(),
                    previous_registry_path=c["previous_registry"], previous_registry_sha256=parent.digest(previous_path))
    previous["objectives"].append({"objective_id": c["objective_id"], "family": "user_fixed_horizon_three_state_logit_boost",
        "registration_path": str((OUT / "registration.json").relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256": parent.digest(OUT / "registration.json"), "historical_fit_budget": 336,
        "synthetic_estimator_fits": 6, "independent_validation": False,
        "historical_adaptive_trial_count_complete": False, "financial_replays": 0, "promotion_authority": False})
    parent.write(REGISTRY, previous)
    print(parent.json.dumps({"registry": "v22", "data_pins": len(data_pins), "fit_budget": 336}))


def load_legacy(minutes, number, times, return_pct):
    legacy = []
    adaptive_no = read(ADAPTIVE / "registration.json")["contract"]["schemes"].index("train_q50")
    for k, path in enumerate(legacy_paths(minutes, number)):
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["times"], times): raise ValueError("legacy forecast times differ")
            if k < 2:
                if not np.array_equal(archive["labels"], tri.labels(return_pct, 0)): raise ValueError("legacy binary label parity failed")
                legacy.append(np.where(archive["probability"] >= .5, 1, -1).astype(np.int8))
            else:
                legacy.append(tri.predict(archive["probability"][adaptive_no if k == 2 else 0]))
    return np.array(legacy)


def fit_origin(number):
    reg, c = registered(), config()
    if not 0 <= number < 21: raise ValueError("origin exceeds registered budget")
    directory = OUT / "origin_{:02d}".format(number); directory.mkdir(exist_ok=True)
    fits, fallback, resumed = 0, 0, 0
    for minutes in c["timeframes_minutes"]:
        prefix = directory / "tf_{}m".format(minutes); path, meta_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
        if meta_path.exists():
            meta = read(meta_path)
            if meta["code_pins"] != pins() or parent.digest(path) != meta["ledger_sha256"]: raise ValueError("completed origin changed")
            for m in meta["models"]:
                if parent.digest(directory / m["file"]) != m["sha256"]: raise ValueError("completed model changed")
            resumed += 1; continue
        if path.exists() or list(directory.glob("tf_{}m__*.joblib".format(minutes))): raise ValueError("partial origin: inspect; do not silently refit")
        for p, sha in reg["data_pins"].items():
            if "tf_{}m/".format(minutes) in p or "tf_{}m_origin_{:02d}_".format(minutes, number) in p or "origin_{:02d}/tf_{}m.npz".format(number, minutes) in p:
                if parent.digest(ROOT / p) != sha: raise ValueError("registered input changed")
        data = parent.load(minutes); times = data["times"]
        origin, width = parent.START+number*3*tri.DAY, minutes*60000
        seed = c["seed"]+minutes*1000+number
        train, cal = core.fold(times, origin, minutes, c, seed)
        test = np.flatnonzero((times >= origin) & (times < min(origin+3*tri.DAY, parent.END)))
        if len(test) != (min(origin+3*tri.DAY, parent.END)-origin)//width or test[-1]+1 >= len(times): raise ValueError("incomplete future label grid")
        threshold = core.band(c, minutes)
        rtrain = (data["closes"][train+1]/data["closes"][train]-1)*100
        rtest = (data["closes"][test+1]/data["closes"][test]-1)*100
        ytrain, ytest = tri.labels(rtrain, threshold), tri.labels(rtest, threshold)
        x, xt = np.asarray(data["features"][train], float), np.asarray(data["features"][test], float)
        legacy = load_legacy(minutes, number, times[test], rtest)
        probability, records, model_shas = [], [], []
        reference_scaler, priors = None, None
        for candidate in c["candidates"]:
            model = core.fit(x, ytrain, candidate, c, seed)
            scaler_hash = parent.identity({"mean": model["mean"], "scale": model["scale"]})
            if reference_scaler is None: reference_scaler = scaler_hash; priors = model["train_priors"]
            assert reference_scaler == scaler_hash and np.array_equal(priors, model["train_priors"])
            model_path = directory / "tf_{}m__{}.joblib".format(minutes, candidate)
            with model_path.open("xb") as stream: joblib.dump(model, stream, compress=3)
            p = core.probabilities(model, xt)
            assert np.array_equal(p, core.probabilities(joblib.load(model_path), xt))
            digest = parent.digest(model_path); probability.append(p); model_shas.append(digest)
            records.append({"candidate": candidate, "file": model_path.name, "sha256": digest, "kind": model["kind"],
                "band_pct": threshold, "scaler_sha256": scaler_hash, "coefficient_count": model["coefficient_count"],
                "train_rows_including_neutral": model["train_rows_including_neutral"],
                "train_counts_down_neutral_up": model["train_counts"], "class_weight_mass": model["class_weight_mass"],
                "fitted_iterations": model["fitted_iterations"], "warnings": model["warnings"]})
            fits += model["estimator"] is not None; fallback += model["estimator"] is None
        uniform = np.random.default_rng(np.random.SeedSequence([seed, 522])).random(ytest.shape)
        with path.open("xb") as stream:
            np.savez_compressed(stream, times=times[test], return_pct=rtest,
                past_return_pct=(data["closes"][test]/data["closes"][test-1]-1)*100,
                labels=ytest, band_pct=np.array(threshold), probability=np.array(probability), train_priors=priors,
                uniform=uniform, legacy_predictions=legacy, model_sha256=np.array(model_shas))
        fold = {"origin_ms": origin, "train_times_sha256": parent.identity(times[train]), "calibration_times_sha256": parent.identity(times[cal]),
            "test_times_sha256": parent.identity(times[test]), "n_train_times": len(train), "n_calibration_times": len(cal), "n_test_times": len(test),
            "train_first_ms": int(times[train[0]]), "train_latest_label_ms": int(times[train[-1]+1]),
            "train_cutoff_exclusive_ms": origin-7*tri.DAY-7*width, "calibration_latest_label_ms": int(times[cal[-1]+1]),
            "calibration_cutoff_exclusive_ms": origin-7*width, "horizon_minutes": minutes, "example_step_minutes": minutes,
            "decision_minutes": minutes, "retrain_days": 3, "purge_bars": 7}
        parent.write(meta_path, {"fold": fold, "fold_sha256": parent.identity(fold), "band_pct": threshold,
            "models": records, "ledger_sha256": parent.digest(path), "code_pins": pins(),
            "model_sha_lookup": "candidate index identifies the model for every decision/asset in its probability slice",
            "independent_validation": False, **{k: False for k in core.SAFETY}})
    print(parent.json.dumps({"origin": number, "fits": int(fits), "prior_only": int(fallback), "resumed_horizons": resumed}))


def score_frame(minutes):
    registered(); c = config(); names = methods(c)
    if minutes not in c["timeframes_minutes"]: raise ValueError("unregistered horizon")
    destination = OUT / "results_{}m.json".format(minutes)
    if destination.exists(): raise ValueError("horizon already scored; no overwrite")
    daily = np.zeros((8, len(names), 61, 3, 3), np.int64)
    proper = np.zeros((4, 8, 3)); model_records, input_pins = [], {}
    for number in range(21):
        prefix = OUT / "origin_{:02d}".format(number) / "tf_{}m".format(minutes)
        meta = read(prefix.with_suffix(".json")); path = prefix.with_suffix(".npz")
        if meta["code_pins"] != pins() or parent.digest(path) != meta["ledger_sha256"]: raise ValueError("origin ledger changed")
        for model in meta["models"]:
            if parent.digest(prefix.parent / model["file"]) != model["sha256"]: raise ValueError("model changed")
            model_records.append(dict(origin=number, **model))
        with np.load(path, allow_pickle=False) as archive: ledger = {k: archive[k].copy() for k in archive.files}
        if float(ledger["band_pct"]) != core.band(c, minutes) or not np.array_equal(ledger["labels"], tri.labels(ledger["return_pct"], core.band(c, minutes))):
            raise ValueError("fixed band or label changed")
        predictions = core.predictions(ledger)
        for asset in range(8):
            for k, pred in enumerate(predictions):
                daily[asset, k] += tri.daily_confusion(ledger["times"], ledger["labels"][:, asset], pred[:, asset], parent.START, 61)
        for k in range(4):
            p, y = ledger["probability"][k], ledger["labels"]
            observed = np.take_along_axis(p, (y+1)[..., None], axis=-1)[..., 0]
            proper[k, :, 0] += -np.log(np.clip(observed, 1e-15, 1)).sum(axis=0)
            proper[k, :, 1] += ((p-np.eye(3)[y+1])**2).sum(axis=-1).sum(axis=0)
            proper[k, :, 2] += len(y)
        for p in [path, prefix.with_suffix(".json")]: input_pins[str(p.relative_to(ROOT)).replace("\\", "/")] = parent.digest(p)
    rows = []
    for asset, symbol in enumerate(c["symbols"]):
        for k, method in enumerate(names):
            metrics = tri.from_confusion(daily[asset, k].sum(axis=0))
            row = dict(minutes=minutes, symbol=symbol, method=method, band_pct=core.band(c, minutes), **metrics)
            if k < 4:
                row.update(log_loss=float(proper[k, asset, 0]/proper[k, asset, 2]), multiclass_brier=float(proper[k, asset, 1]/proper[k, asset, 2]))
            rows.append(row)
    weights, bootstrap_pins = [], {}
    for seed in c["bootstrap"]["seeds"]:
        path = OUT / "bootstrap_bank_{}.npz".format(seed)
        bank = causal.block_bank(61, 1000, 3, seed)
        if path.exists():
            with np.load(path, allow_pickle=False) as a: assert np.array_equal(a["indices"], bank)
        else:
            with path.open("xb") as stream: np.savez_compressed(stream, indices=bank)
        weights.append(tri.calendar_weights(bank, 61))
        bootstrap_pins[str(path.relative_to(ROOT)).replace("\\", "/")] = parent.digest(path)
    comparisons, replicates = [], [[], []]
    for asset, symbol in enumerate(c["symbols"]):
        for k, candidate in enumerate(c["candidates"]):
            model_boot = [tri.bootstrap_metrics(daily[asset, k], w) for w in weights]
            row = next(r for r in rows if r["symbol"] == symbol and r["method"] == candidate)
            row["ci95_by_bank"] = {metric: [np.quantile(a[metric], [.025, .975]).tolist() for a in model_boot] for metric in ["accuracy", "macro_f1"]}
            for j, control in enumerate(c["primary_controls"], start=4):
                control_boot = [tri.bootstrap_metrics(daily[asset, j], w) for w in weights]
                control_row = next(r for r in rows if r["symbol"] == symbol and r["method"] == control)
                for metric in ["accuracy", "macro_f1"]:
                    rep = [model_boot[b][metric]-control_boot[b][metric] for b in range(2)]
                    comparisons.append({"minutes": minutes, "symbol": symbol, "candidate": candidate, "control": control,
                        "metric": metric, "difference": row[metric]-control_row[metric],
                        "paired_lcb95_by_bank": [float(np.quantile(v, .05)) for v in rep]})
                    for b in range(2): replicates[b].append(rep[b])
    draws_path = OUT / "paired_draws_{}m.npz".format(minutes)
    with draws_path.open("xb") as stream: np.savez_compressed(stream, bank0=np.array(replicates[0]).T, bank1=np.array(replicates[1]).T)
    daily_path = OUT / "daily_confusion_{}m.npz".format(minutes)
    with daily_path.open("xb") as stream: np.savez_compressed(stream, daily=daily)
    parent.write(destination, {"rows": rows, "comparisons": comparisons, "bootstrap_pins": bootstrap_pins,
        "paired_draws_sha256": parent.digest(draws_path), "daily_confusion_sha256": parent.digest(daily_path), "input_pins": input_pins,
        "fit_attempts": len(model_records), "fitted_estimators": sum(m["kind"] != "prior_only_sparse_classes" for m in model_records),
        "prior_only": sum(m["kind"] == "prior_only_sparse_classes" for m in model_records),
        "warnings": [dict(origin=m["origin"], candidate=m["candidate"], messages=m["warnings"]) for m in model_records if m["warnings"]],
        "independent_validation": False, **{k: False for k in core.SAFETY}})
    print(parent.json.dumps({"minutes": minutes, "rows": len(rows), "paired_comparisons": len(comparisons)}))


def finalize():
    registered(); c = config()
    if (OUT / "results.json").exists(): raise ValueError("final analysis already exists")
    frames = [read(OUT / "results_{}m.json".format(h)) for h in c["timeframes_minutes"]]
    comparisons = [r for f in frames for r in f["comparisons"]]
    observed, bounds, critical = [r["difference"] for r in comparisons], [], []
    for b in range(2):
        pieces = []
        for h, f in zip(c["timeframes_minutes"], frames):
            path = OUT / "paired_draws_{}m.npz".format(h)
            if parent.digest(path) != f["paired_draws_sha256"]: raise ValueError("bootstrap draws changed")
            with np.load(path, allow_pickle=False) as a: pieces.append(a["bank{}".format(b)].copy())
        lower, q = causal.simultaneous_lcb(observed, np.concatenate(pieces, axis=1)); bounds.append(lower); critical.append(q)
    for k, row in enumerate(comparisons): row["simultaneous_lcb95_by_bank"] = [float(v[k]) for v in bounds]
    parent.write(OUT / "results.json", {"rows": [r for f in frames for r in f["rows"]], "comparisons": comparisons,
        "simultaneous_family_cells": len(comparisons), "max_t_critical_values": critical,
        "fit_attempts": sum(f["fit_attempts"] for f in frames), "fitted_estimators": sum(f["fitted_estimators"] for f in frames),
        "prior_only": sum(f["prior_only"] for f in frames), "warning_count": sum(len(f["warnings"]) for f in frames),
        "fixed_user_bands": c["neutral_bands_pct"], "no_selection_on_test": True,
        "independent_validation": False, **{k: False for k in core.SAFETY}})
    print(parent.json.dumps({"stage": "finalize", "rows": sum(len(f["rows"]) for f in frames),
                            "fits": sum(f["fitted_estimators"] for f in frames), "family": len(comparisons)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["checks", "init", "fit_origin", "score_frame", "finalize"])
    parser.add_argument("--origin", type=int, default=0); parser.add_argument("--minutes", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "fit_origin": fit_origin(args.origin)
    elif args.stage == "score_frame": score_frame(args.minutes)
    else: globals()[args.stage]()
