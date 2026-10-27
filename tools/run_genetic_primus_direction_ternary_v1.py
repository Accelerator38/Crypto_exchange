"""Three-state historical research. Each command handles one origin or one horizon."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_ternary_v1 as core
from exia.genetic_primus import direction_selective_v1 as causal
import run_genetic_primus_direction_selective_v1 as parent

CONFIG = ROOT / "configs/genetic_primus_direction_ternary_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_v1_20261013"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v19.json"
SAFETY = ("credentials_allowed", "orders_enabled", "paper", "live", "runtime_authority", "network_allowed", "promotion_authority")
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_ternary_v1.py",
        ROOT / "src/exia/genetic_primus/direction_selective_v1.py", Path(parent.__file__)]
METHODS = ["ternary_logit", "always_neutral", "majority_train", "random_uniform_3",
           "random_train_prior", "momentum_deadband", "legacy_binary_logistic", "legacy_binary_boosting"]


def read(path):
    value = parent.read(path)
    if Path(path).is_relative_to(OUT) and "sha256" in value:
        payload = {k: v for k, v in value.items() if k != "sha256"}
        if parent.identity(payload) != value["sha256"]: raise ValueError("artifact self-hash mismatch")
    return value


def pins():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): parent.digest(p) for p in CODE}


def config():
    c = read(CONFIG)
    if any(c[k] is not False for k in SAFETY): raise ValueError("all safety permissions must remain false")
    return c


def registered():
    r = read(OUT / "registration.json")
    if r["code_pins"] != pins(): raise ValueError("frozen experiment changed; use a new version")
    return r


def checks():
    c = config(); passed = core.direct_checks()
    rng = np.random.default_rng(18)
    x = rng.normal(size=(160, 8, 12))
    y = core.STATES[(np.arange(160*8).reshape(160, 8) % 3)]
    m1 = core.fit(x[:120], y[:120], c, 11)
    changed = x.copy(); changed[120:] += 1000
    m2 = core.fit(changed[:120], y[:120], c, 11)
    assert np.array_equal(m1["mean"], m2["mean"]) and np.array_equal(m1["scale"], m2["scale"])
    assert np.array_equal(core.probability(m1, x[120:]), core.probability(m2, x[120:]))
    assert m1["nominal_coefficient_count"] == 39
    passed.extend(["three_class_fit_has_39_coefficients", "future_suffix_does_not_change_scaler_or_predictions"])
    sparse = core.fit(x[:120], np.zeros((120, 8), np.int8), c, 11)
    assert sparse["estimator"] is None and np.all(core.predict(core.probability(sparse, x[120:])) == 0)
    passed.append("missing_directional_classes_use_disclosed_train_prior_not_fictitious_training")
    u = np.broadcast_to(np.array([.1, .5, .99])[:, None], (3, 8)).copy()
    fixture = {"probability": np.full((1, 3, 8, 3), 1/3), "uniform": u,
               "train_priors": np.array([np.tile([.2, .5, .3], (8, 1))]),
               "past_return_pct": np.zeros((3, 8)), "bands_pct": np.ones((1, 8)),
               "binary_up_probability": np.ones((2, 3, 8))*.6}
    controls = predictions(fixture, 0)
    assert controls[4][:, 0].tolist() == [-1, 0, 1]
    assert all(np.isin(p, core.STATES).all() for p in controls)
    passed.append("train_prior_random_control_emits_all_three_registered_states")
    OUT.mkdir(exist_ok=True)
    parent.write(OUT / "checks.json", {"checks": passed, "synthetic_estimator_fits": 2,
                 "synthetic_prior_only_calls": 1, "code_pins": pins(), **{k: False for k in SAFETY}})
    print(json.dumps({"checks": len(passed), "synthetic_fits": 2, "status": "PASS"}))


def init():
    c, old = config(), parent.registration()
    check = read(OUT / "checks.json")
    if check["code_pins"] != pins(): raise ValueError("checks are stale")
    data_pins = dict(old["data_pins"])
    for minutes in c["timeframes_minutes"]:
        for origin in range(c["origins"]):
            for kind in ("logistic", "boosting"):
                p = parent.OUT / "tf_{}m_origin_{:02d}_{}.npz".format(minutes, origin, kind)
                data_pins[str(p.relative_to(ROOT)).replace("\\", "/")] = parent.digest(p)
    for path, digest in data_pins.items():
        if parent.digest(ROOT / path) != digest: raise ValueError("parent data changed")
    parent.write(OUT / "registration.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
          "contract": c, "code_pins": pins(), "data_pins": data_pins,
          "parent_registration_sha256": parent.digest(ROOT / c["parent_registration"]),
          "checks_sha256": parent.digest(OUT / "checks.json"), "fit_attempt_budget": c["maximum_fit_attempts"],
          "synthetic_estimator_fits": 2, "independent_validation": False,
          "no_threshold_selection_on_test": True, **{k: False for k in SAFETY}})
    previous_path = ROOT / c["previous_registry"]
    previous = read(previous_path); previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, registration_date_utc=datetime.now(timezone.utc).isoformat(),
                    previous_registry_path=c["previous_registry"], previous_registry_sha256=parent.digest(previous_path))
    previous["objectives"].append({"objective_id": c["objective_id"], "family": "three_state_deadband_semantics",
        "registration_path": str((OUT / "registration.json").relative_to(ROOT)).replace("\\", "/"),
        "registration_sha256": parent.digest(OUT / "registration.json"), "historical_fit_attempt_budget": c["maximum_fit_attempts"],
        "synthetic_estimator_fits": 2, "fixed_thresholds_descriptive_only": True,
        "independent_validation": False, "historical_adaptive_trial_count_complete": False,
        "financial_replays": 0, "promotion_authority": False})
    parent.write(REGISTRY, previous)
    print(json.dumps({"registry": "v19", "data_pins": len(data_pins), "fit_attempt_budget": c["maximum_fit_attempts"]}))


def fit_origin(number):
    reg, c = registered(), config()
    if not 0 <= number < c["origins"]: raise ValueError("origin outside registered budget")
    directory = OUT / "origin_{:02d}".format(number); directory.mkdir(exist_ok=True)
    fitted, fallback, resumed = 0, 0, 0
    for minutes in c["timeframes_minutes"]:
        prefix = directory / "tf_{}m".format(minutes)
        ledger_path, meta_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
        if meta_path.exists():
            meta = read(meta_path)
            if meta["code_pins"] != pins() or parent.digest(ledger_path) != meta["ledger_sha256"]:
                raise ValueError("completed origin receipt/ledger changed")
            for item in meta["models"]:
                if parent.digest(directory / item["file"]) != item["model_sha256"]:
                    raise ValueError("completed model changed")
            resumed += 1; continue
        if ledger_path.exists() or list(directory.glob("tf_{}m__*.joblib".format(minutes))):
            raise ValueError("partial horizon exists; inspect without silently overwriting/refitting")
        for path, digest in reg["data_pins"].items():
            if "tf_{}m/".format(minutes) in path or "tf_{}m_origin_{:02d}_".format(minutes, number) in path:
                if parent.digest(ROOT / path) != digest: raise ValueError("registered data changed")
        bank = parent.load(minutes); times = bank["times"]
        origin, width = parent.START+number*3*core.DAY, minutes*60000
        seed = c["seed"]+minutes*1000+number
        train, calibration = causal.split(times, origin, width, calibration_days=7, seed=seed)
        test = np.flatnonzero((times >= origin) & (times < min(origin+3*core.DAY, parent.END)))
        if len(test) != (min(origin+3*core.DAY, parent.END)-origin)//width or test[-1]+1 >= len(times):
            raise ValueError("incomplete decision/label grid")
        if np.any(times[train]+width >= origin-7*core.DAY-7*width):
            raise ValueError("train/calibration purge violation")
        if np.any(times[calibration]+width >= origin-7*width): raise ValueError("calibration/origin purge violation")
        train_return = (bank["closes"][train+1]/bank["closes"][train]-1)*100
        test_return = (bank["closes"][test+1]/bank["closes"][test]-1)*100
        past_return = (bank["closes"][test]/bank["closes"][test-1]-1)*100
        x, xt = bank["features"][train], bank["features"][test]
        adaptive = core.train_bands(train_return)
        binary = []
        for kind in ("logistic", "boosting"):
            p = parent.OUT / "tf_{}m_origin_{:02d}_{}.npz".format(minutes, number, kind)
            with np.load(p, allow_pickle=False) as archive:
                if not np.array_equal(archive["times"], times[test]): raise ValueError("binary forecast grid mismatch")
                if not np.array_equal(archive["labels"], core.labels(test_return, 0)):
                    raise ValueError("binary target differs from hand-computed price direction")
                binary.append(archive["probability"].copy())
        probability, labels, bands, priors, model_shas, records = [], [], [], [], [], []
        for scheme in c["schemes"]:
            band = adaptive.copy() if scheme == "train_q50" else np.full(8, c["fixed_bands_pct"][scheme])
            train_y, test_y = core.labels(train_return, band), core.labels(test_return, band)
            model = core.fit(x, train_y, c, seed)
            assert np.allclose(model["mean"], np.asarray(x, float).reshape(-1, 12).mean(axis=0))
            path = directory / "tf_{}m__{}.joblib".format(minutes, scheme)
            with path.open("xb") as stream: joblib.dump(model, stream, compress=3)
            p = core.probability(model, xt)
            assert np.array_equal(p, core.probability(joblib.load(path), xt))
            model_sha = parent.digest(path)
            probability.append(p); labels.append(test_y); bands.append(band); priors.append(model["train_priors"])
            model_shas.append(model_sha)
            records.append({"scheme": scheme, "file": path.name, "model_sha256": model_sha,
                "kind": model["kind"], "train_counts_down_neutral_up_by_asset": model["train_counts"].tolist(),
                "nominal_coefficient_count": model["nominal_coefficient_count"], "warnings": model["warnings"],
                "train_rows_including_neutral": model["train_rows_including_neutral"],
                "scaler_sha256": parent.identity({"mean": model["mean"], "scale": model["scale"]})})
            fitted += model["estimator"] is not None; fallback += model["estimator"] is None
        uniform = np.random.default_rng(np.random.SeedSequence([seed, 319])).random(test_return.shape)
        with ledger_path.open("xb") as stream:
            np.savez_compressed(stream, times=times[test], return_pct=test_return, past_return_pct=past_return,
                probability=np.array(probability), labels=np.array(labels), bands_pct=np.array(bands),
                train_priors=np.array(priors), uniform=uniform, binary_up_probability=np.array(binary),
                model_sha256=np.array(model_shas))
        fold = {"origin_ms": origin, "train_first_ms": int(times[train[0]]), "train_last_decision_ms": int(times[train[-1]]),
                "train_label_cutoff_exclusive_ms": origin-7*core.DAY-7*width,
                "calibration_label_cutoff_exclusive_ms": origin-7*width,
                "train_times_sha256": parent.identity(times[train]), "calibration_times_sha256": parent.identity(times[calibration]),
                "test_times_sha256": parent.identity(times[test]), "n_train_times": len(train),
                "n_calibration_times": len(calibration), "n_test_times": len(test),
                "horizon_minutes": minutes, "example_step_minutes": minutes,
                "decision_minutes": minutes, "retrain_days": 3, "purge_bars": 7}
        parent.write(meta_path, {"fold": fold, "fold_sha256": parent.identity(fold), "models": records,
            "bands_pct": np.array(bands), "ledger_sha256": parent.digest(ledger_path), "code_pins": pins(),
            "calibration_role": "temporal gap parity only; no probability/neutral threshold tuning",
            "model_sha_lookup": "scheme index identifies model for every decision/asset in its probability slice",
            "independent_validation": False, **{k: False for k in SAFETY}})
    print(json.dumps({"origin": number, "fitted_estimators": int(fitted), "prior_only": int(fallback), "resumed_horizons": resumed}))


def predictions(ledger, scheme_no):
    p, u = ledger["probability"][scheme_no], ledger["uniform"]
    prior = ledger["train_priors"][scheme_no]
    shape = u.shape
    cdf = prior.cumsum(axis=-1)
    return [core.predict(p), np.zeros(shape, np.int8), np.broadcast_to(core.predict(prior), shape),
            core.STATES[np.minimum((u*3).astype(int), 2)],
            ((u >= cdf[:, 0])+(u >= cdf[:, 1]).astype(np.int8)-1).astype(np.int8),
            core.labels(ledger["past_return_pct"], ledger["bands_pct"][scheme_no]),
            np.where(ledger["binary_up_probability"][0] >= .5, 1, -1).astype(np.int8),
            np.where(ledger["binary_up_probability"][1] >= .5, 1, -1).astype(np.int8)]


def score_frame(minutes):
    registered(); c = config()
    if minutes not in c["timeframes_minutes"]: raise ValueError("unregistered horizon")
    destination = OUT / "results_{}m.json".format(minutes)
    if destination.exists(): raise ValueError("horizon already scored; no overwrite")
    days = (parent.END-parent.START)//core.DAY
    daily = np.zeros((len(c["schemes"]), 8, len(METHODS), days, 3, 3), np.int64)
    threshold_history, model_records, input_pins = [], [], {}
    for number in range(21):
        prefix = OUT / "origin_{:02d}".format(number) / "tf_{}m".format(minutes)
        receipt = read(prefix.with_suffix(".json")); path = prefix.with_suffix(".npz")
        if receipt["code_pins"] != pins() or parent.digest(path) != receipt["ledger_sha256"]:
            raise ValueError("origin receipt/ledger changed")
        for model in receipt["models"]:
            if parent.digest(prefix.parent/model["file"]) != model["model_sha256"]: raise ValueError("model hash changed")
        with np.load(path, allow_pickle=False) as archive:
            ledger = {k: archive[k].copy() for k in archive.files}
        threshold_history.append(ledger["bands_pct"])
        model_records.extend(receipt["models"])
        input_pins[str(path.relative_to(ROOT)).replace("\\", "/")] = parent.digest(path)
        input_pins[str(prefix.with_suffix(".json").relative_to(ROOT)).replace("\\", "/")] = parent.digest(prefix.with_suffix(".json"))
        for s in range(len(c["schemes"])):
            if not np.array_equal(core.labels(ledger["return_pct"], ledger["bands_pct"][s]), ledger["labels"][s]):
                raise ValueError("ternary label mismatch")
            predicted = predictions(ledger, s)
            for asset in range(8):
                for method_no, pred in enumerate(predicted):
                    daily[s, asset, method_no] += core.daily_confusion(ledger["times"], ledger["labels"][s, :, asset],
                        pred[:, asset], parent.START, days)
    rows, bands_array = [], np.array(threshold_history)
    for s, scheme in enumerate(c["schemes"]):
        for asset, symbol in enumerate(c["symbols"]):
            for k, method in enumerate(METHODS):
                metrics = core.from_confusion(daily[s, asset, k].sum(axis=0))
                rows.append(dict(minutes=minutes, symbol=symbol, scheme=scheme, method=method,
                    band_pct_min=float(bands_array[:, s, asset].min()),
                    band_pct_median=float(np.median(bands_array[:, s, asset])),
                    band_pct_max=float(bands_array[:, s, asset].max()), **metrics))
    primary_no = c["schemes"].index(c["primary_scheme"])
    comparisons, draws = [], [[], []]
    banks, weights, bootstrap_pins = [], [], {}
    for seed in c["primary_bootstrap"]["seeds"]:
        bank_path = OUT / "bootstrap_bank_{}.npz".format(seed)
        expected = causal.block_bank(days, 1000, 3, seed)
        if bank_path.exists():
            with np.load(bank_path, allow_pickle=False) as archive: bank = archive["indices"].copy()
            if not np.array_equal(bank, expected): raise ValueError("common bootstrap bank changed")
        else:
            bank = expected
            with bank_path.open("xb") as stream: np.savez_compressed(stream, indices=bank)
        banks.append(bank); weights.append(core.calendar_weights(bank, days))
        bootstrap_pins[str(bank_path.relative_to(ROOT)).replace("\\", "/")] = parent.digest(bank_path)
    for asset, symbol in enumerate(c["symbols"]):
        model_metrics = core.from_confusion(daily[primary_no, asset, 0].sum(axis=0))
        model_boot = [core.bootstrap_metrics(daily[primary_no, asset, 0], w) for w in weights]
        row = next(r for r in rows if r["scheme"] == "train_q50" and r["symbol"] == symbol and r["method"] == "ternary_logit")
        row["calendar_ci95_by_bank"] = {metric: [np.quantile(b[metric], [.025, .975]).tolist() for b in model_boot]
                                        for metric in ("accuracy", "macro_f1")}
        for k, method in enumerate(METHODS[1:], start=1):
            control_metrics = core.from_confusion(daily[primary_no, asset, k].sum(axis=0))
            control_boot = [core.bootstrap_metrics(daily[primary_no, asset, k], w) for w in weights]
            for metric in ("accuracy", "macro_f1"):
                diff = model_metrics[metric]-control_metrics[metric]
                rep = [model_boot[j][metric]-control_boot[j][metric] for j in range(2)]
                comparisons.append({"minutes": minutes, "symbol": symbol, "scheme": "train_q50",
                    "control": method, "metric": metric, "difference": diff,
                    "paired_lcb95_by_bank": [float(np.quantile(v, .05)) for v in rep]})
                for j in range(2): draws[j].append(rep[j])
    draws_path = OUT / "paired_draws_{}m.npz".format(minutes)
    with draws_path.open("xb") as stream: np.savez_compressed(stream, bank0=np.array(draws[0]).T, bank1=np.array(draws[1]).T)
    parent.write(destination, {"rows": rows, "comparisons": comparisons, "bootstrap_pins": bootstrap_pins,
        "paired_draws_sha256": parent.digest(draws_path), "input_pins": input_pins,
        "fit_attempts": len(model_records), "fitted_estimators": sum(r["kind"] == "three_class_balanced_logit" for r in model_records),
        "prior_only": sum(r["kind"] == "prior_only_sparse_classes" for r in model_records),
        "estimator_warnings": [dict(scheme=r["scheme"], warnings=r["warnings"]) for r in model_records if r["warnings"]],
        "independent_validation": False, **{k: False for k in SAFETY}})
    print(json.dumps({"minutes": minutes, "rows": len(rows), "primary_comparisons": len(comparisons)}))


def finalize():
    registered(); c = config()
    if (OUT / "results.json").exists(): raise ValueError("final analysis already exists")
    frames = [read(OUT / "results_{}m.json".format(m)) for m in c["timeframes_minutes"]]
    comparisons = [r for frame in frames for r in frame["comparisons"]]
    observed = [r["difference"] for r in comparisons]
    bounds, critical_values = [], []
    for bank_no in range(2):
        pieces = []
        for minutes, frame in zip(c["timeframes_minutes"], frames):
            path = OUT / "paired_draws_{}m.npz".format(minutes)
            if parent.digest(path) != frame["paired_draws_sha256"]: raise ValueError("paired draws changed")
            with np.load(path, allow_pickle=False) as archive: pieces.append(archive["bank{}".format(bank_no)].copy())
        lower, critical = causal.simultaneous_lcb(observed, np.concatenate(pieces, axis=1))
        bounds.append(lower); critical_values.append(critical)
    for k, comparison in enumerate(comparisons):
        comparison["simultaneous_lcb95_by_bank"] = [float(v[k]) for v in bounds]
    parent.write(OUT / "results.json", {"rows": [r for frame in frames for r in frame["rows"]],
        "comparisons": comparisons, "simultaneous_family_cells": len(comparisons), "max_t_critical_values": critical_values,
        "fit_attempts": sum(f["fit_attempts"] for f in frames), "fitted_estimators": sum(f["fitted_estimators"] for f in frames),
        "prior_only": sum(f["prior_only"] for f in frames), "estimator_warning_count": sum(len(f["estimator_warnings"]) for f in frames),
        "no_threshold_selection_on_test": True, "historical_interval_already_revealed": True,
        "independent_validation": False, **{k: False for k in SAFETY}})
    print(json.dumps({"stage": "finalize", "rows": sum(len(f["rows"]) for f in frames), "family_cells": len(comparisons),
                      "fitted_estimators": sum(f["fitted_estimators"] for f in frames)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["checks", "init", "fit_origin", "score_frame", "finalize"])
    parser.add_argument("--origin", type=int, default=0)
    parser.add_argument("--minutes", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "fit_origin": fit_origin(args.origin)
    elif args.stage == "score_frame": score_frame(args.minutes)
    else: globals()[args.stage]()


if __name__ == "__main__": main()
