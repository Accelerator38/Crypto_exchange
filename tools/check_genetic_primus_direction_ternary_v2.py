"""Recompute ternary labels, model/scaler provenance and metrics without fitting."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import run_genetic_primus_direction_selective_v1 as parent
from exia.genetic_primus import direction_selective_v1 as causal
from exia.genetic_primus import direction_ternary_v1 as core

BASE = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_v2_20261013"
SUP = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_01pct_v1_20261013"
METHODS = ["ternary_logit", "always_neutral", "majority_train", "random_uniform_3",
           "random_train_prior", "momentum_deadband", "legacy_binary_logistic", "legacy_binary_boosting"]


def read(path):
    value = parent.read(path)
    if "sha256" in value:
        assert parent.identity({k: v for k, v in value.items() if k != "sha256"}) == value["sha256"]
    return value


def audit(minutes):
    data = parent.load(minutes); times = data["times"]
    width, days = minutes*60000, 61
    totals, fitted, sparse, cells, source_pins = 0, 0, 0, 0, {}
    primary_daily = np.zeros((8, 8, days, 3, 3), np.int64)
    for out in (BASE, SUP):
        reg = read(out / "registration.json"); c = reg["contract"]
        for p, sha in reg["code_pins"].items(): assert parent.digest(ROOT/p) == sha
        for p, sha in reg["data_pins"].items():
            if "tf_{}m/".format(minutes) in p or "tf_{}m_origin_".format(minutes) in p:
                assert parent.digest(ROOT/p) == sha
                source_pins[p] = sha
        rows = read(out / "results.json")["rows"]
        cm_sum = np.zeros((len(c["schemes"]), 8, 8, 3, 3), np.int64)
        for origin_no in range(21):
            origin = parent.START+origin_no*3*core.DAY
            seed = c["seed"]+minutes*1000+origin_no
            train, cal = causal.split(times, origin, width, calibration_days=7, seed=seed)
            train = train[times[train+1] < origin-7*core.DAY-7*width]
            cal = cal[times[cal+1] < origin-7*width]
            test = np.flatnonzero((times >= origin) & (times < min(origin+3*core.DAY, parent.END)))
            p = out / "origin_{:02d}".format(origin_no) / "tf_{}m".format(minutes)
            meta = read(p.with_suffix(".json"))
            assert meta["code_pins"] == reg["code_pins"] and parent.digest(p.with_suffix(".npz")) == meta["ledger_sha256"]
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as archive:
                ledger = {k: archive[k].copy() for k in archive.files}
            assert np.array_equal(ledger["times"], times[test])
            assert meta["fold"]["train_times_sha256"] == parent.identity(times[train])
            assert meta["fold"]["calibration_times_sha256"] == parent.identity(times[cal])
            actual_return = 100*(data["closes"][test+1]-data["closes"][test])/data["closes"][test]
            assert np.allclose(actual_return, ledger["return_pct"], rtol=1e-9, atol=1e-11)
            train_return = (data["closes"][train+1]/data["closes"][train]-1)*100
            x = np.asarray(data["features"][train], float)
            expected_mean, expected_scale = x.reshape(-1, 12).mean(axis=0), x.reshape(-1, 12).std(axis=0)
            expected_scale[expected_scale < 1e-12] = 1
            for s, scheme in enumerate(c["schemes"]):
                band = np.median(np.abs(train_return), axis=0) if scheme == "train_q50" else np.full(8, c["fixed_bands_pct"][scheme])
                assert np.array_equal(band, ledger["bands_pct"][s])
                # Inclusive boundary semantics, independently of core.labels().
                r = ledger["return_pct"]
                truth = np.where(np.abs(r) <= band, 0, np.where(r > band, 1, -1)).astype(np.int8)
                assert np.array_equal(truth, ledger["labels"][s])
                model_path = p.parent / meta["models"][s]["file"]
                digest = parent.digest(model_path)
                assert digest == str(ledger["model_sha256"][s]) == meta["models"][s]["model_sha256"]
                model = joblib.load(model_path)  # Only pinned, own local experiment artifacts.
                assert np.array_equal(model["mean"], expected_mean) and np.array_equal(model["scale"], expected_scale)
                counts = core.counts_per_asset(core.labels(train_return, band))
                assert np.array_equal(counts, model["train_counts"])
                assert np.array_equal(model["train_priors"], (counts+1)/(len(train)+3))
                assert np.array_equal(core.probability(model, data["features"][test]), ledger["probability"][s])
                assert (model["estimator"] is None) == (np.min(counts.sum(axis=0)) < 32)
                totals += 1; fitted += model["estimator"] is not None; sparse += model["estimator"] is None
                prior, u = model["train_priors"], ledger["uniform"]
                cdf = prior.cumsum(axis=-1)
                predicted = [core.predict(ledger["probability"][s]), np.zeros_like(truth),
                    np.broadcast_to(core.predict(prior), truth.shape), core.STATES[np.minimum((u*3).astype(int), 2)],
                    np.where(u < cdf[:, 0], -1, np.where(u < cdf[:, 1], 0, 1)),
                    core.labels(ledger["past_return_pct"], band),
                    np.where(ledger["binary_up_probability"][0] >= .5, 1, -1),
                    np.where(ledger["binary_up_probability"][1] >= .5, 1, -1)]
                for asset in range(8):
                    for method_no, pred in enumerate(predicted):
                        cm = np.bincount((truth[:, asset].astype(int)+1)*3+pred[:, asset].astype(int)+1, minlength=9).reshape(3, 3)
                        cm_sum[s, asset, method_no] += cm
                        if out == BASE and scheme == "train_q50":
                            primary_daily[asset, method_no] += core.daily_confusion(times[test], truth[:, asset], pred[:, asset], parent.START, days)
        for s, scheme in enumerate(c["schemes"]):
            for asset, symbol in enumerate(c["symbols"]):
                for method_no, method in enumerate(METHODS):
                    row = next(r for r in rows if r["minutes"] == minutes and r["symbol"] == symbol
                               and r["scheme"] == scheme and r["method"] == method)
                    cm = cm_sum[s, asset, method_no]
                    assert row["confusion_down_neutral_up"] == cm.tolist()
                    assert row["accuracy"] == np.trace(cm)/cm.sum()
                    denominator = cm.sum(axis=0)+cm.sum(axis=1)
                    f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator > 0)
                    assert row["macro_f1"] == f1.mean()
                    cells += 1
    frame = read(BASE / "results_{}m.json".format(minutes))
    with np.load(BASE / "paired_draws_{}m.npz".format(minutes), allow_pickle=False) as archive:
        for k, seed in enumerate([2026101320, 2026101321]):
            with np.load(BASE / "bootstrap_bank_{}.npz".format(seed), allow_pickle=False) as a:
                bank = a["indices"].copy()
            assert np.array_equal(bank, causal.block_bank(61, 1000, 3, seed))
            weights = core.calendar_weights(bank, 61)
            expected = []
            for asset in range(8):
                a = core.bootstrap_metrics(primary_daily[asset, 0], weights)
                for control in range(1, 8):
                    b = core.bootstrap_metrics(primary_daily[asset, control], weights)
                    for metric in ("accuracy", "macro_f1"): expected.append(a[metric]-b[metric])
            assert np.array_equal(np.array(expected).T, archive["bank{}".format(k)])
    parent.write(BASE / "audit_tf_{}m.json".format(minutes), {"status": "PASS", "minutes": minutes,
        "models_reloaded": totals, "trained_estimators": int(fitted), "prior_only": int(sparse),
        "metric_cells_recomputed": cells, "paired_bootstrap_cells_recomputed": 112,
        "synthetic_or_historical_refits": 0, "source_pins_verified": len(source_pins),
        "checker_sha256": parent.digest(Path(__file__)), "independent_validation": False})
    print(json.dumps({"audit": "PASS", "minutes": minutes, "models": totals, "metric_cells": cells, "refits": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--minutes", type=int, required=True)
    audit(parser.parse_args().minutes)
