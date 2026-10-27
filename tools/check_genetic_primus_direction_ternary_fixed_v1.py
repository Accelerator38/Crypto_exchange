"""Finite independent ledger/metrics recomputation; never fit a model."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_ternary_fixed_v1 as run
import joblib
import numpy as np


def audit(minutes):
    reg, c = run.registered(), run.config()
    names, threshold = run.methods(c), {1: .020, 10: .064, 30: .109, 240: .320}[minutes]
    for p, sha in reg["data_pins"].items():
        if "tf_{}m/".format(minutes) in p or "tf_{}m_origin_".format(minutes) in p or "/tf_{}m.npz".format(minutes) in p:
            assert run.parent.digest(run.ROOT/p) == sha
    data = run.parent.load(minutes); times = data["times"]
    daily = np.zeros((8, len(names), 61, 3, 3), np.int64)
    loss = np.zeros((4, 8, 3)); model_count, warnings = 0, 0
    for number in range(21):
        origin, width = run.parent.START+number*3*run.tri.DAY, minutes*60000
        seed = c["seed"]+minutes*1000+number
        train, cal = run.causal.split(times, origin, width, calibration_days=7, seed=seed)
        train = train[times[train+1] < origin-7*run.tri.DAY-7*width]
        cal = cal[times[cal+1] < origin-7*width]
        test = np.flatnonzero((times >= origin) & (times < min(origin+3*run.tri.DAY, run.parent.END)))
        prefix = run.OUT / "origin_{:02d}".format(number) / "tf_{}m".format(minutes)
        meta = run.read(prefix.with_suffix(".json")); path = prefix.with_suffix(".npz")
        assert meta["code_pins"] == reg["code_pins"] and run.parent.digest(path) == meta["ledger_sha256"]
        assert meta["fold"]["train_times_sha256"] == run.parent.identity(times[train])
        assert meta["fold"]["calibration_times_sha256"] == run.parent.identity(times[cal])
        assert meta["fold"]["train_latest_label_ms"] < meta["fold"]["train_cutoff_exclusive_ms"]
        assert meta["fold"]["calibration_latest_label_ms"] < meta["fold"]["calibration_cutoff_exclusive_ms"]
        with np.load(path, allow_pickle=False) as a: ledger = {k: a[k].copy() for k in a.files}
        assert np.array_equal(ledger["times"], times[test]) and float(ledger["band_pct"]) == threshold
        r = (data["closes"][test+1]/data["closes"][test]-1)*100
        truth = np.where(np.abs(r) <= threshold, 0, np.where(r > threshold, 1, -1)).astype(np.int8)
        assert np.array_equal(truth, ledger["labels"])
        rtrain = (data["closes"][train+1]/data["closes"][train]-1)*100
        train_labels = np.where(np.abs(rtrain) <= threshold, 0, np.where(rtrain > threshold, 1, -1)).astype(np.int8)
        counts = np.stack([(train_labels == state).sum(axis=0) for state in [-1, 0, 1]], axis=-1)
        x = np.asarray(data["features"][train], float).reshape(-1, 12)
        mean, scale = x.mean(axis=0), x.std(axis=0); scale[scale < 1e-12] = 1
        prior = (counts+1)/(len(train)+3)
        assert np.array_equal(prior, ledger["train_priors"])
        for k, candidate in enumerate(c["candidates"]):
            record = meta["models"][k]; model_path = prefix.parent / record["file"]
            digest = run.parent.digest(model_path)
            assert digest == record["sha256"] == str(ledger["model_sha256"][k])
            model = joblib.load(model_path)
            assert model["candidate"] == candidate
            assert np.array_equal(model["mean"], mean) and np.array_equal(model["scale"], scale)
            assert np.array_equal(model["train_counts"], counts) and np.array_equal(model["train_priors"], prior)
            assert (model["estimator"] is None) == (counts.sum(axis=0).min() < 32)
            if candidate.endswith("_balanced") and model["estimator"] is not None: assert np.allclose(model["class_weight_mass"], [len(x)/3]*3)
            else: assert np.array_equal(model["class_weight_mass"], counts.sum(axis=0))
            p = run.core.probabilities(model, data["features"][test])
            assert np.array_equal(p, ledger["probability"][k])
            observed = np.take_along_axis(p, (truth+1)[..., None], axis=-1)[..., 0]
            loss[k, :, 0] += -np.log(np.clip(observed, 1e-15, 1)).sum(axis=0)
            loss[k, :, 1] += ((p-np.eye(3)[truth+1])**2).sum(axis=-1).sum(axis=0)
            loss[k, :, 2] += len(truth)
            model_count += 1; warnings += bool(model["warnings"])
        u = np.random.default_rng(np.random.SeedSequence([seed, 522])).random(truth.shape)
        assert np.array_equal(u, ledger["uniform"])
        expected_legacy = []
        adaptive_no = run.read(run.ADAPTIVE / "registration.json")["contract"]["schemes"].index("train_q50")
        for k, legacy_path in enumerate(run.legacy_paths(minutes, number)):
            with np.load(legacy_path, allow_pickle=False) as old:
                if k < 2: expected_legacy.append(np.where(old["probability"] >= .5, 1, -1).astype(np.int8))
                else: expected_legacy.append(run.tri.STATES[np.argmax(old["probability"][adaptive_no if k == 2 else 0], axis=-1)])
        assert np.array_equal(expected_legacy, ledger["legacy_predictions"])
        cdf = prior.cumsum(axis=-1)
        predicted = [run.tri.STATES[np.argmax(p, axis=-1)] for p in ledger["probability"]]+[
            np.zeros_like(truth), np.broadcast_to(run.tri.STATES[prior.argmax(axis=-1)], truth.shape),
            run.tri.STATES[np.minimum((u*3).astype(int), 2)],
            np.where(u < cdf[:, 0], -1, np.where(u < cdf[:, 1], 0, 1)),
            run.tri.labels(ledger["past_return_pct"], threshold), *expected_legacy]
        for asset in range(8):
            for k, prediction in enumerate(predicted):
                day = (ledger["times"]-run.parent.START)//run.tri.DAY
                index = day*9+(truth[:, asset].astype(int)+1)*3+prediction[:, asset].astype(int)+1
                daily[asset, k] += np.bincount(index, minlength=61*9).reshape(61, 3, 3)
    frame = run.read(run.OUT / "results_{}m.json".format(minutes)); rows = frame["rows"]
    for asset, coin in enumerate(c["symbols"]):
        for k, method in enumerate(names):
            row = next(r for r in rows if r["symbol"] == coin and r["method"] == method)
            cm = daily[asset, k].sum(axis=0)
            assert cm.tolist() == row["confusion_down_neutral_up"]
            assert np.trace(cm)/cm.sum() == row["accuracy"]
            denominator = cm.sum(axis=0)+cm.sum(axis=1)
            f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(3), where=denominator > 0)
            assert f1.mean() == row["macro_f1"]
            if k < 4:
                assert np.isclose(row["log_loss"], loss[k, asset, 0]/loss[k, asset, 2], rtol=1e-12)
                assert np.isclose(row["multiclass_brier"], loss[k, asset, 1]/loss[k, asset, 2], rtol=1e-12)
    with np.load(run.OUT / "daily_confusion_{}m.npz".format(minutes), allow_pickle=False) as a: assert np.array_equal(daily, a["daily"])
    with np.load(run.OUT / "paired_draws_{}m.npz".format(minutes), allow_pickle=False) as a:
        for bank_no, seed in enumerate(c["bootstrap"]["seeds"]):
            with np.load(run.OUT / "bootstrap_bank_{}.npz".format(seed), allow_pickle=False) as b: bank = b["indices"].copy()
            assert np.array_equal(bank, run.causal.block_bank(61, 1000, 3, seed))
            weights = run.tri.calendar_weights(bank, 61); expected = []
            for asset in range(8):
                for candidate in range(4):
                    model = run.tri.bootstrap_metrics(daily[asset, candidate], weights)
                    for control in range(4, 9):
                        other = run.tri.bootstrap_metrics(daily[asset, control], weights)
                        for metric in ["accuracy", "macro_f1"]: expected.append(model[metric]-other[metric])
            assert np.array_equal(np.array(expected).T, a["bank{}".format(bank_no)])
    if minutes == 240:
        final = run.read(run.OUT / "results.json")
        observed = [r["difference"] for r in final["comparisons"]]
        for bank_no in range(2):
            pieces = []
            for h in [1, 10, 30, 240]:
                with np.load(run.OUT / "paired_draws_{}m.npz".format(h), allow_pickle=False) as a: pieces.append(a["bank{}".format(bank_no)].copy())
            bounds, critical = run.causal.simultaneous_lcb(observed, np.concatenate(pieces, axis=1))
            assert critical == final["max_t_critical_values"][bank_no]
            assert bounds.tolist() == [r["simultaneous_lcb95_by_bank"][bank_no] for r in final["comparisons"]]
    run.parent.write(run.OUT / "audit_{}m.json".format(minutes), {"status": "PASS", "minutes": minutes,
        "band_pct": threshold, "models_verified": model_count, "metric_cells_recomputed": len(rows),
        "paired_bootstrap_cells_recomputed": 320, "warning_count": int(warnings),
        "historical_refits": 0, "checker_sha256": run.parent.digest(Path(__file__)), "independent_validation": False})
    print(run.parent.json.dumps({"audit": "PASS", "minutes": minutes, "models": model_count, "metric_cells": len(rows), "refits": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--minutes", type=int, required=True)
    audit(parser.parse_args().minutes)
