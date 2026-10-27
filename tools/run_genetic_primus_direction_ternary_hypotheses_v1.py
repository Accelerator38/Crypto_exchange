"""Explicit bounded offline stages. Never imports trading runtime or collector."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import joblib
import sklearn
import run_genetic_primus_direction_selective_v1 as parent
from exia.genetic_primus import direction_ternary_v1 as tri
from exia.genetic_primus import direction_selective_v1 as causal
from exia.genetic_primus import direction_hypotheses_v1 as old
from exia.genetic_primus import direction_ternary_fixed_v1 as fixed
from exia.genetic_primus import direction_ternary_hypotheses_v1 as core

CONFIG = ROOT / "configs/genetic_primus_direction_ternary_hypotheses_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_hypotheses_v1_20261014"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v23.json"
CODE = [CONFIG, Path(__file__), ROOT / "src/exia/genetic_primus/direction_ternary_hypotheses_v1.py",
        ROOT / "src/exia/genetic_primus/direction_hypotheses_v1.py", ROOT / "src/exia/genetic_primus/direction_ternary_fixed_v1.py",
        ROOT / "src/exia/genetic_primus/direction_ternary_v1.py", ROOT / "src/exia/genetic_primus/direction_selective_v1.py",
        ROOT / "tools/run_genetic_primus_direction_selective_v1.py"]

def rel(path): return str(path.relative_to(ROOT)).replace("\\", "/")
def pins(): return {rel(p): parent.digest(p) for p in CODE}
def versions(): return dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)

def read(path):
    value = parent.read(path)
    if "sha256" in value:
        payload = {k: v for k, v in value.items() if k != "sha256"}
        if parent.identity(payload) != value["sha256"]: raise ValueError("JSON self-hash mismatch: "+rel(path))
    return value

def config():
    c = read(CONFIG)
    for minutes in c["timeframes_minutes"]: fixed.band(c, minutes)
    if sys.executable.lower() != str(ROOT / ".venv/Scripts/python.exe").lower(): raise ValueError("workspace venv only")
    return c

def baseout(c): return ROOT / c["parent_output"]
def variants(c, minutes): return [v for v in c["variants"] if minutes in v.get("horizons", c["timeframes_minutes"])]
def methods(c, minutes): return [v["id"] for v in variants(c, minutes)] + c["base_methods"] + c["primary_controls"][:5]
def prefix(root, number, minutes): return root / ("origin_%02d" % number) / ("tf_%sm" % minutes)

def registered():
    c = config(); reg = read(OUT / "registration.json")
    if reg["code_pins"] != pins() or reg["versions"] != versions(): raise ValueError("registered code/environment changed")
    if parent.digest(baseout(c) / "delivery_manifest.json") != reg["base_delivery_manifest_sha256"]: raise ValueError("base manifest changed")
    return c, reg

def base_model(c, number, minutes, candidate="logit_uniform"):
    p = prefix(baseout(c), number, minutes)
    meta = read(p.with_suffix(".json"))
    m = next(m for m in meta["models"] if m["candidate"] == candidate)
    path = p.parent / m["file"]
    if parent.digest(path) != m["sha256"]: raise ValueError("frozen base model changed")
    return joblib.load(path), {"file": rel(path), "sha256": m["sha256"], "origin": number}

def predict(payload, x):
    if payload["kind"] != "ensemble": return core.probability(payload, x)
    members = []
    for member in payload["members"]:
        path = ROOT / member["file"]
        if parent.digest(path) != member["sha256"]: raise ValueError("ensemble member changed")
        members.append(fixed.probabilities(joblib.load(path), x))
    return np.mean(members, axis=0)

def scaler_identity(payload):
    if payload["kind"] == "ensemble": return parent.identity(payload["members"])
    stages = [payload["move"], payload["direction"]] if payload["kind"] == "hierarchy" else [payload["model"]]
    return parent.identity([{k: None if m[k] is None else m[k].tolist() for k in ("mean", "scale", "lower", "upper")} for m in stages])

def train_indices(c, bank, number, minutes, variant):
    origin = parent.START + number*3*tri.DAY; seed = c["seed"]+minutes*1000+number
    train, cal = fixed.fold(bank["times"], origin, minutes, c, seed)
    if "train_cap" in variant:
        original = train.copy()
        train = old.larger_train(bank["times"], train, origin, minutes*60000, variant["train_cap"], seed)
        train = train[bank["times"][train+1] < origin-7*tri.DAY-7*minutes*60000]
        if not np.isin(original, train).all(): raise ValueError("large train must contain base train")
    return train, cal

def checks():
    c = config(); OUT.mkdir(parents=True, exist_ok=True)
    pure = core.direct_checks(); rng = np.random.default_rng(140)
    x = rng.normal(size=(90, 8, 6)); y = np.tile(np.array([-1, 0, 1]), 240).reshape(90, 8)
    times = np.arange(90)*tri.DAY
    records = []
    for kind in ("logistic", "boosting", "mlp", "hierarchy"):
        v = dict(estimator=kind)
        payload = core.fit_variant(x, y, v, c, 14, times, 91*tri.DAY)
        p = core.probability(payload, x[:3])
        assert p.shape == (3, 8, 3) and np.allclose(p.sum(axis=-1), 1)
        if kind == "mlp": assert payload["info"]["parameter_count"] == 83
        records.append(payload["info"])
    assert sum(r["new_fits"] for r in records) == 5
    parent.write(OUT / "checks.json", dict(pure_checks=pure, synthetic_estimator_fits=5,
        synthetic_records=records, code_pins=pins(), **{k: False for k in fixed.SAFETY}))
    print(json.dumps(dict(pure_checks=len(pure), synthetic_fits=5)))

def init():
    c = config(); check = read(OUT / "checks.json")
    if check["code_pins"] != pins(): raise ValueError("checks stale")
    manifest = read(baseout(c) / "delivery_manifest.json")
    data_pins = dict(manifest["artifact_pins"])
    base_reg = read(baseout(c) / "registration.json")
    data_pins.update(base_reg["data_pins"])
    for path, sha in data_pins.items():
        if parent.digest(ROOT / path) != sha: raise ValueError("frozen input changed: "+path)
    parent.write(OUT / "registration.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        contract=c, code_pins=pins(), versions=versions(), data_pins=data_pins,
        base_delivery_manifest_sha256=parent.digest(baseout(c) / "delivery_manifest.json"),
        checks_sha256=parent.digest(OUT / "checks.json"),
        historical_fit_budget=882, synthetic_estimator_fits=5, independent_validation=False,
        **{k: False for k in fixed.SAFETY}))
    previous_path = ROOT / c["previous_registry"]; previous = read(previous_path); previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, registration_date_utc=datetime.now(timezone.utc).isoformat(),
        previous_registry_path=c["previous_registry"], previous_registry_sha256=parent.digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"], family="expanded_fixed_band_three_state_hypotheses",
        registration_path=rel(OUT / "registration.json"), registration_sha256=parent.digest(OUT / "registration.json"),
        historical_fit_budget=882, synthetic_estimator_fits=5, independent_validation=False,
        historical_adaptive_trial_count_complete=False, financial_replays=0, promotion_authority=False))
    parent.write(REGISTRY, previous)
    print(json.dumps(dict(registry="v23", pinned_inputs=len(data_pins), fit_budget=882)))

def fit_origin(number):
    c, reg = registered()
    if number not in range(21): raise ValueError("origin 0..20 required")
    fits = warnings_count = resumed = 0
    for minutes in c["timeframes_minutes"]:
        bank = parent.load(minutes); t = bank["times"]; origin = parent.START+number*3*tri.DAY
        test = np.flatnonzero((t >= origin) & (t < min(origin+3*tri.DAY, parent.END)))
        p = prefix(OUT, number, minutes); p.parent.mkdir(exist_ok=True)
        source = prefix(baseout(c), number, minutes); base_meta = read(source.with_suffix(".json"))
        if parent.digest(source.with_suffix(".npz")) != base_meta["ledger_sha256"]: raise ValueError("base ledger changed")
        with np.load(source.with_suffix(".npz"), allow_pickle=False) as a:
            truth = tri.labels(100*(bank["closes"][test+1]/bank["closes"][test]-1), fixed.band(c, minutes))
            if not np.array_equal(a["times"], t[test]) or not np.array_equal(a["labels"], truth): raise ValueError("base label/time parity")
        probabilities, model_meta, all_folds = [], [], []
        for variant in variants(c, minutes):
            model_path = p.parent / (p.name+"__"+variant["id"]+".joblib")
            model_meta_path = model_path.with_suffix(".json")
            train, cal = train_indices(c, bank, number, minutes, variant)
            if not len(train) or not len(cal): raise ValueError("empty causal fold")
            fold = dict(origin_ms=int(origin), horizon_minutes=minutes, example_step_minutes=minutes,
                decision_minutes=minutes, retrain_days=3, purge_bars=7, n_train_times=len(train),
                train_times_sha256=parent.identity(t[train].tolist()), calibration_times_sha256=parent.identity(t[cal].tolist()),
                test_times_sha256=parent.identity(t[test].tolist()), train_first_ms=int(t[train[0]]),
                train_latest_label_ms=int(t[train[-1]+1]), train_cutoff_exclusive_ms=int(origin-7*tri.DAY-7*minutes*60000),
                calibration_latest_label_ms=int(t[cal[-1]+1]), calibration_cutoff_exclusive_ms=int(origin-7*minutes*60000))
            if fold["train_latest_label_ms"] >= fold["train_cutoff_exclusive_ms"] or fold["calibration_latest_label_ms"] >= fold["calibration_cutoff_exclusive_ms"]: raise ValueError("strict causal endpoints")
            if "train_cap" not in variant and fold["train_times_sha256"] != base_meta["fold"]["train_times_sha256"]: raise ValueError("matched parent train required")
            x_test = core.feature_rows(bank, test, variant["features"], minutes)
            if model_meta_path.exists():
                mm = read(model_meta_path)
                if mm["fold_sha256"] != parent.identity(fold) or mm["code_pins"] != pins() or parent.digest(model_path) != mm["sha256_model"]: raise ValueError("stale/changed completed model")
                payload = joblib.load(model_path); resumed += 1
            else:
                if model_path.exists(): raise ValueError("partial model exists; do not silently refit")
                seed = c["seed"]+minutes*1000+number
                if variant["estimator"] == "ensemble":
                    members = [base_model(c, previous, minutes)[1] for previous in range(max(0, number-2), number+1)]
                    payload = dict(kind="ensemble", members=members, info=dict(new_fits=0, warnings=[], member_count=len(members)))
                else:
                    x = core.feature_rows(bank, train, variant["features"], minutes)
                    y = tri.labels(100*(bank["closes"][train+1]/bank["closes"][train]-1), fixed.band(c, minutes))
                    payload = core.fit_variant(x, y, variant, c, seed, t[train], origin)
                with model_path.open("xb") as stream: joblib.dump(payload, stream, compress=3)
                mm = dict(variant=variant, fold=fold, fold_sha256=parent.identity(fold), info=payload["info"],
                    sha256_model=parent.digest(model_path), file=model_path.name, code_pins=pins(),
                    scaler_sha256=scaler_identity(payload),
                    **{k: False for k in fixed.SAFETY})
                parent.write(model_meta_path, mm)
                fits += payload["info"]["new_fits"]; warnings_count += len(payload["info"]["warnings"])
            probabilities.append(predict(payload, x_test))
            model_meta.append(dict(id=variant["id"], file=model_path.name, sha256=parent.digest(model_path), meta_sha256=parent.digest(model_meta_path), info=payload["info"]))
            all_folds.append(parent.identity(fold))
        values = np.stack(probabilities)
        if p.with_suffix(".json").exists():
            completed = read(p.with_suffix(".json"))
            if parent.digest(p.with_suffix(".npz")) != completed["ledger_sha256"]: raise ValueError("completed ledger changed")
            with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
                if not np.array_equal(values, a["probability"]): raise ValueError("recomputed predictions changed")
        else:
            with p.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, times=t[test], labels=truth, probability=values,
                    model_sha256=np.array([m["sha256"] for m in model_meta]), fold_sha256=np.array(all_folds))
            parent.write(p.with_suffix(".json"), dict(minutes=minutes, origin=number, models=model_meta,
                ledger_sha256=parent.digest(p.with_suffix(".npz")), base_ledger_sha256=base_meta["ledger_sha256"],
                code_pins=pins(), **{k: False for k in fixed.SAFETY}))
    print(json.dumps(dict(origin=number, new_fits=fits, warnings=warnings_count, resumed_models=resumed)))

def score(minutes):
    c, reg = registered(); names = methods(c, minutes); vv = variants(c, minutes)
    daily = np.zeros((8, len(names), 61, 3, 3), np.int64); proper = np.zeros((len(vv), 8, 3)); records = []
    for number in range(21):
        p = prefix(OUT, number, minutes); meta = read(p.with_suffix(".json"))
        if parent.digest(p.with_suffix(".npz")) != meta["ledger_sha256"]: raise ValueError("ledger changed")
        source = prefix(baseout(c), number, minutes)
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as a, np.load(source.with_suffix(".npz"), allow_pickle=False) as b:
            if not np.array_equal(a["labels"], b["labels"]) or not np.array_equal(a["times"], b["times"]): raise ValueError("paired observation mismatch")
            # Fixed parent predictions are frozen; only the first nine are the four new-target models and five controls.
            base_predictions = fixed.predictions(b)[:9]
            predictions = [tri.predict(prob) for prob in a["probability"]] + base_predictions
            if len(predictions) != len(names): raise ValueError("method order")
            for k, pred in enumerate(predictions):
                for asset in range(8): daily[asset, k] += tri.daily_confusion(a["times"], a["labels"][:, asset], pred[:, asset], parent.START, 61)
            for k, prob in enumerate(a["probability"]):
                y = a["labels"]; observed = np.take_along_axis(prob, (y+1)[..., None], axis=-1)[..., 0]
                proper[k, :, 0] += -np.log(np.clip(observed, 1e-15, 1)).sum(axis=0)
                proper[k, :, 1] += ((prob-np.eye(3)[y+1])**2).sum(axis=-1).sum(axis=0)
                proper[k, :, 2] += len(y)
        records += [dict(origin=number, **m) for m in meta["models"]]
    rows = []
    for asset, symbol in enumerate(c["symbols"]):
        for k, name in enumerate(names):
            row = dict(minutes=minutes, symbol=symbol, method=name, **tri.from_confusion(daily[asset, k].sum(axis=0)))
            if k < len(vv): row.update(log_loss=float(proper[k, asset, 0]/proper[k, asset, 2]), multiclass_brier=float(proper[k, asset, 1]/proper[k, asset, 2]))
            rows.append(row)
    weights = []
    for seed in c["bootstrap"]["seeds"]:
        bank = causal.block_bank(61, 1000, 3, seed); path = OUT / ("bootstrap_bank_%s.npz" % seed)
        if path.exists():
            with np.load(path, allow_pickle=False) as a: assert np.array_equal(a["indices"], bank)
        else:
            with path.open("xb") as stream: np.savez_compressed(stream, indices=bank)
        weights.append(tri.calendar_weights(bank, 61))
    comparisons, draws = [], [[], []]
    for asset, symbol in enumerate(c["symbols"]):
        boots = [[tri.bootstrap_metrics(daily[asset, k], w) for w in weights] for k in range(len(names))]
        for k, v in enumerate(vv):
            row = next(r for r in rows if r["symbol"] == symbol and r["method"] == v["id"])
            row["ci95_by_bank"] = {metric: [np.quantile(b[metric], [.025, .975]).tolist() for b in boots[k]] for metric in ("accuracy", "macro_f1")}
            for control in c["primary_controls"]:
                j = names.index(control)
                control_row = next(r for r in rows if r["symbol"] == symbol and r["method"] == control)
                for metric in ("accuracy", "macro_f1"):
                    comparisons.append(dict(minutes=minutes, symbol=symbol, candidate=v["id"], control=control, metric=metric, difference=row[metric]-control_row[metric]))
                    for s in range(2): draws[s].append(boots[k][s][metric]-boots[j][s][metric])
    with (OUT / ("daily_confusion_%sm.npz" % minutes)).open("xb") as stream: np.savez_compressed(stream, daily=daily, names=np.array(names))
    with (OUT / ("paired_draws_%sm.npz" % minutes)).open("xb") as stream: np.savez_compressed(stream, bank0=np.array(draws[0]).T, bank1=np.array(draws[1]).T)
    parent.write(OUT / ("results_%sm.json" % minutes), dict(rows=rows, comparisons=comparisons, model_records=records,
        code_pins=pins(), independent_validation=False, **{k: False for k in fixed.SAFETY}))
    print(json.dumps(dict(minutes=minutes, rows=len(rows), comparisons=len(comparisons), model_artifacts=len(records))))

def finalize():
    c, reg = registered(); rows, comparisons, models, draws = [], [], [], [[], []]
    for minutes in c["timeframes_minutes"]:
        result = read(OUT / ("results_%sm.json" % minutes))
        rows += result["rows"]; comparisons += result["comparisons"]; models += result["model_records"]
        with np.load(OUT / ("paired_draws_%sm.npz" % minutes), allow_pickle=False) as a:
            for s in range(2): draws[s].append(a["bank%s" % s])
    observed = np.array([r["difference"] for r in comparisons]); criticals = []
    for s in range(2):
        lower, critical = causal.simultaneous_lcb(observed, np.concatenate(draws[s], axis=1)); criticals.append(float(critical))
        for r, value in zip(comparisons, lower): r["simultaneous_lcb_bank%s" % s] = float(value)
    gate_rows = []
    for row in rows:
        matched = [r for r in comparisons if (r["minutes"], r["symbol"], r["candidate"]) == (row["minutes"], row["symbol"], row["method"])]
        if matched:
            passed = all(r["simultaneous_lcb_bank0"] > 0 and r["simultaneous_lcb_bank1"] > 0 for r in matched)
            gate_rows.append(dict(minutes=row["minutes"], symbol=row["symbol"], candidate=row["method"], all_controls_both_metrics_both_banks_positive=passed))
    summary = dict(rows=len(rows), candidate_asset_horizon_tasks=len(gate_rows), comparisons=len(comparisons),
        model_artifacts=len(models), actual_historical_fits=sum(m["info"]["new_fits"] for m in models),
        warning_count=sum(len(m["info"]["warnings"]) for m in models),
        research_gates_passed=sum(r["all_controls_both_metrics_both_banks_positive"] for r in gate_rows),
        max_t_criticals=criticals, independent_validation=False, **{k: False for k in fixed.SAFETY})
    if len(models) != 882 or len(comparisons) != 4704 or summary["actual_historical_fits"] > 882: raise ValueError("registered budget/family mismatch")
    parent.write(OUT / "results.json", dict(rows=rows, comparisons=comparisons, gates=gate_rows, code_pins=pins()))
    parent.write(OUT / "summary.json", summary)
    print(json.dumps(summary))

def audit(minutes):
    c, reg = registered(); models = 0; max_error = 0.; audits = []
    bank = parent.load(minutes)
    for number in range(21):
        p = prefix(OUT, number, minutes); meta = read(p.with_suffix(".json"))
        if meta["code_pins"] != pins() or parent.digest(p.with_suffix(".npz")) != meta["ledger_sha256"]: raise ValueError("ledger/code changed")
        origin = parent.START+number*3*tri.DAY
        test = np.flatnonzero((bank["times"] >= origin) & (bank["times"] < min(origin+3*tri.DAY, parent.END)))
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
            truth = tri.labels(100*(bank["closes"][test+1]/bank["closes"][test]-1), fixed.band(c, minutes))
            assert np.array_equal(truth, a["labels"]) and np.array_equal(bank["times"][test], a["times"])
            for k, variant in enumerate(variants(c, minutes)):
                m = meta["models"][k]; path = p.parent / m["file"]; mm = read(path.with_suffix(".json"))
                assert parent.digest(path) == m["sha256"] == a["model_sha256"][k]
                assert parent.digest(path.with_suffix(".json")) == m["meta_sha256"]
                train, cal = train_indices(c, bank, number, minutes, variant)
                assert parent.identity(bank["times"][train].tolist()) == mm["fold"]["train_times_sha256"]
                assert parent.identity(bank["times"][cal].tolist()) == mm["fold"]["calibration_times_sha256"]
                assert mm["fold_sha256"] == a["fold_sha256"][k]
                payload = joblib.load(path); assert scaler_identity(payload) == mm["scaler_sha256"]
                x = core.feature_rows(bank, test, variant["features"], minutes)
                fresh = predict(payload, x); error = float(np.max(np.abs(fresh-a["probability"][k])))
                assert error < 1e-14; max_error = max(max_error, error); models += 1
        audits.append(parent.digest(p.with_suffix(".npz")))
    # Independent accumulation of all confusion cells from saved predictions, not the score helper.
    names = methods(c, minutes); cm = np.zeros((8, len(names), 3, 3), np.int64)
    for number in range(21):
        with np.load(prefix(OUT, number, minutes).with_suffix(".npz"), allow_pickle=False) as a, np.load(prefix(baseout(c), number, minutes).with_suffix(".npz"), allow_pickle=False) as b:
            pp = [tri.predict(prob) for prob in a["probability"]]+fixed.predictions(b)[:9]
            for asset in range(8):
                for k, pred in enumerate(pp):
                    np.add.at(cm[asset, k], (a["labels"][:, asset]+1, pred[:, asset]+1), 1)
    with np.load(OUT / ("daily_confusion_%sm.npz" % minutes), allow_pickle=False) as a: assert np.array_equal(cm, a["daily"].sum(axis=2))
    rr = read(OUT / ("results_%sm.json" % minutes))["rows"]
    for asset, symbol in enumerate(c["symbols"]):
        for k, name in enumerate(names):
            row = next(r for r in rr if r["symbol"] == symbol and r["method"] == name)
            assert np.array_equal(row["confusion_down_neutral_up"], cm[asset, k])
            assert row["accuracy"] == np.trace(cm[asset, k])/cm[asset, k].sum()
    parent.write(OUT / ("audit_%sm.json" % minutes), dict(models=models, metric_cells=len(rr), max_probability_error=max_error,
        recomputed_confusion=True, code_pins=pins(), estimator_fits=0, **{k: False for k in fixed.SAFETY}))
    print(json.dumps(dict(minutes=minutes, audited_models=models, metric_cells=len(rr), max_error=max_error)))

def report():
    import matplotlib.pyplot as plt
    c, reg = registered(); r = read(OUT / "results.json"); summary = read(OUT / "summary.json")
    audits = [read(OUT / ("audit_%sm.json" % h)) for h in c["timeframes_minutes"]]
    assert sum(a["models"] for a in audits) == 882
    rows = r["rows"]; labels = {v["id"]: v["id"] for v in c["variants"]}
    medians = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    for ax, h in zip(axes.flat, c["timeframes_minutes"]):
        names = [v["id"] for v in variants(c, h)]+c["base_methods"]+["majority_train"]
        matrix = np.array([[next(x["accuracy"] for x in rows if x["minutes"] == h and x["symbol"] == coin and x["method"] == name) for coin in c["symbols"]] for name in names])*100
        im = ax.imshow(matrix, vmin=25, vmax=60, cmap="viridis", aspect="auto")
        ax.set_xticks(range(8), [s.split('/')[0] for s in c["symbols"]]); ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title("Горизонт %s мин — точность трёх состояний, %%" % h)
        for i in range(len(names)):
            for j in range(8): ax.text(j, i, "%.1f" % matrix[i, j], ha="center", va="center", fontsize=8, color="white" if matrix[i, j] < 43 else "black")
        for name in names:
            selected = [x for x in rows if x["minutes"] == h and x["method"] == name]
            medians.append(dict(minutes=h, method=name, median_accuracy=float(np.median([x["accuracy"] for x in selected])), median_macro_f1=float(np.median([x["macro_f1"] for x in selected]))))
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Доля правильных прогнозов, %", shrink=.6)
    fig.savefig(OUT / "accuracy_by_coin.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained", sharey=True)
    for ax, h in zip(axes, c["timeframes_minutes"]):
        names = [v["id"] for v in variants(c, h)]+["logit_uniform", "boost_uniform", "majority_train"]
        values = []
        for name in names:
            selected = [x for x in rows if x["minutes"] == h and x["method"] == name]
            values.append([np.median([x["recall_down_neutral_up"][k] for x in selected])*100 for k in range(3)])
        matrix = np.array(values)
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="magma", aspect="auto")
        ax.set_xticks(range(3), ["Вниз", "Нейтрально", "Вверх"], rotation=25); ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title("%s мин: медиана recall, %%" % h)
        for i in range(len(names)):
            for j in range(3): ax.text(j, i, "%.0f" % matrix[i,j], ha="center", va="center", fontsize=8, color="white" if matrix[i,j] < 60 else "black")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Полнота истинного класса, %", shrink=.7)
    fig.savefig(OUT / "state_recall.png", dpi=150); plt.close(fig)
    lines = ["# Расширенная трёхрежимная матрица прогнозов", "", "Дата расчёта: 14 октября 2026. Период: 1 августа — 30 сентября 2026 UTC.", "",
        "## Итог", "", "Получено %s модельных артефактов, выполнено %s исторических estimator fits; базовые модели не переобучались. Покрытие 100%%. Проверено %s новых сочетаний модель/валюта/горизонт и %s парных сравнений. Полный исследовательский gate прошли %s сочетаний. Это раскрытый исследовательский период, не независимая валидация и не допуск к торгам." % (summary["model_artifacts"], summary["actual_historical_fits"], summary["candidate_asset_horizon_tasks"], summary["comparisons"], summary["research_gates_passed"]), "",
        "## Данные и расчёт", "", "Исходные memmap OHLCV times/closes/features взяты из direction_multiscale_v1_20261012, хеши данных и родительских результатов закреплены в registration.json. 8 активов; 21 причинный origin; переобучение каждые 3 дня. Окно 30 дней включает 7 дней calibration, которые не используются для подбора параметров этого опыта. Только созревшие labels; purge 7 баров; строгие endpoint cutoffs. Единый scaler обучается на train каждого origin (у иерархии отдельный scaler для каждого этапа). Все исходы, включая нейтральные, входят в общую задачу обучения.", "",
        "r = 100 × (close[t+h] / close[t] − 1). Классы: вниз при r < −δ, нейтрально при |r| ≤ δ, вверх при r > δ. Пороги в процентах: 1m ±0,020; 10m ±0,064; 30m ±0,109; 240m ±0,320. Прогноз на каждом допустимом интервале, итоговый прогноз argmax трёх вероятностей. Accuracy = trace(confusion)/N; macro-F1 = среднее F1 трёх состояний. Таблицы не являются бинарной точностью вверх/вниз.", "",
        "## Гипотезы и бюджет", "", "Лаги: 24 признака, logit 75 коэффициентов; контекст: 20 признаков, logit 63; общая модель с intercept активов: 19 признаков, 60 коэффициентов; robust/recency/large: 12 признаков, 39 коэффициентов. Коэффициенты multiclass logit номинальные, с избыточностью softmax. MLP 6→8→3: 83 параметра, tanh/LBFGS, максимум 80 итераций/160 оценок. Boosting: 32 итерации, 3 дерева на итерацию, до 7 листьев (до 672 leaf values плюс splits). Иерархия: две бинарные logit по 13 параметров, всего 26. Ансамбль использует 1–3 уже сохранённые logit_uniform модели, предсказывающие текущие признаки, без нового fit.", "",
        "Recency90 взвешивает ВСЕ три исхода по exp2(−age/90d), scaler невзвешенный. Robust границы 0,5/99,5 percentiles определяются только на train. Large-train: 16000 временных строк только на 1m, исходный поднабор включён; на остальных горизонтах исходный cap не ограничивал данные. 240m context использует доступный lag 4h вместо недоступного 1h. Прежний binary selective flat gate заменён full-coverage иерархией band-move/neutral → up/down; это изменённая гипотеза, не перенос бинарного результата.", "",
        "Две общие bootstrap banks: календарные блоки 3 дня, 1000 выборок, 61 день. Односторонний max-t LCB одновременно по всем 4704 сравнениям расширения (accuracy и macro-F1, 7 controls, все активы/горизонты/варианты). Второй seed не заменяет новые данные. Семейство не исправляет неизвестное полное число исторических adaptive trials; глобальная исследовательская множественность остаётся ограничением.", "",
        "## Медианы по восьми валютам", "", "| Горизонт, мин | Метод | Accuracy, % | Macro-F1 |", "|---:|---|---:|---:|"]
    for m in medians: lines.append("| %s | %s | %.2f | %.4f |" % (m["minutes"], m["method"], m["median_accuracy"]*100, m["median_macro_f1"]))
    lines += ["", "![Точность по активам](accuracy_by_coin.png)", "", "Рисунок 1. Все расширенные варианты и базовые модели на одинаковых трёхклассовых labels. Цветовая шкала фиксирована, числа показывают точное значение.", "", "![Полнота классов](state_recall.png)", "", "Рисунок 2. Раздельная полнота истинных классов: рост общей accuracy за счёт нейтрального класса не равен точности направления движения.", "", "## Проверки и ограничения", "", "%s synthetic estimator fits; независимое от scorer накопление confusion и повторный predict всех 882 моделей без обучения. Warning count: %s. Предупреждения и достижение MLP лимита не скрываются и не исправляются подбором на этом тесте. Все модельные SHA, hashes folds, scaler, cutoff, data, code, bootstrap фиксируются. В audit проверены SHA и точное воспроизведение вероятностей. Финансовая доходность, fees/funding, стакан, fills, live и prospective не проверялись. Капитальные графики здесь не строятся: пользователь разрешил отдельный прогнозный блок без прибыли." % (c["synthetic_estimator_fits"], summary["warning_count"]), "",
        "## Дальнейшие действия", "", "Не выбирать порог по августу–сентябрю и не интерпретировать neutral accuracy как 80–90% прогноза движения. Выбрать компактный заранее заданный shortlist по устойчивости относительно controls и по обеим направленным recall; заморозить его. Проверить на действительно новом временном отрезке с теми же bands, causal folds и единственным финальным анализом. Для минутного сигнала отдельно нужны многодневные синхронные trades/L2; доступный однодневный BTC пример недостаточен. Не возвращать GA/большую MLP без подтверждения независимой добавки.", "", "## Артефакты", "", "registration.json — контракт и input hashes; configs/genetic_primus_objective_registry_v23.json — append-only связанная регистрация; origin_*/tf_* — модели, fold/scaler/model SHA и probabilities; results.json — все метрики, интервалы и сравнения; summary.json — бюджет и gates; audit_*m.json — независимые проверки; DETAILS_RU.md — подробные таблицы по каждой валюте."]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines)+"\n")
    detail = ["# Все модели: статистика по каждой валюте", "", "Порядок состояний: вниз / нейтрально / вверх. Точность относится ко всем трём классам; neutral band не означает буквально нулевую цену.", ""]
    for h in c["timeframes_minutes"]:
        for coin in c["symbols"]:
            detail += ["## %s — %s мин" % (coin, h), "", "| Метод | N | Accuracy, % | Macro-F1 | Recall вниз/нейтр./вверх, % | Доля направленных вызовов, % | Точность направленных вызовов*, % |", "|---|---:|---:|---:|---|---:|---:|"]
            for row in [x for x in rows if x["minutes"] == h and x["symbol"] == coin]:
                recalls = " / ".join("—" if v is None else "%.1f" % (100*v) for v in row["recall_down_neutral_up"])
                precision = row["directional_precision_including_false_moves"]
                detail.append("| %s | %s | %.2f | %.4f | %s | %.2f | %s |" % (row["method"], row["n"], row["accuracy"]*100, row["macro_f1"], recalls, row["directional_call_fraction"]*100, "—" if precision is None else "%.2f" % (100*precision)))
            detail += ["", "* Верный направленный вызов / все направленные вызовы, включая ложные вызовы движения на neutral. Это условная метрика; не accuracy при 100% покрытии.", ""]
    with (OUT / "DETAILS_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(detail)+"\n")
    parent.write(OUT / "report_statistics.json", dict(medians=medians, code_pins=pins()))
    artifact_pins = {rel(p): parent.digest(p) for p in OUT.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"}
    artifact_pins.update(pins()); artifact_pins[rel(REGISTRY)] = parent.digest(REGISTRY)
    parent.write(OUT / "delivery_manifest.json", dict(artifact_pins=artifact_pins, independent_validation=False,
        **{k: False for k in fixed.SAFETY}))
    print(json.dumps(dict(report=rel(OUT / "REPORT_RU.md"), detailed_metric_cells=len(rows), artifact_pins=len(artifact_pins))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["checks", "init", "origin", "score", "finalize", "audit", "report"])
    parser.add_argument("--number", type=int); parser.add_argument("--minutes", type=int)
    args = parser.parse_args()
    if args.stage == "origin": fit_origin(args.number)
    elif args.stage in ("score", "audit"): globals()[args.stage](args.minutes)
    else: globals()[args.stage]()
