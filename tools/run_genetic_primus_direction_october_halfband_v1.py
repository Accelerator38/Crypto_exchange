"""Offline October rolling target-band experiment. No networking or runtime."""
from __future__ import annotations
import run_genetic_primus_direction_ternary_hypotheses_v1 as reuse
from exia.genetic_primus import direction_multiscale_v1 as features
import argparse
import json
from pathlib import Path
import numpy as np
import joblib

ROOT = reuse.ROOT
CONFIG = ROOT / "configs/genetic_primus_direction_october_halfband_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_october_halfband_v1_20261014"
INTAKE = ROOT / "Retrodate/genetic_primus_direction_october_20261001_20261013_v1"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v25.json"
DAY = 86400000
NAMES = ["original_logit", "original_boost", "half_logit", "half_boost", "always_neutral", "random_uniform_3", "random_prior_original", "random_prior_half", "majority_original", "majority_half", "momentum_original", "momentum_half"]
CODE = [CONFIG, Path(__file__), ROOT / "tools/run_genetic_primus_direction_ternary_hypotheses_v1.py",
    ROOT / "tools/run_genetic_primus_direction_selective_v1.py",
    ROOT / "src/exia/genetic_primus/direction_ternary_fixed_v1.py", ROOT / "src/exia/genetic_primus/direction_ternary_v1.py",
    ROOT / "src/exia/genetic_primus/direction_selective_v1.py", ROOT / "src/exia/genetic_primus/direction_multiscale_v1.py",
    ROOT / "src/exia/genetic_primus/direction_benchmark_v1.py"]

def rel(p): return reuse.rel(p)
def read(p): return reuse.read(p)
def write(p, value): return reuse.parent.write(p, value)
def digest(p): return reuse.parent.digest(p)
def identity(v): return reuse.parent.identity(v)
def pins(): return {rel(p): digest(p) for p in CODE}
def config():
    c = read(CONFIG); reuse.config()
    for h in c["timeframes_minutes"]: reuse.fixed.band(c, h)
    assert NAMES == c["methods"] and c["maximum_historical_fits"] == 80
    return c

def save_array(path, a):
    with path.open("xb") as stream: np.save(stream, a, allow_pickle=False)

def load(h):
    return {k: np.load(OUT / "data" / ("tf_%sm" % h) / (k+".npy"), mmap_mode="r", allow_pickle=False) for k in ("times", "closes", "features")}

def direct_checks():
    c = config(); checks = reuse.tri.direct_checks()
    rng = np.random.default_rng(25); close = np.exp(rng.normal(0, .01, (160, 8)).cumsum(axis=0))
    a = np.stack([close, close*1.01, close*.99, close, np.ones_like(close)], axis=2)
    before = features.features(close, a)
    changed = a.copy(); changed[101:, :, :4] *= 2
    after = features.features(changed[:, :, 3], changed)
    assert np.array_equal(before[:101], after[:101], equal_nan=True)
    checks.append("full_feature_future_suffix_invariance")
    assert np.array_equal(reuse.tri.labels(np.array([-.02, -.01, 0, .01, .02]), .01), [-1, 0, 0, 0, 1])
    checks.append("half_band_inclusive_edges")
    times = c["test_start_ms"]-40*DAY+np.arange(40*6)*240*60000
    train, cal = reuse.fixed.fold(times, c["test_start_ms"], 240, c, 25)
    assert times[train[-1]+1] < c["test_start_ms"]-7*DAY-7*240*60000
    assert times[cal[-1]+1] < c["test_start_ms"]-7*240*60000
    checks.append("strict_rolling_fold_label_maturity")
    return checks

def data():
    import pandas as pd
    c = config(); audit = read(INTAKE / "data_audit.json")
    for p, sha in audit["artifact_pins"].items(): assert digest(ROOT / p) == sha
    path = OUT / "data"; path.mkdir(parents=True, exist_ok=True)
    oldroot = reuse.parent.PARENT
    old_t = np.load(oldroot / "minute_open_times.npy", mmap_mode="r", allow_pickle=False)
    old_a = np.load(oldroot / "minute_ohlcv.npy", mmap_mode="r", allow_pickle=False)
    assert old_t[-1]+60000 == c["test_start_ms"]
    new_t = np.arange(c["test_start_ms"], c["test_end_exclusive_ms"], 60000, dtype=np.int64)
    assets, frames = [], []
    for coin in c["symbols"]:
        symbol = coin.split('/')[0]
        frame = pd.concat([pd.read_parquet(INTAKE / "parts" / ("batch_%02d_%s.parquet" % (i, symbol))) for i in range(5)], ignore_index=True)
        assert np.array_equal(frame.timestamp.to_numpy(), new_t) and frame.symbol.eq(coin).all()
        frames.append(frame); assets.append(frame[["open", "high", "low", "close", "volume"]].to_numpy(float))
    new_a = np.stack(assets, axis=1)
    boundary = (new_a[0, :, 0]/old_a[-1, :, 3]-1)*100
    if np.max(np.abs(boundary)) > 10: raise ValueError("unresolved >10% minute boundary jump; do not train")
    for name, value in [("minute_open_times", np.concatenate([old_t, new_t])), ("minute_ohlcv", np.concatenate([old_a, new_a]))]: save_array(path / (name+".npy"), value)
    panel = INTAKE / "october_full8_1m.parquet"
    if panel.exists(): raise ValueError("exclusive consolidated panel required")
    pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"]).to_parquet(panel, index=False)
    source_pins = dict(audit["artifact_pins"])
    for p in [oldroot / "minute_open_times.npy", oldroot / "minute_ohlcv.npy", INTAKE / "data_audit.json", panel]: source_pins[rel(p)] = digest(p)
    write(OUT / "data_audit.json", dict(source_pins=source_pins, raw_bank_pins={rel(p): digest(p) for p in path.glob("minute_*.npy")},
        new_minute_rows=149760, new_minute_times=len(new_t), boundary_open_gap_pct_by_symbol=dict(zip(c["symbols"], boundary.tolist())),
        exact_old_prefix_preserved=True, code_pins=pins(), **{k: False for k in reuse.fixed.SAFETY}))
    print(json.dumps(dict(new_minute_rows=149760, missing_or_duplicate=0, maximum_boundary_gap_pct=float(np.max(np.abs(boundary))))))

def prepare(h):
    c = config(); assert h in c["timeframes_minutes"]
    audit = read(OUT / "data_audit.json")
    assert audit["code_pins"] == pins()
    for p, sha in audit["raw_bank_pins"].items(): assert digest(ROOT / p) == sha
    times, a = features.aggregate(np.load(OUT / "data/minute_open_times.npy", mmap_mode="r"), np.load(OUT / "data/minute_ohlcv.npy", mmap_mode="r"), h)
    closes = a[:, :, 3]; x = features.features(closes, a)
    old = reuse.parent.load(h); n = len(old["times"])
    assert np.array_equal(times[:n], old["times"]) and np.array_equal(closes[:n], old["closes"])
    assert np.array_equal(x[:n], old["features"], equal_nan=True)
    path = OUT / "data" / ("tf_%sm" % h); path.mkdir()
    for key, value in (("times", times), ("closes", closes), ("features", x)): save_array(path / (key+".npy"), value)
    count = int(((times >= c["test_start_ms"]) & (times < c["test_end_exclusive_ms"])).sum())
    assert count == 13*1440//h
    write(path / "audit.json", dict(minutes=h, old_prefix_feature_parity=True, new_test_times=count, code_pins=pins(),
        bank_pins={rel(p): digest(p) for p in path.glob("*.npy")}, final_label_endpoint_ms=int(times[-1]),
        **{k: False for k in reuse.fixed.SAFETY}))
    print(json.dumps(dict(minutes=h, new_test_times=count, historical_features_exact_match=True)))

def init():
    c = config(); pure = direct_checks(); audit = read(OUT / "data_audit.json")
    data_pins = {**audit["source_pins"], **audit["raw_bank_pins"]}
    for h in c["timeframes_minutes"]:
        p = OUT / "data" / ("tf_%sm" % h) / "audit.json"; a = read(p)
        assert a["code_pins"] == pins(); data_pins.update(a["bank_pins"]); data_pins[rel(p)] = digest(p)
    for p, sha in data_pins.items(): assert digest(ROOT / p) == sha
    write(OUT / "registration.json", dict(contract=c, code_pins=pins(), versions=reuse.versions(), data_pins=data_pins,
        pure_checks=pure, historical_fit_budget=80, synthetic_estimator_fits=0,
        newly_acquired_historical=True, prospective=False, independent_final_validation=False,
        **{k: False for k in reuse.fixed.SAFETY}))
    previous_path = ROOT / c["previous_registry"]; previous = read(previous_path); previous.pop("sha256", None)
    previous.update(registry_id=REGISTRY.stem, previous_registry_path=c["previous_registry"], previous_registry_sha256=digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"], family="october_original_half_band_cross_scored_causal_retraining",
        registration_path=rel(OUT / "registration.json"), registration_sha256=digest(OUT / "registration.json"),
        historical_fit_budget=80, newly_acquired_historical=True, prospective=False, independent_validation=False, promotion_authority=False))
    write(REGISTRY, previous)
    print(json.dumps(dict(registry="v25", pure_checks=len(pure), fit_budget=80, data_pins=len(data_pins))))

def registered():
    c = config(); reg = read(OUT / "registration.json")
    assert reg["code_pins"] == pins() and reg["versions"] == reuse.versions()
    return c, reg

def predictions(a, h, c):
    u = a["uniform"]; priors = a["train_priors"]
    result = [reuse.tri.predict(p) for p in a["probability"]]
    result += [np.zeros(u.shape, np.int8), reuse.tri.STATES[np.minimum((u*3).astype(int), 2)]]
    for prior in priors:
        cdf = prior.cumsum(axis=-1); result.append(np.where(u < cdf[:,0], -1, np.where(u < cdf[:,1], 0, 1)).astype(np.int8))
    for prior in priors: result.append(np.broadcast_to(reuse.tri.predict(prior), u.shape))
    for factor in c["training_band_factors"]: result.append(reuse.tri.labels(a["past_return_pct"], c["neutral_bands_pct"][str(h)]*factor))
    assert len(result) == len(NAMES)
    return result

def origin(number):
    c, reg = registered(); assert number in range(5); total = warning_count = 0
    when = c["test_start_ms"]+number*3*DAY
    for h in c["timeframes_minutes"]:
        bank = load(h); times = bank["times"]; seed = c["seed"]+h*1000+number
        train, cal = reuse.fixed.fold(times, when, h, c, seed)
        test = np.flatnonzero((times >= when) & (times < min(when+3*DAY, c["test_end_exclusive_ms"])))
        fold = dict(origin_ms=when, train_first_ms=int(times[train[0]]), n_train_times=len(train), n_calibration_times=len(cal), n_test_times=len(test),
            train_times_sha256=identity(times[train].tolist()), calibration_times_sha256=identity(times[cal].tolist()), test_times_sha256=identity(times[test].tolist()),
            train_latest_label_ms=int(times[train[-1]+1]), train_cutoff_exclusive_ms=when-7*DAY-7*h*60000,
            calibration_latest_label_ms=int(times[cal[-1]+1]), calibration_cutoff_exclusive_ms=when-7*h*60000,
            horizon_minutes=h, example_step_minutes=h, decision_minutes=h, retrain_days=3, purge_bars=7)
        assert fold["train_latest_label_ms"] < fold["train_cutoff_exclusive_ms"] and fold["calibration_latest_label_ms"] < fold["calibration_cutoff_exclusive_ms"]
        path = OUT / ("origin_%02d" % number); path.mkdir(exist_ok=True)
        prefix = path / ("tf_%sm" % h); probs, models, priors = [], [], []
        x = bank["features"][train]; xt = bank["features"][test]
        train_r = 100*(bank["closes"][train+1]/bank["closes"][train]-1)
        for factor in c["training_band_factors"]:
            y = reuse.tri.labels(train_r, c["neutral_bands_pct"][str(h)]*factor)
            for candidate in c["candidates"]:
                name = NAMES[len(models)]; model_path = path / (prefix.name+"__"+name+".joblib"); meta_path = model_path.with_suffix(".json")
                if meta_path.exists():
                    mm = read(meta_path); assert mm["code_pins"] == pins() and mm["fold_sha256"] == identity(fold) and digest(model_path) == mm["model_sha256"]
                    model = joblib.load(model_path)
                else:
                    if model_path.exists(): raise ValueError("partial model exists; no silent refit")
                    model = reuse.fixed.fit(x, y, candidate, c, seed)
                    with model_path.open("xb") as stream: joblib.dump(model, stream, compress=3)
                    scaler_sha = identity(dict(mean=model["mean"].tolist(), scale=model["scale"].tolist()))
                    write(meta_path, dict(model_sha256=digest(model_path), fold=fold, fold_sha256=identity(fold), training_band_factor=factor,
                        scaler_sha256=scaler_sha, coefficient_count=model["coefficient_count"], train_counts=model["train_counts"].tolist(),
                        class_weight_mass=model["class_weight_mass"].tolist(), warnings=model["warnings"], fitted_iterations=np.asarray(model["fitted_iterations"]).tolist(),
                        estimator_fits=int(model["estimator"] is not None), code_pins=pins(), **{k: False for k in reuse.fixed.SAFETY}))
                    total += int(model["estimator"] is not None); warning_count += len(model["warnings"])
                probs.append(reuse.fixed.probabilities(model, xt))
                models.append(dict(name=name, file=model_path.name, model_sha256=digest(model_path), meta_sha256=digest(meta_path)))
                if candidate == "logit_uniform": priors.append(model["train_priors"])
        if prefix.with_suffix(".json").exists():
            done = read(prefix.with_suffix(".json")); assert digest(prefix.with_suffix(".npz")) == done["ledger_sha256"]
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as a: assert np.array_equal(a["probability"], np.stack(probs))
        else:
            with prefix.with_suffix(".npz").open("xb") as stream:
                np.savez_compressed(stream, times=times[test], return_pct=100*(bank["closes"][test+1]/bank["closes"][test]-1),
                    past_return_pct=100*(bank["closes"][test]/bank["closes"][test-1]-1), probability=np.stack(probs), train_priors=np.array(priors),
                    uniform=np.random.default_rng(np.random.SeedSequence([seed, 522])).random((len(test), 8)), model_sha256=np.array([m["model_sha256"] for m in models]))
            write(prefix.with_suffix(".json"), dict(models=models, fold=fold, fold_sha256=identity(fold), ledger_sha256=digest(prefix.with_suffix(".npz")), code_pins=pins(), **{k: False for k in reuse.fixed.SAFETY}))
    print(json.dumps(dict(origin=number, new_estimator_fits=total, warnings=warning_count)))

def opposite_stats(cm):
    calls = int(cm[:, [0,2]].sum()); wrong = int(cm[0,2]+cm[2,0]); correct = int(cm[0,0]+cm[2,2])
    return dict(directional_calls=calls, opposite_count=wrong, opposite_per_directional_call=None if not calls else wrong/calls,
        opposite_per_all_intervals=wrong/int(cm.sum()), opposite_given_called_and_actual_move=None if not wrong+correct else wrong/(wrong+correct))

def score(h):
    c, reg = registered(); daily = np.zeros((3, 8, 12, 13, 3, 3), np.int64); losses = np.zeros((3, 4, 8, 3))
    for number in range(5):
        p = OUT / ("origin_%02d" % number) / ("tf_%sm" % h); meta = read(p.with_suffix(".json"))
        assert digest(p.with_suffix(".npz")) == meta["ledger_sha256"]
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as a:
            pp = predictions(a, h, c); day = ((a["times"]-c["test_start_ms"])//DAY).astype(int)
            for target, factor in enumerate(c["evaluation_band_factors"]):
                y = reuse.tri.labels(a["return_pct"], c["neutral_bands_pct"][str(h)]*factor)
                for asset in range(8):
                    for k, pred in enumerate(pp): daily[target, asset, k] += np.bincount(day*9+(y[:,asset]+1)*3+pred[:,asset]+1, minlength=13*9).reshape(13,3,3)
                for k, prob in enumerate(a["probability"]):
                    observed = np.take_along_axis(prob, (y+1)[...,None], axis=-1)[...,0]
                    losses[target,k,:,0] += -np.log(np.clip(observed, 1e-15, 1)).sum(axis=0)
                    losses[target,k,:,1] += ((prob-np.eye(3)[y+1])**2).sum(axis=-1).sum(axis=0)
                    losses[target,k,:,2] += len(y)
    rows = []
    weights = []
    for seed in c["bootstrap"]["seeds"]:
        indices = reuse.causal.block_bank(13,1000,3,seed); p = OUT / ("bootstrap_bank_%s.npz" % seed)
        if p.exists():
            with np.load(p,allow_pickle=False) as a: assert np.array_equal(a["indices"],indices)
        else:
            with p.open("xb") as stream: np.savez_compressed(stream,indices=indices)
        weights.append(reuse.tri.calendar_weights(indices,13))
    for target,factor in enumerate(c["evaluation_band_factors"]):
        for asset,coin in enumerate(c["symbols"]):
            for k,name in enumerate(NAMES):
                cm=daily[target,asset,k].sum(axis=0); raw=daily[-1,asset,k].sum(axis=0)
                row=dict(minutes=h,symbol=coin,method=name,evaluation_band_factor=factor,**reuse.tri.from_confusion(cm),**opposite_stats(cm),
                    opposite_raw_sign_per_directional_call=opposite_stats(raw)["opposite_per_directional_call"],
                    opposite_raw_sign_given_actual_move=opposite_stats(raw)["opposite_given_called_and_actual_move"])
                if k<4:
                    row.update(log_loss=float(losses[target,k,asset,0]/losses[target,k,asset,2]),multiclass_brier=float(losses[target,k,asset,1]/losses[target,k,asset,2]))
                    bs=[reuse.tri.bootstrap_metrics(daily[target,asset,k],w) for w in weights]
                    row["ci95_banks"]={metric:[np.quantile(b[metric],[.025,.975]).tolist() for b in bs] for metric in ("accuracy","macro_f1")}
                rows.append(row)
    comparisons,draws=[],[[],[]]
    for asset,coin in enumerate(c["symbols"]):
        boots=[[reuse.tri.bootstrap_metrics(daily[1,asset,k],w) for w in weights] for k in range(12)]
        for candidate,matched in [("half_logit","original_logit"),("half_boost","original_boost")]:
            k=NAMES.index(candidate); row=next(r for r in rows if r["symbol"]==coin and r["method"]==candidate and r["evaluation_band_factor"]==.5)
            for control in (matched,"random_uniform_3","majority_half","momentum_half"):
                j=NAMES.index(control); rr=next(r for r in rows if r["symbol"]==coin and r["method"]==control and r["evaluation_band_factor"]==.5)
                for metric in ("accuracy","macro_f1"):
                    comparisons.append(dict(minutes=h,symbol=coin,candidate=candidate,control=control,metric=metric,difference=row[metric]-rr[metric]))
                    for b in range(2):draws[b].append(boots[k][b][metric]-boots[j][b][metric])
    with (OUT/("daily_confusion_%sm.npz"%h)).open("xb") as stream:np.savez_compressed(stream,daily=daily,names=np.array(NAMES))
    with (OUT/("paired_draws_%sm.npz"%h)).open("xb") as stream:np.savez_compressed(stream,bank0=np.array(draws[0]).T,bank1=np.array(draws[1]).T)
    write(OUT/("results_%sm.json"%h),dict(rows=rows,comparisons=comparisons,code_pins=pins(),**{k:False for k in reuse.fixed.SAFETY}))
    print(json.dumps(dict(minutes=h,metric_cells=len(rows),comparisons=len(comparisons))))

def finalize():
    c,reg=registered();rows=[];comparisons=[];draws=[[],[]];fits=warnings=0
    for h in c["timeframes_minutes"]:
        r=read(OUT/("results_%sm.json"%h));rows+=r["rows"];comparisons+=r["comparisons"]
        with np.load(OUT/("paired_draws_%sm.npz"%h),allow_pickle=False) as a:
            for b in range(2):draws[b].append(a["bank%s"%b])
        for n in range(5):
            for name in NAMES[:4]:
                mm=read(OUT/("origin_%02d"%n)/("tf_%sm__%s.json"%(h,name)))
                fits+=mm["estimator_fits"];warnings+=len(mm["warnings"])
    obs=np.array([r["difference"] for r in comparisons]);criticals=[]
    for b in range(2):
        lower,critical=reuse.causal.simultaneous_lcb(obs,np.concatenate(draws[b],axis=1));criticals.append(float(critical))
        for row,v in zip(comparisons,lower):row["simultaneous_lcb_bank%s"%b]=float(v)
    gates=[]
    for h in c["timeframes_minutes"]:
        for coin in c["symbols"]:
            for candidate in ("half_logit","half_boost"):
                rr=[r for r in comparisons if (r["minutes"],r["symbol"],r["candidate"])==(h,coin,candidate)]
                gates.append(dict(minutes=h,symbol=coin,candidate=candidate,all_registered_comparisons_positive_both_banks=all(r["simultaneous_lcb_bank0"]>0 and r["simultaneous_lcb_bank1"]>0 for r in rr)))
    assert len(rows)==1152 and len(comparisons)==512 and fits<=80
    write(OUT/"results.json",dict(rows=rows,comparisons=comparisons,gates=gates,code_pins=pins()))
    summary=dict(metric_cells=len(rows),paired_comparisons=len(comparisons),estimator_fits=fits,warning_count=warnings,research_gates_passed=sum(g["all_registered_comparisons_positive_both_banks"] for g in gates),
        gate_tasks=64,calendar_days=13,max_t_criticals=criticals,prospective=False,independent_final_validation=False,**{k:False for k in reuse.fixed.SAFETY})
    write(OUT/"summary.json",summary);print(json.dumps(summary))

def audit(h):
    c,reg=registered();bank=load(h);daily=np.zeros((3,8,12,13,3,3),np.int64);model_count=0
    for number in range(5):
        p=OUT/("origin_%02d"%number)/("tf_%sm"%h);meta=read(p.with_suffix(".json"));assert digest(p.with_suffix(".npz"))==meta["ledger_sha256"]
        when=c["test_start_ms"]+number*3*DAY;train,cal=reuse.fixed.fold(bank["times"],when,h,c,c["seed"]+h*1000+number)
        test=np.flatnonzero((bank["times"]>=when)&(bank["times"]<min(when+3*DAY,c["test_end_exclusive_ms"])))
        assert identity(bank["times"][train].tolist())==meta["fold"]["train_times_sha256"]
        assert identity(bank["times"][cal].tolist())==meta["fold"]["calibration_times_sha256"]
        with np.load(p.with_suffix(".npz"),allow_pickle=False) as a:
            assert np.array_equal(a["times"],bank["times"][test])
            assert np.array_equal(a["return_pct"],100*(bank["closes"][test+1]/bank["closes"][test]-1))
            for k,record in enumerate(meta["models"]):
                path=p.parent/record["file"];assert digest(path)==record["model_sha256"]==a["model_sha256"][k]
                mm=read(path.with_suffix(".json"));assert digest(path.with_suffix(".json"))==record["meta_sha256"]
                model=joblib.load(path);assert identity(dict(mean=model["mean"].tolist(),scale=model["scale"].tolist()))==mm["scaler_sha256"]
                assert np.array_equal(reuse.fixed.probabilities(model,bank["features"][test]),a["probability"][k]);model_count+=1
            for target,factor in enumerate(c["evaluation_band_factors"]):
                truth=reuse.tri.labels(a["return_pct"],c["neutral_bands_pct"][str(h)]*factor)
                day=((a["times"]-c["test_start_ms"])//DAY).astype(int)
                for asset in range(8):
                    for k,pred in enumerate(predictions(a,h,c)):np.add.at(daily[target,asset,k],(day,truth[:,asset]+1,pred[:,asset]+1),1)
    with np.load(OUT/("daily_confusion_%sm.npz"%h),allow_pickle=False) as a:assert np.array_equal(daily,a["daily"])
    write(OUT/("audit_%sm.json"%h),dict(models=model_count,metric_cells=288,exact_probability_replay=True,independent_confusion_accumulation=True,estimator_fits=0,code_pins=pins()))
    print(json.dumps(dict(minutes=h,models=model_count,metric_cells=288,exact_replay=True,fits=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["checks","data","prepare","init","origin","score","finalize","audit"])
    parser.add_argument("--minutes",type=int);parser.add_argument("--number",type=int);args=parser.parse_args()
    if args.stage=="checks":print(json.dumps(dict(pure_checks=direct_checks(),estimator_fits=0)))
    elif args.stage in ("prepare","score","audit"):globals()[args.stage](args.minutes)
    elif args.stage=="origin":origin(args.number)
    else:globals()[args.stage]()
