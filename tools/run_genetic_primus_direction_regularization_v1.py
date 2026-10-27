"""Single-variable, bounded offline regularization comparison; no runtime/network."""
from __future__ import annotations
import run_genetic_primus_direction_october_halfband_v2 as adapter
import analyze_genetic_primus_direction_signal_attribution_v1 as attribution
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import joblib

base=adapter.old
ROOT=base.ROOT
PARENT=adapter.OUT
CONFIG=ROOT/"configs/genetic_primus_direction_regularization_v1.json"
OUT=ROOT/"Reports/Exia/Genetic_Primus/direction_regularization_v1_20261018"
REGISTRY=ROOT/"configs/genetic_primus_objective_registry_v28.json"
MODEL_NAMES=["original_C0p1","original_C0p01","half_C0p1","half_C0p01"]
DIRECTION_NAMES=MODEL_NAMES+["train_prior","random_coin"]
CLASS_NAMES=MODEL_NAMES+["majority_original","majority_half","random_uniform_3","momentum_original","momentum_half"]

def config():
    c=base.read(CONFIG);pc,pr=adapter.registered()
    assert all(c[k] is False for k in base.reuse.fixed.SAFETY)
    assert c["challenger_C"]==.01 and c["baseline_C"]==pc["logit"]["C"]==.1
    assert c["model_order"]==MODEL_NAMES and c["classification_methods"]==CLASS_NAMES and c["direction_methods"]==DIRECTION_NAMES
    return c

def pins():
    p=adapter.pins();p.update(attribution.pins())
    for file in (CONFIG,Path(__file__)):p[base.rel(file)]=base.digest(file)
    return p

def binary_losses(prob,truth):
    p,y=np.asarray(prob,float),np.asarray(truth,int)
    assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isin(y,[0,1]).all()
    return (p-y)**2,-y*np.log(np.clip(p,1e-15,1))-(1-y)*np.log(np.clip(1-p,1e-15,1))

def direction_probability(p):
    den=p[...,0]+p[...,2]
    return np.divide(p[...,2],den,out=np.full(den.shape,.5),where=den>0)

def direct_checks():
    from sklearn.linear_model import LogisticRegression
    a=LogisticRegression(C=.1,max_iter=120,tol=1e-6,random_state=7).get_params()
    b=LogisticRegression(C=.01,max_iter=120,tol=1e-6,random_state=7).get_params()
    assert [k for k in a if a[k]!=b[k]]==["C"]
    squared,ll=binary_losses(np.array([.25,.75]),np.array([0,1]))
    assert np.array_equal(squared,[.0625,.0625]) and np.allclose(ll,-np.log(.75))
    assert np.isfinite(binary_losses(np.array([0.,1.]),np.array([1,0]))[1]).all()
    assert direction_probability(np.array([[0.,1.,0.]])).item()==.5
    return ["only_C_changes_in_installed_estimator","hand_binary_brier_and_log_loss","finite_wrong_certain_probability_loss"]

def init():
    c=config();_,parent=adapter.registered();manifest=base.read(PARENT/"delivery_manifest.json")
    att=ROOT/c["attribution_output"];am=base.read(att/"delivery_manifest.json")
    source_pins={**manifest["artifact_pins"],**parent["data_pins"],**am["artifact_pins"]}
    for p,sha in source_pins.items():assert base.digest(ROOT/p)==sha,p
    OUT.mkdir(parents=True)
    base.write(OUT/"registration.json",dict(contract=c,code_pins=pins(),source_pins=source_pins,versions=base.reuse.versions(),
        pure_checks=direct_checks(),historical_fit_budget=10,synthetic_estimator_fits=0,
        independent_validation=False,**{k:False for k in base.reuse.fixed.SAFETY}))
    previous_path=ROOT/c["previous_registry"];previous=base.read(previous_path);previous.pop("sha256",None)
    previous.update(registry_id=REGISTRY.stem,registration_date_utc="2026-10-18",previous_registry_path=c["previous_registry"],previous_registry_sha256=base.digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"],family="single_C_followup_revealed_10m",registration_path=base.rel(OUT/"registration.json"),
        registration_sha256=base.digest(OUT/"registration.json"),historical_fit_budget=10,synthetic_estimator_fits=0,independent_validation=False,promotion_authority=False))
    base.write(REGISTRY,previous);print(json.dumps(dict(registry="v28",fit_budget=10,pure_checks=3,source_pins=len(source_pins))))

def registered():
    c=config();r=base.read(OUT/"registration.json")
    assert r["code_pins"]==pins() and r["versions"]==base.reuse.versions() and r["contract"]==c
    return c,r

def origin(number):
    c,reg=registered();assert number in range(5)
    pc=base.config();h=10;bank=adapter.load(h);when=pc["test_start_ms"]+number*3*base.DAY;seed=pc["seed"]+h*1000+number
    train,cal=base.reuse.fixed.fold(bank["times"],when,h,pc,seed)
    test=np.flatnonzero((bank["times"]>=when)&(bank["times"]<min(when+3*base.DAY,pc["test_end_exclusive_ms"])))
    pp=PARENT/("origin_%02d"%number)/"tf_10m";parent_meta=base.read(pp.with_suffix(".json"))
    assert base.digest(pp.with_suffix(".npz"))==parent_meta["ledger_sha256"]
    assert base.identity(bank["times"][train].tolist())==parent_meta["fold"]["train_times_sha256"]
    assert base.identity(bank["times"][cal].tolist())==parent_meta["fold"]["calibration_times_sha256"]
    assert parent_meta["fold"]["train_latest_label_ms"]<parent_meta["fold"]["train_cutoff_exclusive_ms"]
    path=OUT/("origin_%02d"%number);path.mkdir(exist_ok=True)
    fitted=warnings_count=0;probs=[];records=[];raw_return=bank["closes"][train+1]/bank["closes"][train]-1
    raw_prior_counts=np.stack([(raw_return<0).sum(axis=0),(raw_return==0).sum(axis=0),(raw_return>0).sum(axis=0)],axis=-1)
    raw_priors=(raw_prior_counts+1)/(len(train)+3)
    with np.load(pp.with_suffix(".npz"),allow_pickle=False) as source:
        assert np.array_equal(source["times"],bank["times"][test])
        for group,(factor,oldname,index) in enumerate([(1.,"original_logit",0),(.5,"half_logit",2)]):
            oldpath=pp.parent/(pp.name+"__"+oldname+".joblib");oldmeta=base.read(oldpath.with_suffix(".json"))
            assert base.digest(oldpath)==oldmeta["model_sha256"]
            old=joblib.load(oldpath);assert old["estimator"].C==.1
            assert np.array_equal(base.reuse.fixed.probabilities(old,bank["features"][test]),source["probability"][index])
            name=MODEL_NAMES[group*2+1];model_path=path/(name+".joblib");meta_path=model_path.with_suffix(".json")
            if meta_path.exists():
                mm=base.read(meta_path);assert mm["code_pins"]==pins() and base.digest(model_path)==mm["model_sha256"]
                assert mm["fold_sha256"]==parent_meta["fold_sha256"]
                model=joblib.load(model_path)
            else:
                if model_path.exists():raise ValueError("partial model present; no silent refit")
                fitting=copy.deepcopy(pc);fitting["logit"]["C"]=c["challenger_C"]
                assert {k:v for k,v in fitting["logit"].items() if k!="C"}=={k:v for k,v in pc["logit"].items() if k!="C"}
                labels=base.reuse.tri.labels(raw_return*100,pc["neutral_bands_pct"]["10"]*factor)
                model=base.reuse.fixed.fit(bank["features"][train],labels,"logit_uniform",fitting,seed)
                assert model["estimator"] is not None and model["estimator"].C==.01
                assert np.array_equal(model["mean"],old["mean"]) and np.array_equal(model["scale"],old["scale"])
                assert np.array_equal(model["train_counts"],old["train_counts"])
                with model_path.open("xb") as stream:joblib.dump(model,stream,compress=3)
                mm=dict(name=name,C=.01,training_band_factor=factor,model_sha256=base.digest(model_path),code_pins=pins(),
                    fold=parent_meta["fold"],fold_sha256=parent_meta["fold_sha256"],parent_model=base.rel(oldpath),parent_model_sha256=base.digest(oldpath),
                    scaler_sha256=base.identity(dict(mean=model["mean"].tolist(),scale=model["scale"].tolist())),
                    coefficient_count=model["coefficient_count"],coefficient_l2_norm=float(np.linalg.norm(model["estimator"].coef_)),
                    baseline_coefficient_l2_norm=float(np.linalg.norm(old["estimator"].coef_)),warnings=model["warnings"],fitted_iterations=model["fitted_iterations"],
                    estimator_fits=1,**{k:False for k in base.reuse.fixed.SAFETY})
                base.write(meta_path,mm);fitted+=1;warnings_count+=len(model["warnings"])
            assert np.array_equal(model["mean"],old["mean"]) and np.array_equal(model["scale"],old["scale"])
            probs.extend([source["probability"][index],base.reuse.fixed.probabilities(model,bank["features"][test])])
            records.append(dict(name=name,file=model_path.name,model_sha256=base.digest(model_path),meta_sha256=base.digest(meta_path)))
        values=np.stack(probs);ledger=path/"ledger.npz"
        if ledger.exists():
            done=base.read(path/"ledger.json");assert base.digest(ledger)==done["ledger_sha256"]
            with np.load(ledger,allow_pickle=False) as a:assert np.array_equal(a["probability"],values)
        else:
            with ledger.open("xb") as stream:np.savez_compressed(stream,times=source["times"],return_pct=source["return_pct"],past_return_pct=source["past_return_pct"],
                probability=values,raw_train_priors=raw_priors,train_priors=source["train_priors"],uniform=source["uniform"],
                model_sha256=np.array([r["model_sha256"] for r in records]))
            base.write(path/"ledger.json",dict(ledger_sha256=base.digest(ledger),models=records,parent_ledger_sha256=parent_meta["ledger_sha256"],
                fold=parent_meta["fold"],code_pins=pins(),**{k:False for k in base.reuse.fixed.SAFETY}))
    print(json.dumps(dict(origin=number,new_fits=fitted,warnings=warnings_count,baseline_refits=0)))

def class_predictions(a):
    pc=base.config();shape=a["uniform"].shape
    pp=[base.reuse.tri.predict(p) for p in a["probability"]]
    for prior in a["train_priors"]:pp.append(np.broadcast_to(base.reuse.tri.predict(prior),shape))
    pp.append(base.reuse.tri.STATES[np.minimum((a["uniform"]*3).astype(int),2)])
    for factor in (1.,.5):pp.append(base.reuse.tri.labels(a["past_return_pct"],pc["neutral_bands_pct"]["10"]*factor))
    return pp

def score():
    c,reg=registered();pc=base.config();daily=np.zeros((8,6,13,4));class_daily=np.zeros((3,8,9,13,3,3),np.int64)
    proper=np.zeros((3,4,8,3));pred_chunks=[];truth_chunks=[]
    for origin in range(5):
        path=OUT/("origin_%02d"%origin);meta=base.read(path/"ledger.json");assert base.digest(path/"ledger.npz")==meta["ledger_sha256"]
        with np.load(path/"ledger.npz",allow_pickle=False) as a:
            days=((a["times"]-pc["test_start_ms"])//base.DAY).astype(int);raw_y=a["return_pct"]>0;mask=a["return_pct"]!=0
            probabilities=[direction_probability(p) for p in a["probability"]]
            prior=a["raw_train_priors"][:,2]/(a["raw_train_priors"][:,0]+a["raw_train_priors"][:,2])
            probabilities.extend([np.broadcast_to(prior,raw_y.shape),np.full(raw_y.shape,.5)])
            pred_chunks.append(np.stack(probabilities));truth_chunks.append(a["return_pct"])
            for asset in range(8):
                valid=mask[:,asset];y=raw_y[valid,asset].astype(int);dd=days[valid]
                for k,p in enumerate(probabilities):
                    squared,ll=binary_losses(p[valid,asset],y)
                    correct=((a["uniform"][valid,asset]>.5)==y) if k==5 else ((p[valid,asset]>.5)==y)
                    for day in range(13):
                        chosen=dd==day;daily[asset,k,day]=[chosen.sum(),squared[chosen].sum(),ll[chosen].sum(),correct[chosen].sum()]
            pp=class_predictions(a)
            for target,factor in enumerate(c["evaluation_band_factors"]):
                y=base.reuse.tri.labels(a["return_pct"],pc["neutral_bands_pct"]["10"]*factor)
                for asset in range(8):
                    for k,pred in enumerate(pp):class_daily[target,asset,k]+=np.bincount(days*9+(y[:,asset]+1)*3+pred[:,asset]+1,minlength=13*9).reshape(13,3,3)
                for k,p in enumerate(a["probability"]):
                    observed=np.take_along_axis(p,(y+1)[...,None],axis=-1)[...,0]
                    proper[target,k,:,0]+=-np.log(np.clip(observed,1e-15,1)).sum(axis=0)
                    proper[target,k,:,1]+=((p-np.eye(3)[y+1])**2).sum(axis=-1).sum(axis=0)
                    proper[target,k,:,2]+=len(y)
    probabilities=np.concatenate(pred_chunks,axis=1);returns=np.concatenate(truth_chunks,axis=0)
    weights=[]
    for seed in pc["bootstrap"]["seeds"]:
        with np.load(PARENT/("bootstrap_bank_%s.npz"%seed),allow_pickle=False) as a:weights.append(np.stack([np.bincount(v,minlength=13) for v in a["indices"]]))
    rows=[];class_rows=[]
    for asset,coin in enumerate(pc["symbols"]):
        valid=returns[:,asset]!=0;y=(returns[valid,asset]>0).astype(int)
        prior=probabilities[4,valid,asset]
        for k,name in enumerate(DIRECTION_NAMES):
            stats=attribution.binary_stats(probabilities[k,valid,asset],y,prior)
            totals=daily[asset,k].sum(axis=0)
            stats.update(accuracy=float(totals[3]/totals[0]),brier=float(totals[1]/totals[0]),log_loss=float(totals[2]/totals[0]))
            stats["ci95_banks"]={metric:[np.quantile((w@daily[asset,k,:,column])/(w@daily[asset,k,:,0]),[.025,.975]).tolist() for w in weights] for metric,column in [("brier",1),("log_loss",2),("accuracy",3)]}
            rows.append(dict(symbol=coin,method=name,**stats))
        for target,factor in enumerate(c["evaluation_band_factors"]):
            for k,name in enumerate(CLASS_NAMES):
                row=dict(symbol=coin,method=name,evaluation_band_factor=factor,**base.reuse.tri.from_confusion(class_daily[target,asset,k].sum(axis=0)))
                if k<4:row.update(log_loss=float(proper[target,k,asset,0]/proper[target,k,asset,2]),multiclass_brier=float(proper[target,k,asset,1]/proper[target,k,asset,2]))
                class_rows.append(row)
    comparisons=[];draws=[[],[]]
    for asset,coin in enumerate(pc["symbols"]):
        for candidate,matched in [("original_C0p01","original_C0p1"),("half_C0p01","half_C0p1")]:
            k=DIRECTION_NAMES.index(candidate)
            for control in (matched,"train_prior"):
                j=DIRECTION_NAMES.index(control)
                for metric,column in [("brier",1),("log_loss",2)]:
                    difference=daily[asset,j,:,column]-daily[asset,k,:,column];n=daily[asset,k,:,0]
                    comparisons.append(dict(symbol=coin,candidate=candidate,control=control,metric=metric,improvement=float(difference.sum()/n.sum())))
                    for b,w in enumerate(weights):draws[b].append((w@difference)/(w@n))
    obs=np.array([r["improvement"] for r in comparisons]);criticals=[]
    for b in range(2):
        lower,critical=base.reuse.causal.simultaneous_lcb(obs,np.array(draws[b]).T);criticals.append(float(critical))
        for r,v in zip(comparisons,lower):r["simultaneous_lcb_bank%s"%b]=float(v)
    gates=[]
    for coin in pc["symbols"]:
        for candidate in ("original_C0p01","half_C0p01"):
            rr=[r for r in comparisons if r["symbol"]==coin and r["candidate"]==candidate]
            gates.append(dict(symbol=coin,candidate=candidate,all_losses_controls_banks_positive=all(r["simultaneous_lcb_bank0"]>0 and r["simultaneous_lcb_bank1"]>0 for r in rr)))
    assert len(comparisons)==64 and len(rows)==48 and len(class_rows)==216
    with (OUT/"daily_metrics.npz").open("xb") as stream:np.savez_compressed(stream,direction=daily,classification=class_daily,bank0=np.array(draws[0]).T,bank1=np.array(draws[1]).T)
    base.write(OUT/"results.json",dict(direction=rows,classification=class_rows,comparisons=comparisons,gates=gates,max_t_criticals=criticals,code_pins=pins()))
    base.write(OUT/"summary.json",dict(direction_cells=48,classification_cells=216,comparisons=64,new_model_artifacts=10,
        research_gates_passed=sum(r["all_losses_controls_banks_positive"] for r in gates),gate_tasks=16,independent_validation=False,**{k:False for k in base.reuse.fixed.SAFETY}))
    print(json.dumps(dict(direction_cells=48,classification_cells=216,comparisons=64,research_gates_passed=sum(r["all_losses_controls_banks_positive"] for r in gates))))

def audit():
    c,reg=registered();pc=base.config();bank=adapter.load(10);models=0;warnings=0
    for origin in range(5):
        path=OUT/("origin_%02d"%origin);meta=base.read(path/"ledger.json");assert base.digest(path/"ledger.npz")==meta["ledger_sha256"]
        when=pc["test_start_ms"]+origin*3*base.DAY;seed=pc["seed"]+10000+origin
        train,cal=base.reuse.fixed.fold(bank["times"],when,10,pc,seed)
        test=np.flatnonzero((bank["times"]>=when)&(bank["times"]<min(when+3*base.DAY,pc["test_end_exclusive_ms"])))
        with np.load(path/"ledger.npz",allow_pickle=False) as a:
            assert np.array_equal(a["times"],bank["times"][test])
            assert base.identity(bank["times"][train].tolist())==meta["fold"]["train_times_sha256"]
            for group,record in enumerate(meta["models"]):
                p=path/record["file"];assert base.digest(p)==record["model_sha256"]==a["model_sha256"][group]
                mm=base.read(p.with_suffix(".json"));assert base.digest(p.with_suffix(".json"))==record["meta_sha256"]
                model=joblib.load(p);old=joblib.load(ROOT/mm["parent_model"])
                assert base.digest(ROOT/mm["parent_model"])==mm["parent_model_sha256"]
                assert model["estimator"].C==.01 and old["estimator"].C==.1
                assert np.array_equal(model["mean"],old["mean"]) and np.array_equal(model["scale"],old["scale"])
                assert base.identity(dict(mean=model["mean"].tolist(),scale=model["scale"].tolist()))==mm["scaler_sha256"]
                assert np.array_equal(base.reuse.fixed.probabilities(model,bank["features"][test]),a["probability"][2*group+1])
                assert np.array_equal(base.reuse.fixed.probabilities(old,bank["features"][test]),a["probability"][2*group]);models+=1;warnings+=len(mm["warnings"])
    for p,sha in reg["source_pins"].items():assert base.digest(ROOT/p)==sha,p
    previous=base.read(ROOT/c["previous_registry"]);current=base.read(REGISTRY);assert current["objectives"][:-1]==previous["objectives"]
    base.write(OUT/"model_audit.json",dict(models=models,warnings=warnings,exact_probability_replay=True,identical_scalers_and_folds=True,
        parent_pins_unchanged=True,previous_registry_unchanged=True,new_fits=0,code_pins=pins()))
    print(json.dumps(dict(audited_models=models,warnings=warnings,exact_replay=True,new_fits=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["checks","init","origin","score","audit"]);parser.add_argument("--number",type=int);args=parser.parse_args()
    if args.stage=="checks":print(json.dumps(dict(pure_checks=direct_checks(),fits=0)))
    elif args.stage=="origin":origin(args.number)
    else:globals()[args.stage]()
