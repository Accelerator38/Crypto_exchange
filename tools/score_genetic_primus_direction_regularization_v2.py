"""Correct full-period aggregation of already frozen forecasts. Never fits models."""
from __future__ import annotations
import run_genetic_primus_direction_regularization_v1 as r
import argparse
import json
from pathlib import Path
import numpy as np

ROOT,OUT=r.ROOT,r.OUT
CONFIG=ROOT/"configs/genetic_primus_direction_regularization_scoring_v2.json"
REGISTRY=ROOT/"configs/genetic_primus_objective_registry_v29.json"

def pins():return {r.base.rel(p):r.base.digest(p) for p in (CONFIG,Path(__file__))}

def daily_sums(days,truth,prob,pred=None):
    squared,ll=r.binary_losses(prob,truth)
    pred=(prob>.5) if pred is None else pred
    result=np.stack([np.bincount(days,weights=w,minlength=13) for w in (np.ones(len(days)),squared,ll,(pred==truth).astype(int))],axis=-1)
    assert result[:,0].sum()==len(truth)
    return result

def checks():
    days=np.array([0,0,12,12]);y=np.array([0,1,1,0]);p=np.array([.25,.75,.75,.25]);d=daily_sums(days,y,p)
    assert d[0,0]==d[12,0]==2 and d[:,0].sum()==4 and d[:,3].sum()==4
    assert d[:,1].sum()==.25 and np.allclose(d[:,2].sum(),-4*np.log(.75))
    return ["two_origins_preserve_first_and_last_day","full_denominator_matches_every_observation","hand_proper_loss_sums"]

def init():
    c,reg=r.registered();am=r.base.read(CONFIG)
    assert all(am[k] is False for k in r.base.reuse.fixed.SAFETY)
    assert r.base.read(OUT/"model_audit.json")["models"]==10
    assert not (OUT/"results.json").exists() and not (OUT/"daily_metrics.npz").exists()
    r.base.write(OUT/"score_failure_disclosure.json",dict(old_scoring_calls_failed=1,reason=am["correction"],failed_score_results_persisted=0,models_unchanged=True,new_fits=0))
    sources=dict(reg["source_pins"])
    for p in OUT.rglob("*"):
        if p.is_file():sources[r.base.rel(p)]=r.base.digest(p)
    sources.update(r.pins())
    r.base.write(OUT/"scoring_registration_v2.json",dict(contract=am,code_pins=pins(),source_pins=sources,pure_checks=checks(),new_fits=0))
    previous_path=ROOT/am["previous_registry"];previous=r.base.read(previous_path);previous.pop("sha256",None)
    previous.update(registry_id=REGISTRY.stem,previous_registry_path=am["previous_registry"],previous_registry_sha256=r.base.digest(previous_path))
    previous["objectives"].append(dict(objective_id=am["objective_id"],family="scoring_only_engineering_amendment",
        parent_objective_id=c["objective_id"],registration_path=r.base.rel(OUT/"scoring_registration_v2.json"),
        registration_sha256=r.base.digest(OUT/"scoring_registration_v2.json"),historical_fit_budget=0,statistical_family_changed=False,promotion_authority=False))
    r.base.write(REGISTRY,previous);print(json.dumps(dict(registry="v29",scoring_only=True,new_fits=0,pure_checks=len(checks()))))

def registered():
    c,reg=r.registered();s=r.base.read(OUT/"scoring_registration_v2.json")
    assert s["code_pins"]==pins()
    return c,reg,s

def load():
    chunks={k:[] for k in ("times","return_pct","past_return_pct","probability","uniform","train_priors","raw_train_priors")}
    for origin in range(5):
        path=OUT/("origin_%02d"%origin);meta=r.base.read(path/"ledger.json");assert r.base.digest(path/"ledger.npz")==meta["ledger_sha256"]
        with np.load(path/"ledger.npz",allow_pickle=False) as a:
            n=len(a["times"])
            for key in chunks:
                value=a[key]
                if key=="train_priors":value=np.broadcast_to(value[:,None],(2,n,8,3))
                elif key=="raw_train_priors":value=np.broadcast_to(value,(n,8,3))
                chunks[key].append(value)
    f={k:np.concatenate(v,axis=1 if k in ("probability","train_priors") else 0) for k,v in chunks.items()}
    pc=r.base.config();assert len(f["times"])==1872 and np.array_equal(f["times"],np.arange(pc["test_start_ms"],pc["test_end_exclusive_ms"],600000))
    return f

def score():
    c,reg,s=registered();pc=r.base.config();f=load();days=((f["times"]-pc["test_start_ms"])//r.base.DAY).astype(int)
    raw=f["return_pct"];mask=raw!=0;y=raw>0
    prior=f["raw_train_priors"][:,:,2]/(f["raw_train_priors"][:,:,0]+f["raw_train_priors"][:,:,2])
    probability=np.stack([*[r.direction_probability(p) for p in f["probability"]],prior,np.full(raw.shape,.5)])
    daily=np.zeros((8,6,13,4));class_daily=np.zeros((3,8,9,13,3,3),np.int64)
    direction=[];classification=[];weights=[]
    for seed in pc["bootstrap"]["seeds"]:
        with np.load(r.PARENT/("bootstrap_bank_%s.npz"%seed),allow_pickle=False) as a:weights.append(np.stack([np.bincount(v,minlength=13) for v in a["indices"]]))
    for asset,coin in enumerate(pc["symbols"]):
        valid=mask[:,asset];truth=y[valid,asset].astype(int);dd=days[valid];prior_asset=prior[valid,asset]
        for k,name in enumerate(r.DIRECTION_NAMES):
            prob=probability[k,valid,asset];coin_pred=(f["uniform"][valid,asset]>.5) if k==5 else None
            d=daily_sums(dd,truth,prob,coin_pred);daily[asset,k]=d
            assert d[:,0].sum()==int(valid.sum()) and (d[:,0]>0).all()
            stats=r.attribution.binary_stats(prob,truth,prior_asset);total=d.sum(axis=0)
            stats.update(accuracy=float(total[3]/total[0]),brier=float(total[1]/total[0]),log_loss=float(total[2]/total[0]))
            stats["ci95_banks"]={}
            for metric,column in (("brier",1),("log_loss",2),("accuracy",3)):
                values=[]
                for w in weights:
                    denom=w@d[:,0];assert (denom>0).all()
                    value=(w@d[:,column])/denom;assert np.isfinite(value).all();values.append(np.quantile(value,[.025,.975]).tolist())
                stats["ci95_banks"][metric]=values
            direction.append(dict(symbol=coin,method=name,**stats))
    class_pp=[r.base.reuse.tri.predict(p) for p in f["probability"]]
    for prior_prob in f["train_priors"]:class_pp.append(r.base.reuse.tri.predict(prior_prob))
    class_pp.append(r.base.reuse.tri.STATES[np.minimum((f["uniform"]*3).astype(int),2)])
    for factor in (1.,.5):class_pp.append(r.base.reuse.tri.labels(f["past_return_pct"],pc["neutral_bands_pct"]["10"]*factor))
    for target,factor in enumerate(c["evaluation_band_factors"]):
        truth=r.base.reuse.tri.labels(raw,pc["neutral_bands_pct"]["10"]*factor)
        for asset,coin in enumerate(pc["symbols"]):
            for k,name in enumerate(r.CLASS_NAMES):
                d=np.bincount(days*9+(truth[:,asset]+1)*3+class_pp[k][:,asset]+1,minlength=13*9).reshape(13,3,3)
                assert d.sum()==1872;class_daily[target,asset,k]=d
                row=dict(symbol=coin,method=name,evaluation_band_factor=factor,**r.base.reuse.tri.from_confusion(d.sum(axis=0)))
                if k<4:
                    p=f["probability"][k,:,asset];yy=truth[:,asset]
                    observed=np.take_along_axis(p,(yy+1)[:,None],axis=-1)[:,0]
                    row.update(log_loss=float(-np.log(np.clip(observed,1e-15,1)).mean()),multiclass_brier=float(((p-np.eye(3)[yy+1])**2).sum(axis=-1).mean()))
                classification.append(row)
    comparisons=[];draws=[[],[]]
    for asset,coin in enumerate(pc["symbols"]):
        for candidate,matched in [("original_C0p01","original_C0p1"),("half_C0p01","half_C0p1")]:
            k=r.DIRECTION_NAMES.index(candidate)
            for control in (matched,"train_prior"):
                j=r.DIRECTION_NAMES.index(control)
                for metric,column in (("brier",1),("log_loss",2)):
                    diff=daily[asset,j,:,column]-daily[asset,k,:,column];n=daily[asset,k,:,0]
                    comparisons.append(dict(symbol=coin,candidate=candidate,control=control,metric=metric,improvement=float(diff.sum()/n.sum())))
                    for b,w in enumerate(weights):draws[b].append((w@diff)/(w@n))
    obs=np.array([row["improvement"] for row in comparisons]);criticals=[]
    for b in range(2):
        a=np.array(draws[b]).T;assert np.isfinite(a).all()
        lower,critical=r.base.reuse.causal.simultaneous_lcb(obs,a);criticals.append(float(critical))
        for row,v in zip(comparisons,lower):row["simultaneous_lcb_bank%s"%b]=float(v)
    gates=[]
    for coin in pc["symbols"]:
        for candidate in ("original_C0p01","half_C0p01"):
            rr=[v for v in comparisons if v["symbol"]==coin and v["candidate"]==candidate]
            gates.append(dict(symbol=coin,candidate=candidate,all_losses_controls_banks_positive=all(v["simultaneous_lcb_bank0"]>0 and v["simultaneous_lcb_bank1"]>0 for v in rr)))
    assert len(direction)==48 and len(classification)==216 and len(comparisons)==64
    with (OUT/"daily_metrics.npz").open("xb") as stream:np.savez_compressed(stream,direction=daily,classification=class_daily,bank0=np.array(draws[0]).T,bank1=np.array(draws[1]).T)
    r.base.write(OUT/"results.json",dict(direction=direction,classification=classification,comparisons=comparisons,gates=gates,max_t_criticals=criticals,
        scoring_code_pins=pins(),training_code_pins=r.pins()))
    passed=sum(v["all_losses_controls_banks_positive"] for v in gates)
    r.base.write(OUT/"summary.json",dict(direction_cells=48,classification_cells=216,comparisons=64,new_model_artifacts=10,
        research_gates_passed=passed,gate_tasks=16,full_days=13,failed_scoring_attempts=1,scoring_extra_fits=0,independent_validation=False,
        **{k:False for k in r.base.reuse.fixed.SAFETY}))
    print(json.dumps(dict(direction_cells=48,classification_cells=216,comparisons=64,full_days=13,gates_passed=passed,new_fits=0)))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["checks","init","score"]);a=p.parse_args()
    if a.stage=="checks":print(json.dumps(dict(pure_checks=checks(),fits=0)))
    else:globals()[a.stage]()
