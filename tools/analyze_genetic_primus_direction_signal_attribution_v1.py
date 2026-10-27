"""Read-only frozen-forecast diagnostics. No fitting, networking, or runtime."""
from __future__ import annotations
import run_genetic_primus_direction_october_halfband_v2 as adapter
import argparse
import json
from pathlib import Path
import numpy as np

run=adapter.old
ROOT=run.ROOT
PARENT=adapter.OUT
CONFIG=ROOT/"configs/genetic_primus_direction_signal_attribution_v1.json"
OUT=ROOT/"Reports/Exia/Genetic_Primus/direction_signal_attribution_v1_20261014"
REGISTRY=ROOT/"configs/genetic_primus_objective_registry_v27.json"
METHODS=["original_logit","original_boost","half_logit","half_boost","random_uniform_3","momentum_matching_truth"]

def pins():return {run.rel(p):run.digest(p) for p in (CONFIG,Path(__file__))}
def config():
    c=run.read(CONFIG);run.config()
    assert all(c[k] is False for k in run.reuse.fixed.SAFETY)
    assert c["truth_band_factors"]==[1.,.5,0.] and c["historical_fit_budget"]==0
    return c

def safe_div(a,b):return None if b==0 else float(a/b)

def contingency(truth,pred,control):
    y,p,q=[np.asarray(v).reshape(-1) for v in (truth,pred,control)]
    assert y.shape==p.shape==q.shape and np.isin(y,[-1,0,1]).all()
    good=p==y;baseline=q==y;n=len(y)
    contributions=[int(((good.astype(int)-baseline.astype(int))*(y==state)).sum()) for state in (-1,0,1)]
    wins=int((good&~baseline).sum());losses=int((~good&baseline).sum())
    assert sum(contributions)==wins-losses
    return dict(n=n,model_correct=int(good.sum()),control_correct=int(baseline.sum()),wins=wins,losses=losses,
        both_correct=int((good&baseline).sum()),both_wrong=int((~good&~baseline).sum()),
        correct_delta_down_neutral_up=contributions,model_accuracy=safe_div(int(good.sum()),n),
        control_accuracy=safe_div(int(baseline.sum()),n),accuracy_increment=safe_div(wins-losses,n))

def binary_stats(prob,truth,prior,bins=10):
    p,y,q=[np.asarray(v).reshape(-1) for v in (prob,truth,prior)]
    assert p.shape==y.shape==q.shape and np.isfinite(p).all() and np.isfinite(q).all()
    assert ((p>=0)&(p<=1)).all() and np.isin(y,[0,1]).all()
    if not len(y):return dict(n=0,accuracy=None,prior_accuracy=None,brier=None,prior_brier=None,brier_improvement=None,log_loss=None,prior_log_loss=None,auc=None,bins=[],ece=None,confidence_strata=[])
    def loss(v):return float(-(y*np.log(np.clip(v,1e-15,1))+(1-y)*np.log(np.clip(1-v,1e-15,1))).mean())
    bucket=np.minimum((p*bins).astype(int),bins-1);records=[];ece=0.
    for b in range(bins):
        selected=bucket==b;count=int(selected.sum())
        mean=None if not count else float(p[selected].mean());actual=None if not count else float(y[selected].mean())
        if count:ece+=count/len(y)*abs(mean-actual)
        records.append(dict(bin=b,lower=b/bins,upper=(b+1)/bins,n=count,predicted_mean=mean,actual_positive_fraction=actual))
    from sklearn.metrics import roc_auc_score
    auc=None if len(np.unique(y))<2 else float(roc_auc_score(y,p))
    strata=[];confidence=np.maximum(p,1-p);correct=(p>.5)==y
    for level in (.8,.9):
        selected=confidence>=level;count=int(selected.sum())
        strata.append(dict(minimum_confidence=level,n=count,coverage_of_actual_moves=count/len(y),accuracy=None if not count else float(correct[selected].mean())))
    brier=float(((p-y)**2).mean());prior_brier=float(((q-y)**2).mean())
    return dict(n=len(y),accuracy=float(correct.mean()),prior_accuracy=float(((q>.5)==y).mean()),
        brier=brier,prior_brier=prior_brier,brier_improvement=prior_brier-brier,
        log_loss=loss(p),prior_log_loss=loss(q),auc=auc,bins=records,ece=ece,confidence_strata=strata)

def direct_checks():
    y=np.array([-1,0,1,1]);p=np.array([-1,0,-1,1]);q=np.array([1,-1,1,1])
    a=contingency(y,p,q)
    assert a["wins"]==2 and a["losses"]==1 and a["correct_delta_down_neutral_up"]==[1,1,-1] and a["accuracy_increment"]==.25
    b=binary_stats(np.array([.25,.75]),np.array([0,1]),np.array([.5,.5]))
    assert b["accuracy"]==1 and b["brier"]==.0625 and b["prior_brier"]==.25 and b["auc"]==1
    z=binary_stats(np.array([]),np.array([],int),np.array([]));assert z["n"]==0 and z["accuracy"] is None
    assert sum(r["n"] for r in b["bins"])==2
    return ["hand_exact_attribution_and_wins_losses","hand_direction_brier_auc_and_tie_rule","empty_condition_is_undefined","calibration_counts_partition"]

def init():
    c=config();manifest=run.read(PARENT/"delivery_manifest.json")
    parent_data=run.read(PARENT/"registration.json")["data_pins"]
    for p,sha in {**manifest["artifact_pins"],**parent_data}.items():assert run.digest(ROOT/p)==sha,p
    OUT.mkdir(parents=True)
    run.write(OUT/"registration.json",dict(contract=c,code_pins=pins(),parent_manifest_sha256=run.digest(PARENT/"delivery_manifest.json"),
        parent_artifact_pins=manifest["artifact_pins"],parent_data_pins=parent_data,pure_checks=direct_checks(),estimator_fits=0,**{k:False for k in run.reuse.fixed.SAFETY}))
    previous_path=ROOT/c["previous_registry"];previous=run.read(previous_path);previous.pop("sha256",None)
    previous.update(registry_id=REGISTRY.stem,previous_registry_path=c["previous_registry"],previous_registry_sha256=run.digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"],family="posthoc_signal_attribution_and_duration_scenarios",
        registration_path=run.rel(OUT/"registration.json"),registration_sha256=run.digest(OUT/"registration.json"),historical_fit_budget=0,
        no_parameter_selection=True,posthoc_diagnostic=True,independent_validation=False,promotion_authority=False))
    run.write(REGISTRY,previous);print(json.dumps(dict(registry="v27",pure_checks=4,parent_pins_verified=len(manifest["artifact_pins"]),fits=0)))

def registered():
    c=config();reg=run.read(OUT/"registration.json")
    assert reg["code_pins"]==pins() and reg["parent_manifest_sha256"]==run.digest(PARENT/"delivery_manifest.json")
    return c,reg

def frame(h):
    pc=run.config();chunks={k:[] for k in ("times","returns","past","prob","random","priors","raw_priors")}
    bank=adapter.load(h)
    for origin in range(5):
        prefix=PARENT/("origin_%02d"%origin)/("tf_%sm"%h);meta=run.read(prefix.with_suffix(".json"))
        assert run.digest(prefix.with_suffix(".npz"))==meta["ledger_sha256"]
        when=pc["test_start_ms"]+origin*3*run.DAY
        train,cal=run.reuse.fixed.fold(bank["times"],when,h,pc,pc["seed"]+h*1000+origin)
        mature_return=bank["closes"][train+1]/bank["closes"][train]-1
        raw_counts=np.stack([(mature_return<0).sum(axis=0),(mature_return==0).sum(axis=0),(mature_return>0).sum(axis=0)],axis=-1)
        raw_prior=(raw_counts+1)/(len(train)+3)
        with np.load(prefix.with_suffix(".npz"),allow_pickle=False) as a:
            n=len(a["times"]);pp=run.predictions(a,h,pc)
            chunks["times"].append(a["times"]);chunks["returns"].append(a["return_pct"]);chunks["prob"].append(a["probability"])
            chunks["past"].append(a["past_return_pct"])
            chunks["random"].append(pp[5]);chunks["priors"].append(np.broadcast_to(a["train_priors"][:,None],(2,n,8,3)))
            chunks["raw_priors"].append(np.broadcast_to(raw_prior,(n,8,3)))
    return {k:np.concatenate(v,axis=1 if k in ("prob","priors") else 0) for k,v in chunks.items()}

def interval(daily,weights):
    n=daily[:,0];difference=daily[:,1]-daily[:,2]
    estimates=[(w@difference)/(w@n) for w in weights]
    return [np.quantile(a,[.025,.975]).tolist() for a in estimates]

def calculate(h):
    c,reg=registered();pc=run.config();f=frame(h);days=((f["times"]-pc["test_start_ms"])//run.DAY).astype(int)
    weights=[]
    for seed in pc["bootstrap"]["seeds"]:
        with np.load(PARENT/("bootstrap_bank_%s.npz"%seed),allow_pickle=False) as a:weights.append(np.stack([np.bincount(v,minlength=13) for v in a["indices"]]))
    rows=[];pooled=[];conditional=[];all_daily=np.zeros((3,8,6,13,10),np.int64)
    for target,factor in enumerate(c["truth_band_factors"]):
        y=run.reuse.tri.labels(f["returns"],pc["neutral_bands_pct"][str(h)]*factor)
        prior=f["raw_priors"] if factor==0 else f["priors"][target]
        control=run.reuse.tri.predict(prior)
        # Previous return comes directly from the frozen ledger, not from future truth.
        momentum=run.reuse.tri.labels(f["past"],pc["neutral_bands_pct"][str(h)]*factor)
        pp=[run.reuse.tri.predict(p) for p in f["prob"]]+[f["random"],momentum]
        for asset,coin in enumerate(pc["symbols"]):
            for k,method in enumerate(METHODS):
                pred=pp[k][:,asset];truth=y[:,asset];q=control[:,asset];stats=contingency(truth,pred,q)
                d=all_daily[target,asset,k]
                for day in range(13):
                    a=contingency(truth[days==day],pred[days==day],q[days==day])
                    d[day]=[a[key] for key in ("n","model_correct","control_correct","wins","losses","both_correct","both_wrong")]+a["correct_delta_down_neutral_up"]
                rows.append(dict(minutes=h,symbol=coin,method=method,truth_band_factor=factor,**stats,
                    accuracy_increment_ci95_banks=interval(d,weights)))
                if k<4:
                    p=f["prob"][k,:,asset];moves=truth!=0
                    direction=np.divide(p[:,2],p[:,0]+p[:,2],out=np.full(len(p),.5),where=(p[:,0]+p[:,2])>0)
                    baseline=prior[:,asset,2]/(prior[:,asset,0]+prior[:,asset,2])
                    b=binary_stats(direction[moves],(truth[moves]>0).astype(int),baseline[moves],c["calibration_bins"])
                    move_brier=float(((1-p[:,1]-moves)**2).mean());prior_move_brier=float(((1-prior[:,asset,1]-moves)**2).mean())
                    conditional.append(dict(minutes=h,symbol=coin,method=method,truth_band_factor=factor,
                        n_all_intervals=len(truth),actual_move_fraction=float(moves.mean()),move_brier=move_brier,prior_move_brier=prior_move_brier,
                        move_brier_improvement=prior_move_brier-move_brier,**b))
        for k,method in enumerate(METHODS):
            d=all_daily[target,:,k].sum(axis=0);n=int(d[:,0].sum());delta=int((d[:,1]-d[:,2]).sum())
            blocks=[]
            for block,start in enumerate(range(0,13,3)):
                b=d[start:min(start+3,13)].sum(axis=0)
                blocks.append(dict(block=block,start_day=start+1,end_day=min(start+3,13),n=int(b[0]),correct_difference=int(b[1]-b[2]),accuracy_increment=float((b[1]-b[2])/b[0]),correct_delta_down_neutral_up=b[7:].tolist()))
            loo_asset=[]
            for asset in range(8):
                a=d-all_daily[target,asset,k];loo_asset.append(float((a[:,1]-a[:,2]).sum()/a[:,0].sum()))
            loo_block=[]
            for b in blocks:loo_block.append(float((delta-b["correct_difference"])/(n-b["n"])))
            pooled.append(dict(minutes=h,method=method,truth_band_factor=factor,n=n,model_accuracy=float(d[:,1].sum()/n),control_accuracy=float(d[:,2].sum()/n),
                accuracy_increment=delta/n,accuracy_increment_ci95_banks=interval(d,weights),correct_delta_down_neutral_up=d[:,7:].sum(axis=0).tolist(),
                wins=int(d[:,3].sum()),losses=int(d[:,4].sum()),blocks=blocks,positive_blocks=sum(b["accuracy_increment"]>0 for b in blocks),
                leave_one_asset_out_increment=loo_asset,leave_one_block_out_increment=loo_block))
    with (OUT/("daily_%sm.npz"%h)).open("xb") as stream:np.savez_compressed(stream,counts=all_daily)
    run.write(OUT/("results_%sm.json"%h),dict(rows=rows,pooled=pooled,conditional=conditional,code_pins=pins(),estimator_fits=0))
    print(json.dumps(dict(minutes=h,paired_asset_cells=len(rows),conditional_direction_cells=len(conditional),fits=0)))

def duration():
    from scipy.stats import norm,chi2
    c,reg=registered();pc=run.config();p=c["duration_planning"];rows=[]
    alpha=p["one_sided_family_alpha"]/p["family_comparisons"];z=norm.ppf(1-alpha)+norm.ppf(p["power"])
    for h in pc["timeframes_minutes"]:
        with np.load(OUT/("daily_%sm.npz"%h),allow_pickle=False) as a:d=a["counts"][1]
        for asset,coin in enumerate(pc["symbols"]):
            for method in ("half_logit","half_boost"):
                daily=d[asset,METHODS.index(method)];blocks=daily[:12].reshape(4,3,10).sum(axis=1)
                effects=(blocks[:,1]-blocks[:,2])/blocks[:,0];sigma=float(effects.std(ddof=1))
                upper=sigma*np.sqrt(3/chi2.ppf(1-p["sigma_upper_confidence"],3))
                for delta in p["accuracy_increment_scenarios"]:
                    point=None if sigma==0 else int(np.ceil((z*sigma/delta)**2))*3
                    cautious=None if sigma==0 else int(np.ceil((z*upper/delta)**2))*3
                    rows.append(dict(minutes=h,symbol=coin,method=method,desired_accuracy_increment=delta,observed_full_blocks=4,
                        sigma_three_day_accuracy_difference=sigma,sigma_upper90=upper,normal_approximation_days_point=point,
                        normal_approximation_days_sigma_upper90=cautious,engineering_floor_days=90,
                        proposed_scenario_days=None if cautious is None else max(90,cautious),certified_power=False))
    run.write(OUT/"duration_scenarios.json",dict(rows=rows,assumptions=p,not_a_duration_guarantee=True,estimator_fits=0))
    draft=dict(c["future_draft"]);draft.update(source_contract=run.rel(adapter.PRIOR_CONFIG),metadata_amendment=run.rel(adapter.CONFIG),
        symbol_list=pc["symbols"],original_neutral_bands_pct=pc["neutral_bands_pct"],training_band_factors=[1.,.5],
        scaler="train-only mean/std each origin",purge_bars=7,retrain_days=3,
        analysis_rule="One final analysis after duration and contract are approved and sealed; no intermediate tuning; no autoexecution",
        duration_status="90 days is an engineering minimum proposal, NOT certified power; exact final interval remains unsealed",
        requires_separate_historical_data_authorization=True,**{k:False for k in run.reuse.fixed.SAFETY})
    run.write(OUT/"future_protocol_DRAFT.json",draft)
    print(json.dumps(dict(duration_scenarios=len(rows),full_observed_blocks=4,future_protocol="DRAFT_NOT_SEALED_NOT_SCHEDULED",fits=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["checks","init","calculate","duration"]);parser.add_argument("--minutes",type=int);args=parser.parse_args()
    if args.stage=="checks":print(json.dumps(dict(pure_checks=direct_checks(),fits=0)))
    elif args.stage=="calculate":calculate(args.minutes)
    else:globals()[args.stage]()
