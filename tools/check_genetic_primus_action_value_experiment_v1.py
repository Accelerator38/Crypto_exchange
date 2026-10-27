"""Read-only arithmetic/causality audits, plus generated research diagnostics."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
os.environ["MPLBACKEND"]="Agg"
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_genetic_primus_action_value_experiment_v1 as d
from tools import analyze_genetic_primus_action_value_experiment_v1 as repaired


def main():
    reg=d.registration(); amendment=repaired.verify(); ds,bank,folds=d.checked_dataset()
    data=d.b.load_data(); scaler_checks=0; causal_checks=0; latest_gaps=[]
    for k,fold in enumerate(folds):
        fit=d.sealed(f"fit_{k:02d}.json"); model=fit["model"]
        idx=np.asarray(fold["indices"],int)
        expected=np.flatnonzero(bank["available_us"]<=fold["label_cutoff_us"])
        if not np.array_equal(idx,expected) or model["parameter_count"]!=7:
            raise ValueError("fold index/architecture mismatch")
        x,y=bank["x"][idx],bank["y"][idx]
        digest=d.m.array_identity(bank["asof"][idx],bank["asset"][idx],x,y)
        if digest!=fold["training_samples_sha256"] or digest!=model["training_identity"]["training_samples_sha256"]:
            raise ValueError("training sample provenance mismatch")
        if model["training_identity"]["dataset_registration_sha256"]!=ds["sha256"]:
            raise ValueError("model uses different dataset registration")
        if d.m.array_identity(x,y)!=model["scaler"]["training_sha256"]:
            raise ValueError("scaler training bank mismatch")
        scale=np.where(x.std(axis=0)<1e-12,1.,x.std(axis=0))
        for actual,target in ((model["scaler"]["mean"],x.mean(axis=0)),
                              (model["scaler"]["scale"],scale),
                              (model["scaler"]["target_mean"],y.mean()),
                              (model["scaler"]["target_scale"],y.std())):
            if not np.allclose(actual,target,atol=1e-12,rtol=1e-12):
                raise ValueError("train-only scaler formula")
            scaler_checks+=1
        if int(bank["available_us"][idx].max())>fold["label_cutoff_us"] or fold["origin_ms"]!=fit["origin_ms"]:
            raise ValueError("label leakage")
        latest_gaps.append((fold["origin_ms"]-int(bank["asof"][idx].max()))/d.BAR)
        causal_checks+=3
        d.m.predict(model,x[:1]) # identity/scaler validation, NOT fitting
    for policy in reg["contract"]["policies"]:
        for row in d.sealed(f"vectors_{policy}.json")["rows"]:
            if (row["available_us"]>row["asof_ms"]*1000+2000000
                    or row["fit_ms"]>row["asof_ms"] or row["target_ms"]!=row["asof_ms"]+d.BAR
                    or row["expected_return_4h"]!=row["predicted_close_gain_bps"]):
                raise ValueError("vector causality or transport alias mismatch")
            causal_checks+=1
    covariance=d.covariance(data,reg["start_ms"],reg["end_ms"])
    risk_checks=0; max_risk=0.; max_asset=0.; max_gross=0.
    for policy in reg["contract"]["policies"]:
        for scenario in ("normal","stress"):
            run=d.sealed(f"replay_{policy}_{scenario}.json")
            for row in run["equity"]:
                if row["phase"]!="calendar_bucket_end": continue
                weights=np.asarray(row["asset_exposure"])
                day=row["timestamp_ms"]-row["timestamp_ms"]%d.DAY
                risk=float(np.sqrt(max(0.,weights@covariance[day]@weights)))
                if risk>.005+1e-10 or weights.max()>.1+1e-10 or weights.sum()>.8+1e-10:
                    raise ValueError("post-cost actual risk cap violation")
                max_risk=max(max_risk,risk); max_asset=max(max_asset,float(weights.max())); max_gross=max(max_gross,float(weights.sum()))
                risk_checks+=3
    baseline=d.b.read(d.b.OUT/"replay_R1_EMA_4h_stress.json")
    linear=d.sealed("replay_AV_linear_stress.json")
    decisions=[r for r in linear["decisions"] if r["overlay_rejected_assets"]]
    interventions=sum(len(r["overlay_rejected_assets"]) for r in decisions)
    assets={s: {"linear_net_usdt":linear["per_asset"][s]["net_pnl"],
                "baseline_net_usdt":baseline["per_asset"][s]["net_pnl"],
                "delta_net_usdt":linear["per_asset"][s]["net_pnl"]-baseline["per_asset"][s]["net_pnl"]}
            for s in data["symbols"]}
    months={}
    for policy in ("R0_EMA_daily","R1_EMA_4h")+tuple(reg["contract"]["policies"]):
        run=(d.b.read(d.b.OUT/f"replay_{policy}_stress.json") if policy.startswith("R")
             else d.sealed(f"replay_{policy}_stress.json"))
        daily=np.asarray(run["daily_log_returns"])
        months[policy]={"August_pct":float(np.expm1(daily[:31].sum())*100),
                        "September_pct":float(np.expm1(daily[31:].sum())*100)}
    daily=np.asarray(linear["daily_log_returns"])
    nav=np.r_[10000.,10000.*np.exp(np.cumsum(daily))]
    amounts=np.diff(nav); top=np.argsort(amounts)[-3:][::-1]
    biggest=[{"ending_mark_utc":datetime.fromtimestamp((reg["start_ms"]+(int(i)+1)*d.DAY)/1000,timezone.utc).isoformat(),
              "net_usdt":float(amounts[i])} for i in top]
    paired=daily-np.asarray(baseline["daily_log_returns"])
    fit_count=len(list(d.OUT.glob("fit_*.json"))); replay_count=len(list(d.OUT.glob("replay_*.json")))
    if fit_count!=21 or replay_count!=6:
        raise ValueError("registered budget changed")
    result=d.write("supplementary_checks.json",{
        "source_path":str(Path(__file__).relative_to(ROOT)).replace("\\","/"),
        "source_pin":d.semantic.pin(Path(__file__)),"registration_sha256":reg["sha256"],
        "dataset_registration_sha256":ds["sha256"],"analysis_fix_sha256":amendment["sha256"],
        "train_only_scaler_checks":scaler_checks,"causality_checks":causal_checks,
        "min_latest_training_origin_gap_bars":min(latest_gaps),"post_cost_risk_checks":risk_checks,
        "max_actual_daily_risk":max_risk,"max_actual_asset_exposure":max_asset,"max_actual_gross_exposure":max_gross,
        "fits_completed":fit_count,"financial_replays_completed":replay_count,
        "linear_action_count":interventions,"linear_intervention_timestamps":len(decisions),
        "linear_vs_baseline_paired_positive_days":int(np.sum(paired>0)),
        "linear_vs_baseline_paired_negative_days":int(np.sum(paired<0)),
        "linear_asset_attribution":assets,"stress_monthly_returns":months,
        "largest_linear_daily_capital_gains":biggest,
        "new_training_fits_in_this_check":0,"new_replays_in_this_check":0,
        "network_calls":0,"promotion_authority":False})
    print(d.json.dumps({"scaler_checks":scaler_checks,"causal_checks":causal_checks,
                        "risk_checks":risk_checks,"linear_interventions":interventions,
                        "fits_total":fit_count,"new_fits":0,"new_replays":0}))


if __name__=="__main__": main()
