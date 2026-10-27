"""Analysis-only LCB result-schema fix; frozen training/replay code untouched."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
os.environ["MPLBACKEND"]="Agg"
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_genetic_primus_action_value_experiment_v1 as d


def amendment():
    reg=d.registration()
    pins={p.name:d.semantic.pin(p) for p in d.OUT.iterdir() if p.is_file()}
    d.write("engineering_analysis_amendment_01.json",{
        "registration_sha256":reg["sha256"],
        "reason":"f.lcb returns a dict; analysis must read its lcb field before bps conversion",
        "failed_analysis_created_artifact":False,
        "changes_to_training_replay_contract_threshold_or_banks":False,
        "source_path":str(Path(__file__).relative_to(ROOT)).replace("\\","/"),
        "source_pin":d.semantic.pin(Path(__file__)),"frozen_artifact_pins":pins})
    print('{"analysis_only_amendment":true,"training_or_replay_changed":false}')


def verify():
    d.registration()
    amended=d.sealed("engineering_analysis_amendment_01.json")
    if d.semantic.pin(ROOT/amended["source_path"])!=amended["source_pin"]:
        raise ValueError("analysis source changed")
    for name,pin in amended["frozen_artifact_pins"].items():
        if d.semantic.pin(d.OUT/name)!=pin:
            raise ValueError("frozen training/replay artifact changed: "+name)
    return amended


def analyze():
    amended=verify(); reg=d.registration(); data=d.b.load_data()
    rates=d.rate_map(d.b.sealed(d.b.MODELED_RATES)["events"],data["symbols"])
    bank=d.b.banks(); alpha=reg["contract"]["bootstrap"]["alpha"]
    ref=d.arrays("reference_evaluation.npz"); checks=0; results={}
    for scenario in ("normal","stress"):
        baseline=d.b.read(d.b.OUT/f"replay_R1_EMA_4h_{scenario}.json")
        for policy in reg["contract"]["policies"]:
            run=d.sealed(f"replay_{policy}_{scenario}.json")
            checks+=d.audit_funding(run,data,rates)
            x=np.asarray(run["daily_log_returns"])
            paired=x-np.asarray(baseline["daily_log_returns"])
            robust={name:{"paired_lcb_bps_day":d.f.lcb(paired,idx,alpha)["lcb"]*10000,
                          "absolute_lcb_bps_day":d.f.lcb(x,idx,alpha)["lcb"]*10000}
                    for name,idx in bank.items()}
            metric=run["metrics"]; base=baseline["metrics"]
            entry={"metrics":metric,"paired_daily_mean_bps":float(paired.mean()*10000),"bootstrap":robust,
                   "delta_net_usdt":metric["final_capital"]-base["final_capital"],
                   "delta_gross":metric["gross_pnl"]-base["gross_pnl"],
                   "delta_costs":metric["fees"]+metric["slippage"]-base["fees"]-base["slippage"],
                   "delta_funding":metric["funding"]-base["funding"]}
            if scenario=="stress":
                states=d.snapshots(run,data); held_mismatch=[]; exposure_diff=[]; local=[]; caps=[]
                for k,t in enumerate(ref["times"][:-1]):
                    cash,q=states[int(t)]; reference_q=ref["quantity"][k]
                    held_mismatch.extend(((q>1e-12)!=(reference_q>1e-12)).tolist())
                    idx=ref["indices"][k]; px=data["closes"][idx]
                    nav=cash+float(q@px); refnav=ref["cash"][k]+float(reference_q@px)
                    exposure_diff.extend((q*px/nav-reference_q*px/refnav).tolist())
                    end=data["closes"][ref["indices"][k+1]]
                    fund=end*rates.get(int(t+d.BAR),np.zeros(8))
                    for asset in np.flatnonzero((q>1e-12)&(ref["base"][k]>0)):
                        value=d.a.paired_value(cash,q,px,end,ref["base"][k],ref["base"][k+1],
                                              ref["covariance"][k],fund,int(asset))
                        local.append(value["target_bps"]); caps.append(value["endpoint_capped"])
                entry["state_transfer"]={"held_state_mismatch_share":float(np.mean(held_mismatch)),
                    "mean_abs_asset_exposure_difference":float(np.mean(abs(np.asarray(exposure_diff)))),
                    "policy_state_label_rows":len(local),
                    "policy_state_label_mean_bps":float(np.mean(local)) if local else None,
                    "policy_state_endpoint_cap_share":float(np.mean(caps)) if caps else None,
                    "reference_state_label_mean_bps":d.sealed("labels_evaluation.json")["mean_bps"]}
            results[policy+"_"+scenario]=entry
    quality=d.sealed("forecast_quality.json")["metrics"]
    primary=results["AV_linear_stress"]
    gate=all(primary["bootstrap"][name][key]>0 for name in ("primary","audit")
             for key in ("paired_lcb_bps_day","absolute_lcb_bps_day"))
    d.write("analysis.json",{"results":results,"forecast_quality":quality,"funding_and_cost_checks":checks,
          "research_robustness_gate":gate,"promotion_authority":False,
          "gate_is_not_prospective_or_full_release_gate":True,"registration_sha256":reg["sha256"],
          "engineering_analysis_amendment_sha256":amended["sha256"]})
    print(d.json.dumps({"funding_cost_checks":checks,"linear_stress_delta_usdt":round(primary["delta_net_usdt"],4),
                        "primary_paired_lcb_bps_day":round(primary["bootstrap"]["primary"]["paired_lcb_bps_day"],6),
                        "research_gate":gate,"promotion_authority":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("amendment","analyze","verify"))
    args=parser.parse_args()
    if args.stage=="verify":
        item=verify(); print(d.json.dumps({"frozen_artifacts_verified":len(item["frozen_artifact_pins"])}))
    else: globals()[args.stage]()
