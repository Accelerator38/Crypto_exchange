"""One bounded offline stage; never collector, network, runtime or tuning."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import action_value_portfolio_v1 as a
from tools import run_genetic_primus_forecast4h_v1 as b
from tools import run_genetic_primus_ema_units_v1 as units
from tools import check_genetic_primus_action_value_v1 as semantic
f, m, BAR, DAY = a.f, a.f.m, a.f.BAR_MS, a.f.DAY_MS
OUT = b.BASE / "action_value_experiment_v1_20261012"
CONFIG = ROOT / "configs/genetic_primus_action_value_experiment_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v10.json"


def stamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    value = f.p.c.plain(value)
    value["sha256"] = m.identity(value)
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return value


def npz(name, **arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def arrays(name):
    with np.load(OUT / name, allow_pickle=False) as stream:
        return {key: stream[key] for key in stream.files}


def sealed(name):
    return b.sealed(OUT / name)


def registration():
    reg = sealed("registration.json")
    for name, expected in reg["pins"].items():
        if semantic.pin(ROOT / name) != expected:
            raise ValueError("registered input changed: " + name)
    return reg


def rate_map(events, symbols):
    indices = {s.replace("/", ""): i for i, s in enumerate(symbols)}
    result = {}
    seen = set()
    for row in events:
        key = (int(row["settlement_ms"]), row["symbol"].replace("/", ""))
        if key in seen:
            raise ValueError("duplicate funding input")
        seen.add(key)
        if key[1] in indices:
            result.setdefault(key[0], np.full(len(symbols), np.nan))[indices[key[1]]] = row["rate_fraction"]
    return result


def coverage(rates, start, end):
    first = ((start + 2 * BAR - 1) // (2 * BAR)) * (2 * BAR)
    for t in range(first, end + 1, 2 * BAR):
        if t not in rates or not np.isfinite(rates[t]).all():
            raise ValueError("incomplete funding at " + str(t))


def covariance(data, start, end):
    return {day: f.p.risk_covariance(data["times"], data["closes"], day)[0]
            for day in range(start - start % DAY, end - end % DAY + DAY, DAY)}


def vectors_for(times, predictions, model_shas, origin_times=None):
    rows = []
    for i, t in enumerate(times):
        pred = np.asarray(predictions[i], float).tolist()
        row = {"schema": "action_value_vector/1", "asof_ms": int(t), "target_ms": int(t + BAR),
               "available_us": int(t * 1000 + 1000000), "model_sha256": model_shas[i],
               "predicted_close_gain_bps": pred, "expected_return_4h": pred,
               "legacy_transport_alias_is_not_price_return": True,
               "target_kind": "paired_incremental_close_vs_baseline_rebalance_net_bps",
               "fit_ms": None if origin_times is None else int(origin_times[i]),
               "runtime_authority": False, "promotion_authority": False}
        row["sha256"] = m.identity(row); rows.append(row)
    return rows


def synthetic():
    contract = b.registration()["contract"]
    start = stamp("2026-06-01T00:00:00Z")
    times = np.arange(start, start + 2 * DAY + BAR, BAR, dtype=np.int64)
    prices = np.array([[100. + i * (j + 1) / 20. for j in range(8)] for i in range(len(times))])
    features = np.zeros((len(times), 8, 6)); features[:, :, 0] = 1.
    features[6:9, 1, 0] = -1.
    data = {"times": times, "closes": prices, "features": features, "symbols": list(b.SYMBOLS)}
    cov = {int(day): np.eye(8) * .000004 for day in range(start, start + 3 * DAY, DAY)}
    rates = [{"settlement_ms": int(t), "symbol": symbol, "rate_fraction": .0001}
             for t in times[::2] for symbol in b.SYMBOLS]
    sha = m.identity({"synthetic": True})
    zero = vectors_for(times[:-1], np.zeros((len(times)-1, 8)), [sha] * (len(times)-1))
    old = f.replay(data, zero, cov, rates, contract, "R1_EMA_4h", "stress")
    held = a.action_replay(data, zero, cov, rates, contract, "AV_never", "stress")
    for key in old["metrics"]:
        if not np.isclose(old["metrics"][key], held["metrics"][key], atol=1e-9, rtol=1e-12):
            raise AssertionError("synthetic hold parity: " + key)
    if old["daily_log_returns"] != held["daily_log_returns"]:
        raise AssertionError("synthetic daily path")
    pred = np.zeros((len(times)-1, 8)); pred[1:3, 0] = 1.
    changed = a.action_replay(data, vectors_for(times[:-1], pred, [sha] * (len(times)-1)),
                             cov, rates, contract, "AV_synthetic_close", "stress")
    checks = b.audit_replay(changed, {sha})
    prefix = np.zeros(8); seen = set(); fund_checks = 0
    for row in changed["ledger"]:
        asset = list(b.SYMBOLS).index(row["symbol"])
        t = (row["timestamp_us"] // 1000 // BAR) * BAR
        px = prices[np.searchsorted(times, t), asset]
        if row["kind"] == "modeled_fill":
            prefix[asset] = row["quantity_after"]
            expected = row["quantity"] * px * (.0008 + .0006 * (1.0008 if row["side"] == "BUY" else .9992))
            if abs(row["fee"] + row["slippage"] - expected) > 1e-9:
                raise AssertionError("actual delta-unit costs")
            if row["timestamp_us"] != t * 1000 + contract["fill_delay_us"]:
                raise AssertionError("phantom boundary trade")
        else:
            key = (t, asset)
            if key in seen or abs(row["cash_delta"] + prefix[asset] * px * .0001) > 1e-9:
                raise AssertionError("funding duplication/formula")
            seen.add(key); fund_checks += 1
    asset0 = [r for r in changed["ledger"] if r.get("kind") == "modeled_fill" and r["symbol"] == b.SYMBOLS[0]]
    sell = [r for r in asset0 if r["timestamp_us"] == (start + BAR) * 1000 + contract["fill_delay_us"]][0]
    if sell["side"] != "SELL" or sell["quantity_after"] != 0.:
        raise AssertionError("requested close missing")
    # Following flat-to-positive EMA re-entry is an actual charged BUY, not a
    # restoration secretly inserted by the label endpoint.
    buy = [r for r in asset0 if r["timestamp_us"] == (start + 2*BAR) * 1000 + contract["fill_delay_us"]][0]
    if buy["side"] != "BUY" or buy["fee"] <= 0.:
        raise AssertionError("actual re-entry accounting")
    q = np.full(8, 10.); px = np.full(8, 100.); base = np.full(8, .1)
    counter = 0
    for end_px in (90., 100., 110.):
        for funding in (-.01, 0., .01):
            for asset in range(8):
                result = a.paired_value(2000., q, px, np.full(8, end_px), base, base,
                                        np.eye(8)*.000004, np.full(8, 100.*funding), asset)
                if not np.isfinite(result["target_bps"]):
                    raise AssertionError("joint label")
                counter += 1
    write("synthetic_checks.json", {"passed": True, "hold_parity": True,
          "actual_transition_costs": True, "no_label_boundary_trades": True,
          "no_double_funding": True, "independent_ledger_checks": checks,
          "funding_checks": fund_checks, "joint_counterfactual_cases": counter,
          "new_training_fits": 0, "network_calls": 0, "collector_calls": 0})
    print(json.dumps({"synthetic_passed": True, "ledger_checks": checks,
                      "funding_checks": fund_checks, "joint_cases": counter}))


def prepare():
    contract = b.read(CONFIG); source = units.registration()
    if any(v is not False for v in contract["safety"].values()):
        raise ValueError("safety authorities must remain false")
    if not sealed("synthetic_checks.json")["passed"]:
        raise ValueError("synthetic prerequisite")
    data = b.load_data(); rates = rate_map(b.sealed(b.MODELED_RATES)["events"], data["symbols"])
    for segment in contract["label_reference_segments"]:
        coverage(rates, stamp(segment["start"]), stamp(segment["last_state"]))
    start, end = stamp(contract["evaluation"]["start"]), stamp(contract["evaluation"]["end_exclusive"])
    origins = list(range(start, end, 3*DAY))
    if len(origins) != contract["new_linear_fits_budget"]:
        raise ValueError("fit budget mismatch")
    registry = b.read(REGISTRY)
    if registry["previous_registry_sha256"] != semantic.pin(ROOT/registry["previous_registry_path"])["sha256"]:
        raise ValueError("registry predecessor mismatch")
    for objective in registry["objectives"]:
        if objective["config_sha256"] != semantic.pin(ROOT/objective["config_path"])["sha256"]:
            raise ValueError("objective registry pin mismatch")
    paths = [CONFIG, REGISTRY, ROOT/"configs/genetic_primus_objective_registry_v9.json",
             Path(__file__), ROOT/"src/exia/genetic_primus/action_value_portfolio_v1.py",
             semantic.OUT/"registration.json", semantic.OUT/"receipt.json", b.DATA, b.MODELED_RATES,
             b.OUT/"resample_bank.npz", OUT/"synthetic_checks.json"]
    for folder in (b.OUT, units.previous.OUT, units.OUT):
        receipt = b.read(folder/"receipt.json"); paths.append(folder/"receipt.json")
        for name, pin in receipt["artifact_pins"].items():
            path = folder/name
            if semantic.pin(path) != pin:
                raise ValueError("parent artifact changed")
            paths.append(path)
    bank = b.banks()
    if any(v.shape != (5000,61) for v in bank.values()):
        raise ValueError("common bank shape")
    folds = [{"origin_ms": t, "label_cutoff_us": t*1000-7*BAR*1000,
              "purge_bars": 7, "label_available_delay_us": 2250000} for t in origins]
    write("registration.json", {"schema": "action_value_experiment_registration/1", "contract": contract,
          "baseline_contract": b.registration()["contract"], "start_ms": start, "end_ms": end,
          "folds": folds, "folds_sha256": m.identity(folds), "parent_sha256": source["sha256"],
          "semantic_parent_amended_before_training": True,
          "pins": {str(p.relative_to(ROOT)).replace("\\", "/"): semantic.pin(p) for p in paths}})
    print(json.dumps({"registered": sealed("registration.json")["sha256"],
                      "fits_budget": 21, "policy_replays_budget": 6, "hyperparameter_trials": 0}))


def reference(segment):
    reg = registration(); data = b.load_data()
    spec = next(s for s in reg["contract"]["label_reference_segments"] if s["id"] == segment)
    start, end = stamp(spec["start"]), stamp(spec["last_state"])
    times = np.arange(start, end+BAR, BAR, dtype=np.int64)
    indices = np.searchsorted(data["times"], times)
    if not np.array_equal(data["times"][indices], times):
        raise ValueError("reference price panel gap")
    cov = covariance(data, start, end)
    events = b.sealed(b.MODELED_RATES)["events"]
    rates = rate_map(events, data["symbols"]); coverage(rates,start,end)
    before_cash, before_q, after_cash, after_q, bases, covs = [], [], [], [], [], []
    cash, q = 10000., np.zeros(8)
    for t, idx in zip(times, indices):
        px = data["closes"][idx]
        cash -= float(q @ (px * rates.get(int(t), np.zeros(8))))
        before_cash.append(cash); before_q.append(q.copy())
        sigma = cov[int(t)-int(t)%DAY]
        base = f.p.bounded_weights(data["features"][idx,:,0] > 0, sigma, budget=.005)
        if segment == "evaluation" and t == end:
            base = np.zeros(8)
        bases.append(base); covs.append(sigma)
        cash, q, _ = a.move(cash,q,a.desired(cash,q,px,base,6.,8.),px,6.,8.)
        after_cash.append(cash); after_q.append(q.copy())
    parity = 0
    if segment == "evaluation":
        saved = b.read(b.OUT/"replay_R1_EMA_4h_stress.json")
        calendar = {r["timestamp_ms"]: r for r in saved["equity"] if r["phase"] == "calendar_bucket_end"}
        for k,t in enumerate(times):
            nav = after_cash[k] + float(after_q[k] @ data["closes"][indices[k]])
            if not np.isclose(nav,calendar[int(t)]["nav"],atol=1e-7,rtol=1e-12):
                raise ValueError("reference calendar parity")
            parity += 1
    npz(f"reference_{segment}.npz", times=times, indices=indices, cash=np.array(before_cash),
        quantity=np.array(before_q), cash_after=np.array(after_cash), quantity_after=np.array(after_q),
        base=np.array(bases), covariance=np.array(covs))
    write(f"reference_{segment}.json", {"segment": segment,"states":len(times),
          "funding_complete":True,"calendar_parity_checks":parity,
          "states_pin":semantic.pin(OUT/f"reference_{segment}.npz"),"registration_sha256":reg["sha256"]})
    print(json.dumps({"reference":segment,"states":len(times),"calendar_parity_checks":parity}))


def labels(segment):
    reg=registration(); ref=arrays(f"reference_{segment}.npz"); data=b.load_data()
    rates=rate_map(b.sealed(b.MODELED_RATES)["events"],data["symbols"])
    rows=[]; xs=[]
    for k,t in enumerate(ref["times"][:-1]):
        idx=ref["indices"][k]; end_idx=ref["indices"][k+1]; h=int(t+BAR)
        now,end=data["closes"][idx],data["closes"][end_idx]
        funding=end*rates.get(h,np.zeros(8))
        for asset in np.flatnonzero((ref["quantity"][k]>1e-12)&(ref["base"][k]>0)):
            result=a.paired_value(ref["cash"][k],ref["quantity"][k],now,end,
                                 ref["base"][k],ref["base"][k+1],ref["covariance"][k],funding,int(asset))
            rows.append([t,h*1000+2250000,asset,result["target_bps"],result["endpoint_scale"],
                         result["delta_gross"],result["delta_costs"],result["delta_funding_paid"],
                         result["cross_asset_gross"]])
            xs.append(data["features"][idx,asset])
    panel=np.asarray(rows,float)
    if not len(panel) or not np.isfinite(panel).all() or not np.isfinite(xs).all():
        raise ValueError("empty/nonfinite action labels")
    npz(f"labels_{segment}.npz",asof=panel[:,0].astype(np.int64),available_us=panel[:,1].astype(np.int64),
        asset=panel[:,2].astype(np.int64),y=panel[:,3],endpoint_scale=panel[:,4],x=np.asarray(xs),
        delta_gross=panel[:,5],delta_costs=panel[:,6],delta_funding=panel[:,7],cross_asset_gross=panel[:,8])
    y=panel[:,3]
    write(f"labels_{segment}.json",{"segment":segment,"rows":len(y),"mean_bps":float(y.mean()),
          "quantiles_bps":np.quantile(y,[0,.05,.25,.5,.75,.95,1]),"positive_share":float(np.mean(y>0)),
          "endpoint_cap_share":float(np.mean(panel[:,4]<1.-1e-10)),
          "reference_pin":semantic.pin(OUT/f"reference_{segment}.npz"),
          "labels_pin":semantic.pin(OUT/f"labels_{segment}.npz"),"registration_sha256":reg["sha256"]})
    print(json.dumps({"labels":segment,"rows":len(y),"mean_bps":round(float(y.mean()),4),
                      "positive_pct":round(float(np.mean(y>0))*100,3),"cap_pct":round(float(np.mean(panel[:,4]<1.-1e-10))*100,3)}))


def dataset():
    reg=registration(); banks=[arrays("labels_train.npz"),arrays("labels_evaluation.npz")]
    bank={key:np.concatenate([p[key] for p in banks]) for key in banks[0]}
    order=np.lexsort((bank["asset"],bank["asof"]))
    bank={key:value[order] for key,value in bank.items()}
    folds=[]
    for fold in reg["folds"]:
        idx=np.flatnonzero(bank["available_us"]<=fold["label_cutoff_us"])
        if len(np.unique(bank["asof"][idx]))<50 or np.max(bank["available_us"][idx])>fold["label_cutoff_us"]:
            raise ValueError("insufficient or immature labels")
        folds.append(fold|{"indices":idx.tolist(),"indices_sha256":m.array_identity(idx),
                           "training_samples_sha256":m.array_identity(bank["asof"][idx],bank["asset"][idx],bank["x"][idx],bank["y"][idx])})
    npz("dataset.npz",**bank)
    write("folds.json",{"folds":folds,"registration_sha256":reg["sha256"]})
    write("dataset_registration.json",{"registration_sha256":reg["sha256"],"rows":len(bank["y"]),
          "dataset_pin":semantic.pin(OUT/"dataset.npz"),"folds_pin":semantic.pin(OUT/"folds.json"),
          "labels_pins":{s:semantic.pin(OUT/f"labels_{s}.npz") for s in ("train","evaluation")},
          "references_pins":{s:semantic.pin(OUT/f"reference_{s}.npz") for s in ("train","evaluation")},
          "resample_bank_pin":semantic.pin(b.OUT/"resample_bank.npz"),"fits_completed_before_seal":0})
    print(json.dumps({"dataset_rows":len(bank["y"]),"folds":len(folds),"first_train_rows":len(folds[0]["indices"]),"fits_completed":0}))


def checked_dataset():
    d=sealed("dataset_registration.json")
    if semantic.pin(OUT/"dataset.npz")!=d["dataset_pin"] or semantic.pin(OUT/"folds.json")!=d["folds_pin"]:
        raise ValueError("dataset/fold changed")
    for segment,pin in d["labels_pins"].items():
        if semantic.pin(OUT/f"labels_{segment}.npz")!=pin:
            raise ValueError("source labels changed")
    for segment,pin in d["references_pins"].items():
        if semantic.pin(OUT/f"reference_{segment}.npz")!=pin:
            raise ValueError("reference states changed")
    return d,arrays("dataset.npz"),sealed("folds.json")["folds"]


def fit(first,count):
    reg=registration(); ds,bank,folds=checked_dataset()
    if first<0 or count<1 or count>3 or first+count>21:
        raise ValueError("bounded fit batch 1..3 within registered 21 origins")
    for k in range(first,first+count):
        fold=folds[k]; idx=np.asarray(fold["indices"],int); x,y=bank["x"][idx],bank["y"][idx]
        begin=time.perf_counter()
        scaler=m.fit_scaler(x,y)
        parameters=m.fit_linear(scaler.transform(x),scaler.target(y),np.ones(len(y)),l2=.01)
        identity={"fit_ms":fold["origin_ms"],"label_cutoff_us":fold["label_cutoff_us"],
                  "training_samples_sha256":fold["training_samples_sha256"],"indices_sha256":fold["indices_sha256"],
                  "dataset_registration_sha256":ds["sha256"],"feature_definition_sha256":f.d.FEATURE_SHA256,
                  "uniform_all_outcomes":True,"purge_bars":7}
        model=m.model_payload("linear6",parameters,scaler,weighting="uniform",seed=None,epochs=None,training_identity=identity)
        model["target_kind"]="paired_incremental_close_vs_baseline_rebalance_net_bps"
        model["sha256"]=m.identity({key:v for key,v in model.items() if key!="sha256"})
        constant={"kind":"train_only_mean_action_value","mean_bps":float(y.mean()),"training_identity":identity}
        constant["sha256"]=m.identity(constant)
        elapsed=time.perf_counter()-begin
        if elapsed>=1.:
            raise ValueError("modeled one-second readiness not supported by measured fit")
        write(f"fit_{k:02d}.json",{"origin_index":k,"origin_ms":fold["origin_ms"],"train_rows":len(y),
              "train_origin_clusters":len(np.unique(bank["asof"][idx])),"max_label_available_us":int(bank["available_us"][idx].max()),
              "label_cutoff_us":fold["label_cutoff_us"],"model":model,"constant":constant,
              "fit_seconds":elapsed,"model_ready_us":fold["origin_ms"]*1000+1000000,
              "registration_sha256":reg["sha256"]})
    print(json.dumps({"fits_written":list(range(first,first+count)),"parameter_count":7,"new_threshold_trials":0}))


def predictions():
    reg=registration(); data=b.load_data(); times=np.arange(reg["start_ms"],reg["end_ms"],BAR,dtype=np.int64)
    indices=np.searchsorted(data["times"],times); fits=[sealed(f"fit_{i:02d}.json") for i in range(21)]
    values={name:[] for name in reg["contract"]["policies"]}; shas={name:[] for name in values}; origins=[]
    rule_sha=m.identity({"registered_causal_rule":reg["contract"]["causal_rule_control"],"registration":reg["sha256"]})
    for t,idx in zip(times,indices):
        k=int((t-reg["start_ms"])//(3*DAY)); fitted=fits[k]
        if fitted["origin_ms"]>t or fitted["model_ready_us"]>t*1000+1000000:
            raise ValueError("future/unready model")
        origins.append(fitted["origin_ms"])
        values["AV_linear"].append(m.predict(fitted["model"],data["features"][idx]))
        values["AV_constant"].append(np.full(8,fitted["constant"]["mean_bps"]))
        values["AV_causal_rule"].append(-(data["closes"][idx]/data["closes"][idx-1]-1)*10000.-28.)
        for name,sha in (("AV_linear",fitted["model"]["sha256"]),("AV_constant",fitted["constant"]["sha256"]),("AV_causal_rule",rule_sha)):
            shas[name].append(sha)
    for name in values:
        write(f"vectors_{name}.json",{"policy":name,"rows":vectors_for(times,values[name],shas[name],origins),
              "target_kind":"net_action_value_not_price_return","registration_sha256":reg["sha256"]})
    print(json.dumps({"linear_models":21,"vectors_per_policy":len(times),"asset_predictions_per_policy":len(times)*8}))


def quality():
    reg=registration(); labels=arrays("labels_evaluation.npz"); results={}
    for name in ["AV_never"]+reg["contract"]["policies"]:
        if name=="AV_never": pred=np.zeros(len(labels["y"]))
        else:
            rows=sealed(f"vectors_{name}.json")["rows"]; mapping={r["asof_ms"]:r["predicted_close_gain_bps"] for r in rows}
            pred=np.array([mapping[int(t)][int(asset)] for t,asset in zip(labels["asof"],labels["asset"])])
        y=labels["y"]
        results[name]={"rmse_bps":float(np.sqrt(np.mean((pred-y)**2))),"mae_bps":float(np.mean(abs(pred-y))),
                       "predicted_positive_share":float(np.mean(pred>0)),"prediction_quantiles_bps":np.quantile(pred,[0,.05,.5,.95,1]),
                       "positive_label_mean_bps":float(y[pred>0].mean()) if np.any(pred>0) else None}
    write("forecast_quality.json",{"rows":len(y),"metrics":results,"reference_states_only":True,
          "local_counterfactual_is_not_policy_pnl":True,"registration_sha256":reg["sha256"]})
    print(json.dumps({"quality_rows":len(y),"rmse_bps":{k:round(v["rmse_bps"],4) for k,v in results.items()}}))


def parity(scenario):
    reg=registration(); data=b.load_data()
    # Bridge is zero and no intervention; old price forecasts are NOT reused.
    times=np.arange(reg["start_ms"],reg["end_ms"],BAR,dtype=np.int64)
    sha=m.identity({"never_intervene":reg["sha256"]})
    zero=vectors_for(times,np.zeros((len(times),8)),[sha]*len(times))
    cov=covariance(data,reg["start_ms"],reg["end_ms"])
    result=a.action_replay(data,zero,cov,b.sealed(b.MODELED_RATES)["events"],reg["baseline_contract"],"AV_never",scenario)
    old=b.read(b.OUT/f"replay_R1_EMA_4h_{scenario}.json")
    for key in old["metrics"]:
        if not np.isclose(result["metrics"][key],old["metrics"][key],atol=1e-7,rtol=1e-11):
            raise ValueError("historical hold parity: "+key)
    if not np.array_equal(result["daily_log_returns"],old["daily_log_returns"]):
        raise ValueError("historical daily path not bit exact")
    checks=b.audit_replay(result,{sha})
    write(f"parity_{scenario}.json",{"passed":True,"daily_returns_bit_exact":True,"ledger_checks":checks,
          "metrics":result["metrics"],"registration_sha256":reg["sha256"]})
    print(json.dumps({"parity":scenario,"passed":True,"ledger_checks":checks}))


def replay(policy,scenario):
    reg=registration()
    if policy not in reg["contract"]["policies"]:
        raise ValueError("unregistered policy")
    for cost in ("normal","stress"):
        if not sealed(f"parity_{cost}.json")["passed"]:
            raise ValueError("both parity cases required")
    rows=sealed(f"vectors_{policy}.json")["rows"]; data=b.load_data()
    result=a.action_replay(data,rows,covariance(data,reg["start_ms"],reg["end_ms"]),
                          b.sealed(b.MODELED_RATES)["events"],reg["baseline_contract"],policy,scenario)
    checks=b.audit_replay(result,{r["model_sha256"] for r in rows})
    result["independent_ledger_checks"]=checks; result["registration_sha256"]=reg["sha256"]
    result["vectors_pin"]=semantic.pin(OUT/f"vectors_{policy}.json")
    write(f"replay_{policy}_{scenario}.json",result)
    metric=result["metrics"]
    print(json.dumps({"policy":policy,"costs":scenario,"net_pct":round(metric["net_return"]*100,5),
                      "dd_pct":round(metric["max_drawdown"]*100,5),"fills":metric["fills"]}))


def snapshots(run,data):
    result={}
    for row in run["equity"]:
        if row["phase"] in ("before_funding_and_decision","after_modeled_funding"):
            t=row["timestamp_ms"]; idx=np.searchsorted(data["times"],t)
            result[t]=(row["cash"],np.asarray(row["asset_exposure"])*row["nav"]/data["closes"][idx])
    return result


def audit_funding(run,data,rates):
    by_time=defaultdict(list)
    for row in run["ledger"]:
        t=(row["timestamp_us"]//1000//BAR)*BAR; by_time[t].append(row)
    q=np.zeros(8); checks=0
    for t in range(run["start_ms"],run["end_ms"]+BAR,BAR):
        px=data["closes"][np.searchsorted(data["times"],t)]
        fund=[r for r in by_time[t] if r["kind"]=="modeled_funding"]
        expected=np.flatnonzero(q!=0) if t in rates else np.array([],int)
        if len(fund)!=len(expected) or len({r["symbol"] for r in fund})!=len(fund):
            raise ValueError("missing or double funding")
        for row in fund:
            asset=data["symbols"].index(row["symbol"])
            if abs(row["cash_delta"]+q[asset]*px[asset]*rates[t][asset])>1e-8:
                raise ValueError("independent funding valuation")
            checks+=1
        for row in by_time[t]:
            if row["kind"]=="modeled_fill":
                if row["timestamp_us"]!=t*1000+2250000:
                    raise ValueError("phantom label boundary trade")
                q[data["symbols"].index(row["symbol"])]=row["quantity_after"]
                sign=1 if row["side"]=="BUY" else -1
                fee,drag=(6/10000,(8 if run["scenario"]=="stress" else 2)/10000)
                expected_cost=row["quantity"]*row["reference_close"]*(drag+fee*(1+sign*drag))
                if abs(row["fee"]+row["slippage"]-expected_cost)>1e-8:
                    raise ValueError("actual transition costs")
                checks+=1
    return checks


def analyze():
    reg=registration(); data=b.load_data(); rates=rate_map(b.sealed(b.MODELED_RATES)["events"],data["symbols"])
    bank=b.banks(); alpha=reg["contract"]["bootstrap"]["alpha"]
    ref=arrays("reference_evaluation.npz"); checks=0; results={}
    for scenario in ("normal","stress"):
        baseline=b.read(b.OUT/f"replay_R1_EMA_4h_{scenario}.json")
        for policy in reg["contract"]["policies"]:
            run=sealed(f"replay_{policy}_{scenario}.json"); checks+=audit_funding(run,data,rates)
            x=np.asarray(run["daily_log_returns"]); paired=x-np.asarray(baseline["daily_log_returns"])
            robust={name:{"paired_lcb_bps_day":f.lcb(paired,idx,alpha)*10000,
                          "absolute_lcb_bps_day":f.lcb(x,idx,alpha)*10000} for name,idx in bank.items()}
            metric=run["metrics"]; base=baseline["metrics"]
            entry={"metrics":metric,"paired_daily_mean_bps":float(paired.mean()*10000),"bootstrap":robust,
                   "delta_net_usdt":metric["final_capital"]-base["final_capital"],
                   "delta_gross":metric["gross_pnl"]-base["gross_pnl"],
                   "delta_costs":metric["fees"]+metric["slippage"]-base["fees"]-base["slippage"],
                   "delta_funding":metric["funding"]-base["funding"]}
            if scenario=="stress":
                states=snapshots(run,data); held_mismatch=[]; exposure_diff=[]; local=[]; caps=[]
                for k,t in enumerate(ref["times"][:-1]):
                    cash,q=states[int(t)]; reference_q=ref["quantity"][k]
                    held_mismatch.extend(((q>1e-12)!=(reference_q>1e-12)).tolist())
                    idx=ref["indices"][k]; px=data["closes"][idx]
                    nav=cash+float(q@px); refnav=ref["cash"][k]+float(reference_q@px)
                    exposure_diff.extend((q*px/nav-reference_q*px/refnav).tolist())
                    end=data["closes"][ref["indices"][k+1]]; fund=end*rates.get(int(t+BAR),np.zeros(8))
                    for asset in np.flatnonzero((q>1e-12)&(ref["base"][k]>0)):
                        value=a.paired_value(cash,q,px,end,ref["base"][k],ref["base"][k+1],
                                             ref["covariance"][k],fund,int(asset))
                        local.append(value["target_bps"]); caps.append(value["endpoint_capped"])
                entry["state_transfer"]={"held_state_mismatch_share":float(np.mean(held_mismatch)),
                    "mean_abs_asset_exposure_difference":float(np.mean(abs(np.asarray(exposure_diff)))),
                    "policy_state_label_rows":len(local),"policy_state_label_mean_bps":float(np.mean(local)) if local else None,
                    "policy_state_endpoint_cap_share":float(np.mean(caps)) if caps else None,
                    "reference_state_label_mean_bps":sealed("labels_evaluation.json")["mean_bps"]}
            results[policy+"_"+scenario]=entry
    quality=sealed("forecast_quality.json")["metrics"]
    primary=results["AV_linear_stress"]
    gate=all(primary["bootstrap"][name][key]>0 for name in ("primary","audit")
             for key in ("paired_lcb_bps_day","absolute_lcb_bps_day"))
    write("analysis.json",{"results":results,"forecast_quality":quality,"funding_and_cost_checks":checks,
          "research_robustness_gate":gate,"promotion_authority":False,
          "gate_is_not_prospective_or_full_release_gate":True,"registration_sha256":reg["sha256"]})
    print(json.dumps({"funding_cost_checks":checks,"linear_stress_delta_usdt":round(primary["delta_net_usdt"],4),
                      "research_gate":gate,"promotion_authority":False}))


def charts():
    import matplotlib.pyplot as plt
    reg=registration()
    for scenario in ("normal","stress"):
        runs=[("EMA ежедневно",b.read(b.OUT/f"replay_R0_EMA_daily_{scenario}.json")),
              ("EMA 4ч",b.read(b.OUT/f"replay_R1_EMA_4h_{scenario}.json"))]
        runs += [(name,sealed(f"replay_{name}_{scenario}.json")) for name in reg["contract"]["policies"]]
        fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={"height_ratios":[2,1]})
        for name,run in runs:
            eq=[r for r in run["equity"] if r["phase"]=="calendar_bucket_end"]
            dates=[datetime.fromtimestamp(r["timestamp_ms"]/1000,timezone.utc) for r in eq]
            axes[0].plot(dates,[r["nav"] for r in eq],label=f'{name}: {run["metrics"]["net_return"]*100:+.3f}%')
            axes[1].plot(dates,[r["drawdown"]*100 for r in eq])
        axes[0].set_title(f"Добавочная выгода закрытия | {scenario} | раскрытый август–сентябрь 2026")
        axes[0].set_ylabel("Капитал, USDT (старт 10 000)"); axes[1].set_ylabel("Просадка, %")
        axes[1].set_xlabel("Дата UTC"); axes[0].legend(fontsize=9)
        for ax in axes: ax.grid(alpha=.25)
        fig.text(.5,.015,"Модельное исполнение; непрерывный ledger; не независимая валидация и не реальные fills",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,1)); target=OUT/f"capital_drawdown_{scenario}.png"
        if target.exists(): raise ValueError("chart already exists")
        fig.savefig(target,dpi=130); plt.close(fig)
    print(json.dumps({"charts_created":2}))


def report():
    reg=registration(); analysis=sealed("analysis.json"); results=analysis["results"]
    lines=["# Добавочная выгода действия: итог исследовательского эксперимента", "",
           "Период: 1 августа — 30 сентября 2026, UTC. Период ранее раскрыт. Исполнение OHLCV-модельное, не квалифицированные реальные fills. Все торговые полномочия false.","",
           "## Что изменилось и что проверено","",
           "Одна pooled linear6: шесть признаков + intercept, семь параметров, ridge 0,01, uniform для всех исходов. Переобучение каждые 72 часа; отдельный train-only scaler и SHA на каждом origin. Горизонт, шаг labels и решения — 4 часа. Purge 7 баров плюс точная доступность label после fill-delay.","",
           "Уточнение до обучения: hold означает исходный EMA-rebalance, не сохранение units. В label сравниваются полные портфельные ветки с одним cash, фактическим delta-переходом и общими конечными units. Изменение корреляционного риска других активов входит в цель. Конечное восстановление — только граница label: в replay нет фиктивных trades и принудительного восстановления.","",
           "Funding до весны 2026 отсутствует, 27–31 июля есть пропуск. Net-labels построены на полностью покрытых блоках 1 мая — 27 июля 12:00 и 1 августа — 1 октября. Более ранние цены служат признакам/риску, но не маскируются под net-labels. Отдельные reference-сегменты открыто начинают cash=10 000; финансовый replay августа–сентября не сбрасывает капитал.","",
           "Синтетические проверки, bit-exact hold-parity с EMA в normal/stress, dataset/fold/index/model/scaler hashes и общий primary/audit bootstrap проверены. Бюджет: 21 линейное обучение; три фиксированных политики × два cost-сценария; никаких новых порогов, GA или MLP.","",
           "## Результат после stress-costs","",
           "| Политика | Net, % | DD, % | Экспозиция, % | Оборот / старт | Fills | Удержание, ч |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("R0_EMA_daily","R1_EMA_4h"):
        metric=b.read(b.OUT/f"replay_{name}_stress.json")["metrics"]
        lines.append(f'| {name} | {metric["net_return"]*100:+.4f} | {metric["max_drawdown"]*100:.4f} | {metric["mean_exposure"]*100:.3f} | {metric["turnover_initial_capital"]:.3f} | {metric["fills"]} | {metric["mean_holding_hours"]:.2f} |')
    for name in reg["contract"]["policies"]:
        metric=results[name+"_stress"]["metrics"]
        lines.append(f'| {name} | {metric["net_return"]*100:+.4f} | {metric["max_drawdown"]*100:.4f} | {metric["mean_exposure"]*100:.3f} | {metric["turnover_initial_capital"]:.3f} | {metric["fills"]} | {metric["mean_holding_hours"]:.2f} |')
    lines += ["","AV_constant — train-only средняя net-добавка. AV_causal_rule — отрицательная последняя 4h-доходность в bps минус фиксированные 28 bps stress-roundtrip. Никогда не вмешиваться — EMA 4h, не новый обученный контроль.","",
              "## Labels и качество прогноза","","| Сегмент | Labels | Среднее, bps | Положительные, % | Endpoint caps, % |","|---|---:|---:|---:|---:|"]
    for segment in ("train","evaluation"):
        d=sealed(f"labels_{segment}.json")
        lines.append(f'| {segment} | {d["rows"]} | {d["mean_bps"]:.3f} | {d["positive_share"]*100:.3f} | {d["endpoint_cap_share"]*100:.3f} |')
    lines += ["","| Прогноз | RMSE, bps | MAE, bps | Прогноз >0, % |","|---|---:|---:|---:|"]
    for name,v in analysis["forecast_quality"].items():
        lines.append(f'| {name} | {v["rmse_bps"]:.3f} | {v["mae_bps"]:.3f} | {v["predicted_positive_share"]*100:.3f} |')
    lines += ["","Эта таблица относится к фиксированным reference-состояниям. Локальная добавка с общей конечной позицией не суммируется в фактический PnL многократных вмешательств.","",
              "## Бухгалтерия и парная добавка, stress","",
              "| Политика | Gross | Fees | Slippage | Funding | ΔNet к EMA4h | Primary LCB Δ, bps/день | Audit LCB Δ |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in reg["contract"]["policies"]:
        d=results[name+"_stress"]; v=d["metrics"]
        lines.append(f'| {name} | {v["gross_pnl"]:.2f} | {v["fees"]:.2f} | {v["slippage"]:.2f} | {v["funding"]:.2f} | {d["delta_net_usdt"]:+.2f} | {d["bootstrap"]["primary"]["paired_lcb_bps_day"]:.4f} | {d["bootstrap"]["audit"]["paired_lcb_bps_day"]:.4f} |')
    lines += ["","Общий замороженный resample bank, nominal alpha=0,0025 для раскрытой цепочки 20 contrasts. Это не полный global multiple-testing correction: число прежних адаптивных решений неполно. Audit seed не заменяет новых данных. Absolute LCB и normal/sensitivity расчёты находятся в analysis.json.","",
              "## Перенос состояний reference → новая политика","",
              "| Политика | Несовпадение held-state, % | Средняя абсолютная разница доли актива, п.п. | Local labels в новых состояниях | Caps, % |","|---|---:|---:|---:|---:|"]
    for name in reg["contract"]["policies"]:
        v=results[name+"_stress"]["state_transfer"]
        lines.append(f'| {name} | {v["held_state_mismatch_share"]*100:.3f} | {v["mean_abs_asset_exposure_difference"]*100:.4f} | {v["policy_state_label_rows"]} | {(v["policy_state_endpoint_cap_share"] or 0)*100:.3f} |')
    lines += ["","Модель не получает portfolio-state признаков: изменение распределения состояний — ограничение, а не автоматически исправленная проблема. Неизменные EMA-входы могут возвращать закрытый актив на следующем decision; такой вход оплачивается как настоящий переход.","",
              "## Графики","","![Капитал и просадка: stress](capital_drawdown_stress.png)","",
              "Stress-график показывает итог после повышенного модельного drag; просадка измерена по доступным 4h/funding/fill marks, не по неизвестному внутрисвечному пути.","",
              "![Капитал и просадка: normal](capital_drawdown_normal.png)","",
              "Normal — чувствительность исполнения тех же прогнозов, без нового обучения и выбора нового порога.","",
              "## Итог и дальнейшие действия","",
              f'Исследовательский paired+absolute primary/audit gate для AV_linear: {analysis["research_robustness_gate"]}. Это не полный release gate и не prospective evidence. Promotion всегда false.',"",
              "Не выбирать победителя из трёх политик для реальных денег. Если линейный сигнал не подтверждается, остановить расширение архитектуры и разобрать вклад baseline rebalance, funding и повторных входов. Новая настройка будет отдельной зарегистрированной гипотезой. Если добавка подтверждается, сначала заморозить контракт и оценить длительность нового prospective с учётом зависимости дневных результатов; в этой задаче prospective не открыт.","",
              "Семантика, ограниченный бюджет и происхождение попыток зарегистрированы в objective_registry_v10 и registration.json; фактические train cutoffs/indices — folds.json, fit/scaler/model SHA — fit_*.json, непрерывные ledger — replay_*.json. Новый финансовый результат нельзя приписывать старым price-прогнозам.","",
              f'Дополнительных независимых funding/cost-проверок: {analysis["funding_and_cost_checks"]}. Полномочия credentials/orders/paper/live/runtime/network/promotion остаются false.',""]
    with (OUT/"REPORT_RU.md").open("x",encoding="utf-8") as stream: stream.write("\n".join(lines))
    write("receipt.json",{"registration_sha256":reg["sha256"],
          "artifact_pins":{p.name:semantic.pin(p) for p in OUT.iterdir() if p.is_file()},
          "safety":reg["contract"]["safety"]})
    print(json.dumps({"report":str((OUT/"REPORT_RU.md").relative_to(ROOT)),
                      "linear_stress_delta_usdt":round(results["AV_linear_stress"]["delta_net_usdt"],4),"promotion_authority":False}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("synthetic","prepare","reference","labels","dataset","fit","predictions","quality","parity","replay","analyze","charts","report"))
    parser.add_argument("--segment",choices=("train","evaluation"))
    parser.add_argument("--scenario",choices=("normal","stress"))
    parser.add_argument("--policy",choices=("AV_linear","AV_constant","AV_causal_rule"))
    parser.add_argument("--first",type=int,default=0); parser.add_argument("--count",type=int,default=1)
    args=parser.parse_args()
    if args.stage in ("reference","labels"): globals()[args.stage](args.segment)
    elif args.stage=="fit": fit(args.first,args.count)
    elif args.stage=="parity": parity(args.scenario)
    elif args.stage=="replay": replay(args.policy,args.scenario)
    else: globals()[args.stage]()
