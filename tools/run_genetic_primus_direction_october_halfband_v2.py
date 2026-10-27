"""Metadata-only compatibility amendment. Preserves all frozen v1 artifacts."""
from __future__ import annotations
import run_genetic_primus_direction_october_halfband_v1 as old
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

ROOT = old.ROOT
PRIOR_OUT, PRIOR_CONFIG = old.OUT, old.CONFIG
V1_PINS = old.pins
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_october_halfband_v2_20261014"
CONFIG = ROOT / "configs/genetic_primus_direction_october_halfband_v2.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v26.json"

def config():
    amendment=old.read(CONFIG);c=old.read(PRIOR_CONFIG)
    assert all(amendment[k] is False for k in old.reuse.fixed.SAFETY)
    old.reuse.config()
    before=dict(c["boost"]);c["boost"]["maximum_leaf_values"]=amendment["maximum_leaf_values"]
    assert {k:v for k,v in c["boost"].items() if k!="maximum_leaf_values"}==before
    c.update(objective_id=amendment["objective_id"],previous_registry=amendment["previous_registry"])
    return c

def pins():
    result=V1_PINS()
    for p in (CONFIG,Path(__file__)):result[old.rel(p)]=old.digest(p)
    return result

def load(h):
    return {k:np.load(PRIOR_OUT/"data"/("tf_%sm"%h)/(k+".npy"),mmap_mode="r",allow_pickle=False) for k in ("times","closes","features")}

def registered():
    c=config();r=old.read(OUT/"registration.json")
    assert r["code_pins"]==pins() and r["versions"]==old.reuse.versions() and r["contract"]==c
    return c,r

def init():
    c=config();previous_reg=old.read(PRIOR_OUT/"registration.json")
    assert previous_reg["code_pins"]==V1_PINS()
    models=list(PRIOR_OUT.glob("origin_*/*.joblib"));assert len(models)==1
    original=models[0];assert original.name=="tf_1m__original_logit.joblib"
    source_meta=old.read(original.with_suffix(".json"));assert old.digest(original)==source_meta["model_sha256"]
    bank=load(1);train,cal=old.reuse.fixed.fold(bank["times"],c["test_start_ms"],1,c,c["seed"]+1000)
    assert old.identity(bank["times"][train].tolist())==source_meta["fold"]["train_times_sha256"]
    assert old.identity(bank["times"][cal].tolist())==source_meta["fold"]["calibration_times_sha256"]
    OUT.mkdir();(OUT/"origin_00").mkdir();destination=OUT/"origin_00"/original.name
    shutil.copyfile(original,destination);assert old.digest(destination)==source_meta["model_sha256"]
    derived={k:v for k,v in source_meta.items() if k!="sha256"}
    derived.update(code_pins=pins(),imported_source_model=old.rel(original),imported_source_meta_sha256=old.digest(original.with_suffix(".json")))
    old.write(destination.with_suffix(".json"),derived)
    failure=dict(reason="boost.fit completed, then metadata lookup maximum_leaf_values raised KeyError before persistence",
        persisted_fits=1,not_persisted_fits=1,original_contract_sha256=old.digest(PRIOR_CONFIG),amendment_sha256=old.digest(CONFIG))
    old.write(PRIOR_OUT/"metadata_failure_disclosure.json",failure)
    data_pins=dict(previous_reg["data_pins"])
    for p in (PRIOR_OUT/"registration.json",original,original.with_suffix(".json"),PRIOR_OUT/"metadata_failure_disclosure.json"):
        data_pins[old.rel(p)]=old.digest(p)
    old.write(OUT/"registration.json",dict(contract=c,amendment=old.read(CONFIG),code_pins=pins(),versions=old.reuse.versions(),data_pins=data_pins,
        pure_checks=old.direct_checks(),successful_model_artifact_budget=80,total_executed_fit_budget=81,imported_model_fits=1,
        remaining_fit_budget=79,prospective=False,independent_final_validation=False,**{k:False for k in old.reuse.fixed.SAFETY}))
    previous_path=ROOT/c["previous_registry"];previous=old.read(previous_path);previous.pop("sha256",None)
    previous.update(registry_id=REGISTRY.stem,previous_registry_path=c["previous_registry"],previous_registry_sha256=old.digest(previous_path))
    previous["objectives"].append(dict(objective_id=c["objective_id"],family="metadata_only_amendment_no_hyperparameter_change",
        registration_path=old.rel(OUT/"registration.json"),registration_sha256=old.digest(OUT/"registration.json"),historical_fit_budget=79,
        imported_model_fits=1,prior_unpersisted_fits=1,total_executed_fit_budget=81,prospective=False,independent_validation=False,promotion_authority=False))
    old.write(REGISTRY,previous)
    print(json.dumps(dict(registry="v26",imported_models=1,new_remaining_fits=79,total_fit_accounting=81)))

# Only the adapter bindings change; v1 source, configuration and data remain immutable.
old.OUT=OUT;old.CONFIG=CONFIG;old.REGISTRY=REGISTRY
old.config=config;old.pins=pins;old.load=load;old.registered=registered

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["checks","init","origin","score","finalize","audit"])
    parser.add_argument("--minutes",type=int);parser.add_argument("--number",type=int);args=parser.parse_args()
    if args.stage=="checks":print(json.dumps(dict(pure_checks=old.direct_checks(),estimator_fits=0)))
    elif args.stage=="init":init()
    elif args.stage=="origin":old.origin(args.number)
    elif args.stage in ("score","audit"):getattr(old,args.stage)(args.minutes)
    else:
        old.finalize()
        old.write(OUT/"execution_accounting.json",dict(successful_model_fits=80,imported_successful_fit=1,new_successful_fits=79,
            failed_persistence_fit=1,total_executed_fits_across_v1_v2=81,hyperparameters_changed=False))
