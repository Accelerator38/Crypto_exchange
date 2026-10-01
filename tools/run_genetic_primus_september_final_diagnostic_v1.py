"""Honest post-period diagnostic for origin 10 after prospective manifest cutoff.

The frozen manifest builder rejects origin 10 after October 1. This tool records
that fact, labels the calculation as engineering diagnostic, and uses the same
offline evaluator and unchanged numerical contract fields. No promotion.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.prospective_contract_v1 import validate_contract
from exia.genetic_primus.prospective_evaluator_v1 import evaluate_panel_origin, origin_preflight
from exia.genetic_primus.prospective_manifest_v1 import (
    build_manifest, load_verified_snapshot, save_manifest_exclusive)

CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
CONTRACT_SHA = "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f"
OUT = ROOT / "Reports/Exia/Genetic_Primus/september_final_diagnostic_20261001"
CONFIG = OUT / "diagnostic_contract.json"
MANIFEST = OUT / "data_manifest_origin_10_diagnostic.json"
RESULT = OUT / "origin_10_diagnostic.json"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
FINAL_FOUR = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930/derived_v1/origin_10_240m.parquet"
SEAL = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_final_20260928_20260930/dataset_seal_v1.json"
FLAGS = {k: False for k in ("credentials_allowed", "orders_enabled", "paper_allowed",
         "live_allowed", "promotion_authority", "runtime_authority", "network_allowed")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def base():
    require(Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve(), "wrong Python")
    require(sha(CONTRACT) == CONTRACT_SHA, "frozen contract changed")
    c = read(CONTRACT)
    validate_contract(c)
    require(c["safety"] == FLAGS, "safety flags changed")
    require(read(SEAL)["seal_sha256"] == "a7056c775677cb1a4bdf928247102d2d3a5a7b37f4950e399c5611c0425493ac", "final OHLCV seal changed")
    return c


def diagnostic_contract():
    c = base()
    c["data_already_revealed"] = True
    c["selection_use"] = "engineering_diagnostic_only_not_validation_not_tuning_not_promotion"
    c["diagnostic_reason"] = "origin 10 completed at outer interval end; prospective manifest cutoff makes pre-commit impossible"
    return c


def verified_config():
    c = diagnostic_contract()
    require(read(CONFIG) == c, "diagnostic contract changed")
    return c


def prepare():
    c = diagnostic_contract()
    save(CONFIG, c)
    return {"status": "DIAGNOSTIC_CONTRACT_CREATED", "contract_sha256": sha(CONFIG),
            "prospective_validation": False}


def manifest():
    c = verified_config()
    prior = read(PREVIOUS)
    paths = [ROOT / item["path"] for item in prior["snapshot_files"]] + [FINAL_FOUR]
    require(prior["snapshot_sha256"] == "f548b6af6034b3d59e0c3c64aad287ca2a9b762bca5051331db2e72406861bda", "origin09 snapshot changed")
    m = build_manifest(paths, root=ROOT, contract_path=CONFIG, contract=c,
                       created_at=datetime.now(timezone.utc))
    require(m["data_already_revealed"] is True and m["safety"] == FLAGS, "diagnostic manifest label")
    save_manifest_exclusive(MANIFEST, m)
    return {"status": "DIAGNOSTIC_MANIFEST_COMMITTED", "manifest_sha256": sha(MANIFEST),
            "snapshot_sha256": m["snapshot_sha256"], "prospective_validation": False}


def verified():
    c = verified_config()
    m, panel = load_verified_snapshot(MANIFEST, root=ROOT, contract_path=CONFIG, contract=c)
    require(m["data_already_revealed"] is True and m["safety"] == FLAGS, "diagnostic manifest label")
    a, b = origin_preflight(c, m, 9, now=datetime.now(timezone.utc))
    require(a == 1790553600000 and b == 1790812800000, "origin bounds")
    return c, m, panel


def preflight():
    _, m, panel = verified()
    return {"status": "DIAGNOSTIC_PREFLIGHT_PASS", "rows": len(panel),
            "snapshot_sha256": m["snapshot_sha256"], "prospective_validation": False}


def evaluate():
    require(not RESULT.exists() and not (OUT / "origin_10_diagnostic_started.json").exists(), "diagnostic already attempted")
    c, m, panel = verified()
    save(OUT / "origin_10_diagnostic_started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
         "manifest_sha256": sha(MANIFEST), "prospective_validation": False})
    result = evaluate_panel_origin(panel, c, m, 9, now=datetime.now(timezone.utc))
    result.update({"contract_sha256": sha(CONFIG), "base_contract_sha256": CONTRACT_SHA,
                   "manifest_sha256": sha(MANIFEST), "diagnostic_only": True,
                   "prospective_validation": False, "promotion_eligible": False,
                   "diagnostic_reason": c["diagnostic_reason"]})
    require(result["search_evaluations"] == 0 and result["safety"] == FLAGS, "search/safety")
    save(RESULT, result)
    return {"status": "DIAGNOSTIC_EVALUATED_NO_PROMOTION", "result_sha256": sha(RESULT),
            "search_evaluations": 0, "prospective_validation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "manifest", "preflight", "evaluate"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), sort_keys=True))


if __name__ == "__main__":
    main()
