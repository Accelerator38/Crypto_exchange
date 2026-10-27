"""Validate local daily on-chain artifacts; never synthesize the paper's data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus.direction_selective_v1 import validate_onchain
from run_genetic_primus_direction_selective_v1 import digest, read, write

OUT = ROOT / "Reports/Exia/Genetic_Primus/onchain_replication_v1_20261013"


def register():
    if OUT.exists(): raise ValueError("exclusive on-chain plan registration required")
    OUT.mkdir()
    cfg = read(ROOT / "configs/genetic_primus_direction_selective_v1.json")
    result = write(OUT / "readiness.json", {"paper": cfg["onchain"], "safety": cfg["safety"],
        "status": "awaiting_original_data_code_and_publication_vintages", "trained_models": 0,
        "measured_accuracy": None, "published_accuracy_is_not_our_result": True,
        "required_bank": {"format": "NPZ, allow_pickle=false", "times": "UTC completed-day decision times, int64 ms, consecutive",
            "closes": "positive completed-day close at each decision", "features": "finite daily on-chain matrix, original named schema",
            "available_ms": "per-value actual publication timestamp matrix; <= decision time", "feature_names": "unique Unicode names"},
        "required_receipt": ["bank_sha256", "asset=BTC", "supplier=Glassnode", "point_in_time_verified=true",
                             "author_feature_names", "author_original_data_sha256", "author_original_code_sha256"],
        "paper_target": "next daily close > current daily close; paper negative class includes flat",
        "our_direction_target": "up/down/flat separately; never substitute paper binary accuracy for primary minute metric",
        "replication_stages": ["verify author data/code identities and point-in-time vintages",
            "reproduce original chronological 80/20 paper experiment, disclose original target and original preprocessing",
            "independently audit feature selection and hyperparameter tuning boundaries",
            "run frozen causal variant with train-only scaling/selection and purged calibration",
            "test new temporal interval once; compare random/majority/momentum with calendar-block intervals"],
        "architecture": "Boruta + CNN-LSTM; exact implementation requires the author code; no invented substitute",
        "source_url": "https://link.springer.com/article/10.1186/s40854-024-00643-1",
        "source_availability": "Article Availability of data and materials: available from corresponding author on request",
        "engineering_validator_sha256": digest(Path(__file__))})
    print(json.dumps({"status": result["status"], "fits": 0, "accuracy": None}))


def validate(bank_path, receipt_path):
    receipt = read(receipt_path)
    if receipt.get("bank_sha256") != digest(bank_path): raise ValueError("bank hash mismatch")
    if receipt.get("asset") != "BTC" or receipt.get("supplier") != "Glassnode" or receipt.get("point_in_time_verified") is not True:
        raise ValueError("original BTC source and point-in-time receipt required")
    for key in ("author_original_data_sha256", "author_original_code_sha256"):
        value = receipt.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise ValueError("author artifact identity required")
    with np.load(bank_path, allow_pickle=False) as a:
        names = a["feature_names"].tolist()
        if names != receipt.get("author_feature_names"): raise ValueError("original feature schema mismatch")
        labels = validate_onchain(a["times"], a["closes"], a["features"], a["available_ms"], names)
        summary = {"rows": len(a["times"]), "features": len(names), "up": int((labels > 0).sum()),
                   "down": int((labels < 0).sum()), "flat": int((labels == 0).sum()),
                   "bank_sha256": digest(bank_path), "paper_replication_complete": False}
    write(OUT / "data_validation.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["register", "validate"])
    parser.add_argument("--bank", type=Path); parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.stage == "register": register()
    else:
        if args.bank is None or args.receipt is None: parser.error("local --bank and --receipt required")
        validate(args.bank, args.receipt)
