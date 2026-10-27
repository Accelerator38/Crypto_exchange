"""One metadata-only launch preflight; no experiments, raw data or fits."""
from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/launch_preflight_v1.py"
SELF = "tools/check_genetic_primus_launch_preflight_v1.py"
DOC = "docs/GENETIC_PRIMUS_LAUNCH_PREFLIGHT_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/synthetic_net_accounting_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/launch_preflight_v1_20261007"
SOURCE = {
    "research": "configs/genetic_primus_research_execution_contract_v1.json",
    "record": "Reports/Exia/Genetic_Primus/forward_preregistration_v1_20261007/record.json",
    "availability": "configs/genetic_primus_ohlcv_availability_contract_v1.json",
    "quality": "Reports/Exia/Genetic_Primus/offline_ohlcv_quality_v1_20261007/result.json",
    "origin": "Reports/Exia/Genetic_Primus/origin_feature_prefix_v1_20261007/result.json",
    "matrix": "Reports/Exia/Genetic_Primus/synthetic_label_matrix_v1_20261007/result.json",
    "net": PREVIOUS + "/result.json",
}
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_synthetic_net_accounting_v1.py"))
old, api, quality = prior["old"], prior["api"], prior["quality"]
pure = runpy.run_path(str(ROOT / PURE))


def build():
    previous = quality["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    quality["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous output identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, DOC, *SOURCE.values()):
        bank[path] = old["identity"](path)
    materials = {role: quality["read"](path) for role, path in SOURCE.items()}
    envelope = materials["record"]
    if api["sha"](envelope["record"]) != envelope["record_semantic_sha256"]:
        raise ValueError("frozen preregistration semantic identity")
    materials["record"] = envelope["record"]
    report = pure["assess"](**materials)
    if sum(r["scope"] == "historical_smoke_protocol_requirement" for r in report["requirements"]) != 22:
        raise ValueError("smoke classification count")
    if sum(r["scope"] == "prospective_or_promotion_not_smoke_prerequisite" for r in report["requirements"]) != 8:
        raise ValueError("promotion classification count")
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden module loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.launch_preflight_result/1", "inputs": bank,
              "source_roles": SOURCE, "report": report, "new_synthetic_experiments": 0,
              "parquet_reads": 0, "scaler_fit_count": 0, "model_fit_count": 0,
              "market_financial_evaluation_count": 0, "safety": report["safety"]}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.launch_preflight_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
               "safety": report["safety"]}
    return result, receipt


def main():
    if sys.argv[1:] not in (["--run"], ["--verify"]):
        raise ValueError("use --run or --verify")
    result, receipt = build()
    if sys.argv[1:] == ["--run"]:
        old["local"](DEST + "/result.json").parent.mkdir(exist_ok=False)
        for name, value in (("result.json", result), ("completion_receipt.json", receipt)):
            old["emit"](DEST + "/" + name, value, api)
        print(f"PREFLIGHT: smoke requirements=22; promotion=8; bridges=4; inputs={len(result['inputs'])}; fit=0")
    else:
        if quality["read"](DEST + "/result.json") != result or quality["read"](DEST + "/completion_receipt.json") != receipt:
            raise ValueError("result/receipt mismatch")
        print("VERIFIED: preflight receipt OK; metadata only; parquet=0; fit=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED: " + type(exc).__name__ + "; no training/network")
        raise SystemExit(1)
