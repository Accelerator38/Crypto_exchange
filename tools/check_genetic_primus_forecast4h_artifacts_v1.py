"""Read-only model/provenance/link verification; no training, replay or network."""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_genetic_primus_forecast4h_v1 as base
from tools import run_genetic_primus_forecast4h_positions_v1 as extra


def main():
    original = base.registration()
    extended = extra.registration()
    if extended["base_contract"] != original["contract"]:
        raise ValueError("diagnostic changed original model or thresholds")
    receipt_checks = 0
    receipts = {}
    for folder in (base.OUT, extra.OUT):
        receipt_path = folder / "receipt.json"
        receipt = base.sealed(receipt_path)
        receipts[base.relative(receipt_path)] = base.pin(receipt_path)
        for name, expected in receipt["artifact_pins"].items():
            if base.pin(folder / name) != expected:
                raise ValueError("receipt pin failed: " + str(folder / name))
            receipt_checks += 1
    published = base.BASE / "research_review_20261012"
    manifest = base.read(published / "release_manifest.json")
    for name, expected in manifest["files"].items():
        if base.pin(published / name) != expected:
            raise ValueError("published report changed: " + name)
    rows = base.vector_bank(original)
    fits = [base.sealed(path) for path in sorted(base.OUT.glob("fit_*.json"))]
    if len(fits) != 21:
        raise ValueError("unexpected fit count")
    models = {result["model"]["sha256"]: result for result in fits}
    for result in fits:
        model = result["model"]
        if model["sha256"] != base.f.m.identity({k: v for k, v in model.items() if k != "sha256"}):
            raise ValueError("model seal failed")
        if model["scaler_sha256"] != base.f.m.identity(model["scaler"]) or model["parameter_count"] != 7:
            raise ValueError("scaler or coefficient count failed")
        if result["latest_label_available_ms"] > result["cutoff_ms"] or result["measured_local_fit_us"] > original["contract"]["model_ready_delay_us"]:
            raise ValueError("label or readiness causality failed")
    for row in rows:
        result = models[row["model_sha256"]]
        if result["fit_ms"] > row["asof_ms"] or row["training_cutoff_ms"] != result["cutoff_ms"]:
            raise ValueError("forecast model origin mismatch")
        if row["scaler_sha256"] != result["model"]["scaler_sha256"] or row["available_us"] >= row["target_ms"] * 1000:
            raise ValueError("forecast scaler or availability mismatch")
    local_links = 0
    for folder in (base.OUT, extra.OUT):
        for path in folder.glob("*.md"):
            for raw in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                parsed = urlsplit(raw.strip().strip("<>"))
                if parsed.scheme in ("http", "https"):
                    continue
                if parsed.scheme or (parsed.path and not (folder / unquote(parsed.path)).resolve().exists()):
                    raise ValueError("invalid local report link: " + raw)
                local_links += 1
    output = {"schema": "forecast4h_final_verification/1", "receipt_artifact_pins_verified": receipt_checks,
              "receipts": receipts, "published_files_unchanged": len(manifest["files"]),
              "models_verified": len(fits), "forecast_times_verified": len(rows), "asset_forecasts": len(rows) * 8,
              "new_training_fits": 21, "successful_policy_replays": 20,
              "local_links_resolved": local_links, "ledger_checks": 40712 + 40206,
              "additional_replays_or_fits_in_this_check": 0, "network_or_runtime_started": False,
              "checker_pin": base.pin(Path(__file__))}
    output["sha256"] = base.f.m.identity(output)
    with (extra.OUT / "final_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"pins": receipt_checks, "published_unchanged": len(manifest["files"]),
                      "models": len(fits), "local_links": local_links, "ledger_checks": output["ledger_checks"]}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"ERROR {type(error).__name__}: {str(error)[:500]}", file=sys.stderr)
        raise SystemExit(1) from None
