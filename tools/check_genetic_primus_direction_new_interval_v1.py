"""Reject old-period results; no promotion authority, even on successful research."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_genetic_primus_direction_selective_v1 import read, identity


def check(result, contract):
    reasons = []
    end = datetime.fromisoformat(contract["end_exclusive_utc"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) < end:
        reasons.append("registered_new_period_not_complete")
    payload = {k: v for k, v in result.items() if k != "sha256"}
    if identity(payload) != result.get("sha256"):
        reasons.append("result_identity_mismatch")
    if result.get("test_start_utc") != contract["start_utc"] or result.get("test_end_exclusive_utc") != contract["end_exclusive_utc"]:
        reasons.append("registered_new_period_mismatch")
    if result.get("historical_only") is not False:
        reasons.append("research_period_not_independent")
    if result.get("registration_sha256") != contract["registration_sha256"]:
        reasons.append("contract_identity_mismatch")
    required = [(m, kind, symbol, coverage) for m in [1, 10, 30, 240] for kind in ["logistic", "boosting"]
                for symbol in ["ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT"]
                for coverage in [1, .5, .2, .1]]
    rows = result.get("rows", [])
    ids = [(r["timeframe_minutes"], r["model"], r["symbol"], r["coverage_target"]) for r in rows]
    if sorted(ids) != sorted(required): reasons.append("missing_or_extra_registered_comparisons")
    positive = []
    for row, row_id in zip(rows, ids):
        adequate = (row.get("n_nonflat_emitted", 0) >= contract["minimum_emitted_nonflat"]
                    and row.get("bootstrap", {}).get("emitted_days", 0) >= contract["minimum_emitted_days"])
        controls = row.get("controls", {})
        significant = set(controls) == {"random_coin", "train_majority", "momentum"} and all(
            c.get(b, {}).get("simultaneous_lcb95") is not None and c[b]["simultaneous_lcb95"] > 0
            for c in controls.values() for b in ["bootstrap", "audit_bootstrap"])
        if adequate and significant: positive.append(row_id)
    return {"new_period_qualified": not reasons, "rejection_reasons": reasons,
            "independent_positive_cells": positive if not reasons else [], "promotion_authority": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / "Reports/Exia/Genetic_Primus/direction_selective_v1_20261013"
    contract_path = base / "new_interval_contract_v2.json"
    if not contract_path.exists(): contract_path = base / "new_interval_contract.json"
    contract, result = read(contract_path), read(args.results)
    now = datetime.now(timezone.utc)
    status = check(result, contract)
    status["as_of_utc"] = now.isoformat()
    status["single_final_analysis_enforced_by_this_checker"] = False
    status["qualification"] = "Structural evidence check; cannot attest data provenance or prevent arbitrary repeated analyses."
    print(json.dumps(status))
