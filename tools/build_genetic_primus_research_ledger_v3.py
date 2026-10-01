"""Append the pre-registered, structurally inconclusive episode diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v2_20261001/ledger.json"
PREVIOUS_SHA = "c0f4652ae3e28d21dc1f2a01e7c7dacca536e0cf97afb0e770286756ea2e8186"
CONFIG = ROOT / "configs/genetic_primus_episode_attribution_v2.json"
RESULT = ROOT / "Reports/Exia/Genetic_Primus/episode_attribution_v2_20261001/result.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v3_20261001/ledger.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    if sha(PREVIOUS) != PREVIOUS_SHA:
        raise ValueError("previous ledger bytes changed")
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if (config["objective_id"] != result["objective_id"]
            or result["config_sha256"] != sha(CONFIG)
            or result["structural_hamming_max"] != 2 * (
                result["eligible_bars"] - result["ga_active_bars"])
            or result["structural_hamming_max"] >= config["minimum_changed_decision_rows_per_seed"]
            or result["matched_pnl_evaluated"]
            or result["next_search_gate_passed"]
            or result["paired_lcb"]):
        raise ValueError("structural attribution diagnostic changed")
    gates = dict(prior["stage_gates"])
    if any(gates.values()):
        raise ValueError("research gates unexpectedly opened")
    return {
        "schema_version": "exia.genetic_primus.research_ledger/3",
        "previous_ledger_path": PREVIOUS.relative_to(ROOT).as_posix(),
        "previous_ledger_sha256": PREVIOUS_SHA,
        "new_attempts": [{
            "objective_id": config["objective_id"],
            "config_path": CONFIG.relative_to(ROOT).as_posix(),
            "config_sha256": sha(CONFIG),
            "result_path": RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": sha(RESULT),
            "evidence_class": result["evidence_class"],
            "outcome": "structural_nonidentifiability_no_pnl_evaluated",
            "candidate_selected": False,
        }],
        "verified_recomputations": {
            "structural_hamming_bound": True,
            "no_pnl_on_insufficient_diversity": True,
        },
        "stage_gates": gates,
        "unknown_historical_adaptive_decisions": prior["unknown_historical_adaptive_decisions"],
        "global_dsr_pbo_calibrated": prior["global_dsr_pbo_calibrated"],
        "promotion_authority": False,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("immutable research ledger v3 exists")
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "INCONCLUSIVE_ATTRIBUTION_APPENDED",
                      "attempts": len(result["new_attempts"]),
                      "promotion": result["promotion_authority"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
