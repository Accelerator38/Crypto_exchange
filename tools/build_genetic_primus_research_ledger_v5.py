"""Append the disclosed maximum-diversity attribution attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v4_20261002/ledger.json"
PREVIOUS_SHA = "267370a4e775beff3794802ca3b5f686800a69c957422cbb34ea15d7670670b0"
CONFIG = ROOT / "configs/genetic_primus_episode_attribution_v3.json"
RESULT = ROOT / "Reports/Exia/Genetic_Primus/episode_attribution_v3_20261002/result.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v5_20261002/ledger.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    if sha(PREVIOUS) != PREVIOUS_SHA:
        raise ValueError("previous ledger bytes changed")
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if (result["objective_id"] != config["objective_id"]
            or result["config_sha256"] != sha(CONFIG)
            or result["source_ablation_result_sha256"] != config["source_ablation_result_sha256"]
            or result["source_baseline_audit_sha256"] != config["source_baseline_audit_sha256"]
            or result["maximum_changed_decision_rows"]
            != config["structural_maximum_changed_decision_rows"]
            or result["seeds"] != config["seeds"]
            or len(result["controls"]) != len(config["seeds"])
            or not result["matched_pnl_evaluated"]
            or not result["all_controls_match"]
            or result["historical_support_gate"]
            or result["promotion_authority"]
            or any(result["safety"].values())):
        raise ValueError("attribution lineage or gate changed")
    if (result["distinct_control_targets"] < config["minimum_distinct_control_schedules"]
            or any(row["conservation"]["changed_rows"]
                   != config["structural_maximum_changed_decision_rows"]
                   for row in result["controls"])
            or any(not evaluation["economic_match"] for row in result["controls"]
                   for evaluation in row["evaluations"].values())):
        raise ValueError("matched control geometry changed")
    gates = dict(prior["stage_gates"])
    if any(gates.values()):
        raise ValueError("historical diagnostic cannot open research gates")
    return {
        "schema_version": "exia.genetic_primus.research_ledger/5",
        "previous_ledger_path": PREVIOUS.relative_to(ROOT).as_posix(),
        "previous_ledger_sha256": PREVIOUS_SHA,
        "new_attempts": [{
            "objective_id": config["objective_id"],
            "config_path": CONFIG.relative_to(ROOT).as_posix(),
            "config_sha256": sha(CONFIG),
            "result_path": RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": sha(RESULT),
            "evidence_class": result["evidence_class"],
            "candidate_selected": False,
            "outcome": "matched_historical_controls_do_not_support_GA",
        }],
        "verified_recomputations": {
            "maximum_outcome_blind_episode_diversity": True,
            "per_symbol_episode_exposure_and_trade_count": True,
            "cash_turnover_and_exposure_tolerances": True,
            "fixed_control_and_bootstrap_banks": True,
        },
        "stage_gates": gates,
        "unknown_historical_adaptive_decisions": prior["unknown_historical_adaptive_decisions"],
        "global_dsr_pbo_calibrated": prior["global_dsr_pbo_calibrated"],
        "promotion_authority": False,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("immutable research ledger v5 exists")
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "ATTRIBUTION_ATTEMPT_APPENDED",
                      "support": False, "promotion": result["promotion_authority"]},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
