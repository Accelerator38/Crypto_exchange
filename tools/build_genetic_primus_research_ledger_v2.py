"""Append the episode and rolling-linear trials to the immutable known-trials log."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank, object_sha, paired_lcb
from exia.genetic_primus.research_stage_gates_v1 import stage_gates
from exia.genetic_primus.selection_development_v1 import absolute_score, common_bank

PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v1_20261001/ledger.json"
PREVIOUS_SHA = "b83eeaeec48a31e5dda1f57f148b00f4bab3872012196e500ad02a26a34852e3"
EPISODE = ROOT / "Reports/Exia/Genetic_Primus/episode_attribution_v1_20261001/result.json"
EPISODE_CONFIG = ROOT / "configs/genetic_primus_episode_attribution_v1.json"
LINEAR = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_development_v1_20261001/result.json"
LINEAR_CONFIG = ROOT / "configs/genetic_primus_rolling_linear_development_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v2_20261001/ledger.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict:
    if sha(PREVIOUS) != PREVIOUS_SHA:
        raise ValueError("known-trial ledger ancestry changed")
    prior, episode, linear = read(PREVIOUS), read(EPISODE), read(LINEAR)
    econfig, lconfig = read(EPISODE_CONFIG), read(LINEAR_CONFIG)
    if episode["config_sha256"] != sha(EPISODE_CONFIG) or linear["config_sha256"] != sha(LINEAR_CONFIG):
        raise ValueError("registered development contract changed")
    source = read(ROOT / econfig["source_ablation_result_path"])
    bank = block_bank(27, econfig["paired_bootstrap"]["samples"],
                      econfig["paired_bootstrap"]["block_days"],
                      econfig["paired_bootstrap"]["seed"])
    if array_sha(bank) != episode["paired_bank_sha256"]:
        raise ValueError("episode paired bank changed")
    for label in ("normal", "stress"):
        ga = np.asarray(source["policies"]["GA_target_gate"][0]["evaluations"][label]["daily_returns"])
        controls = np.asarray([item["evaluations"][label]["daily_returns"]
                               for item in episode["controls"]])
        if abs(paired_lcb(ga, np.median(controls, axis=0), bank) - episode["paired_lcb"][label]) > 1e-14:
            raise ValueError("episode paired LCB not reproducible")
    train_bank = common_bank(60, lconfig, validation=False)
    if array_sha(train_bank) != linear["train_bank_sha256"]:
        raise ValueError("linear common bank changed")
    for record in linear["records"]:
        normal = np.asarray(record["evaluations"]["normal"]["daily_returns"])
        stress = np.asarray(record["evaluations"]["stress"]["daily_returns"])
        if absolute_score(normal, stress, train_bank) != record["train_score"]:
            raise ValueError("linear train score not reproducible")
        for origin in record["origin_models"]:
            model = origin["model"]
            if object_sha({k: v for k, v in model.items() if k != "model_sha256"}) != model["model_sha256"]:
                raise ValueError("origin model SHA changed")
            end = model["max_train_label_end_exclusive"]
            if end is not None and end > origin["origin_start"] - lconfig["purge_bars"] * 14_400_000:
                raise ValueError("future label reached an origin fit")
            if abs(model["weight_sum"] - 1.0) > 1e-12 or model["negative_outcome_weight"] < 0:
                raise ValueError("invalid outcome weighting")
    state = stage_gates(episode, linear)
    return {
        "schema_version": "exia.genetic_primus.research_ledger/2",
        "previous_ledger_path": PREVIOUS.relative_to(ROOT).as_posix(),
        "previous_ledger_sha256": PREVIOUS_SHA,
        "prior_objective_entry_count": len(prior["registry_objectives"]),
        "prior_later_attempt_count": len(prior["known_later_attempts"]),
        "new_attempts": [
            {"objective_id": econfig["objective_id"],
             "config_path": EPISODE_CONFIG.relative_to(ROOT).as_posix(),
             "config_sha256": sha(EPISODE_CONFIG),
             "result_path": EPISODE.relative_to(ROOT).as_posix(),
             "result_sha256": sha(EPISODE),
             "fixed_seed_count": len(episode["controls"]),
             "candidate_selected": False,
             "evidence_class": episode["evidence_class"]},
            {"objective_id": lconfig["objective_id"],
             "config_path": LINEAR_CONFIG.relative_to(ROOT).as_posix(),
             "config_sha256": sha(LINEAR_CONFIG),
             "result_path": LINEAR.relative_to(ROOT).as_posix(),
             "result_sha256": sha(LINEAR),
             "candidate_evaluations": linear["search_evaluations"],
             "candidate_selected": linear["selected_variant"] is not None,
             "evidence_class": linear["evidence_class"]},
        ],
        "verified_recomputations": {
            "episode_paired_lcb": True, "episode_resample_bank": True,
            "rolling_linear_train_scores": True, "rolling_linear_model_sha_and_purge": True,
        },
        "stage_gates": state,
        "unknown_historical_adaptive_decisions": True,
        "global_dsr_pbo_calibrated": False,
        "market_data_manifest_sha256": prior["market_data"]["manifest_sha256"],
        "promotion_authority": False,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("immutable research ledger v2 exists")
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "KNOWN_TRIALS_APPENDED_NO_PROMOTION",
                      "new_attempts": len(result["new_attempts"]),
                      "ga_search_allowed": result["stage_gates"]["ga_search_allowed_under_current_plan"],
                      "prospective_registered": result["stage_gates"]["prospective_contract_registered"]},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
