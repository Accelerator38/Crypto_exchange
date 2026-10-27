"""Warm-start-only ablation; reuses frozen cold-start measurement helpers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tools import run_genetic_primus_retrained_august_v1 as cold
from exia.genetic_primus.measurement_lab_v2 import (
    ALLOW_ALL, NO_TRADE, CATALOG, LabScaler, array_sha, daily_returns, object_sha, rank_key,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus_statistics.hac_adapter_v1 import assess_paired_daily

CONFIG = ROOT / "configs/genetic_primus_warmstart_august_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v8.json"
ANCHORS = (NO_TRADE, ALLOW_ALL, cold.FIXED)


def anchored_search(evaluate, budget, seed, population_size=6, anchors=ANCHORS):
    """Same genetic operator sequence as cold search; only initial pending differs."""
    if not 2 <= budget <= len(CATALOG) or not 2 <= population_size <= budget:
        raise ValueError("invalid search budget/population")
    if len(set(anchors)) != len(anchors) or not 2 <= len(anchors) <= population_size:
        raise ValueError("invalid or duplicate anchors")
    if any(g not in CATALOG for g in anchors):
        raise ValueError("anchor outside sparse catalog")
    rng = np.random.default_rng(seed)
    remaining = list(CATALOG[2:])
    selected = [remaining[int(i)] for i in rng.permutation(len(remaining))]
    records = {}
    pending = list(anchors)
    generation_sizes = []
    while len(records) < budget:
        if pending:
            genome = pending.pop(0)
        elif len(records) < population_size:
            genome = next(g for g in selected if g not in records)
        else:
            if not generation_sizes or len(records) >= sum(generation_sizes):
                parents = sorted(records, key=lambda g: rank_key(g, records[g]), reverse=True)[:population_size]
                batch_size = min(population_size, budget - len(records))
                generation_sizes = [len(records), batch_size]
                pending = []
                for _ in range(batch_size):
                    child = None
                    for _attempt in range(32):
                        first, second = (parents[int(rng.integers(len(parents)))] for _ in range(2))
                        raw = np.where(rng.random(5) < 0.5, first, second).astype(int)
                        raw[int(rng.integers(5))] = int(rng.choice([-1, 0, 1]))
                        trial = tuple(int(x) for x in raw)
                        if trial in CATALOG and trial not in records and trial not in pending:
                            child = trial
                            break
                    if child is None:
                        unseen = [g for g in CATALOG if g not in records and g not in pending]
                        child = unseen[int(rng.integers(len(unseen)))]
                    pending.append(child)
            genome = pending.pop(0)
        value = float(evaluate(genome))
        if genome in records or not np.isfinite(value):
            raise ValueError("duplicate evaluation or nonfinite fitness")
        records[genome] = value
    winner = max(records, key=lambda g: rank_key(g, records[g]))
    # Same schema permits exact parity checking when the extra anchor is omitted.
    return {"method": "genetic_search", "winner": winner, "score": records[winner],
            "unique_evaluations": len(records), "seed": seed,
            "evaluations": [{"genome": g, "fitness_lcb": score} for g, score in records.items()]}


def load_config():
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    r = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert c["data_already_revealed"] is True and not any(c["safety"].values())
    assert c["initial_anchors"] == [list(g) for g in ANCHORS]
    assert c["unique_candidates_per_origin"] == 24 and c["origins"] == 10
    assert c["anchor_evaluations_count_toward_budget"] is True and not c["anchor_is_forced_winner"]
    for name in ("base_config", "base_completion", "base_runner"):
        assert file_sha256(ROOT / c[name + "_path"]) == c[name + "_sha256"]
    assert r["objectives"][0]["config_sha256"] == file_sha256(CONFIG)
    assert file_sha256(ROOT / r["previous_registry_path"]) == r["previous_registry_sha256"]
    base, base_output = cold.load_config()
    cold.verify_pre_run(base, base_output)
    output = (ROOT / c["output"]).resolve()
    assert output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus")
    assert output != base_output
    return c, base, output, base_output


def fingerprint():
    return {"config_sha256": file_sha256(CONFIG), "registry_sha256": file_sha256(REGISTRY),
            "warm_runner_sha256": file_sha256(Path(__file__)), "cold": cold.fingerprint()}


def verified_base_artifact(c, name):
    completion_path = ROOT / c["base_completion_path"]
    completion = json.loads(completion_path.read_text())
    path = completion_path.parent / name
    assert file_sha256(path) == completion["artifact_sha256"][name]
    return json.loads(path.read_text())


def verify_receipt(output):
    receipt = json.loads((output / "pre_run_receipt.json").read_text())
    if receipt["fingerprint"] != fingerprint():
        raise ValueError("source/config changed after preparation")
    return receipt


def prepare(_):
    c, base, output, _ = load_config()
    if (output / "pre_run_receipt.json").exists():
        raise FileExistsError("already prepared")
    manifest, _ = cold.panel_and_manifest(base)
    for i in range(1, 11):
        for suffix in (".json", "_selected.json", "_screen.json"):
            verified_base_artifact(c, f"origin_{i:02d}" + suffix)
    cold.save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(),
        "bounds": [cold.bounds(base, i) for i in range(10)],
        "snapshot_sha256": manifest["snapshot_sha256"], "source_files": manifest["snapshot_files"],
        "base_completion_sha256": c["base_completion_sha256"], "initial_anchors": c["initial_anchors"],
        "search_budget_total": 240, "including_cold_search_budget": 480,
        "data_already_revealed": True, "safety": c["safety"],
    })
    print(json.dumps({"status": "PREPARED", "search_budget": 240,
                      "receipt_sha256": file_sha256(output / "pre_run_receipt.json")}))


def evaluate(args):
    c, base, output, _ = load_config()
    receipt = verify_receipt(output)
    b = cold.bounds(base, args.origin)
    stem = f"origin_{args.origin + 1:02d}"
    if (output / (stem + "_started.json")).exists() or (output / (stem + ".json")).exists():
        raise FileExistsError("origin already attempted")
    manifest, panel = cold.panel_and_manifest(base)
    assert manifest["snapshot_sha256"] == receipt["snapshot_sha256"]
    cold.save_new(output / (stem + "_started.json"), {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "bounds": b,
        "pre_run_sha256": file_sha256(output / "pre_run_receipt.json"), "origin_index": args.origin,
    })
    symbols = tuple(manifest["metadata"]["symbols"])
    dt = base["timeframe_minutes"] * 60_000
    train = cold.window(panel, symbols, b["train_start"], b["train_end"], dt)
    scaler = LabScaler.fit(train.values)
    old_selection = verified_base_artifact(c, stem + "_selected.json")
    assert object_sha(scaler.mapping()) == object_sha(old_selection["scaler"])
    assert array_sha(train.values) == old_selection["train_values_sha256"]
    reference = train.simulate(ALLOW_ALL, scaler, 16.0)
    reference_daily = daily_returns(reference)
    training = {}

    def score(genome):
        sim = reference if genome == ALLOW_ALL else train.simulate(genome, scaler, 16.0)
        hac = cold.assessment(sim, reference, b["train_start"])
        value = cold.fitness_value(hac, daily_returns(sim) - reference_daily, base)
        training[str(genome)] = {"metrics": sim.metrics, "hac": hac, "fitness": value}
        return value

    selection = anchored_search(score, 24, base["search"]["seed_base"] + args.origin, 6)
    shared = 0
    for key, value in training.items():
        if key in old_selection["training"]:
            assert value == old_selection["training"][key]
            shared += 1
    winner = tuple(selection["winner"])
    anchor_score = training[str(cold.FIXED)]["fitness"]
    selected_path = output / (stem + "_selected.json")
    cold.save_new(selected_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "bounds": b,
        "search": selection, "initial_anchors": c["initial_anchors"], "training": training,
        "scaler": scaler.mapping(), "train_values_sha256": array_sha(train.values),
        "model": train.model(winner, scaler), "model_sha256": object_sha(train.model(winner, scaler)),
        "anchor_fitness": anchor_score, "shared_candidate_training_parity_count": shared,
    })
    val = cold.window(panel, symbols, b["validation_start"], b["validation_end"], dt)
    validation = {}
    for cost, bps in (("normal", 16.0), ("stress", 24.0)):
        raw = val.simulate(winner, scaler, bps)
        validation[cost] = {"metrics": raw.metrics, "daily_returns": daily_returns(raw).tolist(),
                            "hac": {name: cold.assessment(raw, val.simulate(g, scaler, bps), b["validation_start"])
                                    for name, g in (("EMA", ALLOW_ALL), ("NoTrade", NO_TRADE))}}
    decision = cold.screen(validation, base)
    active = winner if decision["passed"] else NO_TRADE
    screen_path = output / (stem + "_screen.json")
    cold.save_new(screen_path, {"selected_sha256": file_sha256(selected_path), "validation": validation,
                               "screen": decision, "active_genome": active,
                               "created_at_utc": datetime.now(timezone.utc).isoformat()})
    deploy = cold.window(panel, symbols, b["test_start"], b["test_end"], dt)
    genomes = dict(zip(cold.CANDIDATES, (NO_TRADE, ALLOW_ALL, cold.FIXED, winner, active)))
    evaluations = {}
    for cost, bps in (("normal", 16.0), ("stress", 24.0)):
        cache = {g: cold.payload(deploy.simulate(g, scaler, bps, details=True), deploy.model(g, scaler),
                                 b, manifest["snapshot_sha256"], scaler.fit_values_sha256)
                 for g in set(genomes.values())}
        evaluations[cost] = {name: cache[g] for name, g in genomes.items()}
    prior = verified_base_artifact(c, stem + ".json")
    for cost in evaluations:
        for name in ("NoTrade", "EMA", "FixedOverlaySameScaler"):
            assert object_sha(evaluations[cost][name]) == object_sha(prior["evaluations"][cost][name])
    result = {"schema": "exia.genetic_primus.warmstart_origin/1", "origin_index": args.origin,
              "bounds": b, "selected_sha256": file_sha256(selected_path),
              "screen_sha256": file_sha256(screen_path), "winner": winner, "screen": decision,
              "evaluations": evaluations, "scaler": scaler.mapping(), "anchor_fitness": anchor_score,
              "shared_candidate_training_parity_count": shared,
              "config_sha256": file_sha256(CONFIG), "data_already_revealed": True,
              "search_evaluations": 24, "safety": c["safety"]}
    path = output / (stem + ".json")
    cold.save_new(path, result)
    print(json.dumps({"origin": args.origin + 1, "winner": winner, "screen": decision["passed"],
                      "anchor_train_LCB_bps": round(anchor_score * 10000, 4),
                      "warm_return_pct": round(evaluations["normal"]["RetrainedRaw"]["metrics"]["net_return"] * 100, 4),
                      "cold_return_pct": round(prior["evaluations"]["normal"]["RetrainedRaw"]["metrics"]["net_return"] * 100, 4),
                      "result_sha256": file_sha256(path)}))


def finalize(_):
    c, base, output, _ = load_config()
    verify_receipt(output)
    results, priors = [], []
    for i in range(10):
        stem = f"origin_{i + 1:02d}"
        r = json.loads((output / (stem + ".json")).read_text())
        selected = json.loads((output / (stem + "_selected.json")).read_text())
        assert r["bounds"] == cold.bounds(base, i) and r["search_evaluations"] == 24
        assert r["selected_sha256"] == file_sha256(output / (stem + "_selected.json"))
        assert r["screen_sha256"] == file_sha256(output / (stem + "_screen.json"))
        genomes = [tuple(x["genome"]) for x in selected["search"]["evaluations"]]
        assert len(set(genomes)) == 24 and all(genomes.count(g) == 1 for g in ANCHORS)
        assert not any(r["safety"].values()) and r["config_sha256"] == file_sha256(CONFIG)
        results.append(r)
        priors.append(verified_base_artifact(c, stem + ".json"))
    times = cold.millis(base["test_start_utc"]) + np.arange(1, 31, dtype=np.int64) * cold.DAY_MS
    summary = {"origins": 10, "search_evaluations": 240, "warm_plus_cold_evaluations": 480,
               "screen_passes": sum(r["screen"]["passed"] for r in results),
               "same_winner_as_cold_origins": sum(r["winner"] == p["winner"] for r, p in zip(results, priors)),
               "data_already_revealed": True, "promotion_authority": False, "costs": {}}
    for cost in ("normal", "stress"):
        candidates = {name: cold.aggregate(results, cost, name) for name in cold.CANDIDATES}
        candidates["ColdStartRaw"] = cold.aggregate(priors, cost, "RetrainedRaw")
        candidates["ColdStartScreened"] = cold.aggregate(priors, cost, "RetrainedScreened")
        raw = np.array(candidates["RetrainedRaw"]["daily_returns"])
        contrasts = {name: assess_paired_daily(raw, np.array(candidates[name]["daily_returns"]), times).mapping()
                     for name in ("ColdStartRaw", "FixedOverlaySameScaler", "EMA", "NoTrade")}
        summary["costs"][cost] = {"candidates": candidates, "warm_raw_hac_vs": contrasts}
    cold.save_new(output / "summary.json", summary)
    paths = [output / "pre_run_receipt.json", output / "summary.json"]
    paths += [output / (f"origin_{i:02d}" + suffix) for i in range(1, 11)
              for suffix in ("_started.json", "_selected.json", "_screen.json", ".json")]
    cold.save_new(output / "completion.json", {"fingerprint": fingerprint(), "artifact_count": len(paths),
        "artifact_sha256": {p.name: file_sha256(p) for p in paths},
        "status": "REVEALED_WARMSTART_COMPLETE_NO_PROMOTION", "safety": c["safety"]})
    print(json.dumps({"status": "COMPLETE", "origins": 10, "screen_passes": summary["screen_passes"],
                      "same_winners_as_cold": summary["same_winner_as_cold_origins"], "artifacts": len(paths)}))


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(required=True)
    for name, handler in (("prepare", prepare), ("origin", evaluate), ("finalize", finalize)):
        p = subs.add_parser(name)
        if name == "origin":
            p.add_argument("--origin", type=int, required=True)
        p.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
