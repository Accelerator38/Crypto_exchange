"""Bounded historical GA retraining; local data, sequential origin commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from exia.genetic_primus.measurement_lab_v2 import (
    ALLOW_ALL, NO_TRADE, DAY_MS, FEATURES, LabScaler, PreparedWindow,
    array_sha, build_lab_features, daily_returns, object_sha, search,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus_statistics.hac_adapter_v1 import assess_paired_daily

CONFIG = ROOT / "configs/genetic_primus_retrained_august_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v7.json"
CANDIDATES = ("NoTrade", "EMA", "FixedOverlaySameScaler", "RetrainedRaw", "RetrainedScreened")
FIXED = (0, 0, 1, 1, -1)
SOURCES = (
    "tools/run_genetic_primus_retrained_august_v1.py",
    "src/exia/genetic_primus/measurement_lab_v2.py",
    "src/exia/genetic_primus/accounting_v2.py",
    "src/exia/genetic_primus/features.py",
    "src/exia/genetic_primus/contracts.py",
    "src/exia/genetic_primus/prospective_manifest_v1.py",
    "src/exia/genetic_primus/prospective_contract_v1.py",
    "src/exia/genetic_primus_statistics/hac_adapter_v1.py",
)


def millis(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("timezone required")
    return int(date.timestamp() * 1000)


def bounds(config, index):
    if type(index) is not int or index not in range(config["origins"]):
        raise ValueError("unregistered origin")
    start = millis(config["test_start_utc"]) + index * config["retrain_cadence_days"] * DAY_MS
    end = start + config["deployment_days"] * DAY_MS
    gap = config["purge_bars"] * config["timeframe_minutes"] * 60_000
    validation_end = start - gap
    validation_start = validation_end - config["validation_days"] * DAY_MS
    train_end = validation_start - gap
    return {"train_start": train_end - config["train_days"] * DAY_MS,
            "train_end": train_end, "validation_start": validation_start,
            "validation_end": validation_end, "test_start": start, "test_end": end}


def load_config():
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    r = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert c["data_already_revealed"] is True and not any(c["safety"].values())
    assert c["train_days"] == 365 and c["validation_days"] == 30 and c["purge_bars"] == 7
    assert c["deployment_days"] == c["retrain_cadence_days"] == 3 and c["origins"] == 10
    assert c["search"]["unique_candidates_per_origin"] == 24
    assert bounds(c, 9)["test_end"] == millis(c["test_end_exclusive_utc"])
    assert r["objectives"][0]["config_sha256"] == file_sha256(CONFIG)
    assert file_sha256(ROOT / r["previous_registry_path"]) == r["previous_registry_sha256"]
    for field in ("source_config", "source_manifest", "source_completion"):
        assert file_sha256(ROOT / c[field]) == c[field + "_sha256"]
    output = (ROOT / c["output"]).resolve()
    assert output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus")
    return c, output


def fingerprint():
    return {"config_sha256": file_sha256(CONFIG), "registry_sha256": file_sha256(REGISTRY),
            "source_sha256": {p: file_sha256(ROOT / p) for p in SOURCES}}


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def panel_and_manifest(c):
    source_config = json.loads((ROOT / c["source_config"]).read_text(encoding="utf-8"))
    return load_verified_snapshot(ROOT / c["source_manifest"], root=ROOT,
                                  contract_path=ROOT / c["source_config"], contract=source_config)


def verify_pre_run(c, output):
    receipt = json.loads((output / "pre_run_receipt.json").read_text())
    if receipt["fingerprint"] != fingerprint():
        raise ValueError("code/config changed since preregistration")
    return receipt


def prepare(_):
    c, output = load_config()
    if (output / "pre_run_receipt.json").exists():
        raise FileExistsError("already prepared")
    manifest, panel = panel_and_manifest(c)
    assert manifest["metadata"]["coverage_start"] <= bounds(c, 0)["train_start"] - 14_400_000
    assert manifest["metadata"]["coverage_end_exclusive"] >= bounds(c, 9)["test_end"]
    for index in range(c["origins"]):
        b = bounds(c, index)
        assert b["train_end"] < b["validation_start"] < b["validation_end"] < b["test_start"]
    save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(),
        "source_manifest_sha256": c["source_manifest_sha256"],
        "snapshot_sha256": manifest["snapshot_sha256"], "source_files": manifest["snapshot_files"],
        "bounds": [bounds(c, i) for i in range(10)],
        "search_budget_total": 240, "data_already_revealed": True,
        "python": sys.version, "numpy": np.__version__, "safety": c["safety"],
    })
    print(json.dumps({"status": "PREPARED", "origins": 10, "search_budget": 240,
                      "receipt_sha256": file_sha256(output / "pre_run_receipt.json")}))


def window(panel, symbols, start, end, dt):
    # The feature builder only receives a prefix; it cannot inspect later outcomes.
    prefix = panel.loc[panel.timestamp < end].copy()
    frame = build_lab_features(prefix)
    return PreparedWindow.from_frame(frame, symbols, start, end, dt)


def assessment(challenger, reference, start):
    a, b = daily_returns(challenger), daily_returns(reference)
    times = start + np.arange(1, len(a) + 1, dtype=np.int64) * DAY_MS
    return assess_paired_daily(a, b, times).mapping()


def fitness_value(hac, differences, config):
    if np.all(np.asarray(differences) == 0):
        return config["search"]["exact_reference_tie_score"]
    return hac["lcb"] if hac["lcb"] is not None else config["search"]["unidentified_variation_score"]


def screen(validation, config):
    failures = []
    for cost, result in validation.items():
        for reference, hac in result["hac"].items():
            if hac["lcb"] is None or hac["lcb"] <= 0:
                failures.append(cost + "_lcb_vs_" + reference)
        if result["metrics"]["max_drawdown"] > config["internal_screen"]["maximum_validation_drawdown"]:
            failures.append(cost + "_drawdown")
        if result["metrics"]["trades"] < config["internal_screen"]["minimum_validation_trades"]:
            failures.append(cost + "_trades")
    return {"passed": not failures, "reasons": failures, "promotion_authority": False,
            "full_promotion_gate": False}


def payload(simulation, model, b, snapshot_sha, fit_sha):
    led = simulation.decisions.to_dict("records")
    model_sha = object_sha(model)
    assert len(led) == 144 and all(x["model_sha256"] == model_sha for x in led)
    assert all(x["signal_timestamp"] <= x["execution_timestamp"] < b["test_end"] for x in led)
    for x in led:
        x["snapshot_sha256"] = snapshot_sha
        x["scaler_fit_values_sha256"] = fit_sha
    return {"model": model, "model_sha256": model_sha, "metrics": simulation.metrics,
            "daily_returns": daily_returns(simulation).tolist(),
            "portfolio": simulation.portfolio.to_dict("records"),
            "trades": simulation.trades.to_dict("records"), "ledger": led,
            "ledger_sha256": object_sha(led)}


def evaluate(args):
    c, output = load_config()
    receipt = verify_pre_run(c, output)
    b = bounds(c, args.origin)
    stem = f"origin_{args.origin + 1:02d}"
    started = output / (stem + "_started.json")
    if started.exists() or (output / (stem + ".json")).exists():
        raise FileExistsError("origin already attempted")
    manifest, panel = panel_and_manifest(c)
    assert manifest["snapshot_sha256"] == receipt["snapshot_sha256"]
    save_new(started, {"origin_index": args.origin, "bounds": b,
                      "pre_run_sha256": file_sha256(output / "pre_run_receipt.json"),
                      "created_at_utc": datetime.now(timezone.utc).isoformat()})
    symbols = tuple(manifest["metadata"]["symbols"])
    dt = c["timeframe_minutes"] * 60_000
    train = window(panel, symbols, b["train_start"], b["train_end"], dt)
    scaler = LabScaler.fit(train.values)
    train_reference = train.simulate(ALLOW_ALL, scaler, 16.0)
    reference_daily = daily_returns(train_reference)
    training = {}

    def score(genome):
        simulation = train_reference if genome == ALLOW_ALL else train.simulate(genome, scaler, 16.0)
        hac = assessment(simulation, train_reference, b["train_start"])
        fitness = fitness_value(hac, daily_returns(simulation) - reference_daily, c)
        training[str(genome)] = {"metrics": simulation.metrics, "hac": hac, "fitness": fitness}
        return fitness

    search_result = search("genetic_search", score, c["search"]["unique_candidates_per_origin"],
                           c["search"]["seed_base"] + args.origin, c["search"]["population_size"])
    winner = tuple(search_result["winner"])
    selected = {"bounds": b, "search": search_result, "training": training,
                "scaler": scaler.mapping(), "train_values_sha256": array_sha(train.values),
                "model": train.model(winner, scaler), "model_sha256": object_sha(train.model(winner, scaler)),
                "training_fitness_is_selection_not_inference": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    selected_path = output / (stem + "_selected.json")
    save_new(selected_path, selected)  # Commit winner BEFORE validation or test simulation.

    val = window(panel, symbols, b["validation_start"], b["validation_end"], dt)
    validation = {}
    for cost, bps in (("normal", 16.0), ("stress", 24.0)):
        raw = val.simulate(winner, scaler, bps)
        validation[cost] = {"metrics": raw.metrics, "daily_returns": daily_returns(raw).tolist(),
                            "hac": {name: assessment(raw, val.simulate(g, scaler, bps), b["validation_start"])
                                    for name, g in (("EMA", ALLOW_ALL), ("NoTrade", NO_TRADE))}}
    decision = screen(validation, c)
    active = winner if decision["passed"] else NO_TRADE
    screen_path = output / (stem + "_screen.json")
    save_new(screen_path, {"selected_sha256": file_sha256(selected_path), "validation": validation,
                           "screen": decision, "active_genome": active,
                           "created_at_utc": datetime.now(timezone.utc).isoformat()})

    deploy = window(panel, symbols, b["test_start"], b["test_end"], dt)
    genomes = dict(zip(CANDIDATES, (NO_TRADE, ALLOW_ALL, FIXED, winner, active)))
    evaluations = {}
    for cost, bps in (("normal", 16.0), ("stress", 24.0)):
        cache = {}
        for g in set(genomes.values()):
            sim = deploy.simulate(g, scaler, bps, details=True)
            cache[g] = payload(sim, deploy.model(g, scaler), b, manifest["snapshot_sha256"], scaler.fit_values_sha256)
        evaluations[cost] = {name: cache[g] for name, g in genomes.items()}
    # Same zero-weight baselines must reproduce the frozen August diagnostic.
    old_path = (ROOT / c["source_completion"]).parent / (stem + ".json")
    completion = json.loads((ROOT / c["source_completion"]).read_text())
    assert file_sha256(old_path) == completion["artifact_sha256"][old_path.name]
    old = json.loads(old_path.read_text())
    for cost in evaluations:
        for new_name, old_name in (("EMA", "EMA_TrendConsensus"), ("NoTrade", "NoTrade")):
            np.testing.assert_allclose(evaluations[cost][new_name]["daily_returns"],
                old["evaluations"][cost]["candidates"][old_name]["daily_returns"], rtol=0, atol=1e-12)
    result = {"schema": "exia.genetic_primus.retrained_origin/1", "origin_index": args.origin,
              "bounds": b, "selected_sha256": file_sha256(selected_path),
              "screen_sha256": file_sha256(screen_path), "winner": winner,
              "screen": decision, "evaluations": evaluations, "scaler": scaler.mapping(),
              "config_sha256": file_sha256(CONFIG), "data_already_revealed": True,
              "search_evaluations": search_result["unique_evaluations"], "safety": c["safety"]}
    path = output / (stem + ".json")
    save_new(path, result)
    print(json.dumps({"origin": args.origin + 1, "winner": winner, "screen": decision["passed"],
                      "returns_percent": {n: round(v["metrics"]["net_return"] * 100, 4)
                                          for n, v in evaluations["normal"].items()},
                      "result_sha256": file_sha256(path)}))


def aggregate(results, cost, name):
    values = np.concatenate([np.array(r["evaluations"][cost][name]["daily_returns"]) for r in results])
    close_returns = np.concatenate([np.array([p["net_return"] for p in r["evaluations"][cost][name]["portfolio"]])
                                    for r in results])
    equity = np.r_[1.0, np.cumprod(1.0 + close_returns)]
    return {"return": float(equity[-1] - 1), "close_4h_drawdown": float(np.max(1 - equity / np.maximum.accumulate(equity))),
            "trades": sum(r["evaluations"][cost][name]["metrics"]["trades"] for r in results),
            "positive_origins": sum(r["evaluations"][cost][name]["metrics"]["net_return"] > 0 for r in results),
            "daily_returns": values.tolist()}


def finalize(_):
    c, output = load_config()
    verify_pre_run(c, output)
    results = []
    for i in range(10):
        stem = f"origin_{i + 1:02d}"
        r = json.loads((output / (stem + ".json")).read_text())
        assert r["bounds"] == bounds(c, i) and r["search_evaluations"] == 24
        assert r["selected_sha256"] == file_sha256(output / (stem + "_selected.json"))
        assert r["screen_sha256"] == file_sha256(output / (stem + "_screen.json"))
        assert r["config_sha256"] == file_sha256(CONFIG) and not any(r["safety"].values())
        results.append(r)
    summary = {"origins": 10, "search_evaluations": 240, "screen_passes": sum(r["screen"]["passed"] for r in results),
               "data_already_revealed": True, "promotion_authority": False, "costs": {}}
    times = millis(c["test_start_utc"]) + np.arange(1, 31, dtype=np.int64) * DAY_MS
    for cost in ("normal", "stress"):
        candidates = {name: aggregate(results, cost, name) for name in CANDIDATES}
        contrasts = {}
        for ref in ("FixedOverlaySameScaler", "EMA", "NoTrade"):
            contrasts[ref] = assess_paired_daily(np.array(candidates["RetrainedRaw"]["daily_returns"]),
                                                np.array(candidates[ref]["daily_returns"]), times).mapping()
        summary["costs"][cost] = {"candidates": candidates, "retrained_raw_hac_vs": contrasts}
    save_new(output / "summary.json", summary)
    paths = [output / "pre_run_receipt.json", output / "summary.json"]
    paths += [output / (f"origin_{i:02d}" + suffix) for i in range(1, 11)
              for suffix in ("_started.json", "_selected.json", "_screen.json", ".json")]
    save_new(output / "completion.json", {"fingerprint": fingerprint(), "artifact_count": len(paths),
             "artifact_sha256": {p.name: file_sha256(p) for p in paths},
             "status": "REVEALED_RETRAINING_COMPLETE_NO_PROMOTION", "safety": c["safety"]})
    print(json.dumps({"status": "COMPLETE", "search_evaluations": 240,
                      "screen_passes": summary["screen_passes"], "artifacts": len(paths)}))


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
