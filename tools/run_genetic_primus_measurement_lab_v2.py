"""Bounded local measurement experiment. One explicit phase/origin per process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from exia.genetic_primus.measurement_lab_v2 import (
    ALLOW_ALL, DAY_MS, NO_TRADE, LabScaler, PreparedWindow, array_sha, block_bank,
    build_lab_features, daily_returns, object_sha, paired_lcb, search, temporal_bounds,
    validation_gate,
)
from exia.genetic_primus.contracts_v4 import load_v4_config
from exia.genetic_primus.data import load_panel
from exia.genetic_primus.objective_registry import load_immutable_objective_registry

CONFIG_PATH = ROOT / "configs/genetic_primus_measurement_lab_v2.json"
REGISTRY_PATH = ROOT / "configs/genetic_primus_objective_registry_v4.json"


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_new(path: Path, value: object) -> None:
    # Exclusive create: neither prior experiments nor partial results can be overwritten.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint() -> dict:
    paths = sorted((ROOT / "src/exia/genetic_primus").glob("*.py"))
    paths += [Path(__file__).resolve(), CONFIG_PATH, REGISTRY_PATH,
              ROOT / "configs/genetic_primus_v4.json",
              ROOT / "tests/test_genetic_primus_accounting_v2.py",
              ROOT / "tests/test_genetic_primus_measurement_lab_v2.py"]
    return {"files": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths},
            "python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": file_sha(Path(sys.executable)), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow")}}


def load_contract() -> tuple[dict, Path]:
    expected_python = ROOT / ".venv/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise ValueError("only the project .venv Python is authorized")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.measurement_lab/2":
        raise ValueError("wrong contract")
    if any(config["safety"].values()) or config["blind_evaluation"]["allowed_by_this_contract"]:
        raise ValueError("offline-only safety contract violated")
    if config["unique_evaluations_per_method_per_origin"] != 24 or len(config["development_origins"]) != 3:
        raise ValueError("bounded experiment contract violated")
    output = (ROOT / config["result_path"]).resolve()
    if not output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus"):
        raise ValueError("output must stay inside the research report directory")
    return config, output


def prepare(config: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError("result root already exists; no overwrite or silent resume")
    registry = load_immutable_objective_registry(REGISTRY_PATH, workspace=ROOT)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "measurement_lab_pre_run_receipt/2",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "fingerprint": fingerprint(), "config_sha256": file_sha(CONFIG_PATH),
               "registry": registry, "hypotheses_frozen_before_this_run": True,
               "data_is_already_revealed": True, "market_readiness": False,
               "blind_authority": False, "safety": config["safety"]}
    save_new(output / "pre_run_receipt.json", receipt)
    print("prepared: immutable config/source/environment receipt; revealed-data diagnostic only")


def verify_receipt(output: Path) -> dict:
    receipt = json.loads((output / "pre_run_receipt.json").read_text(encoding="utf-8"))
    if receipt["fingerprint"] != fingerprint():
        raise ValueError("source/config/environment changed after preregistration")
    return receipt


def load_features(config: dict) -> tuple[pd.DataFrame, tuple[str, ...]]:
    base = load_v4_config(ROOT / config["dataset_config"], root=ROOT)
    if base.dataset["dataset_sha256"] != config["dataset_sha256"]:
        raise ValueError("unexpected dataset")
    panel = load_panel(base, root=ROOT)  # Validates every parquet SHA; local files only.
    return build_lab_features(panel), tuple(str(s) for s in base.dataset["symbols"])


def make_bank(window: PreparedWindow, config: dict, seed: int) -> np.ndarray:
    count = (window.end - window.start) // DAY_MS
    return block_bank(count, config["fitness"]["bootstrap_samples"], config["fitness"]["block_days"], seed)


def simulation_payload(simulation) -> dict:
    return {"metrics": simulation.metrics,
            "daily_returns": daily_returns(simulation).tolist(),
            "portfolio": simulation.portfolio.to_dict("records"),
            "trades": simulation.trades.to_dict("records"),
            "decisions": simulation.decisions.to_dict("records")}


def evaluate_origin(config: dict, output: Path, index: int) -> None:
    verify_receipt(output)
    if index not in range(len(config["development_origins"])):
        raise ValueError("unregistered origin")
    target = output / f"origin_{index + 1}.json"
    if target.exists() or (output / f"origin_{index + 1}_started.json").exists():
        raise FileExistsError("origin was already attempted; no silent repeat")
    save_new(output / f"origin_{index + 1}_started.json",
             {"created_at_utc": datetime.now(timezone.utc).isoformat(), "origin": index,
              "config_sha256": file_sha(CONFIG_PATH)})
    started = time.perf_counter()
    frame, symbols = load_features(config)
    origin = int(pd.Timestamp(config["development_origins"][index]).timestamp() * 1000)
    dt = config["timeframe_minutes"] * 60_000
    bounds = temporal_bounds(origin, config["training_days"], config["validation_days"],
                             config["boundary_gap_bars"], dt, config["deployment_horizon_days"])
    def window(label):
        return PreparedWindow.from_frame(frame, symbols, bounds[label + "_start"], bounds[label + "_end"], dt)
    train, validation, development = (window(label) for label in ("train", "validation", "development"))
    scaler = LabScaler.fit(train.values)
    seed = config["seed"] + index
    train_bank = make_bank(train, config, seed)
    validation_bank = make_bank(validation, config, seed + config["fitness"]["validation_seed_offset"])
    base_cost, stress_cost = config["costs_bps"]["base_round_trip"], config["costs_bps"]["stress_round_trip"]
    alpha = config["fitness"]["alpha"]
    cache = {}
    def evaluate(genome):
        if genome not in cache:
            simulation = train.simulate(genome, scaler, base_cost)
            returns = daily_returns(simulation)
            score = paired_lcb(returns, np.zeros_like(returns), train_bank, alpha)
            cache[genome] = {"score": score, "metrics": simulation.metrics}
        return cache[genome]["score"]
    results = []
    for method in config["methods"]:
        result = search(method, evaluate, config["unique_evaluations_per_method_per_origin"],
                        seed, config["genetic_population_size"])
        genome = result["winner"]
        normal = validation.simulate(genome, scaler, base_cost)
        stress = validation.simulate(genome, scaler, stress_cost)
        gate = validation_gate(normal, stress, validation_bank, alpha,
                               config["diagnostic_validation_gate"]["maximum_drawdown"])
        active_genome = genome if gate["passed"] else NO_TRADE
        result.update({"model": development.model(genome, scaler),
                       "model_sha256": object_sha(development.model(genome, scaler)),
                       "train_metrics": cache[genome]["metrics"],
                       "validation": {"gate": gate, "base": normal.metrics, "stress": stress.metrics},
                       "diagnostic_development": simulation_payload(development.simulate(genome, scaler, base_cost, details=True)),
                       "diagnostic_development_stress": development.simulate(genome, scaler, stress_cost).metrics,
                       "active_genome": active_genome,
                       "active_development": simulation_payload(development.simulate(active_genome, scaler, base_cost, details=True)),
                       "promotion_authority": False})
        results.append(result)
    baselines = {}
    for name, genome in (("NoTrade", NO_TRADE), ("V7_EMA_TrendConsensus", ALLOW_ALL)):
        baselines[name] = {"development": simulation_payload(development.simulate(genome, scaler, base_cost, details=True)),
                           "development_stress": development.simulate(genome, scaler, stress_cost).metrics}
    payload = {"schema": "measurement_origin_result/2", "origin_index": index,
               "origin": config["development_origins"][index], "bounds": bounds,
               "bounds_sha256": object_sha(bounds), "dataset_sha256": config["dataset_sha256"],
               "config_sha256": file_sha(CONFIG_PATH), "symbols": symbols, "scaler": asdict(scaler),
               "train_bank_sha256": array_sha(train_bank), "validation_bank_sha256": array_sha(validation_bank),
               "train_resample_bank": train_bank.tolist(), "validation_resample_bank": validation_bank.tolist(),
               "train_values_sha256": array_sha(train.values), "validation_values_sha256": array_sha(validation.values),
               "development_values_sha256": array_sha(development.values), "methods": results, "baselines": baselines,
               "logical_candidate_evaluations": sum(r["unique_evaluations"] for r in results),
               "physical_unique_candidate_evaluations": len(cache),
               "elapsed_seconds": time.perf_counter() - started, "safety": config["safety"]}
    save_new(target, payload)
    passed = sum(r["validation"]["gate"]["passed"] for r in results)
    print(f"origin {index + 1}/3 complete: logical=72 physical={len(cache)} validation_pass={passed}/3 seconds={payload['elapsed_seconds']:.1f}")


def ablation(config: dict, output: Path) -> None:
    verify_receipt(output)
    path = output / "execution_ablation.json"
    if path.exists() or (output / "ablation_started.json").exists():
        raise FileExistsError("ablation already attempted")
    save_new(output / "ablation_started.json", {"config_sha256": file_sha(CONFIG_PATH)})
    frame, symbols = load_features(config)
    start = int(pd.Timestamp(config["development_origins"][0]).timestamp() * 1000)
    window = PreparedWindow.from_frame(frame, symbols, start, start + 18 * DAY_MS, config["timeframe_minutes"] * 60_000)
    # ALLOW_ALL has zero weights; scaler values cannot influence the frozen proposal.
    scaler = LabScaler((0.,) * 4, (1.,) * 4, 0, "not_fitted_zero_weight_frozen_policy")
    resets = tuple(range(start + 3 * DAY_MS, window.end, 3 * DAY_MS))
    cost = config["costs_bps"]["base_round_trip"]
    carry = window.simulate(ALLOW_ALL, scaler, cost, details=True)
    reset = window.simulate(ALLOW_ALL, scaler, cost, details=True, force_flat_timestamps=resets)
    save_new(path, {"start": start, "end": window.end, "same_frozen_model": True,
                    "carry": simulation_payload(carry), "reset_every_3d": simulation_payload(reset),
                    "fee_difference": reset.metrics["total_fees"] - carry.metrics["total_fees"],
                    "net_return_difference": reset.metrics["net_return"] - carry.metrics["net_return"],
                    "interpretation": "reset changes both fees and position sizing; not a pure fee-only intervention",
                    "safety": config["safety"]})
    print(f"ablation complete: carry_fills={carry.metrics['fills']} reset_fills={reset.metrics['fills']}")


def finalize(config: dict, output: Path) -> None:
    receipt = verify_receipt(output)
    if (output / "summary.json").exists():
        raise FileExistsError("summary already finalized")
    origins = [json.loads((output / f"origin_{i + 1}.json").read_text(encoding="utf-8")) for i in range(3)]
    if not (output / "execution_ablation.json").exists():
        raise ValueError("ablation missing")
    summary = []
    for method in config["methods"]:
        records = [next(r for r in origin["methods"] if r["method"] == method) for origin in origins]
        summary.append({"method": method, "validation_passes": sum(r["validation"]["gate"]["passed"] for r in records),
                        "origin_count": 3,
                        "train_lcb_daily_bps": [r["score"] * 10_000 for r in records],
                        "diagnostic_development_return_bps": [r["diagnostic_development"]["metrics"]["net_return"] * 10_000 for r in records],
                        "diagnostic_development_stress_bps": [r["diagnostic_development_stress"]["net_return"] * 10_000 for r in records],
                        "active_development_return_bps": [r["active_development"]["metrics"]["net_return"] * 10_000 for r in records],
                        "diagnostic_trades": [r["diagnostic_development"]["metrics"]["trades"] for r in records],
                        "unique_train_evaluations": sum(r["unique_evaluations"] for r in records)})
    blockers = ["only_three_previously_revealed_origins_not_blind_evidence",
                "no_complete_historical_data_exposure_registry", "funding_borrow_execution_latency_not_calibrated",
                "single_seed_per_origin_cannot_establish_optimizer_superiority",
                "no_blind_authority_in_contract"]
    if not any(row["validation_passes"] for row in summary):
        blockers.append("no_train_selected_candidate_passed_independent_validation")
    payload = {"schema": "measurement_lab_summary/2", "methods": summary,
               "logical_candidate_evaluations": sum(o["logical_candidate_evaluations"] for o in origins),
               "physical_candidate_evaluations": sum(o["physical_unique_candidate_evaluations"] for o in origins),
               "baselines": {name: [o["baselines"][name]["development"]["metrics"]["net_return"] * 10_000 for o in origins]
                             for name in ("NoTrade", "V7_EMA_TrendConsensus")},
               "registered_objective_count_lower_bound": len(receipt["registry"]["objective_ids"]),
               "dsr": {"status": "not_estimated", "reason": "no_comparable_independent_global_trial_returns"},
               "pbo": {"status": "not_estimated", "reason": "three_origins_and_duplicate_abstention_not_a_valid_global_pbo"},
               "stage_5": "GA_ADVANTAGE_NOT_ESTABLISHED_NO_AUTOMATIC_NEW_SEARCH",
               "stage_6": {"decision": "NO_GO_BLIND_AND_TRADING", "blockers": blockers},
               "safety": config["safety"]}
    save_new(output / "summary.json", payload)
    artifacts = {p.name: file_sha(p) for p in sorted(output.glob("*.json"))}
    save_new(output / "completion_receipt.json", {"artifacts": artifacts,
             "completed_at_utc": datetime.now(timezone.utc).isoformat(), "safety": config["safety"]})
    print("complete: 3 revealed origins; GA advantage not established; blind/trading NO_GO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "origin", "ablation", "finalize"))
    parser.add_argument("--origin", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    config, output = load_contract()
    if args.phase == "prepare":
        prepare(config, output)
    elif args.phase == "origin":
        if args.origin is None:
            parser.error("origin phase requires --origin 1/2/3")
        evaluate_origin(config, output, args.origin - 1)
    elif args.phase == "ablation":
        ablation(config, output)
    else:
        finalize(config, output)


if __name__ == "__main__":
    main()
