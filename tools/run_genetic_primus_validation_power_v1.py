"""Bounded synthetic-only calibration. Never loads market data or trading runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
from statistics import NormalDist
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank

CONFIG = ROOT / "configs/genetic_primus_validation_power_v1.json"
NORMAL = NormalDist()


def stationary_ar1(replications: int, days: int, rho: float, seed: int) -> np.ndarray:
    if replications < 1 or days < 1 or not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("invalid stationary AR(1) dimensions or correlation")
    innovations = np.random.default_rng(seed).standard_normal((replications, days))
    paths = np.empty_like(innovations)
    paths[:, 0] = innovations[:, 0]  # Stationary initial N(0,1), not a zero transient.
    scale = np.sqrt(1.0 - rho * rho)
    for day in range(1, days):
        paths[:, day] = rho * paths[:, day - 1] + scale * innovations[:, day]
    return paths


def mean_variance(days: int, rho: float, sigma: float = 1.0) -> float:
    """Exact variance of the sample mean under the specified stationary DGP."""
    if days < 1 or abs(rho) >= 1 or sigma <= 0 or not np.isfinite([rho, sigma]).all():
        raise ValueError("invalid Gaussian oracle parameters")
    lags = np.arange(1, days, dtype=float)
    covariance_sum = days + 2.0 * np.sum((days - lags) * np.power(rho, lags))
    return float(sigma * sigma * covariance_sum / (days * days))


def oracle_power(days: int, rho: float, sigma: float, mean: float, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    se = np.sqrt(mean_variance(days, rho, sigma))
    return 1.0 - NORMAL.cdf(NORMAL.inv_cdf(1.0 - alpha) - mean / se)


def rate_interval(successes: int, count: int, confidence: float = .95) -> dict:
    """Wilson Monte Carlo interval; it is NOT a confidence interval for market edge."""
    if count < 1 or not 0 <= successes <= count or not 0 < confidence < 1:
        raise ValueError("invalid binomial observation")
    z = NORMAL.inv_cdf(0.5 + confidence / 2.0)
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return {"successes": successes, "replications": count, "rate": p,
            "lower": max(0.0, float(center - half)), "upper": min(1.0, float(center + half)),
            "confidence": confidence}


def batched_lcbs(paths: np.ndarray, bank: np.ndarray, alpha: float, batch_size: int = 8) -> np.ndarray:
    """Vectorized equivalent of measurement_lab_v2.paired_lcb(paths, zero, bank)."""
    if paths.ndim != 2 or bank.ndim != 2 or bank.shape[1] != paths.shape[1]:
        raise ValueError("unaligned paths and bank")
    if not len(paths) or not bank.size or batch_size < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap dimensions")
    if not np.isfinite(paths).all() or not np.issubdtype(bank.dtype, np.integer):
        raise ValueError("nonfinite observations or noninteger indices")
    if bank.min() < 0 or bank.max() >= paths.shape[1]:
        raise ValueError("resample index out of bounds")
    output = np.empty(len(paths))
    for start in range(0, len(paths), batch_size):
        means = paths[start:start + batch_size, bank].mean(axis=2)
        output[start:start + batch_size] = np.quantile(means, alpha, axis=1)
    return output


def daily_drawdown(paths: np.ndarray) -> np.ndarray:
    if paths.ndim != 2 or not np.isfinite(paths).all() or (paths <= -1.0).any():
        raise ValueError("invalid synthetic daily return path")
    equity = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=1)
    return np.max(1.0 - equity / peak, axis=1)


def classify_null(rate: dict, alpha: float) -> str:
    if rate["lower"] > alpha:
        return "FALSE_POSITIVE_INFLATION_DETECTED"
    if rate["upper"] < alpha:
        return "CONSERVATIVE_IN_THIS_SYNTHETIC_CELL"
    return "INCONCLUSIVE_NOT_PROOF_OF_CALIBRATION"


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(config: dict) -> dict:
    paths = [CONFIG, Path(__file__).resolve(), ROOT / "tests/test_genetic_primus_validation_power_v1.py",
             ROOT / "src/exia/genetic_primus/measurement_lab_v2.py",
             ROOT / "src/exia/genetic_primus/accounting_v2.py",
             ROOT / config["prior_lab_summary"]]
    return {"files": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths},
            "python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": file_sha(Path(sys.executable)),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas")}}


def load_contract() -> tuple[dict, Path]:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only the workspace .venv Python is authorized")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.validation_power/1":
        raise ValueError("invalid schema")
    if config["allowed_data_sources"] != ["synthetic_only"] or any(config["safety"].values()):
        raise ValueError("synthetic-only safety contract violated")
    if config["automatic_threshold_or_block_length_changes"] or config["automatic_validation_horizon_selection"]:
        raise ValueError("this diagnostic cannot select new thresholds")
    if config["maximum_cells"] != 15 or len(cells(config)) != 15:
        raise ValueError("unexpected experiment dimensions")
    if config["replications"] != 1024 or config["batch_size"] > 8 or max(config["horizons_days"]) > 365:
        raise ValueError("memory/work bounds changed")
    if config["null_family_size"] != 3 * len(cells(config)):
        raise ValueError("wrong familywise null count")
    if any(config["bootstrap"][key] != value for key, value in
           (("samples", 256), ("block_days", 5), ("alpha", .05))):
        raise ValueError("the existing bootstrap settings must remain unchanged")
    if config["synthetic_stress_extra_bps_per_day"] < 0:
        raise ValueError("stress cannot lower costs")
    output = (ROOT / config["result_path"]).resolve()
    if not output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus"):
        raise ValueError("unsafe output directory")
    return config, output


def cells(config: dict) -> list[tuple[float, int]]:
    return [(rho, days) for rho in config["ar1_correlations"] for days in config["horizons_days"]]


def verify(config: dict, output: Path) -> dict:
    receipt = json.loads((output / "pre_run_receipt.json").read_text(encoding="utf-8"))
    if receipt["fingerprint"] != fingerprint(config):
        raise ValueError("config/source/environment changed after preregistration")
    return receipt


def prepare(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(config),
        "cells": cells(config), "new_market_observations": 0, "strategy_training_runs": 0,
        "maximum_bootstrap_gather_bytes": config["batch_size"] * config["bootstrap"]["samples"] * 365 * 8,
        "safety": config["safety"],
    })
    print("prepared: 15 synthetic cells; 1024 paths each; two fixed resample banks; no market data")


def run_cell(config: dict, output: Path, index: int) -> None:
    verify(config, output)
    if not 0 <= index < len(cells(config)):
        raise ValueError("unregistered cell")
    rho, days = cells(config)[index]
    prefix = f"cell_{index + 1:02d}"
    if any((output / (prefix + suffix)).exists() for suffix in (".json", ".npz", "_started.json")):
        raise FileExistsError("cell already attempted; no repeat or overwrite")
    save_new(output / (prefix + "_started.json"), {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": file_sha(CONFIG),
        "rho": rho, "days": days,
    })
    started = time.perf_counter()
    bootstrap = config["bootstrap"]
    # Generate the same maximum-length innovations for every horizon; prefixes are paired.
    paths = stationary_ar1(config["replications"], max(config["horizons_days"]), rho,
                           config["generator_seed"])[:, :days].copy()
    banks = {name: block_bank(days, bootstrap["samples"], bootstrap["block_days"], bootstrap[name + "_seed"])
             for name in ("primary", "audit")}
    lcbs = {name: batched_lcbs(paths, bank, bootstrap["alpha"], config["batch_size"])
            for name, bank in banks.items()}
    n = config["replications"]
    confidence = config["pointwise_monte_carlo_confidence"]
    interval = lambda values: rate_interval(int(np.count_nonzero(values)), n, confidence)
    null_confidence = 1.0 - (1.0 - config["null_familywise_confidence"]) / config["null_family_size"]
    null_results = []
    for name in ("primary", "audit", "both_banks"):
        reject = lcbs[name] > 0 if name in lcbs else (lcbs["primary"] > 0) & (lcbs["audit"] > 0)
        rate = rate_interval(int(reject.sum()), n, null_confidence)
        null_results.append({"bank": name, "false_positive_rate": rate,
                             "diagnosis": classify_null(rate, bootstrap["alpha"])})
    scenarios = []
    stress_cost = config["synthetic_stress_extra_bps_per_day"] / 10_000.0
    for sigma_bps in config["daily_volatility_bps"]:
        sigma = sigma_bps / 10_000.0
        for mean_bps in config["daily_mean_bps"]:
            mean = mean_bps / 10_000.0
            dd = daily_drawdown(mean + sigma * paths)
            dd_pass = dd <= config["daily_mark_drawdown_proxy_limit"]
            normal_pass = {name: mean + sigma * value > 0 for name, value in lcbs.items()}
            stress_pass = {name: mean + sigma * value - stress_cost > 0 for name, value in lcbs.items()}
            normal_pass["both_banks"] = normal_pass["primary"] & normal_pass["audit"]
            stress_pass["both_banks"] = stress_pass["primary"] & stress_pass["audit"]
            scenarios.append({"daily_mean_bps": mean_bps, "daily_volatility_bps": sigma_bps,
                              "known_parameter_oracle_power": oracle_power(days, rho, sigma, mean, bootstrap["alpha"]),
                              "known_parameter_oracle_stress_power": oracle_power(days, rho, sigma, mean - stress_cost, bootstrap["alpha"]),
                              "daily_drawdown_proxy_pass_rate": interval(dd_pass),
                              "banks": {name: {"normal_lcb_pass_rate": interval(normal_pass[name]),
                                               "normal_and_stress_lcb_pass_rate": interval(stress_pass[name]),
                                               "lcb_stress_and_daily_drawdown_proxy_pass_rate": interval(stress_pass[name] & dd_pass)}
                                        for name in normal_pass}})
    oracle_threshold = NORMAL.inv_cdf(1.0 - bootstrap["alpha"]) * np.sqrt(mean_variance(days, rho))
    arrays = {"unit_paths": paths, "primary_bank": banks["primary"], "audit_bank": banks["audit"],
              "primary_unit_lcb": lcbs["primary"], "audit_unit_lcb": lcbs["audit"]}
    npz_path = output / (prefix + ".npz")
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    payload = {"cell": index + 1, "rho": rho, "days": days, "config_sha256": file_sha(CONFIG),
               "arrays_sha256": {name: array_sha(value) for name, value in arrays.items()},
               "npz_sha256": file_sha(npz_path), "replications": n, "null_results": null_results,
               "oracle_null_false_positive_rate": interval(paths.mean(axis=1) > oracle_threshold),
               "known_ar1_information_equivalent_days": 1.0 / mean_variance(days, rho),
               "scenarios": scenarios, "elapsed_seconds": time.perf_counter() - started,
               "safety": config["safety"]}
    save_new(output / (prefix + ".json"), payload)
    rates = [round(r["false_positive_rate"]["rate"] * 100, 2) for r in null_results]
    print(f"cell {index + 1}/15 days={days} rho={rho}: null primary/audit/both={rates}% seconds={payload['elapsed_seconds']:.2f}")


def finalize(config: dict, output: Path) -> None:
    verify(config, output)
    if (output / "summary.json").exists():
        raise FileExistsError("already finalized")
    results = []
    for i in range(len(cells(config))):
        prefix = f"cell_{i + 1:02d}"
        value = json.loads((output / (prefix + ".json")).read_text(encoding="utf-8"))
        if value["config_sha256"] != file_sha(CONFIG) or value["npz_sha256"] != file_sha(output / (prefix + ".npz")):
            raise ValueError("cell hash mismatch")
        with np.load(output / (prefix + ".npz"), allow_pickle=False) as arrays:
            if any(array_sha(arrays[name]) != digest for name, digest in value["arrays_sha256"].items()):
                raise ValueError("numeric array hash mismatch")
        results.append(value)
    inflation = [{"days": r["days"], "rho": r["rho"], **null} for r in results for null in r["null_results"]
                 if null["diagnosis"] == "FALSE_POSITIVE_INFLATION_DETECTED"]
    # Analytic reference only. Never select a production horizon from the Monte Carlo table.
    oracle_horizons = []
    for rho in config["ar1_correlations"]:
        for sigma_bps in config["daily_volatility_bps"]:
            for mean_bps in config["daily_mean_bps"]:
                eligible = [days for days in config["horizons_days"] if oracle_power(
                    days, rho, sigma_bps / 10_000.0,
                    (mean_bps - config["synthetic_stress_extra_bps_per_day"]) / 10_000.0,
                    config["bootstrap"]["alpha"]) >= config["target_power"]]
                oracle_horizons.append({"rho": rho, "daily_volatility_bps": sigma_bps,
                                        "daily_mean_bps": mean_bps, "first_grid_horizon_at_80pct_stress_oracle_power": min(eligible) if eligible else None})
    payload = {"schema": "exia.genetic_primus.validation_power_result/1", "cells_completed": len(results),
               "config_sha256": file_sha(CONFIG), "false_positive_inflation_cases": inflation,
               "null_intervals": "Bonferroni-adjusted Wilson Monte Carlo intervals over 45 prespecified tests",
               "oracle_horizons": oracle_horizons, "automatic_parameter_changes": False,
               "decision": "NO_NEW_GA_TRAINING_OR_BLIND_OPENING",
               "limitation": "stationary Gaussian synthetic sensitivity; not empirically fitted to market, no sparse trades or regime drift",
               "safety": config["safety"]}
    save_new(output / "summary.json", payload)
    files = {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.suffix in (".json", ".npz")}
    save_new(output / "completion_receipt.json", {"artifacts": files,
             "completed_at_utc": datetime.now(timezone.utc).isoformat(), "safety": config["safety"]})
    print(f"complete: {len(results)} cells; {len(inflation)} adjusted null inflation cases; no parameter changes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "cell", "finalize"))
    parser.add_argument("--cell", type=int, choices=range(1, 16))
    args = parser.parse_args()
    config, output = load_contract()
    if args.phase == "prepare":
        prepare(config, output)
    elif args.phase == "cell":
        if args.cell is None:
            parser.error("cell phase requires --cell 1..15")
        run_cell(config, output, args.cell - 1)
    else:
        finalize(config, output)


if __name__ == "__main__":
    main()
