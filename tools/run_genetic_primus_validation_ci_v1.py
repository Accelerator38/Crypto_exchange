"""Two experimental confidence bounds; local synthetic data only, sealed audit seeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.stats import t as student_t
from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank

CONFIG = ROOT / "configs/genetic_primus_validation_ci_v1.json"
NORMAL = NormalDist()
EPS = 1e-14


def hac_standard_error(values: np.ndarray, lags: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = values.shape[-1]
    if n < 2 or not 0 <= lags < n or not np.isfinite(values).all():
        raise ValueError("invalid HAC observations or bandwidth")
    centered = values - values.mean(axis=-1, keepdims=True)
    lrv = np.einsum("...i,...i->...", centered, centered) / n
    for lag in range(1, lags + 1):
        covariance = np.einsum("...i,...i->...", centered[..., lag:], centered[..., :-lag]) / n
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    return np.sqrt(np.maximum(lrv, 0.0) / (n - 1.0))


def ci_bounds(paths: np.ndarray, methods: list[str], bank: np.ndarray, alpha: float,
              batch_size: int = 2, minimum_valid: float = .9) -> tuple[dict, dict]:
    if paths.ndim != 2 or not len(paths) or not np.isfinite(paths).all() or not 0 < alpha < .5:
        raise ValueError("invalid CI input")
    n = paths.shape[1]
    block = math.ceil(math.sqrt(n))
    if n < 4 or bank.ndim != 2 or bank.shape[1] != n or not bank.size or bank.min() < 0 or bank.max() >= n:
        raise ValueError("unaligned bank")
    if not np.issubdtype(bank.dtype, np.integer) or batch_size < 1:
        raise ValueError("invalid resample indices or batch size")
    if not set(methods) <= {"hac_small_sample_t", "studentized_block_bootstrap"}:
        raise ValueError("unregistered CI method")
    mean = paths.mean(axis=1)
    se = hac_standard_error(paths, block - 1)
    informative = se > EPS
    bounds, details = {}, {"block_length": block, "hac_lags": block - 1,
                           "uninformative_original_count": int((~informative).sum())}
    df = max(2, n // (2 * block) - 1)
    details["hac_df"] = df
    if "hac_small_sample_t" in methods:
        critical = float(student_t.ppf(1.0 - alpha, df))
        bounds["hac_small_sample_t"] = np.where(informative, mean - critical * se, -np.inf)
        details["hac_critical"] = critical
    if "studentized_block_bootstrap" in methods:
        result = np.full(len(paths), -np.inf)
        supported = 0
        for start in range(0, len(paths), batch_size):
            stop = min(start + batch_size, len(paths))
            sample = paths[start:stop, bank]
            sample_se = hac_standard_error(sample, block - 1)
            good = sample_se > EPS
            roots = np.full(sample_se.shape, np.inf)
            np.divide(sample.mean(axis=2) - mean[start:stop, None], sample_se, out=roots, where=good)
            critical = np.quantile(roots, 1.0 - alpha, axis=1, method="higher")
            critical = np.maximum(critical, NORMAL.inv_cdf(1.0 - alpha))
            usable = informative[start:stop] & np.isfinite(critical) & (good.mean(axis=1) >= minimum_valid)
            local = np.full(stop - start, -np.inf)
            local[usable] = mean[start:stop][usable] - critical[usable] * se[start:stop][usable]
            result[start:stop] = local
            supported += int(usable.sum())
        bounds["studentized_block_bootstrap"] = result
        details["studentized_supported_count"] = supported
    return bounds, details


def legacy_lcb(paths: np.ndarray, bank: np.ndarray, batch_size: int = 2) -> np.ndarray:
    result = np.empty(len(paths))
    for start in range(0, len(paths), batch_size):
        result[start:start + batch_size] = np.quantile(paths[start:start + batch_size, bank].mean(axis=2), .05, axis=1)
    return result


def generate_paths(generator: str, days: int, replications: int, seed: int, burn: int = 256) -> np.ndarray:
    names = ("iid_gaussian", "ar1_gaussian_06", "ar1_t5_06", "sparse_markov25_ar1_06", "volatility_shift_ar1_06")
    if generator not in names or not 4 <= days <= 365 or replications < 1 or burn < 64:
        raise ValueError("unregistered synthetic generator")
    rng = np.random.default_rng(np.random.SeedSequence([seed, names.index(generator)]))
    # Always generate the same maximum-length noise before selecting a horizon prefix.
    length = 365 + burn
    noise = (rng.standard_t(5, (replications, length)) * np.sqrt(3.0 / 5.0)
             if generator == "ar1_t5_06" else rng.standard_normal((replications, length)))
    rho = 0.0 if generator == "iid_gaussian" else .6
    value = np.zeros(replications)
    paths = np.empty((replications, 365))
    for step in range(length):
        value = rho * value + np.sqrt(1.0 - rho * rho) * noise[:, step]
        if step >= burn:
            paths[:, step - burn] = value
    paths = paths[:, :days].copy()
    if generator == "sparse_markov25_ar1_06":
        active = rng.random(replications) < .25
        for day in range(days):
            if day:
                active = rng.random(replications) < np.where(active, .85, .05)
            paths[:, day] *= active / np.sqrt(.25)
    if generator == "volatility_shift_ar1_06":
        multipliers = np.where(np.arange(days) < days // 2, .5, 1.5)
        paths *= multipliers / np.sqrt(np.mean(multipliers ** 2))
    return .01 * paths


def rate_interval(successes: int, total: int, confidence: float = .95) -> dict:
    if total < 1 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("invalid binomial input")
    z = NORMAL.inv_cdf((1.0 + confidence) / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"rate": p, "lower": max(0.0, float(center - radius)), "upper": min(1.0, float(center + radius)),
            "successes": successes, "replications": total, "confidence": confidence}


def select_candidate(evidence: dict, methods: list[str], false_positive_limit: float, power_floor: float) -> tuple[str | None, dict]:
    checks = {}
    for method in methods:
        item = evidence[method]
        reasons = []
        if max(item["null_upper_bounds"]) > false_positive_limit:
            reasons.append("null_upper_bound_exceeds_target")
        if item["positive_control"]["lower"] < power_floor:
            reasons.append("positive_control_power_too_low")
        checks[method] = {"passed": not reasons, "reasons": reasons,
                          "worst_null_upper_bound": max(item["null_upper_bounds"]),
                          "power10_lower": item["power10"]["lower"]}
    eligible = [method for method in methods if checks[method]["passed"]]
    winner = max(eligible, key=lambda method: (checks[method]["power10_lower"], -methods.index(method))) if eligible else None
    return winner, checks


def file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def cases(config: dict) -> list[tuple[str, int]]:
    return [(name, days) for name in config["generators"] for days in config["horizons_days"]]


def fingerprint(config: dict) -> dict:
    paths = [CONFIG, Path(__file__).resolve(), ROOT / "tests/test_genetic_primus_validation_ci_v1.py",
             ROOT / "src/exia/genetic_primus/measurement_lab_v2.py", ROOT / "src/exia/genetic_primus/accounting_v2.py",
             ROOT / config["prior_summary"]]
    return {"files": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}, "python": sys.version,
            "executable": str(Path(sys.executable).resolve()), "executable_sha256": file_sha(Path(sys.executable)),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")}}


def load_contract() -> tuple[dict, Path]:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("workspace .venv Python required")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.validation_ci/1" or config["data_sources"] != ["synthetic_only"]:
        raise ValueError("wrong experiment contract")
    if any(config["safety"].values()) or config["automatic_runtime_integration"] or config["automatic_ga_training"]:
        raise ValueError("research-only safety violated")
    if len(cases(config)) != 10 or config["replications"] != 1024 or config["bootstrap_samples"] != 256 or config["batch_size"] != 2:
        raise ValueError("bounded workload changed")
    if config["horizons_days"] != [30, 365] or config["interval_alpha"] != .025 or config["daily_sigma"] != .01:
        raise ValueError("registered procedure changed")
    output = (ROOT / config["result_path"]).resolve()
    if not output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus"):
        raise ValueError("unsafe output location")
    return config, output


def verify(config: dict, output: Path) -> None:
    receipt = json.loads((output / "pre_run_receipt.json").read_text(encoding="utf-8"))
    if receipt["fingerprint"] != fingerprint(config):
        raise ValueError("frozen source/config/environment changed")


def prepare(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "pre_run_receipt.json", {"fingerprint": fingerprint(config), "cases": cases(config),
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "audit_locked": True, "safety": config["safety"]})
    print("prepared: two CI candidates; 10 primary cases; selected-only audit locked")


def phase_methods(config: dict, output: Path, phase: str) -> list[str]:
    if phase == "primary":
        if (output / "selection.json").exists():
            raise ValueError("primary phase already closed")
        return config["methods"]
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "selection_receipt.json").read_text(encoding="utf-8"))
    if seal["selection_sha256"] != file_sha(output / "selection.json"):
        raise ValueError("selection changed after it was sealed")
    if selection["config_sha256"] != file_sha(CONFIG) or selection["selected_method"] is None:
        raise ValueError("audit not authorized by primary selection")
    if not all(file_sha(output / name) == digest for name, digest in selection["primary_artifacts"].items()):
        raise ValueError("primary evidence changed before audit")
    return [selection["selected_method"]]


def run_case(config: dict, output: Path, phase: str, index: int) -> None:
    verify(config, output)
    methods = phase_methods(config, output, phase)
    if phase not in ("primary", "audit") or not 0 <= index < 10:
        raise ValueError("unregistered phase/case")
    name, days = cases(config)[index]
    prefix = f"{phase}_{index + 1:02d}"
    if any((output / (prefix + suffix)).exists() for suffix in (".json", ".npz", "_started.json")):
        raise FileExistsError("case already attempted; no repeat or overwrite")
    save_new(output / (prefix + "_started.json"), {"config_sha256": file_sha(CONFIG), "methods": methods,
             "created_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    seed = config[phase + "_seed"]
    paths = generate_paths(name, days, config["replications"], seed, config["burn_in_days"])
    bank = block_bank(days, config["bootstrap_samples"], math.ceil(math.sqrt(days)), seed + days)
    bounds, details = ci_bounds(paths, methods, bank, config["interval_alpha"], config["batch_size"], config["minimum_valid_studentized_fraction"])
    old_bank = block_bank(days, config["bootstrap_samples"], 5, seed + days)
    old_bound = legacy_lcb(paths, old_bank, config["batch_size"])
    confidence = 1.0 - (1.0 - config["familywise_monte_carlo_confidence"]) / (10 * len(methods))
    metrics = {}
    for method, bound in bounds.items():
        row = {"null": rate_interval(int((bound > 0).sum()), len(paths), confidence),
               "uninformative_count": int((~np.isfinite(bound)).sum())}
        if name == config["positive_control"]["generator"] and days == config["positive_control"]["days"]:
            row["positive_control"] = rate_interval(int((bound + config["positive_control"]["daily_mean_bps"] / 10_000.0 > 0).sum()), len(paths))
            row["power10"] = rate_interval(int((bound + config["selection_power_mean_bps"] / 10_000.0 > 0).sum()), len(paths))
        metrics[method] = row
    arrays = {"paths": paths, "candidate_bank": bank, "legacy_bank": old_bank,
              "legacy_lcb": old_bound, **{method + "_lcb": value for method, value in bounds.items()}}
    array_path = output / (prefix + ".npz")
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    payload = {"phase": phase, "case": index + 1, "generator": name, "days": days, "methods": metrics,
               "details": details, "legacy_null": rate_interval(int((old_bound > 0).sum()), len(paths)),
               "config_sha256": file_sha(CONFIG), "npz_sha256": file_sha(array_path),
               "arrays_sha256": {key: array_sha(value) for key, value in arrays.items()},
               "elapsed_seconds": time.perf_counter() - started, "safety": config["safety"]}
    save_new(output / (prefix + ".json"), payload)
    rates = {method: round(value["null"]["rate"] * 100, 2) for method, value in metrics.items()}
    print(f"{phase} {index + 1}/10 {name} n={days}: null={rates}; seconds={payload['elapsed_seconds']:.2f}")


def collect(output: Path, phase: str, config: dict) -> tuple[list, dict]:
    rows, hashes = [], {}
    for index in range(1, 11):
        stem = f"{phase}_{index:02d}"
        row = json.loads((output / (stem + ".json")).read_text(encoding="utf-8"))
        if row["config_sha256"] != file_sha(CONFIG) or row["npz_sha256"] != file_sha(output / (stem + ".npz")):
            raise ValueError("result/config hash mismatch")
        with np.load(output / (stem + ".npz"), allow_pickle=False) as arrays:
            if any(array_sha(arrays[key]) != value for key, value in row["arrays_sha256"].items()):
                raise ValueError("numeric array hash mismatch")
        rows.append(row)
        for suffix in (".json", ".npz", "_started.json"):
            hashes[stem + suffix] = file_sha(output / (stem + suffix))
    return rows, hashes


def evidence_from_rows(rows: list, methods: list[str]) -> dict:
    result = {}
    for method in methods:
        control = next(row["methods"][method] for row in rows if "positive_control" in row["methods"][method])
        result[method] = {"null_upper_bounds": [row["methods"][method]["null"]["upper"] for row in rows],
                          "positive_control": control["positive_control"], "power10": control["power10"]}
    return result


def select(config: dict, output: Path) -> None:
    verify(config, output)
    if (output / "selection.json").exists():
        raise FileExistsError("selection already frozen")
    rows, hashes = collect(output, "primary", config)
    evidence = evidence_from_rows(rows, config["methods"])
    winner, checks = select_candidate(evidence, config["methods"], config["target_false_positive_rate"], config["positive_control"]["minimum_power_lower_bound"])
    save_new(output / "selection.json", {"selected_method": winner, "checks": checks, "evidence": evidence,
             "config_sha256": file_sha(CONFIG), "primary_artifacts": hashes,
             "audit_may_open": winner is not None, "runtime_authority": False})
    save_new(output / "selection_receipt.json", {"selection_sha256": file_sha(output / "selection.json"),
             "config_sha256": file_sha(CONFIG), "created_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"primary closed: selected={winner}; selected-only synthetic audit permitted={winner is not None}")


def finalize(config: dict, output: Path) -> None:
    verify(config, output)
    if (output / "summary.json").exists():
        raise FileExistsError("already finalized")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    seal = json.loads((output / "selection_receipt.json").read_text(encoding="utf-8"))
    if seal["selection_sha256"] != file_sha(output / "selection.json"):
        raise ValueError("sealed selection changed")
    winner = selection["selected_method"]
    audit_checks = None
    if winner is not None:
        phase_methods(config, output, "audit")
        rows, _ = collect(output, "audit", config)
        audit_evidence = evidence_from_rows(rows, [winner])
        accepted, audit_checks = select_candidate(audit_evidence, [winner], config["target_false_positive_rate"], config["positive_control"]["minimum_power_lower_bound"])
        decision = "SYNTHETIC_AUDIT_PASSED_NO_RUNTIME_AUTHORITY" if accepted else "SYNTHETIC_AUDIT_FAILED_NO_FALLBACK"
    else:
        collect(output, "primary", config)
        decision = "NO_PRIMARY_ELIGIBLE_CI_AUDIT_NOT_OPENED"
    save_new(output / "summary.json", {"schema": "exia.genetic_primus.validation_ci_result/1", "selected_method": winner,
             "primary_checks": selection["checks"], "audit_checks": audit_checks, "decision": decision,
             "existing_gate_changed": False, "ga_training_runs": 0, "market_data_opened": False,
             "limitation": "finite synthetic domain, approximate Wilson MC bounds; not a universal coverage theorem",
             "safety": config["safety"]})
    hashes = {p.name: file_sha(p) for p in sorted(output.iterdir()) if p.suffix in (".json", ".npz")}
    save_new(output / "completion_receipt.json", {"artifacts": hashes, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "safety": config["safety"]})
    print(decision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "primary", "select", "audit", "finalize"))
    parser.add_argument("--case", type=int, choices=range(1, 11))
    args = parser.parse_args()
    config, output = load_contract()
    if args.phase == "prepare":
        prepare(config, output)
    elif args.phase in ("primary", "audit"):
        if args.case is None:
            parser.error("--case 1..10 required")
        run_case(config, output, args.phase, args.case - 1)
    elif args.phase == "select":
        select(config, output)
    else:
        finalize(config, output)


if __name__ == "__main__":
    main()
