"""Restore frozen validation ledgers, then compare CI versions. No fitting/search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from exia.genetic_primus.contracts_v4 import load_v4_config
from exia.genetic_primus.measurement_lab_v2 import (
    LabScaler, PreparedWindow, array_sha, build_lab_features, daily_returns,
    object_sha, validation_gate,
)
from exia.genetic_primus_statistics.hac_adapter_v1 import (
    assess_paired_daily, complete_daily_returns, diagnostic_gate, qualified_lcb_matrix, specification,
)

CONFIG = ROOT / "configs/genetic_primus_frozen_hac_review_v1.json"
ADAPTER = ROOT / "src/exia/genetic_primus_statistics/hac_adapter_v1.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def bounded_path(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("artifact path escapes its registered root")
    return path


def assert_same(actual, expected, tolerance: float, label: str = "value") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(label + " field mismatch")
        for key in expected:
            assert_same(actual[key], expected[key], tolerance, label + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " list mismatch")
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, tolerance, label + f"[{index}]")
    elif isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(label + " boolean mismatch")
    elif isinstance(expected, (int, float)):
        if not np.isfinite(actual) or abs(actual - expected) > tolerance:
            raise ValueError(label + " numeric mismatch")
    elif actual != expected:
        raise ValueError(label + " mismatch")


def verify_parent(root: Path) -> dict:
    receipt = read_json(root / "pre_run_receipt.json")
    fp = receipt["fingerprint"]
    for name, expected in fp["files"].items():
        if file_sha(bounded_path(ROOT, name)) != expected:
            raise ValueError("parent source/config fingerprint changed: " + name)
    if fp["python"] != sys.version:
        raise ValueError("parent Python version changed")
    expected_executable = fp.get("executable", fp.get("python_executable"))
    expected_binary_sha = fp.get("executable_sha256", fp.get("python_executable_sha256"))
    if Path(expected_executable).resolve() != Path(sys.executable).resolve() or file_sha(Path(sys.executable)) != expected_binary_sha:
        raise ValueError("parent Python executable fingerprint changed")
    for name, version in fp["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("parent dependency version changed: " + name)
    completion = read_json(root / "completion_receipt.json")
    for name, expected in completion["artifacts"].items():
        if file_sha(bounded_path(root, name)) != expected:
            raise ValueError("parent artifact changed: " + name)
    return {"pre_run_receipt_sha256": file_sha(root / "pre_run_receipt.json"),
            "completion_receipt_sha256": file_sha(root / "completion_receipt.json"),
            "verified_artifact_count": len(completion["artifacts"])}


def lineage(config: dict) -> dict:
    if file_sha(ROOT / config["source_lab_config"]) != config["source_lab_config_sha256"]:
        raise ValueError("source lab contract mismatch")
    if file_sha(ROOT / config["qualification_config"]) != config["qualification_config_sha256"]:
        raise ValueError("qualification contract mismatch")
    lab_root, ci_root = ROOT / config["source_lab_root"], ROOT / config["qualification_root"]
    proof = {"lab": verify_parent(lab_root), "qualification": verify_parent(ci_root)}
    summary = read_json(ci_root / "summary.json")
    if summary["decision"] != "SYNTHETIC_AUDIT_PASSED_NO_RUNTIME_AUTHORITY" or summary["selected_method"] != "hac_small_sample_t":
        raise ValueError("HAC qualification is absent")
    return proof


def fingerprint(config: dict) -> dict:
    paths = [CONFIG, Path(__file__).resolve(), ADAPTER, ADAPTER.parent / "__init__.py",
             ROOT / "tests/test_genetic_primus_hac_adapter_v1.py"]
    return {"files": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths},
            "python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": file_sha(Path(sys.executable)),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pyarrow")},
            "parents": lineage(config)}


def load_contract() -> tuple[dict, Path]:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only workspace .venv Python is authorized")
    config = read_json(CONFIG)
    if config["schema_version"] != "exia.genetic_primus.frozen_hac_review/1":
        raise ValueError("invalid schema")
    if any(config["safety"].values()) or config["automatic_runtime_integration"] or config["training_runs"] != 0:
        raise ValueError("offline frozen-review safety violated")
    if config["origins"] != [1, 2, 3] or config["window"] != "original_validation_only" or config["incumbent"] != "NoTrade":
        raise ValueError("registered scope changed")
    output = bounded_path(ROOT / "Reports/Exia/Genetic_Primus", Path(config["result_path"]).name)
    if output != (ROOT / config["result_path"]).resolve():
        raise ValueError("unsafe result path")
    return config, output


def verify(config: dict, output: Path) -> None:
    if read_json(output / "pre_run_receipt.json")["fingerprint"] != fingerprint(config):
        raise ValueError("review source/config/environment/parent changed after freeze")


def prepare(config: dict, output: Path) -> None:
    fp = fingerprint(config)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "pre_run_receipt.json", {"fingerprint": fp, "adapter_specification": specification(),
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "safety": config["safety"]})
    print("prepared: frozen model/scaler review; no fitting, search or new time interval")


def parity(config: dict, output: Path) -> None:
    verify(config, output)
    if (output / "adapter_parity_receipt.json").exists():
        raise FileExistsError("parity already completed")
    ci_root = ROOT / config["qualification_root"]
    checks = []
    for phase in ("primary", "audit"):
        for index in range(1, 11):
            path = ci_root / f"{phase}_{index:02d}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                observed = qualified_lcb_matrix(arrays["paths"])
                expected = arrays["hac_small_sample_t_lcb"]
                if not np.array_equal(observed, expected):
                    raise ValueError("adapter differs from frozen qualified implementation")
                checks.append({"file": path.name, "count": len(expected), "lcb_sha256": array_sha(observed)})
    save_new(output / "adapter_parity_receipt.json", {"passed": True, "adapter_sha256": file_sha(ADAPTER),
             "checks": checks, "total_bounds": sum(row["count"] for row in checks), "safety": config["safety"]})
    print("adapter parity PASS: 20 matrices, 20480 bounds, bitwise equality including insufficient evidence")


def frozen_features(source_config: dict, cutoff: int) -> tuple[pd.DataFrame, dict]:
    base = load_v4_config(ROOT / source_config["dataset_config"], root=ROOT)
    manifest_path = bounded_path(ROOT, base.dataset["manifest"])
    manifest = read_json(manifest_path)
    if manifest["dataset_sha256"] != source_config["dataset_sha256"] or manifest["validation"]["passed"] is not True or manifest["timeframe"] != "240m":
        raise ValueError("dataset contract mismatch")
    columns = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    frames, files = [], []
    for item in manifest["files"]:
        path = bounded_path(ROOT, item["path"])
        digest = file_sha(path)
        if digest != item["sha256"]:
            raise ValueError("frozen parquet SHA mismatch")
        frames.append(pd.read_parquet(path, columns=columns, filters=[("timestamp", "<", cutoff)]))
        files.append({"path": item["path"], "sha256": digest})
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    for key in columns[1:-1]:
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype("float64")
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if frame.empty or int(frame.timestamp.max()) >= cutoff or frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("invalid or future market rows")
    if set(frame.symbol.unique()) != set(base.dataset["symbols"]):
        raise ValueError("fixed universe mismatch")
    proof = {"cutoff_exclusive": cutoff, "maximum_loaded_timestamp": int(frame.timestamp.max()),
             "manifest_sha256": file_sha(manifest_path), "dataset_sha256": manifest["dataset_sha256"], "files": files}
    return build_lab_features(frame), proof


def daily_and_audit(simulation, model_sha: str, start: int, end: int, timeframe: int) -> tuple[np.ndarray, np.ndarray, dict]:
    portfolio = simulation.portfolio
    values, closes = complete_daily_returns(portfolio.timestamp.to_numpy(np.int64), portfolio.net_return.to_numpy(float),
                                             portfolio.equity.to_numpy(float), start=start, end=end, timeframe_ms=timeframe)
    if not np.allclose(values, daily_returns(simulation), rtol=0, atol=1e-14):
        raise ValueError("new daily aggregation differs from legacy aggregation")
    decisions = simulation.decisions
    if not decisions.model_sha256.eq(model_sha).all() or not (decisions.signal_timestamp <= decisions.execution_timestamp).all():
        raise ValueError("frozen decision SHA or causality violation")
    net = sum(row["net_pnl"] for row in simulation.trades.to_dict("records"))
    if abs(net - simulation.metrics["net_return"]) > 1e-12:
        raise ValueError("realized ledger does not reconcile with portfolio")
    evidence = {"metrics": simulation.metrics, "portfolio": portfolio.to_dict("records"),
                "trades": simulation.trades.to_dict("records"), "decisions": decisions.to_dict("records"),
                "events": simulation.events.to_dict("records"), "daily_returns": values.tolist(),
                "daily_close_timestamps": closes.tolist()}
    return values, closes, evidence


def origin(config: dict, output: Path, index: int) -> None:
    verify(config, output)
    parity_receipt = read_json(output / "adapter_parity_receipt.json")
    if parity_receipt["passed"] is not True or parity_receipt["adapter_sha256"] != file_sha(ADAPTER) or parity_receipt["total_bounds"] != 20480:
        raise ValueError("required qualified adapter parity absent")
    if index not in config["origins"]:
        raise ValueError("unregistered origin")
    target = output / f"origin_{index}.json"
    if target.exists() or (output / f"origin_{index}_started.json").exists():
        raise FileExistsError("origin already attempted; no automatic repeat")
    source_path = ROOT / config["source_lab_root"] / f"origin_{index}.json"
    source = read_json(source_path)
    source_config = read_json(ROOT / config["source_lab_config"])
    save_new(output / f"origin_{index}_started.json", {"config_sha256": file_sha(CONFIG), "source_sha256": file_sha(source_path),
             "parity_receipt_sha256": file_sha(output / "adapter_parity_receipt.json"),
             "created_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    bounds = source["bounds"]
    if object_sha(bounds) != source["bounds_sha256"] or source["config_sha256"] != config["source_lab_config_sha256"]:
        raise ValueError("source bounds/config SHA mismatch")
    start, end = bounds["validation_start"], bounds["validation_end"]
    frame, data_proof = frozen_features(source_config, end)
    timeframe = source_config["timeframe_minutes"] * 60_000
    window = PreparedWindow.from_frame(frame, tuple(source["symbols"]), start, end, timeframe)
    if array_sha(window.values) != source["validation_values_sha256"]:
        raise ValueError("reconstructed validation feature SHA mismatch")
    bank = np.asarray(source["validation_resample_bank"], dtype=np.int64)
    if array_sha(bank) != source["validation_bank_sha256"]:
        raise ValueError("legacy validation bank SHA mismatch")
    tolerance = config["reproduction_absolute_tolerance"]
    records, cache, model_files = [], {}, {}
    for model in source["methods"]:
        genome = tuple(model["winner"])
        scaler = LabScaler(**model["model"]["scaler"])
        assert_same(scaler.mapping(), source["scaler"], tolerance, "frozen_scaler")
        reconstructed_model = window.model(genome, scaler)
        if object_sha(reconstructed_model) != model["model_sha256"] or object_sha(model["model"]) != model["model_sha256"]:
            raise ValueError("frozen model SHA mismatch")
        model_sha = model["model_sha256"]
        if model_sha not in cache:
            normal = window.simulate(genome, scaler, source_config["costs_bps"]["base_round_trip"], details=True)
            stress = window.simulate(genome, scaler, source_config["costs_bps"]["stress_round_trip"], details=True)
            old_gate = validation_gate(normal, stress, bank, source_config["fitness"]["alpha"], config["maximum_drawdown"])
            normal_values, times, normal_payload = daily_and_audit(normal, model_sha, start, end, timeframe)
            stress_values, stress_times, stress_payload = daily_and_audit(stress, model_sha, start, end, timeframe)
            if not np.array_equal(times, stress_times):
                raise ValueError("normal/stress timestamps differ")
            normal_hac = assess_paired_daily(normal_values, np.zeros_like(normal_values), times)
            stress_hac = assess_paired_daily(stress_values, np.zeros_like(stress_values), times)
            gate = diagnostic_gate(normal_hac, stress_hac, normal.metrics["max_drawdown"], limit=config["maximum_drawdown"])
            file_name = f"origin_{index}_model_{model_sha}.json"
            payload = {"model": model["model"], "model_sha256": model_sha, "source_origin_sha256": file_sha(source_path),
                       "validation_start": start, "validation_end": end, "normal": normal_payload, "stress": stress_payload,
                       "normal_hac": normal_hac.mapping(), "stress_hac": stress_hac.mapping(),
                       "new_diagnostic_gate": gate, "safety": config["safety"]}
            cache[model_sha] = (normal, stress, old_gate, payload, file_name)
        normal, stress, old_gate, payload, file_name = cache[model_sha]
        # Reproduction must pass before accepting new statistical conclusions.
        assert_same(normal.metrics, model["validation"]["base"], tolerance, "legacy_normal_metrics")
        assert_same(stress.metrics, model["validation"]["stress"], tolerance, "legacy_stress_metrics")
        assert_same(old_gate, model["validation"]["gate"], tolerance, "legacy_gate")
        if file_name not in model_files:
            save_new(output / file_name, payload)
            model_files[file_name] = file_sha(output / file_name)
        records.append({"method": model["method"], "model_sha256": model_sha, "model_artifact": file_name,
                        "legacy_gate": old_gate, "normal_hac": payload["normal_hac"], "stress_hac": payload["stress_hac"],
                        "new_diagnostic_gate": payload["new_diagnostic_gate"], "original_metrics_reproduced": True,
                        "decision_would_change": old_gate["passed"] != payload["new_diagnostic_gate"]["passed"],
                        "frozen_active_genome_unchanged": model["active_genome"], "promotion_authority": False})
    payload = {"origin": index, "source_origin": source["origin"], "source_origin_sha256": file_sha(source_path),
               "config_sha256": file_sha(CONFIG), "data_proof": data_proof, "validation_values_sha256": array_sha(window.values),
               "legacy_bank_sha256": array_sha(bank), "records": records, "model_artifact_hashes": model_files,
               "logical_model_cost_reproductions": 2 * len(records), "physical_cash_simulations": 2 * len(cache),
               "training_runs": 0, "elapsed_seconds": time.perf_counter() - started, "safety": config["safety"]}
    save_new(target, payload)
    print(f"origin {index}/3 reproduced: unique_models={len(cache)} old_pass={sum(r['legacy_gate']['passed'] for r in records)} new_pass={sum(r['new_diagnostic_gate']['passed'] for r in records)}; seconds={payload['elapsed_seconds']:.2f}")


def finalize(config: dict, output: Path) -> None:
    verify(config, output)
    if (output / "summary.json").exists():
        raise FileExistsError("review already finalized")
    origins = [read_json(output / f"origin_{index}.json") for index in config["origins"]]
    for row in origins:
        for name, digest in row["model_artifact_hashes"].items():
            if file_sha(output / name) != digest:
                raise ValueError("reconstructed ledger artifact changed")
    records = [{"origin": row["origin"], **record} for row in origins for record in row["records"]]
    summary = {"schema": "exia.genetic_primus.frozen_hac_review_result/1", "records": records,
               "original_metrics_reproduced": all(r["original_metrics_reproduced"] for r in records),
               "legacy_passes": sum(r["legacy_gate"]["passed"] for r in records),
               "hac_diagnostic_passes": sum(r["new_diagnostic_gate"]["passed"] for r in records),
               "changed_diagnostic_decisions": sum(r["decision_would_change"] for r in records),
               "logical_model_cost_reproductions": sum(o["logical_model_cost_reproductions"] for o in origins),
               "physical_cash_simulations": sum(o["physical_cash_simulations"] for o in origins),
               "training_runs": 0, "legacy_artifacts_overwritten": False, "runtime_changed": False,
               "decision": "FROZEN_STATISTICAL_REVIEW_ONLY_NO_PROMOTION", "safety": config["safety"]}
    save_new(output / "summary.json", summary)
    hashes = {p.name: file_sha(p) for p in sorted(output.glob("*.json"))}
    save_new(output / "completion_receipt.json", {"artifacts": hashes,
             "completed_at_utc": datetime.now(timezone.utc).isoformat(), "safety": config["safety"]})
    print(f"complete: reproduced={len(records)}/9 legacy_pass={summary['legacy_passes']} hac_pass={summary['hac_diagnostic_passes']} changed={summary['changed_diagnostic_decisions']}; no training or promotion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "parity", "origin", "finalize"))
    parser.add_argument("--origin", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    config, output = load_contract()
    if args.phase == "prepare":
        prepare(config, output)
    elif args.phase == "parity":
        parity(config, output)
    elif args.phase == "origin":
        if args.origin is None:
            parser.error("origin phase requires --origin 1/2/3")
        origin(config, output, args.origin)
    else:
        finalize(config, output)


if __name__ == "__main__":
    main()
