"""Offline final-only runner: preflight, commit complete segment, then evaluate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.future_ema_evaluator_v1 import (
    COLUMNS, evaluate_registered_panel, require_completed, utc_ms, validate_market_grid,
)
from exia.genetic_primus.measurement_lab_v2 import object_sha
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from tools.check_genetic_primus_future_ema_viability_v1 import validate_contract

CONFIG = ROOT / "configs/genetic_primus_future_ema_viability_v1.json"
CONFIG_SHA = "11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6"
LOCK = ROOT / "configs/genetic_primus_future_ema_implementation_v1.json"
DESTINATION = ROOT / "Reports/Exia/Genetic_Primus/future_ema_viability_v1_20261005"
MANIFEST = DESTINATION / "data_manifest.json"
RESERVATION = DESTINATION / "analysis_reservation.json"
RESULT = DESTINATION / "result.json"


def _write_exclusive(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("offline input must reside inside the workspace")
    return resolved


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("registered contract bytes changed")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock["schema_version"] != "exia.genetic_primus.future_ema_implementation/1"
            or lock["config_sha256"] != CONFIG_SHA or lock["objective_id"] != c["objective_id"]
            or utc_ms(lock["registered_at_utc"]) >= utc_ms(c["future_start_utc"])):
        raise ValueError("wrong or late implementation commitment")
    for relative, expected in lock["files_sha256"].items():
        if file_sha256(_inside(ROOT / relative)) != expected:
            raise ValueError(f"implementation changed: {relative}")
    manifest_path, contract_path = (ROOT / c["known_revealed_manifest_path"],
                                    ROOT / c["source_contract_path"])
    if (file_sha256(manifest_path) != c["known_revealed_manifest_sha256"]
            or file_sha256(contract_path) != c["source_contract_sha256"]):
        raise ValueError("source lineage changed")
    validate_contract(c, json.loads(manifest_path.read_text(encoding="utf-8")),
                      json.loads(contract_path.read_text(encoding="utf-8")))
    return c


def _read_segment(path: Path, c: dict) -> tuple[pd.DataFrame, str, int]:
    path = _inside(path)
    digest, size = file_sha256(path), path.stat().st_size
    frame = pd.read_parquet(path, columns=list(COLUMNS))
    frame = validate_market_grid(frame, tuple(c["universe"]),
                                 utc_ms(c["offline_future_segment_start_utc"]),
                                 utc_ms(c["future_end_exclusive_utc"]))
    if len(frame) != c["required_new_rows"]:
        raise ValueError("not the complete registered new segment")
    if file_sha256(path) != digest or path.stat().st_size != size:
        raise ValueError("segment changed during validation")
    return frame, digest, size


def commit_segment(c: dict, path: Path, now_utc: datetime) -> dict:
    require_completed(now_utc, utc_ms(c["future_end_exclusive_utc"]))
    if MANIFEST.exists() or RESERVATION.exists() or RESULT.exists():
        raise FileExistsError("future input or analysis already committed")
    frame, digest, size = _read_segment(path, c)
    manifest = {
        "schema_version": "exia.genetic_primus.future_ema_segment/1",
        "objective_id": c["objective_id"], "config_sha256": CONFIG_SHA,
        "implementation_lock_sha256": file_sha256(LOCK),
        "created_at_utc": now_utc.isoformat(), "immutable": True,
        "outcome_metrics_computed": False, "network_used": False,
        "source_snapshot_sha256": c["known_revealed_snapshot_sha256"],
        "segment": {"path": _inside(path).relative_to(ROOT).as_posix(),
                    "sha256": digest, "size_bytes": size, "row_count": len(frame),
                    "canonical_rows_sha256": object_sha(frame.to_dict("records"))},
        "safety": c["safety"],
    }
    _write_exclusive(MANIFEST, manifest)
    return {"status": "DATA_MANIFEST_PREPARED_COMMIT_BEFORE_ANALYSIS",
            "manifest_sha256": file_sha256(MANIFEST), "row_count": len(frame)}


def _require_git_commit(path: Path) -> None:
    relative = _inside(path).relative_to(ROOT).as_posix()
    saved = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                           capture_output=True, check=False)
    if saved.returncode or hashlib.sha256(saved.stdout).hexdigest() != file_sha256(path):
        raise ValueError("data manifest must be committed unchanged before final analysis")


def evaluate_once(c: dict, now_utc: datetime) -> dict:
    require_completed(now_utc, utc_ms(c["future_end_exclusive_utc"]))
    if RESERVATION.exists() or RESULT.exists():
        raise FileExistsError("single final analysis already started; inspect existing artifacts")
    _require_git_commit(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "exia.genetic_primus.future_ema_segment/1"
            or manifest.get("immutable") is not True
            or manifest.get("objective_id") != c["objective_id"]
            or manifest.get("config_sha256") != CONFIG_SHA
            or manifest.get("implementation_lock_sha256") != file_sha256(LOCK)
            or manifest.get("source_snapshot_sha256") != c["known_revealed_snapshot_sha256"]
            or manifest.get("safety") != c["safety"]
            or manifest.get("outcome_metrics_computed") is not False
            or manifest.get("network_used") is not False):
        raise ValueError("future segment commitment mismatch")
    segment = manifest["segment"]
    future, digest, size = _read_segment(ROOT / segment["path"], c)
    if (digest != segment["sha256"] or size != segment["size_bytes"]
            or len(future) != segment["row_count"]
            or object_sha(future.to_dict("records")) != segment["canonical_rows_sha256"]):
        raise ValueError("future market bytes/rows differ from commitment")
    source_contract_path = ROOT / c["source_contract_path"]
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    _, history = load_verified_snapshot(
        ROOT / c["known_revealed_manifest_path"], root=ROOT,
        contract_path=source_contract_path, contract=source_contract)
    commitment = {"objective_id": c["objective_id"], "config_sha256": CONFIG_SHA,
                  "data_manifest_sha256": file_sha256(MANIFEST),
                  "implementation_lock_sha256": file_sha256(LOCK),
                  "analysis_started_at_utc": now_utc.isoformat()}
    _write_exclusive(RESERVATION, commitment)
    try:
        result = evaluate_registered_panel(c, history, future)
        result.update({"schema_version": "exia.genetic_primus.future_ema_result/1",
                       "evidence_class": "registered_future_baseline_screen",
                       **commitment, "safety": c["safety"],
                       "market_readiness": False, "network_used": False,
                       "global_adaptive_trials_DSR_PBO_calibrated": False})
        _write_exclusive(RESULT, result)
    except Exception as exc:
        _write_exclusive(DESTINATION / "analysis_failure.json",
                         {**commitment, "exception_type": type(exc).__name__,
                          "promotion_authority": False})
        raise
    return {"status": result["research_status"], "gate_passed": result["gate"]["passed"],
            "promotion_authority": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "commit", "evaluate"))
    parser.add_argument("--segment", type=Path)
    args = parser.parse_args()
    c = _config()
    now = datetime.now(timezone.utc)
    if args.mode == "preflight":
        status = {"status": "REGISTERED_AWAITING_NEW_OFFLINE_DATA",
                  "earliest_final_analysis_utc": c["future_end_exclusive_utc"],
                  "outcome_metrics_computed": False, "automatic_evaluation": False}
    elif int(now.timestamp() * 1000) < utc_ms(c["future_end_exclusive_utc"]):
        print(json.dumps({"status": "PENDING_INTERVAL_COMPLETE",
                          "earliest_final_analysis_utc": c["future_end_exclusive_utc"],
                          "market_files_opened": False, "outcome_metrics_computed": False},
                         separators=(",", ":")))
        raise SystemExit(2)
    elif args.mode == "commit":
        if args.segment is None:
            parser.error("commit requires --segment inside the workspace")
        status = commit_segment(c, args.segment, now)
    else:
        status = evaluate_once(c, now)
    print(json.dumps(status, separators=(",", ":")))


if __name__ == "__main__":
    main()
