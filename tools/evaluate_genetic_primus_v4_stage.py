from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import load_workbook
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import sha256_file  # noqa: E402
from exia.genetic_primus.contracts_v4 import load_v4_config  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.fitness_v4 import score_simulation_v4  # noqa: E402
from exia.genetic_primus.policy_v4 import GeneticPrimusPolicyV4, V4NetworkShape  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/genetic_primus_v4.json"
DEFAULT_RUN = ROOT / "Reports/Exia/Genetic_Primus/v4_final_20260824"


def _composite_sha(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def assert_stage_authority(run_dir: Path, stage: str) -> dict[str, Any]:
    if stage == "validation":
        authority_path = run_dir / "freeze_manifest.json"
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        if authority.get("status") != "DEVELOPMENT_V4_PASSED_STRICT_GATES":
            raise PermissionError("V4 validation is blocked by rejected development freeze")
        if authority.get("validation_allowed") is not True or authority.get("validation_opened") is not False:
            raise PermissionError("V4 validation authority is absent or already consumed")
        return authority
    validation_path = run_dir / "validation_manifest.json"
    authority = json.loads(validation_path.read_text(encoding="utf-8"))
    if authority.get("status") != "VALIDATION_V4_PASSED_STRICT_GATES":
        raise PermissionError("V4 OOS is blocked by validation status")
    if authority.get("oos_allowed") is not True or authority.get("oos_opened") is not False:
        raise PermissionError("V4 OOS authority is absent or already consumed")
    return authority


def _stage_folds(stage: str) -> list[dict[str, str]]:
    if stage == "validation":
        return [
            {"name": "2025_H1", "start": "2025-01-01T00:00:00Z", "end": "2025-07-01T00:00:00Z"},
            {"name": "2025_H2", "start": "2025-07-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
        ]
    return [
        {"name": "2026_Q1", "start": "2026-01-01T00:00:00Z", "end": "2026-04-01T00:00:00Z"},
        {"name": "2026_Q2_PART", "start": "2026-04-01T00:00:00Z", "end": "2026-07-15T00:00:00Z"},
    ]


def _workbook(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.book[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    workbook = load_workbook(path, read_only=False)
    for name, frame in frames.items():
        sheet = workbook[name]
        if sheet.max_column != len(frame.columns) or sheet.max_column <= 1 or sheet.merged_cells.ranges:
            raise RuntimeError(f"invalid stage workbook: {name}")
    workbook.close()


def run(config_path: Path, run_dir: Path, stage: str) -> dict[str, Any]:
    authority = assert_stage_authority(run_dir, stage)
    output_manifest = run_dir / f"{stage}_manifest.json"
    if output_manifest.exists():
        raise FileExistsError(f"one-time V4 {stage} has already been opened")
    config = load_v4_config(config_path, root=ROOT)
    freeze = json.loads((run_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(config_path) != freeze["config_sha256"]:
        raise ValueError("V4 config SHA changed after freeze")
    if sha256_file(run_dir / "champion.npz") != freeze["champion_file_sha256"]:
        raise ValueError("V4 champion file SHA changed after freeze")
    if sha256_file(run_dir / "scaler.json") != freeze["scaler_sha256"]:
        raise ValueError("V4 scaler SHA changed after freeze")
    feature_sha = _composite_sha((Path("src/exia/genetic_primus/contracts.py"), Path("src/exia/genetic_primus/features.py")))
    execution_sha = _composite_sha((Path("src/exia/genetic_primus/policy_v4.py"), Path("src/exia/genetic_primus/simulator.py")))
    if feature_sha != freeze["feature_contract_sha256"] or execution_sha != freeze["execution_contract_sha256"]:
        raise ValueError("V4 feature/execution contract changed after freeze")
    scaler_payload = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))
    scaler = RobustScaler(np.asarray(scaler_payload["median"]), np.asarray(scaler_payload["scale"]))
    with np.load(run_dir / "champion.npz", allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32)
    architecture = freeze["architecture"]
    shape = V4NetworkShape(30, tuple(architecture["hidden"]), 3)
    features = build_feature_frame(load_panel(config, root=ROOT))
    window = config.payload["windows"][stage]
    folds = _stage_folds(stage)
    strict = dict(config.fitness)
    strict["bootstrap_samples"] = int(config.fitness["strict_bootstrap_samples"])
    strict["minimum_positive_fold_means"] = len(folds)
    scores = {}
    frames = {}
    for label, cost in (("base16", 16.0), ("stress24", 24.0)):
        simulation = simulate_policy(
            features,
            policy=GeneticPrimusPolicyV4(genome, shape=shape),
            scaler=scaler,
            start_timestamp=int(pd.Timestamp(window["start"]).timestamp() * 1000),
            end_timestamp=int(pd.Timestamp(window["end"]).timestamp() * 1000),
            round_trip_cost_bps=cost,
        )
        score = score_simulation_v4(simulation, folds=folds, config=strict, key=f"v4:{stage}:{freeze['champion_genome_sha256']}:{label}")
        scores[label] = score
        frames[f"{label} Folds"] = score.folds
    passed = all(not score.failed_gates for score in scores.values())
    status = f"{stage.upper()}_V4_{'PASSED' if passed else 'REJECTED'}_STRICT_GATES"
    manifest = {
        "schema_version": f"exia.genetic_primus_{stage}.v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "authority_sha256": sha256_file(run_dir / ("freeze_manifest.json" if stage == "validation" else "validation_manifest.json")),
        "champion_genome_sha256": freeze["champion_genome_sha256"],
        "window": window,
        "base16_metrics": dict(scores["base16"].metrics),
        "stress24_metrics": dict(scores["stress24"].metrics),
        "oos_allowed": bool(passed and stage == "validation"),
        "oos_opened": False,
        "paper_canary_allowed": bool(passed and stage == "oos"),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    _workbook(run_dir / f"{stage}_tables.xlsx", frames)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Open one-time Genetic_Primus V4 validation/OOS")
    parser.add_argument("--stage", choices=("validation", "oos"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_dir.resolve(), args.stage), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
