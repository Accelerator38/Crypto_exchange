from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exia.genetic_primus.contracts import load_primus_config, sha256_file  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.fitness import score_simulation  # noqa: E402
from exia.genetic_primus.freeze import (  # noqa: E402
    build_freeze_manifest,
    plot_strict_slices,
    write_json,
    write_strict_report,
    write_strict_workbook,
)
from exia.genetic_primus.policy import GeneticPrimusPolicy  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402
from tools.run_genetic_primus_v1 import CODE_PATHS, _composite_sha  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "genetic_primus_v1.json"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"median", "scale"}:
        raise ValueError("invalid Genetic_Primus scaler schema")
    return RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def _verify_experiment(
    *, run_dir: Path, config_path: Path
) -> tuple[dict[str, object], str, str, str]:
    manifest_path = run_dir / "manifest.json"
    champion_path = run_dir / "champion.npz"
    scaler_path = run_dir / "scaler.json"
    for path in (manifest_path, champion_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DEVELOPMENT_ONLY":
        raise ValueError("strict freeze requires a completed DEVELOPMENT_ONLY run")
    if manifest.get("smoke") is not False:
        raise ValueError("engineering smoke cannot be frozen as development evidence")
    for flag in (
        "validation_opened",
        "oos_opened",
        "sanity_opened",
        "paper_allowed",
        "live_allowed",
        "orders_enabled",
        "promotion_authority",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"unsafe or opened experiment flag: {flag}")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("experiment config SHA does not match current frozen config")
    if manifest.get("code_sha256") != _composite_sha(CODE_PATHS):
        raise ValueError("experiment code SHA does not match current V1 training code")
    experiment_sha = sha256_file(manifest_path)
    champion_sha = sha256_file(champion_path)
    scaler_sha = sha256_file(scaler_path)
    return manifest, experiment_sha, champion_sha, scaler_sha


def run(*, run_dir: Path, config_path: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    config = load_primus_config(config_path, root=ROOT)
    manifest, experiment_sha, champion_sha, scaler_sha = _verify_experiment(
        run_dir=run_dir, config_path=config_path
    )
    if manifest["dataset_sha256"] != config.dataset["dataset_sha256"]:
        raise ValueError("experiment dataset SHA does not match config")

    with np.load(run_dir / "champion.npz", allow_pickle=False) as payload:
        if set(payload.files) != {"genome"}:
            raise ValueError("invalid champion payload")
        genome = np.asarray(payload["genome"], dtype=np.float32).reshape(-1)
    genome_sha = hashlib.sha256(genome.tobytes()).hexdigest()
    scaler = _load_scaler(run_dir / "scaler.json")
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    window = config.windows["development"]
    simulation = simulate_policy(
        features,
        policy=GeneticPrimusPolicy(genome),
        scaler=scaler,
        start_timestamp=_timestamp(window["start"]),
        end_timestamp=_timestamp(window["end"]),
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
    )
    score = score_simulation(
        simulation,
        config=config.fitness,
        key=f"genome:{genome_sha}",
        final_gate=True,
    )
    freeze = build_freeze_manifest(
        experiment_manifest=manifest,
        experiment_manifest_sha256=experiment_sha,
        champion_file_sha256=champion_sha,
        champion_genome_sha256=genome_sha,
        scaler_sha256=scaler_sha,
        score=score,
    )

    score.periods.to_parquet(run_dir / "strict_development_periods.parquet", index=False)
    score.slices.to_parquet(run_dir / "strict_development_slices.parquet", index=False)
    write_json(run_dir / "strict_development_metrics.json", dict(score.metrics))
    write_json(run_dir / "freeze_manifest.json", freeze)
    write_strict_workbook(
        run_dir / "strict_development_tables.xlsx",
        manifest=freeze,
        periods=score.periods,
        slices=score.slices,
    )
    plot_strict_slices(score.slices, run_dir / "strict_development_slices.png")
    write_strict_report(
        run_dir / "strict_development_report.md",
        manifest=freeze,
        slices=score.slices,
    )
    return {
        "run_dir": str(run_dir),
        "status": freeze["status"],
        "validation_allowed": freeze["validation_allowed"],
        "champion_genome_sha256": genome_sha,
        "metrics": dict(score.metrics),
        "failed_gates": list(score.failed_gates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly freeze a Genetic_Primus V1 development champion"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(run_dir=args.run_dir, config_path=args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
