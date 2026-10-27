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
from exia.genetic_primus.fitness_v3 import score_simulation_v3  # noqa: E402
from exia.genetic_primus.freeze import (  # noqa: E402
    plot_strict_slices,
    write_json,
    write_strict_workbook,
)
from exia.genetic_primus.freeze_v3 import build_v3_freeze_manifest  # noqa: E402
from exia.genetic_primus.policy import GeneticPrimusPolicy  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402
from tools.run_genetic_primus_v3 import CODE_PATHS, _composite_sha  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "genetic_primus_v3.json"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _load_scaler(path: Path) -> RobustScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RobustScaler(
        median=np.asarray(payload["median"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
    )


def _write_report(path: Path, manifest: dict[str, object]) -> None:
    metrics = manifest["metrics"]
    temporal = manifest["temporal_evidence"]
    failures = "\n".join(f"- `{value}`" for value in manifest["failed_gates"]) or "- none"
    path.write_text(
        f"""# Genetic_Primus V3 strict development freeze

Status: `{manifest['status']}`. Validation allowed:
`{str(manifest['validation_allowed']).lower()}`; validation was not opened by this command.

## Development evidence

- Selected/configured generation: {manifest['selected_generation']}/{manifest['configured_generations']}
- Closed trades: {int(metrics['trades'])}
- Mean / median / LCB: {metrics['mean_bps']:.4f} / {metrics['median_bps']:.4f} / {metrics['lcb_bps']:.4f} bps
- Temporal margin / monthly fitness: {metrics['temporal_margin_bps']:.4f} / {metrics['fitness']:.4f}
- Failed slices / worst slice LCB: {int(metrics['slice_failure_count'])} / {metrics['worst_supported_slice_lcb_bps']:.4f} bps
- Max drawdown: {metrics['max_drawdown']:.4%}
- Positive/nonpositive period scores: {temporal['positive_period_scores']}/{temporal['nonpositive_period_scores']}
- Positive/nonpositive monthly LCB: {temporal['positive_period_lcb']}/{temporal['nonpositive_period_lcb']}

## Failed gates

{failures}

## Immutable identity

- Champion genome SHA: `{manifest['champion_genome_sha256']}`
- Champion file SHA: `{manifest['champion_file_sha256']}`
- Scaler SHA: `{manifest['scaler_sha256']}`
- Dataset SHA: `{manifest['dataset_sha256']}`

Paper/live authority remains disabled.
""",
        encoding="utf-8",
    )


def run(*, run_dir: Path, config_path: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    config = load_primus_config(config_path, root=ROOT)
    manifest_path = run_dir / "manifest.json"
    champion_path = run_dir / "champion.npz"
    scaler_path = run_dir / "scaler.json"
    summary_path = run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DEVELOPMENT_V3_ONLY" or manifest.get("smoke") is not False:
        raise ValueError("V3 freeze requires a completed non-smoke development run")
    for flag in (
        "validation_opened", "oos_opened", "sanity_opened", "paper_allowed",
        "live_allowed", "orders_enabled", "promotion_authority",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"unsafe or opened V3 experiment flag: {flag}")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("V3 config SHA mismatch")
    if manifest.get("code_sha256") != _composite_sha(CODE_PATHS):
        raise ValueError("V3 training code SHA mismatch")
    if manifest.get("dataset_sha256") != config.dataset["dataset_sha256"]:
        raise ValueError("V3 dataset SHA mismatch")
    with np.load(champion_path, allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32).reshape(-1)
    genome_sha = hashlib.sha256(genome.tobytes()).hexdigest()
    scaler = _load_scaler(scaler_path)
    features = build_feature_frame(load_panel(config, root=ROOT))
    window = config.windows["development"]
    simulation = simulate_policy(
        features,
        policy=GeneticPrimusPolicy(genome),
        scaler=scaler,
        start_timestamp=_timestamp(window["start"]),
        end_timestamp=_timestamp(window["end"]),
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
    )
    score = score_simulation_v3(simulation, config=config.fitness, key=f"genome:{genome_sha}")
    freeze = build_v3_freeze_manifest(
        experiment_manifest=manifest,
        experiment_manifest_sha256=sha256_file(manifest_path),
        champion_file_sha256=sha256_file(champion_path),
        champion_genome_sha256=genome_sha,
        scaler_sha256=sha256_file(scaler_path),
        selected_generation=int(summary["selected_generation"]),
        configured_generations=int(config.evolution["generations"]),
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
    _write_report(run_dir / "strict_development_report.md", freeze)
    return {
        "run_dir": str(run_dir),
        "status": freeze["status"],
        "validation_allowed": freeze["validation_allowed"],
        "champion_genome_sha256": genome_sha,
        "metrics": dict(score.metrics),
        "temporal_evidence": freeze["temporal_evidence"],
        "failed_gates": freeze["failed_gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly freeze Genetic_Primus V3 development")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
