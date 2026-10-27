from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import load_primus_config, sha256_file  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.fitness_v3 import score_simulation_v3  # noqa: E402
from exia.genetic_primus.freeze import (  # noqa: E402
    plot_strict_slices,
    write_json,
    write_strict_workbook,
)
from exia.genetic_primus.freeze_v3 import v3_freeze_failures  # noqa: E402
from exia.genetic_primus.policy import GeneticPrimusPolicy  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "genetic_primus_v3.json"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def run(*, run_dir: Path, config_path: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    config = load_primus_config(config_path.resolve(), root=ROOT)
    validation_manifest_path = run_dir / "validation_manifest.json"
    if validation_manifest_path.exists():
        raise FileExistsError("V3 validation is one-time and has already been opened")
    freeze_path = run_dir / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "DEVELOPMENT_V3_PASSED_STRICT_GATES":
        raise ValueError("development freeze does not authorize V3 validation")
    if freeze.get("validation_allowed") is not True or freeze.get("validation_opened") is not False:
        raise ValueError("invalid V3 validation authority state")
    for flag in ("paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if freeze.get(flag) is not False:
            raise ValueError(f"unsafe V3 freeze flag: {flag}")

    champion_path = run_dir / "champion.npz"
    scaler_path = run_dir / "scaler.json"
    with np.load(champion_path, allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32).reshape(-1)
    genome_sha = hashlib.sha256(genome.tobytes()).hexdigest()
    if genome_sha != freeze["champion_genome_sha256"]:
        raise ValueError("frozen V3 champion genome SHA mismatch")
    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float64),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float64),
    )
    features = build_feature_frame(load_panel(config, root=ROOT))
    window = config.windows["validation"]
    simulation = simulate_policy(
        features,
        policy=GeneticPrimusPolicy(genome),
        scaler=scaler,
        start_timestamp=_timestamp(window["start"]),
        end_timestamp=_timestamp(window["end"]),
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
    )
    score = score_simulation_v3(
        simulation,
        config=config.fitness,
        key=f"genome:{genome_sha}:validation",
    )
    failures = v3_freeze_failures(score)
    status = "VALIDATION_V3_PASSED_STRICT_GATES" if not failures else "VALIDATION_V3_REJECTED"
    periods = score.periods.loc[score.periods["supported"]]
    manifest = {
        "schema_version": "exia.genetic_primus_validation.v3",
        "agent_id": "Genetic_Primus",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "development_freeze_sha256": sha256_file(freeze_path),
        "champion_genome_sha256": genome_sha,
        "champion_file_sha256": sha256_file(champion_path),
        "scaler_sha256": sha256_file(scaler_path),
        "dataset_sha256": freeze["dataset_sha256"],
        "split": "validation",
        "window": dict(window),
        "cost_profile": "stress_round_trip",
        "metrics": dict(score.metrics),
        "temporal_evidence": {
            "supported_periods": int(len(periods)),
            "positive_period_scores": int((periods["period_score"] > 0.0).sum()),
            "nonpositive_period_scores": int((periods["period_score"] <= 0.0).sum()),
            "positive_period_lcb": int((periods["lcb_bps"] > 0.0).sum()),
            "nonpositive_period_lcb": int((periods["lcb_bps"] <= 0.0).sum()),
            "median_period_score": float(score.fitness),
        },
        "failed_gates": list(failures),
        "validation_allowed": True,
        "validation_opened": True,
        "oos_allowed": status == "VALIDATION_V3_PASSED_STRICT_GATES",
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    score.periods.to_parquet(run_dir / "validation_periods.parquet", index=False)
    score.slices.to_parquet(run_dir / "validation_slices.parquet", index=False)
    write_json(run_dir / "validation_metrics.json", dict(score.metrics))
    write_strict_workbook(
        run_dir / "validation_tables.xlsx",
        manifest=manifest,
        periods=score.periods,
        slices=score.slices,
    )
    plot_strict_slices(
        score.slices,
        run_dir / "validation_slices.png",
        title="Genetic_Primus V3 one-time validation slices",
    )
    write_json(validation_manifest_path, manifest)
    (run_dir / "validation_report.md").write_text(
        "\n".join(
            [
                "# Genetic_Primus V3 one-time validation",
                "",
                f"Status: `{status}`.",
                "",
                f"- Closed trades: {int(score.metrics['trades'])}",
                f"- Mean / median / LCB: {score.metrics['mean_bps']:.4f} / {score.metrics['median_bps']:.4f} / {score.metrics['lcb_bps']:.4f} bps",
                f"- Temporal margin / fitness: {score.metrics['temporal_margin_bps']:.4f} / {score.fitness:.4f}",
                f"- Failed slices / worst LCB: {int(score.metrics['slice_failure_count'])} / {score.metrics['worst_supported_slice_lcb_bps']:.4f} bps",
                f"- Max drawdown: {score.metrics['max_drawdown']:.4%}",
                f"- Failed gates: {', '.join(failures) or 'none'}",
                "",
                "OOS, paper and live were not opened.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "status": status,
        "oos_allowed": manifest["oos_allowed"],
        "metrics": dict(score.metrics),
        "temporal_evidence": manifest["temporal_evidence"],
        "failed_gates": list(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Open one-time Genetic_Primus V3 validation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(run_dir=args.run_dir, config_path=args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
