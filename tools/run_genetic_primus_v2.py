from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exia.genetic_primus.contracts import STATIC_FEATURE_NAMES, load_primus_config, sha256_file  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.evolution_v2 import run_evolution_v2  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.reporting import (  # noqa: E402
    save_generation_champion,
    write_evidence_bundle,
    write_progress_snapshot,
)
from exia.genetic_primus.reporting_v2 import plot_gate_progress  # noqa: E402
from tools.run_genetic_primus_v1 import CODE_PATHS as V1_CODE_PATHS  # noqa: E402
from tools.run_genetic_primus_v1 import _composite_sha as v1_composite_sha  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "genetic_primus_v2.json"
DEFAULT_PARENT_RUN = ROOT / "Reports" / "Exia" / "Genetic_Primus" / "development_v1_20260821"
EXPECTED_PARENT_GENOME_SHA = "fe30dd4307c65de112282ee5cbc715c561964dcfd63f1b7a30d59d52050a09a3"
EXPECTED_PARENT_SCALER_SHA = "5be04193487f9a83e355f387066850cff83f1a535e18f75f405f841defcb0793"
CODE_PATHS = (
    Path("src/exia/genetic_primus/contracts.py"),
    Path("src/exia/genetic_primus/data.py"),
    Path("src/exia/genetic_primus/features.py"),
    Path("src/exia/genetic_primus/policy.py"),
    Path("src/exia/genetic_primus/simulator.py"),
    Path("src/exia/genetic_primus/fitness.py"),
    Path("src/exia/genetic_primus/fitness_v2.py"),
    Path("src/exia/genetic_primus/evolution.py"),
    Path("src/exia/genetic_primus/evolution_v2.py"),
    Path("src/exia/genetic_primus/reporting.py"),
    Path("src/exia/genetic_primus/reporting_v2.py"),
    Path("tools/run_genetic_primus_v2.py"),
)


def _composite_sha(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def _load_parent(parent_run: Path) -> tuple[np.ndarray, RobustScaler, dict[str, object]]:
    manifest_path = parent_run / "manifest.json"
    champion_path = parent_run / "champion.npz"
    scaler_path = parent_run / "scaler.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "DEVELOPMENT_ONLY":
        raise ValueError("V2 parent must be a completed V1 development run")
    if manifest.get("code_sha256") != v1_composite_sha(V1_CODE_PATHS):
        raise ValueError("V1 parent training code SHA mismatch")
    if sha256_file(scaler_path) != EXPECTED_PARENT_SCALER_SHA:
        raise ValueError("V1 parent scaler SHA mismatch")
    scaler_payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = RobustScaler(
        median=np.asarray(scaler_payload["median"], dtype=np.float64),
        scale=np.asarray(scaler_payload["scale"], dtype=np.float64),
    )
    with np.load(champion_path, allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32).reshape(-1)
    genome_sha = hashlib.sha256(genome.tobytes()).hexdigest()
    if genome_sha != EXPECTED_PARENT_GENOME_SHA:
        raise ValueError("V1 parent champion genome SHA mismatch")
    identity = {
        "run": str(parent_run.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "champion_file_sha256": sha256_file(champion_path),
        "champion_genome_sha256": genome_sha,
        "scaler_sha256": sha256_file(scaler_path),
    }
    return genome, scaler, identity


def run(
    *,
    config_path: Path,
    parent_run: Path,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, object]:
    config = load_primus_config(config_path, root=ROOT)
    parent_genome, scaler, parent_identity = _load_parent(parent_run)
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    development = config.windows["development"]
    start_timestamp = _timestamp(development["start"])
    configured_end = _timestamp(development["end"])
    end_timestamp = min(
        configured_end,
        start_timestamp + int(pd.Timedelta(days=180).total_seconds() * 1000),
    ) if smoke else configured_end
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory is not empty; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaler.json").write_text(
        json.dumps(scaler.to_mapping(), indent=2) + "\n", encoding="utf-8"
    )
    code_sha = _composite_sha(CODE_PATHS)
    run_fingerprint = hashlib.sha256(
        (
            sha256_file(config_path)
            + code_sha
            + str(config.dataset["dataset_sha256"])
            + str(parent_identity["champion_genome_sha256"])
            + str(smoke)
        ).encode("utf-8")
    ).hexdigest()

    def generation_callback(champion, history) -> None:
        save_generation_champion(output_dir, champion)
        write_progress_snapshot(output_dir, history)
        plot_gate_progress(history, output_dir / "strict_gate_progress_live.png")
        metrics = champion.score.metrics
        print(
            json.dumps(
                {
                    "generation": champion.generation,
                    "trades": metrics["trades"],
                    "mean_bps": metrics["mean_bps"],
                    "median_bps": metrics["median_bps"],
                    "lcb_bps": metrics["lcb_bps"],
                    "max_drawdown": metrics["max_drawdown"],
                    "overall_failure_count": metrics["overall_failure_count"],
                    "slice_failure_count": metrics["slice_failure_count"],
                    "worst_slice_lcb_bps": metrics["worst_supported_slice_lcb_bps"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = run_evolution_v2(
        features,
        scaler=scaler,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
        fitness_config=config.fitness,
        evolution_config=config.evolution,
        warm_start=parent_genome,
        population_size=6 if smoke else None,
        generations=2 if smoke else None,
        on_generation=generation_callback,
        checkpoint_path=output_dir / "evolution_checkpoint.npz",
        resume=resume,
        run_fingerprint=run_fingerprint,
    )
    manifest = {
        "schema_version": "exia.genetic_primus_experiment.v2",
        "agent_id": "Genetic_Primus",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ENGINEERING_SMOKE_V2" if smoke else "DEVELOPMENT_V2_ONLY",
        "selection_contract": "risk>overall_gate>failed_slices>worst_slice_lcb>monthly_fitness",
        "config_sha256": sha256_file(config_path),
        "code_sha256": code_sha,
        "source_revision": _git_revision(),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "parent_v1": parent_identity,
        "development_start": pd.Timestamp(start_timestamp, unit="ms", tz="UTC").isoformat(),
        "development_end": pd.Timestamp(end_timestamp, unit="ms", tz="UTC").isoformat(),
        "timeframe_minutes": 240,
        "network": dict(config.network),
        "feature_names": list(STATIC_FEATURE_NAMES) + [
            "position_direction", "unrealized_net_bps", "holding_bars", "mfe_bps", "mae_bps"
        ],
        "costs_bps": dict(config.costs),
        "smoke": bool(smoke),
        "resumed": bool(resume),
        "run_fingerprint": run_fingerprint,
        "validation_opened": False,
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    write_evidence_bundle(output_dir, result=result, manifest=manifest)
    plot_gate_progress(result.generations, output_dir / "strict_gate_progress.png")
    return {
        "output_dir": str(output_dir),
        "status": manifest["status"],
        "selected_generation": result.champion.generation,
        "metrics": dict(result.champion.score.metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genetic_Primus V2 development evolution")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-run", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = "smoke_v2" if args.smoke else "development_v2"
        output_dir = ROOT / "Reports" / "Exia" / "Genetic_Primus" / f"{suffix}_{stamp}"
    payload = run(
        config_path=args.config.resolve(),
        parent_run=args.parent_run.resolve(),
        output_dir=output_dir.resolve(),
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
