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

from exia.genetic_primus.contracts import STATIC_FEATURE_NAMES, load_primus_config, sha256_file  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.evolution import run_evolution  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.labels import build_forward_labels  # noqa: E402
from exia.genetic_primus.policy import supervised_seed  # noqa: E402
from exia.genetic_primus.reporting import (  # noqa: E402
    save_generation_champion,
    write_evidence_bundle,
    write_progress_snapshot,
)


DEFAULT_CONFIG = ROOT / "configs" / "genetic_primus_v1.json"
CODE_PATHS = (
    Path("src/exia/genetic_primus/contracts.py"),
    Path("src/exia/genetic_primus/data.py"),
    Path("src/exia/genetic_primus/features.py"),
    Path("src/exia/genetic_primus/labels.py"),
    Path("src/exia/genetic_primus/policy.py"),
    Path("src/exia/genetic_primus/simulator.py"),
    Path("src/exia/genetic_primus/fitness.py"),
    Path("src/exia/genetic_primus/evolution.py"),
    Path("src/exia/genetic_primus/reporting.py"),
    Path("tools/run_genetic_primus_v1.py"),
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def run(
    *,
    config_path: Path,
    output_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict[str, object]:
    config = load_primus_config(config_path, root=ROOT)
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    development = config.windows["development"]
    start_timestamp = _timestamp(development["start"])
    configured_end = _timestamp(development["end"])
    end_timestamp = min(
        configured_end,
        start_timestamp + int(pd.Timedelta(days=180).total_seconds() * 1000),
    ) if smoke else configured_end
    development_mask = (
        (features["timestamp"] >= start_timestamp)
        & (features["timestamp"] < end_timestamp)
        & features["feature_supported"]
    )
    scaler = RobustScaler.fit(features.loc[development_mask])

    labels = build_forward_labels(
        panel,
        horizons_bars=tuple(int(value) for value in config.labels["horizons_bars"]),
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
        minimum_edge_bps=float(config.labels["minimum_edge_bps"]),
        minimum_votes=int(config.labels["minimum_votes"]),
    )
    label_lookup = labels.loc[:, ["timestamp", "symbol", "target_position", "label_supported"]]
    training = features.merge(label_lookup, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    seed_mask = development_mask & training["label_supported"].fillna(False)
    static = scaler.transform(training.loc[seed_mask])
    seed_features = np.concatenate(
        [static, np.zeros((len(static), 5), dtype=np.float32)], axis=1
    )
    seed_targets = training.loc[seed_mask, "target_position"].to_numpy(dtype=np.int8)
    if len(seed_features) > 50_000:
        rng = np.random.default_rng(int(config.evolution["seed"]))
        chosen = np.sort(rng.choice(len(seed_features), size=50_000, replace=False))
        seed_features = seed_features[chosen]
        seed_targets = seed_targets[chosen]
    warm_policy = supervised_seed(
        seed_features,
        seed_targets,
        seed=int(config.evolution["seed"]),
        epochs=3 if smoke else 30,
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"output directory is not empty; use --resume or a new directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaler.json").write_text(
        json.dumps(scaler.to_mapping(), indent=2) + "\n", encoding="utf-8"
    )
    label_summary = {
        "samples": int(len(seed_targets)),
        "short": int((seed_targets == -1).sum()),
        "flat": int((seed_targets == 0).sum()),
        "long": int((seed_targets == 1).sum()),
        "future_used_as_feature": False,
    }
    (output_dir / "label_summary.json").write_text(
        json.dumps(label_summary, indent=2) + "\n", encoding="utf-8"
    )

    run_fingerprint = hashlib.sha256(
        (
            sha256_file(config_path)
            + _composite_sha(CODE_PATHS)
            + str(config.dataset["dataset_sha256"])
            + str(smoke)
        ).encode("utf-8")
    ).hexdigest()

    def generation_callback(champion, history) -> None:
        save_generation_champion(output_dir, champion)
        write_progress_snapshot(output_dir, history)
        metrics = champion.score.metrics
        print(
            json.dumps(
                {
                    "generation": champion.generation,
                    "fitness": champion.score.fitness,
                    "selection_score": champion.score.selection_score,
                    "constraint_violation": champion.score.constraint_violation,
                    "trades": metrics["trades"],
                    "mean_bps": metrics["mean_bps"],
                    "median_bps": metrics["median_bps"],
                    "lcb_bps": metrics["lcb_bps"],
                    "max_drawdown": metrics["max_drawdown"],
                    "failed_gates": metrics["failed_gates"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = run_evolution(
        features,
        scaler=scaler,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        round_trip_cost_bps=float(config.costs["stress_round_trip"]),
        fitness_config=config.fitness,
        evolution_config=config.evolution,
        warm_start=warm_policy.genome,
        population_size=6 if smoke else None,
        generations=2 if smoke else None,
        on_generation=generation_callback,
        checkpoint_path=output_dir / "evolution_checkpoint.npz",
        resume=resume,
        run_fingerprint=run_fingerprint,
    )
    manifest = {
        "schema_version": "exia.genetic_primus_experiment.v1",
        "agent_id": "Genetic_Primus",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ENGINEERING_SMOKE" if smoke else "DEVELOPMENT_ONLY",
        "config_sha256": sha256_file(config_path),
        "code_sha256": _composite_sha(CODE_PATHS),
        "source_revision": _git_revision(),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "development_start": pd.Timestamp(start_timestamp, unit="ms", tz="UTC").isoformat(),
        "development_end": pd.Timestamp(end_timestamp, unit="ms", tz="UTC").isoformat(),
        "timeframe_minutes": 240,
        "network": dict(config.network),
        "feature_names": list(STATIC_FEATURE_NAMES) + [
            "position_direction", "unrealized_net_bps", "holding_bars", "mfe_bps", "mae_bps"
        ],
        "label_summary": label_summary,
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
    return {
        "output_dir": str(output_dir),
        "status": manifest["status"],
        "selected_generation": result.champion.generation,
        "fitness": result.champion.score.fitness,
        "metrics": dict(result.champion.score.metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genetic_Primus V1 development evolution")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = ROOT / "Reports" / "Exia" / "Genetic_Primus" / f"development_{stamp}"
    payload = run(
        config_path=args.config.resolve(),
        output_dir=output_dir.resolve(),
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
