from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import FEATURE_NAMES, STATIC_FEATURE_NAMES, sha256_file  # noqa: E402
from exia.genetic_primus.contracts_v4 import PrimusV4Config, load_v4_config  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.evolution_v4 import V4EvolutionResult, run_evolution_v4  # noqa: E402
from exia.genetic_primus.features import RobustScaler, build_feature_frame  # noqa: E402
from exia.genetic_primus.fitness_v4 import score_simulation_v4  # noqa: E402
from exia.genetic_primus.labels import build_forward_labels  # noqa: E402
from exia.genetic_primus.measurement import (  # noqa: E402
    BootstrapBank,
    MeasurementProtocol,
    build_bootstrap_bank,
    folds_sha256,
    label_purge_mask,
    load_measurement_protocol,
    save_bootstrap_bank,
)
from exia.genetic_primus.policy_v4 import (  # noqa: E402
    GeneticPrimusPolicyV4,
    V4NetworkShape,
    supervised_seed_v4,
)
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/genetic_primus_v4.json"
DEFAULT_MEASUREMENT = ROOT / "configs/genetic_primus_measurement_3d_v1.json"
DEFAULT_SCREEN = ROOT / "Reports/Exia/Genetic_Primus/v4_architecture_screen_20260824"
DEFAULT_FINAL = ROOT / "Reports/Exia/Genetic_Primus/v4_final_20260824"
CODE_PATHS = (
    Path("src/exia/genetic_primus/contracts.py"),
    Path("src/exia/genetic_primus/contracts_v4.py"),
    Path("src/exia/genetic_primus/features.py"),
    Path("src/exia/genetic_primus/labels.py"),
    Path("src/exia/genetic_primus/measurement.py"),
    Path("src/exia/genetic_primus/policy_v4.py"),
    Path("src/exia/genetic_primus/simulator.py"),
    Path("src/exia/genetic_primus/fitness_v4.py"),
    Path("src/exia/genetic_primus/evolution.py"),
    Path("src/exia/genetic_primus/evolution_v4.py"),
    Path("tools/run_genetic_primus_v4.py"),
)


def _composite_sha(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode())
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


def _shape(item: Mapping[str, Any]) -> V4NetworkShape:
    return V4NetworkShape(
        inputs=int(item["inputs"]), hidden=tuple(int(value) for value in item["hidden"]), outputs=int(item["outputs"])
    )


def _prepare(config: PrimusV4Config, measurement: MeasurementProtocol):
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    research = config.payload["windows"]["research"]
    if measurement.cutoff_ms != _timestamp(research["end"]):
        raise ValueError("measurement cutoff does not match the sealed V4 research end")
    mask = (features["timestamp"] >= _timestamp(research["start"])) & (features["timestamp"] < _timestamp(research["end"]))
    scaler = RobustScaler.fit(features.loc[mask])
    labels = build_forward_labels(
        panel,
        horizons_bars=tuple(config.payload["labels"]["horizons_bars"]),
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        minimum_edge_bps=float(config.payload["labels"]["minimum_edge_bps"]),
        minimum_votes=int(config.payload["labels"]["minimum_votes"]),
    )
    if not features[["timestamp", "symbol"]].equals(labels[["timestamp", "symbol"]]):
        raise ValueError("V4 feature/label rows are not aligned")
    static = scaler.transform(features)
    purged = label_purge_mask(features["timestamp"], measurement)
    supervised_mask = mask & purged & features["feature_supported"] & labels["label_supported"]
    x = np.concatenate(
        [static[supervised_mask.to_numpy()], np.zeros((int(supervised_mask.sum()), 5), dtype=np.float32)], axis=1
    )
    y = labels.loc[supervised_mask, "target_position"].to_numpy(dtype=np.int8)
    return panel, features, scaler, x, y


def _bootstrap_banks(
    features: pd.DataFrame,
    config: PrimusV4Config,
    measurement: MeasurementProtocol,
) -> tuple[BootstrapBank, BootstrapBank]:
    selection = measurement.payload["selection_bootstrap"]
    audit = measurement.payload["audit_bootstrap"]
    if int(selection["samples"]) != int(config.fitness["bootstrap_samples"]):
        raise ValueError("selection bootstrap samples do not match V4 fitness")
    if int(audit["samples"]) != int(config.fitness["strict_bootstrap_samples"]):
        raise ValueError("audit bootstrap samples do not match V4 strict fitness")
    window = config.payload["windows"]["research"]
    common = {
        "frame": features,
        "start_timestamp": _timestamp(window["start"]),
        "cutoff_timestamp": measurement.cutoff_ms,
        "folds": config.folds,
    }
    selection_bank = build_bootstrap_bank(
        **common,
        bank_id="selection",
        seed=int(selection["seed"]),
        samples=int(selection["samples"]),
        block_days=int(selection["block_days"]),
    )
    audit_bank = build_bootstrap_bank(
        **common,
        bank_id="audit",
        seed=int(audit["seed"]),
        samples=int(audit["samples"]),
        block_days=int(audit["block_days"]),
    )
    if selection_bank.content_sha256 == audit_bank.content_sha256:
        raise RuntimeError("selection and audit bootstrap banks are not independent")
    return selection_bank, audit_bank


def _audit_finalists(
    result: V4EvolutionResult,
    *,
    features: pd.DataFrame,
    scaler: RobustScaler,
    shape: V4NetworkShape,
    config: PrimusV4Config,
    measurement: MeasurementProtocol,
    audit_bank: BootstrapBank,
) -> pd.DataFrame:
    strict_config = dict(config.fitness)
    strict_config["bootstrap_samples"] = audit_bank.samples
    window = config.payload["windows"]["research"]
    rows: list[dict[str, Any]] = []
    for rank, genome in enumerate(result.finalists, start=1):
        simulation = simulate_policy(
            features,
            policy=GeneticPrimusPolicyV4(genome, shape=shape),
            scaler=scaler,
            start_timestamp=_timestamp(window["start"]),
            end_timestamp=measurement.cutoff_ms,
            round_trip_cost_bps=float(config.costs["base_round_trip"]),
        )
        score = score_simulation_v4(
            simulation,
            folds=config.folds,
            config=strict_config,
            key="v4-independent-audit",
            bootstrap_bank=audit_bank,
        )
        rows.append(
            {
                "selection_rank": rank,
                "is_selection_champion": rank == 1,
                "genome_sha256": hashlib.sha256(genome.tobytes()).hexdigest(),
                "audit_fitness": score.fitness,
                "audit_lcb_daily_bps": score.metrics["lcb_daily_bps"],
                "audit_worst_fold_mean_bps": score.metrics["worst_fold_mean_bps"],
                "audit_median_fold_lcb_bps": score.metrics["median_fold_lcb_bps"],
                "audit_failed_gates": ";".join(score.failed_gates),
                "audit_passed": not score.failed_gates,
            }
        )
    return pd.DataFrame(rows)


def _write_sheet(writer: pd.ExcelWriter, name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.book[name]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        values = [str(cell.value) if cell.value is not None else "" for cell in column[:200]]
        sheet.column_dimensions[column[0].column_letter].width = min(48, max(10, max(len(value) for value in values) + 2))


def _workbook(path: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            _write_sheet(writer, name, frame)
    workbook = load_workbook(path, read_only=False)
    for name, frame in frames.items():
        sheet = workbook[name]
        if sheet.max_column != len(frame.columns) or sheet.max_column <= 1 or sheet.merged_cells.ranges:
            raise RuntimeError(f"invalid V4 workbook layout: {name}")
    workbook.close()


def _plot_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(history["generation"], history["worst_fold_mean_bps"], label="Worst fold mean")
    axes[0, 0].plot(history["generation"], history["median_fold_lcb_bps"], label="Median fold LCB")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].legend()
    axes[0, 0].set_title("Temporal robustness")
    axes[0, 1].plot(history["generation"], history["lcb_daily_bps"], label="Overall daily LCB")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Portfolio LCB")
    axes[1, 0].plot(history["generation"], history["positive_fold_means"], label="Positive folds")
    axes[1, 0].axhline(5, color="#B42318", linestyle="--")
    axes[1, 0].set_ylim(0, 6.5)
    axes[1, 0].set_title("Positive half-years")
    axes[1, 1].plot(history["generation"], history["max_drawdown"] * 100.0, label="Drawdown %")
    axes[1, 1].plot(history["generation"], history["fills_per_day"], label="Fills/day")
    axes[1, 1].set_title("Risk and turnover")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Generation")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _save_run(
    run_dir: Path,
    *,
    result: V4EvolutionResult,
    config: PrimusV4Config,
    architecture: Mapping[str, Any],
    seed: int,
    scaler: RobustScaler,
    run_fingerprint: str,
    stage: str,
    measurement: MeasurementProtocol,
    selection_bank: BootstrapBank,
    audit_bank: BootstrapBank,
    audit_frame: pd.DataFrame,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    champion = result.champion
    np.savez_compressed(run_dir / "champion.npz", genome=champion.genome)
    (run_dir / "scaler.json").write_text(json.dumps(scaler.to_mapping(), indent=2) + "\n", encoding="utf-8")
    result.generations.to_parquet(run_dir / "generation_metrics.parquet", index=False)
    audit_frame.to_parquet(run_dir / "audit_top_k.parquet", index=False)
    champion.score.folds.to_parquet(run_dir / "fold_metrics.parquet", index=False)
    champion.simulation.ledger.to_parquet(run_dir / "trades.parquet", index=False)
    _workbook(
        run_dir / "tables.xlsx",
        {
            "Generations": result.generations,
            "Audit Top K": audit_frame,
            "Folds": champion.score.folds,
            "Trades": champion.simulation.ledger,
        },
    )
    _plot_history(result.generations, run_dir / "evolution.png")
    genome_sha = hashlib.sha256(champion.genome.tobytes()).hexdigest()
    selection_bank_path = run_dir / "selection_bootstrap_bank.npz"
    audit_bank_path = run_dir / "audit_bootstrap_bank.npz"
    save_bootstrap_bank(selection_bank_path, selection_bank)
    save_bootstrap_bank(audit_bank_path, audit_bank)
    dataset_manifest_path = ROOT / str(config.dataset["manifest"])
    training_identity = {
        "dataset_sha256": config.dataset["dataset_sha256"],
        "dataset_manifest_file_sha256": sha256_file(dataset_manifest_path),
        "folds_sha256": folds_sha256(config.folds),
        "measurement_protocol_sha256": measurement.sha256,
        "training_cutoff": measurement.payload["training_cutoff"],
        "training_cutoff_ms": measurement.cutoff_ms,
        "label_purge_bars": measurement.purge_bars,
        "label_purge_cutoff_ms": measurement.purge_cutoff_ms,
        "deployment_horizon_days": int(measurement.payload["deployment_horizon_days"]),
        "selection_bootstrap_bank": {
            **selection_bank.metadata(),
            "artifact_sha256": sha256_file(selection_bank_path),
        },
        "audit_bootstrap_bank": {
            **audit_bank.metadata(),
            "artifact_sha256": sha256_file(audit_bank_path),
        },
    }
    manifest = {
        "schema_version": "exia.genetic_primus_experiment.v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": "V4_ARCHITECTURE_SCREEN_RUN" if stage == "screen" else "V4_FINAL_RESEARCH_RUN",
        "architecture": dict(architecture),
        "parameter_count": int(_shape(architecture).genome_size),
        "seed": int(seed),
        "selected_generation": int(champion.generation),
        "metrics": dict(champion.score.metrics),
        "failed_gates": list(champion.score.failed_gates),
        "config_sha256": sha256_file(config.source_path),
        "code_sha256": _composite_sha(CODE_PATHS),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "training_data_identity": training_identity,
        "audit_top_k": int(len(audit_frame)),
        "audit_top_k_sha256": sha256_file(run_dir / "audit_top_k.parquet"),
        "audit_used_for_selection": False,
        "run_fingerprint": run_fingerprint,
        "champion_genome_sha256": genome_sha,
        "champion_file_sha256": sha256_file(run_dir / "champion.npz"),
        "scaler_sha256": sha256_file(run_dir / "scaler.json"),
        "source_revision": _git_revision(),
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def _run_one(
    *,
    run_dir: Path,
    config: PrimusV4Config,
    features: pd.DataFrame,
    scaler: RobustScaler,
    x: np.ndarray,
    y: np.ndarray,
    architecture: Mapping[str, Any],
    settings: Mapping[str, Any],
    seed: int,
    stage: str,
    measurement: MeasurementProtocol,
    selection_bank: BootstrapBank,
    audit_bank: BootstrapBank,
) -> dict[str, Any]:
    shape = _shape(architecture)
    evolution = dict(settings)
    evolution["seed"] = int(seed)
    fingerprint = hashlib.sha256(
        (
            sha256_file(config.source_path)
            + _composite_sha(CODE_PATHS)
            + measurement.sha256
            + selection_bank.content_sha256
            + audit_bank.content_sha256
            + str(measurement.cutoff_ms)
            + str(architecture)
            + str(evolution)
        ).encode()
    ).hexdigest()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != fingerprint:
            raise ValueError(f"stale V4 run manifest: {manifest_path}")
        return existing
    warm_start = supervised_seed_v4(x, y, shape=shape, seed=seed).genome
    run_dir.mkdir(parents=True, exist_ok=True)

    def callback(champion, history):
        history.to_parquet(run_dir / "generation_progress.parquet", index=False)
        metrics = champion.score.metrics
        print(json.dumps({"stage": stage, "architecture": architecture["id"], "seed": seed, "generation": champion.generation, "positive_folds": metrics["positive_fold_means"], "worst_fold_mean": metrics["worst_fold_mean_bps"], "median_fold_lcb": metrics["median_fold_lcb_bps"], "overall_lcb": metrics["lcb_daily_bps"], "drawdown": metrics["max_drawdown"]}), flush=True)

    window = config.payload["windows"]["research"]
    result = run_evolution_v4(
        features,
        scaler=scaler,
        shape=shape,
        start_timestamp=_timestamp(window["start"]),
        end_timestamp=measurement.cutoff_ms,
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        folds=config.folds,
        fitness_config=config.fitness,
        bootstrap_bank=selection_bank,
        evolution_config=evolution,
        warm_start=warm_start,
        checkpoint_path=run_dir / "checkpoint.npz",
        run_fingerprint=fingerprint,
        resume=(run_dir / "checkpoint.npz").is_file(),
        finalist_count=measurement.audit_top_k,
        on_generation=callback,
    )
    audit_frame = _audit_finalists(
        result,
        features=features,
        scaler=scaler,
        shape=shape,
        config=config,
        measurement=measurement,
        audit_bank=audit_bank,
    )
    return _save_run(
        run_dir,
        result=result,
        config=config,
        architecture=architecture,
        seed=seed,
        scaler=scaler,
        run_fingerprint=fingerprint,
        stage=stage,
        measurement=measurement,
        selection_bank=selection_bank,
        audit_bank=audit_bank,
        audit_frame=audit_frame,
    )


def run_screen(
    config_path: Path,
    output_dir: Path,
    measurement_path: Path = DEFAULT_MEASUREMENT,
) -> dict[str, Any]:
    config = load_v4_config(config_path, root=ROOT)
    measurement = load_measurement_protocol(measurement_path)
    _, features, scaler, x, y = _prepare(config, measurement)
    selection_bank, audit_bank = _bootstrap_banks(features, config, measurement)
    rows: list[dict[str, Any]] = []
    for architecture in config.architectures:
        for seed in config.payload["screen"]["seeds"]:
            run_dir = output_dir / "runs" / f"{architecture['id']}_seed_{seed}"
            manifest = _run_one(
                run_dir=run_dir,
                config=config,
                features=features,
                scaler=scaler,
                x=x,
                y=y,
                architecture=architecture,
                settings=config.payload["screen"],
                seed=int(seed),
                stage="screen",
                measurement=measurement,
                selection_bank=selection_bank,
                audit_bank=audit_bank,
            )
            rows.append({"architecture_id": architecture["id"], "seed": int(seed), "parameter_count": manifest["parameter_count"], **manifest["metrics"]})
    results = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for architecture_id, subset in results.groupby("architecture_id", sort=True):
        summaries.append({
            "architecture_id": architecture_id,
            "parameter_count": int(subset["parameter_count"].iloc[0]),
            "seeds": int(len(subset)),
            "passing_seeds": int(subset["temporal_gate_passed"].sum()),
            "minimum_positive_fold_means": int(subset["positive_fold_means"].min()),
            "minimum_worst_fold_mean_bps": float(subset["worst_fold_mean_bps"].min()),
            "median_worst_fold_mean_bps": float(subset["worst_fold_mean_bps"].median()),
            "median_fold_lcb_bps": float(subset["median_fold_lcb_bps"].median()),
            "median_overall_lcb_bps": float(subset["lcb_daily_bps"].median()),
            "maximum_drawdown": float(subset["max_drawdown"].max()),
        })
    summary = pd.DataFrame(summaries)
    summary["selection_key"] = summary.apply(lambda row: json.dumps([int(row.passing_seeds), int(row.minimum_positive_fold_means), float(row.minimum_worst_fold_mean_bps), float(row.median_fold_lcb_bps), float(row.median_overall_lcb_bps), -int(row.parameter_count)]), axis=1)
    chosen = max(summaries, key=lambda row: (row["passing_seeds"], row["minimum_positive_fold_means"], row["minimum_worst_fold_mean_bps"], row["median_fold_lcb_bps"], row["median_overall_lcb_bps"], -row["parameter_count"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "screen_results.parquet", index=False)
    summary.to_parquet(output_dir / "architecture_summary.parquet", index=False)
    _workbook(output_dir / "architecture_screen.xlsx", {"Runs": results, "Architectures": summary})
    manifest = {
        "schema_version": "exia.genetic_primus_architecture_screen.v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "V4_ARCHITECTURE_SELECTED_FOR_FINAL_RESEARCH",
        "selection_contract": "all_seeds>worst_fold>median_fold_lcb>overall_lcb>smaller_network",
        "chosen_architecture": chosen["architecture_id"],
        "chosen_summary": chosen,
        "config_sha256": sha256_file(config_path),
        "code_sha256": _composite_sha(CODE_PATHS),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "dataset_manifest_file_sha256": sha256_file(ROOT / str(config.dataset["manifest"])),
        "folds_sha256": folds_sha256(config.folds),
        "measurement_protocol_sha256": measurement.sha256,
        "training_cutoff": measurement.payload["training_cutoff"],
        "label_purge_bars": measurement.purge_bars,
        "deployment_horizon_days": int(measurement.payload["deployment_horizon_days"]),
        "selection_bootstrap_bank_sha256": selection_bank.content_sha256,
        "audit_bootstrap_bank_sha256": audit_bank.content_sha256,
        "audit_used_for_selection": False,
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_final(
    config_path: Path,
    screen_dir: Path,
    output_dir: Path,
    measurement_path: Path = DEFAULT_MEASUREMENT,
) -> dict[str, Any]:
    config = load_v4_config(config_path, root=ROOT)
    measurement = load_measurement_protocol(measurement_path)
    screen_manifest = json.loads((screen_dir / "manifest.json").read_text(encoding="utf-8"))
    architecture = next(item for item in config.architectures if item["id"] == screen_manifest["chosen_architecture"])
    _, features, scaler, x, y = _prepare(config, measurement)
    selection_bank, audit_bank = _bootstrap_banks(features, config, measurement)
    seed = int(config.payload["final"]["seed"])
    manifest = _run_one(
        run_dir=output_dir,
        config=config,
        features=features,
        scaler=scaler,
        x=x,
        y=y,
        architecture=architecture,
        settings=config.payload["final"],
        seed=seed,
        stage="final",
        measurement=measurement,
        selection_bank=selection_bank,
        audit_bank=audit_bank,
    )
    with np.load(output_dir / "champion.npz", allow_pickle=False) as payload:
        genome = np.asarray(payload["genome"], dtype=np.float32)
    shape = _shape(architecture)
    strict_config = dict(config.fitness)
    strict_config["bootstrap_samples"] = int(config.fitness["strict_bootstrap_samples"])
    window = config.payload["windows"]["research"]
    scores = {}
    simulations = {}
    for label, cost in (("base16", 16.0), ("stress24", 24.0)):
        simulation = simulate_policy(features, policy=GeneticPrimusPolicyV4(genome, shape=shape), scaler=scaler, start_timestamp=_timestamp(window["start"]), end_timestamp=measurement.cutoff_ms, round_trip_cost_bps=cost)
        score = score_simulation_v4(
            simulation,
            folds=config.folds,
            config=strict_config,
            key=f"v4-final-independent-audit:{label}",
            bootstrap_bank=audit_bank,
        )
        scores[label] = score
        simulations[label] = simulation
        score.folds.to_parquet(output_dir / f"strict_{label}_folds.parquet", index=False)
        (output_dir / f"strict_{label}_metrics.json").write_text(json.dumps(dict(score.metrics), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    passed = not scores["base16"].failed_gates and not scores["stress24"].failed_gates
    feature_sha = _composite_sha((Path("src/exia/genetic_primus/contracts.py"), Path("src/exia/genetic_primus/features.py")))
    execution_sha = _composite_sha((Path("src/exia/genetic_primus/policy_v4.py"), Path("src/exia/genetic_primus/simulator.py")))
    freeze = {
        "schema_version": "exia.genetic_primus_freeze.v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "DEVELOPMENT_V4_PASSED_STRICT_GATES" if passed else "DEVELOPMENT_V4_REJECTED_STRICT_GATES",
        "architecture": dict(architecture),
        "screen_manifest_sha256": sha256_file(screen_dir / "manifest.json"),
        "experiment_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "config_sha256": sha256_file(config_path),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "dataset_manifest_file_sha256": sha256_file(ROOT / str(config.dataset["manifest"])),
        "folds_sha256": folds_sha256(config.folds),
        "measurement_protocol_sha256": measurement.sha256,
        "training_cutoff": measurement.payload["training_cutoff"],
        "training_cutoff_ms": measurement.cutoff_ms,
        "label_purge_bars": measurement.purge_bars,
        "label_purge_cutoff_ms": measurement.purge_cutoff_ms,
        "deployment_horizon_days": int(measurement.payload["deployment_horizon_days"]),
        "selection_bootstrap_bank_sha256": selection_bank.content_sha256,
        "audit_bootstrap_bank_sha256": audit_bank.content_sha256,
        "audit_used_for_selection": False,
        "champion_file_sha256": sha256_file(output_dir / "champion.npz"),
        "champion_genome_sha256": hashlib.sha256(genome.tobytes()).hexdigest(),
        "scaler_sha256": sha256_file(output_dir / "scaler.json"),
        "feature_contract_sha256": feature_sha,
        "execution_contract_sha256": execution_sha,
        "base16_metrics": dict(scores["base16"].metrics),
        "stress24_metrics": dict(scores["stress24"].metrics),
        "validation_allowed": bool(passed),
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "freeze_manifest.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    _workbook(output_dir / "strict_freeze.xlsx", {"Base 16 Folds": scores["base16"].folds, "Stress 24 Folds": scores["stress24"].folds, "Trades 16": simulations["base16"].ledger})
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genetic_Primus V4 architecture screen/final")
    parser.add_argument("--stage", choices=("screen", "final"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "screen":
        result = run_screen(
            args.config.resolve(),
            (args.output_dir or args.screen_dir).resolve(),
            args.measurement_config.resolve(),
        )
    else:
        result = run_final(
            args.config.resolve(),
            args.screen_dir.resolve(),
            (args.output_dir or DEFAULT_FINAL).resolve(),
            args.measurement_config.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
