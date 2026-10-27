from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.benchmarks_v4 import (  # noqa: E402
    NoTradePolicy,
    mean_reversion_policy,
    momentum_policy,
    summarize_benchmark,
)
from exia.genetic_primus.contracts_v4 import load_v4_config  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.experiment_matrix_v4 import (  # noqa: E402
    derive_rolling_protocol,
    load_experiment_matrix,
    weekly_cells,
)
from exia.genetic_primus.features import build_feature_frame  # noqa: E402
from exia.genetic_primus.fitness_v4 import score_simulation_v4  # noqa: E402
from exia.genetic_primus.labels import build_forward_labels  # noqa: E402
from exia.genetic_primus.measurement import build_bootstrap_bank, load_measurement_protocol  # noqa: E402
from exia.genetic_primus.policy_v4 import (  # noqa: E402
    EnsemblePolicyV4,
    GeneticPrimusPolicyV4,
    V4NetworkShape,
    ensemble_sparsity_v4,
    genome_sparsity_v4,
)
from exia.genetic_primus.rolling import (  # noqa: E402
    OriginTrainingContext,
    _origin_training_data,
    build_rolling_origins,
    load_rolling_protocol,
)
from exia.genetic_primus.rolling_v4 import rolling_training_folds  # noqa: E402
from exia.genetic_primus.simulator import simulate_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed Genetic Primus island checkpoints")
    parser.add_argument("--island-output", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    args = parser.parse_args()

    config = load_v4_config(ROOT / "configs/genetic_primus_v4.json", root=ROOT)
    measurement = load_measurement_protocol(ROOT / "configs/genetic_primus_measurement_3d_v1.json")
    matrix = load_experiment_matrix(ROOT / "configs/genetic_primus_experiment_matrix_v1.json")
    base = load_rolling_protocol(ROOT / "configs/genetic_primus_rolling_3d_v1.json")
    cell = next(cell for cell in weekly_cells(matrix) if cell.cell_id == args.cell_id)
    protocol = derive_rolling_protocol(base, matrix, cell)
    origin = build_rolling_origins(protocol)[0]
    architecture = next(item for item in config.architectures if item["id"] == "linear")
    shape = V4NetworkShape(architecture["inputs"], tuple(architecture["hidden"]), architecture["outputs"])
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    labels = build_forward_labels(
        panel,
        horizons_bars=tuple(config.payload["labels"]["horizons_bars"]),
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        minimum_edge_bps=float(config.payload["labels"]["minimum_edge_bps"]),
        minimum_votes=int(config.payload["labels"]["minimum_votes"]),
    )
    scaler, x, y, _ = _origin_training_data(features, labels, origin)
    context = OriginTrainingContext(origin, scaler, x, y, features, protocol.sha256, None)
    folds = rolling_training_folds(
        train_start=origin.train_start,
        train_end=origin.train_end,
        count=int(protocol.payload["recency_fitness"]["rolling_fold_count"]),
    )
    selection_spec = measurement.payload["selection_bootstrap"]
    audit_spec = measurement.payload["audit_bootstrap"]
    selection_bank = build_bootstrap_bank(
        features,
        start_timestamp=origin.train_start,
        cutoff_timestamp=origin.train_end,
        folds=folds,
        bank_id="recovery-selection",
        seed=int(selection_spec["seed"]),
        samples=int(selection_spec["samples"]),
        block_days=int(selection_spec["block_days"]),
    )
    audit_bank = build_bootstrap_bank(
        features,
        start_timestamp=origin.train_start,
        cutoff_timestamp=origin.train_end,
        folds=folds,
        bank_id="recovery-audit",
        seed=int(audit_spec["seed"]),
        samples=int(audit_spec["samples"]),
        block_days=int(audit_spec["block_days"]),
    )
    fitness = dict(config.fitness)
    fitness["recency"] = dict(protocol.payload["recency_fitness"])
    genomes = []
    for checkpoint in sorted((args.island_output / "origin_0000").glob("island_*/checkpoint.npz")):
        with np.load(checkpoint, allow_pickle=True) as payload:
            if not bool(payload["complete"][0]):
                raise ValueError(f"incomplete island checkpoint: {checkpoint}")
            genomes.append((checkpoint.parent.name, payload["global_genome"].astype(np.float32)))
    ranked = []
    for island_id, genome in genomes:
        simulation = simulate_policy(
            features,
            policy=GeneticPrimusPolicyV4(genome, shape=shape),
            scaler=scaler,
            start_timestamp=origin.train_start,
            end_timestamp=origin.train_end,
            round_trip_cost_bps=float(config.costs["base_round_trip"]),
        )
        score = score_simulation_v4(
            simulation,
            folds=folds,
            config=fitness,
            key=island_id,
            bootstrap_bank=selection_bank,
            complexity_metrics=genome_sparsity_v4(
                genome,
                shape=shape,
                active_weight_threshold=float(fitness["recency"]["active_weight_threshold"]),
            ),
        )
        ranked.append((island_id, genome, score))
    ranked.sort(key=lambda item: item[2].ranking_key, reverse=True)
    ensemble_rows = []
    evaluated = []
    for size in (3, 5):
        members = tuple(item[1] for item in ranked[:size])
        simulation = simulate_policy(
            features,
            policy=EnsemblePolicyV4(members, shape=shape),
            scaler=scaler,
            start_timestamp=origin.train_start,
            end_timestamp=origin.train_end,
            round_trip_cost_bps=float(config.costs["base_round_trip"]),
        )
        complexity = ensemble_sparsity_v4(
            members,
            shape=shape,
            active_weight_threshold=float(fitness["recency"]["active_weight_threshold"]),
        )
        selection_score = score_simulation_v4(
            simulation,
            folds=folds,
            config=fitness,
            key=f"ensemble-{size}-selection",
            bootstrap_bank=selection_bank,
            complexity_metrics=complexity,
        )
        strict = dict(fitness)
        strict["bootstrap_samples"] = audit_bank.samples
        audit_score = score_simulation_v4(
            simulation,
            folds=folds,
            config=strict,
            key=f"ensemble-{size}-audit",
            bootstrap_bank=audit_bank,
            complexity_metrics=complexity,
        )
        evaluated.append((size, members, selection_score, audit_score))
        ensemble_rows.append(
            {
                "ensemble_size": size,
                "selection_fitness": selection_score.fitness,
                "selection_lcb_daily_bps": selection_score.metrics["lcb_daily_bps"],
                "audit_lcb_daily_bps": audit_score.metrics["lcb_daily_bps"],
                "max_drawdown": selection_score.metrics["max_drawdown"],
                "failed_gates": list(selection_score.failed_gates),
                "members": [item[0] for item in ranked[:size]],
            }
        )
    selected = max(evaluated, key=lambda item: item[2].ranking_key)
    selected_size, members = selected[0], selected[1]
    benchmark_rows = []
    policies = [
        (f"GeneticPrimusEnsemble{selected_size}", EnsemblePolicyV4(members, shape=shape)),
        ("NoTrade", NoTradePolicy()),
        ("Momentum_return_1", momentum_policy()),
        ("MeanReversion_return_1", mean_reversion_policy()),
    ]
    for cost in (16.0, 24.0):
        for policy_id, policy in policies:
            simulation = simulate_policy(
                features,
                policy=policy,
                scaler=scaler,
                start_timestamp=origin.deployment_start,
                end_timestamp=origin.deployment_end,
                round_trip_cost_bps=cost,
            )
            benchmark_rows.append(
                {"agent_id": policy_id, "round_trip_cost_bps": cost, **summarize_benchmark(simulation)}
            )
    report = {
        "schema_version": "exia.genetic_primus.completed_islands_analysis/1",
        "cell_id": cell.cell_id,
        "selected_ensemble_size": selected_size,
        "islands": [
            {
                "island_id": island_id,
                "fitness": score.fitness,
                "lcb_daily_bps": score.metrics["lcb_daily_bps"],
                "max_drawdown": score.metrics["max_drawdown"],
                "failed_gates": list(score.failed_gates),
            }
            for island_id, _, score in ranked
        ],
        "ensembles": ensemble_rows,
        "deployment_benchmarks": benchmark_rows,
        "credentials_allowed": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    target = args.island_output / "recovered_analysis.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "RECOVERED_ANALYSIS_COMPLETE", "selected_ensemble_size": selected_size}))


if __name__ == "__main__":
    main()
