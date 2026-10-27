from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .benchmarks_v4 import NoTradePolicy, momentum_policy, relative_momentum_policy
from .evolution_v4 import run_evolution_v4
from .features import RobustScaler
from .fitness_v4 import score_simulation_v4
from .measurement import (
    MeasurementProtocol,
    build_bootstrap_bank,
    save_bootstrap_bank,
)
from .policy_v4 import (
    EnsemblePolicyV4,
    GeneticPrimusPolicyV4,
    V4NetworkShape,
    ensemble_sparsity_v4,
    genome_sparsity_v4,
    select_action_diverse_genomes_v4,
    stability_selection_v4,
    supervised_seed_overlay_v4,
    supervised_seed_v4,
)
from .promotion_v4 import (
    PromotionProtocol,
    ensemble_centroid,
    evaluate_promotion_gate,
    paired_daily_returns,
)
from .rolling import (
    OriginTrainer,
    OriginTrainingContext,
    RollingProtocol,
    TrainedOriginModel,
    fixed_baseline_scaler,
    genetic_simulation_kwargs,
)
from .simulator import overlay_opportunity_mask, simulate_policy
from .temporal_validation_v4 import TemporalValidationSplit, build_temporal_validation_split


def _iso(timestamp_ms: int) -> str:
    return pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").isoformat().replace("+00:00", "Z")


def rolling_training_folds(
    *,
    train_start: int,
    train_end: int,
    count: int = 6,
) -> tuple[dict[str, str], ...]:
    if count < 2 or train_end <= train_start:
        raise ValueError("invalid rolling training fold request")
    boundaries = np.linspace(train_start, train_end, count + 1, dtype=np.int64)
    return tuple(
        {
            "name": f"rolling_{index + 1:02d}",
            "start": _iso(int(boundaries[index])),
            "end": _iso(int(boundaries[index + 1])),
        }
        for index in range(count)
    )


def _temporal_supervised_data(
    context: OriginTrainingContext,
    *,
    split: TemporalValidationSplit,
    overlay_config: Mapping[str, Any],
) -> tuple[RobustScaler, np.ndarray, np.ndarray]:
    if context.label_frame is None:
        raise ValueError("temporal validation requires the aligned label frame")
    features = context.feature_frame
    labels = context.label_frame
    scaler_mask = features["timestamp"].ge(split.inner_start) & features["timestamp"].lt(
        split.inner_end
    )
    scaler = RobustScaler.fit(features.loc[scaler_mask])
    supervised_mask = (
        scaler_mask & features["feature_supported"] & labels["label_supported"]
    )
    baseline = features.loc[
        supervised_mask, str(overlay_config["baseline_feature"])
    ].to_numpy(np.float64)
    opportunity = overlay_opportunity_mask(
        baseline,
        threshold_mode=str(overlay_config.get("baseline_threshold_mode", "return_bps")),
        threshold_value=float(
            overlay_config.get(
                "minimum_baseline_abs_value",
                overlay_config.get("minimum_baseline_move_bps", 0.0),
            )
        ),
    )
    if not opportunity.any():
        raise ValueError("temporal inner train has no overlay opportunities")
    static = scaler.transform(features.loc[supervised_mask])
    x = np.concatenate(
        [static, np.zeros((len(static), 5), dtype=np.float32)], axis=1
    )[opportunity]
    target = labels.loc[supervised_mask, "target_position"].to_numpy(np.int8)[opportunity]
    baseline_target = np.sign(baseline[opportunity]).astype(np.int8)
    allow = ((target != 0) & (target == baseline_target)).astype(np.int8)
    return scaler, x, allow


def _diversity_anchor_features(
    frame: pd.DataFrame,
    *,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    maximum_rows: int,
) -> np.ndarray:
    mask = (
        frame["timestamp"].ge(start_timestamp)
        & frame["timestamp"].lt(end_timestamp)
        & frame["feature_supported"]
    )
    static = scaler.transform(frame.loc[mask])
    features = np.concatenate(
        [static, np.zeros((len(static), 5), dtype=np.float32)], axis=1
    )
    if not len(features):
        raise ValueError("diversity anchor contains no supported rows")
    if len(features) > int(maximum_rows):
        index = np.linspace(0, len(features) - 1, int(maximum_rows), dtype=np.int64)
        features = features[index]
    return features


_FINALIST_STRUCTURAL_FAILURES = frozenset(
    {
        "minimum_trades",
        "minimum_opportunities",
        "minimum_trade_days",
        "max_drawdown",
        "turnover",
        "direction_collapse",
        "regime_collapse",
        "symbol_collapse",
    }
)


def validation_finalist_eligibility_v4(
    metrics: Mapping[str, Any],
    failed_gates: Sequence[str],
) -> tuple[bool, tuple[str, ...]]:
    """Apply only deployability gates; temporal audit remains independent evidence."""
    blockers = {
        failure
        for failure in failed_gates
        if failure in _FINALIST_STRUCTURAL_FAILURES
        or failure.startswith("overlay_action_coverage")
        or failure.startswith("opportunity_normalized_")
        or failure.startswith("nonpositive_paired_lcb:")
    }
    if not bool(metrics.get("coverage_constraint_passed", False)):
        blockers.add("coverage_constraint")
    if float(metrics.get("paired_incremental_min_lcb_bps", -1.0e9)) <= 0.0:
        blockers.add("paired_incremental_lcb")
    ordered = tuple(sorted(blockers))
    return not ordered, ordered


def validation_candidate_status_v4(eligible_finalists: int) -> str:
    if eligible_finalists < 0:
        raise ValueError("eligible finalist count cannot be negative")
    return (
        "NO_ELIGIBLE_CHALLENGER"
        if eligible_finalists == 0
        else "DIAGNOSTIC_CHALLENGER"
    )


def _audit_finalists(
    finalists: Sequence[np.ndarray],
    *,
    context: OriginTrainingContext,
    shape: V4NetworkShape,
    folds: Sequence[Mapping[str, str]],
    fitness_config: Mapping[str, Any],
    audit_bank,
    round_trip_cost_bps: float,
    simulation_kwargs: Mapping[str, Any] | None = None,
    paired_reference_portfolios: Mapping[str, pd.DataFrame] | None = None,
    scaler: RobustScaler | None = None,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strict = dict(fitness_config)
    strict["bootstrap_samples"] = audit_bank.samples
    for rank, genome in enumerate(finalists, start=1):
        simulation = simulate_policy(
            context.feature_frame,
            policy=GeneticPrimusPolicyV4(genome, shape=shape),
            scaler=scaler if scaler is not None else context.scaler,
            start_timestamp=(
                context.origin.train_start if start_timestamp is None else start_timestamp
            ),
            end_timestamp=context.origin.train_end if end_timestamp is None else end_timestamp,
            round_trip_cost_bps=round_trip_cost_bps,
            **dict(simulation_kwargs or {}),
        )
        score = score_simulation_v4(
            simulation,
            folds=folds,
            config=strict,
            key="rolling-independent-audit",
            bootstrap_bank=audit_bank,
            complexity_metrics=genome_sparsity_v4(
                genome,
                shape=shape,
                active_weight_threshold=float(strict["recency"]["active_weight_threshold"]),
            ),
            paired_reference_portfolios=paired_reference_portfolios,
        )
        deployable, eligibility_blockers = validation_finalist_eligibility_v4(
            score.metrics,
            score.failed_gates,
        )
        rows.append(
            {
                "selection_rank": rank,
                "is_selection_champion": rank == 1,
                "genome_sha256": hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest(),
                "audit_fitness": score.fitness,
                "_ranking_key": score.ranking_key,
                "audit_lcb_daily_bps": score.metrics["lcb_daily_bps"],
                "audit_worst_fold_mean_bps": score.metrics["worst_fold_mean_bps"],
                "audit_median_fold_lcb_bps": score.metrics["median_fold_lcb_bps"],
                "audit_recency_primary_lcb_bps": score.metrics.get("recency_primary_lcb_bps"),
                "audit_recency_worst_window_lcb_bps": score.metrics.get(
                    "recency_worst_window_lcb_bps"
                ),
                "audit_recency_total_penalty_bps": score.metrics.get(
                    "recency_total_penalty_bps"
                ),
                "audit_active_parameter_count": score.metrics.get(
                    "recency_active_parameter_count"
                ),
                "audit_active_feature_count": score.metrics.get("recency_active_feature_count"),
                "audit_feature_sparsity": score.metrics.get("recency_feature_sparsity"),
                "audit_fills_per_day": score.metrics.get("fills_per_day"),
                "audit_max_drawdown": score.metrics.get("max_drawdown"),
                "audit_max_direction_share": score.metrics.get("max_direction_share"),
                "audit_max_state_share": score.metrics.get("max_state_share"),
                "audit_max_symbol_share": score.metrics.get("max_symbol_share"),
                "audit_overlay_allowed_fraction": score.metrics.get(
                    "overlay_allowed_fraction"
                ),
                "audit_coverage_constraint_passed": score.metrics.get(
                    "coverage_constraint_passed", False
                ),
                "audit_paired_incremental_min_lcb_bps": score.metrics.get(
                    "paired_incremental_min_lcb_bps"
                ),
                "audit_paired_objective_lcb_bps": score.metrics.get(
                    "paired_objective_lcb_bps"
                ),
                "audit_paired_absolute_lcb_bps": score.metrics.get(
                    "paired_absolute_lcb_bps"
                ),
                "audit_paired_baseline_lcb_bps": score.metrics.get(
                    "paired_baseline_lcb_bps"
                ),
                "audit_deployment_eligible": deployable,
                "audit_eligibility_blockers": ";".join(eligibility_blockers),
                "audit_failed_gates": ";".join(score.failed_gates),
            }
        )
    return pd.DataFrame(rows)


def _score_ensemble(
    genomes: tuple[np.ndarray, ...],
    *,
    context: OriginTrainingContext,
    shape: V4NetworkShape,
    folds: Sequence[Mapping[str, str]],
    fitness_config: Mapping[str, Any],
    bootstrap_bank,
    round_trip_cost_bps: float,
    key: str,
    scaler=None,
    simulation_kwargs: Mapping[str, Any] | None = None,
    paired_reference_portfolios: Mapping[str, pd.DataFrame] | None = None,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
):
    simulation = simulate_policy(
        context.feature_frame,
        policy=EnsemblePolicyV4(genomes, shape=shape),
        scaler=scaler if scaler is not None else context.scaler,
        start_timestamp=(context.origin.train_start if start_timestamp is None else start_timestamp),
        end_timestamp=context.origin.train_end if end_timestamp is None else end_timestamp,
        round_trip_cost_bps=round_trip_cost_bps,
        **dict(simulation_kwargs or {}),
    )
    strict = dict(fitness_config)
    strict["bootstrap_samples"] = bootstrap_bank.samples
    score = score_simulation_v4(
        simulation,
        folds=folds,
        config=strict,
        key=key,
        bootstrap_bank=bootstrap_bank,
        complexity_metrics=ensemble_sparsity_v4(
            genomes,
            shape=shape,
            active_weight_threshold=float(strict["recency"]["active_weight_threshold"]),
        ),
        paired_reference_portfolios=paired_reference_portfolios,
    )
    return simulation, score


def should_resume_island(checkpoint_path: Path) -> bool:
    """Resume only an explicitly present island checkpoint."""
    return checkpoint_path.is_file()


def bounded_v4_origin_trainer(
    *,
    rolling_protocol: RollingProtocol,
    measurement_protocol: MeasurementProtocol,
    shape: V4NetworkShape,
    fitness_config: Mapping[str, Any],
    round_trip_cost_bps: float,
    promotion_protocol: PromotionProtocol,
) -> OriginTrainer:
    settings = dict(rolling_protocol.payload["genetic_training"])
    schema_version = rolling_protocol.payload.get("schema_version")
    overlay_v2 = schema_version == "exia.genetic_primus.rolling.overlay_3d/2"
    overlay_v3 = schema_version == "exia.genetic_primus.rolling.overlay_3d/3"
    overlay_v4 = schema_version == "exia.genetic_primus.rolling.overlay_3d/4"
    overlay_v5 = schema_version == "exia.genetic_primus.rolling.overlay_3d/5"
    overlay_v6 = schema_version == "exia.genetic_primus.rolling.overlay_3d/6"
    overlay_temporal = overlay_v3 or overlay_v4 or overlay_v5 or overlay_v6
    overlay_modern = overlay_v2 or overlay_temporal
    if int(settings.get("workers", 0)) != 1:
        raise PermissionError("rolling trainer is restricted to one local worker")
    ensemble_sizes = tuple(int(value) for value in settings["ensemble_sizes"])
    expected_ensembles = (1, 3) if overlay_modern else (3, 5)
    if ensemble_sizes != expected_ensembles or int(settings["island_count"]) < max(ensemble_sizes):
        raise ValueError("rolling islands do not support the sealed ensemble sizes")
    if not 5 <= int(settings["generations"]) <= 10:
        raise ValueError("rolling retraining must remain within five to ten generations")
    if not 0.2 <= float(settings["random_immigrant_fraction"]) <= 0.3:
        raise ValueError("rolling random immigrants must remain between 20% and 30%")
    selection_spec = measurement_protocol.payload["selection_bootstrap"]
    audit_spec = measurement_protocol.payload["audit_bootstrap"]
    if int(selection_spec["samples"]) != int(fitness_config["bootstrap_samples"]):
        raise ValueError("rolling selection bootstrap size mismatch")
    if int(audit_spec["samples"]) != int(fitness_config["strict_bootstrap_samples"]):
        raise ValueError("rolling audit bootstrap size mismatch")
    recency_config = dict(rolling_protocol.payload["recency_fitness"])
    rolling_fitness = dict(fitness_config)
    rolling_fitness["recency"] = recency_config
    rolling_fitness["complexity_parameter_count"] = shape.genome_size
    simulation_kwargs = genetic_simulation_kwargs(rolling_protocol)
    overlay_config = rolling_protocol.payload.get("policy_overlay")
    paired_config = rolling_protocol.payload.get("paired_incremental_fitness")
    multiple_testing_config = rolling_protocol.payload.get("adaptive_multiple_testing")
    evidence_config = rolling_protocol.payload.get("abstention_evidence")
    concentration_config = rolling_protocol.payload.get(
        "opportunity_normalized_concentration"
    )
    if overlay_config is not None:
        rolling_fitness["abstention"] = {
            "minimum_allowed_fraction": overlay_config.get("minimum_allowed_fraction", 0.0),
            "maximum_allowed_fraction": overlay_config["maximum_allowed_fraction"],
            "coverage_penalty_bps": overlay_config.get(
                "coverage_penalty_bps",
                overlay_config.get("excess_coverage_penalty_bps", 0.0),
            ),
        }
    if evidence_config is not None:
        rolling_fitness["minimum_trades"] = int(evidence_config["minimum_trades"])
        rolling_fitness["abstention_evidence"] = dict(evidence_config)
    if concentration_config is not None:
        rolling_fitness["opportunity_normalized_concentration"] = dict(
            concentration_config
        )
    if paired_config is not None:
        rolling_fitness["paired_incremental"] = dict(paired_config)
    if multiple_testing_config is not None:
        planned_hypotheses = (
            int(settings["island_count"])
            * int(settings["population_size"])
            * int(settings["generations"])
        )
        if overlay_modern:
            planned_hypotheses += int(multiple_testing_config["audit_finalists"])
            planned_hypotheses += len(ensemble_sizes)
        if planned_hypotheses != int(
            multiple_testing_config["max_hypotheses_per_origin"]
        ):
            raise ValueError("evolution plan does not match the sealed hypothesis budget")
        if int(multiple_testing_config["audit_finalists"]) != int(
            settings["island_count"]
        ):
            raise ValueError("audit finalist count does not match the sealed island count")
        rolling_fitness["adaptive_multiple_testing"] = dict(multiple_testing_config)

    def train(context: OriginTrainingContext) -> TrainedOriginModel:
        origin = context.origin
        seed = int(settings["seed_base"]) + origin.index
        temporal_config = rolling_protocol.payload.get("temporal_validation")
        temporal_split: TemporalValidationSplit | None = None
        training_scaler = context.scaler
        training_x = context.overlay_supervised_features
        training_y = context.overlay_supervised_targets
        evolution_start = origin.train_start
        evolution_end = origin.train_end
        if overlay_temporal:
            if temporal_config is None:
                raise ValueError("overlay v3 requires temporal validation settings")
            temporal_split = build_temporal_validation_split(
                train_start=origin.train_start,
                train_end=origin.train_end,
                timeframe_ms=rolling_protocol.timeframe_ms,
                config=temporal_config,
            )
            training_scaler, training_x, training_y = _temporal_supervised_data(
                context,
                split=temporal_split,
                overlay_config=overlay_config,
            )
            evolution_start = temporal_split.inner_start
            evolution_end = temporal_split.inner_end
        folds = rolling_training_folds(
            train_start=evolution_start,
            train_end=evolution_end,
            count=int(
                temporal_config["inner_fold_count"]
                if temporal_config is not None
                else recency_config["rolling_fold_count"]
            ),
        )
        selection_bank = build_bootstrap_bank(
            context.feature_frame,
            start_timestamp=evolution_start,
            cutoff_timestamp=evolution_end,
            folds=folds,
            bank_id=f"origin_{origin.index:04d}_selection",
            seed=int(selection_spec["seed"]) + origin.index,
            samples=int(selection_spec["samples"]),
            block_days=int(selection_spec["block_days"]),
        )
        validation_folds = folds
        validation_bank = selection_bank
        if temporal_split is not None:
            validation_folds = rolling_training_folds(
                train_start=temporal_split.validation_start,
                train_end=temporal_split.validation_end,
                count=int(temporal_config["validation_fold_count"]),
            )
            validation_bank = build_bootstrap_bank(
                context.feature_frame,
                start_timestamp=temporal_split.validation_start,
                cutoff_timestamp=temporal_split.validation_end,
                folds=validation_folds,
                bank_id=f"origin_{origin.index:04d}_temporal_validation",
                seed=int(selection_spec["seed"]) + 100_000 + origin.index,
                samples=int(selection_spec["samples"]),
                block_days=int(selection_spec["block_days"]),
            )
        audit_folds = folds
        audit_start = evolution_start
        audit_end = evolution_end
        if temporal_split is not None:
            audit_folds = rolling_training_folds(
                train_start=temporal_split.audit_start,
                train_end=temporal_split.audit_end,
                count=int(temporal_config["audit_fold_count"]),
            )
            audit_start = temporal_split.audit_start
            audit_end = temporal_split.audit_end
        audit_bank = build_bootstrap_bank(
            context.feature_frame,
            start_timestamp=audit_start,
            cutoff_timestamp=audit_end,
            folds=audit_folds,
            bank_id=f"origin_{origin.index:04d}_audit",
            seed=int(audit_spec["seed"]) + origin.index,
            samples=int(audit_spec["samples"]),
            block_days=int(audit_spec["block_days"]),
        )
        promotion_resample_bank = audit_bank
        if temporal_split is not None:
            promotion_resample_bank = build_bootstrap_bank(
                context.feature_frame,
                start_timestamp=temporal_split.validation_start,
                cutoff_timestamp=temporal_split.validation_end,
                folds=validation_folds,
                bank_id=f"origin_{origin.index:04d}_promotion_resample_audit",
                seed=int(audit_spec["seed"]) + 200_000 + origin.index,
                samples=int(audit_spec["samples"]),
                block_days=int(audit_spec["block_days"]),
            )
        def reference_portfolios(start_timestamp: int, end_timestamp: int) -> dict[str, pd.DataFrame]:
            materialized: dict[str, pd.DataFrame] = {}
            if paired_config is None:
                return materialized
            baseline_scaler = fixed_baseline_scaler()
            baseline_policies = [NoTradePolicy(), momentum_policy()]
            if "RelativeMomentumRank_return_12" in paired_config["references"]:
                baseline_policies.append(relative_momentum_policy())
            for baseline_policy in baseline_policies:
                reference = simulate_policy(
                    context.feature_frame,
                    policy=baseline_policy,
                    scaler=baseline_scaler,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    round_trip_cost_bps=round_trip_cost_bps,
                )
                materialized[baseline_policy.policy_id] = reference.portfolio
            if context.incumbent_genomes:
                if context.incumbent_scaler is None:
                    raise ValueError("incumbent genomes require a frozen scaler")
                incumbent_policy = (
                    EnsemblePolicyV4(context.incumbent_genomes, shape=shape)
                    if len(context.incumbent_genomes) > 1
                    else GeneticPrimusPolicyV4(context.incumbent_genomes[0], shape=shape)
                )
                incumbent_reference = simulate_policy(
                    context.feature_frame,
                    policy=incumbent_policy,
                    scaler=context.incumbent_scaler,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    round_trip_cost_bps=round_trip_cost_bps,
                    **simulation_kwargs,
                )
                materialized["Incumbent"] = incumbent_reference.portfolio
            return materialized

        paired_reference_portfolios = reference_portfolios(evolution_start, evolution_end)
        validation_reference_portfolios = reference_portfolios(
            temporal_split.validation_start if temporal_split is not None else evolution_start,
            temporal_split.validation_end if temporal_split is not None else evolution_end,
        )
        audit_reference_portfolios = reference_portfolios(audit_start, audit_end)
        if overlay_modern:
            if training_x is None or training_y is None:
                raise ValueError("binary overlay requires ALLOW/ABSTAIN supervised data")
            warm_start = supervised_seed_overlay_v4(
                training_x,
                training_y,
                shape=shape,
                seed=seed,
                epochs=5,
            ).genome
        else:
            warm_start = supervised_seed_v4(
                context.supervised_features,
                context.supervised_targets,
                shape=shape,
                seed=seed,
                epochs=5,
            ).genome
        evolution_template = {
            key: settings[key]
            for key in (
                "workers",
                "population_size",
                "generations",
                "elite_count",
                "tournament_size",
                "crossover_rate",
                "mutation_rate",
                "mutation_sigma",
            )
        }
        temporary = tempfile.TemporaryDirectory() if context.output_dir is None else None
        manager = temporary if temporary is not None else nullcontext()
        with manager:
            run_dir = context.output_dir if context.output_dir is not None else Path(temporary.name)
            assert run_dir is not None
            run_dir.mkdir(parents=True, exist_ok=True)
            island_results = []
            generation_frames: list[pd.DataFrame] = []
            for island_index in range(int(settings["island_count"])):
                island_seed = seed + island_index * int(settings["seed_stride"])
                evolution = dict(evolution_template)
                evolution["seed"] = island_seed
                island_dir = run_dir / f"island_{island_index:02d}"
                island_dir.mkdir(parents=True, exist_ok=True)
                fingerprint_payload = {
                    "rolling_protocol_sha256": rolling_protocol.sha256,
                    "measurement_protocol_sha256": measurement_protocol.sha256,
                    "origin": origin.to_mapping(),
                    "temporal_split": (
                        temporal_split.to_mapping() if temporal_split is not None else None
                    ),
                    "island_index": island_index,
                    "shape": {
                        "inputs": shape.inputs,
                        "hidden": list(shape.hidden),
                        "outputs": shape.outputs,
                        "maximum_active_features": shape.maximum_active_features,
                    },
                    "fitness": rolling_fitness,
                    "evolution": evolution,
                    "random_immigrant_fraction": settings["random_immigrant_fraction"],
                    "supervised_warm_start_sha256": hashlib.sha256(
                        np.asarray(warm_start, np.float32).tobytes()
                    ).hexdigest(),
                    "incumbent_model_sha256": context.incumbent_model_sha256,
                    "selection_bank_sha256": selection_bank.content_sha256,
                    "audit_bank_sha256": audit_bank.content_sha256,
                    "training_scaler_sha256": hashlib.sha256(
                        json.dumps(
                            training_scaler.to_mapping(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
                fingerprint = hashlib.sha256(
                    json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                checkpoint_path = island_dir / "checkpoint.npz"
                result = run_evolution_v4(
                    context.feature_frame,
                    scaler=training_scaler,
                    shape=shape,
                    start_timestamp=evolution_start,
                    end_timestamp=evolution_end,
                    round_trip_cost_bps=round_trip_cost_bps,
                    folds=folds,
                    fitness_config=rolling_fitness,
                    bootstrap_bank=selection_bank,
                    evolution_config=evolution,
                    warm_start=warm_start,
                    additional_warm_starts=(
                        (
                            context.incumbent_genomes[
                                island_index % len(context.incumbent_genomes)
                            ],
                        )
                        if context.incumbent_genomes
                        else ()
                    ),
                    random_immigrant_fraction=float(settings["random_immigrant_fraction"]),
                    checkpoint_path=checkpoint_path,
                    run_fingerprint=fingerprint,
                    resume=should_resume_island(checkpoint_path),
                    finalist_count=1,
                    simulation_kwargs=simulation_kwargs,
                    paired_reference_portfolios=paired_reference_portfolios,
                )
                island_results.append((island_index, island_seed, result))
                generation_frames.append(
                    result.generations.assign(island_index=island_index, island_seed=island_seed)
                )
            ranked_islands = sorted(
                island_results,
                key=lambda item: item[2].champion.score.ranking_key,
                reverse=True,
            )
            island_champions = tuple(item[2].champion.genome.copy() for item in ranked_islands)
            stability_config = rolling_protocol.payload.get("stability_selection")
            stability_metrics: dict[str, Any] = {"passed": True, "stable_feature_count": 0}
            if stability_config is not None:
                reference_signs = None
                if context.incumbent_genomes:
                    incumbent_coefficients = np.stack(
                        [
                            GeneticPrimusPolicyV4(genome, shape=shape).weights[0][:, 0]
                            for genome in context.incumbent_genomes
                        ]
                    )
                    reference_signs = np.sign(incumbent_coefficients.mean(axis=0)).astype(np.int8)
                stability_metrics = stability_selection_v4(
                    island_champions,
                    shape=shape,
                    minimum_active_frequency=float(stability_config["minimum_active_frequency"]),
                    minimum_sign_agreement=float(stability_config["minimum_sign_agreement"]),
                    active_weight_threshold=float(stability_config["active_weight_threshold"]),
                    reference_signs=reference_signs,
                )
            diversity_metrics: dict[str, Any] = {
                "diversity_gate_passed": True,
                "selection_method": "ranking_only",
            }
            audit = _audit_finalists(
                island_champions,
                context=context,
                shape=shape,
                folds=validation_folds,
                fitness_config=rolling_fitness,
                audit_bank=validation_bank,
                round_trip_cost_bps=round_trip_cost_bps,
                simulation_kwargs=simulation_kwargs,
                paired_reference_portfolios=validation_reference_portfolios,
                scaler=training_scaler,
                start_timestamp=(
                    temporal_split.validation_start if temporal_split is not None else evolution_start
                ),
                end_timestamp=(
                    temporal_split.validation_end if temporal_split is not None else evolution_end
                ),
            )
            if overlay_temporal:
                validation_order = sorted(
                    range(len(audit)),
                    key=lambda index: audit.iloc[index]["_ranking_key"],
                    reverse=True,
                )
                validation_ranked = tuple(island_champions[index] for index in validation_order)
                audit = audit.iloc[validation_order].reset_index(drop=True)
                audit["selection_rank"] = np.arange(1, len(audit) + 1)
                audit["is_selection_champion"] = audit["selection_rank"].eq(1)
                diversity_config = rolling_protocol.payload["search_diversity"]
                assert temporal_split is not None
                anchor = _diversity_anchor_features(
                    context.feature_frame,
                    scaler=training_scaler,
                    start_timestamp=temporal_split.validation_start,
                    end_timestamp=temporal_split.validation_end,
                    maximum_rows=int(diversity_config["anchor_rows"]),
                )
                diverse, diversity_metrics = select_action_diverse_genomes_v4(
                    validation_ranked,
                    shape=shape,
                    anchor_features=anchor,
                    count=max(ensemble_sizes),
                    minimum_disagreement=float(
                        diversity_config["minimum_action_disagreement"]
                    ),
                )
                selected_sha = {
                    hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest()
                    for genome in diverse
                }
                remaining = tuple(
                    genome
                    for genome in validation_ranked
                    if hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest()
                    not in selected_sha
                )
                island_champions = (*diverse, *remaining)
            validation_eligible_finalists = int(audit["audit_deployment_eligible"].sum())
            if not bool(stability_metrics["passed"]):
                audit["audit_deployment_eligible"] = False
                audit["audit_eligibility_blockers"] = audit[
                    "audit_eligibility_blockers"
                ].map(lambda value: ";".join(filter(None, (str(value), "feature_stability"))))
                validation_eligible_finalists = 0
            no_eligible_validation_finalist = bool(
                overlay_temporal and validation_eligible_finalists == 0
            )
            audit = audit.drop(columns=["_ranking_key"])
            ensemble_evaluations = []
            for ensemble_size in ensemble_sizes:
                genomes = island_champions[:ensemble_size]
                selection_simulation, selection_score = _score_ensemble(
                    genomes,
                    context=context,
                    shape=shape,
                    folds=validation_folds,
                    fitness_config=rolling_fitness,
                    bootstrap_bank=validation_bank,
                    round_trip_cost_bps=round_trip_cost_bps,
                    key=f"ensemble-{ensemble_size}-selection",
                    simulation_kwargs=simulation_kwargs,
                    paired_reference_portfolios=validation_reference_portfolios,
                    scaler=training_scaler,
                    start_timestamp=(
                        temporal_split.validation_start if temporal_split is not None else evolution_start
                    ),
                    end_timestamp=(
                        temporal_split.validation_end if temporal_split is not None else evolution_end
                    ),
                )
                _, audit_score = _score_ensemble(
                    genomes,
                    context=context,
                    shape=shape,
                    folds=audit_folds,
                    fitness_config=rolling_fitness,
                    bootstrap_bank=audit_bank,
                    round_trip_cost_bps=round_trip_cost_bps,
                    key=f"ensemble-{ensemble_size}-audit",
                    simulation_kwargs=simulation_kwargs,
                    paired_reference_portfolios=audit_reference_portfolios,
                    scaler=training_scaler,
                    start_timestamp=audit_start,
                    end_timestamp=audit_end,
                )
                ensemble_evaluations.append(
                    (ensemble_size, genomes, selection_simulation, selection_score, audit_score)
                )
            eligible_ensembles = ensemble_evaluations
            if overlay_temporal and not bool(diversity_metrics["diversity_gate_passed"]):
                eligible_ensembles = [
                    item for item in ensemble_evaluations if item[0] == 1
                ]
            (
                selected_size,
                selected_genomes,
                selected_simulation,
                selected_score,
                selected_audit_score,
            ) = max(
                eligible_ensembles, key=lambda item: item[3].ranking_key
            )
            promotion_approved = True
            promotion_metrics: dict[str, Any] = {
                "promotion_gate_status": "BOOTSTRAP_NO_INCUMBENT"
            }
            if paired_config is not None and not context.incumbent_genomes:
                selection_paired_lcb = float(
                    selected_score.metrics["paired_incremental_min_lcb_bps"]
                )
                audit_paired_lcb = float(
                    selected_audit_score.metrics["paired_incremental_min_lcb_bps"]
                )
                allowed_fraction = float(
                    selected_audit_score.metrics["overlay_allowed_fraction"]
                )
                maximum_fraction = float(overlay_config["maximum_allowed_fraction"])
                promotion_approved = bool(
                    selection_paired_lcb > 0.0
                    and audit_paired_lcb > 0.0
                    and allowed_fraction <= maximum_fraction
                )
                promotion_metrics = {
                    "promotion_gate_status": "BOOTSTRAP_PAIRED_REFERENCE_GATE",
                    "paired_selection_lcb_bps": selection_paired_lcb,
                    "paired_audit_lcb_bps": audit_paired_lcb,
                    "overlay_allowed_fraction": allowed_fraction,
                    "overlay_maximum_allowed_fraction": maximum_fraction,
                    "promotion_gate_passed": promotion_approved,
                    "failed_gates": [
                        name
                        for name, passed in (
                            ("paired_selection_lcb", selection_paired_lcb > 0.0),
                            ("paired_audit_lcb", audit_paired_lcb > 0.0),
                            ("overlay_action_coverage", allowed_fraction <= maximum_fraction),
                        )
                        if not passed
                    ],
                }
            if context.incumbent_genomes and not overlay_temporal:
                if context.incumbent_scaler is None:
                    raise ValueError("incumbent genomes require their frozen scaler")
                incumbent_simulation, incumbent_score = _score_ensemble(
                    context.incumbent_genomes,
                    context=context,
                    shape=shape,
                    folds=validation_folds,
                    fitness_config=rolling_fitness,
                    bootstrap_bank=validation_bank,
                    round_trip_cost_bps=round_trip_cost_bps,
                    key="incumbent-selection",
                    scaler=context.incumbent_scaler,
                    simulation_kwargs=simulation_kwargs,
                    paired_reference_portfolios={
                        key: value
                        for key, value in validation_reference_portfolios.items()
                        if key != "Incumbent"
                    },
                    start_timestamp=(
                        temporal_split.validation_start if temporal_split is not None else evolution_start
                    ),
                    end_timestamp=(
                        temporal_split.validation_end if temporal_split is not None else evolution_end
                    ),
                )
                stress_cost = float(promotion_protocol.payload["stress_round_trip_cost_bps"])
                challenger_stress = simulate_policy(
                    context.feature_frame,
                    policy=EnsemblePolicyV4(selected_genomes, shape=shape),
                    scaler=training_scaler,
                    start_timestamp=(
                        temporal_split.validation_start if temporal_split is not None else evolution_start
                    ),
                    end_timestamp=(
                        temporal_split.validation_end if temporal_split is not None else evolution_end
                    ),
                    round_trip_cost_bps=stress_cost,
                    **simulation_kwargs,
                )
                incumbent_stress = simulate_policy(
                    context.feature_frame,
                    policy=EnsemblePolicyV4(context.incumbent_genomes, shape=shape),
                    scaler=context.incumbent_scaler,
                    start_timestamp=(
                        temporal_split.validation_start if temporal_split is not None else evolution_start
                    ),
                    end_timestamp=(
                        temporal_split.validation_end if temporal_split is not None else evolution_end
                    ),
                    round_trip_cost_bps=stress_cost,
                    **simulation_kwargs,
                )
                drift_start = audit_start
                challenger_drift = simulate_policy(
                    context.feature_frame,
                    policy=EnsemblePolicyV4(selected_genomes, shape=shape),
                    scaler=training_scaler,
                    start_timestamp=drift_start,
                    end_timestamp=audit_end,
                    round_trip_cost_bps=round_trip_cost_bps,
                    **simulation_kwargs,
                )
                incumbent_drift = simulate_policy(
                    context.feature_frame,
                    policy=EnsemblePolicyV4(context.incumbent_genomes, shape=shape),
                    scaler=context.incumbent_scaler,
                    start_timestamp=drift_start,
                    end_timestamp=audit_end,
                    round_trip_cost_bps=round_trip_cost_bps,
                    **simulation_kwargs,
                )
                promotion = evaluate_promotion_gate(
                    paired=paired_daily_returns(selected_simulation, incumbent_simulation),
                    stress_paired=paired_daily_returns(challenger_stress, incumbent_stress),
                    outer_slices=validation_folds,
                    challenger_metrics=selected_score.metrics,
                    incumbent_metrics=incumbent_score.metrics,
                    challenger_actions=challenger_drift.action_counts,
                    incumbent_actions=incumbent_drift.action_counts,
                    challenger_centroid=ensemble_centroid(selected_genomes),
                    incumbent_centroid=ensemble_centroid(context.incumbent_genomes),
                    selection_bank=selection_bank,
                    audit_bank=promotion_resample_bank,
                    protocol=promotion_protocol,
                )
                promotion_approved = promotion.passed
                promotion_metrics = dict(promotion.metrics)
            if overlay_modern:
                selection_lcb = float(
                    selected_score.metrics["paired_incremental_min_lcb_bps"]
                )
                temporal_audit_lcb = float(
                    selected_audit_score.metrics["paired_incremental_min_lcb_bps"]
                )
                temporal_audit_passed = bool(
                    temporal_audit_lcb > 0.0
                    and selected_audit_score.metrics.get(
                        "coverage_constraint_passed", True
                    )
                )
                predeployment_passed = bool(
                    selection_lcb > 0.0
                    and temporal_audit_passed
                    and diversity_metrics.get("diversity_gate_passed", True)
                    and stability_metrics.get("passed", True)
                    and not no_eligible_validation_finalist
                )
                promotion_approved = False
                failed_predeployment = []
                if selection_lcb <= 0.0:
                    failed_predeployment.append("temporal_validation_lcb")
                if not temporal_audit_passed:
                    failed_predeployment.append("temporal_audit_lcb")
                if not diversity_metrics.get("diversity_gate_passed", True):
                    failed_predeployment.append("action_diversity")
                if not stability_metrics.get("passed", True):
                    failed_predeployment.append("feature_stability")
                if no_eligible_validation_finalist:
                    failed_predeployment.append("no_eligible_validation_finalist")
                promotion_metrics = {
                    "promotion_gate_status": (
                        validation_candidate_status_v4(validation_eligible_finalists)
                        if no_eligible_validation_finalist
                        else "TRAINING_DIAGNOSTIC_ONLY"
                    ),
                    "candidate_status": validation_candidate_status_v4(
                        validation_eligible_finalists
                    ),
                    "candidate_deployable": False,
                    "validation_eligible_finalists": validation_eligible_finalists,
                    "predeployment_training_gate_passed": predeployment_passed,
                    "temporal_audit_passed": temporal_audit_passed,
                    "training_fold_count": int(len(folds)),
                    "training_folds_are_outer_slices": False,
                    "paired_selection_lcb_bps": selection_lcb,
                    "paired_audit_lcb_bps": temporal_audit_lcb,
                    "promotion_gate_passed": False,
                    "failed_gates": [
                        *failed_predeployment,
                        "insufficient_completed_outer_origins",
                    ],
                }
            ensemble_frame = pd.DataFrame(
                [
                    {
                        "ensemble_size": size,
                        "selected_by_selection_bank": size == selected_size,
                        "selection_fitness": selection_score.fitness,
                        "selection_failed_gates": ";".join(selection_score.failed_gates),
                        "audit_fitness": audit_score.fitness,
                        "audit_failed_gates": ";".join(audit_score.failed_gates),
                        "member_genome_sha256": ";".join(
                            hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest()
                            for genome in genomes
                        ),
                    }
                    for size, genomes, _, selection_score, audit_score in ensemble_evaluations
                ]
            )
            generations = pd.concat(generation_frames, ignore_index=True)
            selection_upper_bound = (
                int(settings["island_count"])
                * int(settings["population_size"])
                * int(settings["generations"])
            )
            audit_hypotheses = int(len(island_champions)) if overlay_modern else 0
            ensemble_hypotheses = int(len(ensemble_evaluations)) if overlay_modern else 0
            registered_upper_bound = (
                selection_upper_bound + audit_hypotheses + ensemble_hypotheses
            )
            finalist_hashes = {
                hashlib.sha256(np.asarray(genome, np.float32).tobytes()).hexdigest()
                for genome in island_champions
            }
            evaluated_hashes = {
                genome_sha
                for _, _, result in island_results
                for genome_sha in result.evaluated_genome_sha256
            }
            hypothesis_budget = {
                "schema_version": "exia.genetic_primus.hypothesis_budget/1",
                "origin_index": origin.index,
                "island_count": int(settings["island_count"]),
                "population_size": int(settings["population_size"]),
                "generations": int(settings["generations"]),
                "selection_candidate_upper_bound": selection_upper_bound,
                "actual_unique_selection_genomes": int(len(evaluated_hashes)),
                "evaluated_genome_sha256": sorted(evaluated_hashes),
                "audit_hypotheses": audit_hypotheses,
                "ensemble_hypotheses": ensemble_hypotheses,
                "registered_unique_hypothesis_upper_bound": registered_upper_bound,
                "independent_audit_finalists": int(len(island_champions)),
                "unique_audit_finalist_sha256": sorted(finalist_hashes),
                "selection_bias_penalty": (
                    dict(multiple_testing_config)
                    if multiple_testing_config is not None
                    else None
                ),
                "within_registered_budget": bool(
                    multiple_testing_config is None
                    or (
                        len(evaluated_hashes) <= selection_upper_bound
                        and registered_upper_bound
                        <= int(multiple_testing_config["max_hypotheses_per_origin"])
                    )
                ),
            }
            hypothesis_budget["payload_sha256"] = hashlib.sha256(
                json.dumps(
                    hypothesis_budget, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if context.output_dir is not None:
                save_bootstrap_bank(run_dir / "selection_bootstrap_bank.npz", selection_bank)
                if temporal_split is not None:
                    save_bootstrap_bank(
                        run_dir / "temporal_validation_bootstrap_bank.npz",
                        validation_bank,
                    )
                save_bootstrap_bank(run_dir / "audit_bootstrap_bank.npz", audit_bank)
                audit.to_parquet(
                    run_dir
                    / (
                        "temporal_validation_top_k.parquet"
                        if overlay_temporal
                        else "audit_top_k.parquet"
                    ),
                    index=False,
                )
                ensemble_frame.to_parquet(run_dir / "ensemble_evaluation.parquet", index=False)
                generations.to_parquet(run_dir / "generation_metrics.parquet", index=False)
                (run_dir / "hypothesis_budget.json").write_text(
                    json.dumps(hypothesis_budget, indent=2) + "\n", encoding="utf-8"
                )
                (run_dir / "scaler.json").write_text(
                    json.dumps(training_scaler.to_mapping(), indent=2) + "\n", encoding="utf-8"
                )
            audit_champion = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in audit.iloc[0].to_dict().items()
            }
            metadata = {
                "seed_base": seed,
                "island_seeds": [item[1] for item in ranked_islands],
                "island_count": int(settings["island_count"]),
                "random_immigrant_fraction": float(settings["random_immigrant_fraction"]),
                "incumbent_model_sha256": context.incumbent_model_sha256,
                "selection_bank_sha256": selection_bank.content_sha256,
                "temporal_validation_bank_sha256": validation_bank.content_sha256,
                "audit_bank_sha256": audit_bank.content_sha256,
                "folds": list(folds),
                "training_folds": list(folds),
                "temporal_validation_folds": list(validation_folds),
                "temporal_audit_folds": list(audit_folds),
                "temporal_split": (
                    temporal_split.to_mapping() if temporal_split is not None else None
                ),
                "scaler_fit_scope": (
                    temporal_config["scaler_fit_scope"]
                    if temporal_config is not None
                    else "full_training_window"
                ),
                "search_diversity": dict(diversity_metrics),
                "stability_selection": dict(stability_metrics),
                "selected_ensemble_size": selected_size,
                "selection_metrics": dict(selected_score.metrics),
                "audit_selected_ensemble_metrics": dict(selected_audit_score.metrics),
                "ensemble_candidates": ensemble_frame.to_dict(orient="records"),
                "promotion_protocol_sha256": promotion_protocol.sha256,
                "promotion_approved": promotion_approved,
                "promotion_metrics": promotion_metrics,
                "candidate_status": promotion_metrics.get(
                    "candidate_status", "LEGACY_PROMOTION_EVALUATION"
                ),
                "candidate_deployable": bool(
                    promotion_approved
                    and promotion_metrics.get("candidate_deployable", True)
                ),
                "audit_selection_champion": audit_champion,
                "temporal_validation_champion": (
                    audit_champion if overlay_temporal else None
                ),
                "audit_used_for_selection": False,
                "policy_overlay": dict(overlay_config) if overlay_config is not None else None,
                "paired_incremental_fitness": (
                    dict(paired_config) if paired_config is not None else None
                ),
                "adaptive_multiple_testing": hypothesis_budget,
            }
            if context.output_dir is not None:
                (run_dir / "promotion_gate.json").write_text(
                    json.dumps(
                        {
                            "promotion_approved": promotion_approved,
                            "promotion_protocol_sha256": promotion_protocol.sha256,
                            "metrics": promotion_metrics,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return TrainedOriginModel(
                genome=selected_genomes[0].copy(),
                shape=shape,
                trainer_id="bounded_v4_island_ensemble_origin_trainer/1",
                metadata=metadata,
                ensemble_genomes=tuple(genome.copy() for genome in selected_genomes),
                promotion_approved=promotion_approved,
                trained_scaler=training_scaler,
            )

    return train
