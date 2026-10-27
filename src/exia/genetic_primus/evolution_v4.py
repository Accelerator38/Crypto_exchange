from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .evolution import _breed, _load_checkpoint, _save_checkpoint, _score_key
from .features import RobustScaler
from .fitness_v4 import V4Score, score_simulation_v4
from .measurement import BootstrapBank
from .policy_v4 import GeneticPrimusPolicyV4, V4NetworkShape, genome_sparsity_v4
from .simulator import PrimusSimulation, simulate_policy


@dataclass(frozen=True)
class V4Champion:
    generation: int
    genome: np.ndarray
    simulation: PrimusSimulation
    score: V4Score


@dataclass(frozen=True)
class V4EvolutionResult:
    champion: V4Champion
    generations: pd.DataFrame
    finalists: tuple[np.ndarray, ...]
    evaluated_genome_sha256: tuple[str, ...] = ()


_WORKER_CONTEXT: dict[str, Any] = {}


def _initialize_worker(
    frame: pd.DataFrame,
    scaler: RobustScaler,
    shape: V4NetworkShape,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    folds: Sequence[Mapping[str, str]],
    fitness_config: Mapping[str, Any],
    bootstrap_bank: BootstrapBank,
    simulation_kwargs: Mapping[str, Any],
    paired_reference_portfolios: Mapping[str, pd.DataFrame],
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "frame": frame,
        "scaler": scaler,
        "shape": shape,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "folds": tuple(dict(item) for item in folds),
        "fitness_config": dict(fitness_config),
        "bootstrap_bank": bootstrap_bank,
        "simulation_kwargs": dict(simulation_kwargs),
        "paired_reference_portfolios": dict(paired_reference_portfolios),
    }


def _evaluate_full(
    genome: np.ndarray,
    *,
    frame: pd.DataFrame,
    scaler: RobustScaler,
    shape: V4NetworkShape,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    folds: Sequence[Mapping[str, str]],
    fitness_config: Mapping[str, Any],
    bootstrap_bank: BootstrapBank,
    simulation_kwargs: Mapping[str, Any] | None = None,
    paired_reference_portfolios: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[PrimusSimulation, V4Score]:
    simulation = simulate_policy(
        frame,
        policy=GeneticPrimusPolicyV4(genome, shape=shape),
        scaler=scaler,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        round_trip_cost_bps=round_trip_cost_bps,
        **dict(simulation_kwargs or {}),
    )
    score = score_simulation_v4(
        simulation,
        folds=folds,
        config=fitness_config,
        key=_score_key(genome),
        bootstrap_bank=bootstrap_bank,
        complexity_metrics=(
            genome_sparsity_v4(
                genome,
                shape=shape,
                active_weight_threshold=float(fitness_config["recency"]["active_weight_threshold"]),
            )
            if "recency" in fitness_config
            else None
        ),
        paired_reference_portfolios=paired_reference_portfolios,
    )
    return simulation, score


def _evaluate_worker(item: tuple[int, np.ndarray]):
    index, genome = item
    _, score = _evaluate_full(genome, **_WORKER_CONTEXT)
    return int(index), float(score.fitness), score.ranking_key


def _ordinal(keys: list[tuple[float, ...]]) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda index: (keys[index], -index))
    ranks = np.empty(len(keys), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def _initial_population(
    rng: np.random.Generator,
    *,
    shape: V4NetworkShape,
    population_size: int,
    warm_start: np.ndarray,
    mutation_sigma: float,
    additional_warm_starts: Sequence[np.ndarray] = (),
    random_immigrant_fraction: float | None = None,
) -> np.ndarray:
    bases = [np.asarray(warm_start, dtype=np.float32).reshape(-1)]
    bases.extend(np.asarray(value, dtype=np.float32).reshape(-1) for value in additional_warm_starts)
    if any(len(base) != shape.genome_size for base in bases):
        raise ValueError("V4 warm-start shape mismatch")
    if random_immigrant_fraction is not None:
        fraction = float(random_immigrant_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("V4 random immigrant fraction must be in [0, 1)")
        immigrant_count = int(np.ceil(population_size * fraction))
        seeded_count = population_size - immigrant_count
        genomes = [base.copy() for base in bases[:seeded_count]]
        mutation_index = 0
        while len(genomes) < seeded_count:
            base = bases[mutation_index % len(bases)]
            genomes.append(
                (base + rng.normal(0.0, mutation_sigma, len(base))).astype(np.float32)
            )
            mutation_index += 1
        while len(genomes) < population_size:
            genomes.append(GeneticPrimusPolicyV4.random(rng, shape=shape).genome)
        return np.stack(genomes)[rng.permutation(population_size)]
    base = bases[0]
    genomes = [base.copy()]
    warm_count = max(1, population_size // 4)
    for _ in range(warm_count - 1):
        genomes.append((base + rng.normal(0.0, mutation_sigma, len(base))).astype(np.float32))
    while len(genomes) < population_size:
        genomes.append(GeneticPrimusPolicyV4.random(rng, shape=shape).genome)
    return np.stack(genomes)


def _generation_row(
    generation: int,
    raw_fitness: np.ndarray,
    champion: V4Champion,
) -> dict[str, Any]:
    metrics = champion.score.metrics
    row = {
        "generation": int(generation),
        "best_fitness": float(champion.score.fitness),
        "median_population_fitness": float(np.median(raw_fitness)),
        "p90_population_fitness": float(np.quantile(raw_fitness, 0.9)),
        "trades": int(metrics["trades"]),
        "mean_daily_bps": metrics["mean_daily_bps"],
        "median_daily_bps": metrics["median_daily_bps"],
        "lcb_daily_bps": metrics["lcb_daily_bps"],
        "positive_fold_means": int(metrics["positive_fold_means"]),
        "worst_fold_mean_bps": float(metrics["worst_fold_mean_bps"]),
        "median_fold_lcb_bps": float(metrics["median_fold_lcb_bps"]) if metrics["median_fold_lcb_bps"] is not None else None,
        "worst_fold_lcb_bps": float(metrics["worst_fold_lcb_bps"]) if metrics["worst_fold_lcb_bps"] is not None else None,
        "max_drawdown": float(metrics["max_drawdown"]),
        "fills_per_day": float(metrics["fills_per_day"]),
        "temporal_gate_passed": bool(metrics["temporal_gate_passed"]),
        "failed_gates": ";".join(metrics["failed_gates"]),
        "ranking_key": "|".join(f"{value:.12g}" for value in champion.score.ranking_key),
    }
    if "search_objective" in metrics:
        row.update({name: metrics[name] for name in (
            "search_objective", "objective_sha256", "drawdown_budget_percent",
            "raw_net_log_growth_bps_per_day", "final_capital_quote",
            "validation_status", "stability_bootstrap_evaluated", "ranking_semantics",
        )})
        row["validation_failed_gates"] = ";".join(metrics["validation_failed_gates"])
    if "recency_primary_lcb_bps" in metrics:
        row.update(
            {
                "recency_primary_lcb_bps": float(metrics["recency_primary_lcb_bps"]),
                "recency_worst_window_lcb_bps": float(
                    metrics["recency_worst_window_lcb_bps"]
                ),
                "active_parameter_count": int(metrics["recency_active_parameter_count"]),
                "active_feature_count": int(metrics["recency_active_feature_count"]),
                "feature_sparsity": float(metrics["recency_feature_sparsity"]),
                "parsimony_penalty_bps": float(metrics["recency_parsimony_penalty_bps"]),
                "recency_total_penalty_bps": float(metrics["recency_total_penalty_bps"]),
            }
        )
        row.update(
            {
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("recency_ess_")
            }
        )
    return row


def run_evolution_v4(
    frame: pd.DataFrame,
    *,
    scaler: RobustScaler,
    shape: V4NetworkShape,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    folds: Sequence[Mapping[str, str]],
    fitness_config: Mapping[str, Any],
    bootstrap_bank: BootstrapBank,
    evolution_config: Mapping[str, Any],
    warm_start: np.ndarray,
    checkpoint_path: Path,
    run_fingerprint: str,
    resume: bool,
    finalist_count: int,
    additional_warm_starts: Sequence[np.ndarray] = (),
    random_immigrant_fraction: float | None = None,
    on_generation: Callable[[V4Champion, pd.DataFrame], None] | None = None,
    simulation_kwargs: Mapping[str, Any] | None = None,
    paired_reference_portfolios: Mapping[str, pd.DataFrame] | None = None,
) -> V4EvolutionResult:
    settings = dict(evolution_config)
    rng = np.random.default_rng(int(settings["seed"]))
    history: list[dict[str, Any]] = []
    evaluated_genome_sha256: set[str] = set()
    global_champion: V4Champion | None = None
    start_generation = 0
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(checkpoint_path, fingerprint=run_fingerprint)
        population = checkpoint["population"]
        rng.bit_generator.state = checkpoint["rng_state"]
        history = list(checkpoint["history"])
        evaluated_genome_sha256.update(checkpoint["evaluated_genome_sha256"])
        start_generation = int(checkpoint["next_generation"])
        simulation, score = _evaluate_full(
            checkpoint["global_genome"],
            frame=frame,
            scaler=scaler,
            shape=shape,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            round_trip_cost_bps=round_trip_cost_bps,
            folds=folds,
            fitness_config=fitness_config,
            bootstrap_bank=bootstrap_bank,
            simulation_kwargs=simulation_kwargs,
            paired_reference_portfolios=paired_reference_portfolios,
        )
        global_champion = V4Champion(
            generation=int(checkpoint["global_generation"]),
            genome=checkpoint["global_genome"],
            simulation=simulation,
            score=score,
        )
    else:
        population = _initial_population(
            rng,
            shape=shape,
            population_size=int(settings["population_size"]),
            warm_start=warm_start,
            mutation_sigma=float(settings["mutation_sigma"]),
            additional_warm_starts=additional_warm_starts,
            random_immigrant_fraction=random_immigrant_fraction,
        )
    workers = max(1, int(settings.get("workers", 1)))
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            frame,
            scaler,
            shape,
            start_timestamp,
            end_timestamp,
            round_trip_cost_bps,
            folds,
            dict(fitness_config),
            bootstrap_bank,
            dict(simulation_kwargs or {}),
            dict(paired_reference_portfolios or {}),
        ),
    ) if workers > 1 else None
    final_population: np.ndarray | None = None
    final_keys: list[tuple[float, ...]] | None = None
    try:
        for generation in range(start_generation, int(settings["generations"])):
            evaluated_genome_sha256.update(
                _score_key(genome).removeprefix("genome:") for genome in population
            )
            raw = np.empty(len(population), dtype=np.float64)
            keys: list[tuple[float, ...] | None] = [None] * len(population)
            if executor is not None:
                for index, fitness, ranking_key in executor.map(
                    _evaluate_worker,
                    ((index, genome) for index, genome in enumerate(population)),
                    chunksize=1,
                ):
                    raw[index] = fitness
                    keys[index] = ranking_key
            else:
                for index, genome in enumerate(population):
                    _, score = _evaluate_full(
                        genome,
                        frame=frame,
                        scaler=scaler,
                        shape=shape,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        round_trip_cost_bps=round_trip_cost_bps,
                        folds=folds,
                        fitness_config=fitness_config,
                        bootstrap_bank=bootstrap_bank,
                        simulation_kwargs=simulation_kwargs,
                        paired_reference_portfolios=paired_reference_portfolios,
                    )
                    raw[index] = score.fitness
                    keys[index] = score.ranking_key
            complete = [key for key in keys if key is not None]
            if len(complete) != len(population):
                raise RuntimeError("V4 population evaluation incomplete")
            ordinal = _ordinal(complete)
            final_population = population.copy()
            final_keys = complete
            best_index = max(range(len(population)), key=lambda index: complete[index])
            simulation, score = _evaluate_full(
                population[best_index],
                frame=frame,
                scaler=scaler,
                shape=shape,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                round_trip_cost_bps=round_trip_cost_bps,
                folds=folds,
                fitness_config=fitness_config,
                bootstrap_bank=bootstrap_bank,
                simulation_kwargs=simulation_kwargs,
                paired_reference_portfolios=paired_reference_portfolios,
            )
            champion = V4Champion(generation, population[best_index].copy(), simulation, score)
            if global_champion is None or score.ranking_key > global_champion.score.ranking_key:
                global_champion = champion
            history.append(_generation_row(generation, raw, champion))
            next_population = population
            if generation + 1 < int(settings["generations"]):
                next_population = _breed(population, ordinal, rng=rng, settings=settings)
            _save_checkpoint(
                checkpoint_path,
                fingerprint=run_fingerprint,
                next_generation=generation + 1,
                population=next_population,
                rng=rng,
                history=history,
                global_champion=global_champion,
                complete=generation + 1 >= int(settings["generations"]),
                evaluated_genome_sha256=evaluated_genome_sha256,
            )
            frame_history = pd.DataFrame(history)
            if on_generation is not None:
                on_generation(champion, frame_history)
            population = next_population
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if global_champion is None:
        raise RuntimeError("V4 evolution produced no champion")
    if finalist_count < 1:
        raise ValueError("V4 audit finalist count must be positive")
    if final_population is None or final_keys is None:
        final_population = population.copy()
        final_keys = []
        for genome in final_population:
            _, score = _evaluate_full(
                genome,
                frame=frame,
                scaler=scaler,
                shape=shape,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                round_trip_cost_bps=round_trip_cost_bps,
                folds=folds,
            fitness_config=fitness_config,
            bootstrap_bank=bootstrap_bank,
            simulation_kwargs=simulation_kwargs,
            paired_reference_portfolios=paired_reference_portfolios,
            )
            final_keys.append(score.ranking_key)
    order = sorted(range(len(final_population)), key=lambda index: final_keys[index], reverse=True)
    finalists: list[np.ndarray] = [global_champion.genome.copy()]
    seen = {_score_key(global_champion.genome)}
    for index in order:
        genome = final_population[index]
        identity = _score_key(genome)
        if identity in seen:
            continue
        finalists.append(genome.copy())
        seen.add(identity)
        if len(finalists) >= finalist_count:
            break
    return V4EvolutionResult(
        global_champion,
        pd.DataFrame(history),
        tuple(finalists),
        tuple(sorted(evaluated_genome_sha256)),
    )
