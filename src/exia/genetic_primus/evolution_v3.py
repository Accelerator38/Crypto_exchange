from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .evolution import (
    Champion,
    EvolutionResult,
    _breed,
    _generation_row,
    _initial_population,
    _load_checkpoint,
    _save_checkpoint,
    _score_key,
)
from .features import RobustScaler
from .fitness_v3 import (
    ordinal_selection_scores_v3,
    score_simulation_v3,
    v3_ranking_key,
)
from .policy import GeneticPrimusPolicy
from .simulator import simulate_policy


_WORKER_CONTEXT: dict[str, Any] = {}


def _initialize_worker(
    frame: pd.DataFrame,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    fitness_config: Mapping[str, Any],
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "frame": frame,
        "scaler": scaler,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "fitness_config": dict(fitness_config),
    }


def _evaluate_full(
    genome: np.ndarray,
    *,
    frame: pd.DataFrame,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    fitness_config: Mapping[str, Any],
):
    simulation = simulate_policy(
        frame,
        policy=GeneticPrimusPolicy(genome),
        scaler=scaler,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    score = score_simulation_v3(
        simulation,
        config=fitness_config,
        key=_score_key(genome),
    )
    return simulation, score


def _evaluate_worker(item: tuple[int, np.ndarray]):
    index, genome = item
    simulation, score = _evaluate_full(genome, **_WORKER_CONTEXT)
    del simulation
    return int(index), float(score.fitness), v3_ranking_key(score)


def _v3_generation_row(
    generation: int,
    raw_fitness: np.ndarray,
    ordinal_scores: np.ndarray,
    champion: Champion,
) -> dict[str, Any]:
    row = _generation_row(generation, raw_fitness, ordinal_scores, champion)
    metrics = champion.score.metrics
    row.update(
        {
            "overall_failure_count": int(metrics["overall_failure_count"]),
            "slice_failure_count": int(metrics["slice_failure_count"]),
            "supported_slice_count": int(metrics["supported_slice_count"]),
            "worst_supported_slice_lcb_bps": float(
                metrics["worst_supported_slice_lcb_bps"]
            ),
            "temporal_gate_passed": bool(metrics["temporal_gate_passed"]),
            "temporal_margin_bps": float(metrics["temporal_margin_bps"]),
            "v3_ranking_key": "|".join(
                f"{value:.12g}" for value in v3_ranking_key(champion.score)
            ),
        }
    )
    return row


def run_evolution_v3(
    frame: pd.DataFrame,
    *,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    fitness_config: Mapping[str, Any],
    evolution_config: Mapping[str, Any],
    warm_start: np.ndarray,
    population_size: int | None = None,
    generations: int | None = None,
    on_generation: Callable[[Champion, pd.DataFrame], None] | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    run_fingerprint: str = "",
) -> EvolutionResult:
    settings = dict(evolution_config)
    if population_size is not None:
        settings["population_size"] = int(population_size)
        settings["elite_count"] = min(
            int(settings["elite_count"]), max(1, int(population_size) // 4)
        )
    if generations is not None:
        settings["generations"] = int(generations)
    rng = np.random.default_rng(int(settings["seed"]))
    history: list[dict[str, Any]] = []
    champions: list[Champion] = []
    global_champion: Champion | None = None
    start_generation = 0
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("resume requested without a Genetic_Primus V3 checkpoint")
        checkpoint = _load_checkpoint(checkpoint_path, fingerprint=run_fingerprint)
        population = checkpoint["population"]
        if len(population) != int(settings["population_size"]):
            raise ValueError("checkpoint population size does not match V3 settings")
        rng.bit_generator.state = checkpoint["rng_state"]
        history = list(checkpoint["history"])
        start_generation = int(checkpoint["next_generation"])
        simulation, score = _evaluate_full(
            checkpoint["global_genome"],
            frame=frame,
            scaler=scaler,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            round_trip_cost_bps=round_trip_cost_bps,
            fitness_config=fitness_config,
        )
        global_champion = Champion(
            generation=int(checkpoint["global_generation"]),
            genome=checkpoint["global_genome"],
            simulation=simulation,
            score=score,
        )
    else:
        population = _initial_population(
            rng,
            population_size=int(settings["population_size"]),
            warm_start=np.asarray(warm_start, dtype=np.float32),
            mutation_sigma=float(settings["mutation_sigma"]),
        )

    workers = max(1, int(settings.get("workers", 1)))
    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(
                frame,
                scaler,
                start_timestamp,
                end_timestamp,
                round_trip_cost_bps,
                dict(fitness_config),
            ),
        )
        if workers > 1
        else None
    )
    try:
        for generation in range(start_generation, int(settings["generations"])):
            raw_fitness = np.empty(len(population), dtype=np.float64)
            keys: list[tuple[float, ...] | None] = [None] * len(population)
            if executor is not None:
                evaluated = executor.map(
                    _evaluate_worker,
                    ((index, genome) for index, genome in enumerate(population)),
                    chunksize=1,
                )
                for index, fitness, ranking_key in evaluated:
                    raw_fitness[index] = fitness
                    keys[index] = ranking_key
            else:
                for index, genome in enumerate(population):
                    _, score = _evaluate_full(
                        genome,
                        frame=frame,
                        scaler=scaler,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        round_trip_cost_bps=round_trip_cost_bps,
                        fitness_config=fitness_config,
                    )
                    raw_fitness[index] = score.fitness
                    keys[index] = v3_ranking_key(score)
            complete_keys = [key for key in keys if key is not None]
            if len(complete_keys) != len(population):
                raise RuntimeError("V3 population evaluation is incomplete")
            ordinal = ordinal_selection_scores_v3(complete_keys)
            best_index = max(range(len(population)), key=lambda index: complete_keys[index])
            simulation, score = _evaluate_full(
                population[best_index],
                frame=frame,
                scaler=scaler,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                round_trip_cost_bps=round_trip_cost_bps,
                fitness_config=fitness_config,
            )
            champion = Champion(
                generation=generation,
                genome=population[best_index].copy(),
                simulation=simulation,
                score=score,
            )
            champions.append(champion)
            if global_champion is None or v3_ranking_key(score) > v3_ranking_key(
                global_champion.score
            ):
                global_champion = champion
            history.append(_v3_generation_row(generation, raw_fitness, ordinal, champion))
            generation_frame = pd.DataFrame(history)
            next_population = population
            if generation + 1 < int(settings["generations"]):
                next_population = _breed(population, ordinal, rng=rng, settings=settings)
            if checkpoint_path is not None:
                _save_checkpoint(
                    checkpoint_path,
                    fingerprint=run_fingerprint,
                    next_generation=generation + 1,
                    population=next_population,
                    rng=rng,
                    history=history,
                    global_champion=global_champion,
                    complete=generation + 1 >= int(settings["generations"]),
                )
            if on_generation is not None:
                on_generation(champion, generation_frame)
            population = next_population
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if global_champion is None:
        raise RuntimeError("V3 evolution produced no champion")
    return EvolutionResult(
        champion=global_champion,
        generations=pd.DataFrame(history),
        generation_champions=tuple(champions),
    )
