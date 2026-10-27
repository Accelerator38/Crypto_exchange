from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import RobustScaler
from .fitness import FitnessResult, score_simulation
from .policy import GeneticPrimusPolicy, NetworkShape
from .simulator import PrimusSimulation, simulate_policy


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


def _evaluate_worker(item: tuple[int, np.ndarray]) -> tuple[int, float, float]:
    index, genome = item
    context = _WORKER_CONTEXT
    simulation = simulate_policy(
        context["frame"],
        policy=GeneticPrimusPolicy(genome, shape=NetworkShape()),
        scaler=context["scaler"],
        start_timestamp=context["start_timestamp"],
        end_timestamp=context["end_timestamp"],
        round_trip_cost_bps=context["round_trip_cost_bps"],
    )
    score = score_simulation(
        simulation,
        config=context["fitness_config"],
        key=_score_key(genome),
    )
    return int(index), float(score.fitness), float(score.selection_score)


def _evaluate_full(
    genome: np.ndarray,
    *,
    frame: pd.DataFrame,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    fitness_config: Mapping[str, Any],
) -> tuple[PrimusSimulation, FitnessResult]:
    simulation = simulate_policy(
        frame,
        policy=GeneticPrimusPolicy(genome, shape=NetworkShape()),
        scaler=scaler,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    score = score_simulation(
        simulation,
        config=fitness_config,
        key=_score_key(genome),
    )
    return simulation, score


@dataclass(frozen=True)
class Champion:
    generation: int
    genome: np.ndarray
    simulation: PrimusSimulation
    score: FitnessResult


@dataclass(frozen=True)
class EvolutionResult:
    champion: Champion
    generations: pd.DataFrame
    generation_champions: tuple[Champion, ...]


def _random_genome(rng: np.random.Generator) -> np.ndarray:
    return GeneticPrimusPolicy.random(rng).genome


def _initial_population(
    rng: np.random.Generator,
    *,
    population_size: int,
    warm_start: np.ndarray | None,
    mutation_sigma: float,
) -> np.ndarray:
    genomes: list[np.ndarray] = []
    if warm_start is not None:
        base = np.asarray(warm_start, dtype=np.float32)
        genomes.append(base.copy())
        warm_count = max(1, population_size // 4)
        for _ in range(warm_count - 1):
            genomes.append(
                (base + rng.normal(0.0, mutation_sigma, size=len(base))).astype(np.float32)
            )
    while len(genomes) < population_size:
        genomes.append(_random_genome(rng))
    return np.stack(genomes[:population_size])


def _tournament(
    rng: np.random.Generator,
    fitness: np.ndarray,
    size: int,
) -> int:
    choices = rng.integers(0, len(fitness), size=int(size))
    return int(choices[int(np.argmax(fitness[choices]))])


def _breed(
    population: np.ndarray,
    fitness: np.ndarray,
    *,
    rng: np.random.Generator,
    settings: Mapping[str, Any],
) -> np.ndarray:
    elite_count = int(settings["elite_count"])
    elite_indices = np.argsort(fitness)[-elite_count:][::-1]
    next_population = [population[index].copy() for index in elite_indices]
    while len(next_population) < len(population):
        first = population[_tournament(rng, fitness, int(settings["tournament_size"]))]
        second = population[_tournament(rng, fitness, int(settings["tournament_size"]))]
        child = first.copy()
        if rng.random() < float(settings["crossover_rate"]):
            alpha = rng.random(len(child), dtype=np.float32)
            child = alpha * first + (1.0 - alpha) * second
        mask = rng.random(len(child)) < float(settings["mutation_rate"])
        if np.any(mask):
            child[mask] += rng.normal(
                0.0, float(settings["mutation_sigma"]), size=int(mask.sum())
            ).astype(np.float32)
        next_population.append(child.astype(np.float32))
    return np.stack(next_population)


def _generation_row(
    generation: int,
    raw_fitness_values: np.ndarray,
    selection_values: np.ndarray,
    champion: Champion,
) -> dict[str, Any]:
    metrics = champion.score.metrics
    row: dict[str, Any] = {
        "generation": int(generation),
        "best_fitness": float(champion.score.fitness),
        "best_selection_score": float(champion.score.selection_score),
        "median_fitness": float(np.median(raw_fitness_values)),
        "p90_fitness": float(np.quantile(raw_fitness_values, 0.9)),
        "champion_trades": int(metrics["trades"]),
        "champion_mean_bps": metrics["mean_bps"],
        "champion_median_bps": metrics["median_bps"],
        "champion_lcb_bps": metrics["lcb_bps"],
        "champion_max_drawdown": float(metrics["max_drawdown"]),
        "champion_constraint_violation": float(metrics["constraint_violation"]),
        "champion_turnover_fills": int(metrics["turnover_fills"]),
        "supported_period_rate": float(metrics["supported_period_rate"]),
        "max_direction_share": float(metrics["max_direction_share"]),
        "max_state_share": float(metrics["max_state_share"]),
        "max_symbol_share": float(metrics["max_symbol_share"]),
        "short_actions": int(metrics["action_counts"]["SHORT"]),
        "flat_actions": int(metrics["action_counts"]["FLAT"]),
        "long_actions": int(metrics["action_counts"]["LONG"]),
        "failed_gates": ";".join(metrics["failed_gates"]),
    }
    for slice_row in champion.score.slices.itertuples(index=False):
        if slice_row.scope != "market_state":
            continue
        prefix = f"state_{slice_row.value}"
        row[f"{prefix}_trades"] = int(slice_row.trades)
        row[f"{prefix}_mean_bps"] = slice_row.mean_bps
        row[f"{prefix}_median_bps"] = slice_row.median_bps
        row[f"{prefix}_lcb_bps"] = slice_row.lcb_bps
    return row


def _score_key(genome: np.ndarray) -> str:
    return f"genome:{hashlib.sha256(np.asarray(genome, dtype=np.float32).tobytes()).hexdigest()}"


def _save_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    next_generation: int,
    population: np.ndarray,
    rng: np.random.Generator,
    history: list[dict[str, Any]],
    global_champion: Champion,
    complete: bool,
    evaluated_genome_sha256: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            fingerprint=np.asarray([fingerprint]),
            next_generation=np.asarray([next_generation], dtype=np.int32),
            population=np.asarray(population, dtype=np.float32),
            rng_state=np.asarray([json.dumps(rng.bit_generator.state)]),
            history=np.asarray([json.dumps(history)]),
            global_generation=np.asarray([global_champion.generation], dtype=np.int32),
            global_genome=np.asarray(global_champion.genome, dtype=np.float32),
            complete=np.asarray([bool(complete)]),
            evaluated_genome_sha256=np.asarray(
                sorted(set(evaluated_genome_sha256)), dtype="U64"
            ),
        )
    os.replace(temporary, path)


def _load_checkpoint(path: Path, *, fingerprint: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        stored_fingerprint = str(payload["fingerprint"][0])
        if stored_fingerprint != fingerprint:
            raise ValueError("Genetic_Primus checkpoint fingerprint mismatch")
        return {
            "next_generation": int(payload["next_generation"][0]),
            "population": payload["population"].astype(np.float32),
            "rng_state": json.loads(str(payload["rng_state"][0])),
            "history": json.loads(str(payload["history"][0])),
            "global_generation": int(payload["global_generation"][0]),
            "global_genome": payload["global_genome"].astype(np.float32),
            "complete": bool(payload["complete"][0]),
            "evaluated_genome_sha256": (
                tuple(str(value) for value in payload["evaluated_genome_sha256"])
                if "evaluated_genome_sha256" in payload.files
                else ()
            ),
        }


def run_evolution(
    frame: pd.DataFrame,
    *,
    scaler: RobustScaler,
    start_timestamp: int,
    end_timestamp: int,
    round_trip_cost_bps: float,
    fitness_config: Mapping[str, Any],
    evolution_config: Mapping[str, Any],
    warm_start: np.ndarray | None = None,
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
        settings["elite_count"] = min(int(settings["elite_count"]), max(1, int(population_size) // 4))
    if generations is not None:
        settings["generations"] = int(generations)
    rng = np.random.default_rng(int(settings["seed"]))
    history: list[dict[str, Any]] = []
    champions: list[Champion] = []
    global_champion: Champion | None = None
    start_generation = 0
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("resume requested without a Genetic_Primus checkpoint")
        checkpoint = _load_checkpoint(checkpoint_path, fingerprint=run_fingerprint)
        population = checkpoint["population"]
        if len(population) != int(settings["population_size"]):
            raise ValueError("checkpoint population size does not match evolution settings")
        rng.bit_generator.state = checkpoint["rng_state"]
        history = list(checkpoint["history"])
        start_generation = int(checkpoint["next_generation"])
        restored_genome = checkpoint["global_genome"]
        restored_simulation, restored_score = _evaluate_full(
            restored_genome,
            frame=frame,
            scaler=scaler,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            round_trip_cost_bps=round_trip_cost_bps,
            fitness_config=fitness_config,
        )
        global_champion = Champion(
            generation=int(checkpoint["global_generation"]),
            genome=restored_genome,
            simulation=restored_simulation,
            score=restored_score,
        )
    else:
        population = _initial_population(
            rng,
            population_size=int(settings["population_size"]),
            warm_start=warm_start,
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
            raw_fitness_values = np.empty(len(population), dtype=np.float64)
            selection_values = np.empty(len(population), dtype=np.float64)
            if executor is not None:
                evaluated = executor.map(
                    _evaluate_worker,
                    ((index, genome) for index, genome in enumerate(population)),
                    chunksize=1,
                )
                for index, raw_fitness, selection_score in evaluated:
                    raw_fitness_values[index] = raw_fitness
                    selection_values[index] = selection_score
            else:
                for index, genome in enumerate(population):
                    simulation, score = _evaluate_full(
                        genome,
                        frame=frame,
                        scaler=scaler,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                        round_trip_cost_bps=round_trip_cost_bps,
                        fitness_config=fitness_config,
                    )
                    raw_fitness_values[index] = score.fitness
                    selection_values[index] = score.selection_score
            best_index = int(np.argmax(selection_values))
            best_simulation, best_score = _evaluate_full(
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
                simulation=best_simulation,
                score=best_score,
            )
            champions.append(champion)
            if (
                global_champion is None
                or champion.score.selection_score > global_champion.score.selection_score
            ):
                global_champion = champion
            history.append(
                _generation_row(generation, raw_fitness_values, selection_values, champion)
            )
            generation_frame = pd.DataFrame(history)
            next_population = population
            if generation + 1 < int(settings["generations"]):
                next_population = _breed(population, selection_values, rng=rng, settings=settings)
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
        raise RuntimeError("evolution produced no champion")
    return EvolutionResult(
        champion=global_champion,
        generations=pd.DataFrame(history),
        generation_champions=tuple(champions),
    )
