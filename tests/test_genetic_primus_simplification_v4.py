from __future__ import annotations

import numpy as np

from exia.genetic_primus.benchmarks_v4 import (
    NoTradePolicy,
    mean_reversion_policy,
    momentum_policy,
)
from exia.genetic_primus.contracts import FEATURE_NAMES
from exia.genetic_primus.evolution_v4 import _initial_population
from exia.genetic_primus.policy_v4 import (
    EnsemblePolicyV4,
    V4NetworkShape,
    ensemble_sparsity_v4,
    genome_sparsity_v4,
)


def test_only_small_architectures_have_expected_parameter_counts() -> None:
    assert V4NetworkShape(inputs=30, hidden=()).genome_size == 93
    assert V4NetworkShape(inputs=30, hidden=(8,)).genome_size == 275


def test_linear_sparsity_counts_effective_input_features() -> None:
    shape = V4NetworkShape(inputs=30, hidden=())
    genome = np.zeros(shape.genome_size, dtype=np.float32)
    genome[3 * 4 + 1] = 0.1
    metrics = genome_sparsity_v4(genome, shape=shape, active_weight_threshold=0.05)
    assert metrics["active_parameter_count"] == 1
    assert metrics["active_feature_count"] == 1
    assert np.isclose(metrics["feature_sparsity"], 29 / 30)


def test_hidden_feature_requires_an_active_path_to_output() -> None:
    shape = V4NetworkShape(inputs=30, hidden=(8,))
    genome = np.zeros(shape.genome_size, dtype=np.float32)
    genome[2 * 8 + 3] = 0.1
    metrics = genome_sparsity_v4(genome, shape=shape, active_weight_threshold=0.05)
    assert metrics["active_feature_count"] == 0
    output_weight_start = 30 * 8 + 8
    genome[output_weight_start + 3 * 3 + 1] = 0.1
    metrics = genome_sparsity_v4(genome, shape=shape, active_weight_threshold=0.05)
    assert metrics["active_parameter_count"] == 2
    assert metrics["active_feature_count"] == 1


def test_fixed_baselines_are_deterministic_and_opposed() -> None:
    positive = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    positive[0] = 1.0
    negative = -positive
    assert NoTradePolicy().target(positive) == 0
    assert momentum_policy().target(positive) == 1
    assert momentum_policy().target(negative) == -1
    assert mean_reversion_policy().target(positive) == -1
    assert mean_reversion_policy().target(negative) == 1


def test_retrain_population_has_exactly_25_percent_random_immigrants() -> None:
    shape = V4NetworkShape(inputs=30, hidden=())
    warm = np.zeros(shape.genome_size, dtype=np.float32)
    population = _initial_population(
        np.random.default_rng(9),
        shape=shape,
        population_size=4,
        warm_start=warm,
        mutation_sigma=0.0,
        random_immigrant_fraction=0.25,
    )
    assert population.shape == (4, shape.genome_size)
    assert int(np.all(population == 0.0, axis=1).sum()) == 3


def test_retrain_mix_keeps_supervised_incumbent_mutation_and_random_paths() -> None:
    shape = V4NetworkShape(inputs=30, hidden=())
    supervised = np.zeros(shape.genome_size, dtype=np.float32)
    incumbent = np.ones(shape.genome_size, dtype=np.float32)
    population = _initial_population(
        np.random.default_rng(19),
        shape=shape,
        population_size=4,
        warm_start=supervised,
        additional_warm_starts=(incumbent,),
        mutation_sigma=0.01,
        random_immigrant_fraction=0.25,
    )
    assert int(np.all(population == 0.0, axis=1).sum()) == 1
    assert int(np.all(population == 1.0, axis=1).sum()) == 1
    assert int((~np.all(population == 0.0, axis=1) & ~np.all(population == 1.0, axis=1)).sum()) == 2


def test_top_three_ensemble_averages_logits_and_counts_all_parameters() -> None:
    shape = V4NetworkShape(inputs=30, hidden=())
    long = np.zeros(shape.genome_size, dtype=np.float32)
    short = np.zeros(shape.genome_size, dtype=np.float32)
    long[-1] = 2.0
    short[-3] = 2.0
    genomes = (long, long.copy(), short)
    ensemble = EnsemblePolicyV4(genomes, shape=shape)
    assert ensemble.target(np.zeros(30, dtype=np.float32)) == 1
    metrics = ensemble_sparsity_v4(
        genomes, shape=shape, active_weight_threshold=0.05
    )
    assert metrics["parameter_count"] == 3 * shape.genome_size
    assert metrics["active_parameter_count"] == 3
    assert metrics["ensemble_size"] == 3
