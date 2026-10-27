from __future__ import annotations

from pathlib import Path

from exia.genetic_primus.contracts import load_primus_config
from exia.genetic_primus.fitness_v2 import (
    lexicographic_key_from_values,
    ordinal_selection_scores,
)
from exia.genetic_primus.freeze_v2 import v2_freeze_status


ROOT = Path(__file__).resolve().parents[1]


def ranking_key(
    *,
    violation: float = 0.0,
    overall_failures: int = 0,
    slice_failures: int = 0,
    worst_lcb: float = 1.0,
    fitness: float = 1.0,
    overall_lcb: float = 1.0,
) -> tuple[float, ...]:
    return lexicographic_key_from_values(
        constraint_violation=violation,
        overall_failure_count=overall_failures,
        slice_failure_count=slice_failures,
        worst_slice_lcb_bps=worst_lcb,
        fitness=fitness,
        overall_lcb_bps=overall_lcb,
    )


def test_v2_keeps_the_v1_physical_and_safety_contract() -> None:
    config = load_primus_config(ROOT / "configs/genetic_primus_v2.json", root=ROOT)
    assert config.dataset["timeframe_minutes"] == 240
    assert config.network == {
        "inputs": 30,
        "hidden": [32, 16],
        "outputs": 3,
        "activation": "tanh",
    }
    assert config.payload["safety"] == {
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def test_v2_feasible_candidate_beats_infeasible_candidate() -> None:
    feasible = ranking_key(slice_failures=10, worst_lcb=-100.0, fitness=-100.0)
    infeasible = ranking_key(
        violation=0.001,
        slice_failures=0,
        worst_lcb=100.0,
        fitness=100.0,
    )
    assert feasible > infeasible


def test_v2_positive_overall_gate_precedes_slice_optimization() -> None:
    positive_overall = ranking_key(slice_failures=10, worst_lcb=-100.0)
    failed_overall = ranking_key(
        overall_failures=1,
        slice_failures=0,
        worst_lcb=100.0,
    )
    assert positive_overall > failed_overall


def test_v2_fewer_failed_slices_precede_fitness_and_worst_lcb() -> None:
    fewer_failures = ranking_key(slice_failures=2, worst_lcb=-100.0, fitness=-100.0)
    higher_fitness = ranking_key(slice_failures=3, worst_lcb=100.0, fitness=100.0)
    assert fewer_failures > higher_fitness


def test_v2_worst_slice_lcb_precedes_monthly_fitness() -> None:
    stronger_worst_slice = ranking_key(slice_failures=2, worst_lcb=-2.0, fitness=-100.0)
    higher_fitness = ranking_key(slice_failures=2, worst_lcb=-3.0, fitness=100.0)
    assert stronger_worst_slice > higher_fitness


def test_v2_ordinal_scores_preserve_lexicographic_order() -> None:
    keys = [
        ranking_key(slice_failures=4),
        ranking_key(slice_failures=2, worst_lcb=-5.0),
        ranking_key(slice_failures=2, worst_lcb=1.0),
    ]
    ranks = ordinal_selection_scores(keys)
    assert ranks.tolist() == [0.0, 1.0, 2.0]


def test_v2_temporal_rejection_cannot_open_validation() -> None:
    assert v2_freeze_status(("nonpositive_temporal_fitness",)) == (
        "DEVELOPMENT_V2_REJECTED_TEMPORAL_ROBUSTNESS"
    )
    assert v2_freeze_status(()) == "DEVELOPMENT_V2_PASSED_STRICT_GATES"
