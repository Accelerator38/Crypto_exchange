from __future__ import annotations

from pathlib import Path

from exia.genetic_primus.contracts import load_primus_config
from exia.genetic_primus.fitness_v3 import (
    lexicographic_key_v3_from_values,
    ordinal_selection_scores_v3,
)
from exia.genetic_primus.freeze_v3 import v3_freeze_status


ROOT = Path(__file__).resolve().parents[1]


def key(
    *,
    violation: float = 0.0,
    overall_failures: int = 0,
    temporal_fitness: float = 1.0,
    median: float = 1.0,
    slice_failures: int = 0,
    worst_slice: float = 1.0,
    overall_lcb: float = 1.0,
) -> tuple[float, ...]:
    return lexicographic_key_v3_from_values(
        constraint_violation=violation,
        overall_failure_count=overall_failures,
        temporal_fitness=temporal_fitness,
        overall_median_bps=median,
        slice_failure_count=slice_failures,
        worst_slice_lcb_bps=worst_slice,
        overall_lcb_bps=overall_lcb,
    )


def test_v3_contract_fixes_one_hundred_generations_before_search() -> None:
    config = load_primus_config(ROOT / "configs/genetic_primus_v3.json", root=ROOT)
    assert config.evolution["population_size"] == 64
    assert config.evolution["generations"] == 100
    assert config.payload["safety"] == {
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def test_v3_temporal_pass_precedes_cross_sectional_quality() -> None:
    temporal_pass = key(temporal_fitness=0.1, median=0.1, slice_failures=10, worst_slice=-100.0)
    temporal_fail = key(temporal_fitness=-0.1, median=100.0, slice_failures=0, worst_slice=100.0)
    assert temporal_pass > temporal_fail


def test_v3_temporal_margin_is_optimized_while_gate_is_failed() -> None:
    closer_to_pass = key(temporal_fitness=-1.0, median=10.0, slice_failures=10)
    better_slices = key(temporal_fitness=-2.0, median=10.0, slice_failures=0)
    assert closer_to_pass > better_slices


def test_v3_slice_failures_resume_priority_after_temporal_pass() -> None:
    fewer_slices = key(temporal_fitness=0.1, median=0.1, slice_failures=1, worst_slice=-10.0)
    higher_temporal = key(temporal_fitness=100.0, median=100.0, slice_failures=2, worst_slice=100.0)
    assert fewer_slices > higher_temporal


def test_v3_risk_and_overall_gates_remain_higher_priority() -> None:
    feasible = key(temporal_fitness=-100.0, median=-100.0, slice_failures=10)
    risk_failed = key(violation=0.001, temporal_fitness=100.0, median=100.0)
    overall_pass = key(temporal_fitness=-100.0, median=-100.0)
    overall_failed = key(overall_failures=1, temporal_fitness=100.0, median=100.0)
    assert feasible > risk_failed
    assert overall_pass > overall_failed


def test_v3_ordinal_scores_follow_tuple_order() -> None:
    keys = [
        key(temporal_fitness=-2.0, median=10.0),
        key(temporal_fitness=-1.0, median=10.0),
        key(temporal_fitness=1.0, median=1.0),
    ]
    assert ordinal_selection_scores_v3(keys).tolist() == [0.0, 1.0, 2.0]


def test_v3_freeze_only_allows_validation_without_failures() -> None:
    assert v3_freeze_status(()) == "DEVELOPMENT_V3_PASSED_STRICT_GATES"
    assert v3_freeze_status(("nonpositive_temporal_gate",)) == (
        "DEVELOPMENT_V3_REJECTED_STRICT_GATES"
    )
