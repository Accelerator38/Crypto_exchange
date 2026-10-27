from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.fitness_recency import anchored_recency_weights
from exia.genetic_primus.policy_v4 import V4NetworkShape, functional_policy_drift_v4
from exia.genetic_primus.promotion_v4 import evaluate_outer_promotion_gate
from exia.genetic_primus.rolling import (
    build_functional_drift_anchor,
    load_rolling_protocol,
    summarize_outer_deployment_evidence,
)
from exia.genetic_primus.rolling_v4 import (
    validation_candidate_status_v4,
    validation_finalist_eligibility_v4,
)


ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000


def test_v4_stabilization_contract_is_tighter_without_more_hypotheses() -> None:
    v3 = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v3.json")
    v4 = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v4.json")
    assert v4.payload["adaptive_multiple_testing"] == v3.payload["adaptive_multiple_testing"]
    assert v4.payload["genetic_training"]["maximum_active_features"] == 6
    assert v4.payload["policy_overlay"]["maximum_allowed_fraction"] == 0.25
    assert v4.payload["policy_overlay"]["minimum_baseline_move_bps"] == 24.0
    assert v4.payload["recency_fitness"]["half_life_days"] == [90]
    assert v4.payload["recency_fitness"]["uniform_anchor"] == 0.2
    assert v4.payload["recency_fitness"]["turnover_penalty_bps_per_fill_per_day"] == 2.0
    assert not any(v4.payload["safety"].values())


def test_v4_anchored_recency_weights_are_normalized_and_recent_heavier() -> None:
    timestamps = np.arange(180, dtype=np.int64) * DAY_MS
    weights = anchored_recency_weights(
        timestamps,
        cutoff_ms=int(timestamps[-1]),
        half_life_days=90,
        uniform_anchor=0.2,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)
    assert weights[-1] > weights[0]
    assert 1.0 / np.square(weights).sum() >= 120.0


def test_v4_no_eligible_challenger_is_explicit_and_fail_closed() -> None:
    eligible, blockers = validation_finalist_eligibility_v4(
        {
            "coverage_constraint_passed": False,
            "paired_incremental_min_lcb_bps": -0.1,
        },
        ("turnover", "nonpositive_paired_lcb:NoTrade"),
    )
    assert eligible is False
    assert "turnover" in blockers
    assert "coverage_constraint" in blockers
    assert "paired_incremental_lcb" in blockers
    assert validation_candidate_status_v4(0) == "NO_ELIGIBLE_CHALLENGER"
    assert validation_candidate_status_v4(1) == "DIAGNOSTIC_CHALLENGER"
    clean, clean_blockers = validation_finalist_eligibility_v4(
        {
            "coverage_constraint_passed": True,
            "paired_incremental_min_lcb_bps": 0.1,
        },
        (),
    )
    assert clean is True
    assert clean_blockers == ()


def test_v4_functional_drift_uses_same_hash_anchored_rows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": np.arange(20, dtype=np.int64) * DAY_MS,
            "symbol": [f"S{index:02d}" for index in range(20)],
        }
    )
    anchor, anchor_sha = build_functional_drift_anchor(
        frame,
        start_timestamp=0,
        cutoff_timestamp=19 * DAY_MS,
        maximum_rows=8,
    )
    reversed_anchor, reversed_sha = build_functional_drift_anchor(
        frame.iloc[::-1].reset_index(drop=True),
        start_timestamp=0,
        cutoff_timestamp=19 * DAY_MS,
        maximum_rows=8,
    )
    assert anchor_sha == reversed_sha
    assert anchor[["timestamp", "symbol"]].equals(
        reversed_anchor[["timestamp", "symbol"]]
    )

    shape = V4NetworkShape(inputs=2, hidden=(), outputs=1)
    positive = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    negative = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    features = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    same = functional_policy_drift_v4(
        (positive,),
        (positive,),
        shape=shape,
        challenger_features=features,
        incumbent_features=features,
    )
    opposite = functional_policy_drift_v4(
        (positive,),
        (negative,),
        shape=shape,
        challenger_features=features,
        incumbent_features=features,
    )
    assert same == {"functional_logit_drift": 0.0, "functional_action_drift": 0.0}
    assert opposite["functional_logit_drift"] > 0.9
    assert opposite["functional_action_drift"] == 1.0


def test_v4_diagnostic_challenger_is_excluded_from_outer_evidence() -> None:
    protocol = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v4.json")
    benchmarks = pd.DataFrame(
        [
            {
                "origin_index": 0,
                "agent_id": "GeneticPrimusChallenger",
                "net_bps": 999.0,
                "evidence_role": "diagnostic_outer_deployment",
            },
            {
                "origin_index": 0,
                "agent_id": "GeneticPrimusActive",
                "net_bps": 0.0,
                "evidence_role": "outer_deployment",
            },
        ]
    )
    evidence = summarize_outer_deployment_evidence(
        benchmarks,
        minimum_completed_origins=60,
        minimum_positive_fraction=0.65,
    )
    assert evidence["completed_outer_origins"] == 0
    assert evidence["no_eligible_challenger_evidence"] is True
    origins = pd.DataFrame(
        [{"origin_index": 0, "challenger_deployable": False}]
    )
    decision = evaluate_outer_promotion_gate(
        benchmarks=benchmarks,
        origins=origins,
        config=protocol.payload["promotion_evidence"],
    )
    assert decision.passed is False
    assert decision.metrics["no_eligible_challenger_evidence"] is True
    assert "no_eligible_challenger_evidence" in decision.failed_gates
