from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.policy_v4 import (
    V4NetworkShape,
    select_action_diverse_genomes_v4,
)
from exia.genetic_primus.promotion_v4 import evaluate_outer_promotion_gate
from exia.genetic_primus.rolling import build_rolling_origins, load_rolling_protocol
from exia.genetic_primus.temporal_validation_v4 import build_temporal_validation_split


ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000


def test_v3_temporal_layers_are_disjoint_and_purged() -> None:
    protocol = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v3.json")
    origin = build_rolling_origins(protocol)[0]
    split = build_temporal_validation_split(
        train_start=origin.train_start,
        train_end=origin.train_end,
        timeframe_ms=protocol.timeframe_ms,
        config=protocol.payload["temporal_validation"],
    )
    assert (origin.train_end - origin.train_start) // DAY_MS == 730
    assert (split.validation_end - split.validation_start) // DAY_MS == 60
    assert (split.audit_end - split.audit_start) // DAY_MS == 60
    assert split.validation_start - split.inner_end == 7 * protocol.timeframe_ms
    assert split.audit_start - split.validation_end == 7 * protocol.timeframe_ms


def test_action_diversity_selects_without_creating_new_genomes() -> None:
    shape = V4NetworkShape(30, (), 1, maximum_active_features=8)
    genomes = []
    for weight, bias in ((2.0, 0.0), (-2.0, 0.0), (0.5, 0.2), (0.1, 1.0), (0.1, -1.0)):
        genome = np.zeros(shape.genome_size, dtype=np.float32)
        genome[0] = weight
        genome[-1] = bias
        genomes.append(genome)
    anchor = np.zeros((64, 30), dtype=np.float32)
    anchor[:, 0] = np.linspace(-1.0, 1.0, len(anchor))
    selected, metrics = select_action_diverse_genomes_v4(
        tuple(genomes),
        shape=shape,
        anchor_features=anchor,
        count=3,
        minimum_disagreement=0.10,
    )
    source_sha = {genome.tobytes() for genome in genomes}
    assert len(selected) == 3
    assert all(genome.tobytes() in source_sha for genome in selected)
    assert metrics["diversity_gate_passed"] is True


def _outer_frames(*, origins_count: int = 60, stress_delta: float = 1.0):
    rows = []
    for origin in range(origins_count):
        for agent, net_bps, drawdown in (
            ("GeneticPrimusChallenger", 2.0, 0.01),
            ("GeneticPrimusActive", 0.0, 0.02),
            ("GeneticPrimusChallengerStress", stress_delta, 0.01),
            ("GeneticPrimusActiveStress", 0.0, 0.02),
        ):
            rows.append(
                {
                    "origin_index": origin,
                    "agent_id": agent,
                    "net_bps": net_bps,
                    "max_drawdown": drawdown,
                    "turnover_fills": 6,
                    "max_direction_share": 0.5,
                    "max_state_share": 0.5,
                    "max_symbol_share": 0.25,
                    "evidence_role": "outer_deployment",
                }
            )
    origins = pd.DataFrame(
        {
            "origin_index": np.arange(origins_count),
            "predeployment_training_gate_passed": True,
            "challenger_retrain_action_drift": 0.10,
            "challenger_retrain_model_drift": 0.20,
        }
    )
    return pd.DataFrame(rows), origins


def test_outer_gate_can_pass_but_never_grants_runtime_authority() -> None:
    protocol = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v3.json")
    benchmarks, origins = _outer_frames()
    decision = evaluate_outer_promotion_gate(
        benchmarks=benchmarks,
        origins=origins,
        config=protocol.payload["promotion_evidence"],
    )
    assert decision.passed is True
    assert decision.metrics["promotion_gate_passed"] is True
    assert decision.metrics["promotion_approved"] is False
    assert decision.metrics["promotion_authority"] is False


def test_outer_gate_fails_closed_on_insufficient_origins_and_stress() -> None:
    protocol = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v3.json")
    benchmarks, origins = _outer_frames(origins_count=59, stress_delta=-1.0)
    decision = evaluate_outer_promotion_gate(
        benchmarks=benchmarks,
        origins=origins,
        config=protocol.payload["promotion_evidence"],
    )
    assert decision.passed is False
    assert "minimum_completed_outer_origins" in decision.failed_gates
    assert "stress_cost_lcb" in decision.failed_gates
