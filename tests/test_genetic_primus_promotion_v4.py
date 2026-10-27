from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.measurement import BootstrapBank
from exia.genetic_primus.promotion_v4 import evaluate_promotion_gate, load_promotion_protocol


ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000


def _bank(timestamps: np.ndarray, *, seed: int, indices: np.ndarray, identity: str) -> BootstrapBank:
    return BootstrapBank(
        bank_id=identity,
        seed=seed,
        samples=len(indices),
        block_days=1,
        cutoff_ms=int(timestamps[-1] + DAY_MS),
        folds_sha256="unit",
        scope_timestamps={"overall": timestamps},
        scope_indices={"overall": indices.astype(np.int64)},
        content_sha256=identity,
    )


def _metrics(drawdown: float) -> dict[str, float]:
    return {
        "max_drawdown": drawdown,
        "fills_per_day": 1.0,
        "max_direction_share": 0.5,
        "max_state_share": 0.5,
        "max_symbol_share": 0.5,
    }


def _slices(days: int) -> list[dict[str, str]]:
    return [
        {
            "start": pd.Timestamp(start * DAY_MS, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp(min(days, start + 10) * DAY_MS, unit="ms", tz="UTC").isoformat(),
        }
        for start in range(0, days, 10)
    ]


def test_promotion_passes_only_when_every_gate_is_positive() -> None:
    days = 70
    timestamps = np.arange(days, dtype=np.int64) * DAY_MS
    paired = pd.DataFrame(
        {"timestamp": timestamps, "challenger_net_bps": 2.0, "incumbent_net_bps": 1.0}
    )
    stress = paired.assign(challenger_net_bps=1.5)
    identity = np.tile(np.arange(days), (20, 1))
    decision = evaluate_promotion_gate(
        paired=paired,
        stress_paired=stress,
        outer_slices=_slices(days),
        challenger_metrics=_metrics(0.10),
        incumbent_metrics=_metrics(0.12),
        challenger_actions={"SHORT": 30, "FLAT": 40, "LONG": 30},
        incumbent_actions={"SHORT": 30, "FLAT": 40, "LONG": 30},
        challenger_centroid=np.ones(10) * 0.11,
        incumbent_centroid=np.ones(10) * 0.10,
        selection_bank=_bank(timestamps, seed=1, indices=identity, identity="selection"),
        audit_bank=_bank(timestamps, seed=2, indices=identity[::-1], identity="audit"),
        protocol=load_promotion_protocol(ROOT / "configs/genetic_primus_promotion_gate_v1.json"),
    )
    assert decision.passed
    assert decision.metrics["positive_outer_slice_fraction"] == 1.0


def test_independent_audit_can_veto_selection_success() -> None:
    days = 20
    timestamps = np.arange(days, dtype=np.int64) * DAY_MS
    difference = np.r_[np.full(10, 2.0), np.full(10, -2.0)]
    paired = pd.DataFrame(
        {
            "timestamp": timestamps,
            "challenger_net_bps": difference,
            "incumbent_net_bps": np.zeros(days),
        }
    )
    positive_indices = np.tile(np.arange(10), (20, 2))
    negative_indices = np.tile(np.arange(10, 20), (20, 2))
    decision = evaluate_promotion_gate(
        paired=paired,
        stress_paired=paired.assign(challenger_net_bps=1.0),
        outer_slices=_slices(days),
        challenger_metrics=_metrics(0.10),
        incumbent_metrics=_metrics(0.12),
        challenger_actions={"SHORT": 30, "FLAT": 40, "LONG": 30},
        incumbent_actions={"SHORT": 30, "FLAT": 40, "LONG": 30},
        challenger_centroid=np.ones(10) * 0.11,
        incumbent_centroid=np.ones(10) * 0.10,
        selection_bank=_bank(timestamps, seed=1, indices=positive_indices, identity="selection"),
        audit_bank=_bank(timestamps, seed=2, indices=negative_indices, identity="audit"),
        protocol=load_promotion_protocol(ROOT / "configs/genetic_primus_promotion_gate_v1.json"),
    )
    assert not decision.passed
    assert decision.metrics["paired_selection_lcb_bps"] > 0.0
    assert decision.metrics["paired_audit_lcb_bps"] < 0.0
    assert "paired_audit_lcb" in decision.failed_gates


def test_risk_concentration_and_drift_gates_all_fail_closed() -> None:
    days = 20
    timestamps = np.arange(days, dtype=np.int64) * DAY_MS
    paired = pd.DataFrame(
        {"timestamp": timestamps, "challenger_net_bps": 2.0, "incumbent_net_bps": 1.0}
    )
    indices = np.tile(np.arange(days), (20, 1))
    risky = {
        "max_drawdown": 0.20,
        "fills_per_day": 20.0,
        "max_direction_share": 0.95,
        "max_state_share": 0.90,
        "max_symbol_share": 0.80,
    }
    decision = evaluate_promotion_gate(
        paired=paired,
        stress_paired=paired.assign(challenger_net_bps=0.0),
        outer_slices=_slices(days),
        challenger_metrics=risky,
        incumbent_metrics=_metrics(0.10),
        challenger_actions={"SHORT": 0, "FLAT": 0, "LONG": 100},
        incumbent_actions={"SHORT": 0, "FLAT": 100, "LONG": 0},
        challenger_centroid=np.ones(10),
        incumbent_centroid=np.zeros(10),
        selection_bank=_bank(timestamps, seed=1, indices=indices, identity="selection"),
        audit_bank=_bank(timestamps, seed=2, indices=indices, identity="audit"),
        protocol=load_promotion_protocol(ROOT / "configs/genetic_primus_promotion_gate_v1.json"),
    )
    assert not decision.passed
    assert {
        "stress_cost_lcb",
        "drawdown_vs_incumbent",
        "turnover",
        "direction_concentration",
        "state_concentration",
        "symbol_concentration",
        "action_drift",
        "model_drift",
    } <= set(decision.failed_gates)
