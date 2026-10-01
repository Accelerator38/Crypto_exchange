"""Bounded, chronological research selection with an explicit NoTrade fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

from .measurement_lab_v2 import ALLOW_ALL, CATALOG, NO_TRADE, block_bank, paired_lcb
from .prospective_evaluator_v1 import OVERLAY


def fixed_candidates(budget: int, seed: int) -> tuple[tuple[int, ...], ...]:
    """One registered draw; all sampled genomes count against the budget."""
    if type(budget) is not int or not 1 <= budget <= len(CATALOG) - 2:
        raise ValueError("invalid fixed candidate budget")
    pool = [genome for genome in CATALOG if genome not in (NO_TRADE, ALLOW_ALL, OVERLAY)]
    count = budget - 1
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=count, replace=False)
    return (OVERLAY, *(pool[int(index)] for index in indices))


def common_bank(days: int, config: dict[str, Any], *, validation: bool) -> np.ndarray:
    seed = int(config["bootstrap_seed"]) + (100_000 if validation else 0)
    return block_bank(days, int(config["bootstrap_samples"]),
                      int(config["bootstrap_block_days"]), seed)


def absolute_score(normal: np.ndarray, stress: np.ndarray,
                   bank: np.ndarray) -> dict[str, float]:
    """Absolute paired daily LCB to the zero-return incumbent."""
    normal = np.asarray(normal, dtype=float)
    stress = np.asarray(stress, dtype=float)
    if normal.shape != stress.shape or normal.ndim != 1 or not len(normal):
        raise ValueError("aligned daily returns required")
    zero = np.zeros_like(normal)
    normal_lcb = paired_lcb(normal, zero, bank)
    stress_lcb = paired_lcb(stress, zero, bank)
    return {"normal_lcb_vs_no_trade": normal_lcb,
            "stress_lcb_vs_no_trade": stress_lcb,
            "rank_score": min(normal_lcb, stress_lcb)}


def choose_train_winner(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Inspect the fixed budget once; no replacement chosen on validation."""
    if not records or len({tuple(row["genome"]) for row in records}) != len(records):
        raise ValueError("unique nonempty candidate records required")
    winner = max(records, key=lambda row: (
        float(row["training"]["rank_score"]),
        -sum(weight != 0 for weight in row["genome"][:-1]),
        tuple(-weight for weight in row["genome"])))
    return winner if winner["training"]["rank_score"] > 0.0 else None


def validation_gate(
    normal: np.ndarray, stress: np.ndarray,
    baseline_normal: np.ndarray, baseline_stress: np.ndarray,
    bank: np.ndarray, *, max_drawdown_normal: float,
    max_drawdown_stress: float, drawdown_limit: float,
) -> dict[str, Any]:
    """Absolute survival first; paired incremental evidence is additional."""
    normal = np.asarray(normal, dtype=float)
    stress = np.asarray(stress, dtype=float)
    baseline_normal = np.asarray(baseline_normal, dtype=float)
    baseline_stress = np.asarray(baseline_stress, dtype=float)
    absolute = absolute_score(normal, stress, bank)
    paired_normal = paired_lcb(normal, baseline_normal, bank)
    paired_stress = paired_lcb(stress, baseline_stress, bank)
    normal_net = float(np.prod(1.0 + normal) - 1.0)
    stress_net = float(np.prod(1.0 + stress) - 1.0)
    checks = {
        "normal_net_positive": normal_net > 0.0,
        "stress_net_positive": stress_net > 0.0,
        "normal_lcb_vs_NoTrade_positive": absolute["normal_lcb_vs_no_trade"] > 0.0,
        "stress_lcb_vs_NoTrade_positive": absolute["stress_lcb_vs_no_trade"] > 0.0,
        "normal_paired_lcb_vs_EMA_long_only_positive": paired_normal > 0.0,
        "stress_paired_lcb_vs_EMA_long_only_positive": paired_stress > 0.0,
        "drawdown_below_absolute_limit": (
            max_drawdown_normal <= drawdown_limit
            and max_drawdown_stress <= drawdown_limit),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "normal_net": normal_net, "stress_net": stress_net,
            **absolute,
            "normal_paired_lcb_vs_ema_long_only": paired_normal,
            "stress_paired_lcb_vs_ema_long_only": paired_stress}
