from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import FEATURE_NAMES
from .simulator import PrimusSimulation


RETURN_1_INDEX = FEATURE_NAMES.index("return_1")
RELATIVE_MOMENTUM_RANK_INDEX = FEATURE_NAMES.index("relative_momentum_rank")
RETURN_12_INDEX = FEATURE_NAMES.index("return_12")
EMA_GAP_INDEX = FEATURE_NAMES.index("ema_gap_12_48")
EMA_SLOPE_INDEX = FEATURE_NAMES.index("ema_slope_12_3")
EFFICIENCY_INDEX = FEATURE_NAMES.index("efficiency_12")
DONCHIAN_POSITION_INDEX = FEATURE_NAMES.index("donchian_position_20")
MARKET_RETURN_12_INDEX = FEATURE_NAMES.index("market_return_12")
MARKET_BREADTH_12_INDEX = FEATURE_NAMES.index("market_breadth_12")
RESIDUAL_RETURN_3_INDEX = FEATURE_NAMES.index("residual_return_3")


@dataclass(frozen=True)
class NoTradePolicy:
    policy_id: str = "NoTrade"

    def target(self, features: np.ndarray) -> int:
        return 0


@dataclass(frozen=True)
class SignedReturnPolicy:
    multiplier: int
    policy_id: str
    deadband: float = 0.0

    def __post_init__(self) -> None:
        if self.multiplier not in {-1, 1}:
            raise ValueError("signed-return multiplier must be -1 or 1")
        if self.deadband < 0.0:
            raise ValueError("signed-return deadband cannot be negative")

    def target(self, features: np.ndarray) -> int:
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(values) != len(FEATURE_NAMES):
            raise ValueError("benchmark feature vector does not match the V4 contract")
        signal = float(values[RETURN_1_INDEX])
        if not np.isfinite(signal) or abs(signal) <= self.deadband:
            return 0
        return int(self.multiplier * np.sign(signal))


@dataclass(frozen=True)
class SignedFeaturePolicy:
    feature_index: int
    policy_id: str
    deadband: float

    def target(self, features: np.ndarray) -> int:
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(values) != len(FEATURE_NAMES):
            raise ValueError("benchmark feature vector does not match the V4 contract")
        signal = float(values[self.feature_index])
        if not np.isfinite(signal) or abs(signal) <= self.deadband:
            return 0
        return int(np.sign(signal))


@dataclass(frozen=True)
class V7RegisteredProposalPolicy:
    policy_id: str
    rule: str

    def target(self, features: np.ndarray) -> int:
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if len(values) != len(FEATURE_NAMES):
            raise ValueError("benchmark feature vector does not match the V4 contract")

        if self.rule == "ema_trend_consensus":
            required = values[
                [EMA_GAP_INDEX, EMA_SLOPE_INDEX, RETURN_12_INDEX, EFFICIENCY_INDEX]
            ]
            if not np.isfinite(required).all() or required[3] < 0.30:
                return 0
            direction = int(np.sign(required[0]))
            if direction == 0 or np.sign(required[1]) != direction or np.sign(required[2]) != direction:
                return 0
            return direction

        if self.rule == "residual_rank_consensus":
            required = values[
                [RELATIVE_MOMENTUM_RANK_INDEX, RESIDUAL_RETURN_3_INDEX, EFFICIENCY_INDEX]
            ]
            if not np.isfinite(required).all() or abs(required[0]) <= 0.50 or required[2] < 0.20:
                return 0
            direction = int(np.sign(required[0]))
            return direction if np.sign(required[1]) == direction else 0

        if self.rule == "donchian_breakout":
            required = values[[DONCHIAN_POSITION_INDEX, EMA_SLOPE_INDEX]]
            if not np.isfinite(required).all() or abs(required[0]) <= 1.0:
                return 0
            direction = int(np.sign(required[0]))
            return direction if np.sign(required[1]) == direction else 0

        if self.rule == "market_aligned_trend":
            required = values[
                [RETURN_12_INDEX, MARKET_RETURN_12_INDEX, MARKET_BREADTH_12_INDEX, EFFICIENCY_INDEX]
            ]
            if (
                not np.isfinite(required).all()
                or abs(required[2]) < 0.25
                or required[3] < 0.30
            ):
                return 0
            direction = int(np.sign(required[0]))
            if direction == 0 or np.sign(required[1]) != direction or np.sign(required[2]) != direction:
                return 0
            return direction

        raise ValueError(f"unknown registered V7 proposal rule: {self.rule}")


def momentum_policy() -> SignedReturnPolicy:
    return SignedReturnPolicy(multiplier=1, policy_id="Momentum_return_1")


def mean_reversion_policy() -> SignedReturnPolicy:
    return SignedReturnPolicy(multiplier=-1, policy_id="MeanReversion_return_1")


def relative_momentum_policy() -> SignedFeaturePolicy:
    return SignedFeaturePolicy(
        feature_index=RELATIVE_MOMENTUM_RANK_INDEX,
        policy_id="RelativeMomentumRank_return_12",
        deadband=0.5,
    )


def v7_registered_proposal_policies() -> tuple[V7RegisteredProposalPolicy, ...]:
    return (
        V7RegisteredProposalPolicy("V7_EMA_TrendConsensus", "ema_trend_consensus"),
        V7RegisteredProposalPolicy("V7_ResidualRankConsensus", "residual_rank_consensus"),
        V7RegisteredProposalPolicy("V7_DonchianBreakout", "donchian_breakout"),
        V7RegisteredProposalPolicy("V7_MarketAlignedTrend", "market_aligned_trend"),
    )


def summarize_benchmark(simulation: PrimusSimulation) -> dict[str, float | int]:
    if simulation.portfolio.empty:
        net_return = 0.0
        max_drawdown = 0.0
    else:
        net_return = float(np.prod(1.0 + simulation.portfolio["net_return"].to_numpy(float)) - 1.0)
        max_drawdown = float(-simulation.portfolio["drawdown"].min())
    ledger = simulation.ledger
    max_share = lambda column: (
        float(ledger[column].value_counts(normalize=True).max()) if not ledger.empty else 0.0
    )
    return {
        "net_return": net_return,
        "net_bps": net_return * 10_000.0,
        "max_drawdown": max_drawdown,
        "trades": int(len(simulation.ledger)),
        "turnover_fills": int(simulation.turnover_fills),
        "actions_short": int(simulation.action_counts["SHORT"]),
        "actions_flat": int(simulation.action_counts["FLAT"]),
        "actions_long": int(simulation.action_counts["LONG"]),
        "max_direction_share": max_share("direction"),
        "max_state_share": max_share("market_state"),
        "max_symbol_share": max_share("symbol"),
    }
