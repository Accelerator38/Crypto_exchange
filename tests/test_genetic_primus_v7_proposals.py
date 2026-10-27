import json
from pathlib import Path

import numpy as np

from exia.genetic_primus.benchmarks_v4 import (
    DONCHIAN_POSITION_INDEX,
    EFFICIENCY_INDEX,
    EMA_GAP_INDEX,
    EMA_SLOPE_INDEX,
    MARKET_BREADTH_12_INDEX,
    MARKET_RETURN_12_INDEX,
    RELATIVE_MOMENTUM_RANK_INDEX,
    RESIDUAL_RETURN_3_INDEX,
    RETURN_12_INDEX,
    v7_registered_proposal_policies,
)
from exia.genetic_primus.contracts import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def _features() -> np.ndarray:
    return np.zeros(len(FEATURE_NAMES), dtype=np.float32)


def _policies() -> dict[str, object]:
    return {policy.policy_id: policy for policy in v7_registered_proposal_policies()}


def test_v7_catalog_and_policy_factory_match() -> None:
    catalog = json.loads(
        (ROOT / "configs/genetic_primus_v7_proposal_matrix.json").read_text(encoding="utf-8")
    )
    expected = [(item["policy_id"], item["rule"]) for item in catalog["proposals"]]
    actual = [(policy.policy_id, policy.rule) for policy in v7_registered_proposal_policies()]
    assert actual == expected
    assert catalog["testing_budget"]["fixed_proposals"] == 4
    assert not any(catalog["safety"].values())


def test_v7_ema_trend_requires_full_sign_consensus() -> None:
    policy = _policies()["V7_EMA_TrendConsensus"]
    values = _features()
    values[EMA_GAP_INDEX] = 0.01
    values[EMA_SLOPE_INDEX] = 0.005
    values[RETURN_12_INDEX] = 0.02
    values[EFFICIENCY_INDEX] = 0.30
    assert policy.target(values) == 1
    values[EMA_SLOPE_INDEX] = -0.005
    assert policy.target(values) == 0


def test_v7_residual_rank_uses_strict_quartile_entry() -> None:
    policy = _policies()["V7_ResidualRankConsensus"]
    values = _features()
    values[RELATIVE_MOMENTUM_RANK_INDEX] = -0.75
    values[RESIDUAL_RETURN_3_INDEX] = -0.01
    values[EFFICIENCY_INDEX] = 0.20
    assert policy.target(values) == -1
    values[RELATIVE_MOMENTUM_RANK_INDEX] = -0.50
    assert policy.target(values) == 0


def test_v7_donchian_requires_true_breakout_and_slope_confirmation() -> None:
    policy = _policies()["V7_DonchianBreakout"]
    values = _features()
    values[DONCHIAN_POSITION_INDEX] = 1.01
    values[EMA_SLOPE_INDEX] = 0.001
    assert policy.target(values) == 1
    values[DONCHIAN_POSITION_INDEX] = 1.0
    assert policy.target(values) == 0


def test_v7_market_aligned_trend_accepts_symmetric_short_signal() -> None:
    policy = _policies()["V7_MarketAlignedTrend"]
    values = _features()
    values[RETURN_12_INDEX] = -0.02
    values[MARKET_RETURN_12_INDEX] = -0.01
    values[MARKET_BREADTH_12_INDEX] = -0.25
    values[EFFICIENCY_INDEX] = 0.30
    assert policy.target(values) == -1
    values[MARKET_BREADTH_12_INDEX] = 0.25
    assert policy.target(values) == 0
