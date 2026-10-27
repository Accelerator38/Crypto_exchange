from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.contracts_v4 import load_v4_config
from exia.genetic_primus.fitness_v4 import contiguous_block_lcb, daily_portfolio
from exia.genetic_primus.policy_v4 import GeneticPrimusPolicyV4, V4NetworkShape


ROOT = Path(__file__).resolve().parents[1]


def test_v4_contract_seals_folds_architectures_and_authority() -> None:
    config = load_v4_config(ROOT / "configs/genetic_primus_v4.json", root=ROOT)
    assert len(config.folds) == 6
    assert [item["hidden"] for item in config.architectures] == [[], [8]]
    assert set(config.payload["safety"].values()) == {False}


def test_v4_policy_supports_linear_and_compact_shapes() -> None:
    rng = np.random.default_rng(4)
    for hidden in ((), (8,)):
        shape = V4NetworkShape(inputs=30, hidden=hidden)
        policy = GeneticPrimusPolicyV4.random(rng, shape=shape)
        assert len(policy.genome) == shape.genome_size
        assert policy.target(np.zeros(30, dtype=np.float32)) in {-1, 0, 1}


def test_daily_portfolio_compounds_intraday_returns() -> None:
    portfolio = pd.DataFrame(
        {
            "timestamp": [0, 4 * 3600 * 1000, 24 * 3600 * 1000],
            "net_return": [0.01, -0.005, 0.02],
        }
    )
    daily = daily_portfolio(portfolio)
    assert len(daily) == 2
    assert daily["timestamp"].tolist() == [0, 86_400_000]
    assert np.isclose(daily.iloc[0]["net_return"], (1.01 * 0.995) - 1.0)


def test_contiguous_block_lcb_is_positive_for_positive_series() -> None:
    value = contiguous_block_lcb(
        np.full(100, 2.0), samples=200, alpha=0.05, block_size=7, seed=1
    )
    assert value == 2.0
