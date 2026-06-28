from __future__ import annotations

from panteon_v2.selection.causal_actor_router import (
    CausalActorRouterConfig,
    route_actor,
)


def test_router_selects_best_prior_causal_actor_for_symbol_and_regime():
    candidates = [
        {"label": "MeanEnsemble", "actor_type": "ensemble", "score": 9.0, "risk_mult": 1.0},
        {"label": "RegimeSpecialist", "actor_type": "agent", "score": 2.0, "risk_mult": 1.0},
    ]
    memory = {
        "MeanEnsemble": [
            {
                "bar": 20,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 12,
                "expectancy": 0.05,
                "pnl_lcb": 0.01,
            }
        ],
        "RegimeSpecialist": [
            {
                "bar": 20,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 8,
                "expectancy": 0.35,
                "pnl_lcb": 0.12,
            }
        ],
    }

    routed = route_actor(
        candidates,
        memory,
        symbol="BTC",
        regime="bullish",
        bar=30,
        config=CausalActorRouterConfig(min_closed_trades=5),
    )

    assert routed.label == "RegimeSpecialist"
    assert routed.score_source == "causal_memory"
    assert routed.sample_closed == 8
    assert routed.expectancy == 0.35


def test_router_ignores_current_or_future_metrics_to_avoid_ex_post_oracle():
    candidates = [
        {"label": "FutureWinner", "actor_type": "agent", "score": 10.0},
        {"label": "PriorWinner", "actor_type": "agent", "score": 1.0},
    ]
    memory = {
        "FutureWinner": [
            {
                "bar": 30,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 20,
                "expectancy": 5.0,
                "pnl_lcb": 2.0,
            }
        ],
        "PriorWinner": [
            {
                "bar": 29,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 6,
                "expectancy": 0.2,
                "pnl_lcb": 0.05,
            }
        ],
    }

    routed = route_actor(
        candidates,
        memory,
        symbol="BTC",
        regime="bullish",
        bar=30,
        config=CausalActorRouterConfig(min_closed_trades=5),
    )

    assert routed.label == "PriorWinner"


def test_router_uses_exploration_floor_for_sparse_positive_actor():
    candidates = [
        {"label": "SparsePositive", "actor_type": "agent", "score": 0.2, "risk_mult": 1.0},
        {"label": "NoSample", "actor_type": "agent", "score": 2.0, "risk_mult": 1.0},
    ]
    memory = {
        "SparsePositive": [
            {
                "bar": 10,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 2,
                "expectancy": 0.15,
                "pnl_lcb": 0.01,
            }
        ]
    }

    routed = route_actor(
        candidates,
        memory,
        symbol="BTC",
        regime="bullish",
        bar=20,
        config=CausalActorRouterConfig(
            min_closed_trades=5,
            exploration_enabled=True,
            exploration_risk_mult=0.05,
        ),
    )

    assert routed.label == "SparsePositive"
    assert routed.reason == "exploration_floor"
    assert routed.risk_mult == 0.05


def test_router_does_not_use_component_mean_for_ensemble_without_own_sample():
    candidates = [
        {
            "label": "MeanOnlyEnsemble",
            "actor_type": "ensemble",
            "score": 10.0,
            "component_scores": [10.0, 9.0],
        },
        {"label": "CausalAgent", "actor_type": "agent", "score": 1.0},
    ]
    memory = {
        "CausalAgent": [
            {
                "bar": 5,
                "symbol": "BTC",
                "regime": "bullish",
                "closed_trades": 5,
                "expectancy": 0.12,
                "pnl_lcb": 0.04,
            }
        ]
    }

    routed = route_actor(
        candidates,
        memory,
        symbol="BTC",
        regime="bullish",
        bar=10,
        config=CausalActorRouterConfig(min_closed_trades=5),
    )

    assert routed.label == "CausalAgent"
    assert routed.score_source == "causal_memory"
