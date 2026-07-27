from __future__ import annotations

import pytest

from panteon_v2.app.agent_bootstrap import _ensure_paths


_ensure_paths()

import panteon_agents
from carryflow_policy import (
    apply_carryflow_profile,
    carryflow_profile_ids,
    get_carryflow_profile,
)
from panteon_agents import CarryFlowAgentV2


def test_profiles_compile_to_one_manifest_knob():
    assert carryflow_profile_ids() == (
        "screened_short_v1",
        "divergence_short_v1",
        "divergence_short_systemic_guard_v1",
        "divergence_short_non_range_v1",
        "divergence_short_strict_v1",
        "crowded_overextension_short_v1",
    )
    for profile_id in carryflow_profile_ids():
        profile = get_carryflow_profile(profile_id)
        assert {"PROFILE_ID": profile.profile_id} == {"PROFILE_ID": profile_id}
        assert profile.actor_values()["ALLOW_LONG"] is False
        assert profile.actor_values()["ALLOW_SHORT"] is True


def test_divergence_profile_uses_multi_hour_oi_and_price_confirmation():
    context = {
        "funding_rate": 0.0002,
        "open_interest_usdt": 100.0,
        "long_ratio": 0.65,
        "short_ratio": 0.35,
        "mark_price": 99.95,
        "index_price": 100.0,
        "last_price": 100.0,
        "context_complete": True,
    }

    class Fetcher:
        def get(self, symbol):
            return dict(context)

    def evaluate(last_price: float):
        actor = CarryFlowAgentV2()
        apply_carryflow_profile(actor, "divergence_short_v1")
        oi_values = (100.0, 100.1, 100.2, 102.0)
        prices = (100.0, 100.0, 100.0, last_price)
        result = None
        for bar, (price, oi_value) in enumerate(
            zip(prices, oi_values),
            start=1,
        ):
            context["open_interest_usdt"] = oi_value
            result = actor.act({"BTC": price}, {"BTC": 1.0}, bar_index=bar)
        return actor, result

    panteon_agents.set_fetcher(Fetcher())
    try:
        allowed_actor, allowed = evaluate(100.0)
        blocked_actor, blocked = evaluate(101.0)
    finally:
        panteon_agents.set_fetcher(None)

    assert allowed["BTC"] == 7
    assert allowed_actor.last_signal_diagnostics["BTC"]["oi_lookback_bars"] == 3
    assert allowed_actor.last_signal_diagnostics["BTC"]["price_return"] == 0.0
    assert allowed_actor.last_signal_diagnostics["BTC"]["ranking_score"] == (
        allowed_actor.last_signal_diagnostics["BTC"]["edge"]
    )
    assert allowed_actor.last_signal_diagnostics["BTC"][
        "price_dislocation_bps"
    ] == 50.0
    assert allowed_actor.last_signal_diagnostics["BTC"][
        "oi_excess_bps"
    ] == pytest.approx(50.0)
    assert blocked["BTC"] == 0
    assert (
        blocked_actor.last_signal_diagnostics["BTC"]["reason"]
        == "short_price_divergence_missing"
    )


def test_systemic_guard_requires_majority_breadth_and_excludes_bullish():
    symbols = ("BTC", "ETH", "SOL", "XRP")
    contexts = {
        symbol: {
            "funding_rate": 0.0,
            "open_interest_usdt": 100.0,
            "long_ratio": 0.65,
            "short_ratio": 0.35,
            "mark_price": 99.95,
            "index_price": 100.0,
            "last_price": 100.0,
            "context_complete": True,
        }
        for symbol in symbols
    }
    contexts["BTC"]["funding_rate"] = 0.0002

    class Fetcher:
        def get(self, symbol):
            return dict(contexts[symbol])

    def evaluate(final_prices):
        actor = CarryFlowAgentV2()
        profile = apply_carryflow_profile(
            actor,
            "divergence_short_systemic_guard_v1",
        )
        actions = None
        for bar in range(1, 5):
            contexts["BTC"]["open_interest_usdt"] = (
                102.0 if bar == 4 else 100.0
            )
            prices = (
                dict(final_prices)
                if bar == 4
                else {symbol: 100.0 for symbol in symbols}
            )
            actions = actor.act(
                prices,
                {symbol: 1.0 for symbol in symbols},
                bar_index=bar,
            )
        return profile, actor, actions

    panteon_agents.set_fetcher(Fetcher())
    try:
        blocked_profile, blocked_actor, blocked = evaluate(
            {"BTC": 100.0, "ETH": 98.0, "SOL": 98.0, "XRP": 98.0}
        )
        allowed_profile, allowed_actor, allowed = evaluate(
            {"BTC": 100.0, "ETH": 101.0, "SOL": 100.0, "XRP": 98.0}
        )
    finally:
        panteon_agents.set_fetcher(None)

    assert "bullish" not in blocked_profile.allowed_regimes
    assert allowed_profile.allowed_regimes == blocked_profile.allowed_regimes
    assert blocked["BTC"] == 0
    assert (
        blocked_actor.last_signal_diagnostics["BTC"]["reason"]
        == "short_systemic_selloff_guard"
    )
    assert blocked_actor.last_signal_diagnostics["BTC"][
        "market_positive_breadth"
    ] == 0.25
    assert allowed["BTC"] == 7
    assert allowed_actor.last_signal_diagnostics["BTC"][
        "market_positive_breadth"
    ] == 0.75


def test_overextension_profile_does_not_require_oi_expansion():
    context = {
        "funding_rate": 0.00006,
        "open_interest_usdt": 100.0,
        "long_ratio": 0.70,
        "short_ratio": 0.30,
        "mark_price": 99.95,
        "index_price": 100.0,
        "last_price": 100.0,
        "context_complete": True,
    }

    class Fetcher:
        def get(self, symbol):
            return dict(context)

    def evaluate(last_price: float):
        actor = CarryFlowAgentV2()
        apply_carryflow_profile(actor, "crowded_overextension_short_v1")
        result = None
        for bar, price in enumerate((100.0, 100.0, 100.0, last_price), start=1):
            result = actor.act({"BTC": price}, {"BTC": 1.0}, bar_index=bar)
        return actor, result

    panteon_agents.set_fetcher(Fetcher())
    try:
        allowed_actor, allowed = evaluate(100.6)
        blocked_actor, blocked = evaluate(100.4)
    finally:
        panteon_agents.set_fetcher(None)

    assert allowed["BTC"] == 7
    assert allowed_actor.last_signal_diagnostics["BTC"]["oi_change"] == 0.0
    assert allowed_actor.last_signal_diagnostics["BTC"]["price_return"] > 0.005
    assert blocked["BTC"] == 0
    assert (
        blocked_actor.last_signal_diagnostics["BTC"]["reason"]
        == "short_price_overextension_missing"
    )


def test_overextension_profile_ranks_by_price_extension():
    contexts = {
        "AAA": {
            "funding_rate": 0.0001,
            "open_interest_usdt": 100.0,
            "long_ratio": 0.90,
            "short_ratio": 0.10,
            "mark_price": 100.0,
            "index_price": 100.0,
            "last_price": 100.6,
            "context_complete": True,
        },
        "BBB": {
            "funding_rate": 0.00005,
            "open_interest_usdt": 100.0,
            "long_ratio": 0.68,
            "short_ratio": 0.32,
            "mark_price": 99.86,
            "index_price": 100.0,
            "last_price": 101.0,
            "context_complete": True,
        },
    }

    class Fetcher:
        def get(self, symbol):
            return dict(contexts[symbol])

    actor = CarryFlowAgentV2()
    apply_carryflow_profile(actor, "crowded_overextension_short_v1")
    panteon_agents.set_fetcher(Fetcher())
    try:
        actions = None
        for bar, prices in enumerate(
            (
                {"AAA": 100.0, "BBB": 100.0},
                {"AAA": 100.0, "BBB": 100.0},
                {"AAA": 100.0, "BBB": 100.0},
                {"AAA": 100.6, "BBB": 101.0},
            ),
            start=1,
        ):
            actions = actor.act(prices, {"AAA": 1.0, "BBB": 1.0}, bar_index=bar)
    finally:
        panteon_agents.set_fetcher(None)

    assert actions == {"AAA": 0, "BBB": 7}
