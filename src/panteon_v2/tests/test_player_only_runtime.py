from __future__ import annotations

from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.app.main_loop import main_loop
from panteon_v2.app.shadow_tournament import ProductionShadowTournament
from panteon_v2.domain.types import Action, Regime
from panteon_v2.execution.exchange import FakeExchange
from panteon_v2.execution import RiskLimitsConfig
from panteon_v2.selection.agent import AgentRegistry
from panteon_v2.selection.player_regime import PlayerRegimeConfig
from panteon_v2.shadow.adapters import make_market_snapshot
from panteon_v2.shadow.feed import ReplayFeed


class _ThreeBarEdgePlayer:
    label = "EdgePlayer"
    live_trading_eligible = True
    shadow_only = False

    def act(self, market):
        if market.bar in {1, 3}:
            return {"BTC": Action.FUT_LONG_FULL}
        if market.bar == 2:
            return {"BTC": Action.FUT_CLOSE_ALL}
        return {}


class _LifecyclePlayer:
    label = "LifecyclePlayer"
    live_trading_eligible = True
    shadow_only = False

    def act(self, market):
        actions = {
            1: {"ETH": Action.FUT_LONG_FULL},
            2: {"ETH": Action.FUT_CLOSE_ALL},
            3: {
                "BTC": Action.FUT_LONG_FULL,
                "ETH": Action.FUT_LONG_FULL,
            },
            4: {"ETH": Action.FUT_CLOSE_ALL},
            5: {"SOL": Action.FUT_LONG_FULL},
            6: {"BTC": Action.FUT_CLOSE_ALL},
        }
        return actions.get(market.bar, {})


def test_multi_uses_only_prior_bar_shadow_results_for_real_selection():
    """A close observed on bar 2 may promote a player only from bar 3."""
    registry = AgentRegistry()
    registry.register(_ThreeBarEdgePlayer())
    exchange = FakeExchange(name="BITGET_TEST")
    risk = RiskLimitsConfig(
        capital_fraction=0.10,
        min_notional_usd=0.0,
    )
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=1_000.0,
        risk_config=risk,
        player_only_runtime=True,
        trade_mode="multi",
        player_regime_config=PlayerRegimeConfig(
            min_closed_trades=1,
            min_global_closed_trades=0,
            global_score_weight=0.0,
            recency_decay=1.0,
            downside_penalty=0.0,
            switch_margin=0.0,
            cooldown_bars=0,
        ),
    )
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=registry,
        perf=pipeline.virtual_perf,
        risk_config=risk,
        players_only=True,
        event_log=pipeline.event_log,
    )
    feed = ReplayFeed([
        make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bullish",
        ),
        make_market_snapshot(
            bar=2,
            prices={"BTC": 110.0},
            regime="bullish",
        ),
        make_market_snapshot(
            bar=3,
            prices={"BTC": 111.0},
            regime="bullish",
        ),
    ])

    steps = main_loop(pipeline, feed)

    assert [step.leader for step in steps] == ["NoTrade", "NoTrade", "EdgePlayer"]
    assert steps[1].n_filled == 0
    assert steps[2].n_filled == 1
    assert len(exchange.orders_log) == 1
    assert pipeline.flash_allocator is None
    assert pipeline.profiles == []
    assert pipeline.regime_switch_player_sets == ()
    assert pipeline.rotating_agent_player_sets == ()
    assert pipeline.virtual_perf.recent_returns(
        "EdgePlayer",
        Regime.BULLISH,
        limit=10,
    )
    assert pipeline.virtual_perf.recent_returns(
        "",
        Regime.BULLISH,
        limit=10,
    ) == ()
    assert all(
        update.actor_type == "player"
        for update in pipeline.shadow_tournament.last_actor_updates()
    )
    assert pipeline.shadow_tournament.last_agent_signals() == {}


def test_player_switch_waits_for_owner_to_close_real_position():
    registry = AgentRegistry()
    registry.register(_LifecyclePlayer())
    exchange = FakeExchange(name="BITGET_TEST")
    risk = RiskLimitsConfig(
        capital_fraction=0.10,
        min_notional_usd=0.0,
    )
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=1_000.0,
        risk_config=risk,
        player_only_runtime=True,
        trade_mode="multi",
        player_regime_config=PlayerRegimeConfig(
            min_closed_trades=1,
            min_global_closed_trades=0,
            global_score_weight=0.0,
            recency_decay=1.0,
            downside_penalty=0.0,
            switch_margin=0.0,
            cooldown_bars=0,
        ),
    )
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=registry,
        perf=pipeline.virtual_perf,
        risk_config=risk,
        players_only=True,
        event_log=pipeline.event_log,
    )
    feed = ReplayFeed([
        make_market_snapshot(
            bar=bar,
            prices={
                "BTC": 100.0 + bar,
                "ETH": {1: 100.0, 2: 120.0, 3: 120.0, 4: 40.0}.get(bar, 40.0),
                "SOL": 50.0,
            },
            regime="bullish",
        )
        for bar in range(1, 8)
    ])

    steps = main_loop(pipeline, feed)

    assert [step.leader for step in steps] == [
        "NoTrade",
        "NoTrade",
        "LifecyclePlayer",
        "LifecyclePlayer",
        "LifecyclePlayer",
        "LifecyclePlayer",
        "NoTrade",
    ]
    assert "portfolio is flat" in steps[4].fallback_reason
    assert any(
        detail.startswith("manage_only_open:SOL")
        for detail in steps[4].signal_filter_details
    )
    assert [(order.trade.bar, order.trade.sym) for order in exchange.orders_log] == [
        (3, "BTC"),
        (6, "BTC"),
    ]
