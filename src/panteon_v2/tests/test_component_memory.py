from __future__ import annotations

from panteon_v2.selection.component_memory import ComponentMemory, ComponentStat


def test_component_memory_returns_only_prior_bar_matching_context():
    memory = ComponentMemory([
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="BTC/USDT",
            regime="bearish",
            action="FUT_SHORT_HALF",
            bar=9,
            closed_trades=5,
            expectancy=0.25,
        ),
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="BTC/USDT",
            regime="bearish",
            action="FUT_SHORT_HALF",
            bar=10,
            closed_trades=50,
            expectancy=9.0,
        ),
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="ETH/USDT",
            regime="bearish",
            action="FUT_SHORT_HALF",
            bar=8,
            closed_trades=10,
            expectancy=1.0,
        ),
    ])

    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="bearish",
        action="FUT_SHORT_HALF",
        bar=10,
    )

    assert stat is not None
    assert stat.bar == 9
    assert stat.expectancy == 0.25


def test_component_memory_ignores_current_bar_to_avoid_ex_post_oracle():
    memory = ComponentMemory([
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="BTC/USDT",
            regime="bearish",
            action="FUT_SHORT_HALF",
            bar=10,
            closed_trades=50,
            expectancy=9.0,
        )
    ])

    assert memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="bearish",
        action="FUT_SHORT_HALF",
        bar=10,
    ) is None


def test_component_memory_uses_prior_wildcard_context_when_exact_missing():
    memory = ComponentMemory([
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="*",
            regime="range_low_vol",
            action="*",
            bar=9,
            closed_trades=6,
            expectancy=0.08,
            pnl_lcb=0.02,
        )
    ])

    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="range_low_vol",
        action="FUT_SHORT_FULL",
        bar=10,
    )

    assert stat is not None
    assert stat.symbol == "*"
    assert stat.action == "*"
    assert stat.expectancy == 0.08


def test_component_memory_prefers_exact_context_over_wildcard():
    memory = ComponentMemory([
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="*",
            regime="range_low_vol",
            action="*",
            bar=9,
            closed_trades=20,
            expectancy=0.50,
            pnl_lcb=0.10,
        ),
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="BTC/USDT",
            regime="range_low_vol",
            action="FUT_SHORT_FULL",
            bar=8,
            closed_trades=6,
            expectancy=0.08,
            pnl_lcb=0.02,
        ),
    ])

    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="range_low_vol",
        action="FUT_SHORT_FULL",
        bar=10,
    )

    assert stat is not None
    assert stat.symbol == "BTC/USDT"
    assert stat.action == "FUT_SHORT_FULL"
    assert stat.expectancy == 0.08


def test_component_memory_can_fallback_to_wildcard_when_exact_context_has_too_few_trades():
    memory = ComponentMemory([
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="*",
            regime="neutral",
            action="*",
            bar=8,
            closed_trades=1,
            expectancy=0.50,
        ),
        ComponentStat(
            actor_label="LiveVolCompress",
            symbol="*",
            regime="*",
            action="*",
            bar=8,
            closed_trades=7,
            expectancy=0.12,
        ),
    ])

    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="neutral",
        action="FUT_SHORT_FULL",
        bar=9,
        min_closed_trades=5,
    )

    assert stat is not None
    assert stat.regime == "*"
    assert stat.closed_trades == 7
