"""Тесты Phase 8: adapters, feed, runner, comparator."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Dict, List

from panteon_v2.domain.types import Action, MarketSnapshot, Regime
from panteon_v2.replay.v1_parser import V1Session, V1Signal
from panteon_v2.selection import AgentRegistry
from panteon_v2.shadow import (
    CallableFeed,
    GeneticsV2AgentAdapter,
    ReplayFeed,
    ShadowRunner,
    V1AgentAdapter,
    compare_v1_vs_v2,
    make_market_snapshot,
    regime_to_v1_string,
)
from panteon_v2.tests._helpers import FakeAgent


# ════════════════════════════════════════════════════════════════════
# adapters
# ════════════════════════════════════════════════════════════════════


class TestMakeMarketSnapshot(unittest.TestCase):
    def test_basic(self):
        snap = make_market_snapshot(
            bar=10, prices={"BTC": 100.0, "ETH": 50.0},
            volumes={"BTC": 1000.0, "ETH": 500.0},
            regime="bullish",
        )
        self.assertEqual(snap.bar, 10)
        self.assertEqual(snap.regime, Regime.BULLISH)
        self.assertEqual(snap.prices["BTC"], 100.0)

    def test_filters_zero_prices(self):
        snap = make_market_snapshot(
            bar=1, prices={"BTC": 100.0, "BAD": 0.0, "NEG": -1.0},
            regime="neutral",
        )
        self.assertNotIn("BAD", snap.prices)
        self.assertNotIn("NEG", snap.prices)

    def test_uppercase_symbols(self):
        snap = make_market_snapshot(
            bar=1, prices={"btc": 100.0}, regime="neutral",
        )
        self.assertIn("BTC", snap.prices)
        self.assertNotIn("btc", snap.prices)

    def test_regime_canonization(self):
        for v1, v2 in [
            ("bull", Regime.BULLISH), ("UP", Regime.BULLISH),
            ("bear", Regime.BEARISH), ("flat", Regime.NEUTRAL),
            ("crash", Regime.CRASH), ("garbage", Regime.NEUTRAL),
        ]:
            snap = make_market_snapshot(bar=1, prices={"X": 1.0}, regime=v1)
            self.assertEqual(snap.regime, v2)

    def test_funding_filtered(self):
        snap = make_market_snapshot(
            bar=1, prices={"BTC": 100.0},
            funding={"BTC": 0.001, "UNKNOWN": 0.005},
            regime="neutral",
        )
        self.assertIn("BTC", snap.funding)
        self.assertNotIn("UNKNOWN", snap.funding)

    def test_regime_to_v1_string_roundtrip(self):
        for r in (Regime.BULLISH, Regime.BEARISH, Regime.NEUTRAL, Regime.CRASH):
            self.assertEqual(Regime.from_string(regime_to_v1_string(r)), r)


class TestV1AgentAdapter(unittest.TestCase):
    def test_wraps_v1_act(self):
        class V1FakeAgent:
            def act(self, prices, volumes, month=None, portfolio_value=None,
                   bar_index=None):
                return {sym: 5 for sym in prices}  # full long на всём
        wrapped = V1AgentAdapter(label="V1", v1_agent=V1FakeAgent())
        snap = make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish")
        result = wrapped.act(snap)
        self.assertEqual(result["BTC"], Action.FUT_LONG_FULL)

    def test_filters_unknown_symbols(self):
        class V1Agent:
            def act(self, prices, volumes, **kw):
                return {"BTC": 5, "UNKNOWN": 5}
        wrapped = V1AgentAdapter(label="V1", v1_agent=V1Agent())
        snap = make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish")
        result = wrapped.act(snap)
        self.assertIn("BTC", result)
        self.assertNotIn("UNKNOWN", result)

    def test_handles_invalid_action_codes(self):
        class V1Agent:
            def act(self, prices, volumes, **kw):
                return {"BTC": 999, "ETH": "junk"}
        wrapped = V1AgentAdapter(label="V1", v1_agent=V1Agent())
        snap = make_market_snapshot(bar=1, prices={"BTC": 100.0, "ETH": 50.0},
                                    regime="bullish")
        result = wrapped.act(snap)
        # Оба невалидны → отфильтрованы
        self.assertEqual(result, {})

    def test_handles_v1_exception(self):
        class V1Agent:
            def act(self, prices, volumes, **kw):
                raise RuntimeError("boom")
        wrapped = V1AgentAdapter(label="V1", v1_agent=V1Agent())
        snap = make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish")
        result = wrapped.act(snap)
        self.assertEqual(result, {})


class TestGeneticsV2AgentAdapter(unittest.TestCase):
    def test_maps_collapsed_genetics_actions_to_v2_actions(self):
        class GeneticsLikeAgent:
            def act(self, prices, volumes, **kw):
                return {
                    "BTC": 0,
                    "ETH": 1,
                    "SOL": 2,
                    "XRP": 3,
                    "DOGE": 4,
                    "ADA": 5,
                }

        wrapped = GeneticsV2AgentAdapter(
            label="GeneticsNeutral",
            v1_agent=GeneticsLikeAgent(),
        )
        snap = make_market_snapshot(
            bar=1,
            prices={
                "BTC": 100.0,
                "ETH": 50.0,
                "SOL": 20.0,
                "XRP": 1.0,
                "DOGE": 0.2,
                "ADA": 0.4,
            },
            regime="neutral",
        )

        result = wrapped.act(snap)

        self.assertEqual(result["BTC"], Action.HOLD)
        self.assertEqual(result["ETH"], Action.SPOT_BUY_FULL)
        self.assertEqual(result["SOL"], Action.SPOT_SELL_ALL)
        self.assertEqual(result["XRP"], Action.FUT_LONG_FULL)
        self.assertEqual(result["DOGE"], Action.FUT_SHORT_FULL)
        self.assertEqual(result["ADA"], Action.FUT_CLOSE_ALL)

    def test_does_not_treat_genetics_short_as_v2_long_half(self):
        class GeneticsLikeAgent:
            def act(self, prices, volumes, **kw):
                return {"DOGE": 4}

        wrapped = GeneticsV2AgentAdapter(
            label="GeneticsBearish",
            v1_agent=GeneticsLikeAgent(),
        )
        snap = make_market_snapshot(
            bar=1,
            prices={"DOGE": 0.2},
            regime="bearish",
        )

        result = wrapped.act(snap)

        self.assertEqual(result["DOGE"], Action.FUT_SHORT_FULL)

    def test_reuses_cached_genetics_action_for_same_bar_across_shadow_clone(self):
        class GeneticsLikeAgent:
            total_calls = 0

            def __init__(self):
                self.calls = 0

            def act(self, prices, volumes, **kw):
                self.calls += 1
                type(self).total_calls += 1
                return {"BTC": 3}

        agent = GeneticsLikeAgent()
        wrapped = GeneticsV2AgentAdapter(
            label="GeneticsCacheProbe",
            v1_agent=agent,
        )
        clone = wrapped.clone_for_shadow()
        snap = make_market_snapshot(
            bar=777,
            prices={"BTC": 100.0},
            regime="bullish",
        )

        self.assertEqual(wrapped.act(snap)["BTC"], Action.FUT_LONG_FULL)
        self.assertEqual(clone.act(snap)["BTC"], Action.FUT_LONG_FULL)

        self.assertEqual(agent.calls, 1)
        self.assertEqual(GeneticsLikeAgent.total_calls, 1)


# ════════════════════════════════════════════════════════════════════
# feed
# ════════════════════════════════════════════════════════════════════


class TestReplayFeed(unittest.TestCase):
    def test_empty_returns_none(self):
        feed = ReplayFeed()
        self.assertIsNone(feed.next_bar())

    def test_iterates_in_order(self):
        feed = ReplayFeed()
        for bar in (1, 2, 3):
            feed.append(make_market_snapshot(
                bar=bar, prices={"BTC": float(bar)}, regime="neutral",
            ))
        bars = []
        while True:
            s = feed.next_bar()
            if s is None: break
            bars.append(s.bar)
        self.assertEqual(bars, [1, 2, 3])

    def test_remaining(self):
        feed = ReplayFeed()
        feed.append(make_market_snapshot(bar=1, prices={"X": 1.0}, regime="neutral"))
        self.assertEqual(feed.remaining, 1)
        feed.next_bar()
        self.assertEqual(feed.remaining, 0)


class TestCallableFeed(unittest.TestCase):
    def test_calls_function(self):
        snaps = [make_market_snapshot(bar=1, prices={"X": 1.0}, regime="neutral")]
        def fn():
            return snaps.pop(0) if snaps else None
        feed = CallableFeed(fn=fn)
        self.assertIsNotNone(feed.next_bar())
        self.assertIsNone(feed.next_bar())

    def test_swallows_exceptions(self):
        def fn():
            raise RuntimeError("boom")
        feed = CallableFeed(fn=fn)
        self.assertIsNone(feed.next_bar())


# ════════════════════════════════════════════════════════════════════
# runner
# ════════════════════════════════════════════════════════════════════


class TestShadowRunnerBasic(unittest.TestCase):
    def _setup(self, *, agent_action: Action = Action.FUT_LONG_FULL):
        registry = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveOIBreakout"]:
            registry.register(FakeAgent(label, {"BTC": agent_action}))
        feed = ReplayFeed()
        for bar in range(1, 5):
            feed.append(make_market_snapshot(
                bar=bar, prices={"BTC": 100.0 + bar},
                regime="bullish",
            ))
        runner = ShadowRunner.with_defaults(
            registry=registry, feed=feed,
            balance_usd=1000.0,
        )
        return runner, feed

    def test_step_returns_none_when_exhausted(self):
        runner, feed = self._setup()
        feed._index = len(feed.snapshots)  # эмулируем конец
        self.assertIsNone(runner.step())

    def test_first_step_emits_bar_started(self):
        from panteon_v2.attribution import BarStarted
        runner, _ = self._setup()
        runner.step()
        events = list(runner.event_log.query(event_types=[BarStarted]))
        self.assertEqual(len(events), 1)

    def test_run_to_exhaustion(self):
        runner, _ = self._setup()
        steps = runner.run_until_exhausted()
        # 4 баров → 4 step results
        self.assertEqual(len(steps), 4)

    def test_quarantine_blocks_leader(self):
        """Q4: если все агенты в карантине, runner не выбирает их в leader."""
        registry = AgentRegistry()
        registry.register(FakeAgent("BadAgent",
                                    {"BTC": Action.FUT_LONG_FULL}))
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1, prices={"BTC": 100.0}, regime="bullish",
        ))
        runner = ShadowRunner.with_defaults(
            registry=registry, feed=feed,
            seed_quarantine={"BadAgent"},
        )
        step = runner.step()
        # Карантинный → composer не сможет создать ensemble (min_agents)
        # → leader=None
        self.assertIsNone(step.leader)


# ════════════════════════════════════════════════════════════════════
# comparator
# ════════════════════════════════════════════════════════════════════


def _make_v1_session(sigs: List[V1Signal], pnl: float = 0.0,
                    quarantined: List[str] = None) -> V1Session:
    return V1Session(
        session_path="/test",
        exchange="MEXC",
        initial_capital=100.0,
        current_balance=100.0 * (1 + pnl / 100),
        pnl_usd=pnl,
        pnl_pct=pnl,
        bar_count=10, live_bar_count=10,
        n_signals=len(sigs), n_trades=0,
        current_regime="bullish",
        real_signals=sigs,
        recent_trades=[],
        agents={}, players={},
        quarantined=quarantined or [],
    )


class TestComparator(unittest.TestCase):
    def test_empty_v1_no_match(self):
        session = _make_v1_session([])
        from panteon_v2.attribution import AttributionLedger
        from panteon_v2.memory import QuarantineManager
        report = compare_v1_vs_v2(
            session, v2_steps=[],
            v2_qm=QuarantineManager(seed=set()),
            v2_ledger=AttributionLedger(),
        )
        self.assertEqual(report.n_bars_compared, 0)
        self.assertEqual(report.n_v1_signals, 0)
        self.assertEqual(report.n_v2_signals, 0)

    def test_quarantine_diff_in_notes(self):
        session = _make_v1_session([], quarantined=["FundingArb"])
        from panteon_v2.attribution import AttributionLedger
        from panteon_v2.memory import QuarantineManager
        # v2: пустой qm
        report = compare_v1_vs_v2(
            session, v2_steps=[],
            v2_qm=QuarantineManager(seed=set()),
            v2_ledger=AttributionLedger(),
        )
        # v1 имел FundingArb в карантине, v2 — нет → нота
        self.assertTrue(any("FundingArb" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
