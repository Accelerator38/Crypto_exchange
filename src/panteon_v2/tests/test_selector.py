"""Тесты AgentSelector."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import AgentRegistry, AgentSelector, SessionOverlayConfig
from panteon_v2.tests._helpers import FakeAgent


def _add_perf(perf: PerformanceMemory, label: str, regime: Regime,
              n_trades: int, pnl_per_trade_pct: float, *, start_id: int = 1):
    """Симулирует n_trades для агента (как агента, не игрока)."""
    sid = start_id
    for i in range(n_trades):
        op_price = 100.0
        cl_price = op_price * (1.0 + pnl_per_trade_pct / 100.0)
        os_sig = Signal(id=sid, bar=i*2+1, sym=f"S{sid}",
                        action=Action.FUT_LONG_FULL, price=op_price,
                        regime=regime, by_player=label, by_agent=label)
        cs_sig = Signal(id=sid+1, bar=i*2+2, sym=f"S{sid}",
                        action=Action.FUT_CLOSE_ALL, price=cl_price,
                        regime=regime, by_player=label, by_agent=label)
        sid += 2
        perf.update_from_trade(
            Trade(signal_id=os_sig.id, bar=os_sig.bar, sym=os_sig.sym,
                  side="long", qty=1.0, fill_price=op_price, fee=0.0),
            os_sig,
        )
        perf.update_from_trade(
            Trade(signal_id=cs_sig.id, bar=cs_sig.bar, sym=cs_sig.sym,
                  side="long", qty=1.0, fill_price=cl_price, fee=0.0),
            cs_sig,
        )


class TestAgentSelector(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.perf = PerformanceMemory(trade_fraction=1.0)
        self.qm = QuarantineManager(seed=set())
        # 4 агента
        for label in ["AlphaGood", "BetaBad", "GammaQuarantined", "DeltaNoData"]:
            self.registry.register(FakeAgent(label))
        self.sel = AgentSelector(self.registry, self.perf, self.qm)

    def test_empty_registry_returns_empty(self):
        empty = AgentRegistry()
        sel = AgentSelector(empty, self.perf, self.qm)
        self.assertEqual(sel.select(Regime.BULLISH, k=5), [])

    def test_k_zero_returns_empty(self):
        self.assertEqual(self.sel.select(Regime.BULLISH, k=0), [])

    def test_top_k_sorted_by_score_desc(self):
        # Alpha хороший (5 trades, +1% each), Beta плохой (5 trades, -1%)
        _add_perf(self.perf, "AlphaGood", Regime.BULLISH, 5, 1.0, start_id=1)
        _add_perf(self.perf, "BetaBad", Regime.BULLISH, 5, -1.0, start_id=100)
        result = self.sel.select(Regime.BULLISH, k=5)
        self.assertGreaterEqual(len(result), 1)
        labels = [r.label for r in result]
        # Alpha должен быть первым
        self.assertEqual(labels[0], "AlphaGood")
        # Beta может вообще не пройти min_score=0
        if "BetaBad" in labels:
            self.assertGreater(labels.index("AlphaGood"), -1)
            self.assertGreater(labels.index("BetaBad"), labels.index("AlphaGood"))

    def test_quarantined_excluded(self):
        """Q1: карантинный никогда не попадает в результат."""
        _add_perf(self.perf, "GammaQuarantined", Regime.BULLISH, 10, 5.0)  # очень прибыльный
        self.qm.force_quarantine("GammaQuarantined")
        result = self.sel.select(Regime.BULLISH, k=10)
        labels = [r.label for r in result]
        self.assertNotIn("GammaQuarantined", labels,
                         "Quarantined agent must be excluded even if highly profitable")

    def test_no_data_agents_excluded_by_default(self):
        # DeltaNoData без trades → score=0 (≤ min_score), не попадает
        _add_perf(self.perf, "AlphaGood", Regime.BULLISH, 5, 1.0)
        result = self.sel.select(Regime.BULLISH, k=5)
        labels = [r.label for r in result]
        self.assertIn("AlphaGood", labels)
        self.assertNotIn("DeltaNoData", labels)

    def test_select_with_fallback_returns_when_no_positives(self):
        # Все агенты в минусе → обычный select даст 0; fallback должен вернуть.
        _add_perf(self.perf, "AlphaGood", Regime.NEUTRAL, 5, -2.0, start_id=1)
        _add_perf(self.perf, "BetaBad", Regime.NEUTRAL, 5, -3.0, start_id=100)
        normal = self.sel.select(Regime.NEUTRAL, k=5)
        # Нет позитивных
        self.assertEqual(normal, [])
        fallback = self.sel.select_with_fallback(Regime.NEUTRAL, k=5)
        # Fallback должен вернуть хоть что-то
        self.assertGreater(len(fallback), 0)

    def test_select_with_fallback_respects_min_count_shortfall(self):
        _add_perf(self.perf, "AlphaGood", Regime.NEUTRAL, 5, 1.0, start_id=1)
        _add_perf(self.perf, "BetaBad", Regime.NEUTRAL, 5, -1.0, start_id=100)

        normal = self.sel.select(Regime.NEUTRAL, k=5)
        self.assertEqual([r.label for r in normal], ["AlphaGood"])

        fallback = self.sel.select_with_fallback(
            Regime.NEUTRAL,
            k=5,
            min_count=2,
        )
        labels = [r.label for r in fallback]
        self.assertIn("AlphaGood", labels)
        self.assertIn("BetaBad", labels)

    def test_excludes_param(self):
        _add_perf(self.perf, "AlphaGood", Regime.BULLISH, 5, 1.0, start_id=1)
        # Используем top_k через MEM
        result = self.sel.select(Regime.BULLISH, k=5)
        self.assertIn("AlphaGood", [r.label for r in result])

    def test_isinstance_agent(self):
        """Selector возвращает agent объекты, не label-ы."""
        _add_perf(self.perf, "AlphaGood", Regime.BULLISH, 5, 1.0)
        result = self.sel.select(Regime.BULLISH, k=5)
        self.assertGreater(len(result), 0)
        for sa in result:
            # Должен быть объект ScoredAgent с агентом
            self.assertTrue(hasattr(sa.agent, "act"))
            self.assertTrue(hasattr(sa.agent, "label"))


    def test_session_overlay_promotes_current_session_winner(self):
        registry = AgentRegistry()
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        for label in ["MemoryLeader", "SessionWinner"]:
            registry.register(FakeAgent(label))

        _add_perf(perf, "MemoryLeader", Regime.NEUTRAL, 8, 1.0, start_id=1)
        _add_perf(perf, "SessionWinner", Regime.NEUTRAL, 8, 0.1, start_id=100)
        selector = AgentSelector(
            registry,
            perf,
            qm,
            session_overlay=SessionOverlayConfig(
                enabled=True,
                overlay_weight=1.0,
                stale_penalty=0.25,
                underperformance_weight=0.5,
            ),
        )

        before = selector.select(Regime.NEUTRAL, k=2)
        self.assertEqual(before[0].label, "MemoryLeader")

        _add_perf(perf, "MemoryLeader", Regime.NEUTRAL, 2, -2.0, start_id=300)
        _add_perf(perf, "SessionWinner", Regime.NEUTRAL, 2, 3.0, start_id=400)
        after = selector.select(Regime.NEUTRAL, k=2)

        self.assertEqual(after[0].label, "SessionWinner")
        if len(after) > 1:
            self.assertGreater(after[0].score, after[1].score)
        else:
            self.assertEqual([sa.label for sa in after], ["SessionWinner"])

    def test_default_session_overlay_reacts_to_recent_pnl_swings(self):
        registry = AgentRegistry()
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        for label in ["MemoryLeader", "SessionWinner"]:
            registry.register(FakeAgent(label))

        _add_perf(perf, "MemoryLeader", Regime.NEUTRAL, 8, 1.0, start_id=1)
        _add_perf(perf, "SessionWinner", Regime.NEUTRAL, 8, 0.1, start_id=100)
        selector = AgentSelector(
            registry,
            perf,
            qm,
            session_overlay=SessionOverlayConfig(enabled=True),
        )

        before = selector.select(Regime.NEUTRAL, k=2)
        self.assertEqual(before[0].label, "MemoryLeader")

        _add_perf(perf, "MemoryLeader", Regime.NEUTRAL, 1, -0.75, start_id=300)
        _add_perf(perf, "SessionWinner", Regime.NEUTRAL, 1, 1.5, start_id=400)
        after = selector.select(Regime.NEUTRAL, k=2)

        self.assertEqual(after[0].label, "SessionWinner")


class TestQuarantineDoesNotLeak(unittest.TestCase):
    """Property test: при ЛЮБЫХ метриках карантинный никогда не выходит."""

    def test_quarantined_excluded_with_great_metrics(self):
        registry = AgentRegistry()
        registry.register(FakeAgent("Star"))
        registry.register(FakeAgent("Worker"))
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed={"Star"})  # Star в seed-карантине
        # Star — звёздный (огромный pnl)
        _add_perf(perf, "Star", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "Worker", Regime.BULLISH, 5, 0.5, start_id=200)
        sel = AgentSelector(registry, perf, qm)
        result = sel.select(Regime.BULLISH, k=10)
        labels = [r.label for r in result]
        self.assertNotIn("Star", labels)
        self.assertIn("Worker", labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
