"""Тесты QuarantineManager."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.memory import (
    PerformanceMemory,
    QuarantineManager,
    RecomputeResult,
)


def _seal_trade_for_label(perf: PerformanceMemory,
                          label: str,
                          regime: Regime,
                          pnl_pct: float,
                          closed_trades: int = 5,
                          *,
                          start_id: int = 1):
    """Хелпер: симулируем `closed_trades` сделок с заданным средним pnl_pct.

    Каждая сделка даёт ~ pnl_pct/closed_trades. Ограничиваемся long-открытием.
    """
    avg_per = pnl_pct / max(closed_trades, 1) / 100.0
    sid = start_id
    for i in range(closed_trades):
        op_price = 100.0
        cl_price = op_price * (1.0 + avg_per)
        os_sig = Signal(id=sid, bar=i*2+1, sym=f"S{i}",
                        action=Action.FUT_LONG_FULL, price=op_price,
                        regime=regime, by_player=label, by_agent=label)
        cs_sig = Signal(id=sid+1, bar=i*2+2, sym=f"S{i}",
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


class TestSeed(unittest.TestCase):
    def test_seed_is_quarantined_initially(self):
        qm = QuarantineManager(seed={"FundingArb", "RichardDennis"})
        self.assertTrue(qm.is_quarantined("FundingArb"))
        self.assertTrue(qm.is_quarantined("RichardDennis"))
        self.assertFalse(qm.is_quarantined("LiveAfterShock"))

    def test_seed_immutable(self):
        original = {"X", "Y"}
        qm = QuarantineManager(seed=original)
        original.add("Z")  # mutating outside
        self.assertFalse(qm.is_quarantined("Z"),
                         "seed must be a copy, not reference")

    def test_all_quarantined_returns_immutable(self):
        qm = QuarantineManager(seed={"A", "B"})
        snap = qm.all_quarantined()
        self.assertIsInstance(snap, frozenset)


class TestRecompute(unittest.TestCase):
    def test_recompute_empty_keeps_seed(self):
        """Без данных в perf, seed остаётся в карантине."""
        qm = QuarantineManager(seed={"FundingArb"})
        perf = PerformanceMemory(trade_fraction=1.0)
        result = qm.recompute(perf)
        self.assertTrue(qm.is_quarantined("FundingArb"))
        # No-op (ничего не добавилось/удалилось):
        self.assertTrue(result.is_no_op)

    def test_seed_released_when_locally_proven(self):
        """Seed-агент с положительным per-regime опытом → выходит."""
        qm = QuarantineManager(seed={"FundingArb"})
        perf = PerformanceMemory(trade_fraction=1.0)
        # +0.5% в bullish (5 closed) — выше recovery_pnl=0.10%
        _seal_trade_for_label(perf, "FundingArb", Regime.BULLISH,
                              pnl_pct=2.5, closed_trades=5)
        result = qm.recompute(perf)
        self.assertFalse(qm.is_quarantined("FundingArb"))
        self.assertIn("FundingArb", result.removed)

    def test_new_agent_added_when_hopeless(self):
        """Не-seed агент со всеми режимами в минусе → добавляется."""
        qm = QuarantineManager(seed=set())
        perf = PerformanceMemory(trade_fraction=1.0)
        # Все режимы в минусе, выборка достаточна
        _seal_trade_for_label(perf, "BadAgent", Regime.BULLISH,
                              pnl_pct=-2.0, closed_trades=3, start_id=1)
        _seal_trade_for_label(perf, "BadAgent", Regime.BEARISH,
                              pnl_pct=-1.5, closed_trades=2, start_id=100)
        _seal_trade_for_label(perf, "BadAgent", Regime.NEUTRAL,
                              pnl_pct=-1.0, closed_trades=2, start_id=200)
        # total closed = 7 ≥ 5; worst pnl_pct = -2.0 ≤ -0.30
        result = qm.recompute(perf)
        self.assertTrue(qm.is_quarantined("BadAgent"))
        self.assertIn("BadAgent", result.added)

    def test_new_agent_added_when_one_regime_dominates_catastrophic_loss(self):
        qm = QuarantineManager(seed=set())
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(perf, "LiveMeanRev", Regime.NEUTRAL,
                              pnl_pct=-100.0, closed_trades=80, start_id=1)
        _seal_trade_for_label(perf, "LiveMeanRev", Regime.BULLISH,
                              pnl_pct=0.02, closed_trades=8, start_id=1000)
        _seal_trade_for_label(perf, "LiveMeanRev", Regime.CRASH,
                              pnl_pct=0.12, closed_trades=4, start_id=2000)

        result = qm.recompute(perf)

        self.assertTrue(qm.is_quarantined("LiveMeanRev"))
        self.assertIn("LiveMeanRev", result.added)

    def test_idempotent(self):
        """Повторный recompute без изменений perf → no_op."""
        qm = QuarantineManager(seed={"FundingArb"})
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(perf, "FundingArb", Regime.BULLISH,
                              pnl_pct=2.5, closed_trades=5)
        first = qm.recompute(perf)
        second = qm.recompute(perf)
        self.assertFalse(first.is_no_op)
        self.assertTrue(second.is_no_op)

    def test_deterministic(self):
        """Один и тот же perf → одинаковый результат."""
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(perf, "BadAgent", Regime.BULLISH,
                              pnl_pct=-2.0, closed_trades=5)
        _seal_trade_for_label(perf, "BadAgent", Regime.BEARISH,
                              pnl_pct=-1.5, closed_trades=2, start_id=100)

        qm1 = QuarantineManager(seed=set())
        qm2 = QuarantineManager(seed=set())
        r1 = qm1.recompute(perf)
        r2 = qm2.recompute(perf)
        self.assertEqual(r1.current, r2.current)

    def test_seed_not_in_perf_stays_quarantined(self):
        qm = QuarantineManager(seed={"NeverTraded"})
        perf = PerformanceMemory(trade_fraction=1.0)
        result = qm.recompute(perf)
        # Никаких сделок — остаётся в карантине
        self.assertTrue(qm.is_quarantined("NeverTraded"))


class TestObservers(unittest.TestCase):
    def test_observer_fires_on_change(self):
        qm = QuarantineManager(seed={"X"})
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(perf, "X", Regime.BULLISH,
                              pnl_pct=2.0, closed_trades=5)
        events = []

        def cb(current, result):
            events.append((current, result))

        qm.subscribe(cb)
        qm.recompute(perf)
        self.assertEqual(len(events), 1)
        current, result = events[0]
        self.assertNotIn("X", current)
        self.assertIn("X", result.removed)

    def test_observer_not_fired_on_no_op(self):
        qm = QuarantineManager(seed={"X"})
        perf = PerformanceMemory(trade_fraction=1.0)
        events = []
        qm.subscribe(lambda c, r: events.append(r))
        qm.recompute(perf)
        # Никаких изменений → observer не должен быть вызван
        self.assertEqual(len(events), 0)

    def test_unsubscribe(self):
        qm = QuarantineManager(seed={"X"})
        events = []
        cb = lambda c, r: events.append(r)
        qm.subscribe(cb)
        self.assertTrue(qm.unsubscribe(cb))
        # Force change
        qm.force_quarantine("Y")
        self.assertEqual(len(events), 0, "unsubscribe didn't work")

    def test_observer_exception_doesnt_break(self):
        qm = QuarantineManager(seed=set())
        def bad_cb(c, r):
            raise RuntimeError("boom")
        good_events = []
        def good_cb(c, r):
            good_events.append(1)
        qm.subscribe(bad_cb)
        qm.subscribe(good_cb)
        qm.force_quarantine("X")
        self.assertEqual(len(good_events), 1,
                         "bad observer must not block good ones")


class TestForceOps(unittest.TestCase):
    def test_force_quarantine(self):
        qm = QuarantineManager(seed=set())
        self.assertFalse(qm.is_quarantined("X"))
        qm.force_quarantine("X")
        self.assertTrue(qm.is_quarantined("X"))

    def test_force_quarantine_idempotent(self):
        qm = QuarantineManager(seed={"X"})
        events = []
        qm.subscribe(lambda c, r: events.append(r))
        qm.force_quarantine("X")  # уже в карантине
        self.assertEqual(len(events), 0, "no event for no-op")

    def test_force_quarantine_survives_recompute_until_release(self):
        qm = QuarantineManager(seed=set())
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(
            perf,
            "GeneticsBullish",
            Regime.BULLISH,
            pnl_pct=5.0,
            closed_trades=5,
        )

        qm.force_quarantine(
            "GeneticsBullish",
            reason="degradation_gate:blocked_signal_rate",
            bar=10,
        )
        result = qm.recompute(perf)

        self.assertTrue(qm.is_quarantined("GeneticsBullish"))
        self.assertNotIn("GeneticsBullish", result.removed)
        record = qm.record_for("GeneticsBullish")
        self.assertIsNotNone(record)
        self.assertEqual(record.reason, "degradation_gate:blocked_signal_rate")

    def test_force_release(self):
        qm = QuarantineManager(seed={"X"})
        self.assertTrue(qm.is_quarantined("X"))
        qm.force_release("X")
        self.assertFalse(qm.is_quarantined("X"))

    def test_force_release_idempotent(self):
        qm = QuarantineManager(seed=set())
        events = []
        qm.subscribe(lambda c, r: events.append(r))
        qm.force_release("X")  # не был в карантине
        self.assertEqual(len(events), 0)


    def test_release_override_prevents_hopeless_requarantine(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(
            perf,
            "LiveVolCompress",
            Regime.NEUTRAL,
            pnl_pct=-1.0,
            closed_trades=5,
        )
        qm = QuarantineManager(seed=set())
        first = qm.recompute(perf)
        self.assertIn("LiveVolCompress", first.added)
        self.assertTrue(qm.is_quarantined("LiveVolCompress"))

        qm.force_release_override(
            "LiveVolCompress",
            reason="session_recovery:pnl_pct=0.50;closed=3",
            bar=12,
        )
        self.assertFalse(qm.is_quarantined("LiveVolCompress"))

        second = qm.recompute(perf)

        self.assertFalse(qm.is_quarantined("LiveVolCompress"))
        self.assertTrue(second.is_no_op)
        self.assertIn("LiveVolCompress", qm.release_override_labels())
        record = qm.record_for("LiveVolCompress")
        self.assertIsNotNone(record)
        self.assertIn("session_recovery", record.reason)

    def test_release_override_does_not_override_degradation_quarantine(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _seal_trade_for_label(
            perf,
            "LiveVolCompress",
            Regime.NEUTRAL,
            pnl_pct=-1.0,
            closed_trades=5,
        )
        qm = QuarantineManager(seed=set())
        qm.recompute(perf)
        qm.force_release_override(
            "LiveVolCompress",
            reason="session_recovery:pnl_pct=0.50;closed=3",
            bar=12,
        )

        qm.force_quarantine(
            "LiveVolCompress",
            reason="degradation_gate:session_loss_pct",
            bar=20,
        )
        result = qm.recompute(perf)

        self.assertTrue(qm.is_quarantined("LiveVolCompress"))
        self.assertTrue(result.is_no_op)
        record = qm.record_for("LiveVolCompress")
        self.assertIsNotNone(record)
        self.assertEqual(record.reason, "degradation_gate:session_loss_pct")


class TestRecomputeResult(unittest.TestCase):
    def test_no_op_predicate(self):
        r = RecomputeResult(
            added=frozenset(), removed=frozenset(), current=frozenset({"X"}),
        )
        self.assertTrue(r.is_no_op)
        r2 = RecomputeResult(
            added=frozenset({"Y"}), removed=frozenset(), current=frozenset({"X", "Y"}),
        )
        self.assertFalse(r2.is_no_op)


if __name__ == "__main__":
    unittest.main(verbosity=2)
