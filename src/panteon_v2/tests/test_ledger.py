"""Тесты AttributionLedger.

Цель: зафиксировать гарантию Q2 — сумма AttributionLedger.player_pnl ==
sum(realized_pnl from PositionClosed events).
Плюс: replay детерминирован, идемпотентен.
"""

from __future__ import annotations

import unittest

from panteon_v2.attribution import (
    AttributionLedger,
    BarStarted,
    EventLog,
    OrderFilled,
    PositionClosed,
    PositionOpened,
)


def _open_event(bar: int, sym: str, signal_id: int,
                side: str = "long", entry: float = 100.0,
                qty: float = 1.0, trace_id: str = "") -> PositionOpened:
    return PositionOpened(
        bar=bar, trace_id=trace_id or f"{sym}-{bar}",
        signal_id=signal_id, sym=sym, side=side, entry=entry, qty=qty,
    )


def _close_event(bar: int, sym: str, open_signal_id: int, close_signal_id: int,
                 side: str = "long", entry: float = 100.0, exit_: float = 110.0,
                 qty: float = 1.0, realized_pnl: float = 10.0,
                 by_player: str = "P", by_agent: str = "A",
                 trace_id: str = "") -> PositionClosed:
    return PositionClosed(
        bar=bar, trace_id=trace_id or f"{sym}-{bar}",
        open_signal_id=open_signal_id, close_signal_id=close_signal_id,
        sym=sym, side=side, entry=entry, exit=exit_, qty=qty,
        realized_pnl=realized_pnl, by_player=by_player, by_agent=by_agent,
    )


class TestEmptyReplay(unittest.TestCase):
    def test_empty_log(self):
        ledger = AttributionLedger()
        ledger.replay_from_event_log(EventLog())
        self.assertEqual(ledger.total_pnl_by_player(), {})
        self.assertEqual(ledger.total_pnl_by_agent(), {})
        self.assertEqual(ledger.total_realized_pnl, 0.0)
        self.assertEqual(ledger.closed_count, 0)
        self.assertEqual(ledger.open_count, 0)


class TestSingleClose(unittest.TestCase):
    def test_one_open_one_close(self):
        log = EventLog()
        log.emit(_open_event(bar=1, sym="BTC", signal_id=1))
        log.emit(_close_event(bar=2, sym="BTC", open_signal_id=1,
                              close_signal_id=2, realized_pnl=15.0,
                              by_player="Alpha", by_agent="X"))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.closed_count, 1)
        self.assertEqual(ledger.open_count, 0)
        self.assertAlmostEqual(ledger.total_realized_pnl, 15.0)
        self.assertEqual(ledger.total_pnl_by_player(), {"Alpha": 15.0})
        self.assertEqual(ledger.total_pnl_by_agent(), {"X": 15.0})

    def test_open_without_close(self):
        log = EventLog()
        log.emit(_open_event(bar=1, sym="BTC", signal_id=1))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.closed_count, 0)
        self.assertEqual(ledger.open_count, 1)
        self.assertAlmostEqual(ledger.total_realized_pnl, 0.0)


class TestQ2GuaranteeSumEqualsRealized(unittest.TestCase):
    """Гарантия Q2: сумма attribution = sum realized_pnl."""

    def test_multiple_players_sum_equals_total(self):
        log = EventLog()
        scenarios = [
            ("BTC",   1,  2, "Alpha", "AgA", +5.0),
            ("ETH",   3,  4, "Beta",  "AgB", -3.0),
            ("DOGE",  5,  6, "Alpha", "AgA", +2.0),
            ("SOL",   7,  8, "Gamma", "AgC", -1.5),
            ("ADA",   9, 10, "Beta",  "AgB", +0.5),
        ]
        for sym, op_id, cl_id, player, agent, pnl in scenarios:
            log.emit(_open_event(op_id, sym, op_id))
            log.emit(_close_event(cl_id, sym, op_id, cl_id,
                                  realized_pnl=pnl, by_player=player,
                                  by_agent=agent))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        # По каждому игроку
        by_player = ledger.total_pnl_by_player()
        # Alpha: 5 + 2 = 7
        # Beta: -3 + 0.5 = -2.5
        # Gamma: -1.5
        self.assertAlmostEqual(by_player["Alpha"], 7.0)
        self.assertAlmostEqual(by_player["Beta"], -2.5)
        self.assertAlmostEqual(by_player["Gamma"], -1.5)
        # Сумма по players == total realized
        self.assertAlmostEqual(
            sum(by_player.values()),
            ledger.total_realized_pnl,
            places=8,
        )
        # Q2 формальная: общая сумма
        expected = sum(pnl for _, _, _, _, _, pnl in scenarios)
        self.assertAlmostEqual(ledger.total_realized_pnl, expected)

    def test_consistency_check_passes(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        log.emit(_close_event(2, "BTC", 1, 2, realized_pnl=5.0))
        log.emit(_open_event(3, "ETH", 3))
        log.emit(_close_event(4, "ETH", 3, 4, realized_pnl=-2.0))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertTrue(ledger.consistency_check())


class TestReplayDeterministic(unittest.TestCase):
    """INV-5: replay одного и того же лога даёт одинаковый результат."""

    def setUp(self):
        self.log = EventLog()
        for sid in (1, 3, 5):
            self.log.emit(_open_event(sid, f"S{sid}", sid))
            self.log.emit(_close_event(sid + 1, f"S{sid}", sid, sid + 1,
                                       realized_pnl=sid * 1.5,
                                       by_player=f"P{sid}",
                                       by_agent=f"A{sid}"))

    def test_two_replays_identical(self):
        l1 = AttributionLedger()
        l1.replay_from_event_log(self.log)
        l2 = AttributionLedger()
        l2.replay_from_event_log(self.log)
        self.assertEqual(l1.total_pnl_by_player(), l2.total_pnl_by_player())
        self.assertEqual(l1.total_pnl_by_agent(), l2.total_pnl_by_agent())
        self.assertEqual(l1.closed_count, l2.closed_count)

    def test_replay_idempotent(self):
        """Вызов replay дважды на одном инстансе даёт тот же результат."""
        ledger = AttributionLedger()
        ledger.replay_from_event_log(self.log)
        snapshot1 = (ledger.total_pnl_by_player(), ledger.total_realized_pnl)
        ledger.replay_from_event_log(self.log)
        snapshot2 = (ledger.total_pnl_by_player(), ledger.total_realized_pnl)
        self.assertEqual(snapshot1, snapshot2)


class TestUpToBar(unittest.TestCase):
    def test_filter_by_bar(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        log.emit(_close_event(2, "BTC", 1, 2, realized_pnl=5.0,
                              by_player="P1"))
        log.emit(_open_event(10, "ETH", 10))
        log.emit(_close_event(11, "ETH", 10, 11, realized_pnl=3.0,
                              by_player="P2"))
        # Только до бара 5 → видим только BTC закрытие
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log, up_to_bar=5)
        self.assertAlmostEqual(ledger.total_realized_pnl, 5.0)
        self.assertEqual(set(ledger.total_pnl_by_player().keys()), {"P1"})

        # Полный replay → всё
        ledger.replay_from_event_log(log)
        self.assertAlmostEqual(ledger.total_realized_pnl, 8.0)
        self.assertEqual(set(ledger.total_pnl_by_player().keys()), {"P1", "P2"})


class TestFiltering(unittest.TestCase):
    def setUp(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        log.emit(_close_event(2, "BTC", 1, 2, realized_pnl=10.0,
                              by_player="Alpha", by_agent="X"))
        log.emit(_open_event(3, "BTC", 3))
        log.emit(_close_event(4, "BTC", 3, 4, realized_pnl=-5.0,
                              by_player="Beta", by_agent="Y"))
        log.emit(_open_event(5, "ETH", 5))
        log.emit(_close_event(6, "ETH", 5, 6, realized_pnl=2.0,
                              by_player="Alpha", by_agent="Z"))
        self.log = log

    def test_filter_by_sym(self):
        ledger = AttributionLedger()
        ledger.replay_from_event_log(self.log)
        btc = ledger.realized_attributions(sym="BTC")
        self.assertEqual(len(btc), 2)
        eth = ledger.realized_attributions(sym="ETH")
        self.assertEqual(len(eth), 1)

    def test_filter_by_player(self):
        ledger = AttributionLedger()
        ledger.replay_from_event_log(self.log)
        alpha = ledger.realized_attributions(player="Alpha")
        self.assertEqual(len(alpha), 2)
        beta = ledger.realized_attributions(player="Beta")
        self.assertEqual(len(beta), 1)

    def test_filter_by_agent(self):
        ledger = AttributionLedger()
        ledger.replay_from_event_log(self.log)
        x = ledger.realized_attributions(agent="X")
        self.assertEqual(len(x), 1)


class TestPnLByWindow(unittest.TestCase):
    def test_window_filter(self):
        log = EventLog()
        for sid in (1, 3, 5, 7):
            log.emit(_open_event(sid, f"S{sid}", sid))
            log.emit(_close_event(sid + 1, f"S{sid}", sid, sid + 1,
                                  realized_pnl=1.0, by_player="P"))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        # closed events на барах 2, 4, 6, 8 → window [3, 7) → бары 4, 6
        win = ledger.pnl_by_player_in_window(from_bar=3, to_bar=7)
        self.assertAlmostEqual(win.get("P", 0.0), 2.0)


class TestCounters(unittest.TestCase):
    def test_trade_counts(self):
        log = EventLog()
        for sid in (1, 3, 5):
            log.emit(_open_event(sid, "BTC", sid))
            log.emit(_close_event(sid + 1, "BTC", sid, sid + 1,
                                  realized_pnl=1.0 if sid != 3 else -1.0,
                                  by_player="P"))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.trade_counts_by_player(), {"P": 3})
        self.assertEqual(ledger.win_counts_by_player(), {"P": 2})

    def test_trade_counts_by_agent(self):
        log = EventLog()
        for sid, agent in ((1, "A"), (3, "A"), (5, "B")):
            log.emit(_open_event(sid, "BTC", sid))
            log.emit(_close_event(sid + 1, "BTC", sid, sid + 1,
                                  realized_pnl=1.0 if sid != 3 else -1.0,
                                  by_player="P", by_agent=agent))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.trade_counts_by_agent(), {"A": 2, "B": 1})
        self.assertEqual(ledger.win_counts_by_agent(), {"A": 1, "B": 1})


class TestOpenAttributions(unittest.TestCase):
    def test_open_position_present(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.open_count, 1)
        self.assertIsNotNone(ledger.open_attribution_for_sym("BTC"))
        self.assertIsNone(ledger.open_attribution_for_sym("ETH"))

    def test_close_removes_open(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        log.emit(_close_event(2, "BTC", 1, 2, realized_pnl=5.0))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        self.assertEqual(ledger.open_count, 0)


class TestUnattributedPnL(unittest.TestCase):
    def test_close_without_player_still_counted(self):
        log = EventLog()
        log.emit(_open_event(1, "BTC", 1))
        log.emit(_close_event(2, "BTC", 1, 2, realized_pnl=5.0,
                              by_player="", by_agent=""))
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        # total_realized_pnl всё равно учитывает
        self.assertAlmostEqual(ledger.total_realized_pnl, 5.0)
        # А by_player пуст
        self.assertEqual(ledger.total_pnl_by_player(), {})
        # consistency_check тоже OK (учитывает unattributed)
        self.assertTrue(ledger.consistency_check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
