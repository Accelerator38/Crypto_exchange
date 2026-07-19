"""Тесты PerformanceMemory."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unittest

from panteon_v2.domain.types import Action, Metrics, Regime, Signal, Trade
from panteon_v2.memory import PerformanceMemory
from panteon_v2.scoring import regime_score


def _open_signal(sid: int, sym: str, regime: Regime, by_agent: str = "AgentA",
                 by_player: str = "PlayerP", price: float = 100.0,
                 long_side: bool = True, bar: int = 1,
                 position_scope: str = "") -> Signal:
    return Signal(
        id=sid, bar=bar, sym=sym,
        action=Action.FUT_LONG_FULL if long_side else Action.FUT_SHORT_FULL,
        price=price, regime=regime,
        by_player=by_player, by_agent=by_agent,
        position_scope=position_scope,
    )


def _close_signal(sid: int, sym: str, regime: Regime, by_agent: str = "AgentA",
                  by_player: str = "PlayerP", price: float = 100.0,
                  bar: int = 2, position_scope: str = "",
                  close_fraction: float = 1.0) -> Signal:
    return Signal(
        id=sid, bar=bar, sym=sym, action=Action.FUT_CLOSE_ALL,
        price=price, regime=regime,
        by_player=by_player, by_agent=by_agent,
        position_scope=position_scope,
        close_fraction=close_fraction,
    )


def _open_trade(sig: Signal, fill_price: float = None, qty: float = 1.0,
                fee: float = 0.0, funding: float = 0.0) -> Trade:
    side = sig.action.side or "long"
    return Trade(
        signal_id=sig.id, bar=sig.bar, sym=sig.sym, side=side,
        qty=qty, fill_price=fill_price or sig.price, fee=fee, funding=funding,
    )


def _close_trade(open_sig: Signal, close_sig: Signal, fill_price: float,
                 qty: float = 1.0, fee: float = 0.0,
                 funding: float = 0.0) -> Trade:
    return Trade(
        signal_id=close_sig.id, bar=close_sig.bar, sym=close_sig.sym,
        side=open_sig.action.side or "long",
        qty=qty, fill_price=fill_price, fee=fee, funding=funding,
    )


class TestPerformanceMemoryBasic(unittest.TestCase):
    def test_metrics_pnl_net_docstring_states_pnl_pct_is_net(self):
        doc = Metrics.pnl_net_pct.fget.__doc__ or ""
        self.assertIn("pnl_pct is already net", doc)

    def test_get_empty(self):
        perf = PerformanceMemory()
        self.assertFalse(perf.get("X").has_data)
        self.assertFalse(perf.get("X", regime=Regime.BULLISH).has_data)

    def test_invalid_trade_fraction(self):
        with self.assertRaises(ValueError):
            PerformanceMemory(trade_fraction=0)
        with self.assertRaises(ValueError):
            PerformanceMemory(trade_fraction=1.5)

    def test_invalid_max_returns_history(self):
        with self.assertRaises(ValueError):
            PerformanceMemory(max_returns_history=-1)

    def test_bounded_returns_history_keeps_last_n(self):
        # Phase 3 / C6: при max_returns_history=3 храним последние 3 returns.
        perf = PerformanceMemory(trade_fraction=1.0, max_returns_history=3)
        prices = [110.0, 95.0, 120.0, 90.0, 130.0]  # 5 закрытий
        for i, exit_price in enumerate(prices):
            os = _open_signal(2 * i + 1, "BTC", Regime.NEUTRAL, bar=2 * i + 1)
            cs = _close_signal(2 * i + 2, "BTC", Regime.NEUTRAL, bar=2 * i + 2)
            perf.update_from_trade(_open_trade(os), os)
            perf.update_from_trade(_close_trade(os, cs, exit_price), cs)
        state = perf._state[("AgentA", Regime.NEUTRAL)]
        self.assertEqual(len(state.returns), 3)
        # closed_trades счётчик не обрезается — это полная история.
        self.assertEqual(perf.get("AgentA", regime=Regime.NEUTRAL).closed_trades, 5)

    def test_unbounded_returns_history_by_default(self):
        perf = PerformanceMemory(trade_fraction=1.0)  # default 0 = безлимит
        for i in range(5):
            os = _open_signal(2 * i + 1, "BTC", Regime.NEUTRAL, bar=2 * i + 1)
            cs = _close_signal(2 * i + 2, "BTC", Regime.NEUTRAL, bar=2 * i + 2)
            perf.update_from_trade(_open_trade(os), os)
            perf.update_from_trade(_close_trade(os, cs, 110.0), cs)
        state = perf._state[("AgentA", Regime.NEUTRAL)]
        self.assertEqual(len(state.returns), 5)

    def test_recent_returns_are_ordered_and_isolated_by_player_and_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0, max_returns_history=3)
        observations = (
            (Regime.BULLISH, 110.0),
            (Regime.BEARISH, 90.0),
            (Regime.BULLISH, 120.0),
            (Regime.BULLISH, 105.0),
            (Regime.BULLISH, 130.0),
        )
        for index, (regime, exit_price) in enumerate(observations):
            open_signal = _open_signal(
                2 * index + 1,
                "BTC",
                regime,
                by_agent="",
                by_player="PlayerOnly",
                bar=2 * index + 1,
            )
            close_signal = _close_signal(
                2 * index + 2,
                "BTC",
                regime,
                by_agent="",
                by_player="PlayerOnly",
                bar=2 * index + 2,
            )
            perf.update_from_trade(_open_trade(open_signal), open_signal)
            perf.update_from_trade(
                _close_trade(open_signal, close_signal, exit_price),
                close_signal,
            )

        self.assertEqual(
            perf.recent_returns("PlayerOnly", Regime.BULLISH),
            (20.0, 5.0, 30.0),
        )
        self.assertEqual(
            perf.recent_returns("PlayerOnly", Regime.BULLISH, limit=2),
            (5.0, 30.0),
        )
        self.assertEqual(
            perf.recent_returns("PlayerOnly", Regime.BEARISH),
            (-10.0,),
        )
        aggregate = perf.recent_returns("PlayerOnly", None)
        self.assertEqual(len(aggregate), 5)
        for actual, expected in zip(aggregate, (10.0, -10.0, 20.0, 5.0, 30.0)):
            self.assertAlmostEqual(actual, expected)
        aggregate_tail = perf.recent_returns("PlayerOnly", None, limit=2)
        self.assertEqual(len(aggregate_tail), 2)
        self.assertAlmostEqual(aggregate_tail[0], 5.0)
        self.assertAlmostEqual(aggregate_tail[1], 30.0)
        self.assertEqual(perf.recent_returns("Unknown", Regime.BULLISH), ())

    def test_signal_id_mismatch_rejected(self):
        perf = PerformanceMemory()
        sig = _open_signal(1, "BTC", Regime.BULLISH)
        bad_trade = Trade(signal_id=999, bar=1, sym="BTC", side="long",
                          qty=1.0, fill_price=100.0, fee=0.0)
        with self.assertRaises(ValueError):
            perf.update_from_trade(bad_trade, sig)

    def test_actor_failure_is_visible_to_scoring_metrics(self):
        perf = PerformanceMemory()

        perf.record_execution_outcome(
            _open_signal(1, "BTC", Regime.BULLISH, by_agent="BrokenAgent", by_player="BrokenPlayer"),
            status="rejected",
            reason="exchange rejected",
        )

        metrics = perf.get("BrokenAgent", Regime.BULLISH)
        self.assertEqual(metrics.signals, 1)
        self.assertEqual(metrics.closed_trades, 0)
        self.assertEqual(metrics.losses, 0)
        self.assertAlmostEqual(metrics.pnl_pct, 0.0)
        self.assertEqual(metrics.rejected_signals, 1)
        self.assertEqual(metrics.execution_failures, 1)
        player_metrics = perf.get("BrokenPlayer", Regime.BULLISH)
        self.assertEqual(player_metrics.rejected_signals, 1)


class TestPerformanceMemoryOpenClose(unittest.TestCase):
    def test_long_winning_trade(self):
        perf = PerformanceMemory(trade_fraction=1.0)  # 100% для простоты вычислений
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        m = perf.get("AgentA", regime=Regime.BULLISH)
        # +10% при trade_fraction=1.0 → pnl_pct = 10%
        self.assertEqual(m.closed_trades, 1)
        self.assertEqual(m.entries, 1)
        self.assertEqual(m.wins, 1)
        self.assertEqual(m.losses, 0)
        self.assertAlmostEqual(m.pnl_pct, 10.0, places=5)
        self.assertEqual(m.win_rate, 100.0)

    def test_equity_curve_records_real_closed_trade_path(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        os2 = _open_signal(3, "BTC", Regime.BEARISH, price=100.0, bar=3)
        cs2 = _close_signal(4, "BTC", Regime.BEARISH, price=95.0, bar=4)

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 110.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 95.0), cs2)

        self.assertEqual(perf.equity_curve("AgentA"), [100.0, 110.0, 104.5])
        self.assertEqual(perf.equity_curve("AgentA", Regime.BULLISH), [100.0, 110.0])
        restored = PerformanceMemory(trade_fraction=1.0)
        restored.restore(perf.snapshot())
        self.assertEqual(restored.equity_curve("AgentA"), [100.0, 110.0, 104.5])

    def test_equity_curve_timestamps_record_close_times(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        ts1 = datetime(2026, 6, 6, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 6, 6, 10, 5, tzinfo=timezone.utc)
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = replace(
            _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2),
            timestamp=ts1,
        )
        os2 = _open_signal(3, "BTC", Regime.BEARISH, price=100.0, bar=3)
        cs2 = replace(
            _close_signal(4, "BTC", Regime.BEARISH, price=95.0, bar=4),
            timestamp=ts2,
        )

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 110.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 95.0), cs2)

        self.assertEqual(
            perf.equity_curve_timestamps("AgentA"),
            [ts1.isoformat(), ts1.isoformat(), ts2.isoformat()],
        )
        self.assertEqual(
            perf.equity_curve_timestamps("AgentA", Regime.BULLISH),
            [ts1.isoformat(), ts1.isoformat()],
        )
        restored = PerformanceMemory(trade_fraction=1.0)
        restored.restore(perf.snapshot())
        self.assertEqual(
            restored.equity_curve_timestamps("AgentA"),
            [ts1.isoformat(), ts1.isoformat(), ts2.isoformat()],
        )

    def test_new_regimes_seed_scoring_from_neutral_memory(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os1 = _open_signal(1, "BTC", Regime.NEUTRAL, price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.NEUTRAL, price=105.0, bar=2)
        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 105.0), cs1)

        seeded = perf.get("AgentA", Regime.RANGE_LOW_VOL)
        self.assertEqual(seeded.closed_trades, 1)
        self.assertAlmostEqual(seeded.pnl_pct, 5.0)
        top = perf.top_k_for_regime(
            Regime.CHOPPY_DOWN,
            k=1,
            scorer=regime_score,
        )
        self.assertEqual(top[0][0], "AgentA")

    def test_partial_close_keeps_open_lot_for_later_realized_pnl(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = _close_signal(
            2,
            "BTC",
            Regime.BULLISH,
            price=110.0,
            bar=2,
            close_fraction=0.25,
        )
        cs2 = _close_signal(3, "BTC", Regime.BULLISH, price=120.0, bar=3)
        perf.update_from_trade(_open_trade(os, qty=2.0), os)
        perf.update_from_trade(_close_trade(os, cs1, 110.0, qty=0.5), cs1)
        perf.update_from_trade(_close_trade(os, cs2, 120.0, qty=1.5), cs2)

        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 2)
        self.assertEqual(m.wins, 2)
        self.assertAlmostEqual(m.pnl_pct, 30.0, places=5)

    def test_default_close_all_removes_open_lot_even_when_trade_qty_is_smaller(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)

        perf.update_from_trade(_open_trade(os, qty=2.0), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0, qty=0.5), cs)

        self.assertNotIn(("", "AgentA", "BTC"), perf._open)
        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 1)
        self.assertAlmostEqual(m.pnl_pct, 10.0, places=5)

    def test_spot_sell_all_does_not_close_futures_open_lot_when_family_strict(self):
        perf = PerformanceMemory(
            trade_fraction=1.0,
            strict_close_action_family=True,
        )
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = Signal(
            id=2,
            bar=2,
            sym="BTC",
            action=Action.SPOT_SELL_ALL,
            price=110.0,
            regime=Regime.BULLISH,
            by_player="PlayerP",
            by_agent="AgentA",
        )

        perf.update_from_trade(_open_trade(os, qty=2.0), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0, qty=2.0), cs)

        self.assertIn(("", "AgentA", "BTC"), perf._open)
        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 0)
        self.assertAlmostEqual(m.pnl_pct, 0.0, places=5)

    def test_default_close_family_keeps_legacy_shadow_performance(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = Signal(
            id=2,
            bar=2,
            sym="BTC",
            action=Action.SPOT_SELL_ALL,
            price=110.0,
            regime=Regime.BULLISH,
            by_player="PlayerP",
            by_agent="AgentA",
        )

        perf.update_from_trade(_open_trade(os, qty=2.0), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0, qty=2.0), cs)

        self.assertNotIn(("", "AgentA", "BTC"), perf._open)
        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 1)
        self.assertAlmostEqual(m.pnl_pct, 10.0, places=5)

    def test_spot_sell_all_uses_trade_qty_when_smaller_than_open_lot(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = Signal(
            id=2,
            bar=2,
            sym="BTC",
            action=Action.SPOT_SELL_ALL,
            price=110.0,
            regime=Regime.BULLISH,
            by_player="PlayerP",
            by_agent="AgentA",
        )

        perf.update_from_trade(_open_trade(os, qty=2.0), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0, qty=0.5), cs)

        remaining = perf._open[("", "AgentA", "BTC")]
        self.assertAlmostEqual(remaining.qty, 1.5)
        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 1)
        self.assertAlmostEqual(m.pnl_pct, 10.0, places=5)

    def test_long_losing_trade(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.NEUTRAL, price=100.0)
        cs = _close_signal(2, "BTC", Regime.NEUTRAL, price=95.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 95.0), cs)

        m = perf.get("AgentA", regime=Regime.NEUTRAL)
        self.assertAlmostEqual(m.pnl_pct, -5.0, places=5)
        self.assertEqual(m.wins, 0)
        self.assertEqual(m.losses, 1)

    def test_short_winning_trade(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BEARISH, price=100.0, long_side=False)
        cs = _close_signal(2, "BTC", Regime.BEARISH, price=90.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 90.0), cs)

        m = perf.get("AgentA", regime=Regime.BEARISH)
        # short от 100 до 90 → +10%
        self.assertAlmostEqual(m.pnl_pct, 10.0, places=5)
        self.assertEqual(m.wins, 1)

    def test_close_without_open_records_close_only(self):
        """Если close без open — записываем close, но не PnL (orphan close)."""
        perf = PerformanceMemory()
        cs = _close_signal(1, "BTC", Regime.NEUTRAL, price=100.0)
        # Нужен открытый trade с тем же sym, но без _record_open
        # Создаём только close trade
        ct = Trade(signal_id=1, bar=2, sym="BTC", side="long",
                   qty=1.0, fill_price=100.0, fee=0.0)
        perf.update_from_trade(ct, cs)
        m = perf.get("AgentA", regime=Regime.NEUTRAL)
        self.assertEqual(m.closed_trades, 1)
        self.assertEqual(m.pnl_pct, 0.0)  # нет realized PnL

    def test_trade_fraction_scales_pnl(self):
        perf = PerformanceMemory(trade_fraction=0.10)  # 10%
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        m = perf.get("AgentA", regime=Regime.BULLISH)
        # +10% raw return × 10% fraction = 1% PnL
        self.assertAlmostEqual(m.pnl_pct, 1.0, places=5)

    def test_closed_trade_exposes_net_gross_and_cost_metrics(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)

        perf.update_from_trade(_open_trade(os, fee=0.25, funding=0.10), os)
        perf.update_from_trade(
            _close_trade(os, cs, 110.0, fee=0.25, funding=0.40),
            cs,
        )

        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertAlmostEqual(m.pnl_gross_pct, 10.0, places=5)
        self.assertAlmostEqual(m.fee_pct, 0.50, places=5)
        self.assertAlmostEqual(m.funding_pct, 0.50, places=5)
        self.assertAlmostEqual(m.trading_cost_pct, 1.00, places=5)
        self.assertAlmostEqual(m.pnl_net_pct, 9.00, places=5)
        self.assertAlmostEqual(m.pnl_pct, m.pnl_net_pct, places=5)

    def test_negative_funding_rebate_increases_net_pnl(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)

        perf.update_from_trade(_open_trade(os, funding=-0.25), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0, funding=-0.25), cs)

        m = perf.get("AgentA", regime=Regime.BULLISH)
        self.assertAlmostEqual(m.pnl_gross_pct, 10.0, places=5)
        self.assertAlmostEqual(m.funding_pct, -0.50, places=5)
        self.assertAlmostEqual(m.trading_cost_pct, -0.50, places=5)
        self.assertAlmostEqual(m.pnl_net_pct, 10.50, places=5)


    def test_cash_flat_close_is_attributed_to_original_opener(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(
            1,
            "BTC",
            Regime.BULLISH,
            by_agent="OpenerAgent",
            by_player="OpenerPlayer",
            price=100.0,
        )
        cs = _close_signal(
            2,
            "BTC",
            Regime.BULLISH,
            by_agent="CashFlat",
            by_player="NoTrade",
            price=110.0,
            bar=2,
        )

        perf.update_from_trade(_open_trade(os), os)
        perf.record_signal(cs)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        agent = perf.get("OpenerAgent", regime=Regime.BULLISH)
        player = perf.get("OpenerPlayer", regime=Regime.BULLISH)
        self.assertEqual(agent.closed_trades, 1)
        self.assertEqual(player.closed_trades, 1)
        self.assertEqual(agent.signals, 2)
        self.assertEqual(player.signals, 2)
        self.assertAlmostEqual(agent.pnl_pct, 10.0, places=5)
        self.assertAlmostEqual(player.pnl_pct, 10.0, places=5)

        self.assertFalse(perf.get("CashFlat", regime=Regime.BULLISH).has_data)
        self.assertFalse(perf.get("NoTrade", regime=Regime.BULLISH).has_data)

    def test_guard_close_is_attributed_to_original_opener(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(
            1,
            "BTC",
            Regime.BULLISH,
            by_agent="AdoptedExchangePosition",
            by_player="PanteonFlashAdopted",
            price=100.0,
        )
        cs = _close_signal(
            2,
            "BTC",
            Regime.BULLISH,
            by_agent="PartialProfitLock",
            by_player="Panteon_Flash",
            price=110.0,
            bar=2,
        )

        perf.update_from_trade(_open_trade(os), os)
        perf.record_signal(cs)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        adopted_agent = perf.get("AdoptedExchangePosition", regime=Regime.BULLISH)
        adopted_player = perf.get("PanteonFlashAdopted", regime=Regime.BULLISH)
        self.assertEqual(adopted_agent.closed_trades, 1)
        self.assertEqual(adopted_player.closed_trades, 1)
        self.assertAlmostEqual(adopted_agent.pnl_pct, 10.0, places=5)
        self.assertAlmostEqual(adopted_player.pnl_pct, 10.0, places=5)

        self.assertFalse(perf.get("PartialProfitLock", regime=Regime.BULLISH).has_data)
        self.assertFalse(perf.get("Panteon_Flash", regime=Regime.BULLISH).has_data)

    def test_actor_close_from_other_label_does_not_update_opener(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(
            1,
            "BTC",
            Regime.BEARISH,
            by_agent="OpeningAgent",
            by_player="OpeningPlayer",
            price=100.0,
            long_side=False,
        )
        cs = _close_signal(
            2,
            "BTC",
            Regime.BEARISH,
            by_agent="ClosingAgent",
            by_player="ClosingPlayer",
            price=90.0,
            bar=2,
        )

        perf.update_from_trade(_open_trade(os), os)
        perf.record_signal(cs)
        perf.update_from_trade(_close_trade(os, cs, 90.0), cs)

        opener_agent = perf.get("OpeningAgent", regime=Regime.BEARISH)
        opener_player = perf.get("OpeningPlayer", regime=Regime.BEARISH)
        closer_agent = perf.get("ClosingAgent", regime=Regime.BEARISH)
        closer_player = perf.get("ClosingPlayer", regime=Regime.BEARISH)

        self.assertEqual(opener_agent.closed_trades, 0)
        self.assertEqual(opener_player.closed_trades, 0)
        self.assertEqual(closer_agent.closed_trades, 1)
        self.assertEqual(closer_player.closed_trades, 1)
        self.assertAlmostEqual(opener_agent.pnl_pct, 0.0, places=5)
        self.assertAlmostEqual(opener_player.pnl_pct, 0.0, places=5)
        self.assertAlmostEqual(closer_agent.pnl_pct, 0.0, places=5)
        self.assertAlmostEqual(closer_player.pnl_pct, 0.0, places=5)


class TestPerRegimeIsolation(unittest.TestCase):
    def test_closed_trade_is_attributed_to_entry_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        open_signal = _open_signal(
            1,
            "BTC",
            Regime.BULLISH,
            by_agent="",
            by_player="EntryRegimePlayer",
            price=100.0,
        )
        close_signal = _close_signal(
            2,
            "BTC",
            Regime.BEARISH,
            by_agent="",
            by_player="EntryRegimePlayer",
            price=110.0,
        )

        perf.update_from_trade(_open_trade(open_signal), open_signal)
        perf.update_from_trade(
            _close_trade(open_signal, close_signal, 110.0),
            close_signal,
        )

        bullish = perf.get("EntryRegimePlayer", regime=Regime.BULLISH)
        bearish = perf.get("EntryRegimePlayer", regime=Regime.BEARISH)
        self.assertEqual(bullish.closed_trades, 1)
        self.assertAlmostEqual(bullish.pnl_pct, 10.0, places=5)
        self.assertEqual(bearish.closed_trades, 0)
        self.assertEqual(
            perf.recent_returns("EntryRegimePlayer", Regime.BULLISH),
            (10.0,),
        )
        self.assertEqual(
            perf.recent_returns("EntryRegimePlayer", Regime.BEARISH),
            (),
        )

    def test_regimes_isolated(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        # Прибыль в bullish
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0, bar=1)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        # Потеря в bearish
        os2 = _open_signal(3, "ETH", Regime.BEARISH, price=100.0, bar=3)
        cs2 = _close_signal(4, "ETH", Regime.BEARISH, price=95.0, bar=4)

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 110.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 95.0), cs2)

        m_bull = perf.get("AgentA", regime=Regime.BULLISH)
        m_bear = perf.get("AgentA", regime=Regime.BEARISH)
        self.assertAlmostEqual(m_bull.pnl_pct, 10.0, places=5)
        self.assertAlmostEqual(m_bear.pnl_pct, -5.0, places=5)

    def test_aggregate_sums(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH, price=105.0, bar=2)
        os2 = _open_signal(3, "ETH", Regime.NEUTRAL, price=200.0, bar=3)
        cs2 = _close_signal(4, "ETH", Regime.NEUTRAL, price=210.0, bar=4)

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 105.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 210.0), cs2)

        agg = perf.get("AgentA")  # без regime → агрегат
        # 5% + 5% = 10%
        self.assertAlmostEqual(agg.pnl_pct, 10.0, places=5)
        self.assertEqual(agg.closed_trades, 2)
        self.assertEqual(agg.wins, 2)

    def test_aggregate_get_is_cached_until_label_changes(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH, price=105.0, bar=2)
        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 105.0), cs1)

        calls = 0
        original = perf._aggregate_metrics

        def wrapped(label):
            nonlocal calls
            calls += 1
            return original(label)

        perf._aggregate_metrics = wrapped

        self.assertAlmostEqual(perf.get("AgentA").pnl_pct, 5.0, places=5)
        self.assertAlmostEqual(perf.get("AgentA").pnl_pct, 5.0, places=5)
        self.assertEqual(calls, 1)

        os2 = _open_signal(3, "ETH", Regime.BEARISH, price=100.0, bar=3)
        cs2 = _close_signal(4, "ETH", Regime.BEARISH, price=90.0, bar=4)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 90.0), cs2)

        self.assertAlmostEqual(perf.get("AgentA").pnl_pct, -5.0, places=5)
        self.assertEqual(calls, 2)


class TestContextScopedMemory(unittest.TestCase):
    def test_symbol_regime_and_exchange_scopes_are_isolated(self):
        perf = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
        btc_open = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        btc_close = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        eth_open = _open_signal(3, "ETH", Regime.BULLISH, price=100.0, bar=3)
        eth_close = _close_signal(4, "ETH", Regime.BULLISH, price=95.0, bar=4)
        neutral_open = _open_signal(5, "BTC", Regime.NEUTRAL, price=100.0, bar=5)
        neutral_close = _close_signal(6, "BTC", Regime.NEUTRAL, price=102.0, bar=6)

        perf.update_from_trade(_open_trade(btc_open), btc_open)
        perf.update_from_trade(_close_trade(btc_open, btc_close, 110.0), btc_close)
        perf.update_from_trade(_open_trade(eth_open), eth_open)
        perf.update_from_trade(_close_trade(eth_open, eth_close, 95.0), eth_close)
        perf.update_from_trade(_open_trade(neutral_open), neutral_open)
        perf.update_from_trade(_close_trade(neutral_open, neutral_close, 102.0), neutral_close)

        self.assertAlmostEqual(
            perf.get("AgentA", Regime.BULLISH, symbol="BTC").pnl_pct,
            10.0,
            places=5,
        )
        self.assertAlmostEqual(
            perf.get("AgentA", Regime.BULLISH, symbol="ETH").pnl_pct,
            -5.0,
            places=5,
        )
        self.assertAlmostEqual(
            perf.get("AgentA", Regime.NEUTRAL, symbol="BTC").pnl_pct,
            2.0,
            places=5,
        )
        self.assertFalse(
            perf.get("AgentA", Regime.BULLISH, exchange="BITGET", symbol="BTC").has_data
        )

    def test_legacy_regime_memory_transfers_once_as_symbol_initial_state(self):
        legacy = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "OLD", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "OLD", Regime.BULLISH, price=110.0, bar=2)
        legacy.update_from_trade(_open_trade(os), os)
        legacy.update_from_trade(_close_trade(os, cs, 110.0), cs)
        snapshot = legacy.snapshot()
        snapshot.pop("context_state", None)

        perf = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
        perf.restore(snapshot)
        btc_initial = perf.get("AgentA", Regime.BULLISH, symbol="BTC")
        eth_initial = perf.get("AgentA", Regime.BULLISH, symbol="ETH")
        self.assertAlmostEqual(btc_initial.pnl_pct, 10.0, places=5)
        self.assertAlmostEqual(eth_initial.pnl_pct, 10.0, places=5)

        btc_open = _open_signal(3, "BTC", Regime.BULLISH, price=100.0, bar=3)
        btc_close = _close_signal(4, "BTC", Regime.BULLISH, price=90.0, bar=4)
        perf.update_from_trade(_open_trade(btc_open), btc_open)
        perf.update_from_trade(_close_trade(btc_open, btc_close, 90.0), btc_close)

        self.assertAlmostEqual(
            perf.get("AgentA", Regime.BULLISH, symbol="BTC").pnl_pct,
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            perf.get("AgentA", Regime.BULLISH, symbol="ETH").pnl_pct,
            10.0,
            places=5,
        )

    def test_context_state_roundtrip(self):
        perf1 = PerformanceMemory(trade_fraction=1.0, exchange_scope="MEXC")
        os = _open_signal(1, "BTC", Regime.BEARISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BEARISH, price=90.0, bar=2)
        perf1.update_from_trade(_open_trade(os), os)
        perf1.update_from_trade(_close_trade(os, cs, 90.0), cs)

        perf2 = PerformanceMemory()
        perf2.restore(perf1.snapshot())

        self.assertAlmostEqual(
            perf2.get("AgentA", Regime.BEARISH, exchange="MEXC", symbol="BTC").pnl_pct,
            -10.0,
            places=5,
        )


class TestPlayerAndAgentSeparately(unittest.TestCase):
    def test_both_label_levels_recorded(self):
        """update_from_trade обновляет обоих by_agent и by_player."""
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH,
                          by_agent="LiveAfterShock", by_player="PanteonResearch",
                          price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH,
                           by_agent="LiveAfterShock", by_player="PanteonResearch",
                           price=110.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        m_a = perf.get("LiveAfterShock", regime=Regime.BULLISH)
        m_p = perf.get("PanteonResearch", regime=Regime.BULLISH)
        self.assertAlmostEqual(m_a.pnl_pct, 10.0, places=5)
        self.assertAlmostEqual(m_p.pnl_pct, 10.0, places=5)

    def test_same_agent_and_player_not_double_counted(self):
        """Если by_agent == by_player → счётчик инкремент один раз."""
        perf = PerformanceMemory(trade_fraction=1.0)
        os = _open_signal(1, "BTC", Regime.BULLISH,
                          by_agent="X", by_player="X", price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH,
                           by_agent="X", by_player="X", price=110.0, bar=2)
        perf.update_from_trade(_open_trade(os), os)
        perf.update_from_trade(_close_trade(os, cs, 110.0), cs)

        m = perf.get("X", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 1)
        self.assertEqual(m.entries, 1)

    def test_position_scope_keeps_virtual_and_real_positions_independent(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        virtual_open = _open_signal(
            1, "BTC", Regime.BULLISH,
            by_agent="", by_player="Leader",
            price=100.0, position_scope="shadow:Leader",
        )
        real_open = _open_signal(
            2, "BTC", Regime.BULLISH,
            by_agent="", by_player="Leader",
            price=200.0,
        )
        virtual_close = _close_signal(
            3, "BTC", Regime.BULLISH,
            by_agent="", by_player="Leader",
            price=110.0, bar=3, position_scope="shadow:Leader",
        )
        real_close = _close_signal(
            4, "BTC", Regime.BULLISH,
            by_agent="", by_player="Leader",
            price=210.0, bar=4,
        )

        perf.update_from_trade(_open_trade(virtual_open), virtual_open)
        perf.update_from_trade(_open_trade(real_open), real_open)
        perf.update_from_trade(_close_trade(virtual_open, virtual_close, 110.0), virtual_close)
        perf.update_from_trade(_close_trade(real_open, real_close, 210.0), real_close)

        metrics = perf.get("Leader", regime=Regime.BULLISH)
        self.assertEqual(metrics.closed_trades, 2)
        self.assertAlmostEqual(metrics.pnl_pct, 15.0, places=5)


class TestPerRegimeAndTopK(unittest.TestCase):
    def test_per_regime_for_label(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        os1 = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH, price=105.0, bar=2)
        os2 = _open_signal(3, "ETH", Regime.NEUTRAL, price=100.0, bar=3)
        cs2 = _close_signal(4, "ETH", Regime.NEUTRAL, price=98.0, bar=4)

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 105.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 98.0), cs2)

        per_regime = perf.per_regime_for_label("AgentA")
        self.assertIn(Regime.BULLISH, per_regime)
        self.assertIn(Regime.NEUTRAL, per_regime)
        self.assertAlmostEqual(per_regime[Regime.BULLISH].pnl_pct, 5.0, places=5)
        self.assertAlmostEqual(per_regime[Regime.NEUTRAL].pnl_pct, -2.0, places=5)

    def test_top_k_for_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        # A: +5%, B: -3%, C: +1%
        scenarios = [
            ("AgentA", 100, 105),
            ("AgentB", 100, 97),
            ("AgentC", 100, 101),
        ]
        sid = 1
        for label, op_p, cl_p in scenarios:
            os = Signal(id=sid, bar=1, sym=f"S{sid}",
                        action=Action.FUT_LONG_FULL, price=op_p,
                        regime=Regime.BULLISH, by_player=label, by_agent=label)
            cs = Signal(id=sid + 1, bar=2, sym=f"S{sid}",
                        action=Action.FUT_CLOSE_ALL, price=cl_p,
                        regime=Regime.BULLISH, by_player=label, by_agent=label)
            perf.update_from_trade(_open_trade(os), os)
            perf.update_from_trade(_close_trade(os, cs, cl_p), cs)
            sid += 2

        top = perf.top_k_for_regime(Regime.BULLISH, k=3, scorer=regime_score)
        # Должны быть отсортированы по убыванию score
        labels = [t[0] for t in top]
        self.assertEqual(labels[0], "AgentA")
        self.assertIn("AgentC", labels)
        self.assertIn("AgentB", labels)
        # AgentB должен быть последним (отрицательный pnl)
        self.assertEqual(labels[-1], "AgentB")

    def test_top_k_excludes(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        for label, op_p, cl_p in [("A", 100, 105), ("B", 100, 110)]:
            os = Signal(id=hash(label) % 1000, bar=1, sym="X",
                        action=Action.FUT_LONG_FULL, price=op_p,
                        regime=Regime.BULLISH, by_player=label, by_agent=label)
            cs = Signal(id=hash(label) % 1000 + 1, bar=2, sym="X",
                        action=Action.FUT_CLOSE_ALL, price=cl_p,
                        regime=Regime.BULLISH, by_player=label, by_agent=label)
            perf.update_from_trade(_open_trade(os), os)
            perf.update_from_trade(_close_trade(os, cs, cl_p), cs)

        top = perf.top_k_for_regime(Regime.BULLISH, k=5, scorer=regime_score,
                                    exclude={"B"})
        labels = [t[0] for t in top]
        self.assertNotIn("B", labels)
        self.assertIn("A", labels)


class TestSnapshotRestore(unittest.TestCase):
    def test_roundtrip(self):
        perf1 = PerformanceMemory(trade_fraction=0.10)
        os = _open_signal(1, "BTC", Regime.BULLISH, price=100.0)
        cs = _close_signal(2, "BTC", Regime.BULLISH, price=110.0, bar=2)
        perf1.update_from_trade(_open_trade(os), os)
        perf1.update_from_trade(_close_trade(os, cs, 110.0), cs)

        snap = perf1.snapshot()
        perf2 = PerformanceMemory()
        perf2.restore(snap)

        m1 = perf1.get("AgentA", regime=Regime.BULLISH)
        m2 = perf2.get("AgentA", regime=Regime.BULLISH)
        self.assertAlmostEqual(m1.pnl_pct, m2.pnl_pct, places=8)
        self.assertEqual(m1.closed_trades, m2.closed_trades)
        self.assertEqual(m1.entries, m2.entries)


class TestAllLabels(unittest.TestCase):
    def test_all_labels_distinct(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        # AgentA + PlayerP (одна сделка), AgentB + PlayerP (другая)
        os1 = _open_signal(1, "BTC", Regime.BULLISH,
                           by_agent="AgentA", by_player="PlayerP", price=100.0)
        cs1 = _close_signal(2, "BTC", Regime.BULLISH,
                            by_agent="AgentA", by_player="PlayerP", price=105.0, bar=2)
        os2 = _open_signal(3, "ETH", Regime.NEUTRAL,
                           by_agent="AgentB", by_player="PlayerP", price=100.0, bar=3)
        cs2 = _close_signal(4, "ETH", Regime.NEUTRAL,
                            by_agent="AgentB", by_player="PlayerP", price=110.0, bar=4)

        perf.update_from_trade(_open_trade(os1), os1)
        perf.update_from_trade(_close_trade(os1, cs1, 105.0), cs1)
        perf.update_from_trade(_open_trade(os2), os2)
        perf.update_from_trade(_close_trade(os2, cs2, 110.0), cs2)

        labels = perf.all_labels()
        self.assertEqual(set(labels), {"AgentA", "AgentB", "PlayerP"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
