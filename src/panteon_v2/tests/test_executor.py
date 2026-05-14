"""Тесты TradeExecutor — полный pipeline signal → events."""

from __future__ import annotations

import unittest

from panteon_v2.attribution import (
    EventLog,
    MemoryUpdateFailed,
    OrderFilled,
    OrderRejected,
    OrderSent,
    PositionClosed,
    PositionOpened,
    SymbolBlocked,
)
from panteon_v2.domain.types import Action, Regime, Signal
from panteon_v2.execution import (
    ExecutionStatus,
    FakeExchange,
    OrderResult,
    OrderStatus,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from panteon_v2.memory import PerformanceMemory


def _make_signal(sid: int = 1, sym: str = "BTC",
                 action: Action = Action.FUT_LONG_FULL,
                 price: float = 100.0) -> Signal:
    return Signal(
        id=sid, bar=1, sym=sym, action=action, price=price,
        regime=Regime.BULLISH, by_player="P", by_agent="A",
    )


def _make_executor():
    return {
        "exchange":         FakeExchange(),
        "health":           SymbolHealthMonitor(),
        "risk_limits":      RiskLimits(),
        "position_tracker": PositionTracker(),
        "perf":             PerformanceMemory(trade_fraction=1.0),
        "event_log":        EventLog(),
    }


class TestExecuteSuccess(unittest.TestCase):
    def setUp(self):
        self.deps = _make_executor()
        self.exec = TradeExecutor(**self.deps)

    def test_simple_open(self):
        sig = _make_signal()
        result = self.exec.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.FILLED)
        self.assertIsNotNone(result.trade)
        # Position учтена
        self.assertTrue(self.deps["position_tracker"].has("BTC"))
        # PerfMemory обновился
        agg = self.deps["perf"].get("A")
        self.assertEqual(agg.entries, 1)
        # Events логированы
        log = self.deps["event_log"]
        events_sent = list(log.query(event_types=[OrderSent]))
        events_filled = list(log.query(event_types=[OrderFilled]))
        events_opened = list(log.query(event_types=[PositionOpened]))
        self.assertEqual(len(events_sent), 1)
        self.assertEqual(len(events_filled), 1)
        self.assertEqual(len(events_opened), 1)

    def test_open_then_close(self):
        op_sig = _make_signal(sid=1)
        cl_sig = _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=110.0)
        self.exec.execute(op_sig, balance_usd=1000.0)
        result = self.exec.execute(cl_sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.FILLED)
        self.assertFalse(self.deps["position_tracker"].has("BTC"))
        # PositionClosed event с realized_pnl
        closed = list(self.deps["event_log"].query(event_types=[PositionClosed]))
        self.assertEqual(len(closed), 1)
        self.assertGreater(closed[0].realized_pnl, 0)

    def test_signal_id_in_trade(self):
        """Гарантия: Trade.signal_id == Signal.id."""
        sig = _make_signal(sid=42)
        result = self.exec.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.trade.signal_id, 42)


class TestExecuteBlocked(unittest.TestCase):
    def setUp(self):
        self.deps = _make_executor()
        self.exec = TradeExecutor(**self.deps)

    def test_hold_blocked(self):
        sig = _make_signal(action=Action.HOLD)
        result = self.exec.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertIn("hold", result.reason)

    def test_health_block_stops_open(self):
        # Заранее блокируем символ через SymbolHealthMonitor
        for _ in range(3):
            self.deps["health"].record_pending_failure("BTC")
        self.assertTrue(self.deps["health"].is_blocked("BTC"))
        sig = _make_signal()
        result = self.exec.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertIn("blocklist", result.reason)
        # Биржа НЕ вызывалась
        self.assertEqual(len(self.deps["exchange"].orders_log), 0)

    def test_close_not_blocked_by_health(self):
        """Health-block не должен блокировать close (можно закрыть всегда)."""
        # Открываем позицию
        op_sig = _make_signal(sid=1)
        self.exec.execute(op_sig, balance_usd=1000.0)
        # Блокируем символ
        for _ in range(3):
            self.deps["health"].record_pending_failure("BTC")
        # Close должен пройти
        cl_sig = _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=110.0)
        result = self.exec.execute(cl_sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.FILLED)

    def test_risk_limit_blocked(self):
        # Создаём с очень малым балансом — notional ниже min
        sig = _make_signal()
        result = self.exec.execute(sig, balance_usd=1.0)
        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertIn("risk_limits", result.reason)


class TestExecuteRejected(unittest.TestCase):
    def setUp(self):
        self.deps = _make_executor()
        self.exec = TradeExecutor(**self.deps)

    def test_reject_logs_OrderRejected(self):
        self.deps["exchange"].configure_reject("BTC", count=1)
        sig = _make_signal()
        result = self.exec.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.REJECTED)
        rejected = list(self.deps["event_log"].query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)

    def test_reject_increments_health(self):
        self.deps["exchange"].configure_reject("BTC", count=2)
        for sid in (1, 2):
            self.exec.execute(_make_signal(sid=sid), balance_usd=1000.0)
        st = self.deps["health"].status("BTC")
        self.assertEqual(st.failures_in_window, 2)
        self.assertFalse(st.is_blocked)

    def test_third_reject_triggers_block_event(self):
        self.deps["exchange"].configure_reject("BTC", count=3)
        for sid in (1, 2, 3):
            self.exec.execute(_make_signal(sid=sid), balance_usd=1000.0)
        # Должен быть один SymbolBlocked event
        blocked = list(self.deps["event_log"].query(event_types=[SymbolBlocked]))
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].sym, "BTC")

    def test_exchange_exception_handled(self):
        class CrashExchange(FakeExchange):
            def send_order(self, signal, *, qty):
                raise RuntimeError("API down")
        deps = self.deps.copy()
        deps["exchange"] = CrashExchange()
        ex = TradeExecutor(**deps)
        sig = _make_signal()
        result = ex.execute(sig, balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.REJECTED)
        rejected = list(deps["event_log"].query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)
        self.assertIn("API down", rejected[0].reason)


class TestExecutePending(unittest.TestCase):
    def setUp(self):
        self.deps = _make_executor()
        self.exec = TradeExecutor(**self.deps)

    def test_pending_increments_failure(self):
        self.deps["exchange"].configure_pending("BTC")
        result = self.exec.execute(_make_signal(), balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.PENDING)
        self.assertIsNone(result.trade)
        st = self.deps["health"].status("BTC")
        self.assertEqual(st.failures_in_window, 1)

    def test_pending_order_is_polled_and_accounted_when_fill_is_available(self):
        class PollingExchange(FakeExchange):
            def __init__(self):
                super().__init__()
                self.last_qty = 0.0
                self.polled = []

            def send_order(self, signal, *, qty):
                self.last_qty = qty
                return OrderResult(
                    status=OrderStatus.PENDING,
                    signal_id=signal.id,
                    sym=signal.sym,
                    exchange_order_id="PENDING-1",
                    message="accepted",
                )

            def poll_order(self, order_id, signal):
                self.polled.append(order_id)
                return FakeExchange.send_order(self, signal, qty=self.last_qty)

        deps = _make_executor()
        deps["exchange"] = PollingExchange()
        ex = TradeExecutor(**deps)

        result = ex.execute(_make_signal(), balance_usd=1000.0)

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        self.assertEqual(deps["exchange"].polled, ["PENDING-1"])
        self.assertTrue(deps["position_tracker"].has("BTC"))
        self.assertEqual(deps["perf"].get("A").entries, 1)


class TestExecuteRiskQuantization(unittest.TestCase):
    def test_executor_blocks_exchange_quantized_qty_above_approved_risk_notional(self):
        class MinLotExchange(FakeExchange):
            def __init__(self):
                super().__init__()
                self.sent = 0

            def quantize_order_qty(self, signal, qty):
                return 0.01

            def send_order(self, signal, *, qty):
                self.sent += 1
                return super().send_order(signal, qty=qty)

        deps = _make_executor()
        deps["exchange"] = MinLotExchange()
        deps["risk_limits"] = RiskLimits(config=RiskLimitsConfig(
            capital_fraction=0.10,
            min_notional_usd=0.0,
            max_notional_usd=5.0,
        ))
        ex = TradeExecutor(**deps)
        sig = _make_signal(price=1000.0)

        result = ex.execute(sig, balance_usd=50.0)

        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertIn("quantized notional", result.reason)
        self.assertEqual(deps["exchange"].sent, 0)

    def test_executor_allows_exchange_quantized_qty_inside_upscale_limit(self):
        class MinLotExchange(FakeExchange):
            def __init__(self):
                super().__init__()
                self.sent_qty = 0.0

            def quantize_order_qty(self, signal, qty):
                return 0.01

            def send_order(self, signal, *, qty):
                self.sent_qty = qty
                return super().send_order(signal, qty=qty)

        deps = _make_executor()
        deps["exchange"] = MinLotExchange()
        deps["risk_limits"] = RiskLimits(config=RiskLimitsConfig(
            capital_fraction=0.10,
            min_notional_usd=0.0,
            max_notional_usd=20.0,
            max_min_notional_upscale=2.0,
        ))
        ex = TradeExecutor(**deps)
        sig = _make_signal(price=813.0)

        result = ex.execute(sig, balance_usd=51.0)

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        self.assertEqual(deps["exchange"].sent_qty, 0.01)


class TestExecuteMemoryUpdateFailure(unittest.TestCase):
    def test_memory_update_failure_is_emitted_after_filled_trade(self):
        class BrokenMemory(PerformanceMemory):
            def update_from_trade(self, trade, signal):
                raise RuntimeError("memory disk full")

        deps = _make_executor()
        deps["perf"] = BrokenMemory(trade_fraction=1.0)
        ex = TradeExecutor(**deps)

        result = ex.execute(_make_signal(), balance_usd=1000.0)

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        failures = list(deps["event_log"].query(event_types=[MemoryUpdateFailed]))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].signal_id, 1)
        self.assertIn("memory disk full", failures[0].reason)


class TestExecuteSuccessClearsHealth(unittest.TestCase):
    def test_success_resets_failures(self):
        deps = _make_executor()
        ex = TradeExecutor(**deps)
        # 2 reject подряд
        deps["exchange"].configure_reject("BTC", count=2)
        ex.execute(_make_signal(sid=1), balance_usd=1000.0)
        ex.execute(_make_signal(sid=2), balance_usd=1000.0)
        st_before = deps["health"].status("BTC")
        self.assertEqual(st_before.failures_in_window, 2)
        # Третий — успех
        result = ex.execute(_make_signal(sid=3), balance_usd=1000.0)
        self.assertEqual(result.status, ExecutionStatus.FILLED)
        # Failures сбросились
        st_after = deps["health"].status("BTC")
        self.assertEqual(st_after.failures_in_window, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
