"""Тесты TradeExecutor — полный pipeline signal → events."""

from __future__ import annotations

import unittest
import json
import os
import tempfile
from dataclasses import fields

from panteon_v2.attribution import (
    EventLog,
    ExecutionAttributed,
    MemoryUpdateFailed,
    OrderFilled,
    OrderRejected,
    OrderSent,
    PositionClosed,
    PositionOpened,
    SignalEmitted,
    SymbolBlocked,
)
from panteon_v2.domain.types import Action, Regime, Signal
from panteon_v2.execution import (
    ExecutionStatus,
    FakeExchange,
    OrderLedger,
    OrderResult,
    OrderStatus,
    OrderStage,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from panteon_v2.memory import PerformanceMemory


def _make_signal(sid: int = 1, sym: str = "BTC",
                 action: Action = Action.FUT_LONG_FULL,
                 price: float = 100.0,
                 bar: int = 1) -> Signal:
    return Signal(
        id=sid, bar=bar, sym=sym, action=action, price=price,
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

    def test_events_include_forensic_runtime_context(self):
        self.exec.set_event_context({
            "decision_id": "BITGET-1",
            "exchange": "BITGET",
            "symbol": "BTC",
            "timeframe": "1m",
            "mode": "live",
            "run_id": "run-1",
            "session_id": "session-1",
        })

        self.exec.execute(_make_signal(), balance_usd=1000.0)

        log = self.deps["event_log"]
        sent = list(log.query(event_types=[OrderSent]))[0]
        filled = list(log.query(event_types=[OrderFilled]))[0]
        opened = list(log.query(event_types=[PositionOpened]))[0]
        for event in (sent, filled, opened):
            self.assertEqual(event.decision_id, "BITGET-1")
            self.assertEqual(event.exchange, "BITGET")
            self.assertEqual(event.symbol, "BTC")
            self.assertEqual(event.timeframe, "1m")
            self.assertEqual(event.mode, "live")
            self.assertEqual(event.run_id, "run-1")
            self.assertEqual(event.session_id, "session-1")

    def test_execution_event_classes_define_audit_fields(self):
        required = {
            "fees",
            "slippage_pct",
            "latency_ms",
            "order_id",
            "pending_age_sec",
            "owner_scope",
        }
        for cls in (
            SignalEmitted,
            OrderSent,
            OrderFilled,
            OrderRejected,
            PositionOpened,
            PositionClosed,
            SymbolBlocked,
            MemoryUpdateFailed,
            ExecutionAttributed,
        ):
            with self.subTest(event=cls.__name__):
                self.assertTrue(required.issubset({f.name for f in fields(cls)}))

    def test_execution_events_include_audit_values(self):
        self.deps["exchange"].set_slippage_pct(0.01)
        self.deps["exchange"].set_fee_rate(0.001)
        self.exec.set_event_context({
            "decision_id": "MEXC-1",
            "exchange": "MEXC",
            "symbol": "BTC",
            "timeframe": "1m",
            "mode": "live",
            "run_id": "run-2",
            "session_id": "session-2",
        })

        result = self.exec.execute(_make_signal(price=100.0), balance_usd=1000.0)

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        log = self.deps["event_log"]
        sent = list(log.query(event_types=[OrderSent]))[0]
        filled = list(log.query(event_types=[OrderFilled]))[0]
        opened = list(log.query(event_types=[PositionOpened]))[0]
        self.assertTrue(sent.order_id.startswith("MEXC:BTC:long:MEXC-1:1"))
        self.assertEqual(sent.owner_scope, "panteon_owned")
        self.assertGreaterEqual(sent.latency_ms, 0.0)
        self.assertEqual(filled.order_id, result.trade.exchange_order_id)
        self.assertAlmostEqual(filled.fees, result.trade.fee)
        self.assertAlmostEqual(filled.slippage_pct, 1.0, places=6)
        self.assertGreaterEqual(filled.latency_ms, 0.0)
        self.assertEqual(filled.owner_scope, "panteon_owned")
        self.assertEqual(opened.order_id, result.trade.exchange_order_id)
        self.assertAlmostEqual(opened.fees, result.trade.fee)
        self.assertAlmostEqual(opened.slippage_pct, 1.0, places=6)

    def test_execution_attribution_is_emitted_for_filled_and_blocked_outcomes(self):
        self.exec.set_event_context({
            "decision_id": "MEXC-ATTR",
            "exchange": "MEXC",
            "symbol": "BTC",
            "timeframe": "1m",
            "mode": "live",
            "run_id": "run-attr",
            "session_id": "session-attr",
        })

        filled = self.exec.execute(_make_signal(sid=10), balance_usd=1000.0)
        blocked = self.exec.execute(
            _make_signal(sid=11, action=Action.HOLD, bar=2),
            balance_usd=1000.0,
        )

        self.assertEqual(filled.status, ExecutionStatus.FILLED)
        self.assertEqual(blocked.status, ExecutionStatus.BLOCKED)
        attrs = list(self.deps["event_log"].query(event_types=[ExecutionAttributed]))
        self.assertEqual([a.attribution_bucket for a in attrs], ["MARKET_PNL", "BLOCKED"])
        self.assertEqual(attrs[0].decision_id, "MEXC-ATTR")
        self.assertEqual(attrs[0].status, "filled")
        self.assertGreaterEqual(attrs[0].fees, 0.0)
        self.assertEqual(attrs[1].reason, "hold action")
        self.assertEqual(attrs[1].order_id, "")

    def test_live_jsonl_execution_events_include_mandatory_audit_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            deps = _make_executor()
            deps["event_log"] = EventLog(jsonl_path=path)
            ex = TradeExecutor(**deps)
            ex.set_event_context({
                "decision_id": "AUDIT-1",
                "exchange": "BITGET",
                "symbol": "BTC",
                "timeframe": "1m",
                "mode": "live",
                "run_id": "run-audit",
                "session_id": "session-audit",
            })

            ex.execute(_make_signal(price=100.0), balance_usd=1000.0)

            mandatory = {
                "decision_id",
                "exchange",
                "symbol",
                "timeframe",
                "mode",
                "run_id",
                "session_id",
                "fees",
                "slippage_pct",
                "latency_ms",
                "order_id",
                "pending_age_sec",
                "owner_scope",
            }
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            execution_rows = [
                row for row in rows
                if row.get("_type") in {"OrderSent", "OrderFilled", "PositionOpened"}
            ]
            self.assertEqual({row["_type"] for row in execution_rows},
                             {"OrderSent", "OrderFilled", "PositionOpened"})
            for row in execution_rows:
                self.assertTrue(mandatory.issubset(row.keys()), row)

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
        self.assertEqual(self.deps["perf"].get("A", Regime.BULLISH).blocked_signals, 1)
        self.assertAlmostEqual(self.deps["perf"].get("A", Regime.BULLISH).pnl_pct, 0.0)

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

    def test_close_does_not_require_min_notional_metadata(self):
        class MetadataFlakyExchange(FakeExchange):
            fail_metadata = False

            def get_min_notional(self, sym: str) -> float:
                if self.fail_metadata:
                    raise RuntimeError("metadata API down")
                return super().get_min_notional(sym)

        deps = _make_executor()
        deps["exchange"] = MetadataFlakyExchange()
        ex = TradeExecutor(**deps)

        ex.execute(_make_signal(sid=1), balance_usd=1000.0)
        deps["exchange"].fail_metadata = True

        result = ex.execute(
            _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=110.0),
            balance_usd=1000.0,
        )

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        self.assertFalse(deps["position_tracker"].has("BTC"))

    def test_risk_limit_blocked(self):
        # Создаём с очень малым балансом — notional ниже min
        sig = _make_signal()
        result = self.exec.execute(sig, balance_usd=1.0)
        self.assertEqual(result.status, ExecutionStatus.BLOCKED)
        self.assertIn("risk_limits", result.reason)
        self.assertEqual(self.deps["perf"].get("A", Regime.BULLISH).blocked_signals, 1)
        self.assertAlmostEqual(self.deps["perf"].get("A", Regime.BULLISH).pnl_pct, 0.0)


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
        self.assertEqual(self.deps["perf"].get("A", Regime.BULLISH).rejected_signals, 1)
        self.assertAlmostEqual(self.deps["perf"].get("A", Regime.BULLISH).pnl_pct, 0.0)

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

    def test_open_metadata_exception_is_rejected_without_sending_order(self):
        class MetadataDownExchange(FakeExchange):
            def __init__(self):
                super().__init__()
                self.sent = 0

            def get_min_notional(self, sym: str) -> float:
                raise RuntimeError("metadata API down")

            def send_order(self, signal, *, qty):
                self.sent += 1
                return super().send_order(signal, qty=qty)

        deps = self.deps.copy()
        deps["exchange"] = MetadataDownExchange()
        ex = TradeExecutor(**deps)

        result = ex.execute(_make_signal(), balance_usd=1000.0)

        self.assertEqual(result.status, ExecutionStatus.REJECTED)
        self.assertEqual(deps["exchange"].sent, 0)
        rejected = list(deps["event_log"].query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)
        self.assertIn("metadata API down", rejected[0].reason)


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
        self.assertEqual(self.deps["perf"].get("A", Regime.BULLISH).pending_signals, 1)
        self.assertAlmostEqual(self.deps["perf"].get("A", Regime.BULLISH).pnl_pct, 0.0)

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

    def test_pending_order_timeout_marks_ledger_and_rejects_once(self):
        ledger = OrderLedger()
        deps = _make_executor()
        deps["order_ledger"] = ledger
        ex = TradeExecutor(**deps)

        deps["exchange"].configure_pending("BTC")
        result = ex.execute(_make_signal(), balance_usd=1000.0)
        timed_out = ex.expire_stale_pending_orders(max_age_sec=0.0)

        self.assertEqual(result.status, ExecutionStatus.PENDING)
        self.assertEqual(timed_out, 1)
        record = ledger.pending_records()
        self.assertEqual(record, [])
        all_records = ledger.snapshot()
        self.assertEqual(len(all_records), 1)
        stored = next(iter(all_records.values()))
        self.assertEqual(stored["stage"], OrderStage.TIMED_OUT.value)
        rejected = list(deps["event_log"].query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)
        self.assertIn("pending timeout", rejected[0].reason)
        self.assertEqual(rejected[0].order_id, "FAKE-PENDING-1")
        self.assertGreaterEqual(rejected[0].pending_age_sec, 0.0)


class TestExecutionIdempotency(unittest.TestCase):
    def test_duplicate_execution_key_is_blocked_before_second_exchange_send(self):
        deps = _make_executor()
        ex = TradeExecutor(**deps)
        context = {
            "decision_id": "BITGET-DUP",
            "exchange": "BITGET",
            "symbol": "BTC",
            "timeframe": "1m",
            "mode": "live",
            "run_id": "run-dup",
            "session_id": "session-dup",
        }
        ex.set_event_context(context)
        ex.execute(_make_signal(sid=1, action=Action.FUT_LONG_FULL), balance_usd=1000.0)

        deps["exchange"].configure_pending("BTC")
        close_one = _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=101.0, bar=9)
        close_two = _make_signal(sid=3, action=Action.FUT_CLOSE_ALL, price=101.0, bar=9)
        first = ex.execute(close_one, balance_usd=1000.0)
        second = ex.execute(close_two, balance_usd=1000.0)

        self.assertEqual(first.status, ExecutionStatus.PENDING)
        self.assertEqual(second.status, ExecutionStatus.BLOCKED)
        self.assertIn("duplicate execution key", second.reason)
        self.assertEqual(len(deps["exchange"].orders_log), 2)
        rejected = list(deps["event_log"].query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)
        self.assertIn("duplicate execution key", rejected[0].reason)
        self.assertEqual(rejected[0].owner_scope, "panteon_owned")


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
