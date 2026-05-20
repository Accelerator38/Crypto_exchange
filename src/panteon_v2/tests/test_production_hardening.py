"""Regression tests for live-production hardening."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from panteon_v2.domain.types import Action, MarketSnapshot, Regime, Signal, Trade
from panteon_v2.execution import (
    ExchangePosition,
    ExecutionStatus,
    FakeExchange,
    OrderStatus,
    PositionTracker,
)
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import (
    EnsemblePlayer,
    Strategist,
    StrategistConfig,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.tests._helpers import FakeAgent


def _signal(action: Action, *, sid: int = 1, sym: str = "BTC", price: float = 100.0) -> Signal:
    return Signal(
        id=sid,
        bar=1,
        sym=sym,
        action=action,
        price=price,
        regime=Regime.BULLISH,
        by_player="Leader",
        by_agent="Agent",
    )


class TestExchangeProfiles(unittest.TestCase):
    def test_runtime_settings_are_loaded_for_requested_exchange(self):
        from panteon_v2.app.v1_futures_adapter import (
            load_runtime_leverage,
            load_runtime_trade_fraction,
        )

        calls = []

        class Runtime:
            display_name = "BITGET"
            connector_module_name = "bitget_connector"
            api_module_name = "bitget_api"

            def load_settings(self):
                return {
                    "trade_fraction": "0.06",
                    "bitget_trade_fraction": "0.035",
                    "leverage": "2",
                    "bitget_leverage": "5",
                }

            def parse_settings(self, raw_cfg):
                return {
                    "trade_fraction": float(raw_cfg["bitget_trade_fraction"]),
                    "leverage": int(raw_cfg["bitget_leverage"]),
                    "symbols": ["BTC", "ETH"],
                    "futures_fee": 0.0006,
                }

        fake_registry = types.ModuleType("exchange_registry")

        def load_exchange_runtime(exchange_id=None, **_kwargs):
            calls.append(exchange_id)
            return Runtime()

        fake_registry.load_exchange_runtime = load_exchange_runtime
        old_registry = sys.modules.get("exchange_registry")
        sys.modules["exchange_registry"] = fake_registry
        try:
            self.assertAlmostEqual(load_runtime_trade_fraction("BITGET", default=0.10), 0.035)
            self.assertEqual(load_runtime_leverage("BITGET", default=2), 5)
            self.assertEqual(calls, ["bitget", "bitget"])
        finally:
            if old_registry is None:
                sys.modules.pop("exchange_registry", None)
            else:
                sys.modules["exchange_registry"] = old_registry


class TestPendingOrders(unittest.TestCase):
    def test_v1_adapter_poll_order_converts_ack_to_filled_from_order_detail(self):
        from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter

        class AckThenFilledClient:
            BITGET_MIN_NOTIONAL_USDT = 5.0

            def __init__(self):
                self.polls = 0

            def _get_contract_meta(self, symbol: str) -> dict:
                return {"amountStep": 0.01, "minVol": 1, "volUnit": 1}

            def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
                return {"success": True, "order_id": "BG-ACK-1"}

            def get_order(self, order_id: str, symbol: str | None = None) -> dict:
                self.polls += 1
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "filled",
                    "avgPrice": 101.0,
                    "amount": 0.03,
                    "fee": 0.002,
                }

            def get_futures_positions(self):
                return []

        client = AckThenFilledClient()
        adapter = BitgetExchangeAdapter(order_client=client, read_client=client, leverage=2)
        sig = _signal(Action.FUT_LONG_FULL, sym="ETH", price=100.0)

        pending = adapter.send_order(sig, qty=0.03)
        filled = adapter.poll_order(pending.exchange_order_id, sig)

        self.assertEqual(pending.status, OrderStatus.PENDING)
        self.assertEqual(filled.status, OrderStatus.FILLED)
        self.assertEqual(filled.trade.exchange_order_id, "BG-ACK-1")
        self.assertAlmostEqual(filled.trade.fill_price, 101.0)
        self.assertEqual(client.polls, 1)

    def test_trade_executor_records_pending_to_filled_in_order_ledger(self):
        from panteon_v2.attribution import EventLog
        from panteon_v2.execution import RiskLimits, RiskLimitsConfig, SymbolHealthMonitor, TradeExecutor
        from panteon_v2.execution.order_ledger import OrderLedger, OrderStage

        class PendingThenFilledExchange(FakeExchange):
            def send_order(self, signal, *, qty):
                return super().send_order(signal, qty=qty)._replace()  # type: ignore[attr-defined]

        class Exchange(FakeExchange):
            def send_order(self, signal, *, qty):
                from panteon_v2.execution import OrderResult

                return OrderResult(
                    status=OrderStatus.PENDING,
                    signal_id=signal.id,
                    sym=signal.sym,
                    exchange_order_id="PENDING-1",
                    message="accepted",
                )

            def poll_order(self, order_id, signal):
                from panteon_v2.execution import OrderResult

                trade = Trade(
                    signal_id=signal.id,
                    bar=signal.bar,
                    sym=signal.sym,
                    side="long",
                    qty=0.1,
                    fill_price=signal.price,
                    fee=0.01,
                    exchange_order_id=order_id,
                )
                return OrderResult(
                    status=OrderStatus.FILLED,
                    signal_id=signal.id,
                    sym=signal.sym,
                    exchange_order_id=order_id,
                    trade=trade,
                )

        ledger = OrderLedger()
        executor = TradeExecutor(
            exchange=Exchange(name="REAL"),
            health=SymbolHealthMonitor(),
            risk_limits=RiskLimits(config=RiskLimitsConfig(capital_fraction=0.10)),
            position_tracker=PositionTracker(),
            perf=PerformanceMemory(),
            event_log=EventLog(),
            order_ledger=ledger,
        )

        result = executor.execute(_signal(Action.FUT_LONG_FULL), balance_usd=1000.0)

        self.assertEqual(result.status, ExecutionStatus.FILLED)
        record = ledger.get("PENDING-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.stage, OrderStage.FILLED)
        self.assertEqual(record.signal_id, 1)


class TestBidirectionalReconcile(unittest.TestCase):
    def test_reconcile_removes_tracker_positions_absent_from_exchange(self):
        from panteon_v2.app.live_state import reconcile_tracker_with_exchange
        from panteon_v2.attribution import EventLog, PositionClosed

        tracker = PositionTracker()
        tracker.force_set(TrackedPosition(
            open_signal_id=10,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=0.25,
            fee_open=0.01,
            by_player="Leader",
            by_agent="Agent",
            opened_at=datetime.now(timezone.utc),
        ))

        class Exchange:
            def get_all_positions(self):
                return {}

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()
            event_log = EventLog()

        pipeline = Pipeline()
        summary = reconcile_tracker_with_exchange(pipeline, bar_index=7)

        self.assertEqual(summary["removed"], 1)
        self.assertIsNone(tracker.get("BTC"))
        events = list(pipeline.event_log.query(event_types=[PositionClosed]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].by_player, "Leader")
        self.assertEqual(events[0].close_signal_id, 0)


class TestRegimeDetectorHardening(unittest.TestCase):
    def test_single_alt_crash_does_not_flip_entire_market_to_crash(self):
        from panteon_v2.app.regime_detector import PriceRegimeDetector

        detector = PriceRegimeDetector(lookback=2, hysteresis_bars=1)
        detector.update({"BTC": 100.0, "ETH": 100.0, "ALT": 100.0})
        regime = detector.update({"BTC": 100.2, "ETH": 99.9, "ALT": 90.0})

        self.assertNotEqual(regime, Regime.CRASH)

    def test_anchor_crash_flips_to_crash_with_high_confidence(self):
        from panteon_v2.app.regime_detector import PriceRegimeDetector

        detector = PriceRegimeDetector(lookback=2, hysteresis_bars=1)
        detector.update({"BTC": 100.0, "ETH": 100.0, "ALT": 100.0})
        regime = detector.update({"BTC": 92.0, "ETH": 93.0, "ALT": 99.0})

        self.assertEqual(regime, Regime.CRASH)
        self.assertGreaterEqual(detector.confidence, 0.5)


class TestPromotionGate(unittest.TestCase):
    def test_strategist_blocks_cold_player_until_virtual_record_is_proven(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        player = EnsemblePlayer(
            label="ColdPlayer",
            agents=[FakeAgent("AgentA")],
            weights={"AgentA": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        strategist = Strategist(
            perf,
            qm,
            candidates=[player],
            config=StrategistConfig(
                min_live_closed_trades=2,
                min_live_score=0.0,
                max_live_drawdown_pct=10.0,
            ),
        )

        with self.assertRaises(ValueError):
            strategist.consider_switch(Regime.BULLISH, current_bar=1)


class TestShadowAttribution(unittest.TestCase):
    def test_player_shadow_trade_preserves_agent_contributor(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        from panteon_v2.attribution import EventLog
        from panteon_v2.execution import RiskLimitsConfig

        agent = FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL})
        registry = types.SimpleNamespace(
            all_agents=lambda: [agent],
            all_labels=lambda: ["AgentA"],
            has=lambda label: False,
            register=lambda clone, replace=False: None,
            __len__=lambda self: 1,
        )
        from panteon_v2.selection import AgentRegistry

        real_registry = AgentRegistry()
        real_registry.register(agent)
        perf = PerformanceMemory(trade_fraction=1.0)
        tournament = ProductionShadowTournament(
            registry=real_registry,
            perf=perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        player = EnsemblePlayer(
            label="PlayerA",
            agents=[agent],
            weights={"AgentA": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(open_single=0.10, open_multi=0.10, open_floor=0.10),
        )
        market = MarketSnapshot(
            bar=1,
            timestamp=datetime.now(timezone.utc),
            regime=Regime.BULLISH,
            prices={"BTC": 100.0},
            volumes={"BTC": 1.0},
        )

        tournament.run_bar(market, players=[player], balance_usd=1000.0)

        self.assertEqual(perf.get("AgentA", Regime.BULLISH).entries, 2)
        self.assertEqual(perf.get("PlayerA", Regime.BULLISH).entries, 1)

    def test_shadow_agent_failure_emits_event_and_penalizes_actor(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        from panteon_v2.attribution import AgentVoteFailed, EventLog
        from panteon_v2.execution import RiskLimitsConfig
        from panteon_v2.selection import AgentRegistry

        class CrashingAgent:
            label = "BrokenAgent"

            def act(self, market):
                raise RuntimeError("model exploded")

        registry = AgentRegistry()
        registry.register(CrashingAgent())
        perf = PerformanceMemory()
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=registry,
            perf=perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=event_log,
        )
        market = MarketSnapshot(
            bar=1,
            timestamp=datetime.now(timezone.utc),
            regime=Regime.BULLISH,
            prices={"BTC": 100.0},
            volumes={"BTC": 1.0},
        )

        tournament.run_bar(market, players=[], balance_usd=1000.0)

        events = list(event_log.query(event_types=[AgentVoteFailed]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].agent_label, "BrokenAgent")
        self.assertIn("model exploded", events[0].reason)
        metrics = perf.get("BrokenAgent", Regime.BULLISH)
        self.assertEqual(metrics.losses, 0)
        self.assertEqual(metrics.pnl_pct, 0.0)
        self.assertEqual(metrics.rejected_signals, 1)
        self.assertEqual(metrics.execution_failures, 1)


class TestStartupFailClosed(unittest.TestCase):
    def test_live_bridge_failure_fails_closed_without_explicit_fallback(self):
        from panteon_v2.app.startup import _should_fail_closed_after_bridge_error

        self.assertTrue(
            _should_fail_closed_after_bridge_error(
                mode="live_futures",
                polling_session_dir=None,
                allow_live_feed_fallback=False,
            )
        )
        self.assertFalse(
            _should_fail_closed_after_bridge_error(
                mode="live_futures",
                polling_session_dir="session",
                allow_live_feed_fallback=False,
            )
        )

    def test_live_guardrails_are_resolved_from_settings(self):
        from panteon_v2.app.startup import _live_execution_config_from_settings

        cfg = _live_execution_config_from_settings({
            "v2_max_daily_loss_pct": 4.5,
            "v2_max_slippage_pct": 0.35,
            "v2_max_api_error_streak": 2,
            "v2_max_stale_feed_polls": 7,
            "v2_pending_order_timeout_sec": 90,
            "v2_genetics_probation_execution_enabled": "on",
            "v2_genetics_probation_risk_mult": 0.2,
            "v2_genetics_probation_max_real_trades": 7,
            "v2_genetics_probation_require_shadow_confirmation": "off",
        })

        self.assertEqual(cfg.max_daily_loss_pct, 4.5)
        self.assertEqual(cfg.max_slippage_pct, 0.35)
        self.assertEqual(cfg.max_api_error_streak, 2)
        self.assertEqual(cfg.max_stale_feed_polls, 7)
        self.assertEqual(cfg.pending_order_timeout_sec, 90)
        self.assertTrue(cfg.genetics_probation_execution_enabled)
        self.assertEqual(cfg.genetics_probation_risk_mult, 0.2)
        self.assertEqual(cfg.genetics_probation_max_real_trades, 7)
        self.assertFalse(cfg.genetics_probation_require_shadow_confirmation)

    def test_executable_soft_top1_strategy_is_resolved_from_settings(self):
        from panteon_v2.app.startup import _strategist_config_from_settings

        cfg = _strategist_config_from_settings({
            "v2_executable_soft_top1_enabled": "on",
            "v2_probation_loss_kill_min_closed_trades": 2,
            "v2_probation_loss_kill_pnl_pct": -0.15,
            "v2_probation_loss_kill_win_rate_pct": 50,
            "v2_probation_loss_kill_all_labels": "on",
        })

        self.assertTrue(cfg.use_v3_rolling_score)
        self.assertTrue(cfg.use_v3_soft_shadow_score)
        self.assertTrue(cfg.v3_current_actionable_gate_enabled)
        self.assertEqual(cfg.v3_shadow_rolling_window_bars, 24)
        self.assertEqual(cfg.v3_shadow_rolling_min_closed_trades, 20)
        self.assertEqual(cfg.v3_probation_loss_kill_min_closed_trades, 2)
        self.assertEqual(cfg.v3_probation_loss_kill_pnl_pct, -0.15)
        self.assertEqual(cfg.v3_probation_loss_kill_win_rate_pct, 50)
        self.assertEqual(cfg.v3_probation_loss_kill_label_prefixes, ())


class TestKillSwitches(unittest.TestCase):
    def test_failed_orders_stale_feed_and_slippage_disable_real_trading(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import (
            _kill_switch_reason,
            _record_order_failure,
            _record_order_success,
            _record_stale_feed_poll,
        )
        from panteon_v2.execution import ExecutionResult, ExecutionStatus

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                max_consecutive_failed_orders=2,
                max_stale_feed_polls=3,
                max_slippage_pct=1.0,
            ),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )

        _record_order_failure(pipeline, "exchange error: timeout")
        self.assertEqual(_kill_switch_reason(pipeline), "")
        _record_order_failure(pipeline, "exchange error: timeout")
        self.assertIn("consecutive failed orders", _kill_switch_reason(pipeline))

        stale_pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(max_stale_feed_polls=3),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        _record_stale_feed_poll(stale_pipeline, 3)
        self.assertIn("stale feed", _kill_switch_reason(stale_pipeline))

        slippage_pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(max_slippage_pct=1.0),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        sig = _signal(Action.FUT_LONG_FULL, price=100.0)
        result = ExecutionResult(
            status=ExecutionStatus.FILLED,
            signal=sig,
            trade=Trade(
                signal_id=sig.id,
                bar=sig.bar,
                sym=sig.sym,
                side="long",
                qty=0.1,
                fill_price=103.0,
                fee=0.01,
            ),
        )
        _record_order_success(slippage_pipeline, sig, result)
        self.assertIn("excessive slippage", _kill_switch_reason(slippage_pipeline))


class TestMemorySeparationAndQuarantineState(unittest.TestCase):
    def test_pipeline_separates_virtual_selection_memory_from_real_execution_memory(self):
        from panteon_v2.app.bootstrap import build_production_pipeline
        from panteon_v2.selection import AgentRegistry

        registry = AgentRegistry()
        registry.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
        )

        self.assertIs(pipeline.perf, pipeline.virtual_perf)
        self.assertIsNot(pipeline.real_perf, pipeline.virtual_perf)
        self.assertIs(pipeline.executor._perf, pipeline.real_perf)

    def test_snapshot_persists_real_memory_and_order_ledger(self):
        from panteon_v2.app.migration import load_v2_snapshot, save_v2_snapshot
        from panteon_v2.execution.order_ledger import OrderLedger, OrderStage

        virtual = PerformanceMemory()
        real = PerformanceMemory()
        ledger = OrderLedger()
        sig = _signal(Action.FUT_LONG_FULL)
        real.record_signal(sig)
        ledger.record_submitted(sig)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            save_v2_snapshot(
                virtual,
                path,
                real_perf=real,
                order_ledger=ledger,
            )

            restored_virtual = PerformanceMemory()
            restored_real = PerformanceMemory()
            restored_ledger = OrderLedger()
            self.assertTrue(load_v2_snapshot(
                restored_virtual,
                path,
                real_perf=restored_real,
                order_ledger=restored_ledger,
            ))

        self.assertEqual(restored_real.get("Leader", Regime.BULLISH).signals, 1)
        restored_record = restored_ledger.get("signal:1")
        self.assertEqual(restored_record.stage, OrderStage.SUBMITTED)
        self.assertEqual(restored_record.signal.sym, "BTC")

    def test_quarantine_records_reason_and_probation_on_release(self):
        qm = QuarantineManager(seed=set())
        qm.force_quarantine("Bad", reason="hopeless_in_all_regimes", bar=3)

        record = qm.record_for("Bad")
        self.assertEqual(record.state, "quarantined")
        self.assertEqual(record.reason, "hopeless_in_all_regimes")
        self.assertEqual(record.entered_bar, 3)

        qm.force_release("Bad", reason="locally_proven", bar=8)
        record = qm.record_for("Bad")
        self.assertEqual(record.state, "probation")
        self.assertFalse(qm.is_quarantined("Bad"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
