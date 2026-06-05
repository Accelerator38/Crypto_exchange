"""Regression tests for live-production hardening."""

from __future__ import annotations

import os
import logging
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PANTEON_RUNTIME = PROJECT_ROOT / "src" / "panteon_runtime"


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
    def test_mexc_futures_domain_defaults_to_current_api_domain(self):
        import importlib

        sys.path.insert(0, str(PANTEON_RUNTIME))
        mexc_connector = importlib.import_module("mexc_connector")
        exchange_api_runtime = importlib.import_module("exchange_api_runtime")
        mexc_funding = importlib.import_module("mexc_funding")

        with patch.dict(os.environ, {"MEXC_FUTURES_BASE_URL": ""}):
            self.assertEqual(
                mexc_connector._mexc_futures_live_base_url(),
                "https://api.mexc.com",
            )
            self.assertEqual(
                exchange_api_runtime._mexc_futures_base_url(),
                "https://api.mexc.com",
            )
            self.assertEqual(
                mexc_funding._mexc_futures_base_url(),
                "https://api.mexc.com",
            )

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

    def test_bitget_top_symbols_exclude_static_symbol_blocklist(self):
        import importlib

        sys.path.insert(0, str(PANTEON_RUNTIME))
        bitget_connector = importlib.import_module("bitget_connector")

        class PublicClient:
            def load_markets(self):
                return {
                    "BTC/USDT": {
                        "spot": True,
                        "quote": "USDT",
                        "base": "BTC",
                        "active": True,
                    },
                    "BSB/USDT": {
                        "spot": True,
                        "quote": "USDT",
                        "base": "BSB",
                        "active": True,
                    },
                    "ETH/USDT": {
                        "spot": True,
                        "quote": "USDT",
                        "base": "ETH",
                        "active": True,
                    },
                }

            def fetch_tickers(self):
                return {
                    "BTC/USDT": {"quoteVolume": 100.0},
                    "BSB/USDT": {"quoteVolume": 1000.0},
                    "ETH/USDT": {"quoteVolume": 50.0},
                }

        with patch.object(bitget_connector, "BITGET_SYMBOL_BLOCKLIST", frozenset({"BSB"})):
            with patch.object(bitget_connector, "_public_client", return_value=PublicClient()):
                self.assertEqual(bitget_connector.fetch_top_symbols(3), ["BTC", "ETH"])

    def test_bitget_futures_client_seeds_bad_symbols_from_static_blocklist(self):
        import importlib

        sys.path.insert(0, str(PANTEON_RUNTIME))
        bitget_connector = importlib.import_module("bitget_connector")

        class PrivateClient:
            def load_markets(self):
                return {}

        with patch.object(bitget_connector, "BITGET_SYMBOL_BLOCKLIST", frozenset({"BSB"})):
            with patch.object(bitget_connector, "_private_client", return_value=PrivateClient()):
                client = bitget_connector.BitgetFuturesClient("key", "secret", "pass")

        self.assertIn("BSB", client._bad_symbols)


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
        self.assertEqual(detector.symbol_regimes["ALT"], Regime.CRASH)
        self.assertNotEqual(detector.symbol_regimes["BTC"], Regime.CRASH)

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
            "v2_max_equity_peak_drawdown_pct": 3.0,
            "v2_max_slippage_pct": 0.35,
            "v2_max_api_error_streak": 2,
            "v2_max_stale_feed_polls": 7,
            "v2_pending_order_timeout_sec": 90,
            "v2_genetics_probation_execution_enabled": "on",
            "v2_genetics_probation_labels": "GeneticsCore, GeneticsRouter",
            "v2_genetics_probation_allowed_regimes": "neutral,bearish",
            "v2_genetics_probation_allowed_signal_keys": (
                "agent:GeneticsCore|BTC|FUT_LONG_FULL"
            ),
            "v2_genetics_probation_risk_mult": 0.2,
            "v2_genetics_probation_min_regime_confidence": 0.75,
            "v2_genetics_probation_max_real_trades": 7,
            "v2_genetics_probation_max_daily_trades": 3,
            "v2_genetics_probation_require_shadow_confirmation": "off",
            "v2_genetics_probation_max_consecutive_failed_orders": 2,
            "v2_genetics_probation_max_realized_loss_pct": 0.25,
            "v2_max_real_symbol_min_executable_notional_usd": 40.0,
        })

        self.assertEqual(cfg.max_daily_loss_pct, 4.5)
        self.assertEqual(cfg.max_equity_peak_drawdown_pct, 3.0)
        self.assertEqual(cfg.max_slippage_pct, 0.35)
        self.assertEqual(cfg.max_api_error_streak, 2)
        self.assertEqual(cfg.max_stale_feed_polls, 7)
        self.assertEqual(cfg.pending_order_timeout_sec, 90)
        self.assertTrue(cfg.genetics_probation_execution_enabled)
        self.assertEqual(cfg.genetics_probation_labels, ("GeneticsCore", "GeneticsRouter"))
        self.assertEqual(cfg.genetics_probation_allowed_regimes, ("neutral", "bearish"))
        self.assertEqual(
            cfg.genetics_probation_allowed_signal_keys,
            ("agent:GeneticsCore|BTC|FUT_LONG_FULL",),
        )
        self.assertEqual(cfg.genetics_probation_risk_mult, 0.2)
        self.assertEqual(cfg.genetics_probation_min_regime_confidence, 0.75)
        self.assertEqual(cfg.genetics_probation_max_real_trades, 7)
        self.assertEqual(cfg.genetics_probation_max_daily_trades, 3)
        self.assertFalse(cfg.genetics_probation_require_shadow_confirmation)
        self.assertEqual(cfg.genetics_probation_max_consecutive_failed_orders, 2)
        self.assertEqual(cfg.genetics_probation_max_realized_loss_pct, 0.25)
        self.assertEqual(cfg.max_real_symbol_min_executable_notional_usd, 40.0)

    def test_live_guardrails_accept_exchange_scoped_probation_overrides(self):
        from panteon_v2.app.startup import _live_execution_config_from_settings

        cfg = _live_execution_config_from_settings({
            "v2_max_consecutive_failed_orders": 3,
            "mexc_v2_max_consecutive_failed_orders": 8,
            "v2_genetics_probation_execution_enabled": "off",
            "bitget_v2_genetics_probation_execution_enabled": "on",
            "bitget_v2_genetics_probation_labels": "GeneticsCore",
            "bitget_v2_genetics_probation_allowed_regimes": "neutral",
            "bitget_v2_genetics_probation_max_real_trades": 7,
            "bitget_v2_genetics_probation_max_daily_trades": 3,
            "bitget_v2_max_real_symbol_min_executable_notional_usd": 40.0,
        }, exchange_name="BITGET")

        self.assertEqual(cfg.max_consecutive_failed_orders, 3)
        self.assertTrue(cfg.genetics_probation_execution_enabled)
        self.assertEqual(cfg.genetics_probation_labels, ("GeneticsCore",))
        self.assertEqual(cfg.genetics_probation_allowed_regimes, ("neutral",))
        self.assertEqual(cfg.genetics_probation_max_real_trades, 7)
        self.assertEqual(cfg.genetics_probation_max_daily_trades, 3)
        self.assertEqual(cfg.max_real_symbol_min_executable_notional_usd, 40.0)

        mexc_cfg = _live_execution_config_from_settings({
            "v2_max_consecutive_failed_orders": 3,
            "mexc_v2_max_consecutive_failed_orders": 8,
            "v2_genetics_probation_execution_enabled": "on",
            "mexc_v2_genetics_probation_execution_enabled": "off",
        }, exchange_name="MEXC")

        self.assertEqual(mexc_cfg.max_consecutive_failed_orders, 8)
        self.assertFalse(mexc_cfg.genetics_probation_execution_enabled)

    def test_v1_settings_parser_accepts_exchange_scoped_symbols(self):
        sys.path.insert(0, str(PANTEON_RUNTIME))
        try:
            import mexc_connector

            raw = {
                "symbols": "all",
                "bitget_symbols": "BTC,ETH,SOL",
            }
            with patch.dict(os.environ, {"CRYPTO_EXCHANGE": "bitget"}):
                bitget_cfg = mexc_connector._parse_settings(raw)
            with patch.dict(os.environ, {"CRYPTO_EXCHANGE": "mexc"}):
                mexc_cfg = mexc_connector._parse_settings(raw)
        finally:
            try:
                sys.path.remove(str(PANTEON_RUNTIME))
            except ValueError:
                pass

        self.assertEqual(bitget_cfg["symbols"], ["BTC", "ETH", "SOL"])
        self.assertIsNone(mexc_cfg["symbols"])

    def test_bitget_bridge_uses_scoped_symbols_before_auto_top_fallback(self):
        sys.path.insert(0, str(PANTEON_RUNTIME))
        try:
            import bitget_connector

            cfg = {
                "initial_capital": 100.0,
                "trade_fraction": 0.10,
                "leverage": 2,
                "poll_interval": 60,
                "liquidity_min_adv": 0.0,
                "spot_fee": 0.001,
                "futures_fee": 0.0002,
                "slippage": 0.0001,
                "tf_kline": "1m",
                "bar": 60,
                "symbols": None,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    with patch.object(
                        bitget_connector,
                        "_BITGET_SETTINGS_RAW",
                        {"bitget_symbols": "BTC,ETH,SOL"},
                    ), patch.object(
                        bitget_connector,
                        "_public_client",
                        return_value=object(),
                    ), patch.object(
                        bitget_connector,
                        "fetch_top_symbols",
                        side_effect=AssertionError("auto-top should not be used"),
                    ):
                        bridge = bitget_connector.AgentBitgetBridge(
                            {},
                            cfg,
                            mode="paper",
                            output_dir=tmpdir,
                        )
                finally:
                    tmp_path = str(Path(tmpdir).resolve())
                    root_logger = logging.getLogger()
                    for handler in list(root_logger.handlers):
                        if not isinstance(handler, logging.FileHandler):
                            continue
                        if str(getattr(handler, "baseFilename", "")).startswith(tmp_path):
                            root_logger.removeHandler(handler)
                            handler.close()
        finally:
            try:
                sys.path.remove(str(PANTEON_RUNTIME))
            except ValueError:
                pass

        self.assertEqual(bridge.symbols, ["BTC", "ETH", "SOL"])

    def test_exchange_settings_merge_raw_v2_keys_with_parsed_settings(self):
        from panteon_v2.app.startup import _load_exchange_settings

        profile = types.SimpleNamespace(
            raw_settings={
                "v2_flash_enabled": "on",
                "v2_max_equity_peak_drawdown_pct": "5",
            },
            parsed_settings={
                "trade_fraction": 0.06,
            },
        )

        with patch(
            "panteon_v2.app.exchange_profile.load_exchange_profile",
            return_value=profile,
        ):
            settings = _load_exchange_settings("MEXC")

        self.assertEqual(settings["trade_fraction"], 0.06)
        self.assertEqual(settings["v2_flash_enabled"], "on")
        self.assertEqual(settings["v2_max_equity_peak_drawdown_pct"], "5")

    def test_flash_genetics_probation_bypass_flags_are_resolved_from_settings(self):
        from panteon_v2.app.startup import _flash_allocator_config_from_settings

        cfg = _flash_allocator_config_from_settings({
            "v2_flash_genetics_probation_bypass_min_closed_enabled": "on",
            "v2_flash_genetics_probation_bypass_trend_gate_enabled": "on",
        })

        self.assertTrue(cfg.genetics_probation_bypass_min_closed_enabled)
        self.assertTrue(cfg.genetics_probation_bypass_trend_gate_enabled)

    def test_risk_max_open_positions_is_resolved_from_settings(self):
        from panteon_v2.app.startup import _risk_config_from_settings

        cfg = _risk_config_from_settings({
            "v2_risk_max_open_positions": "8",
        }, trade_fraction=0.06)

        self.assertEqual(cfg.capital_fraction, 0.06)
        self.assertEqual(cfg.max_open_positions, 8)
        self.assertTrue(cfg.floor_to_exchange_min_notional)

    def test_executable_soft_top1_strategy_is_resolved_from_settings(self):
        from panteon_v2.app.startup import _strategist_config_from_settings

        cfg = _strategist_config_from_settings({
            "v2_executable_soft_top1_enabled": "on",
            "v2_shadow_fresh_handoff_enabled": "on",
            "v2_shadow_fresh_handoff_max_age_bars": 24,
            "v2_shadow_fresh_handoff_require_positive_unrealized": "off",
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
        self.assertTrue(cfg.v3_shadow_fresh_handoff_enabled)
        self.assertEqual(cfg.v3_shadow_fresh_handoff_max_age_bars, 24)
        self.assertFalse(cfg.v3_shadow_fresh_handoff_require_positive_unrealized)
        self.assertEqual(cfg.v3_probation_loss_kill_min_closed_trades, 2)
        self.assertEqual(cfg.v3_probation_loss_kill_pnl_pct, -0.15)
        self.assertEqual(cfg.v3_probation_loss_kill_win_rate_pct, 50)
        self.assertEqual(cfg.v3_probation_loss_kill_label_prefixes, ())


class TestKillSwitches(unittest.TestCase):
    def test_failed_orders_stale_feed_and_slippage_disable_real_trading(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import (
            _kill_switch_reason,
            _record_exchange_desync,
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

        desync_pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(max_exchange_desync_events=1),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        _record_exchange_desync(desync_pipeline, {
            "owned_removed": 3,
            "snapshot_unreliable": True,
            "guard_reason": "cached_futures_assets",
        })
        self.assertEqual(desync_pipeline.kill_switch.exchange_desync_events, 0)
        self.assertEqual(_kill_switch_reason(desync_pipeline), "")

    def test_unhealthy_balance_snapshot_is_preserved_for_open_blocking(self):
        from panteon_v2.app.bootstrap import KillSwitchState
        from panteon_v2.app.main_loop import (
            _exchange_health_open_block_reason,
            sync_pipeline_balance,
        )

        class Exchange:
            def get_account_snapshot(self):
                return {
                    "current_balance": 0.0,
                    "futures_equity": 0.0,
                    "available_balance": 0.0,
                    "data_health": {
                        "last_data_error_reason": "cached_futures_assets",
                        "snapshot_healthy": False,
                    },
                }

        pipeline = types.SimpleNamespace(
            executor=types.SimpleNamespace(_exchange=Exchange()),
            kill_switch=KillSwitchState(),
            current_balance=100.0,
        )

        self.assertIsNone(sync_pipeline_balance(pipeline))
        self.assertEqual(
            pipeline.account_snapshot["data_health"]["last_data_error_reason"],
            "cached_futures_assets",
        )
        self.assertEqual(
            _exchange_health_open_block_reason(pipeline),
            "cached_futures_assets",
        )

    def test_genetics_probation_failed_orders_disable_only_probation(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import _kill_switch_reason, _record_order_failure

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                max_consecutive_failed_orders=5,
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsCore",),
                genetics_probation_max_consecutive_failed_orders=2,
            ),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        signal = Signal(
            id=777,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.NEUTRAL,
            by_player="Panteon_Flash",
            by_agent="GeneticsCore",
        )

        _record_order_failure(pipeline, "exchange rejected", signal=signal)
        self.assertEqual(_kill_switch_reason(pipeline), "")
        self.assertEqual(pipeline.kill_switch.genetics_probation_disabled_reason, "")

        _record_order_failure(pipeline, "exchange rejected", signal=signal)

        self.assertEqual(_kill_switch_reason(pipeline), "")
        self.assertIn(
            "GeneticsCore consecutive failed orders",
            pipeline.kill_switch.genetics_probation_disabled_reason,
        )

    def test_genetics_probation_failure_disables_only_matching_label(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import (
            _genetics_probation_disabled_reason,
            _record_order_failure,
        )

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                max_consecutive_failed_orders=5,
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsCore", "GeneticsRegimeAdaptiveBias"),
                genetics_probation_max_consecutive_failed_orders=2,
                genetics_probation_max_consecutive_failed_orders_by_label={
                    "GeneticsRegimeAdaptiveBias": 1,
                },
            ),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        adaptive_signal = Signal(
            id=778,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.NEUTRAL,
            by_player="GeneticsRegimeAdaptiveBias",
            by_agent="GeneticsRegimeAdaptiveBias",
        )

        _record_order_failure(pipeline, "exchange rejected", signal=adaptive_signal)

        self.assertIn(
            "GeneticsRegimeAdaptiveBias consecutive failed orders",
            _genetics_probation_disabled_reason(
                pipeline,
                "GeneticsRegimeAdaptiveBias",
            ),
        )
        self.assertEqual(_genetics_probation_disabled_reason(pipeline, "GeneticsCore"), "")

    def test_default_ensemble_origin_does_not_count_as_genetics_probation_failure(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import _kill_switch_reason, _record_order_failure

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                max_consecutive_failed_orders=5,
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsCore",),
                genetics_probation_max_consecutive_failed_orders=1,
            ),
            kill_switch=KillSwitchState(),
            initial_capital=1000.0,
            current_balance=1000.0,
        )
        signal = Signal(
            id=778,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.NEUTRAL,
            by_player="DefaultEnsemble",
            by_agent="GeneticsCore",
        )

        _record_order_failure(pipeline, "exchange rejected", signal=signal)

        self.assertEqual(_kill_switch_reason(pipeline), "")
        self.assertEqual(pipeline.kill_switch.genetics_probation_disabled_reason, "")

    def test_genetics_probation_realized_loss_disables_probation(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import (
            _disable_genetics_probation_if_realized_loss_exceeded,
        )

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsCore",),
                genetics_probation_max_realized_loss_pct=0.25,
            ),
            kill_switch=KillSwitchState(),
            real_perf=types.SimpleNamespace(
                get=lambda label: types.SimpleNamespace(
                    closed_trades=1,
                    pnl_pct=-0.30 if label == "GeneticsCore" else 0.0,
                ),
            ),
        )

        _disable_genetics_probation_if_realized_loss_exceeded(pipeline)

        self.assertIn(
            "GeneticsCore realized loss",
            pipeline.kill_switch.genetics_probation_disabled_reason,
        )

    def test_genetics_probation_realized_loss_disables_only_losing_label(self):
        from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
        from panteon_v2.app.main_loop import (
            _disable_genetics_probation_if_realized_loss_exceeded,
            _genetics_probation_disabled_reason,
        )

        pipeline = types.SimpleNamespace(
            live_execution=LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsCore", "GeneticsRegimeAdaptiveBias"),
                genetics_probation_max_realized_loss_pct=0.25,
                genetics_probation_max_realized_loss_pct_by_label={
                    "GeneticsRegimeAdaptiveBias": 0.15,
                },
            ),
            kill_switch=KillSwitchState(),
            real_perf=types.SimpleNamespace(
                get=lambda label: types.SimpleNamespace(
                    closed_trades=1,
                    pnl_pct=-0.20 if label == "GeneticsRegimeAdaptiveBias" else 0.10,
                ),
            ),
        )

        _disable_genetics_probation_if_realized_loss_exceeded(pipeline)

        self.assertIn(
            "GeneticsRegimeAdaptiveBias realized loss",
            _genetics_probation_disabled_reason(
                pipeline,
                "GeneticsRegimeAdaptiveBias",
            ),
        )
        self.assertEqual(_genetics_probation_disabled_reason(pipeline, "GeneticsCore"), "")


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

    def test_snapshot_persists_position_tracker_provenance(self):
        from panteon_v2.app.migration import load_v2_snapshot, save_v2_snapshot

        virtual = PerformanceMemory()
        tracker = PositionTracker()
        opened_at = datetime(2026, 5, 27, 7, 0, tzinfo=timezone.utc)
        tracker.force_set(TrackedPosition(
            open_signal_id=42,
            sym="BTC",
            side="long",
            entry_price=100.5,
            qty=0.25,
            fee_open=0.01,
            by_player="Panteon_Flash",
            by_agent="BullRotationAgent",
            opened_at=opened_at,
            opened_bar=5761,
            open_action="FUT_LONG_FULL",
            open_regime="bullish",
            funding_open=0.002,
        ))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            save_v2_snapshot(virtual, path, position_tracker=tracker)

            restored_tracker = PositionTracker()
            self.assertTrue(load_v2_snapshot(
                PerformanceMemory(),
                path,
                position_tracker=restored_tracker,
            ))

        restored = restored_tracker.get("BTC")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.open_signal_id, 42)
        self.assertEqual(restored.by_player, "Panteon_Flash")
        self.assertEqual(restored.by_agent, "BullRotationAgent")
        self.assertEqual(restored.opened_bar, 5761)
        self.assertEqual(restored.opened_at, opened_at)
        self.assertEqual(restored.open_action, "FUT_LONG_FULL")
        self.assertEqual(restored.open_regime, "bullish")
        self.assertAlmostEqual(restored.qty, 0.25)
        self.assertAlmostEqual(restored.entry_price, 100.5)

    def test_startup_recovery_preserves_existing_tracker_provenance(self):
        from panteon_v2.app.bootstrap import build_production_pipeline
        from panteon_v2.app.startup import _recover_exchange_positions
        from panteon_v2.selection import AgentRegistry

        class LiveExchange(FakeExchange):
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(
                        sym="BTC",
                        side="long",
                        qty=0.35,
                        entry=101.25,
                        leverage=2,
                    ),
                }

        registry = AgentRegistry()
        registry.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=LiveExchange(name="REAL"),
            initial_capital=1000.0,
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=42,
            sym="BTC",
            side="long",
            entry_price=100.5,
            qty=0.25,
            fee_open=0.01,
            by_player="Panteon_Flash",
            by_agent="BullRotationAgent",
            opened_at=datetime(2026, 5, 27, 7, 0, tzinfo=timezone.utc),
            opened_bar=5761,
            open_action="FUT_LONG_FULL",
            open_regime="bullish",
        ))

        recovered = _recover_exchange_positions(pipeline)

        restored = pipeline.executor._tracker.get("BTC")
        self.assertEqual(recovered, 1)
        self.assertEqual(restored.by_player, "Panteon_Flash")
        self.assertEqual(restored.by_agent, "BullRotationAgent")
        self.assertEqual(restored.open_signal_id, 42)
        self.assertEqual(restored.opened_bar, 5761)
        self.assertEqual(restored.open_action, "FUT_LONG_FULL")
        self.assertEqual(restored.open_regime, "bullish")
        self.assertAlmostEqual(restored.qty, 0.35)
        self.assertAlmostEqual(restored.entry_price, 101.25)

    def test_snapshot_persists_shadow_position_handoff(self):
        from panteon_v2.app.migration import load_v2_snapshot, save_v2_snapshot

        shadow_positions = {
            "Solo_BullRotationAgent": ({
                "sym": "BTC",
                "side": "long",
                "opened_bar": 5761,
                "age_bars": 3,
                "entry_price": 100.5,
                "current_price": 101.0,
                "qty": 0.25,
                "unrealized_pnl_usd": 0.125,
                "fresh": True,
            },),
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snapshot.json")
            save_v2_snapshot(
                PerformanceMemory(),
                path,
                shadow_positions=shadow_positions,
            )
            restored_positions = {}
            self.assertTrue(load_v2_snapshot(
                PerformanceMemory(),
                path,
                shadow_positions_target=restored_positions,
            ))

        self.assertEqual(restored_positions, shadow_positions)

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
