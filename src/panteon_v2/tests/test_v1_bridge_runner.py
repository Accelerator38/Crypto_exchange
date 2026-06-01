"""Tests for v1 bridge market-feed integration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from panteon_v2.domain.types import Regime


class FakeRuntime:
    api_key_env = "FAKE_API_KEY"
    api_secret_env = "FAKE_SECRET_KEY"
    api_passphrase_env = "FAKE_PASSPHRASE"

    def load_settings(self):
        return {"initial_capital": "100"}

    def parse_settings(self, raw_cfg):
        return {
            "initial_capital": float(raw_cfg["initial_capital"]),
            "trade_fraction": 0.05,
            "leverage": 2,
            "poll_interval": 5,
            "liquidity_min_adv": 0,
            "spot_fee": 0.001,
            "futures_fee": 0.0002,
            "slippage": 0.0,
            "symbols": ["BTC"],
            "paper_capital": 100,
        }

    def resolve_bridge_class(self):
        return FakeBridge

    def create_direct_client(self, api_key, api_secret, api_passphrase=""):
        return {
            "api_key": api_key,
            "api_secret": api_secret,
            "api_passphrase": api_passphrase,
        }


class FakeBridge:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._bar = 10
        self._price_hist = []
        self._volume_hist = []
        self.warmup_calls = []
        self.fetch_calls = 0
        FakeBridge.instances.append(self)

    def warmup(self, n_bars):
        self.warmup_calls.append(n_bars)

    def _fetch_market(self):
        self.fetch_calls += 1
        return {"BTC": 100.0}, {"BTC": 42.0}

    def _run_cycle(self):
        raise AssertionError("v2 feed must not execute v1 trading cycle")

    def run_once(self):
        raise AssertionError("v2 feed must not execute v1 trading cycle")


class V1BridgeRunnerTests(unittest.TestCase):
    def setUp(self):
        self._old_exchange_registry = sys.modules.get("exchange_registry")
        fake_registry = types.ModuleType("exchange_registry")
        fake_registry.load_exchange_runtime = lambda exchange_name: FakeRuntime()
        sys.modules["exchange_registry"] = fake_registry
        os.environ["FAKE_API_KEY"] = "key"
        os.environ["FAKE_SECRET_KEY"] = "secret"
        os.environ["FAKE_PASSPHRASE"] = "pass"
        FakeBridge.instances.clear()

    def tearDown(self):
        if self._old_exchange_registry is None:
            sys.modules.pop("exchange_registry", None)
        else:
            sys.modules["exchange_registry"] = self._old_exchange_registry
        for name in ("FAKE_API_KEY", "FAKE_SECRET_KEY", "FAKE_PASSPHRASE"):
            os.environ.pop(name, None)

    def test_create_bridge_uses_exchange_runtime_bridge_class(self):
        from panteon_v2.app.v1_bridge_runner import _create_bridge

        bridge = _create_bridge(
            "MEXC",
            mode="live_futures",
            output_dir="out-dir",
            initial_capital=40.25,
        )

        self.assertIsInstance(bridge, FakeBridge)
        self.assertEqual(bridge.kwargs["agents"], {})
        self.assertEqual(bridge.kwargs["cfg"]["initial_capital"], 40.25)
        self.assertEqual(bridge.kwargs["cfg"]["paper_capital"], 40.25)
        self.assertEqual(bridge.kwargs["mode"], "live_futures")
        self.assertEqual(bridge.kwargs["api_key"], "key")
        self.assertEqual(bridge.kwargs["api_secret"], "secret")
        self.assertEqual(bridge.kwargs["api_passphrase"], "pass")
        self.assertEqual(bridge.kwargs["output_dir"], "out-dir")
        self.assertEqual(bridge.kwargs["direct_client"]["api_key"], "key")

    def test_bridge_feed_uses_fetch_market_without_v1_order_cycle(self):
        from panteon_v2.app.v1_bridge_runner import V1BridgeFeed

        bridge = FakeBridge()
        feed = V1BridgeFeed(bridge, exchange_name="MEXC")

        snap = feed.next_bar()

        self.assertEqual(bridge.warmup_calls, [5760])
        self.assertEqual(bridge.fetch_calls, 1)
        self.assertEqual(snap.bar, 11)
        self.assertEqual(snap.prices, {"BTC": 100.0})
        self.assertEqual(snap.volumes, {"BTC": 42.0})
        self.assertEqual(bridge._price_hist[-1], {"BTC": 100.0})
        self.assertEqual(bridge._volume_hist[-1], {"BTC": 42.0})

    def test_bridge_feed_passes_funding_rates_into_market_snapshot(self):
        from panteon_v2.app.v1_bridge_runner import V1BridgeFeed

        class FundingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self.funding = type(
                    "Funding",
                    (),
                    {
                        "funding_rate": lambda self, symbol: {
                            "BTC": 0.001,
                            "ETH": -0.0005,
                        }.get(symbol, 0.0)
                    },
                )()

            def _fetch_market(self):
                self.fetch_calls += 1
                return {"BTC": 100.0, "ETH": 50.0}, {"BTC": 42.0, "ETH": 21.0}

        bridge = FundingBridge()
        feed = V1BridgeFeed(bridge, exchange_name="MEXC")

        snap = feed.next_bar()

        self.assertEqual(snap.funding, {"BTC": 0.001, "ETH": -0.0005})

    def test_bridge_feed_detects_regime_from_market_history_without_bridge_agents(self):
        from panteon_v2.app.v1_bridge_runner import V1BridgeFeed

        class RisingBridge(FakeBridge):
            def _fetch_market(self):
                self.fetch_calls += 1
                return {"BTC": 115.0}, {"BTC": 42.0}

        bridge = RisingBridge()
        bridge._price_hist = [
            {"BTC": 100.0},
            {"BTC": 105.0},
            {"BTC": 110.0},
        ]
        bridge.agents = {}
        feed = V1BridgeFeed(bridge, exchange_name="MEXC")

        snap = feed.next_bar()

        self.assertEqual(snap.regime, Regime.BULLISH)

    def test_bridge_feed_populates_technical_indicators_from_warmup_history(self):
        from panteon_v2.app.v1_bridge_runner import V1BridgeFeed

        class TrendingBridge(FakeBridge):
            def __init__(self):
                super().__init__()
                self._price_hist = [
                    {"BTC": 100.0 + idx}
                    for idx in range(40)
                ]
                self._volume_hist = [
                    {"BTC": 10.0 + idx}
                    for idx in range(40)
                ]

            def _fetch_market(self):
                self.fetch_calls += 1
                return {"BTC": 140.0}, {"BTC": 50.0}

        bridge = TrendingBridge()
        feed = V1BridgeFeed(bridge, exchange_name="MEXC")

        snap = feed.next_bar()

        tech = snap.technicals_by_symbol["BTC"]
        self.assertIsNotNone(tech.rsi_14)
        self.assertIsNotNone(tech.macd_histogram_pct)
        self.assertIsNotNone(tech.atr_14_pct)
        self.assertGreater(tech.atr_14_pct, 0.0)

    def test_bitget_bridge_feed_excludes_non_contract_symbols_before_snapshot(self):
        from panteon_v2.app.v1_bridge_runner import V1BridgeFeed

        class FakeFuturesClient:
            def __init__(self):
                self._bad_symbols = set()

            def _get_contract_meta(self, symbol):
                if symbol == "SPOTONLY":
                    return {
                        "apiAllowed": False,
                        "state": "offline",
                        "amountStep": 1.0,
                        "metadataSource": "exchange",
                    }
                return {
                    "apiAllowed": True,
                    "state": "normal",
                    "amountStep": 1.0,
                    "metadataSource": "exchange",
                }

        class BitgetBridge(FakeBridge):
            mode = "live_futures"

            def __init__(self):
                super().__init__()
                self.futures_client = FakeFuturesClient()

            def _fetch_market(self):
                self.fetch_calls += 1
                return {"BTC": 100.0, "SPOTONLY": 1.0}, {"BTC": 42.0, "SPOTONLY": 99.0}

        bridge = BitgetBridge()
        feed = V1BridgeFeed(bridge, exchange_name="BITGET")

        with self.assertLogs("panteon_v2.app.v1_bridge_runner", level="WARNING") as logs:
            snap = feed.next_bar()

        self.assertEqual(snap.prices, {"BTC": 100.0})
        self.assertEqual(snap.volumes, {"BTC": 42.0})
        self.assertIn("SPOTONLY", bridge.futures_client._bad_symbols)
        self.assertTrue(any("excluded before selector/risk" in line for line in logs.output))

    def test_warmup_replays_bridge_history_into_v2_agents(self):
        from panteon_v2.app.v1_bridge_runner import _warmup_v2_agents_from_bridge
        from panteon_v2.selection import AgentRegistry

        class RecordingAgent:
            label = "Recorder"

            def __init__(self):
                self.calls = []

            def act(self, market):
                self.calls.append((
                    market.bar,
                    dict(market.prices),
                    dict(market.volumes),
                    market.regime,
                ))
                return {}

        bridge = FakeBridge()
        bridge._price_hist = [
            {"BTC": 100.0},
            {"BTC": 101.0},
            {"BTC": 102.0},
        ]
        bridge._volume_hist = [
            {"BTC": 10.0},
            {"BTC": 11.0},
            {"BTC": 12.0},
        ]
        registry = AgentRegistry()
        agent = RecordingAgent()
        registry.register(agent)

        warmed = _warmup_v2_agents_from_bridge(
            bridge,
            registry,
            exchange_name="MEXC",
        )

        self.assertEqual(warmed, 3)
        self.assertEqual(
            agent.calls,
            [
                (1, {"BTC": 100.0}, {"BTC": 10.0}, Regime.NEUTRAL),
                (2, {"BTC": 101.0}, {"BTC": 11.0}, Regime.BULLISH),
                (3, {"BTC": 102.0}, {"BTC": 12.0}, Regime.BULLISH),
            ],
        )

    def test_warmup_skips_heavy_optional_genetics_agents(self):
        from panteon_v2.app.v1_bridge_runner import _warmup_v2_agents_from_bridge
        from panteon_v2.selection import AgentRegistry

        class RecordingAgent:
            def __init__(self, label):
                self.label = label
                self.calls = 0

            def act(self, market):
                self.calls += 1
                return {}

        bridge = FakeBridge()
        bridge._price_hist = [
            {"BTC": 100.0},
            {"BTC": 101.0},
        ]
        registry = AgentRegistry()
        normal = RecordingAgent("LiveTrendFollow")
        genetics = RecordingAgent("GeneticsCore")
        registry.register(normal)
        registry.register(genetics)

        warmed = _warmup_v2_agents_from_bridge(
            bridge,
            registry,
            exchange_name="MEXC",
        )

        self.assertEqual(warmed, 2)
        self.assertEqual(normal.calls, 2)
        self.assertEqual(genetics.calls, 0)

    def test_prepare_live_agents_clears_warmup_positions_and_injects_real_positions(self):
        from panteon_v2.app.v1_bridge_runner import (
            _prepare_v2_agents_for_live_after_warmup,
            _warmup_v2_agents_from_bridge,
        )
        from panteon_v2.selection import AgentRegistry

        class StatefulAgent:
            label = "Stateful"

            def __init__(self):
                self.h = {}
                self.pos = {}
                self.ep = {}
                self.et = {}
                self.reset_calls = []

            def act(self, market):
                for sym, price in market.prices.items():
                    self.h.setdefault(sym, []).append(price)
                    if price >= 102.0:
                        self.pos[sym] = "long"
                        self.ep[sym] = price
                        self.et[sym] = market.bar
                return {}

            def reset_for_live(self, bar_index):
                self.reset_calls.append(bar_index)
                for sym in list(self.pos):
                    self.pos[sym] = None
                for sym in list(self.ep):
                    self.ep[sym] = 0.0
                for sym in list(self.et):
                    self.et[sym] = 0

        class Adapter:
            label = "WrappedStateful"

            def __init__(self, inner):
                self.v1_agent = inner

            def act(self, market):
                return self.v1_agent.act(market)

        class Tracker:
            def all_open(self):
                return {
                    "ETH": types.SimpleNamespace(
                        sym="ETH",
                        side="short",
                        entry_price=222.0,
                        qty=0.4,
                    ),
                }

        class Executor:
            _tracker = Tracker()

        class Pipeline:
            executor = Executor()

            def __init__(self, registry):
                self.registry = registry

        bridge = FakeBridge()
        bridge._price_hist = [
            {"BTC": 100.0},
            {"BTC": 101.0},
            {"BTC": 102.0},
        ]
        registry = AgentRegistry()
        inner = StatefulAgent()
        registry.register(Adapter(inner))

        _warmup_v2_agents_from_bridge(
            bridge,
            registry,
            exchange_name="MEXC",
        )
        self.assertEqual(inner.pos["BTC"], "long")
        self.assertEqual(inner.h["BTC"], [100.0, 101.0, 102.0])

        summary = _prepare_v2_agents_for_live_after_warmup(
            Pipeline(registry),
            exchange_name="MEXC",
            bar_index=3,
        )

        self.assertEqual(summary["agents"], 1)
        self.assertEqual(summary["real_positions"], 1)
        self.assertEqual(inner.reset_calls, [3])
        self.assertEqual(inner.h["BTC"], [100.0, 101.0, 102.0])
        self.assertIsNone(inner.pos["BTC"])
        self.assertEqual(inner.ep["BTC"], 0.0)
        self.assertEqual(inner.et["BTC"], 0)
        self.assertEqual(inner.pos["ETH"], "short")
        self.assertEqual(inner.ep["ETH"], 222.0)
        self.assertEqual(inner.et["ETH"], 3)

    def test_account_log_fields_include_balance_and_position_count(self):
        from panteon_v2.app.v1_bridge_runner import _account_log_fields

        class Exchange:
            def get_all_positions(self):
                return {"BTC": object()}

        class Executor:
            _exchange = Exchange()

        class Pipeline:
            current_balance = 120.5
            account_snapshot = {"total_assets": 128.0}
            executor = Executor()

        self.assertEqual(
            _account_log_fields(Pipeline()),
            "balance=$120.50 positions=1",
        )

    def test_run_with_v1_bridge_publishes_warmup_heartbeats(self):
        from panteon_v2.app.v1_bridge_runner import run_with_v1_bridge
        from panteon_v2.app.bootstrap import build_production_pipeline
        from panteon_v2.execution import FakeExchange
        from panteon_v2.selection import AgentRegistry
        from panteon_v2.tests._helpers import FakeAgent

        class Writer:
            output_dir = "out-dir"

            def __init__(self):
                self.heartbeats = []
                self.steps = []
                self.closed = False

            def write_heartbeat(self, **kwargs):
                self.heartbeats.append(kwargs)

            def write(self, step):
                self.steps.append(step)

            def close(self):
                self.closed = True

        registry = AgentRegistry()
        registry.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="MEXC"),
            initial_capital=100.0,
        )
        bridge = FakeBridge()
        bridge.warmup_bars = 12
        writer = Writer()

        rc = run_with_v1_bridge(
            pipeline,
            exchange_name="MEXC",
            bridge=bridge,
            output_writer=writer,
            max_bars=1,
            sleep_between_polls_sec=0.0,
        )

        self.assertEqual(rc, 0)
        self.assertEqual(
            [hb["feed_status"] for hb in writer.heartbeats],
            ["warmup_loading", "agent_warmup", "active"],
        )
        self.assertIn("12 historical bars", writer.heartbeats[0]["message"])
        self.assertEqual(len(writer.steps), 1)
        self.assertTrue(writer.closed)

    def test_run_with_v1_bridge_uses_cached_warmup_without_remote_warmup(self):
        from panteon_v2.app.v1_bridge_runner import run_with_v1_bridge
        from panteon_v2.app.bootstrap import build_production_pipeline
        from panteon_v2.execution import FakeExchange
        from panteon_v2.selection import AgentRegistry
        from panteon_v2.tests._helpers import FakeAgent

        class Writer:
            output_dir = "out-dir"

            def __init__(self):
                self.heartbeats = []
                self.steps = []
                self.closed = False

            def write_heartbeat(self, **kwargs):
                self.heartbeats.append(kwargs)

            def write(self, step):
                self.steps.append(step)

            def close(self):
                self.closed = True

        registry = AgentRegistry()
        registry.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="MEXC"),
            initial_capital=100.0,
        )
        bridge = FakeBridge()
        bridge.symbols = ["BTC"]
        writer = Writer()

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "mexc_bridge_cache.json")
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "schema_version": 1,
                    "exchange": "MEXC",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bar": 99,
                    "symbols": ["BTC"],
                    "price_hist": [{"BTC": 100.0}, {"BTC": 101.0}],
                    "volume_hist": [{"BTC": 10.0}, {"BTC": 11.0}],
                }, fh)

            rc = run_with_v1_bridge(
                pipeline,
                exchange_name="MEXC",
                bridge=bridge,
                output_writer=writer,
                max_bars=1,
                sleep_between_polls_sec=0.0,
                bridge_cache_path=cache_path,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(bridge.warmup_calls, [])
        self.assertEqual(bridge._bar, 100)
        self.assertEqual(bridge._price_hist[:2], [{"BTC": 100.0}, {"BTC": 101.0}])
        self.assertEqual(
            [hb["feed_status"] for hb in writer.heartbeats],
            ["warmup_cached", "agent_warmup", "active"],
        )
        self.assertIn("cached warmup", writer.heartbeats[0]["message"])
        self.assertEqual(len(writer.steps), 1)

    def test_bridge_warmup_cache_rejects_stale_or_symbol_mismatch(self):
        from panteon_v2.app.v1_bridge_runner import restore_bridge_state_cache

        bridge = FakeBridge()
        bridge.symbols = ["BTC", "ETH"]

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "mexc_bridge_cache.json")
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "schema_version": 1,
                    "exchange": "MEXC",
                    "generated_at": "2026-05-27T07:00:00+00:00",
                    "bar": 99,
                    "symbols": ["BTC"],
                    "price_hist": [{"BTC": 100.0}],
                    "volume_hist": [{"BTC": 10.0}],
                }, fh)

            status = restore_bridge_state_cache(
                bridge,
                cache_path,
                exchange_name="MEXC",
                max_age_sec=3600,
                now=datetime(2026, 5, 27, 7, 1, tzinfo=timezone.utc),
            )

        self.assertFalse(status["restored"])
        self.assertEqual(status["reason"], "symbols_changed")

    def test_run_with_v1_bridge_reports_processed_steps_after_keyboard_interrupt(self):
        from panteon_v2.app.v1_bridge_runner import run_with_v1_bridge
        from panteon_v2.app.bootstrap import build_production_pipeline
        from panteon_v2.execution import FakeExchange
        from panteon_v2.selection import AgentRegistry
        from panteon_v2.tests._helpers import FakeAgent

        class Writer:
            output_dir = "out-dir"

            def __init__(self):
                self.steps = []
                self.closed = False

            def write_heartbeat(self, **kwargs):
                pass

            def write(self, step):
                self.steps.append(step)

            def close(self):
                self.closed = True

        def interrupted_loop(*args, on_step=None, **kwargs):
            if on_step:
                on_step(types.SimpleNamespace(
                    bar=11,
                    leader="DefaultEnsemble",
                    n_signals=0,
                    n_filled=0,
                    n_rejected=0,
                    n_blocked=0,
                    n_filtered_real_signals=0,
                ))
            raise KeyboardInterrupt()

        registry = AgentRegistry()
        registry.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="MEXC"),
            initial_capital=100.0,
        )
        writer = Writer()

        with patch("panteon_v2.app.v1_bridge_runner.main_loop", side_effect=interrupted_loop):
            with self.assertLogs("panteon_v2.app.v1_bridge_runner", level="INFO") as logs:
                rc = run_with_v1_bridge(
                    pipeline,
                    exchange_name="MEXC",
                    bridge=FakeBridge(),
                    output_writer=writer,
                    sleep_between_polls_sec=0.0,
                )

        self.assertEqual(rc, 0)
        self.assertTrue(writer.closed)
        self.assertEqual(len(writer.steps), 1)
        self.assertTrue(any("processed 1 bars" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
