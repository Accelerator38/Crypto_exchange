"""Tests for v1 bridge market-feed integration."""

from __future__ import annotations

import os
import sys
import types
import unittest


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

        bridge = _create_bridge("MEXC", mode="live_futures", output_dir="out-dir")

        self.assertIsInstance(bridge, FakeBridge)
        self.assertEqual(bridge.kwargs["agents"], {})
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


if __name__ == "__main__":
    unittest.main()
