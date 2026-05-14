import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "panteon_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from agent_safety import PositionExitGovernor  # noqa: E402
from exchange_api_runtime import (  # noqa: E402
    POSITION_SOURCE_PANTEON_ORDER,
    POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP,
    PositionSyncHealth,
    MexcDirectClient,
    recovered_position_policy,
    safe_reconcile_open_positions,
)


def _load_build_dashboard():
    path = ROOT / "tools" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PositionExitGovernorTests(unittest.TestCase):
    def test_position_specific_stop_loss_override_closes_inherited_position(self):
        book = {
            "AVNT": {
                "entry": 100.0,
                "side": "long",
                "bar": 10,
                "peak": 100.0,
                "external": True,
                "tight_sl_pct": 0.015,
            }
        }

        out = PositionExitGovernor().apply_exit_actions(
            book,
            {"AVNT": 98.0},
            current_bar=11,
            stop_loss_pct=0.04,
            take_profit_pct=0.06,
            trail_pct=0.025,
            stale_bars=720,
        )

        self.assertEqual(out, {"AVNT": 8})
        self.assertNotIn("AVNT", book)

    def test_position_specific_stale_override_closes_quiet_inherited_position(self):
        book = {
            "LAB": {
                "entry": 100.0,
                "side": "long",
                "bar": 10,
                "peak": 100.0,
                "external": True,
                "stale_bars": 0,
                "stale_move_threshold": 0.06,
            }
        }

        out = PositionExitGovernor().apply_exit_actions(
            book,
            {"LAB": 103.0},
            current_bar=11,
            stop_loss_pct=0.04,
            take_profit_pct=0.06,
            trail_pct=0.025,
            stale_bars=720,
        )

        self.assertEqual(out, {"LAB": 8})
        self.assertNotIn("LAB", book)

    def test_recovered_snapshot_gap_position_is_not_closed_by_zero_stale_override(self):
        book = {
            "LAB": {
                "entry": 100.0,
                "side": "long",
                "bar": 10,
                "peak": 100.0,
                "external": True,
                "source": POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP,
                "stale_bars": 0,
                "stale_move_threshold": 0.06,
                "suppress_stale_until_bar": 130,
            }
        }

        out = PositionExitGovernor().apply_exit_actions(
            book,
            {"LAB": 103.0},
            current_bar=11,
            stop_loss_pct=0.04,
            take_profit_pct=0.06,
            trail_pct=0.025,
            stale_bars=720,
        )

        self.assertEqual(out, {})
        self.assertIn("LAB", book)


class RuntimeHttpClientTests(unittest.TestCase):
    def test_mexc_runtime_http_ignores_environment_proxy(self):
        import exchange_api_runtime
        import mexc_connector

        direct = MexcDirectClient("key", "secret")
        spot = mexc_connector.MexcSpotClient("key", "secret")
        futures = mexc_connector.MexcFuturesClient("key", "secret")

        self.assertFalse(exchange_api_runtime.MEXC_DIRECT_HTTP.trust_env)
        self.assertFalse(mexc_connector.MEXC_CONNECTOR_HTTP.trust_env)
        self.assertFalse(direct._session.trust_env)
        self.assertFalse(spot.s.trust_env)
        self.assertFalse(futures.s.trust_env)

    def test_mexc_futures_post_sends_the_exact_signed_json_body(self):
        import mexc_connector

        captured = {}

        class FakeResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {"code": 200, "data": "ok"}

        class FakeSession:
            trust_env = False

            def request(self, method, url, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        client = mexc_connector.MexcFuturesClient("key", "secret")
        client.s = FakeSession()

        client._req("POST", "/api/v1/private/order/submit", {"b": 2, "a": 1})

        self.assertEqual(captured["data"], '{"b":2,"a":1}')
        self.assertNotIn("json", captured)


class PositionReconcileTests(unittest.TestCase):
    def test_reconcile_requires_repeated_healthy_zero_snapshots_before_removal(self):
        book = {
            "BTC": {
                "entry": 100.0,
                "side": "long",
                "bar": 1,
                "peak": 100.0,
                "source": POSITION_SOURCE_PANTEON_ORDER,
            }
        }
        health = PositionSyncHealth(zero_confirmations=3, now_fn=lambda: 100.0)

        first = safe_reconcile_open_positions(book, [], current_bar=20, health=health)
        second = safe_reconcile_open_positions(book, [], current_bar=21, health=health)

        self.assertIn("BTC", book)
        self.assertEqual(first["removed"], 0)
        self.assertEqual(second["removed"], 0)

        third = safe_reconcile_open_positions(book, [], current_bar=22, health=health)

        self.assertNotIn("BTC", book)
        self.assertEqual(third["removed"], 1)

    def test_recent_data_error_blocks_destructive_reconcile(self):
        now = [100.0]
        book = {
            "ETH": {
                "entry": 100.0,
                "side": "long",
                "bar": 1,
                "peak": 100.0,
                "source": POSITION_SOURCE_PANTEON_ORDER,
            }
        }
        health = PositionSyncHealth(zero_confirmations=2, cooldown_sec=120, now_fn=lambda: now[0])
        health.mark_data_error("price_error")

        for bar in range(20, 25):
            safe_reconcile_open_positions(book, [], current_bar=bar, health=health)

        self.assertIn("ETH", book)

    def test_recovered_position_uses_recovered_source_and_long_stale_policy(self):
        book = {
            "ARB": {
                "entry": 100.0,
                "side": "long",
                "bar": 10,
                "peak": 100.0,
                "source": POSITION_SOURCE_PANTEON_ORDER,
            }
        }
        health = PositionSyncHealth(zero_confirmations=3, now_fn=lambda: 100.0)

        safe_reconcile_open_positions(book, [], current_bar=20, health=health)
        safe_reconcile_open_positions(
            book,
            [{"symbol": "ARB", "side": "long", "entry": 100.0, "leverage": 2}],
            current_bar=21,
            health=health,
        )

        self.assertEqual(book["ARB"]["source"], POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP)
        self.assertGreaterEqual(int(book["ARB"]["stale_bars"]), 120)
        self.assertEqual(
            recovered_position_policy(current_bar=21)["source"],
            POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP,
        )


class DashboardBuilderTests(unittest.TestCase):
    def test_build_dashboard_collects_latest_date_named_sessions(self):
        dashboard = _load_build_dashboard()
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "Results"
            session_dir = results / "BITGET" / "2027-01-02_03-04-05"
            session_dir.mkdir(parents=True)
            (results / "MEXC").mkdir(parents=True)
            (session_dir / "status.json").write_text(
                json.dumps({"pnl_pct": 1.23, "open_positions": {"BTC": {"side": "LONG"}}}),
                encoding="utf-8",
            )

            out_html = results / "dashboard.html"
            payload = dashboard.build_dashboard(
                results_dir=results,
                out_html=out_html,
                quiet=True,
            )

            self.assertEqual(payload["BITGET"][0]["name"], "2027-01-02_03-04-05")
            self.assertIn("2027-01-02_03-04-05", out_html.read_text(encoding="utf-8"))

    def test_build_dashboard_collects_v2_suffixed_sessions(self):
        dashboard = _load_build_dashboard()
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "Results"
            session_dir = results / "BITGET" / "2027-01-02_03-04-05_v2"
            session_dir.mkdir(parents=True)
            (results / "MEXC").mkdir(parents=True)
            (session_dir / "status.json").write_text(
                json.dumps({"version": "v2", "pnl_pct": 1.23, "bar_count": 7}),
                encoding="utf-8",
            )

            out_html = results / "dashboard.html"
            payload = dashboard.build_dashboard(
                results_dir=results,
                out_html=out_html,
                quiet=True,
            )

            self.assertEqual(payload["BITGET"][0]["name"], "2027-01-02_03-04-05_v2")
            self.assertEqual(payload["BITGET"][0]["summary"]["version"], "v2")
            self.assertIn("2027-01-02_03-04-05_v2", out_html.read_text(encoding="utf-8"))


class WarmupDashboardTests(unittest.TestCase):
    def _sample_warmup(self):
        prices_bars = []
        volumes_bars = []
        symbols = ["BTC", "ETH", "SOL"]
        for idx in range(12):
            prices_bars.append({
                "BTC": 100.0 + idx,
                "ETH": 50.0 + idx * 0.5,
                "SOL": 20.0 - idx * 0.1,
            })
            volumes_bars.append({
                "BTC": 1000.0 + idx * 10,
                "ETH": 700.0 + idx * 7,
                "SOL": 300.0 + idx * 3,
            })
        return prices_bars, volumes_bars, symbols

    def test_warmup_renderer_writes_single_combined_png(self):
        from mexc_connector import plot_warmup_dashboard

        prices_bars, volumes_bars, symbols = self._sample_warmup()

        with tempfile.TemporaryDirectory() as td:
            plot_warmup_dashboard(prices_bars, volumes_bars, symbols, td, tf_kline="1m")

            self.assertTrue((Path(td) / "warmup_dashboard.png").exists())
            self.assertFalse((Path(td) / "warmup_data.png").exists())
            self.assertFalse((Path(td) / "warmup_market_analysis.png").exists())

    def test_legacy_warmup_market_analysis_uses_combined_png(self):
        from mexc_connector import plot_warmup_market_analysis

        prices_bars, volumes_bars, symbols = self._sample_warmup()

        with tempfile.TemporaryDirectory() as td:
            plot_warmup_market_analysis(prices_bars, volumes_bars, symbols, td, tf_kline="1m", bar=12)

            self.assertTrue((Path(td) / "warmup_dashboard.png").exists())
            self.assertFalse((Path(td) / "warmup_market_analysis.png").exists())


if __name__ == "__main__":
    unittest.main()
