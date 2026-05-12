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


if __name__ == "__main__":
    unittest.main()
