"""Тесты Phase 7: парсер, synthesizer, validator."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from panteon_v2.replay.synthesizer import synthesize_v2_state
from panteon_v2.replay.v1_parser import (
    V1AgentEntry,
    V1Session,
    V1Signal,
    V1Trade,
    parse_all_signals_csv,
    parse_session,
    _action_to_code,
    _parse_timestamp,
)
from panteon_v2.replay.validator import (
    ReplayValidationReport,
    ValidationResult,
    validate_session,
)


class TestActionMapping(unittest.TestCase):
    def test_string_names(self):
        self.assertEqual(_action_to_code("hold"), 0)
        self.assertEqual(_action_to_code("spot_buy_half"), 1)
        self.assertEqual(_action_to_code("close_fut"), 8)
        self.assertEqual(_action_to_code("FL_FULL"), 5)

    def test_int_passthrough(self):
        self.assertEqual(_action_to_code(5), 5)
        self.assertEqual(_action_to_code("5"), 5)

    def test_unknown_returns_hold(self):
        self.assertEqual(_action_to_code("unknown"), 0)


class TestTimestampParsing(unittest.TestCase):
    def test_iso_format(self):
        ts = _parse_timestamp("2026-05-05T09:23:33.224708+00:00")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.year, 2026)

    def test_space_separator(self):
        ts = _parse_timestamp("2026-05-05 09:23:33.224708+00:00")
        self.assertIsNotNone(ts)

    def test_none(self):
        self.assertIsNone(_parse_timestamp(None))
        self.assertIsNone(_parse_timestamp(""))


class TestParseSessionFromMinimalJSON(unittest.TestCase):
    """Создаёт mini v1 сессию, парсит, ожидает корректный V1Session."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # status.json
        status = {
            "initial_capital": 100.0,
            "current_balance": 102.5,
            "pnl_usd": 2.5,
            "pnl_pct": 2.5,
            "bar_count": 5000,
            "live_bar_count": 100,
            "n_signals": 5,
            "n_trades": 2,
            "data_health": {"failed_orders": 1},
            "recent_signals": [
                {"bar": 1, "sym": "BTC", "action": "fl_full",
                 "price": 100.0, "selected_player": "V_X", "selected_agents": "X",
                 "time": "2026-05-05T09:00:00+00:00"},
                {"bar": 2, "sym": "BTC", "action": "close_fut",
                 "price": 110.0, "selected_player": "V_X", "selected_agents": "X",
                 "time": "2026-05-05T09:01:00+00:00"},
            ],
            "recent_trades": [
                {"sym": "BTC", "side": "LONG", "qty": 1.0, "price": 100.0,
                 "fee": 0.06, "funding": 0.0, "value": 100.0,
                 "time": "2026-05-05T09:00:01+00:00"},
            ],
        }
        with open(os.path.join(self.tmp, "status.json"), "w") as f:
            json.dump(status, f)
        # leaderboard_agents.json
        agents = {
            "metadata": {"regime": "bullish"},
            "agents": {
                "V_X": {"pnl_pct": 1.0, "closed_trades": 5, "win_rate": 60,
                        "sharpe": 0.5, "max_drawdown_pct": 1.0,
                        "status": "live", "status_reason": "",
                        "per_regime": {}},
                "V_BadAgent": {"pnl_pct": -1.0, "closed_trades": 5, "win_rate": 30,
                               "sharpe": -0.5, "max_drawdown_pct": 5.0,
                               "status": "quarantine", "status_reason": "test",
                               "per_regime": {}},
            },
        }
        with open(os.path.join(self.tmp, "leaderboard_agents.json"), "w") as f:
            json.dump(agents, f)
        # leaderboard_players.json
        players = {
            "metadata": {"regime": "bullish"},
            "players": {
                "V_X": {"pnl_pct": 1.0, "closed_trades": 5, "win_rate": 60,
                        "sharpe": 0.5, "max_drawdown_pct": 1.0,
                        "status": "live", "status_reason": "",
                        "per_regime": {}},
            },
        }
        with open(os.path.join(self.tmp, "leaderboard_players.json"), "w") as f:
            json.dump(players, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse(self):
        session = parse_session(self.tmp, prefer_csv=False)
        self.assertEqual(session.initial_capital, 100.0)
        self.assertEqual(session.pnl_pct, 2.5)
        self.assertEqual(session.current_regime, "bullish")
        self.assertEqual(len(session.real_signals), 2)
        self.assertEqual(len(session.recent_trades), 1)
        self.assertEqual(session.failed_orders, 1)
        # quarantined: BadAgent
        self.assertIn("BadAgent", session.quarantined)


class TestSynthesizer(unittest.TestCase):
    def _make_session(self, signals):
        """Helper для создания V1Session с указанными signals."""
        return V1Session(
            session_path="/test",
            exchange="MEXC",
            initial_capital=100.0,
            current_balance=100.0,
            pnl_usd=0.0,
            pnl_pct=0.0,
            bar_count=10,
            live_bar_count=10,
            n_signals=len(signals),
            n_trades=0,
            current_regime="bullish",
            real_signals=signals,
            recent_trades=[],
            agents={}, players={},
            quarantined=[],
        )

    def test_open_close_pair(self):
        signals = [
            V1Signal(bar=1, sym="BTC", action_code=5, action_name="fl_full",
                     price=100.0, timestamp=datetime.now(timezone.utc),
                     selected_player="P", selected_agents="A",
                     regime="bullish"),
            V1Signal(bar=2, sym="BTC", action_code=8, action_name="close_fut",
                     price=110.0, timestamp=datetime.now(timezone.utc),
                     selected_player="P", selected_agents="A",
                     regime="bullish"),
        ]
        session = self._make_session(signals)
        state = synthesize_v2_state(session)
        self.assertEqual(state.n_open, 1)
        self.assertEqual(state.n_close, 1)
        self.assertEqual(state.n_orphan_close, 0)
        # Realized PnL = (110 - 100) × 1 - fees ≈ +10
        attr = state.ledger.total_pnl_by_player()
        self.assertIn("P", attr)
        self.assertGreater(attr["P"], 9.0)  # close to +10 минус fees

    def test_orphan_close(self):
        signals = [
            V1Signal(bar=1, sym="BTC", action_code=8, action_name="close_fut",
                     price=110.0, timestamp=None,
                     selected_player="P", selected_agents="A",
                     regime="bullish"),
        ]
        session = self._make_session(signals)
        state = synthesize_v2_state(session)
        self.assertEqual(state.n_orphan_close, 1)

    def test_quarantine_seed_passed(self):
        session = V1Session(
            session_path="/test", exchange="MEXC",
            initial_capital=100.0, current_balance=100.0,
            pnl_usd=0.0, pnl_pct=0.0, bar_count=0, live_bar_count=0,
            n_signals=0, n_trades=0, current_regime="bullish",
            real_signals=[], recent_trades=[],
            agents={}, players={},
            quarantined=["FundingArb", "RichardDennis"],
        )
        state = synthesize_v2_state(session)
        self.assertTrue(state.qm.is_quarantined("FundingArb"))
        self.assertTrue(state.qm.is_quarantined("RichardDennis"))


class TestValidator(unittest.TestCase):
    def test_q1_pass_when_no_quarantined(self):
        session = V1Session(
            session_path="/x", exchange="MEXC",
            initial_capital=100, current_balance=100,
            pnl_usd=0, pnl_pct=0, bar_count=0, live_bar_count=0,
            n_signals=0, n_trades=0, current_regime="neutral",
            real_signals=[], recent_trades=[],
            agents={}, players={}, quarantined=[],
        )
        state = synthesize_v2_state(session)
        report = validate_session(session, state)
        # Должен PASS, потому что нет карантинных
        q1 = next(r for r in report.results if r.name.startswith("Q1"))
        self.assertTrue(q1.passed)

    def test_q1_fails_when_quarantined_player_signals(self):
        # selected_player="V_BadGuy" + BadGuy в карантине → Q1 FAIL
        sig = V1Signal(
            bar=1, sym="BTC", action_code=5, action_name="fl_full",
            price=100.0, timestamp=None,
            selected_player="V_BadGuy", selected_agents="A",
            regime="neutral",
        )
        session = V1Session(
            session_path="/x", exchange="MEXC",
            initial_capital=100, current_balance=100,
            pnl_usd=0, pnl_pct=0, bar_count=1, live_bar_count=1,
            n_signals=1, n_trades=0, current_regime="neutral",
            real_signals=[sig], recent_trades=[],
            agents={"V_A": V1AgentEntry(
                label="V_A", pnl_pct=0, closed_trades=0, win_rate=0,
                sharpe=0, max_dd_pct=0, status="live", status_reason="",
                per_regime={},
            )},
            players={"V_BadGuy": V1AgentEntry(
                label="V_BadGuy", pnl_pct=-1, closed_trades=10, win_rate=20,
                sharpe=-1, max_dd_pct=5, status="quarantine",
                status_reason="bad", per_regime={},
            )},
            quarantined=["BadGuy"],
        )
        state = synthesize_v2_state(session)
        report = validate_session(session, state)
        q1 = next(r for r in report.results if r.name.startswith("Q1"))
        self.assertFalse(q1.passed)
        self.assertIn("BadGuy", q1.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
