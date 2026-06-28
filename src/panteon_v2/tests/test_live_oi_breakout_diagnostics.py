from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "src" / "panteon_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from panteon_agents import LiveOIBreakout  # noqa: E402


def test_live_oi_breakout_reports_insufficient_history_reason():
    agent = LiveOIBreakout()
    agent.CHECK_INT = 1

    out = agent.act(
        {"BTC": 100.0},
        {"BTC": 10.0},
        bar_index=100,
    )

    assert out["BTC"] == 0
    assert agent.last_signal_diagnostics["BTC"]["reason"] == "insufficient_history"
    assert agent.last_signal_diagnostics["BTC"]["history_len"] == 1
    assert agent.last_signal_diagnostics["BTC"]["required_history"] == agent.MOM_N + 1


def test_live_oi_breakout_reports_check_interval_wait_reason():
    agent = LiveOIBreakout()
    agent.CHECK_INT = 60

    agent.act({"BTC": 100.0}, {"BTC": 10.0}, bar_index=100)
    out = agent.act({"BTC": 100.1}, {"BTC": 11.0}, bar_index=101)

    assert out["BTC"] == 0
    diag = agent.last_signal_diagnostics["BTC"]
    assert diag["reason"] == "check_interval_wait"
    assert diag["check_interval"] == 60
    assert diag["bars_since_check"] == 1
    assert diag["wait_bars"] == 59
