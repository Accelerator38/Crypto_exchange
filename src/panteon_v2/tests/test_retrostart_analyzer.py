from pathlib import Path
from unittest.mock import patch

import Retrostart_BITGET
import Retrostart_MEXC
import panteon_runtime.retrostart_analyzer as retrostart_analyzer


def test_analyze_session_uses_status_uptime_for_current_v2_session():
    with patch.object(
        retrostart_analyzer,
        "_load_json",
        return_value={"uptime": "05:40:56"},
    ):
        duration, sources = retrostart_analyzer._infer_duration_hours(
            None,
            Path("missing_all_signals.csv"),
            Path("missing_trading.log"),
            Path("status.json"),
        )

    assert duration > 5.6
    assert sources["status_uptime"] > 5.6


def test_analyze_session_uses_v2_status_panteon_owned_pnl_when_report_missing(tmp_path):
    session = tmp_path / "2026-05-15_08-14-56_v2"
    session.mkdir()
    (session / "trading.log").write_text(
        "2026-05-15 08:14:56 start\n2026-05-15 14:12:31 stop\n",
        encoding="utf-8",
    )
    (session / "status.json").write_text(
        """
{
  "version": "v2",
  "uptime": "05:57:34",
  "pnl_pct": -0.348355769560104,
  "live_session": {
    "panteon_owned_pnl_pct": -0.2033570076813642,
    "account_pnl_pct": -0.348355769560104
  }
}
""",
        encoding="utf-8",
    )

    diag = retrostart_analyzer.analyze_session(session)

    assert diag is not None
    assert diag["perf"]["real_pnl_pct"] == -0.2033570076813642
    assert diag["live_consistency"]["comparison_scope"] == "panteon_owned"
    assert diag["live_consistency"]["account_vs_panteon_delta_pct"] == -0.145
    assert diag["live_consistency"]["has_external_positions"] is False


def test_analyze_session_reports_external_position_scope_for_live_consistency(tmp_path):
    session = tmp_path / "2026-05-16_13-31-43_v2"
    session.mkdir()
    (session / "trading.log").write_text(
        "2026-05-16 13:31:43 start\n2026-05-16 20:31:43 stop\n",
        encoding="utf-8",
    )
    (session / "status.json").write_text(
        """
{
  "version": "v2",
  "uptime": "07:00:00",
  "live_session": {
    "account_pnl_pct": -1.25,
    "panteon_owned_pnl_pct": 0.0,
    "panteon_owned_positions_count": 0,
    "external_positions_count": 2,
    "external_unrealized_pnl_usd": -0.42,
    "comparison_scope": "panteon_owned"
  }
}
""",
        encoding="utf-8",
    )

    diag = retrostart_analyzer.analyze_session(session)

    assert diag is not None
    live = diag["live_consistency"]
    assert live["checked"] is True
    assert live["has_external_positions"] is True
    assert live["external_positions_count"] == 2
    assert live["account_vs_panteon_delta_pct"] == -1.25


def _diag(
    session: str,
    start: str,
    end: str,
    duration_hours: float,
    real_pnl: float,
) -> dict:
    return {
        "session": session,
        "duration_hours": duration_hours,
        "period_start": start,
        "period_end": end,
        "short_duration": duration_hours < retrostart_analyzer.MIN_DURATION_HOURS,
        "is_outlier": False,
        "has_shadow_players": True,
        "perf": {
            "real_pnl_pct": real_pnl,
            "panteon_shadow_pnl_pct": real_pnl / 2.0,
            "delta_real_vs_shadow_panteon": real_pnl / 2.0,
        },
        "replay": {
            "checked": True,
            "replay_pnl_pct": real_pnl - 0.1,
            "delta_replay_vs_report": -0.1,
            "rejected_duplicate_opens": 1,
            "avg_entry_risk_multiplier": 1.0,
            "risk_multiplier_pnl_impact": 0.0,
        },
        "signal_alignment": {"checked": False},
        "dup_opens": {"checked": True, "duplicate_open_events": 0},
        "regime_consistency": {"consistent": True},
    }


def test_aggregate_combines_short_sessions_with_small_gaps():
    diags = [
        _diag("s1", "2026-05-15T10:00:00", "2026-05-15T11:00:00", 1.0, 0.4),
        _diag("s2", "2026-05-15T11:30:00", "2026-05-15T12:45:00", 1.25, -0.1),
        _diag("s3", "2026-05-15T13:00:00", "2026-05-15T14:00:00", 1.0, 0.2),
        _diag("too_far", "2026-05-15T18:30:00", "2026-05-15T19:15:00", 0.75, 5.0),
    ]

    agg = retrostart_analyzer._aggregate(diags)

    assert agg["n_sessions"] == 0
    combined = agg["combined_periods"]
    assert combined["max_gap_hours"] == retrostart_analyzer.COMBINE_MAX_GAP_HOURS
    assert combined["n_periods"] == 1
    assert combined["n_sessions"] == 3
    period = combined["periods"][0]
    assert period["sessions"] == ["s1", "s2", "s3"]
    assert period["total_hours"] == 3.25
    assert period["gap_hours"] == [0.5, 0.25]
    assert period["cumulative_real_pnl_pct"] == 0.5
    assert period["cumulative_replay_pnl_pct"] == 0.2


def test_retrostart_entrypoints_treat_legacy_findings_as_success():
    with (
        patch.object(Retrostart_MEXC, "run_retrostart", return_value=1),
        patch.object(Retrostart_MEXC, "run_retrodate_whatif_for_exchange", return_value=0),
    ):
        assert Retrostart_MEXC.main(["--no-png"]) == 0

    with (
        patch.object(Retrostart_BITGET, "run_retrostart", return_value=1),
        patch.object(Retrostart_BITGET, "run_retrodate_whatif_for_exchange", return_value=0),
    ):
        assert Retrostart_BITGET.main(["--no-png"]) == 0


def test_retrostart_entrypoints_preserve_technical_failures():
    with (
        patch.object(Retrostart_MEXC, "run_retrostart", return_value=2),
        patch.object(Retrostart_MEXC, "run_retrodate_whatif_for_exchange", return_value=0),
    ):
        assert Retrostart_MEXC.main(["--no-png"]) == 2

    with (
        patch.object(Retrostart_BITGET, "run_retrostart", return_value=0),
        patch.object(Retrostart_BITGET, "run_retrodate_whatif_for_exchange", return_value=1),
    ):
        assert Retrostart_BITGET.main(["--no-png"]) == 1
