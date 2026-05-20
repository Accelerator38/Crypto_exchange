"""Tests for Retrodate market benchmark helpers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import panteon_v2.analysis.retrodate_market_runner as runner
from panteon_v2.analysis.soft_allocator import write_shadow_pnl_events
from panteon_v2.analysis.retrodate_market_runner import (
    RetrodateFileSelection,
    RetrodateMarketConfig,
    RetrodateRunSummary,
    RetrodateSnapshotState,
    load_retrodate_year_snapshots,
    select_retrodate_files,
)
from panteon_v2.analysis.retrodate_validator import RetrodateDirReport
from panteon_v2.analysis.retrodate_validator import RetrodateValidationError
from panteon_v2.attribution import CandidateRejected, CandidateScored, ShadowActorUpdated
from panteon_v2.domain.types import Regime


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume", "symbol", "datetime"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(timestamp: int, close: float, symbol: str) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": close * 10,
        "symbol": symbol,
        "datetime": "",
    }


def test_load_retrodate_year_snapshots_builds_stride_multi_symbol_bars(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    _write_rows(
        csv_path,
        [
            _row(1735689600000, 100.0, "BTC/USDT"),
            _row(1735689600000, 10.0, "ETH/USDT"),
            _row(1735691400000, 999.0, "BTC/USDT"),
            _row(1735693200000, 110.0, "BTC/USDT"),
            _row(1735693200000, 11.0, "ETH/USDT"),
        ],
    )

    snapshots = load_retrodate_year_snapshots(csv_path, stride_minutes=60, state=RetrodateSnapshotState())

    assert [snap.bar for snap in snapshots] == [1, 2]
    assert snapshots[0].prices == {"BTC/USDT": 100.0, "ETH/USDT": 10.0}
    assert snapshots[0].volumes == {"BTC/USDT": 1000.0, "ETH/USDT": 100.0}
    assert snapshots[0].timestamp.isoformat() == "2025-01-01T00:00:00+00:00"
    assert snapshots[0].month == 1
    assert snapshots[1].prices["BTC/USDT"] == 110.0


def test_load_retrodate_year_snapshots_keeps_state_across_files(tmp_path):
    first = tmp_path / "crypto_1m_2025_all_symbols.csv"
    second = tmp_path / "crypto_1m_2026_all_symbols.csv"
    _write_rows(first, [_row(1735689600000, 100.0, "BTC/USDT")])
    _write_rows(second, [_row(1767225600000, 120.0, "BTC/USDT")])
    state = RetrodateSnapshotState()

    first_snapshots = load_retrodate_year_snapshots(first, stride_minutes=60, state=state)
    second_snapshots = load_retrodate_year_snapshots(second, stride_minutes=60, state=state)

    assert first_snapshots[0].bar == 1
    assert second_snapshots[0].bar == 2


def test_select_retrodate_files_excludes_invalid_requested_years(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2025_all_symbols.csv",
        [_row(1735689600000, 100.0, "BTC/USDT")],
    )
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, 100.0, "BTC/USDT")],
    )

    selection = select_retrodate_files(
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025, 2026), stride_minutes=60)
    )

    assert [path.name for path in selection.valid_files] == ["crypto_1m_2025_all_symbols.csv"]
    assert [item.path.name for item in selection.excluded_files] == ["crypto_1m_2026_all_symbols.csv"]
    assert selection.executed_years == (2025,)
    assert selection.missing_years == (2026,)


def test_select_retrodate_files_can_fail_on_invalid_requested_years(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, 100.0, "BTC/USDT")],
    )

    with pytest.raises(RetrodateValidationError):
        select_retrodate_files(
            RetrodateMarketConfig(
                data_dir=tmp_path,
                years=(2026,),
                stride_minutes=60,
                invalid_policy="fail",
            )
        )


def test_retrodate_regime_classifier_detects_large_btc_drop(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    rows = []
    timestamp = 1735689600000
    for index in range(25):
        close = 100.0 if index < 24 else 90.0
        rows.append(_row(timestamp + index * 60 * 60 * 1000, close, "BTC/USDT"))
    _write_rows(csv_path, rows)

    snapshots = load_retrodate_year_snapshots(csv_path, stride_minutes=60, state=RetrodateSnapshotState())

    assert snapshots[-1].regime == Regime.CRASH


def test_cli_config_accepts_use_v3_rolling_score_flag():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--use-v3-rolling-score",
        "--use-v3-shadow-rolling-score",
        "--use-v3-soft-shadow-score",
        "--use-v3-entry-causal-score",
        "--enable-v3-shadow-flat-handoff",
        "--enable-v3-shadow-fresh-handoff",
        "--v3-shadow-fresh-handoff-max-age-bars",
        "1",
        "--v3-shadow-rolling-window-bars",
        "24",
        "--v3-shadow-rolling-min-closed-trades",
        "50",
        "--v3-entry-causal-min-filled",
        "2",
        "--v3-entry-causal-actionability-weight",
        "0.5",
        "--enable-actionable-fallback",
        "--actionable-fallback-min-score",
        "0.25",
        "--actionable-fallback-require-has-data",
        "--enable-current-actionable-candidate-layer",
        "--use-v3-executable-soft-confirmed-score",
        "--solo-agent-candidate-limit",
        "6",
        "--enable-fixed-agent-players",
        "--fixed-agent-player-set",
        "Fixed_AB=AgentA,AgentB",
    ])

    assert config.use_v3_rolling_score is True
    assert config.use_v3_shadow_rolling_score is True
    assert config.use_v3_soft_shadow_score is True
    assert config.use_v3_entry_causal_score is True
    assert config.v3_shadow_position_gate_enabled is True
    assert config.v3_shadow_flat_handoff_enabled is True
    assert config.v3_shadow_fresh_handoff_enabled is True
    assert config.v3_shadow_fresh_handoff_max_age_bars == 1
    assert config.v3_shadow_rolling_window_bars == 24
    assert config.v3_shadow_rolling_min_closed_trades == 50
    assert config.v3_entry_causal_min_filled == 2
    assert config.v3_entry_causal_actionability_weight == 0.5
    assert config.actionable_fallback_enabled is True
    assert config.actionable_fallback_min_score == 0.25
    assert config.actionable_fallback_require_has_data is True
    assert config.current_actionable_candidate_layer_enabled is True
    assert config.use_v3_executable_soft_confirmed_score is True
    assert config.solo_agent_candidate_limit == 6
    assert config.fixed_agent_players_enabled is True
    assert config.fixed_agent_player_sets == (("Fixed_AB", ("AgentA", "AgentB")),)


def test_fixed_agent_players_default_to_multiple_experiment_sets():
    config = RetrodateMarketConfig(fixed_agent_players_enabled=True)

    assert config.fixed_agent_players_enabled is True
    assert len(config.fixed_agent_player_sets) >= 5
    assert any(label == "Fixed_BullBreakout" for label, _agents in config.fixed_agent_player_sets)
    assert any(label == "Fixed_NeutralValidator" for label, _agents in config.fixed_agent_player_sets)


def test_probation_loss_kill_defaults_include_experimental_composites():
    config = runner._parse_cli_config(["--years", "2025"])
    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.v3_probation_loss_kill_label_prefixes == (
        "Solo_",
        "Fixed_",
        "Antonius_",
        "Optimal_",
    )


def test_retrodate_strategist_config_denies_weak_production_profiles():
    config = runner._parse_cli_config(["--years", "2025"])
    strategist_config = runner._build_strategist_config(config)

    deny = set(strategist_config.hard_policy_deny_labels)
    assert {"DefaultEnsemble", "DefensiveResearch", "TrendResearch"}.issubset(deny)
    assert {"Solo_PlayerFunding", "Solo_LiveAfterShock"}.issubset(deny)
    assert strategist_config.hard_policy_experimental_min_bar == 0


def test_build_strategist_config_passes_use_v3_rolling_score():
    config = RetrodateMarketConfig(
        use_v3_rolling_score=True,
        use_v3_shadow_rolling_score=True,
        use_v3_soft_shadow_score=True,
        use_v3_executable_soft_confirmed_score=True,
        use_v3_entry_causal_score=True,
        current_actionable_candidate_layer_enabled=True,
        v3_shadow_position_gate_enabled=True,
        v3_shadow_flat_handoff_enabled=True,
        v3_shadow_fresh_handoff_enabled=True,
        v3_shadow_fresh_handoff_max_age_bars=1,
        v3_shadow_rolling_window_bars=24,
        v3_shadow_rolling_min_closed_trades=50,
        v3_entry_causal_min_filled=2,
        v3_entry_causal_actionability_weight=0.5,
        solo_agent_candidate_limit=6,
    )

    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.use_v3_rolling_score is True
    assert strategist_config.use_v3_shadow_rolling_score is True
    assert strategist_config.use_v3_soft_shadow_score is True
    assert strategist_config.v3_current_actionable_gate_enabled is True
    assert strategist_config.use_v3_entry_causal_score is True
    assert strategist_config.v3_shadow_position_gate_enabled is True
    assert strategist_config.v3_shadow_flat_handoff_enabled is True
    assert strategist_config.v3_shadow_fresh_handoff_enabled is True
    assert strategist_config.v3_shadow_fresh_handoff_max_age_bars == 1
    assert strategist_config.v3_shadow_rolling_window_bars == 24
    assert strategist_config.v3_shadow_rolling_min_closed_trades == 50
    assert strategist_config.v3_entry_causal_min_filled == 2
    assert strategist_config.v3_entry_causal_actionability_weight == 0.5
    assert config.solo_agent_candidate_limit == 6


def test_cli_and_strategist_config_accept_executable_soft_top1_score_flag():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--use-v3-executable-soft-top1-score",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.use_v3_executable_soft_top1_score is True
    assert strategist_config.use_v3_rolling_score is True
    assert strategist_config.use_v3_soft_shadow_score is True
    assert strategist_config.v3_current_actionable_gate_enabled is True
    assert strategist_config.v3_shadow_rolling_window_bars == 24
    assert strategist_config.v3_shadow_rolling_min_closed_trades == 20


def test_run_summary_records_effective_executable_soft_top1_strategy_config(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=0,
        registered_agents=(),
        player_profile_count=0,
        first_timestamp="",
        last_timestamp="",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    config = RetrodateMarketConfig(use_v3_executable_soft_top1_score=True)
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(summary.summary_path, summary, selection, config)

    data = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    assert data["use_v3_executable_soft_top1_score"] is True
    assert data["effective_v3_shadow_rolling_window_bars"] == 24
    assert data["effective_v3_shadow_rolling_min_closed_trades"] == 20
    assert data["effective_v3_current_actionable_gate_enabled"] is True


def test_cli_and_strategist_config_accept_real_loss_rescue_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-v3-real-loss-rescue",
        "--v3-real-loss-rescue-min-virtual-pnl-pct",
        "12.5",
        "--v3-real-loss-rescue-max-virtual-dd-pct",
        "40",
        "--v3-real-loss-rescue-min-actionable-share",
        "0.15",
        "--v3-real-loss-rescue-min-recent-filled",
        "3",
        "--v3-real-loss-rescue-max-real-loss-pct",
        "-2.5",
        "--allow-genetics-real-loss-rescue",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_real_loss_rescue_enabled is True
    assert strategist_config.v3_real_loss_rescue_enabled is True
    assert strategist_config.v3_real_loss_rescue_min_virtual_pnl_pct == 12.5
    assert strategist_config.v3_real_loss_rescue_max_virtual_dd_pct == 40.0
    assert strategist_config.v3_real_loss_rescue_min_actionable_share == 0.15
    assert strategist_config.v3_real_loss_rescue_min_recent_filled == 3
    assert strategist_config.v3_real_loss_rescue_max_real_loss_pct == -2.5
    assert config.v3_real_loss_rescue_allow_genetics is True
    assert strategist_config.v3_real_loss_rescue_allow_genetics is True


def test_cli_and_strategist_config_accept_probation_shadow_rescue_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-v3-probation-shadow-rescue",
        "--v3-probation-shadow-rescue-min-virtual-pnl-pct",
        "12.5",
        "--v3-probation-shadow-rescue-max-virtual-dd-pct",
        "35",
        "--v3-probation-shadow-rescue-min-actionable-share",
        "0.2",
        "--v3-probation-shadow-rescue-min-recent-filled",
        "4",
        "--v3-probation-shadow-rescue-min-recent-pnl-usd",
        "3.5",
        "--allow-genetics-probation-shadow-rescue",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_shadow_rescue_enabled is True
    assert strategist_config.v3_probation_shadow_rescue_enabled is True
    assert strategist_config.v3_probation_shadow_rescue_min_virtual_pnl_pct == 12.5
    assert strategist_config.v3_probation_shadow_rescue_max_virtual_dd_pct == 35.0
    assert strategist_config.v3_probation_shadow_rescue_min_actionable_share == 0.2
    assert strategist_config.v3_probation_shadow_rescue_min_recent_filled == 4
    assert strategist_config.v3_probation_shadow_rescue_min_recent_pnl_usd == 3.5
    assert config.v3_probation_shadow_rescue_allow_genetics is True
    assert strategist_config.v3_probation_shadow_rescue_allow_genetics is True


def test_cli_and_live_execution_config_accept_genetics_probation_execution_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--enable-genetics-probation-execution",
        "--genetics-probation-risk-mult",
        "0.20",
        "--genetics-probation-max-real-trades",
        "7",
        "--disable-genetics-probation-shadow-confirmation",
    ])

    live_config = runner._build_live_execution_config(config)

    assert config.genetics_probation_execution_enabled is True
    assert config.genetics_probation_risk_mult == 0.20
    assert config.genetics_probation_max_real_trades == 7
    assert config.genetics_probation_require_shadow_confirmation is False
    assert live_config.genetics_probation_execution_enabled is True
    assert live_config.genetics_probation_labels == ("GeneticsResearch",)
    assert live_config.genetics_probation_allowed_regimes == ("bearish", "crash")
    assert live_config.genetics_probation_risk_mult == 0.20
    assert live_config.genetics_probation_max_real_trades == 7
    assert live_config.genetics_probation_require_shadow_confirmation is False


def test_cli_and_strategist_config_accept_profit_lock_and_probation_kill_flags():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--v3-probation-loss-kill-min-closed-trades",
        "2",
        "--v3-probation-loss-kill-pnl-pct",
        "-0.15",
        "--v3-probation-loss-kill-win-rate-pct",
        "50",
        "--v3-probation-loss-kill-label-prefix",
        "Solo_",
        "--v3-probation-loss-kill-label-prefix",
        "MeanRevResearch",
        "--v3-realized-profit-lock-min-closed-trades",
        "20",
        "--v3-realized-profit-lock-min-peak-pnl-pct",
        "0.75",
        "--v3-realized-profit-lock-max-giveback-pct",
        "0.55",
        "--v3-realized-profit-lock-floor-pnl-pct",
        "0.25",
        "--enable-v3-panteon-equity-guard",
        "--v3-panteon-equity-guard-min-peak-pnl-pct",
        "3.0",
        "--v3-panteon-equity-guard-max-giveback-pct",
        "1.1",
        "--v3-panteon-equity-guard-floor-pnl-pct",
        "2.4",
        "--v3-panteon-equity-guard-cooldown-bars",
        "168",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_loss_kill_min_closed_trades == 2
    assert strategist_config.v3_probation_loss_kill_min_closed_trades == 2
    assert strategist_config.v3_probation_loss_kill_pnl_pct == -0.15
    assert strategist_config.v3_probation_loss_kill_win_rate_pct == 50.0
    assert strategist_config.v3_probation_loss_kill_label_prefixes == (
        "Solo_",
        "MeanRevResearch",
    )
    assert strategist_config.v3_realized_profit_lock_min_closed_trades == 20
    assert strategist_config.v3_realized_profit_lock_min_peak_pnl_pct == 0.75
    assert strategist_config.v3_realized_profit_lock_max_giveback_pct == 0.55
    assert strategist_config.v3_realized_profit_lock_floor_pnl_pct == 0.25
    assert strategist_config.v3_panteon_equity_guard_enabled is True
    assert strategist_config.v3_panteon_equity_guard_min_peak_pnl_pct == 3.0
    assert strategist_config.v3_panteon_equity_guard_max_giveback_pct == 1.1
    assert strategist_config.v3_panteon_equity_guard_floor_pnl_pct == 2.4
    assert strategist_config.v3_panteon_equity_guard_cooldown_bars == 168


def test_cli_can_apply_probation_loss_kill_to_all_labels():
    config = runner._parse_cli_config([
        "--years",
        "2025",
        "--v3-probation-loss-kill-all-labels",
    ])

    strategist_config = runner._build_strategist_config(config)

    assert config.v3_probation_loss_kill_label_prefixes == ()
    assert strategist_config.v3_probation_loss_kill_label_prefixes == ()


def test_analysis_report_includes_soft_allocator_summary(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "Leader",
            "live_session": {
                "panteon_owned_pnl_pct": 1.5,
                "panteon_owned_realized_pnl_usd": 15.0,
                "real_closed_trades": 3,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "soft_allocator_report.json").write_text(
        json.dumps({
            "best_single_label": "Solo_A",
            "best_single_pnl_usd": 100.0,
            "best_single_pnl_pct": 10.0,
            "best_single_max_drawdown_pct": 4.0,
            "best_policy_name": "soft_top3_decayed",
            "best_policy_pnl_usd": 130.0,
            "best_policy_pnl_pct": 13.0,
            "best_policy_max_drawdown_pct": 3.0,
            "regret_vs_best_single_usd": 0.0,
            "beats_best_single": True,
        }),
        encoding="utf-8",
    )
    (output_dir / "perfect_panteon_report.json").write_text(
        json.dumps({
            "pnl_usd": 200.0,
            "pnl_pct": 20.0,
            "max_drawdown_pct": 2.5,
            "month_count": 2,
            "profitable_months": 2,
            "cash_months": 0,
            "closed_trades": 11.0,
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=1,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T01:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_analysis_report(summary.report_path, summary, selection)

    report = summary.report_path.read_text(encoding="utf-8")
    assert "## Soft Allocator Simulation" in report
    assert "`soft_top3_decayed`" in report
    assert "Beats best single: yes" in report
    assert "## Perfect Panteon Monthly Oracle" in report
    assert "$200.00" in report
    assert "available 2026 file contains 2022-2023 timestamps" not in report


def test_analysis_report_and_summary_include_allocation_diagnostics(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({
            "current_leader": "NoTrade",
            "live_session": {
                "panteon_owned_pnl_pct": -1.0,
                "panteon_owned_realized_pnl_usd": -10.0,
                "real_closed_trades": 2,
                "panteon_owned_positions_count": 0,
            },
            "shadow": {"actors": 2},
        }),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_agents.json").write_text(
        json.dumps({"agents": {}}),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_players.json").write_text(
        json.dumps({"players": {}}),
        encoding="utf-8",
    )
    (output_dir / "allocation_diagnostics.json").write_text(
        json.dumps({
            "bars": 4,
            "no_trade_bars": 1,
            "no_trade_share_pct": 25.0,
            "raw_zero_bars": 3,
            "raw_zero_share_pct": 75.0,
            "filled_zero_bars": 3,
            "filled_zero_share_pct": 75.0,
            "leaders": {
                "Alpha": {
                    "bars": 3,
                    "bar_share_pct": 75.0,
                    "raw_zero_share_pct": 66.6666666667,
                    "filled_zero_share_pct": 66.6666666667,
                    "raw_signals": 2,
                    "signals": 1,
                    "filled": 1,
                }
            },
        }),
        encoding="utf-8",
    )
    summary = RetrodateRunSummary(
        output_dir=output_dir,
        report_path=output_dir / "analysis_report.md",
        summary_path=output_dir / "run_summary.json",
        requested_years=(2025,),
        executed_years=(2025,),
        excluded_files=(),
        missing_years=(),
        bars_processed=4,
        registered_agents=("A",),
        player_profile_count=1,
        first_timestamp="2025-01-01T00:00:00+00:00",
        last_timestamp="2025-01-01T04:00:00+00:00",
        stride_minutes=60,
        max_bars=None,
        step_errors=(),
    )
    selection = RetrodateFileSelection(
        report=RetrodateDirReport(path=tmp_path, files=[]),
        valid_reports=(),
        excluded_files=(),
        missing_years=(),
    )

    runner._write_run_summary(
        summary.summary_path,
        summary,
        selection,
        RetrodateMarketConfig(data_dir=tmp_path, years=(2025,), solo_agent_candidate_limit=6),
    )
    runner._write_analysis_report(summary.report_path, summary, selection)

    run_summary = json.loads(summary.summary_path.read_text(encoding="utf-8"))
    report = summary.report_path.read_text(encoding="utf-8")
    assert run_summary["solo_agent_candidate_limit"] == 6
    assert run_summary["fixed_agent_players_enabled"] is False
    assert run_summary["fixed_agent_player_count"] == 0
    assert run_summary["allocation_diagnostics"]["no_trade_share_pct"] == 25.0
    assert "## Allocation Diagnostics" in report
    assert "NoTrade share: 25.00%" in report
    assert "`allocation_diagnostics.json`" in report


def test_shadow_pnl_export_can_write_agent_stream(tmp_path):
    path = write_shadow_pnl_events(
        tmp_path,
        runner.shadow_pnl_events_from_shadow_updates(
            [
                ShadowActorUpdated(
                    bar=1,
                    actor_type="agent",
                    actor_label="MomentumScalper",
                    regime="bearish",
                    realized_pnl_usd=4.0,
                    closed_trades=2,
                    winning_trades=1,
                ),
                ShadowActorUpdated(
                    bar=1,
                    actor_type="player",
                    actor_label="MeanRevResearch",
                    regime="bearish",
                    realized_pnl_usd=8.0,
                    closed_trades=3,
                    winning_trades=2,
                ),
            ],
            actor_type="agent",
        ),
        filename="shadow_agent_pnl_events.jsonl",
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert path.name == "shadow_agent_pnl_events.jsonl"
    assert len(rows) == 1
    assert rows[0]["timestamp"]
    rows[0]["timestamp"] = "<timestamp>"
    assert rows == [{
        "bar": 1,
        "label": "MomentumScalper",
        "timestamp": "<timestamp>",
        "regime": "bearish",
        "pnl_usd": 4.0,
        "closed_trades": 2,
        "wins": 1,
    }]


def test_candidate_diagnostics_export_summarizes_actionability(tmp_path):
    path = runner.write_candidate_diagnostics(
        tmp_path,
        candidate_events=[
            CandidateScored(
                bar=1,
                player_label="Fixed_OIBreakoutCore",
                selected_by_pantheon=True,
                score=2.0,
                has_data=True,
                closed_trades=12,
                signals=8,
                recent_bars=4,
                recent_actionable_bars=2,
                actionable_share=0.5,
                recent_filled=3,
                recent_pnl_usd=7.0,
            ),
            CandidateScored(
                bar=1,
                player_label="Solo_LiveOIBreakout",
                selected_by_pantheon=False,
                score=1.0,
                has_data=True,
                closed_trades=10,
                signals=4,
                recent_bars=4,
                recent_actionable_bars=4,
                actionable_share=1.0,
                recent_filled=4,
                recent_pnl_usd=5.0,
            ),
        ],
        rejection_events=[
            CandidateRejected(
                bar=1,
                player_label="Fixed_Bad",
                reason="hard policy test",
            )
        ],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = data["candidates"]["Fixed_OIBreakoutCore"]
    solo = data["candidates"]["Solo_LiveOIBreakout"]

    assert data["candidate_rows"] == 2
    assert data["selected_rows"] == 1
    assert data["rejection_rows"] == 1
    assert fixed["group_type"] == "fixed"
    assert fixed["selected_bars"] == 1
    assert fixed["avg_actionable_share"] == 0.5
    assert fixed["avg_recent_filled"] == 3.0
    assert fixed["avg_recent_pnl_usd"] == 7.0
    assert solo["group_type"] == "solo"
    assert data["rejections"]["Fixed_Bad"]["count"] == 1


def test_candidate_group_type_marks_regime_switch_and_rotator() -> None:
    assert runner._candidate_group_type("Antonius_strategy") == "regime_switch"
    assert runner._candidate_group_type("Perfect_OIBreakout") == "regime_switch"
    assert runner._candidate_group_type("Optimal_StaticRotator") == "rotating"


def test_oracle_mismatch_report_marks_perfect_leader_without_current_signal(tmp_path):
    trading_log = tmp_path / "trading.log"
    trading_log.write_text(
        "2025-01-01 00:00:00  bar=1  regime=bullish  leader=NoTrade  "
        "selected_leader=NoTrade  executed_leader=NoTrade  raw_signals=0  "
        "signals=0  filled=0\n",
        encoding="utf-8",
    )

    path = runner.write_oracle_mismatch_report(
        tmp_path,
        trading_log,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                actor_type="player",
                actor_label="PerfectPlayer",
                regime="bullish",
                signals=0,
                filled=0,
                realized_pnl_usd=10.0,
                closed_trades=2,
                winning_trades=2,
            )
        ],
        candidate_rejections=[],
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["mismatch_rows"] == 1
    assert data["summary"]["reason_counts"]["no current signal"] == 1
    row = data["rows"][0]
    assert row["real_leader"] == "NoTrade"
    assert row["perfect_leader"] == "PerfectPlayer"
    assert row["reason"] == "no current signal"
