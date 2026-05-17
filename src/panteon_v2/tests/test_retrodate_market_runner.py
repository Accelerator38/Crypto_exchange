"""Tests for Retrodate market benchmark helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import panteon_v2.analysis.retrodate_market_runner as runner
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
        "--v3-shadow-rolling-window-bars",
        "24",
        "--v3-shadow-rolling-min-closed-trades",
        "50",
    ])

    assert config.use_v3_rolling_score is True
    assert config.use_v3_shadow_rolling_score is True
    assert config.v3_shadow_rolling_window_bars == 24
    assert config.v3_shadow_rolling_min_closed_trades == 50


def test_build_strategist_config_passes_use_v3_rolling_score():
    config = RetrodateMarketConfig(
        use_v3_rolling_score=True,
        use_v3_shadow_rolling_score=True,
        v3_shadow_rolling_window_bars=24,
        v3_shadow_rolling_min_closed_trades=50,
    )

    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.use_v3_rolling_score is True
    assert strategist_config.use_v3_shadow_rolling_score is True
    assert strategist_config.v3_shadow_rolling_window_bars == 24
    assert strategist_config.v3_shadow_rolling_min_closed_trades == 50


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
