"""Tests for Retrodate market benchmark helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import panteon_v2.analysis.retrodate_market_runner as runner
from panteon_v2.analysis.retrodate_market_runner import (
    RetrodateMarketConfig,
    RetrodateSnapshotState,
    load_retrodate_year_snapshots,
    select_retrodate_files,
)
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
    config = runner._parse_cli_config(["--years", "2025", "--use-v3-rolling-score"])

    assert config.use_v3_rolling_score is True


def test_build_strategist_config_passes_use_v3_rolling_score():
    config = RetrodateMarketConfig(use_v3_rolling_score=True)

    strategist_config = runner._build_strategist_config(config)

    assert strategist_config.use_v3_rolling_score is True
