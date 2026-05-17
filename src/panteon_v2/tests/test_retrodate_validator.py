"""Tests for Retrodate OHLCV file validation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from panteon_v2.analysis.retrodate_validator import (
    RetrodateValidationError,
    ensure_retrodate_dir_valid,
    validate_retrodate_dir,
    validate_retrodate_file,
)


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


def _row(timestamp: int, dt: str, symbol: str = "BTC/USDT") -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 10.0,
        "symbol": symbol,
        "datetime": dt,
    }


def test_validate_retrodate_file_blocks_rows_outside_filename_year(tmp_path):
    csv_path = tmp_path / "crypto_1m_2026_all_symbols.csv"
    _write_rows(
        csv_path,
        [
            _row(1640995200000, "2022-01-01 00:00:00+00:00"),
            _row(1676419140000, "2023-02-14 23:59:00+00:00", "ETH/USDT"),
        ],
    )

    report = validate_retrodate_file(csv_path)

    assert report.is_valid is False
    assert report.expected_year == 2026
    assert report.first_datetime == "2022-01-01 00:00:00+00:00"
    assert report.last_datetime == "2023-02-14 23:59:00+00:00"
    assert report.rows == 2
    assert report.symbols == 2
    assert [issue.code for issue in report.issues] == ["year_mismatch"]
    assert "expected 2026" in report.issues[0].message
    assert "2022" in report.issues[0].message
    assert "2023" in report.issues[0].message


def test_validate_retrodate_file_accepts_rows_inside_filename_year(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    _write_rows(
        csv_path,
        [
            _row(1735689600000, "2025-01-01 00:00:00+00:00"),
            _row(1767225540000, "2025-12-31 23:59:00+00:00", "ETH/USDT"),
        ],
    )

    report = validate_retrodate_file(csv_path)

    assert report.is_valid is True
    assert report.expected_year == 2025
    assert report.first_datetime == "2025-01-01 00:00:00+00:00"
    assert report.last_datetime == "2025-12-31 23:59:00+00:00"
    assert report.rows == 2
    assert report.symbols == 2
    assert report.issues == []


def test_validate_retrodate_dir_marks_directory_invalid_when_any_file_is_invalid(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2025_all_symbols.csv",
        [_row(1735689600000, "2025-01-01 00:00:00+00:00")],
    )
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, "2022-01-01 00:00:00+00:00")],
    )

    report = validate_retrodate_dir(tmp_path)

    assert report.is_valid is False
    assert len(report.files) == 2
    assert [item.path.name for item in report.invalid_files] == ["crypto_1m_2026_all_symbols.csv"]


def test_ensure_retrodate_dir_valid_blocks_invalid_directory(tmp_path):
    _write_rows(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        [_row(1640995200000, "2022-01-01 00:00:00+00:00")],
    )

    with pytest.raises(RetrodateValidationError) as exc:
        ensure_retrodate_dir_valid(tmp_path)

    assert "crypto_1m_2026_all_symbols.csv" in str(exc.value)
    assert "year_mismatch" in str(exc.value)
