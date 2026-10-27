from __future__ import annotations

import io
from zipfile import ZipFile

import numpy as np
import pandas as pd

from exia.genetic_primus.data_source_v5 import aggregate_symbol_metrics, parse_checksum, read_metric_zip


def _raw_zip() -> bytes:
    timestamps = pd.date_range("2024-01-01", periods=96, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "create_time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": "BTCUSDT",
            "sum_open_interest": np.arange(96, dtype=float) + 1000.0,
            "sum_open_interest_value": np.arange(96, dtype=float) + 5000.0,
            "count_toptrader_long_short_ratio": 1.1,
            "sum_toptrader_long_short_ratio": 1.2,
            "count_long_short_ratio": 0.9,
            "sum_taker_long_short_vol_ratio": 1.25,
        }
    )
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2024-01-01.csv", frame.to_csv(index=False))
    return payload.getvalue()


def test_checksum_parser_requires_expected_filename() -> None:
    digest = "a" * 64
    assert parse_checksum(f"{digest}  file.zip\n".encode(), "file.zip") == digest


def test_raw_metrics_are_aggregated_to_causal_four_hour_bars() -> None:
    raw = read_metric_zip(_raw_zip(), expected_symbol="BTCUSDT")
    result = aggregate_symbol_metrics(raw, source_symbol="BTCUSDT", target_symbol="BTC/USDT")
    assert len(result) == 2
    assert result["samples"].tolist() == [48, 48]
    assert result["complete_bar"].all()
    assert (result["available_at_ms"] - result["timestamp"] == 4 * 60 * 60 * 1000).all()
    assert result["oi_change_4h"].notna().all()
    assert result["taker_imbalance_mean"].gt(0.0).all()


def test_unsorted_official_rows_are_repaired_but_recorded() -> None:
    payload = _raw_zip()
    raw = read_metric_zip(payload, expected_symbol="BTCUSDT")
    assert raw.attrs["source_was_unsorted"] is False
    with ZipFile(io.BytesIO(payload)) as archive:
        frame = pd.read_csv(archive.open(archive.namelist()[0])).iloc[::-1]
    changed = io.BytesIO()
    with ZipFile(changed, "w") as archive:
        archive.writestr("BTCUSDT-metrics-2024-01-01.csv", frame.to_csv(index=False))
    repaired = read_metric_zip(changed.getvalue(), expected_symbol="BTCUSDT")
    assert repaired.attrs["source_was_unsorted"] is True
    assert repaired["create_time"].is_monotonic_increasing
