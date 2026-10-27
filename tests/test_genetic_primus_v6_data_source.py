from __future__ import annotations

import io
from zipfile import ZipFile

import pandas as pd

from exia.genetic_primus.data_source_v6 import KLINE_COLUMNS, build_flow_frame, read_kline_zip


def _zip() -> bytes:
    frame = pd.DataFrame(
        [
            [1640995200000, 100, 110, 90, 105, 10, 1641009599999, 1000, 50, 6, 600, 0],
            [1641009600000, 105, 115, 95, 110, 12, 1641023999999, 1200, 60, 5, 500, 0],
        ],
        columns=KLINE_COLUMNS,
    )
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSDT-4h-2022-01.csv", frame.to_csv(index=False))
    return payload.getvalue()


def test_v6_kline_flow_is_causal_and_exact() -> None:
    raw = read_kline_zip(_zip())
    flow = build_flow_frame(raw, source_symbol="BTCUSDT", target_symbol="BTC/USDT")
    assert flow["kline_taker_imbalance_4h"].round(6).tolist() == [0.2, -0.166667]
    assert (flow["available_at_ms"] - flow["timestamp"] == 4 * 60 * 60 * 1000).all()
    assert flow["symbol"].tolist() == ["BTC/USDT", "BTC/USDT"]


def test_v6_accepts_numeric_headerless_official_format() -> None:
    frame = pd.DataFrame(
        [[1640995200000, 100, 110, 90, 105, 10, 1641009599999, 1000, 50, 6, 600, 0]],
        columns=KLINE_COLUMNS,
    )
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("SOLUSDT-4h-2022-03.csv", frame.to_csv(index=False, header=False))
    parsed = read_kline_zip(payload.getvalue())
    assert parsed.attrs["source_header_missing"] is True
    assert parsed["open_time"].tolist() == [1640995200000]
