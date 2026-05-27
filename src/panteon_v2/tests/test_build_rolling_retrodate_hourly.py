from __future__ import annotations

import importlib.util
import sys
from datetime import date, datetime, timezone
from pathlib import Path


def _load_tool():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_rolling_retrodate_hourly.py"
    spec = importlib.util.spec_from_file_location(
        "build_rolling_retrodate_hourly",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iter_months_spans_partial_boundaries():
    tool = _load_tool()

    months = list(tool.iter_months(date(2021, 5, 27), date(2022, 1, 2)))

    assert months == [
        (2021, 5),
        (2021, 6),
        (2021, 7),
        (2021, 8),
        (2021, 9),
        (2021, 10),
        (2021, 11),
        (2021, 12),
        (2022, 1),
    ]


def test_iter_dates_is_inclusive():
    tool = _load_tool()

    days = list(tool.iter_dates(date(2026, 5, 19), date(2026, 5, 21)))

    assert days == [date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 21)]


def test_parse_binance_kline_row_to_retrodate_row():
    tool = _load_tool()
    row = [
        "1622073600000",
        "38810.99",
        "38900.00",
        "38700.00",
        "38888.00",
        "123.45",
        "1622073659999",
    ]

    parsed = tool.parse_binance_kline_row("BTCUSDT", row)

    assert parsed == {
        "timestamp": 1622073600000,
        "open": "38810.99",
        "high": "38900.00",
        "low": "38700.00",
        "close": "38888.00",
        "volume": "123.45",
        "symbol": "BTC/USDT",
        "datetime": datetime.fromtimestamp(
            1622073600000 / 1000.0,
            timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S+00:00"),
    }


def test_parse_binance_kline_row_normalizes_microsecond_timestamps():
    tool = _load_tool()
    row = [
        "1779148800000000",
        "0.25170000",
        "0.25180000",
        "0.25150000",
        "0.25170000",
        "111713.40000000",
    ]

    parsed = tool.parse_binance_kline_row("ADAUSDT", row)

    assert parsed is not None
    assert parsed["timestamp"] == 1779148800000
    assert parsed["datetime"] == "2026-05-19 00:00:00+00:00"
