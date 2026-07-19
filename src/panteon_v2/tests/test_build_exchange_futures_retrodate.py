from __future__ import annotations

import importlib.util
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "build_exchange_futures_retrodate.py"
    spec = importlib.util.spec_from_file_location("build_exchange_futures_retrodate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_ohlcv_row_outputs_retrodate_schema():
    module = _load_tool()

    row = module.normalize_ohlcv_row(
        exchange="MEXC",
        symbol="BTC/USDT:USDT",
        ohlcv=[1_766_188_800_000, 100.0, 110.0, 95.0, 105.0, 123.45],
    )

    assert row["timestamp"] == 1_766_188_800_000
    assert row["symbol"] == "BTC/USDT"
    assert row["open"] == "100.0"
    assert row["high"] == "110.0"
    assert row["low"] == "95.0"
    assert row["close"] == "105.0"
    assert row["volume"] == "123.45"
    assert row["source_exchange"] == "MEXC"
    assert row["market_type"] == "futures"


def test_write_year_files_keeps_exchange_specific_metadata(tmp_path):
    module = _load_tool()
    rows = [
        module.normalize_ohlcv_row(
            exchange="BITGET",
            symbol="ETH/USDT:USDT",
            ohlcv=[1_766_188_800_000, 200.0, 210.0, 190.0, 205.0, 10.0],
        )
    ]

    counts = module.write_year_files(tmp_path, rows)

    assert counts == {2025: 1}
    text = (tmp_path / "crypto_1m_2025_all_symbols.csv").read_text(encoding="utf-8")
    assert "source_exchange,market_type" in text.splitlines()[0]
    assert "BITGET,futures" in text


def test_write_year_files_merges_existing_exchange_rows(tmp_path):
    module = _load_tool()

    module.write_year_files(
        tmp_path,
        [
            module.normalize_ohlcv_row(
                exchange="MEXC",
                symbol="BTC/USDT:USDT",
                ohlcv=[1_766_188_800_000, 100.0, 101.0, 99.0, 100.5, 1.0],
            )
        ],
    )
    counts = module.write_year_files(
        tmp_path,
        [
            module.normalize_ohlcv_row(
                exchange="BITGET",
                symbol="BTC/USDT:USDT",
                ohlcv=[1_766_188_800_000, 100.1, 101.1, 99.1, 100.6, 2.0],
            )
        ],
    )

    assert counts == {2025: 2}
    text = (tmp_path / "crypto_1m_2025_all_symbols.csv").read_text(encoding="utf-8")
    assert "MEXC,futures" in text
    assert "BITGET,futures" in text


def test_bitget_v3_history_paginates_backwards_and_sorts_rows():
    module = _load_tool()
    start = date(2022, 1, 1)
    end = date(2022, 1, 1)
    start_ms = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    calls = []

    class Response:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return {"code": "00000", "msg": "success", "data": self._data}

    def request_get(_url, *, params, timeout):
        assert timeout == 30
        calls.append(dict(params))
        if len(calls) == 1:
            hours = (20, 21)
        else:
            hours = (0, 1)
        return Response([
            [start_ms + hour * 3_600_000, "1", "2", "0.5", "1.5", "10", "15"]
            for hour in reversed(hours)
        ])

    rows, report = module.fetch_bitget_v3_history_rows(
        symbols=("BTC/USDT",),
        start=start,
        end=end,
        timeframe="1h",
        sleep_ms=0,
        request_get=request_get,
    )

    assert [row["timestamp"] for row in rows] == [
        start_ms,
        start_ms + 3_600_000,
        start_ms + 20 * 3_600_000,
        start_ms + 21 * 3_600_000,
    ]
    assert int(calls[1]["endTime"]) < int(calls[0]["endTime"])
    assert all(
        int(call["endTime"]) - int(call["startTime"])
        <= module.BITGET_MAX_QUERY_SPAN_MS
        for call in calls
    )
    assert report["public_only"] is True
    assert report["requests"] == 2


def test_integrity_manifest_reports_hashes_and_internal_gaps(tmp_path):
    module = _load_tool()
    start_ms = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [
        module.normalize_ohlcv_row(
            exchange="BITGET",
            symbol="BTC/USDT",
            ohlcv=[start_ms + offset, 1, 2, 0.5, 1.5, 10],
        )
        for offset in (0, 2 * 3_600_000)
    ]
    module.write_year_files(tmp_path, rows)

    manifest = module.build_integrity_manifest(
        tmp_path,
        exchange="BITGET",
        symbols=("BTC/USDT",),
        start=date(2022, 1, 1),
        end=date(2022, 1, 1),
        timeframe="1h",
        source={"endpoint": "test", "public_only": True},
    )

    assert manifest["validation"]["duplicate_rows"] == 0
    assert manifest["validation"]["missing_bars_inside_coverage"] == 1
    assert manifest["validation"]["passed"] is False
    assert manifest["coverage"]["BTC/USDT"]["rows"] == 2
    assert len(manifest["files"][0]["sha256"]) == 64
    assert len(manifest["dataset_sha256"]) == 64
