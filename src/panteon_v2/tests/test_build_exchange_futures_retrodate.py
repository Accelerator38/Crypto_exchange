from __future__ import annotations

import importlib.util
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
