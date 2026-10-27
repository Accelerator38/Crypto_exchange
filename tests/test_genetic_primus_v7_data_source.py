from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

from exia.genetic_primus.data_source_v7 import aggregate_liquidations, four_hour_grid, read_liquidation_zip


HEADER = "time,side,order_type,time_in_force,original_quantity,price,average_price,order_status,last_fill_quantity,accumulated_fill_quantity\n"


def _archive(text: str) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSD_PERP-liquidationSnapshot-2024-03-31.csv", text)
    return buffer.getvalue()


def test_v7_reader_deduplicates_exact_source_rows() -> None:
    row = "1711855127079,SELL,LIMIT,IOC,49,69597.2,69863.6,FILLED,44,49\n"
    frame = read_liquidation_zip(_archive(HEADER + row + row), expected_date="2024-03-31")
    assert len(frame) == 1
    assert frame.attrs["exact_duplicates_removed"] == 1
    bars = aggregate_liquidations(
        frame,
        target_symbol="BTC/USDT",
        source_symbol="BTCUSD_PERP",
        contract_size_usd=100,
    )
    assert bars.iloc[0]["sell_liquidation_usd"] == 4900
    assert bars.iloc[0]["buy_liquidation_usd"] == 0


def test_v7_four_hour_grid_uses_millisecond_timestamps() -> None:
    grid = four_hour_grid("2023-06-25", "2023-06-25")
    assert len(grid) == 6
    assert grid.iloc[0]["timestamp"] == 1_687_651_200_000
    assert grid.iloc[-1]["date"] == "2023-06-25"
