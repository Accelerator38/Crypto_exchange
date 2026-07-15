from __future__ import annotations

import csv
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

from panteon_v2.policy.derivatives_context import (
    DERIVATIVES_CONTEXT_CSV_FIELDS,
    HistoricalDerivativesContext,
)


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "collect_bitget_derivatives_context.py"
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "collect_bitget_derivatives_context", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Fetcher:
    def __init__(self):
        self.calls = 0

    def fetch_all(self, symbols):
        assert tuple(symbols) == ("BTC", "ETH")
        self.calls += 1
        return {
            "BTC": {
                "funding_rate": 0.0001,
                "open_interest_usdt": 1_000_000.0 + self.calls,
                "long_ratio": 0.60,
                "short_ratio": 0.40,
                "mark_price": 101.0,
                "index_price": 100.0,
                "last_price": 100.5,
                "updated_ts": NOW.timestamp(),
                "context_complete": True,
            }
        }


def test_collector_appends_all_symbols_and_output_loads_point_in_time(tmp_path):
    tool = _load_tool()
    output = tmp_path / "context.csv"
    timestamps = iter((NOW, NOW + timedelta(minutes=1)))
    sleeps = []

    summary = tool.collect_samples(
        fetcher=_Fetcher(),
        symbols=("BTC", "ETH"),
        output=output,
        samples=2,
        interval_sec=0.25,
        now_fn=lambda: next(timestamps),
        sleep_fn=sleeps.append,
    )

    assert summary["orders_enabled"] is False
    assert summary["rows"] == 4
    assert summary["complete_rows"] == 2
    assert summary["missing_rows"] == 2
    assert sleeps == [0.25]
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == DERIVATIVES_CONTEXT_CSV_FIELDS
        rows = list(reader)
    assert [row["retrieval_status"] for row in rows] == [
        "complete",
        "missing",
        "complete",
        "missing",
    ]

    context = HistoricalDerivativesContext.from_csv(output, symbols=("BTC",))
    context.advance(NOW + timedelta(seconds=30))
    assert context.get("BTC")["open_interest_usdt"] == 1_000_001.0
    context.advance(NOW + timedelta(minutes=1))
    assert context.get("BTC")["open_interest_usdt"] == 1_000_002.0
