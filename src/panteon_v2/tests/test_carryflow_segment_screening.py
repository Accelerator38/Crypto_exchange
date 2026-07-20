from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

from panteon_v2.analysis.carryflow_segment_screening import (
    CarryFlowScreeningConfig,
    screen_carryflow_segments,
)
from panteon_v2.policy import CarryFlowEvidenceTape, append_tape_sample


ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_PATH = ROOT / "tools" / "collect_bitget_carryflow_tape.py"


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_bitget_carryflow_tape_for_screening_test",
        COLLECTOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_tape(path: Path, *, run_id: str, start_price: float) -> None:
    collector = _load_collector()
    base_close = datetime(2026, 7, 1, tzinfo=timezone.utc)
    open_interest = 1_000_000.0
    for index in range(28):
        if index in {1, 15}:
            open_interest *= 1.03
        price = start_price * (1.0 - index * 0.0018)
        close = base_close + timedelta(hours=index)
        observed = close + timedelta(seconds=30)
        close_ms = int(close.timestamp() * 1000)
        candle_start = close_ms - 3_600_000
        payload = collector.build_tape_payload(
            collector_run_id=run_id,
            source_revision="d863585",
            collector_code_sha256="1" * 64,
            funding_fetcher_code_sha256="2" * 64,
            observed_at=observed,
            bar_interval_seconds=3600,
            bar_close_timestamp_ms=close_ms,
            max_bar_close_lag_seconds=120,
            max_derivatives_age_seconds=1200,
            symbols=("BTC",),
            candles={
                "BTC": {
                    "candle_timestamp_ms": candle_start,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 500.0,
                }
            },
            derivatives={
                "BTC": {
                    "funding_rate": 0.0002,
                    "open_interest_usdt": open_interest,
                    "long_ratio": 0.65,
                    "short_ratio": 0.35,
                    "mark_price": price * 0.9995,
                    "index_price": price,
                    "last_price": price,
                    "next_funding_ts": 0,
                    "long_short_ratio_ts": int(observed.timestamp() * 1000),
                    "long_short_ratio_age_sec": 0.0,
                    "long_short_source": "account_long_short_v2",
                    "updated_ts": observed.timestamp() - 5.0,
                    "context_complete": True,
                }
            },
        )
        append_tape_sample(path, payload)


def test_segment_screening_detects_structural_basis_block_without_joining(tmp_path):
    first_path = tmp_path / "one.jsonl"
    second_path = tmp_path / "two.jsonl"
    _build_tape(first_path, run_id="segment-one", start_price=100.0)
    _build_tape(second_path, run_id="segment-two", start_price=200.0)
    tapes = [
        CarryFlowEvidenceTape.from_jsonl(first_path),
        CarryFlowEvidenceTape.from_jsonl(second_path),
    ]

    report = screen_carryflow_segments(
        tapes,
        config=CarryFlowScreeningConfig(hold_bars=12),
    )

    assert report["segments_concatenated"] is False
    assert report["segment_count"] == 2
    assert report["current_policy"]["structural_activation_block"] is True
    assert report["current_policy"]["precondition_events"] == 0
    assert report["research_candidate"]["precondition_events"] == 4
    assert report["research_candidate"]["trade_metrics"]["closed_trades"] == 4
    assert set(
        report["research_candidate"]["trade_metrics"]["trades_by_segment"]
    ) == {"segment-one", "segment-two"}
    assert report["decision"]["research_candidate_ready_for_promotion"] is False
