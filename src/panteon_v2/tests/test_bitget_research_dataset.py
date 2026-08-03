from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from panteon_v2.data.bitget import (
    event_study,
    maker_event_study,
    microstructure_baseline,
    research_dataset,
)


def _build_frame_fixture(path: Path, *, minutes: int = 20) -> Path:
    path.mkdir(parents=True)
    database = path / "frame_1s_v1.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE frames (
                frame_timestamp_ms INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                continuity_id INTEGER NOT NULL,
                bid_price REAL,
                ask_price REAL,
                mid_price REAL,
                spread_bps REAL,
                microprice REAL,
                bid_depth_5_usd REAL,
                ask_depth_5_usd REAL,
                book_imbalance_5 REAL,
                execution_curve_json TEXT NOT NULL,
                basis_bps REAL,
                funding_rate REAL,
                open_interest_base REAL,
                rest_open_interest_base REAL,
                trade_count INTEGER NOT NULL,
                buy_trade_notional_usd REAL NOT NULL,
                sell_trade_notional_usd REAL NOT NULL,
                signed_trade_notional_usd REAL NOT NULL,
                complete INTEGER NOT NULL
            )
            """
        )
        rows = []
        for second in range(1, minutes * 60 + 1):
            minute = (second - 1) // 60
            mid = 100.0 + minute * 0.10
            bid = mid - 0.01
            ask = mid + 0.01
            curve = json.dumps(
                {
                    "25": {
                        "buy_vwap": ask,
                        "buy_slippage_bps": (ask / mid - 1.0) * 10_000.0,
                        "sell_vwap": bid,
                        "sell_slippage_bps": (1.0 - bid / mid) * 10_000.0,
                    }
                }
            )
            rows.append(
                (
                    second * 1_000,
                    "BTCUSDT",
                    1,
                    bid,
                    ask,
                    mid,
                    (ask - bid) / mid * 10_000.0,
                    mid + 0.001,
                    10_000.0,
                    9_000.0,
                    0.05,
                    curve,
                    1.0,
                    0.0001,
                    1_000.0 + minute,
                    1_000.0 + minute,
                    1,
                    100.0,
                    80.0,
                    20.0,
                    1,
                )
            )
        connection.executemany(
            "INSERT INTO frames VALUES ("
            + ",".join("?" for _ in rows[0])
            + ")",
            rows,
        )
    (path / "frame_1s_v1.manifest.json").write_text(
        json.dumps(
            {
                "database_file": database.name,
                "symbols": ["BTCUSDT"],
                "source_session_dir": str(path / "raw"),
            }
        ),
        encoding="utf-8",
    )
    return database


def test_research_dataset_uses_executable_prices_and_captured_fees(
    tmp_path,
    monkeypatch,
):
    frames = tmp_path / "frames"
    _build_frame_fixture(frames)
    frame_validation = {
        "manifest_sha256": "a" * 64,
        "database_sha256": "b" * 64,
        "source_session_id": "fixture-session",
        "source_head_manifest_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        research_dataset,
        "validate_frame_dataset",
        lambda _: frame_validation,
    )
    monkeypatch.setattr(
        research_dataset,
        "_load_taker_fee_bps",
        lambda **_: {"BTCUSDT": 6.0},
    )

    summary = research_dataset.materialize_research_dataset(
        frame_dataset_dir=frames,
        output_dir=tmp_path / "research",
    )

    assert summary.bars == 20
    assert summary.eligible_bars == 20
    assert summary.labels == 80
    assert summary.eligible_labels == 40
    validation = research_dataset.validate_research_dataset(
        tmp_path / "research"
    )
    assert validation["dataset_valid"] is True
    assert validation["promotion_authority"] is False
    with sqlite3.connect(summary.database_path) as connection:
        label = connection.execute(
            """
            SELECT gross_mid_bps, net_execution_bps, total_cost_bps,
                   taker_fee_bps
            FROM labels
            WHERE entry_bar_timestamp_ms = 60000
              AND horizon_minutes = 5
              AND direction = 'LONG'
            """
        ).fetchone()
    assert label is not None
    assert label[0] > 0.0
    assert label[1] < label[0]
    assert label[2] > 12.0
    assert label[3] == pytest.approx(6.0)


def test_fixed_baseline_report_is_never_promotion_authority(
    tmp_path,
    monkeypatch,
):
    frames = tmp_path / "frames"
    _build_frame_fixture(frames, minutes=90)
    frame_validation = {
        "manifest_sha256": "d" * 64,
        "database_sha256": "e" * 64,
        "source_session_id": "fixture-session",
        "source_head_manifest_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        research_dataset,
        "validate_frame_dataset",
        lambda _: frame_validation,
    )
    monkeypatch.setattr(
        research_dataset,
        "_load_taker_fee_bps",
        lambda **_: {"BTCUSDT": 6.0},
    )
    research_dataset.materialize_research_dataset(
        frame_dataset_dir=frames,
        output_dir=tmp_path / "research",
    )
    research_validation = research_dataset.validate_research_dataset(
        tmp_path / "research"
    )
    monkeypatch.setattr(
        microstructure_baseline,
        "validate_research_dataset",
        lambda _: research_validation,
    )

    report = microstructure_baseline.evaluate_fixed_baselines(
        tmp_path / "research"
    )

    assert report["verdict"] == "research_only_not_promotion_evidence"
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert "evidence_duration_below_72h" in report["campaign_failures"]
    assert report["candidates"]


def test_event_study_uses_costed_barriers_and_remains_research_only(
    tmp_path,
    monkeypatch,
):
    frames = tmp_path / "frames"
    _build_frame_fixture(frames, minutes=120)
    frame_validation = {
        "manifest_sha256": "1" * 64,
        "database_sha256": "2" * 64,
        "source_session_id": "fixture-session",
        "source_head_manifest_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        research_dataset,
        "validate_frame_dataset",
        lambda _: frame_validation,
    )
    monkeypatch.setattr(
        research_dataset,
        "_load_taker_fee_bps",
        lambda **_: {"BTCUSDT": 6.0},
    )
    research_dataset.materialize_research_dataset(
        frame_dataset_dir=frames,
        output_dir=tmp_path / "research",
    )
    research_validation = research_dataset.validate_research_dataset(
        tmp_path / "research"
    )
    monkeypatch.setattr(
        event_study,
        "validate_research_dataset",
        lambda _: research_validation,
    )

    report = event_study.evaluate_event_study(tmp_path / "research")

    assert report["verdict"] == "research_only_not_promotion_evidence"
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert "evidence_duration_below_72h" in report["campaign_failures"]
    assert report["outcomes"]["15"]["exit_reasons"]["take_profit"] > 0
    long_breakout = next(
        row
        for row in report["candidates"]
        if row["candidate_id"] == "cost_gate_trend_v1@15m"
    )
    assert long_breakout["aggregate"]["closed_trades"] >= 10
    assert long_breakout["aggregate"]["mean_net_bps"] >= 10.0


def test_maker_study_requires_confirmed_trade_notional_and_blocks_promotion(
    tmp_path,
    monkeypatch,
):
    frames = tmp_path / "frames"
    _build_frame_fixture(frames, minutes=120)
    frame_validation = {
        "manifest_sha256": "4" * 64,
        "database_sha256": "5" * 64,
        "source_session_id": "fixture-session",
        "source_head_manifest_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        research_dataset,
        "validate_frame_dataset",
        lambda _: frame_validation,
    )
    monkeypatch.setattr(
        research_dataset,
        "_load_taker_fee_bps",
        lambda **_: {"BTCUSDT": 6.0},
    )
    research_dataset.materialize_research_dataset(
        frame_dataset_dir=frames,
        output_dir=tmp_path / "research",
    )
    research_validation = research_dataset.validate_research_dataset(
        tmp_path / "research"
    )
    monkeypatch.setattr(
        maker_event_study,
        "validate_research_dataset",
        lambda _: research_validation,
    )
    monkeypatch.setattr(
        maker_event_study,
        "validate_data_session",
        lambda _: {"segment_rows": [], "head_manifest_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        maker_event_study,
        "_load_maker_fee_bps",
        lambda **_: {"BTCUSDT": 2.0},
    )
    synthetic_fills = {}
    for minute in range(120):
        signal_timestamp = (minute + 1) * 60_000
        synthetic_fills[("BTCUSDT", signal_timestamp, "LONG")] = {
            "fill_timestamp_ms": signal_timestamp + 10_000,
            "qualifying_notional_usd": 30.0,
        }
        synthetic_fills[("BTCUSDT", signal_timestamp, "SHORT")] = {
            "fill_timestamp_ms": signal_timestamp + 20_000,
            "qualifying_notional_usd": 30.0,
        }
    monkeypatch.setattr(
        maker_event_study,
        "_build_fill_index",
        lambda **_: (synthetic_fills, {"BTCUSDT": 240}),
    )

    report = maker_event_study.evaluate_maker_event_study(
        tmp_path / "research"
    )

    assert report["promotion_authority"] is False
    assert "maker_queue_position_unverified" in report["campaign_failures"]
    assert report["attempts"]["15"]["maker_fill_rate"] == pytest.approx(1.0)
    candidate = next(
        row
        for row in report["candidates"]
        if row["candidate_id"] == "cost_gate_trend_v1@15m"
    )
    assert candidate["aggregate"]["closed_trades"] >= 10
    assert candidate["aggregate"]["mean_total_cost_bps"] < 12.0


def test_maker_fill_index_requires_aggressor_side_price_and_notional(tmp_path):
    database = tmp_path / "segment.sqlite"
    signal_timestamp = 60_000
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                channel TEXT NOT NULL,
                symbol TEXT NOT NULL,
                received_timestamp_utc_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        payloads = [
            {"price": "99.99", "size": "0.10", "side": "sell"},
            {"price": "99.99", "size": "0.20", "side": "sell"},
            {"price": "100.00", "size": "1.00", "side": "buy"},
        ]
        for index, payload in enumerate(payloads, start=1):
            connection.execute(
                "INSERT INTO events VALUES (?, 'trade', 'BTCUSDT', ?, ?)",
                (
                    index,
                    (signal_timestamp + index * 1_000) * 1_000_000,
                    json.dumps(payload),
                ),
            )
    bar = {
        "bar_timestamp_ms": signal_timestamp,
        "eligible": 1,
        "bid_close": 99.99,
        "ask_close": 100.01,
        "mid_close": 100.0,
        "return_5m_bps": 10.0,
        "realized_vol_5m_bps": 5.0,
        "book_imbalance_mean": 0.1,
        "microprice_edge_mean_bps": 0.1,
    }

    fills, counts = maker_event_study._build_fill_index(
        segment_rows=[{"database": str(database)}],
        bars_by_symbol={"BTCUSDT": [bar]},
    )

    assert counts == {"BTCUSDT": 3}
    assert ("BTCUSDT", signal_timestamp, "LONG") in fills
    assert (
        fills[("BTCUSDT", signal_timestamp, "LONG")][
            "qualifying_notional_usd"
        ]
        >= 25.0
    )
    assert ("BTCUSDT", signal_timestamp, "SHORT") not in fills
