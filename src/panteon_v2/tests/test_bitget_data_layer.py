from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from panteon_v2.data.bitget import (
    BitgetDataProfileError,
    BitgetDataCollectorLock,
    BitgetPublicMessageDecoder,
    BitgetPublicRestReconciler,
    BitgetSegmentStore,
    SegmentValidationError,
    build_subscription_request,
    load_bitget_data_profile,
    materialize_frame_1s,
    recover_unsealed_sessions,
    validate_data_session,
    validate_frame_dataset,
)
from panteon_v2.data.bitget import materializer as frame_materializer


ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = (
    ROOT
    / "configs"
    / "bitget_data_profiles"
    / "bitget_full8_microstructure_v1.json"
)
REVISION = "a" * 40
FINGERPRINT = "b" * 64


def _profile():
    return load_bitget_data_profile(PROFILE_PATH)


def test_profile_is_exact_public_full8_contract(tmp_path):
    profile = _profile()

    assert profile.orders_enabled is False
    assert profile.promotion_authority is False
    assert profile.channels == ("trade", "books5", "ticker")
    assert len(profile.symbols) == 8
    subscription = build_subscription_request(profile)
    assert subscription["op"] == "subscribe"
    assert len(subscription["args"]) == 24
    assert {
        item["channel"] for item in subscription["args"]
    } == {"trade", "books5", "ticker"}

    unsafe = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    unsafe["orders_enabled"] = True
    unsafe_path = tmp_path / "unsafe.json"
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(BitgetDataProfileError, match="orders_enabled"):
        load_bitget_data_profile(unsafe_path)


def test_data_directory_lock_rejects_second_collector(tmp_path):
    with BitgetDataCollectorLock(tmp_path):
        with pytest.raises(RuntimeError, match="another Bitget data collector"):
            with BitgetDataCollectorLock(tmp_path):
                pass


def test_stale_legacy_lock_survives_pid_reuse_only_with_fresh_heartbeat(
    tmp_path,
):
    lock_path = tmp_path / ".collector.lock"
    lock_path.write_text(f"{os.getpid()}:old-token", encoding="ascii")
    old = time.time() - BitgetDataCollectorLock.STALE_HEARTBEAT_SECONDS - 10
    os.utime(lock_path, (old, old))

    with BitgetDataCollectorLock(tmp_path):
        assert json.loads(lock_path.read_text(encoding="ascii"))["pid"] == os.getpid()


def test_decoder_preserves_trade_book_and_ticker_exchange_time():
    decoder = BitgetPublicMessageDecoder(_profile())
    trade, trade_quality = decoder.decode(
        json.dumps(
            {
                "action": "snapshot",
                "arg": {
                    "instType": "USDT-FUTURES",
                    "channel": "trade",
                    "instId": "BTCUSDT",
                },
                "data": [
                    {
                        "ts": "1760000000002",
                        "price": "60000.2",
                        "size": "0.02",
                        "side": "sell",
                        "tradeId": "t-2",
                    },
                    {
                        "ts": "1760000000001",
                        "price": "60000.1",
                        "size": "0.01",
                        "side": "buy",
                        "tradeId": "t-1",
                    }
                ],
                "ts": 1760000000010,
            }
        )
    )
    assert trade_quality == []
    assert trade[0].event_key == "trade:t-1"
    assert trade[0].exchange_timestamp_ms == 1760000000001
    assert trade[1].event_key == "trade:t-2"

    book, book_quality = decoder.decode(
        json.dumps(
            {
                "action": "snapshot",
                "arg": {
                    "instType": "USDT-FUTURES",
                    "channel": "books5",
                    "instId": "BTCUSDT",
                },
                "data": [
                    {
                        "asks": [["60000.2", "2"]],
                        "bids": [["60000.1", "3"]],
                        "seq": 123,
                        "ts": "1760000000002",
                    }
                ],
            }
        )
    )
    assert book_quality == []
    assert book[0].sequence == 123
    assert book[0].event_key == "book:123"

    ticker, ticker_quality = decoder.decode(
        json.dumps(
            {
                "action": "snapshot",
                "arg": {
                    "instType": "USDT-FUTURES",
                    "channel": "ticker",
                    "instId": "BTCUSDT",
                },
                "data": [
                    {
                        "bidPr": "60000.1",
                        "askPr": "60000.2",
                        "markPrice": "60000.15",
                        "indexPrice": "60000.0",
                        "holdingAmount": "1000",
                        "fundingRate": "0.0001",
                        "ts": "1760000000003",
                    }
                ],
            }
        )
    )
    assert ticker_quality == []
    assert ticker[0].channel == "ticker"
    assert ticker[0].exchange_timestamp_ms == 1760000000003


def test_decoder_reports_nonincreasing_book_sequence():
    decoder = BitgetPublicMessageDecoder(_profile())

    def message(sequence):
        return json.dumps(
            {
                "action": "snapshot",
                "arg": {"channel": "books5", "instId": "BTCUSDT"},
                "data": [
                    {
                        "asks": [["2", "1"]],
                        "bids": [["1", "1"]],
                        "seq": sequence,
                        "ts": "1760000000000",
                    }
                ],
            }
        )

    decoder.decode(message(100))
    _, quality = decoder.decode(message(99))

    assert [item["kind"] for item in quality] == [
        "book_sequence_nonincreasing"
    ]


def test_segment_store_rotates_seals_and_validates_hash_chain(tmp_path):
    profile = _profile()
    store = BitgetSegmentStore(
        data_dir=tmp_path,
        profile=profile,
        source_revision=REVISION,
        collector_fingerprint_sha256=FINGERPRINT,
        session_id="session-1",
    )
    first_ns = 1_760_000_000_000_000_000
    second_ns = first_ns + profile.segment_seconds * 1_000_000_000
    for index, received_ns in enumerate((first_ns, second_ns), start=1):
        store.append_event(
            channel="trade",
            symbol="BTCUSDT",
            exchange_timestamp_ms=received_ns // 1_000_000,
            received_timestamp_utc_ns=received_ns,
            received_timestamp_monotonic_ns=index,
            event_key=f"trade:{index}",
            action="snapshot",
            sequence=None,
            payload={"tradeId": str(index)},
        )
    store.close(reason="test_complete")

    report = validate_data_session(store.session_dir)

    assert report["segments_valid"] is True
    assert report["segments"] == 2
    assert report["events"] == 2
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert report["collector_fingerprint_sha256"] == FINGERPRINT
    assert len(report["head_manifest_sha256"]) == 64

    first_database = Path(report["segment_rows"][0]["database"])
    with sqlite3.connect(first_database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_segment_validator_rejects_database_tampering(tmp_path):
    store = BitgetSegmentStore(
        data_dir=tmp_path,
        profile=_profile(),
        source_revision=REVISION,
        collector_fingerprint_sha256=FINGERPRINT,
        session_id="session-tamper",
    )
    now_ns = 1_760_000_000_000_000_000
    store.append_event(
        channel="ticker",
        symbol="BTCUSDT",
        exchange_timestamp_ms=now_ns // 1_000_000,
        received_timestamp_utc_ns=now_ns,
        received_timestamp_monotonic_ns=1,
        event_key="ticker:1",
        action="snapshot",
        sequence=None,
        payload={"bidPr": "1", "askPr": "2"},
    )
    store.close(reason="test_complete")
    report = validate_data_session(store.session_dir)
    database = Path(report["segment_rows"][0]["database"])
    with database.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(SegmentValidationError, match="database hash mismatch"):
        validate_data_session(store.session_dir)


def test_orphan_wal_is_recovered_and_sealed(tmp_path):
    store = BitgetSegmentStore(
        data_dir=tmp_path,
        profile=_profile(),
        source_revision=REVISION,
        collector_fingerprint_sha256=FINGERPRINT,
        session_id="session-recovery",
    )
    now_ns = 1_760_000_000_000_000_000
    store.append_event(
        channel="trade",
        symbol="BTCUSDT",
        exchange_timestamp_ms=now_ns // 1_000_000,
        received_timestamp_utc_ns=now_ns,
        received_timestamp_monotonic_ns=1,
        event_key="trade:recovery",
        action="snapshot",
        sequence=None,
        payload={"tradeId": "recovery"},
    )
    assert store._connection is not None
    store._connection.commit()
    store._connection.close()
    store._connection = None

    recovered = recover_unsealed_sessions(tmp_path)
    report = validate_data_session(store.session_dir)
    manifest = json.loads(
        Path(report["segment_rows"][0]["manifest"]).read_text(encoding="utf-8")
    )

    assert len(recovered) == 1
    assert recovered[0].session_id == "session-recovery"
    assert report["segments_valid"] is True
    assert report["events"] == 1
    assert manifest["recovery"]["recovered"] is True
    assert manifest["seal_reason"] == "recovered_after_unclean_shutdown"


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeRestSession:
    def __init__(self, *, now_ms, symbols):
        self.now_ms = now_ms
        self.symbols = tuple(symbols)

    def get(self, url, params, timeout):
        assert url.startswith("https://api.bitget.com/")
        assert timeout == 10.0
        if url.endswith("/candles"):
            start = int(params["startTime"]) + 60_000
            return _FakeResponse(
                {
                    "code": "00000",
                    "msg": "success",
                    "requestTime": self.now_ms,
                    "data": [
                        [
                            str(start),
                            "100",
                            "102",
                            "99",
                            "101",
                            "10",
                            "1010",
                        ]
                    ],
                }
            )
        if url.endswith("/open-interest"):
            return _FakeResponse(
                {
                    "code": "00000",
                    "msg": "success",
                    "requestTime": self.now_ms,
                    "data": {
                        "openInterestList": [
                            {"symbol": params["symbol"], "size": "1000"}
                        ],
                        "ts": str(self.now_ms),
                    },
                }
            )
        if url.endswith("/current-fund-rate"):
            return _FakeResponse(
                {
                    "code": "00000",
                    "msg": "success",
                    "requestTime": self.now_ms,
                    "data": [
                        {
                            "symbol": symbol,
                            "fundingRate": "0.0001",
                            "fundingRateInterval": "8",
                            "nextUpdate": str(self.now_ms + 3_600_000),
                            "minFundingRate": "-0.003",
                            "maxFundingRate": "0.003",
                        }
                        for symbol in self.symbols
                    ],
                }
            )
        if url.endswith("/contracts"):
            return _FakeResponse(
                {
                    "code": "00000",
                    "msg": "success",
                    "requestTime": self.now_ms,
                    "data": [
                        {
                            "symbol": symbol,
                            "symbolStatus": "normal",
                            "minTradeNum": "0.001",
                            "minTradeUSDT": "5",
                            "sizeMultiplier": "0.001",
                            "pricePlace": "1",
                            "priceEndStep": "1",
                            "volumePlace": "3",
                            "makerFeeRate": "0.0002",
                            "takerFeeRate": "0.0006",
                            "maxMarketOrderQty": "100",
                            "maxOrderQty": "200",
                            "fundInterval": "8",
                            "maintainTime": "0",
                        }
                        for symbol in self.symbols
                    ],
                }
            )
        raise AssertionError(url)


def test_public_rest_reconciliation_produces_all_fixed_facts():
    profile = _profile()
    now_ms = 1_760_000_040_500
    reconciler = BitgetPublicRestReconciler(
        profile,
        session=_FakeRestSession(now_ms=now_ms, symbols=profile.symbols),
    )

    events, quality = reconciler.fetch_cycle(
        include_rules=True,
        now_ms=now_ms,
    )

    assert quality == []
    assert len(events) == 32
    assert {event.channel for event in events} == {
        "rest_candle_1m",
        "rest_open_interest",
        "rest_funding",
        "rest_instrument_rules",
    }
    assert {event.symbol for event in events} == set(profile.symbols)


def test_frame_materializer_uses_receive_time_and_executable_book(tmp_path):
    store = BitgetSegmentStore(
        data_dir=tmp_path / "raw",
        profile=_profile(),
        source_revision=REVISION,
        collector_fingerprint_sha256=FINGERPRINT,
        session_id="session-frame",
    )
    base_ns = 1_760_000_000_100_000_000
    store.append_event(
        channel="books5",
        symbol="BTCUSDT",
        exchange_timestamp_ms=1_760_000_000_050,
        received_timestamp_utc_ns=base_ns,
        received_timestamp_monotonic_ns=1,
        event_key="book:1",
        action="snapshot",
        sequence=1,
        payload={
            "bids": [["99.9", "10"], ["99.8", "10"]],
            "asks": [["100.1", "10"], ["100.2", "10"]],
            "seq": 1,
            "ts": "1760000000050",
        },
    )
    store.append_event(
        channel="ticker",
        symbol="BTCUSDT",
        exchange_timestamp_ms=1_760_000_000_100,
        received_timestamp_utc_ns=base_ns + 100_000_000,
        received_timestamp_monotonic_ns=2,
        event_key="ticker:1",
        action="snapshot",
        sequence=None,
        payload={
            "bidPr": "99.9",
            "askPr": "100.1",
            "markPrice": "100.05",
            "indexPrice": "100",
            "holdingAmount": "1000",
            "fundingRate": "0.0001",
            "ts": "1760000000100",
        },
    )
    store.append_event(
        channel="trade",
        symbol="BTCUSDT",
        exchange_timestamp_ms=1_760_000_000_200,
        received_timestamp_utc_ns=base_ns + 200_000_000,
        received_timestamp_monotonic_ns=3,
        event_key="trade:1",
        action="snapshot",
        sequence=None,
        payload={
            "price": "100.1",
            "size": "1",
            "side": "buy",
            "tradeId": "1",
            "ts": "1760000000200",
        },
    )
    store.close(reason="test_complete")

    summary = materialize_frame_1s(
        session_dir=store.session_dir,
        output_dir=tmp_path / "frames",
    )

    assert summary.frames == 8
    assert summary.complete_frames == 1
    validation = validate_frame_dataset(tmp_path / "frames")
    assert validation["dataset_valid"] is True
    assert validation["source_session_id"] == "session-frame"
    assert validation["frames"] == 8
    assert validation["complete_frames"] == 1
    with sqlite3.connect(summary.database_path) as connection:
        row = connection.execute(
            """
            SELECT mid_price, spread_bps, basis_bps, trade_count,
                   signed_trade_notional_usd, complete, execution_curve_json
            FROM frames
            WHERE symbol = 'BTCUSDT'
            """
        ).fetchone()
    assert row[0] == pytest.approx(100.0)
    assert row[1] == pytest.approx(20.0)
    assert row[2] == pytest.approx(5.0)
    assert row[3] == 1
    assert row[4] == pytest.approx(100.1)
    assert row[5] == 1
    execution = json.loads(row[6])
    assert execution["100"]["buy_slippage_bps"] == pytest.approx(10.0)

    with summary.database_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(
        SegmentValidationError,
        match="frame dataset database hash mismatch",
    ):
        validate_frame_dataset(tmp_path / "frames")


def test_frame_v2_keeps_unchanged_book_inside_stream_continuity():
    state = {
        "book": {
            "bids": [["99.9", "10"], ["99.8", "10"]],
            "asks": [["100.1", "10"], ["100.2", "10"]],
        },
        "ticker": {
            "bidPr": "99.9",
            "askPr": "100.1",
            "markPrice": "100.05",
            "indexPrice": "100",
            "holdingAmount": "1000",
            "fundingRate": "0.0001",
        },
        "book_received_ms": 1_000,
        "ticker_received_ms": 9_000,
    }
    kwargs = {
        "frame_end_ms": 10_000,
        "symbol": "BTCUSDT",
        "continuity_id": 1,
        "state": state,
        "trades": (),
        "source_session_id": "fixture",
        "source_profile_sha256": "a" * 64,
        "source_head_manifest_sha256": "b" * 64,
    }

    v1 = frame_materializer._build_frame(**kwargs)
    v2 = frame_materializer._build_frame(
        **kwargs,
        book_max_age_ms=frame_materializer.BOOK_MAX_AGE_MS_V2,
    )

    assert v1[26] == 0
    assert json.loads(v1[27]) == ["book_stale_or_missing"]
    assert v2[26] == 1
    assert json.loads(v2[27]) == []
