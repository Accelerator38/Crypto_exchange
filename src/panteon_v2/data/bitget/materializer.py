"""Deterministic one-second research frames from sealed Bitget sessions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .segment_store import SegmentValidationError, validate_data_session


BITGET_FRAME_SCHEMA_VERSION = "panteon.bitget_frame_1s.v1"
BITGET_FRAME_DATASET_SCHEMA_VERSION = "panteon.bitget_frame_dataset.v1"
BITGET_FRAME_SCHEMA_VERSION_V2 = "panteon.bitget_frame_1s.v2"
BITGET_FRAME_DATASET_SCHEMA_VERSION_V2 = "panteon.bitget_frame_dataset.v2"
EXECUTION_NOTIONAL_LADDER_USD = (5.0, 10.0, 25.0, 50.0, 100.0)
BOOK_MAX_AGE_MS = 2_000
BOOK_MAX_AGE_MS_V2 = 60_000
TICKER_MAX_AGE_MS = 5_000
STREAM_GAP_RESET_MS = 5_000


@dataclass(frozen=True)
class FrameDatasetSummary:
    database_path: Path
    manifest_path: Path
    database_sha256: str
    manifest_sha256: str
    frames: int
    complete_frames: int


@dataclass(frozen=True)
class _FrameContract:
    frame_schema_version: str
    dataset_schema_version: str
    output_stem: str
    book_max_age_ms: int
    book_freshness_semantics: str


_FRAME_CONTRACT_V1 = _FrameContract(
    frame_schema_version=BITGET_FRAME_SCHEMA_VERSION,
    dataset_schema_version=BITGET_FRAME_DATASET_SCHEMA_VERSION,
    output_stem="frame_1s_v1",
    book_max_age_ms=BOOK_MAX_AGE_MS,
    book_freshness_semantics="fixed_received_age_v1",
)
_FRAME_CONTRACT_V2 = _FrameContract(
    frame_schema_version=BITGET_FRAME_SCHEMA_VERSION_V2,
    dataset_schema_version=BITGET_FRAME_DATASET_SCHEMA_VERSION_V2,
    output_stem="frame_1s_v2",
    book_max_age_ms=BOOK_MAX_AGE_MS_V2,
    book_freshness_semantics=(
        "event_snapshot_valid_within_stream_continuity_bounded_v2"
    ),
)


def materialize_frame_1s(
    *,
    session_dir: str | Path,
    output_dir: str | Path,
) -> FrameDatasetSummary:
    return _materialize_frame_1s(
        session_dir=session_dir,
        output_dir=output_dir,
        contract=_FRAME_CONTRACT_V1,
    )


def materialize_frame_1s_v2(
    *,
    session_dir: str | Path,
    output_dir: str | Path,
) -> FrameDatasetSummary:
    """Build event-aware frames without changing immutable v1 artifacts."""
    return _materialize_frame_1s(
        session_dir=session_dir,
        output_dir=output_dir,
        contract=_FRAME_CONTRACT_V2,
    )


def _materialize_frame_1s(
    *,
    session_dir: str | Path,
    output_dir: str | Path,
    contract: _FrameContract,
) -> FrameDatasetSummary:
    validation = validate_data_session(session_dir)
    if validation["segments_valid"] is not True:
        raise SegmentValidationError("only a sealed valid session can be materialized")
    source = Path(session_dir).resolve()
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    database_path = target / f"{contract.output_stem}.sqlite"
    manifest_path = target / f"{contract.output_stem}.manifest.json"
    if database_path.exists() or manifest_path.exists():
        raise FileExistsError(f"{contract.output_stem} output already exists")
    session = json.loads(
        (source / "session_manifest.json").read_text(encoding="utf-8")
    )
    symbols = tuple(str(item) for item in session["symbols"])
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        _create_frame_schema(connection)
        states = {symbol: _empty_state() for symbol in symbols}
        trade_buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        current_frame_ms: int | None = None
        last_received_ms: int | None = None
        continuity_id = 1
        frames = 0
        complete_frames = 0
        first_frame_ms: int | None = None
        last_frame_ms: int | None = None
        for event in _iter_events(validation["segment_rows"]):
            received_ms = int(event["received_timestamp_utc_ns"]) // 1_000_000
            if (
                last_received_ms is not None
                and received_ms - last_received_ms > STREAM_GAP_RESET_MS
            ):
                continuity_id += 1
                states = {symbol: _empty_state() for symbol in symbols}
                trade_buckets.clear()
                current_frame_ms = None
            if current_frame_ms is None:
                current_frame_ms = received_ms - received_ms % 1_000 + 1_000
            while current_frame_ms <= received_ms:
                inserted, complete = _emit_frame_batch(
                    connection=connection,
                    frame_end_ms=current_frame_ms,
                    continuity_id=continuity_id,
                    symbols=symbols,
                    states=states,
                    trade_buckets=trade_buckets,
                    source_session_id=str(session["session_id"]),
                    source_profile_sha256=str(session["profile_sha256"]),
                    source_head_manifest_sha256=str(
                        validation["head_manifest_sha256"]
                    ),
                    book_max_age_ms=contract.book_max_age_ms,
                )
                frames += inserted
                complete_frames += complete
                first_frame_ms = (
                    current_frame_ms if first_frame_ms is None else first_frame_ms
                )
                last_frame_ms = current_frame_ms
                trade_buckets.clear()
                current_frame_ms += 1_000
            _apply_event(states, trade_buckets, event, received_ms)
            last_received_ms = received_ms
        if current_frame_ms is not None:
            inserted, complete = _emit_frame_batch(
                connection=connection,
                frame_end_ms=current_frame_ms,
                continuity_id=continuity_id,
                symbols=symbols,
                states=states,
                trade_buckets=trade_buckets,
                source_session_id=str(session["session_id"]),
                source_profile_sha256=str(session["profile_sha256"]),
                source_head_manifest_sha256=str(
                    validation["head_manifest_sha256"]
                ),
                book_max_age_ms=contract.book_max_age_ms,
            )
            frames += inserted
            complete_frames += complete
            first_frame_ms = (
                current_frame_ms if first_frame_ms is None else first_frame_ms
            )
            last_frame_ms = current_frame_ms
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    if integrity != "ok":
        raise SegmentValidationError("materialized SQLite integrity failed")
    database_sha = _sha256_file(database_path)
    materializer_sha = _sha256_file(Path(__file__).resolve())
    manifest = {
        "schema_version": contract.dataset_schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_schema_version": contract.frame_schema_version,
        "source_session_dir": str(source),
        "source_session_id": session["session_id"],
        "source_profile_id": session["profile_id"],
        "source_profile_sha256": session["profile_sha256"],
        "source_revision": session["source_revision"],
        "source_collector_fingerprint_sha256": session[
            "collector_fingerprint_sha256"
        ],
        "source_head_manifest_sha256": validation["head_manifest_sha256"],
        "materializer_sha256": materializer_sha,
        "database_file": database_path.name,
        "database_sha256": database_sha,
        "sqlite_integrity": integrity,
        "symbols": list(symbols),
        "frames": frames,
        "complete_frames": complete_frames,
        "complete_frame_coverage_pct": (
            complete_frames / frames * 100.0 if frames else 0.0
        ),
        "continuity_windows": continuity_id,
        "first_frame_timestamp_ms": first_frame_ms,
        "last_frame_timestamp_ms": last_frame_ms,
        "book_max_age_ms": contract.book_max_age_ms,
        "book_freshness_semantics": contract.book_freshness_semantics,
        "ticker_max_age_ms": TICKER_MAX_AGE_MS,
        "stream_gap_reset_ms": STREAM_GAP_RESET_MS,
        "execution_notional_ladder_usd": list(EXECUTION_NOTIONAL_LADDER_USD),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_write_json(manifest_path, manifest)
    return FrameDatasetSummary(
        database_path=database_path,
        manifest_path=manifest_path,
        database_sha256=database_sha,
        manifest_sha256=str(manifest["manifest_sha256"]),
        frames=frames,
        complete_frames=complete_frames,
    )


def validate_frame_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    return _validate_frame_dataset(dataset_dir, contract=_FRAME_CONTRACT_V1)


def validate_frame_dataset_v2(dataset_dir: str | Path) -> dict[str, Any]:
    return _validate_frame_dataset(dataset_dir, contract=_FRAME_CONTRACT_V2)


def _validate_frame_dataset(
    dataset_dir: str | Path,
    *,
    contract: _FrameContract,
) -> dict[str, Any]:
    source = Path(dataset_dir).resolve()
    manifest_path = source / f"{contract.output_stem}.manifest.json"
    if not manifest_path.is_file():
        raise SegmentValidationError("frame dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed_sha = str(manifest.pop("manifest_sha256", ""))
    if _sha256_json(manifest) != claimed_sha:
        raise SegmentValidationError("frame dataset manifest hash mismatch")
    if manifest.get("schema_version") != contract.dataset_schema_version:
        raise SegmentValidationError("frame dataset schema mismatch")
    if manifest.get("frame_schema_version") != contract.frame_schema_version:
        raise SegmentValidationError("frame schema mismatch")
    if contract is _FRAME_CONTRACT_V2:
        if int(manifest.get("book_max_age_ms", -1)) != contract.book_max_age_ms:
            raise SegmentValidationError("frame book freshness ceiling mismatch")
        if (
            manifest.get("book_freshness_semantics")
            != contract.book_freshness_semantics
        ):
            raise SegmentValidationError("frame book freshness semantics mismatch")
    if manifest.get("orders_enabled") is not False:
        raise SegmentValidationError("frame dataset orders_enabled is not false")
    if manifest.get("promotion_authority") is not False:
        raise SegmentValidationError(
            "frame dataset promotion_authority is not false"
        )
    database_path = source / str(manifest["database_file"])
    if not database_path.is_file():
        raise SegmentValidationError("frame dataset database is missing")
    database_sha = _sha256_file(database_path)
    if database_sha != manifest.get("database_sha256"):
        raise SegmentValidationError("frame dataset database hash mismatch")
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        frames = int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        complete = int(
            connection.execute(
                "SELECT COUNT(*) FROM frames WHERE complete = 1"
            ).fetchone()[0]
        )
    if integrity != "ok" or manifest.get("sqlite_integrity") != "ok":
        raise SegmentValidationError("frame dataset SQLite integrity failed")
    if frames != int(manifest.get("frames", -1)):
        raise SegmentValidationError("frame dataset row count mismatch")
    if complete != int(manifest.get("complete_frames", -1)):
        raise SegmentValidationError("frame dataset complete count mismatch")
    source_validation = validate_data_session(manifest["source_session_dir"])
    if (
        source_validation["head_manifest_sha256"]
        != manifest["source_head_manifest_sha256"]
    ):
        raise SegmentValidationError("frame source session head changed")
    if source_validation["profile_sha256"] != manifest["source_profile_sha256"]:
        raise SegmentValidationError("frame source profile mismatch")
    return {
        "schema_version": "panteon.bitget_frame_validation.v1",
        "dataset_dir": str(source),
        "database_sha256": database_sha,
        "manifest_sha256": claimed_sha,
        "source_session_id": manifest["source_session_id"],
        "source_head_manifest_sha256": manifest[
            "source_head_manifest_sha256"
        ],
        "frames": frames,
        "complete_frames": complete,
        "complete_frame_coverage_pct": (
            complete / frames * 100.0 if frames else 0.0
        ),
        "continuity_windows": int(manifest["continuity_windows"]),
        "orders_enabled": False,
        "promotion_authority": False,
        "dataset_valid": True,
    }


def _create_frame_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
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
            mark_price REAL,
            index_price REAL,
            basis_bps REAL,
            funding_rate REAL,
            open_interest_base REAL,
            rest_open_interest_base REAL,
            rest_funding_rate REAL,
            trade_count INTEGER NOT NULL,
            buy_trade_notional_usd REAL NOT NULL,
            sell_trade_notional_usd REAL NOT NULL,
            signed_trade_notional_usd REAL NOT NULL,
            trade_vwap REAL,
            book_age_ms INTEGER,
            ticker_age_ms INTEGER,
            complete INTEGER NOT NULL,
            reasons_json TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_profile_sha256 TEXT NOT NULL,
            source_head_manifest_sha256 TEXT NOT NULL,
            PRIMARY KEY(frame_timestamp_ms, symbol)
        );
        CREATE INDEX frames_symbol_time_idx
            ON frames(symbol, frame_timestamp_ms);
        """
    )


def _iter_events(segment_rows: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for segment in segment_rows:
        database = Path(str(segment["database"]))
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT channel, symbol, exchange_timestamp_ms,
                       received_timestamp_utc_ns, payload_json
                FROM events
                ORDER BY received_timestamp_utc_ns, id
                """
            )
            for row in rows:
                yield {
                    "channel": str(row["channel"]),
                    "symbol": str(row["symbol"]),
                    "exchange_timestamp_ms": int(row["exchange_timestamp_ms"]),
                    "received_timestamp_utc_ns": int(
                        row["received_timestamp_utc_ns"]
                    ),
                    "payload": json.loads(str(row["payload_json"])),
                }


def _apply_event(
    states: dict[str, dict[str, Any]],
    trades: dict[str, list[Mapping[str, Any]]],
    event: Mapping[str, Any],
    received_ms: int,
) -> None:
    symbol = str(event["symbol"])
    if symbol not in states:
        return
    channel = str(event["channel"])
    payload = event["payload"]
    if channel == "trade":
        trades[symbol].append(payload)
    elif channel == "books5":
        states[symbol]["book"] = payload
        states[symbol]["book_received_ms"] = received_ms
    elif channel == "ticker":
        states[symbol]["ticker"] = payload
        states[symbol]["ticker_received_ms"] = received_ms
    elif channel == "rest_open_interest":
        states[symbol]["rest_oi"] = payload
    elif channel == "rest_funding":
        states[symbol]["rest_funding"] = payload


def _emit_frame_batch(
    *,
    connection: sqlite3.Connection,
    frame_end_ms: int,
    continuity_id: int,
    symbols: Sequence[str],
    states: Mapping[str, Mapping[str, Any]],
    trade_buckets: Mapping[str, Sequence[Mapping[str, Any]]],
    source_session_id: str,
    source_profile_sha256: str,
    source_head_manifest_sha256: str,
    book_max_age_ms: int = BOOK_MAX_AGE_MS,
) -> tuple[int, int]:
    rows = []
    complete_count = 0
    for symbol in symbols:
        row = _build_frame(
            frame_end_ms=frame_end_ms,
            symbol=symbol,
            continuity_id=continuity_id,
            state=states[symbol],
            trades=trade_buckets.get(symbol, ()),
            source_session_id=source_session_id,
            source_profile_sha256=source_profile_sha256,
            source_head_manifest_sha256=source_head_manifest_sha256,
            book_max_age_ms=book_max_age_ms,
        )
        rows.append(row)
        complete_count += int(row[26])
    placeholders = ", ".join("?" for _ in rows[0])
    connection.executemany(
        f"INSERT INTO frames VALUES ({placeholders})",
        rows,
    )
    return len(rows), complete_count


def _build_frame(
    *,
    frame_end_ms: int,
    symbol: str,
    continuity_id: int,
    state: Mapping[str, Any],
    trades: Sequence[Mapping[str, Any]],
    source_session_id: str,
    source_profile_sha256: str,
    source_head_manifest_sha256: str,
    book_max_age_ms: int = BOOK_MAX_AGE_MS,
) -> tuple[Any, ...]:
    book = state.get("book") or {}
    ticker = state.get("ticker") or {}
    bids = _levels(book.get("bids"))
    asks = _levels(book.get("asks"))
    bid = bids[0][0] if bids else _positive_float(ticker.get("bidPr"))
    ask = asks[0][0] if asks else _positive_float(ticker.get("askPr"))
    mid = (bid + ask) / 2.0 if bid > 0.0 and ask >= bid else 0.0
    spread = (ask - bid) / mid * 10_000.0 if mid > 0.0 else None
    bid_size = bids[0][1] if bids else 0.0
    ask_size = asks[0][1] if asks else 0.0
    microprice = (
        (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
        if bid_size + ask_size > 0.0
        else None
    )
    bid_depth = sum(price * size for price, size in bids)
    ask_depth = sum(price * size for price, size in asks)
    imbalance = (
        (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if bid_depth + ask_depth > 0.0
        else None
    )
    execution_curve = _execution_curve(bids=bids, asks=asks, mid=mid)
    mark = _positive_float(ticker.get("markPrice"))
    index_price = _positive_float(ticker.get("indexPrice"))
    basis = (
        (mark / index_price - 1.0) * 10_000.0
        if mark > 0.0 and index_price > 0.0
        else None
    )
    funding = _finite_float(ticker.get("fundingRate"))
    open_interest = _positive_float(ticker.get("holdingAmount"))
    rest_oi = _positive_float(
        (state.get("rest_oi") or {}).get("open_interest_base")
    )
    rest_funding = _finite_float(
        (state.get("rest_funding") or {}).get("funding_rate")
    )
    buy_notional = 0.0
    sell_notional = 0.0
    total_base = 0.0
    total_notional = 0.0
    valid_trade_count = 0
    for trade in trades:
        price = _positive_float(trade.get("price"))
        size = _positive_float(trade.get("size"))
        if price <= 0.0 or size <= 0.0:
            continue
        notional = price * size
        if str(trade.get("side") or "").lower() == "buy":
            buy_notional += notional
        else:
            sell_notional += notional
        total_base += size
        total_notional += notional
        valid_trade_count += 1
    trade_vwap = total_notional / total_base if total_base > 0.0 else None
    book_received = state.get("book_received_ms")
    ticker_received = state.get("ticker_received_ms")
    book_age = (
        frame_end_ms - int(book_received) if book_received is not None else None
    )
    ticker_age = (
        frame_end_ms - int(ticker_received)
        if ticker_received is not None
        else None
    )
    reasons = []
    if book_age is None or not 0 <= book_age <= book_max_age_ms:
        reasons.append("book_stale_or_missing")
    if ticker_age is None or not 0 <= ticker_age <= TICKER_MAX_AGE_MS:
        reasons.append("ticker_stale_or_missing")
    if mid <= 0.0:
        reasons.append("invalid_top_of_book")
    five = execution_curve.get("5")
    if not five or five["buy_slippage_bps"] is None:
        reasons.append("buy_depth_below_5_usd")
    if not five or five["sell_slippage_bps"] is None:
        reasons.append("sell_depth_below_5_usd")
    complete = not reasons
    return (
        frame_end_ms,
        symbol,
        continuity_id,
        bid or None,
        ask or None,
        mid or None,
        spread,
        microprice,
        bid_depth or None,
        ask_depth or None,
        imbalance,
        _canonical_json(execution_curve),
        mark or None,
        index_price or None,
        basis,
        funding,
        open_interest or None,
        rest_oi or None,
        rest_funding,
        valid_trade_count,
        buy_notional,
        sell_notional,
        buy_notional - sell_notional,
        trade_vwap,
        book_age,
        ticker_age,
        int(complete),
        _canonical_json(reasons),
        source_session_id,
        source_profile_sha256,
        source_head_manifest_sha256,
    )


def _execution_curve(
    *,
    bids: Sequence[tuple[float, float]],
    asks: Sequence[tuple[float, float]],
    mid: float,
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for notional in EXECUTION_NOTIONAL_LADDER_USD:
        buy_vwap = _book_vwap(asks, notional, mid)
        sell_vwap = _book_vwap(bids, notional, mid)
        result[f"{notional:g}"] = {
            "buy_vwap": buy_vwap,
            "buy_slippage_bps": (
                (buy_vwap / mid - 1.0) * 10_000.0
                if buy_vwap is not None and mid > 0.0
                else None
            ),
            "sell_vwap": sell_vwap,
            "sell_slippage_bps": (
                (1.0 - sell_vwap / mid) * 10_000.0
                if sell_vwap is not None and mid > 0.0
                else None
            ),
        }
    return result


def _book_vwap(
    levels: Sequence[tuple[float, float]],
    target_notional_usd: float,
    mid: float,
) -> float | None:
    if mid <= 0.0:
        return None
    remaining_base = float(target_notional_usd) / mid
    filled_base = 0.0
    quote = 0.0
    for price, size in levels:
        take = min(remaining_base, size)
        filled_base += take
        quote += take * price
        remaining_base -= take
        if remaining_base <= 1e-12:
            break
    if remaining_base > 1e-12 or filled_base <= 0.0:
        return None
    return quote / filled_base


def _levels(raw: Any) -> list[tuple[float, float]]:
    result = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return result
    for level in raw:
        if not isinstance(level, Sequence) or len(level) < 2:
            continue
        price = _positive_float(level[0])
        size = _positive_float(level[1])
        if price > 0.0 and size > 0.0:
            result.append((price, size))
    return result


def _empty_state() -> dict[str, Any]:
    return {
        "book": None,
        "book_received_ms": None,
        "ticker": None,
        "ticker_received_ms": None,
        "rest_oi": None,
        "rest_funding": None,
    }


def _positive_float(value: Any) -> float:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed > 0.0 else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
