"""Deterministic minute features and executable forward labels for Bitget."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .materializer import validate_frame_dataset
from .segment_store import SegmentValidationError, validate_data_session


RESEARCH_SCHEMA_VERSION = "panteon.bitget_microstructure_research.v1"
RESEARCH_DATASET_SCHEMA_VERSION = "panteon.bitget_research_dataset.v1"
BAR_SECONDS = 60
MIN_COMPLETE_SECONDS = 57
FORWARD_HORIZONS_MINUTES = (5, 15)
EXECUTION_NOTIONAL_USD = 25.0


@dataclass(frozen=True)
class ResearchDatasetSummary:
    database_path: Path
    manifest_path: Path
    database_sha256: str
    manifest_sha256: str
    bars: int
    eligible_bars: int
    labels: int
    eligible_labels: int


def materialize_research_dataset(
    *,
    frame_dataset_dir: str | Path,
    output_dir: str | Path,
) -> ResearchDatasetSummary:
    """Build one fixed research dataset from an immutable frame dataset."""
    validation = validate_frame_dataset(frame_dataset_dir)
    source = Path(frame_dataset_dir).resolve()
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    database_path = target / "microstructure_research_v1.sqlite"
    manifest_path = target / "microstructure_research_v1.manifest.json"
    if database_path.exists() or manifest_path.exists():
        raise FileExistsError("microstructure research output already exists")

    frame_manifest = _read_json(source / "frame_1s_v1.manifest.json")
    symbols = tuple(str(item) for item in frame_manifest["symbols"])
    source_session_dir = Path(str(frame_manifest["source_session_dir"]))
    fee_bps = _load_taker_fee_bps(
        session_dir=source_session_dir,
        symbols=symbols,
    )

    connection = sqlite3.connect(database_path)
    bars: list[dict[str, Any]] = []
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        _create_schema(connection)
        with sqlite3.connect(
            f"file:{source / str(frame_manifest['database_file'])}?mode=ro",
            uri=True,
        ) as frame_connection:
            frame_connection.row_factory = sqlite3.Row
            for symbol in symbols:
                rows = frame_connection.execute(
                    """
                    SELECT frame_timestamp_ms, continuity_id, bid_price,
                           ask_price, mid_price, spread_bps, microprice,
                           bid_depth_5_usd, ask_depth_5_usd,
                           book_imbalance_5, execution_curve_json, basis_bps,
                           funding_rate, open_interest_base,
                           rest_open_interest_base, trade_count,
                           buy_trade_notional_usd,
                           sell_trade_notional_usd,
                           signed_trade_notional_usd, complete
                    FROM frames
                    WHERE symbol = ?
                    ORDER BY frame_timestamp_ms
                    """,
                    (symbol,),
                )
                symbol_bars = _build_symbol_bars(
                    symbol=symbol,
                    frame_rows=rows,
                    taker_fee_bps=fee_bps[symbol],
                )
                bars.extend(symbol_bars)
                connection.executemany(
                    """
                    INSERT INTO bars VALUES (
                        :bar_timestamp_ms, :symbol, :continuity_id,
                        :observed_seconds, :complete_seconds, :eligible,
                        :reasons_json, :mid_open, :mid_close, :bid_close,
                        :ask_close, :spread_mean_bps, :spread_max_bps,
                        :microprice_edge_mean_bps,
                        :book_imbalance_mean, :bid_depth_close_usd,
                        :ask_depth_close_usd, :trade_count,
                        :buy_trade_notional_usd,
                        :sell_trade_notional_usd,
                        :signed_trade_notional_usd, :flow_imbalance,
                        :basis_close_bps, :funding_close,
                        :open_interest_close_base, :return_1m_bps,
                        :return_5m_bps, :realized_vol_5m_bps,
                        :oi_change_5m_bps, :execution_curve_json,
                        :taker_fee_bps
                    )
                    """,
                    symbol_bars,
                )
                label_rows = _build_forward_labels(symbol_bars)
                connection.executemany(
                    """
                    INSERT INTO labels VALUES (
                        :entry_bar_timestamp_ms, :symbol,
                        :horizon_minutes, :direction, :eligible,
                        :reasons_json, :notional_usd, :entry_mid,
                        :exit_mid, :entry_execution_price,
                        :exit_execution_price, :gross_mid_bps,
                        :net_execution_bps, :total_cost_bps,
                        :taker_fee_bps
                    )
                    """,
                    label_rows,
                )
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        counts = _dataset_counts(connection)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    if integrity != "ok":
        raise SegmentValidationError("research dataset SQLite integrity failed")

    database_sha = _sha256_file(database_path)
    manifest: dict[str, Any] = {
        "schema_version": RESEARCH_DATASET_SCHEMA_VERSION,
        "research_schema_version": RESEARCH_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_frame_dataset_dir": str(source),
        "source_frame_manifest_sha256": validation["manifest_sha256"],
        "source_frame_database_sha256": validation["database_sha256"],
        "source_session_id": validation["source_session_id"],
        "source_head_manifest_sha256": validation[
            "source_head_manifest_sha256"
        ],
        "database_file": database_path.name,
        "database_sha256": database_sha,
        "sqlite_integrity": integrity,
        "symbols": list(symbols),
        "bar_seconds": BAR_SECONDS,
        "min_complete_seconds": MIN_COMPLETE_SECONDS,
        "forward_horizons_minutes": list(FORWARD_HORIZONS_MINUTES),
        "execution_notional_usd": EXECUTION_NOTIONAL_USD,
        "taker_fee_bps_by_symbol": fee_bps,
        **counts,
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_write_json(manifest_path, manifest)
    return ResearchDatasetSummary(
        database_path=database_path,
        manifest_path=manifest_path,
        database_sha256=database_sha,
        manifest_sha256=str(manifest["manifest_sha256"]),
        bars=int(counts["bars"]),
        eligible_bars=int(counts["eligible_bars"]),
        labels=int(counts["labels"]),
        eligible_labels=int(counts["eligible_labels"]),
    )


def validate_research_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    source = Path(dataset_dir).resolve()
    manifest_path = source / "microstructure_research_v1.manifest.json"
    if not manifest_path.is_file():
        raise SegmentValidationError("research dataset manifest is missing")
    manifest = _read_json(manifest_path)
    claimed_sha = str(manifest.pop("manifest_sha256", ""))
    if _sha256_json(manifest) != claimed_sha:
        raise SegmentValidationError("research dataset manifest hash mismatch")
    if manifest.get("research_only") is not True:
        raise SegmentValidationError("research_only is not true")
    if manifest.get("orders_enabled") is not False:
        raise SegmentValidationError("research dataset orders_enabled is not false")
    if manifest.get("promotion_authority") is not False:
        raise SegmentValidationError(
            "research dataset promotion_authority is not false"
        )
    database_path = source / str(manifest["database_file"])
    if not database_path.is_file():
        raise SegmentValidationError("research dataset database is missing")
    database_sha = _sha256_file(database_path)
    if database_sha != manifest.get("database_sha256"):
        raise SegmentValidationError("research dataset database hash mismatch")
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        counts = _dataset_counts(connection)
    if integrity != "ok" or manifest.get("sqlite_integrity") != "ok":
        raise SegmentValidationError("research dataset SQLite integrity failed")
    for key, value in counts.items():
        if int(manifest.get(key, -1)) != int(value):
            raise SegmentValidationError(f"research dataset {key} mismatch")
    frame_validation = validate_frame_dataset(
        manifest["source_frame_dataset_dir"]
    )
    if (
        frame_validation["manifest_sha256"]
        != manifest["source_frame_manifest_sha256"]
    ):
        raise SegmentValidationError("research source frame manifest changed")
    if (
        frame_validation["database_sha256"]
        != manifest["source_frame_database_sha256"]
    ):
        raise SegmentValidationError("research source frame database changed")
    return {
        "schema_version": "panteon.bitget_research_validation.v1",
        "dataset_dir": str(source),
        "database_sha256": database_sha,
        "manifest_sha256": claimed_sha,
        **counts,
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
        "dataset_valid": True,
    }


def _build_symbol_bars(
    *,
    symbol: str,
    frame_rows: Iterable[sqlite3.Row],
    taker_fee_bps: float,
) -> list[dict[str, Any]]:
    buckets: list[list[sqlite3.Row]] = []
    current_key: int | None = None
    current: list[sqlite3.Row] = []
    for row in frame_rows:
        timestamp = int(row["frame_timestamp_ms"])
        bar_end = ((timestamp - 1) // 60_000 + 1) * 60_000
        if current_key is not None and bar_end != current_key:
            buckets.append(current)
            current = []
        current_key = bar_end
        current.append(row)
    if current:
        buckets.append(current)

    bars = [
        _aggregate_minute(
            symbol=symbol,
            rows=rows,
            taker_fee_bps=taker_fee_bps,
        )
        for rows in buckets
    ]
    returns: list[float | None] = []
    for index, bar in enumerate(bars):
        previous = bars[index - 1] if index else None
        return_1m = _return_bps(
            float(previous["mid_close"]) if previous else None,
            float(bar["mid_close"]) if bar["mid_close"] is not None else None,
        )
        returns.append(return_1m)
        bar["return_1m_bps"] = return_1m
        bar["return_5m_bps"] = (
            _return_bps(
                float(bars[index - 5]["mid_close"]),
                float(bar["mid_close"]),
            )
            if index >= 5
            and bars[index - 5]["mid_close"] is not None
            and bar["mid_close"] is not None
            else None
        )
        recent_returns = [
            float(value)
            for value in returns[max(0, index - 4) : index + 1]
            if value is not None
        ]
        bar["realized_vol_5m_bps"] = (
            statistics.pstdev(recent_returns)
            if len(recent_returns) == 5
            else None
        )
        bar["oi_change_5m_bps"] = (
            _return_bps(
                _as_float(bars[index - 5]["open_interest_close_base"]),
                _as_float(bar["open_interest_close_base"]),
            )
            if index >= 5
            else None
        )
    return bars


def _aggregate_minute(
    *,
    symbol: str,
    rows: Sequence[sqlite3.Row],
    taker_fee_bps: float,
) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    continuity = {int(row["continuity_id"]) for row in rows}
    complete_seconds = sum(int(row["complete"]) for row in rows)
    reasons: list[str] = []
    if len(rows) < MIN_COMPLETE_SECONDS:
        reasons.append("observed_seconds_below_57")
    if complete_seconds < MIN_COMPLETE_SECONDS:
        reasons.append("complete_seconds_below_57")
    if len(continuity) != 1:
        reasons.append("continuity_change_inside_bar")
    if not int(last["complete"]):
        reasons.append("decision_frame_incomplete")
    if last["execution_curve_json"] is None:
        reasons.append("execution_curve_missing")
    mids = [_as_float(row["mid_price"]) for row in rows]
    spreads = [_as_float(row["spread_bps"]) for row in rows]
    imbalances = [_as_float(row["book_imbalance_5"]) for row in rows]
    micro_edges = [
        (float(row["microprice"]) / float(row["mid_price"]) - 1.0) * 10_000.0
        for row in rows
        if _as_float(row["microprice"]) is not None
        and _as_float(row["mid_price"]) is not None
        and float(row["mid_price"]) > 0.0
    ]
    buy_notional = sum(float(row["buy_trade_notional_usd"]) for row in rows)
    sell_notional = sum(float(row["sell_trade_notional_usd"]) for row in rows)
    total_notional = buy_notional + sell_notional
    oi_close = (
        _as_float(last["rest_open_interest_base"])
        or _as_float(last["open_interest_base"])
    )
    return {
        "bar_timestamp_ms": ((int(last["frame_timestamp_ms"]) - 1) // 60_000 + 1)
        * 60_000,
        "symbol": symbol,
        "continuity_id": int(last["continuity_id"]),
        "observed_seconds": len(rows),
        "complete_seconds": complete_seconds,
        "eligible": int(not reasons),
        "reasons_json": _canonical_json(reasons),
        "mid_open": _first_not_none(mids),
        "mid_close": _last_not_none(mids),
        "bid_close": _as_float(last["bid_price"]),
        "ask_close": _as_float(last["ask_price"]),
        "spread_mean_bps": _mean_not_none(spreads),
        "spread_max_bps": max(
            (value for value in spreads if value is not None),
            default=None,
        ),
        "microprice_edge_mean_bps": _mean_not_none(micro_edges),
        "book_imbalance_mean": _mean_not_none(imbalances),
        "bid_depth_close_usd": _as_float(last["bid_depth_5_usd"]),
        "ask_depth_close_usd": _as_float(last["ask_depth_5_usd"]),
        "trade_count": sum(int(row["trade_count"]) for row in rows),
        "buy_trade_notional_usd": buy_notional,
        "sell_trade_notional_usd": sell_notional,
        "signed_trade_notional_usd": buy_notional - sell_notional,
        "flow_imbalance": (
            (buy_notional - sell_notional) / total_notional
            if total_notional > 0.0
            else 0.0
        ),
        "basis_close_bps": _as_float(last["basis_bps"]),
        "funding_close": _as_float(last["funding_rate"]),
        "open_interest_close_base": oi_close,
        "return_1m_bps": None,
        "return_5m_bps": None,
        "realized_vol_5m_bps": None,
        "oi_change_5m_bps": None,
        "execution_curve_json": str(last["execution_curve_json"] or "{}"),
        "taker_fee_bps": taker_fee_bps,
    }


def _build_forward_labels(
    bars: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = FORWARD_HORIZONS_MINUTES,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(bars):
        for horizon in horizons:
            exit_index = index + horizon
            for direction in ("LONG", "SHORT"):
                reasons: list[str] = []
                if exit_index >= len(bars):
                    reasons.append("future_bar_missing")
                    exit_bar = None
                else:
                    exit_bar = bars[exit_index]
                if not int(entry["eligible"]):
                    reasons.append("entry_bar_ineligible")
                if exit_bar is not None:
                    window = bars[index : exit_index + 1]
                    if not all(int(row["eligible"]) for row in window):
                        reasons.append("incomplete_forward_window")
                    if len({int(row["continuity_id"]) for row in window}) != 1:
                        reasons.append("continuity_change")
                    expected_exit = int(entry["bar_timestamp_ms"]) + horizon * 60_000
                    if int(exit_bar["bar_timestamp_ms"]) != expected_exit:
                        reasons.append("noncontiguous_time")
                entry_mid = _as_float(entry["mid_close"])
                exit_mid = (
                    _as_float(exit_bar["mid_close"])
                    if exit_bar is not None
                    else None
                )
                entry_exec = _execution_price(
                    entry.get("execution_curve_json"),
                    side="buy" if direction == "LONG" else "sell",
                )
                exit_exec = (
                    _execution_price(
                        exit_bar.get("execution_curve_json"),
                        side="sell" if direction == "LONG" else "buy",
                    )
                    if exit_bar is not None
                    else None
                )
                if entry_mid is None or exit_mid is None:
                    reasons.append("mid_price_missing")
                if entry_exec is None or exit_exec is None:
                    reasons.append("execution_depth_missing")
                gross_mid_bps = None
                net_execution_bps = None
                total_cost_bps = None
                if not reasons:
                    sign = 1.0 if direction == "LONG" else -1.0
                    gross_mid_bps = (
                        sign * (float(exit_mid) / float(entry_mid) - 1.0) * 10_000.0
                    )
                    net_execution_bps = _net_execution_bps(
                        direction=direction,
                        entry_price=float(entry_exec),
                        exit_price=float(exit_exec),
                        notional_usd=EXECUTION_NOTIONAL_USD,
                        taker_fee_bps=float(entry["taker_fee_bps"]),
                    )
                    total_cost_bps = gross_mid_bps - net_execution_bps
                result.append(
                    {
                        "entry_bar_timestamp_ms": int(
                            entry["bar_timestamp_ms"]
                        ),
                        "symbol": str(entry["symbol"]),
                        "horizon_minutes": horizon,
                        "direction": direction,
                        "eligible": int(not reasons),
                        "reasons_json": _canonical_json(reasons),
                        "notional_usd": EXECUTION_NOTIONAL_USD,
                        "entry_mid": entry_mid,
                        "exit_mid": exit_mid,
                        "entry_execution_price": entry_exec,
                        "exit_execution_price": exit_exec,
                        "gross_mid_bps": gross_mid_bps,
                        "net_execution_bps": net_execution_bps,
                        "total_cost_bps": total_cost_bps,
                        "taker_fee_bps": float(entry["taker_fee_bps"]),
                    }
                )
    return result


def _net_execution_bps(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    notional_usd: float,
    taker_fee_bps: float,
) -> float:
    fee_rate = taker_fee_bps / 10_000.0
    quantity = notional_usd / entry_price
    exit_notional = quantity * exit_price
    if direction == "LONG":
        gross_pnl = exit_notional - notional_usd
    else:
        gross_pnl = notional_usd - exit_notional
    fees = notional_usd * fee_rate + exit_notional * fee_rate
    return (gross_pnl - fees) / notional_usd * 10_000.0


def _load_taker_fee_bps(
    *,
    session_dir: Path,
    symbols: Sequence[str],
) -> dict[str, float]:
    validation = validate_data_session(session_dir)
    latest: dict[str, tuple[int, float]] = {}
    for segment in validation["segment_rows"]:
        database = Path(str(segment["database"]))
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for symbol, received_ns, payload_json in connection.execute(
                """
                SELECT symbol, received_timestamp_utc_ns, payload_json
                FROM events
                WHERE channel = 'rest_instrument_rules'
                ORDER BY received_timestamp_utc_ns
                """
            ):
                payload = json.loads(str(payload_json))
                rate = _as_float(payload.get("taker_fee_rate"))
                if rate is None or rate <= 0.0:
                    continue
                latest[str(symbol)] = (int(received_ns), rate * 10_000.0)
    missing = sorted(set(symbols) - set(latest))
    if missing:
        raise SegmentValidationError(
            "taker fee rules missing for: " + ", ".join(missing)
        )
    return {
        symbol: latest[symbol][1]
        for symbol in symbols
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE bars (
            bar_timestamp_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            continuity_id INTEGER NOT NULL,
            observed_seconds INTEGER NOT NULL,
            complete_seconds INTEGER NOT NULL,
            eligible INTEGER NOT NULL,
            reasons_json TEXT NOT NULL,
            mid_open REAL,
            mid_close REAL,
            bid_close REAL,
            ask_close REAL,
            spread_mean_bps REAL,
            spread_max_bps REAL,
            microprice_edge_mean_bps REAL,
            book_imbalance_mean REAL,
            bid_depth_close_usd REAL,
            ask_depth_close_usd REAL,
            trade_count INTEGER NOT NULL,
            buy_trade_notional_usd REAL NOT NULL,
            sell_trade_notional_usd REAL NOT NULL,
            signed_trade_notional_usd REAL NOT NULL,
            flow_imbalance REAL NOT NULL,
            basis_close_bps REAL,
            funding_close REAL,
            open_interest_close_base REAL,
            return_1m_bps REAL,
            return_5m_bps REAL,
            realized_vol_5m_bps REAL,
            oi_change_5m_bps REAL,
            execution_curve_json TEXT NOT NULL,
            taker_fee_bps REAL NOT NULL,
            PRIMARY KEY(bar_timestamp_ms, symbol)
        );
        CREATE INDEX bars_symbol_time_idx
            ON bars(symbol, bar_timestamp_ms);
        CREATE TABLE labels (
            entry_bar_timestamp_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            direction TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            reasons_json TEXT NOT NULL,
            notional_usd REAL NOT NULL,
            entry_mid REAL,
            exit_mid REAL,
            entry_execution_price REAL,
            exit_execution_price REAL,
            gross_mid_bps REAL,
            net_execution_bps REAL,
            total_cost_bps REAL,
            taker_fee_bps REAL NOT NULL,
            PRIMARY KEY(
                entry_bar_timestamp_ms, symbol, horizon_minutes, direction
            )
        );
        CREATE INDEX labels_horizon_time_idx
            ON labels(horizon_minutes, entry_bar_timestamp_ms);
        """
    )


def _dataset_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        "bars": int(connection.execute("SELECT COUNT(*) FROM bars").fetchone()[0]),
        "eligible_bars": int(
            connection.execute(
                "SELECT COUNT(*) FROM bars WHERE eligible = 1"
            ).fetchone()[0]
        ),
        "labels": int(
            connection.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        ),
        "eligible_labels": int(
            connection.execute(
                "SELECT COUNT(*) FROM labels WHERE eligible = 1"
            ).fetchone()[0]
        ),
    }


def _execution_price(raw: Any, *, side: str) -> float | None:
    try:
        curve = json.loads(str(raw or "{}"))
        return _as_float(curve["25"][f"{side}_vwap"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _return_bps(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0.0 or end <= 0.0:
        return None
    return (end / start - 1.0) * 10_000.0


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean_not_none(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return statistics.fmean(selected) if selected else None


def _first_not_none(values: Sequence[float | None]) -> float | None:
    return next((value for value in values if value is not None), None)


def _last_not_none(values: Sequence[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


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
