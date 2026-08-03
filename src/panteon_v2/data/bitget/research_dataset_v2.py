"""Research dataset bound to the event-aware Bitget frame v2 contract."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .materializer import validate_frame_dataset_v2
from .research_dataset import (
    BAR_SECONDS,
    EXECUTION_NOTIONAL_USD,
    FORWARD_HORIZONS_MINUTES,
    MIN_COMPLETE_SECONDS,
    ResearchDatasetSummary,
    _atomic_write_json,
    _build_forward_labels,
    _build_symbol_bars,
    _create_schema,
    _dataset_counts,
    _load_taker_fee_bps,
    _read_json,
    _sha256_file,
    _sha256_json,
)
from .segment_store import SegmentValidationError


RESEARCH_SCHEMA_VERSION_V2 = "panteon.bitget_microstructure_research.v2"
RESEARCH_DATASET_SCHEMA_VERSION_V2 = "panteon.bitget_research_dataset.v2"
OUTPUT_STEM = "microstructure_research_v2"
RESEARCH_SCHEMA_VERSION_V3 = "panteon.bitget_microstructure_research.v3"
RESEARCH_DATASET_SCHEMA_VERSION_V3 = "panteon.bitget_research_dataset.v3"
OUTPUT_STEM_V3 = "microstructure_research_v3"
FORWARD_HORIZONS_MINUTES_V3 = (120,)


def materialize_research_dataset_v2(
    *,
    frame_dataset_dir: str | Path,
    output_dir: str | Path,
) -> ResearchDatasetSummary:
    return _materialize_research_dataset(
        frame_dataset_dir=frame_dataset_dir,
        output_dir=output_dir,
        output_stem=OUTPUT_STEM,
        dataset_schema_version=RESEARCH_DATASET_SCHEMA_VERSION_V2,
        research_schema_version=RESEARCH_SCHEMA_VERSION_V2,
        forward_horizons=FORWARD_HORIZONS_MINUTES,
        source_frame_contract="event_snapshot_continuity_bounded_v2",
    )


def materialize_research_dataset_v3(
    *,
    frame_dataset_dir: str | Path,
    output_dir: str | Path,
) -> ResearchDatasetSummary:
    return _materialize_research_dataset(
        frame_dataset_dir=frame_dataset_dir,
        output_dir=output_dir,
        output_stem=OUTPUT_STEM_V3,
        dataset_schema_version=RESEARCH_DATASET_SCHEMA_VERSION_V3,
        research_schema_version=RESEARCH_SCHEMA_VERSION_V3,
        forward_horizons=FORWARD_HORIZONS_MINUTES_V3,
        source_frame_contract="event_snapshot_continuity_bounded_v2_h120",
    )


def _materialize_research_dataset(
    *,
    frame_dataset_dir: str | Path,
    output_dir: str | Path,
    output_stem: str,
    dataset_schema_version: str,
    research_schema_version: str,
    forward_horizons: tuple[int, ...],
    source_frame_contract: str,
) -> ResearchDatasetSummary:
    validation = validate_frame_dataset_v2(frame_dataset_dir)
    source = Path(frame_dataset_dir).resolve()
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    database_path = target / f"{output_stem}.sqlite"
    manifest_path = target / f"{output_stem}.manifest.json"
    if database_path.exists() or manifest_path.exists():
        raise FileExistsError(f"{output_stem} output already exists")

    frame_manifest = _read_json(source / "frame_1s_v2.manifest.json")
    symbols = tuple(str(item) for item in frame_manifest["symbols"])
    fee_bps = _load_taker_fee_bps(
        session_dir=Path(str(frame_manifest["source_session_dir"])),
        symbols=symbols,
    )

    connection = sqlite3.connect(database_path)
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
                frame_rows = frame_connection.execute(
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
                    frame_rows=frame_rows,
                    taker_fee_bps=fee_bps[symbol],
                )
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
                    _build_forward_labels(
                        symbol_bars,
                        horizons=forward_horizons,
                    ),
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
        "schema_version": dataset_schema_version,
        "research_schema_version": research_schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_frame_dataset_dir": str(source),
        "source_frame_manifest_sha256": validation["manifest_sha256"],
        "source_frame_database_sha256": validation["database_sha256"],
        "source_session_id": validation["source_session_id"],
        "source_head_manifest_sha256": validation[
            "source_head_manifest_sha256"
        ],
        "source_frame_contract": source_frame_contract,
        "database_file": database_path.name,
        "database_sha256": database_sha,
        "sqlite_integrity": integrity,
        "symbols": list(symbols),
        "bar_seconds": BAR_SECONDS,
        "min_complete_seconds": MIN_COMPLETE_SECONDS,
        "forward_horizons_minutes": list(forward_horizons),
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


def validate_research_dataset_v2(dataset_dir: str | Path) -> dict[str, Any]:
    return _validate_research_dataset(
        dataset_dir,
        output_stem=OUTPUT_STEM,
        dataset_schema_version=RESEARCH_DATASET_SCHEMA_VERSION_V2,
        research_schema_version=RESEARCH_SCHEMA_VERSION_V2,
        source_frame_contract="event_snapshot_continuity_bounded_v2",
        validation_schema_version="panteon.bitget_research_validation.v2",
    )


def validate_research_dataset_v3(dataset_dir: str | Path) -> dict[str, Any]:
    return _validate_research_dataset(
        dataset_dir,
        output_stem=OUTPUT_STEM_V3,
        dataset_schema_version=RESEARCH_DATASET_SCHEMA_VERSION_V3,
        research_schema_version=RESEARCH_SCHEMA_VERSION_V3,
        source_frame_contract="event_snapshot_continuity_bounded_v2_h120",
        validation_schema_version="panteon.bitget_research_validation.v3",
    )


def _validate_research_dataset(
    dataset_dir: str | Path,
    *,
    output_stem: str,
    dataset_schema_version: str,
    research_schema_version: str,
    source_frame_contract: str,
    validation_schema_version: str,
) -> dict[str, Any]:
    source = Path(dataset_dir).resolve()
    manifest_path = source / f"{output_stem}.manifest.json"
    if not manifest_path.is_file():
        raise SegmentValidationError("research v2 dataset manifest is missing")
    manifest = _read_json(manifest_path)
    claimed_sha = str(manifest.pop("manifest_sha256", ""))
    if _sha256_json(manifest) != claimed_sha:
        raise SegmentValidationError("research v2 dataset manifest hash mismatch")
    if manifest.get("schema_version") != dataset_schema_version:
        raise SegmentValidationError("research v2 dataset schema mismatch")
    if manifest.get("research_schema_version") != research_schema_version:
        raise SegmentValidationError("research v2 schema mismatch")
    if manifest.get("source_frame_contract") != source_frame_contract:
        raise SegmentValidationError("research v2 frame contract mismatch")
    if manifest.get("research_only") is not True:
        raise SegmentValidationError("research_only is not true")
    if manifest.get("orders_enabled") is not False:
        raise SegmentValidationError("research dataset orders_enabled is not false")
    if manifest.get("promotion_authority") is not False:
        raise SegmentValidationError("research dataset promotion_authority is not false")
    database_path = source / str(manifest["database_file"])
    if not database_path.is_file():
        raise SegmentValidationError("research v2 database is missing")
    database_sha = _sha256_file(database_path)
    if database_sha != manifest.get("database_sha256"):
        raise SegmentValidationError("research v2 database hash mismatch")
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        counts = _dataset_counts(connection)
    if integrity != "ok" or manifest.get("sqlite_integrity") != "ok":
        raise SegmentValidationError("research v2 SQLite integrity failed")
    for key, value in counts.items():
        if int(manifest.get(key, -1)) != int(value):
            raise SegmentValidationError(f"research v2 {key} mismatch")
    frame = validate_frame_dataset_v2(manifest["source_frame_dataset_dir"])
    if frame["manifest_sha256"] != manifest["source_frame_manifest_sha256"]:
        raise SegmentValidationError("research v2 source frame manifest changed")
    if frame["database_sha256"] != manifest["source_frame_database_sha256"]:
        raise SegmentValidationError("research v2 source frame database changed")
    return {
        "schema_version": validation_schema_version,
        "dataset_dir": str(source),
        "database_sha256": database_sha,
        "manifest_sha256": claimed_sha,
        "source_session_id": manifest["source_session_id"],
        "source_head_manifest_sha256": manifest[
            "source_head_manifest_sha256"
        ],
        **counts,
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
        "dataset_valid": True,
    }
