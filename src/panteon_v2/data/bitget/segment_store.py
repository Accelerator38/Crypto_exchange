"""Crash-safe SQLite segments and immutable manifests for Bitget events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profile import BitgetDataProfile


BITGET_SESSION_SCHEMA_VERSION = "panteon.bitget_data_session.v1"
BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION = "panteon.bitget_data_segment.v1"
BITGET_COLLECTOR_STATUS_SCHEMA_VERSION = "panteon.bitget_data_status.v1"
ZERO_SHA256 = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")


class SegmentValidationError(ValueError):
    """A data session or segment cannot be trusted."""


@dataclass(frozen=True)
class SegmentSummary:
    database_path: Path
    manifest_path: Path
    manifest_sha256: str
    database_sha256: str
    event_count: int
    quality_event_count: int


@dataclass(frozen=True)
class RecoverySummary:
    session_id: str
    recovered_segments: int
    manifest_sha256: str


class BitgetDataCollectorLock:
    """Allow exactly one collector/recovery owner for a data directory."""

    STALE_HEARTBEAT_SECONDS = 120.0

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir).resolve() / ".collector.lock"
        self.owner_token = uuid.uuid4().hex
        self.token = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "owner_token": self.owner_token,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def __enter__(self) -> "BitgetDataCollectorLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as exc:
            if not self._remove_stale_lock():
                raise RuntimeError(
                    f"another Bitget data collector owns {self.path}"
                ) from exc
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        try:
            os.write(descriptor, self.token.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        try:
            if self.path.read_text(encoding="ascii").strip() == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def _remove_stale_lock(self) -> bool:
        try:
            raw = self.path.read_text(encoding="ascii").strip()
            try:
                payload = json.loads(raw)
                pid = int(payload["pid"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pid = int(raw.split(":", 1)[0])
            lock_age = max(0.0, time.time() - self.path.stat().st_mtime)
        except (OSError, ValueError):
            return False
        if lock_age <= self.STALE_HEARTBEAT_SECONDS:
            return False
        if _pid_exists(pid) and self._fresh_collector_heartbeat(pid):
            return False
        try:
            self.path.unlink()
        except OSError:
            return False
        return True

    def _fresh_collector_heartbeat(self, pid: int) -> bool:
        sessions_dir = self.path.parent / "sessions"
        if not sessions_dir.is_dir():
            return False
        now = datetime.now(timezone.utc)
        for status_path in sessions_dir.glob("*/status.json"):
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                if int(payload.get("pid") or 0) != pid:
                    continue
                updated = datetime.fromisoformat(str(payload["updated_at"]))
                if updated.tzinfo is None:
                    continue
                age = (now - updated.astimezone(timezone.utc)).total_seconds()
                if (
                    0.0 <= age <= self.STALE_HEARTBEAT_SECONDS
                    and payload.get("run_state")
                    in {"starting", "collecting", "reconnecting"}
                ):
                    return True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return False


class BitgetSegmentStore:
    """Own one collector session and rotate received events by UTC time."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        profile: BitgetDataProfile,
        source_revision: str,
        collector_fingerprint_sha256: str,
        session_id: str | None = None,
        now_ns_fn: Any = time.time_ns,
        monotonic_ns_fn: Any = time.monotonic_ns,
    ) -> None:
        revision = str(source_revision).strip().lower()
        if not _REVISION_RE.fullmatch(revision):
            raise ValueError("source_revision is invalid")
        fingerprint = str(collector_fingerprint_sha256).strip().lower()
        if not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("collector_fingerprint_sha256 is invalid")
        self.profile = profile
        self.source_revision = revision
        self.collector_fingerprint_sha256 = fingerprint
        self.session_id = session_id or (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        )
        self.session_dir = Path(data_dir).resolve() / "sessions" / self.session_id
        self.segments_dir = self.session_dir / "segments"
        self.status_path = self.session_dir / "status.json"
        self.session_manifest_path = self.session_dir / "session_manifest.json"
        self.now_ns_fn = now_ns_fn
        self.monotonic_ns_fn = monotonic_ns_fn
        self.previous_manifest_sha256 = ZERO_SHA256
        self._connection: sqlite3.Connection | None = None
        self._partial_path: Path | None = None
        self._segment_start_ms: int | None = None
        self._segment_end_ms: int | None = None
        self._pending_events = 0
        self._last_commit_monotonic_ns = self.monotonic_ns_fn()
        self._event_counts: Counter[tuple[str, str]] = Counter()
        self._quality_counts: Counter[str] = Counter()
        self._duplicates = 0
        self._last_exchange_ts: dict[tuple[str, str], int] = {}
        self._session_event_count = 0
        self._session_quality_count = 0
        self._session_duplicate_count = 0
        self.segments_dir.mkdir(parents=True, exist_ok=False)
        self._write_session_manifest()
        self.write_status(run_state="starting")

    def append_event(
        self,
        *,
        channel: str,
        symbol: str,
        exchange_timestamp_ms: int,
        received_timestamp_utc_ns: int,
        received_timestamp_monotonic_ns: int,
        event_key: str,
        action: str,
        sequence: int | None,
        payload: Mapping[str, Any],
    ) -> None:
        self._ensure_segment(received_timestamp_utc_ns)
        assert self._connection is not None
        payload_json = _canonical_json(payload)
        before = self._connection.total_changes
        self._connection.execute(
            """
            INSERT OR IGNORE INTO events (
                channel, symbol, exchange_timestamp_ms,
                received_timestamp_utc_ns, received_timestamp_monotonic_ns,
                event_key, action, sequence, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(channel),
                str(symbol),
                int(exchange_timestamp_ms),
                int(received_timestamp_utc_ns),
                int(received_timestamp_monotonic_ns),
                str(event_key),
                str(action),
                int(sequence) if sequence is not None else None,
                payload_json,
            ),
        )
        if self._connection.total_changes == before:
            self._duplicates += 1
            self._session_duplicate_count += 1
        else:
            self._event_counts[(str(channel), str(symbol))] += 1
            self._session_event_count += 1
            self._pending_events += 1
            key = (str(channel), str(symbol))
            previous = self._last_exchange_ts.get(key)
            if previous is not None and int(exchange_timestamp_ms) < previous:
                self.append_quality_event(
                    kind="exchange_timestamp_regression",
                    channel=channel,
                    symbol=symbol,
                    exchange_timestamp_ms=exchange_timestamp_ms,
                    details={"previous_exchange_timestamp_ms": previous},
                )
            self._last_exchange_ts[key] = max(
                int(exchange_timestamp_ms),
                previous or int(exchange_timestamp_ms),
            )
        self._commit_if_due()

    def append_quality_event(
        self,
        *,
        kind: str,
        channel: str = "",
        symbol: str = "",
        exchange_timestamp_ms: int | None = None,
        details: Mapping[str, Any] | None = None,
        received_timestamp_utc_ns: int | None = None,
    ) -> None:
        received_ns = (
            int(received_timestamp_utc_ns)
            if received_timestamp_utc_ns is not None
            else int(self.now_ns_fn())
        )
        self._ensure_segment(received_ns)
        assert self._connection is not None
        self._connection.execute(
            """
            INSERT INTO quality_events (
                kind, channel, symbol, exchange_timestamp_ms,
                received_timestamp_utc_ns, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(kind),
                str(channel),
                str(symbol),
                (
                    int(exchange_timestamp_ms)
                    if exchange_timestamp_ms is not None
                    else None
                ),
                received_ns,
                _canonical_json(details or {}),
            ),
        )
        self._quality_counts[str(kind)] += 1
        self._session_quality_count += 1
        self._pending_events += 1
        self._commit_if_due()

    def write_status(self, *, run_state: str, **values: Any) -> None:
        payload = {
            "schema_version": BITGET_COLLECTOR_STATUS_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "session_id": self.session_id,
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.profile_sha256,
            "source_revision": self.source_revision,
            "run_state": str(run_state),
            "events": self._session_event_count,
            "quality_events": self._session_quality_count,
            "duplicate_events": self._session_duplicate_count,
            "current_segment_start_ms": self._segment_start_ms,
            "orders_enabled": False,
            "promotion_authority": False,
            **values,
        }
        _atomic_write_json(self.status_path, payload)

    def close(self, *, reason: str = "collector_shutdown") -> SegmentSummary | None:
        summary = self._seal_current(reason=reason)
        self.write_status(
            run_state="stopped",
            stop_reason=reason,
            last_manifest_sha256=self.previous_manifest_sha256,
        )
        return summary

    def _ensure_segment(self, received_timestamp_utc_ns: int) -> None:
        received_ms = int(received_timestamp_utc_ns) // 1_000_000
        interval_ms = int(self.profile.segment_seconds) * 1000
        start_ms = received_ms - received_ms % interval_ms
        if self._connection is not None and start_ms == self._segment_start_ms:
            return
        if self._connection is not None:
            self._seal_current(reason="utc_segment_boundary")
        self._open_segment(start_ms=start_ms, end_ms=start_ms + interval_ms)

    def _open_segment(self, *, start_ms: int, end_ms: int) -> None:
        stamp = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self._partial_path = self.segments_dir / f"{stamp}.sqlite.partial"
        if self._partial_path.exists():
            raise RuntimeError(f"partial segment already exists: {self._partial_path}")
        connection = sqlite3.connect(self._partial_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange_timestamp_ms INTEGER NOT NULL,
                received_timestamp_utc_ns INTEGER NOT NULL,
                received_timestamp_monotonic_ns INTEGER NOT NULL,
                event_key TEXT NOT NULL,
                action TEXT NOT NULL,
                sequence INTEGER,
                payload_json TEXT NOT NULL,
                UNIQUE(channel, symbol, event_key)
            );
            CREATE INDEX events_time_idx
                ON events(exchange_timestamp_ms, channel, symbol);
            CREATE TABLE quality_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                channel TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange_timestamp_ms INTEGER,
                received_timestamp_utc_ns INTEGER NOT NULL,
                details_json TEXT NOT NULL
            );
            """
        )
        metadata = {
            "schema_version": BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION,
            "session_id": self.session_id,
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.profile_sha256,
            "source_revision": self.source_revision,
            "collector_fingerprint_sha256": self.collector_fingerprint_sha256,
            "segment_start_ms": str(start_ms),
            "segment_end_ms": str(end_ms),
            "orders_enabled": "false",
            "promotion_authority": "false",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        self._connection = connection
        self._segment_start_ms = start_ms
        self._segment_end_ms = end_ms
        self._pending_events = 0
        self._last_commit_monotonic_ns = self.monotonic_ns_fn()
        self._event_counts.clear()
        self._quality_counts.clear()
        self._duplicates = 0
        self._last_exchange_ts.clear()

    def _commit_if_due(self) -> None:
        assert self._connection is not None
        elapsed_ms = (
            self.monotonic_ns_fn() - self._last_commit_monotonic_ns
        ) / 1_000_000.0
        if (
            self._pending_events >= self.profile.commit_batch_events
            or elapsed_ms >= self.profile.commit_interval_ms
        ):
            self._connection.commit()
            self._pending_events = 0
            self._last_commit_monotonic_ns = self.monotonic_ns_fn()

    def _seal_current(self, *, reason: str) -> SegmentSummary | None:
        if self._connection is None:
            return None
        connection = self._connection
        partial_path = self._partial_path
        start_ms = self._segment_start_ms
        end_ms = self._segment_end_ms
        assert partial_path is not None and start_ms is not None and end_ms is not None
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        event_count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        quality_count = int(
            connection.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0]
        )
        bounds = connection.execute(
            """
            SELECT MIN(exchange_timestamp_ms), MAX(exchange_timestamp_ms),
                   MIN(received_timestamp_utc_ns), MAX(received_timestamp_utc_ns)
            FROM events
            """
        ).fetchone()
        counts = {
            f"{row[0]}:{row[1]}": int(row[2])
            for row in connection.execute(
                """
                SELECT channel, symbol, COUNT(*)
                FROM events
                GROUP BY channel, symbol
                ORDER BY channel, symbol
                """
            )
        }
        quality_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT kind, COUNT(*)
                FROM quality_events
                GROUP BY kind
                ORDER BY kind
                """
            )
        }
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        database_path = partial_path.with_suffix("")
        os.replace(partial_path, database_path)
        database_sha = _sha256_file(database_path)
        manifest = {
            "schema_version": BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.profile_sha256,
            "source_revision": self.source_revision,
            "collector_fingerprint_sha256": self.collector_fingerprint_sha256,
            "database_file": database_path.name,
            "database_sha256": database_sha,
            "previous_manifest_sha256": self.previous_manifest_sha256,
            "segment_start_ms": start_ms,
            "segment_end_ms": end_ms,
            "seal_reason": str(reason),
            "sqlite_integrity": integrity,
            "event_count": event_count,
            "quality_event_count": quality_count,
            "duplicate_event_count": self._duplicates,
            "event_counts": counts,
            "quality_counts": quality_counts,
            "recovery": {
                "recovered": False,
                "reason": "",
            },
            "min_exchange_timestamp_ms": bounds[0],
            "max_exchange_timestamp_ms": bounds[1],
            "min_received_timestamp_utc_ns": bounds[2],
            "max_received_timestamp_utc_ns": bounds[3],
            "orders_enabled": False,
            "promotion_authority": False,
        }
        manifest["manifest_sha256"] = _sha256_json(manifest)
        manifest_path = database_path.with_suffix(".manifest.json")
        _atomic_write_json(manifest_path, manifest)
        self.previous_manifest_sha256 = str(manifest["manifest_sha256"])
        self._connection = None
        self._partial_path = None
        self._segment_start_ms = None
        self._segment_end_ms = None
        return SegmentSummary(
            database_path=database_path,
            manifest_path=manifest_path,
            manifest_sha256=str(manifest["manifest_sha256"]),
            database_sha256=database_sha,
            event_count=event_count,
            quality_event_count=quality_count,
        )

    def _write_session_manifest(self) -> None:
        payload = {
            "schema_version": BITGET_SESSION_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.profile_sha256,
            "profile_file": str(self.profile.path),
            "source_revision": self.source_revision,
            "collector_fingerprint_sha256": self.collector_fingerprint_sha256,
            "collector_mode": "public_websocket_read_only",
            "segment_seconds": self.profile.segment_seconds,
            "symbols": list(self.profile.symbols),
            "channels": list(self.profile.channels),
            "orders_enabled": False,
            "promotion_authority": False,
        }
        payload["session_manifest_sha256"] = _sha256_json(payload)
        _atomic_write_json(self.session_manifest_path, payload)


def validate_data_session(session_dir: str | Path) -> dict[str, Any]:
    source = Path(session_dir).resolve()
    session_path = source / "session_manifest.json"
    if not session_path.is_file():
        raise SegmentValidationError("session_manifest.json is missing")
    session = _load_json(session_path)
    expected_session_sha = str(session.pop("session_manifest_sha256", ""))
    if not _SHA256_RE.fullmatch(expected_session_sha):
        raise SegmentValidationError("session manifest SHA is invalid")
    if _sha256_json(session) != expected_session_sha:
        raise SegmentValidationError("session manifest hash mismatch")
    if session.get("orders_enabled") is not False:
        raise SegmentValidationError("session orders_enabled is not false")
    if session.get("promotion_authority") is not False:
        raise SegmentValidationError("session promotion_authority is not false")
    partials = sorted((source / "segments").glob("*.partial*"))
    manifests = sorted((source / "segments").glob("*.manifest.json"))
    previous_sha = ZERO_SHA256
    total_events = 0
    total_quality = 0
    channel_symbol_counts: Counter[str] = Counter()
    first_start: int | None = None
    last_end: int | None = None
    segment_rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        row = _load_json(manifest_path)
        claimed_sha = str(row.pop("manifest_sha256", ""))
        if not _SHA256_RE.fullmatch(claimed_sha):
            raise SegmentValidationError(
                f"invalid manifest SHA: {manifest_path.name}"
            )
        if _sha256_json(row) != claimed_sha:
            raise SegmentValidationError(
                f"manifest hash mismatch: {manifest_path.name}"
            )
        if row.get("previous_manifest_sha256") != previous_sha:
            raise SegmentValidationError(
                f"manifest chain mismatch: {manifest_path.name}"
            )
        if row.get("session_id") != session.get("session_id"):
            raise SegmentValidationError(
                f"segment session mismatch: {manifest_path.name}"
            )
        if row.get("profile_sha256") != session.get("profile_sha256"):
            raise SegmentValidationError(
                f"segment profile mismatch: {manifest_path.name}"
            )
        if row.get("collector_fingerprint_sha256") != session.get(
            "collector_fingerprint_sha256"
        ):
            raise SegmentValidationError(
                f"segment collector fingerprint mismatch: {manifest_path.name}"
            )
        if row.get("orders_enabled") is not False:
            raise SegmentValidationError("segment orders_enabled is not false")
        if row.get("promotion_authority") is not False:
            raise SegmentValidationError(
                "segment promotion_authority is not false"
            )
        database_path = manifest_path.parent / str(row["database_file"])
        if not database_path.is_file():
            raise SegmentValidationError(
                f"segment database missing: {database_path.name}"
            )
        if _sha256_file(database_path) != row.get("database_sha256"):
            raise SegmentValidationError(
                f"segment database hash mismatch: {database_path.name}"
            )
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
            integrity = str(
                connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            event_count = int(
                connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            )
            quality_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM quality_events"
                ).fetchone()[0]
            )
        if integrity != "ok" or row.get("sqlite_integrity") != "ok":
            raise SegmentValidationError(
                f"SQLite integrity failed: {database_path.name}"
            )
        if event_count != int(row.get("event_count", -1)):
            raise SegmentValidationError(
                f"event count mismatch: {database_path.name}"
            )
        if quality_count != int(row.get("quality_event_count", -1)):
            raise SegmentValidationError(
                f"quality count mismatch: {database_path.name}"
            )
        start_ms = int(row["segment_start_ms"])
        end_ms = int(row["segment_end_ms"])
        if end_ms <= start_ms:
            raise SegmentValidationError("invalid segment interval")
        if last_end is not None and start_ms < last_end:
            raise SegmentValidationError("overlapping segment intervals")
        first_start = start_ms if first_start is None else first_start
        last_end = end_ms
        previous_sha = claimed_sha
        total_events += event_count
        total_quality += quality_count
        channel_symbol_counts.update(
            {
                str(key): int(value)
                for key, value in dict(row.get("event_counts") or {}).items()
            }
        )
        segment_rows.append(
            {
                "database": str(database_path),
                "manifest": str(manifest_path),
                "manifest_sha256": claimed_sha,
                "event_count": event_count,
                "quality_event_count": quality_count,
                "segment_start_ms": start_ms,
                "segment_end_ms": end_ms,
            }
        )
    required_pairs = {
        f"{channel}:{symbol}"
        for channel in session["channels"]
        for symbol in session["symbols"]
    }
    observed_pairs = {
        pair for pair, count in channel_symbol_counts.items() if count > 0
    }
    return {
        "schema_version": "panteon.bitget_data_session_validation.v1",
        "session_dir": str(source),
        "session_id": session["session_id"],
        "profile_id": session["profile_id"],
        "profile_sha256": session["profile_sha256"],
        "source_revision": session["source_revision"],
        "collector_fingerprint_sha256": session[
            "collector_fingerprint_sha256"
        ],
        "segments": len(segment_rows),
        "events": total_events,
        "quality_events": total_quality,
        "required_channel_symbol_pairs": len(required_pairs),
        "observed_channel_symbol_pairs": len(required_pairs & observed_pairs),
        "missing_channel_symbol_pairs": sorted(required_pairs - observed_pairs),
        "channel_symbol_coverage_pct": (
            len(required_pairs & observed_pairs) / len(required_pairs) * 100.0
            if required_pairs
            else 0.0
        ),
        "first_segment_start_ms": first_start,
        "last_segment_end_ms": last_end,
        "head_manifest_sha256": previous_sha,
        "unsealed_partial_files": [str(path) for path in partials],
        "segments_valid": bool(segment_rows) and not partials,
        "orders_enabled": False,
        "promotion_authority": False,
        "segment_rows": segment_rows,
    }


def recover_unsealed_sessions(data_dir: str | Path) -> list[RecoverySummary]:
    """Seal committed orphan WAL segments without changing their old contract."""

    sessions_dir = Path(data_dir).resolve() / "sessions"
    if not sessions_dir.is_dir():
        return []
    recovered: list[RecoverySummary] = []
    for session_dir in sorted(path for path in sessions_dir.iterdir() if path.is_dir()):
        partials = sorted((session_dir / "segments").glob("*.sqlite.partial"))
        if not partials:
            continue
        if len(partials) > 1:
            raise SegmentValidationError(
                f"multiple orphan partial segments: {session_dir.name}"
            )
        session = _validated_session_manifest(session_dir)
        previous_sha = _validated_existing_chain(session_dir, session)
        partial_path = partials[0]
        manifest_sha = _recover_partial_segment(
            partial_path=partial_path,
            session=session,
            previous_manifest_sha256=previous_sha,
        )
        status_path = session_dir / "status.json"
        _atomic_write_json(
            status_path,
            {
                "schema_version": BITGET_COLLECTOR_STATUS_SCHEMA_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "session_id": session["session_id"],
                "profile_id": session["profile_id"],
                "profile_sha256": session["profile_sha256"],
                "source_revision": session["source_revision"],
                "run_state": "recovered_stopped",
                "stop_reason": "recovered_after_unclean_shutdown",
                "last_manifest_sha256": manifest_sha,
                "orders_enabled": False,
                "promotion_authority": False,
            },
        )
        recovered.append(
            RecoverySummary(
                session_id=str(session["session_id"]),
                recovered_segments=1,
                manifest_sha256=manifest_sha,
            )
        )
    return recovered


def _recover_partial_segment(
    *,
    partial_path: Path,
    session: Mapping[str, Any],
    previous_manifest_sha256: str,
) -> str:
    connection = sqlite3.connect(partial_path)
    try:
        connection.commit()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise SegmentValidationError(
                f"orphan SQLite integrity failed: {partial_path.name}"
            )
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        for key in (
            "session_id",
            "profile_id",
            "profile_sha256",
            "source_revision",
            "collector_fingerprint_sha256",
        ):
            if metadata.get(key) != str(session.get(key)):
                raise SegmentValidationError(
                    f"orphan metadata mismatch for {key}: {partial_path.name}"
                )
        event_count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        quality_count = int(
            connection.execute("SELECT COUNT(*) FROM quality_events").fetchone()[0]
        )
        bounds = connection.execute(
            """
            SELECT MIN(exchange_timestamp_ms), MAX(exchange_timestamp_ms),
                   MIN(received_timestamp_utc_ns), MAX(received_timestamp_utc_ns)
            FROM events
            """
        ).fetchone()
        event_counts = {
            f"{row[0]}:{row[1]}": int(row[2])
            for row in connection.execute(
                """
                SELECT channel, symbol, COUNT(*)
                FROM events
                GROUP BY channel, symbol
                ORDER BY channel, symbol
                """
            )
        }
        quality_counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT kind, COUNT(*)
                FROM quality_events
                GROUP BY kind
                ORDER BY kind
                """
            )
        }
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        orphan = Path(str(partial_path) + suffix)
        if orphan.exists():
            raise SegmentValidationError(
                f"orphan WAL sidecar remains after recovery: {orphan.name}"
            )
    database_path = partial_path.with_suffix("")
    os.replace(partial_path, database_path)
    database_sha = _sha256_file(database_path)
    start_ms = int(metadata["segment_start_ms"])
    end_ms = int(metadata["segment_end_ms"])
    manifest = {
        "schema_version": BITGET_SEGMENT_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session["session_id"],
        "profile_id": session["profile_id"],
        "profile_sha256": session["profile_sha256"],
        "source_revision": session["source_revision"],
        "collector_fingerprint_sha256": session[
            "collector_fingerprint_sha256"
        ],
        "database_file": database_path.name,
        "database_sha256": database_sha,
        "previous_manifest_sha256": previous_manifest_sha256,
        "segment_start_ms": start_ms,
        "segment_end_ms": end_ms,
        "seal_reason": "recovered_after_unclean_shutdown",
        "sqlite_integrity": integrity,
        "event_count": event_count,
        "quality_event_count": quality_count,
        "duplicate_event_count": 0,
        "event_counts": event_counts,
        "quality_counts": quality_counts,
        "recovery": {
            "recovered": True,
            "reason": "unclean_collector_shutdown",
        },
        "min_exchange_timestamp_ms": bounds[0],
        "max_exchange_timestamp_ms": bounds[1],
        "min_received_timestamp_utc_ns": bounds[2],
        "max_received_timestamp_utc_ns": bounds[3],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    manifest_path = database_path.with_suffix(".manifest.json")
    _atomic_write_json(manifest_path, manifest)
    return str(manifest["manifest_sha256"])


def _validated_session_manifest(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "session_manifest.json"
    session = _load_json(path)
    claimed = str(session.pop("session_manifest_sha256", ""))
    if not _SHA256_RE.fullmatch(claimed) or _sha256_json(session) != claimed:
        raise SegmentValidationError(
            f"session manifest hash mismatch: {session_dir.name}"
        )
    if session.get("orders_enabled") is not False:
        raise SegmentValidationError("recovery refused unsafe session")
    if session.get("promotion_authority") is not False:
        raise SegmentValidationError("recovery refused promotion authority")
    return session


def _validated_existing_chain(
    session_dir: Path,
    session: Mapping[str, Any],
) -> str:
    previous_sha = ZERO_SHA256
    for manifest_path in sorted((session_dir / "segments").glob("*.manifest.json")):
        row = _load_json(manifest_path)
        claimed = str(row.pop("manifest_sha256", ""))
        if not _SHA256_RE.fullmatch(claimed) or _sha256_json(row) != claimed:
            raise SegmentValidationError(
                f"existing manifest hash mismatch: {manifest_path.name}"
            )
        if row.get("previous_manifest_sha256") != previous_sha:
            raise SegmentValidationError(
                f"existing manifest chain mismatch: {manifest_path.name}"
            )
        if row.get("session_id") != session.get("session_id"):
            raise SegmentValidationError(
                f"existing segment session mismatch: {manifest_path.name}"
            )
        database_path = manifest_path.parent / str(row["database_file"])
        if (
            not database_path.is_file()
            or _sha256_file(database_path) != row.get("database_sha256")
        ):
            raise SegmentValidationError(
                f"existing segment database mismatch: {manifest_path.name}"
            )
        previous_sha = claimed
    return previous_sha


def current_git_revision(root: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = result.stdout.strip().lower()
    if result.returncode != 0 or not _REVISION_RE.fullmatch(revision):
        raise RuntimeError("git revision unavailable")
    return revision


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return ctypes.windll.kernel32.GetLastError() == 5  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def compute_collector_fingerprint(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    resolved = sorted(Path(path).resolve() for path in paths)
    if not resolved:
        raise ValueError("collector fingerprint needs at least one file")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SegmentValidationError(f"invalid JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SegmentValidationError(f"JSON object required: {path.name}")
    return value


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
