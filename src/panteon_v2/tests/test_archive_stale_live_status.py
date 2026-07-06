from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone


def test_archive_status_copy_preserves_original_and_writes_manifest(tmp_path):
    tool = importlib.import_module("tools.archive_stale_live_status")
    status_path = tmp_path / "Results" / "MEXC" / "2026-06-25_14-23-38_v2" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"run_state":"running"}', encoding="utf-8")
    out_dir = tmp_path / "Reports" / "PreLive" / "stale_status_archive"

    manifest = tool.archive_status_files(
        [status_path],
        out_dir=out_dir,
        now=datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc),
        move=False,
    )

    archived_path = tmp_path / manifest["files"][0]["archived_path"]
    assert status_path.exists()
    assert archived_path.exists()
    assert archived_path.read_text(encoding="utf-8") == '{"run_state":"running"}'
    assert manifest["files"][0]["original_path"] == str(status_path)
    assert manifest["files"][0]["sha256"] == hashlib.sha256(
        b'{"run_state":"running"}'
    ).hexdigest()
    assert (out_dir / "20260702_120000_utc" / "archive_manifest.json").exists()


def test_archive_status_move_removes_only_requested_file(tmp_path):
    tool = importlib.import_module("tools.archive_stale_live_status")
    status_path = tmp_path / "Results" / "BITGET" / "2026-06-25_14-23-48_v2" / "status.json"
    neighbor = status_path.parent / "trading.log"
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"exchange":"BITGET"}', encoding="utf-8")
    neighbor.write_text("keep", encoding="utf-8")

    manifest = tool.archive_status_files(
        [status_path],
        out_dir=tmp_path / "archive",
        now=datetime(2026, 7, 2, 12, 1, tzinfo=timezone.utc),
        move=True,
    )

    assert not status_path.exists()
    assert neighbor.exists()
    archived_path = tmp_path / manifest["files"][0]["archived_path"]
    assert archived_path.exists()
    assert archived_path.read_text(encoding="utf-8") == '{"exchange":"BITGET"}'


def test_archive_status_can_write_fresh_cleanup_tombstone(tmp_path):
    tool = importlib.import_module("tools.archive_stale_live_status")
    status_path = tmp_path / "Results" / "MEXC" / "2026-06-25_14-23-38_v2" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        '{"exchange":"MEXC","run_state":"running","feed_status":"active","pid":12345}',
        encoding="utf-8",
    )

    manifest = tool.archive_status_files(
        [status_path],
        out_dir=tmp_path / "archive",
        now=datetime(2026, 7, 2, 12, 2, tzinfo=timezone.utc),
        move=False,
        write_tombstone=True,
    )

    tombstone_path = tmp_path / manifest["files"][0]["tombstone_path"]
    tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
    assert tombstone_path == (
        tmp_path
        / "Results"
        / "MEXC"
        / "20260702_120200_utc_stale_status_cleanup"
        / "status.json"
    )
    assert tombstone["exchange"] == "MEXC"
    assert tombstone["run_state"] == "stopped"
    assert tombstone["feed_status"] == "inactive"
    assert tombstone["pid"] == 0
    assert tombstone["archived_status_path"] == manifest["files"][0]["archived_path"]
    assert tombstone["original_status_path"] == str(status_path)
