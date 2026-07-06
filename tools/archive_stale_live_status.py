from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUT_DIR = Path("Reports") / "PreLive" / "stale_status_archive"


def _timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_name(path: Path, index: int) -> Path:
    parts = path.parts
    exchange = "unknown"
    session = path.parent.name
    if "Results" in parts:
        result_index = parts.index("Results")
        if len(parts) > result_index + 1:
            exchange = parts[result_index + 1]
    return Path(f"{index:02d}_{exchange}_{session}_status.json")


def _results_exchange_root(path: Path) -> tuple[Path, str]:
    parts = path.parts
    if "Results" not in parts:
        raise ValueError(f"status path is not under Results/<exchange>: {path}")
    result_index = parts.index("Results")
    if len(parts) <= result_index + 1:
        raise ValueError(f"status path is missing exchange segment: {path}")
    exchange = parts[result_index + 1]
    exchange_root = Path(*parts[: result_index + 2])
    return exchange_root, str(exchange).upper()


def _write_tombstone_status(
    *,
    source: Path,
    archived_path: Path,
    archive_sha256: str,
    now: datetime | None,
) -> Path:
    archive_time = now or datetime.now(timezone.utc)
    if archive_time.tzinfo is None:
        archive_time = archive_time.replace(tzinfo=timezone.utc)
    archive_time = archive_time.astimezone(timezone.utc)
    exchange_root, exchange = _results_exchange_root(source)
    tombstone_dir = exchange_root / f"{_timestamp(archive_time)}_stale_status_cleanup"
    tombstone_dir.mkdir(parents=True, exist_ok=True)
    tombstone_path = tombstone_dir / "status.json"
    payload = {
        "timestamp_utc": archive_time.isoformat(),
        "exchange": exchange,
        "mode": "stale_status_cleanup",
        "run_state": "stopped",
        "feed_status": "inactive",
        "pid": 0,
        "cleanup_reason": "stale_orphan_status_archived",
        "original_status_path": str(source),
        "archived_status_path": str(archived_path),
        "archived_status_sha256": archive_sha256,
    }
    tombstone_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return tombstone_path


def archive_status_files(
    status_paths: Sequence[Path | str],
    *,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    now: datetime | None = None,
    move: bool = False,
    write_tombstone: bool = False,
) -> dict[str, Any]:
    archive_root = Path(out_dir) / _timestamp(now)
    archive_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for index, raw_path in enumerate(status_paths, start=1):
        source = Path(raw_path)
        if not source.exists():
            raise FileNotFoundError(str(source))
        if source.name != "status.json":
            raise ValueError(f"expected status.json, got: {source}")
        target = archive_root / _archive_name(source, index)
        if move:
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
        archive_sha256 = _sha256(target)
        tombstone_path = ""
        if write_tombstone:
            tombstone_path = str(
                _write_tombstone_status(
                    source=source,
                    archived_path=target,
                    archive_sha256=archive_sha256,
                    now=now,
                )
            )
        files.append({
            "original_path": str(source),
            "archived_path": str(target),
            "sha256": archive_sha256,
            "mode": "move" if move else "copy",
            "tombstone_path": tombstone_path,
        })
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_dir": str(archive_root),
        "move": bool(move),
        "write_tombstone": bool(write_tombstone),
        "files": files,
    }
    manifest_path = archive_root / "archive_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive stale/orphan Panteon live status.json files.",
    )
    parser.add_argument("--status-path", action="append", default=[], required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--write-tombstone", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = archive_status_files(
        args.status_path,
        out_dir=Path(args.out_dir),
        move=bool(args.move),
        write_tombstone=bool(args.write_tombstone),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
