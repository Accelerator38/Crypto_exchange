"""Per-origin data commitment for a future registered three-day protocol.

An origin is committed after its last candle closes and before its evaluation.
The fixed deadline is relative to that origin, including the final origin.
This module never treats a V1 manifest as prospective V2 evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .prospective_contract_v1 import derive_origins, file_sha256
from .prospective_contract_v2 import validate_contract
from .prospective_manifest_v1 import _canonical_sha, _inside, _read_parquet, _validate_panel


SCHEMA = "exia.genetic_primus.prospective_data_manifest/2"
RULE = "after_origin_complete_before_evaluation"
MAX_DEADLINE_HOURS = 72


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc(value: datetime) -> datetime:
    _require(isinstance(value, datetime) and value.tzinfo is not None
             and value.utcoffset() == timedelta(0), "time must be aware UTC")
    return value.astimezone(timezone.utc)


def _parse_utc(value: str) -> datetime:
    _require(isinstance(value, str), "timestamp must be text")
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _protocol(contract: dict[str, Any]) -> int:
    validate_contract(contract)
    policy = contract["data_commitment"]
    _require(policy.get("manifest_version") == 2 and
             policy.get("origin_snapshot_commit_rule") == RULE,
             "V2 per-origin commitment policy required")
    hours = policy.get("origin_commit_deadline_hours")
    _require(type(hours) is int and 0 < hours <= MAX_DEADLINE_HOURS,
             "deadline must be an integer from 1 to 72 hours")
    return hours


def origin_commit_window(
    contract: dict[str, Any], origin_index: int, committed_at: datetime,
) -> tuple[datetime, datetime]:
    """Pure deadline check. The final origin may be committed after outer end."""
    hours = _protocol(contract)
    origins = derive_origins(contract)
    _require(type(origin_index) is int and 0 <= origin_index < len(origins),
             "origin index outside registered interval")
    closed_at = origins[origin_index][1]
    deadline = closed_at + timedelta(hours=hours)
    committed_at = _utc(committed_at)
    _require(closed_at <= committed_at < deadline,
             "origin snapshot must be committed after close and before its deadline")
    return closed_at, deadline


def _contract_identity(root: Path, contract_path: Path, contract: dict[str, Any]) -> tuple[str, str]:
    path = _inside(contract_path, root)
    _require(json.loads(path.read_text(encoding="utf-8")) == contract,
             "in-memory contract differs from saved bytes")
    return path.relative_to(root).as_posix(), file_sha256(path)


def _previous(
    path: Path | None, *, root: Path, contract: dict[str, Any],
    contract_sha: str, origin_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if origin_index == 0:
        _require(path is None, "first origin cannot have a previous manifest")
        return None, None
    _require(path is not None, "previous V2 manifest is required")
    path = _inside(path, root)
    previous = json.loads(path.read_text(encoding="utf-8"))
    _require(previous.get("schema_version") == SCHEMA
             and previous.get("origin_index") == origin_index - 1
             and previous.get("contract_sha256") == contract_sha
             and previous.get("immutable") is True,
             "previous manifest identity changed")
    origin_commit_window(contract, origin_index - 1,
                         _parse_utc(previous["created_at_utc"]))
    expected_end = int(derive_origins(contract)[origin_index - 1][1].timestamp() * 1000)
    _require(previous.get("metadata", {}).get("coverage_end_exclusive") == expected_end,
             "previous origin coverage changed")
    entries = previous.get("snapshot_files")
    _require(isinstance(entries, list) and _canonical_sha(entries) == previous.get("snapshot_sha256"),
             "previous snapshot commitment malformed")
    return previous, file_sha256(path)


def _entries(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    resolved = [_inside(Path(path), root) for path in paths]
    _require(resolved and len(resolved) == len(set(resolved)),
             "at least one distinct Parquet file is required")
    result = []
    for path in sorted(resolved, key=lambda item: item.as_posix()):
        _require(path.suffix.lower() == ".parquet", "only Parquet snapshots are accepted")
        result.append({"path": path.relative_to(root).as_posix(),
                       "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    return result


def _snapshot(
    entries: list[dict[str, Any]], *, root: Path, contract: dict[str, Any],
    origin_index: int, previous: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if previous is not None:
        old = {item["path"]: item for item in previous["snapshot_files"]}
        new = {item["path"]: item for item in entries}
        _require(all(new.get(path) == item for path, item in old.items()),
                 "snapshot is not append-only or revises older files")
    panel = pd.concat((_read_parquet(_inside(root / item["path"], root)) for item in entries),
                      ignore_index=True)
    timeframe_ms = int(contract["training"]["timeframe_minutes"]) * 60_000
    metadata = _validate_panel(panel, tuple(contract["universe"]), timeframe_ms)
    _require((panel["high"] >= panel[["open", "close", "low"]].max(axis=1)).all()
             and (panel["low"] <= panel[["open", "close"]].min(axis=1)).all(),
             "inconsistent OHLC range")
    origins = derive_origins(contract)
    expected_end = int(origins[origin_index][1].timestamp() * 1000)
    history_start = datetime.fromisoformat(
        contract["training"]["initial_history_start_utc"].replace("Z", "+00:00"))
    _require(metadata["coverage_start"] <= int(history_start.timestamp() * 1000),
             "snapshot does not cover registered training history")
    _require(metadata["coverage_end_exclusive"] == expected_end,
             "snapshot must end at this origin with no future bars")
    return metadata, panel.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def build_origin_manifest(
    paths: Iterable[Path], *, root: Path, contract_path: Path,
    contract: dict[str, Any], origin_index: int,
    previous_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Create an in-memory commitment using the actual current UTC clock."""
    root = root.resolve()
    contract_relpath, contract_sha = _contract_identity(root, contract_path, contract)
    previous, previous_sha = _previous(previous_manifest_path, root=root,
                                      contract=contract, contract_sha=contract_sha,
                                      origin_index=origin_index)
    entries = _entries(paths, root)
    metadata, _ = _snapshot(entries, root=root, contract=contract,
                            origin_index=origin_index, previous=previous)
    created = datetime.now(timezone.utc)
    closed_at, deadline = origin_commit_window(contract, origin_index, created)
    if previous is not None:
        _require(_parse_utc(previous["created_at_utc"]) < created,
                 "manifest chain is not chronological")
    return {"schema_version": SCHEMA, "origin_index": origin_index,
            "origin_end_exclusive_utc": closed_at.isoformat().replace("+00:00", "Z"),
            "commit_deadline_exclusive_utc": deadline.isoformat().replace("+00:00", "Z"),
            "created_at_utc": created.isoformat().replace("+00:00", "Z"),
            "contract_path": contract_relpath, "contract_sha256": contract_sha,
            "snapshot_files": entries, "snapshot_sha256": _canonical_sha(entries),
            "previous_manifest_sha256": previous_sha, "metadata": metadata,
            "immutable": True, "data_already_revealed": False,
            "outcome_metrics_computed_by_builder": False, "network_used": False,
            "safety": dict(contract["safety"])}


def commit_origin_manifest_exclusive(
    target: Path, paths: Iterable[Path], *, root: Path, contract_path: Path,
    contract: dict[str, Any], origin_index: int,
    previous_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build and persist a commitment before an evaluator can start."""
    root = root.resolve()
    target = _inside(target, root)
    _require(not target.exists(), "origin manifest already exists")
    manifest = build_origin_manifest(
        paths, root=root, contract_path=contract_path, contract=contract,
        origin_index=origin_index, previous_manifest_path=previous_manifest_path)
    committed = datetime.now(timezone.utc)
    origin_commit_window(contract, origin_index, committed)
    manifest["created_at_utc"] = committed.isoformat().replace("+00:00", "Z")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def load_verified_origin(
    manifest_path: Path, *, root: Path, contract_path: Path,
    contract: dict[str, Any], origin_index: int,
    previous_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fail closed on deadline, source bytes, future leakage and chain mismatch."""
    root = root.resolve()
    manifest = json.loads(_inside(manifest_path, root).read_text(encoding="utf-8"))
    contract_relpath, contract_sha = _contract_identity(root, contract_path, contract)
    _require(manifest.get("schema_version") == SCHEMA and manifest.get("immutable") is True
             and manifest.get("origin_index") == origin_index,
             "wrong origin or mutable manifest")
    _require(manifest.get("contract_path") == contract_relpath
             and manifest.get("contract_sha256") == contract_sha,
             "manifest contract identity mismatch")
    closed_at, deadline = origin_commit_window(
        contract, origin_index, _parse_utc(manifest["created_at_utc"]))
    _require(_parse_utc(manifest["created_at_utc"]) <= datetime.now(timezone.utc),
             "manifest timestamp is in the future")
    _require(manifest.get("origin_end_exclusive_utc") == closed_at.isoformat().replace("+00:00", "Z")
             and manifest.get("commit_deadline_exclusive_utc") == deadline.isoformat().replace("+00:00", "Z"),
             "manifest origin/deadline mismatch")
    _require(manifest.get("safety") == contract["safety"]
             and not any(manifest["safety"].values())
             and manifest.get("data_already_revealed") is False
             and manifest.get("outcome_metrics_computed_by_builder") is False
             and manifest.get("network_used") is False,
             "manifest safety or evidence label mismatch")
    previous, previous_sha = _previous(previous_manifest_path, root=root,
                                      contract=contract, contract_sha=contract_sha,
                                      origin_index=origin_index)
    _require(manifest.get("previous_manifest_sha256") == previous_sha,
             "previous manifest hash mismatch")
    if previous is not None:
        _require(_parse_utc(previous["created_at_utc"]) < _parse_utc(manifest["created_at_utc"]),
                 "manifest chain is not chronological")
    entries = manifest.get("snapshot_files")
    _require(isinstance(entries, list) and entries
             and _canonical_sha(entries) == manifest.get("snapshot_sha256"),
             "snapshot commitment malformed")
    _require(len(entries) == len({item["path"] for item in entries}),
             "duplicate snapshot paths")
    for item in entries:
        path = _inside(root / item["path"], root)
        _require(path.suffix.lower() == ".parquet" and path.stat().st_size == item["size_bytes"]
                 and file_sha256(path) == item["sha256"], "snapshot file changed")
    metadata, panel = _snapshot(entries, root=root, contract=contract,
                                origin_index=origin_index, previous=previous)
    _require(metadata == manifest.get("metadata"), "snapshot coverage changed")
    return manifest, panel


def reserve_origin_evaluation(
    started_path: Path, result_path: Path, manifest_path: Path, *,
    root: Path, contract_path: Path, contract: dict[str, Any],
    origin_index: int, previous_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify the commitment and write an exclusive marker before any PnL run."""
    root = root.resolve()
    started_path = _inside(started_path, root)
    result_path = _inside(result_path, root)
    _require(not started_path.exists() and not result_path.exists(),
             "origin evaluation already attempted")
    manifest, panel = load_verified_origin(
        manifest_path, root=root, contract_path=contract_path,
        contract=contract, origin_index=origin_index,
        previous_manifest_path=previous_manifest_path)
    started_at = datetime.now(timezone.utc)
    _require(_parse_utc(manifest["created_at_utc"]) < started_at,
             "evaluation cannot precede data commitment")
    marker = {"schema_version": "exia.genetic_primus.prospective_evaluation_started/2",
              "origin_index": origin_index,
              "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
              "manifest_sha256": file_sha256(_inside(manifest_path, root)),
              "contract_sha256": manifest["contract_sha256"],
              "outcome_metrics_available_before_marker": False,
              "safety": dict(contract["safety"])}
    started_path.parent.mkdir(parents=True, exist_ok=True)
    with started_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest, panel
