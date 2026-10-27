from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core import load_contract, verify_tape


ACTIVE_STATES = frozenset({"starting", "collecting", "reconnecting"})
SNAPSHOT_FILES = frozenset({"collector.py", "core.py", "contract.json"})


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _verify_source_snapshot(run_dir: Path) -> str:
    snapshot_dir = run_dir / "source_snapshot"
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "genetic_primus.v8.clean.source_snapshot/1":
        raise ValueError("source snapshot schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or frozenset(files) != SNAPSHOT_FILES:
        raise ValueError("source snapshot file set mismatch")
    digest = hashlib.sha256()
    for name in sorted(SNAPSHOT_FILES):
        data = (snapshot_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != files[name]:
            raise ValueError(f"source snapshot hash mismatch: {name}")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    fingerprint = digest.hexdigest()
    if fingerprint != manifest.get("collector_fingerprint_sha256"):
        raise ValueError("collector fingerprint mismatch")
    return fingerprint


def check_run(*, run_dir: Path, contract_path: Path, root: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    contract_path = Path(contract_path).resolve()
    contract = load_contract(contract_path, root=Path(root).resolve())
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    blockers: list[str] = []

    collector_fingerprint: str | None = None
    try:
        collector_fingerprint = _verify_source_snapshot(run_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        blockers.append(f"invalid_source_snapshot:{type(exc).__name__}:{exc}")

    try:
        status = json.loads((run_dir / "collector_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "result": "FAILURE",
            "root_id": run_dir.name,
            "blockers": [f"invalid_status:{type(exc).__name__}"],
            "orders_enabled": False,
            "promotion_authority": False,
        }

    if status.get("schema") != "genetic_primus.v8.clean.status/2":
        blockers.append("status_schema_mismatch")
    if status.get("root_id") != run_dir.name:
        blockers.append("status_root_mismatch")
    if status.get("contract_sha256") != contract_sha256:
        blockers.append("status_contract_mismatch")
    if collector_fingerprint is not None and status.get("collector_fingerprint_sha256") != collector_fingerprint:
        blockers.append("status_collector_fingerprint_mismatch")
    for key in ("credentials_allowed", "orders_enabled", "promotion_authority"):
        if status.get(key) is not False:
            blockers.append(f"status_not_fail_closed:{key}")

    tape: dict[str, Any] | None = None
    try:
        tape = verify_tape(run_dir / "liquidation_tape.jsonl", contract_sha256=contract_sha256)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        blockers.append(f"invalid_tape:{type(exc).__name__}:{exc}")
    if tape is not None and tape.get("root_id") != run_dir.name:
        blockers.append("tape_root_mismatch")
    if (
        tape is not None
        and collector_fingerprint is not None
        and tape.get("session_payload", {}).get("collector_fingerprint_sha256") != collector_fingerprint
    ):
        blockers.append("tape_collector_fingerprint_mismatch")

    age_seconds: float | None = None
    try:
        updated = datetime.fromisoformat(str(status["updated_at"]))
        if updated.tzinfo is None:
            raise ValueError("naive timestamp")
        age_seconds = (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds()
    except (KeyError, TypeError, ValueError):
        blockers.append("invalid_status_timestamp")

    state = str(status.get("state", ""))
    process_running = _pid_exists(int(status.get("pid") or 0))
    if state in ACTIVE_STATES:
        stale_after = float(contract["source"]["heartbeat_seconds"]) * 3
        if not process_running:
            blockers.append("collector_process_not_running")
        if age_seconds is None or age_seconds > stale_after:
            blockers.append("collector_status_stale")
        if tape is not None and tape.get("last_kind") in {"session_stop", "session_abort"}:
            blockers.append("active_status_has_terminal_tape")
    elif state == "completed":
        if tape is not None and tape.get("last_kind") != "session_stop":
            blockers.append("completed_tape_missing_session_stop")
    elif state == "failed":
        blockers.append("collector_failed")
        if tape is not None and tape.get("last_kind") != "session_abort":
            blockers.append("failed_tape_missing_session_abort")
    else:
        blockers.append("unknown_collector_state")

    return {
        "result": "SUCCESS" if not blockers else "FAILURE",
        "root_id": run_dir.name,
        "state": state,
        "process_running": process_running,
        "status_age_seconds": age_seconds,
        "records": None if tape is None else tape["records"],
        "last_kind": None if tape is None else tape["last_kind"],
        "collector_fingerprint_sha256": collector_fingerprint,
        "events": int(status.get("events") or 0),
        "link_events": int(status.get("link_events") or 0),
        "source_gaps": int(status.get("source_gaps") or 0),
        "blockers": blockers,
        "orders_enabled": False,
        "promotion_authority": False,
    }
