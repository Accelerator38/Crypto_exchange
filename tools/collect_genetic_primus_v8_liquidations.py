from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import sha256_file  # noqa: E402
from exia.genetic_primus.prospective_v8 import (  # noqa: E402
    LiquidationTape,
    canonical_json,
    decode_force_order,
    load_v8_contract,
)


DEFAULT_CONTRACT = ROOT / "configs/genetic_primus_v8_prospective_link_contract.json"
DEFAULT_BASE_DIR = ROOT / "Retrodate/genetic_primus_v8_liquidation_tape"
MODULE_PATH = ROOT / "src/exia/genetic_primus/prospective_v8.py"


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def collector_fingerprint(contract_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), MODULE_PATH.resolve(), contract_path.resolve()):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    sealed = {
        "schema_version": "exia.genetic_primus_liquidation_status.v8",
        **payload,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sealed, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RootLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "RootLock":
        self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(self.descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(self.descriptor)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


async def collect(
    *,
    contract_path: Path,
    run_dir: Path,
    duration_seconds: float | None,
    connect_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    contract = load_v8_contract(contract_path, root=ROOT)
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    prospective_start = int(pd_timestamp_ms(contract.payload["prospective_not_before"]))
    if _now_ms() < prospective_start:
        raise PermissionError("V8 prospective collection cannot start before contract freeze")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    tape_path = run_dir / "liquidation_tape.jsonl"
    status_path = run_dir / "collector_status.json"
    lock_path = run_dir / ".collector.lock"
    fingerprint = collector_fingerprint(contract_path)
    revision = _source_revision()
    run_id = run_dir.name
    tape = LiquidationTape(
        tape_path,
        contract_sha256=contract.sha256,
        collector_fingerprint_sha256=fingerprint,
        run_id=run_id,
    )
    if connect_fn is None:
        from websockets.asyncio.client import connect

        connect_fn = connect
    started_monotonic = time.monotonic()
    deadline = started_monotonic + duration_seconds if duration_seconds is not None else None
    connections = 0
    liquidation_events = 0
    link_events = 0
    duplicate_events = 0
    quality_events = 0
    connected_seconds = 0.0
    seen_event_ids: set[str] = set()

    def status(run_state: str, *, connected: bool, error: str | None = None) -> None:
        _write_status(
            status_path,
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "run_state": run_state,
                "connected": connected,
                "updated_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract.sha256,
                "collector_fingerprint_sha256": fingerprint,
                "source_revision": revision,
                "connections": connections,
                "liquidation_events": liquidation_events,
                "link_events": link_events,
                "duplicate_events": duplicate_events,
                "quality_events": quality_events,
                "connected_seconds": connected_seconds,
                "uptime_seconds": max(0.0, time.monotonic() - started_monotonic),
                "last_record_sha256": tape.previous_sha256,
                "error": error,
            },
        )

    with RootLock(lock_path):
        tape.append(
            record_type="session_start",
            observed_at_ms=_now_ms(),
            payload={"source_revision": revision, "stream": contract.stream["stream"], "connected": False},
        )
        status("starting", connected=False)
        reconnect_delay = 1.0
        stop_reason = "duration_complete" if duration_seconds is not None else "collector_shutdown"
        try:
            while deadline is None or time.monotonic() < deadline:
                connection_started: float | None = None
                try:
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    if remaining == 0.0:
                        break
                    async with connect_fn(
                        contract.stream["url"],
                        ping_interval=20,
                        ping_timeout=float(contract.stream["ping_timeout_seconds"]),
                        close_timeout=5,
                        max_size=4 * 1024 * 1024,
                    ) as websocket:
                        connections += 1
                        connection_started = time.monotonic()
                        reconnect_delay = 1.0
                        tape.append(
                            record_type="connection_open",
                            observed_at_ms=_now_ms(),
                            payload={"connection": connections, "connected": True},
                        )
                        status("collecting", connected=True)
                        heartbeat_deadline = time.monotonic() + float(contract.stream["heartbeat_seconds"])
                        while deadline is None or time.monotonic() < deadline:
                            timeout = max(0.01, heartbeat_deadline - time.monotonic())
                            if deadline is not None:
                                timeout = min(timeout, max(0.01, deadline - time.monotonic()))
                            try:
                                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                            except asyncio.TimeoutError:
                                if deadline is not None and time.monotonic() >= deadline:
                                    break
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=float(contract.stream["ping_timeout_seconds"]))
                                connected_seconds += max(0.0, time.monotonic() - connection_started)
                                connection_started = time.monotonic()
                                tape.append(
                                    record_type="heartbeat",
                                    observed_at_ms=_now_ms(),
                                    payload={"connection": connections, "connected": True, "ping_confirmed": True},
                                )
                                status("collecting", connected=True)
                                heartbeat_deadline = time.monotonic() + float(contract.stream["heartbeat_seconds"])
                                continue
                            events, quality = decode_force_order(raw, contract)
                            for kind in quality:
                                quality_events += 1
                                tape.append(
                                    record_type="quality",
                                    observed_at_ms=_now_ms(),
                                    payload={"kind": kind, "connection": connections},
                                )
                            for event in events:
                                identity_payload = {key: value for key, value in event.items() if key != "raw"}
                                event_id = hashlib.sha256(canonical_json(identity_payload).encode("ascii")).hexdigest()
                                if event_id in seen_event_ids:
                                    duplicate_events += 1
                                    continue
                                seen_event_ids.add(event_id)
                                liquidation_events += 1
                                link_events += event["target_symbol"] == "LINK/USDT"
                                tape.append(
                                    record_type="liquidation",
                                    observed_at_ms=_now_ms(),
                                    payload={"event_id": event_id, **event},
                                )
                            if time.monotonic() >= heartbeat_deadline:
                                connected_seconds += max(0.0, time.monotonic() - connection_started)
                                connection_started = time.monotonic()
                                tape.append(
                                    record_type="heartbeat",
                                    observed_at_ms=_now_ms(),
                                    payload={"connection": connections, "connected": True, "ping_confirmed": False},
                                )
                                status("collecting", connected=True)
                                heartbeat_deadline = time.monotonic() + float(contract.stream["heartbeat_seconds"])
                        if connection_started is not None:
                            connected_seconds += max(0.0, time.monotonic() - connection_started)
                            connection_started = None
                except asyncio.CancelledError:
                    stop_reason = "cancelled"
                    raise
                except Exception as exc:
                    if connection_started is not None:
                        connected_seconds += max(0.0, time.monotonic() - connection_started)
                    quality_events += 1
                    tape.append(
                        record_type="connection_closed",
                        observed_at_ms=_now_ms(),
                        payload={
                            "connection": connections,
                            "connected": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "reconnect_delay_seconds": reconnect_delay,
                        },
                    )
                    status("reconnecting", connected=False, error=f"{type(exc).__name__}: {exc}"[:500])
                    remaining = reconnect_delay if deadline is None else min(reconnect_delay, max(0.0, deadline - time.monotonic()))
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    reconnect_delay = min(30.0, reconnect_delay * 2.0)
            tape.append(
                record_type="session_stop",
                observed_at_ms=_now_ms(),
                payload={"reason": stop_reason, "connected": False},
            )
            status("completed" if duration_seconds is not None else "stopped", connected=False)
        except BaseException as exc:
            status("failed", connected=False, error=f"{type(exc).__name__}: {exc}"[:500])
            raise
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "contract_sha256": contract.sha256,
        "collector_fingerprint_sha256": fingerprint,
        "connections": connections,
        "liquidation_events": liquidation_events,
        "link_events": link_events,
        "quality_events": quality_events,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def pd_timestamp_ms(value: str) -> int:
    from datetime import datetime

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("V8 prospective timestamp must be timezone-aware")
    return int(parsed.timestamp() * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect read-only Genetic_Primus V8 liquidation evidence")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--duration-seconds", type=float)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = load_v8_contract(contract_path, root=ROOT)
    run_id = f"root_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{contract.sha256[:8]}_{uuid.uuid4().hex[:6]}"
    run_dir = args.run_dir.resolve() if args.run_dir else DEFAULT_BASE_DIR / run_id
    result = asyncio.run(
        collect(
            contract_path=contract_path,
            run_dir=run_dir,
            duration_seconds=args.duration_seconds,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
