from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .core import ALLOWED_SYMBOLS, HashChainTape, load_contract, parse_force_order


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _atomic_status(path: Path, payload: dict[str, Any]) -> None:
    status = {
        "schema": "genetic_primus.v8.clean.status/2",
        **payload,
        "credentials_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _snapshot_sources(run_dir: Path, contract_path: Path) -> tuple[str, dict[str, str]]:
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=False)
    sources = {
        "collector.py": Path(__file__).resolve(),
        "core.py": Path(__file__).with_name("core.py").resolve(),
        "contract.json": contract_path.resolve(),
    }
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for name, source in sorted(sources.items()):
        data = source.read_bytes()
        (snapshot_dir / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    fingerprint = digest.hexdigest()
    manifest = {
        "schema": "genetic_primus.v8.clean.source_snapshot/1",
        "collector_fingerprint_sha256": fingerprint,
        "files": hashes,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint, hashes


async def collect_public_liquidations(
    *,
    contract_path: Path,
    run_dir: Path,
    duration_seconds: float,
    connect_fn: Callable[..., Any] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if duration_seconds > 300:
        raise ValueError("clean-room collector is intentionally bounded to 300 seconds")
    contract_bytes = contract_path.read_bytes()
    contract = load_contract(contract_path, root=contract_path.resolve().parent.parent)
    source = contract["source"]
    run_dir.mkdir(parents=True, exist_ok=False)
    collector_fingerprint, source_hashes = _snapshot_sources(run_dir, contract_path)
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    tape = HashChainTape(
        run_dir / "liquidation_tape.jsonl",
        root_id=run_dir.name,
        contract_sha256=contract_hash,
    )
    status_path = run_dir / "collector_status.json"
    monotonic = monotonic_fn or time.monotonic
    started = monotonic()
    deadline = started + duration_seconds
    connections = 0
    events = 0
    link_events = 0
    duplicate_events = 0
    source_gaps = 0
    connected_seconds = 0.0
    error: str | None = None
    seen_events: set[tuple[Any, ...]] = set()

    def update(state: str, connected: bool) -> None:
        _atomic_status(
            status_path,
            {
                "root_id": run_dir.name,
                "pid": os.getpid(),
                "state": state,
                "connected": connected,
                "updated_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_hash,
                "collector_fingerprint_sha256": collector_fingerprint,
                "connections": connections,
                "events": events,
                "link_events": link_events,
                "duplicate_events": duplicate_events,
                "source_gaps": source_gaps,
                "connected_seconds": connected_seconds,
                "uptime_seconds": monotonic() - started,
                "error": error,
            },
        )

    tape.append(
        kind="session_start",
        observed_at_ms=_now_ms(),
        payload={
            "connected": False,
            "stream": source["stream"],
            "collector_fingerprint_sha256": collector_fingerprint,
            "source_snapshot_sha256": source_hashes,
        },
    )
    update("starting", False)
    if connect_fn is None:
        from websockets.asyncio.client import connect

        connect_fn = connect

    try:
        reconnect_delay = 1.0
        while monotonic() < deadline:
            connection_started: float | None = None
            try:
                async with connect_fn(
                    source["url"],
                    ping_interval=20,
                    ping_timeout=float(source["ping_timeout_seconds"]),
                    close_timeout=5,
                    max_size=4 * 1024 * 1024,
                ) as websocket:
                    connections += 1
                    connection_started = monotonic()
                    reconnect_delay = 1.0
                    tape.append(
                        kind="connection_open",
                        observed_at_ms=_now_ms(),
                        payload={"connection": connections},
                    )
                    update("collecting", True)
                    heartbeat_at = monotonic() + float(source["heartbeat_seconds"])
                    while monotonic() < deadline:
                        now = monotonic()
                        timeout = min(deadline - now, max(0.01, heartbeat_at - now))
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=max(0.01, timeout))
                        except asyncio.TimeoutError:
                            if monotonic() >= deadline:
                                break
                            pong = await websocket.ping()
                            await asyncio.wait_for(
                                pong,
                                timeout=float(source["ping_timeout_seconds"]),
                            )
                            tape.append(
                                kind="heartbeat",
                                observed_at_ms=_now_ms(),
                                payload={"connection": connections, "ping_confirmed": True},
                            )
                            heartbeat_at = monotonic() + float(source["heartbeat_seconds"])
                            update("collecting", True)
                            continue
                        raw_text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        parsed = parse_force_order(
                            raw_text,
                            required_symbol_type=int(source["required_symbol_type"]),
                            allowed_symbols=ALLOWED_SYMBOLS,
                        )
                        tape.append(
                            kind="source_message",
                            observed_at_ms=_now_ms(),
                            payload={"connection": connections, "raw": raw_text, "decoded_events": len(parsed)},
                        )
                        for event in parsed:
                            identity = (
                                event.symbol,
                                event.symbol_type,
                                event.side,
                                event.event_time_ms,
                                event.order_time_ms,
                                event.quantity_contracts,
                                event.average_price,
                                event.status,
                            )
                            if identity in seen_events:
                                duplicate_events += 1
                                continue
                            seen_events.add(identity)
                            tape.append_liquidation(event, observed_at_ms=_now_ms())
                            events += 1
                            link_events += int(event.symbol == "LINKUSD_PERP")
                        update("collecting", True)
                    connected_seconds += max(0.0, monotonic() - connection_started)
                    connection_started = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if connection_started is not None:
                    connected_seconds += max(0.0, monotonic() - connection_started)
                source_gaps += 1
                error = f"{type(exc).__name__}: {exc}"[:500]
                tape.append(
                    kind="connection_closed",
                    observed_at_ms=_now_ms(),
                    payload={
                        "connection": connections,
                        "error": error,
                        "reconnect_delay_seconds": reconnect_delay,
                    },
                )
                update("reconnecting", False)
                remaining = deadline - monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(reconnect_delay, remaining))
                reconnect_delay = min(
                    float(source["maximum_reconnect_delay_seconds"]),
                    reconnect_delay * 2,
                )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        tape.append(
            kind="session_abort",
            observed_at_ms=_now_ms(),
            payload={"error": error},
        )
        update("failed", False)
        raise
    tape.append(kind="session_stop", observed_at_ms=_now_ms(), payload={"reason": "duration_complete"})
    update("completed", False)
    return {
        "root_id": run_dir.name,
        "contract_sha256": contract_hash,
        "collector_fingerprint_sha256": collector_fingerprint,
        "connections": connections,
        "events": events,
        "link_events": link_events,
        "duplicate_events": duplicate_events,
        "source_gaps": source_gaps,
        "connected_seconds": connected_seconds,
        "orders_enabled": False,
        "promotion_authority": False,
    }
