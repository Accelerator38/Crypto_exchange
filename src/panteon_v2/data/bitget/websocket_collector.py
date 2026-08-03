"""Public Bitget WebSocket decoder and read-only collection loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .profile import BitgetDataProfile
from .rest_reconciler import BitgetPublicRestReconciler
from .segment_store import BitgetSegmentStore


@dataclass(frozen=True)
class DecodedEvent:
    channel: str
    symbol: str
    exchange_timestamp_ms: int
    event_key: str
    action: str
    sequence: int | None
    payload: Mapping[str, Any]


class BitgetPublicMessageDecoder:
    def __init__(self, profile: BitgetDataProfile) -> None:
        self.profile = profile
        self.allowed_symbols = frozenset(profile.symbols)
        self.allowed_channels = frozenset(profile.channels)
        self.last_book_sequence: dict[str, int] = {}

    def decode(
        self,
        raw_message: str | bytes,
    ) -> tuple[list[DecodedEvent], list[dict[str, Any]]]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        if raw_message == "pong":
            return [], []
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            return [], [{"kind": "invalid_json", "details": {"raw": raw_message[:200]}}]
        if not isinstance(message, Mapping):
            return [], [{"kind": "invalid_message_type", "details": {}}]
        if "event" in message:
            event = str(message.get("event") or "")
            if event == "error" or str(message.get("code") or "") not in {"", "0"}:
                return [], [
                    {
                        "kind": "websocket_control_error",
                        "details": dict(message),
                    }
                ]
            return [], []
        arg = message.get("arg")
        data = message.get("data")
        if (
            not isinstance(arg, Mapping)
            or not isinstance(data, Sequence)
            or isinstance(data, (str, bytes))
        ):
            return [], [{"kind": "invalid_data_message", "details": dict(message)}]
        channel = str(arg.get("channel") or "")
        symbol = str(arg.get("instId") or "").upper()
        if channel not in self.allowed_channels:
            return [], [
                {
                    "kind": "unexpected_channel",
                    "channel": channel,
                    "symbol": symbol,
                    "details": {},
                }
            ]
        if symbol not in self.allowed_symbols:
            return [], [
                {
                    "kind": "unexpected_symbol",
                    "channel": channel,
                    "symbol": symbol,
                    "details": {},
                }
            ]
        action = str(message.get("action") or "snapshot")
        top_timestamp = _positive_int(message.get("ts"))
        events: list[DecodedEvent] = []
        quality: list[dict[str, Any]] = []
        for index, payload in enumerate(data):
            if not isinstance(payload, Mapping):
                quality.append(
                    {
                        "kind": "invalid_channel_payload",
                        "channel": channel,
                        "symbol": symbol,
                        "details": {"index": index},
                    }
                )
                continue
            row = dict(payload)
            exchange_ts = _positive_int(row.get("ts")) or top_timestamp
            if exchange_ts <= 0:
                quality.append(
                    {
                        "kind": "missing_exchange_timestamp",
                        "channel": channel,
                        "symbol": symbol,
                        "details": {"index": index},
                    }
                )
                continue
            sequence: int | None = None
            if channel == "trade":
                trade_id = str(row.get("tradeId") or "").strip()
                if not trade_id or not _positive_number(row.get("price")) or not _positive_number(
                    row.get("size")
                ):
                    quality.append(
                        {
                            "kind": "invalid_trade",
                            "channel": channel,
                            "symbol": symbol,
                            "exchange_timestamp_ms": exchange_ts,
                            "details": {"index": index},
                        }
                    )
                    continue
                event_key = f"trade:{trade_id}"
            elif channel == "books5":
                sequence = _positive_int(row.get("seq")) or None
                bids = _levels(row.get("bids"))
                asks = _levels(row.get("asks"))
                if not bids or not asks:
                    quality.append(
                        {
                            "kind": "empty_book",
                            "channel": channel,
                            "symbol": symbol,
                            "exchange_timestamp_ms": exchange_ts,
                            "details": {},
                        }
                    )
                    continue
                if asks[0][0] < bids[0][0]:
                    quality.append(
                        {
                            "kind": "crossed_book",
                            "channel": channel,
                            "symbol": symbol,
                            "exchange_timestamp_ms": exchange_ts,
                            "details": {
                                "best_bid": bids[0][0],
                                "best_ask": asks[0][0],
                            },
                        }
                    )
                if sequence is not None:
                    previous = self.last_book_sequence.get(symbol)
                    if previous is not None and sequence <= previous:
                        quality.append(
                            {
                                "kind": "book_sequence_nonincreasing",
                                "channel": channel,
                                "symbol": symbol,
                                "exchange_timestamp_ms": exchange_ts,
                                "details": {
                                    "previous_sequence": previous,
                                    "sequence": sequence,
                                },
                            }
                        )
                    self.last_book_sequence[symbol] = max(sequence, previous or sequence)
                event_key = (
                    f"book:{sequence}"
                    if sequence is not None
                    else f"book:{exchange_ts}:{_payload_sha(row)[:16]}"
                )
            else:
                event_key = f"ticker:{exchange_ts}:{_payload_sha(row)[:16]}"
                bid = _number(row.get("bidPr"))
                ask = _number(row.get("askPr"))
                mark = _number(row.get("markPrice"))
                index_price = _number(row.get("indexPrice"))
                if bid <= 0.0 or ask <= 0.0 or mark <= 0.0 or index_price <= 0.0:
                    quality.append(
                        {
                            "kind": "incomplete_ticker",
                            "channel": channel,
                            "symbol": symbol,
                            "exchange_timestamp_ms": exchange_ts,
                            "details": {
                                "bid": bid,
                                "ask": ask,
                                "mark": mark,
                                "index": index_price,
                            },
                        }
                    )
                elif ask < bid:
                    quality.append(
                        {
                            "kind": "crossed_ticker",
                            "channel": channel,
                            "symbol": symbol,
                            "exchange_timestamp_ms": exchange_ts,
                            "details": {"bid": bid, "ask": ask},
                        }
                    )
            events.append(
                DecodedEvent(
                    channel=channel,
                    symbol=symbol,
                    exchange_timestamp_ms=exchange_ts,
                    event_key=event_key,
                    action=action,
                    sequence=sequence,
                    payload=row,
                )
            )
        if channel == "trade":
            # Bitget may batch public trades newest-first. Store the batch in
            # exchange-time order so a normal batch does not look like a feed
            # timestamp regression.
            events.sort(
                key=lambda item: (
                    item.exchange_timestamp_ms,
                    item.event_key,
                )
            )
        return events, quality


def build_subscription_request(profile: BitgetDataProfile) -> dict[str, Any]:
    return {
        "op": "subscribe",
        "args": [
            {
                "instType": profile.instrument_type,
                "channel": channel,
                "instId": symbol,
            }
            for channel in profile.channels
            for symbol in profile.symbols
        ],
    }


async def run_public_collection(
    *,
    profile: BitgetDataProfile,
    store: BitgetSegmentStore,
    duration_seconds: float | None = None,
    connect_fn: Callable[..., Any] | None = None,
    rest_reconciler: Any | None = None,
) -> dict[str, Any]:
    """Collect public market data; this function has no order client or keys."""

    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if connect_fn is None:
        from websockets.asyncio.client import connect

        connect_fn = connect
    decoder = BitgetPublicMessageDecoder(profile)
    rest_client = rest_reconciler or BitgetPublicRestReconciler(profile)
    owns_rest_client = rest_reconciler is None
    rest_stats = {"cycles": 0, "events": 0}
    rest_task = asyncio.create_task(
        _run_rest_reconciliation(
            profile=profile,
            store=store,
            reconciler=rest_client,
            stats=rest_stats,
        )
    )
    subscription = json.dumps(
        build_subscription_request(profile),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    started = time.monotonic()
    deadline = started + duration_seconds if duration_seconds is not None else None
    reconnect_delay = 1.0
    connection_count = 0
    received_messages = 0
    decoded_events = 0
    last_status = 0.0
    stop_reason = "duration_complete" if deadline is not None else "collector_shutdown"
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    break
                async with connect_fn(
                    profile.websocket_url,
                    ping_interval=None,
                    close_timeout=5,
                    max_size=4 * 1024 * 1024,
                ) as websocket:
                    connection_count += 1
                    await websocket.send(subscription)
                    store.write_status(
                        run_state="collecting",
                        connections=connection_count,
                        received_messages=received_messages,
                        decoded_events=decoded_events,
                    )
                    reconnect_delay = 1.0
                    last_ping = time.monotonic()
                    while deadline is None or time.monotonic() < deadline:
                        timeout = float(profile.heartbeat_seconds)
                        if deadline is not None:
                            timeout = min(timeout, max(0.01, deadline - time.monotonic()))
                        try:
                            raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            if deadline is not None and time.monotonic() >= deadline:
                                break
                            await websocket.send("ping")
                            last_ping = time.monotonic()
                            raw = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=min(10.0, float(profile.heartbeat_seconds)),
                            )
                        received_messages += 1
                        received_utc_ns = time.time_ns()
                        received_monotonic_ns = time.monotonic_ns()
                        events, quality = decoder.decode(raw)
                        for item in quality:
                            store.append_quality_event(
                                kind=str(item["kind"]),
                                channel=str(item.get("channel") or ""),
                                symbol=str(item.get("symbol") or ""),
                                exchange_timestamp_ms=item.get(
                                    "exchange_timestamp_ms"
                                ),
                                details=item.get("details") or {},
                                received_timestamp_utc_ns=received_utc_ns,
                            )
                        for event in events:
                            store.append_event(
                                channel=event.channel,
                                symbol=event.symbol,
                                exchange_timestamp_ms=event.exchange_timestamp_ms,
                                received_timestamp_utc_ns=received_utc_ns,
                                received_timestamp_monotonic_ns=received_monotonic_ns,
                                event_key=event.event_key,
                                action=event.action,
                                sequence=event.sequence,
                                payload=event.payload,
                            )
                            decoded_events += 1
                        if (
                            time.monotonic() - last_ping
                            >= float(profile.heartbeat_seconds)
                        ):
                            await websocket.send("ping")
                            last_ping = time.monotonic()
                        if time.monotonic() - last_status >= 5.0:
                            store.write_status(
                                run_state="collecting",
                                connections=connection_count,
                                received_messages=received_messages,
                                decoded_events=decoded_events,
                            )
                            last_status = time.monotonic()
            except asyncio.CancelledError:
                stop_reason = "cancelled"
                raise
            except Exception as exc:
                store.append_quality_event(
                    kind="websocket_disconnect",
                    details={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "retry_seconds": reconnect_delay,
                    },
                )
                store.write_status(
                    run_state="reconnecting",
                    connections=connection_count,
                    received_messages=received_messages,
                    decoded_events=decoded_events,
                    last_error=f"{type(exc).__name__}: {exc}",
                    retry_seconds=reconnect_delay,
                )
                sleep_for = reconnect_delay
                if deadline is not None:
                    sleep_for = min(sleep_for, max(0.0, deadline - time.monotonic()))
                if sleep_for > 0.0:
                    await asyncio.sleep(sleep_for)
                reconnect_delay = min(
                    reconnect_delay * 2.0,
                    float(profile.reconnect_max_seconds),
                )
    finally:
        rest_task.cancel()
        try:
            await rest_task
        except asyncio.CancelledError:
            pass
        if owns_rest_client:
            rest_client.close()
        summary = store.close(reason=stop_reason)
    return {
        "schema_version": "panteon.bitget_public_collection_summary.v1",
        "session_id": store.session_id,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "source_revision": store.source_revision,
        "connections": connection_count,
        "received_messages": received_messages,
        "decoded_events": decoded_events,
        "rest_reconciliation_cycles": rest_stats["cycles"],
        "rest_reconciliation_events": rest_stats["events"],
        "duration_seconds": time.monotonic() - started,
        "last_manifest_sha256": (
            summary.manifest_sha256 if summary is not None else None
        ),
        "orders_enabled": False,
        "promotion_authority": False,
    }


async def _run_rest_reconciliation(
    *,
    profile: BitgetDataProfile,
    store: BitgetSegmentStore,
    reconciler: Any,
    stats: dict[str, int],
) -> None:
    include_rules = True
    last_rules_at = 0.0
    while True:
        cycle_started = time.monotonic()
        if cycle_started - last_rules_at >= float(profile.rules_refresh_seconds):
            include_rules = True
        try:
            events, quality = await asyncio.to_thread(
                reconciler.fetch_cycle,
                include_rules=include_rules,
            )
            now_monotonic_ns = time.monotonic_ns()
            for item in quality:
                store.append_quality_event(
                    kind=str(item["kind"]),
                    channel=str(item.get("channel") or ""),
                    symbol=str(item.get("symbol") or ""),
                    details=item.get("details") or {},
                )
            for event in events:
                store.append_event(
                    channel=event.channel,
                    symbol=event.symbol,
                    exchange_timestamp_ms=event.exchange_timestamp_ms,
                    received_timestamp_utc_ns=event.received_timestamp_utc_ns,
                    received_timestamp_monotonic_ns=now_monotonic_ns,
                    event_key=event.event_key,
                    action="rest_snapshot",
                    sequence=None,
                    payload=event.payload,
                )
            stats["cycles"] += 1
            stats["events"] += len(events)
            if include_rules:
                last_rules_at = cycle_started
                include_rules = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            store.append_quality_event(
                kind="rest_reconciliation_cycle_failed",
                details={
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                },
            )
        elapsed = time.monotonic() - cycle_started
        await asyncio.sleep(
            max(0.0, float(profile.rest_reconcile_seconds) - elapsed)
        )


def _levels(raw: Any) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return result
    for level in raw:
        if not isinstance(level, Sequence) or len(level) < 2:
            continue
        price = _number(level[0])
        size = _number(level[1])
        if price > 0.0 and size >= 0.0:
            result.append((price, size))
    return result


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _positive_number(value: Any) -> bool:
    return _number(value) > 0.0


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _payload_sha(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()
