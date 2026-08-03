"""Sealed operational profile for the read-only Bitget data collector."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


BITGET_DATA_PROFILE_SCHEMA_VERSION = "panteon.bitget_data_profile.v1"
BITGET_DATA_PROFILE_IDS = frozenset({"bitget_full8_microstructure_v1"})
BITGET_DATA_CHANNELS = ("trade", "books5", "ticker")
BITGET_FULL8_MARKETS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
)
_PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "source_exchange",
        "market_type",
        "websocket_url",
        "rest_base_url",
        "instrument_type",
        "symbols",
        "channels",
        "segment_seconds",
        "commit_batch_events",
        "commit_interval_ms",
        "heartbeat_seconds",
        "reconnect_max_seconds",
        "rest_reconcile_seconds",
        "rules_refresh_seconds",
        "orders_enabled",
        "promotion_authority",
    }
)
_MARKET_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")


class BitgetDataProfileError(ValueError):
    """A Bitget data profile is ambiguous or unsafe."""


@dataclass(frozen=True)
class BitgetDataProfile:
    path: Path
    profile_sha256: str
    schema_version: str
    profile_id: str
    source_exchange: str
    market_type: str
    websocket_url: str
    rest_base_url: str
    instrument_type: str
    symbols: tuple[str, ...]
    channels: tuple[str, ...]
    segment_seconds: int
    commit_batch_events: int
    commit_interval_ms: int
    heartbeat_seconds: int
    reconnect_max_seconds: int
    rest_reconcile_seconds: int
    rules_refresh_seconds: int
    orders_enabled: bool
    promotion_authority: bool


def load_bitget_data_profile(path: str | Path) -> BitgetDataProfile:
    source = Path(path)
    if not source.is_file():
        raise BitgetDataProfileError(f"data profile not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BitgetDataProfileError(f"invalid data profile JSON: {exc}") from exc
    row = validate_bitget_data_profile(raw)
    canonical = json.dumps(
        row,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return BitgetDataProfile(
        path=source.resolve(),
        profile_sha256=hashlib.sha256(canonical).hexdigest(),
        schema_version=str(row["schema_version"]),
        profile_id=str(row["profile_id"]),
        source_exchange=str(row["source_exchange"]),
        market_type=str(row["market_type"]),
        websocket_url=str(row["websocket_url"]),
        rest_base_url=str(row["rest_base_url"]),
        instrument_type=str(row["instrument_type"]),
        symbols=tuple(str(item) for item in row["symbols"]),
        channels=tuple(str(item) for item in row["channels"]),
        segment_seconds=int(row["segment_seconds"]),
        commit_batch_events=int(row["commit_batch_events"]),
        commit_interval_ms=int(row["commit_interval_ms"]),
        heartbeat_seconds=int(row["heartbeat_seconds"]),
        reconnect_max_seconds=int(row["reconnect_max_seconds"]),
        rest_reconcile_seconds=int(row["rest_reconcile_seconds"]),
        rules_refresh_seconds=int(row["rules_refresh_seconds"]),
        orders_enabled=bool(row["orders_enabled"]),
        promotion_authority=bool(row["promotion_authority"]),
    )


def validate_bitget_data_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = json.loads(json.dumps(dict(payload)))
    if set(row) != _PROFILE_KEYS:
        missing = sorted(_PROFILE_KEYS - set(row))
        extra = sorted(set(row) - _PROFILE_KEYS)
        raise BitgetDataProfileError(
            f"profile keys mismatch: missing={missing}, extra={extra}"
        )
    if row["schema_version"] != BITGET_DATA_PROFILE_SCHEMA_VERSION:
        raise BitgetDataProfileError("unsupported Bitget data profile schema")
    if row["profile_id"] not in BITGET_DATA_PROFILE_IDS:
        raise BitgetDataProfileError("unregistered Bitget data profile")
    if row["source_exchange"] != "BITGET":
        raise BitgetDataProfileError("source_exchange must be BITGET")
    if row["market_type"] != "usdt_futures":
        raise BitgetDataProfileError("market_type must be usdt_futures")
    if row["websocket_url"] != "wss://ws.bitget.com/v2/ws/public":
        raise BitgetDataProfileError("only the public Bitget websocket is allowed")
    if row["rest_base_url"] != "https://api.bitget.com":
        raise BitgetDataProfileError("only the public Bitget REST API is allowed")
    if row["instrument_type"] != "USDT-FUTURES":
        raise BitgetDataProfileError("instrument_type must be USDT-FUTURES")
    symbols = tuple(str(item).strip().upper() for item in row["symbols"])
    if symbols != BITGET_FULL8_MARKETS:
        raise BitgetDataProfileError("profile must use the exact Bitget full8 set")
    if len(set(symbols)) != len(symbols) or not all(
        _MARKET_RE.fullmatch(symbol) for symbol in symbols
    ):
        raise BitgetDataProfileError("invalid or duplicate Bitget symbol")
    channels = tuple(str(item).strip() for item in row["channels"])
    if channels != BITGET_DATA_CHANNELS:
        raise BitgetDataProfileError(
            "profile must use exact trade/books5/ticker channels"
        )
    _bounded_int(row, "segment_seconds", 60, 3600)
    if int(row["segment_seconds"]) % 60:
        raise BitgetDataProfileError("segment_seconds must align to minutes")
    _bounded_int(row, "commit_batch_events", 1, 10_000)
    _bounded_int(row, "commit_interval_ms", 100, 10_000)
    _bounded_int(row, "heartbeat_seconds", 10, 60)
    _bounded_int(row, "reconnect_max_seconds", 1, 300)
    _bounded_int(row, "rest_reconcile_seconds", 30, 300)
    _bounded_int(row, "rules_refresh_seconds", 3600, 86_400)
    if row["orders_enabled"] is not False:
        raise BitgetDataProfileError("orders_enabled must be false")
    if row["promotion_authority"] is not False:
        raise BitgetDataProfileError("promotion_authority must be false")
    row["symbols"] = list(symbols)
    row["channels"] = list(channels)
    return row


def _bounded_int(
    row: Mapping[str, Any],
    key: str,
    minimum: int,
    maximum: int,
) -> None:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise BitgetDataProfileError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise BitgetDataProfileError(f"{key} is outside [{minimum}, {maximum}]")
