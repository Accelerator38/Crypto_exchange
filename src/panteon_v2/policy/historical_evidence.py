"""Fail-closed qualification of historical CarryFlow source capabilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


HISTORICAL_CAPABILITY_SCHEMA_VERSION = (
    "panteon.bitget_historical_evidence_capability.v1"
)

_REQUIREMENTS = (
    (
        "market_ohlcv",
        "fetchOHLCV",
        "fetch_ohlcv",
        "https://www.bitget.com/api-doc/classic/contract/market/Get-Candle-Data",
    ),
    (
        "mark_ohlcv",
        "fetchMarkOHLCV",
        "fetch_mark_ohlcv",
        "https://www.bitget.com/api-doc/classic/contract/market/Get-Candle-Data",
    ),
    (
        "index_ohlcv",
        "fetchIndexOHLCV",
        "fetch_index_ohlcv",
        "https://www.bitget.com/api-doc/classic/contract/market/Get-Candle-Data",
    ),
    (
        "funding_rate",
        "fetchFundingRateHistory",
        "fetch_funding_rate_history",
        "https://www.bitget.com/api-doc/classic/contract/market/Get-History-Funding-Rate",
    ),
    (
        "long_short_ratio",
        "fetchLongShortRatioHistory",
        "fetch_long_short_ratio_history",
        "https://www.bitget.com/api-doc/classic/common/apidata/Account-Long-Short",
    ),
    (
        "open_interest",
        "fetchOpenInterestHistory",
        "fetch_open_interest_history",
        "https://www.bitget.com/api-doc/classic/contract/market/Get-Open-Interest",
    ),
)


def audit_bitget_historical_capabilities(
    exchange: Any,
    *,
    symbol: str,
    timeframe: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Probe every field required to reconstruct one historical decision row."""

    generated = (generated_at or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    )
    advertised = getattr(exchange, "has", {})
    advertised = advertised if isinstance(advertised, Mapping) else {}
    capabilities: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for field, capability, method_name, documentation in _REQUIREMENTS:
        is_advertised = advertised.get(capability) is True
        row = {
            "ccxt_capability": capability,
            "ccxt_method": method_name,
            "official_documentation": documentation,
            "advertised": is_advertised,
            "status": "not_supported",
            "sample_count": 0,
            "oldest_timestamp_ms": None,
            "newest_timestamp_ms": None,
            "error": "",
        }
        if is_advertised:
            method = getattr(exchange, method_name, None)
            if not callable(method):
                row["status"] = "client_method_missing"
            else:
                try:
                    records = _probe(
                        method,
                        field=field,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    timestamps = _timestamps(records)
                    row["sample_count"] = len(records)
                    if timestamps:
                        row["status"] = "available"
                        row["oldest_timestamp_ms"] = min(timestamps)
                        row["newest_timestamp_ms"] = max(timestamps)
                    else:
                        row["status"] = "timestamped_history_missing"
                except Exception as exc:  # exchange clients use many error types
                    row["status"] = "probe_failed"
                    row["error"] = f"{type(exc).__name__}: {exc}"
        if row["status"] != "available":
            missing.append(field)
        capabilities[field] = row

    promotion_possible = not missing
    primary_failure = ""
    if missing == ["open_interest"]:
        primary_failure = "historical_open_interest_unavailable"
    elif missing:
        primary_failure = "historical_required_context_unavailable"
    return {
        "schema_version": HISTORICAL_CAPABILITY_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "source_exchange": "BITGET",
        "market_type": "swap",
        "probe_symbol": symbol,
        "timeframe": timeframe,
        "required_fields": [item[0] for item in _REQUIREMENTS],
        "capabilities": capabilities,
        "missing_required_fields": missing,
        "promotion_backfill_possible": promotion_possible,
        "evidence_tape_creation_allowed": promotion_possible,
        "research_backfill_possible": all(
            capabilities[field]["status"] == "available"
            for field in (
                "market_ohlcv",
                "mark_ohlcv",
                "index_ohlcv",
                "funding_rate",
                "long_short_ratio",
            )
        ),
        "allowed_use": (
            "promotion_evidence" if promotion_possible else "diagnostic_only"
        ),
        "primary_failure": primary_failure,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def _probe(
    method: Any,
    *,
    field: str,
    symbol: str,
    timeframe: str,
) -> list[Any]:
    if field in {"market_ohlcv", "mark_ohlcv", "index_ohlcv"}:
        records = method(symbol, timeframe=timeframe, limit=2)
    elif field in {"long_short_ratio", "open_interest"}:
        records = method(symbol, timeframe=timeframe, limit=2)
    else:
        records = method(symbol, limit=2)
    return list(records or [])


def _timestamps(records: list[Any]) -> list[int]:
    timestamps: list[int] = []
    for record in records:
        raw = record.get("timestamp") if isinstance(record, Mapping) else (
            record[0]
            if isinstance(record, (list, tuple)) and record
            else None
        )
        try:
            timestamp = int(float(raw))
        except (TypeError, ValueError):
            continue
        if timestamp > 0:
            timestamps.append(timestamp)
    return timestamps
