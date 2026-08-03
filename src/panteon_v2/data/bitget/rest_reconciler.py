"""Public REST reconciliation for the Bitget WebSocket evidence stream."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import requests

from .profile import BitgetDataProfile


@dataclass(frozen=True)
class RestEvent:
    channel: str
    symbol: str
    exchange_timestamp_ms: int
    received_timestamp_utc_ns: int
    event_key: str
    payload: Mapping[str, Any]


class BitgetPublicRestReconciler:
    """Fetch public facts only; no credentials or private endpoints."""

    def __init__(
        self,
        profile: BitgetDataProfile,
        *,
        session: Any | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.profile = profile
        self.session = session or requests.Session()
        self.timeout_seconds = float(timeout_seconds)
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def fetch_cycle(
        self,
        *,
        include_rules: bool,
        now_ms: int | None = None,
    ) -> tuple[list[RestEvent], list[dict[str, Any]]]:
        observed_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        closed_minute_ms = observed_ms - observed_ms % 60_000
        candle_start_ms = closed_minute_ms - 60_000
        events: list[RestEvent] = []
        quality: list[dict[str, Any]] = []
        for symbol in self.profile.symbols:
            try:
                response, received_ns = self._get(
                    "/api/v2/mix/market/candles",
                    {
                        "symbol": symbol,
                        "productType": "usdt-futures",
                        "granularity": "1m",
                        # Bitget's startTime boundary is exclusive for this
                        # endpoint. Query one interval earlier, then select the
                        # exact closed candle by timestamp.
                        "startTime": str(candle_start_ms - 60_000),
                        "endTime": str(closed_minute_ms),
                        "limit": "2",
                    },
                )
                candle = _select_candle(response["data"], candle_start_ms)
                if candle is None:
                    raise ValueError("exact closed 1m candle unavailable")
                payload = {
                    "source_endpoint": "/api/v2/mix/market/candles",
                    "request_time_ms": int(response["requestTime"]),
                    "candle_start_timestamp_ms": candle_start_ms,
                    "bar_close_timestamp_ms": closed_minute_ms,
                    "open": str(candle[1]),
                    "high": str(candle[2]),
                    "low": str(candle[3]),
                    "close": str(candle[4]),
                    "base_volume": str(candle[5]),
                    "quote_volume": str(candle[6]) if len(candle) > 6 else "",
                }
                events.append(
                    RestEvent(
                        channel="rest_candle_1m",
                        symbol=symbol,
                        exchange_timestamp_ms=candle_start_ms,
                        received_timestamp_utc_ns=received_ns,
                        event_key=f"rest_candle_1m:{candle_start_ms}",
                        payload=payload,
                    )
                )
            except Exception as exc:
                quality.append(_rest_error("rest_candle_1m", symbol, exc))
            try:
                response, received_ns = self._get(
                    "/api/v2/mix/market/open-interest",
                    {
                        "symbol": symbol,
                        "productType": "usdt-futures",
                    },
                )
                data = response["data"]
                rows = data.get("openInterestList") if isinstance(data, Mapping) else None
                selected = next(
                    (
                        row
                        for row in rows or ()
                        if isinstance(row, Mapping)
                        and str(row.get("symbol") or "").upper() == symbol
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("open interest symbol missing")
                exchange_ts = _positive_int(data.get("ts")) or int(
                    response["requestTime"]
                )
                payload = {
                    "source_endpoint": "/api/v2/mix/market/open-interest",
                    "request_time_ms": int(response["requestTime"]),
                    "source_timestamp_ms": exchange_ts,
                    "open_interest_base": str(selected.get("size") or ""),
                }
                events.append(
                    RestEvent(
                        channel="rest_open_interest",
                        symbol=symbol,
                        exchange_timestamp_ms=exchange_ts,
                        received_timestamp_utc_ns=received_ns,
                        event_key=(
                            f"rest_open_interest:{exchange_ts}:"
                            f"{_payload_sha(payload)[:16]}"
                        ),
                        payload=payload,
                    )
                )
            except Exception as exc:
                quality.append(_rest_error("rest_open_interest", symbol, exc))
        try:
            response, received_ns = self._get(
                "/api/v2/mix/market/current-fund-rate",
                {"productType": "usdt-futures"},
            )
            funding_rows = {
                str(row.get("symbol") or "").upper(): row
                for row in response["data"]
                if isinstance(row, Mapping)
            }
            for symbol in self.profile.symbols:
                row = funding_rows.get(symbol)
                if row is None:
                    quality.append(
                        _rest_error(
                            "rest_funding",
                            symbol,
                            ValueError("funding symbol missing"),
                        )
                    )
                    continue
                exchange_ts = int(response["requestTime"])
                payload = {
                    "source_endpoint": "/api/v2/mix/market/current-fund-rate",
                    "request_time_ms": exchange_ts,
                    "funding_rate": str(row.get("fundingRate") or ""),
                    "funding_interval_hours": str(
                        row.get("fundingRateInterval") or ""
                    ),
                    "next_update_ms": _positive_int(row.get("nextUpdate")),
                    "min_funding_rate": str(row.get("minFundingRate") or ""),
                    "max_funding_rate": str(row.get("maxFundingRate") or ""),
                }
                events.append(
                    RestEvent(
                        channel="rest_funding",
                        symbol=symbol,
                        exchange_timestamp_ms=exchange_ts,
                        received_timestamp_utc_ns=received_ns,
                        event_key=(
                            f"rest_funding:{exchange_ts}:"
                            f"{_payload_sha(payload)[:16]}"
                        ),
                        payload=payload,
                    )
                )
        except Exception as exc:
            quality.append(_rest_error("rest_funding", "", exc))
        if include_rules:
            try:
                response, received_ns = self._get(
                    "/api/v2/mix/market/contracts",
                    {"productType": "usdt-futures"},
                )
                rule_rows = {
                    str(row.get("symbol") or "").upper(): row
                    for row in response["data"]
                    if isinstance(row, Mapping)
                }
                for symbol in self.profile.symbols:
                    row = rule_rows.get(symbol)
                    if row is None:
                        quality.append(
                            _rest_error(
                                "rest_instrument_rules",
                                symbol,
                                ValueError("instrument rules missing"),
                            )
                        )
                        continue
                    exchange_ts = int(response["requestTime"])
                    payload = {
                        "source_endpoint": "/api/v2/mix/market/contracts",
                        "request_time_ms": exchange_ts,
                        "symbol_status": str(row.get("symbolStatus") or ""),
                        "min_trade_base": str(row.get("minTradeNum") or ""),
                        "min_trade_usdt": str(row.get("minTradeUSDT") or ""),
                        "size_multiplier": str(row.get("sizeMultiplier") or ""),
                        "price_place": str(row.get("pricePlace") or ""),
                        "price_end_step": str(row.get("priceEndStep") or ""),
                        "volume_place": str(row.get("volumePlace") or ""),
                        "maker_fee_rate": str(row.get("makerFeeRate") or ""),
                        "taker_fee_rate": str(row.get("takerFeeRate") or ""),
                        "max_market_order_qty": str(
                            row.get("maxMarketOrderQty") or ""
                        ),
                        "max_order_qty": str(row.get("maxOrderQty") or ""),
                        "funding_interval_hours": str(row.get("fundInterval") or ""),
                        "maintain_time_ms": str(row.get("maintainTime") or ""),
                    }
                    events.append(
                        RestEvent(
                            channel="rest_instrument_rules",
                            symbol=symbol,
                            exchange_timestamp_ms=exchange_ts,
                            received_timestamp_utc_ns=received_ns,
                            event_key=(
                                f"rest_instrument_rules:{exchange_ts}:"
                                f"{_payload_sha(payload)[:16]}"
                            ),
                            payload=payload,
                        )
                    )
            except Exception as exc:
                quality.append(_rest_error("rest_instrument_rules", "", exc))
        return events, quality

    def _get(
        self,
        endpoint: str,
        params: Mapping[str, str],
    ) -> tuple[dict[str, Any], int]:
        response = self.session.get(
            self.profile.rest_base_url + endpoint,
            params=dict(params),
            timeout=self.timeout_seconds,
        )
        received_ns = time.time_ns()
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("Bitget REST response is not an object")
        if str(payload.get("code") or "") != "00000":
            raise ValueError(
                f"Bitget REST error {payload.get('code')}: {payload.get('msg')}"
            )
        request_time = _positive_int(payload.get("requestTime"))
        if request_time <= 0:
            raise ValueError("Bitget REST requestTime missing")
        return dict(payload), received_ns


def _select_candle(data: Any, start_ms: int) -> Sequence[Any] | None:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        return None
    return next(
        (
            row
            for row in data
            if isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) >= 6
            and _positive_int(row[0]) == start_ms
        ),
        None,
    )


def _rest_error(channel: str, symbol: str, exc: Exception) -> dict[str, Any]:
    return {
        "kind": "rest_reconciliation_error",
        "channel": channel,
        "symbol": symbol,
        "details": {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        },
    }


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
