"""Bitget live futures adapter for Panteon v2."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .v1_futures_adapter import V1FuturesExchangeAdapter, load_runtime_leverage


class BitgetExchangeAdapter(V1FuturesExchangeAdapter):
    """Wrap the existing v1 Bitget futures client as a v2 Exchange."""

    def __init__(
        self,
        *,
        order_client: Optional[Any] = None,
        read_client: Optional[Any] = None,
        leverage: Optional[int] = None,
        demo: Optional[bool] = None,
    ) -> None:
        if demo is None:
            demo = os.getenv("BITGET_TRADING_MODE", "live_futures").strip().lower() == "demo_futures"
        self.demo = bool(demo)
        if order_client is None:
            prefix = "BITGET_DEMO" if self.demo else "BITGET"
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            api_secret = os.getenv(f"{prefix}_SECRET_KEY", "")
            api_passphrase = os.getenv(f"{prefix}_PASSPHRASE", "")
            if not api_key or not api_secret or not api_passphrase:
                raise RuntimeError(
                    f"{prefix}_API_KEY, {prefix}_SECRET_KEY and "
                    f"{prefix}_PASSPHRASE are required"
                )
            from bitget_connector import BitgetFuturesClient  # type: ignore

            order_client = BitgetFuturesClient(
                api_key,
                api_secret,
                api_passphrase,
                demo=self.demo,
            )

        if read_client is None:
            try:
                prefix = "BITGET_DEMO" if self.demo else "BITGET"
                api_key = os.getenv(f"{prefix}_API_KEY", "")
                api_secret = os.getenv(f"{prefix}_SECRET_KEY", "")
                api_passphrase = os.getenv(f"{prefix}_PASSPHRASE", "")
                from bitget_api import BitgetDirectClient  # type: ignore

                read_client = BitgetDirectClient(
                    api_key,
                    api_secret,
                    api_passphrase,
                    demo=self.demo,
                )
            except Exception:
                read_client = order_client

        super().__init__(
            name="BITGET",
            order_client=order_client,
            read_client=read_client,
            leverage=leverage if leverage is not None else load_runtime_leverage("BITGET"),
            default_min_notional=5.10,
            default_fee_rate=0.0006,
            close_via_place_order=False,
        )

    def get_policy_market_frame(
        self,
        *,
        symbols: Sequence[str],
        bar_interval_seconds: int,
        bar_close_timestamp: datetime,
        max_notional_usd: float,
    ) -> dict[str, Any]:
        """Read one exact closed swap bar plus current executable book quality.

        This is a public-data operation. It never creates, changes or cancels
        an order. The returned frame is complete only when every manifest
        symbol has the exact expected candle and enough visible depth for the
        manifest maximum notional in both directions.
        """

        interval = int(bar_interval_seconds)
        timeframe = _POLICY_TIMEFRAMES.get(interval)
        if timeframe is None:
            raise ValueError("unsupported Bitget policy bar interval")
        if bar_close_timestamp.tzinfo is None:
            raise ValueError("bar_close_timestamp must be timezone-aware")
        boundary = bar_close_timestamp.astimezone(timezone.utc)
        expected_start_ms = int(boundary.timestamp() * 1000) - interval * 1000
        source = self._policy_public_swap_client()
        rows: dict[str, dict[str, Any]] = {}

        for raw_symbol in symbols:
            symbol = self._normalize_symbol(raw_symbol)
            market_symbol = self._policy_market_symbol(symbol)
            try:
                ticker = source.fetch_ticker(market_symbol) or {}
                order_book = source.fetch_order_book(market_symbol, limit=20) or {}
                bids = _book_levels(order_book.get("bids"))
                asks = _book_levels(order_book.get("asks"))
                bid = bids[0][0] if bids else _finite_positive(ticker.get("bid"))
                ask = asks[0][0] if asks else _finite_positive(ticker.get("ask"))
                if bid <= 0.0 or ask <= 0.0 or ask < bid:
                    raise ValueError("invalid top of book")
                midpoint = (bid + ask) / 2.0
                decision_price = (
                    _finite_positive(ticker.get("last"))
                    or _finite_positive(ticker.get("close"))
                    or midpoint
                )
                candles = source.fetch_ohlcv(
                    market_symbol,
                    timeframe=timeframe,
                    since=expected_start_ms,
                    limit=2,
                ) or []
                candle = next(
                    (
                        item
                        for item in candles
                        if isinstance(item, (list, tuple))
                        and len(item) >= 6
                        and int(float(item[0])) == expected_start_ms
                    ),
                    None,
                )
                if candle is None:
                    raise ValueError("exact closed candle unavailable")
                open_price = _finite_positive(candle[1])
                high = _finite_positive(candle[2])
                low = _finite_positive(candle[3])
                close = _finite_positive(candle[4])
                volume = _finite_nonnegative(candle[5])
                if not (
                    open_price > 0.0
                    and high >= low > 0.0
                    and low <= open_price <= high
                    and low <= close <= high
                ):
                    raise ValueError("closed candle is invalid")
                notional = max(0.0, float(max_notional_usd))
                buy_impact = _book_impact_bps(asks, notional, midpoint, is_buy=True)
                sell_impact = _book_impact_bps(bids, notional, midpoint, is_buy=False)
                if not math.isfinite(buy_impact) or not math.isfinite(sell_impact):
                    raise ValueError("insufficient order-book depth")
                spread_bps = (ask - bid) / midpoint * 10_000.0
                # Manifest slippage is round-trip, so convert the worst
                # one-side visible-book impact to the same unit.
                estimated_slippage_bps = 2.0 * max(buy_impact, sell_impact)
                rows[symbol] = {
                    "market_symbol": market_symbol,
                    "candle_timestamp_ms": expected_start_ms,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "decision_price": decision_price,
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": spread_bps,
                    "estimated_slippage_bps": estimated_slippage_bps,
                    "complete": True,
                }
            except Exception as exc:
                rows[symbol] = {
                    "market_symbol": market_symbol,
                    "complete": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        incomplete = [
            symbol for symbol, row in rows.items() if not bool(row.get("complete"))
        ]
        return {
            "source": "BITGET_PUBLIC_SWAP",
            "observed_at": datetime.now(timezone.utc),
            "bar_close_timestamp": boundary,
            "bar_interval_seconds": interval,
            "symbols": rows,
            "complete": not incomplete and len(rows) == len(tuple(symbols)),
            "reason": (
                "incomplete_symbols:" + ",".join(incomplete)
                if incomplete
                else ""
            ),
        }

    def get_policy_warmup_history(
        self,
        *,
        symbols: Sequence[str],
        bar_interval_seconds: int,
        required_bars: int,
    ) -> dict[str, Any]:
        """Read an exact, synchronized closed-bar warmup window.

        CarryFlow state is cadence-sensitive. Downsampling the bridge's short
        1-minute cache can leave an hourly policy cold for days after every
        restart, so live startup requires this direct public OHLCV window.
        """

        interval = int(bar_interval_seconds)
        timeframe = _POLICY_TIMEFRAMES.get(interval)
        if timeframe is None:
            raise ValueError("unsupported Bitget policy bar interval")
        limit = int(required_bars)
        if limit <= 0 or limit > 500:
            raise ValueError("required_bars must be in [1, 500]")

        now = datetime.now(timezone.utc)
        boundary_epoch = int(now.timestamp())
        boundary_epoch -= boundary_epoch % interval
        boundary = datetime.fromtimestamp(boundary_epoch, tz=timezone.utc)
        first_start_ms = int(boundary.timestamp() * 1000) - limit * interval * 1000
        expected_timestamps = tuple(
            first_start_ms + index * interval * 1000 for index in range(limit)
        )
        source = self._policy_public_swap_client()
        by_symbol: dict[str, dict[int, tuple[float, float]]] = {}
        errors: dict[str, str] = {}

        for raw_symbol in symbols:
            symbol = self._normalize_symbol(raw_symbol)
            market_symbol = self._policy_market_symbol(symbol)
            try:
                candles = source.fetch_ohlcv(
                    market_symbol,
                    timeframe=timeframe,
                    since=first_start_ms,
                    limit=limit,
                ) or []
                parsed: dict[int, tuple[float, float]] = {}
                for candle in candles:
                    if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                        continue
                    timestamp_ms = int(float(candle[0]))
                    close = _finite_positive(candle[4])
                    volume = _finite_nonnegative(candle[5])
                    if timestamp_ms in expected_timestamps and close > 0.0:
                        parsed[timestamp_ms] = (close, volume)
                missing = [ts for ts in expected_timestamps if ts not in parsed]
                if missing:
                    raise ValueError(
                        f"closed warmup window missing {len(missing)}/{limit} bars"
                    )
                by_symbol[symbol] = parsed
            except Exception as exc:
                errors[symbol] = f"{type(exc).__name__}: {exc}"

        normalized_symbols = tuple(self._normalize_symbol(item) for item in symbols)
        if errors or len(by_symbol) != len(normalized_symbols):
            return {
                "source": "BITGET_PUBLIC_SWAP",
                "observed_at": datetime.now(timezone.utc),
                "bar_close_timestamp": boundary,
                "bar_interval_seconds": interval,
                "required_bars": limit,
                "prices": [],
                "volumes": [],
                "complete": False,
                "reason": "warmup_incomplete",
                "errors": errors,
            }

        prices: list[dict[str, float]] = []
        volumes: list[dict[str, float]] = []
        for timestamp_ms in expected_timestamps:
            prices.append(
                {
                    symbol: by_symbol[symbol][timestamp_ms][0]
                    for symbol in normalized_symbols
                }
            )
            volumes.append(
                {
                    symbol: by_symbol[symbol][timestamp_ms][1]
                    for symbol in normalized_symbols
                }
            )
        return {
            "source": "BITGET_PUBLIC_SWAP",
            "observed_at": datetime.now(timezone.utc),
            "bar_close_timestamp": boundary,
            "bar_interval_seconds": interval,
            "required_bars": limit,
            "first_candle_timestamp_ms": expected_timestamps[0],
            "last_candle_timestamp_ms": expected_timestamps[-1],
            "prices": prices,
            "volumes": volumes,
            "complete": True,
            "reason": "",
            "errors": {},
        }

    def _policy_public_swap_client(self) -> Any:
        candidates = (
            getattr(self._read_client, "swap", None),
            getattr(self._order_client, "exchange", None),
        )
        for candidate in candidates:
            if (
                candidate is not None
                and callable(getattr(candidate, "fetch_ticker", None))
                and callable(getattr(candidate, "fetch_order_book", None))
                and callable(getattr(candidate, "fetch_ohlcv", None))
            ):
                return candidate
        raise RuntimeError("Bitget public swap market client is unavailable")

    def _policy_market_symbol(self, symbol: str) -> str:
        mapper = getattr(self._order_client, "_market_symbol", None)
        if callable(mapper):
            return str(mapper(symbol))
        return f"{symbol}/USDT:USDT"


_POLICY_TIMEFRAMES: Mapping[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    14_400: "4h",
    86_400: "1d",
}


def _finite_positive(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0


def _finite_nonnegative(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else 0.0


def _book_levels(raw: Any) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for item in raw or ():
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        price = _finite_positive(item[0])
        qty = _finite_positive(item[1])
        if price > 0.0 and qty > 0.0:
            levels.append((price, qty))
    return levels


def _book_impact_bps(
    levels: Sequence[tuple[float, float]],
    notional_usd: float,
    midpoint: float,
    *,
    is_buy: bool,
) -> float:
    if notional_usd <= 0.0 or midpoint <= 0.0:
        return 0.0
    remaining = float(notional_usd)
    acquired_qty = 0.0
    spent = 0.0
    for price, available_qty in levels:
        level_notional = price * available_qty
        take_notional = min(remaining, level_notional)
        if take_notional <= 0.0:
            continue
        acquired_qty += take_notional / price
        spent += take_notional
        remaining -= take_notional
        if remaining <= 1e-9:
            break
    if remaining > 1e-6 or acquired_qty <= 0.0:
        return float("inf")
    average_price = spent / acquired_qty
    signed = (
        (average_price / midpoint - 1.0)
        if is_buy
        else (1.0 - average_price / midpoint)
    )
    return max(0.0, signed * 10_000.0)
