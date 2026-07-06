"""Bitget public funding and open-interest fetcher for v1 agents.

The legacy agents read market derivatives context through an object with
``get(sym)`` and ``get_global()`` methods.  MEXC already has that adapter; this
module provides the same small contract for Bitget futures.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence

import ccxt  # type: ignore


log = logging.getLogger("bitget_funding")

REFRESH_SEC = 60


def _public_swap_client() -> ccxt.bitget:
    return ccxt.bitget(
        {
            "enableRateLimit": True,
            "options": {"defaultType": "swap", "defaultSubType": "linear"},
        }
    )


def _normalize_base_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if ":" in text:
        text = text.split(":", 1)[0]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.endswith("_USDT"):
        text = text[:-5]
    if text.endswith("USDT") and len(text) > 4:
        text = text[:-4]
    return text


def _market_symbol(symbol: str) -> str:
    return f"{_normalize_base_symbol(symbol)}/USDT:USDT"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _nested(payload: dict, *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_oi_base(open_interest: dict, ticker: dict) -> float:
    candidates: list[Any] = [
        open_interest.get("openInterestAmount"),
        _nested(ticker, "info", "holdingAmount"),
        _nested(open_interest, "info", "holdingAmount"),
    ]
    oi_list = _nested(open_interest, "info", "openInterestList")
    if isinstance(oi_list, list) and oi_list:
        candidates.append((oi_list[0] or {}).get("size"))
    for value in candidates:
        parsed = _to_float(value)
        if parsed > 0:
            return parsed
    return 0.0


class BitgetFundingDataFetcher:
    """Small polling cache for Bitget swap funding and open interest."""

    def __init__(
        self,
        symbols: Sequence[str] | None = None,
        refresh_sec: int = REFRESH_SEC,
        exchange: Any | None = None,
    ) -> None:
        self.symbols = list(symbols or [])
        self.refresh_sec = int(refresh_sec)
        self.exchange = exchange or _public_swap_client()
        self._cache: Dict[str, dict] = {}
        self._global: dict = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update = 0.0
        self._markets_loaded = False

    def get(self, sym: str) -> dict:
        return dict(self._cache.get(_normalize_base_symbol(sym), {}))

    def get_all(self) -> Dict[str, dict]:
        return dict(self._cache)

    def get_global(self) -> dict:
        return dict(self._global)

    def funding_rate(self, sym: str) -> float:
        return _to_float(self._cache.get(_normalize_base_symbol(sym), {}).get("funding_rate"))

    def open_interest(self, sym: str) -> float:
        return _to_float(self._cache.get(_normalize_base_symbol(sym), {}).get("open_interest_usdt"))

    def fetch_all(self, symbols: Sequence[str] | None = None) -> Dict[str, dict]:
        syms = list(symbols or self.symbols)
        if not syms:
            return {}
        self._load_markets()

        new_cache: Dict[str, dict] = {}
        for raw_sym in syms:
            sym = _normalize_base_symbol(raw_sym)
            if not sym:
                continue
            try:
                data = self._fetch_one(sym)
                if data:
                    new_cache[sym] = data
            except Exception as exc:
                log.debug("[bitget funding] fetch %s: %s", sym, exc)

        self._cache = new_cache
        self._last_update = time.time()
        self._global = self._build_global(new_cache)
        return dict(new_cache)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="bitget_funding_fetcher",
        )
        self._thread.start()
        log.info("[bitget funding] background refresh started (%ss)", self.refresh_sec)

    def stop(self) -> None:
        self._running = False

    def _load_markets(self) -> None:
        if self._markets_loaded:
            return
        loader = getattr(self.exchange, "load_markets", None)
        if callable(loader):
            loader()
        self._markets_loaded = True

    def _fetch_one(self, sym: str) -> dict:
        market = _market_symbol(sym)
        ticker: dict = {}
        funding: dict = {}
        open_interest: dict = {}

        try:
            ticker = self.exchange.fetch_ticker(market) or {}
        except Exception as exc:
            log.debug("[bitget funding] ticker %s: %s", market, exc)
        try:
            funding = self.exchange.fetch_funding_rate(market) or {}
        except Exception as exc:
            log.debug("[bitget funding] funding %s: %s", market, exc)
        try:
            open_interest = self.exchange.fetch_open_interest(market) or {}
        except Exception as exc:
            log.debug("[bitget funding] open interest %s: %s", market, exc)

        last_price = _to_float(
            ticker.get("last")
            or ticker.get("markPrice")
            or _nested(ticker, "info", "lastPr")
            or _nested(ticker, "info", "markPrice")
        )
        mark_price = _to_float(
            ticker.get("markPrice")
            or funding.get("markPrice")
            or _nested(ticker, "info", "markPrice")
        )
        index_price = _to_float(ticker.get("indexPrice") or funding.get("indexPrice"))
        price_for_oi = mark_price or last_price

        oi_base = _extract_oi_base(open_interest, ticker)
        oi_usdt = _to_float(open_interest.get("openInterestValue"))
        if oi_usdt <= 0 and oi_base > 0 and price_for_oi > 0:
            oi_usdt = oi_base * price_for_oi

        funding_ts = int(
            _to_float(
                funding.get("fundingTimestamp")
                or funding.get("nextFundingTimestamp")
                or _nested(funding, "info", "nextUpdate")
            )
        )
        next_funding_sec = 0
        if funding_ts > 0:
            next_funding_sec = max(0, int((funding_ts / 1000.0) - time.time()))

        result = {
            "symbol": sym,
            "funding_rate": _to_float(
                funding.get("fundingRate")
                or _nested(funding, "info", "fundingRate")
                or _nested(ticker, "info", "fundingRate")
            ),
            "next_funding_ts": funding_ts,
            "next_funding_sec": next_funding_sec,
            "last_price": last_price,
            "mark_price": mark_price,
            "index_price": index_price,
            "volume_24h_usdt": _to_float(
                ticker.get("quoteVolume") or _nested(ticker, "info", "usdtVolume")
            ),
            "volume_24h_base": _to_float(ticker.get("baseVolume")),
            "open_interest_base": oi_base,
            "open_interest_usdt": oi_usdt,
            "updated_ts": time.time(),
        }
        if not (
            result["funding_rate"]
            or result["open_interest_usdt"]
            or result["volume_24h_usdt"]
            or result["last_price"]
        ):
            return {}
        return result

    @staticmethod
    def _build_global(cache: Dict[str, dict]) -> dict:
        if not cache:
            return {"timestamp": time.time(), "total_oi_usdt": 0.0, "avg_funding": 0.0}
        rates = [_to_float(d.get("funding_rate")) for d in cache.values()]
        ois = [_to_float(d.get("open_interest_usdt")) for d in cache.values()]
        return {
            "avg_funding": float(sum(rates) / len(rates)) if rates else 0.0,
            "max_funding": float(max(rates, key=abs)) if rates else 0.0,
            "total_oi_usdt": float(sum(ois)) if ois else 0.0,
            "n_positive_funding": sum(1 for rate in rates if rate > 0),
            "n_negative_funding": sum(1 for rate in rates if rate < 0),
            "timestamp": time.time(),
        }

    def _loop(self) -> None:
        while self._running:
            try:
                self.fetch_all()
                log.info(
                    "[bitget funding] updated %d symbols | total_oi=%.0f avg_rate=%.4f%%",
                    len(self._cache),
                    self._global.get("total_oi_usdt", 0.0),
                    self._global.get("avg_funding", 0.0) * 100,
                )
            except Exception as exc:
                log.warning("[bitget funding] refresh failed: %s", exc)
            time.sleep(max(1, self.refresh_sec))
