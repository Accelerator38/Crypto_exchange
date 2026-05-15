"""Bitget connector compatible with the existing combo/live bridge."""

from __future__ import annotations

import logging
import math
import os
import json
import re
import sys
import time
from datetime import datetime
from typing import Dict, List

from project_paths import PROJECT_ROOT, add_runtime_paths


def _bootstrap_venv_packages():
    add_runtime_paths()


_bootstrap_venv_packages()

import ccxt  # type: ignore

import mexc_connector as mexc
from bitget_api import BitgetDirectClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bitget_connector")

_load_settings = mexc._load_settings
_parse_settings = mexc._parse_settings
PaperPortfolio = mexc.PaperPortfolio
plot_dashboard = mexc.plot_dashboard
plot_compound_growth = mexc.plot_compound_growth
plot_usd_dashboard_mexc = mexc.plot_usd_dashboard_mexc
save_history_csv = mexc.save_history_csv
save_stats_json = mexc.save_stats_json
ACTION_NAMES = mexc.ACTION_NAMES
MIN_ORDER_USDT = mexc.MIN_ORDER_USDT
TOP_N_SYMBOLS = mexc.TOP_N_SYMBOLS
WARMUP_BARS = mexc.WARMUP_BARS
_BITGET_SETTINGS_RAW = _load_settings()


def _bitget_setting_float(name: str, default: float) -> float:
    for key in (f"bitget_{name}", f"{name}_bitget", name):
        if key in _BITGET_SETTINGS_RAW:
            try:
                return float(_BITGET_SETTINGS_RAW[key])
            except (TypeError, ValueError):
                return float(default)
    return float(default)


def _bitget_setting_int(name: str, default: int) -> int:
    return int(_bitget_setting_float(name, default))


def _bitget_setting_symbols(name: str) -> set[str]:
    raw = _BITGET_SETTINGS_RAW.get(f"bitget_{name}") or _BITGET_SETTINGS_RAW.get(f"{name}_bitget") or ""
    return {s.strip().upper() for s in str(raw).split(",") if s.strip()}


BITGET_SYMBOL_BLOCKLIST = frozenset(_bitget_setting_symbols("symbol_blocklist"))
BITGET_MAX_LIVE_POSITIONS_NEGATIVE_PNL = _bitget_setting_int("max_live_positions_negative_pnl", 3)

API_KEY = os.getenv("BITGET_API_KEY", "")
API_SECRET = os.getenv("BITGET_SECRET_KEY", "")
API_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
TRADING_MODE = os.getenv("BITGET_TRADING_MODE", "live_futures")


def _public_client(default_type: str = "spot") -> ccxt.bitget:
    options = {"defaultType": default_type, "defaultSubType": "linear"}
    return ccxt.bitget({"enableRateLimit": True, "options": options})


def _private_client(default_type: str, api_key: str, api_secret: str, api_passphrase: str) -> ccxt.bitget:
    if api_key and api_secret and not api_passphrase:
        raise ValueError("Bitget requires BITGET_PASSPHRASE for authenticated API access.")
    options = {"defaultType": default_type, "defaultSubType": "linear"}
    return ccxt.bitget(
        {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,
            "options": options,
        }
    )


def _normalize_kline_interval(interval: str) -> str:
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    return mapping.get(interval, "1m")


def _interval_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "60m": 3_600_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(interval, 60_000)


def fetch_klines(symbol: str, interval: str = "1m", n_bars: int = 5760) -> List[dict]:
    exchange = _public_client("spot")
    unified_symbol = f"{symbol.replace('USDT', '')}/USDT"
    timeframe = _normalize_kline_interval(interval)
    step_ms = _interval_ms(interval)
    now_ms = int(time.time() * 1000)
    since = now_ms - n_bars * step_ms
    rows: List[List[float]] = []
    limit = 200
    max_retries = 3
    while len(rows) < n_bars:
        batch = None
        for attempt in range(max_retries):
            try:
                batch = exchange.fetch_ohlcv(
                    unified_symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=min(limit, n_bars - len(rows)),
                )
                break
            except Exception as e:
                err_name = type(e).__name__
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    log.warning(
                        "[klines] %s %s попытка %d/%d: %s — повтор через %dс",
                        symbol, interval, attempt + 1, max_retries, err_name, wait,
                    )
                    time.sleep(wait)
                else:
                    log.warning(
                        "[klines] %s %s — все %d попытки исчерпаны (%s), пропускаю",
                        symbol, interval, max_retries, err_name,
                    )
                    return []
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + step_ms
        if len(batch) < min(limit, n_bars - len(rows) + len(batch)):
            break
        if len(batch) == 1:
            break
    result = []
    for row in rows[:n_bars]:
        if len(row) >= 6:
            result.append({"close": float(row[4]), "volume": float(row[5])})
    return result


def build_warmup_data(symbols: List[str], interval: str, n_bars: int):
    log.info(
        "[warmup] Загружаю %d баров истории для %d символов Bitget (interval=%s)...",
        n_bars,
        len(symbols),
        interval,
    )
    prices_per_sym: Dict[str, List[float]] = {}
    volumes_per_sym: Dict[str, List[float]] = {}
    for i, sym in enumerate(symbols):
        try:
            klines = fetch_klines(f"{sym}USDT", interval, n_bars)
        except Exception as e:
            log.warning("[warmup] Ошибка загрузки %s: %s — пропускаю", sym, e)
            continue
        if not klines:
            log.warning("[warmup] Нет данных для %s — пропускаю", sym)
            continue
        prices_per_sym[sym] = [k["close"] for k in klines]
        volumes_per_sym[sym] = [k["volume"] for k in klines]
        if (i + 1) % 5 == 0 or i == len(symbols) - 1:
            log.info(
                "[warmup]   Загружено %d/%d символов (%s: %d баров)",
                i + 1,
                len(symbols),
                sym,
                len(klines),
            )
    if not prices_per_sym:
        return [], []
    min_len = min(len(v) for v in prices_per_sym.values())
    prices_bars = []
    volumes_bars = []
    for idx in range(min_len):
        prices_bars.append({s: prices_per_sym[s][idx] for s in prices_per_sym})
        volumes_bars.append({s: volumes_per_sym[s][idx] for s in volumes_per_sym})
    log.info("[warmup] Готово: %d баров × %d символов", min_len, len(prices_per_sym))
    return prices_bars, volumes_bars


def fetch_top_symbols(n: int = TOP_N_SYMBOLS) -> List[str]:
    exchange = _public_client("spot")
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    stables = {"USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP"}
    exclude_suffixes = ("UP", "DOWN", "BULL", "BEAR", "LONG", "SHORT", "3L", "3S", "5L", "5S")
    candidates = []
    for symbol, market in markets.items():
        if not market.get("spot") or market.get("quote") != "USDT" or not market.get("active", True):
            continue
        base = str(market.get("base") or "").upper()
        if not base or base in stables or any(base.endswith(sfx) for sfx in exclude_suffixes):
            continue
        ticker = tickers.get(symbol, {}) or {}
        quote_vol = float(
            ticker.get("quoteVolume")
            or ((ticker.get("baseVolume") or 0.0) * (ticker.get("last") or 0.0))
            or 0.0
        )
        if quote_vol > 0:
            candidates.append((base, quote_vol))
    candidates.sort(key=lambda item: item[1], reverse=True)
    top = [base for base, _ in candidates[:n]]
    log.info("[symbols] Топ-%d Bitget по объёму: %s", len(top), top)
    return top


class BitgetSpotClient:
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        self.exchange = _private_client("spot", api_key, api_secret, api_passphrase)
        self.exchange.load_markets()

    def _symbol(self, base: str) -> str:
        return f"{base}/USDT"

    def account_balance(self):
        bal = self.exchange.fetch_balance({"type": "spot"})
        result = {}
        totals = bal.get("total") or {}
        free = bal.get("free") or {}
        for asset, total in totals.items():
            total_f = float(total or 0.0)
            if total_f > 0:
                result[asset.upper()] = float(free.get(asset, total_f) or 0.0)
        return result

    def buy_usdt(self, sym: str, usdt: float):
        market = self._symbol(sym)
        ticker = self.exchange.fetch_ticker(market)
        last = float(ticker.get("last", 0.0) or 0.0)
        if last <= 0:
            raise ValueError(f"ticker price is zero for {market}")
        qty = max(usdt / last, 0.0)
        return self.exchange.create_order(market, "market", "buy", qty)

    def sell_qty(self, sym: str, qty: float):
        return self.exchange.create_order(self._symbol(sym), "market", "sell", qty)


class BitgetFuturesClient:
    _FUTURES_BLACKLIST = BITGET_SYMBOL_BLOCKLIST
    _POSITION_MODE_CACHE_TTL = 60.0

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str, testnet: bool = False):
        del testnet
        self.exchange = _private_client("swap", api_key, api_secret, api_passphrase)
        self.exchange.load_markets()
        self._contract_meta_cache: dict = {}
        self._insufficient_symbol_until: dict = {}
        self._position_mode_cache: str | None = None
        self._position_mode_checked_at = 0.0
        self._open_position_symbols: set[str] = set()
        self._open_positions_checked_at = 0.0
        self._close_skip_until: dict[str, float] = {}
        self._close_rate_limited_until = 0.0
        self._last_data_error_at = 0.0
        self._last_data_error_reason = ""
        # Runtime-блэклист: символы которые вернули BadSymbol/NotFound.
        # Попавший сюда символ исключается из всех последующих запросов
        # до рестарта бота. Это защищает от делистнутых контрактов.
        self._bad_symbols: set[str] = set()

    def _mark_data_error(self, reason: str) -> None:
        self._last_data_error_at = time.time()
        self._last_data_error_reason = str(reason or "data_error")

    def recent_data_error(self, max_age_sec: float = 120.0) -> str:
        if self._last_data_error_at <= 0:
            return ""
        if time.time() - float(self._last_data_error_at) <= float(max_age_sec or 120.0):
            return self._last_data_error_reason or "data_error"
        return ""

    def _market_symbol(self, base_sym: str) -> str:
        return f"{base_sym}/USDT:USDT"

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        sym = str(symbol or "").upper()
        if ":" in sym:
            sym = sym.split(":", 1)[0]
        if "/" in sym:
            sym = sym.split("/", 1)[0]
        if sym.endswith("_USDT"):
            sym = sym[:-5]
        if sym.endswith("USDT") and len(sym) > 4:
            sym = sym[:-4]
        return sym

    def ticker_price(self, symbol: str):
        """
        Получить последнюю цену фьючерсного контракта.

        Возвращает 0.0 и добавляет символ в runtime-блэклист при:
          - BadSymbol (контракт не существует или делистнут),
          - NotSupported (биржа не даёт тикер для этого контракта).
        Это позволяет вызывающему коду (_fetch_market) продолжить работу
        с остальными символами, вместо падения всего цикла.
        """
        base = symbol.replace("_USDT", "").replace("USDT", "")
        if base in self._bad_symbols:
            return 0.0
        try:
            ticker = self.exchange.fetch_ticker(self._market_symbol(base))
            return float(ticker.get("last", 0.0) or 0.0)
        except (ccxt.BadSymbol, ccxt.NotSupported) as e:
            # Делистнутый или несуществующий контракт — запоминаем и не трогаем больше
            self._bad_symbols.add(base)
            log.warning(
                "[bitget] ticker %s → %s: контракт помечен как bad, исключаю до рестарта",
                base, type(e).__name__,
            )
            return 0.0

    def account_assets(self):
        result = {}
        balance = self.exchange.fetch_balance({"type": "swap", "productType": "USDT-FUTURES"})
        usdt = balance.get("USDT", {}) or {}
        total = float(usdt.get("total", 0.0) or 0.0)
        free = float(usdt.get("free", 0.0) or 0.0)
        if total > 0 or free > 0:
            result["USDT"] = total
            result["USDT_AVAIL"] = free
        return result

    def is_symbol_on_margin_cooldown(self, symbol: str) -> bool:
        return time.time() < float(self._insufficient_symbol_until.get(symbol, 0) or 0)

    def margin_cooldown_remaining(self, symbol: str) -> int:
        remaining = float(self._insufficient_symbol_until.get(symbol, 0) or 0) - time.time()
        return max(0, int(math.ceil(remaining)))

    def _get_open_position_symbols(self, force: bool = False) -> set[str]:
        now = time.time()
        if (not force) and (now - self._open_positions_checked_at) < 2.0:
            return set(self._open_position_symbols)
        try:
            positions = self.exchange.fetch_positions(
                params={"productType": "USDT-FUTURES", "marginCoin": "USDT"}
            )
            open_symbols = set()
            for pos in positions:
                contracts = float(pos.get("contracts", 0.0) or 0.0)
                if contracts <= 0:
                    continue
                base = self._normalize_symbol(pos.get("symbol", ""))
                if base:
                    open_symbols.add(base)
            self._open_position_symbols = open_symbols
            self._open_positions_checked_at = now
        except Exception as e:
            self._mark_data_error("positions_fetch_failed")
            log.debug("BITGET positions snapshot failed: %s", e)
        return set(self._open_position_symbols)

    def _get_contract_meta(self, symbol: str) -> dict:
        cached = self._contract_meta_cache.get(symbol)
        if cached is not None:
            return cached
        market = self.exchange.market(self._market_symbol(symbol))
        info = market.get("info", {}) or {}
        size_multiplier = float(info.get("sizeMultiplier") or 1.0)
        native_contract_size = float(market.get("contractSize", 1.0) or 1.0)
        min_trade_num = float(
            info.get("minTradeNum")
            or (market.get("limits", {}).get("amount", {}) or {}).get("min")
            or size_multiplier
            or 1.0
        )
        max_trade_amount = float(
            info.get("maxTradeAmount")
            or (market.get("limits", {}).get("amount", {}) or {}).get("max")
            or 0.0
        )
        min_vol = max(1, int(math.ceil(min_trade_num / max(size_multiplier, 1e-12))))
        max_vol = 10_000_000
        if max_trade_amount > 0:
            max_vol = max(min_vol, int(math.floor(max_trade_amount / max(size_multiplier, 1e-12))))
        meta = {
            "symbol": market.get("id", symbol),
            "contractSize": size_multiplier,
            "nativeContractSize": native_contract_size,
            "amountStep": size_multiplier,
            "minVol": min_vol,
            "volUnit": 1,
            "maxVol": max_vol,
            "takerFeeRate": float(market.get("taker", 0.0006) or 0.0006),
            "apiAllowed": bool(market.get("active", True)),
            "state": 0 if market.get("active", True) else 1,
            "metadataFallback": False,
            "metadataSource": "exchange",
        }
        self._contract_meta_cache[symbol] = meta
        return meta

    def _get_position_mode(self, symbol: str, force: bool = False) -> str:
        now = time.time()
        cached = self._position_mode_cache
        if (
            (not force)
            and cached in ("hedge_mode", "one_way_mode")
            and (now - self._position_mode_checked_at) < self._POSITION_MODE_CACHE_TTL
        ):
            return cached
        market = self.exchange.market(self._market_symbol(symbol))
        try:
            response = self.exchange.privateMixGetV2MixAccountAccount(
                {
                    "symbol": market["id"],
                    "marginCoin": market.get("settleId") or "USDT",
                    "productType": "USDT-FUTURES",
                }
            )
            data = response.get("data") or {}
            pos_mode = str(data.get("posMode") or "").lower()
            if pos_mode not in ("hedge_mode", "one_way_mode"):
                pos_mode = cached or "one_way_mode"
            self._position_mode_cache = pos_mode
            self._position_mode_checked_at = now
            return pos_mode
        except Exception as e:
            fallback = cached or "one_way_mode"
            self._mark_data_error("position_mode_fetch_failed")
            log.warning(
                "  [BITGET] position mode fetch failed for %s: %s (fallback=%s)",
                symbol,
                e,
                fallback,
            )
            self._position_mode_cache = fallback
            self._position_mode_checked_at = now
            return fallback

    @staticmethod
    def _is_position_mode_mismatch(msg: str) -> bool:
        lowered = str(msg or "").lower()
        return "40774" in lowered or "unilateral position" in lowered

    def _build_open_order_params(self, position_mode: str) -> dict:
        return {
            "productType": "USDT-FUTURES",
            "marginMode": "cross",
            "hedged": position_mode == "hedge_mode",
        }

    def _submit_open_order(self, market_symbol: str, side: int, amount: float, position_mode: str) -> dict:
        return self.exchange.create_order(
            market_symbol,
            "market",
            "buy" if side == 1 else "sell",
            amount,
            None,
            self._build_open_order_params(position_mode),
        )

    @staticmethod
    def _exchange_error_payload(exc: Exception) -> dict:
        raw = str(exc)
        payload = {
            "http_status": getattr(exc, "http_status", None) or getattr(exc, "status_code", None),
            "code": getattr(exc, "code", None),
            "msg": raw,
            "body": getattr(exc, "body", None) or getattr(exc, "response", None),
            "type": type(exc).__name__,
        }
        if payload["body"] is not None and not isinstance(payload["body"], str):
            payload["body"] = str(payload["body"])
        for candidate in reversed(re.findall(r"\{.*?\}", raw)):
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                payload["body"] = parsed
                payload["code"] = parsed.get("code", payload["code"])
                payload["msg"] = parsed.get("msg") or parsed.get("message") or payload["msg"]
                break
        if payload["code"] is None:
            payload["code"] = -1
        return payload

    def _is_safe_transient_order_error(self, exc: Exception) -> bool:
        payload = self._exchange_error_payload(exc)
        text = f"{payload.get('code')} {payload.get('msg')} {payload.get('body')}".lower()
        status = payload.get("http_status")
        if status == 429:
            return True
        safe_markers = (
            "too many requests",
            "rate limit",
            "request frequency",
        )
        unsafe_unknowns = (
            "timeout",
            "timed out",
            "network",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "system busy",
            "temporarily unavailable",
            "unknown",
        )
        return any(marker in text for marker in safe_markers) and not any(marker in text for marker in unsafe_unknowns)

    def _submit_open_order_safely(self, market_symbol: str, side: int, amount: float, position_mode: str) -> dict:
        last_exc = None
        for attempt in range(2):
            try:
                return self._submit_open_order(market_symbol, side, amount, position_mode)
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and self._is_safe_transient_order_error(exc):
                    payload = self._exchange_error_payload(exc)
                    log.warning(
                        "  [BITGET] transient order error, safe retry 1x: status=%s code=%s msg=%s",
                        payload.get("http_status"),
                        payload.get("code"),
                        payload.get("msg"),
                    )
                    time.sleep(1.0)
                    continue
                raise
        raise last_exc

    # FIX A4: BITGET min notional = 5 USDT. Без этой проверки бот регулярно
    # ловит {"code":"45110","msg":"less than the minimum amount 5 USDT"} —
    # 26 раз в сессии 2026-04-25_15-24. Каждое такое событие = бот «думает»
    # что открыл позицию и обновляет внутренний _open_pos, а на бирже её нет
    # → потом этот символ «зависает» в leaderboard как открытый.
    BITGET_MIN_NOTIONAL_USDT = 5.10  # +0.10 запас

    def _last_price_quick(self, market_symbol: str) -> float:
        """Быстрая (best-effort) котировка для проверки notional."""
        try:
            ticker = self.exchange.fetch_ticker(market_symbol)
            return float(ticker.get("last", 0.0) or 0.0)
        except Exception:
            return 0.0

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        if self.is_symbol_on_margin_cooldown(symbol):
            return {
                "success": False,
                "data": None,
                "code": 2005,
                "skip": True,
                "cooldown": True,
            }
        if side not in (1, 3):
            return {"success": False, "data": None, "code": -1, "msg": f"unsupported side={side}"}
        market_symbol = self._market_symbol(symbol)
        meta = self._get_contract_meta(symbol)
        amount_step = float(meta.get("amountStep", meta.get("contractSize", 1.0)) or 1.0)
        amount = float(vol) * amount_step
        if amount <= 0:
            return {"success": False, "data": None, "code": -1, "msg": "amount <= 0"}
        # FIX A4: проверяем notional до отправки и при необходимости докручиваем
        # объём до минимального. Если докрутка > 2× (значит сигнал просил
        # очень мало) — пропускаем, чтобы не открывать позицию сильно крупнее
        # запрошенной.
        try:
            last_px = self._last_price_quick(market_symbol)
            if last_px > 0:
                notional = amount * last_px
                min_notional = float(self.BITGET_MIN_NOTIONAL_USDT)
                if notional < min_notional:
                    needed_amount = min_notional / last_px
                    # подтягиваем к шагу
                    if amount_step > 0:
                        steps = int((needed_amount / amount_step) + 0.999999)
                        adj_amount = steps * amount_step
                    else:
                        adj_amount = needed_amount
                    if adj_amount <= 0:
                        return {
                            "success": False, "data": None, "code": 45110,
                            "skip": True,
                            "msg": f"min-notional skip {symbol}: notional=${notional:.2f}<{min_notional:.2f}",
                        }
                    if adj_amount > amount * 2.0:
                        # Слишком большая корректировка — отказываемся, чтобы
                        # не нарушить sizing, который рассчитал игрок.
                        return {
                            "success": False, "data": None, "code": 45110,
                            "skip": True,
                            "msg": (
                                f"min-notional skip {symbol}: "
                                f"req=${notional:.2f}, would-need ${adj_amount*last_px:.2f} "
                                f"(>2× upscale)"
                            ),
                        }
                    log.info(
                        "  [BITGET] %s notional %.2f < %.2f USDT → bumping vol %s → %.4f",
                        symbol, notional, min_notional, vol, adj_amount,
                    )
                    amount = adj_amount
                    vol = max(int(adj_amount / max(amount_step, 1e-9) + 0.5), 1)
        except Exception as exc:
            log.debug("  [BITGET] min-notional check failed for %s: %s", symbol, exc)
        side_name = "LONG" if side == 1 else "SHORT"
        try:
            try:
                self.exchange.set_margin_mode(
                    "cross",
                    market_symbol,
                    {"productType": "USDT-FUTURES"},
                )
            except Exception:
                pass
            try:
                self.exchange.set_leverage(
                    leverage,
                    market_symbol,
                    {"productType": "USDT-FUTURES"},
                )
            except Exception:
                pass
            detected_mode = self._get_position_mode(symbol)
            modes_to_try = [detected_mode]
            attempted_modes = []
            order = None
            last_error = None
            while modes_to_try:
                position_mode = modes_to_try.pop(0)
                attempted_modes.append(position_mode)
                try:
                    order = self._submit_open_order_safely(
                        market_symbol,
                        side,
                        amount,
                        position_mode,
                    )
                    self._position_mode_cache = position_mode
                    self._position_mode_checked_at = time.time()
                    break
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    if self._is_position_mode_mismatch(msg):
                        refreshed_mode = self._get_position_mode(symbol, force=True)
                        for candidate in (
                            refreshed_mode,
                            "hedge_mode",
                            "one_way_mode",
                        ):
                            if candidate not in attempted_modes and candidate not in modes_to_try:
                                modes_to_try.append(candidate)
                        if modes_to_try:
                            log.warning(
                                "  [BITGET] retry %s %s with position_mode=%s after mismatch",
                                symbol,
                                side_name,
                                modes_to_try[0],
                            )
                            continue
                    raise
            if order is None and last_error is not None:
                raise last_error
            order_id = order.get("id") or ((order.get("info") or {}).get("orderId"))
            log.info(
                "  ✅ BITGET order OK: %s %s amount=%s vol=%s lev=%dx mode=%s → order_id=%s",
                symbol,
                side_name,
                amount,
                vol,
                leverage,
                self._position_mode_cache or "?",
                order_id or "?",
            )
            self._open_position_symbols.add(symbol)
            self._open_positions_checked_at = time.time()
            self._close_skip_until.pop(symbol, None)
            return {
                "success": True,
                "data": order,
                "code": 0,
                "amount": amount,
                "order_id": order_id,
                "position_mode": self._position_mode_cache,
            }
        except Exception as e:
            msg = str(e)
            err_payload = self._exchange_error_payload(e)
            code = err_payload.get("code", -1)
            if "insufficient" in msg.lower() or "margin" in msg.lower():
                code = 2005
                self._insufficient_symbol_until[symbol] = time.time() + 180
            log.error(
                "  ❌ BITGET order failed: %s %s amount=%s vol=%s lev=%dx mode=%s → code=%s msg=%s",
                symbol,
                side_name,
                amount,
                vol,
                leverage,
                self._position_mode_cache or "?",
                code,
                msg,
            )
            log.error(
                "  [BITGET] order failed payload: http_status=%s code=%s msg=%s body=%s type=%s",
                err_payload.get("http_status"),
                code,
                err_payload.get("msg"),
                err_payload.get("body"),
                err_payload.get("type"),
            )
            return {
                "success": False,
                "data": None,
                "code": code,
                "msg": err_payload.get("msg", msg),
                "http_status": err_payload.get("http_status"),
                "body": err_payload.get("body"),
                "error_type": err_payload.get("type"),
                "amount": amount,
            }

    def close_all(self, symbol: str):
        now = time.time()
        if now < self._close_rate_limited_until:
            retry_in = max(1, int(math.ceil(self._close_rate_limited_until - now)))
            log.info("  BITGET close_all cooldown: %s skipped for %ds after rate-limit", symbol, retry_in)
            return {"success": False, "skip": True, "code": 429, "msg": "rate-limit cooldown"}
        if now < float(self._close_skip_until.get(symbol, 0) or 0):
            return {"success": False, "skip": True, "code": 22002, "msg": "recently confirmed no position"}
        if symbol not in self._get_open_position_symbols():
            self._close_skip_until[symbol] = now + 45
            log.info("  BITGET close_all skip: %s -> no live position", symbol)
            return {"success": False, "skip": True, "code": 22002, "msg": "No position to close"}
        try:
            order = self.exchange.close_position(
                self._market_symbol(symbol),
                None,
                {"productType": "USDT-FUTURES"},
            )
            order_id = order.get("id") or ((order.get("info") or {}).get("orderId"))
            self._open_position_symbols.discard(symbol)
            self._open_positions_checked_at = time.time()
            self._close_skip_until.pop(symbol, None)
            log.info("  ✅ BITGET close_all OK: %s → order_id=%s", symbol, order_id or "?")
            return {"success": True, "data": order, "code": 0, "order_id": order_id}
        except Exception as e:
            msg = str(e)
            if "22002" in msg or "No position to close" in msg:
                self._open_position_symbols.discard(symbol)
                self._open_positions_checked_at = time.time()
                self._close_skip_until[symbol] = time.time() + 60
                log.info("  BITGET close_all skip confirmed: %s -> no exchange position", symbol)
                return {"success": False, "skip": True, "code": 22002, "msg": msg}
            if "429" in msg or "Too Many Requests" in msg:
                self._close_rate_limited_until = time.time() + 8
            log.warning("close_all %s: %s", symbol, e)
            return {"success": False, "code": -1, "msg": msg}


class AgentBitgetBridge(mexc.AgentMexcBridge):
    def __init__(
        self,
        agents: dict,
        cfg: dict,
        mode=TRADING_MODE,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
        output_dir: str = None,
        direct_client=None,
    ):
        self.agents = agents
        self.mode = mode
        self.exchange_name = "BITGET"
        self._bar = 0
        self._warmup_end = 0
        self._price_hist: List[dict] = []
        self._volume_hist: List[dict] = []
        self._direct_client = direct_client
        self._spot_exchange = _public_client("spot")
        self._spot_markets_loaded = False
        self._last_market_prices: Dict[str, float] = {}
        self._last_market_volumes: Dict[str, float] = {}
        self._last_market_snapshot_at = 0.0
        self._last_data_error_at = 0.0
        self._last_data_error_reason = ""

        self.initial_capital = cfg["initial_capital"]
        self.trade_fraction = _bitget_setting_float("trade_fraction", cfg["trade_fraction"])
        self.leverage = cfg["leverage"]
        self.poll_interval = cfg["poll_interval"]
        self.liquidity_min_adv = _bitget_setting_float("liquidity_min_adv", cfg["liquidity_min_adv"])
        self.spot_fee = cfg["spot_fee"]
        self.futures_fee = cfg["futures_fee"]
        self.slippage = cfg["slippage"]
        self._tf_kline = cfg.get("tf_kline", "1m")
        self._cfg_bar = cfg.get("bar", 60)
        self.funding = None
        self.max_live_positions_negative_pnl = BITGET_MAX_LIVE_POSITIONS_NEGATIVE_PNL

        if cfg["symbols"]:
            self.symbols = cfg["symbols"]
        else:
            self.symbols = fetch_top_symbols(TOP_N_SYMBOLS)

        if output_dir:
            self.output_dir = output_dir
        else:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base = str(PROJECT_ROOT)
            self.output_dir = os.path.join(base, "Results", "BITGET", ts)
        os.makedirs(self.output_dir, exist_ok=True)
        log.info("📁 Результаты: %s", self.output_dir)

        log_path = os.path.join(self.output_dir, "trading.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        root = logging.getLogger()
        for old in list(root.handlers):
            if isinstance(old, logging.FileHandler) and getattr(old, "baseFilename", "") == os.path.abspath(log_path):
                root.removeHandler(old)
                old.close()
        root.addHandler(fh)

        self.paper_pf: Dict[str, PaperPortfolio] = {}
        if mode == "paper":
            paper_cap = cfg.get("paper_capital", cfg["initial_capital"])
            if api_key and api_secret and api_passphrase:
                try:
                    dc = direct_client or BitgetDirectClient(api_key, api_secret, api_passphrase)
                    snap = dc.get_full_snapshot()
                    real = float(snap.get("total_equity", 0.0) or 0.0)
                    if real > 1.0:
                        paper_cap = real
                except Exception as e:
                    log.info("💰 Paper-капитал из settings: $%.2f  (авто-детект: %s)", paper_cap, e)
            self.initial_capital = paper_cap
            for name in agents:
                self.paper_pf[name] = PaperPortfolio(
                    paper_cap,
                    self.trade_fraction,
                    self.leverage,
                    self.spot_fee,
                    self.futures_fee,
                    self.slippage,
                )

        if mode == "paper":
            self.spot = None
            self.futures_client = None
            log.info("📝 Режим PAPER — виртуальный портфель на ценах Bitget.")
        elif mode == "live_spot":
            self.spot = BitgetSpotClient(api_key, api_secret, api_passphrase)
            self.futures_client = None
            log.warning("💵 BITGET LIVE SPOT — торговля на реальные деньги!")
        else:
            self.spot = None
            self.futures_client = BitgetFuturesClient(api_key, api_secret, api_passphrase)
            log.warning("🔴 BITGET LIVE FUTURES — торговля фьючерсами на реальные деньги!")
            # ── Предвалидация символов на доступность фьючерсного контракта ──
            # Отсекает "мёртвые" монеты ДО warmup-а и live-цикла, чтобы
            # BadSymbol не падал внутри _fetch_market каждую минуту.
            # (Если markets не загружены — пропускаем, валидация лениво сработает
            # в ticker_price при первой ошибке.)
            try:
                available_markets = set(self.futures_client.exchange.markets.keys())
                validated = []
                dropped = []
                for sym in self.symbols:
                    market_sym = self.futures_client._market_symbol(sym)
                    if market_sym in available_markets:
                        validated.append(sym)
                    else:
                        dropped.append(sym)
                        # Сразу в блэклист, на случай если где-то ещё попробуют
                        self.futures_client._bad_symbols.add(sym)
                if dropped:
                    log.warning(
                        "[bitget] Нет фьючерсных контрактов для %d символов — "
                        "исключаю до рестарта: %s",
                        len(dropped), ", ".join(dropped),
                    )
                if validated:
                    self.symbols = validated
            except Exception as e:
                log.debug("[bitget] Предвалидация символов пропущена: %s", e)

        log.info(
            "📋 Настройки:\n"
            "   exchange      = %s\n"
            "   symbols       = %s%s\n"
            "   trade_fraction= %.2f (%.0f%%)  leverage=%dx\n"
            "   poll_interval = %d сек  capital=%.0f USDT\n"
            "   fees          = spot=%.4f  fut=%.4f  slip=%.4f",
            self.exchange_name,
            self.symbols[:8],
            f" +{len(self.symbols)-8}" if len(self.symbols) > 8 else "",
            self.trade_fraction,
            self.trade_fraction * 100,
            self.leverage,
            self.poll_interval,
            self.initial_capital,
            self.spot_fee,
            self.futures_fee,
            self.slippage,
        )

    def warmup(self, n_bars: int = WARMUP_BARS) -> None:
        BAR = self._cfg_bar
        ACTIVE_WARMUP_BARS = max(3 * 24 * BAR, 1)
        log.info(
            "🔄 Прогрев Bitget: %d баров (≈ %d ч)  |  тихий=%d  активный=%d",
            n_bars,
            n_bars // 60,
            max(0, n_bars - ACTIVE_WARMUP_BARS),
            ACTIVE_WARMUP_BARS,
        )
        prices_bars, volumes_bars = build_warmup_data(self.symbols, self._tf_kline, n_bars)
        if not prices_bars:
            log.warning("[warmup] Нет данных — пропускаем.")
            return
        actual_bars = len(prices_bars)
        active_start = max(0, actual_bars - ACTIVE_WARMUP_BARS)
        month = datetime.utcnow().month
        for i, (prices, volumes) in enumerate(zip(prices_bars, volumes_bars)):
            self._bar += 1
            self._price_hist.append(dict(prices))
            self._volume_hist.append(dict(volumes))
            is_active = i >= active_start
            for agent_name, agent in self.agents.items():
                pv = (
                    self.paper_pf[agent_name].portfolio_value(prices)
                    if is_active and agent_name in self.paper_pf
                    else self.initial_capital
                )
                try:
                    actions = agent.act(
                        prices=prices,
                        volumes=volumes,
                        month=month,
                        portfolio_value=pv,
                        bar_index=self._bar,
                    )
                except Exception:
                    actions = {}
                if is_active and self.mode == "paper" and actions:
                    for sym, action in actions.items():
                        if action == 0 or sym not in prices:
                            continue
                        pf = self.paper_pf.get(agent_name)
                        if pf:
                            pf.execute(sym, action, prices[sym], pv)
            if is_active and self.mode == "paper":
                for agent_name, pf in self.paper_pf.items():
                    pf.snapshot(prices)
        self._warmup_end = self._bar

    def _fetch_market(self) -> tuple:
        prices = {}
        volumes = {}
        spot_batch_failed = False
        # Получаем blacklist один раз на цикл: если символ помечен как "мёртвый" —
        # не тратим на него запросы.
        bad_symbols = (
            self.futures_client._bad_symbols if self.futures_client is not None else set()
        )
        active_symbols = [s for s in self.symbols if s not in bad_symbols]

        # ── SPOT-тикеры батчем (дешевле чем по одному) ──
        try:
            exchange = self._spot_exchange
            if not self._spot_markets_loaded:
                exchange.load_markets()
                self._spot_markets_loaded = True
            spot_tickers = (
                exchange.fetch_tickers([f"{sym}/USDT" for sym in active_symbols])
                if active_symbols
                else {}
            )
            for sym in active_symbols:
                ticker = spot_tickers.get(f"{sym}/USDT", {}) or {}
                last = float(ticker.get("last", 0.0) or 0.0)
                quote_vol = float(
                    ticker.get("quoteVolume")
                    or ((ticker.get("baseVolume") or 0.0) * last)
                    or 0.0
                )
                if last > 0:
                    prices[sym] = last
                    volumes[sym] = quote_vol / last if last > 0 else 0.0
        except Exception as e:
            spot_batch_failed = True
            self._mark_data_error("price_error")
            log.error("Ошибка цен Bitget (spot batch): %s", e)
            self._spot_exchange = _public_client("spot")
            self._spot_markets_loaded = False

        # ── FUTURES-тикеры по одному (нужны для live_futures) ──
        # Раньше одна BadSymbol ронял весь цикл. Теперь каждый символ
        # в своём try/except: сломанный помечается bad и исключается,
        # остальные продолжают работать.
        if self.mode == "live_futures" and self.futures_client:
            for sym in active_symbols:
                if sym in prices:
                    continue  # уже получили с spot
                if sym in self.futures_client._bad_symbols:
                    continue
                try:
                    price = self.futures_client.ticker_price(sym)
                    if price > 0:
                        prices[sym] = price
                except Exception as e:
                    # Любая оставшаяся ошибка — не ломает цикл.
                    # ticker_price сам уже поймал BadSymbol/NotSupported;
                    # здесь ловим сеть, таймауты, rate-limits и пр.
                    log.debug("[bitget] ticker %s failed: %s — skip this cycle", sym, e)

            for sym in active_symbols:
                if sym in self.futures_client._bad_symbols:
                    continue
                try:
                    fut_price = self.futures_client.ticker_price(sym)
                    if fut_price > 0:
                        prices[sym] = fut_price
                except Exception as e:
                    log.debug("[bitget] futures ticker override %s failed: %s", sym, e)

        if spot_batch_failed and self._last_market_volumes:
            for sym in list(prices):
                if sym not in volumes:
                    cached_volume = float(self._last_market_volumes.get(sym, 0.0) or 0.0)
                    if cached_volume > 0.0:
                        volumes[sym] = cached_volume

        if self.liquidity_min_adv > 0:
            prices = {
                s: p
                for s, p in prices.items()
                if volumes.get(s, 0) * p >= self.liquidity_min_adv
            }
            volumes = {s: v for s, v in volumes.items() if s in prices}
        if prices:
            self._last_market_prices = dict(prices)
            self._last_market_volumes = dict(volumes)
            self._last_market_snapshot_at = time.time()
        elif self._last_market_prices:
            self._mark_data_error("price_error")
            cache_age = max(0.0, time.time() - float(self._last_market_snapshot_at or 0.0))
            reason = "spot batch failure" if spot_batch_failed else "empty market fetch"
            prices = dict(self._last_market_prices)
            volumes = dict(self._last_market_volumes)
            log.warning(
                "[bitget] market fetch returned 0 symbols (%s) -> reuse cached snapshot "
                "(age=%.1fs, symbols=%d)",
                reason,
                cache_age,
                len(prices),
            )
        return prices, volumes


AgentExchangeBridge = AgentBitgetBridge


def check_connectivity() -> bool:
    try:
        exchange = _public_client("spot")
        exchange.load_markets()
        exchange.fetch_ticker("BTC/USDT")
        log.info("✅ Bitget API доступен.")
        return True
    except Exception as e:
        log.error("❌ %s", e)
        return False
