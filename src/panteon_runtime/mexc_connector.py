"""Связка между агентами и MEXC: читает settings.txt и ведёт paper/live исполнение."""

from __future__ import annotations

import os, sys, time, hmac, hashlib, logging, json, threading, math
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from project_paths import PROJECT_ROOT, add_runtime_paths


def _bootstrap_project_paths():
    add_runtime_paths()


_bootstrap_project_paths()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mexc_connector")


# ══════════════════════════════════════════════════════════════════════════════
# НАСТРОЙКИ
# ══════════════════════════════════════════════════════════════════════════════

def _load_settings(path=None) -> dict:
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for cand in [os.path.join(str(PROJECT_ROOT), "settings.txt"), "settings.txt", os.path.join(script_dir, "settings.txt")]:
            if os.path.exists(cand):
                path = cand; break
    if not path:
        log.warning("[settings] settings.txt не найден — значения по умолчанию")
        return {}
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip().lower(); v = v.strip()
            if "#" in v: v = v[:v.index("#")].strip()
            if k and v: cfg[k] = v
    log.info("[settings] Загружено %d параметров из %s", len(cfg), path)
    return cfg


def _parse_settings(cfg: dict) -> dict:
    def g(k, d=None): return cfg.get(k.lower(), d)
    def gf(k, d):
        try: return float(g(k, d))
        except: return float(d)
    def gi(k, d):
        try: return int(g(k, d))
        except: return int(d)

    tf = str(g("timeframe", "minute")).lower()
    if   tf in ("minute","1m","min"):    poll, bar, tf_kline = 60,    60, "1m"
    elif tf in ("hour","1h","hourly"):   poll, bar, tf_kline = 3600,  1,  "60m"
    elif tf in ("day","1d","daily"):     poll, bar, tf_kline = 86400, 1,  "1d"
    else:                                poll, bar, tf_kline = 60,    60, "1m"

    act_every = max(1, gi("act_every", 1))
    poll *= act_every

    sym_raw = g("symbols", "all")
    symbols = None if (not sym_raw or sym_raw.lower() == "all") else \
              [s.strip().upper() for s in sym_raw.split(",") if s.strip()]

    def gb(k, d):
        v = g(k, str(d)).lower().strip()
        return v in ("on","true","yes","1")
    exchange_id = str(os.getenv("CRYPTO_EXCHANGE", "") or "").strip().lower()
    def gf_exchange(k, d):
        if exchange_id:
            for key in (f"{exchange_id}_{k}", f"{k}_{exchange_id}"):
                if g(key, None) is not None:
                    return gf(key, d)
        return gf(k, d)

    # Обновляем глобальные настройки
    import mexc_connector as _self_mc
    _self_mc.DAY_STOP_ENABLED     = gb("day_stop",          False)
    _self_mc.DAY_STOP_PCT         = gf("day_stop_pct",      10.0)
    _self_mc.PEAK_STOP_ENABLED    = gb("peak_stop",         False)
    _self_mc.PEAK_STOP_PCT        = gf("peak_stop_pct",     20.0)
    _self_mc.PROFIT_LOCK_ENABLED  = gb("profit_lock",       False)
    _self_mc.PROFIT_LOCK_TRIGGER  = gf("profit_lock_trigger", 5.0)
    _self_mc.PROFIT_LOCK_STEP     = gf("profit_lock_step",    5.0)
    _self_mc.PROFIT_LOCK_FRACTION = gf("profit_lock_fraction", 0.30)
    _self_mc.PROFIT_LOCK_MAX      = gf("profit_lock_max",     0.70)

    return {
        "initial_capital":   gf("initial_capital",   10_000.0),
        # paper_capital — для paper-режима; если не задан явно, = initial_capital
        # Позволяет задать paper_capital = 125 в settings.txt, не меняя
        # initial_capital = 1_000_000 (для симуляции crypto_exchange.py)
        "paper_capital":     gf("paper_capital",     gf("initial_capital", 10_000.0)),
        "trade_fraction":    gf_exchange("trade_fraction", 0.10),
        "leverage":          gf("leverage",           3.0),
        "liquidity_min_adv": gf_exchange("liquidity_min_adv", 0.0),
        "spot_fee":          gf("spot_fee",            0.001),
        "futures_fee":       gf("futures_fee",         0.0002),
        "slippage":          gf("slippage",            0.0001),
        "poll_interval":     poll,
        "bar":               bar,
        "act_every":         act_every,
        "symbols":           symbols,
        "timeframe":         tf,
        "tf_kline":          tf_kline,
        # Новые параметры
        "day_stop_enabled":     gb("day_stop",          False),
        "day_stop_pct":         gf("day_stop_pct",      10.0),
        "peak_stop_enabled":    gb("peak_stop",         False),
        "peak_stop_pct":        gf("peak_stop_pct",     20.0),
        "profit_lock_enabled":  gb("profit_lock",       False),
        "profit_lock_trigger":  gf("profit_lock_trigger", 5.0),
        "profit_lock_fraction": gf("profit_lock_fraction", 0.30),
        "profit_lock_max":      gf("profit_lock_max",     0.70),
    }


# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

API_KEY      = os.getenv("MEXC_API_KEY",    "YOUR_ACCESS_KEY_HERE")
API_SECRET   = os.getenv("MEXC_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
TRADING_MODE = os.getenv("MEXC_TRADING_MODE", "paper")
MIN_ORDER_USDT = 5.0

# Сколько символов брать из "all" (топ по объёму торгов за 24h).
# Было 30 → теперь 60: расширенная торговая вселенная с жёсткими фильтрами
# ликвидности. Больше пар = больше сигналов и возможностей, но бот по-прежнему
# держит одновременно только те, по которым появился сигнал, а не все сразу.
TOP_N_SYMBOLS = 60

# Жёсткие фильтры ликвидности для авто-выбора символов (fetch_top_symbols).
# Защищают от "мусорных" пар с низкой ликвидностью, где:
#   - ордер 5$ не наберёт минимального объёма контракта,
#   - спред слишком большой (проскальзывание съедает прибыль),
#   - монета может быть делистнута в любой момент.
MIN_QUOTE_VOLUME_24H_USDT = 10_000_000.0   # минимум $10M суточного оборота
# Минимальная цена: 8 знаков после запятой — предел точности большинства бирж.
# PEPE/SHIB/BONK ~$0.00001 — их мы хотим торговать (высокая ликвидность).
# Отсекаем только совсем "мусор" с ценой где округление ломает расчёты.
MIN_LAST_PRICE_USDT       = 1e-9
MAX_SPREAD_PCT            = 0.5             # максимум 0.5% относительного спреда

# ── Внутридневной стоп, пиковый стоп, фиксация прибыли ────────────────────
# Читаются из settings.txt через _parse_settings, дефолты ниже
DAY_STOP_ENABLED     = False
DAY_STOP_PCT         = 10.0
PEAK_STOP_ENABLED    = False
PEAK_STOP_PCT        = 20.0
PROFIT_LOCK_ENABLED  = False
PROFIT_LOCK_TRIGGER  = 5.0
PROFIT_LOCK_STEP     = 5.0
PROFIT_LOCK_FRACTION = 0.30
PROFIT_LOCK_MAX      = 0.70

# Минимальный прогрев: сколько баров истории нужно агентам.
# 96*BAR=5760 покрывает самый долгий warmup (96 часов при 1m).
# MEXC klines отдаёт max 1000 баров за запрос → 6 запросов на символ.
WARMUP_BARS = 5760   # 96 часов × 60 минут/час

SPOT_BASE_URL         = "https://api.mexc.com"
FUTURES_LIVE_BASE_URL = "https://contract.mexc.com"
FUTURES_TEST_BASE_URL = "https://futures.testnet.mexc.com"

MEXC_CONNECTOR_HTTP = requests.Session()
MEXC_CONNECTOR_HTTP.trust_env = False


def _mexc_get(url: str, **kwargs):
    return MEXC_CONNECTOR_HTTP.get(url, **kwargs)

# Подключение MEXC-специфичных модулей
try:
    from mexc_funding import FundingDataFetcher, funding_signal
    _HAS_FUNDING = True
except ImportError:
    FundingDataFetcher = None
    _HAS_FUNDING = False
    log.warning("[connector] mexc_funding.py не найден")

try:
    from panteon_agents import (
        make_panteon_agents as make_mexc_agents,
        make_panteon_players as make_mexc_players,
        configure as _configure_mexc_agents,
        set_fetcher as _set_mexc_fetcher,
        wrap_existing_agents as _wrap_agents,
    )
    _HAS_MEXC_AGENTS = True
except ImportError:
    _HAS_MEXC_AGENTS = False
    log.warning("[connector] panteon_agents.py не найден")

ACTION_NAMES = {
    0:"hold", 1:"spot_buy_half", 2:"spot_buy_full", 3:"spot_sell_all",
    4:"fut_long_half", 5:"fut_long_full", 6:"fut_short_half",
    7:"fut_short_full", 8:"fut_close_all",
}


# ══════════════════════════════════════════════════════════════════════════════
# ИСТОРИЧЕСКИЕ ДАННЫЕ  (warmup)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol: str, interval: str = "1m", n_bars: int = 5760) -> List[dict]:
    """
    Загружает исторические klines с MEXC Spot API.
    Возвращает список dict: {"close": float, "volume": float}
    Делает несколько запросов если n_bars > 1000 (лимит API).

    symbol: "BTCUSDT" (с USDT суффиксом)
    interval: "1m" | "5m" | "15m" | "30m" | "60m" | "4h" | "1d"
    """
    all_klines = []
    limit      = 1000   # максимум за один запрос
    requests_n = (n_bars + limit - 1) // limit   # ceil

    # Начинаем с самых старых баров нужного периода
    end_time   = int(time.time() * 1000)
    start_time = end_time - n_bars * _interval_ms(interval)

    for i in range(requests_n):
        chunk_end = start_time + (i + 1) * limit * _interval_ms(interval)
        chunk_end = min(chunk_end, end_time)
        try:
            r = _mexc_get(
                f"{SPOT_BASE_URL}/api/v3/klines",
                params={
                    "symbol":    symbol,
                    "interval":  interval,
                    "startTime": start_time + i * limit * _interval_ms(interval),
                    "endTime":   chunk_end,
                    "limit":     limit,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            # MEXC klines format: [openTime, open, high, low, close, volume, ...]
            for row in data:
                try:
                    all_klines.append({
                        "close":  float(row[4]),
                        "volume": float(row[5]),
                    })
                except (IndexError, ValueError):
                    pass
        except Exception as e:
            log.warning("[klines] %s %s chunk %d/%d: %s", symbol, interval, i+1, requests_n, e)
            break

    return all_klines


def _interval_ms(interval: str) -> int:
    """Длина интервала kline в миллисекундах."""
    mapping = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "60m": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(interval, 60_000)


def build_warmup_data(symbols: List[str], interval: str, n_bars: int) -> List[dict]:
    """
    Загружает исторические klines для всех символов и собирает
    список баров в формате [{sym: price, ...}, ...].

    Возвращает список словарей prices_per_bar и volumes_per_bar.
    """
    log.info("[warmup] Загружаю %d баров истории для %d символов (interval=%s)...",
             n_bars, len(symbols), interval)

    # prices_per_sym[sym] = [p0, p1, ..., pN]
    prices_per_sym  : Dict[str, List[float]] = {}
    volumes_per_sym : Dict[str, List[float]] = {}

    for i, sym in enumerate(symbols):
        mexc_sym = f"{sym}USDT"
        klines   = fetch_klines(mexc_sym, interval, n_bars)
        if not klines:
            log.warning("[warmup] Нет данных для %s — пропускаю", sym)
            continue
        prices_per_sym[sym]  = [k["close"]  for k in klines]
        volumes_per_sym[sym] = [k["volume"] for k in klines]
        if (i + 1) % 5 == 0 or i == len(symbols) - 1:
            log.info("[warmup]   Загружено %d/%d символов (%s: %d баров)",
                     i + 1, len(symbols), sym, len(klines))

    # Находим минимальную длину (у всех символов должно быть одинаково)
    if not prices_per_sym:
        return [], []
    min_len = min(len(v) for v in prices_per_sym.values())

    # Собираем в список баров [{sym: price}]
    prices_bars  = []
    volumes_bars = []
    for i in range(min_len):
        pb = {s: prices_per_sym[s][i]  for s in prices_per_sym}
        vb = {s: volumes_per_sym[s][i] for s in volumes_per_sym}
        prices_bars.append(pb)
        volumes_bars.append(vb)

    log.info("[warmup] Готово: %d баров × %d символов", min_len, len(prices_per_sym))
    return prices_bars, volumes_bars


# ══════════════════════════════════════════════════════════════════════════════
# ФИЛЬТР СИМВОЛОВ
# ══════════════════════════════════════════════════════════════════════════════

# ── Блэклист ДЛЯ АВТО-СПИСКА (дублирует MexcFuturesClient._FUTURES_BLACKLIST) ──
# Здесь, чтобы fetch_top_symbols не вернул монеты, которые потом заблокируются
# в futures-клиенте (close_all() не работает → маржа зависает).
# Держим в синхронизации с MexcFuturesClient._FUTURES_BLACKLIST.
_AUTO_SYMBOL_BLACKLIST = frozenset({
    "EUR", "USD1", "USDT", "USDC", "STABLE", "BUSD",
    "GOLD(XAUT)", "GOLD(PAXG)",
    "FIDA", "WXT", "PSAI", "PE", "ATLA", "META", "ONT",
    "HYPE", "BULLA", "STO", "SOLV", "RED",
})


def fetch_top_symbols(
    n: int = TOP_N_SYMBOLS,
    min_quote_volume: float = MIN_QUOTE_VOLUME_24H_USDT,
    min_price: float = MIN_LAST_PRICE_USDT,
    max_spread_pct: float = MAX_SPREAD_PCT,
    extra_blacklist: set | None = None,
) -> List[str]:
    """
    Берёт топ-N USDT-пар с MEXC Spot API по объёму торгов за 24h с жёсткими
    фильтрами ликвидности. Расширенная версия для большего покрытия рынка.

    Применяет последовательно:
      1. Только *USDT пары (без leveraged tokens и стейблкоинов).
      2. Базовый символ не в FUTURES_BLACKLIST (нет фьючерса или баги закрытия).
      3. Суточный оборот >= min_quote_volume USDT (ликвидность).
      4. Цена >= min_price (отсечь микро-токены с округлением до нуля).
      5. Относительный спред (bid/ask) <= max_spread_pct % (узкий рынок).
    Затем сортировка по объёму и возврат top-N.
    """
    STABLECOINS = {"USDT","USDC","BUSD","TUSD","DAI","FDUSD","USDP","UST",
                   "SUSD","GUSD","PAXG","LUSD","CRVUSD","PYUSD","AEUR"}
    EXCLUDE_PATTERNS = ("UP","DOWN","BULL","BEAR","LONG","SHORT","3L","3S","5L","5S")

    blacklist = set(_AUTO_SYMBOL_BLACKLIST)
    if extra_blacklist:
        blacklist.update(extra_blacklist)

    try:
        r = _mexc_get(f"{SPOT_BASE_URL}/api/v3/ticker/24hr", timeout=10)
        r.raise_for_status()
        items = r.json()
    except Exception as e:
        log.error("[symbols] Ошибка получения тикеров: %s", e)
        # Запасной список — известные ликвидные монеты (не трогаем на ошибке сети)
        return ["BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOT","MATIC","LINK",
                "TON","DOGE","SUI","APT","ARB","OP","INJ","TIA","SEI","RENDER"]

    # Статистика для лога
    n_total = len(items)
    n_not_usdt = n_stable = n_leveraged = n_bad_name = 0
    n_blacklisted = n_low_vol = n_low_price = n_wide_spread = 0

    candidates: List[Tuple[str, float]] = []
    for item in items:
        sym = item.get("symbol", "")
        if not sym.endswith("USDT"):
            n_not_usdt += 1
            continue
        base = sym[:-4]

        # 1. Стейблкоины
        if base in STABLECOINS:
            n_stable += 1
            continue

        # 2. Leveraged tokens
        if any(base.endswith(p) for p in EXCLUDE_PATTERNS):
            n_leveraged += 1
            continue

        # 3. Невалидные имена
        if not base or base[0].isdigit() or len(base) < 2:
            n_bad_name += 1
            continue

        # 4. Блэклист фьючерсных контрактов (баги закрытия позиций)
        if base in blacklist:
            n_blacklisted += 1
            continue

        # 5. Парсим численные поля
        try:
            quote_vol = float(item.get("quoteVolume", 0) or 0)
            price     = float(item.get("lastPrice",   0) or 0)
            bid       = float(item.get("bidPrice",    0) or 0)
            ask       = float(item.get("askPrice",    0) or 0)
        except (ValueError, TypeError):
            continue

        # 6. Минимальная ликвидность
        if quote_vol < min_quote_volume:
            n_low_vol += 1
            continue

        # 7. Минимальная цена
        if price < min_price:
            n_low_price += 1
            continue

        # 8. Спред: (ask - bid) / price * 100%
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / price * 100.0
            if spread_pct > max_spread_pct:
                n_wide_spread += 1
                continue

        candidates.append((base, quote_vol))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [sym for sym, _ in candidates[:n]]

    log.info(
        "[symbols] Фильтрация: %d тикеров → %d прошли фильтры → топ-%d выбрано",
        n_total, len(candidates), len(top),
    )
    log.info(
        "[symbols]   отсеяно: non-USDT=%d, stable=%d, leveraged=%d, "
        "bad_name=%d, blacklist=%d, low_vol=%d, low_price=%d, wide_spread=%d",
        n_not_usdt, n_stable, n_leveraged, n_bad_name,
        n_blacklisted, n_low_vol, n_low_price, n_wide_spread,
    )
    log.info("[symbols] Топ-%d по объёму: %s", len(top), top)
    return top


# ══════════════════════════════════════════════════════════════════════════════
# PAPER PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

class PaperPortfolio:
    def __init__(self, initial_capital, trade_fraction, leverage,
                 spot_fee, futures_fee, slippage):
        self.initial_capital = initial_capital
        self.trade_fraction  = trade_fraction
        self.leverage        = leverage
        self.spot_fee        = spot_fee
        self.futures_fee     = futures_fee
        self.slippage        = slippage

        self.cash    : float           = initial_capital
        self.holdings: Dict[str,float] = {}
        self.futures : Dict[str,dict]  = {}

        self.history : List[float]     = [initial_capital]
        self.trades  : List[dict]      = []
        self.op_count = {"spot_buy":0,"spot_sell":0,"fut_long":0,"fut_short":0,"fut_close":0}
        self.op_fees  = {"spot_buy":0.0,"spot_sell":0.0,"fut_long":0.0,"fut_short":0.0}

        # ── DayStop / PeakStop / ProfitLock ────────────────────────
        self._day_start_pv      = initial_capital
        self._day_start_bar     = 0
        self._day_stopped       = False
        self._abs_peak_pv       = initial_capital
        self._peak_stopped      = False
        self._locked_profit     = 0.0
        self._lock_next_at      = initial_capital * (1.0 + PROFIT_LOCK_TRIGGER / 100.0)

    def portfolio_value(self, prices: Dict[str,float]) -> float:
        v = self.cash
        for sym, qty in self.holdings.items():
            v += qty * prices.get(sym, 0)
        for sym, pos in self.futures.items():
            px  = prices.get(sym, pos["entry_price"])
            pct = (px - pos["entry_price"]) / (pos["entry_price"] + 1e-12)
            if pos["side"] == "short": pct = -pct
            v  += pos["margin"] * (1 + pct * pos["leverage"])
        return max(v, 0.0)

    def execute(self, sym: str, action: int, price: float, pv: float,
                risk_multiplier: float = 1.0) -> str | None:
        slip = price * (1 + self.slippage) if action in (1,2,4,5) \
               else price * (1 - self.slippage)
        size_mult = min(max(float(risk_multiplier or 1.0), 0.35), 1.50)
        usdt = pv * self.trade_fraction * size_mult

        if action in (1, 2):
            amount = usdt * (0.5 if action==1 else 1.0)
            fee    = amount * self.spot_fee
            if self.cash < amount: return None
            qty    = (amount - fee) / slip
            self.cash -= amount
            self.holdings[sym] = self.holdings.get(sym, 0) + qty
            self.op_count["spot_buy"] += 1; self.op_fees["spot_buy"] += fee
            desc = f"BUY {sym:6s} {qty:.6f} @ {slip:,.2f}  fee={fee:.2f}"
            self.trades.append({"bar":len(self.history), "desc":desc})
            return desc

        elif action == 3:
            qty = self.holdings.pop(sym, 0)
            if qty <= 0: return None
            gross = qty * slip; fee = gross * self.spot_fee
            self.cash += gross - fee
            self.op_count["spot_sell"] += 1; self.op_fees["spot_sell"] += fee
            desc = f"SELL {sym:5s} {qty:.6f} @ {slip:,.2f}  fee={fee:.2f}"
            self.trades.append({"bar":len(self.history), "desc":desc})
            return desc

        elif action in (4, 5, 6, 7):
            frac   = 0.5 if action in (4,6) else 1.0
            side   = "long" if action in (4,5) else "short"
            margin = usdt * frac; fee = margin * self.futures_fee
            if self.cash < margin + fee or sym in self.futures: return None
            self.cash -= (margin + fee)
            self.futures[sym] = {"side":side, "entry_price":slip,
                                  "margin":margin, "leverage":self.leverage}
            key = "fut_long" if side=="long" else "fut_short"
            self.op_count[key] += 1; self.op_fees[key] += fee
            desc = f"{side.upper():5s} {sym:5s} margin={margin:.2f} lev={self.leverage}x @ {slip:,.2f}"
            self.trades.append({"bar":len(self.history), "desc":desc})
            return desc

        elif action == 8:
            pos = self.futures.pop(sym, None)
            if not pos: return None
            pct  = (slip - pos["entry_price"]) / (pos["entry_price"] + 1e-12)
            if pos["side"] == "short": pct = -pct
            pnl  = pos["margin"] * pct * pos["leverage"]
            fee  = pos["margin"] * self.futures_fee
            self.cash += pos["margin"] + pnl - fee
            self.op_count["fut_close"] += 1
            desc = f"CLOSE {sym:5s} pnl={pnl:+.2f}  fee={fee:.2f}"
            self.trades.append({"bar":len(self.history), "desc":desc})
            return desc

        return None

    def snapshot(self, prices: Dict[str,float]):
        pv = self.portfolio_value(prices)
        self.history.append(pv)
        bar = len(self.history) - 1

        # Обновляем день-базис каждые 24*BAR
        from mexc_connector import DAY_STOP_ENABLED, DAY_STOP_PCT, PEAK_STOP_ENABLED
        from mexc_connector import PEAK_STOP_PCT, PROFIT_LOCK_ENABLED
        from mexc_connector import PROFIT_LOCK_TRIGGER, PROFIT_LOCK_STEP
        from mexc_connector import PROFIT_LOCK_FRACTION, PROFIT_LOCK_MAX
        import mexc_connector as _mc

        _bar_module = getattr(_mc, 'BAR', 60) if hasattr(_mc, 'BAR') else 60

        # DayStop: обновление базиса
        if DAY_STOP_ENABLED:
            if bar - self._day_start_bar >= 24 * _bar_module:
                self._day_start_pv  = pv
                self._day_start_bar = bar
                self._day_stopped   = False

        # ProfitLock
        if PROFIT_LOCK_ENABLED and not self._day_stopped:
            if pv >= self._lock_next_at:
                _profit  = pv - self.initial_capital
                _max_lk  = self.initial_capital * PROFIT_LOCK_MAX - self._locked_profit
                if _profit > 0 and _max_lk > 0:
                    _to_lock = min(_profit * PROFIT_LOCK_FRACTION, _max_lk)
                    self._locked_profit += _to_lock
                    self.cash = max(0.0, self.cash - _to_lock)
                    log.info("  [ProfitLock] +$%.2f → locked=%.2f", _to_lock, self._locked_profit)
                self._lock_next_at = max(self._lock_next_at * (1.0 + PROFIT_LOCK_STEP / 100.0),
                                         pv * (1.0 + PROFIT_LOCK_STEP / 100.0))

        # PeakStop: обновление пика
        if PEAK_STOP_ENABLED:
            if pv > self._abs_peak_pv:
                self._abs_peak_pv = pv

    def get_usd_breakdown(self, prices: Dict[str, float]) -> dict:
        """
        Возвращает разбивку активов на 2 категории:
          cash_usdt    — свободный USDT + locked (не в крипто)
          crypto_value — текущая рыночная стоимость всех крипто-позиций
        """
        # Cash USDT — то что не в крипто
        cash_usdt = float(self.cash) + self._locked_profit

        # Стоимость спотовой крипты
        spot_value = 0.0
        for sym, qty in self.holdings.items():
            p = prices.get(sym, 0.0)
            if p > 0:
                spot_value += qty * p

        # Стоимость фьючерсов (маржа + нереализованный PnL)
        fut_value = 0.0
        for sym, pos in self.futures.items():
            p = prices.get(sym, pos["entry_price"])
            if p > 0:
                pct = (p - pos["entry_price"]) / (pos["entry_price"] + 1e-12)
                if pos["side"] == "short": pct = -pct
                fut_value += pos["margin"] * (1 + pct * pos["leverage"])

        crypto_value = spot_value + max(0.0, fut_value)
        total_active = float(self.cash) + crypto_value  # торгуемый капитал

        # Разбивка spot по символам
        spot_positions = {sym: {"qty": qty, "value": qty * prices.get(sym, 0)}
                          for sym, qty in self.holdings.items() if qty > 0}
        # Разбивка futures
        fut_positions = {}
        for sym, pos in self.futures.items():
            p = prices.get(sym, pos["entry_price"])
            pct = (p - pos["entry_price"]) / (pos["entry_price"] + 1e-12)
            if pos["side"] == "short": pct = -pct
            cur_val = pos["margin"] * (1 + pct * pos["leverage"])
            pnl_pct = pct * pos["leverage"] * 100
            fut_positions[sym] = {"side": pos["side"], "margin": pos["margin"],
                                   "value": cur_val, "pnl_pct": round(pnl_pct, 2)}
        return {
            "cash_usdt":      round(cash_usdt, 4),
            "crypto_value":   round(crypto_value, 4),
            "locked_profit":  round(self._locked_profit, 4),
            "total_active":   round(total_active, 4),
            "total_with_locked": round(total_active + self._locked_profit, 4),
            "spot_positions": spot_positions,
            "fut_positions":  fut_positions,
        }

    def stats(self) -> dict:
        arr = np.array(self.history, dtype=float)
        if len(arr) < 2: return {}
        final  = arr[-1]
        ret    = (final / self.initial_capital - 1) * 100
        peaks  = np.maximum.accumulate(arr)
        maxdd  = float(((peaks - arr) / (peaks + 1e-9) * 100).max())
        ra     = np.diff(arr) / (arr[:-1] + 1e-9)
        ra     = ra[np.isfinite(ra)]
        sharpe = float(np.mean(ra) / (np.std(ra) + 1e-9) * np.sqrt(365*24)) \
                 if len(ra) > 1 else 0.0
        calmar = (ret / maxdd) if maxdd > 0.1 else 0.0
        return dict(
            final=final, ret=ret, maxdd=maxdd, sharpe=sharpe, calmar=calmar,
            trades=len(self.trades),
            spot_buys=self.op_count["spot_buy"], spot_sells=self.op_count["spot_sell"],
            fut_longs=self.op_count["fut_long"], fut_shorts=self.op_count["fut_short"],
            total_fees=sum(self.op_fees.values()),
            locked_profit=self._locked_profit,
            total_with_locked=final + self._locked_profit,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HTTP-КЛИЕНТЫ
# ══════════════════════════════════════════════════════════════════════════════

class MexcSpotClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key; self.api_secret = api_secret
        self.s = requests.Session()
        self.s.trust_env = False
        self.s.headers.update({"X-MEXC-APIKEY": api_key})

    def _sign(self, p):
        p["timestamp"] = int(time.time()*1000)
        q = "&".join(f"{k}={v}" for k,v in sorted(p.items()))
        p["signature"] = hmac.new(self.api_secret.encode(),q.encode(),hashlib.sha256).hexdigest()
        return p

    def get(self, path, params=None, signed=False):
        p = params or {}
        if signed: p = self._sign(p)
        r = self.s.get(SPOT_BASE_URL+path, params=p, timeout=10)
        r.raise_for_status(); return r.json()

    def post(self, path, params=None):
        r = self.s.post(SPOT_BASE_URL+path, params=self._sign(params or {}), timeout=10)
        r.raise_for_status(); return r.json()

    def account_balance(self):
        return {b["asset"]:float(b["free"])
                for b in self.get("/api/v3/account",signed=True).get("balances",[])
                if float(b["free"])>0}

    def buy_usdt(self, sym, usdt):
        return self.post("/api/v3/order",
               {"symbol":sym,"side":"BUY","type":"MARKET","quoteOrderQty":f"{usdt:.2f}"})

    def sell_qty(self, sym, qty):
        return self.post("/api/v3/order",
               {"symbol":sym,"side":"SELL","type":"MARKET","quantity":f"{qty:.8f}"})


class MexcFuturesClient:
    """
    Клиент MEXC Contract API (фьючерсы).

    Подпись:
      GET  → HMAC-SHA256(apiKey + timestamp + queryString)
      POST → HMAC-SHA256(apiKey + timestamp + jsonBody)

    FIX v5: POST ранее подписывался как query-string вместо JSON-body →
    все place_order возвращали 401/403 и ордера не исполнялись.
    """

    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key; self.api_secret = api_secret
        self.base = FUTURES_TEST_BASE_URL if testnet else FUTURES_LIVE_BASE_URL
        self.s = requests.Session()
        self.s.trust_env = False
        # Runtime-блэклист: символы которые получили ошибку "symbol not found"
        self._bad_symbols: set = set()
        self._contract_meta_cache: dict = {}
        self._insufficient_symbol_until: dict = {}

    def _sign_get(self, params: dict, ts: str) -> str:
        """Подпись для GET: apiKey + timestamp + queryString."""
        q = "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else ""
        return hmac.new(
            self.api_secret.encode(),
            f"{self.api_key}{ts}{q}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _sign_post(self, body: dict, ts: str) -> str:
        """Подпись для POST: apiKey + timestamp + jsonBody."""
        import json as _json
        body_str = _json.dumps(body, separators=(",", ":")) if body else ""
        return hmac.new(
            self.api_secret.encode(),
            f"{self.api_key}{ts}{body_str}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _req(self, method: str, path: str, params: dict = None):
        # FIX v6: глобальный rate limiter — не более 2 запросов в секунду
        # MEXC возвращает code=510 при превышении. Задержка ПЕРЕД запросом
        # гарантирует что никакие два запроса не идут быстрее 0.6с.
        now = time.time()
        if hasattr(self, '_last_req_time'):
            elapsed = now - self._last_req_time
            if elapsed < 0.6:
                time.sleep(0.6 - elapsed)
        self._last_req_time = time.time()

        p = params or {}
        ts = str(int(time.time() * 1000))

        # User-Agent обязателен — без него Cloudflare WAF блокирует POST с 403
        UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")

        if method == "POST":
            sig = self._sign_post(p, ts)
            h = {"ApiKey": self.api_key, "Request-Time": ts,
                 "Signature": sig, "Content-Type": "application/json",
                 "User-Agent": UA}
            body_str = json.dumps(p, separators=(",", ":")) if p else ""
            r = self.s.request(method, self.base + path, data=body_str, headers=h, timeout=10)
        else:
            sig = self._sign_get(p, ts)
            h = {"ApiKey": self.api_key, "Request-Time": ts,
                 "Signature": sig, "Content-Type": "application/json",
                 "User-Agent": UA}
            r = self.s.request(method, self.base + path, params=p, headers=h, timeout=10)

        # Логируем тело ответа при ошибке ПЕРЕД raise_for_status
        if r.status_code >= 400:
            try:
                err_body = r.json()
            except Exception:
                err_body = r.text[:300]
            log.error("MEXC API %s %s → HTTP %s: %s", method, path, r.status_code, err_body)

        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and data.get("code") not in (None, 0, 200):
            log.debug("MEXC futures API: code=%s  msg=%s", data.get("code"), data.get("message"))

        return data

    def ticker_price(self, symbol):
        try:
            return float(self._req("GET", f"/api/v1/contract/ticker?symbol={symbol}")["data"]["lastPrice"])
        except Exception:
            return 0.0

    def account_assets(self):
        """Возвращает {USDT: equity, USDT_AVAIL: availableBalance}."""
        result = {}
        try:
            resp = self._req("GET", "/api/v1/private/account/assets")
            raw_data = resp.get("data", [])
            if not raw_data:
                log.warning("account_assets: пустой data=[]. code=%s msg=%s",
                            resp.get("code"), resp.get("message") or resp.get("msg"))
            for a in raw_data:
                equity = float(a.get("equity", 0) or 0)
                avail  = float(a.get("availableBalance", 0) or 0)
                cur    = a.get("currency", "")
                if equity > 0 or avail > 0:
                    result[cur] = equity          # total equity
                    result[f"{cur}_AVAIL"] = avail  # available for new positions
        except Exception as e:
            log.debug("account_assets: %s", e)
        return result

    def is_symbol_on_margin_cooldown(self, symbol: str) -> bool:
        return time.time() < float(self._insufficient_symbol_until.get(symbol, 0) or 0)

    def margin_cooldown_remaining(self, symbol: str) -> int:
        remaining = float(self._insufficient_symbol_until.get(symbol, 0) or 0) - time.time()
        return max(0, int(math.ceil(remaining)))

    def _get_contract_meta(self, symbol: str) -> dict:
        cached = self._contract_meta_cache.get(symbol)
        if cached:
            return cached

        fallback_sizes = {
            "BTC": 0.0001, "ETH": 0.01, "BNB": 0.01,
            "SOL": 0.1, "XRP": 10.0, "ADA": 10.0,
        }
        fallback = {
            "symbol": f"{symbol}_USDT",
            "contractSize": float(fallback_sizes.get(symbol, 1.0)),
            "minVol": 1,
            "volUnit": 1,
            "maxVol": 10_000_000,
            "takerFeeRate": 0.0006,
            "apiAllowed": True,
            "state": 0,
            "metadataFallback": True,
            "metadataSource": "fallback",
        }
        contract = self._contract_name(symbol)
        if contract is None:
            return fallback

        try:
            r = self.s.get(
                f"{self.base}/api/v1/contract/detail",
                params={"symbol": contract},
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", [])
            if isinstance(data, list):
                raw = data[0] if data else {}
            elif isinstance(data, dict):
                raw = data
            else:
                raw = {}
            if not raw:
                raise ValueError(f"empty contract meta for {contract}")

            meta = {
                "symbol": raw.get("symbol", contract),
                "contractSize": float(raw.get("contractSize", fallback["contractSize"]) or fallback["contractSize"]),
                "minVol": int(math.ceil(float(raw.get("minVol", fallback["minVol"]) or fallback["minVol"]))),
                "volUnit": int(max(1, round(float(raw.get("volUnit", fallback["volUnit"]) or fallback["volUnit"])))),
                "maxVol": int(max(1, float(raw.get("maxVol", fallback["maxVol"]) or fallback["maxVol"]))),
                "takerFeeRate": float(raw.get("takerFeeRate", fallback["takerFeeRate"]) or fallback["takerFeeRate"]),
                "apiAllowed": bool(raw.get("apiAllowed", True)),
                "state": int(raw.get("state", 0) or 0),
                "metadataFallback": False,
                "metadataSource": "exchange",
            }
            self._contract_meta_cache[symbol] = meta
            return meta
        except Exception as e:
            log.debug("contract detail fallback for %s: %s", symbol, e)
            self._contract_meta_cache[symbol] = fallback
            return fallback

    # Символы-исключения: не существуют как фьючерсные контракты на MEXC
    # или имеют нестандартные имена контрактов
    # FIX v9.2: кэш плечей открытых позиций {symbol: leverage}
    # При code=2021 ("leverage inconsistent") — новый ордер должен использовать
    # то же плечо что у существующей позиции.
    _pos_leverage_cache: dict = {}

    _FUTURES_BLACKLIST = frozenset({
        "EUR", "USD1", "USDT", "USDC", "STABLE", "BUSD",
        "GOLD(XAUT)", "GOLD(PAXG)",  # скобки недопустимы в contract name
        "FIDA", "WXT", "PSAI", "PE", "ATLA", "META", "ONT",
        "HYPE",  # HyperLiquid, может отсутствовать на MEXC futures
        # Ультра-волатильные/малоликвидные — постоянные MASTER-SL
        "BULLA", "STO", "SOLV", "RED",
    })

    def _contract_name(self, base_sym: str) -> str | None:
        """
        Преобразует базовый символ (например 'BTC') в имя фьючерсного контракта ('BTC_USDT').
        Возвращает None если символ точно не торгуется.
        """
        if base_sym in self._FUTURES_BLACKLIST:
            return None
        if base_sym in self._bad_symbols:
            return None
        # Символы со скобками — недопустимы
        if "(" in base_sym or ")" in base_sym:
            return None
        return f"{base_sym}_USDT"

    def place_order(self, symbol: str, side: int, vol: int, leverage: int = 2) -> dict:
        """
        Размещает ордер.
        MEXC side: 1=open long, 2=close short, 3=open short, 4=close long.
        Сначала пробует через pymexc (обходит Cloudflare WAF).
        Если pymexc не установлен — прямой запрос.
        """
        contract = self._contract_name(symbol)
        if contract is None:
            log.debug("place_order: %s пропущен (блэклист)", symbol)
            return {"success": False, "data": None, "skip": True}
        meta = self._get_contract_meta(symbol)
        contract_size = max(float(meta.get("contractSize", 1.0) or 1.0), 1e-12)
        base_amount = float(vol) * contract_size
        notional = 0.0

        if self.is_symbol_on_margin_cooldown(symbol):
            remain = self.margin_cooldown_remaining(symbol)
            log.info("  ⏳ order skip: %s side=%d → cooldown after insufficient margin (%ds)",
                     contract, side, remain)
            return {"success": False, "data": None, "code": 2005, "skip": True, "cooldown": True}

        # FIX v9.2: code=2021 "leverage inconsistent with existing position"
        # Если на данном символе уже есть позиция с другим плечом — используем его.
        # Кэш обновляется при успешных ордерах и при get_positions().
        cached_lev = self._pos_leverage_cache.get(symbol)
        if cached_lev:
            # Кэш есть → используем его, не делаем лишний GET
            if cached_lev != leverage:
                log.info("  [leverage] %s: кэш=%dx, запрошено=%dx → используем кэш",
                         symbol, cached_lev, leverage)
            leverage = cached_lev
        else:
            # Кэш пуст для этого символа → проверяем живые позиции (1 GET)
            try:
                live_positions = self._req("GET", "/api/v1/private/position/open_positions")
                for pos in live_positions.get("data", []):
                    sym_base = pos.get("symbol", "").replace("_USDT", "")
                    lev_val  = int(pos.get("leverage", leverage))
                    if sym_base:
                        self._pos_leverage_cache[sym_base] = lev_val
                pos_lev = self._pos_leverage_cache.get(symbol)
                if pos_lev and pos_lev != leverage:
                    log.info("  [leverage] %s: позиция=%dx, запрошено=%dx → меняем",
                             symbol, pos_lev, leverage)
                    leverage = pos_lev
            except Exception:
                pass  # не критично — продолжаем с запрошенным leverage

        # ── Запрос через pymexc session + SHA256 подпись ────────
        # WAF пропускает запросы через pymexc session (проверено HTTP 200).
        # pymexc внутри использует SHA256 — 602 была из-за неверной сигнатуры вызова.
        # Решение: берём session из pymexc, подпись строим сами (SHA256 + leverage).
        try:
            import json as _json
            from pymexc import futures as _pymexc_fut
            if not hasattr(self, '_pymexc_client'):
                self._pymexc_client = _pymexc_fut.HTTP(
                    api_key=self.api_key, api_secret=self.api_secret
                )
            _sess = (getattr(self._pymexc_client, 'session', None) or
                     getattr(self._pymexc_client, '_session', None))
            if _sess is None:
                raise RuntimeError("pymexc session not found")

            _ts   = str(int(time.time() * 1000))
            _body = {"symbol": contract, "side": side, "vol": vol,
                     "type": 5, "openType": 1,
                     "leverage": max(1, int(leverage)) if leverage else 20}
            _bstr = _json.dumps(_body, separators=(",", ":"))
            _msg  = f"{self.api_key}{_ts}{_bstr}"
            _sig  = hmac.new(self.api_secret.encode(), _msg.encode(), hashlib.sha256).hexdigest()
            _h = {"ApiKey": self.api_key, "Request-Time": _ts, "Signature": _sig,
                  "Content-Type": "application/json",
                  "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/124.0.0.0 Safari/537.36")}
            # FIX v6: rate limiter для pymexc path (обходит _req)
            now = time.time()
            if hasattr(self, '_last_req_time'):
                elapsed = now - self._last_req_time
                if elapsed < 0.6:
                    time.sleep(0.6 - elapsed)
            self._last_req_time = time.time()

            _r   = _sess.post(f"{FUTURES_LIVE_BASE_URL}/api/v1/private/order/submit",
                              data=_bstr, headers=_h, timeout=10)
            _r.raise_for_status()
            res  = _r.json()
            code = res.get("code", -1)
            success = (code == 200 or res.get("data") is not None)
            order_id = None
            if success:
                order_id = res.get("data")
                # Обновляем кэш плечей при успешном ордере
                self._pos_leverage_cache[symbol] = _body["leverage"]
                self._insufficient_symbol_until.pop(symbol, None)
                log.info("  ✅ order OK: %s mexc_side=%d vol=%d lev=%dx → order_id=%s",
                         contract, side, vol, _body["leverage"], order_id)
            else:
                # FIX v7: code=2005 "Balance insufficient" — ожидаемо при tight margin,
                # не ERROR а INFO. Позиция просто пропускается до следующего бара.
                _log_fn = log.info if code == 2005 else log.error
                _log_fn("  ❌ order FAIL: %s side=%d → code=%s %s",
                        contract, side, code,
                        res.get("message") or res.get("msg", ""))
                log.error("  [MEXC] order failed payload: http_status=%s code=%s msg=%s body=%s",
                          getattr(_r, "status_code", None), code,
                          res.get("message") or res.get("msg", ""), res)
                if code == 2021:
                    # code=2021: leverage inconsistent → узнать актуальное и retry
                    log.warning("  [leverage] code=2021 на %s → сброс кэша плеча", symbol)
                    self._pos_leverage_cache.pop(symbol, None)
                elif code in (2005, 2018):
                    self._insufficient_symbol_until[symbol] = time.time() + 180
                elif code in (2011, 2013, 2019):
                    self._bad_symbols.add(symbol)
            return {
                "success": success,
                "data": res.get("data"),
                "code": code,
                "order_id": order_id if success else None,
                "msg": res.get("message") or res.get("msg", ""),
                "http_status": getattr(_r, "status_code", None),
                "body": res,
                "contracts": vol,
                "contractSize": contract_size,
                "amount": base_amount,
                "notional": notional,
            }
        except ImportError:
            log.debug("pymexc не установлен, fallback на прямой запрос")
        except Exception as e:
            log.warning("  pymexc/session ошибка: %s — прямой запрос", e)

    # ── Прямой запасной запрос ──────────────────────────────
        body = {
            "symbol":   contract,
            "side":     side,
            "vol":      vol,
            "type":     5,
            "openType": 1,
            "leverage": max(1, int(leverage)) if leverage else 20,
        }
        try:
            res = self._req("POST", "/api/v1/private/order/submit", body)
            code = res.get("code", -1)
            success = (code == 200 or res.get("success") is True)
            order_id = None
            if success:
                order_id = res.get("data")
                self._insufficient_symbol_until.pop(symbol, None)
                log.info("  ✅ order OK: %s mexc_side=%d vol=%d → order_id=%s",
                         contract, side, vol, order_id)
            else:
                msg_text = res.get("message") or res.get("msg") or str(res)
                # FIX v7: code=2005 → INFO (expected with tight margin)
                _log_fn = log.info if code == 2005 else log.error
                _log_fn("  ❌ order FAIL: %s side=%d vol=%d → code=%s  %s",
                         contract, side, vol, code, msg_text)
                log.error("  [MEXC] order failed payload: http_status=%s code=%s msg=%s body=%s",
                          res.get("http_status") or res.get("status"), code, msg_text, res)
                if code in (2005, 2018):
                    self._insufficient_symbol_until[symbol] = time.time() + 180
                if code in (2011, 2013, 2019, 429, 400):
                    self._bad_symbols.add(symbol)
            return {
                "success": success,
                "data": res.get("data"),
                "code": code,
                "order_id": order_id if success else None,
                "msg": res.get("message") or res.get("msg", ""),
                "http_status": res.get("http_status") or res.get("status"),
                "body": res,
                "contracts": vol,
                "contractSize": contract_size,
                "amount": base_amount,
                "notional": notional,
            }
        except requests.HTTPError as e:
            log.error("  ❌ order HTTP error: %s side=%d → %s", contract, side, e)
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            try:
                body_text = getattr(resp, "text", None) if resp is not None else None
            except Exception:
                body_text = None
            body_json = {}
            if body_text:
                try:
                    parsed = json.loads(body_text)
                    if isinstance(parsed, dict):
                        body_json = parsed
                except Exception:
                    body_json = {}
            code = body_json.get("code", status if status is not None else -1)
            msg = body_json.get("message") or body_json.get("msg") or str(e)
            log.error("  [MEXC] order failed payload: http_status=%s code=%s msg=%s body=%s",
                      status, code, msg, body_text)
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "code": code,
                "msg": msg,
                "http_status": status,
                "body": body_json or body_text,
                "order_id": None,
            }

    def close_all(self, symbol: str) -> None:
        contract = self._contract_name(symbol)
        if contract is None:
            return
        try:
            positions = self._req("GET", "/api/v1/private/position/open_positions").get("data", [])
            matching = [p for p in positions if p.get("symbol") == contract]
            for i, pos in enumerate(matching):
                vol = int(pos.get("holdVol", 0))
                if vol > 0:
                    # MEXC: positionType 1=long, 2=short; close side 4=close long, 2=close short.
                    side = 4 if pos.get("positionType") == 1 else 2
                    lev = int(pos.get("leverage", 2))
                    # FIX v6: кэшируем leverage чтобы place_order не делал
                    # дополнительный GET запрос → экономим 1 API call на ордер
                    self._pos_leverage_cache[symbol] = lev
                    self.place_order(symbol, side, vol, leverage=lev)
                    # Rate limiting обеспечивается глобальным rate limiter в _req()
        except Exception as e:
            log.warning("close_all %s: %s", contract, e)


# ══════════════════════════════════════════════════════════════════════════════
# ДАШБОРДЫ
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(portfolios: Dict[str, PaperPortfolio],
                   price_history: List[dict],
                   output_dir: str, run_ts: str,
                   warmup_end: int = 0) -> None:
    """Обновляемый PNG-дашборд. warmup_end — граница прогрева на графиках."""
    if len(price_history) < 2:
        return

    symbols  = sorted(price_history[0].keys()) if price_history else []
    n_bars   = len(price_history)
    palette  = plt.cm.tab20(np.linspace(0,1,max(len(portfolios),1)))

    from matplotlib.figure import Figure as _Fig
    fig = _Fig(figsize=(26, 20))
    fig.suptitle(f"MEXC Paper Trading  [{run_ts}]",
                 fontsize=15, fontweight="bold", y=0.998)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.25,
                           height_ratios=[1.0, 1.2, 1.5])

    # ── ROW 0: Цены ───────────────────────────────────────────────────────────
    ax_px = fig.add_subplot(gs[0, :])
    sym_clr = plt.cm.tab10(np.linspace(0,0.9,min(len(symbols),10)))
    for i, sym in enumerate(symbols[:10]):
        vals = [ph.get(sym, np.nan) for ph in price_history]
        base = next((v for v in vals if v and v>0), None)
        if base:
            norm = [v/base if (v and v>0) else np.nan for v in vals]
            ax_px.plot(range(n_bars), norm, lw=1.4, alpha=0.85,
                       label=sym, color=sym_clr[i%10])
    if warmup_end > 0:
        ax_px.axvline(warmup_end, color="gray", ls=":", lw=1.5, alpha=0.7,
                      label=f"warmup end ({warmup_end})")
    ax_px.set_title("Crypto Prices (normalized)", fontsize=11, fontweight="bold")
    ax_px.set_ylabel("Norm. price"); ax_px.grid(True, alpha=0.2)
    ax_px.legend(loc="upper left", fontsize=7, ncol=6)
    ax_px.set_xlabel("Bar #")

    # ── ROW 1: Портфели (нормализованные — все из 0% после прогрева) ─────────
    ax_pf = fig.add_subplot(gs[1, :])
    sorted_pf = sorted(portfolios.items(),
                       key=lambda kv: (kv[1].history[-1] if kv[1].history else 0),
                       reverse=True)
    initial = sorted_pf[0][1].initial_capital if sorted_pf else 10_000
    for idx, (name, pf) in enumerate(sorted_pf):
        h = pf.history
        if len(h) < 2: continue
        # Берём только live-часть (после прогрева)
        h_live = h[warmup_end:] if warmup_end < len(h) else h
        if len(h_live) < 1: continue
        # FIX: Нормализуем по initial_capital → все кривые стартуют из одной точки
        base = h_live[0] if h_live[0] > 0 else initial
        h_norm = [v / base * initial for v in h_live]
        ret = (h_live[-1] / base - 1) * 100
        x = range(warmup_end, warmup_end + len(h_norm))
        ax_pf.plot(x, h_norm,
                   label=f"{idx+1:02d}.{name}({ret:+.0f}%)",
                   lw=1.6, alpha=0.85, color=palette[idx%20])
    ax_pf.axhline(initial, color="black", ls="--", lw=1.5, alpha=0.5)
    if warmup_end > 0:
        ax_pf.axvline(warmup_end, color="gray", ls=":", lw=1.5, alpha=0.7)
    ax_pf.set_title("Portfolio Values — paper trading (normalized, all start from same point)",
                    fontsize=11, fontweight="bold")
    ax_pf.set_yscale("log"); ax_pf.grid(True, alpha=0.25)
    ax_pf.legend(loc="upper left", fontsize=6, ncol=4)
    ax_pf.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M" if x>=1e6 else f"{x/1e3:.0f}K"))
    ax_pf.set_xlabel("Bar #")

    # ── ROW 2 LEFT: Returns (live only — from warmup_end) ────────────────────
    ax_bar = fig.add_subplot(gs[2, 0])
    names = [n for n,_ in sorted_pf][::-1]
    rets = []
    for _, pf in sorted_pf:
        if not pf.history:
            rets.append(0.0)
            continue
        h_live = pf.history[warmup_end:] if warmup_end < len(pf.history) else pf.history
        base = h_live[0] if (h_live and h_live[0] > 0) else pf.initial_capital
        rets.append((h_live[-1] / base - 1) * 100)
    rets = rets[::-1]
    clrs  = ["#2ecc71" if r>0 else "#e74c3c" for r in rets]
    bars  = ax_bar.barh(names, rets, color=clrs, alpha=0.85, height=0.7)
    ax_bar.axvline(0, color="black", lw=1.2)
    ax_bar.set_title("Return %", fontsize=11, fontweight="bold")
    ax_bar.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, rets):
        lbl = f"{val:+.1f}%" if abs(val)>=0.1 else ""
        xpos = val + abs(val)*0.02 if val>=0 else val - abs(val)*0.02
        ax_bar.text(xpos, bar.get_y()+bar.get_height()/2,
                    lbl, va="center", fontsize=7,
                    ha="left" if val>=0 else "right", fontweight="bold")

    # ── ROW 2 RIGHT: Stats ────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.axis("off")
    HDR  = f"{'Strategy':<22}{'Final':>9}{'Ret%':>8}{'MaxDD%':>8}{'Sharpe':>8}{'Trades':>8}{'Fees':>10}"
    sep  = "-" * len(HDR)
    rows = ""
    for name, pf in sorted_pf:
        st = pf.stats()
        if not st: continue
        rows += (f"{name:<22}{st['final']:>9,.0f}{st['ret']:>+7.1f}%"
                 f"{st['maxdd']:>7.1f}%{st['sharpe']:>8.2f}"
                 f"{st['trades']:>8}{st['total_fees']:>10,.0f}\n")
    live_bars = n_bars - warmup_end
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    ax_tbl.text(0.01, 0.99,
                f"PAPER TRADING STATS  [{now_str}]\n"
                f"Warmup: {warmup_end} bars  |  Live: {live_bars} bars\n\n"
                f"{HDR}\n{sep}\n{rows}",
                transform=ax_tbl.transAxes, fontsize=7.5, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#fffde7", alpha=0.95))

    out = os.path.join(output_dir, "dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.clf()
    log.info("  [dashboard] → %s", out)


def plot_compound_growth(portfolios: Dict[str, PaperPortfolio], output_dir: str,
                         warmup_end: int = 0) -> None:
    from matplotlib.figure import Figure as _Fig
    fig = _Fig(figsize=(18, 8))
    ax  = fig.add_subplot(111)
    palette = plt.cm.tab20(np.linspace(0,1,max(len(portfolios),1)))
    for idx, (name, pf) in enumerate(sorted(portfolios.items(),
            key=lambda kv: kv[1].history[-1] if kv[1].history else 0, reverse=True)):
        if len(pf.history) < 2: continue
        # FIX: Только live-часть, нормализация по initial_capital → все из 0%
        h_live = pf.history[warmup_end:] if warmup_end < len(pf.history) else pf.history
        if len(h_live) < 2: continue
        h      = np.array(h_live)
        # Используем initial_capital как базу — все стартуют из 0% одинаково
        base   = h[0] if h[0] > 0 else pf.initial_capital
        cumret = (h / base - 1) * 100
        final_ret = cumret[-1]
        ax.plot(range(len(h)), cumret, lw=1.6,
                label=f"{name} ({final_ret:+.1f}%)", color=palette[idx%20])
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set_title("Cumulative Return % — MEXC Paper Trading (all start from 0%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Return %"); ax.set_xlabel("Bar (live)"); ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    out = os.path.join(output_dir, "compound_growth.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); fig.clf()
    log.info("  [compound_growth] → %s", out)


def _plot_combined_warmup_dashboard(prices_bars: List[dict], volumes_bars: List[dict],
                                    symbols: List[str], output_dir: str,
                                    tf_kline: str = "1m") -> None:
    if not prices_bars or not symbols:
        return

    n_bars = len(prices_bars)
    hours = n_bars / 60 if tf_kline == "1m" else n_bars
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    x = np.arange(n_bars)

    price_arr: Dict[str, np.ndarray] = {}
    volume_arr: Dict[str, np.ndarray] = {}
    for sym in symbols:
        px = np.array([b.get(sym, np.nan) for b in prices_bars], dtype=float)
        vol = np.array([b.get(sym, 0.0) for b in (volumes_bars or prices_bars)], dtype=float)
        valid = px[np.isfinite(px)]
        if len(valid) >= 2 and valid[0] > 0:
            price_arr[sym] = px
            volume_arr[sym] = vol
    if not price_arr:
        return

    changes: Dict[str, float] = {}
    norm_arrays = []
    for sym, px in price_arr.items():
        valid = px[np.isfinite(px)]
        if len(valid) < 2 or valid[0] <= 0:
            continue
        changes[sym] = (valid[-1] / valid[0] - 1) * 100
        norm_arrays.append(np.where(np.isfinite(px) & (px > 0), px / valid[0], np.nan))
    market_avg = np.nanmean(norm_arrays, axis=0) if norm_arrays else np.ones(n_bars)

    valid_mask = np.isfinite(market_avg)
    if valid_mask.sum() > 2:
        xv = x[valid_mask]
        yv = market_avg[valid_mask]
        slope, intercept = np.polyfit(xv, yv, 1)
        trend_line = slope * x + intercept
    else:
        slope = 0.0
        trend_line = market_avg.copy()

    market_ret = np.diff(market_avg) / (market_avg[:-1] + 1e-12)
    market_ret = market_ret[np.isfinite(market_ret)]
    sigma = float(np.std(market_ret)) if len(market_ret) > 1 else 0.0
    spike_pct = float(np.mean(np.abs(market_ret) > 2 * sigma) * 100) if sigma > 0 else 0.0

    def _autocorr(arr, lag=1):
        if len(arr) < lag + 2:
            return 0.0
        centered = arr - np.mean(arr)
        if np.std(centered) <= 0:
            return 0.0
        return float(np.corrcoef(centered[:-lag], centered[lag:])[0, 1])

    autocorr = _autocorr(market_ret)
    snr = float(abs(slope) / (sigma + 1e-12)) * 1000 if sigma > 0 else 0.0
    noise_score = min(10.0, max(0.0, spike_pct * 2 - min(abs(autocorr), 0.5) * 3 + 2))
    total_ret = float(np.nanmean(list(changes.values()))) if changes else 0.0
    regime_lbl = (
        "CRASH" if total_ret < -10 else
        "BEAR" if total_ret < -3 else
        "SIDEWAYS" if abs(total_ret) < 3 else
        "BULL" if total_ret < 15 else
        "STRONG BULL"
    )

    leaders = sorted(changes, key=changes.get, reverse=True)
    laggards = sorted(changes, key=changes.get)
    ranked_bars = list(dict.fromkeys(leaders[:8] + laggards[:5]))[::-1]
    curve_syms = leaders[:6]
    volume_syms = sorted(
        volume_arr,
        key=lambda sym: float(np.nanmean(volume_arr.get(sym, np.array([0.0])))),
        reverse=True,
    )[:6]

    fig = plt.figure(figsize=(24, 16), facecolor="#0D1117")
    fig.suptitle(
        f"Warmup Dashboard | {regime_lbl} | Noise {noise_score:.1f}/10 | "
        f"{n_bars} bars ({hours:.0f}h x {tf_kline}) | {ts_str}",
        fontsize=14, fontweight="bold", color="#E6EDF3", y=0.985,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.36, wspace=0.28,
        height_ratios=[1.25, 1.0, 0.72],
    )

    def _style(ax, title: str):
        ax.set_facecolor("#161B22")
        ax.set_title(title, color="#E6EDF3", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors="#8B949E", labelsize=8)
        ax.grid(True, alpha=0.22, color="#30363D")
        for spine in ax.spines.values():
            spine.set_color("#30363D")

    ax_market = fig.add_subplot(gs[0, :2])
    _style(ax_market, "Market Curve and Trend")
    ax_market.plot(x, market_avg, color="#58A6FF", lw=1.5, label="Market avg")
    ax_market.plot(x, trend_line, color="#F0C040", lw=1.4, ls="--", label="Trend")
    ax_market.axhline(1.0, color="#E6EDF3", lw=0.8, alpha=0.35, ls=":")
    ax_market.set_ylabel("Normalized price", color="#8B949E", fontsize=8)
    ax_market.set_xlabel("Bar", color="#8B949E", fontsize=8)
    ax_market.legend(loc="upper left", fontsize=8, frameon=False, labelcolor="#E6EDF3")

    ax_noise = fig.add_subplot(gs[0, 2])
    _style(ax_noise, "Market Noise")
    metric_labels = ["Noise", "AutoCorr", "SNR", "Spike %"]
    metric_values = [noise_score / 10.0, autocorr, min(snr, 1.0), min(spike_pct / 10.0, 1.0)]
    metric_colors = [
        "#F85149" if noise_score >= 6.5 else "#D29922" if noise_score >= 3.5 else "#3FB950",
        "#BC8CFF", "#58A6FF", "#F85149",
    ]
    y_pos = np.arange(len(metric_labels))[::-1]
    ax_noise.barh(y_pos, metric_values, color=metric_colors, alpha=0.85)
    ax_noise.set_yticks(y_pos)
    ax_noise.set_yticklabels(metric_labels, color="#E6EDF3")
    ax_noise.axvline(0, color="#E6EDF3", lw=0.6, alpha=0.3)
    ax_noise.set_xlim(min(-1.0, min(metric_values) - 0.1), 1.05)
    raw_text = [f"{noise_score:.1f}/10", f"{autocorr:+.3f}", f"{snr:.3f}", f"{spike_pct:.1f}%"]
    for idx, (value, text) in enumerate(zip(metric_values[::-1], raw_text[::-1])):
        ax_noise.text(value + 0.03, idx, text, color="#E6EDF3", va="center", fontsize=8)

    ax_rank = fig.add_subplot(gs[1, 0])
    _style(ax_rank, "Warmup Winners and Losers")
    vals = [changes.get(sym, 0.0) for sym in ranked_bars]
    colors = ["#3FB950" if val >= 0 else "#F85149" for val in vals]
    ax_rank.barh(ranked_bars, vals, color=colors, alpha=0.86)
    ax_rank.axvline(0, color="#E6EDF3", lw=0.8, alpha=0.45)
    ax_rank.set_xlabel("Change %", color="#8B949E", fontsize=8)
    for idx, val in enumerate(vals):
        ax_rank.text(val, idx, f" {val:+.1f}%", color="#E6EDF3", va="center", fontsize=7)

    ax_curves = fig.add_subplot(gs[1, 1])
    _style(ax_curves, "Top Price Curves")
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(curve_syms), 1)))
    for idx, sym in enumerate(curve_syms):
        px = price_arr.get(sym)
        if px is None:
            continue
        valid = px[np.isfinite(px)]
        if len(valid) < 2 or valid[0] <= 0:
            continue
        norm = np.where(np.isfinite(px) & (px > 0), px / valid[0], np.nan)
        ax_curves.plot(x, norm, lw=1.5, color=palette[idx],
                       label=f"{sym} ({changes.get(sym, 0):+.1f}%)")
    ax_curves.axhline(1.0, color="#E6EDF3", lw=0.8, alpha=0.35, ls=":")
    ax_curves.set_ylabel("Normalized price", color="#8B949E", fontsize=8)
    ax_curves.set_xlabel("Bar", color="#8B949E", fontsize=8)
    ax_curves.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#E6EDF3")

    ax_volume = fig.add_subplot(gs[1, 2])
    _style(ax_volume, "Top Volumes")
    vol_palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(volume_syms), 1)))
    for idx, sym in enumerate(volume_syms):
        vol = volume_arr.get(sym)
        if vol is None:
            continue
        window = min(60, max(1, len(vol) // 4))
        vol_s = np.convolve(vol, np.ones(window) / window, mode="same") if window > 1 else vol
        ax_volume.plot(x, vol_s, lw=1.1, alpha=0.85, color=vol_palette[idx], label=sym)
    ax_volume.set_xlabel("Bar", color="#8B949E", fontsize=8)
    ax_volume.set_ylabel("Volume", color="#8B949E", fontsize=8)
    ax_volume.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="#E6EDF3")
    ax_volume.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else
                          f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}"))

    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.set_facecolor("#161B22")
    ax_summary.axis("off")
    best = leaders[0] if leaders else "-"
    worst = laggards[0] if laggards else "-"
    leader_s = "  ".join(f"{sym}({changes.get(sym, 0):+.1f}%)" for sym in leaders[:5])
    laggard_s = "  ".join(f"{sym}({changes.get(sym, 0):+.1f}%)" for sym in laggards[:5])
    summary = (
        f"WARMUP SUMMARY [{ts_str}]\n"
        f"Bars: {n_bars} ({hours:.0f}h x {tf_kline}) | Symbols: {len(price_arr)} | "
        f"Market: {total_ret:+.2f}% | Type: {regime_lbl} | Noise: {noise_score:.1f}/10\n"
        f"Best: {best} ({changes.get(best, 0):+.2f}%) | "
        f"Worst: {worst} ({changes.get(worst, 0):+.2f}%) | "
        f"AutoCorr: {autocorr:+.3f} | SNR: {snr:.3f} | Spike%: {spike_pct:.1f}%\n"
        f"Leaders:  {leader_s}\n"
        f"Laggards: {laggard_s}"
    )
    ax_summary.text(
        0.015, 0.92, summary, transform=ax_summary.transAxes,
        fontsize=9, va="top", family="monospace", color="#E6EDF3",
        bbox=dict(boxstyle="round", facecolor="#0D1117", alpha=0.95, edgecolor="#30363D"),
    )

    out = os.path.join(output_dir, "warmup_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.clf()
    log.info("  [warmup_dashboard] -> %s", out)


def plot_warmup_dashboard(prices_bars: List[dict], volumes_bars: List[dict],
                          symbols: List[str], output_dir: str,
                          tf_kline: str = "1m") -> None:
    """
    Сохраняет дашборд загруженных исторических данных прогрева.

    Layout:
      ROW 0  Цены всех символов (normalized, все бары прогрева)
      ROW 1  Топ-10 по доходности за период прогрева (bar chart)
      ROW 2  Объёмы топ-10 символов (area chart)
      ROW 3  Сводная таблица: цена start/end, change%, avg_volume
    """
    if not prices_bars or not symbols:
        return
    _plot_combined_warmup_dashboard(
        prices_bars=prices_bars,
        volumes_bars=volumes_bars,
        symbols=symbols,
        output_dir=output_dir,
        tf_kline=tf_kline,
    )
    return

    n_bars  = len(prices_bars)
    n_syms  = len(symbols)
    palette = plt.cm.tab20(np.linspace(0, 1, min(n_syms, 20)))

    # ── Собираем массивы ──────────────────────────────────────────────────────
    price_arr  : Dict[str, np.ndarray] = {}
    volume_arr : Dict[str, np.ndarray] = {}
    for sym in symbols:
        px  = np.array([b.get(sym, np.nan) for b in prices_bars], dtype=float)
        vol = np.array([b.get(sym, 0.0)    for b in (volumes_bars or prices_bars)],
                       dtype=float)
        if np.any(np.isfinite(px)) and px[np.isfinite(px)][0] > 0:
            price_arr[sym]  = px
            volume_arr[sym] = vol

    if not price_arr:
        return

    # Доходность за период прогрева
    changes = {}
    for sym, px in price_arr.items():
        valid = px[np.isfinite(px)]
        if len(valid) >= 2 and valid[0] > 0:
            changes[sym] = (valid[-1] / valid[0] - 1) * 100

    top10_by_ret = sorted(changes, key=changes.get, reverse=True)[:10]
    bot10        = sorted(changes, key=changes.get)[:5]

    # ── Фигура ────────────────────────────────────────────────────────────────
    from matplotlib.figure import Figure as _Fig
    fig = _Fig(figsize=(26, 24))
    ts_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    hours   = n_bars / 60 if tf_kline == "1m" else n_bars
    fig.suptitle(
        f"MEXC Warmup Data  [{ts_str}]  —  {n_bars} баров ({hours:.0f}ч)  "
        f"×  {len(price_arr)} символов",
        fontsize=14, fontweight="bold", y=0.998,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.25,
                           height_ratios=[1.4, 1.0, 1.0, 1.2])

    x = np.arange(n_bars)

    # ── ROW 0: Все цены normalized (full width) ───────────────────────────────
    ax_all = fig.add_subplot(gs[0, :])
    for idx, sym in enumerate(list(price_arr.keys())[:20]):
        px   = price_arr[sym]
        base = px[np.isfinite(px)][0] if np.any(np.isfinite(px)) else 1.0
        norm = np.where(np.isfinite(px) & (base > 0), px / base, np.nan)
        ax_all.plot(x, norm, lw=0.9, alpha=0.75,
                    label=sym, color=palette[idx % 20])
    ax_all.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.4)
    ax_all.set_title(f"Исторические цены (normalized to 1.0) — {tf_kline} бары",
                     fontsize=11, fontweight="bold")
    ax_all.set_ylabel("Norm. price"); ax_all.grid(True, alpha=0.2)
    ax_all.legend(loc="upper left", fontsize=6, ncol=6)
    ax_all.set_xlabel("Bar #")

    # ── ROW 1 LEFT: Топ-10 winners bar chart ──────────────────────────────────
    ax_top = fig.add_subplot(gs[1, 0])
    top_syms = (top10_by_ret + bot10)[::-1]
    top_vals = [changes.get(s, 0) for s in top_syms]
    top_clrs = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top_vals]
    bars = ax_top.barh(top_syms, top_vals, color=top_clrs, alpha=0.85, height=0.7)
    ax_top.axvline(0, color="black", lw=1.2)
    ax_top.set_title("Топ winners & losers за период прогрева",
                     fontsize=11, fontweight="bold")
    ax_top.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, top_vals):
        lbl  = f"{val:+.1f}%"
        xpos = val + abs(val)*0.02 if val >= 0 else val - abs(val)*0.02
        ax_top.text(xpos, bar.get_y() + bar.get_height()/2,
                    lbl, va="center", fontsize=7,
                    ha="left" if val >= 0 else "right", fontweight="bold")

    # ── ROW 1 RIGHT: Объёмы топ-10 (area) ────────────────────────────────────
    ax_vol = fig.add_subplot(gs[1, 1])
    vol_palette = plt.cm.tab10(np.linspace(0, 0.9, min(len(top10_by_ret), 10)))
    for idx, sym in enumerate(top10_by_ret[:10]):
        if sym not in volume_arr: continue
        vol = volume_arr[sym]
        # Сглаживание (rolling mean 60 баров)
        window = min(60, len(vol) // 4) if len(vol) > 4 else 1
        if window > 1:
            kernel = np.ones(window) / window
            vol_s  = np.convolve(vol, kernel, mode="same")
        else:
            vol_s = vol
        ax_vol.fill_between(x, vol_s, alpha=0.3, color=vol_palette[idx])
        ax_vol.plot(x, vol_s, lw=0.8, alpha=0.8,
                    label=sym, color=vol_palette[idx])
    ax_vol.set_title("Объёмы топ-10 символов (сглаж. 60 баров)",
                     fontsize=11, fontweight="bold")
    ax_vol.set_ylabel("Volume"); ax_vol.grid(True, alpha=0.2)
    ax_vol.legend(loc="upper left", fontsize=7, ncol=2)
    ax_vol.set_xlabel("Bar #")
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f"{v/1e6:.1f}M" if v >= 1e6 else
                                       f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}"))

    # ── ROW 2: Топ-8 цены детально (normalized) ──────────────────────────────
    ax_det = fig.add_subplot(gs[2, :])
    det_palette = plt.cm.tab10(np.linspace(0, 0.9, min(len(top10_by_ret), 10)))
    for idx, sym in enumerate(top10_by_ret[:8]):
        if sym not in price_arr: continue
        px   = price_arr[sym]
        base = px[np.isfinite(px)][0]
        norm = np.where(np.isfinite(px) & (base > 0), px / base, np.nan)
        chg  = changes.get(sym, 0)
        ax_det.plot(x, norm, lw=1.5, alpha=0.9,
                    label=f"{sym} ({chg:+.1f}%)", color=det_palette[idx])
    ax_det.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.4)
    ax_det.set_title("Топ-8 winners: детальные цены (normalized)",
                     fontsize=11, fontweight="bold")
    ax_det.set_ylabel("Norm. price"); ax_det.grid(True, alpha=0.2)
    ax_det.legend(loc="upper left", fontsize=8, ncol=4)
    ax_det.set_xlabel("Bar #")

    # ── ROW 3: Сводная таблица ────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    HDR  = (f"{'Symbol':<10}{'Start':>12}{'End':>12}{'Change%':>10}"
            f"{'Min':>12}{'Max':>12}{'AvgVol/bar':>14}")
    sep  = "-" * len(HDR)
    rows = ""
    all_syms_sorted = sorted(changes, key=changes.get, reverse=True)
    for sym in all_syms_sorted[:30]:
        px  = price_arr.get(sym)
        vol = volume_arr.get(sym)
        if px is None: continue
        valid = px[np.isfinite(px)]
        if len(valid) < 2: continue
        p0      = valid[0]; p1 = valid[-1]
        pmin    = float(np.nanmin(px)); pmax = float(np.nanmax(px))
        avg_vol = float(np.mean(vol)) if vol is not None and len(vol) > 0 else 0
        chg     = changes.get(sym, 0)
        rows += (f"{sym:<10}{p0:>12,.4f}{p1:>12,.4f}{chg:>+9.2f}%"
                 f"{pmin:>12,.4f}{pmax:>12,.4f}{avg_vol:>14,.0f}\n")

    summary = (
        f"WARMUP DATA SUMMARY  [{ts_str}]\n"
        f"Период: {n_bars} баров × {tf_kline}  ({hours:.0f} часов)  |  "
        f"Символов: {len(price_arr)}  |  "
        f"Лучший: {all_syms_sorted[0] if all_syms_sorted else '—'} "
        f"({changes.get(all_syms_sorted[0],0):+.1f}%)  |  "
        f"Худший: {all_syms_sorted[-1] if all_syms_sorted else '—'} "
        f"({changes.get(all_syms_sorted[-1],0):+.1f}%)\n\n"
        f"{HDR}\n{sep}\n{rows}"
    )
    ax_tbl.text(0.01, 0.99, summary,
                transform=ax_tbl.transAxes, fontsize=7.0, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#e8f4f8", alpha=0.95))

    out = os.path.join(output_dir, "warmup_data.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.clf()
    log.info("  [warmup_dashboard] → %s", out)


def plot_warmup_market_analysis(prices_bars: List[dict], volumes_bars: List[dict],
                                symbols: List[str], output_dir: str,
                                tf_kline: str = "1m", bar: int = 0) -> None:
    """
    Дашборд анализа рынка за период прогрева — аналог market_YYYY-MM.png
    из crypto_exchange.py.

    Layout:
      ROW 0  [Market Curve + Trend  (full width)]  |  [Noise Gauge]
      ROW 1  [Per-Coin Return %]  |  [Hourly Returns]  |  [Noise Metrics]
      ROW 2  [Summary Text Box (full width)]
    """
    if not prices_bars or not symbols:
        return
    _plot_combined_warmup_dashboard(
        prices_bars=prices_bars,
        volumes_bars=volumes_bars,
        symbols=symbols,
        output_dir=output_dir,
        tf_kline=tf_kline,
    )
    return

    n_bars  = len(prices_bars)
    hours   = n_bars / 60 if tf_kline == "1m" else n_bars
    ts_str  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Собираем данные ───────────────────────────────────────────────────────
    price_arr  : Dict[str, np.ndarray] = {}
    volume_arr : Dict[str, np.ndarray] = {}
    for sym in symbols:
        px  = np.array([b.get(sym, np.nan) for b in prices_bars], dtype=float)
        vol = np.array([b.get(sym, 0.0)    for b in (volumes_bars or prices_bars)], dtype=float)
        valid = px[np.isfinite(px)]
        if len(valid) >= 2 and valid[0] > 0:
            price_arr[sym]  = px
            volume_arr[sym] = vol

    if not price_arr:
        return

    # Средняя нормализованная цена рынка
    norm_arrays = []
    for sym, px in price_arr.items():
        v = px[np.isfinite(px)]
        if len(v) > 0 and v[0] > 0:
            norm = np.where(np.isfinite(px) & (px > 0), px / v[0], np.nan)
            norm_arrays.append(norm)
    market_avg = np.nanmean(norm_arrays, axis=0) if norm_arrays else np.ones(n_bars)
    x = np.arange(n_bars)

    # Линия тренда (линейная регрессия)
    valid_mask = np.isfinite(market_avg)
    if valid_mask.sum() > 2:
        xv = x[valid_mask]; yv = market_avg[valid_mask]
        slope, intercept = np.polyfit(xv, yv, 1)
        trend_line = slope * x + intercept
    else:
        trend_line = market_avg.copy()
        slope = 0.0

    # Доходности баров (returns)
    market_ret = np.diff(market_avg) / (market_avg[:-1] + 1e-12)
    market_ret = market_ret[np.isfinite(market_ret)]

    # Метрики шума
    def _autocorr(arr, lag=1):
        if len(arr) < lag + 2: return 0.0
        a = arr - np.mean(arr)
        return float(np.corrcoef(a[:-lag], a[lag:])[0, 1]) if np.std(a) > 0 else 0.0

    def _hurst(arr):
        """Упрощённый Hurst через R/S анализ."""
        if len(arr) < 20: return 0.5
        lags = [2, 4, 8, 16, 32]
        rs = []
        for lag in lags:
            if lag >= len(arr): continue
            chunks = [arr[i:i+lag] for i in range(0, len(arr)-lag, lag)]
            rs_vals = []
            for chunk in chunks:
                if len(chunk) < 2: continue
                mean_c = np.mean(chunk)
                dev    = np.cumsum(chunk - mean_c)
                r      = np.max(dev) - np.min(dev)
                s      = np.std(chunk)
                if s > 0: rs_vals.append(r / s)
            if rs_vals: rs.append((lag, np.mean(rs_vals)))
        if len(rs) < 2: return 0.5
        lags_v = np.log([r[0] for r in rs])
        rs_v   = np.log([r[1] for r in rs])
        h = float(np.polyfit(lags_v, rs_v, 1)[0])
        return float(np.clip(h, 0, 1))

    sigma     = float(np.std(market_ret)) if len(market_ret) > 1 else 0.0
    autocorr  = _autocorr(market_ret)
    hurst_h   = _hurst(market_ret)
    spike_pct = float(np.mean(np.abs(market_ret) > 2 * sigma) * 100) if sigma > 0 else 0.0
    snr       = float(abs(slope) / (sigma + 1e-12)) * 1000 if sigma > 0 else 0.0

    # Noise score (0–10): высокий autocorr → тихо, высокий spike% → шумно
    noise_score = min(10.0, max(0.0,
        spike_pct * 2                          # спайки → шум
        - min(abs(autocorr), 0.5) * 3         # структура → тихо
        + (1 - abs(hurst_h - 0.5) * 2) * 2   # близко к 0.5 → шум
        + 2                                    # базовая составляющая
    ))
    if   noise_score < 3.5: noise_label = "ТИХИЙ"
    elif noise_score < 6.5: noise_label = "ШУМНЫЙ"
    else:                   noise_label = "ХАОС"

    # Изменение цен за период
    changes = {}
    for sym, px in price_arr.items():
        v = px[np.isfinite(px)]
        if len(v) >= 2 and v[0] > 0:
            changes[sym] = (v[-1] / v[0] - 1) * 100

    total_ret = float(np.nanmean(list(changes.values()))) if changes else 0.0

    # Внутрибарные доходности (hourly returns for visualization)
    step     = max(1, 60 if tf_kline == "1m" else 1)
    bar_rets = []
    for i in range(step, n_bars, step):
        r = market_avg[i] / (market_avg[i - step] + 1e-12) - 1
        if np.isfinite(r): bar_rets.append(r * 100)

    # ── Фигура ────────────────────────────────────────────────────────────────
    from matplotlib.figure import Figure as _Fig
    import matplotlib.patches as mpatches

    fig = _Fig(figsize=(24, 18))
    fig.patch.set_facecolor("#1a1a2e")

    trend_pct  = (trend_line[-1] / (trend_line[0] + 1e-12) - 1) * 100 * (365 * 24 / hours)
    regime_lbl = ("CRASH" if total_ret < -10 else "BEAR" if total_ret < -3
                  else "SIDEWAYS" if abs(total_ret) < 3 else "BULL"
                  if total_ret < 15 else "STRONG BULL")
    fig.suptitle(
        f"Market Analysis — MEXC Warmup  |  {regime_lbl}  |  "
        f"Noise: {noise_score:.1f}/10  |  {ts_str}",
        fontsize=13, fontweight="bold", color="white", y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           height_ratios=[1.3, 1.0, 0.7])

    TEXT_KW = dict(color="white")
    GRID_KW = dict(alpha=0.15, color="white")

    # ── ROW 0 LEFT+MID: Market Curve (2 cols) ────────────────────────────────
    ax_mc = fig.add_subplot(gs[0, :2])
    ax_mc.set_facecolor("#0d1117")
    ax_mc.plot(x, market_avg, color="#4fc3f7", lw=1.4, label="Market avg")
    ax_mc.plot(x, trend_line, color="#ffd54f", lw=1.5, ls="--", label="Trend")
    ax_mc.fill_between(x, market_avg, trend_line,
                       where=market_avg >= trend_line, alpha=0.08, color="#4caf50")
    ax_mc.fill_between(x, market_avg, trend_line,
                       where=market_avg < trend_line,  alpha=0.08, color="#f44336")
    title_col = "#f44336" if total_ret < 0 else "#4caf50"
    ax_mc.set_title(
        f"Market Curve — {n_bars} bars ({hours:.0f}h) × {tf_kline}  |  "
        f"{total_ret:+.1f}%  |  Trend: {trend_pct:+.0f}%/yr  |  Type: {regime_lbl}",
        fontsize=10, fontweight="bold", color=title_col,
    )
    ax_mc.tick_params(colors="gray"); ax_mc.grid(**GRID_KW)
    ax_mc.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax_mc.set_ylabel("Norm. price", **TEXT_KW)
    for sp in ax_mc.spines.values(): sp.set_color("#333")

    # ── ROW 0 RIGHT: Noise Gauge ──────────────────────────────────────────────
    ax_g = fig.add_subplot(gs[0, 2])
    ax_g.set_facecolor("#0d1117")
    ax_g.set_aspect("equal"); ax_g.set_xlim(-1.2, 1.2); ax_g.set_ylim(-0.3, 1.2)
    ax_g.axis("off")
    # Дуга gauge
    theta = np.linspace(np.pi, 0, 300)
    for i in range(len(theta) - 1):
        t = i / (len(theta) - 1)
        col = (t * 0.8, (1 - t) * 0.7, 0.1)
        ax_g.plot([0.85 * np.cos(theta[i]), 0.85 * np.cos(theta[i+1])],
                  [0.85 * np.sin(theta[i]), 0.85 * np.sin(theta[i+1])],
                  lw=12, color=col, solid_capstyle="round")
    # Стрелка
    angle = np.pi - noise_score / 10 * np.pi
    ax_g.annotate("", xy=(0.65 * np.cos(angle), 0.65 * np.sin(angle)),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="white",
                                  lw=2.5, mutation_scale=18))
    noise_col = "#4caf50" if noise_score < 3.5 else "#ff9800" if noise_score < 6.5 else "#f44336"
    ax_g.text(0, -0.05, f"{noise_score:.1f}/10", ha="center", va="center",
              fontsize=22, fontweight="bold", color=noise_col)
    ax_g.text(0, -0.22, noise_label, ha="center", va="center",
              fontsize=11, color=noise_col, fontweight="bold")
    ax_g.set_title("Уровень шума рынка", fontsize=10, color="white")
    for lbl, ang in [("Тихий\n0-3.5", 0.80 * np.pi), ("Шумный\n3.5-6.5", 0.5 * np.pi),
                     ("Хаос\n6.5-10", 0.18 * np.pi)]:
        ax_g.text(0.95 * np.cos(ang), 0.95 * np.sin(ang), lbl,
                  ha="center", va="center", fontsize=6.5, color="gray")

    # ── ROW 1 LEFT: Per-Coin Return % ────────────────────────────────────────
    ax_ret = fig.add_subplot(gs[1, 0])
    ax_ret.set_facecolor("#0d1117")
    sorted_syms = sorted(changes, key=changes.get)
    vals = [changes[s] for s in sorted_syms]
    clrs = ["#f44336" if v < 0 else "#4caf50" for v in vals]
    ax_ret.barh(sorted_syms, vals, color=clrs, alpha=0.85, height=0.7)
    ax_ret.axvline(0, color="white", lw=0.8)
    ax_ret.set_title("Per-Coin Return %", fontsize=10, fontweight="bold", color="white")
    ax_ret.tick_params(colors="gray", labelsize=7); ax_ret.grid(**GRID_KW, axis="x")
    for sp in ax_ret.spines.values(): sp.set_color("#333")
    for bar_h, val in zip(ax_ret.patches, vals):
        lbl = f"{val:+.1f}%"
        xpos = val + abs(val)*0.04 if val >= 0 else val - abs(val)*0.04
        ax_ret.text(xpos, bar_h.get_y() + bar_h.get_height()/2,
                    lbl, va="center", fontsize=6, color="white",
                    ha="left" if val >= 0 else "right")

    # ── ROW 1 MID: Hourly Returns % ──────────────────────────────────────────
    ax_hr = fig.add_subplot(gs[1, 1])
    ax_hr.set_facecolor("#0d1117")
    if bar_rets:
        clrs_hr = ["#4caf50" if v >= 0 else "#f44336" for v in bar_rets]
        ax_hr.bar(range(len(bar_rets)), bar_rets, color=clrs_hr, alpha=0.85, width=0.8)
    ax_hr.axhline(0, color="white", lw=0.8)
    lbl_period = "Hourly" if tf_kline == "1m" else "Daily"
    ax_hr.set_title(f"{lbl_period} Returns %", fontsize=10, fontweight="bold", color="white")
    ax_hr.tick_params(colors="gray"); ax_hr.grid(**GRID_KW, axis="y")
    for sp in ax_hr.spines.values(): sp.set_color("#333")

    # ── ROW 1 RIGHT: Noise Metrics ────────────────────────────────────────────
    ax_nm = fig.add_subplot(gs[1, 2])
    ax_nm.set_facecolor("#0d1117")
    metrics = [
        ("AutoCorr", autocorr,  -1, 1,   "#ffd54f"),
        ("SNR",      min(snr,1), 0, 1,   "#4fc3f7"),
        ("Spike %",  spike_pct/10,0,1,   "#f44336"),
        ("Hurst H",  hurst_h,   0, 1,   "#4caf50"),
    ]
    y_pos = list(range(len(metrics)))[::-1]
    labels_r = [m[0] for m in metrics]
    vals_r   = [m[1] for m in metrics]
    clrs_r   = [m[4] for m in metrics]
    raw_vals = [autocorr, snr, spike_pct, hurst_h]
    ax_nm.barh(y_pos, vals_r, color=clrs_r, alpha=0.85, height=0.6)
    ax_nm.set_yticks(y_pos); ax_nm.set_yticks(y_pos)
    ax_nm.set_yticklabels(labels_r, color="gray", fontsize=8)
    ax_nm.set_title("Шумовые метрики", fontsize=10, fontweight="bold", color="white")
    ax_nm.tick_params(colors="gray"); ax_nm.grid(**GRID_KW, axis="x")
    ax_nm.set_xlim(-1.1, 1.3)
    for yi, (val, raw) in enumerate(zip(vals_r[::-1], raw_vals)):
        ax_nm.text(val + 0.05, yi, f"{raw:.3f}", va="center", fontsize=8, color="white")
    for sp in ax_nm.spines.values(): sp.set_color("#333")

    # ── ROW 2: Summary Text ───────────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[2, :])
    ax_sum.set_facecolor("#0d0d1a"); ax_sum.axis("off")
    leaders  = sorted(changes, key=changes.get, reverse=True)[:3]
    laggards = sorted(changes, key=changes.get)[:3]
    leader_s  = "  ".join(f"{s}({changes[s]:+.1f}%)" for s in leaders)
    laggard_s = "  ".join(f"{s}({changes[s]:+.1f}%)" for s in laggards)
    summary = (
        f"══════════════════════════════════════════════════════\n"
        f"  MARKET ANALYSIS  |  MEXC Warmup  |  {ts_str}\n"
        f"{'═'*54}\n"
        f"  Total: {total_ret:+.1f}%  |  Bars: {n_bars} ({hours:.0f}h × {tf_kline})"
        f"  |  Symbols: {len(price_arr)}\n"
        f"  MarketType: {regime_lbl}  |  Noise: {noise_score:.1f}/10 [{noise_label}]\n"
        f"  AutoCorr: {autocorr:.3f}  SNR: {snr:.3f}  Spike%: {spike_pct:.1f}%  "
        f"Hurst H: {hurst_h:.3f}\n"
        f"  Trend slope: {slope:+.5f}/bar  ({trend_pct:+.0f}%/yr annualised)\n"
        f"{'─'*54}\n"
        f"  Leaders:  {leader_s}\n"
        f"  Laggards: {laggard_s}\n"
        f"══════════════════════════════════════════════════════"
    )
    ax_sum.text(0.02, 0.95, summary, transform=ax_sum.transAxes,
                fontsize=8.5, va="top", family="monospace", color="#e0e0e0",
                bbox=dict(boxstyle="round", facecolor="#0a0a1a", alpha=0.9,
                          edgecolor="#333"))

    out = os.path.join(output_dir, "warmup_market_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.clf()
    log.info("  [warmup_market_analysis] → %s", out)



def plot_usd_dashboard_mexc(portfolios: Dict[str, "PaperPortfolio"],
                             prices: dict, output_dir: str,
                             bar: int = 0, warmup_end: int = 0) -> None:
    """
    USD дашборд для MEXC paper/live:
      ─ Левая часть: Cash USDT vs Крипто-активы (два отдельных значения)
      ─ Правая часть: детальная таблица со всеми позициями
    Сохраняет usd_dashboard.png в output_dir.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        live_bar = max(0, bar - warmup_end)

        fig = plt.figure(figsize=(20, 11))
        fig.suptitle(f"USD Дашборд  [{now_str}]  live_bar={live_bar}",
                     fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                               height_ratios=[1.2, 1.0])

        names   = list(portfolios.keys())
        n       = len(names)
        initial = portfolios[names[0]].initial_capital if names else 10000.0

    # Собираем USD-разбивку по агентам
        breakdowns = {}
        for aname, pf in portfolios.items():
            try:
                bd = pf.get_usd_breakdown(prices)
            except Exception:
                bd = {"cash_usdt": pf.cash, "crypto_value": 0.0,
                      "locked_profit": pf._locked_profit,
                      "total_active": pf.portfolio_value(prices),
                      "total_with_locked": pf.portfolio_value(prices) + pf._locked_profit,
                      "spot_positions": {}, "fut_positions": {}}
            breakdowns[aname] = bd

        # ── [0, :] PV history curves (live only, all start from 0%) ─────
        ax_pv = fig.add_subplot(gs[0, :])
        cmap  = plt.get_cmap("tab10")
        for i, (aname, pf) in enumerate(portfolios.items()):
            h = pf.history
            if len(h) < 2: continue
            # Берём только live-часть (после прогрева), нормализуем по первому live-значению
            h_live = h[warmup_end:] if warmup_end < len(h) else h
            if len(h_live) < 2: continue
            base = h_live[0] if h_live[0] > 0 else initial
            ret_h = [(v / base - 1) * 100 for v in h_live]
            clr   = cmap(i % 10)
            ax_pv.plot(ret_h, lw=1.6, color=clr,
                       label=f"{aname} {ret_h[-1]:+.2f}%")
        ax_pv.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax_pv.set_title("Доходность % — история портфелей", fontsize=10)
        ax_pv.set_xlabel("Bar"); ax_pv.set_ylabel("Return %")
        ax_pv.legend(fontsize=8, loc='upper left'); ax_pv.grid(True, alpha=0.25)

        # ── [1, 0] Stacked bar: Cash USDT vs Crypto ────────────────────────
        ax_bar = fig.add_subplot(gs[1, 0])
        cash_v   = np.array([breakdowns[a]["cash_usdt"]    for a in names])
        crypto_v = np.array([breakdowns[a]["crypto_value"] for a in names])
        locked_v = np.array([breakdowns[a]["locked_profit"] for a in names])
        y_pos    = np.arange(n)

        ax_bar.barh(y_pos, cash_v,   left=0,               color='#27ae60', alpha=0.85, label='Cash USDT')
        ax_bar.barh(y_pos, crypto_v, left=cash_v,          color='#e67e22', alpha=0.75, label='Крипто-активы')
        ax_bar.barh(y_pos, locked_v, left=cash_v+crypto_v, color='#8e44ad', alpha=0.70, label='Locked')
        ax_bar.set_yticks(y_pos); ax_bar.set_xticklabels([]); ax_bar.set_xlabel("USDT")
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.axvline(initial, color='red', ls='--', lw=1, alpha=0.5)
        ax_bar.legend(fontsize=7); ax_bar.grid(True, axis='x', alpha=0.3)
        ax_bar.set_title("Cash USDT vs Крипто vs Locked", fontsize=9, fontweight='bold')
        for i, aname in enumerate(names):
            bd    = breakdowns[aname]
            total = bd["total_with_locked"]
            ax_bar.text(total + initial*0.01, i,
                        f"${total:,.1f}", va='center', fontsize=7.5,
                        color='#1a5e2e' if total >= initial else '#7a1a1a')

        # ── [1, 1] USD таблица по агентам ─────────────────────────────────
        ax_tbl = fig.add_subplot(gs[1, 1])
        ax_tbl.axis('off')
        HDR  = f"{'Агент':<18} {'Cash$':>8} {'Крипто$':>9} {'Lock$':>7} {'Ret%':>7}"
        sep  = '─' * len(HDR)
        rows = [f"  СЧЁТ В USD  [{now_str}]", sep, HDR, sep]
        for aname in names:
            bd  = breakdowns[aname]
            ret = (bd["total_with_locked"] / initial - 1) * 100
            rows.append(f"{aname[:18]:<18} {bd['cash_usdt']:>8.2f} "
                        f"{bd['crypto_value']:>9.2f} {bd['locked_profit']:>7.2f} "
                        f"{ret:>+7.2f}%")
        rows.append(sep)
    # Итоговая строка
        t_cash   = sum(breakdowns[a]["cash_usdt"]     for a in names)
        t_crypto = sum(breakdowns[a]["crypto_value"]  for a in names)
        t_locked = sum(breakdowns[a]["locked_profit"] for a in names)
        t_tot    = t_cash + t_crypto + t_locked
        t_ret    = (t_tot/(n*initial) - 1)*100 if n else 0
        rows.append(f"{'ИТОГО':<18} {t_cash:>8.2f} {t_crypto:>9.2f} "
                    f"{t_locked:>7.2f} {t_ret:>+7.2f}%")

    # Состояние фиксации прибыли
        from mexc_connector import PROFIT_LOCK_ENABLED, DAY_STOP_ENABLED, PEAK_STOP_ENABLED
        rows.append("")
        rows.append(f"ProfitLock: {'ON' if PROFIT_LOCK_ENABLED else 'off'}  "
                    f"DayStop: {'ON' if DAY_STOP_ENABLED else 'off'}  "
                    f"PeakStop: {'ON' if PEAK_STOP_ENABLED else 'off'}")

        ax_tbl.text(0.01, 0.99, "\n".join(rows), transform=ax_tbl.transAxes,
                    fontsize=8, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.95))

        # ── [1, 2] Открытые позиции по агентам ────────────────────────────
        ax_pos = fig.add_subplot(gs[1, 2])
        ax_pos.axis('off')
        pos_lines = [f"ОТКРЫТЫЕ ПОЗИЦИИ:", "─" * 36]
        for aname in names:
            bd   = breakdowns[aname]
            spot = bd["spot_positions"]
            futs = bd["fut_positions"]
            if spot or futs:
                pos_lines.append(f"{aname}:")
                for sym, sp in spot.items():
                    pos_lines.append(f"  SPOT  {sym:<8} ${sp['value']:>8.2f}")
                for sym, fp in futs.items():
                    pos_lines.append(f"  {fp['side'].upper()[:5]:<5} {sym:<8} "
                                     f"${fp['value']:>8.2f} ({fp.get('pnl_pct', 0.0):+.2f}%)")
        if len(pos_lines) == 2:
            pos_lines.append("  (нет открытых позиций)")

        ax_pos.text(0.01, 0.99, "\n".join(pos_lines), transform=ax_pos.transAxes,
                    fontsize=8, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#fff9e7', alpha=0.95))

        out = os.path.join(output_dir, "usd_dashboard.png")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        log.info("  [usd_dashboard] → %s", out)
    except Exception as e:
        log.warning("  [usd_dashboard] failed: %s", e)


def save_stats_json(portfolios, output_dir, bar, warmup_end=0):
    data = {
        "bar": bar, "warmup_end": warmup_end, "live_bars": max(0, bar - warmup_end),
        "timestamp": datetime.utcnow().isoformat(),
        "agents": {name: pf.stats() for name, pf in portfolios.items()},
    }
    with open(os.path.join(output_dir, "stats_live.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_history_csv(portfolios, output_dir):
    try:
        import pandas as pd
        maxlen = max((len(pf.history) for pf in portfolios.values()), default=0)
        rows = {n: pf.history + [pf.history[-1]]*(maxlen-len(pf.history))
                for n, pf in portfolios.items()}
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "portfolio_history.csv"), index_label="bar")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# МОСТ
# ══════════════════════════════════════════════════════════════════════════════

class AgentMexcBridge:
    """
    Связывает агентов crypto_agents.py с MEXC.

    Фаза 1 — ПРОГРЕВ (warmup):
      Загружает WARMUP_BARS исторических klines с MEXC и прогоняет их
      через agent.act() без исполнения ордеров. Агенты накапливают
      историю цен и прогревают свои индикаторы (EMA, RSI, ATR...).

    Фаза 2 — LIVE:
      bar_index продолжается с конца прогрева → агенты торгуют.
    """

    DASHBOARD_EVERY = 10   # дашборды (PNG) каждые N live-баров
    CSV_EVERY       = 1    # CSV сохраняется каждый бар (для актуального лога)

    def __init__(self, agents: dict, cfg: dict,
                 mode=TRADING_MODE, api_key=API_KEY, api_secret=API_SECRET,
                 output_dir: str = None, direct_client=None):
        self.agents      = agents
        self.mode        = mode
        self._bar        = 0
        self._warmup_end = 0
        self._price_hist : List[dict] = []
        self._volume_hist: List[dict] = []
        self._last_data_error_at = 0.0
        self._last_data_error_reason = ""
        # FIX v4: direct_client (MexcDirectClient, SHA256) как резервный источник
        # баланса если MexcFuturesClient.account_assets() вернул 0.
        self._direct_client = direct_client

        self.initial_capital   = cfg["initial_capital"]
        self.trade_fraction    = cfg["trade_fraction"]
        self.leverage          = cfg["leverage"]
        self.poll_interval     = cfg["poll_interval"]
        self.liquidity_min_adv = cfg["liquidity_min_adv"]
        self.spot_fee          = cfg["spot_fee"]
        self.futures_fee       = cfg["futures_fee"]
        self.slippage          = cfg["slippage"]
        self._tf_kline         = cfg.get("tf_kline", "1m")
        self._cfg_bar          = cfg.get("bar", 60)   # BAR для расчёта интервалов

        # Символы
        if cfg["symbols"]:
            self.symbols = cfg["symbols"]
        else:
            # symbols=all → берём топ-N по объёму (без мусорных монет)
            self.symbols = fetch_top_symbols(TOP_N_SYMBOLS)
        if mode == "demo_futures":
            self.symbols = [s for s in self.symbols if s in ("BTC","ETH")]

    # Папка вывода
        if output_dir:
            self.output_dir = output_dir
        else:
            ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base = str(PROJECT_ROOT)
            self.output_dir = os.path.join(base, "Results", "MEXC", ts)
        os.makedirs(self.output_dir, exist_ok=True)
        log.info("📁 Результаты: %s", self.output_dir)

        # ── Лог в файл (trading.log рядом с результатами) ────────────────────
        _log_path = os.path.join(self.output_dir, "trading.log")
        _fh = logging.FileHandler(_log_path, encoding="utf-8")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s  [%(levelname)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        # Добавляем в root-логгер чтобы захватить все модули (crypto_agents и т.д.)
        _root = logging.getLogger()
        # Удаляем старый FileHandler в тот же путь если он уже есть (при рестарте)
        for _old in list(_root.handlers):
            if isinstance(_old, logging.FileHandler) and \
               getattr(_old, 'baseFilename', '') == os.path.abspath(_log_path):
                _root.removeHandler(_old)
                _old.close()
        _root.addHandler(_fh)
        log.info("📝 Лог: %s", _log_path)

        # ── Funding Data Fetcher ──────────────────────────────────────────────
        self.funding = None
        if _HAS_FUNDING and mode in ("paper", "demo_futures", "live_spot", "live_futures"):
            self.funding = FundingDataFetcher(symbols=self.symbols)
            self.funding.fetch_all()   # первичное заполнение синхронно
            self.funding.start()       # фоновое обновление
            log.info("[funding] Fetcher запущен для %d символов", len(self.symbols))
            # Передаём fetcher MEXC-агентам
            if _HAS_MEXC_AGENTS:
                try:
                    _set_mexc_fetcher(self.funding)
                    _configure_mexc_agents(
                        BAR=cfg.get("bar", 60),
                        TRADE_FRACTION=cfg["trade_fraction"],
                        INITIAL_CAPITAL=cfg["initial_capital"],
                        LEVERAGE=float(cfg["leverage"]),
                    )
                except Exception as _e:
                    log.warning("[panteon_agents] configure: %s", _e)

        # Paper-портфели
        self.paper_pf: Dict[str, PaperPortfolio] = {}
        if mode == "paper":
            # Пробуем получить реальный баланс MEXC чтобы paper-капитал
            # совпадал с реальным счётом (как в exchange_api_runtime.py).
            # Если ключи не заданы или API недоступен — используем paper_capital из settings.
            _paper_cap = cfg.get("paper_capital", cfg["initial_capital"])
            if api_key and api_key not in ("your_api_key", ""):
                try:
                    from exchange_api_runtime import MexcDirectClient as _MDC
                    _dc = _MDC(api_key, api_secret)
                    _snap = _dc.get_full_snapshot()
                    _fut_eq = _snap.get("futures", {}).get("equity", 0.0)
                    _spot_v = _snap.get("spot", {}).get("total_value", 0.0)
                    _real   = _fut_eq + _spot_v
                    if _real > 1.0:
                        _paper_cap = _real
                        log.info(
                            "💰 Paper-капитал авто-синхронизирован с реальным счётом: "
                            "$%.4f (фьючерсы=$%.4f + спот=$%.4f)",
                            _real, _fut_eq, _spot_v,
                        )
                    else:
                        log.info("💰 Paper-капитал из settings: $%.2f", _paper_cap)
                except Exception as _ce:
                    log.info("💰 Paper-капитал из settings: $%.2f  (авто-детект: %s)",
                             _paper_cap, _ce)
            else:
                log.info("💰 Paper-капитал: $%.2f  (API-ключ не задан, авто-детект пропущен)",
                         _paper_cap)

            self.initial_capital = _paper_cap   # обновляем для дашбордов
            for name in agents:
                self.paper_pf[name] = PaperPortfolio(
                    _paper_cap, self.trade_fraction, self.leverage,
                    self.spot_fee, self.futures_fee, self.slippage,
                )

        # API-клиенты
        if mode == "paper":
            self.spot = self.futures_client = None
            log.info("🗒  Режим PAPER — виртуальный портфель на живых ценах.")
        elif mode == "demo_futures":
            self.spot = None
            self.futures_client = MexcFuturesClient(api_key, api_secret, testnet=True)
            log.info("🧪 DEMO FUTURES (Testnet)  %s", FUTURES_TEST_BASE_URL)
            log.warning("   ⚠️  Нужен ОТДЕЛЬНЫЙ аккаунт на futures.testnet.mexc.com")
        elif mode == "live_spot":
            self.spot = MexcSpotClient(api_key, api_secret)
            self.futures_client = None
            log.warning("💵 LIVE SPOT — торговля на РЕАЛЬНЫЕ деньги!")
        elif mode == "live_futures":
            self.spot = None
            self.futures_client = MexcFuturesClient(api_key, api_secret, testnet=False)
            log.warning("🔴 LIVE FUTURES — торговля фьючерсами на РЕАЛЬНЫЕ деньги! "
                        "Используется НАСТОЯЩИЙ аккаунт contract.mexc.com")
        else:
            raise ValueError(f"Неизвестный режим: {mode!r}")

        log.info(
            "📋 Настройки:\n"
            "   symbols       = %s%s\n"
            "   trade_fraction= %.2f (%.0f%%)  leverage=%dx\n"
            "   poll_interval = %d сек  capital=%.0f USDT\n"
            "   fees          = spot=%.4f  fut=%.4f  slip=%.4f",
            self.symbols[:8],
            f" +{len(self.symbols)-8}" if len(self.symbols)>8 else "",
            self.trade_fraction, self.trade_fraction*100, self.leverage,
            self.poll_interval, self.initial_capital,
            self.spot_fee, self.futures_fee, self.slippage,
        )

    def _mark_data_error(self, reason: str) -> None:
        self._last_data_error_at = time.time()
        self._last_data_error_reason = str(reason or "data_error")

    def recent_data_error(self, max_age_sec: float = 120.0) -> str:
        if self._last_data_error_at <= 0:
            return ""
        if time.time() - float(self._last_data_error_at) <= float(max_age_sec or 120.0):
            return self._last_data_error_reason or "data_error"
        return ""

    # ── Прогрев ───────────────────────────────────────────────────────────────

    def warmup(self, n_bars: int = WARMUP_BARS) -> None:
        """
        Фаза 1: прогрев агентов через исторические klines.

        Два этапа:
          Тихий прогрев  — первые (n_bars - ACTIVE_WARMUP_BARS) баров:
            agent.act() вызывается, но ордера НЕ исполняются.
            Агенты накапливают историю цен и прогревают индикаторы.

          Активный прогрев — последние ACTIVE_WARMUP_BARS баров:
            Агенты торгуют на бумаге. Это критично: агенты с rebal_int=2880
            успевают совершить свой плановый ребаланс ДО начала live-торговли.
            Без этого первая реальная сделка была бы через 46 часов live.
        """
        # Активный прогрев = max rebal_int агентов (≥ 5 дней при 1m, BAR=60)
        # Берём 3 дня × BAR = 4320 баров — покрывает rebal_int=2880 (48h) и 4320 (72h)
        BAR = self._cfg_bar
        ACTIVE_WARMUP_BARS = max(3 * 24 * BAR, 1)  # 3 дня
        silent_bars = max(0, n_bars - ACTIVE_WARMUP_BARS)

        log.info("🔄 Прогрев: %d баров (≈ %d ч)  |  тихий=%d  активный=%d",
                 n_bars, n_bars // 60, silent_bars, ACTIVE_WARMUP_BARS)

        prices_bars, volumes_bars = build_warmup_data(
            self.symbols, self._tf_kline, n_bars
        )

        if not prices_bars:
            log.warning("[warmup] Нет данных — пропускаем.")
            return

        actual_bars = len(prices_bars)
        # Пересчитываем активный этап если данных меньше запрошенного
        active_start = max(0, actual_bars - ACTIVE_WARMUP_BARS)
        log.info("[warmup] Загружено %d баров × %d символов  |  "
                 "активный этап с бара %d",
                 actual_bars, len(self.symbols), active_start)

        month = datetime.utcnow().month

        for i, (prices, volumes) in enumerate(zip(prices_bars, volumes_bars)):
            self._bar += 1
            self._price_hist.append(dict(prices))
            self._volume_hist.append(dict(volumes))
            is_active = (i >= active_start)

            for agent_name, agent in self.agents.items():
                pv = (self.paper_pf[agent_name].portfolio_value(prices)
                      if is_active and agent_name in self.paper_pf
                      else self.initial_capital)
                try:
                    actions = agent.act(
                        prices=prices, volumes=volumes,
                        month=month,
                        portfolio_value=pv,
                        bar_index=self._bar,
                    )
                except Exception:
                    actions = {}

                # Активный этап: исполняем ордера на бумаге
                if is_active and self.mode == "paper" and actions:
                    for sym, action in actions.items():
                        if action == 0 or sym not in prices: continue
                        pf = self.paper_pf.get(agent_name)
                        if pf:
                            desc = pf.execute(sym, action, prices[sym], pv)
                            if desc:
                                log.debug("  [%s][WARMUP-ACTIVE] %s", agent_name, desc)

            # Снэпшот в активном этапе
            if is_active and self.mode == "paper":
                for agent_name, pf in self.paper_pf.items():
                    pf.snapshot(prices)

            # Прогресс
            if (i + 1) % max(1, actual_bars // 10) == 0:
                phase = "активный" if is_active else "тихий"
                log.info("[warmup %s]  %d/%d (%.0f%%)",
                         phase, i + 1, actual_bars, (i+1)/actual_bars*100)

        # ── Padding warmup до кратного COMMON_REBAL ─────────────────────────
        # БАГ #1 FIX: DualMomentum использует `self.t % rebal_int == 0`.
        # После warmup_end=3000 следующее срабатывание: bar 5760 = live bar 2760 (46h).
        # Решение: продолжаем прокрутку с последней известной ценой до bar 5760,
        # чтобы все модуло-таймеры сработали ДО начала live-торговли.
        # Это занимает ~2 секунды (2760 итераций × ~0.7мс).
        COMMON_REBAL = 5760   # LCM(2880, 3600) = 5760 = 96h * BAR
        if self._bar < COMMON_REBAL and prices_bars:
            last_prices  = prices_bars[-1]
            last_volumes = volumes_bars[-1] if volumes_bars else {}
            pad_to       = COMMON_REBAL
            pad_n        = pad_to - self._bar
            log.info("[warmup] Padding до bar %d (+%d баров для синхронизации таймеров)...",
                     pad_to, pad_n)
            month = datetime.utcnow().month
            for _ in range(pad_n):
                self._bar += 1
                self._price_hist.append(dict(last_prices))
                for agent_name, agent in self.agents.items():
                    pv = (self.paper_pf[agent_name].portfolio_value(last_prices)
                          if agent_name in self.paper_pf else self.initial_capital)
                    try:
                        actions = agent.act(prices=last_prices, volumes=last_volumes,
                                            month=month, portfolio_value=pv,
                                            bar_index=self._bar)
                    except Exception:
                        actions = {}
                    if self.mode == "paper" and actions:
                        for sym, action in actions.items():
                            if action == 0 or sym not in last_prices: continue
                            pf = self.paper_pf.get(agent_name)
                            if pf:
                                desc = pf.execute(sym, action, last_prices[sym], pv)
                                if desc: log.debug("  [%s][PAD] %s", agent_name, desc)
                if self.mode == "paper":
                    for agent_name, pf in self.paper_pf.items():
                        pf.snapshot(last_prices)
            log.info("[warmup] Padding завершён. bar_index=%d", self._bar)

        self._warmup_end = self._bar

        # ── Логируем ожидаемые первые сделки ─────────────────────────────────
        self._log_next_trade_schedule(BAR)

        # ── Дашборды ─────────────────────────────────────────────────────────
        log.info("[warmup] Сохраняю дашборды...")
        try:
            plot_warmup_dashboard(
                prices_bars=prices_bars, volumes_bars=volumes_bars,
                symbols=self.symbols, output_dir=self.output_dir,
                tf_kline=self._tf_kline,
            )
        except Exception as e:
            log.warning("[warmup] plot_warmup_dashboard: %s", e)

        # Активный прогрев создал позиции — сохраняем начальный дашборд
        if self.mode == "paper" and self.paper_pf:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            try:
                plot_dashboard(self.paper_pf, self._price_hist,
                               self.output_dir, ts + " [after warmup]",
                               self._warmup_end)
            except Exception as e:
                log.warning("[warmup] plot_dashboard: %s", e)

        log.info("✅ Прогрев завершён. bar_index=%d — агенты в позициях, готовы к live.",
                 self._bar)

    def _log_next_trade_schedule(self, BAR: int) -> None:
        """Выводит расписание ближайших сделок каждого агента.
        BUG FIX v3: Добавлены суб-агенты PlayerStop (STP_*) в таблицу интервалов.
        Без этого PlayerStop отображался с дефолтным CHECK_INT=240, хотя
        TrendCapture и VolBrk внутри него проверяются каждые 60 минут.
        """
        log.info("━" * 60)
        log.info("📅 Расписание следующих проверок агентов (live):")
        CHECK_INTERVALS = {
            # Стандартные агенты
            "DualMomentum":       6 * BAR,
            "CrashHunter":        4 * BAR,
            "MomentumGuard":      4 * BAR,
            "TrendCapture":       BAR,
            "AdaptivePortfolio":  3 * 24 * BAR,
            "MacroRotation":      4 * BAR,
            "TrendFollow":        4 * BAR,
            "VolatilityBreakout": BAR,
            "CrossSectMomentum":  4 * BAR,
            "MeanRevBBRSI":       BAR,
            "VolumeBreakout":     BAR,
            # BUG FIX v3: суб-агенты PlayerStop
            "PlayerStop":         BAR,          # первый сигнал — как самый быстрый суб-агент
            "STP_TrendCapture":   BAR,          # CHECK_INT=1*BAR=60 мин
            "STP_VolBrk":         4 * BAR,      # CHECK_INT=4*BAR=240 мин
            "STP_MomGuard2":      6 * BAR,      # CHECK_INT=6*BAR=360 мин
            "STP_Adaptive":       3 * 24 * BAR, # CHECK_INT=3*24*BAR (редко)
            "STP_TopAgent":       BAR,          # Genetics/TrendFollow — BAR
        }
        w = self._warmup_end
        shown = set()
        for name in self.agents:
            ci = CHECK_INTERVALS.get(name, 4 * BAR)
            last_check = (w // ci) * ci
            next_check = last_check + ci
            live_bar   = next_check - w
            if live_bar not in shown:
                shown.add(live_bar)
            log.info("  %-24s  CHECK_INT=%4d  →  live bar %4d  (~%d мин)",
                     name, ci, live_bar, live_bar)
        earliest = min(((((w // ci) * ci + ci) - w) for ci in CHECK_INTERVALS.values()
                        if ci > 0), default=60)
        log.info("  ⏱  Первая проверка через %d live-баров (~%d мин)",
                 earliest, earliest)
        log.info("━" * 60)

    # ── Рыночные данные ───────────────────────────────────────────────────────

    def _fetch_market(self) -> tuple:
        prices, volumes = {}, {}
        spot_price_error = False
        # FIX v9.1: retry до 3 раз при сетевых ошибках (ConnectTimeout, NameResolutionError)
        for _attempt in range(3):
            try:
                r = _mexc_get(f"{SPOT_BASE_URL}/api/v3/ticker/24hr", timeout=15)
                r.raise_for_status()
                sym_set = set(self.symbols)
                for item in r.json():
                    s = item["symbol"]
                    if s.endswith("USDT"):
                        base = s[:-4]
                        if base in sym_set:
                            prices[base]  = float(item.get("lastPrice",0) or 0)
                            volumes[base] = float(item.get("volume",   0) or 0)
                break  # успех — выходим из retry
            except Exception as e:
                if _attempt < 2:
                    import time as _time
                    log.warning("Ошибка цен (попытка %d/3): %s — повтор через 5с", _attempt+1, e)
                    _time.sleep(5)
                else:
                    log.error("Ошибка цен: %s", e)
                    spot_price_error = True

        if self.mode in ("demo_futures", "live_futures") and self.futures_client:
            for sym in self.symbols:
                try:
                    fut_price = self.futures_client.ticker_price(f"{sym}_USDT")
                    if fut_price > 0:
                        prices[sym] = fut_price
                except Exception:
                    pass

        if self.liquidity_min_adv > 0:
            prices  = {s:p for s,p in prices.items()
                       if volumes.get(s,0)*p >= self.liquidity_min_adv}
            volumes = {s:v for s,v in volumes.items() if s in prices}
        if spot_price_error:
            self._mark_data_error("price_error")

        return prices, volumes

    def _get_pv(self, agent_name: str, prices: dict) -> float:
        if self.mode == "paper":
            pf = self.paper_pf.get(agent_name)
            return pf.portfolio_value(prices) if pf else self.initial_capital
        elif self.mode in ("demo_futures", "live_futures"):
            try:
                cached_pv = float(getattr(self, '_last_good_futures_pv', self.initial_capital) or self.initial_capital)
                assets = self.futures_client.account_assets()
                # FIX: используем availableBalance (свободная маржа) вместо equity.
                # equity включает unrealPnL открытых позиций — если позиция убыточна,
                # FIX v9.1: используем equity как portfolio_value (весь капитал),
                # а available проверяем только при фактическом открытии позиции.
                # Было: usdt_avail*0.90 → pv=44.78 даже при equity=127 (после пополнения).
                # Теперь: pv = equity (реальная стоимость счёта с PnL позиций).
                usdt_avail = assets.get("USDT_AVAIL", 0.0)
                usdt_eq    = assets.get("USDT", 0.0)          # equity = avail + margin + unrealPnL
                self._last_futures_available_margin = float(usdt_avail or 0.0)
                self._last_futures_equity = float(usdt_eq or 0.0)
                # Используем equity как PV — отражает реальный капитал с учётом позиций
                usdt = usdt_eq if usdt_eq >= MIN_ORDER_USDT else usdt_avail
                if usdt >= MIN_ORDER_USDT:
                    self._last_good_futures_pv = float(usdt)
                    return usdt
                # account_assets вернул 0 — пробуем direct_client (если есть)
                if hasattr(self, '_direct_client') and self._direct_client is not None:
                    try:
                        snap = self._direct_client.get_full_snapshot()
                        equity = snap['futures']['equity']
                        if equity >= MIN_ORDER_USDT:
                            self._last_futures_equity = float(equity)
                            self._last_good_futures_pv = float(equity)
                            return equity
                    except Exception:
                        pass
                if cached_pv >= MIN_ORDER_USDT:
                    bar = getattr(self, '_bar', 0)
                    if not hasattr(self, '_pv_cache_warn_bar') or bar - self._pv_cache_warn_bar >= 60:
                        self._pv_cache_warn_bar = bar
                        log.warning(
                            "_get_pv: биржа вернула equity $%.4f, использую последний валидный PV $%.4f",
                            usdt, cached_pv,
                        )
                    return cached_pv
                # Оба источника вернули 0 — WARNING раз в 60 баров чтобы не спамить
                bar = getattr(self, '_bar', 0)
                if not hasattr(self, '_pv_warn_bar') or bar - self._pv_warn_bar >= 60:
                    self._pv_warn_bar = bar
                    log.warning(
                        "_get_pv: фьючерсный equity $%.4f < $%.1f. "
                        "Ордера будут отклонены! Переведите средства: "
                        "MEXC → Кошелёк → Перевод → Спот → Фьючерсы.",
                        usdt, MIN_ORDER_USDT,
                    )
                return self.initial_capital
            except Exception as e:
                log.warning("_get_pv: ошибка баланса: %s — fallback $%.2f", e, self.initial_capital)
                return self.initial_capital
        elif self.mode == "live_spot":
            try:
                bal   = self.spot.account_balance()
                total = bal.get("USDT", 0.0)
                for sym, qty in bal.items():
                    if sym != "USDT": total += qty * prices.get(sym, 0)
        # Если счёт пустой — запасной переход на initial_capital, чтобы
                # агент мог рассчитать размер позиции; MEXC вернёт ошибку
                # "insufficient balance" при попытке ордера, но сигналы
                # будут переданы корректно.
                if total < MIN_ORDER_USDT:
                    log.debug(
                        "_get_pv: реальный баланс %.4f < MIN_ORDER %.1f, "
                        "используем initial_capital=%.0f для расчёта позиций",
                        total, MIN_ORDER_USDT, self.initial_capital,
                    )
                    return self.initial_capital
                return total
            except:
                return self.initial_capital
        return self.initial_capital

    # ── Исполнение ────────────────────────────────────────────────────────────

    def _dispatch(self, agent: str, sym: str, action: int,
                  price: float, pv: float, risk_multiplier: float = 1.0) -> None:
        if action == 0: return

        if self.mode == "paper":
            pf   = self.paper_pf[agent]
            desc = pf.execute(sym, action, price, pv, risk_multiplier=risk_multiplier)
            if desc: log.info("  [%s][PAPER] %s", agent, desc)
            return

        if self.mode in ("demo_futures", "live_futures"):
            # FIX v9.2: проверяем блэклист ДО маппинга spot→fut
            # STO/SOLV/RED/BULLA проникали через spot-коды (action=1,2) минуя blacklist
            if sym in self.futures_client._FUTURES_BLACKLIST:
                log.debug("  [%s] %s в BLACKLIST → пропуск", agent, sym)
                return

            # Маппинг spot-кодов → futures-коды
            # TrendCapture/VolBreakout генерируют spot-коды (1,2,3).
            # В live_futures их нужно конвертировать в fut-эквиваленты.
            _SPOT_TO_FUT = {1: 4, 2: 5, 3: 8}
            if action in _SPOT_TO_FUT:
                mapped = _SPOT_TO_FUT[action]
                log.debug("  [%s] spot→fut: %d→%d для %s", agent, action, mapped, sym)
                action = mapped

            side_map = {4: (1, 0.5), 5: (1, 1.0), 6: (3, 0.5), 7: (3, 1.0)}
            if action == 8:
                self.futures_client.close_all(sym)
                return

            side, frac = side_map.get(action, (None, None))
            if side is None: return
            if (
                getattr(self, "exchange_name", "").upper() == "BITGET"
                and self.mode == "live_futures"
                and action in (4, 5, 6, 7)
            ):
                _stats_for_risk = getattr(self, '_stats_ref', None)
                real_pnl_pct = float(getattr(_stats_for_risk, "pnl_pct", 0.0) or 0.0)
                max_neg_positions = int(getattr(self, "max_live_positions_negative_pnl", 3) or 3)
                try:
                    open_syms = self.futures_client._get_open_position_symbols()
                except Exception:
                    open_syms = set()
                if real_pnl_pct < 0.0 and sym not in open_syms and len(open_syms) >= max_neg_positions:
                    log.info(
                        "  [%s] BITGET risk cap: pnl=%.2f%% open=%d/%d -> skip %s",
                        agent,
                        real_pnl_pct,
                        len(open_syms),
                        max_neg_positions,
                        sym,
                    )
                    return

            # ── FIX v7: Position sizing from FREE MARGIN, not total PV ─────────
            # Проблема: pv=$127, free=$30, но 5 позиций уже открыты.
            # Старый расчёт: $127 * 15% * 50% = $9.50 → больше чем реально доступно.
            # Новый расчёт: min(pv-based, free_margin * 50% для буфера)
            size_mult = min(max(float(risk_multiplier or 1.0), 0.35), 1.50)
            effective_margin = pv * self.trade_fraction * frac * size_mult
            if self.mode == "live_futures":
                try:
                    assets = self.futures_client.account_assets()
                    free_margin = float(assets.get("USDT_AVAIL", 0) or 0)
                    if free_margin < MIN_ORDER_USDT:
                        log.info(
                            "  [%s] ⛔ Нет маржи: free=$%.2f < MIN=$%.1f → пропуск %s",
                            agent, free_margin, MIN_ORDER_USDT, sym)
                        return
                    # FIX v7: потолок = 30% от свободной маржи
                    # При $30 free → max $9 на позицию → хватает на 3 позиции с буфером
                    max_margin = free_margin * 0.30
                    if effective_margin > max_margin:
                        if max_margin < MIN_ORDER_USDT:
                            log.info(
                                "  [%s] ⛔ Маржа=$%.2f×30%%=$%.2f < MIN → пропуск %s",
                                agent, free_margin, max_margin, sym)
                            return
                        log.info("  [%s] 📉 Маржа: $%.1f → $%.1f (free=$%.1f)",
                                 agent, effective_margin, max_margin, free_margin)
                        effective_margin = max_margin
                except Exception as _me:
                    log.debug("  [%s] margin check fail: %s", agent, _me)

            # Размер контракта по умолчанию
            meta = self.futures_client._get_contract_meta(sym)
            cs = max(float(meta.get("contractSize", 1.0) or 1.0), 1e-12)
            min_vol = max(1, int(meta.get("minVol", 1) or 1))
            vol_unit = max(1, int(meta.get("volUnit", 1) or 1))
            max_vol = max(min_vol, int(meta.get("maxVol", 10_000_000) or 10_000_000))
            fee_rate = float(meta.get("takerFeeRate", 0.0006) or 0.0006)
            if not bool(meta.get("apiAllowed", True)) or int(meta.get("state", 0) or 0) != 0:
                log.info("  [%s] ⛔ %s недоступен для API/state=%s → пропуск",
                         agent, sym, meta.get("state"))
                return
            if price > 0:
                # FIX v9.1: vol рассчитывается из effective_margin (≤ available*0.85)
                notional_budget = effective_margin * self.leverage * max(0.75, 1.0 - (fee_rate * 6 + 0.10))
                raw_vol = notional_budget / (price * cs)
                vol = int(math.floor(raw_vol / vol_unit) * vol_unit)
                vol = min(vol, max_vol)
                if vol < min_vol:
                    log.info(
                        "  [%s] ⛔ %s raw_vol=%.2f < minVol=%d (cs=%g) → пропуск",
                        agent, sym, raw_vol, min_vol, cs,
                    )
                    return
            else:
                log.warning("  [%s] price=0 для %s — пропуск", agent, sym)
                return

            prefix = "🔴 LIVE" if self.mode == "live_futures" else "🧪 DEMO"
            log.info("  [%s] %s → %s %s  vol=%d lev=%dx @ %.4f  pv=%.2f  risk=%.2fx",
                     agent, prefix, "LONG" if side == 1 else "SHORT",
                     sym, vol, self.leverage, price, pv, size_mult)

            # Задержка между ордерами обеспечивается глобальным rate limiter в _req()

            # place_order теперь принимает базовый символ (BTC, ETH...) не контракт
            res = self.futures_client.place_order(sym, side, vol, self.leverage)
            # Записываем результат ордера в статистику (для дашборда)
            _stats = getattr(self, '_stats_ref', None)
            order_id = res.get("order_id") or res.get("orderId")
            exchange_ok = res.get("success", False) and not res.get("skip", False)
            confirmed_ok = bool(exchange_ok and order_id)
            if _stats is not None:
                _stats.record_order(
                    sym,
                    confirmed_ok,
                    res.get("code", 0),
                    order_id=order_id,
                    order_result=res,
                    position_source="panteon_order" if confirmed_ok else "pending_order",
                )
                if confirmed_ok:
                    trade_qty = float(res.get("amount", vol) or vol)
                    fee_est = float(trade_qty) * float(price) * float(fee_rate)
                    _stats.record_trade(sym, "LONG" if side == 1 else "SHORT",
                                        trade_qty, price, fee_est, 0.0)
            if not confirmed_ok:
                if exchange_ok:
                    log.error("  [%s] order status unknown: %s %s success without order_id -> pending removed",
                              agent, sym, res)
                for _agent_obj in getattr(self, "agents", {}).values():
                    _candidate = getattr(_agent_obj, "_inner", _agent_obj)
                    if hasattr(_candidate, "_inner"):
                        _candidate = getattr(_candidate, "_inner", _candidate)
                    book = getattr(_candidate, "_open_pos", None)
                    if isinstance(book, dict):
                        rec = book.get(sym) or {}
                        if rec.get("pending_order") or rec.get("source") == "pending_order":
                            book.pop(sym, None)
            return

        if action in (4,5,6,7,8):
            log.warning("  [%s] ⚠️  Фьючерсный сигнал %s (%d) для %s ПРОИГНОРИРОВАН "
                        "— бот в режиме live_spot. Смени режим на live_futures "
                        "в exchange_api_runtime.py чтобы исполнять фьючерсные ордера.",
                        agent, ACTION_NAMES.get(action, action), action, sym)
            return
        mexc_sym = f"{sym}USDT"
        try:
            if action in (1,2):
                size_mult = min(max(float(risk_multiplier or 1.0), 0.35), 1.50)
                usdt = pv * self.trade_fraction * (0.5 if action==1 else 1.0) * size_mult
                if usdt < MIN_ORDER_USDT:
                    log.warning(
                        "  [%s] BUY %s пропущен: размер ордера %.2f USDT < минимум %.1f USDT "
                        "(pv=%.2f). Пополните USDT-баланс на MEXC.",
                        agent, sym, usdt, MIN_ORDER_USDT, pv,
                    )
                    return
                res = self.spot.buy_usdt(mexc_sym, usdt)
                order_id = res.get("orderId")
                if order_id:
                    log.info("  [%s] ✅ BUY %s %.2fU → orderId=%s", agent, sym, usdt, order_id)
                else:
                    # MEXC возвращает HTTP 200 с кодом ошибки в теле
                    log.error(
                        "  [%s] ❌ BUY %s %.2fU НЕУСПЕШЕН → code=%s  msg=%s  (полный ответ: %s)",
                        agent, sym, usdt,
                        res.get("code", "?"), res.get("msg", "нет описания"), res,
                    )
            elif action == 3:
                qty = self.spot.account_balance().get(sym, 0.0)
                if qty*price < MIN_ORDER_USDT:
                    log.warning(
                        "  [%s] SELL %s пропущен: нет позиции (%.8f шт × %.4f = %.4f USDT < %.1f). "
                        "Нечего продавать.",
                        agent, sym, qty, price, qty*price, MIN_ORDER_USDT,
                    )
                    return
                res = self.spot.sell_qty(mexc_sym, qty)
                order_id = res.get("orderId")
                if order_id:
                    log.info("  [%s] ✅ SELL %s %.8f → orderId=%s", agent, sym, qty, order_id)
                else:
                    log.error(
                        "  [%s] ❌ SELL %s НЕУСПЕШЕН → code=%s  msg=%s  (полный ответ: %s)",
                        agent, sym,
                        res.get("code", "?"), res.get("msg", "нет описания"), res,
                    )
        except requests.HTTPError as e:
            log.error("  [%s] spot HTTP err %s: %s", agent, sym, e)

    # ── Один live-цикл ────────────────────────────────────────────────────────

    def _sync_positions_with_exchange(self) -> None:
        """
        FIX v5: После каждого цикла синхронизирует PositionTracker агентов
        с реальными позициями MEXC. Без этого неудачные ордера (403/400)
        оставляют phantom-позиции в _pos → следующие сигналы пытаются их
        закрыть → повторные ошибки → вечный цикл без реальных входов.
        """
        if self.mode not in ("live_futures", "demo_futures"):
            return
        if not self._direct_client:
            return
        try:
            snap = self._direct_client.get_full_snapshot()
            real_syms = {p['symbol'] for p in snap['futures']['positions']}
        except Exception:
            return

        # Ищем PlayerStop в агентах (прямо или через обёртку)
        for agent_name, agent in self.agents.items():
            player = agent
            # Снять обёртку SignalCapturingAgent
            for attr in ('_inner', 'inner', '_agent'):
                try:
                    inner = getattr(agent, attr, None)
                    if inner is not None:
                        player = inner
                        break
                except Exception:
                    pass

            if not hasattr(player, '_pos'):
                continue

            # Обновляем PositionTracker каждого суб-агента
            for name, tracker in player._pos.items():
                # Оставляем только реально открытые позиции
                phantom_fut = tracker.futures - real_syms
                if phantom_fut:
                    log.debug("  [sync] Удаляем phantom фьючерс: %s из %s",
                              phantom_fut, name)
                    tracker.futures -= phantom_fut

                phantom_spot = tracker.spot - real_syms
                if phantom_spot:
                    tracker.spot -= phantom_spot

    def _run_cycle(self) -> None:
        self._bar += 1
        live_bar = self._bar - self._warmup_end
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        log.info("━━━ Bar #%d (live #%d)  %s  [%s] ━━━",
                 self._bar, live_bar, now, self.mode.upper())

        prices, volumes = self._fetch_market()
        if not prices:
            log.warning("Нет данных — пропуск."); return

        self._price_hist.append(dict(prices))

        sample = list(prices.items())[:8]
        def _fmt_p(p):
            """Авто-формат цены: для sub-penny используем больше знаков."""
            if p == 0:
                return "0"
            if p >= 1:
                return f"{p:,.2f}"
            if p >= 0.001:
                return f"{p:.4f}"
            if p >= 0.000001:
                return f"{p:.6f}"
            return f"{p:.2e}"
        log.info("  Цены: %s%s",
                 "  ".join(f"{s}={_fmt_p(p)}" for s, p in sample),
                 f"  +{len(prices)-8}" if len(prices) > 8 else "")

        # Funding info + запись в историю для дашборда
        if self.funding:
            g = self.funding.get_global()
            if g:
                avg_rate = g.get("avg_funding", 0)
                log.info("  Funding: avg=%.4f%%  bias=%s  longs=%.0f%%",
                         avg_rate * 100, g.get("market_bias", "?"),
                         g.get("avg_long_ratio", 0.5) * 100)
                # Записываем в TradingStats если есть ссылка
                _stats = getattr(self, '_stats_ref', None)
                if _stats is not None:
                    _stats.funding_history.append(avg_rate)

        for agent_name, agent in self.agents.items():
            pv = self._get_pv(agent_name, prices)
            log.info("  [%s] portfolio_value=%.2f USDT", agent_name, pv)
            if self.mode in ("demo_futures", "live_futures"):
                available_margin = float(getattr(self, '_last_futures_available_margin', 0.0) or 0.0)
                account_equity = float(getattr(self, '_last_futures_equity', pv) or pv)
                for target in (agent, getattr(agent, '_inner', None)):
                    if target is None:
                        continue
                    try:
                        setattr(target, '_last_available_margin', available_margin)
                        setattr(target, '_last_account_equity', account_equity)
                    except Exception:
                        pass
            try:
                actions: dict = agent.act(
                    prices=prices, volumes=volumes,
                    month=datetime.utcnow().month,
                    portfolio_value=pv, bar_index=self._bar,
                )
            except Exception as e:
                log.error("  [%s] act() err: %s", agent_name, e); continue

            n_active = sum(1 for a in actions.values() if a and a != 0)
            if n_active == 0:
                log.info("  [%s] HOLD — нет активных сигналов (все %d символов = 0)",
                         agent_name, len(actions))
            else:
                log.info("  [%s] активных сигналов: %d из %d",
                         agent_name, n_active, len(actions))

            risk_map = getattr(agent, '_last_risk_multipliers', {}) or {}

            # FIX v7: двухфазный dispatch — CLOSE → пауза → OPEN
            # Проблема: MEXC обновляет баланс с задержкой после close.
            # Если сразу открывать → "Balance insufficient" хотя маржа уже свободна.
            close_actions = [(sym, action) for sym, action in actions.items()
                             if action in (3, 8) and sym in prices]
            open_actions  = [(sym, action) for sym, action in actions.items()
                             if action not in (0, 3, 8) and action != 0 and sym in prices]

            # Фаза 1: все CLOSE
            for sym, action in close_actions:
                log.info("  [%s] → %s %s  (action=%d)",
                         agent_name, ACTION_NAMES.get(action, action), sym, action)
                self._dispatch(agent_name, sym, action, prices[sym], pv)

            # Пауза между фазами: даём MEXC обновить баланс
            if close_actions and open_actions:
                time.sleep(1.5)

            # Фаза 2: OPEN — по одному, с проверкой маржи перед каждым
            for sym, action in open_actions:
                _free = float(pv or 0.0)
                if self.mode in ("demo_futures", "live_futures") and self.futures_client is not None:
                    if self.futures_client.is_symbol_on_margin_cooldown(sym):
                        remain = self.futures_client.margin_cooldown_remaining(sym)
                        log.info("  [%s] ⏳ cooldown после insufficient margin: %s (%ds)",
                                 agent_name, sym, remain)
                        continue
                    # Свежая проверка маржи перед КАЖДЫМ open-ордером
                    try:
                        _assets = self.futures_client.account_assets()
                        _free = float(_assets.get("USDT_AVAIL", 0) or 0)
                        if _free < MIN_ORDER_USDT:
                            log.info("  [%s] ⛔ Маржа $%.2f < MIN $%.1f → пропуск %s",
                                     agent_name, _free, MIN_ORDER_USDT, sym)
                            continue
                    except Exception:
                        pass
                risk_multiplier = float(risk_map.get(sym, 1.0) or 1.0)
                log.info("  [%s] → %s %s  (action=%d)  free=$%.1f  risk=%.2fx",
                         agent_name, ACTION_NAMES.get(action, action), sym, action, _free, risk_multiplier)
                # FIX v8: передаём pv=min(pv, free_margin) чтобы dispatch не
                # рассчитывал размер от полного equity ($127) когда свободно только $10
                _effective_pv = min(pv, _free * 3) if _free > 0 else pv
                self._dispatch(
                    agent_name, sym, action, prices[sym], _effective_pv,
                    risk_multiplier=risk_multiplier,
                )

            if self.mode == "paper":
                self.paper_pf[agent_name].snapshot(prices)

        # FIX v5: синхронизируем phantom-позиции с реальными MEXC после dispatch
        if self.mode in ("live_futures", "demo_futures"):
            self._sync_positions_with_exchange()

        if self.mode == "paper":
            # CSV и JSON — каждый бар (лёгкие операции, нужны для актуального лога)
            if live_bar % self.CSV_EVERY == 0:
                save_history_csv(self.paper_pf, self.output_dir)
                save_stats_json(self.paper_pf, self.output_dir,
                                self._bar, self._warmup_end)

            # Дашборды PNG — каждые DASHBOARD_EVERY баров (тяжёлые операции)
            if live_bar % self.DASHBOARD_EVERY == 0:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                plot_dashboard(self.paper_pf, self._price_hist,
                               self.output_dir, ts, self._warmup_end)
                if not getattr(self, 'skip_compound_growth', False):
                    plot_compound_growth(self.paper_pf, self.output_dir, self._warmup_end)
                plot_usd_dashboard_mexc(self.paper_pf, prices,
                                         self.output_dir, self._bar, self._warmup_end)

        elif self.mode == "live_spot" and live_bar % self.DASHBOARD_EVERY == 0:
            # В live_spot режиме строим эквивалентные дашборды на основе
            # реального баланса MEXC (live_pf заменяет paper_pf).
            try:
                real_bal = self.spot.account_balance()
                # Создаём синтетический портфель для визуализации
                live_cap = sum(
                    real_bal.get(sym, 0) * prices.get(sym, 0)
                    for sym in real_bal if sym != "USDT"
                ) + real_bal.get("USDT", 0)
                if not hasattr(self, "_live_pf"):
                    self._live_pf = PaperPortfolio(
                        initial_capital=self.initial_capital,
                        trade_fraction=self.trade_fraction,
                        leverage=self.leverage,
                        spot_fee=self.spot_fee,
                        futures_fee=self.futures_fee,
                        slippage=self.slippage,
                    )
                # Синхронизируем реальный баланс с виртуальным портфелем
                self._live_pf.cash = real_bal.get("USDT", 0)
                for sym, qty in real_bal.items():
                    if sym != "USDT":
                        self._live_pf.portfolio[sym] = qty
                self._live_pf.snapshot(prices)
                live_pf_map = {"Live SPOT": self._live_pf}
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                plot_dashboard(live_pf_map, self._price_hist,
                               self.output_dir, ts, self._warmup_end)
                plot_usd_dashboard_mexc(live_pf_map, prices,
                                         self.output_dir, self._bar, self._warmup_end)
                save_stats_json(live_pf_map, self.output_dir,
                                self._bar, self._warmup_end)
                log.info("  📊 Live-дашборд обновлён (live_bar=%d  balance=%.2f USDT)",
                         live_bar, live_cap)
            except Exception as e:
                log.debug("Live dashboard err: %s", e)

    def run(self) -> None:
        log.info("🚀 Live-торговля  mode=%s  агентов=%d  интервал=%d сек",
                 self.mode, len(self.agents), self.poll_interval)
        try:
            while True:
                start = time.time()
                try:   self._run_cycle()
                except Exception as e: log.error("Ошибка цикла: %s", e, exc_info=True)
                s = max(0, self.poll_interval - (time.time()-start))
                log.info("  Следующий цикл через %.0f сек.", s)
                time.sleep(s)
        except KeyboardInterrupt:
            log.info("⏹  Остановка.")
            self._finalize()

    def run_once(self) -> None:
        self._run_cycle()

    def _finalize(self) -> None:
        if not self.paper_pf: return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        plot_dashboard(self.paper_pf, self._price_hist,
                       self.output_dir, ts, self._warmup_end)
        if not getattr(self, 'skip_compound_growth', False):
            plot_compound_growth(self.paper_pf, self.output_dir, self._warmup_end)
        try:
            prices_final, _ = self._fetch_market()
        except Exception:
            prices_final = {}
        plot_usd_dashboard_mexc(self.paper_pf, prices_final,
                                 self.output_dir, self._bar, self._warmup_end)
        save_stats_json(self.paper_pf, self.output_dir, self._bar, self._warmup_end)
        save_history_csv(self.paper_pf, self.output_dir)
        log.info("═"*60)
        log.info("  ИТОГОВЫЕ РЕЗУЛЬТАТЫ  |  Начальный капитал: $%.2f  |  "
                 "Live баров: %d",
                 self.initial_capital,
                 max(0, self._bar - self._warmup_end))
        log.info("  %-24s  %8s  %7s  %7s  %7s  %s",
                 "Агент", "Финал $", "P&L %", "MaxDD %", "Sharpe", "Сделок")
        log.info("  " + "─"*72)
        for name, pf in sorted(self.paper_pf.items(),
                key=lambda kv: kv[1].history[-1] if kv[1].history else 0, reverse=True):
            st = pf.stats()
            if not st: continue
            final_val = pf.history[-1] if pf.history else self.initial_capital
            log.info("  %-24s  %8.2f  %+7.2f  %7.2f  %7.2f  %d",
                     name, final_val, st["ret"], st["maxdd"], st["sharpe"], st["trades"])
        log.info("═"*60)
        log.info("📁 Результаты: %s", self.output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

AgentExchangeBridge = AgentMexcBridge


def check_connectivity() -> bool:
    try:
        _mexc_get(f"{SPOT_BASE_URL}/api/v3/ping", timeout=5).raise_for_status()
        log.info("✅ MEXC API доступен."); return True
    except Exception as e:
        log.error("❌ %s", e); return False


if __name__ == "__main__":
    """
    Запуск:
        python mexc_connector.py

    MEXC_TRADING_MODE — paper | demo_futures | live_spot  (default: paper)

    Результаты:  Results/MEXC/<timestamp>/
      dashboard.png, compound_growth.png, stats_live.json, portfolio_history.csv
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    raw_cfg    = _load_settings()
    parsed_cfg = _parse_settings(raw_cfg)

    try:
        import crypto_agents as _ca
        _ca.configure(
            BAR            = parsed_cfg["bar"],
            TRADE_FRACTION = parsed_cfg["trade_fraction"],
            LEVERAGE       = float(parsed_cfg["leverage"]),
            INITIAL_CAPITAL= parsed_cfg["initial_capital"],
            cfg            = raw_cfg,
            ACT_EVERY      = parsed_cfg["act_every"],
        )
        agents = _ca.make_agents()
        log.info("Загружено агентов: %d → %s", len(agents), list(agents.keys()))
    except ImportError:
        log.error("crypto_agents.py не найден!"); sys.exit(1)

    # Патчим баги существующих агентов (#3, #4, #5)
    if _HAS_MEXC_AGENTS:
        log.info("Применяем патчи к существующим агентам...")
        agents = _wrap_agents(agents)

    # Добавляем MEXC-специфичных агентов и игроков
    if _HAS_MEXC_AGENTS:
        from collections import OrderedDict as _OD
        mexc_ag = make_mexc_agents()
        mexc_pl = make_mexc_players()
        try:
            from crypto_agents import DDStopWrapper as _DDW
            mexc_ag = _OD({k: _DDW(v) for k, v in mexc_ag.items()})
            mexc_pl = _OD({k: _DDW(v) for k, v in mexc_pl.items()})
        except Exception:
            pass
        agents.update(mexc_ag)
        agents.update(mexc_pl)
        log.info("MEXC-агентов добавлено: %d агентов + %d игроков",
                 len(mexc_ag), len(mexc_pl))
        log.info("Всего агентов: %d", len(agents))

    if not check_connectivity():
        log.error("Нет связи с MEXC."); sys.exit(1)

    bridge = AgentMexcBridge(
        agents=agents, cfg=parsed_cfg,
        mode=TRADING_MODE, api_key=API_KEY, api_secret=API_SECRET,
    )

    # ── Фаза 1: прогрев агентов через историю ─────────────────────────────────
    bridge.warmup(n_bars=WARMUP_BARS)

    # ── Фаза 2: live-торговля ─────────────────────────────────────────────────
    log.info("Запуск live-торговли (Ctrl+C для остановки)...")
    bridge.run()
