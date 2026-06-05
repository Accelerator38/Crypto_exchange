"""
mexc_funding.py  —  Актуальные рыночные данные с MEXC Futures API

Данные которые подгружаются:
  • Funding rate (ставка финансирования) для каждого контракта
  • Open Interest (открытый интерес)
  • Long/Short ratio (соотношение лонгов/шортов)
  • Mark price и индексная цена

Использование в mexc_connector.py:
    from mexc_funding import FundingDataFetcher
    fetcher = FundingDataFetcher()
    data = fetcher.fetch_all(symbols=["BTC","ETH","SOL"])
    # data["BTC"] = {"funding_rate": -0.00006, "open_interest": 1234.5, ...}

Данные обновляются каждые REFRESH_SEC секунд (по умолчанию 60).
Агенты получают данные через параметр extra_data в act().
"""

from __future__ import annotations

import time
import logging
import threading
import os
from typing import Dict, Optional

import requests

log = logging.getLogger("mexc_funding")

def _mexc_futures_base_url() -> str:
    raw = str(os.getenv("MEXC_FUTURES_BASE_URL", "") or "").strip()
    return (raw or "https://api.mexc.com").rstrip("/")


FUTURES_BASE  = _mexc_futures_base_url()
SPOT_BASE     = "https://api.mexc.com"
REFRESH_SEC   = 60   # обновлять данные раз в минуту

# Контракт имя на MEXC: BTC → BTC_USDT
def _contract(sym: str) -> str:
    return f"{sym}_USDT"


class FundingDataFetcher:
    """
    Фоновый менеджер актуальных рыночных данных с MEXC.

    Запускает фоновый поток для периодического обновления.
    Потокобезопасен: агенты читают _cache без блокировок (атомарное присваивание dict).
    """

    def __init__(self, symbols: list = None, refresh_sec: int = REFRESH_SEC):
        self.symbols     = symbols or []
        self.refresh_sec = refresh_sec
        self._cache: Dict[str, dict] = {}     # sym → данные
        self._global: dict           = {}     # глобальные метрики рынка
        self._running                = False
        self._thread: Optional[threading.Thread] = None
        self._last_update            = 0.0

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def get(self, sym: str) -> dict:
        """Получить данные по символу. Возвращает пустой dict если нет данных."""
        return self._cache.get(sym, {})

    def get_all(self) -> Dict[str, dict]:
        """Снимок всех данных."""
        return dict(self._cache)

    def get_global(self) -> dict:
        """Глобальные рыночные метрики (total OI, avg funding, доля шортов)."""
        return dict(self._global)

    def funding_rate(self, sym: str) -> float:
        """Текущий funding rate (0.0 если нет данных)."""
        return self._cache.get(sym, {}).get("funding_rate", 0.0)

    def open_interest(self, sym: str) -> float:
        """Открытый интерес в USDT."""
        return self._cache.get(sym, {}).get("open_interest_usdt", 0.0)

    def long_short_ratio(self, sym: str) -> float:
        """Доля лонгов (0.0–1.0). 0.5 = паритет."""
        return self._cache.get(sym, {}).get("long_ratio", 0.5)

    def time_to_funding(self, sym: str) -> int:
        """Секунд до следующего funding."""
        return self._cache.get(sym, {}).get("next_funding_sec", 0)

    # ── Однократное получение данных ─────────────────────────────────────────

    def fetch_all(self, symbols: list = None) -> Dict[str, dict]:
        """Разово обновить данные по всем символам."""
        syms = symbols or self.symbols
        if not syms:
            return {}

        new_cache = {}
        for sym in syms:
            try:
                d = self._fetch_one(sym)
                if d:
                    new_cache[sym] = d
            except Exception as e:
                log.debug("fetch %s: %s", sym, e)

        # Глобальные агрегаты
        if new_cache:
            rates = [d["funding_rate"] for d in new_cache.values()
                     if "funding_rate" in d]
            ois   = [d["open_interest_usdt"] for d in new_cache.values()
                     if "open_interest_usdt" in d]
            lrs   = [d["long_ratio"] for d in new_cache.values()
                     if "long_ratio" in d]
            self._global = {
                "avg_funding":    float(sum(rates)/len(rates)) if rates else 0.0,
                "max_funding":    float(max(rates, key=abs)) if rates else 0.0,
                "total_oi_usdt":  float(sum(ois)) if ois else 0.0,
                "avg_long_ratio": float(sum(lrs)/len(lrs)) if lrs else 0.5,
                # Рынок в целом лонговый если avg_long_ratio > 0.55
                "market_bias":    ("long" if (sum(lrs)/len(lrs) > 0.55 if lrs else False)
                                   else "short" if (sum(lrs)/len(lrs) < 0.45 if lrs else False)
                                   else "neutral"),
                "n_positive_funding": sum(1 for r in rates if r > 0),
                "n_negative_funding": sum(1 for r in rates if r < 0),
                "timestamp": time.time(),
            }

        self._cache      = new_cache
        self._last_update = time.time()
        return new_cache

    def _fetch_one(self, sym: str) -> dict:
        """Получить все данные для одного символа."""
        contract = _contract(sym)
        result   = {}

        # 1. Funding rate + mark price + index price
        try:
            r = requests.get(
                f"{FUTURES_BASE}/api/v1/contract/funding_rate/{contract}",
                timeout=8,
            )
            r.raise_for_status()
            d = r.json().get("data", {})
            if d:
                result["funding_rate"]   = float(d.get("fundingRate",    0) or 0)
                result["next_funding_ts"]= int(d.get("nextSettleTime",   0) or 0)
                result["mark_price"]     = float(d.get("markPrice",      0) or 0)
                result["index_price"]    = float(d.get("indexPrice",     0) or 0)
                # Секунд до следующего funding
                next_ts = result["next_funding_ts"] / 1000
                result["next_funding_sec"] = max(0, int(next_ts - time.time()))
        except Exception as e:
            log.debug("funding_rate %s: %s", contract, e)

        # 2. Open Interest
        try:
            r = requests.get(
                f"{FUTURES_BASE}/api/v1/contract/ticker?symbol={contract}",
                timeout=8,
            )
            r.raise_for_status()
            d = r.json().get("data", {})
            if d:
                result["last_price"]         = float(d.get("lastPrice",      0) or 0)
                result["volume_24h_usdt"]    = float(d.get("amount24",       0) or 0)
                result["volume_24h_base"]    = float(d.get("vol24",          0) or 0)
                result["open_interest_base"] = float(d.get("holdVol",        0) or 0)
                price = result.get("last_price") or result.get("mark_price", 0)
                result["open_interest_usdt"] = result["open_interest_base"] * price
                result["price_change_24h"]   = float(d.get("riseFallRate",   0) or 0)
                result["high_24h"]           = float(d.get("high24Price",    0) or 0)
                result["low_24h"]            = float(d.get("low24Price",     0) or 0)
        except Exception as e:
            log.debug("ticker %s: %s", contract, e)

        # 3. Long/Short ratio (топ трейдеры)
        try:
            r = requests.get(
                f"{FUTURES_BASE}/api/v1/contract/long_short",
                params={"symbol": contract},
                timeout=8,
            )
            r.raise_for_status()
            d = r.json().get("data", {})
            if d:
                long_vol  = float(d.get("longVol",  1) or 1)
                short_vol = float(d.get("shortVol", 1) or 1)
                total     = long_vol + short_vol
                result["long_ratio"]  = long_vol  / total if total > 0 else 0.5
                result["short_ratio"] = short_vol / total if total > 0 else 0.5
        except Exception as e:
            log.debug("long_short %s: %s", contract, e)

        result["symbol"]    = sym
        result["updated_ts"]= time.time()
        return result

    # ── Фоновый поток ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Запустить фоновое обновление данных."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="funding_fetcher")
        self._thread.start()
        log.info("[funding] Фоновое обновление запущено (каждые %d сек)", self.refresh_sec)

    def stop(self) -> None:
        self._running = False
        log.info("[funding] Остановлено.")

    def _loop(self) -> None:
        while self._running:
            try:
                self.fetch_all()
                n = len(self._cache)
                g = self._global
                log.info(
                    "[funding] Обновлено %d символов  |  "
                    "avg_rate=%.4f%%  market=%s  longs=%.0f%%",
                    n,
                    g.get("avg_funding", 0) * 100,
                    g.get("market_bias", "?"),
                    g.get("avg_long_ratio", 0.5) * 100,
                )
            except Exception as e:
                log.warning("[funding] Ошибка обновления: %s", e)
            time.sleep(self.refresh_sec)


# ════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции для агентов
# ════════════════════════════════════════════════════════════════════════════

def funding_signal(funding_rate: float) -> str:
    """
    Интерпретирует funding rate как торговый сигнал.

    Логика: при сильно позитивном funding (лонги платят шортам) →
    рынок перегрет вверх → SHORT-сигнал.
    При сильно негативном → рынок перегрет вниз → LONG-сигнал.

    Пороги (стандарт MEXC: интервал 8h):
      > +0.03%  →  strong_short  (лонги платят много)
      > +0.01%  →  weak_short
      < -0.01%  →  weak_long
      < -0.03%  →  strong_long   (шорты платят много)
      иначе     →  neutral
    """
    f = funding_rate
    if   f > 0.0003:  return "strong_short"
    elif f > 0.0001:  return "weak_short"
    elif f < -0.0003: return "strong_long"
    elif f < -0.0001: return "weak_long"
    else:             return "neutral"


def oi_trend(oi_now: float, oi_prev: float) -> str:
    """Тренд открытого интереса: expanding / contracting / stable."""
    if oi_prev <= 0: return "stable"
    change = (oi_now - oi_prev) / oi_prev
    if   change >  0.05: return "expanding"
    elif change < -0.05: return "contracting"
    else:                return "stable"


if __name__ == "__main__":
    """Быстрый тест — запускаем разово."""
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetcher = FundingDataFetcher(symbols=["BTC", "ETH", "SOL", "XRP"])
    print("Получаем данные...")
    data = fetcher.fetch_all()
    for sym, d in data.items():
        fr  = d.get("funding_rate", 0) * 100
        oi  = d.get("open_interest_usdt", 0) / 1e9
        lr  = d.get("long_ratio", 0.5) * 100
        nfs = d.get("next_funding_sec", 0) // 60
        sig = funding_signal(d.get("funding_rate", 0))
        print(f"  {sym:6s}  funding={fr:+.4f}%  OI={oi:.2f}B  longs={lr:.0f}%  "
              f"next={nfs}мин  signal={sig}")
    print("\nГлобально:", json.dumps(fetcher.get_global(), indent=2, default=str))
