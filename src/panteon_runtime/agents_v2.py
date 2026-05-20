"""
Новые агенты v2 (FIX C из плана 2026-04-26).

Все агенты следуют тому же интерфейсу, что и LiveMeanRev / LiveTrendFollow в
panteon_agents.py:
    act(prices, volumes, month=None, portfolio_value=None, bar_index=None)
        → dict {sym: action_int}
с теми же кодами действий:
    1=spot_buy_half, 2=spot_buy_full, 3=spot_sell,
    4=fut_long_half, 5=fut_long_full, 6=fut_short_half, 7=fut_short_full,
    8=fut_close.

Дополнительные классовые атрибуты:
    AGENT_STATUS         = "live" | "shadow_only" | "experimental"
    AGENT_STATUS_REASON  = текстовое объяснение (попадает в leaderboard)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, Optional
import math

import numpy as np


def _atr(highs: list, lows: list, closes: list, n: int = 14) -> float:
    if len(closes) < n + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        h = highs[i] if i < len(highs) else closes[i]
        l = lows[i] if i < len(lows) else closes[i]
        c_prev = closes[i - 1]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
    return float(np.mean(trs[-n:])) if trs else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# C1. MeanRev-Confirmed — расширение LiveMeanRev (66% wr) с volume-фильтром
# ──────────────────────────────────────────────────────────────────────────────

class MeanRevConfirmedAgent:
    """
    LiveMeanRev (z<-2) + подтверждение объёмом (volume ≥ 1.3× медианы за 50 баров).
    Закрытие также подтверждается возвратом z к нулю.

    Цель: брать только «жирные» точки разворота, а не каждую попытку отбоя.
    """
    AGENT_STATUS = "live"
    AGENT_STATUS_REASON = "v2 confirmation layer over LiveMeanRev (66/68% wr base)"

    BB_PERIOD = 50
    ENTRY_Z = 2.0
    EXIT_Z = 0.30
    CHECK_INT = 10
    STOP_PCT = 0.018
    HOLD_BARS = 3 * 60
    MAX_POS = 4
    VOL_LOOKBACK = 50
    VOL_RATIO = 1.30
    ENTRY_COOLDOWN = 90
    NAME = "MeanRevConfirmed"

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.last_entry: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def reset_for_live(self, bar_index: int = 0):
        self.t = bar_index
        self._lc = -9999
        for k in list(self.pos.keys()):
            self.pos[k] = None
            self.ep[k] = 0.0
            self.et[k] = 0

    def _z(self, h):
        if len(h) < self.BB_PERIOD:
            return 0.0
        w = list(h)[-self.BB_PERIOD:]
        mu = float(np.mean(w)); sd = float(np.std(w))
        if sd <= 1e-12:
            return 0.0
        return (h[-1] - mu) / sd

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.BB_PERIOD * 4)).append(float(p))
            self.v.setdefault(s, deque(maxlen=self.VOL_LOOKBACK * 2)).append(float(volumes.get(s, 0.0) or 0.0))
            self.pos.setdefault(s, None); self.ep.setdefault(s, 0.0); self.et.setdefault(s, 0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        n_open = sum(1 for v in self.pos.values() if v is not None)
        candidates = []
        for sym in prices:
            h = list(self.h[sym])
            v = list(self.v[sym])
            cur = self.pos[sym]; px = prices[sym]; ep = self.ep[sym]
            if len(h) < self.BB_PERIOD:
                continue
            z = self._z(h)
            held = self.t - self.et[sym]

            if cur == "long":
                if abs(z) <= self.EXIT_Z or px <= ep * (1 - self.STOP_PCT) or held >= self.HOLD_BARS:
                    actions[sym] = 3
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                    continue
            elif cur == "short":
                if abs(z) <= self.EXIT_Z or px >= ep * (1 + self.STOP_PCT) or held >= self.HOLD_BARS:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                    continue
            if cur is not None:
                continue
            # cooldown
            if self.t - int(self.last_entry.get(sym, -99999)) < self.ENTRY_COOLDOWN:
                continue
            # volume gate
            if len(v) < self.VOL_LOOKBACK:
                continue
            v_med = float(np.median(v[-self.VOL_LOOKBACK:]))
            v_now = float(v[-1])
            if v_med <= 0 or v_now < v_med * self.VOL_RATIO:
                continue

            if z < -self.ENTRY_Z:
                edge = -z - self.ENTRY_Z + (v_now / v_med - self.VOL_RATIO)
                candidates.append((edge, sym, 1, "long", px))   # spot half-buy = mean-rev стиль
            elif z > self.ENTRY_Z:
                edge = z - self.ENTRY_Z + (v_now / v_med - self.VOL_RATIO)
                candidates.append((edge, sym, 6, "short", px))

        slots = max(0, self.MAX_POS - n_open)
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, sym, a, side, px in candidates[:slots]:
            actions[sym] = a
            self.pos[sym] = side
            self.ep[sym] = px
            self.et[sym] = self.t
            self.last_entry[sym] = self.t
        return actions


# ──────────────────────────────────────────────────────────────────────────────
# C2. Adaptive-Vol Trend — TrendFollow с ATR-based трейлингом
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveVolTrendAgent:
    """
    EMA(20) + EMA(80) cross + ATR(20)-trailing.
    В режиме trail: позиция закрывается, когда цена откатывается на k×ATR
    от пика (вместо фиксированного TRAIL_PCT, который выбивает на нормальном
    шуме при низкой волатильности и слишком позно при высокой).
    """
    AGENT_STATUS = "live"
    AGENT_STATUS_REASON = "ATR-based trailing replaces fixed % trailing"

    EMA_FAST = 20
    EMA_SLOW = 80
    ATR_N = 20
    ATR_K_TRAIL = 2.0
    ATR_K_STOP = 1.5
    CHECK_INT = 5
    HOLD_MAX = 6 * 60
    MAX_POS = 5
    NAME = "AdaptiveVolTrend"

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.peak: Dict[str, float] = {}
        self.t = 0
        self._lc = -9999

    def reset_for_live(self, bar_index: int = 0):
        self.t = bar_index
        self._lc = -9999
        for k in list(self.pos.keys()):
            self.pos[k] = None
            self.ep[k] = 0.0
            self.et[k] = 0
            self.peak[k] = 0.0

    @staticmethod
    def _ema(arr, n):
        if len(arr) < n:
            return float(arr[-1]) if arr else 0.0
        a = 2.0 / (n + 1)
        e = float(arr[0])
        for v in arr[1:]:
            e = e * (1.0 - a) + float(v) * a
        return e

    def _atr_proxy(self, h):
        if len(h) < self.ATR_N + 1:
            return 0.0
        diffs = [abs(h[i] - h[i - 1]) for i in range(1, len(h))]
        return float(np.mean(diffs[-self.ATR_N:])) if diffs else 0.0

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.EMA_SLOW * 4)).append(float(p))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self.peak.setdefault(s, 0.0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        n_open = sum(1 for v in self.pos.values() if v is not None)
        candidates = []
        for sym in prices:
            h = list(self.h[sym])
            cur = self.pos[sym]; px = prices[sym]; ep = self.ep[sym]
            if len(h) < self.EMA_SLOW + 1:
                continue
            ema_f = self._ema(h[-self.EMA_FAST * 2:], self.EMA_FAST)
            ema_s = self._ema(h[-self.EMA_SLOW * 2:], self.EMA_SLOW)
            atr = self._atr_proxy(h)
            held = self.t - self.et[sym]

            if cur == "long":
                self.peak[sym] = max(self.peak[sym], px)
                stop = ep - self.ATR_K_STOP * atr
                trail = self.peak[sym] - self.ATR_K_TRAIL * atr
                if px <= stop or px <= trail or held >= self.HOLD_MAX or ema_f < ema_s:
                    actions[sym] = 8
                    self.pos[sym] = None
                    self.peak[sym] = 0.0
                    n_open = max(0, n_open - 1)
                    continue
            elif cur == "short":
                self.peak[sym] = (px if self.peak[sym] == 0 else min(self.peak[sym], px))
                stop = ep + self.ATR_K_STOP * atr
                trail = self.peak[sym] + self.ATR_K_TRAIL * atr
                if px >= stop or px >= trail or held >= self.HOLD_MAX or ema_f > ema_s:
                    actions[sym] = 8
                    self.pos[sym] = None
                    self.peak[sym] = 0.0
                    n_open = max(0, n_open - 1)
                    continue
            if cur is not None:
                continue
            # вход
            if atr <= 0 or ema_s <= 0:
                continue
            if ema_f > ema_s * 1.001:
                edge = (ema_f - ema_s) / max(ema_s, 1e-9)
                candidates.append((edge, sym, 5, "long", px))   # full long
            elif ema_f < ema_s * 0.999:
                edge = (ema_s - ema_f) / max(ema_s, 1e-9)
                candidates.append((edge, sym, 7, "short", px))  # full short

        slots = max(0, self.MAX_POS - n_open)
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, sym, a, side, px in candidates[:slots]:
            actions[sym] = a
            self.pos[sym] = side
            self.ep[sym] = px
            self.et[sym] = self.t
            self.peak[sym] = px
        return actions


# ──────────────────────────────────────────────────────────────────────────────
# C3. CrossExchange-Skew — заготовка статарба BITGET vs MEXC
# ──────────────────────────────────────────────────────────────────────────────

class CrossExchangeSkewAgent:
    """
    Открывает позицию, если в external_quotes (других бирж) для того же
    символа цена отличается на > 0.30%.

    В текущем рантайме каждый бот соединён с одной биржей, поэтому external_quotes
    нужно принести через общий канал (см. mexc_funding.py / bitget_api.py — там
    есть рейт-таблицы). Если канал пуст — агент молчит. Это shadow_only пока
    канал не подключён.
    """
    AGENT_STATUS = "shadow_only"
    AGENT_STATUS_REASON = "needs cross-exchange quote bus (BITGET<->MEXC)"

    SKEW_PCT_OPEN = 0.0030
    SKEW_PCT_CLOSE = 0.0008
    HOLD_MAX = 4 * 60
    MAX_POS = 3
    CHECK_INT = 5
    NAME = "CrossExchangeSkew"

    def __init__(self):
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999
        self.external_quotes: Dict[str, float] = {}  # set externally

    def reset_for_live(self, bar_index: int = 0):
        self.t = bar_index
        self._lc = -9999
        for k in list(self.pos.keys()):
            self.pos[k] = None
            self.ep[k] = 0.0
            self.et[k] = 0

    def set_external_quotes(self, quotes: Dict[str, float]):
        self.external_quotes = dict(quotes or {})

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t
        if not self.external_quotes:
            return actions
        n_open = sum(1 for v in self.pos.values() if v is not None)
        for sym in prices:
            ext = float(self.external_quotes.get(sym, 0.0) or 0.0)
            px = float(prices[sym])
            if ext <= 0 or px <= 0:
                continue
            skew = (ext - px) / px
            cur = self.pos.get(sym)
            ep = self.ep.get(sym, 0.0)
            held = self.t - self.et.get(sym, self.t)
            if cur == "long":
                if abs(skew) <= self.SKEW_PCT_CLOSE or held >= self.HOLD_MAX or skew < -self.SKEW_PCT_OPEN:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if cur == "short":
                if abs(skew) <= self.SKEW_PCT_CLOSE or held >= self.HOLD_MAX or skew > self.SKEW_PCT_OPEN:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if n_open >= self.MAX_POS:
                continue
            if skew > self.SKEW_PCT_OPEN:
                # local price ниже external → лонг local
                actions[sym] = 4  # half long, статарб не требует full
                self.pos[sym] = "long"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
            elif skew < -self.SKEW_PCT_OPEN:
                actions[sym] = 6
                self.pos[sym] = "short"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


# ──────────────────────────────────────────────────────────────────────────────
# C4. Funding-Window — открывает позицию строго в окне фандинга
# ──────────────────────────────────────────────────────────────────────────────

class FundingWindowAgent:
    """
    Открывает позицию за ~5 минут до funding cutoff и закрывает сразу после.
    Условие входа: |funding_rate| ≥ 0.05% и направление получения фандинга
    совпадает с краткосрочным трендом (EMA10 vs EMA40).
    """
    AGENT_STATUS = "live"
    AGENT_STATUS_REASON = "narrow window, low directional risk"

    FUNDING_INTERVAL_MIN = 8 * 60   # 8h в минутах
    PRE_OPEN_MIN = 5
    POST_CLOSE_MIN = 2
    MIN_RATE_ABS = 0.0005
    MAX_POS = 4
    NAME = "FundingWindow"

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.funding_rates: Dict[str, float] = {}   # set externally each bar
        self.t = 0

    def reset_for_live(self, bar_index: int = 0):
        self.t = bar_index
        for k in list(self.pos.keys()):
            self.pos[k] = None
            self.ep[k] = 0.0
            self.et[k] = 0

    def set_funding_rates(self, rates: Dict[str, float]):
        self.funding_rates = dict(rates or {})

    @staticmethod
    def _ema(arr, n):
        if len(arr) < n:
            return float(arr[-1]) if arr else 0.0
        a = 2.0 / (n + 1)
        e = float(arr[0])
        for v in arr[1:]:
            e = e * (1.0 - a) + float(v) * a
        return e

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=200)).append(float(p))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)

        # фаза funding-цикла. self.t в минутах. Funding каждые 8ч = 480 мин.
        # окно входа: minute_of_cycle in [475, 480). Окно выхода: в [0, 2).
        cycle = self.t % self.FUNDING_INTERVAL_MIN
        in_pre = cycle >= (self.FUNDING_INTERVAL_MIN - self.PRE_OPEN_MIN)
        in_post = cycle < self.POST_CLOSE_MIN

        # Закрываем все позиции в post-окно
        if in_post:
            for sym, cur in self.pos.items():
                if cur is not None:
                    actions[sym] = 8
                    self.pos[sym] = None
            return actions

        if not in_pre:
            return actions

        n_open = sum(1 for v in self.pos.values() if v is not None)
        for sym, rate in self.funding_rates.items():
            if sym not in prices or n_open >= self.MAX_POS:
                continue
            if abs(rate) < self.MIN_RATE_ABS or self.pos.get(sym) is not None:
                continue
            h = list(self.h.get(sym, []))
            if len(h) < 40:
                continue
            ema_f = self._ema(h[-20:], 10)
            ema_s = self._ema(h[-80:], 40)
            if rate > 0:
                # short получает фандинг; берём short только если тренд НЕ против
                if ema_f <= ema_s * 1.002:
                    actions[sym] = 6
                    self.pos[sym] = "short"
                    self.ep[sym] = prices[sym]
                    self.et[sym] = self.t
                    n_open += 1
            else:
                if ema_f >= ema_s * 0.998:
                    actions[sym] = 4
                    self.pos[sym] = "long"
                    self.ep[sym] = prices[sym]
                    self.et[sym] = self.t
                    n_open += 1
        return actions


# ──────────────────────────────────────────────────────────────────────────────
# C5. Genome-Ensemble — голосование top-rank геномов
# ──────────────────────────────────────────────────────────────────────────────

class GenomeEnsembleAgent:
    """
    Лёгкая обёртка над несколькими ranked-геномами из Genetics_DL_Agents/Agents/genetics/
    archive_rank{1..5}.npy. Если geometry-движок недоступен — агент сидит тихо.

    Голосование: каждый геном даёт action_int; финальный = мажоритарный, при
    условии что хотя бы 60% голосов согласны.
    """
    AGENT_STATUS = "experimental"
    AGENT_STATUS_REASON = "needs crypto_genetics module on import path"

    NAME = "GenomeEnsemble"

    def __init__(self):
        self._agents = []  # populated lazily
        self._loaded = False

    def reset_for_live(self, bar_index: int = 0):
        for a in self._agents:
            try:
                if hasattr(a, "reset_for_live"):
                    a.reset_for_live(bar_index)
            except Exception:
                pass

    def update_from_exchange(self, symbol: str, spot_qty: float, spot_entry: float,
                              fut_qty: float, fut_entry: float):
        self._load()
        for a in self._agents:
            try:
                update = getattr(a, "update_from_exchange", None)
                if callable(update):
                    update(symbol, spot_qty, spot_entry, fut_qty, fut_entry)
            except Exception:
                pass

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from crypto_genetics import GeneticsBullishAgent, GeneticsBearishAgent, GeneticsNeutralAgent
            for cls in (GeneticsBullishAgent, GeneticsBearishAgent, GeneticsNeutralAgent):
                try:
                    self._agents.append(cls())
                except Exception:
                    pass
        except Exception:
            self._agents = []

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        self._load()
        actions = {s: 0 for s in prices}
        if not self._agents:
            return actions
        votes: Dict[str, Dict[int, int]] = {}
        for a in self._agents:
            try:
                acts = a.act(prices, volumes=volumes, month=month,
                             portfolio_value=portfolio_value, bar_index=bar_index) or {}
            except TypeError:
                try:
                    acts = a.act(prices, volumes, month=month,
                                 portfolio_value=portfolio_value) or {}
                except TypeError:
                    try:
                        acts = a.act(prices, volumes) or {}
                    except Exception:
                        acts = {}
                except Exception:
                    acts = {}
            except Exception:
                acts = {}
            for sym, val in acts.items():
                if val == 0:
                    continue
                votes.setdefault(sym, {})[int(val)] = votes.setdefault(sym, {}).get(int(val), 0) + 1
        threshold = max(2, int(0.6 * max(len(self._agents), 1)))
        for sym, vmap in votes.items():
            best_action, best_count = max(vmap.items(), key=lambda x: x[1])
            if best_count >= threshold:
                actions[sym] = best_action
        return actions


# ──────────────────────────────────────────────────────────────────────────────
# C6. Defensive-Stop Overlay — НЕ открывает, только закрывает по cross-сигналам
# ──────────────────────────────────────────────────────────────────────────────

class DefensiveStopOverlayAgent:
    """
    «Страховка»: если позиция в минусе ≥ STOP_LOSS_PCT и цена ушла ниже
    EMA-короткого, форсируем close. Не открывает позиции вообще.

    Используется как finalizer — последняя линия защиты против зависших
    позиций (таких как KAT/RAVE из сессии 2026-04-25, где обычный
    PositionGovernor не успевал.)
    """
    AGENT_STATUS = "live"
    AGENT_STATUS_REASON = "exit-only safety net; never opens positions"

    EMA_FAST = 20
    STOP_LOSS_PCT = 0.030
    CHECK_INT = 5
    NAME = "DefensiveStopOverlay"

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.t = 0
        self._lc = -9999

    def reset_for_live(self, bar_index: int = 0):
        self.t = bar_index
        self._lc = -9999

    def sync_position(self, sym: str, side: str, entry: float):
        """Внешний апдейт: бридж сообщает, какие позиции у нас открыты."""
        if side in (None, "", "flat"):
            self.pos[sym] = None
            self.ep[sym] = 0.0
        else:
            self.pos[sym] = side
            self.ep[sym] = float(entry)

    @staticmethod
    def _ema(arr, n):
        if len(arr) < n:
            return float(arr[-1]) if arr else 0.0
        a = 2.0 / (n + 1)
        e = float(arr[0])
        for v in arr[1:]:
            e = e * (1.0 - a) + float(v) * a
        return e

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.EMA_FAST * 4)).append(float(p))
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t
        for sym, side in list(self.pos.items()):
            if side is None or sym not in prices:
                continue
            px = float(prices[sym])
            ep = float(self.ep.get(sym, 0.0))
            if ep <= 0 or px <= 0:
                continue
            h = list(self.h.get(sym, []))
            if len(h) < self.EMA_FAST + 1:
                continue
            ema = self._ema(h[-self.EMA_FAST * 2:], self.EMA_FAST)
            move = px / ep - 1 if side == "long" else 1 - px / ep
            if side == "long" and move <= -self.STOP_LOSS_PCT and px < ema:
                actions[sym] = 8
                self.pos[sym] = None
            elif side == "short" and move <= -self.STOP_LOSS_PCT and px > ema:
                actions[sym] = 8
                self.pos[sym] = None
        return actions


__all__ = [
    "MeanRevConfirmedAgent",
    "AdaptiveVolTrendAgent",
    "CrossExchangeSkewAgent",
    "FundingWindowAgent",
    "GenomeEnsembleAgent",
    "DefensiveStopOverlayAgent",
]
