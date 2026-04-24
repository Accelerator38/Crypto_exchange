"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  master_player.py  —  MasterPlayer v7 для crypto exchange  $100            ║
║                                                                              ║
║  Новое в v7 (vs v6):                                                        ║
║    ИСПРАВЛЕНИЯ КРИТИЧЕСКИХ БАГОВ:                                           ║
║    • FIX: Голосование — short (6,7) больше не засчитывается как buy          ║
║      Short только при большинстве голосов за short, не при short_w>=0.10    ║
║    • FIX: Добавлена поддержка fut_long (действия 4,5) в _apply()           ║
║    • FIX: PnL для fut_long корректно считается (зарабатывает на росте)     ║
║    • FIX: reset_after_warmup полностью очищает: profit_lock, позиции,       ║
║      стопы, сделки — warmup больше не загрязняет live                       ║
║    • FIX: Режимный детектор: асимметричные пороги → симметричные (±0.006)   ║
║    • UPGRADE: Подключён мульти-детектор _detect_regime_live из crypto_agents║
║      (3 окна: 168h/72h/24h + halving/macro/seasonal bias)                  ║
║    • FIX: get_status / _check_master_sl — корректная обработка fut_long     ║
║                                                                              ║
║  v6: БЫСТРО торгующие агенты (DualMom, VolBreakout, DCA, Genetics)         ║
║    • DayStop / PeakStop / ProfitLock — без изменений                        ║
║                                                                              ║
║  Запуск:                                                                     ║
║      python master_player.py                    # paper-тест                ║
║      EXCHANGE_TRADING_MODE=live_spot EXCHANGE_API_KEY=xxx EXCHANGE_SECRET=yyy║
║      python master_player.py                    # реальные деньги           ║
║                                                                              ║
║  Выходные файлы:                                                             ║
║    usd_dashboard.png  — USD-дашборд с разбивкой активов                    ║
║    master_detail.png  — детали по суб-игрокам                               ║
║    master_detail.json — полный стейт                                        ║
║    master_live_growth.png — кривая live-доходности                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os, sys, json, logging, traceback
import numpy as np
from collections import deque, OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from project_paths import PROJECT_ROOT, RUNTIME_DIR
from agent_safety import (
    DeclineGuard as SharedDeclineGuard,
    ExtremeMoveGuard as SharedExtremeMoveGuard,
    FeeAwareFilter as SharedFeeAwareFilter,
    PortfolioSafetyGovernor,
)

log = logging.getLogger("master_player")

# ──────────────────────────────────────────────────────────────────────────────
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ
# ──────────────────────────────────────────────────────────────────────────────

BAR             = 60
INITIAL_CAPITAL = 100.0
TRADE_FRACTION  = 0.10      # консервативнее: 10% на позицию
LEVERAGE        = 3.0
SPOT_FEE        = 0.001
FUTURES_FEE     = 0.0002


def configure(BAR=60, INITIAL_CAPITAL=100.0, TRADE_FRACTION=0.10,
              LEVERAGE=3.0, SPOT_FEE=0.001, FUTURES_FEE=0.0002,
              cfg=None, **kwargs):
    import sys as _sys
    m = _sys.modules[__name__]
    m.BAR             = int(BAR)
    m.INITIAL_CAPITAL = float(INITIAL_CAPITAL)
    m.TRADE_FRACTION  = float(TRADE_FRACTION)
    m.LEVERAGE        = float(LEVERAGE)
    m.SPOT_FEE        = float(SPOT_FEE)
    m.FUTURES_FEE     = float(FUTURES_FEE)


# ──────────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
# ──────────────────────────────────────────────────────────────────────────────

def _vol_zscore(v_deque, window: int = 60) -> float:
    lst = list(v_deque)
    if len(lst) < window + 1: return 0.0
    hist, cur = lst[-window - 1:-1], lst[-1]
    mu, sd = float(np.mean(hist)), float(np.std(hist))
    return (cur - mu) / (sd + 1e-9)


def _detect_regime(price_history: Dict[str, deque], lb: int = 60,
                    current_month: Optional[int] = None) -> str:
    """
    Детектор режима рынка.
    Использует продвинутый мульти-детектор из crypto_agents если доступен,
    иначе fallback на простой медианный детектор с симметричными порогами.
    """
    # Пробуем продвинутый детектор
    try:
        from crypto_agents import _detect_regime_live, _r3
        raw = _detect_regime_live(price_history, min_bars=max(lb, 24 * BAR),
                                   current_month=current_month)
        label_3 = _r3(raw)  # bearish / neutral / bullish
        return {'bullish': 'bull', 'bearish': 'bear', 'neutral': 'sideways'}.get(label_3, 'sideways')
    except Exception:
        pass

    # Запасной вариант: простой детектор с симметричными порогами
    moms = []
    for h in price_history.values():
        lst = list(h)
        if len(lst) < lb + 1: continue
        b = lst[-(lb + 1)]
        if b > 0: moms.append(lst[-1] / b - 1)
    if not moms: return 'sideways'
    med = float(np.median(moms))
    if med > 0.006: return 'bull'     # симметричный порог (было 0.008)
    if med < -0.006: return 'bear'    # симметричный порог (было -0.004)
    return 'sideways'


# ──────────────────────────────────────────────────────────────────────────────
# ФИЛЬТРЫ-ОБЁРТКИ
# ──────────────────────────────────────────────────────────────────────────────

class _DeclineGuard(SharedDeclineGuard):
    PERIOD = 3 * 60; THRESH = -0.015

    def __init__(self, a):
        super().__init__(a, period=self.PERIOD, threshold=self.THRESH)
        self._a = a
        self._h = self._history


class _FeeAwareFilter(SharedFeeAwareFilter):
    MIN_FACTOR = 1.5; WIN = 20

    def __init__(self, a, is_futures=False):
        fee = FUTURES_FEE if is_futures else SPOT_FEE
        super().__init__(a, fee=fee, min_factor=self.MIN_FACTOR, window=self.WIN)
        self._a = a
        self._fee = fee
        self._h = self._history


class _MemeFilter(SharedExtremeMoveGuard):
    EXTREME_MOVE = 0.30; LOOKBACK = 24 * 60

    def __init__(self, a):
        super().__init__(a, lookback=self.LOOKBACK, extreme_move=self.EXTREME_MOVE)
        self._a = a
        self._h = self._history


class _Genetics6to9Adapter:
    """
    FIX: маппинг 6-action external → 9-action scheme.

    GeneticsAgent через _GeneticsAdapter возвращает 6-action external коды:
      0=hold, 1=buy_spot, 2=sell_spot, 3=fut_long, 4=fut_short, 5=close_fut

    _SubPlayerBase._vote() ожидает 9-action scheme:
      0=hold, 1=buy_half, 2=buy_full, 3=sell_spot,
      4=fl_half, 5=fl_full, 6=fs_half, 7=fs_full, 8=close_fut

    Без этого маппинга sell=2 → трактуется как buy_full,
    fut_long=3 → трактуется как sell_spot, и т.д. — полная инверсия!
    """
    _MAP = {0: 0, 1: 2, 2: 3, 3: 5, 4: 7, 5: 8}

    def __init__(self, agent):
        self._a = agent

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        r = self._a.act(prices, volumes, month=month,
                        portfolio_value=portfolio_value, bar_index=bar_index)
        return {s: self._MAP.get(a, 0) for s, a in r.items()} if r else {}

    def __getattr__(self, item):
        # Guard: без этого pickle.loads вызывает бесконечную рекурсию
        if item == '_a':
            raise AttributeError(item)
        return getattr(self._a, item)


def _wrap(agent, is_futures=False, decline_guard=False, is_genetics=False):
    """Навешивает стандартный стек фильтров."""
    a = agent
    if is_genetics:
        a = _Genetics6to9Adapter(a)
    a = _MemeFilter(_FeeAwareFilter(a, is_futures=is_futures))
    if decline_guard:
        a = _DeclineGuard(a)
    return a


# ──────────────────────────────────────────────────────────────────────────────
# ЧЁРНЫЙ СПИСОК МОНЕТ
# ──────────────────────────────────────────────────────────────────────────────

BLACKLIST = frozenset({
    'TONIXAI', 'DEGO', 'CRTR', 'ATLA',
    'GOLD(PAXG)', 'GOLD(XAUT)', 'BLINKY', 'WLD',
})


def _reset_agent_inner_state(agent, seen: set = None):
    """
    Рекурсивно сбрасывает внутренние позиции суб-агента и его обёрток.
    Агенты типа DualMomentum, CorrBreakout и др. хранят self.pos — dict
    позиций. После warmup эти позиции должны быть обнулены, иначе агенты
    не будут генерировать новые buy-сигналы (думают что позиция уже открыта).
    """
    if seen is None:
        seen = set()
    aid = id(agent)
    if aid in seen:
        return
    seen.add(aid)

    # Сброс позиций (DualMomentum, CrashHunter, TrendCapture, etc.)
    if hasattr(agent, 'pos') and isinstance(agent.pos, dict):
        for s in list(agent.pos.keys()):
            agent.pos[s] = None

    # Сброс кэшированных действий (_GeneticsAdapter)
    if hasattr(agent, '_cached_acts'):
        try:
            object.__setattr__(agent, '_cached_acts', None)
        except Exception:
            pass

    # Сброс cooldown-таймеров (DualMomentum, CrashHunter)
    if hasattr(agent, '_cooldown_until'):
        agent._cooldown_until = 0
    if hasattr(agent, '_trail_peak_eq'):
        agent._trail_peak_eq = None
    if hasattr(agent, '_month_start_eq'):
        agent._month_start_eq = None

    # Обход вложенных обёрток
    for attr in ('_a', '_inner', '_agent'):
        inner = None
        try:
            inner = getattr(agent, attr, None)
        except Exception:
            pass
        if inner is not None and inner is not agent:
            _reset_agent_inner_state(inner, seen)


# ══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЙ СУБ-ИГРОК
# ══════════════════════════════════════════════════════════════════════════════

class _SubPlayerBase:
    """
    Базовый суб-игрок. Управляет набором агентов, отслеживает:
      • DayStop  : -10% от капитала начала торгового дня
      • PeakStop : -20% от исторического пика
      • ProfitLock: при +LOCK_TRIGGER% прибыли фиксируем LOCK_FRACTION долю
    Каждый суб-игрок получает равную долю от INITIAL_CAPITAL.
    """

    NAME            = "SubPlayer"
    # Дневной стоп: -10% от открытия дня
    DAY_STOP_PCT    = 0.10
    # Стоп от пика: -20%
    PEAK_STOP_PCT   = 0.20
    # Cooldown после стопа: 4 часа
    STOP_COOLDOWN   = 4 * 60
    # === Фиксация прибыли ===
    # При +LOCK_TRIGGER% прибыли — фиксируем LOCK_FRACTION долю
    LOCK_TRIGGER    = 0.05      # +5% → первая фиксация
    LOCK_STEP       = 0.05      # каждые дополнительные +5%
    LOCK_FRACTION   = 0.30      # фиксируем 30% от текущей прибыли
    LOCK_MAX        = 0.70      # максимум 70% капитала в locked_usdt
    # Вотинг
    BUY_THRESHOLD   = 0.25
    MAX_POSITIONS   = 4
    MIN_ORDER_USDT  = 5.0

    def __init__(self, capital_fraction: float = 1.0):
        self._capital = INITIAL_CAPITAL * capital_fraction
        self._pv      = self._capital
        self._pv_peak = self._capital
        self._pv_history: List[float] = [self._capital]
        self._live_start_idx = 0

        # Агенты
        self._agents: Dict[str, object] = {}
        self._build_agents()

        # Позиции
        self._open_pos: Dict[str, dict] = {}

        # Режим
        self._price_hist: Dict[str, deque] = {}
        self._regime    = 'sideways'
        self._regime_buf: List[str] = []
        self._last_regime_check = -9999

        # DayStop
        self._day_start_pv    = self._capital
        self._day_start_bar   = 0
        self._day_stopped     = False
        self._day_stop_until  = -1

        # PeakStop
        self._peak_stopped    = False
        self._peak_stop_until = -1

        # ProfitLock
        self._locked_usdt     = 0.0      # зафиксированный USDT (не торгуется)
        self._lock_next_at    = self._capital * (1 + self.LOCK_TRIGGER)  # следующий триггер

        self.t = 0
        self._all_trades: List[dict] = []
        self._signals_count: Dict[str, int] = {}
        self._safety = PortfolioSafetyGovernor(master_stop_loss=0.08, min_order_cap=120.0)

    # ── Стройка агентов (переопределить в наследнике) ────────────────────────
    def _build_agents(self):
        pass

    def _safe_load(self, name: str, factory):
        try:
            self._agents[name] = factory()
            log.info("    [%s] %-20s OK", self.NAME, name)
        except Exception as e:
            log.warning("    [%s] %-20s FAIL: %s", self.NAME, name, e)

    # ── DayStop ──────────────────────────────────────────────────────────────
    def _update_day_stop(self):
        """Сбрасывает день-статистику каждые 24 часа."""
        self._safety.update_day_boundary(self, bar_span=BAR, logger=log)

    def _check_day_stop(self) -> bool:
        return self._safety.check_day_stop(self, logger=log)

    # ── PeakStop ─────────────────────────────────────────────────────────────
    def _check_peak_stop(self) -> bool:
        return self._safety.check_peak_stop(self, logger=log)

    def _force_close_all(self, prices) -> dict:
        """Принудительное закрытие всех позиций."""
        return self._safety.force_close_all(self, prices)

    # ── Фиксация прибыли (Profit Lock) ───────────────────────────────────────
    def _update_profit_lock(self) -> float:
        return self._safety.update_profit_lock(self, logger=log)

    # ── Режим рынка ──────────────────────────────────────────────────────────
    def _update_regime(self, month: Optional[int] = None):
        check_interval = 20  # 20 баров ≈ 20 мин
        if self.t - self._last_regime_check < check_interval:
            return
        self._last_regime_check = self.t
        new = _detect_regime(self._price_hist, current_month=month)
        self._regime_buf.append(new)
        if len(self._regime_buf) > 2: self._regime_buf.pop(0)
        if len(self._regime_buf) == 2 and len(set(self._regime_buf)) == 1:
            self._regime = self._regime_buf[0]

    # ── Голосование ──────────────────────────────────────────────────────────
    def _active_agents(self) -> Dict[str, object]:
        """Возвращает активные агенты для текущего режима."""
        return self._agents

    def _vote(self, prices, volumes, month, bar_index) -> Tuple[dict, dict]:
        agents  = self._active_agents()
        n = len(agents)
        if n == 0: return {}, {}
        weight = 1.0 / n
        buy_w:  Dict[str, float] = {}
        sell_w: Dict[str, float] = {}
        act_map: Dict[str, Dict[str, int]] = {}

        # Раздельные весы: spot_buy (1,2), fut_long (4,5), fut_short (6,7), sell/close (3,8)
        long_w:  Dict[str, float] = {}   # покупка (spot/fut_long)
        short_w_map: Dict[str, float] = {}   # fut_short

        for name, agent in agents.items():
            try:
                pv_share = self._pv * weight
                acts = agent.act(prices, volumes, month=month,
                                 portfolio_value=pv_share, bar_index=bar_index)
            except Exception as e:
                log.debug("  [%s/%s] error: %s", self.NAME, name, e)
                acts = {}
            for sym, a in acts.items():
                # FIX C13: блокируем ВСЕ действия на blacklisted символах
                if sym in BLACKLIST: continue
                if not (isinstance(a, int) and 1 <= a <= 8): continue
                act_map.setdefault(sym, {})[name] = a
                if a in (1, 2, 4, 5):
                    buy_w[sym]  = buy_w.get(sym, 0) + weight
                    long_w[sym] = long_w.get(sym, 0) + weight
                elif a in (6, 7):
                    # FIX C11: шорты НЕ добавляются в buy_w (было ошибкой)
                    short_w_map[sym] = short_w_map.get(sym, 0) + weight
                elif a in (3, 8):
                    sell_w[sym] = sell_w.get(sym, 0) + weight

        final = {}
        # FIX C11: объединяем buy_w, sell_w И short_w_map для полного покрытия
        for sym in set(buy_w) | set(sell_w) | set(short_w_map):
            in_pos = sym in self._open_pos
            if in_pos and sell_w.get(sym, 0) > 0:
                final[sym] = 3 if self._open_pos[sym]['type'] == 'spot' else 8
            elif not in_pos and len(self._open_pos) < self.MAX_POSITIONS:
                sw = short_w_map.get(sym, 0)
                lw = long_w.get(sym, 0)
                bw = buy_w.get(sym, 0)
                # Шорт: если short_w достаточен и превышает long_w
                if sw >= self.BUY_THRESHOLD and sw > lw:
                    final[sym] = 6
                elif bw >= self.BUY_THRESHOLD:
                    # fut_long (4) если есть голоса за long-futures, иначе spot buy (1)
                    fl_w = sum(weight for _, a in act_map.get(sym, {}).items()
                               if a in (4, 5))
                    final[sym] = 4 if fl_w > lw - fl_w else 1

        return final, act_map

    def _apply(self, actions, act_map, prices):
        for sym, a in actions.items():
            if a in (1, 2):
                agent = max(act_map.get(sym, {'?': 1}).keys(), default='?')
                self._open_pos[sym] = {'type': 'spot', 'agent': agent,
                                        'entry_price': prices.get(sym, 0),
                                        'entry_bar': self.t}
                self._all_trades.append({'bar': self.t, 'sym': sym, 'action': 'BUY',
                                          'agent': agent, 'price': round(prices.get(sym, 0), 6),
                                          'regime': self._regime})
            elif a in (4, 5):
                agent = max(act_map.get(sym, {'?': 1}).keys(), default='?')
                self._open_pos[sym] = {'type': 'fut_long', 'agent': agent,
                                        'entry_price': prices.get(sym, 0),
                                        'entry_bar': self.t}
                self._all_trades.append({'bar': self.t, 'sym': sym, 'action': 'LONG',
                                          'agent': agent, 'price': round(prices.get(sym, 0), 6),
                                          'regime': self._regime})
            elif a in (6, 7):
                agent = max(act_map.get(sym, {'?': 1}).keys(), default='?')
                self._open_pos[sym] = {'type': 'fut_short', 'agent': agent,
                                        'entry_price': prices.get(sym, 0),
                                        'entry_bar': self.t}
                self._all_trades.append({'bar': self.t, 'sym': sym, 'action': 'SHORT',
                                          'agent': agent, 'price': round(prices.get(sym, 0), 6),
                                          'regime': self._regime})
            elif a in (3, 8):
                info = self._open_pos.pop(sym, {})
                ep   = info.get('entry_price', 0)
                cur  = prices.get(sym, ep)
                tp   = info.get('type', 'spot')
                if tp == 'spot':
                    pnl = (cur/ep - 1) * 100 if ep > 0 else 0
                elif tp == 'fut_long':
                    pnl = (cur/ep - 1) * 100 if ep > 0 else 0  # лонг: зарабатываем на росте
                else:  # fut_short
                    pnl = (1 - cur/ep) * 100 if ep > 0 else 0
                self._all_trades.append({'bar': self.t, 'sym': sym, 'action': 'CLOSE',
                                          'agent': info.get('agent','?'),
                                          'price': round(cur, 6), 'pnl_pct': round(pnl,3),
                                          'regime': self._regime})

    # ── Мин. ордер ──────────────────────────────────────────────────────────
    def _filter_min(self, actions):
        return self._safety.filter_min_order(self, actions, trade_fraction=TRADE_FRACTION)

    # ── Принудительный стоп по времени ──────────────────────────────────────
    def _check_master_sl(self, prices) -> dict:
        """Закрываем позиции с -8% от входа."""
        return self._safety.check_master_stop(self, prices, logger=log)

    # ── ГЛАВНЫЙ МЕТОД ────────────────────────────────────────────────────────
    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None) -> dict:
        self.t = bar_index if bar_index is not None else self.t + 1
        if portfolio_value is not None:
            self._pv = max(0.1, float(portfolio_value))
        self._pv_history.append(self._pv)

        for s, p in prices.items():
            self._price_hist.setdefault(s, deque(maxlen=8 * BAR)).append(float(p))

        # Обновляем режим и дневной старт
        self._update_regime(month=month)
        self._update_day_stop()

        # Фиксация прибыли
        self._update_profit_lock()

        # Проверки стопов
        sl_closes = self._check_master_sl(prices)
        if sl_closes:
            result = {s: 0 for s in prices}
            result.update(sl_closes)
            return result

        if self._check_peak_stop():
            return self._force_close_all(prices)

        if self._check_day_stop():
            return {s: 0 for s in prices}

        # Голосование
        raw, act_map = self._vote(prices, volumes, month, self.t)
        final = self._filter_min(raw)
        self._apply(final, act_map, prices)

        return {s: final.get(s, 0) for s in prices}

    def reset_after_warmup(self, prices):
        """Сброс после прогрева — очистить загрязнённую историю."""
        self._price_hist.clear()
        self._regime_buf.clear()
        self._last_regime_check = self.t - 20

        # FIX: сбрасываем PV к начальному капиталу, чтобы все игроки
        # стартовали live-торговлю с одной точки (одинакового капитала)
        self._pv = self._capital
        self._live_start_idx = len(self._pv_history)
        self._pv_history.append(self._capital)  # первая точка live = капитал
        self._day_start_pv   = self._capital
        self._day_start_bar  = self.t

        # Сброс фиксации прибыли — warmup не должен влиять на live
        self._locked_usdt  = 0.0
        self._lock_next_at = self._capital * (1 + self.LOCK_TRIGGER)

        # Сброс открытых позиций от warmup
        self._open_pos.clear()
        self._all_trades.clear()

        # Сброс стопов
        self._day_stopped     = False
        self._day_stop_until  = -1
        self._peak_stopped    = False
        self._peak_stop_until = -1
        self._pv_peak         = self._capital

        for s, p in prices.items():
            self._price_hist.setdefault(s, deque(maxlen=8 * BAR)).append(float(p))

        # FIX CRITICAL: сбрасываем внутренние позиции суб-агентов!
        # Без этого агенты (DualMomentum, CorrBreakout и др.) считают что
        # warmup-позиции всё ещё открыты → не генерируют новые buy-сигналы.
        # Обходим все агенты и вложенные обёртки.
        all_agent_sets = [self._agents]
        for attr in ('_bull_agents', '_bear_agents', '_sideways_agents'):
            d = getattr(self, attr, None)
            if d: all_agent_sets.append(d)
        seen = set()
        for agents_dict in all_agent_sets:
            for name, agent in agents_dict.items():
                _reset_agent_inner_state(agent, seen)

        log.info("  [%s] reset after warmup (live_start=%d, pv=%.2f)", self.NAME, self._live_start_idx, self._pv)

    # ── Статус ───────────────────────────────────────────────────────────────
    def get_status(self, prices: dict) -> dict:
        positions = []
        for sym, info in self._open_pos.items():
            cur = prices.get(sym, info.get('entry_price', 0))
            ep  = info.get('entry_price', 0)
            tp = info['type']
            if tp == 'fut_short':
                pnl = (1 - cur/ep) * 100 if ep > 0 else 0
            else:  # spot, fut_long
                pnl = (cur/ep - 1) * 100 if ep > 0 else 0
            positions.append({'sym': sym, 'type': info['type'],
                               'agent': info.get('agent','?'),
                               'entry_price': round(ep,6), 'cur_price': round(cur,6),
                               'held_bars': self.t - info.get('entry_bar', self.t),
                               'pnl_pct': round(pnl,3)})
        # Стоимость крипто-активов по текущим ценам
        # Спот: qty × current_price, но qty не хранится, поэтому используем оценку через PnL
        TRADE_USD = min(self._capital * TRADE_FRACTION, 120.0)
        crypto_value = 0.0
        for sym, info in self._open_pos.items():
            cur = prices.get(sym, info.get('entry_price', 0))
            ep  = info.get('entry_price', 0)
            if info['type'] == 'spot' and ep > 0:
                crypto_value += TRADE_USD * (cur / ep)
            elif info['type'] == 'fut_long' and ep > 0:
                pnl_pct = (cur/ep - 1)
                crypto_value += TRADE_USD * (1 + LEVERAGE * pnl_pct)  # margin + pnl
            elif info['type'] == 'fut_short' and ep > 0:
                pnl_pct = (1 - cur/ep)
                crypto_value += TRADE_USD * (1 + LEVERAGE * pnl_pct)  # margin + pnl
        # Приближённый кэш: PV - crypto_value
        crypto_value_cur = crypto_value
        dd = (self._pv_peak - self._pv) / max(self._pv_peak, 1)
        day_dd = (self._day_start_pv - self._pv) / max(self._day_start_pv, 1)
        # Разделяем cash и крипто
        cash_usdt_approx = max(0.0, self._pv - crypto_value) + self._locked_usdt
        return {
            'name':           self.NAME,
            'pv':             round(self._pv, 4),
            'pv_peak':        round(self._pv_peak, 4),
            'locked_usdt':    round(self._locked_usdt, 4),
            'cash_usdt':      round(cash_usdt_approx, 4),
            'crypto_value':   round(crypto_value, 4),
            'total_usdt':     round(self._pv + self._locked_usdt, 4),
            'regime':         self._regime,
            'dd_pct':         round(dd * 100, 2),
            'day_dd_pct':     round(day_dd * 100, 2),
            'day_stopped':    self._day_stopped,
            'peak_stopped':   self._peak_stopped,
            'positions':      positions,
            'n_positions':    len(self._open_pos),
            'total_trades':   len(self._all_trades),
            'recent_trades':  self._all_trades[-8:],
            'agents_active':  list(self._active_agents().keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК ALPHA — АДАПТИВНЫЙ (меняет агентов по типу рынка)
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Alpha(_SubPlayerBase):
    """
    Alpha v8: Адаптивный суб-игрок с улучшенной диверсификацией.

    Проблема v6: только DualMom+VolBrk в bear/sideways — одинаковый набор,
      нет специализации. CorrBreakout отсутствовал (некоррелированная стратегия).

    v8 Решение — разные составы для каждого режима:
      Bull:     DualMomentum + VolBreakout + GeneticsBullish + CorrBreakout
      Bear:     DualMomentum + VolBreakout + GeneticsBearish (специалист медвежьего)
      Sideways: DualMomentum + CorrBreakout (breakout работает в range-bound)

    CorrBreakout: торгует разрывы корреляции — некоррелирован с momentum-стратегиями,
    добавляет диверсификацию и снижает drawdown при переключении режимов.
    GeneticsBearish: нейросеть, обученная на медвежьих периодах — шортит эффективно.
    """
    NAME = "Alpha(Adaptive)"

    def _build_agents(self):
        self._bull_agents:     Dict[str, object] = {}
        self._bear_agents:     Dict[str, object] = {}
        self._sideways_agents: Dict[str, object] = {}

        # Общие агенты — быстро торгующие, работают во всех режимах
        try:
            from crypto_agents import DualMomentum
            dm = _wrap(DualMomentum(), is_futures=False)
            self._bull_agents['DualMom'] = dm
            self._bear_agents['DualMom'] = dm
            self._sideways_agents['DualMom'] = dm
        except Exception as e:
            log.warning("  [Alpha] DualMomentum: %s", e)

        try:
            from crypto_agents import VolatilityBreakoutAgent
            vb = _wrap(VolatilityBreakoutAgent(), is_futures=True)
            self._bull_agents['VolBrk'] = vb
            self._bear_agents['VolBrk'] = vb
        except Exception as e:
            log.warning("  [Alpha] VolatilityBreakout: %s", e)

        # CorrBreakout — некоррелированная стратегия для диверсификации
        try:
            from crypto_agents import CorrBreakoutAgent
            cb = _wrap(CorrBreakoutAgent(), is_futures=False)
            self._bull_agents['CorrBrk'] = cb
            self._sideways_agents['CorrBrk'] = cb
        except Exception as e:
            log.warning("  [Alpha] CorrBreakout: %s", e)

        # GeneticsBullish — лучший по прибыли, только в бычьем рынке
        # Исправление: GeneticsBullish заменён на GeneticsBullishAgent + обёртки адаптации
        try:
            from crypto_genetics import GeneticsBullishAgent
            from crypto_agents import _GeneticsAdapter
            self._bull_agents['GenBull'] = _wrap(_GeneticsAdapter(GeneticsBullishAgent()),
                                                  is_futures=False, is_genetics=True)
        except Exception:
            try:
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                self._bull_agents['GenBull'] = _wrap(_GeneticsAdapter(GeneticsAgent()),
                                                      is_futures=False, is_genetics=True)
            except Exception as e:
                log.warning("  [Alpha] GeneticsBullish: %s", e)

        # GeneticsBearish — специалист медвежьего рынка, только в bear-режиме
        # Исправление: GeneticsBearish заменён на GeneticsBearishAgent + обёртки адаптации
        try:
            from crypto_genetics import GeneticsBearishAgent
            from crypto_agents import _GeneticsAdapter
            self._bear_agents['GenBear'] = _wrap(_GeneticsAdapter(GeneticsBearishAgent()),
                                                  is_futures=True, is_genetics=True)
        except Exception:
            try:
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                self._bear_agents['GenBear'] = _wrap(_GeneticsAdapter(GeneticsAgent()),
                                                      is_futures=True, is_genetics=True)
            except Exception as e:
                log.warning("  [Alpha] GeneticsBearish: %s", e)

        # По умолчанию — sideways
        self._agents = dict(self._sideways_agents) or dict(self._bull_agents)
        log.info("  [Alpha] bull=%d  bear=%d  sideways=%d agents",
                 len(self._bull_agents), len(self._bear_agents), len(self._sideways_agents))

    def _active_agents(self) -> Dict[str, object]:
        if self._regime == 'bull':   return self._bull_agents or self._agents
        if self._regime == 'bear':   return self._bear_agents or self._agents
        return self._sideways_agents or self._agents


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК BETA — ЗАЩИТНЫЙ (консерватизм, низкая DD)
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Beta(_SubPlayerBase):
    """
    Beta v8: Защитный суб-игрок с упором на стабильный доход и минимальную DD.

    Проблема v6: DualMom+RevenueDCA+VolBrk — все momentum-стратегии,
      высокая корреляция между собой, не защищают в bear.

    v8 Решение — упор на DCA-стратегии + bear-специалист:
      • ThreeCommasDCA — лучший по Sharpe (1.46), усреднение позиций при падении,
        с адаптированными параметрами для $100 капитала (STEP_PCT=1.5%, MAX_SAFETY=2)
      • GeneticsBearish — нейросеть для медвежьих рынков, шортит эффективно
      • CorrBreakout — некоррелированная стратегия, работает в любых условиях

    Защитные параметры:
      • PEAK_STOP_PCT=12% — жёсткий стоп от пика
      • DAY_STOP_PCT=8% — дневной лимит потерь
      • LOCK_TRIGGER=3% — ранняя фиксация прибыли (40% от прибыли)
      • MAX_POSITIONS=3 — ограниченная экспозиция
    """
    NAME           = "Beta(Defensive)"
    BUY_THRESHOLD  = 0.25      # v8: 0.30→0.25 (3 агента, каждый 0.33)
    PEAK_STOP_PCT  = 0.12      # жёстче: -12%
    DAY_STOP_PCT   = 0.08      # -8% в день
    LOCK_TRIGGER   = 0.03      # раньше начинаем фиксировать: +3%
    LOCK_FRACTION  = 0.40      # 40% прибыли
    MAX_POSITIONS  = 3         # меньше позиций

    def _build_agents(self):
        # ThreeCommasDCA — лучший DCA агент, адаптирован для $100
        self._safe_load('3Commas', lambda: self._make_dca())

        # GeneticsBearish — шортит в медвежьем рынке, защита капитала
        self._safe_load('GenBear', lambda: self._make_gen_bear())

        # CorrBreakout — некоррелированная с momentum, работает в range
        self._safe_load('CorrBrk', lambda: _wrap(
            __import__('crypto_agents', fromlist=['CorrBreakoutAgent']).CorrBreakoutAgent(),
            is_futures=False, decline_guard=True))

    def _make_dca(self):
        from crypto_agents import ThreeCommasDCA
        raw = ThreeCommasDCA()
        # Настройки для малого капитала ($100)
        for attr, val in [('STEP_PCT', 0.015), ('TP_PCT', 0.025),
                          ('MAX_SAFETY', 2), ('SL_PCT', 0.05)]:
            if hasattr(raw, attr): setattr(raw, attr, val)
        return _wrap(raw, is_futures=False, decline_guard=True)

    def _make_gen_bear(self):
        # FIX: GeneticsBearish → GeneticsBearishAgent; +_GeneticsAdapter +_Genetics6to9Adapter
        try:
            from crypto_genetics import GeneticsBearishAgent
            from crypto_agents import _GeneticsAdapter
            return _wrap(_GeneticsAdapter(GeneticsBearishAgent()),
                         is_futures=True, is_genetics=True)
        except Exception:
            try:
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                return _wrap(_GeneticsAdapter(GeneticsAgent()),
                             is_futures=True, is_genetics=True)
            except Exception:
                # Запасной вариант: DualMomentum умеет шортить
                from crypto_agents import DualMomentum
                return _wrap(DualMomentum(), is_futures=False)


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК GAMMA — АГРЕССИВНЫЙ (максимизация прибыли в bull)
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Gamma(_SubPlayerBase):
    """
    Gamma: Агрессивный суб-игрок.
    Максимизация прибыли через лучших по Sharpe быстрых агентов.

    v6: ПОЛНАЯ ПЕРЕСТРОЙКА.
      ПРОБЛЕМА v5: CrossSectMomentum (0 сделок, lookback=7 дней),
        MomentumGuard (0 сделок, REBAL=5 дней), MomentumGuard_v2 (0 сделок).
        3 из 4 агентов мёртвые → Gamma делал 3 сделки через TrendCapture(-20%).
      РЕШЕНИЕ: только агенты с доказанной быстрой торговлей:
        - ThreeCommasDCA: +48.9%, Sharpe 1.46 (ЛУЧШИЙ по ret%)
        - GeneticsAgent: +14.9%, Sharpe 1.00
        - DualMomentum: +7.0%, Sharpe 0.78

    Все режимы: ThreeCommasDCA + GeneticsAgent + DualMomentum
    (режимное переключение убрано — все три агента работают в любом рынке)
    """
    NAME           = "Gamma(Aggressive)"
    BUY_THRESHOLD  = 0.20      # низкий порог: действуем при одном сигнале
    PEAK_STOP_PCT  = 0.25      # терпимее к просадке
    DAY_STOP_PCT   = 0.12      # -12% в день
    LOCK_TRIGGER   = 0.08      # +8% → начало фиксации
    LOCK_FRACTION  = 0.25      # 25% прибыли
    MAX_POSITIONS  = 5

    def _build_agents(self):
        # Главные агенты — все работают во всех режимах
        for name, factory in [
            ('ThreeCommasDCA', lambda: self._make_dca()),
            ('GeneticsAgent', lambda: self._make_genetics()),
            ('DualMom', lambda: _wrap(
                __import__('crypto_agents', fromlist=['DualMomentum']).DualMomentum(),
                is_futures=False)),
        ]:
            self._safe_load(name, factory)

    def _make_dca(self):
        from crypto_agents import ThreeCommasDCA
        raw = ThreeCommasDCA()
        # Настройки для малого капитала ($100):
        for attr, val in [('STEP_PCT', 0.015), ('TP_PCT', 0.025),
                          ('MAX_SAFETY', 2), ('SL_PCT', 0.06)]:
            if hasattr(raw, attr): setattr(raw, attr, val)
        return _wrap(raw, is_futures=False, decline_guard=True)

    def _make_genetics(self):
        # Исправление: используем _GeneticsAdapter (hourly gate) + _Genetics6to9Adapter
        # Исправление: запасной путь больше не создаёт _GeneticsAdapter('neutral') поверх строки
        try:
            from crypto_genetics import GeneticsAgent
            from crypto_agents import _GeneticsAdapter
            return _wrap(_GeneticsAdapter(GeneticsAgent()),
                         is_futures=False, is_genetics=True)
        except Exception:
            try:
                from crypto_agents import DualMomentum
                return _wrap(DualMomentum(), is_futures=False)
            except Exception:
                return None


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК PLAYER7WAYS — обёртка над Player7Ways из crypto_players
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Player7Ways(_SubPlayerBase):
    """
    Player7Ways: режимный суб-игрок из crypto_players.py.

    Один активный агент в каждый момент — специалист текущего режима рынка:
      Bear:    MomentumGuard_v2 (специалист по падениям)
      Neutral: MacroRotationAgent (ротация по макро-фазам)
      Bull:    GeneticsAgent / VolatilityBreakoutAgent (рост)

    Переключение по детектору режима (_detect_regime_live).
    Позиции закрываются при смене режима (минимизация drawdown).
    """
    NAME            = "P7Ways"
    BUY_THRESHOLD   = 0.01    # Player7Ways — один активный агент, всегда проходит
    MAX_POSITIONS   = 4
    PEAK_STOP_PCT   = 0.18
    DAY_STOP_PCT    = 0.10

    def _build_agents(self):
        """Используем Player7Ways как единый агент."""
        try:
            from crypto_players import Player7Ways
            p7w = Player7Ways()
            self._agents['P7Ways'] = p7w
            log.info("    [P7Ways] Player7Ways загружен OK")
        except Exception as e:
            log.warning("    [P7Ways] Player7Ways FAIL: %s — fallback на DualMom+VolBrk", e)
                # Запасной вариант: набор из двух быстрых агентов
            try:
                from crypto_agents import DualMomentum, VolatilityBreakoutAgent
                self._agents['DualMom'] = _wrap(DualMomentum(), is_futures=False)
                self._agents['VolBrk']  = _wrap(VolatilityBreakoutAgent(), is_futures=True)
            except Exception as e2:
                log.warning("    [P7Ways] Fallback FAIL: %s", e2)


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК ULTIMA — универсальный суб-игрок с 6 агентами
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Ultima(_SubPlayerBase):
    """
    Ultima: Универсальный суб-игрок с режимной ротацией 6-ти агентов.

    Состав:
      • GeneticsBullish  — нейросеть-бычий (лучший в bull, +24.7%)
      • GeneticsBearish  — нейросеть-медвежий (шортит в bear)
      • CorrBreakout     — разрывы корреляции (некоррелирован с momentum)
      • TrendCapture     — захват трендов (пик +138.3%)
      • DualMomentum     — кросс-таймфрейм momentum (+7%)
      • ThreeCommasDCA   — DCA-усреднение (Sharpe 1.46)

    Режимная логика:
      Bull:     GenBullish + TrendCapture + DualMom + 3CommasDCA (4 агента, агрессия)
      Bear:     GenBearish + DualMom (2 агента, шорт-фокус)
      Sideways: CorrBreakout + DualMom + 3CommasDCA (3 агента, range-bound)

    Низкий BUY_THRESHOLD (0.20) позволяет действовать при 1 сильном сигнале.
    Высокий MAX_POSITIONS (6) — больше позиций для диверсификации.
    """
    NAME            = "Ultima"
    BUY_THRESHOLD   = 0.20    # низкий порог — при 4 агентах один = 0.25 > 0.20
    MAX_POSITIONS   = 6
    PEAK_STOP_PCT   = 0.20
    DAY_STOP_PCT    = 0.10
    LOCK_TRIGGER    = 0.05
    LOCK_FRACTION   = 0.30

    def _build_agents(self):
        self._bull_agents:     Dict[str, object] = {}
        self._bear_agents:     Dict[str, object] = {}
        self._sideways_agents: Dict[str, object] = {}

        # ── GeneticsBullish ──
        self._safe_load('GenBull', lambda: self._make_gen('bullish'))
        if 'GenBull' in self._agents:
            self._bull_agents['GenBull'] = self._agents['GenBull']

        # ── GeneticsBearish ──
        self._safe_load('GenBear', lambda: self._make_gen('bearish'))
        if 'GenBear' in self._agents:
            self._bear_agents['GenBear'] = self._agents['GenBear']

        # ── CorrBreakout — sideways + bull ──
        self._safe_load('CorrBrk', lambda: _wrap(
            __import__('crypto_agents', fromlist=['CorrBreakoutAgent']).CorrBreakoutAgent(),
            is_futures=False))
        if 'CorrBrk' in self._agents:
            self._sideways_agents['CorrBrk'] = self._agents['CorrBrk']
            self._bull_agents['CorrBrk'] = self._agents['CorrBrk']

        # ── TrendCapture — bull (пик +138%) ──
        self._safe_load('TrendCap', lambda: _wrap(
            __import__('crypto_agents', fromlist=['TrendCapture']).TrendCapture(),
            is_futures=False))
        if 'TrendCap' in self._agents:
            self._bull_agents['TrendCap'] = self._agents['TrendCap']

        # ── DualMomentum — все режимы (быстрый, шортит) ──
        self._safe_load('DualMom', lambda: _wrap(
            __import__('crypto_agents', fromlist=['DualMomentum']).DualMomentum(),
            is_futures=False))
        if 'DualMom' in self._agents:
            self._bull_agents['DualMom'] = self._agents['DualMom']
            self._bear_agents['DualMom'] = self._agents['DualMom']
            self._sideways_agents['DualMom'] = self._agents['DualMom']

        # ── ThreeCommasDCA — bull + sideways (DCA) ──
        self._safe_load('3Commas', lambda: self._make_dca())
        if '3Commas' in self._agents:
            self._bull_agents['3Commas'] = self._agents['3Commas']
            self._sideways_agents['3Commas'] = self._agents['3Commas']

        # По умолчанию — sideways
        self._agents = dict(self._sideways_agents) or dict(self._bull_agents)
        log.info("  [Ultima] bull=%d  bear=%d  sideways=%d agents",
                 len(self._bull_agents), len(self._bear_agents), len(self._sideways_agents))

    def _make_gen(self, regime: str):
        # Исправление: правильные имена — GeneticsBullishAgent / GeneticsBearishAgent
        # Исправление: добавлены _GeneticsAdapter (hourly gate) и _Genetics6to9Adapter
        # Исправление: запасной путь больше не оборачивает строку в _GeneticsAdapter
        is_fut = (regime == 'bearish')
        try:
            import crypto_genetics as _cg
            from crypto_agents import _GeneticsAdapter
            cls_name = f'Genetics{"Bullish" if regime == "bullish" else "Bearish"}Agent'
            cls = getattr(_cg, cls_name)
            return _wrap(_GeneticsAdapter(cls()), is_futures=is_fut, is_genetics=True)
        except Exception:
            try:
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                return _wrap(_GeneticsAdapter(GeneticsAgent()),
                             is_futures=is_fut, is_genetics=True)
            except Exception:
                from crypto_agents import DualMomentum
                return _wrap(DualMomentum(), is_futures=False)

    def _make_dca(self):
        from crypto_agents import ThreeCommasDCA
        raw = ThreeCommasDCA()
        # Настройки для малого капитала ($100)
        for attr, val in [('STEP_PCT', 0.015), ('TP_PCT', 0.025),
                          ('MAX_SAFETY', 2), ('SL_PCT', 0.06)]:
            if hasattr(raw, attr): setattr(raw, attr, val)
        return _wrap(raw, is_futures=False, decline_guard=True)

    def _active_agents(self) -> Dict[str, object]:
        if self._regime == 'bull':   return self._bull_agents or self._agents
        if self._regime == 'bear':   return self._bear_agents or self._agents
        return self._sideways_agents or self._agents


# ══════════════════════════════════════════════════════════════════════════════
# СУБ-ИГРОК NEURO — управляет всеми вариациями генетических агентов
# ══════════════════════════════════════════════════════════════════════════════

class SubPlayer_Neuro(_SubPlayerBase):
    """
    Neuro: Суб-игрок на базе нейросетевых генетических агентов.

    Управляет всеми 4-мя обученными нейросетями:
      • GeneticsAgent    — универсальный (все режимы)
      • GeneticsBullish  — бычий специалист (+24.7% в bull)
      • GeneticsBearish  — медвежий специалист (шортит эффективно)
      • GeneticsNeutral  — sideways специалист (range-bound)

    Режимная логика:
      Bull:     GenBullish + GenAgent + GenNeutral  (3 агента, бычий доминирует)
      Bear:     GenBearish + GenAgent               (2 агента, медвежий доминирует)
      Sideways: GenNeutral + GenAgent + GenBullish  (3 агента, нейтральный доминирует)

    Низкий BUY_THRESHOLD (0.20) — нейросети уже имеют встроенную фильтрацию.
    """
    NAME            = "Neuro"
    BUY_THRESHOLD   = 0.20
    MAX_POSITIONS   = 5
    PEAK_STOP_PCT   = 0.18
    DAY_STOP_PCT    = 0.10
    LOCK_TRIGGER    = 0.05
    LOCK_FRACTION   = 0.30

    def _build_agents(self):
        self._bull_agents:     Dict[str, object] = {}
        self._bear_agents:     Dict[str, object] = {}
        self._sideways_agents: Dict[str, object] = {}

        # ── GeneticsAgent (универсальный) — все режимы ──
        self._safe_load('GenAgent', lambda: self._make_gen('neutral', is_fut=False))
        if 'GenAgent' in self._agents:
            self._bull_agents['GenAgent'] = self._agents['GenAgent']
            self._bear_agents['GenAgent'] = self._agents['GenAgent']
            self._sideways_agents['GenAgent'] = self._agents['GenAgent']

        # ── GeneticsBullish — bull + sideways ──
        self._safe_load('GenBullish', lambda: self._make_gen('bullish', is_fut=False))
        if 'GenBullish' in self._agents:
            self._bull_agents['GenBullish'] = self._agents['GenBullish']
            self._sideways_agents['GenBullish'] = self._agents['GenBullish']

        # ── GeneticsBearish — bear (шортит) ──
        self._safe_load('GenBearish', lambda: self._make_gen('bearish', is_fut=True))
        if 'GenBearish' in self._agents:
            self._bear_agents['GenBearish'] = self._agents['GenBearish']

        # ── GeneticsNeutral — sideways + bull ──
        self._safe_load('GenNeutral', lambda: self._make_gen_neutral())
        if 'GenNeutral' in self._agents:
            self._sideways_agents['GenNeutral'] = self._agents['GenNeutral']
            self._bull_agents['GenNeutral'] = self._agents['GenNeutral']

        # По умолчанию — sideways
        self._agents = dict(self._sideways_agents) or dict(self._bull_agents)
        log.info("  [Neuro] bull=%d  bear=%d  sideways=%d agents",
                 len(self._bull_agents), len(self._bear_agents), len(self._sideways_agents))

    def _make_gen(self, regime: str, is_fut: bool = False):
        # FIX: правильные имена классов: GeneticsBullishAgent (не GeneticsBullish)
        # FIX: оборачиваем в _GeneticsAdapter для hourly gate
        # FIX: _Genetics6to9Adapter для маппинга 6→9 action codes
        # Исправление: запасной путь создавал _GeneticsAdapter(string) и падал на каждом .act()
        try:
            import crypto_genetics as _cg
            from crypto_agents import _GeneticsAdapter
            cls_map = {
                'neutral': 'GeneticsAgent',
                'bullish': 'GeneticsBullishAgent',
                'bearish': 'GeneticsBearishAgent',
            }
            cls = getattr(_cg, cls_map[regime])
            return _wrap(_GeneticsAdapter(cls()), is_futures=is_fut, is_genetics=True)
        except Exception:
            try:
                # Запасной вариант: хотя бы универсальный GeneticsAgent
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                return _wrap(_GeneticsAdapter(GeneticsAgent()), is_futures=is_fut, is_genetics=True)
            except Exception:
                from crypto_agents import DualMomentum
                return _wrap(DualMomentum(), is_futures=False)

    def _make_gen_neutral(self):
        # FIX: правильное имя класса GeneticsNeutralAgent (не GeneticsNeutral)
        # FIX: _Genetics6to9Adapter для маппинга 6→9 action codes
        try:
            import crypto_genetics as _cg
            from crypto_agents import _GeneticsAdapter
            return _wrap(_GeneticsAdapter(_cg.GeneticsNeutralAgent()), is_futures=False, is_genetics=True)
        except Exception:
            try:
                from crypto_genetics import GeneticsAgent
                from crypto_agents import _GeneticsAdapter
                return _wrap(_GeneticsAdapter(GeneticsAgent()), is_futures=False, is_genetics=True)
            except Exception:
                from crypto_agents import CorrBreakoutAgent
                return _wrap(CorrBreakoutAgent(), is_futures=False)

    def _active_agents(self) -> Dict[str, object]:
        if self._regime == 'bull':   return self._bull_agents or self._agents
        if self._regime == 'bear':   return self._bear_agents or self._agents
        return self._sideways_agents or self._agents


# ══════════════════════════════════════════════════════════════════════════════
# MASTERPLAYER v6 — оркестратор 6-ти суб-игроков (+ Neuro)
# ══════════════════════════════════════════════════════════════════════════════

class MasterPlayer:
    """
    MasterPlayer v6: 6 суб-игроков, каждый с $INITIAL_CAPITAL.
    Агрегирует их сигналы взвешенно.
    Новый: Neuro — управляет всеми генетическими нейросетями.
    """

    def __init__(self):
        cap = INITIAL_CAPITAL
        # Каждый суб-игрок работает НЕЗАВИСИМО с полным капиталом $INITIAL_CAPITAL
        # Цель: сравнить 6 стратегий в реальных условиях, выбрать лучшую
        self._players = OrderedDict([
            ('Alpha',  SubPlayer_Alpha(capital_fraction=1.0)),
            ('Beta',   SubPlayer_Beta(capital_fraction=1.0)),
            ('Gamma',  SubPlayer_Gamma(capital_fraction=1.0)),
            ('P7Ways', SubPlayer_Player7Ways(capital_fraction=1.0)),
            ('Ultima', SubPlayer_Ultima(capital_fraction=1.0)),
            ('Neuro',  SubPlayer_Neuro(capital_fraction=1.0)),
        ])
        # Суммарный PV = сумма всех пяти
        n_players = len(self._players)
        self._pv          = INITIAL_CAPITAL * n_players
        self._pv_peak     = INITIAL_CAPITAL * n_players
        self._pv_history  = [INITIAL_CAPITAL * n_players]
        self._live_start_idx = 0
        self.t            = 0
        log.info("MasterPlayer v6 — %d НЕЗАВИСИМЫХ игроков по $%.0f каждый ($%.0f всего)",
                 n_players, INITIAL_CAPITAL, INITIAL_CAPITAL * n_players)

    # ── act ──────────────────────────────────────────────────────────────────
    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None) -> dict:
        self.t = bar_index if bar_index is not None else self.t + 1
        if portfolio_value is not None:
            self._pv = max(0.1, float(portfolio_value))
        self._pv_history.append(self._pv)
        if self._pv > self._pv_peak: self._pv_peak = self._pv

        # v5: Правильная агрегация сигналов вместо last-wins
        # Правило: sell/close (3,8) имеет приоритет над buy (1,2,4-7).
        # При конфликте buy-сигналов: берём сигнал от игрока с лучшим текущим PnL.
        all_acts: Dict[str, Dict[str, int]] = {}  # sym → {player_name: action}
        for pname, player in self._players.items():
            try:
                acts = player.act(prices, volumes, month=month,
                                   portfolio_value=player._pv, bar_index=bar_index)
            except Exception as e:
                log.error("  [MasterPlayer/%s] act error: %s", pname, e)
                acts = {}
            for sym, a in acts.items():
                if a != 0:
                    all_acts.setdefault(sym, {})[pname] = a

        result = {}
        for sym, player_acts in all_acts.items():
            # Приоритет 1: sell/close сигналы (любой игрок хочет закрыть → закрываем)
            sells = {pn: a for pn, a in player_acts.items() if a in (3, 8)}
            buys  = {pn: a for pn, a in player_acts.items() if a not in (3, 8)}

            if sells:
                # Берём первый sell-сигнал (3=spot, 8=futures)
                result[sym] = next(iter(sells.values()))
            elif buys:
                if len(buys) == 1:
                    result[sym] = next(iter(buys.values()))
                else:
                    # При конфликте buy-сигналов: игрок с лучшим текущим ret% решает
                    best_player = max(buys.keys(),
                                       key=lambda pn: (self._players[pn]._pv
                                                        / self._players[pn]._capital - 1))
                    result[sym] = buys[best_player]

        return result

    # ── Reset после warmup ───────────────────────────────────────────────────
    def reset_after_warmup(self, prices):
        self._live_start_idx = len(self._pv_history)
        for player in self._players.values():
            player.reset_after_warmup(prices)
        log.info("MasterPlayer v6: reset_after_warmup OK (live_start=%d)", self._live_start_idx)

    # ── Полный статус ────────────────────────────────────────────────────────
    def get_detailed_status(self, prices: dict) -> dict:
        # Суммарный PV = сумма трёх независимых
        self._pv = sum(p._pv for p in self._players.values())
        if self._pv > self._pv_peak: self._pv_peak = self._pv
        dd = (self._pv_peak - self._pv) / max(self._pv_peak, 1)
        total_locked = sum(p._locked_usdt for p in self._players.values())
        players_status = {name: p.get_status(prices) for name, p in self._players.items()}
        return {
            'timestamp':     datetime.now(timezone.utc).isoformat(),
            'bar':           self.t,
            'portfolio': {
                'value':        round(self._pv, 4),
                'peak':         round(self._pv_peak, 4),
                'dd_pct':       round(dd * 100, 2),
                'ret_pct':      round((self._pv / INITIAL_CAPITAL - 1) * 100, 3),
                'locked_usdt':  round(total_locked, 4),
                'total_usdt':   round(self._pv + total_locked, 4),
            },
            'players': players_status,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ДАШБОРДЫ
# ══════════════════════════════════════════════════════════════════════════════

def _save_usd_dashboard(mp: MasterPlayer, prices: dict, output_dir: str):
    """
    USD-дашборд: показывает состояние счёта в USD с разбивкой.
    4 панели:
      1. График total PV + locked_usdt
      2. Разбивка по суб-игрокам: cash / crypto / locked
      3. Состояние стопов (DayStop / PeakStop)
      4. Текущие позиции с PnL в USD
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        status = mp.get_detailed_status(prices)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(f"MasterPlayer v6  —  USD Дашборд  [{now_str}]",
                     fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                               height_ratios=[1.2, 1.0, 1.0])

        P_CLR = {'Alpha': '#3498db', 'Beta': '#27ae60', 'Gamma': '#e74c3c', 'P7Ways': '#9b59b6', 'Ultima': '#f39c12', 'Neuro': '#1abc9c'}
        R_CLR = {'bull': '#2ecc71', 'bear': '#e74c3c', 'sideways': '#f39c12'}

        # ── [0, :] Кривая PV + locked ──────────────────────────────────────
        ax_pv = fig.add_subplot(gs[0, :])
        pv_h  = mp._pv_history
        xs    = list(range(len(pv_h)))
        if len(pv_h) > 1:
            ax_pv.plot(xs, pv_h, lw=2, color='#2980b9', label='Total PV ($)', zorder=3)
            ax_pv.axhline(INITIAL_CAPITAL, color='gray', ls='--', lw=1, alpha=0.6)
            # locked_usdt track (approximate: sum of players' locked history)
            total_locked = status['portfolio']['locked_usdt']
            if total_locked > 0:
                ax_pv.axhline(INITIAL_CAPITAL + total_locked,
                              color='#8e44ad', ls=':', lw=1.5, alpha=0.7,
                              label=f'locked_USDT =${total_locked:.2f}')
        # Кривые PV по суб-игрокам (нормализуем к INITIAL_CAPITAL на старте live)
        for name, p in mp._players.items():
            pv_live = p._pv_history[p._live_start_idx:]
            if len(pv_live) > 1:
                # FIX: нормализуем чтобы все кривые начинались с INITIAL_CAPITAL
                base = pv_live[0] if pv_live[0] > 0 else 1.0
                pv_norm = [INITIAL_CAPITAL * (v / base) for v in pv_live]
                xs2 = [p._live_start_idx + i for i in range(len(pv_live))]
                ax_pv.plot(xs2, pv_norm, lw=1.2, alpha=0.5,
                           color=P_CLR.get(name,'gray'), label=f'{name}', ls='--')
        ax_pv.set_title(
            f"Портфель: ${status['portfolio']['value']:.2f}  "
            f"({status['portfolio']['ret_pct']:+.2f}%)  "
            f"+ Locked: ${status['portfolio']['locked_usdt']:.2f}  "
            f"= Total USDT: ${status['portfolio']['total_usdt']:.2f}  "
            f"MaxDD: {status['portfolio']['dd_pct']:.2f}%",
            fontsize=10, fontweight='bold')
        ax_pv.set_ylabel('USDT'); ax_pv.grid(True, alpha=0.25)
        ax_pv.legend(loc='upper left', fontsize=8)
        ax_pv.set_xlabel('Bar #')

        # ── [1, 0] USD разбивка по суб-игрокам: Cash vs Крипто ──────────────
        ax_usd = fig.add_subplot(gs[1, 0])
        player_names = list(mp._players.keys())
        pv_vals     = [status['players'][n]['pv'] for n in player_names]
        locked_vals = [status['players'][n]['locked_usdt'] for n in player_names]
        cash_vals   = [status['players'][n].get('cash_usdt',  pv_vals[i]) for i,n in enumerate(player_names)]
        crypto_vals = [status['players'][n].get('crypto_value', 0.0)       for i,n in enumerate(player_names)]
        x = np.arange(len(player_names))

        b1 = ax_usd.bar(x - 0.22, cash_vals,   width=0.38,
                         color='#27ae60', alpha=0.85, label='Cash USDT')
        b2 = ax_usd.bar(x + 0.22, crypto_vals, width=0.38,
                         color='#e67e22', alpha=0.80, label='Крипто-активы')
        ax_usd.bar(x,              locked_vals, width=0.14,
                   color='#8e44ad', alpha=0.70, label='Locked')
        ax_usd.set_xticks(x); ax_usd.set_xticklabels([n[:5] for n in player_names])
        ax_usd.set_title('Cash USDT vs Крипто-активы vs Locked\n(2 отдельных сущности)',
                          fontsize=9, fontweight='bold')
        ax_usd.set_ylabel('USDT'); ax_usd.legend(fontsize=7)
        ax_usd.grid(True, alpha=0.3, axis='y')
        ax_usd.axhline(INITIAL_CAPITAL, color='red', ls='--', lw=0.8, alpha=0.4)
        for bar, v in zip(b1, cash_vals):
            if v > 2:
                ax_usd.text(bar.get_x()+bar.get_width()/2, v+0.5,
                            f'${v:.0f}', ha='center', va='bottom', fontsize=7.5, color='#155733')
        for bar, v in zip(b2, crypto_vals):
            if v > 2:
                ax_usd.text(bar.get_x()+bar.get_width()/2, v+0.5,
                            f'${v:.0f}', ha='center', va='bottom', fontsize=7.5, color='#8a4500')

        # ── [1, 1] Режим и стопы ──────────────────────────────────────────
        ax_st = fig.add_subplot(gs[1, 1])
        ax_st.axis('off')
        lines = ["СТАТУС СУПЛЕЙЕРОВ:\n" + "─"*36]
        for name, player in mp._players.items():
            ps = status['players'][name]
            regime_clr = R_CLR.get(ps['regime'], '#888')
            ds = "⛔ СТОП" if ps['day_stopped']  else f"day_dd={ps['day_dd_pct']:+.1f}%"
            pk = "⛔ СТОП" if ps['peak_stopped'] else f"dd={ps['dd_pct']:.1f}%"
            lines.append(
                f"{name[:5]:5s}  {ps['regime'].upper():8s}  {ds}  {pk}\n"
                f"       Позиций: {ps['n_positions']}  Сделок: {ps['total_trades']}"
            )
        ax_st.text(0.03, 0.97, "\n".join(lines), transform=ax_st.transAxes,
                   fontsize=8.5, va='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.95))

    # ── [1, 2] Прогресс фиксации прибыли ──────────────────────────────
        ax_lk = fig.add_subplot(gs[1, 2])
        ax_lk.axis('off')
        lock_lines = ["ФИКСАЦИЯ ПРИБЫЛИ (Profit Lock):\n" + "─"*36]
        for name, player in mp._players.items():
            ps   = status['players'][name]
            pv   = ps['pv']
            lock = ps['locked_usdt']
            cash = ps.get('cash_usdt', pv)
            cryp = ps.get('crypto_value', 0.0)
            cap = INITIAL_CAPITAL
            ret_pct = (ps.get('total_usdt', pv + lock) / max(cap, 1) - 1) * 100
            lock_pct = lock / max(cap, 1) * 100
            lock_lines.append(
                f"{name[:5]:5s}  Cash=${cash:.1f}  Крипто=${cryp:.1f}\n"
                f"       Locked=${lock:.2f} ({lock_pct:.1f}%)  {ret_pct:+.2f}%"
            )
        lock_lines.append("\n" + "─"*36)
        lock_lines.append(
            f"ИТОГО  Active=${status['portfolio']['value']:.2f}\n"
            f"  + Locked: ${status['portfolio']['locked_usdt']:.2f}\n"
            f"  = TOTAL USDT: ${status['portfolio']['total_usdt']:.2f}"
        )
        ax_lk.text(0.03, 0.97, "\n".join(lock_lines), transform=ax_lk.transAxes,
                   fontsize=8.5, va='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.95))

        # ── [2, :] Текущие позиции с USD ──────────────────────────────────
        ax_pos = fig.add_subplot(gs[2, :])
        ax_pos.axis('off')
        all_positions = []
        for name, player in mp._players.items():
            for pos in status['players'][name]['positions']:
                pos['player'] = name
                all_positions.append(pos)

        if all_positions:
            HDR = (f"{'Игрок':<7}{'Символ':<9}{'Тип':<11}{'Агент':<13}"
                   f"{'Вход$':>7}{'Тек.$':>8}{'PnL%':>7}{'PnL$':>7}{'Hold':>6}")
            sep  = '─' * len(HDR)
            rows = []
            cap_each = INITIAL_CAPITAL
            for p in all_positions:
                held  = p.get('held_bars', 0)
                pnl_p = p.get('pnl_pct', 0)
                pos_usdt = min(cap_each * TRADE_FRACTION, 120.0)
                pnl_usd  = pos_usdt * pnl_p / 100
                rows.append(
                    f"{p['player'][:7]:<7}{p['sym']:<9}{p['type']:<11}"
                    f"{p['agent'][:12]:<13}"
                    f"{pos_usdt:>7.1f}{pos_usdt+pnl_usd:>8.1f}"
                    f"{pnl_p:>+6.2f}%{pnl_usd:>+7.2f}{held:>5}b"
                )
            t_usdt = sum(min(INITIAL_CAPITAL/3*TRADE_FRACTION,120) for _ in all_positions)
            t_pnl  = sum(min(INITIAL_CAPITAL/3*TRADE_FRACTION,120)*p.get('pnl_pct',0)/100 for p in all_positions)
            rows.append(sep)
            rows.append(f"{'Итого крипто-активов:':<40} {t_usdt:>7.1f}{t_usdt+t_pnl:>8.1f}"
                        f"{'':>7}{t_pnl:>+7.2f}")
            text = (f"ПОЗИЦИИ — Крипто-активы в USD ({len(all_positions)} поз.):\n"
                    f"{HDR}\n{sep}\n" + "\n".join(rows))
        else:
            text = "ОТКРЫТЫЕ ПОЗИЦИИ: нет\n(все средства в Cash USDT)"

        ax_pos.text(0.01, 0.99, text, transform=ax_pos.transAxes,
                    fontsize=8.5, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#fff9e7', alpha=0.95))

        out = os.path.join(output_dir, "usd_dashboard.png")
        fig.savefig(out, dpi=130, bbox_inches='tight')
        fig.clf(); plt.close(fig)
        log.info("  [usd_dashboard] → %s", out)
    except Exception as e:
        log.warning("  [usd_dashboard] failed: %s\n%s", e, traceback.format_exc())


def _save_master_detail(mp: MasterPlayer, prices: dict, output_dir: str):
    """Детальный JSON + PNG для отладки суб-игроков.
    Все кривые нормализованы: 0% в начале live-торговли."""
    status = mp.get_detailed_status(prices)
    json_path = os.path.join(output_dir, "master_detail.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_players = len(mp._players)
        n_cols = min(n_players, 5)
        fig = plt.figure(figsize=(5 * n_cols, 14))
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        fig.suptitle(f"MasterPlayer v6  —  Детальный дашборд  [{now_str}]",
                     fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.50, wspace=0.35)
        P_CLR = {'Alpha': '#3498db', 'Beta': '#27ae60', 'Gamma': '#e74c3c', 'P7Ways': '#9b59b6', 'Ultima': '#f39c12', 'Neuro': '#1abc9c'}

    # Истории PV по игрокам — нормализация от точки старта live = 0%
        for i, (name, player) in enumerate(mp._players.items()):
            ax = fig.add_subplot(gs[0, i])
            pv_live = player._pv_history[player._live_start_idx:]
            if len(pv_live) > 1:
                base_val = pv_live[0] if pv_live[0] > 0 else 1.0
                xs = list(range(len(pv_live)))
                ret = [(v / base_val - 1) * 100 for v in pv_live]
                clr = P_CLR.get(name, '#888')
                ax.fill_between(xs, ret, 0, alpha=0.2, color=clr)
                ax.plot(xs, ret, lw=1.8, color=clr)
                ax.axhline(0, color='gray', ls='--', lw=1)
                if player._locked_usdt > 0:
                    ax.axhline(player._locked_usdt / base_val * 100,
                               color='#8e44ad', ls=':', lw=1, alpha=0.7)
            pv_final = player._pv
            base_cap = pv_live[0] if (pv_live and pv_live[0] > 0) else INITIAL_CAPITAL
            ret_val = (pv_final / base_cap - 1) * 100
            ax.set_title(f"{name}\n${pv_final:.2f} ({ret_val:+.2f}%)  "
                         f"Locked:${player._locked_usdt:.1f}  "
                         f"Режим:{player._regime.upper()}",
                         fontsize=9, fontweight='bold')
            ax.set_ylabel('Return %'); ax.grid(True, alpha=0.25)

        # Agents active per player
        for i, (name, player) in enumerate(mp._players.items()):
            ax = fig.add_subplot(gs[1, i])
            ax.axis('off')
            ps = status['players'][name]
            lines = [f"{name} — активные агенты:"]
            lines += [f"  • {a}" for a in ps['agents_active']] or ["  (нет)"]
            lines.append("")
            lines.append(f"Позиций: {ps['n_positions']}  Сделок: {ps['total_trades']}")
            lines.append(f"Day DD: {ps['day_dd_pct']:+.2f}%  {'⛔СТОП' if ps['day_stopped'] else 'OK'}")
            lines.append(f"Peak DD: {ps['dd_pct']:.2f}%  {'⛔СТОП' if ps['peak_stopped'] else 'OK'}")
            lines.append("")
            for tr in ps['recent_trades'][-5:]:
                pnl_s = f" pnl={tr.get('pnl_pct',0):+.2f}%" if tr.get('pnl_pct') is not None else ""
                lines.append(f"  b{tr['bar']} {tr['action']} {tr['sym']}{pnl_s}")
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                    fontsize=9, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.95))

        out = os.path.join(output_dir, "master_detail.png")
        fig.savefig(out, dpi=130, bbox_inches='tight')
        fig.clf(); plt.close(fig)
        log.info("  [master_detail] → %s", out)
    except Exception as e:
        log.warning("  [master_detail] png failed: %s\n%s", e, traceback.format_exc())


def _save_player_agent_dashboards(players: dict, output_dir: str):
    """
    Для каждого игрока (Alpha/Beta/Gamma) генерирует отдельный дашборд
    с анализом агентов, входящих в его состав:
      Верхняя панель — кумулятивный PnL% по каждому агенту (из trade history)
      Нижняя панель — таблица: агент, сделки, win/loss, суммарный PnL$, ср. PnL%
    Удобно для пересмотра состава агентов.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        P_CLR_MAP = {'Alpha': '#3498db', 'Beta': '#27ae60', 'Gamma': '#e74c3c', 'P7Ways': '#9b59b6', 'Ultima': '#f39c12', 'Neuro': '#1abc9c'}
        AGENT_CLRS = plt.cm.tab10(np.linspace(0, 0.9, 10))
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        for pname, player in players.items():
            trades = player._all_trades
            # FIX: Показываем дашборд даже если нет закрытых сделок,
            # но есть открытые позиции или агенты
            has_positions = bool(player._open_pos)
            has_agents = bool(player._agents) or bool(getattr(player, '_bull_agents', {}))
            if not trades and not has_positions and not has_agents:
                continue

            # Собираем PnL по агентам из истории сделок
            agent_names_from_trades = sorted(set(t.get('agent', '?') for t in trades)) if trades else []
            # Также добавляем агентов из открытых позиций
            agent_names_from_pos = sorted(set(p.get('agent', '?') for p in player._open_pos.values()))
            # И из текущих активных агентов
            active_names = list(player._active_agents().keys()) if hasattr(player, '_active_agents') else list(player._agents.keys())
            agent_names = sorted(set(agent_names_from_trades + agent_names_from_pos + active_names))
            # Кумулятивный PnL по каждому агенту (только CLOSE-сделки)
            agent_cum_pnl: Dict[str, List[Tuple[int, float]]] = {a: [] for a in agent_names}
            agent_stats: Dict[str, dict] = {}
            pos_size = min(player._capital * TRADE_FRACTION, 120.0)

            for aname in agent_names:
                closes = [t for t in trades if t.get('agent') == aname and t.get('action') == 'CLOSE']
                cum = 0.0
                wins, losses = 0, 0
                total_pnl_usd = 0.0
                pnl_pcts = []
                cum_series = [(0, 0.0)]  # (trade_index, cum_pnl_pct)
                for idx, tr in enumerate(closes):
                    pnl_p = tr.get('pnl_pct', 0)
                    pnl_pcts.append(pnl_p)
                    pnl_usd = pos_size * pnl_p / 100.0
                    total_pnl_usd += pnl_usd
                    cum += pnl_p
                    cum_series.append((idx + 1, cum))
                    if pnl_p >= 0:
                        wins += 1
                    else:
                        losses += 1
                agent_cum_pnl[aname] = cum_series
                # Открытые позиции этого агента (BUY/LONG/SHORT без CLOSE)
                opens = len([t for t in trades if t.get('agent') == aname and t.get('action') != 'CLOSE'])
                agent_stats[aname] = {
                    'trades': len(closes),
                    'opens': opens,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / max(wins + losses, 1) * 100,
                    'total_pnl_usd': total_pnl_usd,
                    'cum_pnl_pct': cum,
                    'avg_pnl_pct': float(np.mean(pnl_pcts)) if pnl_pcts else 0.0,
                }

            # ── Рисуем дашборд ─────────────────────────────────────────
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f"{pname}  —  Анализ агентов  [{now_str}]",
                         fontsize=13, fontweight='bold',
                         color=P_CLR_MAP.get(pname, '#333'))
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                                   height_ratios=[1.3, 1.0])

            # ── [0, 0] Кумулятивный PnL% по агентам ──────────────────
            ax_cum = fig.add_subplot(gs[0, 0])
            for ci, aname in enumerate(agent_names):
                series = agent_cum_pnl[aname]
                if len(series) > 1:
                    xs = [s[0] for s in series]
                    ys = [s[1] for s in series]
                    st = agent_stats[aname]
                    ax_cum.plot(xs, ys, lw=1.8, color=AGENT_CLRS[ci % 10],
                                marker='o', markersize=3, alpha=0.85,
                                label=f"{aname} ({st['cum_pnl_pct']:+.1f}%)")
            ax_cum.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
            ax_cum.set_title("Кумулятивный PnL% по агентам (closed trades)",
                             fontsize=10, fontweight='bold')
            ax_cum.set_xlabel('Сделка #'); ax_cum.set_ylabel('Cum PnL %')
            ax_cum.legend(fontsize=7, loc='best'); ax_cum.grid(True, alpha=0.25)

    # ── [0, 1] Столбчатый график: суммарный PnL$ по агентам ──────────
            ax_bar = fig.add_subplot(gs[0, 1])
            sorted_agents = sorted(agent_names,
                                    key=lambda a: agent_stats[a]['total_pnl_usd'],
                                    reverse=True)
            y_pos = np.arange(len(sorted_agents))
            pnl_vals = [agent_stats[a]['total_pnl_usd'] for a in sorted_agents]
            bar_clrs = ['#27ae60' if v >= 0 else '#e74c3c' for v in pnl_vals]
            bars = ax_bar.barh(y_pos, pnl_vals, color=bar_clrs, alpha=0.85, height=0.6)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(sorted_agents, fontsize=8)
            ax_bar.axvline(0, color='black', lw=1)
            ax_bar.set_title("Суммарный PnL$ по агентам", fontsize=10, fontweight='bold')
            ax_bar.set_xlabel('PnL (USDT)'); ax_bar.grid(True, axis='x', alpha=0.3)
            for bar, val in zip(bars, pnl_vals):
                if abs(val) > 0.01:
                    ax_bar.text(val + (0.3 if val >= 0 else -0.3),
                                bar.get_y() + bar.get_height() / 2,
                                f"${val:+.2f}", va='center', fontsize=7.5,
                                ha='left' if val >= 0 else 'right', fontweight='bold')

            # ── [1, :] Таблица статистики агентов ─────────────────────
            ax_tbl = fig.add_subplot(gs[1, :])
            ax_tbl.axis('off')
            HDR = (f"{'Агент':<18}{'Закр.':>6}{'Откр.':>6}{'Win':>5}{'Loss':>5}"
                   f"{'WinR%':>7}{'CumPnL%':>9}{'AvgPnL%':>9}{'PnL$':>9}")
            sep = '─' * len(HDR)
            rows = [f"  {pname} — Статистика агентов", sep, HDR, sep]
            for aname in sorted_agents:
                st = agent_stats[aname]
                rows.append(
                    f"{aname:<18}{st['trades']:>6}{st['opens']:>6}"
                    f"{st['wins']:>5}{st['losses']:>5}"
                    f"{st['win_rate']:>6.1f}%{st['cum_pnl_pct']:>+8.2f}%"
                    f"{st['avg_pnl_pct']:>+8.2f}%{st['total_pnl_usd']:>+9.2f}"
                )
            rows.append(sep)
            # Итого по игроку
            total_trades = sum(agent_stats[a]['trades'] for a in agent_names)
            total_wins   = sum(agent_stats[a]['wins'] for a in agent_names)
            total_losses = sum(agent_stats[a]['losses'] for a in agent_names)
            total_pnl    = sum(agent_stats[a]['total_pnl_usd'] for a in agent_names)
            total_wr     = total_wins / max(total_wins + total_losses, 1) * 100
            rows.append(
                f"{'ИТОГО':<18}{total_trades:>6}{'':>6}"
                f"{total_wins:>5}{total_losses:>5}"
                f"{total_wr:>6.1f}%{'':>9}{'':>9}{total_pnl:>+9.2f}"
            )
            rows.append("")
            rows.append(f"Капитал: ${player._capital:.0f}  "
                        f"PV: ${player._pv:.2f}  "
                        f"Locked: ${player._locked_usdt:.2f}  "
                        f"Режим: {player._regime.upper()}")
            # Дополнительно: открытые позиции
            if player._open_pos:
                rows.append("")
                rows.append("Открытые позиции:")
                for sym, info in player._open_pos.items():
                    ep = info.get('entry_price', 0)
                    tp = info.get('type', '?')
                    ag = info.get('agent', '?')
                    rows.append(f"  {sym:8s} {tp:10s} agent={ag}  entry=${ep:.4f}")

            ax_tbl.text(0.02, 0.97, "\n".join(rows), transform=ax_tbl.transAxes,
                        fontsize=8.5, va='top', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='#fafafa', alpha=0.95))

            out = os.path.join(output_dir, f"player_{pname.lower()}_agents.png")
            fig.savefig(out, dpi=130, bbox_inches='tight')
            fig.clf(); plt.close(fig)
            log.info("  [player_agents] → %s", out)

    except Exception as e:
        log.warning("  [player_agent_dashboards] failed: %s\n%s", e, traceback.format_exc())


def _save_live_growth(mp: MasterPlayer, output_dir: str):
    """Live-кривая доходности всех суб-игроков + MasterPlayer.
    Все кривые начинаются из 0% (нормализация по первому live-значению)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 5))
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        P_CLR = {'Alpha': '#3498db', 'Beta': '#27ae60', 'Gamma': '#e74c3c', 'P7Ways': '#9b59b6', 'Ultima': '#f39c12', 'Neuro': '#1abc9c'}

    # Общая кривая MasterPlayer
        pv_h = mp._pv_history[mp._live_start_idx:]
        if len(pv_h) > 1:
            base_mp = pv_h[0] if pv_h[0] > 0 else 1.0
            xs  = list(range(len(pv_h)))
            ret = [(v / base_mp - 1) * 100 for v in pv_h]
            clr = '#2ecc71' if ret[-1] >= 0 else '#e74c3c'
            ax.fill_between(xs, ret, 0, alpha=0.12, color=clr)
            ax.plot(xs, ret, lw=2.5, color=clr, label=f'MasterPlayer {ret[-1]:+.2f}%')

    # Суб-игроки
        for name, player in mp._players.items():
            pv_live = player._pv_history[player._live_start_idx:]
            if len(pv_live) > 1:
                base_p = pv_live[0] if pv_live[0] > 0 else 1.0
                xs2  = list(range(len(pv_live)))
                ret2 = [(v / base_p - 1) * 100 for v in pv_live]
                ax.plot(xs2, ret2, lw=1.2, ls='--', alpha=0.7,
                        color=P_CLR.get(name,'gray'),
                        label=f'{name} {ret2[-1]:+.2f}%')

        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_title(f"MasterPlayer v6 — live доходность [{now_str}]  "
                     f"live_bars={len(pv_h)}",
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Return %'); ax.set_xlabel('Bar (live)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.25)
        out = os.path.join(output_dir, "master_live_growth.png")
        fig.savefig(out, dpi=130, bbox_inches='tight')
        fig.clf(); plt.close(fig)
        log.info("  [live_growth] → %s", out)
    except Exception as e:
        log.warning("  [live_growth] failed: %s\n%s", e, traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
# ФАБРИКА
# ──────────────────────────────────────────────────────────────────────────────

def make_master_player() -> OrderedDict:
    """Один агрегированный MasterPlayer (для совместимости с bridge)."""
    agents = OrderedDict()
    agents['MasterPlayer'] = MasterPlayer()
    return agents


def make_independent_players() -> OrderedDict:
    """
    6 НЕЗАВИСИМЫХ агентов по $INITIAL_CAPITAL каждый.
    Каждый отображается в portfolio_history.csv отдельной колонкой.

    Alpha  — адаптивный (CorrBreakout + GeneticsBull/Bear по режиму)
    Beta   — защитный   (ThreeCommasDCA + GenBearish + CorrBreakout)
    Gamma  — агрессивный (ThreeCommasDCA + GeneticsAgent + DualMom)
    P7Ways — режимный   (Player7Ways: один специалист на режим)
    Ultima — универсальный (GenBull + GenBear + CorrBrk + TrendCap + DualMom + 3Commas)
    Neuro  — нейросетевой (GenAgent + GenBullish + GenBearish + GenNeutral)
    """
    return OrderedDict([
        ('Alpha',  SubPlayer_Alpha(capital_fraction=1.0)),
        ('Beta',   SubPlayer_Beta(capital_fraction=1.0)),
        ('Gamma',  SubPlayer_Gamma(capital_fraction=1.0)),
        ('P7Ways', SubPlayer_Player7Ways(capital_fraction=1.0)),
        ('Ultima', SubPlayer_Ultima(capital_fraction=1.0)),
        ('Neuro',  SubPlayer_Neuro(capital_fraction=1.0)),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# ВСТРОЕННЫЙ ЗАПУСК
# ──────────────────────────────────────────────────────────────────────────────


def _apply_player_settings_single(player: _SubPlayerBase, cfg: dict):
    """Применяет настройки из settings.txt к одному суб-игроку."""
    day_stop_en  = cfg.get('day_stop_enabled',     False)
    day_stop_pct = float(cfg.get('day_stop_pct',   10.0))
    peak_stop_en  = cfg.get('peak_stop_enabled',   False)
    peak_stop_pct = float(cfg.get('peak_stop_pct', 20.0))
    lock_en   = cfg.get('profit_lock_enabled',  False)
    lock_trig = float(cfg.get('profit_lock_trigger', 5.0))
    lock_step = float(cfg.get('profit_lock_step',    5.0))
    lock_frac = float(cfg.get('profit_lock_fraction', 0.30))
    lock_max  = float(cfg.get('profit_lock_max',  0.70))

    player.DAY_STOP_ENABLED = day_stop_en
    player.DAY_STOP_PCT     = day_stop_pct / 100.0
    player.PEAK_STOP_PCT    = peak_stop_pct / 100.0 if peak_stop_en else 99.0
    if lock_en:
        player.LOCK_TRIGGER  = lock_trig / 100.0
        player.LOCK_STEP     = lock_step / 100.0
        player.LOCK_FRACTION = lock_frac
        player.LOCK_MAX      = lock_max
        player._lock_next_at = player._capital * (1.0 + player.LOCK_TRIGGER)
    else:
        player._lock_next_at = float('inf')


class _PlayerWrapper:
    """Обёртка для совместимости с _save_master_detail/_save_live_growth."""
    def __init__(self, players: dict):
        self._players    = players
        # Агрегируем PV-историю из суб-игроков (bar-by-bar sum)
        if players:
            max_len = max(len(p._pv_history) for p in players.values())
            pv_agg = []
            for i in range(max_len):
                total = 0.0
                for p in players.values():
                    if i < len(p._pv_history):
                        total += p._pv_history[i]
                    else:
                        # если у игрока меньше баров — берём последнее значение
                        total += p._pv_history[-1] if p._pv_history else 0.0
                pv_agg.append(total)
            self._pv_history = pv_agg
        else:
            self._pv_history = [0.0]
        self._live_start_idx = min(p._live_start_idx for p in players.values()) if players else 0
        self.t           = max((p.t for p in players.values()), default=0)

    def get_detailed_status(self, prices: dict) -> dict:
        total_pv     = sum(p._pv for p in self._players.values())
        total_peak   = sum(p._pv_peak for p in self._players.values())
        total_locked = sum(p._locked_usdt for p in self._players.values())
        dd = (total_peak - total_pv) / max(total_peak, 1)
        players_status = {name: p.get_status(prices) for name, p in self._players.items()}
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'bar': self.t,
            'portfolio': {
                'value':        round(total_pv, 4),
                'peak':         round(total_peak, 4),
                'dd_pct':       round(dd * 100, 2),
                'ret_pct':      round((total_pv / max(INITIAL_CAPITAL * len(self._players), 1) - 1) * 100, 3),
                'locked_usdt':  round(total_locked, 4),
                'total_usdt':   round(total_pv + total_locked, 4),
            },
            'players': players_status,
        }


def _make_fake_pf(player: _SubPlayerBase, prices: dict):
    """Создаёт PaperPortfolio-подобный объект для plot_usd_dashboard_mexc."""
    class _FakePF:
        def __init__(self, p, px):
            ps = p.get_status(px)
            self.history           = p._pv_history
            self.initial_capital   = p._capital
            self.cash              = ps.get('cash_usdt', p._pv)
            self._locked_profit    = p._locked_usdt
            self.holdings          = {}
            self.futures           = {}
            for pos in ps.get('positions', []):
                sym = pos['sym']
                ep  = pos.get('entry_price', 0)
                cur = pos.get('cur_price', ep)
                if pos['type'] == 'spot' and ep > 0 and cur > 0:
                    trade_usdt = min(p._capital * TRADE_FRACTION, 120.0)
                    self.holdings[sym] = trade_usdt / ep
                elif pos['type'] == 'fut_long' and ep > 0:
                    trade_usdt = min(p._capital * TRADE_FRACTION, 120.0)
                    self.futures[sym] = {'side': 'long', 'entry_price': ep,
                                          'margin': trade_usdt, 'leverage': LEVERAGE}
                elif pos['type'] == 'fut_short' and ep > 0:
                    trade_usdt = min(p._capital * TRADE_FRACTION, 120.0)
                    self.futures[sym] = {'side': 'short', 'entry_price': ep,
                                          'margin': trade_usdt, 'leverage': LEVERAGE}
        def portfolio_value(self, prices): return sum(self.history[-1:] or [self.initial_capital])
        def get_usd_breakdown(self, prices):
            st = {'cash_usdt': self.cash, 'crypto_value': 0.0,
                  'locked_profit': self._locked_profit,
                  'total_active': self.cash,
                  'total_with_locked': self.cash + self._locked_profit,
                  'spot_positions': {}, 'fut_positions': {}}
            for sym, qty in self.holdings.items():
                p = prices.get(sym, 0)
                st['crypto_value'] += qty * p
                st['spot_positions'][sym] = {'qty': qty, 'value': qty * p}
                st['total_active'] += qty * p
                st['total_with_locked'] += qty * p
            # FIX: учитываем фьючерсные позиции (long/short) в крипто-активах
            for sym, fi in self.futures.items():
                p = prices.get(sym, 0)
                ep = fi['entry_price']
                margin = fi['margin']
                lev = fi.get('leverage', LEVERAGE)
                if fi['side'] == 'long':
                    pnl = margin * lev * (p / ep - 1) if ep > 0 else 0
                else:  # short
                    pnl = margin * lev * (1 - p / ep) if ep > 0 else 0
                fut_val = margin + pnl
                st['crypto_value'] += fut_val
                st['fut_positions'][sym] = {'side': fi['side'], 'margin': margin, 'pnl': pnl, 'value': fut_val}
                st['total_active'] += fut_val
                st['total_with_locked'] += fut_val
            return st
    return _FakePF(player, prices)

def _apply_player_settings(mp: MasterPlayer, cfg: dict):
    """Применяет DayStop / PeakStop / ProfitLock из settings.txt к суб-игрокам."""
    day_stop_en  = cfg.get('day_stop_enabled',     False)
    day_stop_pct = float(cfg.get('day_stop_pct',   10.0))
    peak_stop_en  = cfg.get('peak_stop_enabled',   False)
    peak_stop_pct = float(cfg.get('peak_stop_pct', 20.0))
    lock_en   = cfg.get('profit_lock_enabled',  False)
    lock_trig = float(cfg.get('profit_lock_trigger', 5.0))
    lock_step = float(cfg.get('profit_lock_step',    5.0))
    lock_frac = float(cfg.get('profit_lock_fraction', 0.30))
    lock_max  = float(cfg.get('profit_lock_max',  0.70))

    for player in mp._players.values():
        # DayStop
        player.DAY_STOP_ENABLED = day_stop_en
        player.DAY_STOP_PCT     = day_stop_pct / 100.0
        # PeakStop
        player.PEAK_STOP_PCT    = peak_stop_pct / 100.0
        if not peak_stop_en:
            player.PEAK_STOP_PCT = 99.0  # практически отключён
        # ProfitLock
        if lock_en:
            player.LOCK_TRIGGER  = lock_trig / 100.0
            player.LOCK_STEP     = lock_step / 100.0
            player.LOCK_FRACTION = lock_frac
            player.LOCK_MAX      = lock_max
            # Обновляем стартовый триггер
            player._lock_next_at = player._capital * (1.0 + player.LOCK_TRIGGER)
        else:
            # Отключаем: ставим триггер недостижимым
            player._lock_next_at = float('inf')

    log.info("  [settings] DayStop=%s(%.0f%%)  PeakStop=%s(%.0f%%)  ProfitLock=%s(+%.0f%% → %.0f%%)",
             "ON" if day_stop_en  else "off", day_stop_pct,
             "ON" if peak_stop_en else "off", peak_stop_pct,
             "ON" if lock_en      else "off", lock_trig, lock_frac*100)


class _ColorFormatter(logging.Formatter):
    """Цветной форматтер логов с emoji-индикаторами."""
    GREEN  = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    CYAN   = '\033[96m'; RESET  = '\033[0m';  BOLD = '\033[1m'
    COLORS = {logging.DEBUG: '', logging.INFO: '', logging.WARNING: YELLOW, logging.ERROR: RED}
    def format(self, record):
        clr  = self.COLORS.get(record.levelno, '')
        msg  = super().format(record)
        return f"{clr}{msg}{self.RESET}" if clr else msg


def _setup_logging(output_dir: str = None):
    """Настраивает логирование: консоль (цветной) + файл (plain)."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    # Консоль
    ch = logging.StreamHandler()
    ch.setFormatter(_ColorFormatter(
        "%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)
    # Файл
    if output_dir:
        import os as _os
        _os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(
            _os.path.join(output_dir, "master_player.log"), encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)


def _log_player_status(players: dict, prices: dict, bar: int, warmup_end: int):
    """Выводит подробную строку состояния каждого суб-игрока."""
    live = max(0, bar - warmup_end)
    sep  = "─" * 78
    log.info(sep)
    log.info("  Bar %-5d  Live %-5d  %s", bar, live,
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info(sep)
    total_pv     = 0.0
    total_locked = 0.0
    for name, player in players.items():
        ps       = player.get_status(prices)
        pv       = ps['pv']
        locked   = ps['locked_usdt']
        cap      = player._capital
        ret_pct  = (pv + locked) / max(cap, 1) * 100 - 100
        dd_pct   = ps['dd_pct']
        day_pct  = ps['day_dd_pct']
        regime   = ps['regime'].upper()[:4]
        n_pos    = ps['n_positions']
        n_trades = ps['total_trades']
        cash_u   = ps.get('cash_usdt', pv)
        cryp_u   = ps.get('crypto_value', 0.0)
        day_s    = '⛔DAY' if ps['day_stopped']  else f'day{day_pct:+.1f}%'
        peak_s   = '⛔PEAK' if ps['peak_stopped'] else f'dd{dd_pct:.1f}%'
        lock_s   = f'+lock${locked:.1f}' if locked > 0.1 else ''
        log.info("  %-5s  $%-7.2f (%+.2f%%)  Cash$%-7.1f Крипто$%-7.1f  "                 "[%s] %s %s  pos=%d tr=%d%s",
                 name, pv, ret_pct, cash_u, cryp_u,
                 regime, day_s, peak_s, n_pos, n_trades, lock_s)
        # Открытые позиции
        for pos in ps['positions']:
            sym   = pos['sym']
            ptype = pos['type'][:5]
            pnl   = pos.get('pnl_pct', 0)
            held  = pos.get('held_bars', 0)
            agent = pos.get('agent','?')[:10]
            clr   = '↑' if pnl >= 0 else '↓'
            log.info("    %s %-8s %-5s %-10s  pnl=%+.2f%%  %dб  via:%s",
                     clr, sym, ptype, '', pnl, held, agent)
        total_pv     += pv
        total_locked += locked
    log.info(sep)
    total_ret = (total_pv + total_locked) / max(INITIAL_CAPITAL * len(players), 1) * 100 - 100
    log.info("  ИТОГО  Active=$%.2f  Locked=$%.2f  Total=$%.2f  (%+.2f%%)",
             total_pv, total_locked, total_pv + total_locked, total_ret)
    log.info(sep)


def _run():
    import os as _os
    from datetime import timezone as _tz
    _ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _out = _os.path.join(str(PROJECT_ROOT), "Results", "MasterPlayer", _ts)
    _setup_logging(_out)
    script_dir = str(RUNTIME_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import mexc_connector as _mc
    except ImportError:
        log.error("mexc_connector.py не найден рядом с master_player.py"); sys.exit(1)

    raw_cfg    = _mc._load_settings()
    parsed_cfg = _mc._parse_settings(raw_cfg)
    parsed_cfg.update({
        'initial_capital': INITIAL_CAPITAL,
        'trade_fraction':  TRADE_FRACTION,
        'leverage':        int(LEVERAGE),
        'spot_fee':        SPOT_FEE,
        'futures_fee':     FUTURES_FEE,
    })

    log.info("═" * 65)
    log.info("  MasterPlayer v6  5 × $%.0f = $%.0f  mode=%s", INITIAL_CAPITAL, INITIAL_CAPITAL*5, _mc.TRADING_MODE)
    log.info("  trade_fraction=%.0f%%  5 суб-игроков: Alpha/Beta/Gamma/P7Ways/Ultima", TRADE_FRACTION * 100)
    log.info("  Настройки стопов/лока берутся из settings.txt (day_stop / peak_stop / profit_lock)")
    log.info("═" * 65)

    configure(
        BAR=parsed_cfg['bar'], INITIAL_CAPITAL=parsed_cfg['initial_capital'],
        TRADE_FRACTION=parsed_cfg['trade_fraction'],
        LEVERAGE=float(parsed_cfg['leverage']),
        SPOT_FEE=parsed_cfg['spot_fee'], FUTURES_FEE=parsed_cfg['futures_fee'],
    )

    for mod_name in ('crypto_agents', 'panteon_agents'):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, 'configure'):
                mod.configure(BAR=parsed_cfg['bar'],
                              TRADE_FRACTION=parsed_cfg['trade_fraction'],
                              INITIAL_CAPITAL=parsed_cfg['initial_capital'],
                              LEVERAGE=float(parsed_cfg['leverage']),
                              cfg=raw_cfg)
        except ImportError:
            pass

    if not _mc.check_connectivity():
        log.error("Нет связи с API биржи."); sys.exit(1)

    # ── Используем 3 НЕЗАВИСИМЫХ игрока ──────────────────────────────────────
    players = make_independent_players()   # {'Alpha': ..., 'Beta': ..., 'Gamma': ...}

    # Применяем настройки из settings.txt к каждому игроку
    for player in players.values():
        _apply_player_settings_single(player, parsed_cfg)

    bridge = _mc.AgentMexcBridge(
        agents=players, cfg=parsed_cfg,
        mode=_mc.TRADING_MODE, api_key=_mc.API_KEY, api_secret=_mc.API_SECRET,
        output_dir=_out,
    )
    # MasterPlayer: compound_growth дублирует дашборд — пропускаем
    bridge.skip_compound_growth = True

    # Патч: детальные логи + дашборды каждые DASHBOARD_EVERY баров
    _orig_cycle = bridge._run_cycle

    def _patched_cycle():
        _orig_cycle()
        live_bar = bridge._bar - bridge._warmup_end
        if live_bar > 0 and live_bar % bridge.DASHBOARD_EVERY == 0:
            try:
                prices, _ = bridge._fetch_market()
                # Логи по каждому игроку
                _log_player_status(players, prices, bridge._bar, bridge._warmup_end)
                # USD дашборд
                from mexc_connector import plot_usd_dashboard_mexc
                fake_pf = {n: _make_fake_pf(p, prices) for n, p in players.items()}
                plot_usd_dashboard_mexc(fake_pf, prices, bridge.output_dir,
                                         bridge._bar, bridge._warmup_end)
                # Детальный PNG по суб-игрокам
                mp_wrapper = _PlayerWrapper(players)
                _save_master_detail(mp_wrapper, prices, bridge.output_dir)
                _save_live_growth(mp_wrapper, bridge.output_dir)
                # Дашборд по агентам каждого игрока
                _save_player_agent_dashboards(players, bridge.output_dir)
            except Exception as e:
                log.warning("dashboard error: %s\n%s", e, traceback.format_exc())

    bridge._run_cycle = _patched_cycle

    log.info("══ Прогрев %d баров ≈ %dч ══", _mc.WARMUP_BARS, _mc.WARMUP_BARS // 60)
    bridge.warmup(n_bars=_mc.WARMUP_BARS)

    try:
        prices, _ = bridge._fetch_market()
        for player in players.values():
            player.reset_after_warmup(prices)

        # FIX: Сбрасываем PaperPortfolio бриджа — warmup-позиции не должны
        # переходить в live. Без этого portfolio_history.csv показывает
        # PV от warmup-позиций, а player._all_trades = пустой.
        for name, pf in bridge.paper_pf.items():
            pf.cash     = pf.initial_capital
            pf.holdings.clear()
            pf.futures.clear()
            pf.history  = [pf.initial_capital]
            pf.trades.clear()
            pf.op_count = {"spot_buy":0,"spot_sell":0,"fut_long":0,"fut_short":0,"fut_close":0}
            pf.op_fees  = {"spot_buy":0.0,"spot_sell":0.0,"fut_long":0.0,"fut_short":0.0}
            pf._day_start_pv    = pf.initial_capital
            pf._day_start_bar   = 0
            pf._day_stopped     = False
            pf._abs_peak_pv     = pf.initial_capital
            pf._peak_stopped    = False
            pf._locked_profit   = 0.0
            pf._lock_next_at    = pf.initial_capital * 1.05  # default trigger
            log.info("  [reset] PaperPortfolio(%s) → $%.0f", name, pf.initial_capital)

        log.info("══ Прогрев завершён. Старт live-торговли ══")
        _log_player_status(players, prices, bridge._warmup_end, bridge._warmup_end)
    except Exception as e:
        log.warning("post-warmup init: %s\n%s", e, traceback.format_exc())

    log.info("Live-торговля (Ctrl+C для остановки)...")
    bridge.run()


if __name__ == '__main__':
    _run()

