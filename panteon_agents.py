"""Live-агенты и пресеты игроков для Panteon."""

from __future__ import annotations
import json
import os
import sys
import time
import logging
import numpy as np
from collections import deque, OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

from agent_meta import (
    ContextMemoryAgent,
    MemorySnapshotStore,
    PortfolioAllocatorAgent,
    ShadowScoringAgent,
)
from agent_safety import (
    DeclineGuard as SharedDeclineGuard,
    MarketRegimeActionGuard,
    PositionExitGovernor,
    SymbolUniverseGuard,
)


def _bootstrap_project_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = base_dir
    if not os.path.isdir(os.path.join(project_root, "Retrodate_cryptotrade")):
        project_root = os.path.dirname(base_dir)
    for extra_dir in (
        os.path.join(project_root, "Retrodate_cryptotrade"),
        os.path.join(project_root, "Genetics_DL_Agents"),
    ):
        if os.path.isdir(extra_dir) and extra_dir not in sys.path:
            sys.path.insert(0, extra_dir)


_bootstrap_project_paths()

log = logging.getLogger("panteon_agents")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _sanitize_memory_namespace(namespace: Optional[str]) -> str:
    raw = str(namespace or "").strip().upper()
    if not raw:
        return ""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw)
    return cleaned.strip("_")


def _memory_paths_for_namespace(namespace: Optional[str]):
    token = _sanitize_memory_namespace(namespace)
    if not token:
        return (
            os.path.join(SCRIPT_DIR, "Real_Player_Memory.txt"),
            os.path.join(SCRIPT_DIR, "Real_Player_Memory"),
        )
    return (
        os.path.join(SCRIPT_DIR, f"Real_Player_Memory_{token}.txt"),
        os.path.join(SCRIPT_DIR, f"Real_Player_Memory_{token}"),
    )


_DEFAULT_MEMORY_NAMESPACE = os.getenv("CRYPTO_EXCHANGE") or os.getenv("CRYPTO_EXCHANGE_ID")
REAL_PLAYER_MEMORY_FILE, LEGACY_REAL_PLAYER_MEMORY_FILE = _memory_paths_for_namespace(
    _DEFAULT_MEMORY_NAMESPACE
)

# ─────────────────────────────────────────────────────────────────────────────
# ЧЁРНЫЙ СПИСОК МОНЕТ
# ─────────────────────────────────────────────────────────────────────────────
BLACKLIST = frozenset({
    'TONIXAI', 'DEGO', 'CRTR', 'ATLA',
    'GOLD(PAXG)', 'GOLD(XAUT)', 'BLINKY', 'WLD',
})

# Запасные значения; configure() подменяет их из settings.txt.
BAR            = 60
TRADE_FRACTION = 0.10
INITIAL_CAPITAL= 10_000.0
LEVERAGE       = 3
_FUNDING_FETCHER = None
GENETICS_SAFE_SYMBOLS = {
    "BTC","ETH","BNB","SOL","XRP","ADA","AVAX","DOT","LINK","MATIC",
    "TON","DOGE","SUI","APT","ARB","OP","INJ","TIA","SEI","NEAR",
    "ATOM","UNI","LTC","ETC","BCH","FIL","AAVE","SNX","COMP","MKR",
}

def set_fetcher(f): global _FUNDING_FETCHER; _FUNDING_FETCHER = f

def configure(BAR=60, TRADE_FRACTION=0.10, INITIAL_CAPITAL=10_000.0,
              LEVERAGE=3.0, cfg=None, **kw):
    import sys
    m = sys.modules.get(__name__)
    if m:
        m.BAR=int(BAR); m.TRADE_FRACTION=float(TRADE_FRACTION)
        m.INITIAL_CAPITAL=float(INITIAL_CAPITAL); m.LEVERAGE=int(LEVERAGE)

def _get_funding(sym):
    if _FUNDING_FETCHER is None: return {}
    try: return _FUNDING_FETCHER.get(sym) or {}
    except: return {}

def _get_global():
    if _FUNDING_FETCHER is None: return {}
    try: return _FUNDING_FETCHER.get_global() or {}
    except: return {}

def _ema(data, period):
    if not data: return 0.0
    k = 2/(period+1); e = data[0]
    for v in data[1:]: e = v*k + e*(1-k)
    return float(e)

def _mom(arr, lb):
    lst = list(arr)
    if len(lst) < lb+1: return 0.0
    b = lst[-lb-1]
    return (lst[-1]/b - 1) if b > 0 else 0.0

def _regime(h_dict, lb=60):
    """bull/bear/sideways по медиане momentum."""
    moms = []
    for h in h_dict.values():
        lst = list(h)
        if len(lst) < lb+1: continue
        b = lst[-lb-1]
        if b > 0: moms.append(lst[-1]/b - 1)
    if not moms: return 'sideways'
    avg = float(np.median(moms))
    if avg > 0.005: return 'bull'
    if avg < -0.005: return 'bear'
    return 'sideways'

def _vol_zscore(v_deque, window=60):
    """Объём относительно среднего (z-score)."""
    lst = list(v_deque)
    if len(lst) < window+1: return 0.0
    hist = lst[-window-1:-1]; cur = lst[-1]
    mu = float(np.mean(hist)); sd = float(np.std(hist))
    return (cur - mu) / (sd + 1e-9)

# ══════════════════════════════════════════════════════════════════
# ПАТЧ-ОБЁРТКИ
# ══════════════════════════════════════════════════════════════════

class GeneticsSymbolGuard(SymbolUniverseGuard):
    def __init__(self, agent):
        super().__init__(agent, allowed_symbols=GENETICS_SAFE_SYMBOLS)

class MarketRegimeGuard(MarketRegimeActionGuard):
    """PMM только в sideways (threshold расслаблен)."""
    SMA=30; THRESH=0.006

    def __init__(self, a):
        super().__init__(a, sma=self.SMA, threshold=self.THRESH)
        self._a = a
        self._h = self._history

class DeclineGuard(SharedDeclineGuard):
    """Блокирует DCA-покупки при падении >2% за 4ч."""
    PERIOD=4*60; THRESH=-0.02

    def __init__(self, a):
        super().__init__(a, period=self.PERIOD, threshold=self.THRESH)
        self._a = a
        self._h = self._history

def wrap_existing_agents(agents: OrderedDict) -> OrderedDict:
    patched = OrderedDict()
    # FIX: правильные имена классов + _GeneticsAdapter для обёрток
    genetics = {"GeneticsAgent", "GeneticsBearishAgent", "GeneticsNeutralAgent",
                "GeneticsBullishAgent", "_GeneticsAdapter"}
    for name, agent in agents.items():
        inner = getattr(agent,'_agent',agent); cn=type(inner).__name__
        if cn in genetics or name in genetics:
            w=GeneticsSymbolGuard(inner)
            if hasattr(agent,'_agent'): object.__setattr__(agent,'_agent',w); patched[name]=agent
            else: patched[name]=w
            print(f"  [patch] {name} → GeneticsSymbolGuard")
        elif cn=="HummingbotPMM" or name=="HummingbotPMM":
            w=MarketRegimeGuard(inner)
            if hasattr(agent,'_agent'): object.__setattr__(agent,'_agent',w); patched[name]=agent
            else: patched[name]=w
            print(f"  [patch] {name} → MarketRegimeGuard")
        elif cn=="ThreeCommasDCA" or name=="ThreeCommasDCA":
            w=DeclineGuard(inner)
            if hasattr(agent,'_agent'): object.__setattr__(agent,'_agent',w); patched[name]=agent
            else: patched[name]=w
            print(f"  [patch] {name} → DeclineGuard")
        else:
            patched[name]=agent
    return patched

# ══════════════════════════════════════════════════════════════════
# BOMBERMAN v3  (исправлен: меньше сделок, подтверждение объёмом, стоп)
# ══════════════════════════════════════════════════════════════════

class Bomberman:
    """
    Bomberman v3: BOP + MRC + Donchian.

    ИСПРАВЛЕНИЯ vs v2:
    - CHECK_INT: 5→60 мин (снижение числа сделок с 1100 до ~100)
    - Volume confirmation: объём > avg*1.2
    - Hard stop: 1.5% от входа (раньше не было)
    - Profit target: 3% (RR = 2:1)
    - BOP_THRESH: 0.3→0.5 (только сильные сигналы)
    - Режимный фильтр: long только в bull/sideways, short только в bear/sideways
    """
    BOP_PERIOD=30; MRC_PERIOD=50; DONCHIAN_PERIOD=10
    MRC_BANDS=2.0; BOP_THRESH=0.5; CHECK_INT=120  # FIX: снижен с 60→120 для уменьшения overtrading  # раз в час!
    STOP_PCT=0.015; TARGET_PCT=0.03; VOL_CONFIRM=1.2
    MAX_POS=3

    def __init__(self):
        self.h:Dict[str,deque]={}; self.v:Dict[str,deque]={}
        self.pos:Dict[str,str]={}; self.entry_px:Dict[str,float]={}
        self.t=0; self._lc=-999

    def _bop(self, h):
        if len(h)<self.BOP_PERIOD: return 0.0
        w=h[-self.BOP_PERIOD:]; sma=float(np.mean(w))
        atr=float(np.mean(np.abs(np.diff(w))))+1e-12
        return (h[-1]-sma)/atr

    def _mrc(self, h):
        if len(h)<self.MRC_PERIOD: return h[-1]*0.97,h[-1],h[-1]*1.03
        w=h[-self.MRC_PERIOD:]; mid=float(np.mean(w)); sd=float(np.std(w))
        return mid-self.MRC_BANDS*sd, mid, mid+self.MRC_BANDS*sd

    def _donchian(self, h):
        if len(h)<self.DONCHIAN_PERIOD+1: return h[-1]*0.98,h[-1]*1.02
        prev=h[-(self.DONCHIAN_PERIOD+1):-1]
        return float(min(prev)), float(max(prev))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.MRC_PERIOD*4)).append(float(p))
            self.v.setdefault(s,deque(maxlen=self.BOP_PERIOD*3)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.entry_px.setdefault(s,0.)
        self.t = bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        regime=_regime(self.h)
        n_long=sum(1 for v in self.pos.values() if v=='long_fut')
        n_short=sum(1 for v in self.pos.values() if v=='short_fut')
        for sym in prices:
            h=list(self.h[sym]); v_list=list(self.v[sym]); cur=self.pos[sym]; px=prices[sym]
            if len(h)<self.MRC_PERIOD+self.DONCHIAN_PERIOD: continue
            bop=self._bop(h); lo,mid,hi=self._mrc(h); dlo,dhi=self._donchian(h)
            ep=self.entry_px[sym]
            # Стоп-лосс и тейк-профит (обязательные выходы)
            if cur=='long_fut':
                if px<=ep*(1-self.STOP_PCT) or px>=ep*(1+self.TARGET_PCT) or px>=hi:
                    actions[sym]=8; self.pos[sym]=None; n_long-=1; continue
            elif cur=='short_fut':
                if px>=ep*(1+self.STOP_PCT) or px<=ep*(1-self.TARGET_PCT) or px<=lo:
                    actions[sym]=8; self.pos[sym]=None; n_short-=1; continue
            if cur is not None: continue
        # Подтверждение объёмом
            v_avg=float(np.mean(v_list[-self.BOP_PERIOD:])) if len(v_list)>=self.BOP_PERIOD else 0
            v_cur=v_list[-1] if v_list else 0
            vol_ok=v_cur > v_avg*self.VOL_CONFIRM if v_avg>0 else True
        # Вход (только при подтверждении объёмом и подходящем режиме)
            if px>dhi and bop>self.BOP_THRESH and px<hi and vol_ok and regime!='bear' and n_long<self.MAX_POS:
                actions[sym]=5; self.pos[sym]='long_fut'; self.entry_px[sym]=px; n_long+=1
            elif px<dlo and bop<-self.BOP_THRESH and px>lo and vol_ok and regime!='bull' and n_short<self.MAX_POS:
                actions[sym]=7; self.pos[sym]='short_fut'; self.entry_px[sym]=px; n_short+=1
        return actions


# ══════════════════════════════════════════════════════════════════
# MOMENTUM SCALPER  (новый - учится у DualMomentum + VolumeBreakout)
# ══════════════════════════════════════════════════════════════════

class MomentumScalper:
    """
    Тренд-следование на средних таймфреймах.

    Учится у лучших: DualMomentum (53% WR) + VolumeBreakout (Sharpe 1.32).

    Вход: EMA(2h) > EMA(8h) > EMA(24h)  + volume spike (1.3×avg)
    Выход: EMA(2h) < EMA(8h) OR stop-loss 2% OR profit target 4%
    RR = 2:1 минимум.

    Торгует 1 раз в 2 часа → ~50 сделок за 97ч.
    """
    EMA_F=10; EMA_M=30; EMA_S=2*60
    VOL_WIN=30; VOL_MULT=1.05; STOP=0.012; TARGET=0.018
    CHECK_INT=5; MAX_POS=3; MOM_MIN=0.003

    def __init__(self):
        self.h:Dict[str,deque]={}; self.v:Dict[str,deque]={}
        self.pos:Dict[str,str]={}; self.entry_px:Dict[str,float]={}
        self.t=0; self._lc=-9999

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.EMA_S*2)).append(float(p))
            self.v.setdefault(s,deque(maxlen=self.VOL_WIN*3)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.entry_px.setdefault(s,0.)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        n_long=sum(1 for v in self.pos.values() if v=='long')
        n_short=sum(1 for v in self.pos.values() if v=='short')
        for sym in prices:
            h=list(self.h[sym]); vl=list(self.v[sym]); cur=self.pos[sym]; px=prices[sym]
            if len(h)<self.EMA_S+1: continue
            ef=_ema(h[-self.EMA_F*3:],  self.EMA_F)
            em=_ema(h[-self.EMA_M*2:],  self.EMA_M)
            es=_ema(h[-self.EMA_S:],     self.EMA_S)
            ep=self.entry_px[sym]
            # Стоп + тейк
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or ef<em*0.999:
                    actions[sym]=3; self.pos[sym]=None; n_long-=1; continue
            elif cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or ef>em*1.001:
                    actions[sym]=8; self.pos[sym]=None; n_short-=1; continue
            if cur is not None: continue
            mom=_mom(self.h[sym], self.EMA_F)
            v_avg=float(np.mean(vl[-self.VOL_WIN-1:-1])) if len(vl)>self.VOL_WIN else 0
            v_cur=vl[-1] if vl else 0
            vol_spike=v_cur>v_avg*self.VOL_MULT if v_avg>0 else False
            # v5 FIX: vol_spike теперь используется как условие входа (было мёртвый код)
            if ef>em*1.003 and em>es*1.001 and mom>self.MOM_MIN and vol_spike and n_long<self.MAX_POS:
                actions[sym]=2; self.pos[sym]='long'; self.entry_px[sym]=px; n_long+=1
            elif ef<em*0.997 and em<es*0.999 and mom<-self.MOM_MIN and vol_spike and n_short<self.MAX_POS:
                actions[sym]=7; self.pos[sym]='short'; self.entry_px[sym]=px; n_short+=1
        return actions


# ══════════════════════════════════════════════════════════════════
# LIVE-АГЕНТЫ  (исправлены)
# ══════════════════════════════════════════════════════════════════

class LiveAfterShock:
    """
    После сильного падения — ловим отскок.
    Сигнал: цена упала > DROP за LOOKBACK + RSI перепродан.
    Stop/Target: 0.5% / 1.0% (RR=2:1).
    """
    LOOKBACK=45; DROP=0.0035; RSI_N=7; RSI_OS=45
    STOP=0.005; TARGET=0.010; HOLD=3*60; CHECK_INT=10

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.t=0; self._lc=-9999

    def _rsi(self, h, n):
        if len(h)<n*3: return 50.
        d=np.diff(list(h)[-(n*3):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
        ag,al=np.mean(g[:n]),np.mean(l[:n])
        for gi,li in zip(g[n:],l[n:]): ag=(ag*(n-1)+gi)/n; al=(al*(n-1)+li)/n
        return 100-100/(1+ag/(al+1e-9))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.LOOKBACK*4)); self.pos.setdefault(s,None)
            self.ep.setdefault(s,0.); self.et.setdefault(s,0)
            self.h[s].append(float(p))
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if cur=='long':
                held=self.t-self.et[sym]
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; continue
                continue
            if len(h)<self.LOOKBACK+self.RSI_N*3: continue
            lst=list(h); base=lst[-self.LOOKBACK-1]
            drop=(px-base)/base if base>0 else 0
            if drop<-self.DROP and self._rsi(h,self.RSI_N)<self.RSI_OS:
                actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
        return actions


class LiveCrashHunter:
    """
    Торгует во всех режимах (исправление оригинала только-bear).
    Bear: шортит топ-падающих
    Sideways/Bull: mean-reversion по RSI
    Stop: 1.5% обязательный.
    """
    CHECK_INT=30; RSI_N=14; RSI_OB=60; RSI_OS=40; MOM_LB=120
    STOP=0.012; TARGET=0.020; HOLD=4*60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.t=0; self._lc=-9999

    def _rsi(self, h, n=14):
        if len(h)<n+2: return 50.
        d=np.diff(list(h)[-(n*3):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
        ag,al=np.mean(g[:n]),np.mean(l[:n])
        for gi,li in zip(g[n:],l[n:]): ag=(ag*(n-1)+gi)/n; al=(al*(n-1)+li)/n
        return 100-100/(1+ag/(al+1e-9))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=10*BAR)); self.pos.setdefault(s,None)
            self.ep.setdefault(s,0.); self.et.setdefault(s,0)
            self.h[s].append(float(p))
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        regime=_regime(self.h)
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            held=self.t-self.et[sym]
            # Стоп + тейк + время
            if cur=='short_fut':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; continue
            elif cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; continue
            if cur is not None: continue
            if len(h)<max(self.MOM_LB,self.RSI_N*3): continue
            rsi=self._rsi(h,self.RSI_N); mom=_mom(h,self.MOM_LB)
            if regime in ('bear','crash') and mom < -0.005 and rsi > 45:
                actions[sym]=6; self.pos[sym]='short_fut'; self.ep[sym]=px; self.et[sym]=self.t
            elif rsi < self.RSI_OS and mom > -0.012 and regime != 'bear':
                actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
            elif rsi>self.RSI_OB and regime in ('bear','sideways'):
                actions[sym]=6; self.pos[sym]='short_fut'; self.ep[sym]=px; self.et[sym]=self.t
        return actions


class LiveTrendFollow:
    """
    EMA-тренд с расслабленными условиями.
    Stop 2%, Target 4%, CHECK раз в 2ч.
    """
    EMA_F=30; EMA_M=90; EMA_S=4*60
    STOP=0.015; TARGET=0.025; CHECK_INT=15; MAX_POS=3; MIN_MOM=0.0015

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}
        self.t=0; self._lc=-9999

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.EMA_S*2)).append(float(p))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        nl=sum(1 for v in self.pos.values() if v=='long')
        ns=sum(1 for v in self.pos.values() if v=='short')
        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if len(h)<self.EMA_S+1: continue
            ef=_ema(h[-self.EMA_F*3:],self.EMA_F)
            em=_ema(h[-self.EMA_M*2:],self.EMA_M)
            es=_ema(h[-self.EMA_S:],  self.EMA_S)
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or ef<em*0.999:
                    actions[sym]=3; self.pos[sym]=None; nl-=1; continue
            elif cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or ef>em*1.001:
                    actions[sym]=8; self.pos[sym]=None; ns-=1; continue
            if cur is not None: continue
            mom=_mom(self.h[sym],self.EMA_F)
            if ef>em*1.002 and em>es*1.001 and mom>self.MIN_MOM and nl<self.MAX_POS:
                actions[sym]=2; self.pos[sym]='long'; self.ep[sym]=px; nl+=1
            elif ef<em*0.998 and em<es*0.999 and mom<-self.MIN_MOM and ns<self.MAX_POS:
                actions[sym]=7; self.pos[sym]='short'; self.ep[sym]=px; ns+=1
        return actions


class RichardDennisTurtle:
    """
    Richard Dennis / Turtle Trading, адаптированный под минутные бары.

    Основа стратегии:
    - вход по пробою Donchian-каналов 20h/55h;
    - long/short только по направлению устойчивого импульса;
    - выход по каналу 10h/20h или по стопу около 2 ATR.
    """
    FAST_ENTRY = 120
    SLOW_ENTRY = 240
    FAST_EXIT = 60
    SLOW_EXIT = 120
    ATR_LB = 120
    CHECK_INT = 5
    MAX_POS = 3
    MIN_BREAKOUT = 0.0008
    ATR_STOP_MULT = 2.0
    STOP_MIN = 0.012
    STOP_MAX = 0.055

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.pos: Dict[str, str] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.stop: Dict[str, float] = {}
        self.system: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _atr_frac(self, h):
        lst = list(h)
        if len(lst) < self.ATR_LB + 1:
            return self.STOP_MIN
        w = np.array(lst[-self.ATR_LB - 1:], dtype=float)
        rets = np.abs(np.diff(w) / np.maximum(w[:-1], 1e-12))
        atr = float(np.mean(rets))
        return min(max(atr * self.ATR_STOP_MULT, self.STOP_MIN), self.STOP_MAX)

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = self.SLOW_ENTRY + self.ATR_LB + 120
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self.stop.setdefault(s, self.STOP_MIN)
            self.system.setdefault(s, 1)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        n_long = sum(1 for v in self.pos.values() if v == 'long')
        n_short = sum(1 for v in self.pos.values() if v == 'short')

        for sym in prices:
            h = list(self.h[sym])
            cur = self.pos[sym]
            px = float(prices[sym])
            ep = self.ep[sym]
            if len(h) < self.SLOW_ENTRY + 1:
                continue

            stop_pct = self.stop.get(sym, self.STOP_MIN)
            exit_lb = self.FAST_EXIT if self.system.get(sym, 1) == 1 else self.SLOW_EXIT
            long_exit = float(min(h[-exit_lb - 1:-1])) if len(h) >= exit_lb + 1 else px * 0.98
            short_exit = float(max(h[-exit_lb - 1:-1])) if len(h) >= exit_lb + 1 else px * 1.02

            if cur == 'long':
                if px <= max(long_exit, ep * (1 - stop_pct)):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_long = max(0, n_long - 1)
                continue
            if cur == 'short':
                if px >= min(short_exit, ep * (1 + stop_pct)):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_short = max(0, n_short - 1)
                continue

            fast_high = float(max(h[-self.FAST_ENTRY - 1:-1]))
            fast_low = float(min(h[-self.FAST_ENTRY - 1:-1]))
            slow_high = float(max(h[-self.SLOW_ENTRY - 1:-1]))
            slow_low = float(min(h[-self.SLOW_ENTRY - 1:-1]))
            trend_mom = _mom(h, 60)
            regime = _regime(self.h, lb=120)
            stop_pct = self._atr_frac(h)

            long_fast = px >= fast_high * (1 + self.MIN_BREAKOUT)
            long_slow = px >= slow_high
            short_fast = px <= fast_low * (1 - self.MIN_BREAKOUT)
            short_slow = px <= slow_low

            if n_long < self.MAX_POS and regime != 'bear' and (long_fast or long_slow) and trend_mom > 0.0025:
                actions[sym] = 5
                self.pos[sym] = 'long'
                self.ep[sym] = px
                self.et[sym] = self.t
                self.stop[sym] = stop_pct
                self.system[sym] = 1 if long_fast else 2
                n_long += 1
            elif n_short < self.MAX_POS and regime != 'bull' and (short_fast or short_slow) and trend_mom < -0.0025:
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
                self.stop[sym] = stop_pct
                self.system[sym] = 1 if short_fast else 2
                n_short += 1
        return actions


class LiveRegimePullback:
    """
    Trend continuation after a controlled pullback.

    Idea:
    - trade only when fast/mid/slow EMAs are aligned;
    - wait for a real pullback instead of chasing fresh highs/lows;
    - require momentum re-acceleration plus non-weak volume;
    - use volatility-aware stop/target and time stop.
    """
    EMA_FAST=45; EMA_MID=180; EMA_SLOW=12*60
    CHECK_INT=10; PULLBACK_LB=90; TURN_LB=20; TREND_LB=2*60
    STOP_MIN=0.010; STOP_MAX=0.022; TARGET_R=2.2; HOLD=8*60
    MAX_POS=3; PULLBACK_DEPTH=0.0035; BOUNCE_MIN=0.0012
    TURN_MOM_MIN=0.0006; TREND_MOM_MIN=0.0025; VOL_Z_MIN=-0.45

    def __init__(self):
        self.h:Dict[str,deque]={}; self.v:Dict[str,deque]={}
        self.pos:Dict[str,str]={}; self.ep:Dict[str,float]={}
        self.et:Dict[str,int]={}; self.sl:Dict[str,float]={}; self.tp:Dict[str,float]={}
        self.t=0; self._lc=-9999

    def _vol_pct(self, h, lb=60):
        lst=list(h)
        if len(lst) < lb+1:
            return 0.0
        w=np.array(lst[-lb-1:], dtype=float)
        rets=np.diff(w) / np.maximum(w[:-1], 1e-12)
        return float(np.mean(np.abs(rets)))

    def _risk_params(self, h):
        vol=max(self._vol_pct(h, lb=60), self.STOP_MIN/2)
        stop=min(max(vol*2.4, self.STOP_MIN), self.STOP_MAX)
        target=min(max(stop*self.TARGET_R, 0.018), 0.05)
        return stop, target

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.EMA_SLOW*2)).append(float(p))
            self.v.setdefault(s,deque(maxlen=self.EMA_MID*2)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.0); self.et.setdefault(s,0)
            self.sl.setdefault(s,self.STOP_MIN); self.tp.setdefault(s,self.STOP_MIN*self.TARGET_R)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT:
            return actions
        self._lc=self.t

        n_long=sum(1 for v in self.pos.values() if v=='long')
        n_short=sum(1 for v in self.pos.values() if v=='short')

        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=float(prices[sym]); ep=self.ep[sym]
            if len(h) < max(self.EMA_SLOW+1, self.PULLBACK_LB+5, self.TREND_LB+1):
                continue

            ef=_ema(h[-self.EMA_FAST*3:], self.EMA_FAST)
            em=_ema(h[-self.EMA_MID*2:], self.EMA_MID)
            es=_ema(h[-self.EMA_SLOW:],  self.EMA_SLOW)
            held=self.t-self.et[sym]
            stop=self.sl.get(sym, self.STOP_MIN)
            target=self.tp.get(sym, self.STOP_MIN*self.TARGET_R)

            if cur=='long':
                if (px<=ep*(1-stop) or px>=ep*(1+target) or held>=self.HOLD or
                        ef<em*0.9995):
                    actions[sym]=3; self.pos[sym]=None; n_long=max(0,n_long-1)
                continue
            if cur=='short':
                if (px>=ep*(1+stop) or px<=ep*(1-target) or held>=self.HOLD or
                        ef>em*1.0005):
                    actions[sym]=8; self.pos[sym]=None; n_short=max(0,n_short-1)
                continue

            trend_mom=_mom(h, self.TREND_LB)
            turn_mom=_mom(h, self.TURN_LB)
            recent=h[-self.PULLBACK_LB:]
            recent_high=float(max(recent)); recent_low=float(min(recent))
            vol_z=_vol_zscore(self.v[sym], window=min(60, max(20, len(self.v[sym])-1)))

            long_trend = ef>em*1.001 and em>es*1.0005 and trend_mom>self.TREND_MOM_MIN
            short_trend = ef<em*0.999 and em<es*0.9995 and trend_mom<-self.TREND_MOM_MIN

            if long_trend and n_long < self.MAX_POS:
                pullback = 1 - recent_low/max(recent_high, 1e-12)
                bounce = px/max(recent_low, 1e-12) - 1
                drawdown = 1 - px/max(recent_high, 1e-12)
                if (pullback >= self.PULLBACK_DEPTH and bounce >= self.BOUNCE_MIN and
                        -0.002 <= drawdown <= 0.012 and px >= ef*0.999 and
                        turn_mom >= self.TURN_MOM_MIN and vol_z >= self.VOL_Z_MIN):
                    stop, target = self._risk_params(h)
                    actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
                    self.sl[sym]=stop; self.tp[sym]=target
                    n_long += 1
                    continue

            if short_trend and n_short < self.MAX_POS:
                squeeze = recent_high/max(recent_low, 1e-12) - 1
                roll = 1 - recent_low/max(px, 1e-12)
                rebound = recent_high/max(px, 1e-12) - 1
                if (squeeze >= self.PULLBACK_DEPTH and rebound >= self.BOUNCE_MIN and
                        -0.002 <= roll <= 0.012 and px <= ef*1.001 and
                        turn_mom <= -self.TURN_MOM_MIN and vol_z >= self.VOL_Z_MIN):
                    stop, target = self._risk_params(h)
                    actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px; self.et[sym]=self.t
                    self.sl[sym]=stop; self.tp[sym]=target
                    n_short += 1
        return actions


class LiveMeanRev:
    """
    Mean Reversion v2: BB(50) + RSI, только в sideways.

    ИСПРАВЛЕНИЯ:
    - BB(20)→BB(50): меньше ложных сигналов в trending рынке
    - Порог входа: z<-1.5 (было -1.0)
    - Порог RSI: 30 (был 35)
    - Выход: z>-0.3 → exit (было -0.2, слишком быстро)
    - Trend filter: не торгуем в bull (mean-rev ломается)
    - Stop: 1% обязательный
    """
    BB_PERIOD=30; BB_STD=1.5; RSI_N=14; RSI_OB=60; RSI_OS=40
    ENTRY_Z=1.2
    EXIT_Z=0.35; CHECK_INT=10; STOP=0.012; HOLD=3*60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.t=0; self._lc=-9999

    def _bb_z(self,h):
        if len(h)<self.BB_PERIOD: return 0.
        w=h[-self.BB_PERIOD:]; mu=float(np.mean(w)); sd=float(np.std(w))
        return (h[-1]-mu)/(sd*self.BB_STD+1e-12)

    def _rsi(self,h,n):
        if len(h)<n*3: return 50.
        d=np.diff(list(h)[-(n*3):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
        ag,al=np.mean(g[:n]),np.mean(l[:n])
        for gi,li in zip(g[n:],l[n:]): ag=(ag*(n-1)+gi)/n; al=(al*(n-1)+li)/n
        return 100-100/(1+ag/(al+1e-9))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.BB_PERIOD*4)).append(float(p))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.); self.et.setdefault(s,0)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        regime=_regime(self.h,lb=self.BB_PERIOD)
        # Mean-rev плохо работает в сильном тренде вверх (только short заблокирован)
        bull_mode = regime=='bull'
        bear_mode = regime=='bear'
        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if len(h)<self.BB_PERIOD+self.RSI_N*3: continue
            z=self._bb_z(h); r=self._rsi(h,self.RSI_N); held=self.t-self.et[sym]
            # Выход: возврат к среднему ИЛИ стоп ИЛИ время
            if cur=='long':
                if z>-self.EXIT_Z or px<=ep*(1-self.STOP) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; continue
            elif cur=='short':
                if z<self.EXIT_Z or px>=ep*(1+self.STOP) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; continue
            if cur is not None: continue
            if z<-self.ENTRY_Z and r<self.RSI_OS and not bear_mode:
                actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
            elif z>self.ENTRY_Z and r>self.RSI_OB and not bull_mode:
                actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px; self.et[sym]=self.t
        return actions


class LiveVolCompress:
    """
    BB-Squeeze: сжатие → пробой.
    BBW < 2% → ждём пробой.
    Stop 0.5%, Target 1% (RR=2:1), max hold 2ч.
    """
    BB_PERIOD=20; BB_STD=2.; BBW_THRESH=0.015; MIN_BBW=0.003; MOM_N=5; MOM_MIN=0.003
    STOP=0.005; TARGET=0.010; HOLD=4*60; CHECK_INT=3  # FIX: HOLD увеличен 2h→4h

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.t=0; self._lc=-9999

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.BB_PERIOD*4)).append(float(p))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.); self.et.setdefault(s,0)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        regime = _regime(self.h, lb=max(60, self.BB_PERIOD * 2))
        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if len(h)<self.BB_PERIOD+self.MOM_N+2: continue
            w=h[-self.BB_PERIOD:]; mid=float(np.mean(w)); sd=float(np.std(w))
            held=self.t-self.et[sym]
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; continue
                continue
            if cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; continue
                continue
            if mid<1e-9: continue
            bbw=(2*self.BB_STD*sd)/mid
            mom=(h[-1]-h[-self.MOM_N-1])/(h[-self.MOM_N-1]+1e-12)
            if self.MIN_BBW < bbw < self.BBW_THRESH:
                if mom>self.MOM_MIN and regime != 'bear':
                    actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
                elif mom<-self.MOM_MIN and regime != 'bull':
                    actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px; self.et[sym]=self.t
        return actions


class FundingArb:
    """
    Funding Arb с надёжным fallback.

    С fetcher: contrarian по funding rate.
    Без fetcher: EMA(1h)/EMA(4h) crossover — momentum fallback.
    RSI fallback порог снижен 70/30→65/35.
    Stop: 1% обязательный.
    """
    STRONG=0.0002; CLOSE=0.00005; OI_MIN=100_000; CHECK_INT=15
    BULL_SHORT_BLOCK=0.0004; BEAR_LONG_BLOCK=0.0004
    RSI_OB=65; RSI_OS=35; STOP=0.01; TARGET=0.02; EMA_F=60; EMA_S=4*60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}
        self.t=0; self._lc=-9999

    def _rsi(self, h, n=14):
        if len(h)<n*3: return 50.
        d=np.diff(list(h)[-(n*3):]); g=np.where(d>0,d,0); l=np.where(d<0,-d,0)
        ag,al=np.mean(g[:n]),np.mean(l[:n])
        for gi,li in zip(g[n:],l[n:]): ag=(ag*(n-1)+gi)/n; al=(al*(n-1)+li)/n
        return 100-100/(1+ag/(al+1e-9))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.EMA_S*3)).append(float(p))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        has_fd=_FUNDING_FETCHER is not None
        regime = _regime(self.h, lb=max(self.EMA_F, 60))
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            # Стоп + тейк
            if cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET): actions[sym]=8; self.pos[sym]=None; continue
            elif cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET): actions[sym]=3; self.pos[sym]=None; continue
            if cur is not None: continue
            if has_fd:
                fd=_get_funding(sym); rate=fd.get("funding_rate"); oi=fd.get("open_interest_usdt",0)
                if rate is None: continue
                rate = float(rate)
                if oi>0 and oi<self.OI_MIN: continue
                if regime == 'bull':
                    if rate <= -self.STRONG:
                        actions[sym]=5; self.pos[sym]='long'; self.ep[sym]=px
                    elif rate >= self.BULL_SHORT_BLOCK:
                        continue
                elif regime == 'bear':
                    if rate >= self.STRONG * 0.5:
                        actions[sym]=7; self.pos[sym]='short'; self.ep[sym]=px
                    elif rate <= -self.BEAR_LONG_BLOCK:
                        continue
                else:
                    if rate>=self.STRONG:
                        actions[sym]=7; self.pos[sym]='short'; self.ep[sym]=px
                    elif rate<=-self.STRONG:
                        actions[sym]=5; self.pos[sym]='long'; self.ep[sym]=px
            else:
                # Запасной вариант: пересечение EMA (momentum, не contrarian)
                if len(h)<self.EMA_S+1: continue
                ef=_ema(list(h)[-self.EMA_F*3:],self.EMA_F)
                es=_ema(list(h)[-self.EMA_S:],  self.EMA_S)
                r=self._rsi(h)
                if ef>es*1.003 and r<self.RSI_OB and regime != 'bear':
                    actions[sym]=1; self.pos[sym]='long'; self.ep[sym]=px
                elif ef<es*0.997 and r>self.RSI_OS and regime != 'bull':
                    actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px
        return actions


class LiveOIBreakout:
    """
    OI Breakout v2 — ИСПРАВЛЕН.

    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обязательный hard stop 2%.
    Предыдущая версия потеряла -709 USDT без стоп-лосса.
    Leverage убран (spot only для long, spot short для short).
    """
    CHECK_INT=BAR; MOM_N=20; VOL_MULT=1.3; MOM_MIN=0.002
    STOP=0.02; TARGET=0.04; HOLD=3*60  # stop 2%, target 4%, RR=2:1

    def __init__(self):
        self.h:Dict[str,deque]={}; self.v:Dict[str,deque]={}
        self.pos:Dict[str,str]={}; self.ep:Dict[str,float]={}
        self.oi_prev:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.t=0; self._lc=-9999

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=6*BAR)).append(float(p))
            self.v.setdefault(s,deque(maxlen=4*BAR+10)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.)
            self.oi_prev.setdefault(s,0.); self.et.setdefault(s,0)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT: return actions
        self._lc=self.t
        for sym in prices:
            cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]; held=self.t-self.et[sym]
            # ОБЯЗАТЕЛЬНЫЙ стоп + тейк + время
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; continue
            elif cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; continue
            if cur is not None: continue
            if len(self.h[sym])<self.MOM_N+1: continue
            mom=_mom(self.h[sym],self.MOM_N)
            vl=list(self.v[sym])
            v_avg=float(np.mean(vl[-BAR-1:-1])) if len(vl)>BAR else 0
            v_cur=vl[-1] if vl else 0
            spike=v_cur>v_avg*self.VOL_MULT if v_avg>0 else False
            has_fd=_FUNDING_FETCHER is not None
            oi_exp=False
            if has_fd:
                fd=_get_funding(sym); oi=fd.get("open_interest_usdt",0)
                prev=self.oi_prev[sym]
                if oi>0 and prev>0 and (oi-prev)/prev>0.03: oi_exp=True
                self.oi_prev[sym]=oi if oi>0 else prev
            if spike or oi_exp:
                if mom>self.MOM_MIN:
                    actions[sym]=2; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
                elif mom<-self.MOM_MIN:
                    actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px; self.et[sym]=self.t
        return actions


class VolBreakoutHunter:
    """
    Volatility Expansion Breakout (shadow candidate, v1).

    НИША (чего не закрывают существующие агенты):
    - LiveVolCompress ловит пробой ИЗ сжатия (низкая BBW → пробой).
    - LiveTrendFollow и LiveRegimePullback торгуют уже сложившийся тренд.
    - LiveCrashHunter/MeanRev — counter-trend в откатах.
    Никто не ловит МОМЕНТ расширения волатильности с пробоем N-часового
    high/low на повышенном объёме — классический Donchian + ATR-фильтр.

    ЛОГИКА:
    1. Обновляем историю цен и объёмов (2*LOOKBACK для ATR).
    2. Считаем ATR (TR-прокси через |ret|·price) и требуем его расширения
       против медианы (ATR_EXPANSION_MIN).
    3. Требуем пробой LOOKBACK-часового high (для long) или low (для short).
    4. Объём текущего бара ≥ VOL_MULT · средний объём окна.
    5. Режим должен быть bull/bear/crash — в sideways не входим (там ложняк).
    6. Стоп и тейк — в ATR (не фиксированные %), чтобы корректно работать
       и на BTC ($0.3% ATR), и на мем-коинах ($3-5% ATR).
    7. После закрытия по стопу — cooldown на символ, чтобы не чейсить
       пилу при продолжающемся расширении волатильности.

    ПАРАМЕТРЫ ПОДОБРАНЫ КОНСЕРВАТИВНО:
    - LOOKBACK=4*60 (4 часа) — чтобы не быть заваленным ложными внутридневными
      пробоями, но и не отставать как дневной Turtle.
    - ATR_MULT_STOP=1.5, ATR_MULT_TARGET=3.0 — RR=2:1, как у большинства
      агентов в составе.
    - MAX_POS=3 — не больше 3 одновременных позиций (как в LiveRegimePullback).

    SHADOW-ONLY:
    Агент создан как кандидат для теневого контура. Попадёт в V_VolBreakoutHunter
    через SHADOW_MAP. Живые веса получит только после prom.-gate (см.
    promotion_gate.py): нужны ≥3 сигнала, ≥1 закрытая сделка, PnL/Sharpe
    не хуже baseline shadow-игроков.
    """
    # Интервалы и окна
    CHECK_INT    = 5                 # проверяем каждые 5 баров
    LOOKBACK     = 4 * 60            # 4 часа для Donchian high/low
    ATR_WINDOW   = 60                # ATR по последнему часу
    VOL_WINDOW   = 60                # объёмный бенчмарк по последнему часу
    # Условия входа
    VOL_MULT          = 1.4          # объём ≥ 1.4× среднего
    ATR_EXPANSION_MIN = 1.25         # ATR ≥ 1.25× медианы ATR за окно
    BREAKOUT_MARGIN   = 0.0005       # на 0.05% выше/ниже LOOKBACK-high/low
    # Риск-менеджмент (в долях ATR)
    ATR_MULT_STOP   = 1.5
    ATR_MULT_TARGET = 3.0            # RR=2:1
    HOLD            = 6 * 60         # макс. удержание 6 часов
    COOLDOWN_AFTER_STOP = 2 * 60     # 2 часа паузы после стопа по символу
    MAX_POS         = 3

    def __init__(self):
        self.h: Dict[str, deque]  = {}   # цены
        self.v: Dict[str, deque]  = {}   # объёмы
        self.pos: Dict[str, str]  = {}   # 'long' | 'short' | None
        self.ep: Dict[str, float] = {}   # entry price
        self.et: Dict[str, int]   = {}   # entry time (bar_index)
        self.sl: Dict[str, float] = {}   # stop loss (абсолютный в %)
        self.tp: Dict[str, float] = {}   # take profit (абсолютный в %)
        self.cooldown_until: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _atr_proxy(self, h_list: list) -> float:
        """ATR-прокси через среднее абсолютных ретёрнов за ATR_WINDOW баров."""
        if len(h_list) < self.ATR_WINDOW + 1:
            return 0.0
        w = np.array(h_list[-self.ATR_WINDOW - 1:], dtype=float)
        rets = np.abs(np.diff(w) / np.maximum(w[:-1], 1e-12))
        return float(np.mean(rets))

    def _atr_expansion_ratio(self, h_list: list) -> float:
        """
        Отношение текущего ATR к медианному за LOOKBACK.
        > 1 = волатильность расширяется, < 1 = сжимается.
        """
        if len(h_list) < self.LOOKBACK + 1:
            return 0.0
        atr_now = self._atr_proxy(h_list)
        if atr_now <= 0:
            return 0.0
        # Строим ряд скользящих ATR и берём медиану
        arr = np.array(h_list[-self.LOOKBACK - 1:], dtype=float)
        rets = np.abs(np.diff(arr) / np.maximum(arr[:-1], 1e-12))
        # Скользящее среднее длиной ATR_WINDOW
        if len(rets) < self.ATR_WINDOW * 2:
            return 0.0
        atr_series = np.convolve(
            rets,
            np.ones(self.ATR_WINDOW) / self.ATR_WINDOW,
            mode="valid",
        )
        median_atr = float(np.median(atr_series))
        if median_atr <= 0:
            return 0.0
        return atr_now / median_atr

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        # Обновляем историю
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.LOOKBACK * 2 + 10)).append(float(p))
            self.v.setdefault(s, deque(maxlen=self.VOL_WINDOW * 2 + 10)).append(
                float(volumes.get(s, 0) or 0)
            )
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self.sl.setdefault(s, 0.0)
            self.tp.setdefault(s, 0.0)

        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        regime = _regime(self.h)
        # В боковике breakouts чаще ложные — не входим (но управляем открытыми).
        allow_entry = regime in ("bull", "bear", "crash")

        n_open = sum(1 for v in self.pos.values() if v is not None)

        for sym in prices:
            cur = self.pos[sym]
            px = float(prices[sym])
            ep = self.ep[sym]
            held = self.t - self.et[sym]
            stop_pct = self.sl.get(sym, 0.0)
            target_pct = self.tp.get(sym, 0.0)

            # ──── Управление открытыми позициями ────
            if cur == "long":
                if (stop_pct > 0 and px <= ep * (1 - stop_pct)) or \
                   (target_pct > 0 and px >= ep * (1 + target_pct)) or \
                   held >= self.HOLD:
                    # Если по стопу — ставим кулдаун
                    if stop_pct > 0 and px <= ep * (1 - stop_pct):
                        self.cooldown_until[sym] = self.t + self.COOLDOWN_AFTER_STOP
                    actions[sym] = 3
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if cur == "short":
                if (stop_pct > 0 and px >= ep * (1 + stop_pct)) or \
                   (target_pct > 0 and px <= ep * (1 - target_pct)) or \
                   held >= self.HOLD:
                    if stop_pct > 0 and px >= ep * (1 + stop_pct):
                        self.cooldown_until[sym] = self.t + self.COOLDOWN_AFTER_STOP
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            # ──── Поиск новых входов ────
            if not allow_entry:
                continue
            if n_open >= self.MAX_POS:
                continue
            if self.t < self.cooldown_until.get(sym, 0):
                continue

            h_list = list(self.h[sym])
            v_list = list(self.v[sym])
            if len(h_list) < self.LOOKBACK + 1:
                continue
            if len(v_list) < self.VOL_WINDOW + 1:
                continue

            # Donchian: high/low за LOOKBACK (не включая текущий бар)
            window = h_list[-self.LOOKBACK - 1:-1]
            donch_high = float(max(window))
            donch_low  = float(min(window))

            # ATR и его расширение
            atr = self._atr_proxy(h_list)
            if atr <= 0:
                continue
            expansion = self._atr_expansion_ratio(h_list)
            if expansion < self.ATR_EXPANSION_MIN:
                continue

            # Объём
            v_avg = float(np.mean(v_list[-self.VOL_WINDOW - 1:-1]))
            v_cur = float(v_list[-1])
            if v_avg <= 0 or v_cur < v_avg * self.VOL_MULT:
                continue

            stop_pct_new   = atr * self.ATR_MULT_STOP
            target_pct_new = atr * self.ATR_MULT_TARGET
            # Санитарные ограничения: не меньше 0.3% и не больше 5%
            stop_pct_new   = min(max(stop_pct_new, 0.003), 0.05)
            target_pct_new = min(max(target_pct_new, 0.006), 0.10)

            # Пробой high — long
            if px > donch_high * (1 + self.BREAKOUT_MARGIN):
                actions[sym] = 1
                self.pos[sym] = "long"
                self.ep[sym] = px
                self.et[sym] = self.t
                self.sl[sym] = stop_pct_new
                self.tp[sym] = target_pct_new
                n_open += 1
                continue
            # Пробой low — short
            if px < donch_low * (1 - self.BREAKOUT_MARGIN):
                actions[sym] = 6
                self.pos[sym] = "short"
                self.ep[sym] = px
                self.et[sym] = self.t
                self.sl[sym] = stop_pct_new
                self.tp[sym] = target_pct_new
                n_open += 1
        return actions


# ══════════════════════════════════════════════════════════════════
# ИГРОКИ  (переработаны)
# ══════════════════════════════════════════════════════════════════

class CarryFlowAgentV2:
    """
    Carry/flow агент для shadow-контуров.

    Использует funding + crowding + OI expansion и имеет безопасный fallback,
    если деривативные данные недоступны или устарели.
    """
    CHECK_INT = 20
    HOLD = 8 * 60
    STOP = 0.012
    TARGET = 0.024
    FUNDING_ENTRY = 0.00008
    FUNDING_EXIT = 0.00002
    OI_SPIKE = 0.02
    CROWD_RATIO = 0.58
    BASIS_ENTRY = 0.0004
    EXTREME_EXT = 0.008
    EMA_FAST = 60
    EMA_SLOW = 4 * 60
    RSI_N = 14
    RSI_OB = 66
    RSI_OS = 34
    MAX_DATA_AGE_SEC = 20 * 60

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.oi_h: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _rsi(self, h, n=14):
        if len(h) < n * 3:
            return 50.0
        d = np.diff(list(h)[-(n * 3):])
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag, al = np.mean(g[:n]), np.mean(l[:n])
        for gi, li in zip(g[n:], l[n:]):
            ag = (ag * (n - 1) + gi) / n
            al = (al * (n - 1) + li) / n
        return float(100 - 100 / (1 + ag / (al + 1e-9)))

    def _fresh_funding(self, sym: str) -> dict:
        fd = _get_funding(sym)
        if not fd:
            return {}
        updated_ts = float(fd.get("updated_ts", 0) or 0)
        if updated_ts > 0 and (time.time() - updated_ts) > self.MAX_DATA_AGE_SEC:
            return {}
        return fd

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.EMA_SLOW * 2)).append(float(p))
            self.oi_h.setdefault(s, deque(maxlen=16))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            fd = self._fresh_funding(s)
            oi_now = float(fd.get("open_interest_usdt", 0) or 0)
            if oi_now > 0:
                self.oi_h[s].append(oi_now)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t
        for sym in prices:
            h = self.h[sym]
            cur = self.pos[sym]
            px = float(prices[sym])
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            fd = self._fresh_funding(sym)
            rate = float(fd.get("funding_rate", 0) or 0)
            long_ratio = float(fd.get("long_ratio", 0.5) or 0.5)
            short_ratio = float(fd.get("short_ratio", max(0.0, 1.0 - long_ratio)) or 0.5)
            mark = float(fd.get("mark_price", 0) or 0)
            index = float(fd.get("index_price", 0) or 0)
            basis = (mark / index - 1.0) if mark > 0 and index > 0 else 0.0
            ext = 0.0
            if len(h) >= self.EMA_FAST + 3:
                ef = _ema(list(h)[-self.EMA_FAST * 3:], self.EMA_FAST)
                if ef > 0:
                    ext = px / ef - 1.0
            else:
                ef = 0.0
            rsi = self._rsi(h, self.RSI_N)
            oi_vals = list(self.oi_h[sym])
            oi_chg = 0.0
            if len(oi_vals) >= 2 and oi_vals[-2] > 0:
                oi_chg = oi_vals[-1] / oi_vals[-2] - 1.0

            if cur == 'long':
                normalized = (
                    fd
                    and rate >= -self.FUNDING_EXIT
                    and short_ratio < 0.54
                    and basis > -self.BASIS_ENTRY * 0.5
                    and ext >= -self.EXTREME_EXT * 0.25
                )
                if (
                    px <= ep * (1 - self.STOP)
                    or px >= ep * (1 + self.TARGET)
                    or held >= self.HOLD
                    or normalized
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                continue
            if cur == 'short':
                normalized = (
                    fd
                    and rate <= self.FUNDING_EXIT
                    and long_ratio < 0.54
                    and basis < self.BASIS_ENTRY * 0.5
                    and ext <= self.EXTREME_EXT * 0.25
                )
                if (
                    px >= ep * (1 + self.STOP)
                    or px <= ep * (1 - self.TARGET)
                    or held >= self.HOLD
                    or normalized
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                continue

            if len(h) < self.EMA_SLOW + 5:
                continue
            es = _ema(list(h)[-self.EMA_SLOW:], self.EMA_SLOW)
            trend_up = ef > es * 1.001 if es > 0 else False
            trend_dn = ef < es * 0.999 if es > 0 else False

            if fd and oi_chg >= self.OI_SPIKE:
                if (
                    rate >= self.FUNDING_ENTRY
                    and long_ratio >= self.CROWD_RATIO
                    and basis >= self.BASIS_ENTRY
                    and (ext >= self.EXTREME_EXT * 0.5 or rsi >= self.RSI_OB)
                ):
                    actions[sym] = 7
                    self.pos[sym] = 'short'
                    self.ep[sym] = px
                    self.et[sym] = self.t
                elif (
                    rate <= -self.FUNDING_ENTRY
                    and short_ratio >= self.CROWD_RATIO
                    and basis <= -self.BASIS_ENTRY
                    and (ext <= -self.EXTREME_EXT * 0.5 or rsi <= self.RSI_OS)
                ):
                    actions[sym] = 5
                    self.pos[sym] = 'long'
                    self.ep[sym] = px
                    self.et[sym] = self.t
                continue

            if not fd:
                if trend_up and ext <= -self.EXTREME_EXT and rsi <= self.RSI_OS:
                    actions[sym] = 5
                    self.pos[sym] = 'long'
                    self.ep[sym] = px
                    self.et[sym] = self.t
                elif trend_dn and ext >= self.EXTREME_EXT and rsi >= self.RSI_OB:
                    actions[sym] = 7
                    self.pos[sym] = 'short'
                    self.ep[sym] = px
                    self.et[sym] = self.t
        return actions


class ExternalSignalAgent:
    """
    Shadow-only агент для внешних локальных сигналов.

    Поддерживает файл с сигналами и использует свои stop/target/time exits.
    """
    CHECK_INT = 3
    STOP = 0.015
    TARGET = 0.03
    HOLD = 12 * 60
    MIN_CONFIDENCE = 0.55
    MAX_SIGNAL_AGE_SEC = 30 * 60

    def __init__(self):
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self._last_token: Dict[str, str] = {}
        self._signals_cache: Dict[str, dict] = {}
        self._last_load_bar = -99999
        self._last_mtime = 0.0
        self.t = 0

    def _signal_path(self) -> str:
        return os.getenv("EXTERNAL_SIGNALS_FILE") or os.path.join(SCRIPT_DIR, "external_signals.json")

    def _normalize_symbol(self, raw) -> str:
        if raw is None:
            return ""
        sym = str(raw).strip().upper()
        if not sym:
            return ""
        if sym.endswith(":USDT"):
            sym = sym[:-5]
        for suffix in ("/USDT", "_USDT", "USDT", "/USD", "_USD", "USD"):
            if sym.endswith(suffix) and len(sym) > len(suffix):
                sym = sym[:-len(suffix)]
                break
        for sep in ("/", "_", ":"):
            if sep in sym:
                sym = sym.split(sep)[0]
        return sym

    def _parse_ts(self, raw, fallback: float) -> float:
        if raw is None:
            return float(fallback)
        if isinstance(raw, (int, float)):
            val = float(raw)
            return val / 1000.0 if val > 1e12 else val
        try:
            text = str(raw).strip()
            if not text:
                return float(fallback)
            if text.isdigit():
                val = float(text)
                return val / 1000.0 if val > 1e12 else val
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return float(fallback)

    def _normalize_side(self, raw):
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            code = int(raw)
            if code in (4, 5, 1, 2):
                return "long"
            if code in (6, 7):
                return "short"
            if code in (3, 8):
                return "close"
            return None
        side = str(raw).strip().lower()
        if side in {"buy", "long", "open_long", "enter_long", "bull", "bullish"}:
            return "long"
        if side in {"sell", "short", "open_short", "enter_short", "bear", "bearish"}:
            return "short"
        if side in {"close", "exit", "flat", "neutral", "reduce", "close_all"}:
            return "close"
        return None

    def _load_signals(self) -> Dict[str, dict]:
        path = self._signal_path()
        if not os.path.exists(path):
            self._signals_cache = {}
            self._last_mtime = 0.0
            return {}
        try:
            mtime = os.path.getmtime(path)
            if self._signals_cache and self._last_mtime == mtime:
                return self._signals_cache
            with open(path, "r", encoding="utf-8-sig") as fh:
                payload = json.load(fh)
        except Exception as exc:
            log.debug("ExternalSignalAgent load failed: %s", exc)
            self._signals_cache = {}
            return {}

        rows = []
        if isinstance(payload, dict):
            if isinstance(payload.get("signals"), list):
                rows = payload["signals"]
            elif any(k in payload for k in ("symbol", "sym", "side", "action", "signal")):
                rows = [payload]
            else:
                for symbol, meta in payload.items():
                    if isinstance(meta, dict):
                        row = dict(meta)
                        row.setdefault("symbol", symbol)
                        rows.append(row)
                    elif isinstance(meta, str):
                        rows.append({"symbol": symbol, "side": meta})
        elif isinstance(payload, list):
            rows = payload

        normalized = {}
        fallback_ts = self._last_mtime or time.time()
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = self._normalize_symbol(
                row.get("symbol") or row.get("sym") or row.get("ticker") or row.get("pair")
            )
            side = self._normalize_side(row.get("side", row.get("action", row.get("signal"))))
            if not symbol or side is None:
                continue
            confidence = float(row.get("confidence", row.get("score", row.get("strength", 1.0))) or 0)
            ts = self._parse_ts(row.get("ts", row.get("timestamp", row.get("time"))), fallback=mtime)
            token = str(row.get("id") or row.get("token") or f"{symbol}:{side}:{int(ts)}")
            prev = normalized.get(symbol)
            if prev and (prev["ts"], prev["confidence"]) >= (ts, confidence):
                continue
            normalized[symbol] = {
                "side": side,
                "confidence": confidence,
                "ts": ts if ts > 0 else fallback_ts,
                "token": token,
            }
        self._last_mtime = mtime
        self._signals_cache = normalized
        return normalized

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s in prices:
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self._last_token.setdefault(s, "")
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._last_load_bar >= self.CHECK_INT:
            self._load_signals()
            self._last_load_bar = self.t

        now_ts = time.time()
        for sym, px_raw in prices.items():
            px = float(px_raw)
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]

            if cur == 'long':
                if px <= ep * (1 - self.STOP) or px >= ep * (1 + self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    continue
            elif cur == 'short':
                if px >= ep * (1 + self.STOP) or px <= ep * (1 - self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    continue

            sig = self._signals_cache.get(sym)
            if not sig:
                continue
            if sig["confidence"] < self.MIN_CONFIDENCE:
                continue
            if now_ts - sig["ts"] > self.MAX_SIGNAL_AGE_SEC:
                continue
            if sig["token"] == self._last_token.get(sym):
                continue

            side = sig["side"]
            if side == "close":
                if cur is not None:
                    actions[sym] = 8
                    self.pos[sym] = None
                self._last_token[sym] = sig["token"]
                continue
            if cur is not None:
                if (cur == 'long' and side == "short") or (cur == 'short' and side == "long"):
                    actions[sym] = 8
                    self.pos[sym] = None
                    self._last_token[sym] = sig["token"]
                continue
            if side == "long":
                actions[sym] = 5
                self.pos[sym] = 'long'
                self.ep[sym] = px
                self.et[sym] = self.t
                self._last_token[sym] = sig["token"]
            elif side == "short":
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
                self._last_token[sym] = sig["token"]
        return actions


class ResearchValidatorAgent:
    """
    Консервативный benchmark-агент для shadow research.

    Ищет только чистые breakout/trend setups с подтверждением объёмом.
    """
    CHECK_INT = 30
    EMA_FAST = 60
    EMA_MID = 4 * 60
    EMA_SLOW = 12 * 60
    BREAKOUT = 90
    VOL_WIN = 60
    NOISE_WIN = 180
    STOP = 0.015
    TARGET = 0.03
    HOLD = 8 * 60
    VOL_MULT = 1.18
    MAX_NOISE = 0.0068

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _noise(self, hist, win):
        vals = list(hist)
        if len(vals) < win + 1:
            return 1.0
        arr = np.asarray(vals[-(win + 1):], dtype=float)
        ret = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        return float(np.mean(np.abs(ret)))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.EMA_SLOW * 2)).append(float(p))
            self.v.setdefault(s, deque(maxlen=self.VOL_WIN * 6)).append(float(volumes.get(s, 0)))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        for sym, px_raw in prices.items():
            h = self.h[sym]
            v = self.v[sym]
            px = float(px_raw)
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < self.EMA_SLOW + 5 or len(v) < self.VOL_WIN + 3:
                continue

            ef = _ema(list(h)[-self.EMA_FAST * 3:], self.EMA_FAST)
            em = _ema(list(h)[-self.EMA_MID:], self.EMA_MID)
            es = _ema(list(h)[-self.EMA_SLOW:], self.EMA_SLOW)

            if cur == 'long':
                if (
                    px <= ep * (1 - self.STOP)
                    or px >= ep * (1 + self.TARGET)
                    or held >= self.HOLD
                    or ef < em
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                continue
            if cur == 'short':
                if (
                    px >= ep * (1 + self.STOP)
                    or px <= ep * (1 - self.TARGET)
                    or held >= self.HOLD
                    or ef > em
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                continue

            prev = list(h)[-(self.BREAKOUT + 1):-1]
            if len(prev) < self.BREAKOUT:
                continue
            vol_avg = float(np.mean(list(v)[-self.VOL_WIN - 1:-1])) if len(v) > self.VOL_WIN else 0.0
            vol_spike = float(v[-1]) > vol_avg * self.VOL_MULT if vol_avg > 0 else False
            noise = self._noise(h, self.NOISE_WIN)
            if not vol_spike or noise > self.MAX_NOISE:
                continue

            if ef > em > es and px >= max(prev) * 1.001:
                actions[sym] = 5
                self.pos[sym] = 'long'
                self.ep[sym] = px
                self.et[sym] = self.t
            elif ef < em < es and px <= min(prev) * 0.999:
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
        return actions


class PlayerBomberman:
    """
    Bomberman v3 с двойным консенсусом.
    Обе конфигурации должны согласиться → очень мало сделок, высокая точность.
    """
    def __init__(self):
        self._b1=Bomberman()
        self._b2=Bomberman()
        self._b2.BOP_THRESH=0.7; self._b2.DONCHIAN_PERIOD=20; self._b2.MRC_PERIOD=100

    def reset_for_live(self, bar_index: int = 0):
        for agent in (self._b1, self._b2):
            if hasattr(agent, '_lc'):
                agent._lc = -99999
            try:
                agent.t = bar_index
            except (AttributeError, TypeError):
                pass
            pos = getattr(agent, 'pos', None)
            if isinstance(pos, dict):
                for key in pos:
                    pos[key] = None
            entry_px = getattr(agent, 'entry_px', None)
            if isinstance(entry_px, dict):
                for key in entry_px:
                    entry_px[key] = 0.0

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        kw=dict(prices=prices,volumes=volumes,month=month,
                portfolio_value=portfolio_value,bar_index=bar_index)
        a1=self._b1.act(**kw); a2=self._b2.act(**kw)
        r={s:0 for s in prices}
        for sym in prices:
            v1=a1.get(sym,0); v2=a2.get(sym,0)
            if v1==v2 and v1!=0: r[sym]=v1   # консенсус
            elif v1 in (3,8) or v2 in (3,8): r[sym]=v1 or v2  # выход без консенсуса
        return r


class PlayerFunding:
    """
    FundingArb + MomentumScalper — арбитраж финансирования + скальпер.

    FIX v9.1: старая логика (v1==v2) требовала идентичных сигналов — 0 сделок.
    Теперь: weighted voting 60%/40%, buy при w>=0.4, sell при любом агенте.
    """
    def __init__(self):
        self._fa=FundingArb(); self._ms=MomentumScalper()

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        kw=dict(prices=prices,volumes=volumes,month=month,
                portfolio_value=portfolio_value,bar_index=bar_index)
        a1=self._fa.act(**kw); a2=self._ms.act(**kw)
        r={s:0 for s in prices}
        for sym in prices:
            v1=a1.get(sym,0); v2=a2.get(sym,0)
            # FIX: sell/close — приоритет любого агента
            if v1 in (3,8) or v2 in (3,8):
                r[sym] = v1 if v1 in (3,8) else v2
            # Buy/Long: достаточно одного агента (снижен порог с 2/2 до 1/2)
            elif v1!=0 and v1 not in (3,8):
                r[sym] = v1
            elif v2!=0 and v2 not in (3,8):
                r[sym] = v2
        return r


class Panteon:
    """
    Адаптивный игрок v4. Пересмотренный состав по результатам live-тестов.

    LIVE результаты v3 за 2103 баров (35ч paper):
      ★ FundingArb     +91.2%  55 signals  — БЕЗОГОВОРОЧНЫЙ ЛИДЕР
      LiveCrashHunter   +2.2%  139 signals
      LiveAfterShock    +1.9%  378 signals
      MomentumScalper   -9.2%  6 signals   — убыточен, демотирован

    КРИТИЧЕСКИЙ FIX v4:
      BUG: sub-agents сохраняли warmup-позиции в self.pos → HOLD навечно
      FIX: reset_for_live() сбрасывает pos, ep, _lc всех sub-agents
      BUG: _named zip смешивал имена агентов (косметический)

    Новый состав v4 (FundingArb — доминант):
      Bull:     FundingArb (45%) + LiveAfterShock (30%) + LiveCrashHunter (25%)
      Bear:     LiveCrashHunter (40%) + FundingArb (40%) + LiveAfterShock (20%)
      Neutral:  FundingArb (45%) + LiveAfterShock (30%) + LiveCrashHunter (25%)

    ИСКЛЮЧЕНЫ:
      - MomentumScalper (-9.2%) — убыточен, требует vol_spike который редок
      - LiveMeanRev, LiveTrendFollow — убыточны в прошлых тестах
    """
    STATE_INT = 2 * 60
    MAX_POS   = 3     # defensive default: small live accounts degrade quickly above 3 futures legs

    # FIX v7: символы из FUTURES_BLACKLIST — агент не должен их торговать,
    # потому что close_all() молча не работает для них → маржа блокируется навечно
    _BLACKLIST = frozenset({
        "EUR", "USD1", "USDT", "USDC", "STABLE", "BUSD",
        "GOLD(XAUT)", "GOLD(PAXG)", "FIDA", "WXT", "PSAI", "PE",
        "ATLA", "META", "ONT", "HYPE", "BULLA", "STO", "SOLV", "RED",
    })

    def __init__(self):
        self._ms=MomentumScalper()
        self._lch=LiveCrashHunter()
        self._fa=FundingArb()
        self._las=LiveAfterShock()
        self._lrp=LiveRegimePullback()
        self._lmr=LiveMeanRev()       # v7: добавлен в пул для ротации
        self._lvc=LiveVolCompress()    # v7: добавлен в пул для ротации
        self._ltf=LiveTrendFollow()   # v8: доступен для ротации, но не включён в дефолтные веса
        self._rd=RichardDennisTurtle()
        self._loi=LiveOIBreakout()
        self._cf=CarryFlowAgentV2()
        self._rv=ResearchValidatorAgent()
        self._vbh=VolBreakoutHunter()
        self._gb = None
        self._gbr = None
        try:
            from crypto_agents import _GeneticsAdapter
            try:
                from crypto_genetics import GeneticsBullishAgent
                self._gb = _GeneticsAdapter(GeneticsBullishAgent())
            except Exception:
                pass
            try:
                from crypto_genetics import GeneticsBearishAgent
                self._gbr = _GeneticsAdapter(GeneticsBearishAgent())
            except Exception:
                pass
        except Exception:
            pass
        self._r=None; self._lr=0
        self._t=0
        self._ph: Dict[str, deque] = {}
        self._vh: Dict[str, deque] = {}
        self._open_pos: Dict[str, dict] = {}
        self._position_safety = PositionExitGovernor(close_action=8)
        self._meta_logger = log
        self._shadow_scoring = ShadowScoringAgent()
        self._meta_memory = ContextMemoryAgent()
        self._memory_store = MemorySnapshotStore(
            REAL_PLAYER_MEMORY_FILE,
            LEGACY_REAL_PLAYER_MEMORY_FILE,
            save_interval_bars=30,
        )
        self._memory_namespace = _sanitize_memory_namespace(_DEFAULT_MEMORY_NAMESPACE)
        self._portfolio_allocator = PortfolioAllocatorAgent()

        # ═══════════════════════════════════════════════════════════════
        # ADAPTIVE ROTATION v7
        # ═══════════════════════════════════════════════════════════════
        # Полный пул агентов: {label → (agent_obj, attr_name)}
        self._agent_pool = OrderedDict(
            (label, agent) for _, label, agent in self.iter_subagents()
        )

        # Активные агенты с весами (обновляются ротацией)
        # Дефолт: FundingArb доминирует (по результатам 35ч live-теста)
        self._active_weights: Dict[str, float] = {
            'LiveAfterShock':     0.30,
            'LiveMeanRev':        0.25,
            'LiveCrashHunter':    0.20,
            'CarryFlowAgentV2':   0.15,
            'FundingArb':         0.10,
        }
        self._last_rotation = 0
        self._agent_success_memory: Dict[str, float] = {}
        self._last_shadow_snapshot: Dict[str, dict] = {}
        self._last_memory_save_bar = -99999
        self._memory_bootstrap_loaded = False
        self._memory_enabled = False
        self._current_context: Dict[str, str] = {}
        self._last_context_bar = -99999
        self._context_memory: Dict[str, Dict[str, dict]] = {}
        self._factor_memory: Dict[str, Dict[str, Dict[str, dict]]] = {}
        self._regime_memory: Dict[str, Dict[str, dict]] = {}
        self._shadow_window_anchor: Dict[str, dict] = {}
        self._symbol_memory: Dict[str, Dict[str, dict]] = {}
        self._symbol_regime_memory: Dict[str, Dict[str, Dict[str, dict]]] = {}
        self._symbol_factor_memory: Dict[str, Dict[str, Dict[str, dict]]] = {}
        self._shadow_symbol_anchor: Dict[str, Dict[str, dict]] = {}
        self._recent_real_signal_agents: deque = deque(maxlen=self.RECENT_SIGNAL_WINDOW)
        self._label_to_shadow = {
            label: shadow_name for shadow_name, label in self.SHADOW_MAP.items()
        }

        # Теневая статистика доходности — приходит извне через set_shadow_perf()
        self._shadow_perf: Optional[Dict[str, dict]] = None

    def iter_subagents(self):
        """Единый список sub-agent'ов Panteon для reset/logging/live-sync."""
        for attr_name, label in (
            ('_fa', 'FundingArb'),
            ('_ms', 'MomentumScalper'),
            ('_las', 'LiveAfterShock'),
            ('_lch', 'LiveCrashHunter'),
            ('_lrp', 'LiveRegimePullback'),
            ('_lmr', 'LiveMeanRev'),
            ('_ltf', 'LiveTrendFollow'),
            ('_lvc', 'LiveVolCompress'),
            ('_rd', 'RichardDennis'),
            ('_loi', 'LiveOIBreakout'),
            ('_cf', 'CarryFlowAgentV2'),
            ('_rv', 'ResearchValidatorAgent'),
            ('_vbh', 'VolBreakoutHunter'),
            ('_gb', 'GeneticsBullish'),
            ('_gbr', 'GeneticsBearish'),
        ):
            if label in self.LIVE_AGENT_BLOCKLIST:
                continue
            agent = getattr(self, attr_name, None)
            if agent is not None:
                yield attr_name, label, agent

    def subagent_labels(self):
        return [label for _, label, _ in self.iter_subagents()]

    # ── Параметры управления позициями ──────────────────────────────────
    SL_PCT      = 0.04   # v5: стоп-лосс 4%
    TP_PCT      = 0.06   # v5: тейк-профит 6%
    TRAIL_PCT   = 0.025  # v5: trailing stop 2.5%
    STALE_BARS  = 720    # v5: stale exit 12ч

    # ── Параметры адаптивной ротации ─────────────────────────────────
    ROTATION_INT  = 30
    MIN_AGENTS    = 2
    MAX_AGENTS    = 3
    MIN_WEIGHT    = 0.10
    SOFT_MIN_WEIGHT = 0.04
    MAX_WEIGHT    = 0.78
    REGIME_PRIORITY_FLOOR = 0.10
    MEMORY_SCHEMA_VERSION = 6
    USE_REGIME_ONLY_MEMORY = True
    RESTORE_SAVED_ACTIVE_WEIGHTS = False
    RESTORE_GLOBAL_SUCCESS_MEMORY = False
    CONTEXT_REFRESH_INT = 15
    GLOBAL_MEMORY_WEIGHT = 0.0
    CONTEXT_FACTOR_WEIGHT = 0.0
    CONTEXT_FULL_WEIGHT = 0.0
    REGIME_MEMORY_WEIGHT = 1.35
    REGIME_STATIC_PRIOR_WEIGHT = 0.55
    REGIME_MEMORY_CONFIDENCE_SAMPLES = 6
    REGIME_SCORE_CLIP = 4.0
    MAX_CONTEXTS_PER_AGENT = 48
    CONTEXT_FACTORS = (
        'regime', 'trend_local', 'trend_global', 'trend_alignment',
        'noise', 'volatility', 'breadth', 'dispersion',
        'volume_regime', 'session_utc', 'season', 'weekday_type',
    )
    SYMBOL_PROFILE_FACTORS = (
        'asset_group', 'symbol_regime', 'symbol_trend', 'symbol_strength',
        'symbol_volatility', 'symbol_noise', 'symbol_volume',
    )
    SYMBOL_EXACT_WEIGHT = 0.16
    SYMBOL_PROFILE_WEIGHT = 0.10
    SYMBOL_REGIME_WEIGHT = 0.14
    SYMBOL_OPPORTUNITY_WEIGHT = 0.12
    SYMBOL_REGIME_ROTATION_WEIGHT = 0.95
    SYMBOL_REGIME_STATIC_PRIOR_WEIGHT = 0.40
    MAX_MIXED_SYMBOL_REGIME_AGENTS = 5
    SYMBOL_REGIME_COVERAGE_TOLERANCE = 1.25
    SYMBOL_VOTE_STRENGTH = 0.35
    OPEN_SINGLE_THRESHOLD = 0.30
    OPEN_MULTI_THRESHOLD = 0.24
    CLOSE_SINGLE_THRESHOLD = 0.25
    CLOSE_MULTI_THRESHOLD = 0.20
    CLOSE_STRONG_THRESHOLD = 0.32
    SMALL_ACCOUNT_VALUE = 90.0
    SMALL_ACCOUNT_OPEN_BONUS = 0.04
    SMALL_ACCOUNT_CLOSE_BONUS = 0.025
    NOISY_MARKET_OPEN_BONUS = 0.03
    NOISY_MARKET_CLOSE_BONUS = 0.02
    WEAK_LINEUP_OPEN_BONUS = 0.025
    WEAK_LINEUP_CLOSE_BONUS = 0.015
    MIN_MULTI_AGENT_SUPPORT = 2
    MAX_DISCRETIONARY_CLOSES_PER_BAR = 1
    RECENT_SIGNAL_WINDOW = 80
    DOMINANCE_SOFT_SHARE = 0.30
    DOMINANCE_HARD_SHARE = 0.45
    RISK_MULTIPLIER_MIN = 0.55
    RISK_MULTIPLIER_MAX = 1.35
    MAX_SYMBOLS_PER_AGENT = 40
    SINGLE_AGENT_STRONG_THRESHOLD = 0.34
    SINGLE_AGENT_MIN_CLOSED_TRADES = 6
    SINGLE_AGENT_MIN_PNL_PCT = 0.10
    SINGLE_AGENT_MIN_SHARPE = 0.0
    SINGLE_AGENT_MAX_DD_PCT = 12.0
    UNPROVEN_SINGLE_THRESHOLD = 0.72
    UNPROVEN_SINGLE_MIN_ENTRIES = 4
    UNPROVEN_SINGLE_MIN_PNL_PCT = 0.35
    MEMORY_HARD_NEGATIVE_SCORE = -18.0
    MEMORY_RECOVERY_PNL_PCT = 0.35
    MEMORY_RECOVERY_CLOSED_TRADES = 6
    ROTATION_HARD_LOSS_PCT = -0.35
    ROTATION_HARD_NEGATIVE_SHARPE = -8.0
    ROTATION_MIN_CLOSED_FOR_BLOCK = 4
    LIVE_AGENT_BLOCKLIST = frozenset({
        # Multi-run log review: these live sub-agents had persistent negative
        # contribution or clearly bad latest signals. They remain in shadow
        # monitoring, but cannot vote in the real ensemble.
        'MomentumScalper',
        'LiveRegimePullback',
        'LiveVolCompress',
        'RichardDennis',
    })
    REGIME_STATIC_WEIGHTS = {
        'bullish': {
            'LiveAfterShock': 0.28,
            'LiveMeanRev': 0.24,
            'LiveCrashHunter': 0.18,
            'CarryFlowAgentV2': 0.16,
            'FundingArb': 0.08,
            'LiveTrendFollow': 0.06,
        },
        'neutral': {
            'LiveMeanRev': 0.30,
            'LiveAfterShock': 0.22,
            'FundingArb': 0.16,
            'CarryFlowAgentV2': 0.16,
            'LiveCrashHunter': 0.12,
            'ResearchValidatorAgent': 0.04,
        },
        'bearish': {
            'LiveCrashHunter': 0.30,
            'CarryFlowAgentV2': 0.24,
            'LiveMeanRev': 0.18,
            'FundingArb': 0.14,
            'LiveAfterShock': 0.10,
            'LiveOIBreakout': 0.04,
        },
        'crash': {
            # Start crash as regular bearish. It will diverge through
            # regime_memory after real crash observations are collected.
            'LiveCrashHunter': 0.30,
            'CarryFlowAgentV2': 0.24,
            'LiveMeanRev': 0.18,
            'FundingArb': 0.14,
            'LiveAfterShock': 0.10,
            'LiveOIBreakout': 0.04,
        },
    }
    REGIME_PRIORITY_MAP = {
        'bullish': ('LiveCrashHunter', 'LiveTrendFollow', 'LiveOIBreakout', 'CarryFlowAgentV2', 'MomentumScalper', 'LiveRegimePullback', 'RichardDennis', 'FundingArb'),
        'bearish': ('CarryFlowAgentV2', 'LiveTrendFollow', 'LiveOIBreakout', 'LiveCrashHunter', 'FundingArb', 'RichardDennis', 'MomentumScalper'),
        'neutral': ('CarryFlowAgentV2', 'FundingArb', 'LiveCrashHunter', 'LiveOIBreakout', 'MomentumScalper', 'LiveRegimePullback', 'RichardDennis'),
        'crash': ('CarryFlowAgentV2', 'LiveTrendFollow', 'LiveOIBreakout', 'LiveCrashHunter', 'FundingArb', 'RichardDennis', 'MomentumScalper'),
    }
    REGIME_SCORE_BONUS = {
        'bullish': {
            'LiveCrashHunter': 0.16,
            'LiveTrendFollow': 0.12,
            'LiveOIBreakout': 0.11,
            'CarryFlowAgentV2': 0.10,
            'ResearchValidatorAgent': 0.08,
            'VolBreakoutHunter': 0.06,
            'MomentumScalper': 0.14,
            'LiveRegimePullback': 0.12,
            'RichardDennis': 0.08,
            'FundingArb': 0.04,
            'LiveVolCompress': -0.16,
            'LiveMeanRev': -0.08,
            'LiveAfterShock': -0.08,
        },
        'bearish': {
            'CarryFlowAgentV2': 0.14,
            'LiveTrendFollow': 0.11,
            'LiveOIBreakout': 0.10,
            'ResearchValidatorAgent': 0.06,
            'LiveCrashHunter': 0.18,
            'FundingArb': 0.12,
            'RichardDennis': 0.10,
            'MomentumScalper': 0.06,
            'LiveAfterShock': -0.06,
        },
        'neutral': {
            'CarryFlowAgentV2': 0.13,
            'LiveOIBreakout': 0.08,
            'ResearchValidatorAgent': 0.05,
            'VolBreakoutHunter': 0.04,
            'FundingArb': 0.14,
            'LiveCrashHunter': 0.12,
            'MomentumScalper': 0.10,
            'LiveRegimePullback': 0.08,
            'RichardDennis': 0.04,
            'LiveMeanRev': -0.06,
            'LiveAfterShock': -0.06,
            'LiveVolCompress': -0.10,
        },
        'crash': {
            'CarryFlowAgentV2': 0.14,
            'LiveTrendFollow': 0.11,
            'LiveOIBreakout': 0.10,
            'ResearchValidatorAgent': 0.06,
            'LiveCrashHunter': 0.18,
            'FundingArb': 0.12,
            'RichardDennis': 0.10,
            'MomentumScalper': 0.06,
            'LiveAfterShock': -0.06,
        },
    }
    MAJOR_SYMBOLS = frozenset({
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'TRX',
        'AVAX', 'DOT', 'LINK', 'UNI', 'NEAR', 'LTC', 'BCH', 'TON',
    })
    MEME_SYMBOLS = frozenset({
        'DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'FLOKI', 'MEME',
        'BRETT', 'PONKE', 'MOG', 'BOME', 'TURBO',
    })

    def _detect_regime_fast(self) -> str:
        """
        Быстрый мульти-таймфрейм детектор режима.

        3 окна: 4ч, 12ч, 24ч. Голосование большинством.
        Пороги: ±0.3% за 4ч, ±0.8% за 12ч, ±1.5% за 24ч.
        Если нет явного тренда → neutral (не bearish по дефолту).
        """
        short_rets = self._collect_universe_returns(60)
        mid_rets = self._collect_universe_returns(240)
        if len(short_rets) >= 6 or len(mid_rets) >= 6:
            short_med = float(np.median(short_rets)) if short_rets else 0.0
            mid_med = float(np.median(mid_rets)) if mid_rets else 0.0
            short_down = (
                sum(1 for ret in short_rets if ret <= -0.004) / len(short_rets)
                if short_rets else 0.0
            )
            short_hard_down = (
                sum(1 for ret in short_rets if ret <= -0.012) / len(short_rets)
                if short_rets else 0.0
            )
            mid_down = (
                sum(1 for ret in mid_rets if ret <= -0.010) / len(mid_rets)
                if mid_rets else 0.0
            )
            mid_hard_down = (
                sum(1 for ret in mid_rets if ret <= -0.025) / len(mid_rets)
                if mid_rets else 0.0
            )
            abs_rets = self._collect_universe_abs_returns(60)
            volume_ratios = self._collect_volume_ratios()
            volatility = float(np.median(abs_rets)) if abs_rets else 0.0
            volume_ratio = float(np.median(volume_ratios)) if volume_ratios else 1.0
            panic_1h = short_med <= -0.010 and short_down >= 0.70 and short_hard_down >= 0.35
            panic_4h = mid_med <= -0.025 and mid_down >= 0.70 and mid_hard_down >= 0.35
            severe_move = short_med <= -0.020 or mid_med <= -0.045
            risk_expansion = volume_ratio >= 1.25 or volatility >= 0.0035 or severe_move
            if (panic_1h or panic_4h) and risk_expansion:
                return 'crash'

        votes = []  # +1=bull, -1=bear, 0=neutral
        for lb_bars, threshold in [(240, 0.003), (720, 0.008), (1440, 0.015)]:
            moms = []
            for h in self._ph.values():
                lst = list(h)
                if len(lst) < lb_bars + 1:
                    continue
                base = lst[-lb_bars - 1]
                if base > 0:
                    moms.append(lst[-1] / base - 1)
            if not moms:
                votes.append(0)
                continue
            median_mom = float(np.median(moms))
            if median_mom > threshold:
                votes.append(1)
            elif median_mom < -threshold:
                votes.append(-1)
            else:
                votes.append(0)

        s = sum(votes)
        if s >= 2:
            return 'bullish'
        elif s <= -2:
            return 'bearish'
        return 'neutral'

    def _bucket_ternary(self, value: float, low_cut: float, high_cut: float,
                        low_label: str, mid_label: str, high_label: str) -> str:
        if value <= low_cut:
            return low_label
        if value >= high_cut:
            return high_label
        return mid_label

    def _collect_universe_returns(self, lb: int) -> list:
        out = []
        for hist in self._ph.values():
            lst = list(hist)
            if len(lst) < lb + 1:
                continue
            base = lst[-lb - 1]
            if base > 0:
                out.append(lst[-1] / base - 1.0)
        return out

    def _collect_universe_abs_returns(self, lb: int) -> list:
        out = []
        for hist in self._ph.values():
            lst = list(hist)
            if len(lst) < lb + 1:
                continue
            w = np.array(lst[-lb - 1:], dtype=float)
            rets = np.diff(w) / np.maximum(w[:-1], 1e-12)
            if len(rets):
                out.append(float(np.mean(np.abs(rets))))
        return out

    def _collect_universe_noise(self, lb: int) -> list:
        out = []
        for hist in self._ph.values():
            lst = list(hist)
            if len(lst) < lb + 1:
                continue
            w = np.array(lst[-lb - 1:], dtype=float)
            rets = np.diff(w) / np.maximum(w[:-1], 1e-12)
            path = float(np.sum(np.abs(rets)))
            net = abs(float(w[-1] / max(w[0], 1e-12) - 1.0))
            if path > 0:
                out.append(path / max(net, 1e-4))
        return out

    def _collect_volume_ratios(self, fast_lb: int = 60, slow_lb: int = 360) -> list:
        out = []
        for hist in self._vh.values():
            lst = list(hist)
            if len(lst) < slow_lb:
                continue
            fast = float(np.mean(lst[-fast_lb:])) if fast_lb > 0 else 0.0
            slow = float(np.mean(lst[-slow_lb:])) if slow_lb > 0 else 0.0
            if slow > 1e-9:
                out.append(fast / slow)
        return out

    def _detect_session_utc(self, now_utc: datetime) -> str:
        hour = int(now_utc.hour)
        if 0 <= hour < 6:
            return 'asia'
        if 6 <= hour < 12:
            return 'europe'
        if 12 <= hour < 18:
            return 'us'
        return 'late'

    def _detect_season(self, month: int) -> str:
        if month in (12, 1, 2):
            return 'winter'
        if month in (3, 4, 5):
            return 'spring'
        if month in (6, 7, 8):
            return 'summer'
        return 'autumn'

    def _context_key(self, context: Dict[str, str]) -> str:
        return "|".join(
            f"{factor}={context.get(factor, 'unknown')}"
            for factor in self.CONTEXT_FACTORS
        )

    def _context_summary(self, context: Dict[str, str]) -> str:
        if not context:
            return "unknown"
        focus = ('regime', 'trend_local', 'trend_global', 'noise', 'session_utc')
        return ", ".join(f"{k}={context.get(k, 'unknown')}" for k in focus)

    def _canonical_market_regime(self, regime: Optional[str]) -> str:
        key = str(regime or '').strip().lower()
        if key in {'bull', 'bullish', 'up', 'long'}:
            return 'bullish'
        if key in {'bear', 'bearish', 'down', 'short'}:
            return 'bearish'
        if key in {'crash', 'panic', 'capitulation'}:
            return 'crash'
        return 'neutral'

    def _canonical_symbol_regime(self, regime: Optional[str]) -> str:
        key = str(regime or '').strip().lower()
        if key in {'bull', 'bullish', 'up', 'long'}:
            return 'bullish'
        if key in {'bear', 'bearish', 'down', 'short'}:
            return 'bearish'
        if key in {'crash', 'panic', 'capitulation'}:
            return 'crash'
        if key in {'flat', 'sideways', 'neutral', 'range'}:
            return 'neutral'
        return 'unknown'

    def _detect_symbol_regime(self, sym: str, hist: Optional[list] = None) -> str:
        prices = list(hist if hist is not None else (self._ph.get(sym) or []))
        if len(prices) < 61:
            return 'unknown'
        last = float(prices[-1])
        if last <= 0:
            return 'unknown'

        short_ret = last / max(float(prices[-61]), 1e-12) - 1.0
        medium_ret = (
            last / max(float(prices[-241]), 1e-12) - 1.0
            if len(prices) >= 241 else short_ret
        )
        long_ret = (
            last / max(float(prices[-721]), 1e-12) - 1.0
            if len(prices) >= 721 else medium_ret
        )

        if medium_ret <= -0.080 or long_ret <= -0.120 or (
            short_ret <= -0.045 and medium_ret <= -0.020
        ):
            return 'crash'
        if short_ret >= 0.004 and medium_ret >= 0.010:
            return 'bullish'
        if short_ret <= -0.004 and medium_ret <= -0.010:
            return 'bearish'
        if medium_ret >= 0.025 and short_ret > -0.002:
            return 'bullish'
        if medium_ret <= -0.025 and short_ret < 0.002:
            return 'bearish'
        return 'neutral'

    def _build_market_context(self, month: Optional[int] = None) -> Dict[str, str]:
        now_utc = datetime.now(timezone.utc)
        month_val = int(month) if isinstance(month, int) and 1 <= month <= 12 else now_utc.month
        regime = self._canonical_market_regime(self._r or self._detect_regime_fast())

        short_rets = self._collect_universe_returns(60)
        long_rets = self._collect_universe_returns(720)
        breadth_rets = self._collect_universe_returns(240)
        abs_rets = self._collect_universe_abs_returns(60)
        noise_vals = self._collect_universe_noise(180)
        volume_ratios = self._collect_volume_ratios()

        short_ret = float(np.median(short_rets)) if short_rets else 0.0
        long_ret = float(np.median(long_rets)) if long_rets else 0.0
        volatility = float(np.median(abs_rets)) if abs_rets else 0.0
        noise = float(np.median(noise_vals)) if noise_vals else 5.0
        dispersion = float(np.std(short_rets)) if len(short_rets) > 1 else 0.0
        volume_ratio = float(np.median(volume_ratios)) if volume_ratios else 1.0
        breadth_ratio = (
            sum(1 for ret in breadth_rets if ret > 0) / len(breadth_rets)
            if breadth_rets else 0.5
        )

        trend_local = self._bucket_ternary(short_ret, -0.0025, 0.0025, 'down', 'flat', 'up')
        trend_global = self._bucket_ternary(long_ret, -0.0100, 0.0100, 'down', 'flat', 'up')
        trend_alignment = (
            trend_local if trend_local == trend_global and trend_local in ('up', 'down')
            else 'mixed'
        )
        noise_bucket = self._bucket_ternary(noise, 3.0, 7.0, 'clean', 'mixed', 'noisy')
        volatility_bucket = self._bucket_ternary(volatility, 0.0015, 0.0045, 'low', 'normal', 'high')
        dispersion_bucket = self._bucket_ternary(dispersion, 0.0030, 0.0120, 'low', 'normal', 'high')
        volume_bucket = self._bucket_ternary(volume_ratio, 0.85, 1.15, 'quiet', 'normal', 'active')

        if breadth_ratio >= 0.67:
            breadth = 'broad_up'
        elif breadth_ratio <= 0.33:
            breadth = 'broad_down'
        else:
            breadth = 'mixed'

        return {
            'regime': regime,
            'trend_local': trend_local,
            'trend_global': trend_global,
            'trend_alignment': trend_alignment,
            'noise': noise_bucket,
            'volatility': volatility_bucket,
            'breadth': breadth,
            'dispersion': dispersion_bucket,
            'volume_regime': volume_bucket,
            'session_utc': self._detect_session_utc(now_utc),
            'season': self._detect_season(month_val),
            'weekday_type': 'weekend' if now_utc.weekday() >= 5 else 'weekday',
        }

    def _empty_context_stats(self, context: Optional[Dict[str, str]] = None) -> dict:
        return {
            'context': dict(context or {}),
            'ema_score': 0.0,
            'avg_pnl': 0.0,
            'samples': 0,
            'wins': 0,
            'losses': 0,
            'last_window_pnl': 0.0,
            'last_updated': '',
        }

    def _regime_static_prior(self, label: str, regime: Optional[str]) -> float:
        regime_key = self._canonical_market_regime(regime)
        profile = self.REGIME_STATIC_WEIGHTS.get(regime_key, {})
        if label in self.LIVE_AGENT_BLOCKLIST:
            return 0.0
        return float(profile.get(label, 0.0) or 0.0)

    def _regime_memory_score(self, label: str, regime: Optional[str]) -> float:
        regime_key = self._canonical_market_regime(regime)
        stats = self._regime_memory.get(regime_key, {}).get(label)
        if not isinstance(stats, dict):
            return 0.0
        samples = int(stats.get('samples', 0) or 0)
        if samples <= 0:
            return 0.0
        confidence = min(samples / max(float(self.REGIME_MEMORY_CONFIDENCE_SAMPLES), 1.0), 1.0)
        ema_score = float(stats.get('ema_score', 0.0) or 0.0)
        avg_pnl = float(stats.get('avg_pnl', 0.0) or 0.0)
        wins = int(stats.get('wins', 0) or 0)
        losses = int(stats.get('losses', 0) or 0)
        balance = (wins - losses) / max(samples, 1)
        score = ema_score * 0.72 + avg_pnl * 0.18 + balance * 0.45
        return float(np.clip(score, -self.REGIME_SCORE_CLIP, self.REGIME_SCORE_CLIP)) * confidence

    def _update_stats_bucket(self, stats: dict, score: float, pnl: float,
                             context: Optional[Dict[str, str]] = None) -> dict:
        samples = int(stats.get('samples', 0) or 0)
        alpha = 0.22 if samples < 6 else 0.12
        stats['context'] = dict(context or stats.get('context') or {})
        stats['ema_score'] = float(stats.get('ema_score', 0.0)) * (1.0 - alpha) + float(score) * alpha
        stats['avg_pnl'] = (
            (float(stats.get('avg_pnl', 0.0)) * samples + float(pnl)) / (samples + 1)
            if samples >= 0 else float(pnl)
        )
        stats['samples'] = samples + 1
        stats['wins'] = int(stats.get('wins', 0) or 0) + int(score > 0.40 or pnl > 0.15)
        stats['losses'] = int(stats.get('losses', 0) or 0) + int(score < -0.40 or pnl < -0.15)
        stats['last_window_pnl'] = float(pnl)
        stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        return stats

    def _prune_context_memory(self, label: str):
        ctx_map = self._context_memory.get(label)
        if not isinstance(ctx_map, dict) or len(ctx_map) <= self.MAX_CONTEXTS_PER_AGENT:
            return
        ranked = sorted(
            ctx_map.items(),
            key=lambda kv: (int(kv[1].get('samples', 0) or 0), str(kv[1].get('last_updated', ''))),
            reverse=True,
        )
        self._context_memory[label] = dict(ranked[:self.MAX_CONTEXTS_PER_AGENT])

    def _sanitize_shadow_anchor(self, raw: Optional[Dict[str, dict]]) -> Dict[str, dict]:
        out = {}
        if not isinstance(raw, dict):
            return out
        for name, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue
            out[str(name)] = {
                'pnl_pct': float(metrics.get('pnl_pct', 0.0) or 0.0),
                'signals': int(metrics.get('signals', 0) or 0),
                'entries': int(metrics.get('entries', 0) or 0),
                'closed_trades': int(metrics.get('closed_trades', metrics.get('total_trades', 0)) or 0),
                'win_rate': float(metrics.get('win_rate', 0.0) or 0.0),
                'sharpe': float(metrics.get('sharpe', 0.0) or 0.0),
                'max_dd': float(metrics.get('max_dd', metrics.get('max_drawdown_pct', metrics.get('max_dd_pct', 0.0))) or 0.0),
            }
        return out

    def _build_context_observation(self, shadow_name: str, perf: dict) -> Optional[dict]:
        prev = self._shadow_window_anchor.get(shadow_name, {})
        delta_pnl = float(perf.get('pnl_pct', 0.0) or 0.0) - float(prev.get('pnl_pct', 0.0) or 0.0)
        signals = max(int(perf.get('signals', 0) or 0) - int(prev.get('signals', 0) or 0), 0)
        entries = max(int(perf.get('entries', 0) or 0) - int(prev.get('entries', 0) or 0), 0)
        closed = max(
            int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0) -
            int(prev.get('closed_trades', prev.get('total_trades', 0)) or 0),
            0,
        )
        if signals == 0 and entries == 0 and closed == 0 and abs(delta_pnl) < 0.05:
            return None

        win_rate = float(perf.get('win_rate', 0.0) or 0.0)
        sharpe = float(perf.get('sharpe', 0.0) or 0.0)
        max_dd_change = max(
            abs(float(perf.get('max_dd', perf.get('max_drawdown_pct', perf.get('max_dd_pct', 0.0))) or 0.0)) -
            abs(float(prev.get('max_dd', 0.0) or 0.0)),
            0.0,
        )
        activity = min(signals, 20) * 0.05 + min(entries, 12) * 0.08 + min(closed, 8) * 0.10
        win_bonus = ((win_rate - 50.0) / 12.0) if closed >= 2 else 0.0
        score = delta_pnl * 0.85 + sharpe * 0.20 + activity + win_bonus - max_dd_change * 0.35
        return {
            'score': float(score),
            'window_pnl': float(delta_pnl),
            'signals': int(signals),
            'entries': int(entries),
            'closed_trades': int(closed),
        }

    def _record_context_observation(self, label: str, context: Dict[str, str], observation: dict):
        context_key = self._context_key(context)
        agent_ctx = self._context_memory.setdefault(label, {})
        stats = agent_ctx.get(context_key) or self._empty_context_stats(context)
        agent_ctx[context_key] = self._update_stats_bucket(
            stats, observation['score'], observation['window_pnl'], context=context,
        )
        self._prune_context_memory(label)

        for factor in self.CONTEXT_FACTORS:
            bucket = str(context.get(factor, 'unknown'))
            factor_store = self._factor_memory.setdefault(factor, {})
            bucket_store = factor_store.setdefault(bucket, {})
            bucket_stats = bucket_store.get(label) or self._empty_context_stats()
            bucket_store[label] = self._update_stats_bucket(
                bucket_stats, observation['score'], observation['window_pnl'],
            )

    def _record_regime_observation(self, label: str, regime: Optional[str], observation: dict):
        regime_key = self._canonical_market_regime(regime)
        regime_store = self._regime_memory.setdefault(regime_key, {})
        stats = regime_store.get(label) or self._empty_context_stats({'regime': regime_key})
        regime_store[label] = self._update_stats_bucket(
            stats,
            observation['score'],
            observation['window_pnl'],
            context={'regime': regime_key},
        )

    def _update_context_memory(self, context: Optional[Dict[str, str]] = None):
        self._meta_memory.update_context_memory(self, context)

    def _restore_context_memory(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        restored: Dict[str, Dict[str, dict]] = {}
        if not isinstance(raw, dict):
            return restored
        for label, ctx_map in raw.items():
            if label not in self._agent_pool or not isinstance(ctx_map, dict):
                continue
            clean_ctx = {}
            for ctx_key, stats in ctx_map.items():
                if not isinstance(stats, dict):
                    continue
                clean_ctx[str(ctx_key)] = {
                    'context': dict(stats.get('context') or {}),
                    'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                    'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                    'samples': int(stats.get('samples', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'last_window_pnl': float(stats.get('last_window_pnl', 0.0) or 0.0),
                    'last_updated': str(stats.get('last_updated', '') or ''),
                }
            if clean_ctx:
                restored[label] = clean_ctx
        return restored

    def _restore_factor_memory(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, Dict[str, dict]]]:
        restored: Dict[str, Dict[str, Dict[str, dict]]] = {}
        if not isinstance(raw, dict):
            return restored
        for factor, bucket_map in raw.items():
            if not isinstance(bucket_map, dict):
                continue
            clean_buckets = {}
            for bucket, agent_map in bucket_map.items():
                if not isinstance(agent_map, dict):
                    continue
                clean_agents = {}
                for label, stats in agent_map.items():
                    if label not in self._agent_pool or not isinstance(stats, dict):
                        continue
                    clean_agents[label] = {
                        'context': {},
                        'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                        'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                        'samples': int(stats.get('samples', 0) or 0),
                        'wins': int(stats.get('wins', 0) or 0),
                        'losses': int(stats.get('losses', 0) or 0),
                        'last_window_pnl': float(stats.get('last_window_pnl', 0.0) or 0.0),
                        'last_updated': str(stats.get('last_updated', '') or ''),
                    }
                if clean_agents:
                    clean_buckets[str(bucket)] = clean_agents
            if clean_buckets:
                restored[str(factor)] = clean_buckets
        return restored

    def _restore_regime_memory(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        restored: Dict[str, Dict[str, dict]] = {}
        if not isinstance(raw, dict):
            return restored
        for regime, agent_map in raw.items():
            if not isinstance(agent_map, dict):
                continue
            regime_key = self._canonical_market_regime(regime)
            clean_agents = {}
            for label, stats in agent_map.items():
                if label not in self._agent_pool or not isinstance(stats, dict):
                    continue
                clean_agents[label] = {
                    'context': {'regime': regime_key},
                    'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                    'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                    'samples': int(stats.get('samples', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'last_window_pnl': float(stats.get('last_window_pnl', 0.0) or 0.0),
                    'last_updated': str(stats.get('last_updated', '') or ''),
                }
            if clean_agents:
                restored[regime_key] = clean_agents
        return restored

    def _bootstrap_regime_memory_from_factors(
        self,
        factor_memory: Optional[Dict[str, Dict[str, Dict[str, dict]]]],
    ) -> Dict[str, Dict[str, dict]]:
        regime_factor = (factor_memory or {}).get('regime')
        if not isinstance(regime_factor, dict):
            return {}
        restored: Dict[str, Dict[str, dict]] = {}
        for regime, agent_map in regime_factor.items():
            if not isinstance(agent_map, dict):
                continue
            regime_key = self._canonical_market_regime(regime)
            clean_agents = {}
            for label, stats in agent_map.items():
                if label not in self._agent_pool or not isinstance(stats, dict):
                    continue
                clean_agents[label] = {
                    'context': {'regime': regime_key},
                    'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                    'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                    'samples': int(stats.get('samples', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'last_window_pnl': float(stats.get('last_window_pnl', 0.0) or 0.0),
                    'last_updated': str(stats.get('last_updated', '') or ''),
                }
            if clean_agents:
                restored[regime_key] = clean_agents
        return restored

    def _context_memory_score(self, label: str, context: Optional[Dict[str, str]]) -> tuple[float, float]:
        return self._meta_memory.context_memory_score(self, label, context)

    def _asset_group(self, sym: str) -> str:
        base = str(sym or '').replace('_USDT', '').replace('USDT', '')
        if base in self.MEME_SYMBOLS:
            return 'meme'
        if base in self.MAJOR_SYMBOLS:
            return 'major'
        return 'alt'

    def _build_symbol_profile(self, sym: str) -> Dict[str, str]:
        hist = list(self._ph.get(sym) or [])
        vol_hist = list(self._vh.get(sym) or [])
        symbol_regime = self._detect_symbol_regime(sym, hist)
        if len(hist) < 181:
            return {
                'asset_group': self._asset_group(sym),
                'symbol_regime': symbol_regime,
                'symbol_trend': 'unknown',
                'symbol_strength': 'unknown',
                'symbol_volatility': 'unknown',
                'symbol_noise': 'unknown',
                'symbol_volume': 'unknown',
            }

        local_ret = hist[-1] / max(hist[-61], 1e-12) - 1.0 if len(hist) >= 61 else 0.0
        rel_ret = hist[-1] / max(hist[-241], 1e-12) - 1.0 if len(hist) >= 241 else local_ret
        universe_rets = self._collect_universe_returns(240)
        universe_mid = float(np.median(universe_rets)) if universe_rets else 0.0
        strength = rel_ret - universe_mid

        h_arr = np.array(hist[-61:], dtype=float)
        rets = np.diff(h_arr) / np.maximum(h_arr[:-1], 1e-12)
        vol_val = float(np.mean(np.abs(rets))) if len(rets) else 0.0

        h_noise = np.array(hist[-181:], dtype=float)
        noise_rets = np.diff(h_noise) / np.maximum(h_noise[:-1], 1e-12)
        path = float(np.sum(np.abs(noise_rets)))
        net = abs(float(h_noise[-1] / max(h_noise[0], 1e-12) - 1.0))
        noise_val = path / max(net, 1e-4) if len(noise_rets) else 5.0

        if len(vol_hist) >= 360:
            fast_vol = float(np.mean(vol_hist[-60:]))
            slow_vol = float(np.mean(vol_hist[-360:]))
            volume_ratio = fast_vol / max(slow_vol, 1e-9)
        else:
            volume_ratio = 1.0

        return {
            'asset_group': self._asset_group(sym),
            'symbol_regime': symbol_regime,
            'symbol_trend': self._bucket_ternary(local_ret, -0.003, 0.003, 'down', 'flat', 'up'),
            'symbol_strength': self._bucket_ternary(strength, -0.015, 0.015, 'weak', 'mid', 'strong'),
            'symbol_volatility': self._bucket_ternary(vol_val, 0.0018, 0.0055, 'low', 'normal', 'high'),
            'symbol_noise': self._bucket_ternary(noise_val, 3.0, 7.5, 'clean', 'mixed', 'noisy'),
            'symbol_volume': self._bucket_ternary(volume_ratio, 0.85, 1.20, 'quiet', 'normal', 'active'),
        }

    def _symbol_profile_key(self, profile: Dict[str, str]) -> str:
        return "|".join(
            f"{factor}={profile.get(factor, 'unknown')}"
            for factor in self.SYMBOL_PROFILE_FACTORS
        )

    def _empty_symbol_stats(self, profile: Optional[Dict[str, str]] = None) -> dict:
        return {
            'profile': dict(profile or {}),
            'ema_score': 0.0,
            'avg_pnl': 0.0,
            'samples': 0,
            'wins': 0,
            'losses': 0,
            'last_total_pnl': 0.0,
            'last_updated': '',
        }

    def _restore_symbol_memory(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        restored: Dict[str, Dict[str, dict]] = {}
        if not isinstance(raw, dict):
            return restored
        for label, sym_map in raw.items():
            if label not in self._agent_pool or not isinstance(sym_map, dict):
                continue
            clean_map = {}
            for sym, stats in sym_map.items():
                if not isinstance(stats, dict):
                    continue
                clean_map[str(sym)] = {
                    'profile': dict(stats.get('profile') or {}),
                    'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                    'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                    'samples': int(stats.get('samples', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'last_total_pnl': float(stats.get('last_total_pnl', 0.0) or 0.0),
                    'last_updated': str(stats.get('last_updated', '') or ''),
                }
            if clean_map:
                restored[label] = clean_map
        return restored

    def _restore_symbol_regime_memory(
        self,
        raw: Optional[Dict[str, dict]],
    ) -> Dict[str, Dict[str, Dict[str, dict]]]:
        restored: Dict[str, Dict[str, Dict[str, dict]]] = {}
        if not isinstance(raw, dict):
            return restored
        for label, sym_map in raw.items():
            if label not in self._agent_pool or not isinstance(sym_map, dict):
                continue
            clean_symbols = {}
            for sym, regime_map in sym_map.items():
                if not isinstance(regime_map, dict):
                    continue
                clean_regimes = {}
                for regime, stats in regime_map.items():
                    if not isinstance(stats, dict):
                        continue
                    regime_key = self._canonical_symbol_regime(regime)
                    if regime_key == 'unknown':
                        continue
                    clean_regimes[regime_key] = {
                        'profile': dict(stats.get('profile') or {'symbol_regime': regime_key}),
                        'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                        'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                        'samples': int(stats.get('samples', 0) or 0),
                        'wins': int(stats.get('wins', 0) or 0),
                        'losses': int(stats.get('losses', 0) or 0),
                        'last_total_pnl': float(stats.get('last_total_pnl', 0.0) or 0.0),
                        'last_updated': str(stats.get('last_updated', '') or ''),
                    }
                if clean_regimes:
                    clean_symbols[str(sym)] = clean_regimes
            if clean_symbols:
                restored[label] = clean_symbols
        return restored

    def _restore_symbol_factor_memory(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, Dict[str, dict]]]:
        restored: Dict[str, Dict[str, Dict[str, dict]]] = {}
        if not isinstance(raw, dict):
            return restored
        for factor, bucket_map in raw.items():
            if not isinstance(bucket_map, dict):
                continue
            clean_buckets = {}
            for bucket, agent_map in bucket_map.items():
                if not isinstance(agent_map, dict):
                    continue
                clean_agents = {}
                for label, stats in agent_map.items():
                    if label not in self._agent_pool or not isinstance(stats, dict):
                        continue
                    clean_agents[label] = {
                        'profile': {},
                        'ema_score': float(stats.get('ema_score', 0.0) or 0.0),
                        'avg_pnl': float(stats.get('avg_pnl', 0.0) or 0.0),
                        'samples': int(stats.get('samples', 0) or 0),
                        'wins': int(stats.get('wins', 0) or 0),
                        'losses': int(stats.get('losses', 0) or 0),
                        'last_total_pnl': float(stats.get('last_total_pnl', 0.0) or 0.0),
                        'last_updated': str(stats.get('last_updated', '') or ''),
                    }
                if clean_agents:
                    clean_buckets[str(bucket)] = clean_agents
            if clean_buckets:
                restored[str(factor)] = clean_buckets
        return restored

    def _sanitize_shadow_symbol_anchor(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        restored: Dict[str, Dict[str, dict]] = {}
        if not isinstance(raw, dict):
            return restored
        for shadow_name, sym_map in raw.items():
            if not isinstance(sym_map, dict):
                continue
            clean_map = {}
            for sym, stats in sym_map.items():
                if not isinstance(stats, dict):
                    continue
                clean_map[str(sym)] = {
                    'signals': int(stats.get('signals', 0) or 0),
                    'entries': int(stats.get('entries', 0) or 0),
                    'closed_trades': int(stats.get('closed_trades', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'total_pnl_pct': float(stats.get('total_pnl_pct', 0.0) or 0.0),
                    'realized_pnl_pct': float(stats.get('realized_pnl_pct', 0.0) or 0.0),
                    'last_bar': int(stats.get('last_bar', 0) or 0),
                }
            if clean_map:
                restored[str(shadow_name)] = clean_map
        return restored

    def _update_symbol_bucket(self, stats: dict, score: float, pnl: float,
                              profile: Optional[Dict[str, str]] = None) -> dict:
        samples = int(stats.get('samples', 0) or 0)
        alpha = 0.20 if samples < 6 else 0.10
        stats['profile'] = dict(profile or stats.get('profile') or {})
        stats['ema_score'] = float(stats.get('ema_score', 0.0)) * (1.0 - alpha) + float(score) * alpha
        stats['avg_pnl'] = (float(stats.get('avg_pnl', 0.0)) * samples + float(pnl)) / (samples + 1)
        stats['samples'] = samples + 1
        stats['wins'] = int(stats.get('wins', 0) or 0) + int(score > 0.25 or pnl > 0.10)
        stats['losses'] = int(stats.get('losses', 0) or 0) + int(score < -0.25 or pnl < -0.10)
        stats['last_total_pnl'] = float(pnl)
        stats['last_updated'] = datetime.now(timezone.utc).isoformat()
        return stats

    def _prune_symbol_memory(self, label: str):
        sym_map = self._symbol_memory.get(label)
        if not isinstance(sym_map, dict) or len(sym_map) <= self.MAX_SYMBOLS_PER_AGENT:
            return
        ranked = sorted(
            sym_map.items(),
            key=lambda kv: (int(kv[1].get('samples', 0) or 0), str(kv[1].get('last_updated', ''))),
            reverse=True,
        )
        self._symbol_memory[label] = dict(ranked[:self.MAX_SYMBOLS_PER_AGENT])

    def _build_symbol_observation(self, shadow_name: str, sym: str, stats: dict) -> Optional[dict]:
        prev = self._shadow_symbol_anchor.get(shadow_name, {}).get(sym, {})
        delta_total = float(stats.get('total_pnl_pct', 0.0) or 0.0) - float(prev.get('total_pnl_pct', 0.0) or 0.0)
        delta_realized = float(stats.get('realized_pnl_pct', 0.0) or 0.0) - float(prev.get('realized_pnl_pct', 0.0) or 0.0)
        entries = max(int(stats.get('entries', 0) or 0) - int(prev.get('entries', 0) or 0), 0)
        closed = max(int(stats.get('closed_trades', 0) or 0) - int(prev.get('closed_trades', 0) or 0), 0)
        signals = max(int(stats.get('signals', 0) or 0) - int(prev.get('signals', 0) or 0), 0)
        if signals == 0 and entries == 0 and closed == 0 and abs(delta_total) < 0.03:
            return None
        activity = min(signals, 10) * 0.05 + min(entries, 6) * 0.09 + min(closed, 6) * 0.11
        score = delta_realized * 0.75 + delta_total * 0.35 + activity
        return {
            'score': float(score),
            'pnl_pct': float(delta_total),
        }

    def _record_symbol_observation(self, label: str, sym: str, profile: Dict[str, str], observation: dict):
        agent_symbols = self._symbol_memory.setdefault(label, {})
        stats = agent_symbols.get(sym) or self._empty_symbol_stats(profile)
        agent_symbols[sym] = self._update_symbol_bucket(
            stats, observation['score'], observation['pnl_pct'], profile=profile,
        )
        self._prune_symbol_memory(label)

        for factor in self.SYMBOL_PROFILE_FACTORS:
            bucket = str(profile.get(factor, 'unknown'))
            factor_store = self._symbol_factor_memory.setdefault(factor, {})
            bucket_store = factor_store.setdefault(bucket, {})
            bucket_stats = bucket_store.get(label) or self._empty_symbol_stats()
            bucket_store[label] = self._update_symbol_bucket(
                bucket_stats, observation['score'], observation['pnl_pct'],
            )

    def _prune_symbol_regime_memory(self, label: str):
        sym_map = self._symbol_regime_memory.get(label)
        if not isinstance(sym_map, dict) or len(sym_map) <= self.MAX_SYMBOLS_PER_AGENT:
            return
        ranked = sorted(
            sym_map.items(),
            key=lambda kv: (
                sum(int(stats.get('samples', 0) or 0) for stats in kv[1].values() if isinstance(stats, dict)),
                max((str(stats.get('last_updated', '')) for stats in kv[1].values() if isinstance(stats, dict)), default=''),
            ),
            reverse=True,
        )
        self._symbol_regime_memory[label] = dict(ranked[:self.MAX_SYMBOLS_PER_AGENT])

    def _record_symbol_regime_observation(
        self,
        label: str,
        sym: str,
        regime: Optional[str],
        observation: dict,
    ):
        regime_key = self._canonical_symbol_regime(regime)
        if regime_key == 'unknown':
            return
        agent_symbols = self._symbol_regime_memory.setdefault(label, {})
        regime_map = agent_symbols.setdefault(str(sym), {})
        profile = {'symbol': str(sym), 'symbol_regime': regime_key}
        stats = regime_map.get(regime_key) or self._empty_symbol_stats(profile)
        regime_map[regime_key] = self._update_symbol_bucket(
            stats, observation['score'], observation['pnl_pct'], profile=profile,
        )
        self._prune_symbol_regime_memory(label)

    def _symbol_regime_memory_score(self, label: str, sym: str, regime: Optional[str]) -> float:
        regime_key = self._canonical_symbol_regime(regime)
        if regime_key == 'unknown':
            return 0.0
        stats = (
            self._symbol_regime_memory
            .get(label, {})
            .get(str(sym), {})
            .get(regime_key)
        )
        if not isinstance(stats, dict):
            return 0.0
        samples = int(stats.get('samples', 0) or 0)
        if samples <= 0:
            return 0.0
        confidence = min(samples / max(float(self.REGIME_MEMORY_CONFIDENCE_SAMPLES), 1.0), 1.0)
        ema_score = float(stats.get('ema_score', 0.0) or 0.0)
        avg_pnl = float(stats.get('avg_pnl', 0.0) or 0.0)
        wins = int(stats.get('wins', 0) or 0)
        losses = int(stats.get('losses', 0) or 0)
        balance = (wins - losses) / max(samples, 1)
        score = ema_score * 0.70 + avg_pnl * 0.18 + balance * 0.42
        return float(np.clip(score, -self.REGIME_SCORE_CLIP, self.REGIME_SCORE_CLIP)) * confidence

    def _current_symbol_regime_mix(self, max_symbols: int = 80) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for idx, (sym, hist) in enumerate(getattr(self, '_ph', {}).items()):
            if max_symbols > 0 and idx >= max_symbols:
                break
            regime = self._detect_symbol_regime(sym, list(hist or []))
            if regime == 'unknown':
                continue
            counts[regime] = counts.get(regime, 0) + 1
        total = sum(counts.values())
        if total <= 0:
            return {}
        return {
            regime: count / total
            for regime, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        }

    def _priority_agents_for_symbol_regimes(self, regime_mix: Optional[Dict[str, float]] = None):
        labels = []
        for regime, _share in sorted((regime_mix or {}).items(), key=lambda item: item[1], reverse=True):
            for label in self._priority_agents_for_regime(regime):
                if label not in labels:
                    labels.append(label)
        return tuple(labels)

    def _symbol_regime_static_prior_mix(self, label: str, regime_mix: Optional[Dict[str, float]]) -> float:
        scores = []
        weighted = 0.0
        for regime, share in (regime_mix or {}).items():
            score = self._regime_static_prior(label, regime)
            weighted += score * float(share)
            scores.append(score)
        if not scores:
            return 0.0
        return float(max(weighted, max(scores) * 0.65))

    def _symbol_regime_bonus_mix(
        self,
        label: str,
        regime_mix: Optional[Dict[str, float]],
        perf: Optional[dict] = None,
    ) -> float:
        scores = []
        weighted = 0.0
        for regime, share in (regime_mix or {}).items():
            score = self._regime_score_adjustment(label, regime, perf)
            weighted += score * float(share)
            scores.append(score)
        if not scores:
            return 0.0
        return float(max(weighted, max(scores) * 0.65))

    def _symbol_regime_memory_mix(self, label: str, regime_mix: Optional[Dict[str, float]]) -> float:
        if not regime_mix:
            return 0.0
        by_regime: Dict[str, list] = {regime: [] for regime in regime_mix}
        for sym, hist in getattr(self, '_ph', {}).items():
            regime = self._detect_symbol_regime(sym, list(hist or []))
            if regime not in by_regime:
                continue
            score = self._symbol_regime_memory_score(label, sym, regime)
            if score != 0.0:
                by_regime[regime].append(score)

        regime_scores = []
        weighted = 0.0
        for regime, share in regime_mix.items():
            general_score = self._regime_memory_score(label, regime)
            local_scores = sorted(by_regime.get(regime, ()), reverse=True)[:3]
            local_score = float(np.mean(local_scores)) if local_scores else 0.0
            score = (
                general_score * 0.55 + local_score * 0.45
                if local_scores else general_score
            )
            weighted += score * float(share)
            regime_scores.append(score)
        if not regime_scores:
            return 0.0
        return float(max(weighted, max(regime_scores) * 0.70))

    def _agent_symbol_regime_recovered(self, label: str) -> bool:
        regime_mix = self._current_symbol_regime_mix()
        if not regime_mix:
            return False
        for regime in regime_mix:
            regime_stats = self._regime_memory.get(regime, {}).get(label, {})
            regime_samples = int(regime_stats.get('samples', 0) or 0) if isinstance(regime_stats, dict) else 0
            if regime_samples >= 3 and self._regime_memory_score(label, regime) >= 0.80:
                return True
        for sym, hist in getattr(self, '_ph', {}).items():
            regime = self._detect_symbol_regime(sym, list(hist or []))
            if regime not in regime_mix:
                continue
            stats = self._symbol_regime_memory.get(label, {}).get(str(sym), {}).get(regime)
            samples = int(stats.get('samples', 0) or 0) if isinstance(stats, dict) else 0
            if samples >= 3 and self._symbol_regime_memory_score(label, sym, regime) >= 0.80:
                return True
        return False

    def _update_symbol_memory(self):
        self._meta_memory.update_symbol_memory(self)

    def _symbol_memory_score(self, label: str, sym: str,
                             profile: Optional[Dict[str, str]] = None,
                             exact_stats: Optional[dict] = None) -> tuple[float, float]:
        return self._meta_memory.symbol_memory_score(
            self, label, sym, profile=profile, exact_stats=exact_stats,
        )

    def _symbol_vote_weight(self, label: str, sym: str, base_weight: float) -> float:
        return self._meta_memory.symbol_vote_weight(self, label, sym, base_weight)

    def _agent_symbol_opportunity_score(self, label: str) -> float:
        return self._meta_memory.agent_symbol_opportunity_score(self, label)

    def _build_symbol_preferences_snapshot(self, top_n: int = 5) -> Dict[str, dict]:
        snapshot: Dict[str, dict] = {}
        for label, sym_map in sorted(self._symbol_memory.items()):
            if label not in self._agent_pool or not isinstance(sym_map, dict) or not sym_map:
                continue
            ranked = []
            for sym, stats in sym_map.items():
                if not isinstance(stats, dict):
                    continue
                profile = dict(stats.get('profile') or {})
                exact_score, factor_score = self._symbol_memory_score(
                    label, sym, profile=profile, exact_stats=stats,
                )
                combined = exact_score * 0.65 + factor_score * 0.35
                ranked.append({
                    'symbol': str(sym),
                    'score': round(float(combined), 6),
                    'ema_score': round(float(stats.get('ema_score', 0.0) or 0.0), 6),
                    'avg_pnl': round(float(stats.get('avg_pnl', 0.0) or 0.0), 6),
                    'samples': int(stats.get('samples', 0) or 0),
                    'wins': int(stats.get('wins', 0) or 0),
                    'losses': int(stats.get('losses', 0) or 0),
                    'profile': profile,
                })
            if not ranked:
                continue
            ranked.sort(key=lambda item: item['score'], reverse=True)
            snapshot[label] = {
                'best_symbols': ranked[:top_n],
                'worst_symbols': list(reversed(ranked[-top_n:])),
            }
        return snapshot

    SHADOW_MAP = {         # V_Name → internal label
        'V_Panteon_shadow': 'Panteon',
        'V_PanteonResearch': 'PanteonResearch',
        'V_FundingArb':      'FundingArb',
        'V_MomentumScalper': 'MomentumScalper',
        'V_LiveAfterShock':  'LiveAfterShock',
        'V_LiveCrashHunter': 'LiveCrashHunter',
        'V_LiveRegimePullback': 'LiveRegimePullback',
        'V_LiveMeanRev':     'LiveMeanRev',
        'V_LiveTrendFollow': 'LiveTrendFollow',
        'V_LiveVolCompress': 'LiveVolCompress',
        'V_LiveOIBreakout': 'LiveOIBreakout',
        'V_CarryFlowAgentV2': 'CarryFlowAgentV2',
        'V_ResearchValidatorAgent': 'ResearchValidatorAgent',
        'V_RichardDennisTurtle': 'RichardDennis',
        'V_GeneticsBullish': 'GeneticsBullish',
        'V_GeneticsBearish': 'GeneticsBearish',
        # Shadow candidate: новый агент (volatility breakout). Живые веса
        # получит через promotion_gate после набора достаточной статистики.
        'V_VolBreakoutHunter': 'VolBreakoutHunter',
    }

    def _normalize_weights(
        self,
        weights: Dict[str, float],
        priority_labels=None,
    ) -> Dict[str, float]:
        max_weight = float(getattr(self, 'MAX_WEIGHT', 1.0) or 1.0)
        base_floor = max(0.0, float(getattr(self, 'SOFT_MIN_WEIGHT', 0.0) or 0.0))
        required_agents = max(
            int(getattr(self, 'MIN_AGENTS', 1) or 1),
            int(np.ceil(1.0 / max(max_weight, 1e-6))),
        )
        filtered = OrderedDict()
        for label, weight in (weights or {}).items():
            if label not in self._agent_pool or label in self.LIVE_AGENT_BLOCKLIST:
                continue
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                continue
            if weight > 0:
                filtered[label] = weight
        if len(filtered) < required_agents:
            for label, seed_weight in getattr(self, '_active_weights', {}).items():
                if label in filtered or label not in self._agent_pool or label in self.LIVE_AGENT_BLOCKLIST:
                    continue
                filtered[label] = max(float(seed_weight or 0.0), base_floor or 0.01)
                if len(filtered) >= required_agents:
                    break
        total = sum(filtered.values())
        if total <= 0:
            return {}
        normalized = OrderedDict(
            (label, weight / total)
            for label, weight in filtered.items()
        )
        priority_floor = max(
            base_floor,
            float(getattr(self, 'REGIME_PRIORITY_FLOOR', base_floor) or base_floor),
        )
        priorities = {
            label for label in (priority_labels or ())
            if label in normalized
        }
        floor_map = OrderedDict(
            (label, priority_floor if label in priorities else base_floor)
            for label in normalized
        )
        floor_total = sum(floor_map.values())
        if floor_total >= 0.95 and floor_total > 0:
            scale = 0.95 / floor_total
            floor_map = OrderedDict(
                (label, floor * scale)
                for label, floor in floor_map.items()
            )

        weights_map = OrderedDict(
            (label, max(weight, floor_map[label]))
            for label, weight in normalized.items()
        )
        for _ in range(8):
            changed = False
            current_total = sum(weights_map.values())
            if current_total <= 0:
                break
            weights_map = OrderedDict(
                (label, weight / current_total)
                for label, weight in weights_map.items()
            )

            deficit = 0.0
            donors = []
            for label, floor in floor_map.items():
                value = weights_map[label]
                if value < floor:
                    deficit += floor - value
                    weights_map[label] = floor
                    changed = True
                elif value > floor:
                    donors.append(label)
            if deficit > 1e-9 and donors:
                spare_total = sum(
                    max(weights_map[label] - floor_map[label], 0.0)
                    for label in donors
                )
                if spare_total > 1e-9:
                    for label in donors:
                        spare = max(weights_map[label] - floor_map[label], 0.0)
                        weights_map[label] -= deficit * spare / spare_total

            excess = 0.0
            receivers = []
            for label, value in list(weights_map.items()):
                if value > max_weight:
                    excess += value - max_weight
                    weights_map[label] = max_weight
                    changed = True
                elif value < max_weight - 1e-9:
                    receivers.append(label)
            if excess > 1e-9 and receivers:
                room_total = sum(
                    max(max_weight - weights_map[label], 0.0)
                    for label in receivers
                )
                if room_total > 1e-9:
                    for label in receivers:
                        room = max(max_weight - weights_map[label], 0.0)
                        weights_map[label] += excess * room / room_total

            if not changed:
                break

        final_total = sum(max(weight, 0.0) for weight in weights_map.values())
        if final_total <= 0:
            return {}
        return {
            label: max(weight, 0.0) / final_total
            for label, weight in weights_map.items()
            if weight > 1e-8
        }

    def _priority_agents_for_regime(self, regime: str):
        regime_key = self._canonical_market_regime(regime)
        return tuple(
            label for label in self.REGIME_PRIORITY_MAP.get(regime_key, ())
            if label in self._agent_pool
        )

    def _regime_score_adjustment(self, label: str, regime: str, perf: Optional[dict] = None) -> float:
        regime_key = self._canonical_market_regime(regime)
        bonus = float(self.REGIME_SCORE_BONUS.get(regime_key, {}).get(label, 0.0))
        if label == 'FundingArb' and regime_key == 'bullish':
            pnl = float((perf or {}).get('pnl_pct', 0.0) or 0.0)
            if pnl < 0.5:
                bonus -= 0.10
        return bonus

    def _shadow_metrics_for_label(self, label: str) -> dict:
        shadow_name = getattr(self, '_label_to_shadow', {}).get(label)
        if not shadow_name or not isinstance(self._shadow_perf, dict):
            return {}
        perf = self._shadow_perf.get(shadow_name)
        return perf if isinstance(perf, dict) else {}

    def _agent_recent_signal_share(self, label: str) -> float:
        recent = getattr(self, '_recent_real_signal_agents', None)
        if not recent:
            return 0.0
        hits = 0
        total = 0
        for voters in recent:
            if not voters:
                continue
            total += 1
            if label in voters:
                hits += 1
        if total <= 0:
            return 0.0
        return hits / total

    def _record_recent_real_signal(self, voters):
        cleaned = tuple(
            sorted(
                {
                    str(v)
                    for v in (voters or [])
                    if v and v != '(limited)'
                }
            )
        )
        if cleaned:
            self._recent_real_signal_agents.append(cleaned)

    def _agent_currently_blocked(self, label: str, perf: Optional[dict] = None) -> bool:
        if label in self.LIVE_AGENT_BLOCKLIST:
            return True

        perf = perf if isinstance(perf, dict) else self._shadow_metrics_for_label(label)
        pnl = float((perf or {}).get('pnl_pct', 0.0) or 0.0)
        sharpe = float((perf or {}).get('sharpe', 0.0) or 0.0)
        closed = int((perf or {}).get('closed_trades', (perf or {}).get('total_trades', 0)) or 0)
        entries = int((perf or {}).get('entries', 0) or 0)

        if closed >= self.ROTATION_MIN_CLOSED_FOR_BLOCK:
            if pnl <= self.ROTATION_HARD_LOSS_PCT:
                return True
            if pnl < 0.0 and sharpe <= self.ROTATION_HARD_NEGATIVE_SHARPE:
                return True

        memory_score = float(self._agent_success_memory.get(label, 0.0) or 0.0)
        current_regime = self._canonical_market_regime(
            (getattr(self, '_current_context', {}) or {}).get('regime') or getattr(self, '_r', None)
        )
        regime_stats = self._regime_memory.get(current_regime, {}).get(label, {})
        regime_samples = int(regime_stats.get('samples', 0) or 0) if isinstance(regime_stats, dict) else 0
        regime_score = self._regime_memory_score(label, current_regime)
        regime_recovered = regime_samples >= 3 and regime_score >= 0.80
        symbol_regime_recovered = self._agent_symbol_regime_recovered(label)
        recovered = (
            closed >= self.MEMORY_RECOVERY_CLOSED_TRADES
            and pnl >= self.MEMORY_RECOVERY_PNL_PCT
            and sharpe >= 0.0
        )
        if memory_score <= self.MEMORY_HARD_NEGATIVE_SCORE and not (
            recovered or regime_recovered or symbol_regime_recovered
        ):
            return True

        if closed == 0 and entries >= self.UNPROVEN_SINGLE_MIN_ENTRIES and pnl < 0.0:
            return True

        return False

    def _effective_vote_weight(self, label: str, sym: str, base_weight: float, action: int) -> float:
        vote_w = float(base_weight or 0.0)
        if action in (1, 2, 4, 5, 6, 7):
            vote_w = self._symbol_vote_weight(label, sym, vote_w)

        perf = self._shadow_metrics_for_label(label)
        pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
        sharpe = float(perf.get('sharpe', 0.0) or 0.0)
        closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
        dominance = self._agent_recent_signal_share(label)

        mult = 1.0
        current_negative = closed >= 4 and (
            pnl <= -0.20 or (pnl < 0.0 and sharpe < 0.0)
        )
        if action in (1, 2, 4, 5, 6, 7):
            if closed >= 8 and (pnl <= -0.75 or sharpe <= -10.0):
                mult *= 0.35
            elif current_negative:
                mult *= 0.58
            elif closed >= 6 and pnl >= 0.60 and sharpe > 0.0:
                mult *= 1.08
        elif current_negative:
            mult *= 0.82

        if dominance > self.DOMINANCE_SOFT_SHARE and current_negative:
            severity = (dominance - self.DOMINANCE_SOFT_SHARE) / max(
                1e-6, 1.0 - self.DOMINANCE_SOFT_SHARE
            )
            mult *= max(0.35, 1.0 - 0.55 * severity)
        elif dominance > self.DOMINANCE_HARD_SHARE and pnl < 0.20:
            mult *= 0.80

        return max(vote_w * mult, 0.0)

    def _execution_threshold_offsets(self, portfolio_value=None) -> tuple[float, float]:
        open_extra = 0.0
        close_extra = 0.0
        try:
            pv = float(portfolio_value) if portfolio_value is not None else 0.0
        except (TypeError, ValueError):
            pv = 0.0
        if 0.0 < pv < self.SMALL_ACCOUNT_VALUE:
            open_extra += self.SMALL_ACCOUNT_OPEN_BONUS
            close_extra += self.SMALL_ACCOUNT_CLOSE_BONUS

        ctx = getattr(self, '_current_context', {}) or {}
        if str(ctx.get('noise', '') or '') == 'noisy':
            open_extra += self.NOISY_MARKET_OPEN_BONUS
            close_extra += self.NOISY_MARKET_CLOSE_BONUS

        active_labels = [
            label for label in getattr(self, '_active_weights', {})
            if label in getattr(self, '_agent_pool', {})
        ]
        weak_count = 0
        observed_count = 0
        for label in active_labels:
            perf = self._shadow_metrics_for_label(label)
            if not perf:
                continue
            closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
            if closed < 4:
                continue
            observed_count += 1
            pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
            sharpe = float(perf.get('sharpe', 0.0) or 0.0)
            if pnl < -0.15 or (pnl < 0.0 and sharpe < 0.0):
                weak_count += 1
        if observed_count >= 2 and weak_count >= max(1, observed_count // 2):
            open_extra += self.WEAK_LINEUP_OPEN_BONUS
            close_extra += self.WEAK_LINEUP_CLOSE_BONUS

        return float(open_extra), float(close_extra)

    def _single_agent_open_allowed(
        self,
        label: str,
        agreement_weight: float,
        min_open_threshold: Optional[float] = None,
    ) -> bool:
        perf = self._shadow_metrics_for_label(label)
        base_open_threshold = (
            float(min_open_threshold)
            if min_open_threshold is not None
            else float(self.OPEN_SINGLE_THRESHOLD)
        )
        strong_threshold = min(
            float(self.MAX_WEIGHT),
            max(float(self.SINGLE_AGENT_STRONG_THRESHOLD), base_open_threshold),
        )
        if not perf:
            return float(agreement_weight or 0.0) >= strong_threshold

        pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
        sharpe = float(perf.get('sharpe', 0.0) or 0.0)
        max_dd = abs(float(perf.get('max_dd', perf.get('max_drawdown_pct', perf.get('max_dd_pct', 0.0))) or 0.0))
        closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
        entries = int(perf.get('entries', 0) or 0)
        agreement_weight = float(agreement_weight or 0.0)

        if self._agent_currently_blocked(label, perf):
            return False

        if closed < self.SINGLE_AGENT_MIN_CLOSED_TRADES:
            required = max(
                strong_threshold,
                min(float(self.MAX_WEIGHT), float(self.UNPROVEN_SINGLE_THRESHOLD)),
            )
            if (
                entries >= self.UNPROVEN_SINGLE_MIN_ENTRIES
                and pnl >= self.UNPROVEN_SINGLE_MIN_PNL_PCT
                and sharpe >= 0.0
            ):
                required = max(strong_threshold, required - 0.04)
            if float(self._agent_success_memory.get(label, 0.0) or 0.0) > 5.0:
                required = max(strong_threshold, required - 0.03)
            return agreement_weight >= required
        if pnl < self.SINGLE_AGENT_MIN_PNL_PCT:
            return False
        if sharpe < self.SINGLE_AGENT_MIN_SHARPE:
            return False
        if max_dd > self.SINGLE_AGENT_MAX_DD_PCT and pnl < 0.35:
            return False
        return True

    def _signal_risk_multiplier(
        self,
        sym: str,
        action: int,
        agreement_score: float,
        voters,
    ) -> float:
        if action not in (1, 2, 4, 5, 6, 7):
            return 1.0

        agreement = float(agreement_score or 0.0)
        active_voters = []
        has_limited = False
        for voter in (voters or []):
            name = str(voter or '')
            if not name:
                continue
            if name == '(limited)':
                has_limited = True
                continue
            active_voters.append(name)

        support_n = len(set(active_voters))
        regime = self._canonical_market_regime(
            getattr(self, '_r', None) or getattr(self, '_last_regime', '') or 'neutral'
        )
        ctx = getattr(self, '_current_context', {}) or {}
        noise = str(ctx.get('noise', '') or '')
        trend_alignment = str(ctx.get('trend_alignment', '') or '')

        mult = 0.82
        if agreement >= 0.60:
            mult += 0.28
        elif agreement >= 0.45:
            mult += 0.16
        elif agreement >= 0.33:
            mult += 0.06
        elif agreement < 0.26:
            mult -= 0.12
        else:
            mult -= 0.06

        if support_n >= 3:
            mult += 0.12
        elif support_n == 2:
            mult += 0.06
        elif support_n == 0 and has_limited:
            mult -= 0.10

        shadow_edges = []
        dominance = 0.0
        for label in active_voters:
            perf = self._shadow_metrics_for_label(label)
            pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
            sharpe = float(perf.get('sharpe', 0.0) or 0.0)
            closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)

            edge = 0.0
            if closed >= 8 and (pnl <= -0.75 or sharpe <= -10.0):
                edge = -0.10
            elif closed >= 4 and (pnl < 0.0 or sharpe < 0.0):
                edge = -0.06
            elif closed >= 6 and pnl >= 0.75 and sharpe > 0.0:
                edge = 0.08
            elif closed >= 4 and pnl > 0.0 and sharpe >= 0.0:
                edge = 0.04
            shadow_edges.append(edge)
            dominance = max(dominance, self._agent_recent_signal_share(label))

        if shadow_edges:
            mult += sum(shadow_edges) / len(shadow_edges)

        if support_n <= 1 and dominance > self.DOMINANCE_SOFT_SHARE:
            severity = (dominance - self.DOMINANCE_SOFT_SHARE) / max(
                1e-6, 1.0 - self.DOMINANCE_SOFT_SHARE
            )
            mult -= 0.10 * min(1.0, severity)

        if noise == 'noisy':
            mult -= 0.08
        elif noise == 'mixed':
            mult -= 0.04
        elif noise == 'quiet':
            mult += 0.02

        if trend_alignment == 'aligned':
            mult += 0.04
        elif trend_alignment == 'counter':
            mult -= 0.04

        is_long = action in (1, 2, 4, 5)
        is_short = action in (6, 7)
        if (regime == 'bullish' and is_long) or (regime == 'bearish' and is_short):
            mult += 0.04
        elif regime in {'bullish', 'bearish'}:
            mult -= 0.06

        if has_limited and support_n <= 1:
            mult = min(mult, 0.78)

        return float(min(self.RISK_MULTIPLIER_MAX, max(self.RISK_MULTIPLIER_MIN, mult)))

    def enable_real_memory(self, load_existing: bool = True):
        self._memory_store.enable(self, load_existing=load_existing)

    def set_memory_namespace(self, namespace: Optional[str], load_existing: bool = False):
        clean = _sanitize_memory_namespace(namespace)
        if clean == self._memory_namespace:
            return
        memory_file, legacy_file = _memory_paths_for_namespace(clean)
        self._memory_namespace = clean
        self._memory_store = MemorySnapshotStore(
            memory_file,
            legacy_file,
            save_interval_bars=30,
        )
        self._last_memory_save_bar = -99999
        if self._memory_enabled and load_existing:
            self._memory_store.load_snapshot(self, logger=log)

    def _load_memory_snapshot(self):
        self._memory_store.load_snapshot(self, logger=log)

    def set_shadow_perf(self, perf: Dict[str, dict]):
        """Вызывается из COMBO_TRADE для передачи shadow performance."""
        self._shadow_scoring.set_shadow_perf(self, perf)

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        return self._shadow_scoring.score_shadow_candidate(self, label, perf)

    def _success_signal(self, perf: dict) -> float:
        return self._shadow_scoring.success_signal(self, perf)

    def _update_success_memory(self, label: str, perf: dict) -> float:
        return self._shadow_scoring.update_success_memory(self, label, perf)

    def _build_memory_payload(self) -> dict:
        return self._memory_store.build_payload(self)

    def save_memory_snapshot(self, force: bool = False, reason: str = ""):
        self._memory_store.save_snapshot(self, force=force, reason=reason, logger=log)

    def _rotate_agents(self):
        self._portfolio_allocator.rotate(self, logger=log)

    def reset_for_live(self, bar_index: int = 0):
        """
        Сброс всех суб-агентов для чистого старта live-торговли.
        Вызывается ПОСЛЕ warmup, ПЕРЕД live.

        Сбрасывает:
          - self.pos (позиции из warmup — не существуют в реальности)
          - self.ep / entry_px (entry prices от warmup)
          - self._lc (таймер — чтобы первый check сработал сразу)
          - self._open_pos (трекер Panteon)
          - self._r (режим — пересчитается)
        """
        for attr_name, label, agent in self.iter_subagents():
            # Таймер: немедленный первый check
            if hasattr(agent, '_lc'):
                try: agent._lc = -99999
                except (AttributeError, TypeError): pass
            try: agent.t = bar_index
            except (AttributeError, TypeError): pass
            # Позиции: сброс
            pos_d = getattr(agent, 'pos', None)
            if isinstance(pos_d, dict):
                n_held = sum(1 for v in pos_d.values() if v is not None)
                for k in pos_d:
                    pos_d[k] = None
                if n_held:
                    log.info("    [reset] %s: %d warmup-позиций очищено", label, n_held)
        # Цены входа
            for ep_attr in ('ep', 'entry_px'):
                d = getattr(agent, ep_attr, None)
                if isinstance(d, dict):
                    for k in d:
                        d[k] = 0.0
        # Время входа
            et_d = getattr(agent, 'et', None)
            if isinstance(et_d, dict):
                for k in et_d:
                    et_d[k] = 0

        # Сброс Panteon state
        self._open_pos.clear()
        self._r = None
        self._lr = 0
        self._shadow_window_anchor = {}
        self._shadow_symbol_anchor = {}
        self._current_context = {}
        self._last_context_bar = -99999
        if hasattr(self, '_sub_signal_counts'):
            self._sub_signal_counts.clear()
        if hasattr(self, '_recent_real_signal_agents'):
            self._recent_real_signal_agents.clear()
        if hasattr(self, '_last_risk_multipliers'):
            self._last_risk_multipliers.clear()
        log.info("  ✅ Panteon.reset_for_live(): все sub-agents сброшены (bar=%d)", bar_index)

    def act(self,prices,volumes,month=None,portfolio_value=None,bar_index=None):
        self._t = bar_index if bar_index is not None else getattr(self,'_t',0)+1
        t = self._t
        # Обновляем историю цен
        for s,p in prices.items():
            self._ph.setdefault(s, deque(maxlen=5760)).append(float(p))
            self._vh.setdefault(s, deque(maxlen=5760)).append(float(volumes.get(s, 0.0) or 0.0))

        # FIX v8: Режим пересчитывается чаще (каждый час) и использует
        # короткие окна (4h/12h) вместо 7d/3d. Старый _detect_regime_live
        # застревал в bearish на дни после одного падения.
        if t-self._lr>=self.STATE_INT or self._r is None:
            self._r = self._canonical_market_regime(self._detect_regime_fast())
            self._lr=t

        if t - self._last_context_bar >= self.CONTEXT_REFRESH_INT or not self._current_context:
            self._current_context = self._build_market_context(month=month)
            self._last_context_bar = t

        regime=self._canonical_market_regime(self._r or 'neutral')
        kw=dict(volumes=volumes,month=month,portfolio_value=portfolio_value,bar_index=bar_index)

        # ═══════════════════════════════════════════════════════════════
        # ADAPTIVE ROTATION: периодически обновляем состав агентов
        # ═══════════════════════════════════════════════════════════════
        if t - self._last_rotation >= self.ROTATION_INT:
            self._rotate_agents()
            self._last_rotation = t

        # Строим acts_list из _active_weights (обновляются ротацией)
        acts_list = []
        for label, weight in sorted(self._active_weights.items(),
                                     key=lambda x: x[1], reverse=True):
            agent = self._agent_pool.get(label)
            if agent is None:
                continue
            if self._agent_currently_blocked(label):
                continue
            try:
                acts = agent.act(prices, **kw)
            except Exception:
                acts = {}
            acts_list.append((label, acts, weight))

        # Взвешенное голосование: собираем голоса и отмечаем, кто их подал
        buy_w={};sell_w={};long_w={};short_w_map={}
        long_support={};short_support={};sell_support={}
        total_active_weight = max(
            sum(float(weight) for weight in self._active_weights.values()),
            1e-9,
        )
        open_extra, close_extra = self._execution_threshold_offsets(portfolio_value)
        open_single_threshold = min(self.MAX_WEIGHT, self.OPEN_SINGLE_THRESHOLD + open_extra)
        open_multi_threshold = min(self.MAX_WEIGHT, self.OPEN_MULTI_THRESHOLD + open_extra * 0.75)
        single_strong_threshold = min(
            self.MAX_WEIGHT,
            max(self.SINGLE_AGENT_STRONG_THRESHOLD, open_single_threshold),
        )
        close_single_threshold = min(self.MAX_WEIGHT, self.CLOSE_SINGLE_THRESHOLD + close_extra)
        close_multi_threshold = min(self.MAX_WEIGHT, self.CLOSE_MULTI_THRESHOLD + close_extra * 0.75)
        close_strong_threshold = min(self.MAX_WEIGHT, self.CLOSE_STRONG_THRESHOLD + close_extra)
        # {sym: list[agent_name]} — финальные контрибьюторы сигнала
        _contributors: dict = {}
        _agreement_scores: dict = {}
        # FIX v4: _named zip порядок теперь совпадает с acts_list
        for ag_name, acts, w in acts_list:
            if acts is None: continue
            for s, a in acts.items():
                if a == 0: continue
                vote_w = self._effective_vote_weight(ag_name, s, w, a)
                if a in(1,2):
                    buy_w[s]=buy_w.get(s,0)+vote_w; long_w[s]=long_w.get(s,0)+vote_w
                    long_support.setdefault(s, set()).add(ag_name)
                    _contributors.setdefault(s, {})[ag_name] = vote_w
                elif a in(4,5):
                    buy_w[s]=buy_w.get(s,0)+vote_w; long_w[s]=long_w.get(s,0)+vote_w
                    long_support.setdefault(s, set()).add(ag_name)
                    _contributors.setdefault(s, {})[ag_name] = vote_w
                elif a in(6,7):
                    short_w_map[s]=short_w_map.get(s,0)+vote_w
                    short_support.setdefault(s, set()).add(ag_name)
                    _contributors.setdefault(s, {})[ag_name] = vote_w
                elif a in(3,8):
                    sell_w[s]=sell_w.get(s,0)+vote_w
                    sell_support.setdefault(s, set()).add(ag_name)
                    _contributors.setdefault(s, {})[ag_name] = vote_w

        # FIX v6: строим финальные действия с лимитом на новые позиции за бар
        out={s:0 for s in prices}
        n_open = len(self._open_pos)

        # Шаг 1: CLOSE-сигналы — только для реально известных открытых позиций.
        # Иначе суб-агенты со stale-состоянием спамят close_all по символам,
        # которых в live-портфеле уже нет.
        close_candidates = []
        for s in set(sell_w):
            if s not in self._open_pos:
                continue
            support_n = len(sell_support.get(s, ()))
            close_w = sell_w.get(s, 0.0)
            thresh = (
                close_multi_threshold
                if support_n >= self.MIN_MULTI_AGENT_SUPPORT
                else close_single_threshold
            )
            if close_w < thresh:
                continue
            if support_n < self.MIN_MULTI_AGENT_SUPPORT and close_w < close_strong_threshold:
                continue
            close_candidates.append((s, close_w))

        close_candidates.sort(key=lambda item: item[1], reverse=True)
        for s, close_w in close_candidates[:self.MAX_DISCRETIONARY_CLOSES_PER_BAR]:
            out[s]=8
            _agreement_scores[s] = min(1.0, close_w / total_active_weight)

        # Шаг 2: OPEN-сигналы — сортируем по силе, берём только лучшие
        # MAX_NEW_PER_BAR: не более 1 новой позиции за бар
        # $128 аккаунт с $30 free margin → нельзя открывать 2 позиции за раз
        MAX_NEW_PER_BAR = 1
        open_candidates = []  # [(sym, action, weight, support_n)]
        for s in set(buy_w)|set(short_w_map):
            if out[s] == 8:
                continue  # уже close
            if s in self._open_pos:
                continue  # уже есть позиция
            sw = short_w_map.get(s, 0)
            lw = long_w.get(s, 0)
            short_support_n = len(short_support.get(s, ()))
            long_support_n = len(long_support.get(s, ()))
            if sw > lw:
                short_labels = tuple(
                    sorted(
                        short_support.get(s, ()),
                        key=lambda ag: _contributors.get(s, {}).get(ag, 0.0),
                        reverse=True,
                    )
                )
                lead_short = short_labels[0] if short_labels else ''
                thresh = (
                    open_multi_threshold
                    if short_support_n >= self.MIN_MULTI_AGENT_SUPPORT
                    else open_single_threshold
                )
                if sw >= thresh:
                    if short_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                        if sw < single_strong_threshold:
                            continue
                        if lead_short and not self._single_agent_open_allowed(lead_short, sw, open_single_threshold):
                            continue
                    open_candidates.append((s, 6, sw, short_support_n))
            else:
                long_labels = tuple(
                    sorted(
                        long_support.get(s, ()),
                        key=lambda ag: _contributors.get(s, {}).get(ag, 0.0),
                        reverse=True,
                    )
                )
                lead_long = long_labels[0] if long_labels else ''
                thresh = (
                    open_multi_threshold
                    if long_support_n >= self.MIN_MULTI_AGENT_SUPPORT
                    else open_single_threshold
                )
                if lw >= thresh:
                    if long_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                        if lw < single_strong_threshold:
                            continue
                        if lead_long and not self._single_agent_open_allowed(lead_long, lw, open_single_threshold):
                            continue
                    open_candidates.append((s, 4, lw, long_support_n))

        # Сортируем по весу (самые сильные сигналы сначала)
        open_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)

        n_new = 0
        for s, action, w, _support_n in open_candidates:
            if n_new >= MAX_NEW_PER_BAR:
                break
            if n_open >= self.MAX_POS:
                break
            out[s] = action
            n_open += 1
            n_new += 1
            _contributors.setdefault(s, {})['(limited)'] = w
            _agreement_scores[s] = min(1.0, w / total_active_weight)

        # ══════════════════════════════════════════════════════════════════
        # POSITION MANAGER v5: SL + TP + Trailing Stop + Stale Exit
        # Управляет ВСЕМИ позициями включая внешние (инъектированные)
        # ══════════════════════════════════════════════════════════════════
        self._position_safety.apply_exit_actions(
            self._open_pos,
            prices,
            t,
            out,
            stop_loss_pct=self.SL_PCT,
            take_profit_pct=self.TP_PCT,
            trail_pct=self.TRAIL_PCT,
            stale_bars=self.STALE_BARS,
            logger=log,
        )
        self._position_safety.sync_positions(self._open_pos, out, prices, t)
        # Legacy loop intentionally disabled after phase-2 governor extraction.
        for s, info in []:
            cur = prices.get(s, 0)
            ep  = info.get('entry', 0)
            side = info.get('side', 'long')
            bar_opened = info.get('bar', 0)
            peak = info.get('peak', ep)  # лучшая цена за время жизни позиции

            # Если entry=0 — пробуем заполнить текущей ценой
            if ep == 0 and cur > 0:
                info['entry'] = cur
                info['peak'] = cur
                ep = cur
                log.warning("  [Panteon] %s entry=0 → установлен %.4f", s, cur)
                continue

            if cur <= 0 or ep <= 0:
                continue

            # Движение цены
            if side == 'long':
                move = cur / ep - 1
                # Обновляем peak
                if cur > peak:
                    info['peak'] = cur
                    peak = cur
                retreat = 1 - cur / peak if peak > 0 else 0
            else:  # short
                move = 1 - cur / ep
                if cur < peak:  # для short "peak" = минимум цены
                    info['peak'] = cur
                    peak = cur
                retreat = cur / peak - 1 if peak > 0 else 0

            close_reason = None

            # 1. СТОП-ЛОСС: move < -SL_PCT
            if move < -self.SL_PCT:
                close_reason = f"SL({move*100:+.1f}%)"

            # 2. ТЕЙК-ПРОФИТ: move > +TP_PCT
            elif move > self.TP_PCT:
                close_reason = f"TP({move*100:+.1f}%)"

            # 3. TRAILING STOP: позиция была в плюсе, откатилась > TRAIL_PCT
            elif move > 0.01 and retreat > self.TRAIL_PCT:
                close_reason = f"TRAIL(peak_move={((peak/ep-1) if side=='long' else (1-peak/ep))*100:+.1f}% now={move*100:+.1f}%)"

            # 4. STALE EXIT: позиция > STALE_BARS без значимого движения
            elif t - bar_opened > self.STALE_BARS and abs(move) < 0.02:
                close_reason = f"STALE({t-bar_opened}bars, move={move*100:+.1f}%)"

            if close_reason:
                log.warning(
                    "  [Panteon PM] 🔴 CLOSE %s %s  entry=%.4f  cur=%.4f  %s",
                    s, side.upper(), ep, cur, close_reason)
                out[s] = 8
                del self._open_pos[s]
                n_open -= 1

        # Трекинг: обновляем _open_pos по финальным действиям
        for s,a in out.items():
            if a in (4,5,1,2) and s not in self._open_pos:
                px = prices.get(s,0)
                self._open_pos[s] = {'entry': px, 'side': 'long', 'bar': t, 'peak': px}
                n_open += 1
            elif a in (6,7) and s not in self._open_pos:
                px = prices.get(s,0)
                self._open_pos[s] = {'entry': px, 'side': 'short', 'bar': t, 'peak': px}
                n_open += 1
            elif a in (3,8) and s in self._open_pos:
                del self._open_pos[s]; n_open -= 1

        # Сохраняем метаданные сигналов для SignalCapturingAgent:
        #   _last_contributors  = {sym: "AgentA+AgentB"} — кто голосовал
        #   _sub_signal_counts  = {agent_name: total_signals} — накопленный счётчик
        self._last_contributors: dict = {}
        self._last_agreement_scores: dict = {}
        self._last_risk_multipliers: dict = {}
        if not hasattr(self, '_sub_signal_counts'):
            self._sub_signal_counts: dict = {}
        for s, a in out.items():
            if a == 0:
                continue
            voters = []
            if s in _contributors:
                voters = sorted(_contributors[s].keys(),
                                key=lambda k: _contributors[s][k], reverse=True)
                self._last_contributors[s] = '+'.join(voters)
                self._last_agreement_scores[s] = float(_agreement_scores.get(s, 0.0))
            else:
                self._last_agreement_scores[s] = 0.0
            self._last_risk_multipliers[s] = self._signal_risk_multiplier(
                s, a, self._last_agreement_scores.get(s, 0.0), voters,
            )
            if voters:
                self._record_recent_real_signal(voters)
                for ag in voters:
                    if ag == '(limited)':
                        continue
                    self._sub_signal_counts[ag] = self._sub_signal_counts.get(ag, 0) + 1

        # Сохраняем текущий режим для контекста сигналов
        self._last_regime = regime

        # FIX v7: убираем сигналы для blacklisted символов
        for s in list(out.keys()):
            if s in self._BLACKLIST and out[s] != 0:
                out[s] = 0

        if any(a != 0 for a in out.values()):
            self.save_memory_snapshot(reason="real-signal")

        return out


# ══════════════════════════════════════════════════════════════════
# NEURO_PLAYER — управляет всеми вариациями генетических агентов
# ══════════════════════════════════════════════════════════════════

class PanteonResearch(Panteon):
    """
    Research ensemble for shadow-first experimentation.

    It gives startup weight to the new pullback agent and ranks
    shadow candidates by a more robust score than raw P&L alone.
    """
    ROTATION_INT = 60
    MIN_AGENTS = 3
    MAX_AGENTS = 5
    MIN_WEIGHT = 0.12

    def __init__(self):
        super().__init__()
        if self._memory_bootstrap_loaded:
            return
        desired = OrderedDict([
            ('LiveRegimePullback', 0.30),
            ('LiveMeanRev',        0.24),
            ('RichardDennis',      0.18),
            ('GeneticsBullish',    0.14),
            ('GeneticsBearish',    0.14),
            ('LiveCrashHunter',    0.10),
            ('LiveAfterShock',     0.08),
        ])
        filtered = OrderedDict(
            (label, weight) for label, weight in desired.items()
            if label in self._agent_pool
        )
        total = sum(filtered.values())
        if total > 0:
            self._active_weights = {label: weight / total for label, weight in filtered.items()}

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        pnl = float(perf.get('pnl_pct', 0.0))
        sharpe = float(perf.get('sharpe', 0.0))
        max_dd = abs(float(perf.get('max_dd', perf.get('max_drawdown_pct', perf.get('max_dd_pct', 0.0))) or 0.0))
        signals = int(perf.get('signals', 0) or 0)
        entries = int(perf.get('entries', 0) or 0)
        closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
        activity = min(signals, 40) * 0.03 + min(entries, 20) * 0.08 + min(closed, 12) * 0.06
        inactivity_penalty = 1.5 if signals == 0 and entries == 0 else 0.0
        return pnl + sharpe * 1.25 + activity - max_dd * 0.35 - inactivity_penalty


class _PanteonShadowVariant(PanteonResearch):
    """
    Base class for shadow-only Panteon variants.

    Each variant keeps the live-compatible API but starts from a distinct
    bootstrap mix of sub-agents so the shadow leaderboard can compare
    alternative trading styles side by side.
    """

    BOOTSTRAP_WEIGHTS = ()
    SCORE_BONUS_LABELS = frozenset()
    SCORE_BONUS = 0.0
    MAX_ACTIVITY_SIGNALS = 999
    MAX_ACTIVITY_ENTRIES = 999
    ACTIVITY_PENALTY_PER_SIGNAL = 0.0
    ACTIVITY_PENALTY_PER_ENTRY = 0.0
    DRAWDOWN_PENALTY_MULT = 0.35

    def __init__(self):
        super().__init__()
        if self._memory_bootstrap_loaded:
            return
        desired = OrderedDict(self.BOOTSTRAP_WEIGHTS)
        filtered = OrderedDict(
            (label, weight) for label, weight in desired.items()
            if label in self._agent_pool
        )
        total = sum(filtered.values())
        if total > 0:
            self._active_weights = {
                label: weight / total for label, weight in filtered.items()
            }

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
        sharpe = float(perf.get('sharpe', 0.0) or 0.0)
        max_dd = abs(float(perf.get('max_dd', perf.get('max_drawdown_pct', perf.get('max_dd_pct', 0.0))) or 0.0))
        signals = int(perf.get('signals', 0) or 0)
        entries = int(perf.get('entries', 0) or 0)
        closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
        activity = min(signals, 40) * 0.03 + min(entries, 20) * 0.08 + min(closed, 12) * 0.06
        inactivity_penalty = 1.5 if signals == 0 and entries == 0 else 0.0
        score = pnl + sharpe * 1.25 + activity - max_dd * self.DRAWDOWN_PENALTY_MULT - inactivity_penalty
        if label in self.SCORE_BONUS_LABELS and signals > 0:
            score += float(self.SCORE_BONUS)
        if signals > self.MAX_ACTIVITY_SIGNALS:
            score -= (signals - self.MAX_ACTIVITY_SIGNALS) * self.ACTIVITY_PENALTY_PER_SIGNAL
        if entries > self.MAX_ACTIVITY_ENTRIES:
            score -= (entries - self.MAX_ACTIVITY_ENTRIES) * self.ACTIVITY_PENALTY_PER_ENTRY
        return score


class PanteonTrendResearch(_PanteonShadowVariant):
    ROTATION_INT = 45
    MIN_AGENTS = 2
    MAX_AGENTS = 4
    MIN_WEIGHT = 0.14
    MAX_POS = 5
    OPEN_SINGLE_THRESHOLD = 0.32
    OPEN_MULTI_THRESHOLD = 0.26
    CLOSE_SINGLE_THRESHOLD = 0.28
    CLOSE_MULTI_THRESHOLD = 0.23
    BOOTSTRAP_WEIGHTS = (
        ('RichardDennis', 0.28),
        ('LiveTrendFollow', 0.24),
        ('MomentumScalper', 0.16),
        ('FundingArb', 0.12),
        ('LiveAfterShock', 0.10),
        ('LiveCrashHunter', 0.10),
    )
    SCORE_BONUS_LABELS = frozenset({
        'RichardDennis', 'LiveTrendFollow', 'MomentumScalper',
    })
    SCORE_BONUS = 0.12


class PanteonMeanRevResearch(_PanteonShadowVariant):
    ROTATION_INT = 50
    MIN_AGENTS = 3
    MAX_AGENTS = 5
    MIN_WEIGHT = 0.11
    MAX_POS = 4
    OPEN_SINGLE_THRESHOLD = 0.31
    OPEN_MULTI_THRESHOLD = 0.25
    CLOSE_SINGLE_THRESHOLD = 0.27
    CLOSE_MULTI_THRESHOLD = 0.22
    BOOTSTRAP_WEIGHTS = (
        ('LiveMeanRev', 0.30),
        ('LiveVolCompress', 0.24),
        ('LiveRegimePullback', 0.18),
        ('FundingArb', 0.12),
        ('LiveCrashHunter', 0.10),
        ('RichardDennis', 0.06),
    )
    SCORE_BONUS_LABELS = frozenset({
        'LiveMeanRev', 'LiveVolCompress', 'LiveRegimePullback',
    })
    SCORE_BONUS = 0.14
    DRAWDOWN_PENALTY_MULT = 0.40


class PanteonDefensiveResearch(_PanteonShadowVariant):
    ROTATION_INT = 60
    MIN_AGENTS = 2
    MAX_AGENTS = 4
    MIN_WEIGHT = 0.15
    MAX_POS = 3
    OPEN_SINGLE_THRESHOLD = 0.34
    OPEN_MULTI_THRESHOLD = 0.28
    CLOSE_SINGLE_THRESHOLD = 0.25
    CLOSE_MULTI_THRESHOLD = 0.20
    CLOSE_STRONG_THRESHOLD = 0.30
    MAX_DISCRETIONARY_CLOSES_PER_BAR = 1
    BOOTSTRAP_WEIGHTS = (
        ('FundingArb', 0.30),
        ('LiveCrashHunter', 0.26),
        ('LiveRegimePullback', 0.18),
        ('LiveMeanRev', 0.12),
        ('RichardDennis', 0.08),
        ('GeneticsBearish', 0.06),
    )
    SCORE_BONUS_LABELS = frozenset({
        'FundingArb', 'LiveCrashHunter', 'LiveRegimePullback',
    })
    SCORE_BONUS = 0.10
    DRAWDOWN_PENALTY_MULT = 0.52


class PanteonConsensusResearch(_PanteonShadowVariant):
    ROTATION_INT = 45
    MIN_AGENTS = 3
    MAX_AGENTS = 4
    MIN_WEIGHT = 0.14
    MAX_POS = 4
    OPEN_SINGLE_THRESHOLD = 0.35
    OPEN_MULTI_THRESHOLD = 0.28
    CLOSE_SINGLE_THRESHOLD = 0.30
    CLOSE_MULTI_THRESHOLD = 0.24
    MIN_MULTI_AGENT_SUPPORT = 3
    MAX_DISCRETIONARY_CLOSES_PER_BAR = 1
    BOOTSTRAP_WEIGHTS = (
        ('LiveCrashHunter', 0.22),
        ('FundingArb', 0.20),
        ('LiveMeanRev', 0.18),
        ('LiveRegimePullback', 0.16),
        ('LiveTrendFollow', 0.12),
        ('RichardDennis', 0.12),
    )
    MAX_ACTIVITY_SIGNALS = 55
    MAX_ACTIVITY_ENTRIES = 28
    ACTIVITY_PENALTY_PER_SIGNAL = 0.015
    ACTIVITY_PENALTY_PER_ENTRY = 0.03

PanteonPlayer = Panteon


class NeuroPlayer:
    """
    Neuro_Player v1: оркестратор ВСЕХ генетических агентов.

    Управляет 4-мя нейросетевыми агентами обученными на разных режимах:
      • GeneticsAgent    — универсальный (обучен на всех периодах)
      • GeneticsBullish  — специалист бычьего рынка
      • GeneticsBearish  — специалист медвежьего рынка (шортит)
      • GeneticsNeutral  — специалист бокового рынка (range-bound)

    Логика:
      1. Детектирует текущий режим рынка (bull/bear/sideways)
      2. Присваивает веса агентам в зависимости от режима:
         Bull:     GenBullish=0.40, GenAgent=0.30, GenNeutral=0.20, GenBearish=0.10
         Bear:     GenBearish=0.40, GenAgent=0.30, GenNeutral=0.20, GenBullish=0.10
         Sideways: GenNeutral=0.35, GenAgent=0.30, GenBullish=0.20, GenBearish=0.15
      3. Взвешенное голосование: buy при sum_w>=0.30, sell при sum_w>0.15

    Преимущества перед отдельными генетиками:
      - Плавная ротация вместо жёсткого переключения (меньше whipsaw)
      - Все 4 генетики всегда активны, но с разными весами
      - Стоп-лосс на уровне игрока: -8% от входа → принудительное закрытие
    """
    REGIME_INT = 2 * 60     # проверка режима каждые 2 часа
    BUY_THRESH = 0.30       # минимальный вес для входа
    SELL_THRESH = 0.15      # минимальный вес для выхода
    STOP_PCT = 0.08         # стоп-лосс 8% от входа
    MAX_POS = 5             # максимум открытых позиций

    # Веса по режимам: (GenAgent, GenBullish, GenBearish, GenNeutral)
    W_BULL     = (0.30, 0.40, 0.10, 0.20)
    W_BEAR     = (0.30, 0.10, 0.40, 0.20)
    W_SIDEWAYS = (0.30, 0.20, 0.15, 0.35)

    def __init__(self):
        self._agents = OrderedDict()
        self._agent_loaded = False
        self._load_agents()
        self._ph: Dict[str, deque] = {}
        self._regime = 'sideways'
        self._lr = -9999
        self.t = 0
        # Трекинг позиций для стоп-лосса
        self._pos: Dict[str, dict] = {}  # sym → {side, entry_px, agent}

    def _load_agents(self):
        """Загружает все вариации генетических агентов."""
        # FIX: правильные имена классов: GeneticsBullishAgent (не GeneticsBullish)
        # FIX: оборачиваем в _GeneticsAdapter для hourly gate
        # Исправление: запасной путь создавал _GeneticsAdapter(string) и падал на каждом .act()
        try:
            from crypto_genetics import (GeneticsAgent,
                                          GeneticsBullishAgent, GeneticsBearishAgent,
                                          GeneticsNeutralAgent)
            from crypto_agents import _GeneticsAdapter
            self._agents['GenAgent']   = GeneticsSymbolGuard(_GeneticsAdapter(GeneticsAgent()))
            self._agents['GenBullish'] = GeneticsSymbolGuard(_GeneticsAdapter(GeneticsBullishAgent()))
            self._agents['GenBearish'] = GeneticsSymbolGuard(_GeneticsAdapter(GeneticsBearishAgent()))
            self._agents['GenNeutral'] = GeneticsSymbolGuard(_GeneticsAdapter(GeneticsNeutralAgent()))
            self._agent_loaded = True
            return
        except Exception:
            pass

                # Запасной вариант: загружаем агентов по одному
        try:
            from crypto_agents import _GeneticsAdapter
            from crypto_genetics import GeneticsAgent
            self._agents['GenAgent'] = GeneticsSymbolGuard(_GeneticsAdapter(GeneticsAgent()))
            # Пробуем островные агенты
            import crypto_genetics as _cg
            for cls_name, key in [('GeneticsBullishAgent', 'GenBullish'),
                                   ('GeneticsBearishAgent', 'GenBearish'),
                                   ('GeneticsNeutralAgent', 'GenNeutral')]:
                try:
                    cls = getattr(_cg, cls_name)
                    self._agents[key] = GeneticsSymbolGuard(_GeneticsAdapter(cls()))
                except Exception:
                    pass
            self._agent_loaded = len(self._agents) >= 1
        except Exception:
            pass

        if not self._agent_loaded:
                    # Последний запасной вариант: DualMomentum
            try:
                from crypto_agents import DualMomentum
                self._agents['GenAgent'] = DualMomentum()
                self._agent_loaded = True
            except Exception:
                pass

    def _get_weights(self) -> Dict[str, float]:
        """Возвращает веса агентов для текущего режима."""
        names = ['GenAgent', 'GenBullish', 'GenBearish', 'GenNeutral']
        if self._regime == 'bull':
            weights = self.W_BULL
        elif self._regime == 'bear':
            weights = self.W_BEAR
        else:
            weights = self.W_SIDEWAYS

        result = {}
        for name, w in zip(names, weights):
            if name in self._agents:
                result[name] = w
        # Нормализуем если не все агенты загружены
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        self.t = bar_index if bar_index is not None else self.t + 1

        if not self._agent_loaded:
            return {s: 0 for s in prices}

        # Обновляем историю цен
        for s, p in prices.items():
            self._ph.setdefault(s, deque(maxlen=5760)).append(float(p))

        # Детектируем режим
        if self.t - self._lr >= self.REGIME_INT or self._regime is None:
            try:
                from crypto_agents import _detect_regime_live, _r3
                raw = _detect_regime_live(self._ph, current_month=month)
                label = _r3(raw)
                self._regime = {'bullish': 'bull', 'bearish': 'bear'}.get(label, 'sideways')
            except Exception:
                self._regime = _regime(self._ph)
            self._lr = self.t

        # Получаем действия от всех агентов
        weights = self._get_weights()
        kw = dict(volumes=volumes, month=month, portfolio_value=portfolio_value, bar_index=bar_index)

        buy_w: Dict[str, float] = {}
        sell_w: Dict[str, float] = {}
        long_w: Dict[str, float] = {}
        short_w: Dict[str, float] = {}
        best_action: Dict[str, int] = {}  # лучшее действие от самого весомого агента

        for agent_name, w in weights.items():
            agent = self._agents[agent_name]
            try:
                acts = agent.act(prices, **kw)
            except Exception:
                acts = {}
            for sym, a in acts.items():
                if a == 0:
                    continue
                # FIX: GeneticsAgent через _GeneticsAdapter возвращает 6-action external:
                #   1=buy_spot, 2=sell_spot, 3=fut_long, 4=fut_short, 5=close_fut
                # Маппинг на buy/sell/long/short:
                if a == 1:        # buy_spot → long buy
                    buy_w[sym] = buy_w.get(sym, 0) + w
                    long_w[sym] = long_w.get(sym, 0) + w
                    if sym not in best_action or w > weights.get(best_action.get(sym + '_agent', ''), 0):
                        best_action[sym] = 1  # → spot buy для биржи
                        best_action[sym + '_agent'] = agent_name
                elif a == 3:      # fut_long → long buy (через фьючерсы)
                    buy_w[sym] = buy_w.get(sym, 0) + w
                    long_w[sym] = long_w.get(sym, 0) + w
                    if sym not in best_action or w > weights.get(best_action.get(sym + '_agent', ''), 0):
                        best_action[sym] = 4  # → fut_long_half для биржи
                        best_action[sym + '_agent'] = agent_name
                elif a == 4:      # fut_short → short
                    short_w[sym] = short_w.get(sym, 0) + w
                    if sym not in best_action or w > weights.get(best_action.get(sym + '_agent', ''), 0):
                        best_action[sym] = 6  # → fs_half для биржи
                        best_action[sym + '_agent'] = agent_name
                elif a in (2, 5): # sell_spot / close_fut → sell
                    sell_w[sym] = sell_w.get(sym, 0) + w

        # Стоп-лосс проверка
        result = {s: 0 for s in prices}
        for sym, info in list(self._pos.items()):
            px = prices.get(sym, 0)
            ep = info.get('entry_px', 0)
            if px <= 0 or ep <= 0:
                continue
            if info['side'] == 'short':
                loss = (px - ep) / ep
            else:
                loss = (ep - px) / ep
            if loss >= self.STOP_PCT:
                result[sym] = 3 if info['side'] == 'long_spot' else 8
                self._pos.pop(sym, None)

        # Голосование
        n_pos = len(self._pos)
        for sym in set(buy_w) | set(sell_w) | set(short_w):
            if result.get(sym, 0) != 0:
                continue  # уже закрыт стопом

            if sell_w.get(sym, 0) > self.SELL_THRESH and sym in self._pos:
                side = self._pos[sym].get('side', 'long_spot')
                result[sym] = 3 if side == 'long_spot' else 8
                self._pos.pop(sym, None)
            elif short_w.get(sym, 0) >= self.BUY_THRESH and short_w.get(sym, 0) > long_w.get(sym, 0):
                if sym not in self._pos and n_pos < self.MAX_POS:
                    result[sym] = 6
                    self._pos[sym] = {'side': 'short', 'entry_px': prices.get(sym, 0)}
                    n_pos += 1
            elif buy_w.get(sym, 0) >= self.BUY_THRESH:
                if sym not in self._pos and n_pos < self.MAX_POS:
                    action = best_action.get(sym, 1)
                    if action in (4, 5):
                        result[sym] = action
                        self._pos[sym] = {'side': 'long_fut', 'entry_px': prices.get(sym, 0)}
                    else:
                        result[sym] = 1
                        self._pos[sym] = {'side': 'long_spot', 'entry_px': prices.get(sym, 0)}
                    n_pos += 1

        return result


# ══════════════════════════════════════════════════════════════════
# ФАБРИКИ  (вызываются из exchange connector layers)
# ══════════════════════════════════════════════════════════════════

def make_panteon_agents() -> OrderedDict:
    """Создаёт одиночные стратегии для Panteon."""
    agents = OrderedDict()
    for name, cls in [
        ('Bomberman',      Bomberman),
        ('MomentumScalper', MomentumScalper),
        ('LiveAfterShock',  LiveAfterShock),
        ('LiveCrashHunter', LiveCrashHunter),
        ('LiveRegimePullback', LiveRegimePullback),
        ('LiveTrendFollow', LiveTrendFollow),
        ('LiveMeanRev',     LiveMeanRev),
        ('LiveVolCompress', LiveVolCompress),
        ('FundingArb',      FundingArb),
        ('LiveOIBreakout',  LiveOIBreakout),
        # Shadow candidate (v1). Попадает в _agent_pool, но в active_weights
        # не включается по умолчанию — сначала набирает shadow-метрики,
        # затем promotion_gate решает.
        ('VolBreakoutHunter', VolBreakoutHunter),
    ]:
        try:
            agents[name] = cls()
        except Exception as e:
            print(f"  [panteon_agents] {name}  FAIL: {e}")
    return agents


def make_panteon_players() -> OrderedDict:
    """Создаёт композитных игроков Panteon."""
    players = OrderedDict()
    extra_players = []
    try:
        from player_next import PanteonNextResearch
        extra_players.append(('PanteonNextResearch', PanteonNextResearch))
    except Exception as e:
        print(f"  [panteon_players] PanteonNextResearch  FAIL: {e}")
    for name, cls in [
        ('PlayerBomberman', PlayerBomberman),
        ('PlayerFunding',   PlayerFunding),
        ('Panteon',         Panteon),
        ('PanteonResearch', PanteonResearch),
        ('NeuroPlayer',     NeuroPlayer),
        *extra_players,
    ]:
        try:
            players[name] = cls()
        except Exception as e:
            print(f"  [panteon_players] {name}  FAIL: {e}")
    return players
