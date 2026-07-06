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
from typing import Any, Dict, Optional

from agent_meta import (
    ContextMemoryAgent,
    MemorySnapshotStore,
    PortfolioAllocatorAgent,
    ShadowScoringAgent,
    ShadowPlayerMetaSelector,
)
from agent_safety import (
    DeclineGuard as SharedDeclineGuard,
    MarketRegimeActionGuard,
    PositionExitGovernor,
    SymbolUniverseGuard,
)
from project_paths import MEMORY_DIR, RUNTIME_DIR, add_runtime_paths, project_path


def _bootstrap_project_paths():
    add_runtime_paths()


_bootstrap_project_paths()

log = logging.getLogger("panteon_agents")
SCRIPT_DIR = str(RUNTIME_DIR)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


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
            str(MEMORY_DIR / "Real_Player_Memory.txt"),
            str(MEMORY_DIR / "Real_Player_Memory"),
        )
    return (
        str(MEMORY_DIR / f"Real_Player_Memory_{token}.txt"),
        str(MEMORY_DIR / f"Real_Player_Memory_{token}"),
    )


def _aggregator_memory_path_for_namespace(namespace: Optional[str]) -> str:
    token = _sanitize_memory_namespace(namespace)
    if not token:
        return str(MEMORY_DIR / "Real_Player_Aggregator_Memory.txt")
    return str(MEMORY_DIR / f"Real_Player_Aggregator_Memory_{token}.txt")


_DEFAULT_MEMORY_NAMESPACE = os.getenv("CRYPTO_EXCHANGE") or os.getenv("CRYPTO_EXCHANGE_ID")
REAL_PLAYER_MEMORY_FILE, LEGACY_REAL_PLAYER_MEMORY_FILE = _memory_paths_for_namespace(
    _DEFAULT_MEMORY_NAMESPACE
)
REAL_PLAYER_AGGREGATOR_MEMORY_FILE = _aggregator_memory_path_for_namespace(
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


def _entry_cooldown_penalty(last_entry, sym, now, cooldown):
    if cooldown <= 0:
        return 0.0
    try:
        last = int((last_entry or {}).get(sym, -10**9) or -10**9)
        age = max(0, int(now) - last)
    except Exception:
        return 0.0
    if age >= cooldown:
        return 0.0
    return float(cooldown - age) / max(float(cooldown), 1.0)


def _take_ranked_entries(candidates, slots):
    if slots <= 0 or not candidates:
        return []
    ranked = sorted(candidates, key=lambda item: (-float(item[0]), str(item[1])))
    return ranked[:slots]


def _diag_value(value: Any):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _diag_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diag_value(v) for v in value]
    return str(value)


def _set_signal_diag(diags, sym, **payload):
    diags[str(sym).upper()] = {str(k): _diag_value(v) for k, v in payload.items()}


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


class _Genetics6to9Adapter:
    _MAP = {0: 0, 1: 2, 2: 3, 3: 5, 4: 7, 5: 8}

    def __init__(self, agent):
        self._a = agent

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        raw = self._a.act(
            prices,
            volumes,
            month=month,
            portfolio_value=portfolio_value,
            bar_index=bar_index,
        )
        return {s: self._MAP.get(a, 0) for s, a in (raw or {}).items()}

    def __getattr__(self, item):
        if item == '_a':
            raise AttributeError(item)
        return getattr(self._a, item)


def make_genetics_panteon_agent(genetics_cls):
    from crypto_agents import _GeneticsAdapter

    return _Genetics6to9Adapter(
        GeneticsSymbolGuard(_GeneticsAdapter(genetics_cls()))
    )


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
        diagnostics={}
        if self.t-self._lc<self.CHECK_INT:
            wait=max(0,self.CHECK_INT-(self.t-self._lc))
            for sym in prices:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="MomentumScalper",
                    reason="check_interval_wait",
                    action=0,
                    bar=self.t,
                    check_interval=self.CHECK_INT,
                    bars_until_check=wait,
                    history_len=len(self.h.get(sym,())),
                    volume_history_len=len(self.v.get(sym,())),
                    futures_replay_mode=bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
                )
            self.last_signal_diagnostics=diagnostics
            return actions
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
        self.last_signal_diagnostics:Dict[str,dict]={}
        self.t=0; self._lc=-9999

    def _long_open_action(self):
        return 5 if getattr(self, "FUTURES_REPLAY_MODE", False) else 2

    def _long_close_action(self):
        return 8 if getattr(self, "FUTURES_REPLAY_MODE", False) else 3

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.EMA_S*2)).append(float(p))
            self.v.setdefault(s,deque(maxlen=self.VOL_WIN*3)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.entry_px.setdefault(s,0.)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        diagnostics={}
        if self.t-self._lc<self.CHECK_INT:
            wait=max(0,self.CHECK_INT-(self.t-self._lc))
            for sym in prices:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="MomentumScalper",
                    reason="check_interval_wait",
                    action=0,
                    bar=self.t,
                    check_interval=self.CHECK_INT,
                    bars_until_check=wait,
                    history_len=len(self.h.get(sym,())),
                    volume_history_len=len(self.v.get(sym,())),
                    futures_replay_mode=bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
                )
            self.last_signal_diagnostics=diagnostics
            return actions
        self._lc=self.t
        n_long=sum(1 for v in self.pos.values() if v=='long')
        n_short=sum(1 for v in self.pos.values() if v=='short')
        for sym in prices:
            h=list(self.h[sym]); vl=list(self.v[sym]); cur=self.pos[sym]; px=prices[sym]
            if len(h)<self.EMA_S+1:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="MomentumScalper",
                    reason="not_enough_history",
                    action=0,
                    bar=self.t,
                    history_len=len(h),
                    required_history=self.EMA_S+1,
                    futures_replay_mode=bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
                )
                continue
            ef=_ema(h[-self.EMA_F*3:],  self.EMA_F)
            em=_ema(h[-self.EMA_M*2:],  self.EMA_M)
            es=_ema(h[-self.EMA_S:],     self.EMA_S)
            ep=self.entry_px[sym]
            common={
                "agent":"MomentumScalper",
                "bar":self.t,
                "history_len":len(h),
                "volume_history_len":len(vl),
                "price":px,
                "ema_fast":ef,
                "ema_mid":em,
                "ema_slow":es,
                "ema_fast_mid_ratio":(ef/(em+1e-12))-1.0,
                "ema_mid_slow_ratio":(em/(es+1e-12))-1.0,
                "long_fast_mid_gap":(ef/(em+1e-12))-1.003,
                "long_mid_slow_gap":(em/(es+1e-12))-1.001,
                "short_fast_mid_gap":0.997-(ef/(em+1e-12)),
                "short_mid_slow_gap":0.999-(em/(es+1e-12)),
                "n_long":n_long,
                "n_short":n_short,
                "max_pos":self.MAX_POS,
                "futures_replay_mode":bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
            }
            # Стоп + тейк
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or ef<em*0.999:
                    actions[sym]=self._long_close_action(); self.pos[sym]=None; n_long-=1
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_long",
                        action=actions[sym],
                        entry_price=ep,
                        stop_hit=px<=ep*(1-self.STOP),
                        target_hit=px>=ep*(1+self.TARGET),
                        trend_exit=ef<em*0.999,
                    )
                    continue
                _set_signal_diag(
                    diagnostics,
                    sym,
                    **common,
                    reason="in_long_position",
                    action=0,
                    entry_price=ep,
                )
                continue
            elif cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or ef>em*1.001:
                    actions[sym]=8; self.pos[sym]=None; n_short-=1
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_short",
                        action=actions[sym],
                        entry_price=ep,
                        stop_hit=px>=ep*(1+self.STOP),
                        target_hit=px<=ep*(1-self.TARGET),
                        trend_exit=ef>em*1.001,
                    )
                    continue
                _set_signal_diag(
                    diagnostics,
                    sym,
                    **common,
                    reason="in_short_position",
                    action=0,
                    entry_price=ep,
                )
                continue
            mom=_mom(self.h[sym], self.EMA_F)
            v_avg=float(np.mean(vl[-self.VOL_WIN-1:-1])) if len(vl)>self.VOL_WIN else 0
            v_cur=vl[-1] if vl else 0
            vol_spike=v_cur>v_avg*self.VOL_MULT if v_avg>0 else False
            long_trend_ok=ef>em*1.003 and em>es*1.001
            short_trend_ok=ef<em*0.997 and em<es*0.999
            common.update({
                "momentum":mom,
                "momentum_min":self.MOM_MIN,
                "long_momentum_gap":mom-self.MOM_MIN,
                "short_momentum_gap":-mom-self.MOM_MIN,
                "volume_current":v_cur,
                "volume_average":v_avg,
                "volume_mult":self.VOL_MULT,
                "volume_threshold":v_avg*self.VOL_MULT if v_avg>0 else 0.0,
                "volume_gap":v_cur-(v_avg*self.VOL_MULT) if v_avg>0 else None,
                "volume_spike":vol_spike,
                "long_trend_ok":long_trend_ok,
                "short_trend_ok":short_trend_ok,
            })
            # v5 FIX: vol_spike теперь используется как условие входа (было мёртвый код)
            if long_trend_ok and mom>self.MOM_MIN and vol_spike and n_long<self.MAX_POS:
                actions[sym]=self._long_open_action(); self.pos[sym]='long'; self.entry_px[sym]=px; n_long+=1
                _set_signal_diag(diagnostics,sym,**common,reason="candidate_long",action=actions[sym])
            elif short_trend_ok and mom<-self.MOM_MIN and vol_spike and n_short<self.MAX_POS:
                actions[sym]=7; self.pos[sym]='short'; self.entry_px[sym]=px; n_short+=1
                _set_signal_diag(diagnostics,sym,**common,reason="candidate_short",action=actions[sym])
            else:
                if not vol_spike:
                    reason="volume_below_threshold"
                elif mom<=self.MOM_MIN and mom>=-self.MOM_MIN:
                    reason="momentum_below_threshold"
                elif mom>self.MOM_MIN and not long_trend_ok:
                    reason="long_trend_not_aligned"
                elif mom<-self.MOM_MIN and not short_trend_ok:
                    reason="short_trend_not_aligned"
                elif mom>self.MOM_MIN and n_long>=self.MAX_POS:
                    reason="max_long_positions"
                elif mom<-self.MOM_MIN and n_short>=self.MAX_POS:
                    reason="max_short_positions"
                else:
                    reason="entry_conditions_not_met"
                _set_signal_diag(diagnostics,sym,**common,reason=reason,action=0)
        self.last_signal_diagnostics=diagnostics
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
    STOP=0.005; TARGET=0.010; HOLD=3*60; CHECK_INT=10; MAX_POS=4; ENTRY_COOLDOWN=60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.last_entry:Dict[str,int]={}
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
        n_long=sum(1 for v in self.pos.values() if v=='long')
        candidates=[]
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if cur=='long':
                held=self.t-self.et[sym]
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; n_long=max(0,n_long-1); continue
                continue
            if len(h)<self.LOOKBACK+self.RSI_N*3: continue
            lst=list(h); base=lst[-self.LOOKBACK-1]
            drop=(px-base)/base if base>0 else 0
            rsi=self._rsi(h,self.RSI_N)
            if drop<-self.DROP and rsi<self.RSI_OS:
                edge=(-drop-self.DROP)*100.0+(self.RSI_OS-rsi)/100.0
                edge-=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
                candidates.append((edge,sym,1,'long',px))
        slots=max(0,self.MAX_POS-n_long)
        for _,sym,action,side,px in _take_ranked_entries(candidates,slots):
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px; self.et[sym]=self.t
            self.last_entry[sym]=self.t
        return actions


class LiveCrashHunter:
    """
    Торгует во всех режимах (исправление оригинала только-bear).
    Bear: шортит топ-падающих
    Sideways/Bull: mean-reversion по RSI
    Stop: 1.5% обязательный.
    """
    CHECK_INT=30; RSI_N=14; RSI_OB=60; RSI_OS=40; MOM_LB=120; MAX_POS=4; ENTRY_COOLDOWN=90
    STOP=0.012; TARGET=0.020; HOLD=4*60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.last_entry:Dict[str,int]={}
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
        n_open=sum(1 for v in self.pos.values() if v is not None)
        candidates=[]
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            held=self.t-self.et[sym]
            # Стоп + тейк + время
            if cur=='short_fut':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; n_open=max(0,n_open-1); continue
            elif cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; n_open=max(0,n_open-1); continue
            if cur is not None: continue
            if len(h)<max(self.MOM_LB,self.RSI_N*3): continue
            rsi=self._rsi(h,self.RSI_N); mom=_mom(h,self.MOM_LB)
            cooldown=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
            if regime in ('bear','crash') and mom < -0.005 and rsi > 45:
                edge=(-mom-0.005)*100.0+(rsi-45.0)/100.0-cooldown
                candidates.append((edge,sym,6,'short_fut',px))
            elif rsi < self.RSI_OS and mom > -0.012 and regime != 'bear':
                edge=(self.RSI_OS-rsi)/100.0+max(mom+0.012,0.0)*20.0-cooldown
                candidates.append((edge,sym,1,'long',px))
            elif rsi>self.RSI_OB and regime in ('bear','sideways'):
                edge=(rsi-self.RSI_OB)/100.0+max(-mom,0.0)*20.0-cooldown
                candidates.append((edge,sym,6,'short_fut',px))
        slots=max(0,self.MAX_POS-n_open)
        for _,sym,action,side,px in _take_ranked_entries(candidates,slots):
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px; self.et[sym]=self.t
            self.last_entry[sym]=self.t
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
    EXIT_Z=0.35; CHECK_INT=10; STOP=0.012; HOLD=3*60; MAX_POS=4; ENTRY_COOLDOWN=60

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.last_entry:Dict[str,int]={}
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
        n_open=sum(1 for v in self.pos.values() if v is not None)
        candidates=[]
        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if len(h)<self.BB_PERIOD+self.RSI_N*3: continue
            z=self._bb_z(h); r=self._rsi(h,self.RSI_N); held=self.t-self.et[sym]
            # Выход: возврат к среднему ИЛИ стоп ИЛИ время
            if cur=='long':
                if z>-self.EXIT_Z or px<=ep*(1-self.STOP) or held>=self.HOLD:
                    actions[sym]=3; self.pos[sym]=None; n_open=max(0,n_open-1); continue
            elif cur=='short':
                if z<self.EXIT_Z or px>=ep*(1+self.STOP) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; n_open=max(0,n_open-1); continue
            if cur is not None: continue
            cooldown=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
            if z<-self.ENTRY_Z and r<self.RSI_OS and not bear_mode:
                edge=(-z-self.ENTRY_Z)+(self.RSI_OS-r)/100.0-cooldown
                candidates.append((edge,sym,1,'long',px))
            elif z>self.ENTRY_Z and r>self.RSI_OB and not bull_mode:
                edge=(z-self.ENTRY_Z)+(r-self.RSI_OB)/100.0-cooldown
                candidates.append((edge,sym,6,'short',px))
        slots=max(0,self.MAX_POS-n_open)
        for _,sym,action,side,px in _take_ranked_entries(candidates,slots):
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px; self.et[sym]=self.t
            self.last_entry[sym]=self.t
        return actions


class LiveVolCompress:
    """
    BB-Squeeze: сжатие → пробой.
    BBW < 2% → ждём пробой.
    Stop 0.5%, Target 1% (RR=2:1), max hold 2ч.
    """
    BB_PERIOD=20; BB_STD=2.; BBW_THRESH=0.015; MIN_BBW=0.003; MOM_N=5; MOM_MIN=0.003
    STOP=0.005; TARGET=0.010; HOLD=4*60; CHECK_INT=3; MAX_POS=4; ENTRY_COOLDOWN=45  # FIX: HOLD увеличен 2h→4h

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}; self.et:Dict[str,int]={}
        self.last_entry:Dict[str,int]={}
        self.last_signal_diagnostics:Dict[str,dict]={}
        self.t=0; self._lc=-9999

    def _long_open_action(self):
        return 5 if getattr(self, "FUTURES_REPLAY_MODE", False) else 1

    def _long_close_action(self):
        return 8 if getattr(self, "FUTURES_REPLAY_MODE", False) else 3

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=self.BB_PERIOD*4)).append(float(p))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.); self.et.setdefault(s,0)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        diagnostics={}
        if self.t-self._lc<self.CHECK_INT:
            wait=max(0,self.CHECK_INT-(self.t-self._lc))
            for sym in prices:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="LiveVolCompress",
                    reason="check_interval_wait",
                    action=0,
                    bar=self.t,
                    check_interval=self.CHECK_INT,
                    bars_until_check=wait,
                    history_len=len(self.h.get(sym,())),
                    futures_replay_mode=bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
                )
            self.last_signal_diagnostics=diagnostics
            return actions
        self._lc=self.t
        regime = _regime(self.h, lb=max(60, self.BB_PERIOD * 2))
        n_open=sum(1 for v in self.pos.values() if v is not None)
        candidates=[]
        for sym in prices:
            h=list(self.h[sym]); cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            if len(h)<self.BB_PERIOD+self.MOM_N+2:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="LiveVolCompress",
                    reason="not_enough_history",
                    action=0,
                    bar=self.t,
                    history_len=len(h),
                    required_history=self.BB_PERIOD+self.MOM_N+2,
                    regime=regime,
                    futures_replay_mode=bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
                )
                continue
            w=h[-self.BB_PERIOD:]; mid=float(np.mean(w)); sd=float(np.std(w))
            held=self.t-self.et[sym]
            common={
                "agent":"LiveVolCompress",
                "bar":self.t,
                "history_len":len(h),
                "price":px,
                "entry_price":ep,
                "held_bars":held,
                "regime":regime,
                "n_open":n_open,
                "max_pos":self.MAX_POS,
                "bb_period":self.BB_PERIOD,
                "bb_std":self.BB_STD,
                "bb_mid":mid,
                "bb_sd":sd,
                "bbw_thresh":self.BBW_THRESH,
                "min_bbw":self.MIN_BBW,
                "mom_n":self.MOM_N,
                "mom_min":self.MOM_MIN,
                "futures_replay_mode":bool(getattr(self,"FUTURES_REPLAY_MODE",False)),
            }
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    actions[sym]=self._long_close_action(); self.pos[sym]=None; n_open=max(0,n_open-1)
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_long",
                        action=actions[sym],
                        stop_hit=px<=ep*(1-self.STOP),
                        target_hit=px>=ep*(1+self.TARGET),
                        hold_expired=held>=self.HOLD,
                    )
                    continue
                _set_signal_diag(diagnostics,sym,**common,reason="in_long_position",action=0)
                continue
            if cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    actions[sym]=8; self.pos[sym]=None; n_open=max(0,n_open-1)
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_short",
                        action=actions[sym],
                        stop_hit=px>=ep*(1+self.STOP),
                        target_hit=px<=ep*(1-self.TARGET),
                        hold_expired=held>=self.HOLD,
                    )
                    continue
                _set_signal_diag(diagnostics,sym,**common,reason="in_short_position",action=0)
                continue
            if mid<1e-9:
                _set_signal_diag(diagnostics,sym,**common,reason="invalid_bb_mid",action=0)
                continue
            bbw=(2*self.BB_STD*sd)/mid
            mom=(h[-1]-h[-self.MOM_N-1])/(h[-self.MOM_N-1]+1e-12)
            squeeze_ok=self.MIN_BBW < bbw < self.BBW_THRESH
            cooldown=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
            common.update({
                "bbw":bbw,
                "squeeze_ok":squeeze_ok,
                "bbw_above_min_gap":bbw-self.MIN_BBW,
                "bbw_below_max_gap":self.BBW_THRESH-bbw,
                "momentum":mom,
                "long_momentum_gap":mom-self.MOM_MIN,
                "short_momentum_gap":-mom-self.MOM_MIN,
                "cooldown_penalty":cooldown,
                "regime_blocks_long":regime=="bear",
                "regime_blocks_short":regime=="bull",
            })
            if squeeze_ok:
                squeeze=(self.BBW_THRESH-bbw)/max(self.BBW_THRESH-self.MIN_BBW,1e-12)
                common["squeeze_score"]=squeeze
                if mom>self.MOM_MIN and regime != 'bear':
                    edge=(mom-self.MOM_MIN)*100.0+squeeze*0.25-cooldown
                    candidates.append((edge,sym,self._long_open_action(),'long',px))
                    _set_signal_diag(diagnostics,sym,**common,reason="candidate_long_pending_selection",action=0,edge=edge)
                elif mom<-self.MOM_MIN and regime != 'bull':
                    edge=(-mom-self.MOM_MIN)*100.0+squeeze*0.25-cooldown
                    candidates.append((edge,sym,6,'short',px))
                    _set_signal_diag(diagnostics,sym,**common,reason="candidate_short_pending_selection",action=0,edge=edge)
                elif mom>self.MOM_MIN and regime == 'bear':
                    _set_signal_diag(diagnostics,sym,**common,reason="long_regime_blocked",action=0)
                elif mom<-self.MOM_MIN and regime == 'bull':
                    _set_signal_diag(diagnostics,sym,**common,reason="short_regime_blocked",action=0)
                else:
                    _set_signal_diag(diagnostics,sym,**common,reason="momentum_below_threshold",action=0)
            else:
                reason="bbw_below_min" if bbw<=self.MIN_BBW else "bbw_above_threshold"
                _set_signal_diag(diagnostics,sym,**common,reason=reason,action=0)
        slots=max(0,self.MAX_POS-n_open)
        selected_entries=_take_ranked_entries(candidates,slots)
        selected_keys={(sym,action,side) for _,sym,action,side,_px in selected_entries}
        for edge,sym,action,side,_px in candidates:
            if (sym,action,side) in selected_keys:
                continue
            diag=dict(diagnostics.get(str(sym).upper(),{}))
            diag.update({"reason":"candidate_not_selected_slots","action":0,"edge":edge,"slots":slots})
            diagnostics[str(sym).upper()]={str(k):_diag_value(v) for k,v in diag.items()}
        for edge,sym,action,side,px in selected_entries:
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px; self.et[sym]=self.t
            self.last_entry[sym]=self.t
            diag=dict(diagnostics.get(str(sym).upper(),{}))
            diag.update({"reason":f"candidate_{side}","action":action,"edge":edge,"slots":slots})
            diagnostics[str(sym).upper()]={str(k):_diag_value(v) for k,v in diag.items()}
        self.last_signal_diagnostics=diagnostics
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
    RSI_OB=65; RSI_OS=35; STOP=0.01; TARGET=0.02; EMA_F=60; EMA_S=4*60; MAX_POS=4; ENTRY_COOLDOWN=90

    def __init__(self):
        self.h:Dict[str,deque]={}; self.pos:Dict[str,str]={}
        self.ep:Dict[str,float]={}
        self.close_action:Dict[str,int]={}
        self.last_entry:Dict[str,int]={}
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
        n_open=sum(1 for v in self.pos.values() if v is not None)
        candidates=[]
        for sym in prices:
            h=self.h[sym]; cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]
            # Стоп + тейк
            if cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET): actions[sym]=8; self.pos[sym]=None; self.close_action.pop(sym,None); n_open=max(0,n_open-1); continue
            elif cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET): actions[sym]=self.close_action.get(sym,8); self.pos[sym]=None; self.close_action.pop(sym,None); n_open=max(0,n_open-1); continue
            if cur is not None: continue
            if has_fd:
                fd=_get_funding(sym); rate=fd.get("funding_rate"); oi=fd.get("open_interest_usdt",0)
                if rate is None: continue
                rate = float(rate)
                if oi>0 and oi<self.OI_MIN: continue
                oi_score=min(float(oi or 0.0)/max(float(self.OI_MIN),1.0),5.0)*0.03
                cooldown=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
                if regime == 'bull':
                    if rate <= -self.STRONG:
                        edge=(-rate-self.STRONG)*10000.0+oi_score-cooldown
                        candidates.append((edge,sym,5,'long',px,8))
                    elif rate >= self.BULL_SHORT_BLOCK:
                        continue
                elif regime == 'bear':
                    if rate >= self.STRONG * 0.5:
                        edge=(rate-self.STRONG*0.5)*10000.0+oi_score-cooldown
                        candidates.append((edge,sym,7,'short',px,8))
                    elif rate <= -self.BEAR_LONG_BLOCK:
                        continue
                else:
                    if rate>=self.STRONG:
                        edge=(rate-self.STRONG)*10000.0+oi_score-cooldown
                        candidates.append((edge,sym,7,'short',px,8))
                    elif rate<=-self.STRONG:
                        edge=(-rate-self.STRONG)*10000.0+oi_score-cooldown
                        candidates.append((edge,sym,5,'long',px,8))
            else:
                # Запасной вариант: пересечение EMA (momentum, не contrarian)
                if len(h)<self.EMA_S+1: continue
                ef=_ema(list(h)[-self.EMA_F*3:],self.EMA_F)
                es=_ema(list(h)[-self.EMA_S:],  self.EMA_S)
                r=self._rsi(h)
                spread=(ef/es-1.0) if es>0 else 0.0
                cooldown=_entry_cooldown_penalty(self.last_entry,sym,self.t,self.ENTRY_COOLDOWN)*0.75
                if ef>es*1.003 and r<self.RSI_OB and regime != 'bear':
                    edge=(spread-0.003)*100.0+(self.RSI_OB-r)/100.0-cooldown
                    candidates.append((edge,sym,1,'long',px,3))
                elif ef<es*0.997 and r>self.RSI_OS and regime != 'bull':
                    edge=(-spread-0.003)*100.0+(r-self.RSI_OS)/100.0-cooldown
                    candidates.append((edge,sym,6,'short',px,8))
        slots=max(0,self.MAX_POS-n_open)
        for _,sym,action,side,px,close_action in _take_ranked_entries(candidates,slots):
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px
            self.close_action[sym]=close_action; self.last_entry[sym]=self.t
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
        self.last_signal_diagnostics:Dict[str,dict]={}

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        diagnostics={}
        for s,p in prices.items():
            self.h.setdefault(s,deque(maxlen=6*BAR)).append(float(p))
            self.v.setdefault(s,deque(maxlen=4*BAR+10)).append(float(volumes.get(s,0)))
            self.pos.setdefault(s,None); self.ep.setdefault(s,0.)
            self.oi_prev.setdefault(s,0.); self.et.setdefault(s,0)
        self.t=bar_index if bar_index is not None else self.t+1
        actions={s:0 for s in prices}
        if self.t-self._lc<self.CHECK_INT:
            for sym in prices:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    reason="check_interval_wait",
                    action=0,
                    bar=self.t,
                    check_interval=self.CHECK_INT,
                    bars_since_check=self.t-self._lc,
                    wait_bars=max(0,self.CHECK_INT-(self.t-self._lc)),
                )
            self.last_signal_diagnostics=diagnostics
            return actions
        self._lc=self.t
        for sym in prices:
            cur=self.pos[sym]; px=prices[sym]; ep=self.ep[sym]; held=self.t-self.et[sym]
            common=dict(bar=self.t,price=px,check_interval=self.CHECK_INT)
            # ОБЯЗАТЕЛЬНЫЙ стоп + тейк + время
            if cur=='long':
                if px<=ep*(1-self.STOP) or px>=ep*(1+self.TARGET) or held>=self.HOLD:
                    reason="long_stop" if px<=ep*(1-self.STOP) else "long_target" if px>=ep*(1+self.TARGET) else "long_hold_timeout"
                    actions[sym]=3; self.pos[sym]=None
                    _set_signal_diag(diagnostics,sym,**common,reason=reason,action=actions[sym],held_bars=held,entry_price=ep)
                    continue
            elif cur=='short':
                if px>=ep*(1+self.STOP) or px<=ep*(1-self.TARGET) or held>=self.HOLD:
                    reason="short_stop" if px>=ep*(1+self.STOP) else "short_target" if px<=ep*(1-self.TARGET) else "short_hold_timeout"
                    actions[sym]=8; self.pos[sym]=None
                    _set_signal_diag(diagnostics,sym,**common,reason=reason,action=actions[sym],held_bars=held,entry_price=ep)
                    continue
            if cur is not None:
                _set_signal_diag(diagnostics,sym,**common,reason="in_position",action=0,side=cur,held_bars=held,entry_price=ep)
                continue
            if len(self.h[sym])<self.MOM_N+1:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    **common,
                    reason="insufficient_history",
                    action=0,
                    history_len=len(self.h[sym]),
                    required_history=self.MOM_N+1,
                )
                continue
            mom=_mom(self.h[sym],self.MOM_N)
            vl=list(self.v[sym])
            v_avg=float(np.mean(vl[-BAR-1:-1])) if len(vl)>BAR else 0
            v_cur=vl[-1] if vl else 0
            spike=v_cur>v_avg*self.VOL_MULT if v_avg>0 else False
            has_fd=_FUNDING_FETCHER is not None
            oi_exp=False
            oi=0
            oi_change=0.0
            if has_fd:
                fd=_get_funding(sym); oi=fd.get("open_interest_usdt",0)
                prev=self.oi_prev[sym]
                if oi>0 and prev>0:
                    oi_change=(oi-prev)/prev
                    if oi_change>0.03: oi_exp=True
                self.oi_prev[sym]=oi if oi>0 else prev
            if spike or oi_exp:
                if mom>self.MOM_MIN:
                    actions[sym]=2; self.pos[sym]='long'; self.ep[sym]=px; self.et[sym]=self.t
                    _set_signal_diag(diagnostics,sym,**common,reason="candidate_long",action=actions[sym],momentum=mom,volume_current=v_cur,volume_avg=v_avg,volume_spike=spike,oi_expansion=oi_exp,open_interest_usdt=oi,oi_change=oi_change)
                elif mom<-self.MOM_MIN:
                    actions[sym]=6; self.pos[sym]='short'; self.ep[sym]=px; self.et[sym]=self.t
                    _set_signal_diag(diagnostics,sym,**common,reason="candidate_short",action=actions[sym],momentum=mom,volume_current=v_cur,volume_avg=v_avg,volume_spike=spike,oi_expansion=oi_exp,open_interest_usdt=oi,oi_change=oi_change)
                else:
                    _set_signal_diag(diagnostics,sym,**common,reason="momentum_below_threshold",action=0,momentum=mom,momentum_threshold=self.MOM_MIN,volume_current=v_cur,volume_avg=v_avg,volume_spike=spike,oi_expansion=oi_exp,open_interest_usdt=oi,oi_change=oi_change)
            else:
                _set_signal_diag(diagnostics,sym,**common,reason="no_volume_or_oi_breakout",action=0,momentum=mom,momentum_threshold=self.MOM_MIN,volume_current=v_cur,volume_avg=v_avg,volume_spike=spike,volume_multiplier=self.VOL_MULT,funding_fetcher_available=has_fd,oi_expansion=oi_exp,open_interest_usdt=oi,oi_change=oi_change)
        self.last_signal_diagnostics=diagnostics
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

class BullRotationAgent:
    """
    Bull-regime relative-strength rotation.

    Trader pattern: in a broad uptrend, capital rotates into the strongest
    leaders. This agent is long-only and buys strong symbols after a shallow
    pullback/reclaim, instead of chasing every EMA trend like LiveTrendFollow.
    """
    CHECK_INT = 10
    MARKET_LB = 240
    REL_LB = 240
    FAST_EMA = 45
    SLOW_EMA = 180
    TURN_LB = 18
    PULLBACK_LB = 90
    STOP = 0.014
    TARGET = 0.034
    HOLD = 8 * 60
    MAX_POS = 3
    MIN_BREADTH = 0.55
    MIN_REL_STRENGTH = 0.006
    MIN_PULLBACK = 0.0025
    MAX_PULLBACK = 0.018
    MIN_TURN = 0.0006
    MAX_EXTENSION = 0.018
    VOL_Z_MIN = -0.25

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _market_stats(self):
        rets = []
        for hist in self.h.values():
            vals = list(hist)
            if len(vals) < self.MARKET_LB + 1:
                continue
            base = vals[-self.MARKET_LB - 1]
            if base > 0:
                rets.append(vals[-1] / base - 1.0)
        if not rets:
            return 0.0, 0.5
        return float(np.median(rets)), sum(1 for r in rets if r > 0.0) / len(rets)

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.MARKET_LB, self.REL_LB, self.SLOW_EMA, self.PULLBACK_LB) * 2
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=180)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        regime = _regime(self.h, lb=self.MARKET_LB)
        market_ret, breadth = self._market_stats()
        if regime != 'bull' or market_ret <= 0.004 or breadth < self.MIN_BREADTH:
            manage_only = True
        else:
            manage_only = False

        n_open = sum(1 for v in self.pos.values() if v is not None)
        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < max(self.SLOW_EMA + 1, self.REL_LB + 1, self.PULLBACK_LB + 1):
                continue

            ef = _ema(h[-self.FAST_EMA * 3:], self.FAST_EMA)
            es = _ema(h[-self.SLOW_EMA:], self.SLOW_EMA)
            rel_ret = px / max(h[-self.REL_LB - 1], 1e-12) - 1.0
            rel_strength = rel_ret - market_ret

            if cur == 'long':
                if (
                    px <= ep * (1 - self.STOP)
                    or px >= ep * (1 + self.TARGET)
                    or held >= self.HOLD
                    or ef < es * 1.0002
                    or rel_strength < -0.002
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if manage_only or n_open >= self.MAX_POS:
                continue

            recent = h[-self.PULLBACK_LB:]
            high = float(max(recent))
            low = float(min(recent))
            pullback = 1.0 - px / max(high, 1e-12)
            bounce = px / max(low, 1e-12) - 1.0
            turn = _mom(h, self.TURN_LB)
            extension = px / max(ef, 1e-12) - 1.0
            vol_z = _vol_zscore(self.v[sym], window=60)

            if (
                ef > es * 1.002
                and rel_strength >= self.MIN_REL_STRENGTH
                and self.MIN_PULLBACK <= pullback <= self.MAX_PULLBACK
                and bounce >= self.MIN_TURN
                and turn >= self.MIN_TURN
                and extension <= self.MAX_EXTENSION
                and vol_z >= self.VOL_Z_MIN
            ):
                actions[sym] = 5
                self.pos[sym] = 'long'
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


class BearReliefFadeAgent:
    """
    Bear-regime relief-rally fade.

    Trader pattern: in a bear market, weak rallies into overhead supply are
    sold. This is not a top-loser momentum short; it waits for a bounce to fail.
    """
    CHECK_INT = 10
    MARKET_LB = 240
    FAST_EMA = 45
    SLOW_EMA = 180
    BOUNCE_LB = 90
    TURN_LB = 12
    RSI_N = 14
    STOP = 0.013
    TARGET = 0.030
    HOLD = 6 * 60
    MAX_POS = 3
    MIN_BREADTH_DOWN = 0.55
    MIN_BOUNCE = 0.006
    MAX_BOUNCE = 0.045
    MIN_WEAKNESS = 0.003
    RSI_FADE = 52
    VOL_Z_MIN = -0.20

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _rsi(self, h, n=14):
        vals = list(h)
        if len(vals) < n * 3:
            return 50.0
        d = np.diff(vals[-(n * 3):])
        g = np.where(d > 0, d, 0)
        l = np.where(d < 0, -d, 0)
        ag, al = np.mean(g[:n]), np.mean(l[:n])
        for gi, li in zip(g[n:], l[n:]):
            ag = (ag * (n - 1) + gi) / n
            al = (al * (n - 1) + li) / n
        return float(100 - 100 / (1 + ag / (al + 1e-9)))

    def _market_stats(self):
        rets = []
        for hist in self.h.values():
            vals = list(hist)
            if len(vals) < self.MARKET_LB + 1:
                continue
            base = vals[-self.MARKET_LB - 1]
            if base > 0:
                rets.append(vals[-1] / base - 1.0)
        if not rets:
            return 0.0, 0.5
        return float(np.median(rets)), sum(1 for r in rets if r < 0.0) / len(rets)

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.MARKET_LB, self.SLOW_EMA, self.BOUNCE_LB) * 2
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=180)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        regime = _regime(self.h, lb=self.MARKET_LB)
        market_ret, down_breadth = self._market_stats()
        manage_only = regime != 'bear' or market_ret >= -0.004 or down_breadth < self.MIN_BREADTH_DOWN
        n_open = sum(1 for v in self.pos.values() if v is not None)

        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < max(self.SLOW_EMA + 1, self.MARKET_LB + 1, self.BOUNCE_LB + 1):
                continue

            ef = _ema(h[-self.FAST_EMA * 3:], self.FAST_EMA)
            es = _ema(h[-self.SLOW_EMA:], self.SLOW_EMA)
            rel_ret = px / max(h[-self.MARKET_LB - 1], 1e-12) - 1.0
            rel_weakness = market_ret - rel_ret

            if cur == 'short':
                if (
                    px >= ep * (1 + self.STOP)
                    or px <= ep * (1 - self.TARGET)
                    or held >= self.HOLD
                    or ef > es * 1.001
                    or rel_weakness < -0.002
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if manage_only or n_open >= self.MAX_POS:
                continue

            recent = h[-self.BOUNCE_LB:]
            low = float(min(recent))
            high = float(max(recent))
            bounce = px / max(low, 1e-12) - 1.0
            below_high = 1.0 - px / max(high, 1e-12)
            turn = _mom(h, self.TURN_LB)
            rsi = self._rsi(h, self.RSI_N)
            near_fast = abs(px / max(ef, 1e-12) - 1.0) <= 0.012
            vol_z = _vol_zscore(self.v[sym], window=60)

            if (
                ef < es * 0.998
                and rel_weakness >= self.MIN_WEAKNESS
                and self.MIN_BOUNCE <= bounce <= self.MAX_BOUNCE
                and below_high >= 0.0015
                and turn <= -0.0004
                and rsi >= self.RSI_FADE
                and near_fast
                and vol_z >= self.VOL_Z_MIN
            ):
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


class NeutralRangeScalper:
    """
    Neutral-regime range-liquidity fade.

    Trader pattern: in a clean sideways market, fade failed tests of the range
    edges and exit near the middle. It uses channel location and range quality
    instead of the BB/RSI recipe used by LiveMeanRev.
    """
    CHECK_INT = 6
    RANGE_LB = 180
    MARKET_LB = 240
    TURN_LB = 10
    MIN_WIDTH = 0.007
    MAX_WIDTH = 0.055
    EDGE = 0.14
    STOP = 0.008
    TARGET = 0.016
    HOLD = 3 * 60
    MAX_POS = 3
    MIN_NOISE = 2.8
    MAX_MARKET_MOM = 0.006

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.mid: Dict[str, float] = {}
        self.t = 0
        self._lc = -9999

    def _noise_ratio(self, h):
        vals = list(h)
        if len(vals) < self.RANGE_LB + 1:
            return 0.0
        arr = np.asarray(vals[-(self.RANGE_LB + 1):], dtype=float)
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
        path = float(np.sum(np.abs(rets)))
        net = abs(float(arr[-1] / max(arr[0], 1e-12) - 1.0))
        return path / max(net, 1e-4)

    def _market_mom(self):
        vals = []
        for hist in self.h.values():
            h = list(hist)
            if len(h) < self.MARKET_LB + 1:
                continue
            base = h[-self.MARKET_LB - 1]
            if base > 0:
                vals.append(h[-1] / base - 1.0)
        return float(np.median(vals)) if vals else 0.0

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.RANGE_LB, self.MARKET_LB) * 2
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=180)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self.mid.setdefault(s, float(p))
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        regime = _regime(self.h, lb=self.MARKET_LB)
        market_mom = self._market_mom()
        allow_entry = regime == 'sideways' and abs(market_mom) <= self.MAX_MARKET_MOM
        n_open = sum(1 for v in self.pos.values() if v is not None)

        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < self.RANGE_LB + 2:
                continue

            window = h[-self.RANGE_LB - 1:-1]
            lo = float(min(window))
            hi = float(max(window))
            width = (hi / max(lo, 1e-12)) - 1.0
            mid = (hi + lo) / 2.0
            self.mid[sym] = mid

            if cur == 'long':
                if px >= mid or px <= ep * (1 - self.STOP) or px >= ep * (1 + self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if cur == 'short':
                if px <= mid or px >= ep * (1 + self.STOP) or px <= ep * (1 - self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if not allow_entry or n_open >= self.MAX_POS:
                continue
            if width < self.MIN_WIDTH or width > self.MAX_WIDTH:
                continue

            loc = (px - lo) / max(hi - lo, 1e-12)
            turn = _mom(h, self.TURN_LB)
            noise = self._noise_ratio(self.h[sym])
            vol_z = _vol_zscore(self.v[sym], window=60)
            if noise < self.MIN_NOISE:
                continue

            if loc <= self.EDGE and turn >= 0.00035 and vol_z >= -0.35:
                actions[sym] = 5
                self.pos[sym] = 'long'
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
            elif loc >= 1.0 - self.EDGE and turn <= -0.00035 and vol_z >= -0.35:
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


class NeutralLiquiditySweepAgent:
    """
    Neutral-market false-breakout fade.

    This is not a generic RSI/mean-reversion agent. It waits for a symbol to
    sweep beyond a recent range edge on volume and then close back inside the
    range while BTC/ETH and market breadth remain neutral. The edge is the
    failed breakout, not simply "price is low/high".
    """
    CHECK_INT = 3
    LOOKBACK = 90
    MARKET_LB = 180
    VOL_WINDOW = 60
    MAX_POS = 3
    SWEEP_MARGIN = 0.0012
    REENTRY_MARGIN = 0.0004
    MIN_WIDTH = 0.004
    MAX_WIDTH = 0.075
    MIN_VOL_Z = 0.35
    MAX_ANCHOR_MOM = 0.008
    MAX_BREADTH_ABS = 0.55
    STOP = 0.007
    TARGET = 0.014
    HOLD = 90

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _ret(self, sym: str, lb: int) -> Optional[float]:
        hist = list(self.h.get(sym) or [])
        if len(hist) < lb + 1:
            return None
        base = hist[-lb - 1]
        if base <= 0:
            return None
        return hist[-1] / base - 1.0

    def _neutral_context(self) -> bool:
        lb = max(2, int(self.MARKET_LB))
        anchor_rets = [
            r for r in (self._ret("BTC", lb), self._ret("ETH", lb))
            if r is not None
        ]
        if anchor_rets and max(abs(r) for r in anchor_rets) > self.MAX_ANCHOR_MOM:
            return False

        rets = []
        for hist in self.h.values():
            h = list(hist)
            if len(h) < lb + 1:
                continue
            base = h[-lb - 1]
            if base > 0:
                rets.append(h[-1] / base - 1.0)
        if not rets:
            return True
        med = float(np.median(rets))
        directional = [r for r in rets if abs(r) > self.MAX_ANCHOR_MOM * 0.25]
        if len(directional) >= 4:
            up_share = sum(1 for r in directional if r > 0.0) / len(directional)
            breadth_bias = abs(up_share - 0.5) * 2.0
        else:
            breadth_bias = 0.0
        return abs(med) <= self.MAX_ANCHOR_MOM and breadth_bias <= self.MAX_BREADTH_ABS

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.LOOKBACK, self.MARKET_LB, self.VOL_WINDOW) * 2 + 10
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=max_hist)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)

        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        allow_entry = self._neutral_context()
        n_open = sum(1 for v in self.pos.values() if v is not None)

        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < self.LOOKBACK + 3:
                continue

            if cur == "long":
                if px <= ep * (1 - self.STOP) or px >= ep * (1 + self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if cur == "short":
                if px >= ep * (1 + self.STOP) or px <= ep * (1 - self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if not allow_entry or n_open >= self.MAX_POS:
                continue

            base_window = h[-self.LOOKBACK - 2:-2]
            if len(base_window) < self.LOOKBACK:
                continue
            lo = float(min(base_window))
            hi = float(max(base_window))
            width = hi / max(lo, 1e-12) - 1.0
            if width < self.MIN_WIDTH or width > self.MAX_WIDTH:
                continue

            prev = h[-2]
            vol_z = _vol_zscore(self.v[sym], window=max(3, int(self.VOL_WINDOW)))
            if vol_z < self.MIN_VOL_Z:
                continue

            swept_low = prev < lo * (1 - self.SWEEP_MARGIN) and px > lo * (1 - self.REENTRY_MARGIN)
            swept_high = prev > hi * (1 + self.SWEEP_MARGIN) and px < hi * (1 + self.REENTRY_MARGIN)
            strong = vol_z >= self.MIN_VOL_Z + 0.75

            if swept_low:
                actions[sym] = 5 if strong else 4
                self.pos[sym] = "long"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
            elif swept_high:
                actions[sym] = 7 if strong else 6
                self.pos[sym] = "short"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


class AnchorFlowMomentumAgent:
    """
    Anchor/breadth-confirmed momentum continuation.

    The agent trades with BTC/ETH anchor direction only when market breadth
    agrees, then selects symbols with relative momentum and volume expansion.
    This differs from single-symbol trend following and OI breakouts: market
    structure is confirmed by anchors before any symbol is allowed to trade.
    """
    CHECK_INT = 5
    FAST_LB = 24
    SLOW_LB = 96
    MARKET_LB = 120
    VOL_WINDOW = 60
    MAX_POS = 3
    MIN_ANCHOR_MOM = 0.0025
    MIN_BREADTH = 0.56
    MIN_REL_STRENGTH = 0.0015
    MIN_VOL_Z = 0.25
    STOP = 0.012
    TARGET = 0.028
    HOLD = 180

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _ret_from(self, h, lb: int) -> Optional[float]:
        vals = list(h)
        if len(vals) < lb + 1:
            return None
        base = vals[-lb - 1]
        if base <= 0:
            return None
        return vals[-1] / base - 1.0

    def _anchor_direction(self):
        lb = max(2, int(self.MARKET_LB))
        anchor = []
        for sym in ("BTC", "ETH"):
            r = self._ret_from(self.h.get(sym) or [], lb)
            if r is not None:
                anchor.append(r)
        if not anchor:
            return "", 0.0

        rets = []
        for hist in self.h.values():
            r = self._ret_from(hist, lb)
            if r is not None:
                rets.append(r)
        if not rets:
            return "", 0.0

        anchor_med = float(np.median(anchor))
        up_share = sum(1 for r in rets if r > 0.0) / len(rets)
        if anchor_med >= self.MIN_ANCHOR_MOM and up_share >= self.MIN_BREADTH:
            return "long", anchor_med
        if anchor_med <= -self.MIN_ANCHOR_MOM and up_share <= 1.0 - self.MIN_BREADTH:
            return "short", anchor_med
        return "", anchor_med

    def _market_median_return(self, lb: int) -> float:
        vals = []
        for hist in self.h.values():
            r = self._ret_from(hist, lb)
            if r is not None:
                vals.append(r)
        return float(np.median(vals)) if vals else 0.0

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.FAST_LB, self.SLOW_LB, self.MARKET_LB, self.VOL_WINDOW) * 2 + 10
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=max_hist)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)

        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        direction, anchor_mom = self._anchor_direction()
        median_slow = self._market_median_return(max(2, int(self.SLOW_LB)))
        n_open = sum(1 for v in self.pos.values() if v is not None)

        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < max(self.FAST_LB, self.SLOW_LB, self.MARKET_LB) + 1:
                continue

            fast = self._ret_from(self.h[sym], max(2, int(self.FAST_LB))) or 0.0
            slow = self._ret_from(self.h[sym], max(2, int(self.SLOW_LB))) or 0.0
            rel = slow - median_slow

            if cur == "long":
                if (
                    px <= ep * (1 - self.STOP)
                    or px >= ep * (1 + self.TARGET)
                    or held >= self.HOLD
                    or direction == "short"
                    or fast < -self.MIN_ANCHOR_MOM
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue
            if cur == "short":
                if (
                    px >= ep * (1 + self.STOP)
                    or px <= ep * (1 - self.TARGET)
                    or held >= self.HOLD
                    or direction == "long"
                    or fast > self.MIN_ANCHOR_MOM
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if not direction or n_open >= self.MAX_POS:
                continue
            vol_z = _vol_zscore(self.v[sym], window=max(3, int(self.VOL_WINDOW)))
            if vol_z < self.MIN_VOL_Z:
                continue

            strong = abs(rel) >= self.MIN_REL_STRENGTH * 2.0 and vol_z >= self.MIN_VOL_Z + 0.75
            if (
                direction == "long"
                and anchor_mom > 0
                and slow >= self.MIN_ANCHOR_MOM * 0.7
                and fast > 0
                and rel >= self.MIN_REL_STRENGTH
            ):
                actions[sym] = 5 if strong else 4
                self.pos[sym] = "long"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
            elif (
                direction == "short"
                and anchor_mom < 0
                and slow <= -self.MIN_ANCHOR_MOM * 0.7
                and fast < 0
                and rel <= -self.MIN_REL_STRENGTH
            ):
                actions[sym] = 7 if strong else 6
                self.pos[sym] = "short"
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


class CrashPanicShortAgent:
    """
    Crash-regime liquidation cascade short.

    Trader pattern: during broad panic, do not bottom-fish first. Short fresh
    breakdowns with volume expansion and get out quickly on snap-back risk.
    """
    CHECK_INT = 3
    SHORT_LB = 60
    MID_LB = 240
    BREAKDOWN_LB = 45
    FAST_EMA = 20
    SLOW_EMA = 90
    STOP = 0.010
    TARGET = 0.028
    HOLD = 90
    MAX_POS = 3
    MIN_DOWN_SHARE = 0.65
    MIN_VOL_RATIO = 1.12
    MIN_SYMBOL_DROP = -0.006
    BREAK_MARGIN = 0.0008

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.v: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.t = 0
        self._lc = -9999

    def _market_crash(self):
        short_rets = []
        mid_rets = []
        vol_ratios = []
        for sym, hist in self.h.items():
            h = list(hist)
            if len(h) >= self.SHORT_LB + 1:
                base = h[-self.SHORT_LB - 1]
                if base > 0:
                    short_rets.append(h[-1] / base - 1.0)
            if len(h) >= self.MID_LB + 1:
                base = h[-self.MID_LB - 1]
                if base > 0:
                    mid_rets.append(h[-1] / base - 1.0)
            vols = list(self.v.get(sym) or [])
            if len(vols) >= 120:
                fast = float(np.mean(vols[-20:]))
                slow = float(np.mean(vols[-120:]))
                if slow > 1e-9:
                    vol_ratios.append(fast / slow)
        if len(short_rets) < 4:
            return False
        short_med = float(np.median(short_rets))
        short_down = sum(1 for r in short_rets if r <= -0.004) / len(short_rets)
        mid_med = float(np.median(mid_rets)) if mid_rets else short_med
        mid_down = sum(1 for r in mid_rets if r <= -0.010) / len(mid_rets) if mid_rets else short_down
        vol_ratio = float(np.median(vol_ratios)) if vol_ratios else 1.0
        panic_now = short_med <= -0.010 and short_down >= self.MIN_DOWN_SHARE
        panic_mid = mid_med <= -0.025 and mid_down >= self.MIN_DOWN_SHARE
        return (panic_now or panic_mid) and vol_ratio >= self.MIN_VOL_RATIO

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        max_hist = max(self.MID_LB, self.SLOW_EMA, self.BREAKDOWN_LB) * 2
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=max_hist)).append(float(p))
            self.v.setdefault(s, deque(maxlen=240)).append(float(volumes.get(s, 0) or 0))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        crash = self._market_crash()
        n_open = sum(1 for v in self.pos.values() if v is not None)
        for sym in prices:
            h = list(self.h[sym])
            px = float(prices[sym])
            cur = self.pos[sym]
            ep = float(self.ep[sym])
            held = self.t - self.et[sym]
            if len(h) < max(self.SLOW_EMA + 1, self.BREAKDOWN_LB + 1, self.SHORT_LB + 1):
                continue

            ef = _ema(h[-self.FAST_EMA * 3:], self.FAST_EMA)
            es = _ema(h[-self.SLOW_EMA:], self.SLOW_EMA)
            snapback = _mom(h, 8) > 0.004
            if cur == 'short':
                if (
                    px >= ep * (1 + self.STOP)
                    or px <= ep * (1 - self.TARGET)
                    or held >= self.HOLD
                    or px > ef * 1.004
                    or snapback
                ):
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                continue

            if not crash or n_open >= self.MAX_POS:
                continue

            prev = h[-self.BREAKDOWN_LB - 1:-1]
            recent_low = float(min(prev))
            symbol_drop = _mom(h, self.SHORT_LB)
            vol_z = _vol_zscore(self.v[sym], window=60)
            if (
                ef < es * 0.997
                and symbol_drop <= self.MIN_SYMBOL_DROP
                and px < recent_low * (1 - self.BREAK_MARGIN)
                and vol_z >= 0.75
            ):
                actions[sym] = 7
                self.pos[sym] = 'short'
                self.ep[sym] = px
                self.et[sym] = self.t
                n_open += 1
        return actions


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
    MAX_POS = 4
    ENTRY_COOLDOWN = 120
    ALLOW_LONG = False
    ALLOW_SHORT = True

    def __init__(self):
        self.h: Dict[str, deque] = {}
        self.oi_h: Dict[str, deque] = {}
        self.pos: Dict[str, Optional[str]] = {}
        self.ep: Dict[str, float] = {}
        self.et: Dict[str, int] = {}
        self.last_entry: Dict[str, int] = {}
        self.last_signal_diagnostics: Dict[str, dict] = {}
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
        diagnostics: Dict[str, dict] = {}
        if self.t - self._lc < self.CHECK_INT:
            wait = max(0, self.CHECK_INT - (self.t - self._lc))
            for sym in prices:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    agent="CarryFlowAgentV2",
                    reason="check_interval_wait",
                    action=0,
                    bar=self.t,
                    check_interval=self.CHECK_INT,
                    bars_until_check=wait,
                    history_len=len(self.h.get(sym, ())),
                    oi_history_len=len(self.oi_h.get(sym, ())),
                )
            self.last_signal_diagnostics = diagnostics
            return actions
        self._lc = self.t
        n_open = sum(1 for v in self.pos.values() if v is not None)
        candidates = []
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

            common = {
                "agent": "CarryFlowAgentV2",
                "bar": self.t,
                "history_len": len(h),
                "required_history": self.EMA_SLOW + 5,
                "oi_history_len": len(oi_vals),
                "price": px,
                "entry_price": ep,
                "held_bars": held,
                "funding_present": bool(fd),
                "funding_rate": rate,
                "funding_entry": self.FUNDING_ENTRY,
                "long_ratio": long_ratio,
                "short_ratio": short_ratio,
                "crowd_ratio": self.CROWD_RATIO,
                "basis": basis,
                "basis_entry": self.BASIS_ENTRY,
                "extension": ext,
                "extreme_extension": self.EXTREME_EXT,
                "rsi": rsi,
                "rsi_overbought": self.RSI_OB,
                "rsi_oversold": self.RSI_OS,
                "oi_change": oi_chg,
                "oi_spike": self.OI_SPIKE,
                "n_open": n_open,
                "max_pos": self.MAX_POS,
            }

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
                    n_open = max(0, n_open - 1)
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_long",
                        action=actions[sym],
                        stop_hit=px <= ep * (1 - self.STOP),
                        target_hit=px >= ep * (1 + self.TARGET),
                        hold_expired=held >= self.HOLD,
                        normalized=bool(normalized),
                    )
                else:
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="in_long_position",
                        action=0,
                    )
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
                    n_open = max(0, n_open - 1)
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="close_short",
                        action=actions[sym],
                        stop_hit=px >= ep * (1 + self.STOP),
                        target_hit=px <= ep * (1 - self.TARGET),
                        hold_expired=held >= self.HOLD,
                        normalized=bool(normalized),
                    )
                else:
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="in_short_position",
                        action=0,
                    )
                continue

            if len(h) < self.EMA_SLOW + 5:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    **common,
                    reason="not_enough_history",
                    action=0,
                )
                continue
            es = _ema(list(h)[-self.EMA_SLOW:], self.EMA_SLOW)
            trend_up = ef > es * 1.001 if es > 0 else False
            trend_dn = ef < es * 0.999 if es > 0 else False
            cooldown = _entry_cooldown_penalty(
                self.last_entry, sym, self.t, self.ENTRY_COOLDOWN,
            ) * 0.75
            common.update({
                "ema_fast": ef,
                "ema_slow": es,
                "trend_up": trend_up,
                "trend_down": trend_dn,
                "cooldown_penalty": cooldown,
            })

            if fd and oi_chg >= self.OI_SPIKE:
                if (
                    self.ALLOW_SHORT
                    and
                    rate >= self.FUNDING_ENTRY
                    and long_ratio >= self.CROWD_RATIO
                    and basis >= self.BASIS_ENTRY
                    and (ext >= self.EXTREME_EXT * 0.5 or rsi >= self.RSI_OB)
                ):
                    edge=(
                        (rate-self.FUNDING_ENTRY)*10000.0
                        +(long_ratio-self.CROWD_RATIO)*4.0
                        +(basis-self.BASIS_ENTRY)*1000.0
                        +(oi_chg-self.OI_SPIKE)*10.0
                        +max(ext-self.EXTREME_EXT*0.5,0.0)*100.0
                        +max(rsi-self.RSI_OB,0.0)/100.0
                        -cooldown
                    )
                    candidates.append((edge,sym,7,'short',px))
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="candidate_short_pending_selection",
                        action=0,
                        edge=edge,
                    )
                elif (
                    self.ALLOW_LONG
                    and
                    rate <= -self.FUNDING_ENTRY
                    and short_ratio >= self.CROWD_RATIO
                    and basis <= -self.BASIS_ENTRY
                    and (ext <= -self.EXTREME_EXT * 0.5 or rsi <= self.RSI_OS)
                ):
                    edge=(
                        (-rate-self.FUNDING_ENTRY)*10000.0
                        +(short_ratio-self.CROWD_RATIO)*4.0
                        +(-basis-self.BASIS_ENTRY)*1000.0
                        +(oi_chg-self.OI_SPIKE)*10.0
                        +max(-ext-self.EXTREME_EXT*0.5,0.0)*100.0
                        +max(self.RSI_OS-rsi,0.0)/100.0
                        -cooldown
                    )
                    candidates.append((edge,sym,5,'long',px))
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="candidate_long_pending_selection",
                        action=0,
                        edge=edge,
                    )
                else:
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="funding_conditions_not_met",
                        action=0,
                    )
                continue

            if not fd:
                if self.ALLOW_LONG and trend_up and ext <= -self.EXTREME_EXT and rsi <= self.RSI_OS:
                    edge=max(-ext-self.EXTREME_EXT,0.0)*100.0+max(self.RSI_OS-rsi,0.0)/100.0-cooldown
                    candidates.append((edge,sym,5,'long',px))
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="fallback_candidate_long_pending_selection",
                        action=0,
                        edge=edge,
                    )
                elif self.ALLOW_SHORT and trend_dn and ext >= self.EXTREME_EXT and rsi >= self.RSI_OB:
                    edge=max(ext-self.EXTREME_EXT,0.0)*100.0+max(rsi-self.RSI_OB,0.0)/100.0-cooldown
                    candidates.append((edge,sym,7,'short',px))
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="fallback_candidate_short_pending_selection",
                        action=0,
                        edge=edge,
                    )
                else:
                    _set_signal_diag(
                        diagnostics,
                        sym,
                        **common,
                        reason="funding_missing_fallback_conditions_not_met",
                        action=0,
                    )
            else:
                _set_signal_diag(
                    diagnostics,
                    sym,
                    **common,
                    reason="oi_below_threshold",
                    action=0,
                )
        slots=max(0,self.MAX_POS-n_open)
        for _,sym,action,side,px in _take_ranked_entries(candidates,slots):
            actions[sym]=action; self.pos[sym]=side; self.ep[sym]=px; self.et[sym]=self.t
            self.last_entry[sym]=self.t
            diag = dict(diagnostics.get(str(sym).upper(), {}))
            diag.update({"reason": f"candidate_{side}", "action": action, "slots": slots})
            diagnostics[str(sym).upper()] = {str(k): _diag_value(v) for k, v in diag.items()}
        self.last_signal_diagnostics = diagnostics
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
        return os.getenv("EXTERNAL_SIGNALS_FILE") or project_path("state", "external_signals.json")

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
    # FIX 2026-05-03 (dynamic quarantine v3): seed-карантин по плохой истории
    # (0/3 wins on BG, -4.48% pnl). Снимается автоматически
    # `recompute_dynamic_quarantine()`, если в каком-либо режиме рынка появится
    # достаточный положительный опыт.
    PLAYER_STATUS = "live"            # перевычисляется в Panteon_Trade._resolve_status
    PLAYER_STATUS_REASON = ""
    PLAYER_QUARANTINE_SEED = True
    PLAYER_QUARANTINE_SEED_REASON = "0/3 wins on BG (-4.48% pnl); collect more data first"

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
    # FIX 2026-05-03 (dynamic quarantine v3): убрано статическое shadow_only.
    # PlayerFunding часто прибылен в bullish-funding-сессиях даже при том,
    # что отдельный FundingArb на BG/MEXC проигрывал. Решение о допуске в live
    # принимается ДИНАМИЧЕСКИ по per-regime производительности самого игрока,
    # а не его суб-агентов: см. Panteon_Trade._resolve_status / save_leaderboard_json.
    PLAYER_STATUS = "live"
    PLAYER_STATUS_REASON = ""

    def __init__(self):
        self._fa=FundingArb(); self._ms=MomentumScalper()

    def reset_for_live(self, bar_index: int = 0):
        for agent in (self._fa, self._ms):
            if hasattr(agent, '_lc'):
                try:
                    agent._lc = -99999
                except (AttributeError, TypeError):
                    pass
            try:
                agent.t = bar_index
            except (AttributeError, TypeError):
                pass
            for attr in ('pos', 'ep', 'entry_px', 'et'):
                state = getattr(agent, attr, None)
                if not isinstance(state, dict):
                    continue
                for key in state:
                    state[key] = None if attr == 'pos' else 0.0

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


class CandlePatternAgent:
    """
    FIX (2026-04-29): Свечной паттерн-агент. Реконструирует synthetic OHLC bars
    из close-only истории и распознаёт 6 классических паттернов разворота:
      Bullish/Bearish Engulfing, Hammer/ShootingStar, Morning/Evening Star.

    Дизайн:
      • Накапливает close history в self.h (как все Live*-агенты).
      • Каждый AGG_BARS минут собирает synthetic OHLC: open=h[-AGG], close=h[-1],
        high=max(h[-AGG:]), low=min(h[-AGG:]). Это аппроксимация — реальные
        биржевые high/low внутри минут не видны, но при AGG=5..10 минут
        ошибка <30 % амплитуды свечи (приемлемо для confirmation-фильтра).
      • Сигнализирует только когда подтверждается trend-context: bullish
        паттерн в bear-trend = reversal (entry_long), в bull-trend = continuation
        (тоже entry_long); bullish паттерн в pure neutral без объёма = пропуск.
      • Status="live" — может быть выбран real-лидером селектором.
    """
    NAME = "CandlePatternAgent"
    AGG_BARS = 5             # 1 synthetic свеча = 5 минут
    HISTORY_MAX = 600        # держим 600 минут = 120 synthetic свечей
    CHECK_INT = AGG_BARS     # проверяем каждый раз когда формируется новая свеча
    MIN_BODY_RATIO = 0.30    # тело >= 30 % range — иначе свеча "shaky"
    DOJI_BODY_RATIO = 0.10   # тело <= 10 % range = doji
    HAMMER_TAIL_RATIO = 2.0  # tail >= 2x body
    MAX_POS = 3
    ENTRY_COOLDOWN = 60
    STOP = 0.012
    TARGET = 0.020
    HOLD = 3 * 60            # 3 часа max

    def __init__(self):
        self.h: Dict[str, deque] = {}
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
            self.last_entry[k] = 0

    @staticmethod
    def _ohlc(prices_window):
        """Из последовательности close-prices строит synthetic OHLC."""
        op = float(prices_window[0])
        cl = float(prices_window[-1])
        hi = float(max(prices_window))
        lo = float(min(prices_window))
        return op, hi, lo, cl

    def _candles(self, h, n_candles: int):
        """Возвращает последние n_candles synthetic OHLC из price history."""
        h_list = list(h)
        agg = self.AGG_BARS
        if len(h_list) < agg * n_candles:
            return None
        out = []
        for i in range(n_candles, 0, -1):
            window = h_list[-(agg * i):-(agg * (i - 1))] if i > 1 else h_list[-agg:]
            if len(window) < agg:
                return None
            out.append(self._ohlc(window))
        return out  # хронологически: out[0] — самая старая, out[-1] — последняя

    @staticmethod
    def _body(o, h, l, c):
        body = abs(c - o)
        rng = max(h - l, 1e-9)
        return body, rng, body / rng

    @staticmethod
    def _is_bull(o, c):
        return c > o

    @staticmethod
    def _is_bear(o, c):
        return c < o

    def _bullish_engulfing(self, prev, cur):
        """Bull engulfing с допуском на gap: prev bear + cur bull, cur.close >= prev.open,
        и cur тело >= 1.2x prev тела. Жёсткое условие cur.open <= prev.close
        ослаблено, потому что на минутных synthetic OHLC между свечами почти
        всегда micro-gap — и строгое условие никогда бы не сработало."""
        po, ph, pl, pc = prev
        co, ch, cl, cc = cur
        body_prev = abs(pc - po)
        body_cur = abs(cc - co)
        if body_prev <= 1e-9 or body_cur < body_prev * 1.2:
            return False
        return self._is_bear(po, pc) and self._is_bull(co, cc) and cc >= po

    def _bearish_engulfing(self, prev, cur):
        po, ph, pl, pc = prev
        co, ch, cl, cc = cur
        body_prev = abs(pc - po)
        body_cur = abs(cc - co)
        if body_prev <= 1e-9 or body_cur < body_prev * 1.2:
            return False
        return self._is_bull(po, pc) and self._is_bear(co, cc) and cc <= po

    def _hammer(self, cur):
        o, h, l, c = cur
        body, rng, br = self._body(o, h, l, c)
        if rng <= 1e-9 or br < 0.10:
            return False
        upper = h - max(o, c)
        lower = min(o, c) - l
        # Hammer: длинная нижняя тень + маленькая верхняя.
        return lower >= self.HAMMER_TAIL_RATIO * body and upper <= 0.5 * body

    def _shooting_star(self, cur):
        o, h, l, c = cur
        body, rng, br = self._body(o, h, l, c)
        if rng <= 1e-9 or br < 0.10:
            return False
        upper = h - max(o, c)
        lower = min(o, c) - l
        return upper >= self.HAMMER_TAIL_RATIO * body and lower <= 0.5 * body

    def _morning_star(self, c1, c2, c3):
        """Morning star: длинная bear → маленькое тело star → длинная bull.
        Star-тело < 35 % bear-тела, и cur close доходит хотя бы до середины bear-тела.
        Сравниваем размеры тел между собой, не через body/range."""
        o1, h1, l1, cc1 = c1
        o3, h3, l3, cc3 = c3
        body1 = abs(cc1 - o1)
        body2 = abs(c2[3] - c2[0])
        body3 = abs(cc3 - o3)
        if body1 <= 1e-9 or body3 <= 1e-9:
            return False
        return (self._is_bear(o1, cc1) and self._is_bull(o3, cc3) and
                body2 < body1 * 0.35 and
                body3 >= body1 * 0.7 and
                cc3 > (o1 + cc1) / 2)

    def _evening_star(self, c1, c2, c3):
        o1, h1, l1, cc1 = c1
        o3, h3, l3, cc3 = c3
        body1 = abs(cc1 - o1)
        body2 = abs(c2[3] - c2[0])
        body3 = abs(cc3 - o3)
        if body1 <= 1e-9 or body3 <= 1e-9:
            return False
        return (self._is_bull(o1, cc1) and self._is_bear(o3, cc3) and
                body2 < body1 * 0.35 and
                body3 >= body1 * 0.7 and
                cc3 < (o1 + cc1) / 2)

    def _detect(self, candles):
        """Возвращает 'long' / 'short' / None по сумме весов паттернов.

        Веса:
          • Engulfing (2-свечный, классический разворот) = 1.0
          • Hammer / Shooting Star (1-свечный, на synthetic OHLC слабее) = 0.5
          • Morning / Evening Star (3-свечный, сильнейший) = 1.5
        Порог входа: score >= 1.0 (т.е. engulfing один или ≥2 малых).
        """
        if not candles or len(candles) < 3:
            return None
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        bull_score = 0.0
        bear_score = 0.0
        if self._bullish_engulfing(c2, c3):
            bull_score += 1.0
        if self._bearish_engulfing(c2, c3):
            bear_score += 1.0
        if self._hammer(c3):
            bull_score += 0.5
        if self._shooting_star(c3):
            bear_score += 0.5
        if self._morning_star(c1, c2, c3):
            bull_score += 1.5
        if self._evening_star(c1, c2, c3):
            bear_score += 1.5
        if bull_score >= 1.0 and bull_score > bear_score:
            return ('long', bull_score)
        if bear_score >= 1.0 and bear_score > bull_score:
            return ('short', bear_score)
        return None

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for s, p in prices.items():
            self.h.setdefault(s, deque(maxlen=self.HISTORY_MAX))
            self.pos.setdefault(s, None)
            self.ep.setdefault(s, 0.0)
            self.et.setdefault(s, 0)
            self.last_entry.setdefault(s, -9999)
            self.h[s].append(float(p))
        self.t = bar_index if bar_index is not None else self.t + 1
        actions = {s: 0 for s in prices}
        if self.t - self._lc < self.CHECK_INT:
            return actions
        self._lc = self.t

        n_open = sum(1 for v in self.pos.values() if v is not None)
        candidates = []
        for sym in prices:
            cur_pos = self.pos[sym]
            px = float(prices[sym])
            ep = self.ep[sym]
            held = self.t - self.et[sym]
            # Exit-логика: SL/TP/Hold
            if cur_pos == 'long':
                if px <= ep * (1 - self.STOP) or px >= ep * (1 + self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                    continue
            elif cur_pos == 'short':
                if px >= ep * (1 + self.STOP) or px <= ep * (1 - self.TARGET) or held >= self.HOLD:
                    actions[sym] = 8
                    self.pos[sym] = None
                    n_open = max(0, n_open - 1)
                    continue
            if cur_pos is not None:
                continue
            # Cooldown между входами
            if self.t - self.last_entry[sym] < self.ENTRY_COOLDOWN:
                continue
            # Реконструируем последние 3 synthetic свечи
            candles = self._candles(self.h[sym], 3)
            if candles is None:
                continue
            sig = self._detect(candles)
            if sig is None:
                continue
            side, score = sig
            edge = float(score)  # ranking — морnar/evening star (1.5) выигрывает над hammer (0.5)
            if side == 'long':
                candidates.append((edge, sym, 4, 'long', px))   # 4 = fut_long_half
            else:
                candidates.append((edge, sym, 6, 'short', px))  # 6 = fut_short_half

        slots = max(0, self.MAX_POS - n_open)
        for _, sym, action, side, px in _take_ranked_entries(candidates, slots):
            actions[sym] = action
            self.pos[sym] = side
            self.ep[sym] = px
            self.et[sym] = self.t
            self.last_entry[sym] = self.t
        return actions


class _SoloPlayerWrapper:
    """
    FIX H1 (2026-04-27): тонкий wrapper, превращающий одиночного агента
    (FundingArb, LiveVolCompress, ...) в player-уровень для shadow-pool.

    Мотивация: 12 из 14 проанализированных сессий показали, что лучший shadow-AGENT
    в среднем обыгрывает реального Panteon на ~3 пп. Селектор не мог выбрать
    одиночек напрямую, потому что pool содержал только Panteon-style ансамбли,
    а внутри ансамбля (champion_blend: selected 70% + challenger 20% + rest 10%)
    одиночка получал ≤10% веса и тонул в голосовании коррелированных
    Pantheon-копий (внутрикластерная корреляция действий ≥80% по all_signals.csv).

    Безопасность:
    - PLAYER_STATUS = "live" — селектор имеет право выбрать этот player лидером.
    - act() делегируется внутреннему агенту 1-в-1; его внутреннее состояние pos
      живёт раздельно от self._open_pos real Panteon. Дубль-open и close-when-empty
      фильтруются стандартным _act_via_shadow_players-фильтром (см.
      project_panteon_action_filter в auto-memory).
    - PLAYER_MIN_LIVE_CLOSED_TRADES (=5) защищает от лаки-первого-трейда.
    """
    PLAYER_STATUS = "live"
    NAME = "_SoloBase"

    def __init__(self, agent_factory, name: str):
        self._inner = agent_factory()
        self.NAME = name

    def reset_for_live(self, bar_index: int = 0):
        reset = getattr(self._inner, "reset_for_live", None)
        if callable(reset):
            try:
                reset(bar_index)
            except Exception:
                pass
        # Принудительный сброс позиций sub-agent'а (на случай, если у агента
        # не определён reset_for_live).
        for attr in ("pos", "ep", "entry_px", "et"):
            d = getattr(self._inner, attr, None)
            if isinstance(d, dict):
                for k in d:
                    d[k] = None if attr == "pos" else 0.0
        if hasattr(self._inner, "_lc"):
            try:
                self._inner._lc = -99999
            except Exception:
                pass

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        return self._inner.act(
            prices, volumes,
            month=month,
            portfolio_value=portfolio_value,
            bar_index=bar_index,
        )


class SoloFundingArb(_SoloPlayerWrapper):
    def __init__(self):
        super().__init__(FundingArb, "SoloFundingArb")


class SoloLiveVolCompress(_SoloPlayerWrapper):
    def __init__(self):
        super().__init__(LiveVolCompress, "SoloLiveVolCompress")


class SoloLiveRegimePullback(_SoloPlayerWrapper):
    def __init__(self):
        super().__init__(LiveRegimePullback, "SoloLiveRegimePullback")


class SoloLiveTrendFollow(_SoloPlayerWrapper):
    def __init__(self):
        super().__init__(LiveTrendFollow, "SoloLiveTrendFollow")


class SoloLiveCrashHunter(_SoloPlayerWrapper):
    def __init__(self):
        super().__init__(LiveCrashHunter, "SoloLiveCrashHunter")


class SoloCandlePattern(_SoloPlayerWrapper):
    """FIX (2026-04-29): обёртка над CandlePatternAgent для shadow-pool."""
    def __init__(self):
        super().__init__(CandlePatternAgent, "SoloCandlePattern")


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
    MAX_POS   = 8
    ENABLE_META_PLAYERS = True
    PLAYER_ROTATION_INT = 30
    # FIX 2026-05-04 (faster lineup rotation): cooldown 90→30, hard_neg −1.25→−0.50,
    # margin 0.45→0.30. Без этого Пантеон 1.5 часа держался за лидера с pnl=−0.30%
    # пока тот формально не достигнет −1.25. Теперь смена быстрее.
    PLAYER_SWITCH_COOLDOWN_BARS = 30
    PLAYER_SWITCH_CONFIRMATIONS = 2
    PLAYER_SWITCH_MARGIN = 0.30           # FIX 2026-05-04: 0.45→0.30 — быстрее уходим от плохого лидера
    PLAYER_MIN_SCORE_TO_SWITCH = -0.05    # FIX 2026-05-04: 0.05→−0.05 — допускаем смену на чуть-минусового, если он лучший
    PLAYER_HARD_NEGATIVE_SCORE = -0.50    # FIX 2026-05-04: −1.25→−0.50 — раньше срабатывало только при катастрофе
    SYMBOL_PLAYER_SWITCH_MARGIN = 0.30    # FIX 2026-05-04: 0.42→0.30 — быстрая ротация на per-symbol
    # FIX: гистерезис режима против частых флипов bullish↔neutral
    # (было ~15-60 сек между переключениями во время warmup).
    # 12 баров × 1 мин = 12 мин устойчивости перед сменой режима.
    REGIME_HYSTERESIS_BARS = 12
    # FIX: порог PnL ниже которого игрок не может быть выбран для реальной
    # торговли, даже если у него высокая активность.
    PLAYER_HARD_NEGATIVE_PNL = -1.5
    # FIX A2 (2026-04-26) + FIX 2026-05-04: было 5 — блокировало ярких новичков
    # с локально доказанной прибылью в текущем режиме. Снижено до 3, плюс
    # в agent_meta.select() добавлен «pass» если per_regime[regime].pnl_pct>0
    # при closed>=2.
    PLAYER_MIN_LIVE_CLOSED_TRADES = 3
    # FIX B1/B3 (2026-04-26): режим работы ансамбля shadow-игроков
    #   "single"          — текущее поведение (один selected игрок исполняет всё)
    #   "champion_blend"  — selected = 70%, challenger = 20%, прочие = 10%
    #   "risk_parity"     — веса игроков ~ 1/vol(equity), нормированы
    ENSEMBLE_MODE = "champion_blend"
    ENSEMBLE_PURGATORY_PNL = -1.5
    ENSEMBLE_PURGATORY_BARS = 240
    ENSEMBLE_BLEND_WEIGHTS = {"selected": 0.70, "challenger": 0.20, "rest": 0.10}
    # FIX B4: per-symbol лидер должен иметь хотя бы N закрытых сделок по символу
    PER_SYMBOL_MIN_CLOSED = 3
    PER_SYMBOL_MIN_WIN_RATE = 50.0
    # FIX 2026-05-03 (post-variant-2): grace-окно после reset_for_live
    # — сколько баров после старта live полностью пропускать гейт
    # `_symbol_live_allowed`. Нужно для bootstrap: пока shadow-игроки
    # не накопили закрытых сделок по конкретным символам, гейт по
    # умолчанию глушит почти всё. 60 бар ≈ 1 час 1m-таймфрейма.
    LIVE_GATE_GRACE_BARS = 60

    # FIX v7: символы из FUTURES_BLACKLIST — агент не должен их торговать,
    # потому что close_all() молча не работает для них → маржа блокируется навечно
    _BLACKLIST = frozenset({
        "EUR", "USD1", "USDT", "USDC", "STABLE", "BUSD",
        "GOLD(XAUT)", "GOLD(PAXG)", "FIDA", "WXT", "PSAI", "PE",
        "ATLA", "META", "ONT", "HYPE", "BULLA", "STO", "SOLV", "RED",
    })

    def __init__(self, enable_meta_players: Optional[bool] = None):
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
        self._bra=BullRotationAgent()
        self._brf=BearReliefFadeAgent()
        self._nrs=NeutralRangeScalper()
        self._cps=CrashPanicShortAgent()
        self._pf=PlayerFunding()
        self._gb = None
        self._gbr = None
        self._gn = None
        try:
            try:
                from crypto_genetics import GeneticsBullishAgent
                self._gb = make_genetics_panteon_agent(GeneticsBullishAgent)
            except Exception:
                pass
            try:
                from crypto_genetics import GeneticsBearishAgent
                self._gbr = make_genetics_panteon_agent(GeneticsBearishAgent)
            except Exception:
                pass
            try:
                from crypto_genetics import GeneticsNeutralAgent
                self._gn = make_genetics_panteon_agent(GeneticsNeutralAgent)
            except Exception:
                pass
        except Exception:
            pass

        # FIX C (2026-04-26): новые агенты v2 (см. agents_v2.py).
        self._mrc = None; self._avt = None; self._ces = None
        self._fw  = None; self._gens = None; self._dso = None
        try:
            from agents_v2 import (
                MeanRevConfirmedAgent, AdaptiveVolTrendAgent,
                CrossExchangeSkewAgent, FundingWindowAgent,
                GenomeEnsembleAgent, DefensiveStopOverlayAgent,
            )
            try: self._mrc = MeanRevConfirmedAgent()
            except Exception: pass
            try: self._avt = AdaptiveVolTrendAgent()
            except Exception: pass
            try: self._ces = CrossExchangeSkewAgent()
            except Exception: pass
            try: self._fw  = FundingWindowAgent()
            except Exception: pass
            try: self._gens = GenomeEnsembleAgent()
            except Exception: pass
            try: self._dso = DefensiveStopOverlayAgent()
            except Exception: pass
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
        self._shadow_player_selector = ShadowPlayerMetaSelector(
            REAL_PLAYER_AGGREGATOR_MEMORY_FILE,
            save_interval_bars=30,
        )
        self._shadow_players_enabled = (
            bool(self.ENABLE_META_PLAYERS)
            if enable_meta_players is None
            else bool(enable_meta_players)
        )
        self._shadow_player_pool: OrderedDict[str, object] = OrderedDict()
        self._shadow_player_perf: Optional[Dict[str, dict]] = None
        self._last_shadow_player_snapshot: Dict[str, dict] = {}
        self._shadow_player_regime_memory: Dict[str, Dict[str, dict]] = {}
        self._shadow_player_symbol_memory: Dict[str, Dict[str, Dict[str, dict]]] = {}
        # FIX 2026-05-03 (variant 2): per-bar raw per-symbol stats для
        # _symbol_live_allowed (питается из _publish_player_perf,
        # сериализуется в Real_Player_Aggregator_Memory_*.txt как
        # ключ `player_symbol_memory`). НЕ путать с
        # _shadow_player_symbol_memory выше — там regime-EMA store.
        self._shadow_player_symbol_perf: Dict[str, Dict[str, dict]] = {}
        self._shadow_player_window_anchor: Dict[str, dict] = {}
        self._shadow_player_symbol_anchor: Dict[str, Dict[str, dict]] = {}
        self._selected_shadow_player: str = ""
        self._selected_symbol_shadow_players: Dict[str, str] = {}
        self._shadow_player_scores: Dict[str, float] = {}
        self._shadow_player_challenger: str = ""
        self._shadow_player_challenger_streak = 0
        self._last_shadow_player_switch_bar = -99999
        self._last_shadow_player_rotation = -99999
        self._last_shadow_player_memory_save_bar = -99999
        # FIX: гистерезис режима — инициализация
        self._stable_regime: Optional[str] = None
        self._pending_regime: Optional[str] = None
        self._pending_regime_since: int = -99999
        self._last_regime_switch_bar: int = -99999

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
            'BullRotationAgent':   0.18,
            'BearReliefFadeAgent': 0.18,
            'NeutralRangeScalper': 0.18,
            'CrashPanicShortAgent': 0.18,
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
        self._regime_leaders: Dict[str, str] = self._default_regime_leaders()
        self._regime_leader_scores: Dict[str, float] = {
            regime: 0.0 for regime in self.REGIME_TOPS
        }
        self._last_symbol_regime: Dict[str, str] = {}
        self._shadow_perf: Optional[Dict[str, dict]] = None
        self._shadow_bootstrap_mode = False
        self._suppress_info_logs = False
        if self._shadow_players_enabled:
            self._init_shadow_player_pool()

    def set_external_shadow_player_actions(self, actions_by_player: dict, bar_index: int) -> None:
        """Use shadow-player actions already emitted by the runtime for this bar."""
        try:
            self._external_shadow_player_actions = {
                str(name): dict(actions or {})
                for name, actions in (actions_by_player or {}).items()
            }
            self._external_shadow_player_actions_bar = int(bar_index)
        except Exception:
            self._external_shadow_player_actions = {}
            self._external_shadow_player_actions_bar = -1

    def _init_shadow_player_pool(self):
        if self._shadow_player_pool:
            return
        specs = (
            ("V_Panteon_shadow", Panteon),
            ("V_PlayerFunding", globals().get("PlayerFunding")),
            ("V_PlayerBomberman", globals().get("PlayerBomberman")),
            ("V_PanteonResearch", globals().get("PanteonResearch")),
            ("V_PanteonTrendResearch", globals().get("PanteonTrendResearch")),
            ("V_PanteonMeanRevResearch", globals().get("PanteonMeanRevResearch")),
            ("V_PanteonDefensiveResearch", globals().get("PanteonDefensiveResearch")),
            ("V_PanteonConsensusResearch", globals().get("PanteonConsensusResearch")),
            # FIX (2026-04-26): новый устойчивый игрок
            ("V_PanteonResilient", globals().get("PanteonResilient")),
            ("V_NeuroPlayer", globals().get("NeuroPlayer")),
            # FIX H1 (2026-04-27): солисты в live-pool. До этого фикса все
            # ансамбли в pool были взаимно скоррелированы на ≥80% (см. отчёт),
            # и независимые источники alpha (FundingArb +3.05%, LiveVolCompress
            # +1.69% в shadow MEXC 19h vs PanteonResearch -0.41%) никогда не
            # выбирались селектором лидером — они получали ≤10% веса как rest
            # в champion_blend. Расширение pool солистами восстанавливает
            # право селектора выбрать одиночку.
            ("V_SoloFundingArb",         globals().get("SoloFundingArb")),
            ("V_SoloLiveVolCompress",    globals().get("SoloLiveVolCompress")),
            ("V_SoloLiveRegimePullback", globals().get("SoloLiveRegimePullback")),
            ("V_SoloLiveTrendFollow",    globals().get("SoloLiveTrendFollow")),
            ("V_SoloLiveCrashHunter",    globals().get("SoloLiveCrashHunter")),
            # FIX (2026-04-29): свечной паттерн-агент.
            ("V_SoloCandlePattern",      globals().get("SoloCandlePattern")),
        )
        try:
            from player_next import PanteonNextResearch as _PanteonNextResearch
            specs = specs + (("V_PanteonNextResearch", _PanteonNextResearch),)
        except Exception:
            pass
        for shadow_name, cls in specs:
            if cls is None:
                continue
            try:
                player = cls(enable_meta_players=False) if cls is Panteon else cls()
                if hasattr(player, "set_shadow_bootstrap_mode"):
                    player.set_shadow_bootstrap_mode(self._shadow_bootstrap_mode)
                else:
                    setattr(player, "_shadow_bootstrap_mode", bool(self._shadow_bootstrap_mode))
                if hasattr(player, "set_shadow_perf") and self._shadow_perf:
                    player.set_shadow_perf(self._shadow_perf)
                setattr(player, "_suppress_info_logs", True)
                self._shadow_player_pool[shadow_name] = player
            except Exception as exc:
                log.debug("  [PLAYER meta] %s init skipped: %s", shadow_name, exc)
        if self._shadow_player_pool and not self._selected_shadow_player:
            self._selected_shadow_player = next(iter(self._shadow_player_pool))

    def set_shadow_bootstrap_mode(self, enabled: bool = True):
        enabled_flag = bool(enabled)
        self._shadow_bootstrap_mode = enabled_flag
        self._suppress_info_logs = enabled_flag
        for player in getattr(self, "_shadow_player_pool", {}).values():
            try:
                if hasattr(player, "set_shadow_bootstrap_mode"):
                    player.set_shadow_bootstrap_mode(enabled_flag)
                else:
                    setattr(player, "_shadow_bootstrap_mode", enabled_flag)
            except Exception:
                pass

    def _quiet_info_logs(self) -> bool:
        return bool(getattr(self, "_suppress_info_logs", False))

    def iter_subagents(self):
        """Единый список sub-agent'ов Panteon для reset/logging/live-sync.

        FIX A3 (2026-04-26): НЕ фильтруем здесь LIVE_AGENT_BLOCKLIST.
        Карантинные агенты должны оставаться в pool для shadow-leaderboard и
        копить статистику. Фильтр применяется в местах, где формируются
        live-веса (rotation/portfolio_allocator) и в селекторе.
        """
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
            ('_bra', 'BullRotationAgent'),
            ('_brf', 'BearReliefFadeAgent'),
            ('_nrs', 'NeutralRangeScalper'),
            ('_cps', 'CrashPanicShortAgent'),
            ('_pf', 'PlayerFunding'),
            ('_gb', 'GeneticsBullish'),
            ('_gbr', 'GeneticsBearish'),
            ('_gn', 'GeneticsNeutral'),
            # FIX C (2026-04-26): новые агенты v2
            ('_mrc',  'MeanRevConfirmed'),
            ('_avt',  'AdaptiveVolTrend'),
            ('_ces',  'CrossExchangeSkew'),
            ('_fw',   'FundingWindow'),
            ('_gens', 'GenomeEnsemble'),
            ('_dso',  'DefensiveStopOverlay'),
        ):
            agent = getattr(self, attr_name, None)
            if agent is not None:
                yield attr_name, label, agent

    def agent_status(self, label: str) -> str:
        """Статус агента: live | shadow_only | quarantine | experimental.
        Используется для маркировки в leaderboard.
        """
        if label in self.LIVE_AGENT_BLOCKLIST:
            return "quarantine"
        agent = None
        for _, lbl, ag in self.iter_subagents():
            if lbl == label:
                agent = ag; break
        if agent is None:
            return "live"
        return str(getattr(agent, "AGENT_STATUS", "live") or "live")

    def subagent_labels(self):
        return [label for _, label, _ in self.iter_subagents()]

    # ── Динамический карантин (FIX 2026-05-03 v3) ─────────────────────────
    def recompute_dynamic_quarantine(
        self,
        per_regime_by_label: Dict[str, Dict[str, dict]],
    ) -> Dict[str, str]:
        """Пересчитать живой blocklist агентов на основе per-regime памяти.

        Параметры
        ---------
        per_regime_by_label: dict
            Словарь label -> {regime -> {pnl_pct, closed_trades, ...}}.
            Значения берутся из VirtualPortfolio.export_regime_stats() по
            каждому V_<label> shadow-портфелю. Содержит поля как минимум
            'pnl_pct' и 'closed_trades' для каждого режима.

        Возвращает
        ----------
        decisions: dict label -> 'live' | 'quarantine'
            Решение по каждому агенту. Параллельно обновляет
            self.LIVE_AGENT_BLOCKLIST (frozenset на инстансе) для
            использования всеми остальными местами кода.
        """
        if not isinstance(per_regime_by_label, dict):
            return {}
        decisions: Dict[str, str] = {}
        new_q: set = set()
        # 1) Стартовый seed: всё, что НИКОГДА не торговало в shadow, остаётся
        #    в карантине, чтобы не пускать в бой "чёрный ящик".
        seed = set(getattr(type(self), '_SEED_AGENT_QUARANTINE', frozenset()))
        for label in seed:
            new_q.add(label)
        # 2) Проходим по фактической per-regime статистике.
        try:
            recovery_pnl    = float(self.QUARANTINE_RECOVERY_PNL_PCT)
            recovery_closed = int(self.QUARANTINE_RECOVERY_CLOSED)
            hard_pnl        = float(self.QUARANTINE_HARD_NEG_PNL_PCT)
            hard_min_closed = int(self.QUARANTINE_HARD_MIN_CLOSED)
        except Exception:
            recovery_pnl, recovery_closed = 0.10, 3
            hard_pnl, hard_min_closed = -0.30, 5
        for label, per_regime in per_regime_by_label.items():
            if not isinstance(per_regime, dict) or not per_regime:
                # Нет shadow-данных вообще → не трогаем. Если был в seed —
                # останется в карантине; если нет — останется live.
                if label in seed:
                    decisions[label] = 'quarantine'
                continue
            any_positive = False
            all_negative = True
            total_closed = 0
            worst_pnl    = 0.0
            for regime, stats in per_regime.items():
                if not isinstance(stats, dict):
                    continue
                pnl_pct = float(stats.get('pnl_pct', 0.0) or 0.0)
                closed  = int(stats.get('closed_trades', 0) or 0)
                total_closed += closed
                if pnl_pct >= recovery_pnl and closed >= recovery_closed:
                    any_positive = True
                if pnl_pct > 0.0:
                    all_negative = False
                if pnl_pct < worst_pnl:
                    worst_pnl = pnl_pct
            if any_positive:
                # выпустить из карантина даже если был в seed
                decisions[label] = 'live'
                new_q.discard(label)
                continue
            # Если нет положительного опыта НИ в одном режиме И есть достаточная
            # выборка, и худший pnl ниже HARD-порога → автокарантин.
            if (
                all_negative
                and total_closed >= hard_min_closed
                and worst_pnl <= hard_pnl
            ):
                new_q.add(label)
                decisions[label] = 'quarantine'
            else:
                # Не доказал свою прибыльность, но и плохого нет —
                # держим как было (если был в seed — оставляем там,
                # иначе live).
                if label in seed:
                    decisions[label] = 'quarantine'
                else:
                    decisions[label] = 'live'
        # Обновляем динамический список на инстансе (frozenset, чтобы оно
        # вело себя как раньше там, где `if x in self.LIVE_AGENT_BLOCKLIST`).
        new_q_set = frozenset(new_q)
        self.LIVE_AGENT_BLOCKLIST = new_q_set
        # И копируем в AGENT_QUARANTINE для обратной совместимости.
        self.AGENT_QUARANTINE = new_q_set
        # ── FIX 2026-05-04 (carantine propagation v2) ──
        # Распространяем blocklist на ВСЕ shadow-player инстансы (PanteonResearch,
        # PanteonMeanRevResearch, PanteonNextResearch, PanteonDefensiveResearch,
        # PanteonConsensusResearch, PanteonTrendResearch, V_Panteon_shadow и т.д.).
        # Без этого: когда _selected_shadow_player = PanteonMeanRevResearch,
        # его внутренний ансамбль продолжает использовать карантинные агенты
        # (LiveTrendFollow, MomentumScalper, FundingArb, RichardDennis), и они
        # реально торгуют от имени Пантеона. Это и наблюдалось в логах MEXC
        # 14:39 — boevoi blocklist обновлялся, но `by=[PanteonMeanRevResearch]`
        # сигналы продолжали идти через LiveTrendFollow. Теперь карантин
        # синхронизируется на ВСЕ Panteon-варианты + чистит их _active_weights
        # от заблокированных агентов и принудительно перенормирует веса.
        try:
            for shadow_name, player in (getattr(self, "_shadow_player_pool", {}) or {}).items():
                if player is self:
                    continue
                # Применяем blocklist только к Panteon-родственникам, у которых
                # есть нативный LIVE_AGENT_BLOCKLIST. PlayerFunding/PlayerBomberman/
                # NeuroPlayer/Solo* не имеют вложенного ансамбля агентов — их
                # пропускаем.
                if not hasattr(player, "LIVE_AGENT_BLOCKLIST"):
                    continue
                try:
                    player.LIVE_AGENT_BLOCKLIST = new_q_set
                    player.AGENT_QUARANTINE = new_q_set
                except Exception:
                    continue
                # Чистим текущие active_weights — выкидываем заблокированных
                aw = getattr(player, "_active_weights", None)
                if isinstance(aw, dict) and aw:
                    removed = [
                        label for label in list(aw.keys()) if label in new_q_set
                    ]
                    for label in removed:
                        aw.pop(label, None)
                    total = sum(float(v or 0.0) for v in aw.values())
                    if total > 0 and removed:
                        for label in list(aw.keys()):
                            aw[label] = float(aw[label] or 0.0) / total
                    elif total <= 0 and removed:
                        # Все веса были карантинными → сбрасываем в пустоту,
                        # пусть _rotate_agents в следующем тике перезаполнит.
                        player._active_weights = {}
        except Exception as exc:
            try:
                log.debug("  [quarantine] propagation failed: %s", exc)
            except Exception:
                pass
        return decisions

    # ── Параметры управления позициями ──────────────────────────────────
    SL_PCT      = 0.04   # v5: стоп-лосс 4%
    TP_PCT      = 0.06   # v5: тейк-профит 6%
    TRAIL_PCT   = 0.025  # v5: trailing stop 2.5%
    STALE_BARS  = 720    # v5: stale exit 12ч

    # ── Параметры адаптивной ротации ─────────────────────────────────
    ROTATION_INT  = 30
    MIN_AGENTS    = 2
    MAX_AGENTS    = 4
    MIN_WEIGHT    = 0.10
    SOFT_MIN_WEIGHT = 0.04
    MAX_WEIGHT    = 0.78
    REGIME_PRIORITY_FLOOR = 0.10
    MEMORY_SCHEMA_VERSION = 6
    USE_REGIME_ONLY_MEMORY = True
    AGGREGATION_MODE = 'risk_adjusted_top_positive'
    USE_REGIME_TOP_TRADER = False
    RISK_ADJUSTED_MIN_AGENTS = 2
    RISK_ADJUSTED_MAX_AGENTS = 5
    REGIME_TOPS = (
        'bullish',
        'bearish',
        'neutral',
        'crash',
        'range_low_vol',
        'choppy_down',
        'choppy_up',
        'mixed_rotational',
    )
    REGIME_TOP_MIN_POSITION_TICKS = 12
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
    OPEN_SINGLE_THRESHOLD = 0.28
    OPEN_MULTI_THRESHOLD = 0.21
    MIN_OPEN_SINGLE_THRESHOLD = 0.22
    MIN_OPEN_MULTI_THRESHOLD = 0.16
    LOW_UTIL_OPEN_DISCOUNT = 0.06
    LOW_UTIL_SINGLE_DISCOUNT = 0.05
    TARGET_MARGIN_UTILIZATION = 0.72
    MAX_NEW_PER_BAR = 3
    FULL_SIZE_OPEN_THRESHOLD = 0.42
    FULL_SIZE_MULTI_THRESHOLD = 0.34
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
    MAX_DISCRETIONARY_CLOSES_PER_BAR = 2
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
    # FIX 2026-05-03 (dynamic quarantine v3): карантин теперь ДИНАМИЧЕСКИЙ.
    # Стартовый seed (плохая история на BG/MEXC) — те же три агента, но
    # `Panteon.recompute_dynamic_quarantine()` пересчитывает live-blocklist на
    # каждом тике на основе per-regime памяти shadow-портфелей: если у агента
    # хотя бы в одном из канонических режимов рынка есть
    # положительный накопленный pnl при достаточной выборке (>= QUARANTINE_RECOVERY_CLOSED
    # closed-trades), он автоматически выпускается из карантина и снова имеет
    # право быть live-лидером.  И, наоборот, любой агент/игрок, у которого ВО
    # ВСЕХ режимах накопленный pnl_pct отрицателен при достаточной общей выборке
    # — попадает в карантин автоматически, без ручного редактирования кода.
    _SEED_AGENT_QUARANTINE = frozenset({
        'FundingArb',
        'RichardDennis',
        'MomentumScalper',
    })
    LIVE_AGENT_BLOCKLIST = frozenset(_SEED_AGENT_QUARANTINE)
    AGENT_QUARANTINE = frozenset(_SEED_AGENT_QUARANTINE)
    # Параметры выхода из карантина: достаточно ОДНОГО режима с положительным опытом
    QUARANTINE_RECOVERY_PNL_PCT = 0.10   # >0.10% net PnL в каком-либо режиме
    QUARANTINE_RECOVERY_CLOSED  = 3      # минимум закрытых сделок в этом режиме
    # Параметры попадания в карантин (если ни в одном режиме нет плюса)
    QUARANTINE_HARD_NEG_PNL_PCT = -0.30  # PnL хуже -0.30% во всех режимах
    QUARANTINE_HARD_MIN_CLOSED  = 5      # суммарно закрытых сделок
    AGENT_QUARANTINE_PROBATION_PNL = 0.40
    AGENT_QUARANTINE_PROBATION_CLOSED = 5
    REGIME_STATIC_WEIGHTS = {
        'bullish': {
            'BullRotationAgent': 0.32,
            'LiveAfterShock': 0.34,
            'LiveMeanRev': 0.24,
            'LiveCrashHunter': 0.08,
            'CarryFlowAgentV2': 0.22,
            'LiveOIBreakout': 0.14,
            'LiveVolCompress': 0.12,
            'FundingArb': 0.08,
            'LiveTrendFollow': 0.03,
        },
        'neutral': {
            'NeutralRangeScalper': 0.42,
            'LiveMeanRev': 0.30,
            'LiveVolCompress': 0.28,
            'LiveAfterShock': 0.26,
            'FundingArb': 0.04,
            'CarryFlowAgentV2': 0.20,
            'LiveCrashHunter': 0.06,
            'ResearchValidatorAgent': 0.04,
        },
        'bearish': {
            'BearReliefFadeAgent': 0.32,
            'LiveCrashHunter': 0.12,
            'CarryFlowAgentV2': 0.30,
            'LiveMeanRev': 0.18,
            'FundingArb': 0.04,
            'LiveAfterShock': 0.18,
            'LiveOIBreakout': 0.14,
        },
        'crash': {
            # Start crash as regular bearish. It will diverge through
            # regime_memory after real crash observations are collected.
            'CrashPanicShortAgent': 0.36,
            'LiveCrashHunter': 0.12,
            'CarryFlowAgentV2': 0.30,
            'LiveMeanRev': 0.18,
            'FundingArb': 0.04,
            'LiveAfterShock': 0.18,
            'LiveOIBreakout': 0.14,
        },
    }
    REGIME_STATIC_WEIGHTS['range_low_vol'] = dict(REGIME_STATIC_WEIGHTS['neutral'])
    REGIME_STATIC_WEIGHTS['choppy_down'] = dict(REGIME_STATIC_WEIGHTS['bearish'])
    REGIME_STATIC_WEIGHTS['choppy_up'] = dict(REGIME_STATIC_WEIGHTS['bullish'])
    REGIME_STATIC_WEIGHTS['mixed_rotational'] = dict(REGIME_STATIC_WEIGHTS['neutral'])

    REGIME_PRIORITY_MAP = {
        'bullish': ('LiveAfterShock', 'BullRotationAgent', 'LiveOIBreakout', 'CarryFlowAgentV2', 'LiveVolCompress', 'LiveRegimePullback', 'LiveTrendFollow', 'LiveCrashHunter'),
        'bearish': ('CarryFlowAgentV2', 'BearReliefFadeAgent', 'LiveOIBreakout', 'LiveAfterShock', 'LiveCrashHunter', 'LiveTrendFollow'),
        'neutral': ('NeutralRangeScalper', 'LiveVolCompress', 'LiveAfterShock', 'CarryFlowAgentV2', 'LiveMeanRev', 'LiveOIBreakout', 'LiveCrashHunter', 'LiveRegimePullback'),
        'crash': ('CrashPanicShortAgent', 'CarryFlowAgentV2', 'LiveOIBreakout', 'LiveAfterShock', 'LiveCrashHunter', 'LiveTrendFollow'),
    }
    REGIME_PRIORITY_MAP['range_low_vol'] = REGIME_PRIORITY_MAP['neutral']
    REGIME_PRIORITY_MAP['choppy_down'] = REGIME_PRIORITY_MAP['bearish']
    REGIME_PRIORITY_MAP['choppy_up'] = REGIME_PRIORITY_MAP['bullish']
    REGIME_PRIORITY_MAP['mixed_rotational'] = REGIME_PRIORITY_MAP['neutral']

    REGIME_SCORE_BONUS = {
        'bullish': {
            'BullRotationAgent': 0.24,
            'LiveCrashHunter': -0.04,
            'LiveTrendFollow': -0.04,
            'LiveOIBreakout': 0.16,
            'CarryFlowAgentV2': 0.14,
            'ResearchValidatorAgent': 0.08,
            'VolBreakoutHunter': 0.06,
            'MomentumScalper': 0.14,
            'LiveRegimePullback': -0.02,
            'RichardDennis': 0.08,
            'FundingArb': 0.04,
            'BearReliefFadeAgent': -0.10,
            'NeutralRangeScalper': -0.04,
            'CrashPanicShortAgent': -0.14,
            'LiveVolCompress': 0.08,
            'LiveMeanRev': -0.08,
            'LiveAfterShock': 0.16,
        },
        'bearish': {
            'BearReliefFadeAgent': 0.24,
            'CarryFlowAgentV2': 0.14,
            'LiveTrendFollow': -0.04,
            'LiveOIBreakout': 0.14,
            'ResearchValidatorAgent': 0.06,
            'LiveCrashHunter': -0.02,
            'FundingArb': -0.02,
            'RichardDennis': 0.10,
            'MomentumScalper': -0.04,
            'CrashPanicShortAgent': 0.06,
            'BullRotationAgent': -0.12,
            'NeutralRangeScalper': -0.04,
            'LiveVolCompress': -0.08,
            'LiveAfterShock': 0.12,
        },
        'neutral': {
            'NeutralRangeScalper': 0.24,
            'CarryFlowAgentV2': 0.16,
            'LiveOIBreakout': 0.12,
            'ResearchValidatorAgent': 0.05,
            'VolBreakoutHunter': 0.04,
            'FundingArb': -0.02,
            'LiveCrashHunter': -0.04,
            'MomentumScalper': -0.04,
            'LiveRegimePullback': -0.02,
            'RichardDennis': 0.04,
            'BullRotationAgent': -0.04,
            'BearReliefFadeAgent': -0.04,
            'CrashPanicShortAgent': -0.14,
            'LiveMeanRev': -0.06,
            'LiveAfterShock': 0.14,
            'LiveVolCompress': 0.16,
        },
        'crash': {
            'CrashPanicShortAgent': 0.30,
            'CarryFlowAgentV2': 0.14,
            'LiveTrendFollow': -0.04,
            'LiveOIBreakout': 0.14,
            'ResearchValidatorAgent': 0.06,
            'LiveCrashHunter': -0.02,
            'FundingArb': -0.02,
            'RichardDennis': 0.10,
            'MomentumScalper': -0.04,
            'BearReliefFadeAgent': 0.06,
            'BullRotationAgent': -0.16,
            'NeutralRangeScalper': -0.12,
            'LiveVolCompress': -0.10,
            'LiveAfterShock': 0.12,
        },
    }
    REGIME_SCORE_BONUS['range_low_vol'] = dict(REGIME_SCORE_BONUS['neutral'])
    REGIME_SCORE_BONUS['choppy_down'] = dict(REGIME_SCORE_BONUS['bearish'])
    REGIME_SCORE_BONUS['choppy_up'] = dict(REGIME_SCORE_BONUS['bullish'])
    REGIME_SCORE_BONUS['mixed_rotational'] = dict(REGIME_SCORE_BONUS['neutral'])

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

        FIX 2026-04-27: первые 60 live-баров после reset_for_live игнорируем
        crash-trigger — переход с архивных warmup-цен на свежие live-цены
        часто даёт ложное -1..2% short_med, и бот зависает в crash.
        """
        live_started_at = int(getattr(self, "_live_start_bar", 0) or 0)
        bars_since_live = (int(getattr(self, "_t", 0) or 0) - live_started_at) if live_started_at else 99999
        suppress_crash = bool(getattr(self, "_live_started", False)) and bars_since_live < 60

        short_rets = self._collect_universe_returns(60)
        mid_rets = self._collect_universe_returns(240)
        if not suppress_crash and (len(short_rets) >= 6 or len(mid_rets) >= 6):
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
        normalized = key.replace('-', '_').replace(' ', '_').replace('/', '_')
        if key in {'bull', 'bullish', 'up', 'long'}:
            return 'bullish'
        if key in {'bear', 'bearish', 'down', 'short'}:
            return 'bearish'
        if key in {'crash', 'panic', 'capitulation'}:
            return 'crash'
        if normalized in {'range_low_vol', 'low_vol_range', 'low_vol'}:
            return 'range_low_vol'
        if normalized in {'choppy_down', 'chop_down', 'volatile_down'}:
            return 'choppy_down'
        if normalized in {'choppy_up', 'chop_up', 'volatile_up'}:
            return 'choppy_up'
        if normalized in {'mixed_rotational', 'mixed_rotation', 'rotational', 'mixed'}:
            return 'mixed_rotational'
        return 'neutral'

    def _canonical_symbol_regime(self, regime: Optional[str]) -> str:
        key = str(regime or '').strip().lower()
        normalized = key.replace('-', '_').replace(' ', '_').replace('/', '_')
        if key in {'bull', 'bullish', 'up', 'long'}:
            return 'bullish'
        if key in {'bear', 'bearish', 'down', 'short'}:
            return 'bearish'
        if key in {'crash', 'panic', 'capitulation'}:
            return 'crash'
        if normalized in {'range_low_vol', 'low_vol_range', 'low_vol'}:
            return 'range_low_vol'
        if normalized in {'choppy_down', 'chop_down', 'volatile_down'}:
            return 'choppy_down'
        if normalized in {'choppy_up', 'chop_up', 'volatile_up'}:
            return 'choppy_up'
        if normalized in {'mixed_rotational', 'mixed_rotation', 'rotational', 'mixed'}:
            return 'mixed_rotational'
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
        'V_GeneticsNeutral': 'GeneticsNeutral',
        # Shadow candidate: новый агент (volatility breakout). Живые веса
        # получит через promotion_gate после набора достаточной статистики.
        'V_VolBreakoutHunter': 'VolBreakoutHunter',
        'V_BullRotationAgent': 'BullRotationAgent',
        'V_BearReliefFadeAgent': 'BearReliefFadeAgent',
        'V_NeutralRangeScalper': 'NeutralRangeScalper',
        'V_CrashPanicShortAgent': 'CrashPanicShortAgent',
        'V_PlayerFunding': 'PlayerFunding',
        # FIX C (2026-04-26): новые агенты v2
        'V_MeanRevConfirmed':     'MeanRevConfirmed',
        'V_AdaptiveVolTrend':     'AdaptiveVolTrend',
        'V_CrossExchangeSkew':    'CrossExchangeSkew',
        'V_FundingWindow':        'FundingWindow',
        'V_GenomeEnsemble':       'GenomeEnsemble',
        'V_DefensiveStopOverlay': 'DefensiveStopOverlay',
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

    def _default_regime_leaders(self) -> Dict[str, str]:
        leaders: Dict[str, str] = {}
        for regime in self.REGIME_TOPS:
            for label in self.REGIME_PRIORITY_MAP.get(regime, ()):
                if label in self._agent_pool and label not in self.LIVE_AGENT_BLOCKLIST:
                    leaders[regime] = label
                    break
        fallback_labels = list(self._active_weights) + list(self._agent_pool)
        for regime in self.REGIME_TOPS:
            if regime in leaders:
                continue
            for label in fallback_labels:
                if label in self._agent_pool and label not in self.LIVE_AGENT_BLOCKLIST:
                    leaders[regime] = label
                    break
        return leaders

    def _symbol_execution_regime(self, sym: str, fallback: Optional[str] = None) -> str:
        regime = self._detect_symbol_regime(sym, list(self._ph.get(sym) or []))
        if regime == 'unknown':
            regime = self._canonical_market_regime(fallback or self._r or 'neutral')
        return self._canonical_symbol_regime(regime)

    def _regime_shadow_score(self, label: str, perf: dict, regime: str) -> Optional[float]:
        per_regime = perf.get('per_regime') if isinstance(perf, dict) else None
        stats = (per_regime or {}).get(regime) if isinstance(per_regime, dict) else None
        if not isinstance(stats, dict):
            return None

        signals = int(stats.get('signals', 0) or 0)
        entries = int(stats.get('entries', 0) or 0)
        closed = int(stats.get('closed_trades', 0) or 0)
        pos_ticks = int(stats.get('position_ticks', 0) or 0)
        if signals <= 0 and entries <= 0 and closed <= 0 and pos_ticks < self.REGIME_TOP_MIN_POSITION_TICKS:
            return None

        pnl = float(stats.get('pnl_pct', 0.0) or 0.0)
        win_rate = float(stats.get('win_rate', 0.0) or 0.0)
        score = pnl
        if closed >= 2:
            score += (win_rate - 50.0) * 0.004
        elif entries > 0:
            score -= 0.05
        score += min(closed, 8) * 0.006
        return float(score)

    def _fallback_regime_leader(self, regime: str) -> Optional[str]:
        ranked = []
        for shadow_name, perf in (self._shadow_perf or {}).items():
            label = self.SHADOW_MAP.get(shadow_name)
            if not label or label not in self._agent_pool or label in self.LIVE_AGENT_BLOCKLIST:
                continue
            if self._agent_currently_blocked(label, perf):
                continue
            pnl = float((perf or {}).get('pnl_pct', 0.0) or 0.0)
            sharpe = float((perf or {}).get('sharpe', 0.0) or 0.0)
            entries = int((perf or {}).get('entries', 0) or 0)
            closed = int((perf or {}).get('closed_trades', (perf or {}).get('total_trades', 0)) or 0)
            if entries <= 0 and closed <= 0:
                continue
            ranked.append((pnl + sharpe * 0.08 + min(closed, 8) * 0.03, label))
        if ranked:
            ranked.sort(reverse=True)
            return ranked[0][1]

        current = self._regime_leaders.get(regime)
        if current in self._agent_pool and current not in self.LIVE_AGENT_BLOCKLIST:
            return current

        defaults = self._default_regime_leaders()
        return defaults.get(regime)

    def _update_regime_leaders(self, logger=None) -> bool:
        if not isinstance(self._shadow_perf, dict) or not self._shadow_perf:
            return False

        old = dict(getattr(self, '_regime_leaders', {}) or {})
        leaders = dict(old or self._default_regime_leaders())
        scores = dict(getattr(self, '_regime_leader_scores', {}) or {})

        for regime in self.REGIME_TOPS:
            ranked = []
            for shadow_name, perf in self._shadow_perf.items():
                label = self.SHADOW_MAP.get(shadow_name)
                if not label or label not in self._agent_pool or label in self.LIVE_AGENT_BLOCKLIST:
                    continue
                score = self._regime_shadow_score(label, perf, regime)
                if score is None:
                    continue
                if score <= 0.0 and self._agent_currently_blocked(label, perf):
                    continue
                stats = ((perf.get('per_regime') or {}).get(regime) or {})
                pnl = float(stats.get('pnl_pct', 0.0) or 0.0)
                ranked.append((score, pnl, label))

            if ranked:
                ranked.sort(reverse=True)
                scores[regime] = float(ranked[0][0])
                leaders[regime] = ranked[0][2]
            else:
                fallback = self._fallback_regime_leader(regime)
                if fallback:
                    leaders[regime] = fallback
                    scores.setdefault(regime, 0.0)

        changed = leaders != old
        self._regime_leaders = leaders
        self._regime_leader_scores = scores

        unique = OrderedDict()
        for regime in self.REGIME_TOPS:
            label = leaders.get(regime)
            if label and label in self._agent_pool:
                unique.setdefault(label, 1.0)
        if unique:
            equal_weight = 1.0 / len(unique)
            self._active_weights = {label: equal_weight for label in unique}

        if changed and logger is not None:
            leader_msg = ", ".join(
                f"{regime}={leaders.get(regime, '-')}"
                for regime in self.REGIME_TOPS
            )
            logger.info("  [REAL regime-top] leaders: %s", leader_msg)
        return changed

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

    def _set_agent_position_state(
        self,
        agent,
        sym: str,
        side: Optional[str],
        entry: float = 0.0,
        bar_index: int = 0,
        _depth: int = 0,
    ) -> None:
        if agent is None or _depth > 5:
            return
        try:
            pos_dict = getattr(agent, 'pos', None)
            if isinstance(pos_dict, dict):
                pos_dict[sym] = side
        except Exception:
            pass
        for attr_name in ('ep', 'entry_px'):
            try:
                entry_dict = getattr(agent, attr_name, None)
                if isinstance(entry_dict, dict):
                    entry_dict[sym] = float(entry or 0.0) if side else 0.0
            except Exception:
                pass
        try:
            et_dict = getattr(agent, 'et', None)
            if isinstance(et_dict, dict):
                et_dict[sym] = int(bar_index or 0) if side else 0
        except Exception:
            pass
        for wrap_attr in ('_inner', '_a', '_agent', 'agent', '_fa', '_ms', '_b1', '_b2'):
            try:
                inner = getattr(agent, wrap_attr, None)
                if inner is not None and inner is not agent:
                    self._set_agent_position_state(
                        inner, sym, side, entry=entry,
                        bar_index=bar_index, _depth=_depth + 1,
                    )
            except Exception:
                pass

    def _sync_subagents_to_real_positions(self, prices: dict, bar_index: int) -> None:
        real_positions = dict(getattr(self, '_open_pos', {}) or {})
        symbols = set(prices or {}) | set(real_positions)
        for _, _label, agent in self.iter_subagents():
            pos_dict = getattr(agent, 'pos', None)
            if isinstance(pos_dict, dict):
                symbols.update(pos_dict.keys())
            for sym in list(symbols):
                info = real_positions.get(sym)
                if info:
                    side = str(info.get('side') or 'long').lower()
                    entry = float(info.get('entry') or prices.get(sym, 0.0) or 0.0)
                    opened_bar = int(info.get('bar', bar_index) or bar_index)
                    self._set_agent_position_state(
                        agent, sym, side, entry=entry, bar_index=opened_bar,
                    )
                else:
                    self._set_agent_position_state(agent, sym, None)

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

    def _account_margin_utilization(self, portfolio_value=None) -> Optional[float]:
        try:
            equity = float(getattr(self, '_last_account_equity', 0.0) or portfolio_value or 0.0)
            available = getattr(self, '_last_available_margin', None)
            if available is None or equity <= 0:
                return None
            available = max(float(available or 0.0), 0.0)
        except (TypeError, ValueError):
            return None
        used = max(equity - available, 0.0)
        return float(np.clip(used / max(equity, 1e-9), 0.0, 1.0))

    def _turnover_pressure(self, portfolio_value=None, n_open: int = 0) -> float:
        target = float(getattr(self, 'TARGET_MARGIN_UTILIZATION', 0.72) or 0.72)
        util = self._account_margin_utilization(portfolio_value)
        if util is not None:
            return float(np.clip((target - util) / max(target, 1e-9), 0.0, 1.0))
        target_open = max(1.0, float(self.MAX_POS) * target)
        return float(np.clip((target_open - float(n_open)) / target_open, 0.0, 1.0))

    def _open_thresholds_for_turnover(
        self,
        open_extra: float,
        turnover_pressure: float,
    ) -> tuple[float, float, float]:
        pressure = float(np.clip(turnover_pressure, 0.0, 1.0))
        discount = float(getattr(self, 'LOW_UTIL_OPEN_DISCOUNT', 0.0) or 0.0) * pressure
        single_discount = float(getattr(self, 'LOW_UTIL_SINGLE_DISCOUNT', 0.0) or 0.0) * pressure
        open_single = max(
            float(getattr(self, 'MIN_OPEN_SINGLE_THRESHOLD', 0.0) or 0.0),
            min(float(self.MAX_WEIGHT), float(self.OPEN_SINGLE_THRESHOLD) + float(open_extra) - discount),
        )
        open_multi = max(
            float(getattr(self, 'MIN_OPEN_MULTI_THRESHOLD', 0.0) or 0.0),
            min(float(self.MAX_WEIGHT), float(self.OPEN_MULTI_THRESHOLD) + float(open_extra) * 0.75 - discount * 0.85),
        )
        single_strong = min(
            float(self.MAX_WEIGHT),
            max(float(self.SINGLE_AGENT_STRONG_THRESHOLD) - single_discount, open_single),
        )
        return float(open_single), float(open_multi), float(single_strong)

    def _shadow_bootstrap_seed_weight(self, label: str) -> float:
        if not getattr(self, "_shadow_bootstrap_mode", False):
            return 0.0
        try:
            seed = float((getattr(self, "_active_weights", {}) or {}).get(label, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return seed if seed > 0.0 else 0.0

    def _single_agent_thresholds_for_label(
        self,
        label: str,
        open_single_threshold: float,
        single_strong_threshold: float,
    ) -> tuple[float, float]:
        open_single = float(open_single_threshold or 0.0)
        single_strong = float(single_strong_threshold or open_single)
        seed = self._shadow_bootstrap_seed_weight(label)
        if seed <= 0.0:
            return open_single, single_strong
        floor = float(getattr(self, "MIN_OPEN_SINGLE_THRESHOLD", 0.0) or 0.0)
        bootstrap_open = max(floor, min(open_single, seed))
        bootstrap_strong = max(bootstrap_open, min(single_strong, seed))
        return float(bootstrap_open), float(bootstrap_strong)

    def _max_new_positions_for_bar(
        self,
        portfolio_value=None,
        n_open: int = 0,
        candidate_count: int = 0,
    ) -> int:
        free_slots = max(int(self.MAX_POS) - int(n_open), 0)
        if free_slots <= 0 or candidate_count <= 0:
            return 0

        pressure = self._turnover_pressure(portfolio_value, n_open)
        desired = 1
        try:
            equity = float(getattr(self, '_last_account_equity', 0.0) or portfolio_value or 0.0)
            available = float(getattr(self, '_last_available_margin', 0.0) or 0.0)
            free_ratio = available / max(equity, 1e-9) if equity > 0 else 0.0
        except (TypeError, ValueError):
            free_ratio = 0.0

        max_new = max(1, int(getattr(self, 'MAX_NEW_PER_BAR', 1) or 1))
        if pressure >= 0.65 or free_ratio >= 0.60:
            desired = max_new
        elif pressure >= 0.30 or free_ratio >= 0.35:
            desired = min(2, max_new)
        if n_open == 0 and candidate_count >= 2:
            desired = max(desired, min(2, max_new))
        return int(min(desired, free_slots, candidate_count))

    def _full_size_open_action(
        self,
        base_action: int,
        score: float,
        support_n: int,
        turnover_pressure: float,
    ) -> int:
        if base_action not in (4, 6):
            return int(base_action)
        pressure = float(np.clip(turnover_pressure, 0.0, 1.0))
        single_threshold = float(getattr(self, 'FULL_SIZE_OPEN_THRESHOLD', 0.42) or 0.42) - 0.04 * pressure
        multi_threshold = float(getattr(self, 'FULL_SIZE_MULTI_THRESHOLD', 0.34) or 0.34) - 0.04 * pressure
        if support_n >= self.MIN_MULTI_AGENT_SUPPORT:
            full_size = float(score) >= max(multi_threshold, self.OPEN_MULTI_THRESHOLD)
        else:
            full_size = float(score) >= max(single_threshold, self.OPEN_SINGLE_THRESHOLD)
        if not full_size:
            return int(base_action)
        return 5 if base_action == 4 else 7

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
        if min_open_threshold is not None:
            strong_threshold = min(
                float(self.MAX_WEIGHT),
                max(
                    base_open_threshold,
                    float(getattr(self, 'MIN_OPEN_SINGLE_THRESHOLD', 0.0) or 0.0),
                ),
            )
        else:
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
        bootstrap_seed = self._shadow_bootstrap_seed_weight(label)

        if self._agent_currently_blocked(label, perf):
            return False

        if closed < self.SINGLE_AGENT_MIN_CLOSED_TRADES:
            required = max(
                strong_threshold,
                min(float(self.MAX_WEIGHT), float(self.UNPROVEN_SINGLE_THRESHOLD)),
            )
            if bootstrap_seed > 0.0:
                required = max(
                    float(getattr(self, 'MIN_OPEN_SINGLE_THRESHOLD', 0.0) or 0.0),
                    min(required, max(strong_threshold, bootstrap_seed)),
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
        if self._shadow_players_enabled and load_existing:
            self._shadow_player_selector.load_snapshot(self, logger=log)

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
        self._shadow_player_selector = ShadowPlayerMetaSelector(
            _aggregator_memory_path_for_namespace(clean),
            save_interval_bars=30,
        )
        self._last_memory_save_bar = -99999
        self._last_shadow_player_memory_save_bar = -99999
        # FIX 2026-05-03 (variant 2): не пробрасываем per-symbol perf
        # из предыдущего namespace (например, переключение BITGET→MEXC).
        self._shadow_player_symbol_perf = {}
        if self._memory_enabled and load_existing:
            self._memory_store.load_snapshot(self, logger=log)
            if self._shadow_players_enabled:
                self._shadow_player_selector.load_snapshot(self, logger=log)

    def _load_memory_snapshot(self):
        self._memory_store.load_snapshot(self, logger=log)

    def set_shadow_perf(self, perf: Dict[str, dict]):
        """Вызывается из COMBO_TRADE для передачи shadow performance."""
        self._shadow_scoring.set_shadow_perf(self, perf)
        for player in getattr(self, "_shadow_player_pool", {}).values():
            try:
                if hasattr(player, "set_shadow_perf"):
                    player.set_shadow_perf(perf)
            except Exception:
                pass

    def set_shadow_player_perf(self, perf: Dict[str, dict]):
        self._shadow_player_selector.set_shadow_player_perf(self, perf)

    def set_shadow_player_symbol_memory(self, perf: Dict[str, dict]):
        """FIX 2026-05-03 (variant 2): обновить per-symbol stats для
        Panteon._symbol_live_allowed из shadow-player perf.

        Вызывается из Panteon_Trade._publish_player_perf один раз за бар.
        Формат `perf` — выход `_build_shadow_perf_from_shadows`, где
        у каждого игрока есть поле `per_symbol` со статами
        `vp.export_symbol_stats(prices)`.
        """
        try:
            built = self._shadow_player_selector._build_player_symbol_perf(
                self, perf,
            )
        except Exception as exc:
            try:
                log.debug("  [PLAYER meta] symbol_perf rebuild err: %s", exc)
            except Exception:
                pass
            return
        # Полная замена, а не merge: per_symbol — кумулятивные счётчики,
        # которые ShadowPlayer держит сам, поэтому свежий снимок уже
        # содержит всю историю. Игроки, по которым в этом баре нет
        # данных, выпадают — это OK, они вернутся через бар.
        self._shadow_player_symbol_perf = built

    def _rotate_shadow_players(self, prices: Optional[dict] = None):
        if not self._shadow_players_enabled:
            return
        self._shadow_player_selector.select(
            self,
            context=self._current_context,
            prices=prices or {},
            logger=log,
        )

    def _shadow_player_static_prior(self, name: str, regime: str) -> float:
        name = str(name or "")
        regime = self._canonical_market_regime(regime)
        priors = {
            "bullish": {
                "V_PanteonTrendResearch": 0.22,
                "V_PanteonResearch": 0.12,
                "V_PanteonNextResearch": 0.08,
                "V_PanteonConsensusResearch": 0.06,
                "V_Panteon_shadow": 0.04,
                "V_PanteonDefensiveResearch": -0.05,
            },
            "bearish": {
                "V_PanteonDefensiveResearch": 0.18,
                "V_PanteonMeanRevResearch": 0.10,
                "V_PanteonConsensusResearch": 0.08,
                "V_PanteonNextResearch": 0.06,
                "V_PanteonResearch": 0.05,
                "V_PanteonTrendResearch": -0.06,
            },
            "neutral": {
                "V_PanteonMeanRevResearch": 0.18,
                "V_PanteonConsensusResearch": 0.12,
                "V_PanteonResearch": 0.08,
                "V_PanteonNextResearch": 0.06,
                "V_Panteon_shadow": 0.04,
            },
            "crash": {
                "V_PanteonDefensiveResearch": 0.24,
                "V_PanteonConsensusResearch": 0.10,
                "V_PanteonNextResearch": 0.06,
                "V_PanteonMeanRevResearch": 0.05,
                "V_PanteonTrendResearch": -0.14,
            },
        }
        return float(priors.get(regime, {}).get(name, 0.0))

    def _sync_shadow_player_state(self, player, portfolio_value=None):
        # Keep delegated players on their own virtual trajectory. Real position
        # safety is applied after act(); copying live _open_pos here made the
        # live Panteon and the shadow leaderboard compare different paths.
        for attr in ("_last_account_equity", "_last_available_margin"):
            if hasattr(self, attr):
                try:
                    setattr(player, attr, getattr(self, attr))
                except Exception:
                    pass
        if portfolio_value is not None:
            try:
                player._last_account_equity = float(portfolio_value)
            except Exception:
                pass
        try:
            player._shadow_perf = self._shadow_perf
            player._last_shadow_snapshot = self._last_shadow_snapshot
        except Exception:
            pass

    def _shadow_player_signal_confidence(self, player_name: str, sym: str = "") -> float:
        score = float((getattr(self, "_shadow_player_scores", {}) or {}).get(player_name, 0.0) or 0.0)
        perf = (getattr(self, "_shadow_player_perf", {}) or {}).get(player_name, {}) or {}
        closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0) if isinstance(perf, dict) else 0
        sample = min(max(closed, 0) / 8.0, 1.0)
        conf = 0.35 + max(score, -1.0) * 0.14 + sample * 0.18
        if sym and player_name == (getattr(self, "_selected_symbol_shadow_players", {}) or {}).get(sym):
            conf += 0.06
        return float(np.clip(conf, 0.20, 1.0))

    def _symbol_live_allowed(self, player_name: str, sym: str) -> bool:
        """Per-symbol live-gate для делегированных shadow-сигналов.

        История правок:
          • 2026-04-30: метод добавлен; trailing `return False` глушил все
            entry-сигналы → real Panteon перестал торговать.
          • 2026-05-03 (variant 1): trailing `return True` — fail-open,
            чтобы немедленно вернуть торговлю.
          • 2026-05-03 (variant 2): подключили реальный источник
            `_shadow_player_symbol_perf` через _publish_player_perf и
            persist в Real_Player_Aggregator_Memory_*.txt.
          • 2026-05-03 (post-variant-2):
              - инвертирован порог: гейт блокирует ТОЛЬКО при
                достаточной выборке (closed >= MIN_CLOSED) И низком
                win-rate. Раньше блокировались все символы с малой
                выборкой и низким wr — это создавало chicken-and-egg
                (shadow-игрок впервые играет → 1 закрытие с 0% wr →
                гейт навсегда блокирует символ).
              - добавлено grace-окно LIVE_GATE_GRACE_BARS после
                reset_for_live: первые ~60 бар все символы пропускаются,
                чтобы shadow-игроки успели набрать статистику.

        Логика:
          1. Grace-окно после live-старта → ALLOW.
          2. Достаточная выборка + низкий win-rate → BLOCK
             (явно убыточный игрок на этом символе).
          3. Иначе (мало данных или прибыльный) → ALLOW.
          4. Полное отсутствие данных в любых сторах → ALLOW + WARNING
             (degraded-режим, см. variant 1 backstop).
        """
        # 1) Grace-window: пока shadow-игроки не успели накопить closed_trades
        # по символам, гейт пропускает всё. Защита от bootstrap-deadlock.
        grace_bars = int(getattr(self, "LIVE_GATE_GRACE_BARS", 60) or 0)
        live_start = int(getattr(self, "_live_start_bar", 0) or 0)
        cur_bar = int(getattr(self, "_t", 0) or 0)
        bars_since_live = cur_bar - live_start
        if grace_bars > 0 and 0 <= bars_since_live < grace_bars:
            if not getattr(self, "_symbol_gate_grace_logged", False):
                try:
                    log.info(
                        "  [PLAYER meta] symbol live-gate in grace window (%d bars) — allowing all entries until shadow-stats accumulate",
                        grace_bars,
                    )
                except Exception:
                    pass
                try:
                    self._symbol_gate_grace_logged = True
                except Exception:
                    pass
            return True

        min_closed = int(getattr(self, "PER_SYMBOL_MIN_CLOSED", 3) or 3)
        min_wr = float(getattr(self, "PER_SYMBOL_MIN_WIN_RATE", 50.0) or 50.0)
        names = [str(player_name or ""), str(player_name or "").replace("V_", "")]
        # variant 2: первый источник — свежий per-bar снимок
        # `_shadow_player_symbol_perf`, питается из
        # set_shadow_player_symbol_memory. Старые сторы оставлены
        # fallback-ом для первых секунд до публикации perf.
        stores = (
            getattr(self, "_shadow_player_symbol_perf", {}) or {},
            getattr(self, "_shadow_player_symbol_memory", {}) or {},
            getattr(self, "_symbol_memory", {}) or {},
        )
        for store in stores:
            for name in names:
                stats = ((store.get(name) or {}) if isinstance(store, dict) else {}).get(sym, {})
                if not isinstance(stats, dict) or not stats:
                    continue
                closed = int(stats.get("closed_trades", stats.get("total_trades", stats.get("samples", 0))) or 0)
                win_rate = float(stats.get("win_rate", stats.get("win_rate_pct", 0.0)) or 0.0)
                # FIX 2026-05-03 (post-variant-2): инверсия порога.
                # Блокируем только когда есть ДОСТАТОЧНАЯ выборка
                # И player реально проигрывает на этом символе.
                # Малая выборка = "ещё не доказано" → ALLOW.
                if closed >= min_closed and win_rate < min_wr:
                    return False
                return True
        # Нет per-symbol истории → fail-open + одноразовый WARNING.
        if not getattr(self, "_symbol_gate_degraded_warned", False):
            try:
                log.warning(
                    "  [PLAYER meta] symbol live-gate degraded: no per-symbol memory for %s — fail-open",
                    player_name,
                )
            except Exception:
                pass
            try:
                self._symbol_gate_degraded_warned = True
            except Exception:
                pass
        return True

    def _shadow_player_risk_multiplier(self, player_name: str, action: int, confidence: float) -> float:
        if action not in (1, 2, 4, 5, 6, 7):
            return 1.0
        # Shadow-player delegation must stay position-size comparable with the
        # shadow leaderboard. Let player selection decide direction; do not add
        # another live-only leverage layer on top of the selected player.
        return 1.0

    # FIX B3 (2026-04-26): purgatory bookkeeping для убыточных игроков.
    def _ensemble_purgatory(self, current_bar: int) -> set:
        purgatory = getattr(self, "_player_purgatory_until", None)
        if purgatory is None:
            purgatory = {}
            self._player_purgatory_until = purgatory
        purg_pnl = float(getattr(self, "ENSEMBLE_PURGATORY_PNL", -1.5) or -1.5)
        purg_bars = int(getattr(self, "ENSEMBLE_PURGATORY_BARS", 240) or 240)
        perf_map = getattr(self, "_shadow_player_perf", {}) or {}
        if isinstance(perf_map, dict):
            for name, perf in perf_map.items():
                if not isinstance(perf, dict):
                    continue
                pnl = float(perf.get("pnl_pct", 0.0) or 0.0)
                closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
                if closed >= 2 and pnl <= purg_pnl and name not in purgatory:
                    purgatory[name] = current_bar + purg_bars
        for name in list(purgatory.keys()):
            if current_bar >= purgatory[name]:
                purgatory.pop(name, None)
        return set(purgatory.keys())

    # FIX B1/B2/B3 (2026-04-26): расчёт весов игроков для ensemble-голосования.
    def _ensemble_weights(self, regime: str) -> Dict[str, float]:
        mode = str(getattr(self, "ENSEMBLE_MODE", "single") or "single").lower()
        pool = getattr(self, "_shadow_player_pool", {}) or {}
        if not pool:
            return {}
        eligible = []
        for name, instance in pool.items():
            status = str(getattr(instance, "PLAYER_STATUS", "live") or "live").lower()
            if status in ("shadow_only", "quarantine"):
                continue
            eligible.append(name)
        if not eligible:
            return {}
        if mode == "single":
            sel = self._selected_shadow_player
            return {sel: 1.0} if sel in eligible else {eligible[0]: 1.0}
        scores = getattr(self, "_shadow_player_scores", {}) or {}
        perf_map = getattr(self, "_shadow_player_perf", {}) or {}
        if mode == "champion_blend":
            blend = getattr(self, "ENSEMBLE_BLEND_WEIGHTS",
                            {"selected": 0.70, "challenger": 0.20, "rest": 0.10})
            sel = self._selected_shadow_player if self._selected_shadow_player in eligible else eligible[0]
            ranked = sorted(
                ((scores.get(n, -10.0), n) for n in eligible if n != sel),
                key=lambda x: x[0], reverse=True,
            )
            challenger = ranked[0][1] if ranked else None
            rest = [n for _, n in ranked[1:]]
            weights: Dict[str, float] = {}
            weights[sel] = float(blend.get("selected", 0.70))
            if challenger is not None:
                weights[challenger] = float(blend.get("challenger", 0.20))
            if rest:
                share = float(blend.get("rest", 0.10)) / max(len(rest), 1)
                for n in rest:
                    weights[n] = share
            min_score = float(getattr(self, "PLAYER_HARD_NEGATIVE_SCORE", -1.25) or -1.25)
            weights = {n: w for n, w in weights.items()
                       if scores.get(n, 0.0) > min_score or n == sel}
        elif mode == "risk_parity":
            inv = {}
            for n in eligible:
                p = perf_map.get(n) or {}
                dd = abs(float(p.get("max_drawdown_pct", p.get("max_dd", 0.0)) or 0.0))
                pnl = float(p.get("pnl_pct", 0.0) or 0.0)
                vol_proxy = max(dd, 0.5)
                w = 1.0 / vol_proxy
                if pnl < 0.0:
                    w *= max(0.1, 1.0 + pnl * 0.20)
                inv[n] = max(w, 0.05)
            total = sum(inv.values())
            weights = {n: w / total for n, w in inv.items()} if total > 0 else {}
        else:
            sel = self._selected_shadow_player if self._selected_shadow_player in eligible else eligible[0]
            weights = {sel: 1.0}
        purg = self._ensemble_purgatory(int(getattr(self, "_t", 0) or 0))
        if purg:
            for n in list(weights.keys()):
                if n in purg:
                    weights.pop(n, None)
        total = sum(weights.values())
        if total > 0:
            weights = {n: w / total for n, w in weights.items()}
        return weights

    @staticmethod
    def _action_class(a: int) -> str:
        if a in (1, 2): return "spot_buy"
        if a == 3: return "spot_sell"
        if a in (4, 5): return "fut_long"
        if a in (6, 7): return "fut_short"
        if a == 8: return "fut_close"
        return ""

    def _act_via_shadow_players(self, prices, volumes, month=None,
                                portfolio_value=None, bar_index=None,
                                regime: str = "neutral") -> dict:
        if not self._shadow_player_pool:
            return {s: 0 for s in prices}

        selected = self._selected_shadow_player
        if selected not in self._shadow_player_pool:
            selected = next(iter(self._shadow_player_pool))
            self._selected_shadow_player = selected

        current_bar = int(bar_index if bar_index is not None else (self._t or 0))
        cached_bar = int(getattr(self, "_external_shadow_player_actions_bar", -1) or -1)
        cached_actions = getattr(self, "_external_shadow_player_actions", {}) or {}
        if cached_actions and cached_bar == current_bar:
            player_actions = {
                name: dict(actions or {})
                for name, actions in cached_actions.items()
            }
        else:
            kw = dict(volumes=volumes, month=month, portfolio_value=portfolio_value, bar_index=bar_index)
            player_actions: Dict[str, dict] = {}
            for name, player in self._shadow_player_pool.items():
                try:
                    self._sync_shadow_player_state(player, portfolio_value=portfolio_value)
                    player_actions[name] = player.act(prices, **kw) or {}
                except Exception as exc:
                    log.debug("  [PLAYER meta/%s] act error: %s", name, exc)
                    player_actions[name] = {}

        # FIX B1/B3 (2026-04-26): weighted ensemble голосование.
        # При single-mode — старое поведение, при champion_blend / risk_parity —
        # для каждого символа выбирается категория действий с наибольшим
        # суммарным весом, при условии веса ≥ min_agree.
        weights = self._ensemble_weights(regime)
        out = {s: 0 for s in prices}
        source_for_symbol = {}
        symbol_leaders = getattr(self, "_selected_symbol_shadow_players", {}) or {}
        ensemble_mode = str(getattr(self, "ENSEMBLE_MODE", "single") or "single").lower()
        min_agree = 0.45

        for sym in prices:
            sym_leader = symbol_leaders.get(sym)
            if sym_leader and (player_actions.get(sym_leader) or {}).get(sym, 0):
                action = player_actions[sym_leader][sym]
                out[sym] = action
                source_for_symbol[sym] = sym_leader
                continue
            if ensemble_mode == "single" or len(weights) <= 1:
                leader = sym_leader or selected
                action = (player_actions.get(leader) or {}).get(sym, 0)
                source = leader
                if not action and leader != selected:
                    action = (player_actions.get(selected) or {}).get(sym, 0)
                    source = selected if action else leader
                if action:
                    out[sym] = action
                    source_for_symbol[sym] = source
                continue
            cat_weight: Dict[str, float] = {}
            cat_specific: Dict[str, Dict[int, float]] = {}
            cat_voters: Dict[str, list] = {}
            for name, w in weights.items():
                if w <= 0: continue
                a = (player_actions.get(name) or {}).get(sym, 0)
                if a == 0: continue
                cat = self._action_class(a)
                if not cat: continue
                cat_weight[cat] = cat_weight.get(cat, 0.0) + w
                cat_specific.setdefault(cat, {})[a] = cat_specific.setdefault(cat, {}).get(a, 0.0) + w
                cat_voters.setdefault(cat, []).append(name)
            if not cat_weight:
                continue
            # FIX H2/H4 (2026-04-27): симметричный close/open threshold.
            # Старый порог close_w >= 0.30 был ниже open-порога 0.45 → любые
            # 30 % коррелированных голосов закрывали позицию преждевременно.
            # Теперь close требует тот же вес большинства, что и open.
            # Аварийные exit-ы остаются за PositionSafety (SL/TP/trail), они
            # независимы от голосования.
            close_w = cat_weight.get("fut_close", 0.0) + cat_weight.get("spot_sell", 0.0)
            non_close_cats = {c: w for c, w in cat_weight.items()
                              if c not in ("fut_close", "spot_sell")}
            non_close_max = max(non_close_cats.values()) if non_close_cats else 0.0
            if close_w >= min_agree and close_w >= non_close_max:
                if cat_weight.get("fut_close", 0.0) >= cat_weight.get("spot_sell", 0.0):
                    out[sym] = 8
                else:
                    out[sym] = 3
                voters = cat_voters.get("fut_close", []) or cat_voters.get("spot_sell", [])
                if voters:
                    source_for_symbol[sym] = voters[0]
                continue
            best_cat = max(cat_weight, key=cat_weight.get)
            if cat_weight[best_cat] < min_agree:
                continue
            spec = cat_specific.get(best_cat, {})
            if not spec:
                continue
            best_action = max(spec, key=spec.get)
            out[sym] = int(best_action)
            voters = cat_voters.get(best_cat, [])
            source_for_symbol[sym] = voters[0] if voters else (selected or "ensemble")

        # FIX A1 (2026-04-26): снимок out ДО PositionGovernor. Любой close=8/3,
        # дописанный PositionGovernor по актуальному self._open_pos, должен
        # пройти насквозь — иначе race с reconcile приводил к зависанию KAT/RAVE.
        out_before_safety = dict(out)
        self._position_safety.apply_exit_actions(
            self._open_pos,
            prices,
            int(self._t or 0),
            out,
            stop_loss_pct=self.SL_PCT,
            take_profit_pct=self.TP_PCT,
            trail_pct=self.TRAIL_PCT,
            stale_bars=self.STALE_BARS,
            logger=log,
        )
        safety_forced_close: set = set()
        for sym, action in out.items():
            prev = out_before_safety.get(sym, 0)
            if action in (3, 8) and prev != action:
                safety_forced_close.add(sym)
                source_for_symbol[sym] = "PositionSafety"

        for sym in list(out.keys()):
            if sym in self._BLACKLIST and out[sym] != 0:
                out[sym] = 0
                source_for_symbol.pop(sym, None)
        for sym in list(out.keys()):
            action = out[sym]
            if action not in (1, 2, 4, 5, 6, 7):
                continue
            source = source_for_symbol.get(sym) or selected
            if not self._symbol_live_allowed(source, sym):
                out[sym] = 0
                source_for_symbol.pop(sym, None)
        # FIX: фильтруем конфликтные действия от делегированного игрока против
        # реального состояния позиций. Внутреннее состояние суб-агентов игрока
        # (FundingArb.pos и т.п.) живёт отдельно от self._open_pos реального
        # Пантеона: игрок может выдать "open long BTC" пока реальный Пантеон
        # уже держит BTC long (открыл его на прошлом тике через другого
        # игрока). Без этого фильтра в hedge-режиме MEXC/Bitget биржа
        # стэкает позицию — удваивает экспозицию без предупреждения.
        # Аналогично "close" при отсутствии позиции — нет смысла слать на биржу.
        for sym in list(out.keys()):
            action = out[sym]
            if action == 0:
                continue
            # FIX A1: PositionSafety уже прочитал _open_pos и сказал «закрывай».
            # Не блокируем такие close, даже если _open_pos снизу окажется пуст.
            if sym in safety_forced_close and action in (3, 8):
                continue
            existing = self._open_pos.get(sym)
            if existing:
                existing_side = str(existing.get("side", "long") or "long")
                # Уже открыта позиция в ту же сторону → блокируем дубль open
                if action in (1, 2, 4, 5) and existing_side == "long":
                    out[sym] = 0
                    source_for_symbol.pop(sym, None)
                elif action in (6, 7) and existing_side == "short":
                    out[sym] = 0
                    source_for_symbol.pop(sym, None)
                # Противоположная сторона → принудительно close (3/8) вместо
                # flip-в-противоположную, так как биржа в hedge-режиме не
                # закроет старую сторону автоматически. Игрок может на
                # следующем тике переоткрыть нужную сторону.
                elif action in (1, 2, 4, 5) and existing_side == "short":
                    out[sym] = 8
                elif action in (6, 7) and existing_side == "long":
                    out[sym] = 8
            else:
                # Нет позиции, а игрок шлёт close → дроп.
                # ВНИМАНИЕ: PositionSafety case обработан выше отдельно.
                if action in (3, 8):
                    out[sym] = 0
                    source_for_symbol.pop(sym, None)
        self._position_safety.sync_positions(self._open_pos, out, prices, int(self._t or 0))

        self._last_contributors = {}
        self._last_agreement_scores = {}
        self._last_risk_multipliers = {}
        if not hasattr(self, "_sub_signal_counts"):
            self._sub_signal_counts = {}
        for sym, action in out.items():
            if action == 0:
                continue
            source = source_for_symbol.get(sym) or selected
            confidence = self._shadow_player_signal_confidence(source, sym)
            clean_name = source.replace("V_", "")
            self._last_contributors[sym] = clean_name
            self._last_agreement_scores[sym] = confidence
            self._last_risk_multipliers[sym] = self._shadow_player_risk_multiplier(
                source, action, confidence,
            )
            self._sub_signal_counts[clean_name] = self._sub_signal_counts.get(clean_name, 0) + 1

        self._last_regime = regime
        if any(a != 0 for a in out.values()):
            self._shadow_player_selector.save_snapshot(
                self, reason="player-signal", logger=log,
            )
        return out

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
        if self._shadow_players_enabled:
            self._shadow_player_selector.save_snapshot(
                self, force=force, reason=reason, logger=log,
            )

    def _rotate_agents(self):
        logger = None if self._quiet_info_logs() else log
        if self.USE_REGIME_TOP_TRADER:
            self._update_regime_leaders(logger=logger)
            self._update_context_memory(self._current_context)
            self._update_symbol_memory()
            self._last_rotation = int(getattr(self, "_t", getattr(self, "_last_rotation", 0)) or 0)
            self.save_memory_snapshot(force=True, reason="rotation")
            return
        self._portfolio_allocator.rotate(self, logger=logger)

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

        FIX 2026-04-27: помечаем переход в live-фазу. До этого момента
        regime-rotations подавляются (warmup-чурн в логах не нужен).
        """
        # Помечаем live-старт у себя и у всех shadow-players
        self._live_started = True
        self._live_start_bar = int(bar_index or 0)
        for _name, _player in (getattr(self, "_shadow_player_pool", {}) or {}).items():
            try:
                setattr(_player, "_live_started", True)
                setattr(_player, "_live_start_bar", int(bar_index or 0))
            except Exception:
                pass
        quiet_logs = self._quiet_info_logs()
        for attr_name, label, agent in self.iter_subagents():
            reset = getattr(agent, 'reset_for_live', None)
            if callable(reset):
                try:
                    reset(bar_index)
                except Exception:
                    pass
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
                if n_held and not quiet_logs:
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
        self._last_rotation = -99999
        self._last_shadow_player_rotation = -99999
        self._shadow_window_anchor = {}
        self._shadow_symbol_anchor = {}
        self._current_context = {}
        self._last_context_bar = -99999
        self._shadow_player_challenger = ""
        self._shadow_player_challenger_streak = 0
        self._shadow_player_scores = {}
        self._selected_symbol_shadow_players = {}
        # FIX: сбрасываем выбор лидера и счётчик кулдауна, чтобы первая
        # live-ротация выбрала лучшего игрока по свежим warmup-метрикам.
        # Иначе стейл _selected_shadow_player из warmup (например,
        # V_PlayerFunding с пиковой активностью) может править первые
        # live-бары до истечения PLAYER_SWITCH_COOLDOWN_BARS.
        self._selected_shadow_player = ""
        self._last_shadow_player_switch_bar = -99999
        self._last_regime_switch_bar = -99999
        self._pending_regime = None
        self._pending_regime_since = -99999
        if hasattr(self, '_sub_signal_counts'):
            self._sub_signal_counts.clear()
        if hasattr(self, '_recent_real_signal_agents'):
            self._recent_real_signal_agents.clear()
        if hasattr(self, '_last_risk_multipliers'):
            self._last_risk_multipliers.clear()
        self._external_shadow_player_actions = {}
        self._external_shadow_player_actions_bar = -1
        for name, player in getattr(self, "_shadow_player_pool", {}).items():
            reset = getattr(player, "reset_for_live", None)
            if callable(reset):
                try:
                    reset(bar_index)
                except Exception as exc:
                    log.debug("    [reset/player] %s failed: %s", name, exc)
        if not quiet_logs:
            log.info("  ✅ Panteon.reset_for_live(): все sub-agents сброшены (bar=%d)", bar_index)

    def act(self,prices,volumes,month=None,portfolio_value=None,bar_index=None):
        self._t = bar_index if bar_index is not None else getattr(self,'_t',0)+1
        t = self._t
        prev_regime = self._canonical_market_regime(self._r) if self._r is not None else None
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

        raw_regime = self._canonical_market_regime(self._r or 'neutral')
        # FIX: гистерезис режима. _detect_regime_fast часто скачет bullish↔neutral
        # по коротким окнам, что каждые 30-60 сек форсировало ротацию игроков и
        # агентов (тысячи CLOSE-сообщений в warmup). Новая логика:
        #   1. Мгновенно применяем новый режим только при переходе в/из crash
        #      (crash — это защитная мера, её нельзя откладывать).
        #   2. Остальные флипы требуют REGIME_HYSTERESIS_BARS устойчивости.
        hysteresis = int(getattr(self, "REGIME_HYSTERESIS_BARS", 12) or 0)
        effective_prev = self._canonical_market_regime(
            getattr(self, "_stable_regime", None) or prev_regime or raw_regime
        )
        stable_regime = getattr(self, "_stable_regime", None)
        if stable_regime is None:
            stable_regime = raw_regime
            self._stable_regime = stable_regime
        if hysteresis <= 0 or raw_regime == "crash":
            # вход в crash — мгновенный (защитная мера)
            new_stable = raw_regime
        elif stable_regime == "crash":
            # FIX 2026-04-27: выход ИЗ crash тоже требует hysteresis-баров
            # стабильности. Раньше тут был мгновенный exit, и бот мог скакать
            # между crash и neutral в нестабильных warmup→live переходах.
            pending = getattr(self, "_pending_regime", None)
            pending_since = int(getattr(self, "_pending_regime_since", -99999) or -99999)
            if raw_regime == stable_regime:
                new_stable = stable_regime
                self._pending_regime = None
                self._pending_regime_since = -99999
            else:
                if pending != raw_regime:
                    self._pending_regime = raw_regime
                    self._pending_regime_since = t
                    new_stable = stable_regime
                elif t - pending_since >= hysteresis:
                    new_stable = raw_regime
                    self._pending_regime = None
                    self._pending_regime_since = -99999
                else:
                    new_stable = stable_regime
        else:
            pending = getattr(self, "_pending_regime", None)
            pending_since = int(getattr(self, "_pending_regime_since", -99999) or -99999)
            if raw_regime == stable_regime:
                new_stable = stable_regime
                self._pending_regime = None
                self._pending_regime_since = -99999
            else:
                if pending != raw_regime:
                    self._pending_regime = raw_regime
                    self._pending_regime_since = t
                    new_stable = stable_regime
                elif t - pending_since >= hysteresis:
                    new_stable = raw_regime
                    self._pending_regime = None
                    self._pending_regime_since = -99999
                else:
                    new_stable = stable_regime
        self._stable_regime = new_stable
        regime = new_stable
        regime_changed = effective_prev != regime
        kw=dict(volumes=volumes,month=month,portfolio_value=portfolio_value,bar_index=bar_index)

        # FIX 2026-04-27: подавляем regime-rotation/log-spam во время warmup.
        # Регим там скачет каждые несколько секунд из-за компрессии warmup-time
        # → бесполезный churn в логах и ложные ротации.
        in_warmup = not bool(getattr(self, "_live_started", False))

        if self._shadow_players_enabled and self._shadow_player_pool:
            # FIX H5 (2026-04-27): anti-flapping. В сессии MEXC 2026-04-25
            # за 1 минуту было 8 force-rotations из-за быстрого
            # neutral↔bullish↔bearish скачка → noisy селект. Теперь
            # forced rotation не чаще, чем PLAYER_SWITCH_COOLDOWN_BARS // 3.
            cooldown = int(getattr(self, "PLAYER_SWITCH_COOLDOWN_BARS", 90) or 90)
            min_gap = max(1, cooldown // 3)
            last_force = int(getattr(self, "_last_force_player_rotation_bar", -99999) or -99999)
            recently_forced = (t - last_force) < min_gap
            if regime_changed and not self._quiet_info_logs() and not in_warmup and not recently_forced:
                log.info("  [PLAYER meta] regime changed %s -> %s; forcing player rotation", effective_prev, regime)
            should_force = regime_changed and not in_warmup and not recently_forced
            if should_force:
                self._last_force_player_rotation_bar = t
            if should_force or t - self._last_shadow_player_rotation >= self.PLAYER_ROTATION_INT:
                self._rotate_shadow_players(prices)
                self._last_shadow_player_rotation = t
            return self._act_via_shadow_players(
                prices,
                volumes,
                month=month,
                portfolio_value=portfolio_value,
                bar_index=bar_index,
                regime=regime,
            )

        # ═══════════════════════════════════════════════════════════════
        # ADAPTIVE ROTATION: периодически обновляем состав агентов
        # ═══════════════════════════════════════════════════════════════
        if self.USE_REGIME_TOP_TRADER:
            self._update_regime_leaders(logger=log)

        # FIX H5 (2026-04-27) + FIX 2026-05-04: default cooldown 90→30.
        # Anti-flapping для agent-rotation сохраняется через min_gap_a.
        cooldown_a = int(getattr(self, "PLAYER_SWITCH_COOLDOWN_BARS", 30) or 30)
        min_gap_a = max(1, cooldown_a // 3)
        last_force_a = int(getattr(self, "_last_force_agent_rotation_bar", -99999) or -99999)
        recently_forced_a = (t - last_force_a) < min_gap_a
        if regime_changed and not self._quiet_info_logs() and not in_warmup and not recently_forced_a:
            log.info("  [Panteon] regime changed %s -> %s; forcing agent rotation", effective_prev, regime)
        should_force_agent = regime_changed and not in_warmup and not recently_forced_a
        if should_force_agent:
            self._last_force_agent_rotation_bar = t
        if should_force_agent or t - self._last_rotation >= self.ROTATION_INT:
            self._rotate_agents()
            self._last_rotation = t

        # Строим acts_list из _active_weights (обновляются ротацией)
        acts_list = []
        if self.USE_REGIME_TOP_TRADER:
            symbol_regimes = {
                s: self._symbol_execution_regime(s, regime)
                for s in prices
            }
            self._last_symbol_regime = dict(symbol_regimes)
            agent_actions = {}
            for label, agent in self._agent_pool.items():
                if label in self.LIVE_AGENT_BLOCKLIST:
                    continue
                try:
                    acts = agent.act(prices, **kw)
                except Exception:
                    acts = {}
                agent_actions[label] = acts or {}

            for label, acts in agent_actions.items():
                filtered = {}
                for s, a in (acts or {}).items():
                    if not a or s not in prices:
                        continue
                    sym_regime = symbol_regimes.get(s) or regime
                    if self._regime_leaders.get(sym_regime) == label:
                        filtered[s] = a
                if filtered:
                    acts_list.append((label, filtered, 1.0))
        else:
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
        total_active_weight = 1.0 if self.USE_REGIME_TOP_TRADER else max(
            sum(float(weight) for weight in self._active_weights.values()),
            1e-9,
        )
        n_open = len(self._open_pos)
        turnover_pressure = self._turnover_pressure(portfolio_value, n_open)
        open_extra, close_extra = self._execution_threshold_offsets(portfolio_value)
        open_single_threshold, open_multi_threshold, single_strong_threshold = (
            self._open_thresholds_for_turnover(open_extra, turnover_pressure)
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
                vote_w = 1.0 if self.USE_REGIME_TOP_TRADER else self._effective_vote_weight(ag_name, s, w, a)
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
                candidate_open_threshold = open_single_threshold
                candidate_strong_threshold = single_strong_threshold
                if short_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                    candidate_open_threshold, candidate_strong_threshold = (
                        self._single_agent_thresholds_for_label(
                            lead_short,
                            open_single_threshold,
                            single_strong_threshold,
                        )
                    )
                thresh = (
                    open_multi_threshold
                    if short_support_n >= self.MIN_MULTI_AGENT_SUPPORT
                    else candidate_open_threshold
                )
                if sw >= thresh:
                    if short_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                        if sw < candidate_strong_threshold:
                            continue
                        if (
                            lead_short
                            and not self.USE_REGIME_TOP_TRADER
                            and not self._single_agent_open_allowed(lead_short, sw, candidate_strong_threshold)
                        ):
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
                candidate_open_threshold = open_single_threshold
                candidate_strong_threshold = single_strong_threshold
                if long_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                    candidate_open_threshold, candidate_strong_threshold = (
                        self._single_agent_thresholds_for_label(
                            lead_long,
                            open_single_threshold,
                            single_strong_threshold,
                        )
                    )
                thresh = (
                    open_multi_threshold
                    if long_support_n >= self.MIN_MULTI_AGENT_SUPPORT
                    else candidate_open_threshold
                )
                if lw >= thresh:
                    if long_support_n < self.MIN_MULTI_AGENT_SUPPORT:
                        if lw < candidate_strong_threshold:
                            continue
                        if (
                            lead_long
                            and not self.USE_REGIME_TOP_TRADER
                            and not self._single_agent_open_allowed(lead_long, lw, candidate_strong_threshold)
                        ):
                            continue
                    open_candidates.append((s, 4, lw, long_support_n))

        # Сортируем по весу (самые сильные сигналы сначала)
        open_candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
        max_new_per_bar = self._max_new_positions_for_bar(
            portfolio_value,
            n_open=n_open,
            candidate_count=len(open_candidates),
        )

        n_new = 0
        for s, action, w, _support_n in open_candidates:
            if n_new >= max_new_per_bar:
                break
            if n_open >= self.MAX_POS:
                break
            out[s] = self._full_size_open_action(action, w, _support_n, turnover_pressure)
            n_open += 1
            n_new += 1
            if not self.USE_REGIME_TOP_TRADER:
                _contributors.setdefault(s, {})['(limited)'] = w
            _agreement_scores[s] = min(1.0, w / total_active_weight)

        # ══════════════════════════════════════════════════════════════════
        # POSITION MANAGER v5: SL + TP + Trailing Stop + Stale Exit
        # Управляет ВСЕМИ позициями включая внешние (инъектированные)
        # ══════════════════════════════════════════════════════════════════
        # Suppress verbose CLOSE log when this Panteon instance is itself a shadow
        # player running inside the meta ensemble — otherwise each rotation event
        # produces N× duplicate warnings (one per shadow player running the same tick).
        _is_shadow_child = self._quiet_info_logs()
        self._position_safety.apply_exit_actions(
            self._open_pos,
            prices,
            t,
            out,
            stop_loss_pct=self.SL_PCT,
            take_profit_pct=self.TP_PCT,
            trail_pct=self.TRAIL_PCT,
            stale_bars=self.STALE_BARS,
            logger=(None if _is_shadow_child else log),
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

        self._sync_subagents_to_real_positions(prices, t)

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
            self._last_risk_multipliers[s] = (
                1.0 if self.USE_REGIME_TOP_TRADER else self._signal_risk_multiplier(
                    s, a, self._last_agreement_scores.get(s, 0.0), voters,
                )
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
    ENABLE_META_PLAYERS = False
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
        # FIX 2026-05-05 (carantine-aware bootstrap): отфильтровываем
        # карантинных. Без этого PanteonResearch стартует с RichardDennis=18%
        # (карантин), и до первого _rotate_agents использует его в торговле.
        blocklist = (
            getattr(self, "LIVE_AGENT_BLOCKLIST", frozenset())
            or getattr(type(self), "_SEED_AGENT_QUARANTINE", frozenset())
        )
        filtered = OrderedDict(
            (label, weight) for label, weight in desired.items()
            if label in self._agent_pool and label not in blocklist
        )
        total = sum(filtered.values())
        if total > 0:
            self._active_weights = {label: weight / total for label, weight in filtered.items()}

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        # FIX 2026-05-05: карантинный → сильный минус, не выберется
        if label in (
            getattr(self, "LIVE_AGENT_BLOCKLIST", frozenset())
            or getattr(type(self), "_SEED_AGENT_QUARANTINE", frozenset())
        ):
            return -10.0
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
        # FIX 2026-05-05 (carantine-aware bootstrap): отфильтровываем агентов,
        # которые сейчас в карантине (LIVE_AGENT_BLOCKLIST или
        # _SEED_AGENT_QUARANTINE). Без этого, например, PanteonTrendResearch
        # стартует с 80% веса на RichardDennis/LiveTrendFollow/MomentumScalper/
        # FundingArb — все четверо карантинные. Вариант сразу торгует через
        # них, и реальный Пантеон (когда выбирает этого варианта делегатом)
        # тащит их сигналы в боевую торговлю.
        blocklist = (
            getattr(self, "LIVE_AGENT_BLOCKLIST", frozenset())
            or getattr(type(self), "_SEED_AGENT_QUARANTINE", frozenset())
        )
        filtered = OrderedDict(
            (label, weight) for label, weight in desired.items()
            if label in self._agent_pool and label not in blocklist
        )
        total = sum(filtered.values())
        if total > 0:
            self._active_weights = {
                label: weight / total for label, weight in filtered.items()
            }
        else:
            # Все из BOOTSTRAP_WEIGHTS оказались в карантине → пустой
            # ансамбль, _rotate_agents в первом цикле перезаполнит
            # активными живыми агентами.
            self._active_weights = {}

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        # FIX 2026-05-05 (carantine-aware scoring): карантинный агент
        # не получает СКОРИНГ — иначе вариант делает его лидером.
        # Возвращаем сильно отрицательное значение, чтобы он гарантированно
        # ушёл вниз ranking-а в ShadowPlayerMetaSelector / _rotate.
        if label in (
            getattr(self, "LIVE_AGENT_BLOCKLIST", frozenset())
            or getattr(type(self), "_SEED_AGENT_QUARANTINE", frozenset())
        ):
            return -10.0
        pnl = float(perf.get('pnl_pct', 0.0) or 0.0)
        sharpe = float(perf.get('sharpe', 0.0) or 0.0)
        max_dd = abs(float(perf.get('max_dd', perf.get('max_drawdown_pct', perf.get('max_dd_pct', 0.0))) or 0.0))
        signals = int(perf.get('signals', 0) or 0)
        entries = int(perf.get('entries', 0) or 0)
        closed = int(perf.get('closed_trades', perf.get('total_trades', 0)) or 0)
        activity = min(signals, 40) * 0.03 + min(entries, 20) * 0.08 + min(closed, 12) * 0.06
        inactivity_penalty = 1.5 if signals == 0 and entries == 0 else 0.0
        score = pnl + sharpe * 1.25 + activity - max_dd * self.DRAWDOWN_PENALTY_MULT - inactivity_penalty
        # FIX 2026-05-05: бонус только если label НЕ в карантине
        if (
            label in self.SCORE_BONUS_LABELS
            and signals > 0
            and label not in (
                getattr(self, "LIVE_AGENT_BLOCKLIST", frozenset())
                or getattr(type(self), "_SEED_AGENT_QUARANTINE", frozenset())
            )
        ):
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


# ═══════════
# PanteonResilient — устойчивый игрок (FIX 2026-04-26)
# ════════════════════════════════════════════════════════════════════════
class PanteonResilient(_PanteonShadowVariant):
    """
    Состав построен ТОЛЬКО на агентах с положительным результатом одновременно
    на BITGET и на MEXC за анализируемую сессию 2026-04-25_15-24:

      Cross-exchange winners (BG / MEXC pnl_pct):
        LiveMeanRev          +3.63 / +0.34   wr 66/68%
        LiveVolCompress      +1.18 / +1.81   wr 41/47%
        LiveTrendFollow      +6.44 / +1.16   wr 44/35%
        LiveRegimePullback   +2.11 / +1.67   wr 34/38%
        Genetics{Bull,Bear,Neutral} — стабильный фон.

    Дополнительно (только если живут):
        MeanRevConfirmed   — confirmation-обёртка над LiveMeanRev
        AdaptiveVolTrend   — ATR-trail-обёртка над LiveTrendFollow

    Защита (всегда активна):
        DefensiveStopOverlay — last-line exit, не открывает.

    Bootstrap-веса распределяют ~70% между MeanRev/Trend/VolCompress (где
    win-rate подтверждён), ~16% — генетика, 8% — защитный оверлей.
    """
    ROTATION_INT = 90
    ENABLE_META_PLAYERS = False
    MIN_AGENTS = 3
    MAX_AGENTS = 6
    MIN_WEIGHT = 0.10
    MAX_POS = 5
    OPEN_SINGLE_THRESHOLD = 0.30
    OPEN_MULTI_THRESHOLD = 0.24
    CLOSE_SINGLE_THRESHOLD = 0.24
    CLOSE_MULTI_THRESHOLD = 0.20
    BOOTSTRAP_WEIGHTS = (
        ('LiveMeanRev',         0.20),
        ('MeanRevConfirmed',    0.14),
        ('AdaptiveVolTrend',    0.12),
        ('LiveTrendFollow',     0.10),
        ('LiveVolCompress',     0.10),
        ('LiveRegimePullback',  0.10),
        ('GeneticsBullish',     0.06),
        ('GeneticsBearish',     0.06),
        ('GeneticsNeutral',     0.04),
        ('DefensiveStopOverlay', 0.08),
    )
    SCORE_BONUS_LABELS = frozenset({
        'LiveMeanRev', 'MeanRevConfirmed', 'LiveVolCompress',
        'LiveRegimePullback', 'AdaptiveVolTrend', 'LiveTrendFollow',
    })
    SCORE_BONUS = 0.16
    DRAWDOWN_PENALTY_MULT = 0.45
    MAX_ACTIVITY_SIGNALS = 80
    MAX_ACTIVITY_ENTRIES = 36
    ACTIVITY_PENALTY_PER_SIGNAL = 0.010
    ACTIVITY_PENALTY_PER_ENTRY = 0.020


PanteonPlayer = Panteon


# ────────────────────────────────────────────────────────────────────
# NeuroPlayer-stub (восстановление после обрыва файла 2026-04-26).
# Полная реализация хранится в панти-pyc; этот класс — fallback,
# чтобы _init_shadow_player_pool не падал при globals().get("NeuroPlayer").
# ────────────────────────────────────────────────────────────────────
class NeuroPlayer:
    """Fallback-stub: реальная реализация в .pyc-кэше."""
    PLAYER_STATUS = "shadow_only"
    PLAYER_STATUS_REASON = "stub after file truncation; original logic in pyc"
    REGIME_INT = 2 * 60
    BUY_THRESH = 0.30
    SELL_THRESH = 0.15
    STOP_PCT = 0.08
    MAX_POS = 5

    def __init__(self, *args, **kwargs):
        self.pos = {}
        self.ep = {}
        self._t = 0

    def reset_for_live(self, bar_index: int = 0):
        self._t = bar_index

    def act(self, prices, volumes=None, month=None, portfolio_value=None, bar_index=None):
        if bar_index is not None:
            self._t = bar_index
        return {s: 0 for s in (prices or {})}


# ────────────────────────────────────────────────────────────────────────
# Фабрики, которых ждёт mexc_connector (FIX 2026-04-26).
# Утрачены при обрыве файла. Безопасные минимальные реализации:
# возвращают пустой OrderedDict, чтобы вызов был валидным, а сама работа
# с агентами шла через главный Panteon-игрок, который bridge создаёт сам.
# Если в будущем нужна интеграция MEXC-специфичных агентов, расширьте.
# ────────────────────────────────────────────────────────────────────────
def make_panteon_agents():
    """Empty by design (FIX 2026-04-26). См. комментарий выше."""
    from collections import OrderedDict as _OD
    return _OD()


def make_panteon_players():
    """Empty by design (FIX 2026-04-26). См. комментарий выше."""
    from collections import OrderedDict as _OD
    return _OD()
