"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Panteon_Trade.py  —  Panteon Live Trading + Shadow Analytics               ║
║                                                                              ║
║  На основе exchange_api_runtime.py (API_TRADE).                             ║
║                                                                              ║
║  АРХИТЕКТУРА:                                                                ║
║    1. Panteon — торгует реальными деньгами (как в API_TRADE)                ║
║    2. Shadow-агенты — внутренний аналитический контур                       ║
║       на тех же рыночных данных и в том же тайминге.                         ║
║    3. Расширенные логи — для отладки, анализа и адаптации стратегии:        ║
║       • Таблица виртуального P&L всех агентов (leaderboard)                 ║
║       • Матрица согласия агентов (кто с кем совпадает)                      ║
║       • CSV-лог каждого сигнала с полным контекстом                         ║
║       • Детекция смены режима (bull/bear/sideways)                          ║
║       • Win-rate и RR по каждому виртуальному агенту                        ║
║       • Рекомендации по ротации состава Panteon                            ║
║                                                                              ║
║  Цель: адаптировать состав агентов Panteon под текущий рынок,               ║
║  используя live-данные внутреннего shadow-контура.                          ║
║                                                                              ║
║  Запуск:  python Panteon_Trade.py                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import csv
import time
import json
import copy
import logging
import threading
import traceback
from datetime import datetime, timezone, timedelta
from collections import OrderedDict, deque, defaultdict
from typing import Dict, List, Optional, Tuple

from exchange_registry import load_exchange_runtime


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


def _write_json_atomic(path: str, payload, *, log_context: str = "json") -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ (идентична exchange_api_runtime.py)
# ══════════════════════════════════════════════════════════════════════════════

_exchange_runtime = load_exchange_runtime(
    exchange_id=os.getenv("CRYPTO_EXCHANGE"),
    connector_module=os.getenv("CRYPTO_EXCHANGE_CONNECTOR_MODULE"),
    api_module=os.getenv("CRYPTO_EXCHANGE_API_MODULE"),
    adapter_module=os.getenv("CRYPTO_EXCHANGE_ADAPTER_MODULE"),
)
_exchange_adapter = _exchange_runtime.adapter

EXCHANGE_NAME = _exchange_runtime.display_name.strip().upper()
EXCHANGE_CONNECTOR_MODULE = _exchange_runtime.connector_module_name
EXCHANGE_API_MODULE = _exchange_runtime.api_module_name

_exchange_connector = _exchange_adapter.connector_module
_exchange_api = _exchange_adapter.api_module

API_KEY = os.getenv(_exchange_adapter.api_key_env) or getattr(_exchange_api, "API_KEY", "")
API_SECRET = os.getenv(_exchange_adapter.api_secret_env) or getattr(_exchange_api, "API_SECRET", "")
API_PASSPHRASE = os.getenv(_exchange_adapter.api_passphrase_env) or getattr(
    _exchange_api, "API_PASSPHRASE", ""
)
TRADING_MODE = os.getenv(_exchange_adapter.trading_mode_env) or getattr(
    _exchange_api, "TRADING_MODE", "live_futures"
)

# Загружаем settings.txt
_settings_raw = _exchange_adapter.load_settings()
_settings_parsed = _exchange_adapter.parse_settings(_settings_raw)

INITIAL_CAPITAL  = 50.0
TRADE_FRACTION   = _settings_parsed.get("trade_fraction", 0.15)
LEVERAGE         = int(_settings_parsed.get("leverage", 4))
SPOT_FEE         = _settings_parsed.get("spot_fee",     0.001)
FUTURES_FEE      = _settings_parsed.get("futures_fee",  0.0002)

print(f"[settings] {EXCHANGE_NAME} COMBO: leverage={LEVERAGE}x  trade_fraction={TRADE_FRACTION:.0%}"
      f"  spot_fee={SPOT_FEE:.4f}  futures_fee={FUTURES_FEE:.4f}")

RESULTS_ROOT = os.path.join("Results", EXCHANGE_NAME)
DASHBOARD_INTERVAL_SEC = 1800
STATUS_REFRESH_INTERVAL_SEC = 60
LEADERBOARD_INTERVAL_SEC = 600   # Таблица лидеров каждые 10 мин
ROTATION_CHANGE_LOGS_ONLY = True
SIGNAL_CSV_FILE = "all_signals.csv"

FIXED_SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "BNB",
    "ADA", "DOGE", "LTC", "LINK", "UNI",
    "ALGO", "ZEC", "TAO", "NEAR", "ENA",
    "TRX", "AVAX", "DOT",
    # УБРАНЫ: STO, SOLV — в FUTURES_BLACKLIST → close_all() молча не работает
    # Агент открывает позиции, но закрыть не может → маржа заблокирована навечно
]

# ── Расширение торговой вселенной (v5) ──────────────────────────────────────
# FIXED_SYMBOLS остаётся ядром (проверенные монеты). Дополнительно загружаем
# топ-N с биржи по объёму (с жёсткими фильтрами ликвидности) чтобы увеличить
# количество торговых возможностей.
#
# Управление:
#   AUTO_EXPAND_SYMBOLS=0 → только FIXED_SYMBOLS (старое поведение)
#   AUTO_EXPAND_SYMBOLS=1 → FIXED_SYMBOLS + топ-N (по умолчанию)
#
# Переменная TRADING_UNIVERSE_SIZE задаёт целевой размер.
TRADING_UNIVERSE_SIZE = int(os.environ.get("TRADING_UNIVERSE_SIZE", "60"))
_AUTO_EXPAND_SYMBOLS = os.environ.get("AUTO_EXPAND_SYMBOLS", "1") == "1"


def build_trading_universe(
    core: List[str] = None,
    target_size: int = None,
    auto_expand: bool = None,
) -> List[str]:
    """
    Объединяет ядро проверенных монет (core) с топ-N биржи по объёму.
    Применяет фильтры ликвидности (через fetch_top_symbols коннектора).
    При ошибках сети безопасно возвращает только core.
    """
    core = list(core or FIXED_SYMBOLS)
    if target_size is None:
        target_size = TRADING_UNIVERSE_SIZE
    if auto_expand is None:
        auto_expand = _AUTO_EXPAND_SYMBOLS

    if not auto_expand or target_size <= len(core):
        log.info("[universe] Использую только ядро: %d символов", len(core))
        return core

    # Пытаемся взять топ с биржи через коннектор
    top_from_market: List[str] = []
    try:
        fetch_fn = getattr(_exchange_connector, "fetch_top_symbols", None)
        if fetch_fn is not None:
            top_from_market = fetch_fn(n=target_size + len(core))
    except Exception as e:
        log.warning(
            "[universe] fetch_top_symbols упал (%s) — использую только ядро (%d)",
            e, len(core),
        )
        return core

    if not top_from_market:
        log.warning("[universe] Топ пуст — использую только ядро (%d)", len(core))
        return core

    # Объединяем: ядро → остальные по объёму
    seen = {s.upper() for s in core}
    result = list(core)
    for sym in top_from_market:
        sym_upper = sym.upper()
        if sym_upper not in seen:
            result.append(sym_upper)
            seen.add(sym_upper)
            if len(result) >= target_size:
                break

    log.info(
        "[universe] Торговая вселенная %s: %d символов (ядро=%d, рынок=%d)",
        EXCHANGE_NAME, len(result), len(core), len(result) - len(core),
    )
    log.info("[universe] Состав: %s", result)
    return result

MIN_BALANCE_USD = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

if API_KEY:
    os.environ[f"{EXCHANGE_NAME}_API_KEY"] = API_KEY
if API_SECRET:
    os.environ[f"{EXCHANGE_NAME}_SECRET_KEY"] = API_SECRET
if API_PASSPHRASE:
    os.environ[f"{EXCHANGE_NAME}_PASSPHRASE"] = API_PASSPHRASE
os.environ[f"{EXCHANGE_NAME}_TRADING_MODE"] = TRADING_MODE

# ══════════════════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("combo_trade")


def _setup_file_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(output_dir, "combo_trading.log"), encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)  # DEBUG уровень в файл для полной диагностики
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)


ACTION_NAMES = _exchange_adapter.action_names or {
    0: "hold", 1: "buy_half", 2: "buy_full", 3: "sell_spot",
    4: "fl_half", 5: "fl_full", 6: "fs_half", 7: "fs_full", 8: "close_fut",
}


# ══════════════════════════════════════════════════════════════════════════════
# ВИРТУАЛЬНЫЙ ПОРТФЕЛЬ ДЛЯ SHADOW-АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

class VirtualPortfolio:
    """
    Легковесный виртуальный портфель для shadow-агента.
    Трекает виртуальные позиции и P&L без реальных ордеров.
    """

    def __init__(self, initial_capital: float, leverage: int = 4,
                 trade_fraction: float = 0.15, fee: float = 0.0002):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.leverage = leverage
        self.trade_fraction = trade_fraction
        self.fee = fee

        # {sym: {side: 'long'/'short', qty: float, entry: float, bar: int}}
        self.positions: Dict[str, dict] = {}
        self.equity_history: List[float] = [initial_capital]
        self.trades_log: List[dict] = []
        self.signal_count = 0
        self.entry_count = 0
        self.close_count = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self._peak = initial_capital
        self.symbol_stats: Dict[str, dict] = {}

    def reset(self):
        """Сбрасывает shadow-портфель после warmup, оставляя только live-статистику."""
        self.positions.clear()
        self.cash = self.initial_capital
        self.equity_history = [self.initial_capital]
        self.trades_log.clear()
        self.signal_count = 0
        self.entry_count = 0
        self.close_count = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self._peak = self.initial_capital
        self.symbol_stats.clear()

    def _sym_stats(self, sym: str) -> dict:
        stats = self.symbol_stats.get(sym)
        if stats is None:
            stats = {
                'signals': 0,
                'entries': 0,
                'closed_trades': 0,
                'wins': 0,
                'losses': 0,
                'realized_pnl': 0.0,
                'last_pnl': 0.0,
                'last_bar': 0,
            }
            self.symbol_stats[sym] = stats
        return stats

    def execute(self, sym: str, action: int, price: float, bar: int,
                risk_multiplier: float = 1.0):
        """Виртуально исполняет сигнал агента."""
        if action == 0 or price <= 0:
            return

        self.signal_count += 1
        sym_stats = self._sym_stats(sym)
        sym_stats['signals'] += 1
        sym_stats['last_bar'] = int(bar)

        # Close actions (3, 8)
        if action in (3, 8) and sym in self.positions:
            pos = self.positions.pop(sym)
            pnl = self._calc_pnl(pos, price)
            fee_cost = abs(pos['qty'] * price * self.fee)
            net_pnl = pnl - fee_cost
            # FIX v8: возвращаем МАРЖУ (не полную нотионал стоимость!)
            # Было: qty * entry = notional = margin * leverage → cash раздувался в leverage раз
            margin = pos['qty'] * pos['entry'] / self.leverage
            self.cash += margin + net_pnl
            self.total_pnl += net_pnl
            self.close_count += 1
            self.total_trades += 1
            sym_stats['closed_trades'] += 1
            sym_stats['realized_pnl'] += float(net_pnl)
            sym_stats['last_pnl'] = float(net_pnl)
            if net_pnl > 0:
                self.wins += 1
                sym_stats['wins'] += 1
            else:
                self.losses += 1
                sym_stats['losses'] += 1
            self.trades_log.append({
                'bar': bar, 'sym': sym, 'side': pos['side'],
                'entry': pos['entry'], 'exit': price,
                'pnl': net_pnl, 'hold_bars': bar - pos['bar'],
            })
            return

        # Open actions — only if no position in this sym
        if sym in self.positions:
            return

        if action in (1, 2, 4, 5):
            side = 'long'
        elif action in (6, 7):
            side = 'short'
        else:
            return

        # FIX v8: Position sizing от INITIAL_CAPITAL, не от текущего equity
        # Real bot размер позиции ограничен физической свободной маржой.
        # Shadow должен имитировать это, а не реинвестировать нереализованную прибыль.
        base_capital = self.initial_capital
        size_mult = min(max(float(risk_multiplier or 1.0), 0.35), 1.50)
        pos_value = base_capital * self.trade_fraction * self.leverage * size_mult
        qty = pos_value / price
        margin = pos_value / self.leverage   # = base_capital * trade_fraction

        if margin > self.cash * 0.90:
            return  # недостаточно кэша

        fee_cost = qty * price * self.fee
        self.cash -= margin + fee_cost
        self.entry_count += 1
        sym_stats['entries'] += 1
        self.positions[sym] = {
            'side': side, 'qty': qty, 'entry': price, 'bar': bar,
        }

    def _calc_pnl(self, pos: dict, cur_price: float) -> float:
        qty = pos['qty']
        if pos['side'] == 'long':
            return qty * (cur_price - pos['entry'])
        else:
            return qty * (pos['entry'] - cur_price)

    def get_equity(self, prices: dict) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            p = prices.get(sym, pos['entry'])
            eq += pos['qty'] * pos['entry'] / self.leverage  # маржа
            eq += self._calc_pnl(pos, p)                      # unrealized PnL
        return eq

    def snapshot(self, prices: dict):
        eq = self.get_equity(prices)
        self.equity_history.append(eq)
        if eq > self._peak:
            self._peak = eq
        dd = (self._peak - eq) / (self._peak + 1e-9)
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def export_symbol_stats(self, prices: Optional[dict] = None) -> Dict[str, dict]:
        prices = prices or {}
        out: Dict[str, dict] = {}
        symbols = set(self.symbol_stats) | set(self.positions)
        for sym in symbols:
            stats = dict(self.symbol_stats.get(sym) or {})
            pos = self.positions.get(sym)
            open_pnl = 0.0
            if pos is not None:
                open_pnl = float(self._calc_pnl(pos, float(prices.get(sym, pos['entry']))))
            realized_pnl = float(stats.get('realized_pnl', 0.0) or 0.0)
            total_pnl = realized_pnl + open_pnl
            out[sym] = {
                'signals': int(stats.get('signals', 0) or 0),
                'entries': int(stats.get('entries', 0) or 0),
                'closed_trades': int(stats.get('closed_trades', 0) or 0),
                'wins': int(stats.get('wins', 0) or 0),
                'losses': int(stats.get('losses', 0) or 0),
                'realized_pnl': realized_pnl,
                'open_pnl': open_pnl,
                'total_pnl': total_pnl,
                'realized_pnl_pct': realized_pnl / max(self.initial_capital, 1e-9) * 100.0,
                'total_pnl_pct': total_pnl / max(self.initial_capital, 1e-9) * 100.0,
                'last_pnl': float(stats.get('last_pnl', 0.0) or 0.0),
                'last_bar': int(stats.get('last_bar', 0) or 0),
                'has_open_position': int(pos is not None),
                'position_side': str(pos['side']) if pos is not None else '',
            }
        return out

    @property
    def pnl_pct(self) -> float:
        if not self.equity_history:
            return 0.0
        return (self.equity_history[-1] / self.initial_capital - 1) * 100

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades * 100

    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def sharpe(self) -> float:
        if len(self.equity_history) < 10:
            return 0.0
        import numpy as np
        rets = np.diff(self.equity_history) / (np.array(self.equity_history[:-1]) + 1e-9)
        if np.std(rets) < 1e-9:
            return 0.0
        return float(np.mean(rets) / np.std(rets) * (252 * 24 * 60) ** 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL CSV LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class SignalCSVLogger:
    """Пишет каждый сигнал каждого агента в CSV для офлайн-анализа."""

    FIELDS = [
        'timestamp', 'bar', 'live_bar', 'agent', 'is_real',
        'symbol', 'action', 'action_name', 'price',
        'regime', 'contributors', 'portfolio_value',
        'n_open_positions', 'agreement_score', 'risk_multiplier',
    ]

    def __init__(self, output_dir: str):
        self._path = os.path.join(output_dir, SIGNAL_CSV_FILE)
        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._implicit_risk_multiplier = 1.0

    def log(self, bar: int, live_bar: int, agent: str, is_real: bool,
            sym: str, action: int, price: float, regime: str = '',
            contributors: str = '', pv: float = 0.0,
            n_pos: int = 0, agreement: float = 0.0,
            risk_multiplier=None):
        risk_value = self._implicit_risk_multiplier if risk_multiplier is None else risk_multiplier
        self._writer.writerow({
            'timestamp':       datetime.now(tz=timezone.utc).isoformat(),
            'bar':             bar,
            'live_bar':        live_bar,
            'agent':           agent,
            'is_real':         is_real,
            'symbol':          sym,
            'action':          action,
            'action_name':     ACTION_NAMES.get(action, f'act{action}'),
            'price':           f'{price:.6f}',
            'regime':          regime,
            'contributors':    contributors,
            'portfolio_value': f'{pv:.2f}',
            'n_open_positions': n_pos,
            'agreement_score':  f'{agreement:.2f}',
            'risk_multiplier':  f'{float(risk_value or 1.0):.2f}',
        })
        self._file.flush()
        self._implicit_risk_multiplier = 1.0

    def close(self):
        self._file.close()


# ══════════════════════════════════════════════════════════════════════════════
# AGREEMENT MATRIX — кто с кем совпадает по направлению
# ══════════════════════════════════════════════════════════════════════════════

class AgreementTracker:
    """Отслеживает согласованность сигналов между агентами."""

    def __init__(self):
        # {(agent_a, agent_b): {'agree': int, 'total': int}}
        self._pairs: Dict[tuple, dict] = defaultdict(lambda: {'agree': 0, 'total': 0})
        # {agent: {sym: direction}} — текущий тик
        self._current_tick: Dict[str, Dict[str, int]] = {}

    def record(self, agent: str, sym: str, direction: int):
        """direction: +1 long, -1 short, 0 close/hold"""
        if agent not in self._current_tick:
            self._current_tick[agent] = {}
        self._current_tick[agent][sym] = direction

    def flush_tick(self):
        """Вызвать после обработки всех агентов на текущем баре."""
        agents = list(self._current_tick.keys())
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                # Общие символы с ненулевыми сигналами
                syms_a = {s for s, d in self._current_tick[a].items() if d != 0}
                syms_b = {s for s, d in self._current_tick[b].items() if d != 0}
                common = syms_a & syms_b
                for sym in common:
                    key = tuple(sorted([a, b]))
                    self._pairs[key]['total'] += 1
                    if self._current_tick[a][sym] == self._current_tick[b][sym]:
                        self._pairs[key]['agree'] += 1
        self._current_tick.clear()

    def get_agreement_for(self, agent: str, all_agents: dict,
                          sym: str) -> float:
        """Доля агентов согласных с данным по символу."""
        if sym not in self._current_tick.get(agent, {}):
            return 0.0
        my_dir = self._current_tick[agent][sym]
        if my_dir == 0:
            return 0.0
        agree = 0
        total = 0
        for other, sigs in self._current_tick.items():
            if other == agent:
                continue
            if sym in sigs and sigs[sym] != 0:
                total += 1
                if sigs[sym] == my_dir:
                    agree += 1
        return agree / max(total, 1)

    def log_matrix(self, top_n: int = 15):
        """Возвращает строку с матрицей согласия для лога."""
        if not self._pairs:
            return "  Матрица согласия: нет данных"
        lines = ["  ┌── МАТРИЦА СОГЛАСИЯ АГЕНТОВ (топ-{}) ──┐".format(top_n)]
        sorted_pairs = sorted(self._pairs.items(),
                              key=lambda x: x[1]['total'], reverse=True)[:top_n]
        for (a, b), v in sorted_pairs:
            pct = v['agree'] / max(v['total'], 1) * 100
            lines.append(f"  │ {a:>20s} ↔ {b:<20s}  "
                         f"agree={pct:5.1f}%  ({v['agree']}/{v['total']})")
        lines.append("  └" + "─" * 56 + "┘")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME TRACKER — логирует смены режимов
# ══════════════════════════════════════════════════════════════════════════════

class RegimeTracker:
    """Логирует смены рыночного режима."""

    def __init__(self):
        self.current = 'unknown'
        self.history: List[dict] = []   # [{bar, regime, time}]
        self.durations: Dict[str, int] = defaultdict(int)  # режим → кол-во баров

    def update(self, regime: str, bar: int):
        self.durations[regime] = self.durations.get(regime, 0) + 1
        if regime != self.current:
            prev = self.current
            self.current = regime
            self.history.append({
                'bar': bar, 'from': prev, 'to': regime,
                'time': datetime.now(tz=timezone.utc),
            })
            log.info("  🔄 РЕЖИМ СМЕНИЛСЯ: %s → %s  (bar=%d)", prev, regime, bar)
            return True
        return False

    def summary(self) -> str:
        total = sum(self.durations.values()) or 1
        parts = []
        for r in ('bullish', 'bearish', 'neutral', 'sideways'):
            pct = self.durations.get(r, 0) / total * 100
            if pct > 0:
                parts.append(f"{r}={pct:.0f}%")
        return "  ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD — таблица виртуальных агентов
# ══════════════════════════════════════════════════════════════════════════════

def _log_leaderboard_legacy(virtual_portfolios: Dict[str, VirtualPortfolio],
                    real_pnl_pct: float, regime: str, bar: int,
                    regime_tracker: RegimeTracker):
    """Логирует таблицу лидеров + рекомендации."""
    log.info("═" * 78)
    log.info("  📊 LEADERBOARD — ВИРТУАЛЬНЫЕ АГЕНТЫ vs РЕАЛЬНЫЙ  (bar=%d  режим=%s)", bar, regime)
    log.info("  Распределение режимов: %s", regime_tracker.summary())
    log.info("─" * 78)
    log.info("  %-22s  %8s  %7s  %6s  %6s  %6s  %5s  %s",
             "Агент", "Equity$", "P&L%", "WinR%", "Sharpe", "MaxDD%", "Trd", "Позиции")
    log.info("─" * 78)

    # Сортируем по P&L%
    ranked = sorted(virtual_portfolios.items(),
                    key=lambda kv: kv[1].pnl_pct, reverse=True)

    for name, vp in ranked:
        eq = vp.equity_history[-1] if vp.equity_history else vp.initial_capital
        n_pos = len(vp.positions)
        pos_str = ",".join(f"{s}({p['side'][0]})" for s, p in list(vp.positions.items())[:4])
        if len(vp.positions) > 4:
            pos_str += f"+{len(vp.positions) - 4}"

        log.info("  %-22s  %8.2f  %+7.2f  %5.1f%%  %6.2f  %5.1f%%  %5d  %s",
                 name, eq, vp.pnl_pct, vp.win_rate,
                 vp.sharpe(), vp.max_drawdown * 100, vp.total_trades, pos_str)

    log.info("─" * 78)
    log.info("  %-22s  %8s  %+7.2f  %6s  %6s  %6s  %5s  %s",
             "★ РЕАЛЬНЫЙ (Panteon)", "—", real_pnl_pct, "—", "—", "—", "—", "LIVE")
    log.info("═" * 78)

    # Рекомендации: какие агенты лучше всего в текущем режиме
    if ranked:
        top3 = [name for name, vp in ranked[:3] if vp.pnl_pct > 0]
        bottom3 = [name for name, vp in ranked[-3:] if vp.pnl_pct < -2]
        if top3:
            log.info("  💡 РЕКОМЕНДАЦИЯ: лучшие агенты для ротации → %s", ", ".join(top3))
        if bottom3:
            log.info("  ⚠️  УБЫТОЧНЫЕ агенты (рассмотреть исключение) → %s", ", ".join(bottom3))

    # Анализ сигналов: какие агенты генерируют больше всего
    signal_counts = {name: vp.total_trades for name, vp in ranked}
    max_signals = max(signal_counts.values()) if signal_counts else 0
    if max_signals > 0:
        hyperactive = [n for n, c in signal_counts.items()
                       if c > max_signals * 0.7 and virtual_portfolios[n].pnl_pct < 0]
        if hyperactive:
            log.info("  📉 Гиперактивные убыточные (много сделок + минус): %s",
                     ", ".join(hyperactive))


# ══════════════════════════════════════════════════════════════════════════════
# СОЗДАНИЕ SHADOW-АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

def log_leaderboard(virtual_portfolios: Dict[str, VirtualPortfolio],
                    real_pnl_pct: float, regime: str, bar: int,
                    regime_tracker: RegimeTracker):
    """Лидерборд shadow-агентов с честными метриками активности после reset."""
    table_width = 108
    log.info("═" * table_width)
    log.info("  📊 LEADERBOARD — ВИРТУАЛЬНЫЕ АГЕНТЫ vs РЕАЛЬНЫЙ  (bar=%d  режим=%s)", bar, regime)
    log.info("  Распределение режимов: %s", regime_tracker.summary())
    log.info("─" * table_width)
    log.info("  %-22s  %8s  %7s  %6s  %6s  %6s  %5s  %5s  %5s  %s",
             "Агент", "Equity$", "P&L%", "WinR%", "Sharpe", "MaxDD%",
             "Sig", "Ent", "Cls", "Позиции")
    log.info("─" * table_width)

    ranked = sorted(virtual_portfolios.items(), key=lambda kv: kv[1].pnl_pct, reverse=True)
    for name, vp in ranked:
        eq = vp.equity_history[-1] if vp.equity_history else vp.initial_capital
        pos_str = ",".join(f"{s}({p['side'][0]})" for s, p in list(vp.positions.items())[:4])
        if len(vp.positions) > 4:
            pos_str += f"+{len(vp.positions) - 4}"
        log.info("  %-22s  %8.2f  %+7.2f  %5.1f%%  %6.2f  %5.1f%%  %5d  %5d  %5d  %s",
                 name, eq, vp.pnl_pct, vp.win_rate, vp.sharpe(),
                 vp.max_drawdown * 100, vp.signal_count, vp.entry_count,
                 vp.close_count, pos_str or "-")

    log.info("─" * table_width)
    log.info("  %-22s  %8s  %+7.2f  %6s  %6s  %6s  %5s  %5s  %5s  %s",
             "★ РЕАЛЬНЫЙ (Panteon)", "—", real_pnl_pct,
             "—", "—", "—", "—", "—", "—", "LIVE")
    log.info("═" * table_width)

    if ranked:
        top3 = [name for name, vp in ranked[:3] if vp.pnl_pct > 0]
        bottom3 = [name for name, vp in ranked[-3:] if vp.pnl_pct < -2]
        if top3:
            log.info("  💡 РЕКОМЕНДАЦИЯ: лучшие агенты для ротации → %s", ", ".join(top3))
        if bottom3:
            log.info("  ⚠️  УБЫТОЧНЫЕ агенты (рассмотреть исключение) → %s", ", ".join(bottom3))

    signal_counts = {name: vp.signal_count for name, vp in ranked}
    max_signals = max(signal_counts.values()) if signal_counts else 0
    if max_signals > 0:
        hyperactive = [n for n, c in signal_counts.items()
                       if c > max_signals * 0.7 and virtual_portfolios[n].pnl_pct < 0]
        if hyperactive:
            log.info("  📉 Гиперактивные убыточные (много сигналов + минус): %s",
                     ", ".join(hyperactive))

    inactive = [name for name, vp in ranked if vp.signal_count == 0 and not vp.positions]
    if inactive:
        preview = ", ".join(inactive[:6])
        if len(inactive) > 6:
            preview += f" +{len(inactive) - 6}"
        log.info("  ℹ️ Без live-сигналов после reset: %s", preview)


def _create_shadow_agents(initial_capital: float) -> Dict[str, tuple]:
    """
    Возвращает {name: (agent_instance, VirtualPortfolio)}.
Все доступные агенты из panteon_agents + crypto_players.
    """
    shadows = OrderedDict()

# --- panteon_agents: индивидуальные агенты ---
    try:
        from panteon import (MomentumScalper, FundingArb, LiveAfterShock,
                             LiveCrashHunter, LiveRegimePullback,
                             LiveMeanRev, LiveTrendFollow,
                             LiveVolCompress, LiveOIBreakout, CarryFlowAgentV2,
                             ExternalSignalAgent, ResearchValidatorAgent,
                             RichardDennisTurtle, Bomberman,
                             PlayerBomberman, PlayerFunding,
                             NeuroPlayer, PanteonResearch,
                             PanteonTrendResearch, PanteonMeanRevResearch,
                             PanteonDefensiveResearch, PanteonConsensusResearch)
        agent_classes = [
            ("V_MomentumScalper",  MomentumScalper),
            ("V_FundingArb",       FundingArb),
            ("V_LiveAfterShock",   LiveAfterShock),
            ("V_LiveCrashHunter",  LiveCrashHunter),
            ("V_LiveRegimePullback", LiveRegimePullback),
            ("V_LiveMeanRev",      LiveMeanRev),
            ("V_LiveTrendFollow",  LiveTrendFollow),
            ("V_LiveVolCompress",  LiveVolCompress),
            ("V_LiveOIBreakout",   LiveOIBreakout),
            ("V_CarryFlowAgentV2", CarryFlowAgentV2),
            ("V_ExternalSignalAgent", ExternalSignalAgent),
            ("V_ResearchValidatorAgent", ResearchValidatorAgent),
            ("V_RichardDennisTurtle", RichardDennisTurtle),
            ("V_Bomberman",        Bomberman),
            ("V_PlayerBomberman",  PlayerBomberman),
            ("V_PlayerFunding",    PlayerFunding),
            ("V_PanteonResearch", PanteonResearch),
            ("V_PanteonTrendResearch", PanteonTrendResearch),
            ("V_PanteonMeanRevResearch", PanteonMeanRevResearch),
            ("V_PanteonDefensiveResearch", PanteonDefensiveResearch),
            ("V_PanteonConsensusResearch", PanteonConsensusResearch),
            ("V_NeuroPlayer",      NeuroPlayer),
        ]
        for name, cls in agent_classes:
            try:
                agent = cls()
                vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
                shadows[name] = (agent, vp)
            except Exception as e:
                log.debug("  Shadow %s skip: %s", name, e)
    except ImportError as e:
        log.warning("  panteon_agents import failed: %s", e)

    # --- crypto_players: PlayerStop и другие ---
    try:
        from crypto_players import PlayerStop
        agent = PlayerStop()
        vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
        shadows["V_PlayerStop"] = (agent, vp)
    except Exception as e:
        log.debug("  PlayerStop skip: %s", e)

    # --- Альтернативный Panteon (для сравнения с самим собой) ---
    try:
        from panteon import Panteon
        agent = Panteon()
        vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
        shadows["V_Panteon_shadow"] = (agent, vp)
    except Exception as e:
        log.debug("  Panteon shadow skip: %s", e)

    # --- Новый pipeline player для shadow-first сравнения ---
    try:
        from panteon import PanteonNextResearch
        agent = PanteonNextResearch()
        vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
        shadows["V_PanteonNextResearch"] = (agent, vp)
    except Exception as e:
        log.debug("  PanteonNextResearch shadow skip: %s", e)

    # --- Генетические агенты ---
    try:
        from crypto_genetics import GeneticsBullishAgent, GeneticsBearishAgent
        from crypto_agents import _GeneticsAdapter
        for name, cls in [("V_GeneticsBullish", GeneticsBullishAgent),
                          ("V_GeneticsBearish", GeneticsBearishAgent)]:
            try:
                agent = _GeneticsAdapter(cls())
                vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
                shadows[name] = (agent, vp)
            except Exception as e:
                log.debug("  %s skip: %s", name, e)
    except ImportError:
        pass

    log.info("  Shadow-агенты создано: %d  →  %s", len(shadows), list(shadows.keys()))
    return shadows


# ══════════════════════════════════════════════════════════════════════════════
# РАСШИРЕННЫЙ SIGNAL-CAPTURING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedSignalCapture:
    """
    Обёртка для реального Panteon с расширенным логированием.
    Записывает:
      - каждый сигнал в TradingStats
      - полный контекст (режим, суб-агенты, цены, объёмы)
      - сравнение с решениями shadow-агентов
    """

    def __init__(self, inner, stats, csv_logger: SignalCSVLogger,
                 agreement: AgreementTracker, regime_tracker: RegimeTracker):
        self._inner = inner
        self._stats = stats
        self._csv = csv_logger
        self._agreement = agreement
        self._regime = regime_tracker
        self._label = type(inner).__name__
        self._warmup_end = 0
        self._capture_enabled = False

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def set_inner(self, inner):
        self._inner = inner
        self._label = type(inner).__name__

    def enable_capture(self, warmup_end: int = 0):
        self._warmup_end = int(warmup_end or 0)
        self._capture_enabled = True

    def disable_capture(self):
        self._capture_enabled = False

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        acts = self._inner.act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        )

        if not self._capture_enabled:
            return acts

        # Метаданные Panteon
        _contributors = getattr(self._inner, '_last_contributors', {})
        _agreement_scores = getattr(self._inner, '_last_agreement_scores', {})
        _risk_multipliers = getattr(self._inner, '_last_risk_multipliers', {})
        _regime = getattr(self._inner, '_last_regime', '')
        _open_pos = getattr(self._inner, '_open_pos', {})

        # Обновляем режим
        if _regime:
            self._regime.update(_regime, bar_index or 0)

        # Логируем каждый ненулевой сигнал
        if acts:
            n_active = sum(1 for a in acts.values() if a and a != 0)
            if n_active > 0:
                log.info("  [REAL %s] режим=%s  открытых_позиций=%d  "
                         "сигналов=%d",
                         self._label, _regime, len(_open_pos), n_active)

            for sym, action in acts.items():
                if action and action != 0:
                    price = prices.get(sym, 0.0)
                    agent = _contributors.get(sym, '')
                    risk_multiplier = float(_risk_multipliers.get(sym, 1.0) or 1.0)
                    # Записываем в stats
                    self._stats.record_signal(
                        bar_index or 0, sym, action, price,
                        agent=agent, regime=_regime,
                        risk_multiplier=risk_multiplier,
                    )
                    # Определяем direction для agreement
                    if action in (1, 2, 4, 5):
                        direction = 1
                    elif action in (6, 7):
                        direction = -1
                    else:
                        direction = 0
                    self._agreement.record(self._label, sym, direction)

                    # Подробный лог сигнала
                    agr_score = float(
                        _agreement_scores.get(
                            sym,
                            self._agreement.get_agreement_for(self._label, {}, sym),
                        )
                    )
                    log.info(
                        "  ▶ REAL SIGNAL: %s %s  price=%.6f  action=%d(%s)  "
                        "by=[%s]  agree=%.0f%%  risk=%.2fx  regime=%s  open_pos=%d",
                        ACTION_NAMES.get(action, '?'), sym, price,
                        action, ACTION_NAMES.get(action, '?'),
                        agent, agr_score * 100, risk_multiplier, _regime, len(_open_pos),
                    )

                    # CSV
                    warmup_end = getattr(self, '_warmup_end', 0)
                    self._csv.log(
                        bar=bar_index or 0,
                        live_bar=max(0, (bar_index or 0) - warmup_end),
                        agent=self._label,
                        is_real=True,
                        sym=sym, action=action, price=price,
                        regime=_regime, contributors=agent,
                        pv=portfolio_value or 0,
                        n_pos=len(_open_pos),
                        agreement=agr_score,
                        risk_multiplier=risk_multiplier,
                    )
        return acts


# ══════════════════════════════════════════════════════════════════════════════
# ИМПОРТ exchange_api_runtime КОМПОНЕНТОВ (переиспользуем)
# ══════════════════════════════════════════════════════════════════════════════

# Импортируем из exchange_api_runtime.py нужные классы
sys.path.insert(0, script_dir)
try:
    from exchange_api_runtime import (
        MexcDirectClient, TradingStats, inject_live_state_into_player,
        save_dashboard, save_summary_json, _check_futures_api_permission,
    )
except ImportError as _ie:
    log.error("Не удалось импортировать exchange_api_runtime.py — убедитесь что файл "
              "находится в той же директории: %s", _ie)
    sys.exit(1)

# Переопределяем exchange-компоненты через registry/runtime. Так текущий
# entrypoint остаётся совместимым с MEXC/Bitget и готов к следующим биржам.
MexcDirectClient = _exchange_adapter.resolve_direct_client_class()
TradingStats = _exchange_adapter.trading_stats_cls
inject_live_state_into_player = _exchange_adapter.inject_live_state_into_player
save_dashboard = _exchange_adapter.save_dashboard
save_summary_json = _exchange_adapter.save_summary_json
_check_futures_api_permission = _exchange_adapter.check_futures_api_permission


# ══════════════════════════════════════════════════════════════════════════════
# РАСШИРЕННЫЙ MONITOR THREAD
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SHADOW DASHBOARD — PNG визуализация прогресса теневых агентов
# ══════════════════════════════════════════════════════════════════════════════

def save_shadow_dashboard(vp_map: Dict[str, 'VirtualPortfolio'],
                          stats, regime_tracker,
                          output_dir: str):
    """Генерирует PNG-дашборд с прогрессом всех shadow-агентов vs реального."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        return

    if not vp_map:
        return

    # Цвета
    BG   = '#0D1117'
    MID  = '#161B22'
    GRN  = '#3FB950'
    RED  = '#F85149'
    BLU  = '#58A6FF'
    YLW  = '#D29922'
    WHT  = '#E6EDF3'
    GRY  = '#8B949E'
    GRID = '#21262D'

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle("SHADOW AGENTS DASHBOARD — Virtual Trading Leaderboard",
                 fontsize=14, fontweight='bold', color=WHT, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                           height_ratios=[1.2, 1.0])

    def _style(ax, title=''):
        ax.set_facecolor(MID)
        ax.tick_params(colors=GRY, labelsize=7)
        ax.set_title(title, color=WHT, fontsize=10, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, alpha=0.15, color=GRID)

    def _clean_label(name: str, max_len: int = 18) -> str:
        txt = str(name or '').replace('V_', '').replace('STP_', '')
        if len(txt) <= max_len:
            return txt
        return txt[: max(0, max_len - 3)] + "..."

    def _repair_curve_spikes(values, floor: float = 0.0):
        if not values:
            return []
        raw = []
        for v in values:
            try:
                raw.append(float(v))
            except Exception:
                raw.append(float("nan"))

        cleaned = list(raw)
        n = len(raw)

        def _nearest_valid(idx: int, step: int):
            j = idx + step
            while 0 <= j < n:
                cand = raw[j]
                if np.isfinite(cand):
                    return cand
                j += step
            return None

        for i, val in enumerate(raw):
            prev_v = _nearest_valid(i, -1)
            next_v = _nearest_valid(i, +1)

            if not np.isfinite(val):
                if prev_v is not None and next_v is not None:
                    cleaned[i] = (prev_v + next_v) / 2.0
                elif prev_v is not None:
                    cleaned[i] = prev_v
                elif next_v is not None:
                    cleaned[i] = next_v
                continue

            if val <= floor:
                if prev_v is not None and prev_v > floor and next_v is not None and next_v > floor:
                    cleaned[i] = (prev_v + next_v) / 2.0
                elif prev_v is not None and prev_v > floor:
                    cleaned[i] = prev_v
                elif next_v is not None and next_v > floor:
                    cleaned[i] = next_v
                continue

            if prev_v is None or next_v is None:
                continue
            ref_low = min(prev_v, next_v)
            ref_high = max(prev_v, next_v)
            if ref_low <= floor:
                continue
            close_neighbors = abs(prev_v - next_v) / max(ref_high, 1e-9) <= 0.08
            deep_drop = val < ref_low * 0.35
            if close_neighbors and deep_drop:
                cleaned[i] = (prev_v + next_v) / 2.0

        return cleaned

    def _compress_real_curve():
        eq = _repair_curve_spikes(list(getattr(stats, 'equity_curve', []) or []))
        if not eq:
            return [], []

        raw_x = getattr(stats, 'live_bar_curve', None)
        if not isinstance(raw_x, list) or len(raw_x) != len(eq):
            raw_x = list(range(len(eq)))

        out_x, out_y = [], []
        for raw_bar, equity in zip(raw_x, eq):
            try:
                bar = int(raw_bar)
            except Exception:
                bar = len(out_x)

            if out_x and bar == out_x[-1]:
                out_y[-1] = equity
            else:
                out_x.append(bar)
                out_y.append(equity)
        return out_x, out_y

    real_bar_x, real_bar_eq = _compress_real_curve()

    def _shadow_x(eq_len: int):
        if eq_len <= 0:
            return []
        if real_bar_x:
            if len(real_bar_x) >= eq_len:
                return list(real_bar_x[-eq_len:])
            if len(real_bar_x) == 1:
                start = real_bar_x[0]
                return [start + i for i in range(eq_len)]
            return np.linspace(real_bar_x[0], real_bar_x[-1], eq_len).tolist()
        return list(range(eq_len))

    # Ранжируем по P&L%
    ranked = sorted(vp_map.items(), key=lambda kv: kv[1].pnl_pct, reverse=True)
    real_ic = float(getattr(stats, 'initial_capital', 0.0) or 0.0)
    real_pnl = ((real_bar_eq[-1] / real_ic) - 1.0) * 100.0 if real_bar_eq and real_ic > 0 else (
        stats.pnl_pct if hasattr(stats, 'pnl_pct') else 0.0
    )
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(ranked), 1)))

    # ═══════════════════════════════════════════════════════════
    # PANEL 1: Equity Curves (top-left, spans 2 cols)
    # ═══════════════════════════════════════════════════════════
    ax_eq = fig.add_subplot(gs[0, :2])
    _style(ax_eq, 'Equity Curves — Shadow Agents vs Real')

    for i, (name, vp) in enumerate(ranked):
        eq = _repair_curve_spikes(vp.equity_history)
        if len(eq) < 2:
            continue
        alpha = 0.9 if i < 5 else 0.3
        lw = 1.8 if i < 3 else 0.8
        label = f"{_clean_label(name)} ({vp.pnl_pct:+.1f}%)" if i < 6 else None
        ax_eq.plot(_shadow_x(len(eq)), eq, color=palette[i % 20],
                   alpha=alpha, lw=lw, label=label)

    # Реальный агент
    if real_bar_eq:
        ax_eq.plot(real_bar_x, real_bar_eq,
                   color=YLW, lw=2.5, alpha=1.0, linestyle='--',
                   label=f'★ REAL ({real_pnl:+.1f}%)')

    ic = ranked[0][1].initial_capital if ranked else 100
    ax_eq.axhline(ic, color=WHT, lw=0.8, alpha=0.3, linestyle=':')
    ax_eq.set_ylabel('Equity ($)', color=GRY, fontsize=8)
    ax_eq.set_xlabel('Bar', color=GRY, fontsize=8)
    ax_eq.legend(loc='upper left', fontsize=6, ncol=2,
                 facecolor=MID, edgecolor=GRID, labelcolor=WHT)

    # ═══════════════════════════════════════════════════════════
    # PANEL 2: P&L% Ranking (top-right)
    # ═══════════════════════════════════════════════════════════
    ax_bar = fig.add_subplot(gs[0, 2])
    _style(ax_bar, 'P&L% Ranking')

    names = [_clean_label(n, max_len=16) for n, _ in ranked]
    pnls = [vp.pnl_pct for _, vp in ranked]
    colors = [GRN if p > 0 else RED for p in pnls]

    y_pos = range(len(names))
    ax_bar.barh(y_pos, pnls, color=colors, alpha=0.8, height=0.7)
    # Линия реального агента
    ax_bar.axvline(real_pnl, color=YLW, lw=2, linestyle='--',
                   label=f'REAL {real_pnl:+.1f}%')
    ax_bar.axvline(0, color=WHT, lw=0.5, alpha=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=6.5, color=WHT)
    ax_bar.set_xlabel('P&L %', color=GRY, fontsize=8)
    ax_bar.legend(fontsize=7, facecolor=MID, edgecolor=GRID, labelcolor=WHT)
    ax_bar.invert_yaxis()

    # ═══════════════════════════════════════════════════════════
    # PANEL 3: Win Rate + Trades (bottom-left)
    # ═══════════════════════════════════════════════════════════
    ax_wr = fig.add_subplot(gs[1, 0])
    _style(ax_wr, 'Win Rate % vs Closed Trades')

    for i, (name, vp) in enumerate(ranked):
        if vp.total_trades == 0:
            continue
        ax_wr.scatter(vp.total_trades, vp.win_rate, color=palette[i % 20],
                      s=60, alpha=0.8, edgecolors=WHT, linewidth=0.5,
                      zorder=3)
        ax_wr.annotate(_clean_label(name, max_len=14), (vp.total_trades, vp.win_rate),
                       fontsize=5, color=GRY, ha='left',
                       xytext=(4, 2), textcoords='offset points')

    ax_wr.axhline(50, color=WHT, lw=0.8, alpha=0.3, linestyle=':')
    ax_wr.set_xlabel('Closed Trades', color=GRY, fontsize=8)
    ax_wr.set_ylabel('Win Rate %', color=GRY, fontsize=8)

    # ═══════════════════════════════════════════════════════════
    # PANEL 4: Sharpe + MaxDD (bottom-center)
    # ═══════════════════════════════════════════════════════════
    ax_sh = fig.add_subplot(gs[1, 1])
    _style(ax_sh, 'Sharpe Ratio vs Max Drawdown')

    for i, (name, vp) in enumerate(ranked):
        sh = vp.sharpe()
        dd = vp.max_drawdown * 100
        ax_sh.scatter(dd, sh, color=palette[i % 20],
                      s=60, alpha=0.8, edgecolors=WHT, linewidth=0.5,
                      zorder=3)
        ax_sh.annotate(_clean_label(name, max_len=14), (dd, sh),
                       fontsize=5, color=GRY, ha='left',
                       xytext=(4, 2), textcoords='offset points')

    ax_sh.axhline(0, color=WHT, lw=0.8, alpha=0.3, linestyle=':')
    ax_sh.set_xlabel('Max Drawdown %', color=GRY, fontsize=8)
    ax_sh.set_ylabel('Sharpe Ratio', color=GRY, fontsize=8)

    # ═══════════════════════════════════════════════════════════
    # PANEL 5: Summary Table (bottom-right)
    # ═══════════════════════════════════════════════════════════
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.set_facecolor(MID)
    ax_tbl.axis('off')
    ax_tbl.set_title('Summary', color=WHT, fontsize=10, fontweight='bold', pad=8)

    regime = regime_tracker.current if regime_tracker else '?'
    uptime = stats.uptime_str if hasattr(stats, 'uptime_str') else '?'
    n_signals = len(stats.signals) if hasattr(stats, 'signals') else 0
    regime_summary = regime_tracker.summary() if regime_tracker else ''

    lines = [
        f"Режим: {regime}",
        f"Аптайм: {uptime}",
        f"Сигналов (real): {n_signals}",
        f"Режимы: {regime_summary}",
        f"Shadow агентов: {len(vp_map)}",
        "",
        "── ТОП-3 ──",
    ]
    for i, (name, vp) in enumerate(ranked[:3]):
        lines.append(f"  {i+1}. {_clean_label(name, max_len=16)}: {vp.pnl_pct:+.1f}% "
                     f"(Sig={vp.signal_count} Cls={vp.close_count})")
    lines.append("")
    lines.append("── ДНО-3 ──")
    for name, vp in ranked[-3:]:
        if vp.pnl_pct < 0:
            lines.append(f"  ✗ {name.replace('V_','')}: {vp.pnl_pct:+.1f}%")
    lines.append("")
    lines.append(f"★ REAL: {real_pnl:+.2f}%")

    line_step = 0.80 / max(len(lines), 1)
    for i, line in enumerate(lines):
        clr = YLW if '★' in line else (GRN if '──' in line else WHT)
        ax_tbl.text(0.05, 0.95 - i * line_step, line,
                    transform=ax_tbl.transAxes, fontsize=7.5,
                    color=clr, va='top', fontfamily='monospace')

    # Сохраняем
    path = os.path.join(output_dir, 'shadow_dashboard.png')
    fig.savefig(path, dpi=130, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    log.debug("  📊 Shadow dashboard → %s", path)


if False and EXCHANGE_NAME != "MEXC":
    try:
        MexcDirectClient = getattr(
            _exchange_api,
            "MexcDirectClient",
            getattr(_exchange_api, "BitgetDirectClient", MexcDirectClient),
        )
        TradingStats = _exchange_api.TradingStats
        inject_live_state_into_player = _exchange_api.inject_live_state_into_player
        save_dashboard = _exchange_api.save_dashboard
        save_summary_json = _exchange_api.save_summary_json
        _check_futures_api_permission = _exchange_api._check_futures_api_permission
    except Exception as _exchange_import_err:
        log.error(
            "РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РіСЂСѓР·РёС‚СЊ exchange API module %s: %s",
            EXCHANGE_API_MODULE,
            _exchange_import_err,
        )
        sys.exit(1)


def _create_direct_client():
    return _exchange_adapter.create_direct_client(API_KEY, API_SECRET, API_PASSPHRASE)


def _resolve_bridge_class(connector_module):
    del connector_module
    return _exchange_adapter.resolve_bridge_class()


class ComboMonitorThread(threading.Thread):
    """
    Фоновый поток: всё что делает MonitorThread + leaderboard + agreement.
    """

    def __init__(self, client: MexcDirectClient, stats: TradingStats,
                 output_dir: str, shadows: Dict[str, tuple],
                 agreement: AgreementTracker, regime_tracker: RegimeTracker):
        super().__init__(daemon=True, name="combo_monitor")
        self._client = client
        self._stats = stats
        self._output_dir = output_dir
        self._shadows = shadows
        self._agreement = agreement
        self._regime_tracker = regime_tracker
        self._bridge = None
        self._stop_evt = threading.Event()
        self._counter = 0
        self._last_leaderboard = 0
        self._last_dashboard = 0.0

    def run(self):
        while not self._stop_evt.wait(timeout=STATUS_REFRESH_INTERVAL_SEC):
            self._counter += 1
            try:
                self._refresh()
            except Exception as e:
                log.debug("ComboMonitor error: %s", e)

    def _refresh(self, render_dashboards: bool = False):
        # --- Обновляем bar count ---
        if self._bridge is not None:
            bridge_bar = getattr(self._bridge, "_bar", 0)
            if not hasattr(self, '_warmup_end_bar'):
                bridge_warmup = int(getattr(self._bridge, "_warmup_end", 0) or 0)
                if bridge_warmup <= 0 and bridge_bar > 0:
                    return
                self._warmup_end_bar = (
                    bridge_warmup if bridge_warmup > 0 else self._stats.bar_count
                )
            live = max(0, bridge_bar - self._warmup_end_bar)
            self._stats.live_bar_count = live
            self._stats.bar_count = bridge_bar

        # --- Баланс с MEXC ---
        try:
            if TRADING_MODE == "live_futures":
                snap = self._client.get_full_snapshot()
                fut = snap['futures']
                total_stable = fut['equity']
                usdt_val = fut['available']
                unrealized = fut['unrealized']

                log.info(
                    "  💰 Futures equity=$%.4f  available=$%.4f  "
                    "unrealPnL=%+.4f  позиций=%d",
                    total_stable, usdt_val, unrealized, len(fut['positions']),
                )

                # Позиции
                self._stats.current_positions.clear()
                for p in fut['positions']:
                    self._stats.current_positions[p['symbol']] = {
                        "side": p['side'].upper(), "qty": p['qty'],
                        "entry": p['entry'], "pnl": p['unrealized_pnl'],
                        "type": "futures",
                    }

                # Reconcile _open_pos
                self._reconcile_open_pos(fut['positions'])

                self._stats.bonus_info = {
                    "real_usdt": usdt_val, "stable_total": total_stable,
                    "tradeable_total": total_stable, "bonus_tokens": {},
                    "has_funds": total_stable >= MIN_BALANCE_USD,
                }

                self._stats.record_tick(
                    ts=datetime.now(tz=timezone.utc),
                    balance=total_stable, pnl=unrealized, available=usdt_val,
                    live_bar=self._stats.live_bar_count,
                )
        except Exception as e:
            log.debug("Balance refresh: %s", e)

        # --- Dashboard ---
        save_summary_json(self._stats, self._output_dir)

        # --- Shadow Dashboard ---
        vp_map = {name: tup[1] for name, tup in self._shadows.items()}
        try:
            save_shadow_dashboard(
                vp_map, self._stats, self._regime_tracker,
                self._output_dir,
            )
        except Exception as e:
            log.debug("Shadow dashboard err: %s", e)

        # --- Leaderboard (каждые LEADERBOARD_INTERVAL_SEC) ---
        now = time.time()
        if render_dashboards or (now - self._last_dashboard >= DASHBOARD_INTERVAL_SEC):
            self._last_dashboard = now
            save_dashboard(self._stats, self._output_dir, "dashboard_latest.png")
        if now - self._last_leaderboard >= LEADERBOARD_INTERVAL_SEC:
            self._last_leaderboard = now
            vp_map = {name: tup[1] for name, tup in self._shadows.items()}
            if not ROTATION_CHANGE_LOGS_ONLY:
                real_pnl = self._stats.pnl_pct
                regime = self._regime_tracker.current
                bar = self._stats.bar_count
                log_leaderboard(vp_map, real_pnl, regime, bar, self._regime_tracker)
                log.info(self._agreement.log_matrix())

            # Сохраняем leaderboard в JSON
            self._save_leaderboard_json(vp_map)

    def _reconcile_open_pos(self, exchange_positions: list):
        """Синхронизирует _open_pos агента с реальными позициями."""
        try:
            if self._bridge is None:
                return
            _player = None
            for _ag in getattr(self._bridge, 'agents', {}).values():
                _candidate = getattr(_ag, '_inner', _ag)
# EnhancedSignalCapture wraps Panteon
                if hasattr(_candidate, '_inner'):
                    _candidate = _candidate._inner
                if hasattr(_candidate, '_open_pos'):
                    _player = _candidate
                    break
            if _player is None:
                return

            real_syms = {p['symbol'] for p in exchange_positions}
            stale = [s for s in list(_player._open_pos) if s not in real_syms]
            for s in stale:
                log.info("  [reconcile] %s удалён из _open_pos (закрыта)", s)
                del _player._open_pos[s]
            for p in exchange_positions:
                sym = p['symbol']
                if sym not in _player._open_pos:
                    _player._open_pos[sym] = {
                        'entry': p['entry'], 'side': p['side'],
                        'bar': getattr(self._bridge, '_bar', 0),
                        'peak': p['entry'],
                        'external': True,
                    }
                    log.info("  [reconcile] %s добавлен (внешняя: %s entry=%.4f)",
                             sym, p['side'], p['entry'])
        except Exception as e:
            log.debug("  [reconcile] %s", e)

    def _save_leaderboard_json(self, vp_map: Dict[str, VirtualPortfolio]):
        try:
            data = {}
            for name, vp in vp_map.items():
                data[name] = {
                    'pnl_pct': round(vp.pnl_pct, 4),
                    'equity': round(vp.equity_history[-1], 4) if vp.equity_history else 0,
                    'win_rate': round(vp.win_rate, 2),
                    'signals': vp.signal_count,
                    'entries': vp.entry_count,
                    'closed_trades': vp.close_count,
                    'total_trades': vp.total_trades,
                    'sharpe': round(vp.sharpe(), 4),
                    'max_drawdown_pct': round(vp.max_drawdown * 100, 2),
                    'open_positions': list(vp.positions.keys()),
                }
            data['_real_pnl_pct'] = round(self._stats.pnl_pct, 4)
            data['_regime'] = self._regime_tracker.current
            data['_regime_history'] = self._regime_tracker.history[-20:]
            data['_bar'] = self._stats.bar_count
            data['_timestamp'] = datetime.now(tz=timezone.utc).isoformat()

            path = os.path.join(self._output_dir, "leaderboard.json")
            _write_json_atomic(path, data, log_context="leaderboard")
        except Exception as exc:
            log.warning("  [leaderboard.json] save failed: %s", exc)

    def stop(self):
        self._stop_evt.set()


# ══════════════════════════════════════════════════════════════════════════════
# SHADOW RUNNER — прогоняет виртуальных агентов на каждом тике
# ══════════════════════════════════════════════════════════════════════════════

def run_shadow_tick(shadows: Dict[str, tuple], prices: dict, volumes: dict,
                    bar: int, live_bar: int, month: int,
                    csv_logger: SignalCSVLogger,
                    agreement: AgreementTracker,
                    regime: str):
    """
    Прогоняет всех shadow-агентов на текущих ценах.
    Записывает сигналы в CSV и обновляет виртуальные портфели.
    """
    for name, (agent, vp) in shadows.items():
        try:
            pv = vp.get_equity(prices)
            acts = agent.act(
                prices=prices, volumes=volumes,
                month=month, portfolio_value=pv, bar_index=bar,
            )
        except Exception as e:
            log.debug("  [shadow %s] act error: %s", name, e)
            continue

        if acts is None:
            continue

        n_active = sum(1 for a in acts.values() if a and a != 0)

        for sym, action in acts.items():
            if action == 0 or sym not in prices:
                continue

            price = prices[sym]
            risk_map = getattr(agent, '_last_risk_multipliers', {}) or {}
            risk_multiplier = float(risk_map.get(sym, 1.0) or 1.0)
            vp.execute(sym, action, price, bar, risk_multiplier=risk_multiplier)
            csv_logger._implicit_risk_multiplier = risk_multiplier

            # Direction for agreement
            if action in (1, 2, 4, 5):
                direction = 1
            elif action in (6, 7):
                direction = -1
            else:
                direction = 0
            agreement.record(name, sym, direction)

            # CSV log
            csv_logger.log(
                bar=bar, live_bar=live_bar,
                agent=name, is_real=False,
                sym=sym, action=action, price=price,
                regime=regime, contributors='',
                pv=pv, n_pos=len(vp.positions),
                agreement=0,  # рассчитываем после flush
            )

        # Snapshot портфеля
        vp.snapshot(prices)

        if n_active > 0:
            log.debug("  [shadow %-22s] tick_signals=%d  signals=%d  entries=%d  closes=%d  "
                      "equity=$%.2f  P&L=%+.2f%%  open=%d",
                      name, n_active, vp.signal_count, vp.entry_count,
                      vp.close_count, vp.equity_history[-1], vp.pnl_pct,
                      len(vp.positions))


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКА BRIDGE._run_cycle ДЛЯ SHADOW-АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

def _patch_bridge_for_shadows(bridge, shadows, csv_logger, agreement, regime_tracker,
                              real_player=None):
    """
    Monkey-patch bridge._fetch_market и bridge._run_cycle:
      1. _fetch_market — сохраняет volumes в bridge._last_volumes
      2. _run_cycle — после реального цикла прогоняет shadow-агентов
      3. Передаёт shadow performance в Panteon для адаптивной ротации
    """
    # --- Патч _fetch_market чтобы сохранять volumes ---
    original_fetch = bridge._fetch_market

    def patched_fetch():
        prices, volumes = original_fetch()
        bridge._last_volumes = volumes
        return prices, volumes

    bridge._fetch_market = patched_fetch
    bridge._last_volumes = {}

    # --- Патч _run_cycle ---
    original_run_cycle = bridge._run_cycle

    def patched_run_cycle():
        original_run_cycle()

        prices = bridge._price_hist[-1] if bridge._price_hist else {}
        if not prices:
            return

        volumes = bridge._last_volumes or {}

        bar = bridge._bar
        live_bar = max(0, bar - bridge._warmup_end)
        month = datetime.utcnow().month
        regime = regime_tracker.current

        # Прогоняем shadow-агентов
        run_shadow_tick(
            shadows, prices, volumes,
            bar, live_bar, month,
            csv_logger, agreement, regime,
        )

        # Flush agreement tracker
        agreement.flush_tick()

        # ═══════════════════════════════════════════════════════════════
        # ADAPTIVE ROTATION: передаём shadow performance в Panteon
        # ═══════════════════════════════════════════════════════════════
        if real_player is not None and hasattr(real_player, 'set_shadow_perf'):
            perf = {}
            for name, (agent, vp) in shadows.items():
                perf[name] = {
                    'pnl_pct':      vp.pnl_pct,
                    'win_rate':     vp.win_rate,
                    'signals':      vp.signal_count,
                    'entries':      vp.entry_count,
                    'closed_trades': vp.close_count,
                    'total_trades': vp.total_trades,
                    'sharpe':       vp.sharpe(),
                    'max_dd':       vp.max_drawdown * 100,
                    'n_positions':  len(vp.positions),
                    'per_symbol':   vp.export_symbol_stats(prices),
                }
            real_player.set_shadow_perf(perf)

        # Периодический подробный лог (каждые 60 live баров = ~1 час)
        if (not ROTATION_CHANGE_LOGS_ONLY) and live_bar > 0 and live_bar % 60 == 0:
            log.info("═" * 78)
            log.info("  📈 HOURLY SUMMARY  (live_bar=%d  bar=%d)", live_bar, bar)
            log.info("  Режим: %s  |  Распределение: %s",
                     regime, regime_tracker.summary())
            log.info("  Смен режима: %d", len(regime_tracker.history))

            vp_map = {name: tup[1] for name, tup in shadows.items()}
            top5 = sorted(vp_map.items(), key=lambda kv: kv[1].pnl_pct, reverse=True)[:5]
            for name, vp in top5:
                log.info("    %s: %+.2f%% (sig=%d, ent=%d, cls=%d, win=%.0f%%)",
                         name, vp.pnl_pct, vp.signal_count, vp.entry_count,
                         vp.close_count, vp.win_rate)

            # Логируем текущие активные веса
            if real_player and hasattr(real_player, '_active_weights'):
                aw = real_player._active_weights
                log.info("  Активные агенты: %s",
                         "  ".join(f"{k}={v:.0%}" for k, v in
                                   sorted(aw.items(), key=lambda x: x[1], reverse=True)))
            log.info("═" * 78)

    bridge._run_cycle = patched_run_cycle


# ══════════════════════════════════════════════════════════════════════════════
# КРИТИЧЕСКИЙ FIX: СБРОС SUB-АГЕНТОВ ПОСЛЕ WARMUP
# ══════════════════════════════════════════════════════════════════════════════

def _reset_agent_internals(agent, bar_index: int, name: str = "?"):
    """
    Полный сброс внутреннего состояния агента после warmup.

Проблема: _reset_subagent_timers() из exchange_api_runtime искал _last_check,
но все panteon_agents используют _lc. Позиционные словари (self.pos,
    self.ep, self.entry_px) вообще не сбрасывались.

    Сбрасывает:
      - Таймеры: _lc, _last_check, last_check, _last_rebal
      - Позиции: pos, entry_px, ep, et
      - Кэши: _open_pos, _sub_signal_counts
    """
    reset_log = []

    custom_reset = getattr(agent, 'reset_for_live', None)
    if callable(custom_reset):
        try:
            custom_reset(bar_index)
            log.info("    [reset] %-22s  сброшено: reset_for_live()", name)
            return
        except TypeError:
            try:
                custom_reset()
                log.info("    [reset] %-22s  сброшено: reset_for_live()", name)
                return
            except Exception as e:
                log.debug("    [reset %s] custom reset_for_live failed: %s", name, e)
        except Exception as e:
            log.debug("    [reset %s] custom reset_for_live failed: %s", name, e)

    sub_agents = getattr(agent, 'sub_agents', None)
    if isinstance(sub_agents, dict):
        for sub_name, sub_agent in sub_agents.items():
            _reset_agent_internals(sub_agent, bar_index, f"{name}.{sub_name}")

        active = getattr(agent, '_active', None)
        if isinstance(active, dict):
            for key in active:
                active[key] = True
            reset_log.append('_active')

        virt = getattr(agent, '_virt', None)
        if isinstance(virt, dict):
            virt.clear()
            reset_log.append('_virt')

        pos_trackers = getattr(agent, '_pos', None)
        if isinstance(pos_trackers, dict):
            try:
                agent._pos = {key: type(val)() for key, val in pos_trackers.items()}
            except Exception:
                agent._pos = {}
            reset_log.append('_pos')

        dd_reactivate = getattr(agent, '_dd_reactivate_at', None)
        if isinstance(dd_reactivate, dict):
            for key in dd_reactivate:
                dd_reactivate[key] = -1
            reset_log.append('_dd_reactivate_at')

        cooldowns = getattr(agent, '_cooldown_until', None)
        if isinstance(cooldowns, dict):
            for key in cooldowns:
                cooldowns[key] = -1
            reset_log.append('_cooldown_until')

        dd_streak = getattr(agent, '_dd_streak', None)
        if isinstance(dd_streak, dict):
            for key in dd_streak:
                dd_streak[key] = 0
            reset_log.append('_dd_streak')

        if hasattr(agent, '_prev_total_pv'):
            agent._prev_total_pv = None
            reset_log.append('_prev_total_pv')
        if hasattr(agent, '_initialized'):
            agent._initialized = False
            reset_log.append('_initialized')
        if hasattr(agent, '_apy_mode'):
            agent._apy_mode = False
        if hasattr(agent, 'pos') and isinstance(agent.pos, dict):
            agent.pos.clear()
            reset_log.append('pos')
        if hasattr(agent, '_current') and sub_agents:
            agent._current = next(iter(sub_agents))
        if hasattr(agent, '_regime'):
            agent._regime = 'unknown'

    # --- Таймеры: ставим так чтобы первый check сработал немедленно ---
    for attr in ('_lc', '_last_check', 'last_check', '_last_rebal',
                 'last_rebal', '_cooldown_until', 'pause_until',
                 '_last_reeval', '_last_regime_check', '_last_switch_t'):
        if hasattr(agent, attr):
            value = getattr(agent, attr, None)
            if isinstance(value, dict):
                continue
            setattr(agent, attr, -99999)
            reset_log.append(attr)

    # --- Позиционные словари: очищаем значения (оставляем ключи) ---
    for attr in ('pos',):
        d = getattr(agent, attr, None)
        if isinstance(d, dict):
            n_held = sum(1 for v in d.values() if v is not None)
            for k in d:
                d[k] = None  # Не удаляем ключи, только сбрасываем значения
            if n_held > 0:
                reset_log.append(f"pos({n_held} held→0)")

    # --- Entry prices: обнуляем ---
    for attr in ('ep', 'entry_px'):
        d = getattr(agent, attr, None)
        if isinstance(d, dict):
            for k in d:
                d[k] = 0.0
            reset_log.append(attr)

    # --- Entry time ---
    for attr in ('et',):
        d = getattr(agent, attr, None)
        if isinstance(d, dict):
            for k in d:
                d[k] = 0
            reset_log.append(attr)

    # --- Внутренний bar counter ---
    try: agent.t = bar_index
    except (AttributeError, TypeError): pass

    if reset_log:
        uniq_log = list(dict.fromkeys(reset_log))
        log.info("    [reset] %-22s  сброшено: %s", name, ", ".join(uniq_log))


def _reset_player_subagents_for_live(player, bar_index: int):
    """
    Сброс всех суб-агентов Panteon для чистого старта live-торговли.
    """
    if hasattr(player, 'iter_subagents'):
        for _, name, agent in player.iter_subagents():
            _reset_agent_internals(agent, bar_index, name)
    else:
        sub_attrs = [
            ('_ms',  'MomentumScalper'),
            ('_fa',  'FundingArb'),
            ('_las', 'LiveAfterShock'),
            ('_lch', 'LiveCrashHunter'),
            ('_gb',  'GeneticsBullish'),
        ]
        for attr, name in sub_attrs:
            agent = getattr(player, attr, None)
            if agent is not None:
                _reset_agent_internals(agent, bar_index, name)

    log.info("  ✅ Все суб-агенты Panteon сброшены для live-торговли")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _instantiate_real_player_by_name(player_name: str):
    player_name = str(player_name).strip()
    import panteon as panteon_mod
    from player_next import PanteonNextResearch
    if player_name == "PanteonNextResearch":
        return PanteonNextResearch()
    player_cls = getattr(panteon_mod, player_name, None)
    if player_cls is None:
        raise AttributeError(f"{player_name} is not available in panteon")
    return player_cls()


def _build_shadow_perf_from_shadows(shadows: Dict[str, tuple], prices: Optional[dict] = None) -> Dict[str, dict]:
    prices = prices or {}
    perf = {}
    for name, (_, vp) in shadows.items():
        perf[name] = {
            'pnl_pct': vp.pnl_pct,
            'win_rate': vp.win_rate,
            'signals': vp.signal_count,
            'entries': vp.entry_count,
            'closed_trades': vp.close_count,
            'total_trades': vp.total_trades,
            'sharpe': vp.sharpe(),
            'max_dd': vp.max_drawdown * 100,
            'n_positions': len(vp.positions),
            'per_symbol': vp.export_symbol_stats(prices),
        }
    return perf


def _log_promotion_gate_summary(summary: Optional[dict], *, forced: bool = False):
    if not isinstance(summary, dict):
        return
    candidate = summary.get("candidate_shadow", "?")
    baseline = summary.get("baseline_shadow", "") or "n/a"
    cand = summary.get("candidate_metrics") or {}
    base = summary.get("baseline_metrics") or {}
    ready = bool(summary.get("ready"))
    log_fn = log.info if (ready or forced) else log.warning
    status = "READY" if ready else "BLOCKED"
    if forced and not ready:
        status = "FORCED"
    log_fn(
        "  [promotion-gate] %s  candidate=%s pnl=%+.2f%% sharpe=%.2f dd=%.2f%%  |  "
        "baseline=%s pnl=%+.2f%% sharpe=%.2f dd=%.2f%%",
        status,
        candidate,
        float(cand.get("pnl_pct", 0.0) or 0.0),
        float(cand.get("sharpe", 0.0) or 0.0),
        float(cand.get("max_dd", 0.0) or 0.0),
        baseline,
        float(base.get("pnl_pct", 0.0) or 0.0),
        float(base.get("sharpe", 0.0) or 0.0),
        float(base.get("max_dd", 0.0) or 0.0),
    )
    for reason in summary.get("reasons") or []:
        log_fn("  [promotion-gate] %s", reason)


def _warmup_player_from_history(player, price_hist: list, month: int):
    base_pv = float(getattr(player, "_pv", INITIAL_CAPITAL) or INITIAL_CAPITAL)
    for i, hist_prices in enumerate(price_hist):
        try:
            player.act(
                prices=hist_prices,
                volumes={},
                month=month,
                portfolio_value=base_pv,
                bar_index=i + 1,
            )
        except Exception:
            pass


def _swap_real_player(agents: OrderedDict, capturing_agent: EnhancedSignalCapture,
                      bridge, new_player_name: str, new_player, stats):
    if hasattr(capturing_agent, "set_inner"):
        capturing_agent.set_inner(new_player)
    else:
        capturing_agent._inner = new_player
    agents.clear()
    agents[new_player_name] = capturing_agent
    bridge.agents = agents
    stats.agent_name = f"{new_player_name} ({EXCHANGE_NAME} COMBO)"


def _create_real_player():
    """
    Creates the live player.

    Default stays Panteon. For safe experiments the caller can set
    PANTEON_REAL_PLAYER=PanteonResearch.
    """
    player_name = (
        os.getenv("PANTEON_REAL_PLAYER")
        or "Panteon"
    ).strip() or "Panteon"
    allowed = {
        "Panteon",
        "PanteonResearch",
        "PlayerFunding",
        "PlayerBomberman",
        "NeuroPlayer",
        "PanteonNextResearch",
    }
    if player_name not in allowed:
        raise ValueError(f"unsupported PANTEON_REAL_PLAYER={player_name!r}; allowed={sorted(allowed)}")
    if player_name == "PanteonNextResearch" and not _env_flag("PANTEON_NEXT_LIVE_OPT_IN"):
        raise ValueError(
            "PanteonNextResearch requires PANTEON_NEXT_LIVE_OPT_IN=1 for any live-session selection"
        )
    return player_name, _instantiate_real_player_by_name(player_name)


def main():
    # ── Папка результатов ──────────────────────────────────────────────────
    session_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(script_dir, RESULTS_ROOT, session_date)
    os.makedirs(out, exist_ok=True)
    _setup_file_logging(out)

    log.info("═" * 78)
    log.info("  🏛 PANTEON TRADE — Live + Shadow Analytics")
    log.info("  Результаты: %s", out)
    log.info("  Дашборд: каждые %d сек  |  Leaderboard: каждые %d сек",
             DASHBOARD_INTERVAL_SEC, LEADERBOARD_INTERVAL_SEC)
    log.info("═" * 78)

    # ── MEXC клиент ────────────────────────────────────────────────────────
    direct_client = _create_direct_client()

    # ── Снапшот аккаунта ───────────────────────────────────────────────────
    log.info("\n  📊 ДИАГНОСТИКА АККАУНТА:")
    snapshot: dict = {}
    try:
        snapshot = direct_client.get_full_snapshot()
        direct_client.log_snapshot(snapshot)
    except Exception as e:
        log.warning("  Ошибка снапшота: %s", e)
        snapshot = {
            'futures': {'equity': 0, 'available': 0, 'unrealized': 0, 'positions': []},
            'spot': {'usdt': 0, 'crypto': {}, 'total_value': 0},
            'total_equity': 0, 'primary_capital': 0,
        }

    detected_capital = snapshot.get('primary_capital', 0.0)
    if detected_capital < 1.0:
        detected_capital = snapshot['spot'].get('usdt', 0) or INITIAL_CAPITAL
    log.info("  💰 Капитал: $%.4f", detected_capital)

    bonus = {
        "real_usdt": detected_capital, "stable_total": detected_capital,
        "bonus_tokens": {}, "locked_total": 0,
        "tradeable_total": detected_capital,
        "has_funds": detected_capital >= MIN_BALANCE_USD,
    }

    # ── Проверка прав API ──────────────────────────────────────────────────
    has_futures_perm = _check_futures_api_permission(direct_client)
    if has_futures_perm:
        log.info("  ✅ Фьючерсная торговля: разрешена")
    else:
        log.error("  ❌ НЕТ ПРАВ НА ФЬЮЧЕРСНУЮ ТОРГОВЛЮ — включите Contract Trading!")

    # ── Инструменты анализа ────────────────────────────────────────────────
    csv_logger = SignalCSVLogger(out)
    agreement = AgreementTracker()
    regime_tracker = RegimeTracker()
    promotion_summary = None
    warmup_shadow_perf: Dict[str, dict] = {}

    # ── Настройка модулей ──────────────────────────────────────────────────
    mc = _exchange_adapter

    raw_cfg = _exchange_adapter.load_settings()
    parsed_cfg = _exchange_adapter.parse_settings(raw_cfg)
    # Строим итоговый список символов: ядро + топ с биржи с фильтрами.
    # Кэшируем в переменной, чтобы не делать HTTP-запрос повторно.
    trading_symbols = build_trading_universe()
    parsed_cfg.update({
        "initial_capital": detected_capital,
        "trade_fraction": TRADE_FRACTION,
        "leverage": LEVERAGE,
        "spot_fee": SPOT_FEE,
        "futures_fee": FUTURES_FEE,
        "symbols": trading_symbols,
    })

    _common_cfg = dict(
        BAR=parsed_cfg.get("bar", 60),
        TRADE_FRACTION=TRADE_FRACTION,
        INITIAL_CAPITAL=detected_capital,
        LEVERAGE=float(LEVERAGE),
        cfg=raw_cfg,
    )
    _master_cfg = {**_common_cfg, "SPOT_FEE": SPOT_FEE, "FUTURES_FEE": FUTURES_FEE}

    for mod_name in ("crypto_agents", "panteon_agents", "master_player"):
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "configure"):
                kw = _master_cfg if mod_name == "master_player" else _common_cfg
                try:
                    mod.configure(**kw)
                except TypeError as _te:
                    import inspect as _inspect
                    _params = set(_inspect.signature(mod.configure).parameters)
                    kw_safe = {k: v for k, v in kw.items() if k in _params}
                    log.debug("configure(%s): TypeError '%s' -> retry with %s",
                              mod_name, _te, list(kw_safe))
                    mod.configure(**kw_safe)
        except Exception as e:
            log.debug("optional module %s skipped during startup: %s", mod_name, e)

    if not _exchange_adapter.check_connectivity():
        log.error("Нет связи с API биржи!")
        sys.exit(1)

    # ── Статистика ─────────────────────────────────────────────────────────
    stats = TradingStats(detected_capital)
    stats.bonus_info = bonus
    stats.has_futures_perm = has_futures_perm
    stats.agent_name = f"Panteon ({EXCHANGE_NAME} TRADE)"
    stats.record_tick(datetime.now(tz=timezone.utc), detected_capital, 0.0, live_bar=0)

    # ── Реальный агент ─────────────────────────────────────────────────────
    try:
        selected_player_name, raw_player = _create_real_player()
        if hasattr(raw_player, 'set_memory_namespace'):
            raw_player.set_memory_namespace(EXCHANGE_NAME)
        if hasattr(raw_player, 'enable_real_memory'):
            raw_player.enable_real_memory()
        log.info("  [real-player] %s", selected_player_name)
        log.info("\n  🤖 Реальный агент: %s", selected_player_name)
    except Exception as e:
        log.warning("  Panteon недоступен (%s) — fallback PlayerStop", e)
        from crypto_players import PlayerStop
        raw_player = PlayerStop()
    selected_player_name = type(raw_player).__name__
    stats.agent_name = f"{selected_player_name} ({EXCHANGE_NAME} COMBO)"

    capturing_agent = EnhancedSignalCapture(
        raw_player, stats, csv_logger, agreement, regime_tracker,
    )
    agents = OrderedDict([(selected_player_name, capturing_agent)])

    # Показываем суб-агенты
    if hasattr(raw_player, 'subagent_labels'):
        _sub_names = raw_player.subagent_labels()
    elif hasattr(raw_player, 'iter_subagents'):
        _sub_names = [name for _, name, _ in raw_player.iter_subagents()]
    else:
        _sub_names = []
        for a in ("_ms", "_lch", "_fa", "_las", "_gb"):
            obj = getattr(raw_player, a, None)
            if obj:
                _sub_names.append(type(obj).__name__)
    if _sub_names:
        log.info("  Суб-агенты: %s", ", ".join(_sub_names))

    # ── Shadow-агенты ──────────────────────────────────────────────────────
    log.info("\n  👻 Создаём shadow-агентов (виртуальная торговля)...")
    shadows = _create_shadow_agents(detected_capital)

    # ── Bridge ─────────────────────────────────────────────────────────────
    bridge_cls = _resolve_bridge_class(mc)
    bridge_kwargs = dict(
        agents=agents,
        cfg=parsed_cfg,
        mode=TRADING_MODE,
        api_key=API_KEY,
        api_secret=API_SECRET,
        output_dir=out,
        direct_client=direct_client,
    )
    if API_PASSPHRASE:
        bridge_kwargs["api_passphrase"] = API_PASSPHRASE
    try:
        bridge = bridge_cls(**bridge_kwargs)
    except TypeError:
        bridge_kwargs.pop("api_passphrase", None)
        bridge = bridge_cls(**bridge_kwargs)
    bridge._stats_ref = stats

    # Патчим bridge для shadow-агентов
    # ВАЖНО: патчим ДО warmup, чтобы shadow-агенты тоже прогрелись
    _patch_bridge_for_shadows(bridge, shadows, csv_logger, agreement, regime_tracker,
                              real_player=raw_player)

    # ── Monitor ────────────────────────────────────────────────────────────
    monitor = ComboMonitorThread(
        direct_client, stats, out, shadows, agreement, regime_tracker,
    )
    monitor._bridge = bridge

    # ── Прогрев ────────────────────────────────────────────────────────────
    log.info("═" * 78)
    log.info("  ФАЗА 1: ПРОГРЕВ (%d баров ≈ %dч)",
             (mc.warmup_bars or 0),
             (mc.warmup_bars or 0) // 60)
    log.info("═" * 78)

    bridge.warmup(n_bars=mc.warmup_bars or 0)
    actual_warmup_end = int(
        getattr(bridge, "_warmup_end", 0)
        or getattr(bridge, "_bar", 0)
        or mc.warmup_bars
        or 0
    )
    stats.bar_count = actual_warmup_end
    stats.live_bar_count = 0
    monitor._warmup_end_bar = actual_warmup_end
    capturing_agent.enable_capture(actual_warmup_end)
    log.info("  Прогрев завершён (bar=%d)", bridge._bar)

    # ══════════════════════════════════════════════════════════════════════
    # КРИТИЧЕСКИЙ FIX: сброс внутренних позиций суб-агентов после warmup
    # ══════════════════════════════════════════════════════════════════════
    # BUG: После warmup sub-agents (FundingArb, MomentumScalper, etc.)
    # внутри Panteon сохраняют позиции в self.pos из warmup-данных.
    # Эти позиции не существуют в реальности. Из-за них sub-agents:
    #   1. Пропускают символы (if cur is not None: continue)
    #   2. Не генерируют новые сигналы
    #   3. Panteon получает пустые голоса → HOLD навсегда
    # Результат: Panteon = НОЛЬ сигналов за 35ч → -35% P&L
    #
    # FIX: Сброс self.pos, self.ep, self.entry_px, self._lc для всех
    # суб-агентов, чтобы они начали live-торговлю с чистого состояния.
    # Также: _reset_subagent_timers использовал _last_check, но агенты
    # используют _lc → таймеры НЕ сбрасывались.
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n  🔧 СБРОС СОСТОЯНИЯ SUB-АГЕНТОВ ПОСЛЕ WARMUP")
    if hasattr(raw_player, 'reset_for_live'):
        raw_player.reset_for_live(bridge._bar)
    else:
        _reset_player_subagents_for_live(raw_player, bridge._bar)
        if hasattr(raw_player, '_open_pos'):
            old_count = len(raw_player._open_pos)
            raw_player._open_pos.clear()
            log.info("  ✅ Panteon._open_pos очищен (%d warmup позиций удалено)", old_count)
        if hasattr(raw_player, '_r'):
            raw_player._r = None
        if hasattr(raw_player, '_lr'):
            raw_player._lr = 0
    log.info("  ✅ Режим Panteon сброшен → пересчитается на первом live баре")

    # ── Прогрев shadow-агентов на тех же исторических данных ───────────
    log.info("  👻 Прогрев shadow-агентов через %d баров истории...",
             len(bridge._price_hist))
    _shadow_warmup_start = time.time()
    month = datetime.utcnow().month
    for i, hist_prices in enumerate(bridge._price_hist):
        bar_idx = i + 1
        for name, (agent, vp) in shadows.items():
            try:
                agent.act(
                    prices=hist_prices, volumes={},
                    month=month, portfolio_value=vp.initial_capital,
                    bar_index=bar_idx,
                )
            except Exception:
                pass
        # Прогресс
        if (i + 1) % max(1, len(bridge._price_hist) // 5) == 0:
            log.info("    shadow warmup: %d/%d (%.0f%%)",
                     i + 1, len(bridge._price_hist),
                     (i + 1) / len(bridge._price_hist) * 100)
    log.info("  ✅ Shadow warmup за %.1f сек", time.time() - _shadow_warmup_start)

    warmup_prices = bridge._price_hist[-1] if bridge._price_hist else {}
    warmup_shadow_perf = _build_shadow_perf_from_shadows(shadows, prices=warmup_prices)
    try:
        from promotion_gate import PromotionGateAgent
        promotion_summary = PromotionGateAgent().evaluate(warmup_shadow_perf)
        _log_promotion_gate_summary(promotion_summary)
    except Exception as e:
        log.debug("promotion gate evaluate failed: %s", e)

    if hasattr(raw_player, 'set_shadow_perf') and warmup_shadow_perf:
        try:
            raw_player.set_shadow_perf(warmup_shadow_perf)
        except Exception as e:
            log.debug("warmup shadow perf inject failed: %s", e)

    if selected_player_name == "PanteonNextResearch":
        force_next_live = _env_flag("PANTEON_NEXT_FORCE_LIVE")
        gate_ready = isinstance(promotion_summary, dict) and bool(promotion_summary.get("ready"))
        if force_next_live and isinstance(promotion_summary, dict) and not gate_ready:
            _log_promotion_gate_summary(promotion_summary, forced=True)
        elif not gate_ready and not force_next_live:
            fallback_name = (
                os.getenv("PANTEON_NEXT_LIVE_FALLBACK")
                or "Panteon"
            ).strip() or "Panteon"
            if fallback_name == "PanteonNextResearch":
                fallback_name = "Panteon"
            log.warning(
                "  [promotion-gate] %s не прошёл gate -> fallback на %s",
                selected_player_name, fallback_name,
            )
            replacement = _instantiate_real_player_by_name(fallback_name)
            if hasattr(replacement, 'set_memory_namespace'):
                replacement.set_memory_namespace(EXCHANGE_NAME)
            if hasattr(replacement, 'enable_real_memory'):
                replacement.enable_real_memory()
            _warmup_player_from_history(replacement, bridge._price_hist, month)
            if hasattr(replacement, 'set_shadow_perf') and warmup_shadow_perf:
                try:
                    replacement.set_shadow_perf(warmup_shadow_perf)
                except Exception:
                    pass
            if hasattr(replacement, 'reset_for_live'):
                replacement.reset_for_live(bridge._bar)
            _swap_real_player(agents, capturing_agent, bridge, fallback_name, replacement, stats)
            raw_player = replacement
            selected_player_name = fallback_name
            log.info("  [real-player] fallback active -> %s", selected_player_name)
        else:
            log.info("  [promotion-gate] %s допущен к live-сессии", selected_player_name)

    # Сброс shadow-агентов после warmup (та же проблема: stale positions)
    log.info("  🔧 Сброс shadow-агентов для live...")
    for name, (agent, vp) in shadows.items():
        _reset_agent_internals(agent, bridge._bar, name)
        # Очищаем виртуальный портфель — warmup P&L не релевантен
        vp.reset()
    log.info("  ✅ Shadow-агенты сброшены: %d агентов", len(shadows))

    # ── Синхронизация ──────────────────────────────────────────────────────
    log.info("\n  🔄 Синхронизация с биржей после прогрева...")
    try:
        live_snapshot = direct_client.get_full_snapshot()
        direct_client.log_snapshot(live_snapshot)

        live_capital = live_snapshot.get('primary_capital', 0.0)
        if live_capital >= 1.0 and abs(live_capital - detected_capital) > 0.01:
            detected_capital = live_capital
            bridge.initial_capital = detected_capital
            log.info("  💡 Капитал обновлён: $%.4f", detected_capital)

        # Инъекция в реального игрока
        raw_player_ref = getattr(capturing_agent, '_inner', raw_player)
        inject_live_state_into_player(raw_player_ref, live_snapshot, bridge._bar)
        log.info("  ✅ Синхронизация завершена")
    except Exception as e:
        log.warning("  Синхронизация не удалась: %s", e)

    try:
        monitor._refresh(render_dashboards=True)
    except Exception as e:
        log.debug("  post-sync monitor refresh failed: %s", e)

    save_dashboard(stats, out, "dashboard_after_warmup.png")
    if not monitor.is_alive():
        monitor.start()

    # ── Live ───────────────────────────────────────────────────────────────
    log.info("═" * 78)
    log.info("  ФАЗА 2: LIVE COMBO TRADING")
    log.info("  Реальный: %s  |  Shadow: %d агентов", selected_player_name, len(shadows))
    log.info("  Ctrl+C для остановки")
    log.info("═" * 78)

    try:
        bridge.run()
    except KeyboardInterrupt:
        log.info("  Остановка по Ctrl+C...")
    except Exception as e:
        log.error("  Критическая ошибка: %s", e, exc_info=True)
    finally:
        monitor.stop()
        csv_logger.close()

        # Финальный отчёт
        try:
            save_dashboard(stats, out, "dashboard_FINAL.png")
            save_summary_json(stats, out)
        except Exception:
            pass

        # Финальный leaderboard
        vp_map = {name: tup[1] for name, tup in shadows.items()}
        try:
            final_shadow_perf = _build_shadow_perf_from_shadows(shadows)
            if hasattr(raw_player, 'set_shadow_perf'):
                raw_player.set_shadow_perf(final_shadow_perf)
            if hasattr(raw_player, 'save_memory_snapshot'):
                raw_player.save_memory_snapshot(force=True, reason="shutdown")
        except Exception as e:
            log.debug("real player memory save failed: %s", e)
        if not ROTATION_CHANGE_LOGS_ONLY:
            log_leaderboard(vp_map, stats.pnl_pct, regime_tracker.current,
                            stats.bar_count, regime_tracker)
            log.info(agreement.log_matrix())

        # Финальный JSON
        try:
            final_report = {
                'session': session_date,
                'real_player': selected_player_name,
                'duration': stats.uptime_str,
                'real_pnl_pct': round(stats.pnl_pct, 4),
                'real_signals': len(stats.signals),
                'real_trades': len(stats.trades),
                'regime_summary': regime_tracker.summary(),
                'regime_changes': len(regime_tracker.history),
                'promotion_gate': promotion_summary,
                'shadows': {},
            }
            for name, vp in vp_map.items():
                final_report['shadows'][name] = {
                    'pnl_pct': round(vp.pnl_pct, 4),
                    'win_rate': round(vp.win_rate, 2),
                    'signals': vp.signal_count,
                    'entries': vp.entry_count,
                    'closed_trades': vp.close_count,
                    'total_trades': vp.total_trades,
                    'sharpe': round(vp.sharpe(), 4),
                    'max_dd_pct': round(vp.max_drawdown * 100, 2),
                    'last_trades': vp.trades_log[-10:],
                }
            path = os.path.join(out, "final_combo_report.json")
            _write_json_atomic(path, final_report, log_context="final_combo_report")
        except Exception as exc:
            log.warning("  [final_combo_report] save failed: %s", exc)

        log.info("═" * 78)
        log.info("  COMBO TRADE завершён.")
        log.info("  Результаты:       %s", out)
        log.info("  Сигналов (real):   %d", len(stats.signals))
        log.info("  Сделок (real):     %d", len(stats.trades))
        log.info("  P&L (real):        $%.2f (%.2f%%)", stats.pnl, stats.pnl_pct)
        log.info("  Shadow-агентов:    %d", len(shadows))
        log.info("  Смен режима:       %d", len(regime_tracker.history))
        log.info("  CSV сигналов:      %s", SIGNAL_CSV_FILE)
        log.info("═" * 78)


if __name__ == "__main__":
    main()
