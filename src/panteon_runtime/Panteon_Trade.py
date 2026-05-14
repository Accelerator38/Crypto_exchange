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

from env_bootstrap import load_local_env
from exchange_registry import load_exchange_runtime
from project_paths import PROJECT_ROOT, RUNTIME_DIR, add_runtime_paths


def _bootstrap_project_paths():
    add_runtime_paths()


_bootstrap_project_paths()
load_local_env(PROJECT_ROOT)


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


def _exchange_settings_float(name: str, default: float) -> float:
    prefix = EXCHANGE_NAME.lower()
    raw_cfg = _settings_raw if isinstance(_settings_raw, dict) else {}
    for key in (f"{prefix}_{name}", f"{name}_{prefix}"):
        if key in raw_cfg:
            try:
                return float(raw_cfg[key])
            except (TypeError, ValueError):
                return float(default)
    try:
        return float(_settings_parsed.get(name, default))
    except (TypeError, ValueError):
        return float(default)


INITIAL_CAPITAL  = 50.0
TRADE_FRACTION   = _exchange_settings_float("trade_fraction", 0.15)
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
MARKET_REGIMES = ("bullish", "bearish", "neutral", "crash")


def _canonical_dashboard_regime(regime: str) -> str:
    text = str(regime or "neutral").strip().lower()
    if text in ("bull", "bullish", "uptrend", "risk_on"):
        return "bullish"
    if text in ("bear", "bearish", "downtrend", "risk_off"):
        return "bearish"
    if text in ("crash", "panic", "capitulation"):
        return "crash"
    return "neutral"


_REGIME_CLASSIFIER_REF = None  # set by Panteon_Trade wiring to the real Panteon


def set_dashboard_regime_classifier(classifier) -> None:
    """Register a fallback per-symbol regime classifier (real Panteon).

    Needed because simple shadow agents lack `_build_symbol_profile` /
    `_detect_symbol_regime`, so without this they all silently fall back to the
    global regime and per_regime stats became NEUTRAL-only for agents while
    players (Panteon instances) saw BULLISH/BEARISH. This split polluted the
    regime-keyed memory and broke comparability between the agent and player
    dashboards.
    """
    global _REGIME_CLASSIFIER_REF
    _REGIME_CLASSIFIER_REF = classifier


def _detect_dashboard_symbol_regime(agent, sym: str, fallback: str = "neutral") -> str:
    """Best-effort per-symbol regime for shadow analytics."""
    try:
        if hasattr(agent, "_build_symbol_profile"):
            profile = agent._build_symbol_profile(sym)
            if isinstance(profile, dict):
                regime = profile.get("symbol_regime")
                if regime and str(regime).lower() != "unknown":
                    return _canonical_dashboard_regime(regime)
    except Exception:
        pass

    hist = None
    try:
        ph = getattr(agent, "_ph", None)
        if isinstance(ph, dict):
            hist = list(ph.get(sym) or [])
        elif hasattr(ph, "get"):
            hist = list(ph.get(sym) or [])
    except Exception:
        hist = None

    try:
        if hasattr(agent, "_detect_symbol_regime"):
            regime = agent._detect_symbol_regime(sym, hist)
            if regime and str(regime).lower() != "unknown":
                return _canonical_dashboard_regime(regime)
    except Exception:
        pass

    # Shadow agents that lack their own MTF classifier must still see the same
    # per-symbol regime as shadow players — otherwise per_regime accounting and
    # the regime-keyed memory diverge. Delegate to the registered real Panteon.
    classifier = _REGIME_CLASSIFIER_REF
    if classifier is not None and classifier is not agent:
        try:
            if hasattr(classifier, "_build_symbol_profile"):
                profile = classifier._build_symbol_profile(sym)
                if isinstance(profile, dict):
                    regime = profile.get("symbol_regime")
                    if regime and str(regime).lower() != "unknown":
                        return _canonical_dashboard_regime(regime)
        except Exception:
            pass
        try:
            if hasattr(classifier, "_detect_symbol_regime"):
                cls_ph = getattr(classifier, "_ph", None)
                cls_hist = None
                if isinstance(cls_ph, dict):
                    cls_hist = list(cls_ph.get(sym) or [])
                regime = classifier._detect_symbol_regime(sym, cls_hist or hist)
                if regime and str(regime).lower() != "unknown":
                    return _canonical_dashboard_regime(regime)
        except Exception:
            pass

    if hist is not None and len(hist) >= 61:
        try:
            last = float(hist[-1])
            if last > 0:
                short_ret = last / max(float(hist[-61]), 1e-12) - 1.0
                medium_ret = (
                    last / max(float(hist[-241]), 1e-12) - 1.0
                    if len(hist) >= 241 else short_ret
                )
                long_ret = (
                    last / max(float(hist[-721]), 1e-12) - 1.0
                    if len(hist) >= 721 else medium_ret
                )
                if medium_ret <= -0.080 or long_ret <= -0.120 or (
                    short_ret <= -0.045 and medium_ret <= -0.020
                ):
                    return "crash"
                if short_ret >= 0.004 and medium_ret >= 0.010:
                    return "bullish"
                if short_ret <= -0.004 and medium_ret <= -0.010:
                    return "bearish"
                if medium_ret >= 0.025 and short_ret > -0.002:
                    return "bullish"
                if medium_ret <= -0.025 and short_ret < 0.002:
                    return "bearish"
                return "neutral"
        except Exception:
            pass
    return _canonical_dashboard_regime(fallback)

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

script_dir = str(RUNTIME_DIR)
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

_HTML_DASHBOARD_LAST_REFRESH = 0.0
_HTML_DASHBOARD_INTERVAL_SEC = float(os.getenv("PANTEON_HTML_DASHBOARD_INTERVAL_SEC", "60"))


def _refresh_html_dashboard_best_effort(force: bool = False) -> None:
    global _HTML_DASHBOARD_LAST_REFRESH
    now = time.time()
    if not force and (now - _HTML_DASHBOARD_LAST_REFRESH) < _HTML_DASHBOARD_INTERVAL_SEC:
        return
    _HTML_DASHBOARD_LAST_REFRESH = now
    try:
        import importlib.util

        dashboard_path = PROJECT_ROOT / "tools" / "build_dashboard.py"
        if not dashboard_path.exists():
            return
        spec = importlib.util.spec_from_file_location("panteon_build_dashboard", dashboard_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "build_dashboard"):
            module.build_dashboard(quiet=True)
    except Exception as exc:
        log.debug("HTML dashboard refresh skipped: %s", exc)


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
                 trade_fraction: float = 0.15, fee: float = 0.0002,
                 # FIX H4 (2026-04-27): моделирование издержек, без которых
                 # shadow-метрики систематически переоценивают churn-агентов.
                 # Дефолты соответствуют settings.txt: slippage=0.0001 = 1 bp,
                 # средний spread по ликвидным USDT-fut ≈ 1 bp,
                 # funding_rate=0.00006/8h → 0.00006/(8*60) на минутном баре.
                 slippage: float = 0.0001,
                 spread: float = 0.0001,
                 funding_per_bar: float = 6e-7):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.leverage = leverage
        self.trade_fraction = trade_fraction
        self.fee = fee
        self.slippage = float(slippage)
        self.spread = float(spread)
        self.funding_per_bar = float(funding_per_bar)

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
        self.regime_stats: Dict[str, dict] = {}

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
        self.regime_stats.clear()

    def _regime_stats(self, regime: str) -> dict:
        key = _canonical_dashboard_regime(regime)
        stats = self.regime_stats.get(key)
        if stats is None:
            stats = {
                'ticks': 0,
                'position_ticks': 0,
                'signals': 0,
                'entries': 0,
                'closed_trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'equity_start': 0.0,
                'equity_end': 0.0,
            }
            self.regime_stats[key] = stats
        return stats

    def _add_regime_pnl(self, regime: str, pnl: float, equity_value: Optional[float] = None):
        stats = self._regime_stats(regime)
        if float(stats.get('equity_start', 0.0) or 0.0) == 0.0:
            stats['equity_start'] = float(
                self.equity_history[-1] if self.equity_history else self.initial_capital
            )
        stats['pnl'] += float(pnl or 0.0)
        if equity_value is not None:
            stats['equity_end'] = float(equity_value)

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
                risk_multiplier: float = 1.0, regime: str = ''):
        """Виртуально исполняет сигнал агента."""
        if action == 0 or price <= 0:
            return

        self.signal_count += 1
        sym_stats = self._sym_stats(sym)
        sym_stats['signals'] += 1
        sym_stats['last_bar'] = int(bar)
        signal_regime = _canonical_dashboard_regime(regime)
        signal_regime_stats = self._regime_stats(signal_regime)
        signal_regime_stats['signals'] += 1

        # FIX H4 (2026-04-27): adverse fill price = mid ± slippage ± spread/2.
        # MARKET-ордера на бирже исполняются в неблагоприятную сторону.
        adverse = self.slippage + self.spread / 2.0

        # Close actions (3, 8)
        if action in (3, 8) and sym in self.positions:
            pos = self.positions.pop(sym)
            # FIX H4: при закрытии long продаём → eff_price ниже mid;
            # при закрытии short покупаем → eff_price выше mid.
            eff_price = price * (1 - adverse) if pos['side'] == 'long' else price * (1 + adverse)
            pnl = self._calc_pnl(pos, eff_price)
            fee_cost = abs(pos['qty'] * eff_price * self.fee)
            net_pnl = pnl - fee_cost
            prev_mark = float(pos.get('last_mark', pos['entry']) or pos['entry'])
            prev_unrealized = self._calc_pnl(pos, prev_mark)
            trade_regime = _canonical_dashboard_regime(pos.get('regime') or signal_regime)
            # FIX v8: возвращаем МАРЖУ (не полную нотионал стоимость!)
            # Было: qty * entry = notional = margin * leverage → cash раздувался в leverage раз
            margin = pos['qty'] * pos['entry'] / self.leverage
            self.cash += margin + net_pnl
            self._add_regime_pnl(trade_regime, pnl - prev_unrealized - fee_cost, self.cash)
            self.total_pnl += net_pnl
            self.close_count += 1
            self.total_trades += 1
            trade_regime_stats = self._regime_stats(trade_regime)
            trade_regime_stats['closed_trades'] += 1
            sym_stats['closed_trades'] += 1
            sym_stats['realized_pnl'] += float(net_pnl)
            sym_stats['last_pnl'] = float(net_pnl)
            if net_pnl > 0:
                self.wins += 1
                trade_regime_stats['wins'] += 1
                sym_stats['wins'] += 1
            else:
                self.losses += 1
                trade_regime_stats['losses'] += 1
                sym_stats['losses'] += 1
            self.trades_log.append({
                'bar': bar, 'sym': sym, 'side': pos['side'],
                'entry': pos['entry'], 'exit': eff_price,
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
        # FIX H4 (2026-04-27): открытие long → покупаем по price выше mid;
        # открытие short → продаём по price ниже mid.
        eff_price = price * (1 + adverse) if side == 'long' else price * (1 - adverse)
        pos_value = base_capital * self.trade_fraction * self.leverage * size_mult
        qty = pos_value / eff_price
        margin = pos_value / self.leverage   # = base_capital * trade_fraction

        if margin > self.cash * 0.90:
            return  # недостаточно кэша

        fee_cost = qty * eff_price * self.fee
        self.cash -= margin + fee_cost
        self._add_regime_pnl(signal_regime, -fee_cost, self.cash)
        self.entry_count += 1
        signal_regime_stats['entries'] += 1
        sym_stats['entries'] += 1
        self.positions[sym] = {
            'side': side, 'qty': qty, 'entry': eff_price, 'bar': bar,
            'last_mark': eff_price, 'regime': signal_regime,
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

    def snapshot(self, prices: dict, regime: str = '',
                 symbol_regimes: Optional[Dict[str, str]] = None):
        # FIX H4 (2026-04-27): списываем funding rate с notional удерживаемых
        # futures-позиций каждый бар. Без этого долгие удержания (carry-style
        # игроки) выглядят в shadow систематически прибыльнее, чем на бирже.
        # Знак funding в текущем коде упрощённо предполагаем положительным
        # для long (longs платят shorts при положительном funding).
        if self.positions and self.funding_per_bar > 0:
            funding_total = 0.0
            for sym, pos in self.positions.items():
                cur_p = float(prices.get(sym, pos['entry']) or pos['entry'])
                notional = abs(pos['qty']) * cur_p
                # Long платит при позитивном funding, short платит при негативном.
                # Без потока реальных ставок аппроксимируем: каждая сторона платит
                # по половине абсолютного среднего расход → симметричный haircut.
                cost = notional * self.funding_per_bar * 0.5
                funding_total += cost
            self.cash -= funding_total
        eq = self.get_equity(prices)
        self.equity_history.append(eq)
        symbol_regimes = symbol_regimes or {}
        seen_regimes = set()
        for sym, pos in self.positions.items():
            try:
                cur_price = float(prices.get(sym, pos['entry']) or pos['entry'])
                prev_mark = float(pos.get('last_mark', pos['entry']) or pos['entry'])
            except (TypeError, ValueError):
                continue
            cur_pnl = self._calc_pnl(pos, cur_price)
            prev_pnl = self._calc_pnl(pos, prev_mark)
            pos['last_mark'] = cur_price
            pos_regime = _canonical_dashboard_regime(
                symbol_regimes.get(sym) or pos.get('regime') or regime
            )
            if pos_regime not in seen_regimes:
                stats = self._regime_stats(pos_regime)
                stats['ticks'] += 1
                stats['position_ticks'] += 1
                seen_regimes.add(pos_regime)
            self._add_regime_pnl(pos_regime, cur_pnl - prev_pnl, eq)
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

    def export_regime_stats(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for regime in MARKET_REGIMES:
            stats = dict(self.regime_stats.get(regime) or {})
            ticks = int(stats.get('ticks', 0) or 0)
            wins = int(stats.get('wins', 0) or 0)
            losses = int(stats.get('losses', 0) or 0)
            closed = int(stats.get('closed_trades', 0) or 0)
            pnl = float(stats.get('pnl', 0.0) or 0.0)
            out[regime] = {
                'ticks': ticks,
                'position_ticks': int(stats.get('position_ticks', 0) or 0),
                'signals': int(stats.get('signals', 0) or 0),
                'entries': int(stats.get('entries', 0) or 0),
                'closed_trades': closed,
                'wins': wins,
                'losses': losses,
                'win_rate': wins / max(closed, 1) * 100.0 if closed else 0.0,
                'pnl': pnl,
                'pnl_pct': pnl / max(self.initial_capital, 1e-9) * 100.0,
                'active_ratio': (
                    float(stats.get('position_ticks', 0) or 0) / max(ticks, 1)
                    if ticks else 0.0
                ),
                'equity_start': float(stats.get('equity_start', 0.0) or 0.0),
                'equity_end': float(stats.get('equity_end', 0.0) or 0.0),
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
        'selected_player', 'selected_agents', 'order_result',
        'order_id', 'position_source',
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
            risk_multiplier=None, selected_player: str = '',
            selected_agents: str = '', order_result: str = '',
            order_id: str = '', position_source: str = ''):
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
            'selected_player': selected_player,
            'selected_agents': selected_agents or contributors,
            'order_result': order_result,
            'order_id': order_id,
            'position_source': position_source,
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
                             VolBreakoutHunter,
                             BullRotationAgent, BearReliefFadeAgent,
                             NeutralRangeScalper, CrashPanicShortAgent,
                             ExternalSignalAgent, ResearchValidatorAgent,
                             RichardDennisTurtle, Bomberman,
                             PlayerBomberman, PlayerFunding)
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
            ("V_VolBreakoutHunter", VolBreakoutHunter),
            ("V_BullRotationAgent", BullRotationAgent),
            ("V_BearReliefFadeAgent", BearReliefFadeAgent),
            ("V_NeutralRangeScalper", NeutralRangeScalper),
            ("V_CrashPanicShortAgent", CrashPanicShortAgent),
            ("V_ExternalSignalAgent", ExternalSignalAgent),
            ("V_ResearchValidatorAgent", ResearchValidatorAgent),
            ("V_RichardDennisTurtle", RichardDennisTurtle),
            ("V_Bomberman",        Bomberman),
            ("V_PlayerBomberman",  PlayerBomberman),
            ("V_PlayerFunding",    PlayerFunding),
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
        pass
    except Exception as e:
        log.debug("  Panteon shadow skip: %s", e)

    # --- Новый pipeline player для shadow-first сравнения ---
    try:
        pass
    except Exception as e:
        log.debug("  PanteonNextResearch shadow skip: %s", e)

    # --- Генетические агенты ---
    try:
        from crypto_genetics import GeneticsBullishAgent, GeneticsBearishAgent, GeneticsNeutralAgent
        from panteon_agents import make_genetics_panteon_agent
        for name, cls in [("V_GeneticsBullish", GeneticsBullishAgent),
                          ("V_GeneticsBearish", GeneticsBearishAgent),
                          ("V_GeneticsNeutral", GeneticsNeutralAgent)]:
            try:
                agent = make_genetics_panteon_agent(cls)
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

def _create_shadow_players(initial_capital: float) -> Dict[str, tuple]:
    """Return player-level aggregation strategies tracked in shadow mode."""
    players = OrderedDict()
    try:
        from panteon import (
            NeuroPlayer,
            Panteon,
            PanteonConsensusResearch,
            PanteonDefensiveResearch,
            PanteonMeanRevResearch,
            PanteonNextResearch,
            PanteonResearch,
            PanteonTrendResearch,
            PlayerBomberman,
            PlayerFunding,
        )
        # FIX (2026-04-29): импортируем Solo*-обёртки из panteon_agents.
        # До этого фикса внешний shadow_players-pool содержал ТОЛЬКО Pantheon-style
        # ансамбли. _create_shadow_players в Panteon_Trade.py — это та структура,
        # из которой строится shadow_perf, который потом видит селектор. Если
        # солистов нет здесь — они не в perf, и селектор никогда их не выберет
        # лидером, даже если они в Panteon._init_shadow_player_pool().
        try:
            from panteon_agents import (
                SoloFundingArb,
                SoloLiveVolCompress,
                SoloLiveRegimePullback,
                SoloLiveTrendFollow,
                SoloLiveCrashHunter,
                SoloCandlePattern,
            )
        except Exception as imp_exc:
            log.warning("  Solo* import failed: %s", imp_exc)
            SoloFundingArb = SoloLiveVolCompress = SoloLiveRegimePullback = None
            SoloLiveTrendFollow = SoloLiveCrashHunter = SoloCandlePattern = None
        specs = [
            ("V_Panteon_shadow", lambda: Panteon(enable_meta_players=False)),
            ("V_PlayerFunding", PlayerFunding),
            ("V_PlayerBomberman", PlayerBomberman),
            ("V_PanteonResearch", PanteonResearch),
            ("V_PanteonTrendResearch", PanteonTrendResearch),
            ("V_PanteonMeanRevResearch", PanteonMeanRevResearch),
            ("V_PanteonDefensiveResearch", PanteonDefensiveResearch),
            ("V_PanteonConsensusResearch", PanteonConsensusResearch),
            ("V_PanteonNextResearch", PanteonNextResearch),
            ("V_NeuroPlayer", NeuroPlayer),
        ]
        # FIX (2026-04-29): солисты добавляются только если их класс импортировался.
        for solo_name, solo_cls in [
            ("V_SoloFundingArb",         SoloFundingArb),
            ("V_SoloLiveVolCompress",    SoloLiveVolCompress),
            ("V_SoloLiveRegimePullback", SoloLiveRegimePullback),
            ("V_SoloLiveTrendFollow",    SoloLiveTrendFollow),
            ("V_SoloLiveCrashHunter",    SoloLiveCrashHunter),
            ("V_SoloCandlePattern",      SoloCandlePattern),
        ]:
            if solo_cls is not None:
                specs.append((solo_name, solo_cls))
        for name, factory in specs:
            try:
                player = factory()
                if hasattr(player, 'set_shadow_bootstrap_mode'):
                    player.set_shadow_bootstrap_mode(True)
                else:
                    setattr(player, '_shadow_bootstrap_mode', True)
                vp = VirtualPortfolio(initial_capital, LEVERAGE, TRADE_FRACTION, FUTURES_FEE)
                players[name] = (player, vp)
            except Exception as exc:
                log.debug("  Shadow player %s skip: %s", name, exc)
    except Exception as exc:
        log.warning("  shadow player import failed: %s", exc)
    log.info("  Shadow-players created: %d -> %s", len(players), list(players.keys()))
    # FIX 2026-04-27: повторно применяем settings к свежесозданным игрокам.
    try:
        from exchange_api_runtime import reapply_settings_to_existing_instances
        reapply_settings_to_existing_instances()
    except Exception as exc:
        log.debug("  reapply_settings_to_existing_instances skipped: %s", exc)
    return players


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

        # FIX 2026-05-04 (per-symbol blocklist): отсекаем сигналы для символов,
        # которые временно заблокированы из-за повторных pending → not confirmed.
        # Это останавливает бесконечную карусель «отправили — биржа не приняла —
        # удалили» для нелинквидных пар (NAORIS, AIGENSYN, BSB, LAB на BITGET).
        # Сигнал заменяется на 0 с предупреждением.
        health = getattr(self, "_position_sync_health", None)
        if health is None:
            owner = getattr(self, "_owner", None) or getattr(self, "_runtime", None)
            if owner is not None:
                health = getattr(owner, "_position_sync_health", None)
        if acts and health is not None and hasattr(health, "is_symbol_blocked"):
            blocked_now = []
            for sym in list(acts.keys()):
                action = acts.get(sym, 0)
                # Пропускаем close-actions (3, 8) и hold (0) — нужно уметь
                # закрывать существующие позиции даже у заблокированного символа.
                if not action or action in (0, 3, 8):
                    continue
                try:
                    if health.is_symbol_blocked(sym):
                        acts[sym] = 0
                        blocked_now.append(sym)
                except Exception:
                    pass
            if blocked_now:
                log.info(
                    "  [Panteon] блок-лист отсёк сигналы по символам: %s "
                    "(временный per-symbol blocklist по pending failures)",
                    ", ".join(blocked_now),
                )

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
                    selected_player = getattr(self._inner, '_selected_shadow_player', '') or self._label
                    pos_info = (_open_pos or {}).get(sym, {}) or {}
                    position_source = pos_info.get('source') or pos_info.get('position_source') or ''
                    if action in (1, 2, 4, 5, 6, 7) and not position_source:
                        position_source = 'pending_order'
                    # Записываем в stats
                    self._stats.record_signal(
                        bar_index or 0, sym, action, price,
                        agent=agent, regime=_regime,
                        risk_multiplier=risk_multiplier,
                    )
                    if self._stats.signals:
                        self._stats.signals[-1].update({
                            'selected_player': selected_player,
                            'selected_agents': agent,
                            'position_source': position_source,
                        })
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
                        selected_player=selected_player,
                        selected_agents=agent,
                        position_source=position_source,
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
                          output_dir: str,
                          output_filename: str = 'shadow_agents_dashboard.png',
                          title: str = "SHADOW AGENTS DASHBOARD - Virtual Trading Leaderboard",
                          status_map: Optional[Dict[str, str]] = None):
    """Генерирует PNG-дашборд с прогрессом всех shadow-агентов vs реального.

    FIX 2026-05-03: status_map отображает участников в карантине серым
    цветом на P&L Ranking-bar (не зелёным/красным), чтобы было сразу видно
    кто отключён от live-выбора.
    """
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

    entity_label = 'players' if 'PLAYER' in str(title or '').upper() else 'agents'
    entity_label_ru = 'игроков' if entity_label == 'players' else 'агентов'

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
    fig.suptitle(title,
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

    _style(ax_eq, f'Equity Curves — Shadow {entity_label.title()} vs Real')
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

    status_map = status_map or {}
    quarantine_statuses = {'quarantine', 'shadow_only', 'purgatory'}
    # Считаем статусы по каждому участнику, чтобы окрасить столбики
    raw_statuses = [str(status_map.get(n, '') or '').lower() for n, _ in ranked]
    names = [
        _clean_label(n, max_len=16) + (' [Q]' if s in quarantine_statuses else '')
        for (n, _), s in zip(ranked, raw_statuses)
    ]
    pnls = [vp.pnl_pct for _, vp in ranked]
    # FIX 2026-05-03: карантинные — всегда серые, не зелёные/красные.
    colors = []
    for p, st in zip(pnls, raw_statuses):
        if st in quarantine_statuses:
            colors.append(GRY)
        elif p > 0:
            colors.append(GRN)
        elif p < 0:
            colors.append(RED)
        else:
            colors.append(GRY)

    y_pos = range(len(names))
    bar_alpha = [0.45 if s in quarantine_statuses else 0.8 for s in raw_statuses]
    # matplotlib не поддерживает per-bar alpha напрямую — рисуем по одному
    bars = ax_bar.barh(y_pos, pnls, color=colors, alpha=0.8, height=0.7)
    for bar, a in zip(bars, bar_alpha):
        bar.set_alpha(a)
    for idx, pnl_val in enumerate(pnls):
        if abs(float(pnl_val or 0.0)) <= 1e-9:
            ax_bar.scatter(0, idx, color=GRY, s=18, zorder=4)
    # Линия реального агента
    ax_bar.axvline(real_pnl, color=YLW, lw=2, linestyle='--',
                   label=f'REAL {real_pnl:+.1f}%')
    ax_bar.axvline(0, color=WHT, lw=0.5, alpha=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=6.5, color=WHT)
    # Перекрасить метки карантинных в серый
    try:
        for tick_lbl, st in zip(ax_bar.get_yticklabels(), raw_statuses):
            if st in quarantine_statuses:
                tick_lbl.set_color(GRY)
    except Exception:
        pass
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
    if len(lines) >= 5:
        lines[4] = f"Shadow {entity_label_ru}: {len(vp_map)}"
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
    path = os.path.join(output_dir, output_filename)
    fig.savefig(path, dpi=130, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    log.debug("  📊 Shadow dashboard → %s", path)


def save_shadow_agents_dashboard(vp_map: Dict[str, 'VirtualPortfolio'], stats,
                                 regime_tracker, output_dir: str,
                                 status_map: Optional[Dict[str, str]] = None):
    save_shadow_dashboard(
        vp_map,
        stats,
        regime_tracker,
        output_dir,
        output_filename='shadow_agents_dashboard.png',
        title="SHADOW AGENTS DASHBOARD - Virtual Trading Leaderboard",
        status_map=status_map,
    )


def save_shadow_player_dashboard(vp_map: Dict[str, 'VirtualPortfolio'], stats,
                                 regime_tracker, output_dir: str,
                                 status_map: Optional[Dict[str, str]] = None):
    save_shadow_dashboard(
        vp_map,
        stats,
        regime_tracker,
        output_dir,
        output_filename='shadow_player_dashboard.png',
        title="SHADOW PLAYERS DASHBOARD - Aggregator Strategy Leaderboard",
        status_map=status_map,
    )


def save_agent_regime_dashboard(vp_map: Dict[str, 'VirtualPortfolio'], output_dir: str,
                                output_filename: str = 'agent_regime_dashboard.png',
                                title: str = 'Agent Efficiency by Market Regime',
                                status_map: Optional[Dict[str, str]] = None):
    """Render regime efficiency heatmap for agents or players.

    FIX 2026-05-03: для участников в карантине (status_map[name] == 'quarantine'/
    'shadow_only'/'purgatory') строка целиком отрисовывается серой (приглушённой)
    с пометкой [Q] в названии, чтобы операторы видели, что это запас, а не
    активный игрок live.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if not vp_map:
        return

    rows = []
    for name, vp in vp_map.items():
        per_regime = vp.export_regime_stats()
        total_abs = sum(abs(float(per_regime[r].get('pnl_pct', 0.0) or 0.0)) for r in MARKET_REGIMES)
        total_signals = sum(int(per_regime[r].get('signals', 0) or 0) for r in MARKET_REGIMES)
        if total_abs <= 0.0 and total_signals <= 0:
            continue
        rows.append((name, vp, per_regime))

    if not rows:
        return

    rows.sort(key=lambda item: item[1].pnl_pct, reverse=True)
    rows = rows[:22]
    matrix = []
    annot = []
    for _name, _vp, per_regime in rows:
        row = []
        ann_row = []
        for regime in MARKET_REGIMES:
            stats = per_regime.get(regime, {})
            pnl_pct = float(stats.get('pnl_pct', 0.0) or 0.0)
            ticks = int(stats.get('ticks', 0) or 0)
            signals = int(stats.get('signals', 0) or 0)
            row.append(np.nan if ticks <= 0 and signals <= 0 else pnl_pct)
            ann_row.append("" if ticks <= 0 and signals <= 0 else f"{pnl_pct:+.2f}%\nS{signals}")
        matrix.append(row)
        annot.append(ann_row)

    arr = np.array(matrix, dtype=float)
    max_abs = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    max_abs = max(float(max_abs), 0.5)

    BG = '#0D1117'
    MID = '#161B22'
    WHT = '#E6EDF3'
    GRY = '#8B949E'
    GRID = '#21262D'

    height = max(7.0, 0.42 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(11, height), facecolor=BG)
    ax.set_facecolor(MID)
    cmap = plt.get_cmap('RdYlGn')
    try:
        cmap = cmap.copy()
        cmap.set_bad('#30363D')
    except Exception:
        pass
    im = ax.imshow(arr, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect='auto')

    status_map = status_map or {}
    quarantine_statuses = {'quarantine', 'shadow_only', 'purgatory'}
    row_statuses = [
        str(status_map.get(name, '') or '').lower()
        for name, _, _ in rows
    ]
    labels = [
        name.replace('V_', '')[:24] + (' [Q]' if st in quarantine_statuses else '')
        for (name, _, _), st in zip(rows, row_statuses)
    ]
    ax.set_xticks(range(len(MARKET_REGIMES)))
    ax.set_xticklabels([r.upper() for r in MARKET_REGIMES], color=WHT, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=WHT, fontsize=8)
    # FIX 2026-05-03: красим метки карантинных строк в серый.
    try:
        for tick_lbl, st in zip(ax.get_yticklabels(), row_statuses):
            if st in quarantine_statuses:
                tick_lbl.set_color(GRY)
    except Exception:
        pass
    ax.set_title(title, color=WHT, fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors=GRY)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isfinite(arr[i, j]):
                continue
            color = '#0D1117' if abs(arr[i, j]) > max_abs * 0.45 else WHT
            ax.text(j, i, annot[i][j], ha='center', va='center', fontsize=7, color=color)
    # Серая полупрозрачная "вуаль" поверх строк карантина, чтобы цвет
    # тепловой карты (RdYlGn) визуально гасился.
    try:
        for i, st in enumerate(row_statuses):
            if st not in quarantine_statuses:
                continue
            ax.axhspan(i - 0.5, i + 0.5, facecolor=GRY, alpha=0.55, zorder=2)
    except Exception:
        pass

    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xticks(np.arange(-.5, len(MARKET_REGIMES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color=GRID, linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=GRY, labelsize=8)
    cbar.set_label('P&L % contribution', color=GRY, fontsize=8)

    path = os.path.join(output_dir, output_filename)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    log.debug("  📊 Agent regime dashboard → %s", path)


def save_player_regime_dashboard(vp_map: Dict[str, 'VirtualPortfolio'], output_dir: str,
                                  status_map: Optional[Dict[str, str]] = None):
    save_agent_regime_dashboard(
        vp_map,
        output_dir,
        output_filename='player_regime_dashboard.png',
        title='Player Efficiency by Market Regime',
        status_map=status_map,
    )


def save_combined_shadow_dashboard(agent_vp_map: Dict[str, 'VirtualPortfolio'],
                                   player_vp_map: Dict[str, 'VirtualPortfolio'],
                                   stats,
                                   regime_tracker,
                                   output_dir: str,
                                   output_filename: str = 'shadow_dashboard.png',
                                   agent_status_map: Optional[Dict[str, str]] = None,
                                   player_status_map: Optional[Dict[str, str]] = None):
    """Render one shadow dashboard: players on top, agents below."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        return

    if not agent_vp_map and not player_vp_map:
        return

    BG = '#0D1117'
    MID = '#161B22'
    GRN = '#3FB950'
    RED = '#F85149'
    BLU = '#58A6FF'
    PRP = '#BC8CFF'
    YLW = '#D29922'
    WHT = '#E6EDF3'
    GRY = '#8B949E'
    GRID = '#21262D'
    quarantine_statuses = {'quarantine', 'shadow_only', 'purgatory'}

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.34, wspace=0.25,
        width_ratios=[1.65, 1.0],
    )

    def _style(ax, title=''):
        ax.set_facecolor(MID)
        ax.tick_params(colors=GRY, labelsize=7)
        ax.set_title(title, color=WHT, fontsize=11, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, alpha=0.18, color=GRID)

    def _clean_label(name: str, max_len: int = 20) -> str:
        txt = str(name or '').replace('V_', '').replace('STP_', '')
        return txt if len(txt) <= max_len else txt[: max(0, max_len - 3)] + '...'

    def _repair_curve(values):
        out = []
        last = None
        for value in list(values or []):
            try:
                cur = float(value)
            except Exception:
                cur = last if last is not None else float('nan')
            if not np.isfinite(cur) or cur <= 0:
                cur = last if last is not None else cur
            out.append(cur)
            if np.isfinite(cur) and cur > 0:
                last = cur
        return out

    def _real_curve():
        eq = _repair_curve(getattr(stats, 'equity_curve', []) or [])
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

    real_x, real_eq = _real_curve()
    real_ic = float(getattr(stats, 'initial_capital', 0.0) or 0.0)
    real_pnl = ((real_eq[-1] / real_ic) - 1.0) * 100.0 if real_eq and real_ic > 0 else (
        float(getattr(stats, 'pnl_pct', 0.0) or 0.0)
    )

    def _curve_x(length: int):
        if length <= 0:
            return []
        if not real_x:
            return list(range(length))
        if len(real_x) >= length:
            return list(real_x[-length:])
        if len(real_x) == 1:
            return [real_x[0] + i for i in range(length)]
        return np.linspace(real_x[0], real_x[-1], length).tolist()

    def _draw_group(row: int, title: str, vp_map: Dict[str, 'VirtualPortfolio'],
                    status_map: Optional[Dict[str, str]], accent: str):
        ax_eq = fig.add_subplot(gs[row, 0])
        ax_rank = fig.add_subplot(gs[row, 1])
        _style(ax_eq, f'{title} Equity Curves')
        _style(ax_rank, f'{title} P&L Ranking')

        if not vp_map:
            for ax in (ax_eq, ax_rank):
                ax.text(0.5, 0.5, 'No data yet', transform=ax.transAxes,
                        ha='center', va='center', color=GRY, fontsize=11)
            return

        ranked = sorted(vp_map.items(), key=lambda kv: kv[1].pnl_pct, reverse=True)
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(ranked), 1)))
        for idx, (name, vp) in enumerate(ranked[:18]):
            eq = _repair_curve(getattr(vp, 'equity_history', []) or [])
            if len(eq) < 2:
                continue
            label = f"{_clean_label(name, 18)} ({vp.pnl_pct:+.1f}%)" if idx < 6 else None
            ax_eq.plot(_curve_x(len(eq)), eq, color=palette[idx % 20],
                       lw=1.8 if idx < 4 else 0.9,
                       alpha=0.9 if idx < 4 else 0.35,
                       label=label)
        if real_eq:
            ax_eq.plot(real_x, real_eq, color=YLW, lw=2.4, alpha=1.0,
                       linestyle='--', label=f'REAL ({real_pnl:+.1f}%)')
        ic = ranked[0][1].initial_capital if ranked else real_ic or 100.0
        ax_eq.axhline(ic, color=WHT, lw=0.8, alpha=0.28, linestyle=':')
        ax_eq.set_ylabel('Equity ($)', color=GRY, fontsize=8)
        ax_eq.set_xlabel('Bar', color=GRY, fontsize=8)
        ax_eq.legend(loc='upper left', fontsize=6, ncol=2,
                     facecolor=MID, edgecolor=GRID, labelcolor=WHT)

        status_map = status_map or {}
        top = ranked[:18]
        raw_statuses = [str(status_map.get(name, '') or '').lower() for name, _ in top]
        labels = [
            _clean_label(name, 18) + (' [Q]' if st in quarantine_statuses else '')
            for (name, _), st in zip(top, raw_statuses)
        ]
        values = [float(vp.pnl_pct or 0.0) for _, vp in top]
        colors = []
        for value, st in zip(values, raw_statuses):
            if st in quarantine_statuses:
                colors.append(GRY)
            elif value > 0:
                colors.append(GRN)
            elif value < 0:
                colors.append(RED)
            else:
                colors.append(accent)
        y_pos = range(len(labels))
        bars = ax_rank.barh(y_pos, values, color=colors, alpha=0.82, height=0.72)
        for bar, st in zip(bars, raw_statuses):
            if st in quarantine_statuses:
                bar.set_alpha(0.45)
        ax_rank.axvline(0, color=WHT, lw=0.6, alpha=0.45)
        ax_rank.axvline(real_pnl, color=YLW, lw=2, linestyle='--',
                        label=f'REAL {real_pnl:+.1f}%')
        ax_rank.set_yticks(y_pos)
        ax_rank.set_yticklabels(labels, fontsize=6.7, color=WHT)
        try:
            for tick_lbl, st in zip(ax_rank.get_yticklabels(), raw_statuses):
                if st in quarantine_statuses:
                    tick_lbl.set_color(GRY)
        except Exception:
            pass
        ax_rank.set_xlabel('P&L %', color=GRY, fontsize=8)
        ax_rank.legend(fontsize=7, facecolor=MID, edgecolor=GRID, labelcolor=WHT)
        ax_rank.invert_yaxis()

    _draw_group(0, 'Shadow Players', player_vp_map, player_status_map, PRP)
    _draw_group(1, 'Shadow Agents', agent_vp_map, agent_status_map, BLU)

    regime = regime_tracker.current if regime_tracker else '?'
    fig.suptitle(f'Shadow Dashboard - Players and Agents | Regime: {regime}',
                 fontsize=15, fontweight='bold', color=WHT, y=0.985)
    path = os.path.join(output_dir, output_filename)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    log.debug("Combined shadow dashboard -> %s", path)


def save_combined_regime_dashboard(agent_vp_map: Dict[str, 'VirtualPortfolio'],
                                   player_vp_map: Dict[str, 'VirtualPortfolio'],
                                   output_dir: str,
                                   output_filename: str = 'regime_dashboard.png',
                                   agent_status_map: Optional[Dict[str, str]] = None,
                                   player_status_map: Optional[Dict[str, str]] = None):
    """Render one regime dashboard: players on top, agents below, curves on the right."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        return

    if not agent_vp_map and not player_vp_map:
        return

    BG = '#0D1117'
    MID = '#161B22'
    WHT = '#E6EDF3'
    GRY = '#8B949E'
    GRID = '#21262D'
    quarantine_statuses = {'quarantine', 'shadow_only', 'purgatory'}

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.34, wspace=0.24,
        width_ratios=[1.55, 1.0],
    )

    def _style(ax, title=''):
        ax.set_facecolor(MID)
        ax.tick_params(colors=GRY, labelsize=7)
        ax.set_title(title, color=WHT, fontsize=11, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(True, alpha=0.18, color=GRID)

    def _clean_label(name: str, max_len: int = 24) -> str:
        txt = str(name or '').replace('V_', '').replace('STP_', '')
        return txt if len(txt) <= max_len else txt[: max(0, max_len - 3)] + '...'

    def _collect_rows(vp_map: Dict[str, 'VirtualPortfolio']):
        rows = []
        for name, vp in (vp_map or {}).items():
            per_regime = vp.export_regime_stats()
            total_abs = sum(abs(float(per_regime[r].get('pnl_pct', 0.0) or 0.0)) for r in MARKET_REGIMES)
            total_signals = sum(int(per_regime[r].get('signals', 0) or 0) for r in MARKET_REGIMES)
            if total_abs <= 0.0 and total_signals <= 0:
                continue
            rows.append((name, vp, per_regime))
        rows.sort(key=lambda item: item[1].pnl_pct, reverse=True)
        return rows[:22]

    def _draw_heatmap(ax, title: str, vp_map: Dict[str, 'VirtualPortfolio'],
                      status_map: Optional[Dict[str, str]]):
        rows = _collect_rows(vp_map)
        ax.set_facecolor(MID)
        if not rows:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'No regime data yet', transform=ax.transAxes,
                    ha='center', va='center', color=GRY, fontsize=11)
            ax.set_title(title, color=WHT, fontsize=11, fontweight='bold', pad=8)
            return None

        matrix, annot = [], []
        for _name, _vp, per_regime in rows:
            row, ann_row = [], []
            for regime in MARKET_REGIMES:
                stats_r = per_regime.get(regime, {})
                pnl_pct = float(stats_r.get('pnl_pct', 0.0) or 0.0)
                ticks = int(stats_r.get('ticks', 0) or 0)
                signals = int(stats_r.get('signals', 0) or 0)
                row.append(np.nan if ticks <= 0 and signals <= 0 else pnl_pct)
                ann_row.append('' if ticks <= 0 and signals <= 0 else f'{pnl_pct:+.2f}%\nS{signals}')
            matrix.append(row)
            annot.append(ann_row)

        arr = np.array(matrix, dtype=float)
        max_abs = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
        max_abs = max(float(max_abs), 0.5)
        cmap = plt.get_cmap('RdYlGn')
        try:
            cmap = cmap.copy()
            cmap.set_bad('#30363D')
        except Exception:
            pass
        image = ax.imshow(arr, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect='auto')
        status_map = status_map or {}
        row_statuses = [str(status_map.get(name, '') or '').lower() for name, _, _ in rows]
        labels = [
            _clean_label(name) + (' [Q]' if st in quarantine_statuses else '')
            for (name, _, _), st in zip(rows, row_statuses)
        ]
        ax.set_xticks(range(len(MARKET_REGIMES)))
        ax.set_xticklabels([r.upper() for r in MARKET_REGIMES], color=WHT, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, color=WHT, fontsize=7)
        ax.set_title(title, color=WHT, fontsize=11, fontweight='bold', pad=8)
        ax.tick_params(colors=GRY)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isfinite(arr[i, j]):
                    continue
                text_color = BG if abs(arr[i, j]) > max_abs * 0.45 else WHT
                ax.text(j, i, annot[i][j], ha='center', va='center',
                        fontsize=6.5, color=text_color)
        try:
            for tick_lbl, st in zip(ax.get_yticklabels(), row_statuses):
                if st in quarantine_statuses:
                    tick_lbl.set_color(GRY)
            for i, st in enumerate(row_statuses):
                if st in quarantine_statuses:
                    ax.axhspan(i - 0.5, i + 0.5, facecolor=GRY, alpha=0.50, zorder=2)
        except Exception:
            pass
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_xticks(np.arange(-.5, len(MARKET_REGIMES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which='minor', color=GRID, linestyle='-', linewidth=0.7)
        ax.tick_params(which='minor', bottom=False, left=False)
        return image

    def _draw_equity(ax, title: str, vp_map: Dict[str, 'VirtualPortfolio']):
        _style(ax, title)
        ranked = sorted((vp_map or {}).items(), key=lambda kv: kv[1].pnl_pct, reverse=True)
        if not ranked:
            ax.text(0.5, 0.5, 'No equity history yet', transform=ax.transAxes,
                    ha='center', va='center', color=GRY, fontsize=11)
            return
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(ranked), 1)))
        plotted = 0
        for idx, (name, vp) in enumerate(ranked[:18]):
            eq = []
            for value in list(getattr(vp, 'equity_history', []) or []):
                try:
                    eq.append(float(value))
                except Exception:
                    continue
            if len(eq) < 2:
                continue
            label = f"{_clean_label(name, 18)} ({vp.pnl_pct:+.1f}%)" if plotted < 6 else None
            ax.plot(range(len(eq)), eq, color=palette[idx % 20],
                    lw=1.8 if plotted < 4 else 0.9,
                    alpha=0.9 if plotted < 4 else 0.35,
                    label=label)
            plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.5, 'No equity history yet', transform=ax.transAxes,
                    ha='center', va='center', color=GRY, fontsize=11)
            return
        first_ic = float(getattr(ranked[0][1], 'initial_capital', 0.0) or 0.0)
        if first_ic > 0:
            ax.axhline(first_ic, color=WHT, lw=0.8, alpha=0.28, linestyle=':')
        ax.set_xlabel('Bar', color=GRY, fontsize=8)
        ax.set_ylabel('Equity ($)', color=GRY, fontsize=8)
        ax.legend(loc='upper left', fontsize=6, ncol=2,
                  facecolor=MID, edgecolor=GRID, labelcolor=WHT)

    heat_player = fig.add_subplot(gs[0, 0])
    curve_player = fig.add_subplot(gs[0, 1])
    heat_agent = fig.add_subplot(gs[1, 0])
    curve_agent = fig.add_subplot(gs[1, 1])
    im_player = _draw_heatmap(heat_player, 'Players by Market Regime', player_vp_map, player_status_map)
    im_agent = _draw_heatmap(heat_agent, 'Agents by Market Regime', agent_vp_map, agent_status_map)
    _draw_equity(curve_player, 'Players Equity Over Time', player_vp_map)
    _draw_equity(curve_agent, 'Agents Equity Over Time', agent_vp_map)
    for ax, image in ((heat_player, im_player), (heat_agent, im_agent)):
        if image is None:
            continue
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(colors=GRY, labelsize=8)
        cbar.set_label('P&L % contribution', color=GRY, fontsize=8)

    fig.suptitle('Regime Dashboard - Players Over Agents',
                 fontsize=15, fontweight='bold', color=WHT, y=0.985)
    path = os.path.join(output_dir, output_filename)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)
    log.debug("Combined regime dashboard -> %s", path)


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
                 agreement: AgreementTracker, regime_tracker: RegimeTracker,
                 shadow_players: Optional[Dict[str, tuple]] = None):
        super().__init__(daemon=True, name="combo_monitor")
        self._client = client
        self._stats = stats
        self._output_dir = output_dir
        self._shadows = shadows
        self._shadow_players = shadow_players or OrderedDict()
        self._agreement = agreement
        self._regime_tracker = regime_tracker
        self._bridge = None
        self._stop_evt = threading.Event()
        self._counter = 0
        self._last_leaderboard = 0
        self._last_dashboard = 0.0
        try:
            from exchange_api_runtime import PositionSyncHealth
            self._position_sync_health = PositionSyncHealth()
        except Exception:
            self._position_sync_health = None

    def _refresh_dashboard_inputs(self):
        bridge = self._bridge
        if bridge is not None and hasattr(self._stats, "capture_price_history"):
            try:
                self._stats.capture_price_history(getattr(bridge, "_price_hist", None), limit=360)
            except Exception:
                pass
        if bridge is not None and str(getattr(bridge, "mode", "") or "").lower() == "paper":
            paper_pf = getattr(bridge, "paper_pf", {}) or {}
            if paper_pf:
                try:
                    agent_name = next(iter(getattr(bridge, "agents", {}) or paper_pf))
                    pf = paper_pf.get(agent_name) or next(iter(paper_pf.values()))
                    prices = (getattr(bridge, "_price_hist", []) or [{}])[-1] or {}
                    balance = float(pf.portfolio_value(prices) if prices else (pf.history[-1] if pf.history else pf.initial_capital))
                    initial = float(getattr(pf, "initial_capital", self._stats.initial_capital) or self._stats.initial_capital)
                    if initial > 0 and abs(float(self._stats.initial_capital) - initial) > 1e-9:
                        self._stats.initial_capital = initial
                    self._stats.current_positions.clear()
                    for sym, pos in (getattr(pf, "futures", {}) or {}).items():
                        entry = float(pos.get("entry", 0.0) or 0.0)
                        qty = float(pos.get("qty", 0.0) or 0.0)
                        side = str(pos.get("side", "") or "").upper()
                        price = float(prices.get(sym, entry) or entry)
                        direction = 1.0 if side == "LONG" else -1.0
                        pnl = qty * (price - entry) * direction
                        self._stats.current_positions[sym] = {
                            "side": side,
                            "qty": qty,
                            "entry": entry,
                            "pnl": pnl,
                            "type": "paper_futures",
                        }
                    self._stats.bonus_info = {
                        "real_usdt": balance,
                        "stable_total": balance,
                        "tradeable_total": balance,
                        "bonus_tokens": {},
                        "has_funds": balance >= MIN_BALANCE_USD,
                    }
                    self._stats.record_tick(
                        ts=datetime.now(tz=timezone.utc),
                        balance=balance,
                        pnl=balance - initial,
                        available=balance,
                        live_bar=self._stats.live_bar_count,
                    )
                except Exception as exc:
                    log.debug("paper dashboard state refresh failed: %s", exc)

        if getattr(self._stats, "sub_agent_pvs", None) is not None:
            shadow_pvs = {}
            shadow_initials = {}
            for name, item in (self._shadows or {}).items():
                try:
                    vp = item[1]
                    hist = list(getattr(vp, "equity_history", []) or [])
                    value = float(hist[-1] if hist else getattr(vp, "initial_capital", 0.0) or 0.0)
                    initial = float(getattr(vp, "initial_capital", 0.0) or 0.0)
                except Exception:
                    continue
                label = str(name)
                if label.startswith("V_"):
                    label = label[2:]
                shadow_pvs[label] = round(value, 4)
                shadow_initials[label] = round(initial, 4)
            self._stats.sub_agent_pvs = shadow_pvs
            self._stats.sub_agent_initials = shadow_initials

        funding = getattr(bridge, "funding", None) if bridge is not None else None
        if funding is None or getattr(self._stats, "funding_history", None) is None:
            return
        try:
            global_funding = funding.get_global()
            if not global_funding:
                return
            avg_rate = float(global_funding.get("avg_funding", 0.0) or 0.0)
            hist = self._stats.funding_history
            if not hist or abs(float(hist[-1]) - avg_rate) > 1e-12:
                hist.append(avg_rate)
        except Exception:
            return

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
                # FIX 2026-04-27: раньше тут был `return` если bridge_warmup<=0
                # и bridge_bar>0 — это блокировало все сохранения во время
                # warmup (status.json/leaderboard не появлялись на диске).
                # Теперь просто откладываем установку _warmup_end_bar до момента
                # когда bridge выставит _warmup_end, но всё равно идём дальше
                # и сохраняем то, что доступно.
                if bridge_warmup > 0:
                    self._warmup_end_bar = bridge_warmup
                else:
                    # tentative — можем переустановить когда warmup завершится
                    self._warmup_end_bar = max(0, self._stats.bar_count or 0)
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
                snapshot_healthy = (
                    float(total_stable or 0.0) > 0.0
                    and float(usdt_val or 0.0) >= 0.0
                )
                if not snapshot_healthy and self._position_sync_health is not None:
                    self._position_sync_health.mark_data_error("balance_error")

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
                self._reconcile_open_pos(fut['positions'], snapshot_healthy=snapshot_healthy)
                # FIX 2026-04-27: tight-stop для legacy-leverage позиций
                try:
                    self._enforce_legacy_leverage_stops(
                        {p['symbol']: p.get('mark_price') or p.get('entry')
                         for p in fut['positions']}
                    )
                except Exception:
                    pass

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
            if self._position_sync_health is not None:
                self._position_sync_health.mark_data_error("balance_error")
                self._stats.data_health = self._position_sync_health.summary()
                self._stats.data_health["failed_orders"] = getattr(self._stats, "orders_fail", 0)
            log.debug("Balance refresh: %s", e)

        # --- Dashboard ---
        self._refresh_dashboard_inputs()
        save_summary_json(self._stats, self._output_dir)
        _refresh_html_dashboard_best_effort(force=render_dashboards)

        # --- Shadow Dashboard ---
        vp_map = {name: tup[1] for name, tup in self._shadows.items()}
        player_vp_map = {name: tup[1] for name, tup in self._shadow_players.items()}
        # FIX 2026-05-03: пересчитать статусы (вкл. динамический карантин)
        # ДО отрисовки, чтобы дашборды уже были раскрашены корректно.
        try:
            self._recompute_status_maps(vp_map, player_vp_map)
        except Exception as exc:
            log.debug("  [status_map] recompute failed: %s", exc)
        agent_status_map  = getattr(self, "_status_map_agents", None) or {}
        player_status_map = getattr(self, "_status_map_players", None) or {}

        # FIX 2026-05-05 (REAL attribution v4): дашборд «Вклад суб-агентов в
        # P&L» теперь показывает РЕАЛЬНЫЙ привклад каждого делегата (player/
        # agent) на основе фактических сделок Пантеона.
        #
        # История изменений этой панели:
        #   v1 — фейк: total_pnl * (sig_count / total_sigs) — все одного знака
        #   v2 — shadow PV: vp.pnl_pct каждого виртуального агента (как если бы
        #         агент торговал в одиночку) — показывает «потенциал», но
        #         противоречит РЕАЛЬНОСТИ: Пантеон выбирает плохо, и сумма
        #         положительных shadow > отрицательных, но live в минусе.
        #   v3 (этот) — REAL attribution: проходим по stats.signals в порядке
        #         времени, отслеживаем открытые позиции, при close-сигнале
        #         считаем PnL = (close_price - open_price) / open_price * side
        #         и атрибутируем его selected_player исходного OPEN-сигнала.
        #         Это РЕАЛЬНЫЙ вклад делегата в портфель Пантеона. Sum(real)
        #         ≈ real PnL Пантеона (с поправкой на fees/slippage/funding).
        try:
            attribution = self._compute_real_attribution()
            ic = float(getattr(self._stats, 'initial_capital', 0.0) or 0.0)
            if attribution:
                # PV = ic + attribution_pnl_usd ; baseline = ic
                real_pvs: Dict[str, float] = {}
                real_inits: Dict[str, float] = {}
                for player, pnl_usd in attribution.items():
                    real_pvs[player] = round(ic + float(pnl_usd or 0.0), 4)
                    real_inits[player] = round(ic, 4)
                self._stats.sub_agent_pvs = real_pvs
                self._stats.sub_agent_initials = real_inits
                # Сохраняем как отдельный атрибут для других потребителей
                self._stats.real_attribution_pnl = dict(attribution)
            else:
                # Без сделок — оставляем пустым, чтобы не вводить в заблуждение
                self._stats.sub_agent_pvs = {}
                self._stats.sub_agent_initials = {}
                self._stats.real_attribution_pnl = {}
        except Exception as exc:
            log.debug("  [real_attribution] failed: %s", exc)
        try:
            save_combined_shadow_dashboard(
                vp_map, player_vp_map, self._stats, self._regime_tracker,
                self._output_dir,
                agent_status_map=agent_status_map,
                player_status_map=player_status_map,
            )
            save_combined_regime_dashboard(
                vp_map, player_vp_map, self._output_dir,
                agent_status_map=agent_status_map,
                player_status_map=player_status_map,
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
            player_vp_map = {name: tup[1] for name, tup in self._shadow_players.items()}
            if not ROTATION_CHANGE_LOGS_ONLY:
                real_pnl = self._stats.pnl_pct
                regime = self._regime_tracker.current
                bar = self._stats.bar_count
                log_leaderboard(vp_map, real_pnl, regime, bar, self._regime_tracker)
                if player_vp_map:
                    log_leaderboard(player_vp_map, real_pnl, regime, bar, self._regime_tracker)
                log.info(self._agreement.log_matrix())

            # Сохраняем leaderboard в JSON
            self._save_leaderboard_json(vp_map, filename="leaderboard_agents.json")
            self._save_leaderboard_json(player_vp_map, filename="leaderboard_players.json")

    def _reconcile_open_pos(self, exchange_positions: list, snapshot_healthy: bool = True):
        """Синхронизирует _open_pos агента с реальными позициями.

        FIX 2026-04-27: дополнительно помечаем позиции, открытые с устаревшим
        leverage (например lev=3 после смены настройки на lev=2). Такие
        позиции — основной источник убытка на REAL: они унаследованы от
        прошлой сессии и продолжают терять деньги. Им назначаем агрессивный
        tight-SL, который форсирует close при первом же откате.
        """
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

            try:
                from exchange_api_runtime import (
                    LEVERAGE as _CURRENT_LEV,
                    inherited_position_policy as _inherited_position_policy,
                    mark_recent_runtime_data_errors as _mark_recent_runtime_data_errors,
                    safe_reconcile_open_positions as _safe_reconcile_open_positions,
                )
            except Exception:
                _CURRENT_LEV = None
                _inherited_position_policy = lambda: {"external": True}
                _mark_recent_runtime_data_errors = None
                _safe_reconcile_open_positions = None

            real_syms = {p['symbol'] for p in exchange_positions}
            if self._position_sync_health is not None and _safe_reconcile_open_positions is not None:
                if _mark_recent_runtime_data_errors is not None:
                    _mark_recent_runtime_data_errors(
                        self._position_sync_health,
                        self._bridge,
                        self._client,
                    )
                _safe_reconcile_open_positions(
                    _player._open_pos,
                    exchange_positions,
                    current_bar=getattr(self._bridge, '_bar', self._stats.bar_count),
                    health=self._position_sync_health,
                    snapshot_healthy=snapshot_healthy,
                    logger=log,
                )
                self._stats.data_health = self._position_sync_health.summary()
                self._stats.data_health["failed_orders"] = getattr(self._stats, "orders_fail", 0)
            stale = []
            for s in stale:
                log.info("  [reconcile] %s удалён из _open_pos (закрыта)", s)
                del _player._open_pos[s]
            for p in exchange_positions:
                sym = p['symbol']
                pos_lev = int(p.get('leverage', _CURRENT_LEV or 1) or 1)
                is_legacy_lev = (
                    _CURRENT_LEV is not None
                    and pos_lev > 0
                    and int(_CURRENT_LEV) != pos_lev
                )
                if sym not in _player._open_pos:
                    rec = {
                        'entry': p['entry'], 'side': p['side'],
                        'bar': getattr(self._bridge, '_bar', 0),
                        'peak': p['entry'],
                        'leverage': pos_lev,
                        **_inherited_position_policy(),
                    }
                    if is_legacy_lev:
                        # tight-SL = 1.5%: чем быстрее закроем legacy с большим
                        # плечом, тем меньше будет drawdown по REAL pnl.
                        rec['tight_sl_pct'] = 0.015
                        rec['legacy_leverage'] = True
                        log.warning(
                            "  [reconcile] %s legacy lev=%dx (current=%dx), "
                            "set tight-SL 1.5%% (entry=%.4f)",
                            sym, pos_lev, int(_CURRENT_LEV or 0), p['entry'],
                        )
                    _player._open_pos[sym] = rec
                    log.info("  [reconcile] %s добавлен (внешняя: %s entry=%.4f lev=%dx%s)",
                             sym, p['side'], p['entry'], pos_lev,
                             " [LEGACY]" if is_legacy_lev else "")
                else:
                    # Обновляем leverage и legacy-флаг даже у уже трекаемых
                    rec = _player._open_pos[sym]
                    rec['leverage'] = pos_lev
                    if rec.get('external') and not rec.get('inherited_from_exchange'):
                        for key, val in _inherited_position_policy().items():
                            rec.setdefault(key, val)
                    if is_legacy_lev and not rec.get('legacy_leverage'):
                        rec['legacy_leverage'] = True
                        rec['tight_sl_pct'] = 0.015
        except Exception as e:
            log.debug("  [reconcile] %s", e)

    def _enforce_legacy_leverage_stops(self, prices: dict):
        """FIX 2026-04-27: проверяет legacy-leverage позиции и форсирует close,
        если цена ушла дальше tight_sl_pct от entry. Запускается после
        reconcile, до основного act() цикла, чтобы успеть высвободить маржу.
        """
        try:
            if self._bridge is None:
                return
            _player = None
            for _ag in getattr(self._bridge, 'agents', {}).values():
                _candidate = getattr(_ag, '_inner', _ag)
                if hasattr(_candidate, '_inner'):
                    _candidate = _candidate._inner
                if hasattr(_candidate, '_open_pos'):
                    _player = _candidate
                    break
            if _player is None:
                return
            forced = []
            for sym, rec in list(_player._open_pos.items()):
                if not rec.get('legacy_leverage'):
                    continue
                if sym not in prices:
                    continue
                tight = float(rec.get('tight_sl_pct', 0.015) or 0.015)
                cur = float(prices[sym] or 0.0)
                ep = float(rec.get('entry', 0.0) or 0.0)
                if ep <= 0 or cur <= 0:
                    continue
                side = str(rec.get('side', 'long') or 'long').lower()
                move = (cur / ep - 1.0) if side == 'long' else (1.0 - cur / ep)
                if move <= -tight:
                    forced.append((sym, side, ep, cur, move))
            if not forced:
                return
            # Высвобождаем маржу через bridge close API
            for sym, side, ep, cur, move in forced:
                log.warning(
                    "  [legacy-lev] FORCE-CLOSE %s %s entry=%.4f cur=%.4f move=%+.2f%% (tight-SL)",
                    sym, side.upper(), ep, cur, move * 100,
                )
                # Используем bridge.close_position если есть
                try:
                    bridge = self._bridge
                    closer = (
                        getattr(bridge, "close_position", None)
                        or getattr(bridge, "_close_position", None)
                    )
                    if callable(closer):
                        closer(sym)
                    else:
                        # Fallback — пометить позицию как закрытую в _open_pos,
                        # чтобы action=8 прошёл при следующем тике
                        _player._open_pos.pop(sym, None)
                except Exception as exc:
                    log.warning("  [legacy-lev] close failed for %s: %s", sym, exc)
        except Exception as exc:
            log.debug("  [legacy-lev] enforce failed: %s", exc)

    def _compute_real_attribution(self) -> Dict[str, float]:
        """FIX 2026-05-05 (REAL attribution v4): атрибуция реального PnL по
        делегатам (selected_player) на основе stats.signals.

        Логика:
          1) Идём по signals в порядке времени.
          2) При OPEN-сигнале (action ∈ {1,2,4,5,6,7}) запоминаем для sym:
             (open_price, side, opener_player).
          3) При CLOSE-сигнале (action ∈ {3,8}) для того же sym:
               pnl_pct = (close_px - open_px)/open_px  if side=long
                       = (open_px - close_px)/open_px  if side=short
               pnl_usd = pnl_pct * size_usd, где size_usd = ic * trade_fraction
             Атрибутируем opener_player (тот кто решил войти).
          4) Открытые-но-не-закрытые позиции игнорируем (нет realised PnL).

        Возвращает словарь: player_name → total_pnl_usd.

        Это даёт реальную картину вклада, в отличие от shadow PV (виртуального
        одиночного PnL) — последний игнорирует, что в реальном Пантеоне
        делегат может быть выбран в плохой момент или его сигналы могут
        перебиваться другими.
        """
        try:
            signals = list(getattr(self._stats, "signals", []) or [])
            ic = float(getattr(self._stats, "initial_capital", 0.0) or 0.0)
            if not signals or ic <= 0:
                return {}
            # Размер позиции по умолчанию (~10% от капитала, как
            # обычно конфигурируется в боевом Пантеоне). Это даёт
            # масштаб PnL близкий к реальному.
            size_per_trade_usd = ic * 0.10
            opens: Dict[str, Dict] = {}  # sym → {price, side, player}
            attrib: Dict[str, float] = {}
            OPEN_LONG = {1, 2, 4, 5}
            OPEN_SHORT = {6, 7}
            CLOSE = {3, 8}
            for sig in signals:
                try:
                    sym = str(sig.get("sym") or "")
                    action = int(sig.get("action") or 0)
                    px = float(sig.get("price") or 0.0)
                    if not sym or px <= 0 or action == 0:
                        continue
                    player = str(
                        sig.get("selected_player")
                        or sig.get("selected_agents")
                        or sig.get("agent")
                        or "Unknown"
                    ) or "Unknown"
                    # Чистим V_ префикс
                    if player.startswith("V_"):
                        player = player[2:]
                    if action in OPEN_LONG:
                        opens[sym] = {"price": px, "side": "long", "player": player}
                    elif action in OPEN_SHORT:
                        opens[sym] = {"price": px, "side": "short", "player": player}
                    elif action in CLOSE:
                        rec = opens.pop(sym, None)
                        if not rec:
                            continue
                        open_px = float(rec["price"] or 0.0)
                        if open_px <= 0:
                            continue
                        side = rec["side"]
                        if side == "long":
                            pnl_pct = (px - open_px) / open_px
                        else:
                            pnl_pct = (open_px - px) / open_px
                        # Атрибутируем тому, кто открыл — он принял решение
                        opener = rec["player"]
                        pnl_usd = pnl_pct * size_per_trade_usd
                        attrib[opener] = attrib.get(opener, 0.0) + pnl_usd
                except Exception:
                    continue
            return attrib
        except Exception as exc:
            log.debug("  [real_attribution] compute failed: %s", exc)
            return {}

    def _get_panteon_inner(self):
        """Поднять внутренний инстанс Panteon (с _agent_pool/_shadow_player_pool)."""
        try:
            if self._bridge is None:
                return None
            for ag in getattr(self._bridge, "agents", {}).values():
                cand = getattr(ag, "_inner", ag)
                if hasattr(cand, "_inner"):
                    cand = cand._inner
                if hasattr(cand, "_shadow_player_pool") or hasattr(cand, "_agent_pool"):
                    return cand
        except Exception:
            return None
        return None

    def _per_regime_summary(self, vp) -> Dict[str, dict]:
        """Извлечь per_regime метрики из VirtualPortfolio безопасно."""
        try:
            return vp.export_regime_stats() or {}
        except Exception:
            return {}

    def _has_positive_regime_experience(self, per_regime: Dict[str, dict],
                                        recovery_pnl: float = 0.10,
                                        recovery_closed: int = 3) -> bool:
        """Возвращает True если хотя бы в одном режиме рынка
        накоплен положительный опыт (>= recovery_pnl% при >= recovery_closed
        закрытых сделках)."""
        if not isinstance(per_regime, dict):
            return False
        for regime, stats in per_regime.items():
            if not isinstance(stats, dict):
                continue
            try:
                pnl_pct = float(stats.get('pnl_pct', 0.0) or 0.0)
                closed  = int(stats.get('closed_trades', 0) or 0)
            except (TypeError, ValueError):
                continue
            if pnl_pct >= recovery_pnl and closed >= recovery_closed:
                return True
        return False

    def _is_hopeless_in_all_regimes(self, per_regime: Dict[str, dict],
                                    hard_pnl: float = -0.30,
                                    min_closed: int = 5) -> bool:
        """Безнадёжен — если ВО ВСЕХ режимах рынка pnl_pct <= 0 и
        накопленный закрытый объём >= min_closed, и худший pnl <= hard_pnl."""
        if not isinstance(per_regime, dict):
            return False
        any_data = False
        all_negative = True
        total_closed = 0
        worst = 0.0
        for regime, stats in per_regime.items():
            if not isinstance(stats, dict):
                continue
            try:
                pnl_pct = float(stats.get('pnl_pct', 0.0) or 0.0)
                closed  = int(stats.get('closed_trades', 0) or 0)
            except (TypeError, ValueError):
                continue
            any_data = True
            total_closed += closed
            if pnl_pct > 0.0:
                all_negative = False
            if pnl_pct < worst:
                worst = pnl_pct
        return (
            any_data
            and all_negative
            and total_closed >= min_closed
            and worst <= hard_pnl
        )

    def _resolve_status(self, name: str, collection_key: str,
                        vp=None) -> dict:
        """FIX #15 (2026-04-26) + FIX 2026-05-03 (dynamic quarantine v3):
        вернуть статус (live/shadow_only/quarantine/experimental/purgatory)
        и причину для записи в leaderboard. Решение принимается с учётом:
          1) PLAYER_STATUS / AGENT_STATUS на инстансе агента/игрока (legacy);
          2) per-regime памяти shadow-портфеля (VirtualPortfolio.export_regime_stats);
          3) seed-карантина PLAYER_QUARANTINE_SEED / LIVE_AGENT_BLOCKLIST.

        Если у участника во ВСЕХ режимах рынка pnl_pct отрицателен и набрана
        достаточная выборка — он автоматически переходит в quarantine
        (отображение серым на дашбордах). Если хотя бы в одном режиме есть
        положительный накопленный опыт — статически объявленный shadow_only/
        quarantine снимается, и участник возвращается к live-выбору.
        """
        try:
            inner = self._get_panteon_inner()
            label_clean = name.replace("V_", "")
            per_regime = self._per_regime_summary(vp) if vp is not None else {}
            # Параметры порога — из Panteon (если доступно), иначе дефолты
            try:
                recovery_pnl    = float(getattr(inner, 'QUARANTINE_RECOVERY_PNL_PCT', 0.10))
                recovery_closed = int(getattr(inner, 'QUARANTINE_RECOVERY_CLOSED', 3))
                hard_pnl        = float(getattr(inner, 'QUARANTINE_HARD_NEG_PNL_PCT', -0.30))
                hard_min_closed = int(getattr(inner, 'QUARANTINE_HARD_MIN_CLOSED', 5))
            except Exception:
                recovery_pnl, recovery_closed = 0.10, 3
                hard_pnl, hard_min_closed = -0.30, 5
            has_positive = self._has_positive_regime_experience(
                per_regime, recovery_pnl, recovery_closed
            )
            hopeless = self._is_hopeless_in_all_regimes(
                per_regime, hard_pnl, hard_min_closed
            )

            if collection_key == "players":
                pool = getattr(inner, "_shadow_player_pool", {}) if inner else {}
                pool = pool or {}
                instance = pool.get(name)
                base_status = (
                    str(getattr(instance, "PLAYER_STATUS", "live") or "live").lower()
                    if instance is not None else "live"
                )
                base_reason = (
                    str(getattr(instance, "PLAYER_STATUS_REASON", "") or "")
                    if instance is not None else ""
                )
                seed_q = bool(getattr(instance, "PLAYER_QUARANTINE_SEED", False))
                seed_reason = str(getattr(instance, "PLAYER_QUARANTINE_SEED_REASON", "") or "")

                # Purgatory имеет приоритет над всем остальным
                purg = getattr(inner, "_player_purgatory_until", {}) if inner else {}
                purg = purg or {}
                if name in purg:
                    until_bar = int(purg.get(name, 0))
                    cur_bar = int(getattr(inner, "_t", 0) or 0) if inner else 0
                    return {"status": "purgatory",
                            "status_reason": f"in purgatory until bar {until_bar} (now {cur_bar})"}

                # Автоматическое снятие seed-карантина при положительном опыте
                if has_positive:
                    return {"status": "live",
                            "status_reason": "released from quarantine — positive regime experience"}

                # Авто-карантин по полностью отрицательной истории
                if hopeless:
                    return {"status": "quarantine",
                            "status_reason": "no positive PnL in any regime (auto)"}

                # Seed-карантин (фиксированный список плохих исторически)
                if seed_q:
                    return {"status": "quarantine",
                            "status_reason": seed_reason or "seed quarantine"}

                # experimental / shadow_only / live — как объявлено в коде агента
                return {"status": base_status, "status_reason": base_reason}
            else:
                # АГЕНТЫ
                pool = getattr(inner, "_agent_pool", {}) if inner else {}
                pool = pool or {}
                instance = pool.get(label_clean)
                base_status = (
                    str(getattr(instance, "AGENT_STATUS", "live") or "live").lower()
                    if instance is not None else "live"
                )
                base_reason = (
                    str(getattr(instance, "AGENT_STATUS_REASON", "") or "")
                    if instance is not None else ""
                )
                seed_q = label_clean in getattr(inner, "_SEED_AGENT_QUARANTINE", frozenset())

                if has_positive:
                    return {"status": "live",
                            "status_reason": "released from quarantine — positive regime experience"}
                if hopeless:
                    return {"status": "quarantine",
                            "status_reason": "no positive PnL in any regime (auto)"}
                # Live-blocklist (динамический): после recompute_dynamic_quarantine
                if label_clean in getattr(inner, "LIVE_AGENT_BLOCKLIST", set()):
                    return {"status": "quarantine",
                            "status_reason": "seed quarantine — awaiting positive regime experience"}
                if seed_q:
                    return {"status": "quarantine",
                            "status_reason": "seed quarantine — awaiting positive regime experience"}
                return {"status": base_status, "status_reason": base_reason}
        except Exception as exc:
            log.debug("  [_resolve_status] %s: %s", name, exc)
            return {"status": "live", "status_reason": ""}

    def _recompute_status_maps(self,
                               vp_map: Dict[str, 'VirtualPortfolio'],
                               player_vp_map: Dict[str, 'VirtualPortfolio']) -> None:
        """FIX 2026-05-03 (dynamic quarantine v3): прогон один раз
        пересчитывает status каждого агента/игрока (с учётом per-regime
        памяти) и сохраняет в self._status_map_agents / _status_map_players.
        Также синхронно обновляет Panteon.LIVE_AGENT_BLOCKLIST через
        recompute_dynamic_quarantine — так боевой выбор лидера на следующем
        тике уже учтёт новый список.
        """
        try:
            # 1) Собрать per_regime для агентов и пересчитать карантин Panteon
            per_regime_by_label: Dict[str, Dict[str, dict]] = {}
            for name, vp in (vp_map or {}).items():
                per_regime_by_label[name.replace("V_", "")] = self._per_regime_summary(vp)
            inner = self._get_panteon_inner()
            if inner is not None and hasattr(inner, "recompute_dynamic_quarantine"):
                try:
                    inner.recompute_dynamic_quarantine(per_regime_by_label)
                except Exception as exc:
                    log.debug("  [quarantine] recompute failed: %s", exc)

            # 2) Заполнить status_map по агентам и игрокам
            self._status_map_agents = {
                name: self._resolve_status(name, "agents", vp=vp).get("status", "live")
                for name, vp in (vp_map or {}).items()
            }
            self._status_map_players = {
                name: self._resolve_status(name, "players", vp=vp).get("status", "live")
                for name, vp in (player_vp_map or {}).items()
            }
            # FIX 2026-05-04 (carantine consistency): пробрасываем объединённый
            # статус-словарь в TradingStats, чтобы api_trading.png тоже видел
            # карантин и красил столбики суб-агентов серым (а не зелёным/
            # красным). Сейчас api_trading.png показывает «вклад суб-агентов»
            # в красном — даже для тех, кто формально в карантине.
            try:
                merged = {}
                merged.update(self._status_map_agents or {})
                merged.update(self._status_map_players or {})
                # Дополнительные ключи без префикса V_, чтобы plot_api_trading_dashboard
                # мог искать по «чистому» имени:
                for k, v in list(merged.items()):
                    if k.startswith("V_"):
                        merged.setdefault(k[2:], v)
                if hasattr(self._stats, "status_map") or self._stats is not None:
                    setattr(self._stats, "status_map", merged)
            except Exception as exc:
                log.debug("  [status_map] stats.status_map propagation failed: %s", exc)
        except Exception as exc:
            log.debug("  [status_map] _recompute failed: %s", exc)

    def _save_leaderboard_json(self, vp_map: Dict[str, VirtualPortfolio],
                               filename: str = "leaderboard.json"):
        try:
            collection_key = "players" if "players" in filename else "agents"
            entries = {}
            status_map: Dict[str, str] = {}
            # FIX 2026-05-03: per_regime по каждому участнику нужен и для
            # leaderboard, и для пересчёта динамического карантина в Panteon.
            per_regime_by_label: Dict[str, Dict[str, dict]] = {}
            for name, vp in vp_map.items():
                status_info = self._resolve_status(name, collection_key, vp=vp)
                per_regime = self._per_regime_summary(vp)
                # ключ для синхронизации с Panteon.LIVE_AGENT_BLOCKLIST — без 'V_'
                label_clean = name.replace("V_", "")
                per_regime_by_label[label_clean] = per_regime
                entries[name] = {
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
                    'per_regime': per_regime,
                    # FIX #15 (2026-04-26): метки статуса
                    'status': status_info["status"],
                    'status_reason': status_info["status_reason"],
                }
                status_map[name] = status_info["status"]

            # ── FIX 2026-05-03 (dynamic quarantine v3) ──
            # Для агентов — пересчитываем live-blocklist на основе per-regime
            # данных и пробрасываем результат внутрь Panteon, чтобы боевой
            # выбор лидера в _update_regime_leaders / _normalize_weights
            # учитывал РЕАЛЬНЫЙ накопленный опыт, а не статический список.
            if collection_key == "agents":
                inner = self._get_panteon_inner()
                if inner is not None and hasattr(inner, "recompute_dynamic_quarantine"):
                    try:
                        decisions = inner.recompute_dynamic_quarantine(per_regime_by_label)
                        # Логируем изменения если они нетривиальные
                        new_q = sorted(getattr(inner, "LIVE_AGENT_BLOCKLIST", set()))
                        if new_q != sorted(getattr(self, "_last_q_set", [])):
                            log.info("  [quarantine] live blocklist обновлён: %s", new_q)
                            self._last_q_set = list(new_q)
                    except Exception as exc:
                        log.debug("  [quarantine] recompute failed: %s", exc)

            # Кэшируем status_map для дашбордов
            cache_key = "_status_map_players" if collection_key == "players" else "_status_map_agents"
            setattr(self, cache_key, dict(status_map))

            data = {
                'metadata': {
                    'schema': 3,
                    'collection': collection_key,
                    'exchange': EXCHANGE_NAME,
                    'real_pnl_pct': round(self._stats.pnl_pct, 4),
                    'regime': self._regime_tracker.current,
                    'regime_history': self._regime_tracker.history[-20:],
                    'bar': self._stats.bar_count,
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'status_legend': {
                        'live': 'торгует на бирже, веса считаются обычным образом',
                        'shadow_only': 'только теневой режим — не выбирается лидером',
                        'quarantine': 'динамический карантин — нет положительного опыта ни в одном режиме',
                        'experimental': 'экспериментальный, требует подключения данных',
                        'purgatory': 'временный карантин по плохому live-результату',
                    },
                },
                collection_key: entries,
            }

            path = os.path.join(self._output_dir, filename)
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
    actions_by_name = {}
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
        actions_by_name[name] = dict(acts or {})

        n_active = sum(1 for a in acts.values() if a and a != 0)
        regime_symbols = set(vp.positions) | set(acts)
        symbol_regimes = {
            sym: _detect_dashboard_symbol_regime(agent, sym, regime)
            for sym in regime_symbols
            if sym in prices
        }

        for sym, action in acts.items():
            if action == 0 or sym not in prices:
                continue

            price = prices[sym]
            symbol_regime = symbol_regimes.get(sym, regime)
            risk_map = getattr(agent, '_last_risk_multipliers', {}) or {}
            risk_multiplier = float(risk_map.get(sym, 1.0) or 1.0)
            vp.execute(
                sym,
                action,
                price,
                bar,
                risk_multiplier=risk_multiplier,
                regime=symbol_regime,
            )
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
                regime=symbol_regime, contributors='',
                pv=pv, n_pos=len(vp.positions),
                agreement=0,  # рассчитываем после flush
            )

        # Snapshot портфеля
        vp.snapshot(prices, regime=regime, symbol_regimes=symbol_regimes)

        if n_active > 0:
            log.debug("  [shadow %-22s] tick_signals=%d  signals=%d  entries=%d  closes=%d  "
                      "equity=$%.2f  P&L=%+.2f%%  open=%d",
                      name, n_active, vp.signal_count, vp.entry_count,
                      vp.close_count, vp.equity_history[-1], vp.pnl_pct,
                      len(vp.positions))
    return actions_by_name


# ══════════════════════════════════════════════════════════════════════════════
# FIX (2026-04-28): ORPHAN POSITION GUARDIAN
# ══════════════════════════════════════════════════════════════════════════════
# Проблема: на BITGET 2026-04-27_20-36-19 4 из 8 открытых позиций были
# "сиротами" — символы, по которым bridge._fetch_market() не возвращает цены
# (символы выпали из топ-78 по объёму). Эти позиции унаследованы от прошлой
# сессии через inject_live_state_into_player. Поскольку prices.get(sym, 0) = 0,
# PositionSafety пропускает их (`if cur <= 0: continue`), а агенты не подают
# сигналов (нет данных). Одна такая позиция (H SHORT) разошлась на -27.8 % за
# 27 часов и съела -4.21 USD ≈ 80 % всей просадки.
#
# Решение: периодически проверяем _open_pos real-плеера на сироты, через
# futures_client.ticker_price() / _last_price_quick() запрашиваем цену
# напрямую, и применяем SL/TP/STALE-логику. Закрытие через
# futures_client.close_all(sym) — тот же путь, что и обычные exit-ы.

def _orphan_get_price(futures_client, sym: str) -> float:
    """One-shot ticker для символа, отсутствующего в feed. 0 при ошибке."""
    if futures_client is None:
        return 0.0
    # MEXC-style: ticker_price("BTC_USDT") -> float
    fn = getattr(futures_client, "ticker_price", None)
    if callable(fn):
        try:
            p = fn(f"{sym}_USDT")
            if p and float(p) > 0:
                return float(p)
        except Exception:
            pass
    # Bitget-style: _last_price_quick(market_symbol) -> float
    if hasattr(futures_client, "_last_price_quick") and hasattr(futures_client, "_market_symbol"):
        try:
            mkt = futures_client._market_symbol(sym)
            p = futures_client._last_price_quick(mkt)
            if p and float(p) > 0:
                return float(p)
        except Exception:
            pass
    return 0.0


def _orphan_position_guardian(real_player, prices: dict, futures_client,
                              sl_pct: float = 0.06, tp_pct: float = 0.06,
                              stale_bars_threshold: int = 720,
                              current_bar: int = 0, logger=None):
    """
    Проверяет _open_pos real_player на сиротские позиции (нет цены в prices).
    Для каждой такой позиции запрашивает цену напрямую и закрывает по
    SL/TP/STALE.
    Returns: list of closed symbols.
    """
    if real_player is None or futures_client is None:
        return []
    open_pos = getattr(real_player, "_open_pos", None)
    if not isinstance(open_pos, dict) or not open_pos:
        return []
    # FIX H4 2026-04-28: SL у Panteon = 0.04, но для сирот используем 0.06 —
    # доп. буфер на slippage при out-of-feed-закрытии и для амортизации
    # лагов ticker_price.
    closed = []
    for sym, info in list(open_pos.items()):
        if not isinstance(info, dict):
            continue
        if sym in prices and float(prices.get(sym, 0) or 0) > 0:
            continue  # символ в feed — обработает обычная PositionSafety
        entry = float(info.get("entry", 0) or 0)
        side = str(info.get("side", "long") or "long")
        bar_opened = int(info.get("bar", current_bar) or current_bar)
        def _float_override(key: str, default: float) -> float:
            val = info.get(key, None)
            if val is None:
                return float(default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        def _int_override(key: str, default: int) -> int:
            val = info.get(key, None)
            if val is None:
                return int(default)
            try:
                return int(val)
            except (TypeError, ValueError):
                return int(default)

        pos_sl_pct = _float_override("tight_sl_pct", sl_pct)
        pos_tp_pct = _float_override("take_profit_pct", tp_pct)
        pos_stale_bars = _int_override("stale_bars", stale_bars_threshold)
        pos_stale_move_threshold = _float_override("stale_move_threshold", 0.02)
        if entry <= 0:
            continue
        cur = _orphan_get_price(futures_client, sym)
        if cur <= 0:
            if logger is not None:
                logger.debug("  [OrphanGuardian] %s: ticker fetch failed, skip", sym)
            continue
        if side == "long":
            move = cur / entry - 1.0
        else:
            move = 1.0 - cur / entry
        held = current_bar - bar_opened
        reason = None
        if move < -pos_sl_pct:
            reason = f"SL_orphan({move*100:+.1f}%)"
        elif move > pos_tp_pct:
            reason = f"TP_orphan({move*100:+.1f}%)"
        elif held > pos_stale_bars and abs(move) < pos_stale_move_threshold:
            reason = f"STALE_orphan({held}bars,{move*100:+.1f}%)"
        if reason and hasattr(futures_client, "close_all"):
            if logger is not None:
                logger.warning(
                    "  [OrphanGuardian] CLOSE %s %s entry=%.6f cur=%.6f %s",
                    sym, side.upper(), entry, cur, reason,
                )
            try:
                res = futures_client.close_all(sym)
                if isinstance(res, dict) and res.get("success"):
                    open_pos.pop(sym, None)
                    closed.append(sym)
            except Exception as exc:
                if logger is not None:
                    logger.warning("  [OrphanGuardian] close_all %s failed: %s", sym, exc)
    return closed


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКА BRIDGE._run_cycle ДЛЯ SHADOW-АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

def _patch_bridge_for_shadows(bridge, shadows, csv_logger, agreement, regime_tracker,
                              real_player=None, shadow_players=None):
    """
    Monkey-patch bridge._fetch_market и bridge._run_cycle:
      1. _fetch_market — сохраняет volumes в bridge._last_volumes
      2. _run_cycle — после реального цикла прогоняет shadow-агентов
      3. Передаёт shadow performance в Panteon для адаптивной ротации
    """
    # Регистрируем real Panteon как фолбэк-классификатор per-symbol режима.
    # Без этого простые агенты (без _build_symbol_profile) относили все сигналы
    # к глобальному "neutral", а игроки-пантеоны видели BULLISH/BEARISH — из-за
    # этого дашборды расходились и regime-memory обучалась на несогласованных
    # метках.
    if real_player is not None:
        try:
            set_dashboard_regime_classifier(real_player)
        except Exception:
            pass

    # --- Патч _fetch_market чтобы сохранять volumes ---
    original_fetch = bridge._fetch_market

    def _publish_agent_perf(prices: dict):
        # FIX H3 (2026-04-27): идемпотентность на бар. До этого фикса
        # _publish_agent_perf вызывался дважды на каждом баре (в patched_fetch
        # и в patched_run_cycle), что приводило к двойному обновлению EMA
        # buckets в ShadowPlayerMetaSelector.update_memory и сдвигало
        # _shadow_player_window_anchor → следующий _build_player_observation
        # видел нулевую дельту pnl и портил обучение скоринга.
        cur_bar = int(getattr(bridge, "_bar", 0) or 0)
        last = int(getattr(bridge, "_last_agent_perf_bar", -1) or -1)
        if cur_bar == last:
            return getattr(bridge, "_last_agent_perf_payload", {}) or {}
        perf = _build_shadow_perf_from_shadows(shadows, prices=prices)
        if real_player is not None and hasattr(real_player, 'set_shadow_perf'):
            real_player.set_shadow_perf(perf)
        for shadow_player, _vp in (shadow_players or {}).values():
            try:
                if hasattr(shadow_player, 'set_shadow_perf'):
                    shadow_player.set_shadow_perf(perf)
            except Exception:
                pass
        bridge._last_agent_perf_bar = cur_bar
        bridge._last_agent_perf_payload = perf
        return perf

    def _publish_player_perf(prices: dict):
        # FIX H3 (2026-04-27): идемпотентность на бар (см. _publish_agent_perf).
        cur_bar = int(getattr(bridge, "_bar", 0) or 0)
        last = int(getattr(bridge, "_last_player_perf_bar", -1) or -1)
        if cur_bar == last:
            return getattr(bridge, "_last_player_perf_payload", {}) or {}
        player_perf = _build_shadow_perf_from_shadows(shadow_players, prices=prices) if shadow_players else {}
        if real_player is not None and hasattr(real_player, 'set_shadow_player_perf') and shadow_players:
            real_player.set_shadow_player_perf(player_perf)
            # FIX 2026-05-03 (variant 2): питаем per-symbol stats для
            # Panteon._symbol_live_allowed. Если у real_player нет seam-
            # метода (старый снапшот, рефакторинг) — тихо пропускаем,
            # чтобы не сломать публикацию остальной перфы.
            if hasattr(real_player, 'set_shadow_player_symbol_memory'):
                try:
                    real_player.set_shadow_player_symbol_memory(player_perf)
                except Exception as exc:
                    log.debug("set_shadow_player_symbol_memory failed: %s", exc)
        for shadow_player, _vp in (shadow_players or {}).values():
            try:
                if hasattr(shadow_player, 'set_shadow_player_perf') and player_perf:
                    shadow_player.set_shadow_player_perf(player_perf)
            except Exception:
                pass
        bridge._last_player_perf_bar = cur_bar
        bridge._last_player_perf_payload = player_perf
        return player_perf

    def patched_fetch():
        prices, volumes = original_fetch()
        bridge._last_volumes = volumes
        if getattr(bridge, '_shadow_prefetch_active', False) and not getattr(bridge, '_shadow_prefetch_done', False) and prices:
            bar = bridge._bar
            live_bar = max(0, bar - bridge._warmup_end)
            month = datetime.utcnow().month
            regime = regime_tracker.current
            run_shadow_tick(
                shadows, prices, volumes,
                bar, live_bar, month,
                csv_logger, agreement, regime,
            )
            _publish_agent_perf(prices)
            if shadow_players:
                player_actions = run_shadow_tick(
                    shadow_players, prices, volumes,
                    bar, live_bar, month,
                    csv_logger, agreement, regime,
                )
                if real_player is not None and hasattr(real_player, 'set_external_shadow_player_actions'):
                    try:
                        real_player.set_external_shadow_player_actions(player_actions, bar)
                    except Exception:
                        pass
                _publish_player_perf(prices)
            bridge._shadow_prefetch_done = True
        return prices, volumes

    bridge._fetch_market = patched_fetch
    bridge._last_volumes = {}
    bridge._shadow_prefetch_active = False
    bridge._shadow_prefetch_done = False
    # FIX H3: anti-double-publish на бар.
    bridge._last_agent_perf_bar = -1
    bridge._last_agent_perf_payload = {}
    bridge._last_player_perf_bar = -1
    bridge._last_player_perf_payload = {}

    # --- Патч _run_cycle ---
    original_run_cycle = bridge._run_cycle

    def patched_run_cycle():
        bridge._shadow_prefetch_active = True
        bridge._shadow_prefetch_done = False
        try:
            original_run_cycle()
        finally:
            bridge._shadow_prefetch_active = False

        prices = bridge._price_hist[-1] if bridge._price_hist else {}
        if not prices:
            return

        volumes = bridge._last_volumes or {}

        bar = bridge._bar
        live_bar = max(0, bar - bridge._warmup_end)
        month = datetime.utcnow().month
        regime = regime_tracker.current

        # Прогоняем shadow-агентов
        if not bridge._shadow_prefetch_done:
            run_shadow_tick(
                shadows, prices, volumes,
                bar, live_bar, month,
                csv_logger, agreement, regime,
            )
            _publish_agent_perf(prices)
            if shadow_players:
                run_shadow_tick(
                    shadow_players, prices, volumes,
                    bar, live_bar, month,
                    csv_logger, agreement, regime,
                )
                _publish_player_perf(prices)

        # Flush agreement tracker
        agreement.flush_tick()

        # ═══════════════════════════════════════════════════════════════
        # ADAPTIVE ROTATION: передаём shadow performance в Panteon
        # ═══════════════════════════════════════════════════════════════
        _publish_agent_perf(prices)
        _publish_player_perf(prices)
        bridge._shadow_prefetch_done = False

        # FIX (2026-04-28): Orphan Position Guardian — раз в 5 live-баров
        # проверяем _open_pos реального игрока на сироты и принудительно
        # закрываем их по SL/TP/STALE через futures_client напрямую.
        if live_bar > 0 and live_bar % 5 == 0:
            try:
                inner_rp = getattr(real_player, "_inner", real_player)
                fc = getattr(bridge, "futures_client", None)
                # SL/TP лучше брать из самого Panteon, чтобы синхронно с
                # обычной PositionSafety (но с +50% буфером на slippage).
                sl = float(getattr(inner_rp, "SL_PCT", 0.04) or 0.04) * 1.5
                tp = float(getattr(inner_rp, "TP_PCT", 0.06) or 0.06)
                stale = int(getattr(inner_rp, "STALE_BARS", 720) or 720)
                _orphan_position_guardian(
                    inner_rp, prices, fc,
                    sl_pct=sl, tp_pct=tp,
                    stale_bars_threshold=stale,
                    current_bar=bar, logger=log,
                )
            except Exception as exc:
                log.debug("  [OrphanGuardian] tick failed: %s", exc)

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
            'per_regime': vp.export_regime_stats(),
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


def _warmup_player_from_history(player, price_hist: list, month: int, volume_hist: list | None = None):
    base_pv = float(getattr(player, "_pv", INITIAL_CAPITAL) or INITIAL_CAPITAL)
    for i, hist_prices in enumerate(price_hist):
        hist_volumes = {}
        if isinstance(volume_hist, list) and i < len(volume_hist):
            candidate = volume_hist[i]
            if isinstance(candidate, dict):
                hist_volumes = candidate
        try:
            player.act(
                prices=hist_prices,
                volumes=hist_volumes,
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
    out = os.path.join(str(PROJECT_ROOT), RESULTS_ROOT, session_date)
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
    shadow_players = _create_shadow_players(detected_capital)

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
    if TRADING_MODE == "paper" and getattr(bridge, "paper_pf", None):
        try:
            paper_pf = next(iter(bridge.paper_pf.values()))
            paper_capital = float(getattr(paper_pf, "initial_capital", 0.0) or 0.0)
            if paper_capital > 0 and abs(float(stats.initial_capital) - paper_capital) > 1e-9:
                stats.initial_capital = paper_capital
                stats.current_balance = paper_capital
                stats.equity_curve = [paper_capital]
                stats.balance_curve = [paper_capital]
                stats.unrealized_curve = [0.0]
                stats.total_assets_curve = [paper_capital]
                stats.pnl_history = [0.0]
                log.info("  [paper] stats capital synced to virtual portfolio: $%.4f", paper_capital)
        except Exception as exc:
            log.debug("paper stats capital sync failed: %s", exc)

    # Патчим bridge для shadow-агентов
    # ВАЖНО: патчим ДО warmup, чтобы shadow-агенты тоже прогрелись
    _patch_bridge_for_shadows(bridge, shadows, csv_logger, agreement, regime_tracker,
                              real_player=raw_player,
                              shadow_players=shadow_players)

    # ── Monitor ────────────────────────────────────────────────────────────
    monitor = ComboMonitorThread(
        direct_client, stats, out, shadows, agreement, regime_tracker,
        shadow_players=shadow_players,
    )
    monitor._bridge = bridge
    # FIX 2026-05-04 (per-symbol blocklist): EnhancedSignalCapture нуждается в
    # ссылке на owner, чтобы достать _position_sync_health и спрашивать его
    # is_symbol_blocked(sym). Без этого block-list существует, но никто не
    # отсекает сигналы для заблокированных символов.
    try:
        capturing_agent._owner = monitor
    except Exception:
        pass

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
    volume_hist = getattr(bridge, '_volume_hist', None)
    for i, hist_prices in enumerate(bridge._price_hist):
        bar_idx = i + 1
        hist_volumes = {}
        if isinstance(volume_hist, list) and i < len(volume_hist):
            candidate = volume_hist[i]
            if isinstance(candidate, dict):
                hist_volumes = candidate
        for name, (agent, vp) in shadows.items():
            try:
                agent.act(
                    prices=hist_prices, volumes=hist_volumes,
                    month=month, portfolio_value=vp.initial_capital,
                    bar_index=bar_idx,
                )
            except Exception:
                pass
        for name, (agent, vp) in shadow_players.items():
            try:
                agent.act(
                    prices=hist_prices, volumes=hist_volumes,
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
    warmup_shadow_player_perf = _build_shadow_perf_from_shadows(shadow_players, prices=warmup_prices)
    try:
        from promotion_gate import PromotionGateAgent
        promotion_summary = PromotionGateAgent().evaluate(warmup_shadow_player_perf)
        _log_promotion_gate_summary(promotion_summary)
    except Exception as e:
        log.debug("promotion gate evaluate failed: %s", e)

    if hasattr(raw_player, 'set_shadow_perf') and warmup_shadow_perf:
        try:
            raw_player.set_shadow_perf(warmup_shadow_perf)
        except Exception as e:
            log.debug("warmup shadow perf inject failed: %s", e)
    if hasattr(raw_player, 'set_shadow_player_perf') and warmup_shadow_player_perf:
        try:
            raw_player.set_shadow_player_perf(warmup_shadow_player_perf)
        except Exception as e:
            log.debug("warmup shadow player perf inject failed: %s", e)
    for shadow_player, _vp in (shadow_players or {}).values():
        try:
            if hasattr(shadow_player, 'set_shadow_perf') and warmup_shadow_perf:
                shadow_player.set_shadow_perf(warmup_shadow_perf)
            if hasattr(shadow_player, 'set_shadow_player_perf') and warmup_shadow_player_perf:
                shadow_player.set_shadow_player_perf(warmup_shadow_player_perf)
        except Exception:
            pass

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
            _warmup_player_from_history(
                replacement,
                bridge._price_hist,
                month,
                getattr(bridge, '_volume_hist', None),
            )
            if hasattr(replacement, 'set_shadow_perf') and warmup_shadow_perf:
                try:
                    replacement.set_shadow_perf(warmup_shadow_perf)
                except Exception:
                    pass
            if hasattr(replacement, 'set_shadow_player_perf') and warmup_shadow_player_perf:
                try:
                    replacement.set_shadow_player_perf(warmup_shadow_player_perf)
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
    for name, (agent, vp) in shadow_players.items():
        _reset_agent_internals(agent, bridge._bar, name)
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
        fut = live_snapshot.get('futures', {}) or {}
        live_balance = float(fut.get('equity', detected_capital) or detected_capital)
        live_available = float(fut.get('available', live_balance) or live_balance)
        live_unrealized = float(fut.get('unrealized', 0.0) or 0.0)
        if hasattr(stats, 'reset_live_baseline'):
            stats.reset_live_baseline(
                detected_capital,
                balance=live_balance,
                available=live_available,
                pnl=live_unrealized,
                live_bar=0,
                reset_activity=True,
            )
        else:
            stats.initial_capital = detected_capital
        raw_player_ref = getattr(capturing_agent, '_inner', raw_player)
        inject_live_state_into_player(raw_player_ref, live_snapshot, bridge._bar)
        log.info("  ✅ Синхронизация завершена")
    except Exception as e:
        log.warning("  Синхронизация не удалась: %s", e)

    try:
        monitor._refresh(render_dashboards=True)
    except Exception as e:
        log.debug("  post-sync monitor refresh failed: %s", e)

    save_dashboard(stats, out, "dashboard_latest.png")
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
            save_dashboard(stats, out, "dashboard_latest.png")
            save_summary_json(stats, out)
            _refresh_html_dashboard_best_effort(force=True)
        except Exception:
            pass

        # Финальный leaderboard
        vp_map = {name: tup[1] for name, tup in shadows.items()}
        try:
            final_shadow_perf = _build_shadow_perf_from_shadows(shadows)
            final_shadow_player_perf = _build_shadow_perf_from_shadows(shadow_players)
            if hasattr(raw_player, 'set_shadow_perf'):
                raw_player.set_shadow_perf(final_shadow_perf)
            if hasattr(raw_player, 'set_shadow_player_perf'):
                raw_player.set_shadow_player_perf(final_shadow_player_perf)
            if hasattr(raw_player, 'save_memory_snapshot'):
                raw_player.save_memory_snapshot(force=True, reason="shutdown")
        except Exception as e:
            log.debug("real player memory save failed: %s", e)
        if not ROTATION_CHANGE_LOGS_ONLY:
            log_leaderboard(vp_map, stats.pnl_pct, regime_tracker.current,
                            stats.bar_count, regime_tracker)
            player_vp_map = {name: tup[1] for name, tup in shadow_players.items()}
            if player_vp_map:
                log_leaderboard(player_vp_map, stats.pnl_pct, regime_tracker.current,
                                stats.bar_count, regime_tracker)
            log.info(agreement.log_matrix())

        # Финальный JSON
        try:
            # FIX (2026-04-27): trades_per_hour и selector_diagnostics для
            # анализа эффекта фиксов H1-H5.
            def _parse_uptime_hours(s):
                try:
                    parts = str(s or "0:0:0").split(":")
                    if len(parts) == 3:
                        h, m, sec = (float(p) for p in parts)
                        return h + m / 60.0 + sec / 3600.0
                    if len(parts) == 2:
                        m, sec = (float(p) for p in parts)
                        return m / 60.0 + sec / 3600.0
                except Exception:
                    pass
                return 0.0
            uptime_h = max(_parse_uptime_hours(getattr(stats, "uptime_str", "0:0:0")), 1e-6)

            def _vp_block(vp):
                tph = float(vp.total_trades) / uptime_h if uptime_h > 0 else 0.0
                rpt = (float(vp.total_pnl) / max(float(vp.initial_capital), 1e-9) * 100.0
                       / max(vp.total_trades, 1)) if vp.total_trades else 0.0
                return {
                    "pnl_pct": round(vp.pnl_pct, 4),
                    "win_rate": round(vp.win_rate, 2),
                    "signals": vp.signal_count,
                    "entries": vp.entry_count,
                    "closed_trades": vp.close_count,
                    "total_trades": vp.total_trades,
                    "sharpe": round(vp.sharpe(), 4),
                    "max_dd_pct": round(vp.max_drawdown * 100, 2),
                    "trades_per_hour": round(tph, 3),
                    "realised_per_trade_pct": round(rpt, 4),
                    "per_regime": vp.export_regime_stats(),
                    "last_trades": vp.trades_log[-10:],
                }

            final_report = {
                "session": session_date,
                "real_player": selected_player_name,
                "duration": stats.uptime_str,
                "real_pnl_pct": round(stats.pnl_pct, 4),
                "real_signals": len(stats.signals),
                "real_trades": len(stats.trades),
                "real_trades_per_hour": round(len(stats.trades) / uptime_h, 3) if uptime_h > 0 else 0.0,
                "regime_summary": regime_tracker.summary(),
                "regime_changes": len(regime_tracker.history),
                "regime_flips_per_hour": round(len(regime_tracker.history) / uptime_h, 3) if uptime_h > 0 else 0.0,
                "promotion_gate": promotion_summary,
                "shadow_agents": {},
                "shadow_players": {},
            }
            try:
                inner_rp = getattr(capturing_agent, "_inner", raw_player)
                scores = dict(getattr(inner_rp, "_shadow_player_scores", {}) or {})
                pool = list((getattr(inner_rp, "_shadow_player_pool", {}) or {}).keys())
                final_report["selector_diagnostics"] = {
                    "pool_size": len(pool),
                    "pool": pool,
                    "currently_selected": str(getattr(inner_rp, "_selected_shadow_player", "") or ""),
                    "currently_challenger": str(getattr(inner_rp, "_shadow_player_challenger", "") or ""),
                    "last_scores": {k: round(float(v), 4) for k, v in scores.items()},
                    "has_solo_in_pool": any(n.startswith("V_Solo") for n in pool),
                    "last_force_player_rotation_bar": int(
                        getattr(inner_rp, "_last_force_player_rotation_bar", -99999) or -99999),
                    "last_force_agent_rotation_bar": int(
                        getattr(inner_rp, "_last_force_agent_rotation_bar", -99999) or -99999),
                }
            except Exception as exc:
                log.debug("selector_diagnostics build failed: %s", exc)
            for name, vp in vp_map.items():
                final_report["shadow_agents"][name] = _vp_block(vp)
            for name, (_, vp) in shadow_players.items():
                final_report["shadow_players"][name] = _vp_block(vp)
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
        log.info("  Shadow-players:    %d", len(shadow_players))
        log.info("  Shadow-агентов:    %d", len(shadows))
        log.info("  Смен режима:       %d", len(regime_tracker.history))
        log.info("  CSV сигналов:      %s", SIGNAL_CSV_FILE)
        log.info("═" * 78)


if __name__ == "__main__":
    main()
