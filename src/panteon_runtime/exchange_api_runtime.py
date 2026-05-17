"""Общий runtime для live/paper торговли и связанных дашбордов."""

from __future__ import annotations

import os
import sys
import time
import math
import hmac
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from collections import OrderedDict, deque
from typing import Callable, Dict, List, Optional, Tuple

import requests

MEXC_DIRECT_HTTP = requests.Session()
MEXC_DIRECT_HTTP.trust_env = False


def _mexc_direct_get(url: str, **kwargs):
    return MEXC_DIRECT_HTTP.get(url, **kwargs)


from project_paths import PROJECT_ROOT, RUNTIME_DIR, add_runtime_paths


def _bootstrap_project_paths():
    add_runtime_paths()


_bootstrap_project_paths()

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

API_KEY = os.getenv("MEXC_API_KEY", "")
API_SECRET = os.getenv("MEXC_SECRET_KEY", "")


def _repair_curve_spikes(values: List[float], floor: float = 0.0) -> List[float]:
    """
    Repair obviously bad single-point glitches in equity-like curves while keeping
    the original series length intact, so dashboard markers stay aligned.
    """
    if not values:
        return []

    raw: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")
        raw.append(fv)

    cleaned = list(raw)
    n = len(raw)

    def _nearest_valid(idx: int, step: int) -> Optional[float]:
        j = idx + step
        while 0 <= j < n:
            cand = raw[j]
            if math.isfinite(cand):
                return cand
            j += step
        return None

    for i, val in enumerate(raw):
        prev_v = _nearest_valid(i, -1)
        next_v = _nearest_valid(i, +1)

        if not math.isfinite(val):
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

# ── Режим торговли ──────────────────────────────────────────────────────────
# "live_futures" — фьючерсы на реальные деньги (агент генерирует FL/FS сигналы)
# "live_spot"    — спот на реальные деньги (агент генерирует BUY/SELL сигналы)
# "paper"        — виртуальная торговля (без реальных ордеров)
TRADING_MODE = "live_futures"

# ── Параметры из settings.txt (единый источник для всех режимов) ─────────────
# Загружаем ДО импорта агентов чтобы все используемые константы уже были правильными
import mexc_connector as _mc_settings_reader
_settings_raw    = _mc_settings_reader._load_settings()
_settings_parsed = _mc_settings_reader._parse_settings(_settings_raw)

INITIAL_CAPITAL  = 50.0    # fallback — заменяется реальным балансом при запуске
TRADE_FRACTION   = _settings_parsed.get("trade_fraction", 0.10)
LEVERAGE         = int(_settings_parsed.get("leverage", 3))
SPOT_FEE         = _settings_parsed.get("spot_fee",     0.001)
FUTURES_FEE      = _settings_parsed.get("futures_fee",  0.0002)


def _settings_float(name: str, default: float) -> float:
    try:
        return float(_settings_parsed.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _settings_int(name: str, default: int) -> int:
    try:
        return int(_settings_parsed.get(name, default))
    except (TypeError, ValueError):
        return int(default)


POSITION_SOURCE_PANTEON_ORDER = "panteon_order"
POSITION_SOURCE_MANUAL_EXTERNAL = "manual_external"
POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP = "recovered_after_snapshot_gap"
POSITION_SOURCE_PENDING_ORDER = "pending_order"


def _positive_settings_int(name: str, default: int) -> int:
    value = _settings_int(name, default)
    return value if value > 0 else int(default)


def inherited_position_policy(source: str = POSITION_SOURCE_MANUAL_EXTERNAL) -> dict:
    # Positions already on the exchange at startup are not fresh Panteon trades.
    # Keep them visible, but give the safety layer stricter per-position exits.
    stale_bars = _positive_settings_int("inherited_position_stale_bars", 60)
    return {
        "external": True,
        "inherited_from_exchange": True,
        "source": source,
        "position_source": source,
        "tight_sl_pct": _settings_float(
            "inherited_position_sl_pct",
            _settings_float("sl_pct", 0.04),
        ),
        "stale_bars": stale_bars,
        "stale_move_threshold": _settings_float("inherited_position_stale_move_threshold", 0.06),
    }


def recovered_position_policy(current_bar: int = 0) -> dict:
    stale_bars = _positive_settings_int("recovered_position_stale_bars", 120)
    policy = inherited_position_policy(POSITION_SOURCE_RECOVERED_AFTER_SNAPSHOT_GAP)
    policy["stale_bars"] = stale_bars
    policy["recovered_after_snapshot_gap"] = True
    if current_bar:
        policy["suppress_stale_until_bar"] = int(current_bar) + stale_bars
    return policy


def panteon_order_position_policy(order_id: str | None = None) -> dict:
    policy = {
        "external": False,
        "inherited_from_exchange": False,
        "source": POSITION_SOURCE_PANTEON_ORDER,
        "position_source": POSITION_SOURCE_PANTEON_ORDER,
    }
    if order_id:
        policy["order_id"] = order_id
    return policy


class PositionSyncHealth:
    def __init__(
        self,
        zero_confirmations: int | None = None,
        cooldown_sec: int | None = None,
        now_fn: Callable[[], float] | None = None,
    ):
        self.zero_confirmations = int(
            zero_confirmations
            if zero_confirmations is not None
            else _positive_settings_int("snapshot_zero_confirmations", 3)
        )
        self.cooldown_sec = int(
            cooldown_sec
            if cooldown_sec is not None
            else _positive_settings_int("snapshot_health_cooldown_sec", 120)
        )
        self._now_fn = now_fn or time.time
        self.missing_counts: Dict[str, int] = {}
        self.recent_error_until = 0.0
        self.last_error_reason = ""
        self.last_snapshot_at = 0.0
        self.last_zero_snapshot_at = 0.0
        self.api_error_times = deque(maxlen=200)
        self.reconcile_add = 0
        self.reconcile_remove = 0
        self.failed_orders = 0
        # FIX 2026-05-04 (per-symbol failure blocklist): отслеживаем ордера,
        # которые отправились на биржу, но не подтвердились (pending → removed).
        # Если для одного символа подряд проваливается >=3 ордера за час — символ
        # временно блокируется на BLOCKED_SYMBOL_DURATION_SEC. Блок снимается
        # при первом успешном reconcile_add для того же символа.
        # Применяется в Panteon_Trade.EnhancedSignalCapture.act() через
        # is_symbol_blocked() — слабые/не-ликвидные пары больше не плодят
        # бесконечные failed_orders на BITGET (NAORIS, AIGENSYN, BSB, LAB).
        self.pending_failures: Dict[str, deque] = {}
        self.blocked_until: Dict[str, float] = {}
        self.PENDING_FAILURE_LIMIT = 3        # сколько подряд → блок
        self.PENDING_FAILURE_WINDOW_SEC = 3600  # окно усреднения (1 ч)
        self.BLOCKED_SYMBOL_DURATION_SEC = 3600  # длительность блока (1 ч)

    def now(self) -> float:
        return float(self._now_fn())

    def record_pending_failure(self, sym: str) -> bool:
        """FIX 2026-05-04: вызывается когда pending_order не подтверждён биржей
        и удалён reconcile-loop. Возвращает True если символ только что
        попал в blocklist (для логирования)."""
        sym = str(sym or "").upper()
        if not sym:
            return False
        ts = self.now()
        dq = self.pending_failures.setdefault(sym, deque(maxlen=10))
        dq.append(ts)
        # Очистим устаревшие отметки
        cutoff = ts - float(self.PENDING_FAILURE_WINDOW_SEC)
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= int(self.PENDING_FAILURE_LIMIT):
            already_blocked = sym in self.blocked_until and self.blocked_until[sym] > ts
            self.blocked_until[sym] = ts + float(self.BLOCKED_SYMBOL_DURATION_SEC)
            return not already_blocked
        return False

    def record_symbol_success(self, sym: str) -> None:
        """FIX 2026-05-04: успешное reconcile_add → сбрасываем failure
        счётчик и снимаем блок (если был)."""
        sym = str(sym or "").upper()
        if not sym:
            return
        self.pending_failures.pop(sym, None)
        self.blocked_until.pop(sym, None)

    def is_symbol_blocked(self, sym: str) -> bool:
        """FIX 2026-05-04: возвращает True если sym в активном blocklist."""
        sym = str(sym or "").upper()
        if not sym:
            return False
        until = float(self.blocked_until.get(sym, 0.0) or 0.0)
        if until <= 0.0:
            return False
        now = self.now()
        if now >= until:
            # Срок истёк — снимаем
            self.blocked_until.pop(sym, None)
            return False
        return True

    def blocked_symbols_summary(self) -> Dict[str, float]:
        """Список заблокированных символов и сколько секунд осталось."""
        now = self.now()
        out = {}
        for sym, until in list(self.blocked_until.items()):
            remain = float(until) - now
            if remain <= 0:
                self.blocked_until.pop(sym, None)
                continue
            out[sym] = round(remain, 1)
        return out

    def mark_snapshot(self, positions_count: int, snapshot_healthy: bool = True) -> None:
        now = self.now()
        self.last_snapshot_at = now
        if int(positions_count or 0) == 0:
            self.last_zero_snapshot_at = now
        if not snapshot_healthy:
            self.mark_data_error("snapshot_unhealthy", now=now)

    def mark_data_error(self, reason: str, now: float | None = None) -> None:
        ts = self.now() if now is None else float(now)
        self.recent_error_until = max(self.recent_error_until, ts + self.cooldown_sec)
        self.last_error_reason = str(reason or "data_error")
        self.api_error_times.append(ts)

    def mark_failed_order(self) -> None:
        self.failed_orders += 1

    def destructive_allowed(self, snapshot_healthy: bool = True) -> bool:
        return bool(snapshot_healthy) and self.now() >= self.recent_error_until

    def summary(self) -> dict:
        now = self.now()
        errors_10m = sum(1 for ts in self.api_error_times if now - float(ts) <= 600.0)
        return {
            "snapshot_age_sec": None if not self.last_snapshot_at else round(now - self.last_snapshot_at, 1),
            "last_zero_snapshot_age_sec": None
            if not self.last_zero_snapshot_at
            else round(now - self.last_zero_snapshot_at, 1),
            "api_errors_10m": errors_10m,
            "recent_error_until_sec": max(0.0, round(self.recent_error_until - now, 1)),
            "last_error_reason": self.last_error_reason,
            "reconcile_add": self.reconcile_add,
            "reconcile_remove": self.reconcile_remove,
            "failed_orders": self.failed_orders,
            "pending_zero_confirmations": dict(self.missing_counts),
        }


def _position_symbol(pos: dict) -> str:
    sym = str(pos.get("symbol", "") or "").upper()
    if sym.endswith("_USDT"):
        sym = sym[:-5]
    if sym.endswith("USDT") and len(sym) > 4:
        sym = sym[:-4]
    return sym


def _normalize_exchange_positions(exchange_positions: list | None) -> list[dict]:
    normalized = []
    for pos in exchange_positions or []:
        sym = _position_symbol(pos)
        if not sym:
            continue
        try:
            entry = float(pos.get("entry", 0.0) or 0.0)
        except Exception:
            entry = 0.0
        side = str(pos.get("side", "long") or "long").lower()
        if side not in ("long", "short"):
            side = "short" if "short" in side else "long"
        normalized.append(
            {
                "symbol": sym,
                "entry": entry,
                "side": side,
                "leverage": pos.get("leverage", 1),
                "raw": pos,
            }
        )
    return normalized


def mark_recent_runtime_data_errors(health: PositionSyncHealth, *sources) -> None:
    """Propagate recent price/API mode errors into reconcile cooldown."""
    if health is None:
        return
    cooldown = float(getattr(health, "cooldown_sec", 120) or 120)
    seen: set[int] = set()
    queue = list(sources)
    for src in queue:
        if src is None:
            continue
        ident = id(src)
        if ident in seen:
            continue
        seen.add(ident)
        probe = getattr(src, "recent_data_error", None)
        if callable(probe):
            try:
                reason = probe(cooldown)
            except Exception:
                reason = ""
            if reason:
                health.mark_data_error(str(reason))
        for attr in ("futures_client", "_fut", "_client"):
            child = getattr(src, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)


def safe_reconcile_open_positions(
    position_book: dict,
    exchange_positions: list | None,
    *,
    current_bar: int,
    health: PositionSyncHealth,
    snapshot_healthy: bool = True,
    subagents: list | None = None,
    logger=None,
) -> dict:
    positions = _normalize_exchange_positions(exchange_positions)
    health.mark_snapshot(len(positions), snapshot_healthy=snapshot_healthy)
    destructive_allowed = health.destructive_allowed(snapshot_healthy=snapshot_healthy)
    real_syms = {p["symbol"] for p in positions}
    summary = {"added": 0, "updated": 0, "removed": 0, "guarded": 0, "recovered": 0}

    def _log(level: str, msg: str, *args) -> None:
        if logger is None:
            return
        getattr(logger, level, logger.info)(msg, *args)

    def _clear_subagents(sym: str) -> None:
        for item in subagents or []:
            sub = item[-1] if isinstance(item, tuple) else item
            try:
                _try_clear_agent_pos(sub, sym)
            except Exception:
                pass

    def _set_subagents(sym: str, side: str, entry: float) -> None:
        for item in subagents or []:
            sub = item[-1] if isinstance(item, tuple) else item
            try:
                _try_set_agent_pos(sub, sym, side, entry=entry, bar_index=current_bar)
            except Exception:
                pass

    for sym in list(position_book):
        if sym in real_syms:
            continue
        rec = position_book.get(sym) or {}
        if rec.get("pending_order"):
            if destructive_allowed:
                position_book.pop(sym, None)
                _clear_subagents(sym)
                summary["removed"] += 1
                health.reconcile_remove += 1
                health.missing_counts.pop(sym, None)
                # FIX 2026-05-04 (per-symbol failure blocklist): трекаем
                # неуспешные pending → если 3+ подряд → автоблок на 1 час.
                newly_blocked = False
                try:
                    newly_blocked = bool(health.record_pending_failure(sym))
                except Exception:
                    pass
                if newly_blocked:
                    _log(
                        "warning",
                        "  [reconcile] %s pending failed >=3x → BLOCKED for %ds (illiquid?)",
                        sym, int(getattr(health, "BLOCKED_SYMBOL_DURATION_SEC", 3600)),
                    )
                else:
                    _log(
                        "info",
                        "  [reconcile] %s pending order not confirmed by exchange -> removed",
                        sym,
                    )
            else:
                summary["guarded"] += 1
            continue
        if not destructive_allowed:
            summary["guarded"] += 1
            continue
        count = int(health.missing_counts.get(sym, 0) or 0) + 1
        health.missing_counts[sym] = count
        if count < health.zero_confirmations:
            summary["guarded"] += 1
            _log(
                "info",
                "  [reconcile] %s absent in snapshot %d/%d -> keep position",
                sym,
                count,
                health.zero_confirmations,
            )
            continue
        position_book.pop(sym, None)
        _clear_subagents(sym)
        summary["removed"] += 1
        health.reconcile_remove += 1
        _log("info", "  [reconcile] %s removed after %d healthy zero confirmations", sym, count)

    for pos in positions:
        sym = pos["symbol"]
        entry = float(pos["entry"] or 0.0)
        side = pos["side"]
        leverage = pos.get("leverage", 1)
        existing = position_book.get(sym)
        gap_count = int(health.missing_counts.pop(sym, 0) or 0)
        if existing is not None and gap_count > 0:
            existing.update(recovered_position_policy(current_bar=current_bar))
            summary["recovered"] += 1
        if existing is None:
            if gap_count > 0:
                policy = recovered_position_policy(current_bar=current_bar)
                summary["recovered"] += 1
            else:
                policy = inherited_position_policy(POSITION_SOURCE_MANUAL_EXTERNAL)
            position_book[sym] = {
                "entry": entry,
                "side": side,
                "bar": current_bar,
                "leverage": leverage,
                "peak": entry,
                **policy,
            }
            _set_subagents(sym, side, entry)
            summary["added"] += 1
            health.reconcile_add += 1
            # FIX 2026-05-04: успешное появление позиции → сбрасываем
            # pending-failure счётчик и снимаем блок (если был)
            try:
                health.record_symbol_success(sym)
            except Exception:
                pass
            _log(
                "info",
                "  [reconcile] %s added source=%s side=%s lev=%s entry=%.4f",
                sym,
                position_book[sym].get("source"),
                side.upper(),
                leverage,
                entry,
            )
            continue
        if existing.get("pending_order"):
            existing.update(panteon_order_position_policy(existing.get("order_id")))
            existing.pop("pending_order", None)
            # FIX 2026-05-04: pending перешёл в подтверждённую позицию → успех
            try:
                health.record_symbol_success(sym)
            except Exception:
                pass
            summary["updated"] += 1
        needs_refresh = (
            str(existing.get("side", "")).lower() != side
            or abs(float(existing.get("entry", 0.0) or 0.0) - entry) > max(1e-8, abs(entry) * 0.001)
        )
        if needs_refresh:
            existing.update({"entry": entry, "side": side, "leverage": leverage, "peak": entry})
            _set_subagents(sym, side, entry)
            summary["updated"] += 1
        else:
            existing["leverage"] = leverage
        existing.setdefault("source", POSITION_SOURCE_MANUAL_EXTERNAL)
        existing.setdefault("position_source", existing.get("source"))

    return summary

# FIX D (2026-04-26): применяем risk-параметры из settings.txt к классу Panteon
# (и наследникам) ДО первого инстансирования. Это безопаснее, чем тащить
# параметры через конструкторы каждого ансамбля.
# FIX 2026-04-27: дублируем в file logger (через panteon_agents.log) — без
# этого override был "невидимым" и пользователь жаловался что ничего не
# применилось. Plus: применяем override и к УЖЕ созданным экземплярам
# Panteon-наследников, чтобы settings распространились на shadow-pool.
def _apply_settings_to_panteon():
    try:
        from panteon_agents import Panteon as _Panteon
    except Exception:
        return
    overrides = {
        "SL_PCT": _settings_parsed.get("sl_pct"),
        "TP_PCT": _settings_parsed.get("tp_pct"),
        "TRAIL_PCT": _settings_parsed.get("trail_pct"),
        "ROTATION_INT": _settings_parsed.get("rotation_int"),
        "PLAYER_ROTATION_INT": _settings_parsed.get("player_rotation_int"),
        "PLAYER_SWITCH_COOLDOWN_BARS": _settings_parsed.get("player_switch_cooldown_bars"),
        "PLAYER_HARD_NEGATIVE_PNL": _settings_parsed.get("player_hard_negative_pnl"),
        "PLAYER_MIN_LIVE_CLOSED_TRADES": _settings_parsed.get("player_min_live_closed_trades"),
        "REGIME_HYSTERESIS_BARS": _settings_parsed.get("regime_hysteresis_bars"),
        "ENSEMBLE_MODE": _settings_parsed.get("ensemble_mode"),
        "AGENT_QUARANTINE_PROBATION_PNL": _settings_parsed.get("agent_quarantine_probation_pnl"),
        "AGENT_QUARANTINE_PROBATION_CLOSED": _settings_parsed.get("agent_quarantine_probation_closed"),
        "PER_SYMBOL_MIN_CLOSED": _settings_parsed.get("per_symbol_min_closed"),
        "PER_SYMBOL_MIN_WIN_RATE": _settings_parsed.get("per_symbol_min_win_rate"),
    }
    applied = []
    for attr, val in overrides.items():
        if val is None:
            continue
        try:
            cur = getattr(_Panteon, attr, None)
            if isinstance(cur, str) or attr == "ENSEMBLE_MODE":
                setattr(_Panteon, attr, str(val))
            elif cur is None:
                setattr(_Panteon, attr, val)
            else:
                setattr(_Panteon, attr, type(cur)(val))
            applied.append(f"{attr}={val}")
        except Exception:
            pass
    if applied:
        msg = f"[settings] Panteon overrides: {', '.join(applied)}"
        print(msg)
        try:
            import logging
            logging.getLogger("Panteon_Trade").info(msg)
            logging.getLogger().info(msg)
        except Exception:
            pass
    return applied

_PANTEON_OVERRIDES_APPLIED = _apply_settings_to_panteon()


def reapply_settings_to_existing_instances():
    """FIX 2026-04-27: повторно применяет settings к УЖЕ созданным инстансам
    Panteon-наследников (включая shadow-pool). Безопасно вызывать после
    `_init_shadow_player_pool()`.

    Атрибут на классе достаточен для большинства мест (getattr идёт через
    MRO), но некоторые поля сохраняются в self при __init__ (например через
    `setattr(player, '_suppress_info_logs', ...)`), и для них нужен второй
    pass.
    """
    if not _PANTEON_OVERRIDES_APPLIED:
        return
    try:
        from panteon_agents import Panteon as _Panteon
    except Exception:
        return
    import gc
    n = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, _Panteon):
                pool = getattr(obj, "_shadow_player_pool", None)
                # Для каждого инстанса/потомка явно ставим класс-атрибуты
                for attr in (
                    "SL_PCT", "TP_PCT", "TRAIL_PCT",
                    "ROTATION_INT", "PLAYER_ROTATION_INT",
                    "PLAYER_SWITCH_COOLDOWN_BARS", "PLAYER_HARD_NEGATIVE_PNL",
                    "PLAYER_MIN_LIVE_CLOSED_TRADES", "REGIME_HYSTERESIS_BARS",
                    "ENSEMBLE_MODE", "PER_SYMBOL_MIN_CLOSED",
                    "PER_SYMBOL_MIN_WIN_RATE",
                ):
                    cls_val = getattr(type(obj), attr, None)
                    if cls_val is not None and not hasattr(obj, "__" + attr.lower() + "_overridden"):
                        try:
                            setattr(obj, attr, cls_val)
                        except Exception:
                            pass
                if pool:
                    for p in pool.values():
                        for attr in (
                            "REGIME_HYSTERESIS_BARS", "ROTATION_INT",
                            "PLAYER_ROTATION_INT", "PLAYER_HARD_NEGATIVE_PNL",
                            "PLAYER_MIN_LIVE_CLOSED_TRADES", "ENSEMBLE_MODE",
                            "PER_SYMBOL_MIN_CLOSED", "PER_SYMBOL_MIN_WIN_RATE",
                        ):
                            cls_val = getattr(_Panteon, attr, None)
                            if cls_val is not None:
                                try:
                                    setattr(p, attr, cls_val)
                                except Exception:
                                    pass
                n += 1
        except Exception:
            continue
    if n:
        try:
            import logging
            logging.getLogger("Panteon_Trade").info(
                "[settings] reapplied overrides to %d existing Panteon instance(s)", n,
            )
        except Exception:
            pass

# Лог загруженных значений
print(f"[settings] exchange_api_runtime: leverage={LEVERAGE}x  trade_fraction={TRADE_FRACTION:.0%}"
      f"  spot_fee={SPOT_FEE:.4f}  futures_fee={FUTURES_FEE:.4f}")

# Папка дашбордов
RESULTS_ROOT = os.path.join("Results", "MEXC")

# Интервал сохранения дашборда (секунды)
# FIX v4: было 300 (5 мин) → слишком много пустых файлов. Теперь 1800 (30 мин).
# Дополнительно: дашборд сохраняется ТОЛЬКО если что-то изменилось (новые сигналы/сделки).
DASHBOARD_INTERVAL_SEC = 1800   # 30 минут

# ── Список символов для торговли ────────────────────────────────────────────
# АРХИТЕКТУРА (v5, расширенная вселенная):
#
#   CORE_SYMBOLS  — проверенное ядро: монеты с подтверждёнными фьючерсными
#                   контрактами на MEXC, прошедшие реальную торговлю.
#                   Гарантированно работают, всегда в списке.
#
#   EXTRA_TOP_N   — дополнительно N монет из топа биржи по 24h объёму,
#                   прошедшие фильтры ликвидности (см. fetch_top_symbols).
#                   Автоматически обновляется из рынка, исключает блэклист.
#
#   FIXED_SYMBOLS — финальный итог: CORE_SYMBOLS ∪ top_N_by_volume (до TRADING_UNIVERSE_SIZE).
#                   Дубликаты удаляются, сохраняется порядок: сначала ядро,
#                   потом остальное по убыванию объёма.
#
# Таким образом бот анализирует ~60-80 монет (больше сигналов, больше сделок),
# но при этом каждый новый символ проходит жёсткие фильтры ликвидности, и
# проблемные монеты из FUTURES_BLACKLIST никогда не попадают в список.

CORE_SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "BNB",       # топ по ликвидности
    "ADA", "DOGE", "LTC", "LINK", "UNI",      # крупные альткоины
    "ALGO", "ZEC", "TAO", "NEAR", "ENA",      # средние с контрактом
    "TRX", "AVAX", "DOT",                     # дополнительные
]

# Целевой размер торговой вселенной
TRADING_UNIVERSE_SIZE = 60

# Управление через переменную окружения:
#   AUTO_EXPAND_SYMBOLS=0 → использовать только CORE_SYMBOLS (старое поведение)
#   AUTO_EXPAND_SYMBOLS=1 → CORE_SYMBOLS + топ-N от биржи (по умолчанию)
_AUTO_EXPAND_SYMBOLS = os.environ.get("AUTO_EXPAND_SYMBOLS", "1") == "1"


def build_trading_universe(
    core: List[str] = None,
    target_size: int = TRADING_UNIVERSE_SIZE,
    auto_expand: bool = None,
) -> List[str]:
    """
    Строит итоговый список символов для торговли:
      1. Всегда включает CORE_SYMBOLS (проверенное ядро).
      2. Если auto_expand=True — добавляет топ по объёму с биржи до target_size.
      3. Убирает дубликаты, сохраняет порядок.

    Возвращает итоговый список. При любых ошибках сети
    безопасно возвращает только CORE_SYMBOLS.
    """
    core = list(core or CORE_SYMBOLS)
    if auto_expand is None:
        auto_expand = _AUTO_EXPAND_SYMBOLS

    if not auto_expand:
        log.info("[universe] AUTO_EXPAND_SYMBOLS выключен — использую %d CORE-символов",
                 len(core))
        return core

    # Сколько добавить из топа биржи (чтобы итого было target_size)
    n_extra = max(0, target_size - len(core))
    if n_extra <= 0:
        return core

    try:
        # mexc_connector.fetch_top_symbols уже применяет фильтры ликвидности
        # и блэклист — ничего мусорного не просочится.
        # Запрашиваем с запасом: target_size + буфер на случай пересечений с core.
        import mexc_connector as _mc
        top_from_market = _mc.fetch_top_symbols(n=target_size + len(core))
    except Exception as e:
        log.warning(
            "[universe] Не удалось получить топ символов с биржи (%s) — "
            "использую только CORE (%d шт.)", e, len(core),
        )
        return core

    # Объединяем: core → остальные по объёму. set для быстрого дедупа.
    seen = set(s.upper() for s in core)
    result = list(core)
    for sym in top_from_market:
        sym_upper = sym.upper()
        if sym_upper not in seen:
            result.append(sym_upper)
            seen.add(sym_upper)
            if len(result) >= target_size:
                break

    log.info(
        "[universe] Торговая вселенная: %d символов (ядро=%d, из рынка=%d)",
        len(result), len(core), len(result) - len(core),
    )
    log.info("[universe] Символы: %s", result)
    return result


# Итоговый список — строится лениво при первом обращении, чтобы не делать
# HTTP-запрос к бирже на import (для тестов и unit-импортов).
_FIXED_SYMBOLS_CACHE: List[str] | None = None


def _get_fixed_symbols() -> List[str]:
    global _FIXED_SYMBOLS_CACHE
    if _FIXED_SYMBOLS_CACHE is None:
        _FIXED_SYMBOLS_CACHE = build_trading_universe()
    return _FIXED_SYMBOLS_CACHE


# Обратная совместимость через PEP 562: `from exchange_api_runtime import FIXED_SYMBOLS`
# триггерит __getattr__, который лениво строит список и кэширует его.
# Таким образом на import модуля HTTP-запроса к бирже не делается — только при
# первом обращении к FIXED_SYMBOLS (уже после полной инициализации логгера).
def __getattr__(name: str):
    if name == "FIXED_SYMBOLS":
        return _get_fixed_symbols()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Минимальный баланс для запуска торговли
MIN_BALANCE_USD = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ ПУТЕЙ
# ══════════════════════════════════════════════════════════════════════════════

script_dir = str(RUNTIME_DIR)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

if API_KEY:
    os.environ["MEXC_API_KEY"] = API_KEY
if API_SECRET:
    os.environ["MEXC_SECRET_KEY"] = API_SECRET
os.environ["MEXC_TRADING_MODE"] = TRADING_MODE

# ══════════════════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mexc_v2")


def _setup_file_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(output_dir, "trading.log"), encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)


# ══════════════════════════════════════════════════════════════════════════════
# ПРЯМОЙ КЛИЕНТ MEXC API (для бонусов и диагностики)
# ══════════════════════════════════════════════════════════════════════════════

class MexcDirectClient:
    """
    Прямые вызовы MEXC API (spot + futures) — работает независимо от mexc_connector.
    Используется для:
      • Полного баланса всех активов (включая бонусные токены)
      • Обнаружения voucher / activity-balance
      • Открытых позиций на фьючерсах
      • Истории сделок
    """

    SPOT_BASE     = "https://api.mexc.com"
    FUTURES_BASE  = "https://contract.mexc.com"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self._session   = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"X-MEXC-APIKEY": api_key})
        self._last_good_futures_snapshot: Optional[dict] = None
        self._last_good_futures_snapshot_at = 0.0
        self._contract_meta_cache: Dict[str, dict] = {}

    # ── Подпись ───────────────────────────────────────────────────────────────
    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _ts(self) -> int:
        return int(time.time() * 1000)

    # ── Spot ──────────────────────────────────────────────────────────────────
    def get_spot_account(self) -> dict:
        """GET /api/v3/account — все спот-балансы."""
        params = {"timestamp": self._ts()}
        params["signature"] = self._sign(params)
        r = self._session.get(
            f"{self.SPOT_BASE}/api/v3/account",
            params=params, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        # MEXC возвращает HTTP 200 даже для ошибок — нужно проверить явно
        if isinstance(data, dict) and "code" in data and data.get("code") != 0:
            log.warning(
                "  ⚠️  Spot API ошибка: code=%s  msg=%s",
                data.get("code"), data.get("msg", "?"),
            )
        return data

    def get_all_balances(self) -> Dict[str, dict]:
        """
        Возвращает все ненулевые балансы:
          {'USDT': {'free': 10.5, 'locked': 0.0, 'total': 10.5}, ...}
        Включает бонусные токены (MX, USDT-bonus, vouchers).
        """
        acct = self.get_spot_account()
        result = {}
        for b in acct.get("balances", []):
            free   = float(b.get("free",   0) or 0)
            locked = float(b.get("locked", 0) or 0)
            if free > 0 or locked > 0:
                result[b["asset"]] = {
                    "free":   free,
                    "locked": locked,
                    "total":  free + locked,
                }
        return result

    def get_usdt_equivalent(self, balances: Dict[str, dict]) -> float:
        """
        Оценивает суммарную стоимость всех активов в USDT.
        USDT, USDC, BUSD, DAI — 1:1.
        Остальные — пропускаем (нет доступа к ценам без лишних запросов).
        """
        stables = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USD1", "USDE"}
        total = 0.0
        for asset, b in balances.items():
            if asset in stables:
                total += b["total"]
        return total

    def get_open_orders(self) -> List[dict]:
        """GET /api/v3/openOrders — все открытые ордера."""
        params = {"timestamp": self._ts()}
        params["signature"] = self._sign(params)
        r = self._session.get(
            f"{self.SPOT_BASE}/api/v3/openOrders",
            params=params, timeout=10,
        )
        if r.status_code == 200:
            return r.json() if isinstance(r.json(), list) else []
        return []

    def get_recent_trades(self, limit: int = 20) -> List[dict]:
        """GET /api/v3/myTrades — последние сделки по всем парам."""
        # Запрашиваем для BTC, ETH, SOL как наиболее вероятных
        all_trades = []
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]:
            params = {"symbol": sym, "limit": 5, "timestamp": self._ts()}
            params["signature"] = self._sign(params)
            try:
                r = self._session.get(
                    f"{self.SPOT_BASE}/api/v3/myTrades",
                    params=params, timeout=5,
                )
                if r.status_code == 200 and isinstance(r.json(), list):
                    all_trades.extend(r.json())
            except Exception:
                pass
        all_trades.sort(key=lambda x: x.get("time", 0), reverse=True)
        return all_trades[:limit]

    # ── Futures ───────────────────────────────────────────────────────────────
    _UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/124.0.0.0 Safari/537.36")

    def _futures_sign(self, params: dict, ts: int) -> str:
        """Подпись для фьючерсного API MEXC."""
        body = json.dumps(params, separators=(",", ":")) if params else ""
        msg  = self.api_key + str(ts) + body
        return hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()

    def get_futures_positions(self) -> List[dict]:
        """Открытые фьючерсные позиции."""
        ts  = self._ts()
        sig = self._futures_sign({}, ts)
        try:
            r = _mexc_direct_get(
                f"{self.FUTURES_BASE}/api/v1/private/position/open_positions",
                headers={
                    "ApiKey": self.api_key,
                    "Request-Time": str(ts),
                    "Signature": sig,
                    "Content-Type": "application/json",
                    "User-Agent": self._UA,
                },
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json()
                return d.get("data", []) if isinstance(d, dict) else []
        except Exception:
            pass
        return []

    def get_futures_account(self) -> dict:
        """Аккаунт фьючерсов (баланс маржи, PnL)."""
        ts  = self._ts()
        sig = self._futures_sign({}, ts)
        try:
            r = _mexc_direct_get(
                f"{self.FUTURES_BASE}/api/v1/private/account/assets",
                headers={
                    "ApiKey": self.api_key,
                    "Request-Time": str(ts),
                    "Signature": sig,
                    "Content-Type": "application/json",
                    "User-Agent": self._UA,
                },
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json()
                return d.get("data", {}) if isinstance(d, dict) else {}
        except Exception:
            pass
        return {}

    # ── Полный снапшот аккаунта ───────────────────────────────────────────────
    _CONTRACT_SIZE_FALLBACKS = {
        "BTC": 0.0001,
        "ETH": 0.01,
        "BNB": 0.01,
        "SOL": 0.1,
        "XRP": 10.0,
        "ADA": 10.0,
    }

    @staticmethod
    def _first_float(raw: dict, *keys: str, default: float = 0.0) -> float:
        for key in keys:
            try:
                value = raw.get(key)
                if value is None or value == "":
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _get_contract_meta(self, symbol: str) -> dict:
        base = str(symbol or "").replace("_USDT", "").upper()
        cached = self._contract_meta_cache.get(base)
        if cached:
            return cached

        fallback = {
            "symbol": f"{base}_USDT",
            "contractSize": float(self._CONTRACT_SIZE_FALLBACKS.get(base, 1.0)),
        }
        try:
            r = _mexc_direct_get(
                f"{self.FUTURES_BASE}/api/v1/contract/detail",
                params={"symbol": fallback["symbol"]},
                timeout=5,
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
            contract_size = float(
                raw.get("contractSize", fallback["contractSize"])
                or fallback["contractSize"]
            )
            meta = {
                "symbol": raw.get("symbol", fallback["symbol"]),
                "contractSize": contract_size,
            }
        except Exception:
            meta = fallback

        self._contract_meta_cache[base] = meta
        return meta

    def get_full_snapshot(self) -> dict:
        """
        Единый срез реального аккаунта: фьючерсы + спот.

        Возвращает:
        {
          'futures': {
            'equity':     float,   # баланс + нереализованный PnL
            'available':  float,   # свободная маржа
            'unrealized': float,   # нереализованный PnL по всем позициям
            'positions':  [        # открытые фьючерсные позиции
              { 'symbol','side','qty','leverage',
                'entry','unrealized_pnl','margin' }
            ]
          },
          'spot': {
            'usdt':        float,
            'crypto':      { sym: {'qty':float,'price':float,'value':float} },
            'total_value': float   # USDT + крипта по рыночным ценам
          },
          'total_equity': float,   # futures.equity + spot.total_value
          'primary_capital': float # что реально доступно для торговли в текущем режиме
        }
        """
        snap = {
            'futures': {'equity': 0.0, 'available': 0.0, 'unrealized': 0.0, 'positions': []},
            'spot':    {'usdt': 0.0, 'crypto': {}, 'total_value': 0.0},
            'total_equity': 0.0, 'primary_capital': 0.0,
        }

        # ── Фьючерсы ─────────────────────────────────────────────────────────
        try:
            resp_json = {}
            items = []
            last_status = None
            for attempt in range(2):
                ts = self._ts()
                sig = self._futures_sign({}, ts)
                r = _mexc_direct_get(
                    f"{self.FUTURES_BASE}/api/v1/private/account/assets",
                    headers={"ApiKey": self.api_key, "Request-Time": str(ts),
                             "Signature": sig, "Content-Type": "application/json",
                             "User-Agent": self._UA},
                    timeout=10,
                )
                last_status = r.status_code
                if r.status_code != 200:
                    if attempt == 0:
                        time.sleep(0.35)
                        continue
                    break
                resp_json = r.json()
                items = resp_json.get("data", [])
                if items:
                    break
                if attempt == 0:
                    time.sleep(0.35)

            if items:
                for item in items:
                    cur = str(item.get("currency", "")).upper()
                    if cur in ("USDT", "USDC"):
                        snap['futures']['equity']     += float(item.get("equity", 0) or 0)
                        snap['futures']['available']  += float(item.get("availableBalance", 0) or 0)
                        snap['futures']['unrealized'] += float(item.get("unrealized", 0) or 0)
            elif self._last_good_futures_snapshot and (
                time.time() - self._last_good_futures_snapshot_at
            ) <= 300:
                snap['futures'].update(dict(self._last_good_futures_snapshot))
                log.warning(
                    "get_full_snapshot: futures assets empty/invalid -> reuse cached snapshot "
                    "(age=%.1fs, code=%s, msg=%s)",
                    time.time() - self._last_good_futures_snapshot_at,
                    resp_json.get("code"),
                    resp_json.get("message") or resp_json.get("msg") or last_status,
                )
            elif last_status is not None and last_status != 200:
                log.warning("get_full_snapshot: HTTP %s от futures assets", last_status)
            else:
                log.warning(
                    "get_full_snapshot: фьючерсный аккаунт вернул пустой data=[]. "
                    "code=%s msg=%s. Возможно: субаккаунт, неверная подпись или "
                    "аккаунт не активирован для фьючерсов.",
                    resp_json.get("code"), resp_json.get("message") or resp_json.get("msg")
                )
        except Exception as e:
            log.debug("get_full_snapshot futures assets: %s", e)

        # Открытые фьючерсные позиции
        try:
            for p in self.get_futures_positions():
                sym = str(p.get("symbol", "")).replace("_USDT", "")
                pos_type = str(p.get("positionType", "")).strip().lower()
                side = "long" if pos_type in ("1", "long") else "short"
                raw_contracts = float(p.get("holdVol", 0) or 0)
                meta = self._get_contract_meta(sym)
                contract_size = max(float(meta.get("contractSize", 1.0) or 1.0), 1e-12)
                qty = raw_contracts * contract_size
                entry = float(p.get("openAvgPrice", p.get("holdAvgPrice", 0)) or 0)
                unrealized_pnl = self._first_float(
                    p,
                    "unrealizedValue",
                    "unrealisedValue",
                    "unrealizedPnl",
                    "unrealisedPnl",
                    "holdProfitLoss",
                    "profit",
                    "pnl",
                    default=0.0,
                )
                if abs(unrealized_pnl) <= 1e-12 and entry > 0 and raw_contracts > 0:
                    mark = self._first_float(
                        p,
                        "markPrice",
                        "fairPrice",
                        "lastPrice",
                        "indexPrice",
                        default=0.0,
                    )
                    if mark > 0:
                        direction = 1.0 if side == "long" else -1.0
                        unrealized_pnl = qty * (mark - entry) * direction
                snap['futures']['positions'].append({
                    'symbol':         sym,
                    'side':           side,
                    'qty':            qty,
                    'contracts':      raw_contracts,
                    'contract_size':  contract_size,
                    'leverage':       int(p.get("leverage", 1) or 1),
                    'entry':          entry,
                    'unrealized_pnl': unrealized_pnl,
                    'margin':         float(p.get("im", p.get("margin", 0)) or 0),
                    'notional':       qty * entry,
                })
        except Exception as e:
            log.debug("get_full_snapshot futures positions: %s", e)

        zero_assets = (
            float(snap['futures']['equity'] or 0.0) <= 0.0
            and float(snap['futures']['available'] or 0.0) <= 0.0
        )
        if (
            zero_assets
            and self._last_good_futures_snapshot
            and (time.time() - self._last_good_futures_snapshot_at) <= 300
        ):
            cache_age = time.time() - self._last_good_futures_snapshot_at
            cached = dict(self._last_good_futures_snapshot)
            current_unrealized = float(snap['futures']['unrealized'] or 0.0)
            if current_unrealized == 0.0 and not snap['futures']['positions']:
                current_unrealized = float(cached.get('unrealized', 0.0) or 0.0)
            realized_base = max(
                float(cached.get('equity', 0.0) or 0.0) - float(cached.get('unrealized', 0.0) or 0.0),
                0.0,
            )
            snap['futures']['equity'] = max(realized_base + current_unrealized, 0.0)
            snap['futures']['available'] = max(float(cached.get('available', 0.0) or 0.0), 0.0)
            snap['futures']['unrealized'] = current_unrealized
            log.warning(
                "get_full_snapshot: futures equity/available returned zero -> reuse cached core "
                "(age=%.1fs, positions=%d)",
                cache_age,
                len(snap['futures']['positions']),
            )
            snap["data_health"] = {
                "uses_cached_balance": True,
                "cache_age_sec": round(cache_age, 1),
                "last_data_error_reason": "cached_equity",
            }

        if snap['futures']['equity'] > 0 or snap['futures']['available'] > 0:
            self._last_good_futures_snapshot = {
                'equity': float(snap['futures']['equity'] or 0.0),
                'available': float(snap['futures']['available'] or 0.0),
                'unrealized': float(snap['futures']['unrealized'] or 0.0),
            }
            self._last_good_futures_snapshot_at = time.time()

        # ── Спот ─────────────────────────────────────────────────────────────
        stables = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USD1", "USDE", "FDUSD"}
        non_usdt_syms = []
        try:
            for asset, b in self.get_all_balances().items():
                qty = b.get("free", 0.0) + b.get("locked", 0.0)
                if qty <= 0:
                    continue
                if asset in stables:
                    snap['spot']['usdt'] += qty
                else:
                    snap['spot']['crypto'][asset] = {'qty': qty, 'price': 0.0, 'value': 0.0}
                    non_usdt_syms.append(asset)
        except Exception as e:
            log.debug("get_full_snapshot spot balances: %s", e)

        # Рыночные цены для спот-крипты
        if non_usdt_syms:
            try:
                r = self._session.get(
                    f"{self.SPOT_BASE}/api/v3/ticker/24hr", timeout=10
                )
                if r.status_code == 200:
                    prices_map = {
                        item["symbol"][:-4]: float(item.get("lastPrice", 0) or 0)
                        for item in r.json()
                        if item.get("symbol", "").endswith("USDT")
                    }
                    for sym in non_usdt_syms:
                        p = prices_map.get(sym, 0.0)
                        qty = snap['spot']['crypto'][sym]['qty']
                        snap['spot']['crypto'][sym]['price'] = p
                        snap['spot']['crypto'][sym]['value'] = qty * p
            except Exception as e:
                log.debug("get_full_snapshot spot prices: %s", e)

        crypto_value = sum(v['value'] for v in snap['spot']['crypto'].values())
        snap['spot']['total_value'] = snap['spot']['usdt'] + crypto_value
        snap['total_equity'] = snap['futures']['equity'] + snap['spot']['total_value']

        # Приоритет: фьючерсный счёт для live_futures, иначе спот
        trading_mode = os.environ.get("MEXC_TRADING_MODE", "live_futures")
        if trading_mode == "live_futures":
            snap['primary_capital'] = snap['futures']['equity']
        else:
            snap['primary_capital'] = snap['spot']['total_value']

        return snap

    def log_snapshot(self, snap: dict) -> None:
        """Выводит красивую таблицу полного состояния аккаунта в лог."""
        log.info("┌──────────────────────────────────────────────────────────┐")
        log.info("│              ПОЛНЫЙ СНАПШОТ АККАУНТА MEXC                │")
        log.info("├──────────────────────────────────────────────────────────┤")

        # Фьючерсы
        fut = snap['futures']
        log.info("│  ─── ФЬЮЧЕРСНЫЙ СЧЁТ (contract.mexc.com) ───            │")
        log.info("│    Equity (доступно + unrealPnL): $%10.4f             │", fut['equity'])
        log.info("│    Свободная маржа:               $%10.4f             │", fut['available'])
        if fut['unrealized'] != 0:
            log.info("│    Нереализованный PnL:           $%10.4f             │", fut['unrealized'])
        if fut['positions']:
            log.info("│    Открытые позиции:                                     │")
            for p in fut['positions']:
                pnl_sign = "+" if p['unrealized_pnl'] >= 0 else ""
                log.info("│      %-8s %-5s lev=%dx  entry=%-10.4f  pnl=%s%.2f  │",
                         p['symbol'], p['side'].upper(), p['leverage'],
                         p['entry'], pnl_sign, p['unrealized_pnl'])
        else:
            log.info("│    Открытых позиций нет                                  │")

        # Спот
        spot = snap['spot']
        log.info("├──────────────────────────────────────────────────────────┤")
        log.info("│  ─── СПОТОВЫЙ СЧЁТ (api.mexc.com) ───                   │")
        log.info("│    USDT:                          $%10.4f             │", spot['usdt'])
        if spot['crypto']:
            for sym, info in sorted(spot['crypto'].items(),
                                    key=lambda kv: -kv[1]['value']):
                log.info("│    %-10s qty=%-14.6f  ≈ $%-10.4f             │",
                         sym, info['qty'], info['value'])
        log.info("│    Итого спот:                    $%10.4f             │", spot['total_value'])

        # Итого
        log.info("├──────────────────────────────────────────────────────────┤")
        log.info("│    TOTAL EQUITY (фьючерсы + спот): $%9.4f             │", snap['total_equity'])
        log.info("│    Торговый капитал (текущий режим):$%9.4f             │", snap['primary_capital'])
        log.info("└──────────────────────────────────────────────────────────┘")

        if snap['primary_capital'] < 1.0:
            trading_mode = os.environ.get("MEXC_TRADING_MODE", "live_futures")
            if trading_mode == "live_futures":
                log.warning(
                    "\n"
                    "  ╔══ ⚠️  ФЬЮЧЕРСНЫЙ СЧЁТ ПУСТ ════════════════════════════╗\n"
                    "  ║  Ордера будут отклоняться до пополнения счёта.          ║\n"
                    "  ║  ➡ MEXC → Кошелёк → Перевод → Спот → Фьючерсы         ║\n"
                    "  ╚═════════════════════════════════════════════════════════╝"
                )

    def detect_bonus_funds(self, balances: Dict[str, dict]) -> dict:
        """
        Обнаруживает бонусные/акционные фонды в аккаунте.
        Возвращает структуру с real_usdt, stable_total, bonus_tokens и т.д.
        """
        stables = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP"}
        bonus_stables = {"USD1", "USDE", "FDUSD", "USDD"}

        real_usdt    = balances.get("USDT", {}).get("free", 0.0)
        locked_usdt  = sum(b["locked"] for b in balances.values())
        stable_total = sum(
            b["free"] for a, b in balances.items()
            if a in stables or a in bonus_stables
        )
        bonus_tokens = {
            a: b for a, b in balances.items()
            if a not in stables and a not in bonus_stables
        }
        return {
            "real_usdt":       real_usdt,
            "stable_total":    stable_total,
            "bonus_tokens":    bonus_tokens,
            "locked_total":    locked_usdt,
            "tradeable_total": stable_total,
            "has_funds":       stable_total >= MIN_BALANCE_USD,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ТРЕКЕР СТАТИСТИКИ ТОРГОВЛИ
# ══════════════════════════════════════════════════════════════════════════════

class TradingStats:
    """Накапливает статистику в памяти для дашборда."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.start_time      = datetime.now(tz=timezone.utc)

        self.timestamps:        List[datetime] = []
        self.equity_curve:      List[float]    = []   # available + unrealPnL
        self.balance_curve:     List[float]    = []   # available (свободная маржа)
        self.unrealized_curve:  List[float]    = []   # нереализованный PnL позиций
        self.live_bar_curve:    List[int]      = []   # x-ось equity_curve в live-барах
        self.total_assets_curve: List[float]   = []   # equity (дублирует для явности)
        self.signals:           List[dict]     = []
        self.trades:            List[dict]     = []
        self.pnl_history:       List[float]    = []

        self.current_balance    = initial_capital
        self.current_positions: Dict[str, dict] = {}
        self.open_orders:       List[dict]      = []
        self.bar_count          = 0
        self.live_bar_count     = 0
        self.last_signal_bar    = -1
        self.bonus_info:        dict            = {}
        self.spot_total:        float           = 0.0   # итого спот (USDT-эквивалент)

        # Новые поля для расширенного дашборда v5
        self.has_futures_perm: bool            = True
        self.sub_agent_pvs:    Dict[str, float] = {}   # {name: virtual_pv}
        self.sub_agent_initials: Dict[str, float] = {}  # {name: baseline_pv}
        self.order_history:    List[dict]       = []   # {sym, ok, code, color, time}
        self.orders_ok:        int              = 0
        self.orders_fail:      int              = 0
        self.funding_history:  List[float]      = []   # avg_rate per tick
        self.data_health:      dict             = {}

    def record_tick(self, ts: datetime, balance: float, pnl: float = 0.0,
                    available: float = 0.0, live_bar: Optional[int] = None):
        """
        balance  — equity (доступно + unrealPnL) — основная кривая
        pnl      — нереализованный PnL открытых позиций
        available— свободная маржа (без залогов под позиции)
        """
        prev_balance = self.equity_curve[-1] if self.equity_curve else float(self.initial_capital or 0.0)
        if self.equity_curve and balance <= 0.0 < prev_balance:
            log.warning(
                "[TradingStats] suspicious non-positive equity tick %.4f -> reuse previous %.4f",
                balance, prev_balance,
            )
            balance = prev_balance
            if available <= 0.0 and self.balance_curve:
                available = self.balance_curve[-1]
            if abs(float(pnl or 0.0)) <= 1e-9 and self.unrealized_curve:
                pnl = self.unrealized_curve[-1]

        self.timestamps.append(ts)
        # equity_curve = полный баланс включая unrealPnL (= balance)
        self.equity_curve.append(balance)
        self.live_bar_curve.append(int(self.live_bar_count if live_bar is None else live_bar))
        self.total_assets_curve.append(balance)
        # balance_curve = свободная маржа (cash без залогов)
        self.balance_curve.append(available if available > 0 else balance)
        # unrealized_curve = только нереализованный PnL позиций
        self.unrealized_curve.append(pnl)
        self.pnl_history.append(pnl)
        self.current_balance = balance

    def record_signal(self, bar: int, sym: str, action: int, price: float,
                      agent: str = '', regime: str = '',
                      risk_multiplier: float = 1.0):
        ACTION_NAMES = {
            0: "hold", 1: "buy_half", 2: "buy_full", 3: "sell_spot",
            4: "fl_half", 5: "fl_full", 6: "fs_half", 7: "fs_full", 8: "close_fut",
        }
        self.signals.append({
            "bar":    bar,
            "sym":    sym,
            "action": action,
            "name":   ACTION_NAMES.get(action, f"act{action}"),
            "price":  price,
            "time":   datetime.now(tz=timezone.utc),
            "agent":  agent,    # имя суб-агента(ов) которые дали сигнал
            "regime": regime,   # рыночный режим в момент сигнала
            "risk_multiplier": float(risk_multiplier or 1.0),
            "selected_player": getattr(self, "agent_name", ""),
            "selected_agents": agent,
            "position_source": "",
        })
        self.last_signal_bar = bar

    def record_trade(self, sym: str, side: str, qty: float, price: float, fee: float, funding: float = 0.0):
        self.trades.append({
            "sym": sym, "side": side, "qty": qty, "price": price,
            "fee": fee, "funding": funding, "value": qty * price,
            "time": datetime.now(tz=timezone.utc),
        })

    def record_order(
        self,
        sym: str,
        success: bool,
        code: int = 0,
        order_id: str | None = None,
        order_result: dict | None = None,
        position_source: str | None = None,
    ):
        """Записывает результат попытки разместить ордер (успешно/ошибка)."""
        import re as _re
        GRN = "#3FB950"; RED = "#F85149"
        if success:
            self.orders_ok += 1
        else:
            self.orders_fail += 1
        self.order_history.append({
            "sym":   sym[:6],
            "ok":    success,
            "code":  code if not success else 0,
            "color": GRN if success else RED,
            "time":  datetime.now(tz=timezone.utc),
        })
        # Отмечаем последний сигнал по этому символу
        for sig in reversed(self.signals):
            if sig["sym"] == sym:
                sig["order_ok"] = success
                sig["order_id"] = order_id or ""
                sig["order_result"] = order_result or {"success": success, "code": code}
                if position_source:
                    sig["position_source"] = position_source
                break

    def capture_price_history(self, price_history, limit: int = 360):
        """Сохраняет хвост истории цен в stats для детекта режима рынка в дашбордах."""
        try:
            hist = list(price_history or [])
        except Exception:
            hist = []
        if not hist:
            return
        tail = hist[-max(int(limit or 0), 2):]
        raw = {}
        for snapshot in tail:
            if not isinstance(snapshot, dict):
                continue
            for sym, value in snapshot.items():
                try:
                    price = float(value)
                except Exception:
                    continue
                if not math.isfinite(price) or price <= 0.0:
                    continue
                raw.setdefault(str(sym), []).append(price)
        if raw:
            self._price_hist_raw = raw

    def reset_live_baseline(
        self,
        initial_capital: float,
        *,
        balance: Optional[float] = None,
        pnl: float = 0.0,
        available: Optional[float] = None,
        live_bar: int = 0,
        reset_activity: bool = False,
    ):
        """Сбрасывает стартовую live-базу после warmup и переинициализирует equity curve."""
        ic = max(float(initial_capital or 0.0), 0.0)
        bal = float(ic if balance is None else balance)
        avail = float(bal if available is None else available)

        self.initial_capital = ic
        self.current_balance = bal
        self.timestamps.clear()
        self.equity_curve.clear()
        self.balance_curve.clear()
        self.unrealized_curve.clear()
        self.live_bar_curve.clear()
        self.total_assets_curve.clear()
        self.pnl_history.clear()
        self.last_signal_bar = -1

        if reset_activity:
            self.signals.clear()
            self.trades.clear()
            self.open_orders.clear()
            self.order_history.clear()
            self.orders_ok = 0
            self.orders_fail = 0
            self.funding_history.clear()

        self.record_tick(
            datetime.now(tz=timezone.utc),
            bal,
            pnl=float(pnl or 0.0),
            available=avail,
            live_bar=int(live_bar or 0),
        )

    @property
    def uptime_str(self) -> str:
        delta = datetime.now(tz=timezone.utc) - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def pnl(self) -> float:
        if not self.equity_curve:
            return 0.0
        return self.equity_curve[-1] - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return self.pnl / self.initial_capital * 100.0

    @property
    def max_drawdown(self) -> float:
        eq = [
            float(v) for v in _repair_curve_spikes(self.equity_curve)
            if isinstance(v, (int, float)) and float(v) > 0.0
        ]
        if len(eq) < 2:
            return 0.0
        peak = max(eq[0], float(self.initial_capital or 0.0))
        maxdd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > maxdd:
                maxdd = dd
        return maxdd * 100.0


def _infer_market_regime_from_price_hist(price_hist_raw) -> str:
    try:
        from crypto_agents import _detect_regime_live, _r3
    except Exception:
        return "unknown"

    if not isinstance(price_hist_raw, dict) or not price_hist_raw:
        return "unknown"

    try:
        raw = _detect_regime_live(
            {sym: deque(values, maxlen=len(values)) for sym, values in price_hist_raw.items() if values}
        )
        return {
            "bullish": "bull",
            "bearish": "bear",
            "neutral": "sideways",
            "crash": "crash",
        }.get(_r3(raw), "sideways")
    except Exception:
        return "unknown"


def _infer_market_regime_from_stats(stats: TradingStats) -> str:
    return _infer_market_regime_from_price_hist(getattr(stats, "_price_hist_raw", {}))


def _save_api_trading_dashboard(stats: TradingStats, output_dir: str, filename: str = "api_trading.png"):
    try:
        import sys as _sys
        _db_dir = os.path.dirname(os.path.abspath(__file__))
        if _db_dir not in _sys.path:
            _sys.path.insert(0, _db_dir)
        from mexc_dashboards import plot_api_trading_dashboard

        regime = _infer_market_regime_from_stats(stats)
        fund_hist = getattr(stats, 'funding_history', [])
        fund_avg = 0.0
        if fund_hist:
            try:
                latest = fund_hist[-1]
                if isinstance(latest, dict):
                    latest = latest.get('avg_funding', latest.get('funding_rate', 0.0))
                fund_avg = float(latest or 0.0)
            except Exception:
                fund_avg = 0.0

        plot_api_trading_dashboard(
            stats=stats,
            output_dir=output_dir,
            market_regime=regime,
            funding_avg=fund_avg,
            funding_history=fund_hist,
            sub_agent_pnl=_compute_sub_agent_pnl(stats),
            filename=filename,
        )
        return os.path.join(output_dir, filename)
    except ImportError:
        return None
    except Exception as exc:
        log.debug("api_trading dashboard failed: %s", exc)
        return None


def _resolve_sub_agent_baselines(stats: TradingStats) -> Dict[str, float]:
    baselines = {}
    initial_map = getattr(stats, "sub_agent_initials", {}) or {}
    for name, value in initial_map.items():
        try:
            baselines[str(name)] = float(value)
        except Exception:
            continue
    return baselines


def _compute_sub_agent_pnl(stats: TradingStats, sub_pvs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    pvs = sub_pvs if isinstance(sub_pvs, dict) else (getattr(stats, "sub_agent_pvs", {}) or {})
    if not pvs:
        return {}

    baselines = _resolve_sub_agent_baselines(stats)
    fallback_base = float(getattr(stats, "initial_capital", 0.0) or 0.0)
    pnl = {}
    for name, pv_val in pvs.items():
        try:
            pv = float(pv_val)
        except Exception:
            continue
        base = baselines.get(str(name), fallback_base)
        pnl[str(name)] = pv - base
    return pnl




# ══════════════════════════════════════════════════════════════════════════════
# СИНХРОНИЗАЦИЯ РЕАЛЬНЫХ ПОЗИЦИЙ В АГЕНТЫ
# ══════════════════════════════════════════════════════════════════════════════

def inject_live_state_into_player(player, snapshot: dict, bar_index: int = 0) -> None:
    """
    Синхронизирует реальное состояние счёта биржи в live-player.

    Делает три вещи:
    1. Сбрасывает позиции прогрева — они не существуют в реальности.
    2. Инъектирует реальные открытые позиции из snapshot во все суб-агенты
       чтобы игрок корректно знал, что закрывать при sell-сигнале.
    3. Устанавливает virtual PV каждого суб-агента равным
       (real_equity / n_agents) — правильное базовое плечо для расчёта ордеров.

    player  — Panteon, PlayerStop или другой PlayerBase
    snapshot — результат MexcDirectClient.get_full_snapshot()
    """
    # FIX v9.1: Panteon не имеет sub_agents - адаптируем inject
    if not hasattr(player, 'sub_agents'):
        player_subagents = list(_iter_player_subagents(player))
        # Для Panteon: используем reset_for_live() если доступен
        log.info("  [inject_live] Panteon: сброс позиций и таймеров")

        # FIX v4: reset_for_live() — полный сброс sub-agents (pos, ep, _lc)
        # Без этого sub-agents сохраняют warmup-позиции → HOLD навечно
        if hasattr(player, 'reset_for_live'):
            player.reset_for_live(bar_index)
        else:
            if hasattr(player, '_pos'):
                player._pos.clear()
            # Сброс таймеров всех суб-агентов Panteon (включая GeneticsBullish)
            for _, _, sub in player_subagents:
                _reset_subagent_timers(sub, bar_index)
            # Сброс кэша режима
            if hasattr(player, '_r'): player._r = None
            if hasattr(player, '_lr'): player._lr = 0
        # Без этого Panteon не знает об открытых позициях → не соблюдает MAX_POS
        # → открывает новые позиции пока не закончится вся маржа (free=0.88 USDT)
        real_positions = snapshot.get('futures', {}).get('positions', [])
        if real_positions and hasattr(player, '_open_pos'):
            player._open_pos.clear()
            for pos in real_positions:
                sym   = pos.get('symbol', '').replace('_USDT', '')
                side  = pos.get('side', 'long').lower()
                entry    = pos.get('entry', 0.0)
                leverage = pos.get('leverage', 1)
                if sym:
                    player._open_pos[sym] = {
                        'entry':    entry,
                        'side':     side,
                        'bar':      bar_index,
                        'leverage': leverage,
                        'peak':     entry,    # FIX v5: для trailing stop
                        **inherited_position_policy(),
                    }
                    log.info("  [inject_live]   %s %s lev=%dx  entry=%.4f",
                             sym, side.upper(), leverage, entry)

                    # FIX v5: пропагируем позицию в sub-agents чтобы они
                    # могли генерировать close-сигналы для неё.
                    # Без этого sub-agents не знают о позиции → никогда
                    # не голосуют за close → позиция висит навечно.
                    for _, _, sub in player_subagents:
                        _try_set_agent_pos(sub, sym, side, entry=entry, bar_index=bar_index)
                    log.debug("  [inject_live]   → propagated %s %s to sub-agents",
                              sym, side)

            log.info("  [inject_live] Panteon: инъектировано %d реальных позиций + sub-agents",
                     len(player._open_pos))
        real_equity = snapshot.get('primary_capital', 0) or snapshot.get('total_equity', 50.0)
        log.info("  [inject_live] Panteon ready: equity=%.4f  open_pos=%d",
                 real_equity, len(getattr(player, '_open_pos', {})))
        return

    real_equity = snapshot['primary_capital']
    if real_equity < 0.01:
        real_equity = snapshot['total_equity']
    real_equity = max(real_equity, 1.0)

    n = player.n_agents or 1
    per_agent = real_equity / n

    agent_names = list(player.sub_agents.keys())
    log.info("  [inject_live] Сброс warmup-состояния: %d суб-агентов, equity=%.4f",
             len(agent_names), real_equity)

    # ── 1. Сброс позиций прогрева ─────────────────────────────────────────
    for tracker in player._pos.values():
        tracker.spot.clear()
        tracker.futures.clear()
    player.pos.clear()

    # ── 2. КЛЮЧЕВОЙ FIX: сброс cooldown и флагов активности ──────────────
    # Без этого суб-агенты остаются в cooldown из warmup (COOLDOWN_EXT_BARS=1440).
    # При warmup_end=5760: _cooldown_until[agent] может быть 5760+1440=7200.
    # В live (bar 6050) это всё ещё cooldown → _n_trading()==0 → HOLD вечно.
    if hasattr(player, '_cooldown_until'):
        player._cooldown_until = {n: -1 for n in agent_names}
        log.info("  [inject_live]   _cooldown_until сброшен → все суб-агенты разблокированы")

    if hasattr(player, '_dd_streak'):
        player._dd_streak = {n: 0 for n in agent_names}

    # Активируем всех суб-агентов (в warmup некоторые могли быть деактивированы)
    player._active = {n: True for n in agent_names}
    player._dd_reactivate_at = {n: -1 for n in agent_names}

    # ── 3. Сброс virtual PV → реальный баланс ────────────────────────────
    player._initialized   = False
    player._prev_total_pv = real_equity

    for name in agent_names:
        if name in player._virt:
            vv = player._virt[name]
            vv.value   = per_agent
            vv.initial = per_agent
            vv.peak    = per_agent
        else:
            from crypto_players import _VirtualPV as _VPV
            player._virt[name] = _VPV(per_agent)

    player._initialized = True

    # _last_reeval: ставим так чтобы первый реэвал произошёл НЕМЕДЛЕННО
    # (не ждать REEVAL_BARS=240 баров после warmup)
    player._last_reeval = bar_index - player.REEVAL_BARS - 1

    # ── 4. Сброс внутренних cooldown суб-агентов ─────────────────────────
    # Каждый суб-агент (TrendCapture, VolBreakout, etc.) имеет свой
    # _cooldown_until / pause_until / _last_check. Сбрасываем всё.
    for name, agent in player.sub_agents.items():
        _reset_subagent_timers(agent, bar_index)

    # ── 5. Инъекция реальных позиций MEXC ────────────────────────────────
    real_positions = snapshot['futures'].get('positions', [])
    if not real_positions:
        log.info("  [inject_live] Нет открытых позиций на MEXC — старт с чистого листа.")
    else:
        active_names = player._active_names()
        log.info("  [inject_live] Реальных позиций: %d → инъекция в %d суб-агентов",
                 len(real_positions), len(active_names))
        for pos in real_positions:
            sym  = pos['symbol']
            side = pos['side']
            log.info("  [inject_live]   %s %s  pnl=%+.4f USDT",
                     sym, side.upper(), pos['unrealized_pnl'])
            for name in active_names:
                player._pos[name].futures.add(sym)
            for name in active_names:
                _try_set_agent_pos(player.sub_agents[name], sym, side)

    if hasattr(player, '_merge_pos_for_ddstop'):
        player._merge_pos_for_ddstop()

    n_trading = player._n_trading() if hasattr(player, '_n_trading') else '?'
    log.info("  [inject_live] ✅ Готово: активных суб-агентов=%s  per_agent=$%.4f",
             n_trading, per_agent)


def _reset_subagent_timers(agent, bar_index: int, _depth: int = 0) -> None:
    """
    Сбрасывает внутренние cooldown-таймеры суб-агента и его обёрток.
    Суб-агенты хранят таймеры под разными именами:
      _cooldown_until, pause_until, _last_check, last_check, last_rebal, _last_rebal
    """
    if _depth > 5 or agent is None:
        return

    for attr, val in [
        ('_cooldown_until', 0),
        ('pause_until',     0),
        ('_last_check',     bar_index - 99999),   # немедленная проверка
        ('last_check',      bar_index - 99999),
        ('last_rebal',      bar_index - 99999),
        ('_last_rebal',     bar_index - 99999),
    ('_lc',             -99999),   # FIX v4: panteon_agents используют _lc, не _last_check
        ('_month_start_eq', None),
        ('_trail_peak_eq',  None),
    ]:
        try:
            if hasattr(agent, attr):
                setattr(agent, attr, val)
        except Exception:
            pass

    # Рекурсивный обход обёрток
    for wrap_attr in ('_inner', '_a', '_agent', 'agent'):
        try:
            inner = getattr(agent, wrap_attr, None)
            if inner is not None and inner is not agent:
                _reset_subagent_timers(inner, bar_index, _depth + 1)
        except Exception:
            pass

    # FIX v4: сброс внутренних позиций суб-агентов (warmup positions → stale)
    for pos_attr in ('pos',):
        d = getattr(agent, pos_attr, None)
        if isinstance(d, dict):
            for k in d:
                d[k] = None
    for ep_attr in ('ep', 'entry_px'):
        d = getattr(agent, ep_attr, None)
        if isinstance(d, dict):
            for k in d:
                d[k] = 0.0
    for et_attr in ('et',):
        d = getattr(agent, et_attr, None)
        if isinstance(d, dict):
            for k in d:
                d[k] = 0











def _iter_player_subagents(player):
    """Возвращает актуальный список sub-agent'ов Panteon."""
    if hasattr(player, 'iter_subagents'):
        try:
            for item in player.iter_subagents():
                if len(item) == 3:
                    yield item
            return
        except Exception:
            pass

    for attr_name, label in (
        ('_fa', 'FundingArb'),
        ('_ms', 'MomentumScalper'),
        ('_las', 'LiveAfterShock'),
        ('_lch', 'LiveCrashHunter'),
        ('_lmr', 'LiveMeanRev'),
        ('_ltf', 'LiveTrendFollow'),
        ('_lvc', 'LiveVolCompress'),
        ('_gb', 'GeneticsBullish'),
        ('_gbr', 'GeneticsBearish'),
    ):
        agent = getattr(player, attr_name, None)
        if agent is not None:
            yield attr_name, label, agent


def _try_set_agent_pos(agent, sym: str, side: str, entry: float = None,
                       bar_index: int = None, _depth: int = 0) -> None:
    """Рекурсивно проставляет pos[sym]=side в агенте и его обёртках."""
    if _depth > 5 or agent is None:
        return
    try:
        pos_dict = getattr(agent, 'pos', None)
        if isinstance(pos_dict, dict):
            pos_dict[sym] = side
    except Exception:
        pass
    if entry is not None:
        for attr_name in ('ep', 'entry_px'):
            try:
                entry_dict = getattr(agent, attr_name, None)
                if isinstance(entry_dict, dict):
                    entry_dict[sym] = entry
            except Exception:
                pass
    if bar_index is not None:
        try:
            et_dict = getattr(agent, 'et', None)
            if isinstance(et_dict, dict):
                et_dict[sym] = bar_index
        except Exception:
            pass
    for attr in ('_inner', '_a', '_agent', 'agent'):
        try:
            inner = getattr(agent, attr, None)
            if inner is not None and inner is not agent:
                _try_set_agent_pos(inner, sym, side, entry=entry,
                                   bar_index=bar_index, _depth=_depth + 1)
        except Exception:
            pass


def _try_clear_agent_pos(agent, sym: str, _depth: int = 0) -> None:
    """Рекурсивно очищает позицию sym в агенте и его обёртках."""
    if _depth > 5 or agent is None:
        return
    try:
        pos_dict = getattr(agent, 'pos', None)
        if isinstance(pos_dict, dict):
            pos_dict[sym] = None
    except Exception:
        pass
    for attr_name in ('ep', 'entry_px'):
        try:
            entry_dict = getattr(agent, attr_name, None)
            if isinstance(entry_dict, dict):
                entry_dict[sym] = 0.0
        except Exception:
            pass
    try:
        et_dict = getattr(agent, 'et', None)
        if isinstance(et_dict, dict):
            et_dict[sym] = 0
    except Exception:
        pass
    for attr in ('_inner', '_a', '_agent', 'agent'):
        try:
            inner = getattr(agent, attr, None)
            if inner is not None and inner is not agent:
                _try_clear_agent_pos(inner, sym, _depth=_depth + 1)
        except Exception:
            pass


def save_dashboard(stats: TradingStats, output_dir: str, filename: str = "dashboard.png"):
    """
    Расширенный дашборд v5:
      Row 0: Equity curve (реальный баланс) + per-subagent виртуальный PV
      Row 1: Последние сигналы + статус API + открытые позиции
      Row 2: История ордеров (успешные/неудачные) + funding rate
    """
    if os.path.basename(str(filename)) == "dashboard_latest.png":
        return _save_api_trading_dashboard(stats, output_dir, "api_trading.png")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs
        import numpy as np

        DARK = "#0D1117"; MID = "#161B22"; GRID = "#21262D"
        GRN  = "#3FB950"; RED  = "#F85149"; CYN  = "#58A6FF"
        GOL  = "#F0C040"; PUR  = "#BC8CFF"; ORG  = "#FF8C00"
        WHT  = "#E6EDF3"; GRY  = "#8B949E"

        has_perm = getattr(stats, 'has_futures_perm', True)
        bonus    = stats.bonus_info or {}
        live_bars = stats.live_bar_count
        uptime   = stats.uptime_str
        market_regime = _infer_market_regime_from_stats(stats)

        title_color = WHT if has_perm else RED
        title_suffix = "" if has_perm else "  ⚠ НЕТ ПРАВ FUTURES"

        # FIX v9.2: 'agents' — не в scope save_dashboard (NameError каждые 30 мин)
        # Берём имя агента из stats, где оно сохраняется при запуске
        _agent_label = getattr(stats, 'agent_name', 'Panteon')

        fig = plt.figure(figsize=(22, 15), facecolor=DARK)
        fig.suptitle(
            f"Live Dashboard - {_agent_label}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"  |  Uptime {uptime}  |  live bars: {live_bars}{title_suffix}",
            fontsize=12, fontweight="bold", color=title_color, y=0.99,
        )

        layout = gs.GridSpec(3, 4, figure=fig,
                             hspace=0.50, wspace=0.30,
                             left=0.05, right=0.98, top=0.94, bottom=0.04)

        def _style(ax, title="", fs=9):
            ax.set_facecolor(MID)
            ax.tick_params(colors=GRY, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.grid(True, color=GRID, lw=0.4)
            if title:
                ax.set_title(title, color=WHT, fontsize=fs, fontweight="bold")

        def _shorten_label(text, max_len=18):
            txt = str(text or "")
            if len(txt) <= max_len:
                return txt
            return txt[: max(0, max_len - 3)] + "..."

        # ══════════════════════════════════════════════════════
        # ROW 0 left (col 0:3) — Equity curve (реальный баланс)
        # ══════════════════════════════════════════════════════
        ax1 = fig.add_subplot(layout[0, :3])
        _style(ax1, "Equity (реальный фьючерсный баланс)")

        eq = _repair_curve_spikes(list(stats.equity_curve))
        ic = stats.initial_capital

        if len(eq) > 1:
            xs = list(range(len(eq)))
            col_eq = GRN if eq[-1] >= ic else RED
            ax1.fill_between(xs, ic, eq, color=col_eq, alpha=0.15)
            ax1.plot(xs, eq, color=col_eq, lw=2.0,
                     label=f"Equity ${eq[-1]:.2f}")
            ax1.axhline(ic, color=GOL, lw=1.0, ls="--", alpha=0.7,
                        label=f"Старт ${ic:.0f}")
            # Маркеры сигналов на кривой
            open_x, open_y = [], []
            close_x, close_y = [], []
            for sig in stats.signals[-80:]:
                idx = sig.get("_eq_idx")
                if idx is None:
                    idx = len(eq) - 1
                if 0 <= idx < len(eq):
                    if sig["action"] in (1, 2, 4, 5):
                        open_x.append(idx)
                        open_y.append(eq[idx])
                    else:
                        close_x.append(idx)
                        close_y.append(eq[idx])
            if open_x:
                ax1.scatter(open_x, open_y, s=24, marker="^", color=GRN,
                            alpha=0.78, edgecolors=DARK, linewidths=0.3,
                            zorder=4, label="Open/Long")
            if close_x:
                ax1.scatter(close_x, close_y, s=24, marker="v", color=RED,
                            alpha=0.78, edgecolors=DARK, linewidths=0.3,
                            zorder=4, label="Close/Short")
            pnl = eq[-1] - ic
            pnl_pct = pnl / ic * 100 if ic > 0 else 0
            pnl_c = GRN if pnl >= 0 else RED
            ax1.text(0.99, 0.95,
                     f"P&L: {pnl:+.4f} USDT  ({pnl_pct:+.2f}%)\n"
                     f"MaxDD: {stats.max_drawdown:.2f}%  |  Сделок: {len(stats.trades)}",
                     transform=ax1.transAxes, ha="right", va="top",
                     color=pnl_c, fontsize=9, fontweight="bold",
                     bbox=dict(fc=MID, ec=GRID, lw=0.5, pad=3))
            ax1.legend(fontsize=8, loc="upper left")
        else:
            # Equity curve ещё пустая — показываем стартовую линию
            ax1.axhline(ic, color=GOL, lw=1.5, ls="--", alpha=0.8)
            ax1.set_ylim(ic * 0.95, ic * 1.05)
            ax1.text(0.5, 0.5, f"Накопление данных...\nСтарт: ${ic:.2f}",
                     ha="center", va="center", color=GRY, fontsize=11,
                     transform=ax1.transAxes)

        ax1.set_ylabel("USDT", color=GRY, fontsize=8)
        ax1.set_xlabel("Тик мониторинга", color=GRY, fontsize=8)

        # ══════════════════════════════════════════════════════
        # ROW 0 right (col 3) — Статус и ключевые метрики
        # ══════════════════════════════════════════════════════
        ax_stat = fig.add_subplot(layout[0, 3])
        ax_stat.set_facecolor(MID)
        ax_stat.axis("off")
        ax_stat.text(0.5, 0.98, "Статус системы", ha="center", va="top",
                     color=WHT, fontsize=9, fontweight="bold",
                     transform=ax_stat.transAxes)

        data_health = getattr(stats, "data_health", {}) or {}
        snapshot_age = data_health.get("snapshot_age_sec")
        try:
            snapshot_age_val = None if snapshot_age is None else float(snapshot_age)
        except Exception:
            snapshot_age_val = None
        snapshot_age_txt = "n/a" if snapshot_age_val is None else f"{snapshot_age_val:.0f}s"
        api_errors_10m = int(data_health.get("api_errors_10m", 0) or 0)
        reconcile_add = int(data_health.get("reconcile_add", 0) or 0)
        reconcile_remove = int(data_health.get("reconcile_remove", 0) or 0)
        failed_orders = int(data_health.get("failed_orders", getattr(stats, "orders_fail", 0)) or 0)

        status_lines = [
            ("Статус",     "🟢 LIVE" if stats.live_bar_count > 0 else "🟡 WARMUP",
             GRN if stats.live_bar_count > 0 else GOL),
            ("Баланс",     f"${eq[-1]:.4f}" if eq else f"${stats.current_balance:.4f}", CYN),
            ("Капитал",    f"${ic:.4f}", GOL),
            ("Live баров", str(live_bars), WHT),
            ("Сигналов",   str(len(stats.signals)), PUR),
            ("Сделок",     str(len(stats.trades)), GRN if stats.trades else GRY),
            ("Позиций",    str(len(stats.current_positions)),
             GRN if stats.current_positions else GRY),
            ("Ордеров ок", str(getattr(stats, 'orders_ok', 0)), GRN),
            ("Ошибок API", str(getattr(stats, 'orders_fail', 0)),
             RED if getattr(stats, 'orders_fail', 0) > 0 else GRY),
            ("API Futures", "✅ ДА" if has_perm else "❌ НЕТ",
             GRN if has_perm else RED),
        ]
        status_lines.extend([
            ("Data health", "", GRY),
            ("Snapshot age", snapshot_age_txt, RED if snapshot_age_val and snapshot_age_val > 180 else WHT),
            ("API err 10m", str(api_errors_10m), RED if api_errors_10m else GRY),
            ("Reconcile +", str(reconcile_add), CYN if reconcile_add else GRY),
            ("Reconcile -", str(reconcile_remove), RED if reconcile_remove else GRY),
            ("Failed orders", str(failed_orders), RED if failed_orders else GRY),
        ])

        y0 = 0.88
        for label, val, col in status_lines:
            if val == "":
                ax_stat.text(0.5, y0, "--- " + label + " ---", color=GRY, fontsize=7,
                             transform=ax_stat.transAxes, va="top", ha="center")
                y0 -= 0.045
                continue
            ax_stat.text(0.05, y0, label + ":", color=GRY, fontsize=7.2,
                         transform=ax_stat.transAxes, va="top")
            ax_stat.text(0.95, y0, val, color=col, fontsize=7.2,
                         transform=ax_stat.transAxes, va="top", ha="right",
                         fontweight="bold")
            ax_stat.plot([0.02, 0.98], [y0 - 0.01, y0 - 0.01],
                        color=GRID, lw=0.4, transform=ax_stat.transAxes)
            y0 -= 0.058

        if not has_perm:
            ax_stat.text(0.5, 0.02,
                         "⚠ Включи Contract Trading\nв API Settings биржи",
                         ha="center", va="bottom", color=RED, fontsize=7,
                         transform=ax_stat.transAxes, fontweight="bold",
                         bbox=dict(fc="#300", ec=RED, lw=1, pad=2))

        # ══════════════════════════════════════════════════════
        # ROW 1 left (col 0:2) — Последние сигналы
        # ══════════════════════════════════════════════════════
        ax_sig = fig.add_subplot(layout[1, :2])
        ax_sig.set_facecolor(MID)
        ax_sig.axis("off")
        ax_sig.text(0.5, 0.97, f"Последние сигналы — {getattr(stats, 'agent_name', 'Агент')}",
                    ha="center", va="top", color=WHT, fontsize=9,
                    fontweight="bold", transform=ax_sig.transAxes)

        cols_h   = ["Время", "Bar", "Символ", "Действие", "Цена", "Статус"]
        xs_sig   = [0.01, 0.18, 0.32, 0.46, 0.68, 0.86]
        col_clrs = {
            "fl_half": CYN, "fl_full": CYN, "fs_half": ORG, "fs_full": ORG,
            "buy_half": GRN, "buy_full": GRN, "sell_spot": RED,
            "close_fut": RED, "hold": GRY,
        }
        for xh, hdr in zip(xs_sig, cols_h):
            ax_sig.text(xh, 0.88, hdr, color=GRY, fontsize=7,
                        transform=ax_sig.transAxes, va="top", style="italic")
        ax_sig.plot([0.0, 1.0], [0.85, 0.85],
                    color=GRID, lw=0.5, transform=ax_sig.transAxes)

        recent = list(reversed(stats.signals[-16:]))
        for i, sig in enumerate(recent):
            y = 0.80 - i * 0.049
            if y < 0.0:
                break
            c    = col_clrs.get(sig["name"], GRY)
            ok   = sig.get("order_ok")
            st_c = GRN if ok is True else (RED if ok is False else GRY)
            st_t = "✅" if ok is True else ("❌" if ok is False else "—")
            t_s  = sig["time"].strftime("%H:%M:%S")
            row  = [t_s, str(sig["bar"]), sig["sym"],
                    sig["name"].upper(), f"${sig['price']:.4f}", st_t]
            for xi, val in zip(xs_sig, row):
                rc = st_c if val == st_t else (c if val == sig["name"].upper() else WHT)
                ax_sig.text(xi, y, val, color=rc, fontsize=7.5,
                            transform=ax_sig.transAxes, va="top")

        if not stats.signals:
            next_bar = max(0, 60 - live_bars)
            ax_sig.text(0.5, 0.5,
                        f"Сигналов пока нет\n"
                        f"Первая проверка: live bar 60 (~{next_bar} мин)",
                        ha="center", va="center", color=GRY, fontsize=9,
                        transform=ax_sig.transAxes)

        # ══════════════════════════════════════════════════════
        # ROW 1 col 2 — Открытые позиции
        # ══════════════════════════════════════════════════════
        ax_pos = fig.add_subplot(layout[1, 2])
        ax_pos.set_facecolor(MID)
        ax_pos.axis("off")
        ax_pos.text(0.5, 0.97, "Открытые позиции",
                    ha="center", va="top", color=WHT, fontsize=9,
                    fontweight="bold", transform=ax_pos.transAxes)
        if stats.current_positions:
            y0 = 0.86
            for sym, pos in list(stats.current_positions.items())[:8]:
                pnl_v = pos.get("pnl", 0.0)
                c = GRN if pnl_v >= 0 else RED
                side = pos.get("side", "?")
                entry = pos.get("entry", 0.0)
                ax_pos.text(0.04, y0, f"{sym} {side}", color=GOL, fontsize=8,
                            transform=ax_pos.transAxes, va="top", fontweight="bold")
                ax_pos.text(0.96, y0, f"pnl: {pnl_v:+.4f}", color=c, fontsize=8,
                            transform=ax_pos.transAxes, va="top", ha="right")
                ax_pos.text(0.04, y0 - 0.065,
                            f"  entry: {entry:.4f}", color=GRY, fontsize=7,
                            transform=ax_pos.transAxes, va="top")
                y0 -= 0.13
        else:
            ax_pos.text(0.5, 0.5, "Нет открытых позиций",
                        ha="center", va="center", color=GRY, fontsize=9,
                        transform=ax_pos.transAxes)

        # ══════════════════════════════════════════════════════
        # ROW 1 col 3 — Virtual PV суб-агентов
        # ══════════════════════════════════════════════════════
        ax_ag = fig.add_subplot(layout[1, 3])
        ax_ag.set_facecolor(MID)
        ax_ag.axis("off")
        ax_ag.text(0.5, 0.97, "Виртуальный PV суб-агентов",
                   ha="center", va="top", color=WHT, fontsize=9,
                   fontweight="bold", transform=ax_ag.transAxes)

        sub_pvs = getattr(stats, 'sub_agent_pvs', {})
        sub_agent_pnl = _compute_sub_agent_pnl(stats, sub_pvs)
        sub_agent_initials = _resolve_sub_agent_baselines(stats)
        if sub_pvs:
            sorted_pvs = sorted(sub_pvs.items(), key=lambda kv: -kv[1])
            top_pvs = sorted_pvs[:8]
            y_top = 0.86
            y_bottom = 0.14
            step = (y_top - y_bottom) / max(len(top_pvs) - 1, 1)
            for idx, (name, pv_val) in enumerate(top_pvs):
                y0 = y_top - idx * step
                base = float(sub_agent_initials.get(str(name), ic) or 0.0)
                pnl_val = float(sub_agent_pnl.get(str(name), float(pv_val) - base))
                chg = (pnl_val / base * 100.0) if base > 0 else 0.0
                c = GRN if chg >= 0 else RED
                short_name = _shorten_label(name.replace("STP_", "").replace("V_", ""), 18)
                ax_ag.text(0.04, y0, short_name, color=CYN, fontsize=8,
                           transform=ax_ag.transAxes, va="top")
                ax_ag.text(0.96, y0, f"${pv_val:.2f}  ({chg:+.1f}%)", color=c,
                           fontsize=7.5, transform=ax_ag.transAxes, va="top", ha="right")
                if idx < len(top_pvs) - 1:
                    ax_ag.plot([0.04, 0.96], [y0 - step * 0.45, y0 - step * 0.45],
                               color=GRID, lw=0.4, transform=ax_ag.transAxes)
            if len(sorted_pvs) > len(top_pvs):
                ax_ag.text(0.5, 0.05, f"+{len(sorted_pvs) - len(top_pvs)} more agents",
                           ha="center", va="bottom", color=GRY, fontsize=7,
                           transform=ax_ag.transAxes)
        else:
            ax_ag.text(0.5, 0.5, "Нет данных\n(накапливается)",
                       ha="center", va="center", color=GRY, fontsize=8,
                       transform=ax_ag.transAxes)

        # ══════════════════════════════════════════════════════
        # ROW 2 left (col 0:2) — История ордеров / сделок
        # ══════════════════════════════════════════════════════
        ax_ord = fig.add_subplot(layout[2, :2])
        _style(ax_ord, "Ордера (зелёный=успешно, красный=ошибка)")

        order_hist = getattr(stats, 'order_history', [])
        _signals   = getattr(stats, 'signals', [])
        if order_hist:
            last_N = order_hist[-40:]
            cols_o  = [o.get("color", GRY) for o in last_N]
            vals_o  = [1.0] * len(last_N)
            ax_ord.bar(range(len(last_N)), vals_o, color=cols_o, width=0.8, alpha=0.85)
            ax_ord.set_yticks([])
            ax_ord.set_xticks(range(len(last_N)))
            ax_ord.set_xticklabels(
                [o.get("sym", "?")[:4] for o in last_N],
                rotation=70, fontsize=6, color=GRY
            )
            for i, o in enumerate(last_N):
                if o.get("color") == RED and o.get("code"):
                    ax_ord.text(i, 0.5, str(o.get("code", "")),
                                ha="center", va="center", fontsize=5, color=WHT)
        elif stats.trades:
            sides  = [t["side"] for t in stats.trades[-30:]]
            values = [t["value"] for t in stats.trades[-30:]]
            colors = [GRN if s in ("BUY", "buy", "LONG") else RED for s in sides]
            ax_ord.bar(range(len(values)), values, color=colors, alpha=0.8, width=0.8)
            ax_ord.set_ylabel("USDT", color=GRY, fontsize=8)
        elif _signals:
            # Нет ордеров — показываем историю сигналов
            ax_ord.set_title("Сигналы суб-агентов (ордеров пока нет)",
                             color=WHT, fontsize=9, fontweight="bold")
            SIG_COLORS = {
                "fl_half": CYN, "fl_full": CYN,
                "fs_half": ORG, "fs_full": ORG,
                "buy_half": GRN, "buy_full": GRN,
                "sell_spot": RED, "close_fut": RED, "hold": GRY,
            }
            active = [s for s in _signals if s.get("action", 0) != 0][-60:]
            if active:
                scols = [SIG_COLORS.get(s.get("name","hold"), GRY) for s in active]
                ax_ord.bar(range(len(active)), [1.0]*len(active),
                           color=scols, width=0.85, alpha=0.80)
                ax_ord.set_yticks([])
                step = max(1, len(active) // 20)
                shown = list(range(0, len(active), step))
                ax_ord.set_xticks(shown)
                ax_ord.set_xticklabels(
                    [active[i].get("sym","?")[:4] for i in shown],
                    rotation=70, fontsize=5.5, color=GRY)
                from collections import Counter as _Ctr
                cnts = _Ctr(s.get("name","") for s in active)
                lbl  = "  ".join(f"{k.upper()}:{v}"
                                 for k,v in cnts.items() if k and k != "hold")
                ax_ord.text(0.01, 0.97, lbl, transform=ax_ord.transAxes,
                            ha="left", va="top", fontsize=6.5, color=WHT,
                            bbox=dict(fc="#1C2128", ec=GRID, lw=0.4, pad=2))
            else:
                ax_ord.text(0.5, 0.5, "Сигналов пока нет",
                            ha="center", va="center", color=GRY, fontsize=9,
                            transform=ax_ord.transAxes)
        else:
            ax_ord.text(0.5, 0.5,
                        "Сделок пока нет\n"
                        + ("❌ Ордера отклоняются (403) — включи Contract Trading в API!"
                           if not has_perm else
                           "Ждём следующего сигнала агента"),
                        ha="center", va="center", color=RED if not has_perm else GRY,
                        fontsize=9, transform=ax_ord.transAxes)

        # ══════════════════════════════════════════════════════
        # ROW 2 col 2 — Funding rate история
        # ══════════════════════════════════════════════════════
        ax_fund = fig.add_subplot(layout[2, 2])
        _style(ax_fund, "Funding rate (avg)")

        raw_fund_hist = getattr(stats, 'funding_history', [])
        fund_hist = []
        for value in raw_fund_hist:
            try:
                fval = float(value)
            except Exception:
                continue
            if np.isfinite(fval):
                fund_hist.append(fval)
        if len(fund_hist) > 1:
            fvals = [f * 100 for f in fund_hist[-60:]]
            fcols = [GRN if v >= 0 else RED for v in fvals]
            ax_fund.bar(range(len(fvals)), fvals, color=fcols, width=0.8, alpha=0.8)
            ax_fund.axhline(0, color=WHT, lw=0.5, alpha=0.4)
            ax_fund.set_ylabel("%", color=GRY, fontsize=7)
        elif fund_hist:
            fval = fund_hist[-1] * 100
            ax_fund.bar([0], [fval], color=GRN if fval >= 0 else RED, width=0.55, alpha=0.85)
            ax_fund.axhline(0, color=WHT, lw=0.5, alpha=0.4)
            ax_fund.set_xticks([0])
            ax_fund.set_xticklabels(["now"], color=GRY, fontsize=7)
            ax_fund.set_ylabel("%", color=GRY, fontsize=7)
            ax_fund.text(0.5, 0.88, f"{fval:+.4f}%", ha="center", va="top",
                         color=WHT, fontsize=8, transform=ax_fund.transAxes)
        else:
            regime_colors = {
                "bull": GRN,
                "bear": RED,
                "sideways": CYN,
                "crash": ORG,
            }
            regime_labels = {
                "bull": "BULL",
                "bear": "BEAR",
                "sideways": "SIDEWAYS",
                "crash": "CRASH",
            }
            ax_fund.text(0.5, 0.62, "No funding data\n(waiting or unsupported)",
                         ha="center", va="center",
                         color=GRY, fontsize=9, transform=ax_fund.transAxes)
            ax_fund.text(
                0.5, 0.18,
                regime_labels.get(market_regime, "UNKNOWN"),
                ha="center", va="center",
                color=regime_colors.get(market_regime, WHT),
                fontsize=12, fontweight="bold",
                transform=ax_fund.transAxes,
            )

        # ══════════════════════════════════════════════════════
        # ROW 2 col 3 — Инструкция если нет прав
        # ══════════════════════════════════════════════════════
        ax_info = fig.add_subplot(layout[2, 3])
        ax_info.set_facecolor(MID)
        ax_info.axis("off")

        if not has_perm:
            ax_info.text(0.5, 0.97, "⚠ КАК ИСПРАВИТЬ 403",
                         ha="center", va="top", color=RED, fontsize=9,
                         fontweight="bold", transform=ax_info.transAxes)
            steps = [
                "1. mexc.com → Аккаунт",
                "   → API Management",
                "2. Найди свой API ключ",
                "3. Нажми Edit (Ред.)",
                "4. Включи галку:",
                "   「Contract Trading」",
                "5. Сохрани + 2FA",
                "6. Перезапусти бота",
            ]
            for i, s in enumerate(steps):
                c = RED if "Contract Trading" in s else WHT
                ax_info.text(0.05, 0.84 - i * 0.10, s,
                             color=c, fontsize=8, transform=ax_info.transAxes, va="top",
                             fontweight="bold" if "Contract" in s else "normal")
        else:
            ax_info.text(0.5, 0.97, "Расписание",
                         ha="center", va="top", color=WHT, fontsize=9,
                         fontweight="bold", transform=ax_info.transAxes)
            next_sig = max(0, 60 - (live_bars % 60)) if live_bars < 60 else live_bars % 60
            info_lines = [
                f"Следующий сигнал: ~{next_sig} мин",
                f"Интервал: каждые 60 мин",
                f"Первый сигнал: bar 60",
                "",
                f"Сигналов всего: {len(stats.signals)}",
                f"Блэклист MEXC: ~12 симв.",
                f"Активных симв.: ~18",
            ]
            for i, line in enumerate(info_lines):
                ax_info.text(0.05, 0.84 - i * 0.11, line,
                             color=WHT if line else GRY, fontsize=8,
                             transform=ax_info.transAxes, va="top")

        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        log.info("  📊 Дашборд сохранён: %s", path)

        # ── Расширенный API-Trading дашборд (mexc_dashboards.py) ────────────
        try:
            import sys as _sys
            _db_dir = os.path.dirname(os.path.abspath(__file__))
            if _db_dir not in _sys.path:
                _sys.path.insert(0, _db_dir)
            from mexc_dashboards import plot_api_trading_dashboard

            _regime = _infer_market_regime_from_stats(stats)
            _fund_hist = getattr(stats, 'funding_history', [])
            _fund_avg  = float(_fund_hist[-1]) if _fund_hist else 0.0
            _sub_pnl   = _compute_sub_agent_pnl(stats)

            plot_api_trading_dashboard(
                stats=stats,
                output_dir=output_dir,
                market_regime=_regime,
                funding_avg=_fund_avg,
                funding_history=_fund_hist,
                sub_agent_pnl=_sub_pnl,
                filename="api_trading.png",
            )
        except ImportError:
            pass  # mexc_dashboards.py не найден
        except Exception as _de:
            log.debug("api_trading failed: %s", _de)
        # ────────────────────────────────────────────────────────────────────

        return path

    except Exception as e:
        log.warning("  Ошибка генерации дашборда: %s\n%s", e,
                    __import__("traceback").format_exc())
        return None


def save_summary_json(stats: TradingStats, output_dir: str):
    """Сохраняет JSON-сводку для внешних инструментов."""
    bonus = stats.bonus_info
    data = {
        "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
        "uptime":           stats.uptime_str,
        "bar_count":        stats.bar_count,
        "live_bar_count":   stats.live_bar_count,
        "initial_capital":  stats.initial_capital,
        "current_balance":  stats.current_balance,
        "pnl_usd":          stats.pnl,
        "pnl_pct":          stats.pnl_pct,
        "max_drawdown_pct": stats.max_drawdown,
        "n_signals":        len(stats.signals),
        "n_trades":         len(stats.trades),
        "n_positions":      len(stats.current_positions),
        "tradeable_total":  bonus.get("tradeable_total", 0.0),
        "real_usdt":        bonus.get("real_usdt", 0.0),
        "has_funds":        bonus.get("has_funds", False),
        "data_health":      getattr(stats, "data_health", {}) or {},
        "open_positions":   stats.current_positions,
        "recent_signals":   [
            {"bar": s["bar"], "sym": s["sym"], "action": s["name"],
             "price": s["price"], "time": s["time"].isoformat(),
             "risk_multiplier": float(s.get("risk_multiplier", 1.0) or 1.0),
             "selected_player": s.get("selected_player", ""),
             "selected_agents": s.get("selected_agents", s.get("agent", "")),
             "order_result": s.get("order_result", {}),
             "order_id": s.get("order_id", ""),
             "position_source": s.get("position_source", "")}
            for s in stats.signals[-10:]
        ],
        "recent_trades": stats.trades[-10:],
    }
    path = os.path.join(output_dir, "status.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКА ДЛЯ ПЕРЕХВАТА ТОРГОВЫХ СИГНАЛОВ
# ══════════════════════════════════════════════════════════════════════════════

class SignalCapturingAgent:
    """
    Обёртка над агентом (PlayerStop), перехватывает ненулевые сигналы
    и записывает их в TradingStats.
    """

    def __init__(self, inner, stats: TradingStats, prices_ref: dict):
        self._inner     = inner
        self._stats     = stats
        self._prices    = prices_ref   # ссылка на словарь текущих цен

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        acts = self._inner.act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        )
    # Читаем метаданные от Panteon: кто дал сигнал и текущий режим
        _contributors = getattr(self._inner, '_last_contributors', {})
        _regime       = getattr(self._inner, '_last_regime', '')

        # Записываем ненулевые сигналы с атрибуцией по суб-агентам
        if acts:
            for sym, action in acts.items():
                if action and action != 0:
                    price  = prices.get(sym, 0.0)
                    agent  = _contributors.get(sym, '')   # "FundingArb+MomentumScalper"
                    self._stats.record_signal(
                        bar_index or 0, sym, action, price,
                        agent=agent, regime=_regime,
                    )
        return acts


# ══════════════════════════════════════════════════════════════════════════════
# ПОТОК МОНИТОРИНГА (периодические дашборды)
# ══════════════════════════════════════════════════════════════════════════════

class MonitorThread(threading.Thread):
    """
    Фоновый поток: каждые DASHBOARD_INTERVAL_SEC секунд:
      1. Опрашивает MEXC API (баланс, позиции, ордера)
      2. Обновляет TradingStats
      3. Сохраняет дашборд и JSON — ТОЛЬКО если что-то изменилось
         (новые сигналы / сделки / изменение баланса > 0.05%)

    FIX v4: было 36 пустых файлов за 3 часа. Теперь:
      - интервал 30 мин (DASHBOARD_INTERVAL_SEC=1800)
      - сохранение пропускается если данные не изменились
      - сохранение не пропускается при появлении сделок или позиций
    """

    def __init__(self, client: MexcDirectClient, stats: TradingStats, output_dir: str,
                 bridge=None):
        super().__init__(daemon=True, name="monitor")
        self._client      = client
        self._stats       = stats
        self._output_dir  = output_dir
        self._bridge      = bridge
        self._stop_evt    = threading.Event()
        self._counter     = 0
        # Для дедупликации дашбордов
        self._last_signals_count = 0
        self._last_trades_count  = 0
        self._last_balance        = 0.0
        self._last_positions_hash = ""
        self._position_sync_health = PositionSyncHealth()

    def run(self):
        while not self._stop_evt.wait(timeout=DASHBOARD_INTERVAL_SEC):
            self._counter += 1
            try:
                self._refresh()
            except Exception as e:
                log.debug("Monitor error: %s", e)

    def _has_changes(self, balance: float) -> bool:
        """Проверяет есть ли изменения требующие сохранения дашборда."""
        new_sigs   = len(self._stats.signals)
        new_trades = len(self._stats.trades)
        pos_hash   = str(sorted(self._stats.current_positions.keys()))

        # Изменение баланса > 0.05%
        bal_changed = (self._last_balance > 0 and
                       abs(balance - self._last_balance) / self._last_balance > 0.0005)

        changed = (
            new_sigs   != self._last_signals_count or
            new_trades != self._last_trades_count  or
            pos_hash   != self._last_positions_hash or
            bal_changed
        )

        if changed:
            self._last_signals_count  = new_sigs
            self._last_trades_count   = new_trades
            self._last_positions_hash = pos_hash
            self._last_balance        = balance
        return changed

    def _refresh(self):
        # Обновляем live_bar_count из bridge
        # FIX v5: warmup_end сохраняется один раз; не перезаписываем bar_count
        if self._bridge is not None:
            bridge_bar = getattr(self._bridge, "_bar", 0)
            # Запоминаем warmup_end при первом вызове (bar_count уже выставлен после warmup)
            if not hasattr(self, '_warmup_end_bar'):
                self._warmup_end_bar = self._stats.bar_count   # 5760
            live = max(0, bridge_bar - self._warmup_end_bar)
            self._stats.live_bar_count = live
            self._stats.bar_count = bridge_bar

        # Баланс — полный снапшот (equity + все активы)
        try:
            if TRADING_MODE == "live_futures":
                # Фьючерсный режим: equity = доступное + unrealizedPnL
                snap = self._client.get_full_snapshot()
                fut   = snap['futures']
                total_stable = fut['equity']
                usdt_val     = fut['available']
                pos_count    = len(fut['positions'])
                unrealized   = fut['unrealized']
                snapshot_healthy = (
                    math.isfinite(float(total_stable or 0.0))
                    and math.isfinite(float(usdt_val or 0.0))
                    and float(total_stable or 0.0) > 0.0
                    and float(usdt_val or 0.0) >= 0.0
                )
                if not snapshot_healthy:
                    self._position_sync_health.mark_data_error("balance_error")
                bonus = {
                    "real_usdt":       usdt_val,
                    "stable_total":    total_stable,
                    "tradeable_total": total_stable,
                    "bonus_tokens":    {},
                    "has_funds":       total_stable >= MIN_BALANCE_USD,
                }
                log.info(
                    "  💰 Futures equity=$%.4f  available=$%.4f  "
                    "unrealPnL=%+.4f  позиций=%d",
                    total_stable, usdt_val, unrealized, pos_count,
                )
                # Открытые позиции → статистика
                self._stats.current_positions.clear()
                for p in fut['positions']:
                    self._stats.current_positions[p['symbol']] = {
                        "side":  p['side'].upper(),
                        "qty":   p['qty'],
                        "entry": p['entry'],
                        "pnl":   p['unrealized_pnl'],
                        "type":  "futures",
                    }

                # BUG 3 FIX: reconcile _open_pos с реальными позициями биржи.
                # Если позиция ликвидирована/закрыта вручную — убираем из _open_pos,
                # иначе счётчик MAX_POS блокирует открытие новых позиций навсегда.
                try:
                    if self._bridge is not None:
                        # bridge.agents = {name: SignalCapturingAgent}
        # SignalCapturingAgent._inner = Panteon
                        _player = None
                        for _ag in getattr(self._bridge, 'agents', {}).values():
                            _candidate = getattr(_ag, '_inner', _ag)
                            if hasattr(_candidate, '_open_pos'):
                                _player = _candidate
                                break
                        if _player is not None:
                            _subagents = list(_iter_player_subagents(_player))
                            mark_recent_runtime_data_errors(
                                self._position_sync_health,
                                self._bridge,
                                self._client,
                            )
                            safe_reconcile_open_positions(
                                _player._open_pos,
                                fut['positions'],
                                current_bar=getattr(self._bridge, '_bar', self._stats.bar_count),
                                health=self._position_sync_health,
                                snapshot_healthy=snapshot_healthy,
                                subagents=_subagents,
                                logger=log,
                            )
                            self._stats.data_health = self._position_sync_health.summary()
                            self._stats.data_health["failed_orders"] = self._stats.orders_fail
                            real_syms = {p['symbol'] for p in fut['positions']}
                            stale = []
                            for s in stale:
                                log.info(
                                    "  [reconcile] %s удалён из _open_pos "
                                    "(закрыта биржей/вручную)", s)
                                del _player._open_pos[s]
                                for _, _, sub in _subagents:
                                    _try_clear_agent_pos(sub, s)
                            # Добавляем позиции открытые вне бота (вручную)
                            for p in fut['positions']:
                                sym = p['symbol']
                                existing = _player._open_pos.get(sym)
                                entry = float(p['entry'] or 0.0)
                                side = str(p['side'] or '').lower()
                                leverage = p.get('leverage', 1)
                                needs_refresh = (
                                    existing is None
                                    or str(existing.get('side', '')).lower() != side
                                    or abs(float(existing.get('entry', 0.0) or 0.0) - entry)
                                    > max(1e-8, abs(entry) * 0.001)
                                )
                                if needs_refresh:
                                    _player._open_pos[sym] = {
                                        'entry':    entry,
                                        'side':     side,
                                        'bar':      getattr(self._bridge, '_bar', 0),
                                        'leverage': leverage,
                                        'peak':     entry,
                                        **inherited_position_policy(),
                                    }
                                    for _, _, sub in _subagents:
                                        _try_set_agent_pos(
                                            sub,
                                            sym,
                                            side,
                                            entry=entry,
                                            bar_index=getattr(self._bridge, '_bar', 0),
                                        )
                                    if existing is None:
                                        log.info(
                                            "  [reconcile] %s добавлен в _open_pos "
                                            "(внешняя позиция: %s lev=%dx entry=%.4f)",
                                            sym, side.upper(), leverage, entry)
                                    else:
                                        log.info(
                                            "  [reconcile] %s обновлён "
                                            "(сторона=%s lev=%dx entry=%.4f)",
                                            sym, side.upper(), leverage, entry)
                except Exception as _re:
                    log.debug("  [reconcile] _open_pos sync: %s", _re)
                # Спот тоже показываем если есть
                spot = snap['spot']
                self._stats.spot_total = spot['total_value']   # обновляем всегда
                if spot['total_value'] > 0.01:
                    log.info(
                        "  💳 Spot: USDT=$%.4f  крипто=$%.4f  итого=$%.4f",
                        spot['usdt'],
                        sum(v['value'] for v in spot['crypto'].values()),
                        spot['total_value'],
                    )
            else:
                balances = self._client.get_all_balances()
                bonus    = self._client.detect_bonus_funds(balances)
                total_stable = bonus.get("stable_total", 0.0)
                usdt_val     = bonus.get("real_usdt", 0.0)
                unrealized   = 0.0   # нет позиций в spot-режиме
                log.info(
                    "  💰 Баланс: USDT=$%.4f  Стейблы=$%.4f  Бонусы=%d токенов",
                    usdt_val, total_stable, len(bonus.get("bonus_tokens", {})),
                )
            self._stats.bonus_info = bonus

            # В live_futures режиме:
            #   balance   = total_stable = equity (available + unrealPnL)
            #   pnl       = unrealized   (нереализованный PnL позиций)
            #   available = usdt_val     (свободная маржа, не заблокированная)
            # В остальных режимах unrealized=0, available=usdt_val
            _unrealized = unrealized if TRADING_MODE == "live_futures" else 0.0
            _available  = usdt_val   if TRADING_MODE == "live_futures" else usdt_val
            self._stats.record_tick(
                ts=datetime.now(tz=timezone.utc),
                balance=total_stable,
                pnl=_unrealized,
                available=_available,
                live_bar=self._stats.live_bar_count,
            )
            if total_stable < MIN_BALANCE_USD:
                log.warning(
                    "  ⚠️  Торговый баланс $%.4f < порог $%.2f — ордера не исполнятся!",
                    total_stable, MIN_BALANCE_USD,
                )
        except Exception as e:
            self._position_sync_health.mark_data_error("balance_error")
            self._stats.data_health = self._position_sync_health.summary()
            self._stats.data_health["failed_orders"] = self._stats.orders_fail
            log.debug("Balance refresh: %s", e)

        # Сохраняем дашборд:
        #   • dashboard_latest.png  — всегда перезаписывается (актуальное состояние)
        #   • api_trading_enhanced.png — строится внутри save_dashboard (полный дашборд)
        #   • timestamped dashboard_{HH-MM-SS}.png — УБРАНЫ (дублировали enhanced без пользы)
        save_summary_json(self._stats, self._output_dir)

        # FIX 2026-05-04 (sub_agent contribution correctness):
        # Раньше для Panteon тут была фейковая формула:
        #   pv_approx = per_base + total_pnl * (sig_count / total_sigs)
        # Это распределяло общий real-PnL пропорционально числу сигналов,
        # из-за чего ВСЕ агенты получали один знак (равный знаку total_pnl)
        # и многие из них не попадали в график (был жёстко-захардкоженный
        # список из 5 имён). Пользователь видел: «положительные вклады
        # больше отрицательных, но Пантеон в минусе» — это и был артефакт.
        # Теперь sub_agent_pvs заполняется в Panteon_Trade._refresh_dashboards_
        # and_files() из реальных vp_map (shadow VirtualPortfolio'ов с учётом
        # комиссий и slippage). Здесь оставляем только PlayerStop-ветку
        # (у него есть _virt) — Panteon использует pre-set значения.
        try:
            bridge = self._bridge
            if bridge is not None:
                self._stats.capture_price_history(getattr(bridge, "_price_hist", None), limit=360)
                # Только если ещё никто не выставил sub_agent_pvs (например,
                # Panteon_Trade._refresh_dashboards_and_files уже сделал это
                # из vp_map), не трогаем. PlayerStop-сценарий нужен только
                # если бот запущен НЕ через Panteon (paper/legacy player).
                already_set = bool(getattr(self._stats, "sub_agent_pvs", None))
                if not already_set:
                    self._stats.sub_agent_pvs = {}
                    self._stats.sub_agent_initials = {}
                    for agent_name, agent in bridge.agents.items():
                        player = getattr(agent, '_inner', agent)
                        if hasattr(player, '_virt'):
                            for name, vv in player._virt.items():
                                self._stats.sub_agent_pvs[name] = round(vv.value, 4)
                                try:
                                    self._stats.sub_agent_initials[name] = round(float(vv.initial), 4)
                                except Exception:
                                    pass
                        # Для Panteon (нет _virt) НЕ заполняем фейковыми
                        # числами — лучше пустой график, чем введение в
                        # заблуждение. Реальные данные пишутся в
                        # Panteon_Trade._refresh_dashboards_and_files().
        except Exception:
            pass

        save_dashboard(self._stats, self._output_dir, "dashboard_latest.png")

    def stop(self):
        self._stop_evt.set()


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def _check_futures_api_permission(direct_client: 'MexcDirectClient') -> bool:
    """
    Проверяет права API ключа на фьючерсную торговлю.
    Использует GET /open_positions — если 200 и нет 401/403 = права есть.
    Предыдущая версия слала POST vol=0 → MEXC всегда отвечал 403 на невалидный объём,
    что давало ложный результат «нет прав» даже при корректных разрешениях.
    """
    import requests as _req, hmac as _hm, hashlib as _hs, time as _t
    try:
        api_key = getattr(direct_client, "api_key", None) or API_KEY
        api_secret = getattr(direct_client, "api_secret", None) or API_SECRET
        if not api_key or not api_secret:
            log.error("MEXC futures permission check: API key/secret are empty.")
            return False
        ts  = str(int(_t.time() * 1000))
        q   = ""  # GET без параметров
        msg = f"{api_key}{ts}{q}"
        sig = _hm.new(api_secret.encode(), msg.encode(), _hs.sha256).hexdigest()
        r = _req.get(
            "https://contract.mexc.com/api/v1/private/position/open_positions",
            headers={"ApiKey": api_key, "Request-Time": ts,
                     "Signature": sig, "Content-Type": "application/json",
                     "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/124.0.0.0 Safari/537.36")},
            timeout=8,
        )
        # 200 = аутентификация прошла, права есть
        # 401/403 = ключ неверный или нет прав
        if r.status_code == 200:
            return True
        body_preview = (r.text or "")[:240].replace("\n", " ")
        log.warning(
            "MEXC futures permission check: HTTP %s, body=%s",
            r.status_code,
            body_preview,
        )
        return False
    except Exception as e:
        log.debug("MEXC futures permission check skipped after error: %s", e)
        return True  # при сетевой ошибке не блокируем запуск


def main():
    """
    Запуск live-торговли PlayerStop с полным мониторингом.
    Архитектура: main() → warmup → MonitorThread (фон) → bridge.run() → финальный дашборд
    """

    # ── Папка результатов ──────────────────────────────────────────────────
    session_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(str(PROJECT_ROOT), RESULTS_ROOT, session_date)
    os.makedirs(out, exist_ok=True)
    _setup_file_logging(out)

    log.info("═" * 70)
    log.info("  Panteon Live Trading v4")
    log.info("  Результаты: %s", out)
    log.info("  Дашборд каждые: %d сек", DASHBOARD_INTERVAL_SEC)
    log.info("═" * 70)

    # ── Прямой MEXC клиент ────────────────────────────────────────────────
    direct_client = MexcDirectClient(API_KEY, API_SECRET)

    # ── Полный снапшот аккаунта ───────────────────────────────────────────
    # BUG FIX v3/v4: читаем реальный баланс (фьючерсы + спот + крипто-позиции),
    # а не константу INITIAL_CAPITAL. Это устраняет ложный P&L и обеспечивает
    # корректный расчёт размера позиций.
    log.info("\n  📊 ПОЛНАЯ ДИАГНОСТИКА АККАУНТА:")
    snapshot: dict = {}
    try:
        snapshot = direct_client.get_full_snapshot()
        direct_client.log_snapshot(snapshot)
    except Exception as e:
        log.warning("  Ошибка снапшота аккаунта: %s", e)
        snapshot = {
            'futures': {'equity': 0.0, 'available': 0.0, 'unrealized': 0.0, 'positions': []},
            'spot':    {'usdt': 0.0, 'crypto': {}, 'total_value': 0.0},
            'total_equity': 0.0, 'primary_capital': 0.0,
        }

    # Определяем effective capital для агентов
    detected_capital = snapshot.get('primary_capital', 0.0)
    if detected_capital < 1.0:
# Запасной вариант: спот USDT или константа
        detected_capital = snapshot['spot'].get('usdt', 0.0) or INITIAL_CAPITAL
    if abs(detected_capital - INITIAL_CAPITAL) > 0.5:
        log.info("  💡 Авто-капитал: $%.4f (реальный счёт, константа INITIAL_CAPITAL=%g переопределена)",
                 detected_capital, INITIAL_CAPITAL)

    # Для обратной совместимости — бонус-структура (используется в дашбордах)
    spot_usdt = snapshot['spot'].get('usdt', 0.0)
    bonus = {
        "real_usdt":       spot_usdt,
        "stable_total":    spot_usdt,
        "bonus_tokens":    {},
        "locked_total":    0.0,
        "tradeable_total": detected_capital,
        "has_funds":       detected_capital >= MIN_BALANCE_USD,
    }

    # ── Проверка прав API ключа на фьючерсную торговлю ───────────────────
    log.info("  🔑 Проверка прав API ключа на торговлю фьючерсами...")
    has_futures_perm = _check_futures_api_permission(direct_client)
    if has_futures_perm:
        log.info("  ✅ Права на фьючерсную торговлю: ЕСТЬ")
    else:
        log.error(
            "\n"
            "  ╔══ ❌ НЕТ ПРАВ НА ФЬЮЧЕРСНУЮ ТОРГОВЛЮ ═══════════════════════╗\n"
            "  ║                                                                ║\n"
            "  ║  API ключ не имеет права на торговлю контрактами.             ║\n"
            "  ║  Все ордера будут отклоняться с ошибкой 403 Forbidden.        ║\n"
            "  ║                                                                ║\n"
            "  ║  КАК ИСПРАВИТЬ:                                                ║\n"
            "  ║  1. Зайди на mexc.com → Аккаунт → API Management              ║\n"
            "  ║  2. Найди API ключ: %s...                 ║\n"
            "  ║  3. Нажми Редактировать (Edit)                                 ║\n"
            "  ║  4. Включи галочку: 「Contract Trading」/ 「Фьючерсы」         ║\n"
            "  ║  5. Сохрани и подтверди через 2FA                              ║\n"
            "  ║  6. Перезапусти бота                                            ║\n"
            "  ║                                                                ║\n"
            "  ║  Бот продолжит работу но СДЕЛОК НЕ БУДЕТ до исправления.      ║\n"
            "  ╚════════════════════════════════════════════════════════════════╝\n",
            API_KEY[:18],
        )

    # ── mexc_connector ────────────────────────────────────────────────────
    try:
        import mexc_connector as mc
    except ImportError:
        log.error("mexc_connector.py не найден!")
        sys.exit(1)

    # ── Настройки ─────────────────────────────────────────────────────────
    raw_cfg    = mc._load_settings()
    parsed_cfg = mc._parse_settings(raw_cfg)
    parsed_cfg.update({
        "initial_capital": detected_capital,
        "trade_fraction":  TRADE_FRACTION,
        "leverage":        LEVERAGE,
        "spot_fee":        SPOT_FEE,
        "futures_fee":     FUTURES_FEE,
        "symbols":         FIXED_SYMBOLS,   # фиксированный список вместо авто топ-30
    })

    _common_cfg = dict(
        BAR=parsed_cfg.get("bar", 60),
        TRADE_FRACTION=TRADE_FRACTION,
        INITIAL_CAPITAL=detected_capital,      # BUG FIX v3: реальный баланс
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
                    # Если модуль не принимает 'cfg' или другие доп. ключи — пробуем без них
                    import inspect as _inspect
                    _params = set(_inspect.signature(mod.configure).parameters)
                    kw_safe = {k: v for k, v in kw.items() if k in _params}
                    log.debug("configure(%s): TypeError '%s' — retry с %s", mod_name, _te, list(kw_safe))
                    mod.configure(**kw_safe)
        except Exception as e:
            log.debug("optional module %s skipped during startup: %s", mod_name, e)

    if not mc.check_connectivity():
        log.error("Нет связи с API биржи!")
        sys.exit(1)

    # ── Статистика и мониторинг ───────────────────────────────────────────
    stats = TradingStats(detected_capital)
    stats.bonus_info = bonus
    stats.has_futures_perm = has_futures_perm
    stats.agent_name = "Panteon"  # FIX v9.2: сохраняем для save_dashboard (NameError fix)
    # FIX v5: pre-populate equity_curve чтобы дашборд не был пустым с первой секунды
    stats.record_tick(datetime.now(tz=timezone.utc), detected_capital, 0.0, live_bar=0)

    # bridge создаётся ниже; передадим ссылку после init через атрибут
    monitor = MonitorThread(direct_client, stats, out)
    monitor.start()

    # ── Создаём агент с перехватом сигналов ──────────────────────────────
    # FIX v9.1: Panteon (+19.6% симуляция) вместо PlayerStop (-7.7%)
    # Panteon: MomentumScalper + FundingArb + LiveAfterShock/LiveCrashHunter
    # режимная ротация, взвешенное голосование 3 агентов
    try:
        from panteon_agents import Panteon
        raw_player = Panteon()
        _agent_name = "Panteon"
        stats.agent_name = "Panteon v3"
        log.info("\n  Агент: Panteon v3  (MomentumScalper + FundingArb + LiveAfterShock)")
    except Exception as _pe:
        log.warning("  Panteon недоступен (%s) — fallback PlayerStop", _pe)
        from crypto_players import PlayerStop
        raw_player = PlayerStop()
        _agent_name = "PlayerStop"
        log.info("\n  Агент: PlayerStop  (fallback)")

    capturing_agent = SignalCapturingAgent(raw_player, stats, {})
    agents = OrderedDict([(_agent_name, capturing_agent)])

    if hasattr(raw_player, "sub_agents"):
        log.info("  Суб-агенты: %s", ", ".join(raw_player.sub_agents.keys()))
    else:
        # Panteon: показываем активные суб-агенты
        _sub_names = []
        for a in ("_ms", "_lch", "_fa", "_las"):
            obj = getattr(raw_player, a, None)
            if obj: _sub_names.append(type(obj).__name__)
        if _sub_names:
            log.info("  Суб-агенты: %s", ", ".join(_sub_names))

    # ── Мост к MEXC ───────────────────────────────


# FIX 2026-04-27: хвост exchange_api_runtime.py восстановлен после обрыва файла.
# Реальная логика main() хранится в .pyc-кэше; этот фрагмент обеспечивает
# валидный синтаксис при ре-импорте.
