from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

import numpy as np


class AgentSafetyProxy:
    def __init__(self, agent):
        self._agent = agent

    def _inner_act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        return self._agent.act(
            prices,
            volumes,
            month=month,
            portfolio_value=portfolio_value,
            bar_index=bar_index,
        )

    def __getattr__(self, item):
        if item == "_agent":
            raise AttributeError(item)
        return getattr(self._agent, item)


class SymbolUniverseGuard(AgentSafetyProxy):
    def __init__(self, agent, allowed_symbols: Iterable[str]):
        super().__init__(agent)
        self._allowed_symbols = frozenset(str(sym).upper() for sym in allowed_symbols if sym)

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        scoped_prices = {s: p for s, p in prices.items() if s in self._allowed_symbols}
        scoped_volumes = {s: v for s, v in volumes.items() if s in self._allowed_symbols}
        if not scoped_prices:
            return {s: 0 for s in prices}
        result = {s: 0 for s in prices}
        result.update(
            {
                s: a
                for s, a in (
                    self._inner_act(
                        scoped_prices,
                        scoped_volumes,
                        month=month,
                        portfolio_value=portfolio_value,
                        bar_index=bar_index,
                    )
                    or {}
                ).items()
                if s in self._allowed_symbols
            }
        )
        return result


class MarketRegimeActionGuard(AgentSafetyProxy):
    def __init__(self, agent, sma: int = 30, threshold: float = 0.006,
                 allowed_exit_actions: Iterable[int] = (3, 8)):
        super().__init__(agent)
        self._sma = int(sma)
        self._threshold = float(threshold)
        self._allowed_exit_actions = tuple(int(v) for v in allowed_exit_actions)
        self._history: Dict[str, deque] = {}

    def _is_sideways(self) -> bool:
        scores = []
        for hist in self._history.values():
            lst = list(hist)
            if len(lst) < self._sma * 2:
                continue
            recent = float(np.mean(lst[-self._sma:]))
            older = float(np.mean(lst[:self._sma]))
            if older > 0:
                scores.append(abs(recent / older - 1.0))
        return float(np.mean(scores)) < self._threshold if scores else True

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for sym, price in prices.items():
            self._history.setdefault(sym, deque(maxlen=self._sma * 4)).append(float(price))
        result = self._inner_act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        ) or {}
        if self._is_sideways():
            return result
        return {s: (a if a in self._allowed_exit_actions else 0) for s, a in result.items()}


class DeclineGuard(AgentSafetyProxy):
    def __init__(self, agent, period: int, threshold: float,
                 allowed_exit_actions: Iterable[int] = (3, 8)):
        super().__init__(agent)
        self._period = int(period)
        self._threshold = float(threshold)
        self._allowed_exit_actions = tuple(int(v) for v in allowed_exit_actions)
        self._history: Dict[str, deque] = {}

    def _declining(self) -> bool:
        changes = []
        for hist in self._history.values():
            lst = list(hist)
            if len(lst) < self._period + 1:
                continue
            base = lst[-self._period - 1]
            if base > 0:
                changes.append(lst[-1] / base - 1.0)
        return float(np.median(changes)) < self._threshold if changes else False

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for sym, price in prices.items():
            self._history.setdefault(sym, deque(maxlen=self._period * 2 + 10)).append(float(price))
        result = self._inner_act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        ) or {}
        if not self._declining():
            return result
        return {s: (a if a in self._allowed_exit_actions else 0) for s, a in result.items()}


class FeeAwareFilter(AgentSafetyProxy):
    def __init__(self, agent, fee: float, min_factor: float = 1.5, window: int = 20,
                 trade_actions: Iterable[int] = (1, 2, 4, 5, 6, 7)):
        super().__init__(agent)
        self._fee = float(fee)
        self._min_factor = float(min_factor)
        self._window = int(window)
        self._trade_actions = tuple(int(v) for v in trade_actions)
        self._history: Dict[str, deque] = {}

    def _expected_move(self, sym: str) -> float:
        hist = self._history.get(sym)
        if hist is None or len(hist) < self._window + 1:
            return 1.0
        lst = list(hist)
        return float(np.mean([abs(lst[i] / lst[i - 1] - 1.0) for i in range(-self._window, 0)]))

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for sym, price in prices.items():
            self._history.setdefault(sym, deque(maxlen=self._window * 2 + 5)).append(float(price))
        result = self._inner_act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        ) or {}
        min_move = self._fee * self._min_factor
        return {
            s: (a if (a not in self._trade_actions or self._expected_move(s) >= min_move) else 0)
            for s, a in result.items()
        }


class ExtremeMoveGuard(AgentSafetyProxy):
    def __init__(self, agent, lookback: int = 24 * 60, extreme_move: float = 0.30,
                 allowed_exit_actions: Iterable[int] = (3, 8)):
        super().__init__(agent)
        self._lookback = int(lookback)
        self._extreme_move = float(extreme_move)
        self._allowed_exit_actions = tuple(int(v) for v in allowed_exit_actions)
        self._history: Dict[str, deque] = {}

    def _is_extreme(self, sym: str) -> bool:
        hist = self._history.get(sym)
        if hist is None or len(hist) < 2:
            return False
        lst = list(hist)
        lb = min(self._lookback, len(lst) - 1)
        base = lst[-lb - 1]
        return base > 0 and abs(lst[-1] / base - 1.0) >= self._extreme_move

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        for sym, price in prices.items():
            self._history.setdefault(sym, deque(maxlen=self._lookback + 10)).append(float(price))
        result = self._inner_act(
            prices, volumes,
            month=month, portfolio_value=portfolio_value, bar_index=bar_index,
        ) or {}
        return {
            s: (a if (a in self._allowed_exit_actions or not self._is_extreme(s)) else 0)
            for s, a in result.items()
        }


class PortfolioSafetyGovernor:
    def __init__(self, master_stop_loss: float = 0.08, min_order_cap: float = 120.0):
        self.master_stop_loss = float(master_stop_loss)
        self.min_order_cap = float(min_order_cap)

    def update_day_boundary(self, player, *, bar_span: int, logger=None) -> None:
        if player.t - player._day_start_bar >= 24 * int(bar_span):
            player._day_start_pv = player._pv
            player._day_start_bar = player.t
            player._day_stopped = False
            if logger is not None:
                logger.debug("  [%s] New day: pv=%.2f", getattr(player, "NAME", "Player"), player._pv)

    def check_day_stop(self, player, logger=None) -> bool:
        if player._day_stopped and player.t < player._day_stop_until:
            return True
        if player._day_stopped and player.t >= player._day_stop_until:
            player._day_stopped = False
        dd = (player._day_start_pv - player._pv) / max(player._day_start_pv, 1.0)
        if dd >= float(player.DAY_STOP_PCT):
            player._day_stopped = True
            player._day_stop_until = player.t + int(player.STOP_COOLDOWN)
            if logger is not None:
                logger.warning(
                    "  [%s] DAY-STOP %.1f%% (pv=%.2f, day_start=%.2f)",
                    getattr(player, "NAME", "Player"), dd * 100.0, player._pv, player._day_start_pv,
                )
            return True
        return False

    def check_peak_stop(self, player, logger=None) -> bool:
        if player._pv > player._pv_peak:
            player._pv_peak = player._pv
        if player._peak_stopped and player.t < player._peak_stop_until:
            return True
        if player._peak_stopped and player.t >= player._peak_stop_until:
            player._peak_stopped = False
        dd = (player._pv_peak - player._pv) / max(player._pv_peak, 1.0)
        if dd >= float(player.PEAK_STOP_PCT):
            player._peak_stopped = True
            player._peak_stop_until = player.t + int(player.STOP_COOLDOWN) * 6
            if logger is not None:
                logger.warning(
                    "  [%s] PEAK-STOP %.1f%% (peak=%.2f, now=%.2f)",
                    getattr(player, "NAME", "Player"), dd * 100.0, player._pv_peak, player._pv,
                )
            return True
        return False

    def force_close_all(self, player, prices: dict) -> dict:
        close = {}
        for sym, info in list(player._open_pos.items()):
            close[sym] = 3 if info["type"] == "spot" else 8
            cur = prices.get(sym, info.get("entry_price", 0))
            ep = info.get("entry_price", 0)
            tp = info["type"]
            if tp == "fut_short":
                pnl_pct = (1 - cur / ep) * 100 if ep > 0 else 0.0
            else:
                pnl_pct = (cur / ep - 1) * 100 if ep > 0 else 0.0
            player._all_trades.append(
                {
                    "bar": player.t,
                    "sym": sym,
                    "action": "CLOSE",
                    "agent": info.get("agent", "?"),
                    "price": round(cur, 6),
                    "pnl_pct": round(pnl_pct, 3),
                    "regime": getattr(player, "_regime", "unknown"),
                    "reason": "FORCE_CLOSE",
                }
            )
            player._open_pos.pop(sym, None)
        return {**{s: 0 for s in prices}, **close}

    def update_profit_lock(self, player, logger=None) -> float:
        if player._pv < player._lock_next_at:
            return 0.0

        tradeable = player._pv
        profit = tradeable - player._capital
        if profit <= 0:
            player._lock_next_at = player._pv * (1 + player.LOCK_STEP)
            return 0.0

        max_lockable = player._capital * player.LOCK_MAX - player._locked_usdt
        if max_lockable <= 0:
            player._lock_next_at = player._pv * (1 + player.LOCK_STEP)
            return 0.0

        to_lock = min(profit * player.LOCK_FRACTION, max_lockable)
        player._locked_usdt += to_lock
        player._lock_next_at = player._pv * (1 + player.LOCK_STEP)

        if logger is not None:
            logger.info(
                "  [%s] PROFIT-LOCK +$%.2f -> locked=%.2f (pv=%.2f)",
                getattr(player, "NAME", "Player"), to_lock, player._locked_usdt, player._pv,
            )
        return to_lock

    def filter_min_order(self, player, actions: dict, *, trade_fraction: float) -> dict:
        pos = min(player._pv * float(trade_fraction), self.min_order_cap)
        return {s: a for s, a in actions.items() if a in (3, 8) or pos >= player.MIN_ORDER_USDT}

    def check_master_stop(self, player, prices: dict, logger=None) -> dict:
        close = {}
        for sym, info in list(player._open_pos.items()):
            cur = prices.get(sym, 0)
            ep = info.get("entry_price", 0)
            if not (cur and ep > 0):
                continue
            tp = info["type"]
            if tp == "fut_short":
                loss = (cur - ep) / ep
                pnl_pct = (1 - cur / ep) * 100
            else:
                loss = (ep - cur) / ep
                pnl_pct = (cur / ep - 1) * 100
            if loss >= self.master_stop_loss:
                close[sym] = 3 if tp == "spot" else 8
                player._all_trades.append(
                    {
                        "bar": player.t,
                        "sym": sym,
                        "action": "CLOSE",
                        "agent": info.get("agent", "?"),
                        "price": round(cur, 6),
                        "pnl_pct": round(pnl_pct, 3),
                        "regime": getattr(player, "_regime", "unknown"),
                        "reason": "SL",
                    }
                )
                player._open_pos.pop(sym, None)
                if logger is not None:
                    logger.info(
                        "  [%s] MASTER-SL %.1f%% -> %s",
                        getattr(player, "NAME", "Player"), loss * 100.0, sym,
                    )
        return close


class PositionExitGovernor:
    def __init__(self, close_action: int = 8,
                 trail_activation: float = 0.01, stale_move_threshold: float = 0.02):
        self.close_action = int(close_action)
        self.trail_activation = float(trail_activation)
        self.stale_move_threshold = float(stale_move_threshold)

    def apply_exit_actions(
        self,
        position_book: dict,
        prices: dict,
        current_bar: int,
        out: Optional[dict] = None,
        *,
        stop_loss_pct: float,
        take_profit_pct: float,
        trail_pct: float,
        stale_bars: int,
        logger=None,
    ) -> dict:
        result = out if out is not None else {}
        for sym, info in list(position_book.items()):
            cur = prices.get(sym, 0)
            entry = info.get("entry", 0)
            side = str(info.get("side", "long") or "long")
            bar_opened = int(info.get("bar", 0) or 0)
            peak = float(info.get("peak", entry) or entry)
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

            pos_stop_loss_pct = _float_override("tight_sl_pct", float(stop_loss_pct))
            pos_take_profit_pct = _float_override("take_profit_pct", float(take_profit_pct))
            pos_trail_pct = _float_override("trail_pct", float(trail_pct))
            pos_stale_bars = _int_override("stale_bars", int(stale_bars))
            pos_stale_move_threshold = _float_override(
                "stale_move_threshold",
                self.stale_move_threshold,
            )
            source = str(info.get("source") or info.get("position_source") or "")
            suppress_stale_until_bar = int(info.get("suppress_stale_until_bar", -1) or -1)
            stale_allowed = not (
                source == "recovered_after_snapshot_gap"
                and current_bar <= suppress_stale_until_bar
            )

            if entry == 0 and cur > 0:
                info["entry"] = cur
                info["peak"] = cur
                if logger is not None:
                    logger.warning("  [PositionGovernor] %s entry=0 -> set %.4f", sym, cur)
                continue

            if cur <= 0 or entry <= 0:
                continue

            if side == "long":
                move = cur / entry - 1.0
                if cur > peak:
                    info["peak"] = cur
                    peak = cur
                retreat = 1.0 - cur / peak if peak > 0 else 0.0
                peak_move = peak / entry - 1.0 if entry > 0 else 0.0
            else:
                move = 1.0 - cur / entry
                if cur < peak:
                    info["peak"] = cur
                    peak = cur
                retreat = cur / peak - 1.0 if peak > 0 else 0.0
                peak_move = 1.0 - peak / entry if entry > 0 else 0.0

            close_reason = None
            if move < -pos_stop_loss_pct:
                close_reason = f"SL({move * 100:+.1f}%)"
            elif move > pos_take_profit_pct:
                close_reason = f"TP({move * 100:+.1f}%)"
            elif move > self.trail_activation and retreat > pos_trail_pct:
                close_reason = (
                    f"TRAIL(peak_move={peak_move * 100:+.1f}% now={move * 100:+.1f}%)"
                )
            elif (
                stale_allowed
                and current_bar - bar_opened > pos_stale_bars
                and abs(move) < pos_stale_move_threshold
            ):
                close_reason = f"STALE({current_bar - bar_opened}bars, move={move * 100:+.1f}%)"

            if close_reason:
                if logger is not None:
                    log_fn = logger.info if close_reason.startswith(("SL", "TP", "TRAIL", "STALE")) else logger.warning
                    log_fn(
                        "  [PositionGovernor] CLOSE %s %s entry=%.4f cur=%.4f %s",
                        sym, side.upper(), entry, cur, close_reason,
                    )
                result[sym] = self.close_action
                position_book.pop(sym, None)
        return result

    def sync_positions(self, position_book: dict, actions: dict, prices: dict, current_bar: int) -> None:
        for sym, action in actions.items():
            if action in (1, 2, 4, 5) and sym not in position_book:
                px = prices.get(sym, 0)
                position_book[sym] = {
                    "entry": px,
                    "side": "long",
                    "bar": current_bar,
                    "peak": px,
                    "source": "pending_order",
                    "position_source": "pending_order",
                    "pending_order": True,
                }
            elif action in (6, 7) and sym not in position_book:
                px = prices.get(sym, 0)
                position_book[sym] = {
                    "entry": px,
                    "side": "short",
                    "bar": current_bar,
                    "peak": px,
                    "source": "pending_order",
                    "position_source": "pending_order",
                    "pending_order": True,
                }
            elif action in (3, 8) and sym in position_book:
                position_book.pop(sym, None)


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
