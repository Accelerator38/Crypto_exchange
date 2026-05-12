"""MarketFeed Protocol — источник market snapshot-ов для shadow run.

Адаптеры:
  ReplayFeed  — список заранее заданных snapshot-ов (для тестов)
  PollingFeed — polling files (status.json + последние сигналы) с диска
                — используется для shadow-run рядом с боевым v1
  CallableFeed — простая обёртка над функцией (для CLI/scripts)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from ..domain.types import MarketSnapshot
from .adapters import make_market_snapshot


# ────────────────────────────────────────────────────────────────────
# Protocol
# ────────────────────────────────────────────────────────────────────


class MarketFeed(Protocol):
    """Источник market data.

    `next_bar()` возвращает MarketSnapshot или None если новых баров нет.
    Не блокирующий: caller сам решает, ждать или нет.
    """

    def next_bar(self) -> Optional[MarketSnapshot]: ...


# ────────────────────────────────────────────────────────────────────
# ReplayFeed — для тестов
# ────────────────────────────────────────────────────────────────────


@dataclass
class ReplayFeed:
    """Простой feed из заранее подготовленных snapshot-ов."""

    snapshots: List[MarketSnapshot] = field(default_factory=list)
    _index:    int = 0

    def append(self, snap: MarketSnapshot) -> None:
        self.snapshots.append(snap)

    def reset(self) -> None:
        self._index = 0

    def next_bar(self) -> Optional[MarketSnapshot]:
        if self._index >= len(self.snapshots):
            return None
        snap = self.snapshots[self._index]
        self._index += 1
        return snap

    @property
    def remaining(self) -> int:
        return max(0, len(self.snapshots) - self._index)


# ────────────────────────────────────────────────────────────────────
# CallableFeed — обёртка над функцией
# ────────────────────────────────────────────────────────────────────


@dataclass
class CallableFeed:
    """Feed-обёртка для случаев, когда callsite сам готов отдать snapshot.

    Полезно когда v1-bridge есть в той же программе и можно подписаться
    на его события.
    """

    fn: Callable[[], Optional[MarketSnapshot]]

    def next_bar(self) -> Optional[MarketSnapshot]:
        try:
            return self.fn()
        except Exception:
            return None


# ────────────────────────────────────────────────────────────────────
# PollingFeed — читает status.json от боевого v1
# ────────────────────────────────────────────────────────────────────


class PollingFeed:
    """Polling-feed: читает status.json из v1-сессии и эмитит MarketSnapshot
    при появлении нового бара.

    Использование:
        feed = PollingFeed(session_dir="/path/to/MEXC/2026-05-05_*/")
        while True:
            snap = feed.next_bar()
            if snap is None:
                time.sleep(5)
                continue
            # process snap
    """

    def __init__(
        self,
        session_dir:  str,
        *,
        prices_fn:    Optional[Callable[[], Dict[str, float]]] = None,
        regime_fn:    Optional[Callable[[], str]] = None,
    ):
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session_dir not found: {session_dir}")
        self._dir = session_dir
        self._status_path = os.path.join(session_dir, "status.json")
        self._leaderboard_path = os.path.join(session_dir, "leaderboard_agents.json")
        self._prices_fn = prices_fn
        self._regime_fn = regime_fn
        self._last_bar: int = -1

    def next_bar(self) -> Optional[MarketSnapshot]:
        # 1. Читаем status.json (если есть)
        try:
            with open(self._status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        bar = int(status.get("bar_count", 0) or 0)
        if bar <= self._last_bar:
            return None  # нет нового бара
        self._last_bar = bar

        # 2. Цены — из callback или из recent_signals[].price
        if self._prices_fn is not None:
            prices = self._prices_fn() or {}
        else:
            # Fallback: из recent_signals возьмём последние цены по символам
            prices = self._extract_recent_prices(status)

        # 3. Regime — из callback или из leaderboard metadata
        regime_str = "neutral"
        if self._regime_fn is not None:
            regime_str = self._regime_fn() or "neutral"
        else:
            try:
                with open(self._leaderboard_path, "r", encoding="utf-8") as f:
                    lb = json.load(f)
                regime_str = str(lb.get("metadata", {}).get("regime") or "neutral")
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

        return make_market_snapshot(
            bar=bar,
            prices=prices,
            regime=regime_str,
        )

    @staticmethod
    def _extract_recent_prices(status: dict) -> Dict[str, float]:
        """Возвращает sym → последняя_цена_из_recent_signals.

        Fallback на случай, если callback prices_fn не подан.
        """
        out: Dict[str, float] = {}
        for sig in (status.get("recent_signals") or []):
            sym = str(sig.get("sym") or "").upper()
            try:
                px = float(sig.get("price") or 0.0)
                if sym and px > 0:
                    out[sym] = px
            except (TypeError, ValueError):
                continue
        # И из open_positions тоже
        for sym, info in (status.get("open_positions") or {}).items():
            sym = str(sym).upper()
            try:
                px = float(info.get("entry") or 0.0)
                if px > 0 and sym not in out:
                    out[sym] = px
            except (TypeError, ValueError):
                continue
        return out
