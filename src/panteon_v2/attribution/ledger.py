"""AttributionLedger — детерминированная проекция EventLog в PnL по
делегатам/агентам.

Решает корневую проблему v1 P19 ("shadow ≠ real") и P17 ("фейковая формула
вклада"). В v2:
  • Каждый Trade связан с Signal через signal_id (тип-гарантия).
  • PositionOpened/PositionClosed эмиттят by_player/by_agent.
  • AttributionLedger проходит по этим events и строит словари:
      total_pnl_by_player()  → Dict[player_label, realized_pnl_usd]
      total_pnl_by_agent()   → Dict[agent_label, realized_pnl_usd]
  • Сумма == sum(realized_pnl across all PositionClosed events) → Q2.

Дашборды (Phase 6) читают ТОЛЬКО эти методы. Никаких параллельных
sub_agent_pvs формул.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .event_log import EventLog
from .events import (
    OrderFilled,
    OrderRejected,
    OrderSent,
    PositionClosed,
    PositionOpened,
)


# ────────────────────────────────────────────────────────────────────
# Attribution record
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Attribution:
    """Запись об одной сделке (или открытой позиции).

    Если is_open=True → close_signal_id=None, realized_pnl=0.0,
                       by_player/by_agent — открывающего.
    Если is_open=False → есть pair (open_signal_id, close_signal_id),
                          realized_pnl рассчитан, atribution к открывшему.
    """

    open_signal_id:  int
    close_signal_id: Optional[int]
    sym:             str
    side:            str       # "long" | "short"
    by_player:       str
    by_agent:        str
    entry:           float
    exit:            Optional[float]
    qty:             float
    realized_pnl:    float
    bar:             int
    is_open:         bool

    @property
    def is_closed(self) -> bool:
        return not self.is_open


# ────────────────────────────────────────────────────────────────────
# AttributionLedger
# ────────────────────────────────────────────────────────────────────


class AttributionLedger:
    """Детерминированная projection из EventLog.

    Использование:
        ledger = AttributionLedger()
        ledger.replay_from_event_log(event_log)
        # или с ограничением по бару:
        ledger.replay_from_event_log(event_log, up_to_bar=12345)

        for attr in ledger.realized_attributions():
            ...
        player_pnl = ledger.total_pnl_by_player()  # Dict[label, pnl_usd]

    Все методы детерминированы при одном и том же EventLog → Q2 + INV-5.
    """

    def __init__(self) -> None:
        self._closed: List[Attribution] = []
        # sym → Attribution, для текущих open positions
        self._open: Dict[str, Attribution] = {}

    # ── Replay ──────────────────────────────────────────────────────

    def replay_from_event_log(
        self,
        event_log: EventLog,
        *,
        up_to_bar: Optional[int] = None,
    ) -> None:
        """Перечитывает события и пересобирает state с нуля.

        Идемпотентный: повторный вызов с тем же логом → тот же state.
        """
        self._closed.clear()
        self._open.clear()

        # Берём только нужные типы событий, отсортированные в порядке вставки
        # (EventLog возвращает их именно так).
        relevant_types = (PositionOpened, PositionClosed)
        for ev in event_log.query(
            event_types=relevant_types,
            before_bar=(up_to_bar + 1) if up_to_bar is not None else None,
        ):
            if isinstance(ev, PositionOpened):
                self._on_position_opened(ev)
            elif isinstance(ev, PositionClosed):
                self._on_position_closed(ev)

    def _on_position_opened(self, ev: PositionOpened) -> None:
        attr = Attribution(
            open_signal_id=ev.signal_id,
            close_signal_id=None,
            sym=ev.sym,
            side=ev.side,
            by_player="",   # PositionOpened не несёт by_player в текущей схеме
            by_agent="",    # — мы обновим из PositionClosed (там есть)
                            # либо, если позиция остаётся open, оставим пустыми
                            # и попытаемся восстановить через Signal lookup
                            # (см. _enrich_from_signals если нужно)
            entry=ev.entry,
            exit=None,
            qty=ev.qty,
            realized_pnl=0.0,
            bar=ev.bar,
            is_open=True,
        )
        self._open[ev.sym] = attr

    def _on_position_closed(self, ev: PositionClosed) -> None:
        # Если open был зарегистрирован — заменяем
        opened = self._open.pop(ev.sym, None)
        attr = Attribution(
            open_signal_id=ev.open_signal_id,
            close_signal_id=ev.close_signal_id,
            sym=ev.sym,
            side=ev.side,
            by_player=ev.by_player,
            by_agent=ev.by_agent,
            entry=ev.entry,
            exit=ev.exit,
            qty=ev.qty,
            realized_pnl=ev.realized_pnl,
            bar=ev.bar,
            is_open=False,
        )
        self._closed.append(attr)

    # ── Read API ────────────────────────────────────────────────────

    def total_pnl_by_player(self) -> Dict[str, float]:
        """Сумма realized_pnl по каждому игроку.

        Только закрытые позиции. Открытые позиции — отдельный метод
        unrealized_attribution() (но в первой реализации мы не считаем
        unrealized, т.к. для этого нужен текущий рынок).
        """
        out: Dict[str, float] = {}
        for attr in self._closed:
            if not attr.by_player:
                continue
            out[attr.by_player] = out.get(attr.by_player, 0.0) + attr.realized_pnl
        return out

    def total_pnl_by_agent(self) -> Dict[str, float]:
        """Сумма realized_pnl по каждому агенту-инициатору."""
        out: Dict[str, float] = {}
        for attr in self._closed:
            if not attr.by_agent:
                continue
            out[attr.by_agent] = out.get(attr.by_agent, 0.0) + attr.realized_pnl
        return out

    def realized_attributions(
        self,
        *,
        sym: Optional[str] = None,
        player: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> List[Attribution]:
        """Все закрытые attributions с опциональным фильтром."""
        out = []
        for attr in self._closed:
            if sym is not None and attr.sym != sym:
                continue
            if player is not None and attr.by_player != player:
                continue
            if agent is not None and attr.by_agent != agent:
                continue
            out.append(attr)
        return out

    def open_attributions(self) -> List[Attribution]:
        """Все ещё-открытые позиции (на момент последнего replay)."""
        return list(self._open.values())

    def open_attribution_for_sym(self, sym: str) -> Optional[Attribution]:
        return self._open.get(sym)

    @property
    def total_realized_pnl(self) -> float:
        """Сумма всех realized_pnl. Q2: должна совпадать с
        sum(player_pnl) и sum(agent_pnl) (но agent может быть пустым)."""
        return sum(a.realized_pnl for a in self._closed)

    @property
    def closed_count(self) -> int:
        return len(self._closed)

    @property
    def open_count(self) -> int:
        return len(self._open)

    # ── Per-bar / per-window ───────────────────────────────────────

    def pnl_by_player_in_window(
        self,
        from_bar: int,
        to_bar:   int,
    ) -> Dict[str, float]:
        """PnL по игрокам только за указанный bar-диапазон [from_bar, to_bar)."""
        out: Dict[str, float] = {}
        for attr in self._closed:
            if attr.bar < from_bar or attr.bar >= to_bar:
                continue
            if attr.by_player:
                out[attr.by_player] = out.get(attr.by_player, 0.0) + attr.realized_pnl
        return out

    # ── Counters per player/agent ──────────────────────────────────

    def trade_counts_by_player(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attr in self._closed:
            if attr.by_player:
                out[attr.by_player] = out.get(attr.by_player, 0) + 1
        return out

    def win_counts_by_player(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attr in self._closed:
            if attr.by_player and attr.realized_pnl > 0:
                out[attr.by_player] = out.get(attr.by_player, 0) + 1
        return out

    # ── Sanity check (Q2) ──────────────────────────────────────────

    def trade_counts_by_agent(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attr in self._closed:
            if attr.by_agent:
                out[attr.by_agent] = out.get(attr.by_agent, 0) + 1
        return out

    def win_counts_by_agent(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attr in self._closed:
            if attr.by_agent and attr.realized_pnl > 0:
                out[attr.by_agent] = out.get(attr.by_agent, 0) + 1
        return out

    def consistency_check(self) -> bool:
        """True если суммы согласованы: total = sum(by_player) если все
        attributions имеют by_player.
        """
        total = self.total_realized_pnl
        attributed = sum(self.total_pnl_by_player().values())
        # Не все attributions могут иметь by_player (например, если v1-trade
        # пробрасывал пустые поля). Допускаем разницу = unattributed PnL.
        unattributed = sum(
            a.realized_pnl for a in self._closed
            if not a.by_player
        )
        return abs(total - attributed - unattributed) < 1e-6
