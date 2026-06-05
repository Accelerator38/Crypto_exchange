"""PerformanceMemory — единый регистр метрик per-label × per-regime.

Заменяет в v1:
- `Panteon._regime_memory` (на каждом наследнике своя копия)
- `Panteon._symbol_regime_memory`
- `Panteon._shadow_player_regime_memory` (отдельный регистр для игроков)
- `Panteon._shadow_player_symbol_memory`
- `Panteon._context_memory` / `_factor_memory`
- `Panteon._global_success_memory`

→ один регистр, единственная точка обновления через `update_from_trade`.

Принципы:
- "label" — это имя агента ИЛИ игрока (для PerformanceMemory они эквивалентны).
  В update_from_trade обновляются метрики и by_agent, и by_player отдельно.
- Метрики хранятся per-regime: ключ (label, regime). Запрос с regime=None
  возвращает агрегированные по всем регимам.
- Mutable internal state (`_LabelRegimeState`) → immutable view (`Metrics`).
- Sharpe считается по логарифмическим returns закрытых сделок.
- max_dd_pct — максимальная просадка от пика equity-curve этого label×regime.
- Открытые позиции отслеживаются для расчёта realized PnL при закрытии.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Tuple

from ..domain.types import Action, Metrics, Regime, Signal, Trade


# ────────────────────────────────────────────────────────────────────
# Внутреннее mutable состояние для одной пары (label, regime)
# ────────────────────────────────────────────────────────────────────


@dataclass
class _LabelRegimeState:
    """Mutable rolling state. Не экспортируется наружу — снаружи отдаётся
    immutable Metrics через ``snapshot_as_metrics``."""

    closed_trades: int = 0
    entries:       int = 0
    signals:       int = 0
    wins:          int = 0
    losses:        int = 0
    pnl_pct:       float = 0.0          # cumulative net
    pnl_gross_pct: float = 0.0
    fee_pct:       float = 0.0
    funding_pct:   float = 0.0
    # Returns каждой закрытой сделки (для Sharpe)
    returns:       List[float] = field(default_factory=list)
    # Пиковое значение equity-curve (1.0 = baseline)
    equity:        float = 1.0
    peak:          float = 1.0
    max_dd_pct:    float = 0.0          # максимальная просадка в %
    equity_curve:  List[float] = field(default_factory=lambda: [100.0])
    blocked_signals: int = 0
    rejected_signals: int = 0
    pending_signals: int = 0
    execution_failures: int = 0

    def snapshot_as_metrics(self) -> Metrics:
        sharpe = _sharpe(self.returns)
        return Metrics(
            pnl_pct=self.pnl_pct,
            closed_trades=self.closed_trades,
            entries=self.entries,
            signals=self.signals,
            wins=self.wins,
            losses=self.losses,
            sharpe=sharpe,
            max_dd_pct=self.max_dd_pct,
            blocked_signals=self.blocked_signals,
            rejected_signals=self.rejected_signals,
            pending_signals=self.pending_signals,
            execution_failures=self.execution_failures,
            pnl_gross_pct=self.pnl_gross_pct,
            fee_pct=self.fee_pct,
            funding_pct=self.funding_pct,
        )


def _sharpe(returns: List[float]) -> float:
    """Простой Sharpe: mean / std (без annualization).

    Возвращает 0.0 если выборка слишком мала или std=0.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return 0.0
    return float(mean / std)


def _uses_neutral_seed(regime: Regime) -> bool:
    return regime in {
        Regime.RANGE_LOW_VOL,
        Regime.CHOPPY_DOWN,
        Regime.CHOPPY_UP,
        Regime.MIXED_ROTATIONAL,
    }


# ────────────────────────────────────────────────────────────────────
# Открытая позиция — для парного расчёта close → realized PnL
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _OpenPosition:
    """Запоминается на open, используется на close."""

    signal_id:    int
    label:        str
    regime:       Regime
    side:         str
    entry_price:  float
    qty:          float
    fee_open:     float
    funding_open: float = 0.0
    open_action:  str = ""


# ────────────────────────────────────────────────────────────────────
# PerformanceMemory — публичный API
# ────────────────────────────────────────────────────────────────────


# Дефолтный фракционный размер позиции (как в v1 — обычно ~10% капитала).
# Используется чтобы абсолютный pnl_pct совпадал с тем, что видит
# Пантеон в реальной торговле. Конфигурируется через __init__.
_DEFAULT_TRADE_FRACTION = 0.10
_DEFAULT_EXCHANGE_SCOPE = "__default__"
_CASH_FLAT_AGENT = "CashFlat"
_NO_TRADE_PLAYER = "NoTrade"
_OPENER_ATTRIBUTED_CLOSE_AGENTS = frozenset({
    _CASH_FLAT_AGENT,
    "GeneticsProbationRegimeExit",
    "PartialProfitLock",
    "StalePositionGuard",
})


class PerformanceMemory:
    """Единый owner per-label × per-regime метрик.

    Использование:
        perf = PerformanceMemory()
        # на open
        perf.update_from_trade(trade=open_trade, signal=open_signal)
        # на close
        perf.update_from_trade(trade=close_trade, signal=close_signal)
        # запрос
        m = perf.get("LiveAfterShock", regime=Regime.BULLISH)
        m_agg = perf.get("LiveAfterShock")            # агрегат по всем
        per_regime = perf.per_regime_for_label("LiveAfterShock")
        top = perf.top_k_for_regime(Regime.NEUTRAL, k=5, scorer=score_fn)
    """

    def __init__(
        self,
        *,
        trade_fraction: float = _DEFAULT_TRADE_FRACTION,
        strict_close_action_family: bool = False,
        exchange_scope: str = _DEFAULT_EXCHANGE_SCOPE,
    ):
        if not 0 < trade_fraction <= 1.0:
            raise ValueError(f"trade_fraction must be in (0, 1], got {trade_fraction}")
        self._trade_fraction = float(trade_fraction)
        self._strict_close_action_family = bool(strict_close_action_family)
        self._exchange_scope = _normalize_exchange_scope(exchange_scope)
        # (label, regime) → state
        self._state: Dict[Tuple[str, Regime], _LabelRegimeState] = {}
        self._context_state: Dict[Tuple[str, str, Regime, str], _LabelRegimeState] = {}
        self._context_bootstrap_state: Dict[Tuple[str, Regime], _LabelRegimeState] = {}
        # Открытые позиции по (label, sym) — нужно знать для close.
        # Один label может иметь только одну открытую позицию по sym
        # (общепринятое допущение, как в v1).
        self._open: Dict[Tuple[str, str, str], _OpenPosition] = {}
        # Записанные signals — для bookkeeping (avoid double-counting if
        # update_from_signal вызывается отдельно):
        self._seen_signal_ids: set = set()
        self._aggregate_cache: Dict[str, Metrics] = {}
        self._context_aggregate_cache: Dict[Tuple[str, str, str], Metrics] = {}
        self._equity_curves: Dict[str, List[float]] = {}

    # ── Обновление ──────────────────────────────────────────────────

    def record_signal(self, signal: Signal) -> None:
        """Только инкремент signal-counter.

        Используется когда сигнал сгенерирован, но **не** факт что он
        дойдёт до биржи (TradeExecutor может отклонить). Полезно для
        диагностики, но `update_from_trade` НЕ требует предварительного
        вызова record_signal.
        """
        self._record_signal_for_labels(signal, self._labels_for_signal(signal))

    def record_actor_failure(
        self,
        label: str,
        regime: Regime,
        *,
        penalty_pct: float = 0.05,
    ) -> None:
        """Record an operational actor failure without mixing it into market PnL."""
        if not label:
            return
        self._record_execution_status(
            str(label),
            regime,
            status="rejected",
            count_signal=True,
        )

    def record_execution_outcome(
        self,
        signal: Signal,
        *,
        status: str,
        reason: str = "",
    ) -> None:
        """Attribute BLOCKED/REJECTED/PENDING separately from market PnL."""
        normalized = str(status or "").strip().lower()
        if normalized not in {"blocked", "rejected", "pending"}:
            return
        labels = self._labels_for_signal(signal)
        self._record_signal_for_labels(signal, labels)
        for label in labels:
            self._record_execution_status(
                label,
                signal.regime,
                status=normalized,
                count_signal=False,
                sym=signal.sym,
            )

    def seed_open_position(
        self,
        *,
        labels: List[str],
        regime: Regime,
        sym: str,
        side: str,
        entry_price: float,
        qty: float,
        fee_open: float = 0.0,
        funding_open: float = 0.0,
        signal_id: int = 0,
        position_scope: str = "",
        count_entry: bool = False,
    ) -> None:
        """Seed an already-open real position for later close-PnL accounting."""
        normalized_labels: List[str] = []
        for label in labels:
            item = str(label or "").strip()
            if item and item not in normalized_labels:
                normalized_labels.append(item)
        symbol = str(sym or "").upper()
        if not normalized_labels or not symbol:
            return
        side_key = str(side or "").lower()
        if side_key not in {"long", "short"}:
            return
        try:
            entry = float(entry_price)
            position_qty = float(qty)
        except (TypeError, ValueError):
            return
        if entry <= 0.0 or position_qty <= 0.0:
            return
        for label in normalized_labels:
            if count_entry:
                for state in self._states_for_update(label, regime, symbol):
                    state.entries += 1
            self._open[(position_scope, label, symbol)] = _OpenPosition(
                signal_id=int(signal_id or 0),
                label=label,
                regime=regime,
                side=side_key,
                entry_price=entry,
                qty=position_qty,
                fee_open=float(fee_open or 0.0),
                funding_open=float(funding_open or 0.0),
            )
            self._invalidate_label(label)

    def update_from_trade(self, trade: Trade, signal: Signal) -> None:
        """Главный entry point. Обновляет метрики на основании trade.

        Сигнал передаётся явно (вместо лукапа в EventLog) — pure API,
        не зависит от внешних регистров.
        """
        if trade.signal_id != signal.id:
            raise ValueError(
                f"Trade.signal_id ({trade.signal_id}) does not match "
                f"Signal.id ({signal.id})"
            )

        # Учитываем signal как increment если ещё не учли
        labels = self._labels_for_signal(signal, trade=trade)
        self._record_signal_for_labels(signal, labels)

        regime = signal.regime

        # Open vs close
        if signal.action.is_open:
            for label in labels:
                self._record_open(label, regime, trade, signal)
        elif signal.action.is_close:
            for label in labels:
                self._record_close(label, regime, trade, signal)
        # action.is_hold → ничего не делаем

    # ── Запросы ─────────────────────────────────────────────────────

    def get(
        self,
        label: str,
        regime: Optional[Regime] = None,
        *,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Metrics:
        """Возвращает immutable Metrics.

        regime=None → агрегат по всем регимам.
        """
        if not label:
            return Metrics.empty()
        if self._context_requested(exchange=exchange, symbol=symbol):
            return self._get_context_metrics(
                label,
                regime=regime,
                exchange=exchange,
                symbol=symbol,
            )
        if regime is not None:
            state = self._state.get((label, regime))
            if state is None and _uses_neutral_seed(regime):
                state = self._state.get((label, Regime.NEUTRAL))
            return state.snapshot_as_metrics() if state else Metrics.empty()
        # Агрегат по всем регимам
        cached = self._aggregate_cache.get(label)
        if cached is not None:
            return cached
        metrics = self._aggregate_metrics(label)
        self._aggregate_cache[label] = metrics
        return metrics

    def per_regime_for_label(
        self,
        label: str,
        *,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Mapping[Regime, Metrics]:
        """Полная карта regime → Metrics для одного label.

        Используется QuarantineManager для is_locally_proven /
        is_hopeless_in_all_regimes.
        """
        if self._context_requested(exchange=exchange, symbol=symbol):
            return self._per_regime_context_for_label(
                label,
                exchange=exchange,
                symbol=symbol,
            )
        out: Dict[Regime, Metrics] = {}
        for (lbl, reg), state in self._state.items():
            if lbl == label:
                out[reg] = state.snapshot_as_metrics()
        return out

    def equity_curve(
        self,
        label: str,
        regime: Optional[Regime] = None,
        *,
        limit: Optional[int] = None,
    ) -> List[float]:
        if not label:
            return []
        if regime is not None:
            state = self._state.get((label, regime))
            if state is None and _uses_neutral_seed(regime):
                state = self._state.get((label, Regime.NEUTRAL))
            curve = list(state.equity_curve) if state is not None else []
        else:
            curve = list(self._equity_curves.get(label) or [])
        if limit is not None and int(limit) > 0:
            curve = curve[-int(limit):]
        return [round(float(item), 10) for item in curve]

    def all_labels(self) -> List[str]:
        labels = {label for (label, _) in self._state.keys()}
        labels.update(label for (_, _, _, label) in self._context_state.keys())
        return sorted(labels)

    def top_k_for_regime(
        self,
        regime: Regime,
        k: int,
        *,
        scorer,                       # Callable[[Metrics, Regime], float]
        exclude: Optional[set] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Tuple[str, float, Metrics]]:
        """Top-k labels для регима, отсортированных по убыванию score.

        scorer обычно = panteon_v2.scoring.regime_score (или замыкание
        с другим ScoringConfig).
        """
        exclude = exclude or set()
        candidates: List[Tuple[str, float, Metrics]] = []
        if self._context_requested(exchange=exchange, symbol=symbol):
            for label in self.all_labels():
                if label in exclude:
                    continue
                metrics = self.get(
                    label,
                    regime=regime,
                    exchange=exchange,
                    symbol=symbol,
                )
                if not metrics.has_data:
                    continue
                score = scorer(metrics, regime)
                candidates.append((label, score, metrics))
            candidates.sort(key=lambda t: -t[1])
            return candidates[: max(0, int(k))]
        if _uses_neutral_seed(regime):
            for label in sorted({label for (label, _) in self._state.keys()}):
                if label in exclude:
                    continue
                metrics = self.get(label, regime=regime)
                if not metrics.has_data:
                    continue
                score = scorer(metrics, regime)
                candidates.append((label, score, metrics))
            candidates.sort(key=lambda t: -t[1])
            return candidates[: max(0, int(k))]
        seen_labels = set()
        for (label, reg), state in self._state.items():
            if reg != regime or label in seen_labels or label in exclude:
                continue
            seen_labels.add(label)
            metrics = state.snapshot_as_metrics()
            score = scorer(metrics, regime)
            candidates.append((label, score, metrics))
        candidates.sort(key=lambda t: -t[1])
        return candidates[: max(0, int(k))]

    # ── Persistence ─────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Сериализуемый снимок состояния (для save/restore)."""
        return {
            "trade_fraction": self._trade_fraction,
            "exchange_scope": self._exchange_scope,
            "state": {
                f"{label}|{regime.label}": _state_to_dict(s)
                for (label, regime), s in self._state.items()
            },
            "context_state": {
                _context_key_to_text(exchange, symbol, label, regime): _state_to_dict(s)
                for (exchange, symbol, regime, label), s in self._context_state.items()
            },
            "open": {
                _open_key_to_text(scope, label, sym): _open_to_dict(p)
                for (scope, label, sym), p in self._open.items()
            },
            "seen_signal_ids": sorted(self._seen_signal_ids),
            "equity_curves": {
                label: list(curve)
                for label, curve in sorted(self._equity_curves.items())
            },
        }

    def restore(self, snapshot: dict) -> None:
        """Восстановление из snapshot()."""
        self._trade_fraction = float(snapshot.get("trade_fraction", _DEFAULT_TRADE_FRACTION))
        snapshot_exchange_scope = _normalize_exchange_scope(
            snapshot.get("exchange_scope", "")
        )
        if snapshot_exchange_scope != _DEFAULT_EXCHANGE_SCOPE:
            self._exchange_scope = snapshot_exchange_scope
        self._state.clear()
        self._context_state.clear()
        self._context_bootstrap_state.clear()
        self._aggregate_cache.clear()
        self._context_aggregate_cache.clear()
        for key, payload in (snapshot.get("state") or {}).items():
            label, regime_str = key.split("|", 1)
            regime = Regime.from_string(regime_str)
            self._state[(label, regime)] = _state_from_dict(payload)
        self._context_bootstrap_state = {
            key: _clone_state(state)
            for key, state in self._state.items()
        }
        for key, payload in (snapshot.get("context_state") or {}).items():
            exchange, symbol, label, regime = _context_key_from_text(key)
            self._context_state[(exchange, symbol, regime, label)] = _state_from_dict(payload)
        self._open.clear()
        for key, payload in (snapshot.get("open") or {}).items():
            scope, label, sym = _open_key_from_text(key)
            self._open[(scope, label, sym)] = _open_from_dict(payload)
        self._seen_signal_ids = set(int(x) for x in snapshot.get("seen_signal_ids", []))
        self._equity_curves = {
            str(label): [float(item) for item in curve]
            for label, curve in dict(snapshot.get("equity_curves") or {}).items()
            if isinstance(curve, list)
        }
        if not self._equity_curves:
            self._equity_curves = self._aggregate_equity_curves_from_states()

    # ── Внутренние helper-методы ───────────────────────────────────

    def _get_or_create(self, label: str, regime: Regime) -> _LabelRegimeState:
        key = (label, regime)
        self._invalidate_label(label)
        if key not in self._state:
            self._state[key] = _LabelRegimeState()
        return self._state[key]

    def _get_or_create_context(
        self,
        label: str,
        regime: Regime,
        sym: str,
        *,
        exchange: Optional[str] = None,
    ) -> Optional[_LabelRegimeState]:
        symbol = _normalize_symbol_scope(sym)
        if not symbol:
            return None
        exchange_scope = _normalize_exchange_scope(exchange or self._exchange_scope)
        key = (exchange_scope, symbol, regime, label)
        self._invalidate_label(label)
        if key not in self._context_state:
            base = self._context_bootstrap_state.get((label, regime))
            self._context_state[key] = (
                _clone_state(base) if base is not None else _LabelRegimeState()
            )
        return self._context_state[key]

    def _states_for_update(
        self,
        label: str,
        regime: Regime,
        sym: str,
    ) -> List[_LabelRegimeState]:
        states = [self._get_or_create(label, regime)]
        context_state = self._get_or_create_context(label, regime, sym)
        if context_state is not None:
            states.append(context_state)
        return states

    def _invalidate_label(self, label: str) -> None:
        self._aggregate_cache.pop(label, None)
        stale = [key for key in self._context_aggregate_cache if key[0] == label]
        for key in stale:
            self._context_aggregate_cache.pop(key, None)

    @staticmethod
    def _labels_from_signal(signal: Signal) -> List[str]:
        """Извлекаем оба label-а: agent и player.

        Если они совпадают (single-agent player), возвращаем один.
        """
        labels = []
        if signal.by_agent:
            labels.append(signal.by_agent)
        if signal.by_player and signal.by_player not in labels:
            labels.append(signal.by_player)
        return labels

    def _record_signal_for_labels(self, signal: Signal, labels: List[str]) -> None:
        if signal.id in self._seen_signal_ids:
            return
        self._seen_signal_ids.add(signal.id)
        for label in labels:
            for state in self._states_for_update(label, signal.regime, signal.sym):
                state.signals += 1

    def _labels_for_signal(self, signal: Signal, *, trade: Optional[Trade] = None) -> List[str]:
        if self._is_opener_attributed_close(signal):
            sym = trade.sym if trade is not None else signal.sym
            opener_labels = self._open_labels_for(signal.position_scope, sym)
            if opener_labels:
                return opener_labels
        return self._labels_from_signal(signal)

    @staticmethod
    def _is_cash_flat_close(signal: Signal) -> bool:
        return (
            signal.action.is_close
            and (
                signal.by_agent == _CASH_FLAT_AGENT
                or signal.by_player == _NO_TRADE_PLAYER
            )
        )

    @staticmethod
    def _is_opener_attributed_close(signal: Signal) -> bool:
        if not signal.action.is_close:
            return False
        if signal.by_agent in _OPENER_ATTRIBUTED_CLOSE_AGENTS:
            return True
        return signal.by_player == _NO_TRADE_PLAYER

    def _open_labels_for(self, position_scope: str, sym: str) -> List[str]:
        labels: List[str] = []
        for scope, label, open_sym in self._open.keys():
            if scope == position_scope and open_sym == sym and label not in labels:
                labels.append(label)
        return labels

    def _record_execution_status(
        self,
        label: str,
        regime: Regime,
        *,
        status: str,
        count_signal: bool,
        sym: str = "",
    ) -> None:
        for state in self._states_for_update(label, regime, sym):
            if count_signal:
                state.signals += 1
            if status == "blocked":
                state.blocked_signals += 1
            elif status == "rejected":
                state.rejected_signals += 1
            elif status == "pending":
                state.pending_signals += 1
            state.execution_failures += 1

    def _record_open(
        self,
        label: str,
        regime: Regime,
        trade: Trade,
        signal: Signal,
    ) -> None:
        for state in self._states_for_update(label, regime, trade.sym):
            state.entries += 1
        # Запоминаем открытую позицию
        self._open[(signal.position_scope, label, trade.sym)] = _OpenPosition(
            signal_id=signal.id,
            label=label,
            regime=regime,
            side=trade.side,
            entry_price=trade.fill_price,
            qty=trade.qty,
            fee_open=trade.fee,
            funding_open=trade.funding,
            open_action=signal.action.name,
        )

    def _record_close(
        self,
        label: str,
        regime: Regime,
        trade: Trade,
        signal: Signal,
    ) -> None:
        key = (signal.position_scope, label, trade.sym)
        opened = self._open.get(key)
        # Считаем close в любом случае (для signal counter etc)
        states = self._states_for_update(label, regime, trade.sym)
        if (
            self._strict_close_action_family
            and opened is not None
            and not _close_matches_open_action(signal.action, opened.open_action)
        ):
            return
        for state in states:
            state.closed_trades += 1
        if opened is None:
            # Пытаемся закрыть несуществующую позицию — записываем close,
            # но без realized PnL (не знаем entry). В v1 такое случалось.
            return
        # Realized PnL pct (long: (exit - entry)/entry; short: наоборот).
        quantity_limited_close = _is_quantity_limited_close(signal)
        close_qty = (
            min(float(trade.qty), float(opened.qty))
            if quantity_limited_close
            else float(opened.qty)
        )
        if close_qty <= 0.0:
            return
        open_fraction = close_qty / float(opened.qty)
        trade_fraction = close_qty / float(trade.qty) if quantity_limited_close else 1.0
        fee_open = opened.fee_open * open_fraction
        funding_open = opened.funding_open * open_fraction
        fee_close = trade.fee * trade_fraction
        funding_close = trade.funding * trade_fraction
        if opened.side == "long":
            return_pct = (trade.fill_price - opened.entry_price) / opened.entry_price
        else:
            return_pct = (opened.entry_price - trade.fill_price) / opened.entry_price
        # Учитываем фракцию + fees
        entry_notional = max(opened.entry_price * close_qty, 1e-12)
        gross_pnl = return_pct * self._trade_fraction
        fee_total_pct = (
            (fee_open + fee_close) / entry_notional * self._trade_fraction
        )
        funding_total_pct = (
            (funding_open + funding_close)
            / entry_notional
            * self._trade_fraction
        )
        net_pnl = gross_pnl - fee_total_pct - funding_total_pct
        net_pnl_pct = net_pnl * 100.0  # переводим в %

        # Обновляем cumulative PnL и returns
        for state in states:
            _apply_realized_close_to_state(
                state,
                net_pnl=net_pnl,
                net_pnl_pct=net_pnl_pct,
                gross_pnl_pct=gross_pnl * 100.0,
                fee_pct=fee_total_pct * 100.0,
                funding_pct=funding_total_pct * 100.0,
            )
        aggregate_curve = self._equity_curves.setdefault(label, [100.0])
        aggregate_curve.append(aggregate_curve[-1] * (1.0 + net_pnl))
        # Wins+Losses == closed_trades (плюс ноль). Ноль не считаем ни win, ни loss
        # — это редкий случай idealized fill.

        # Обновляем equity-curve и max_dd для этой (label, regime) пары.
        # Равитет переменная — стартует с 1.0 = 100%, изменяется на
        # net_pnl (доля). +1% = 1.01.
        remaining_qty = float(opened.qty) - close_qty
        if not quantity_limited_close or remaining_qty <= 1e-12:
            self._open.pop(key, None)
        else:
            self._open[key] = replace(
                opened,
                qty=remaining_qty,
                fee_open=opened.fee_open - fee_open,
                funding_open=opened.funding_open - funding_open,
            )

    def _aggregate_metrics(self, label: str) -> Metrics:
        """Сумма метрик по всем регимам для одного label."""
        agg_pnl = 0.0
        agg_gross = agg_fee = agg_funding = 0.0
        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        all_returns: List[float] = []
        agg_dd = 0.0
        any_data = False
        for (lbl, _), state in self._state.items():
            if lbl != label:
                continue
            any_data = True
            agg_pnl += state.pnl_pct
            agg_gross += state.pnl_gross_pct
            agg_fee += state.fee_pct
            agg_funding += state.funding_pct
            closed += state.closed_trades
            entries += state.entries
            signals += state.signals
            wins += state.wins
            losses += state.losses
            blocked += state.blocked_signals
            rejected += state.rejected_signals
            pending += state.pending_signals
            execution_failures += state.execution_failures
            all_returns.extend(state.returns)
            agg_dd = max(agg_dd, state.max_dd_pct)
        if not any_data:
            return Metrics.empty()
        return Metrics(
            pnl_pct=agg_pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=_sharpe(all_returns),
            max_dd_pct=agg_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
            pnl_gross_pct=agg_gross,
            fee_pct=agg_fee,
            funding_pct=agg_funding,
        )


# ────────────────────────────────────────────────────────────────────
    def _aggregate_equity_curves_from_states(self) -> Dict[str, List[float]]:
        curves: Dict[str, List[float]] = {}
        for (label, _regime), state in self._state.items():
            curve = curves.setdefault(label, [100.0])
            values = list(state.equity_curve)
            for idx in range(1, len(values)):
                previous = curve[-1]
                state_previous = float(values[idx - 1])
                if previous <= 0.0 or state_previous <= 0.0:
                    continue
                curve.append(previous * (float(values[idx]) / state_previous))
        return curves

    @staticmethod
    def _context_requested(
        *,
        exchange: Optional[str],
        symbol: Optional[str],
    ) -> bool:
        return bool(str(exchange or "").strip() or str(symbol or "").strip())

    def _get_context_metrics(
        self,
        label: str,
        *,
        regime: Optional[Regime],
        exchange: Optional[str],
        symbol: Optional[str],
    ) -> Metrics:
        exchange_scope = _normalize_exchange_scope(exchange or self._exchange_scope)
        symbol_scope = _normalize_symbol_scope(symbol)
        if regime is not None and symbol_scope:
            state = self._context_state.get((exchange_scope, symbol_scope, regime, label))
            if state is None:
                state = self._bootstrap_context_state(
                    label,
                    regime,
                    exchange_scope,
                    symbol_scope,
                )
            return state.snapshot_as_metrics() if state else Metrics.empty()
        cache_key = (label, exchange_scope, symbol_scope)
        if regime is None:
            cached = self._context_aggregate_cache.get(cache_key)
            if cached is not None:
                return cached
        metrics = self._aggregate_context_metrics(
            label,
            exchange=exchange_scope,
            symbol=symbol_scope,
            regime=regime,
        )
        if regime is None:
            self._context_aggregate_cache[cache_key] = metrics
        return metrics

    def _bootstrap_context_state(
        self,
        label: str,
        regime: Regime,
        exchange: str,
        symbol: str,
    ) -> Optional[_LabelRegimeState]:
        key = (exchange, symbol, regime, label)
        state = self._context_state.get(key)
        if state is not None:
            return state
        if exchange != self._exchange_scope:
            return None
        self._ensure_context_bootstrap_baseline()
        base = self._context_bootstrap_state.get((label, regime))
        if base is None and _uses_neutral_seed(regime):
            base = self._context_bootstrap_state.get((label, Regime.NEUTRAL))
        if base is None or not base.snapshot_as_metrics().has_data:
            return None
        self._context_state[key] = _clone_state(base)
        self._context_aggregate_cache.pop((label, exchange, symbol), None)
        return self._context_state[key]

    def _ensure_context_bootstrap_baseline(self) -> None:
        if self._context_bootstrap_state or not self._state:
            return
        self._context_bootstrap_state = {
            key: _clone_state(state)
            for key, state in self._state.items()
        }

    def _aggregate_context_metrics(
        self,
        label: str,
        *,
        exchange: str,
        symbol: str,
        regime: Optional[Regime],
    ) -> Metrics:
        return _metrics_from_states(
            state
            for (ex, sym, reg, lbl), state in self._context_state.items()
            if lbl == label
            and ex == exchange
            and (not symbol or sym == symbol)
            and (regime is None or reg == regime)
        )

    def _per_regime_context_for_label(
        self,
        label: str,
        *,
        exchange: Optional[str],
        symbol: Optional[str],
    ) -> Mapping[Regime, Metrics]:
        exchange_scope = _normalize_exchange_scope(exchange or self._exchange_scope)
        symbol_scope = _normalize_symbol_scope(symbol)
        out: Dict[Regime, Metrics] = {}
        for regime in Regime:
            metrics = self.get(
                label,
                regime=regime,
                exchange=exchange_scope,
                symbol=symbol_scope,
            )
            if metrics.has_data:
                out[regime] = metrics
        return out


# (De)serialization helpers
# ────────────────────────────────────────────────────────────────────


def _state_to_dict(s: _LabelRegimeState) -> dict:
    return {
        "closed_trades": s.closed_trades,
        "entries":       s.entries,
        "signals":       s.signals,
        "wins":          s.wins,
        "losses":        s.losses,
        "pnl_pct":       s.pnl_pct,
        "pnl_gross_pct": s.pnl_gross_pct,
        "fee_pct":       s.fee_pct,
        "funding_pct":   s.funding_pct,
        "returns":       list(s.returns),
        "equity":        s.equity,
        "equity_curve":  list(s.equity_curve),
        "peak":          s.peak,
        "max_dd_pct":    s.max_dd_pct,
        "blocked_signals": s.blocked_signals,
        "rejected_signals": s.rejected_signals,
        "pending_signals": s.pending_signals,
        "execution_failures": s.execution_failures,
    }


def _clone_state(s: _LabelRegimeState) -> _LabelRegimeState:
    return _state_from_dict(_state_to_dict(s))


def _normalize_exchange_scope(value: object) -> str:
    clean = str(value or "").strip().upper()
    return clean or _DEFAULT_EXCHANGE_SCOPE


def _normalize_symbol_scope(value: object) -> str:
    return str(value or "").strip().upper()


def _context_key_to_text(
    exchange: str,
    symbol: str,
    label: str,
    regime: Regime,
) -> str:
    return f"{exchange}|{symbol}|{label}|{regime.label}"


def _context_key_from_text(key: str) -> Tuple[str, str, str, Regime]:
    exchange, symbol, rest = str(key).split("|", 2)
    label, regime_str = rest.rsplit("|", 1)
    return (
        _normalize_exchange_scope(exchange),
        _normalize_symbol_scope(symbol),
        label,
        Regime.from_string(regime_str),
    )


def _apply_realized_close_to_state(
    state: _LabelRegimeState,
    *,
    net_pnl: float,
    net_pnl_pct: float,
    gross_pnl_pct: float,
    fee_pct: float,
    funding_pct: float,
) -> None:
    state.pnl_pct += net_pnl_pct
    state.pnl_gross_pct += gross_pnl_pct
    state.fee_pct += fee_pct
    state.funding_pct += funding_pct
    state.returns.append(net_pnl_pct)
    if net_pnl_pct > 0:
        state.wins += 1
    elif net_pnl_pct < 0:
        state.losses += 1
    state.equity *= (1.0 + net_pnl)
    state.equity_curve.append(state.equity * 100.0)
    if state.equity > state.peak:
        state.peak = state.equity
    dd = (state.peak - state.equity) / max(state.peak, 1e-12) * 100.0
    if dd > state.max_dd_pct:
        state.max_dd_pct = dd


def _metrics_from_states(states) -> Metrics:
    agg_pnl = 0.0
    agg_gross = agg_fee = agg_funding = 0.0
    closed = entries = signals = wins = losses = 0
    blocked = rejected = pending = execution_failures = 0
    all_returns: List[float] = []
    agg_dd = 0.0
    any_data = False
    for state in states:
        any_data = True
        agg_pnl += state.pnl_pct
        agg_gross += state.pnl_gross_pct
        agg_fee += state.fee_pct
        agg_funding += state.funding_pct
        closed += state.closed_trades
        entries += state.entries
        signals += state.signals
        wins += state.wins
        losses += state.losses
        blocked += state.blocked_signals
        rejected += state.rejected_signals
        pending += state.pending_signals
        execution_failures += state.execution_failures
        all_returns.extend(state.returns)
        agg_dd = max(agg_dd, state.max_dd_pct)
    if not any_data:
        return Metrics.empty()
    return Metrics(
        pnl_pct=agg_pnl,
        closed_trades=closed,
        entries=entries,
        signals=signals,
        wins=wins,
        losses=losses,
        sharpe=_sharpe(all_returns),
        max_dd_pct=agg_dd,
        blocked_signals=blocked,
        rejected_signals=rejected,
        pending_signals=pending,
        execution_failures=execution_failures,
        pnl_gross_pct=agg_gross,
        fee_pct=agg_fee,
        funding_pct=agg_funding,
    )


def _is_intentional_partial_close(signal: Signal) -> bool:
    if not signal.action.is_close:
        return False
    try:
        return 0.0 < float(getattr(signal, "close_fraction", 1.0) or 1.0) < 1.0
    except (TypeError, ValueError):
        return False


def _is_quantity_limited_close(signal: Signal) -> bool:
    return signal.action == Action.SPOT_SELL_ALL or _is_intentional_partial_close(signal)


def _state_from_dict(d: dict) -> _LabelRegimeState:
    s = _LabelRegimeState()
    s.closed_trades = int(d.get("closed_trades", 0))
    s.entries       = int(d.get("entries", 0))
    s.signals       = int(d.get("signals", 0))
    s.wins          = int(d.get("wins", 0))
    s.losses        = int(d.get("losses", 0))
    s.pnl_pct       = float(d.get("pnl_pct", 0.0))
    s.pnl_gross_pct = float(d.get("pnl_gross_pct", s.pnl_pct))
    s.fee_pct       = float(d.get("fee_pct", 0.0))
    s.funding_pct   = float(d.get("funding_pct", 0.0))
    s.returns       = [float(r) for r in d.get("returns", [])]
    s.equity        = float(d.get("equity", 1.0))
    curve = d.get("equity_curve")
    if isinstance(curve, list) and curve:
        s.equity_curve = [float(item) for item in curve]
    elif s.returns:
        values = [100.0]
        equity = 1.0
        for item in s.returns:
            equity *= 1.0 + (float(item) / 100.0)
            values.append(equity * 100.0)
        s.equity_curve = values
    elif abs(s.equity - 1.0) > 1e-12:
        s.equity_curve = [100.0, s.equity * 100.0]
    s.peak          = float(d.get("peak", 1.0))
    s.max_dd_pct    = float(d.get("max_dd_pct", 0.0))
    s.blocked_signals = int(d.get("blocked_signals", 0))
    s.rejected_signals = int(d.get("rejected_signals", 0))
    s.pending_signals = int(d.get("pending_signals", 0))
    s.execution_failures = int(d.get("execution_failures", 0))
    return s


def _open_to_dict(p: _OpenPosition) -> dict:
    return {
        "signal_id":   p.signal_id,
        "label":       p.label,
        "regime":      p.regime.label,
        "side":        p.side,
        "entry_price": p.entry_price,
        "qty":         p.qty,
        "fee_open":    p.fee_open,
        "funding_open": p.funding_open,
        "open_action": p.open_action,
    }


def _open_key_to_text(scope: str, label: str, sym: str) -> str:
    if scope:
        return f"{scope}|{label}|{sym}"
    return f"{label}|{sym}"


def _open_key_from_text(key: str) -> Tuple[str, str, str]:
    parts = key.split("|", 2)
    if len(parts) == 2:
        label, sym = parts
        return "", label, sym
    if len(parts) == 3:
        scope, label, sym = parts
        return scope, label, sym
    return "", key, ""


def _open_from_dict(d: dict) -> _OpenPosition:
    return _OpenPosition(
        signal_id=int(d["signal_id"]),
        label=str(d["label"]),
        regime=Regime.from_string(str(d["regime"])),
        side=str(d["side"]),
        entry_price=float(d["entry_price"]),
        qty=float(d["qty"]),
        fee_open=float(d["fee_open"]),
        funding_open=float(d.get("funding_open", 0.0)),
        open_action=str(d.get("open_action", "")),
    )


_SPOT_OPEN_ACTION_NAMES = frozenset({
    Action.SPOT_BUY_HALF.name,
    Action.SPOT_BUY_FULL.name,
})
_FUTURE_OPEN_ACTION_NAMES = frozenset({
    Action.FUT_LONG_HALF.name,
    Action.FUT_LONG_FULL.name,
    Action.FUT_SHORT_HALF.name,
    Action.FUT_SHORT_FULL.name,
})


def _close_matches_open_action(close_action: Action, open_action: str) -> bool:
    open_action_name = str(open_action or "")
    if not open_action_name:
        return True
    if close_action == Action.SPOT_SELL_ALL:
        return open_action_name in _SPOT_OPEN_ACTION_NAMES
    if close_action == Action.FUT_CLOSE_ALL:
        return open_action_name in _FUTURE_OPEN_ACTION_NAMES
    return True
