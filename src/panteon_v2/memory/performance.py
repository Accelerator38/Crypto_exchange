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
from dataclasses import dataclass, field
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


# ────────────────────────────────────────────────────────────────────
# PerformanceMemory — публичный API
# ────────────────────────────────────────────────────────────────────


# Дефолтный фракционный размер позиции (как в v1 — обычно ~10% капитала).
# Используется чтобы абсолютный pnl_pct совпадал с тем, что видит
# Пантеон в реальной торговле. Конфигурируется через __init__.
_DEFAULT_TRADE_FRACTION = 0.10
_CASH_FLAT_AGENT = "CashFlat"
_NO_TRADE_PLAYER = "NoTrade"


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

    def __init__(self, *, trade_fraction: float = _DEFAULT_TRADE_FRACTION):
        if not 0 < trade_fraction <= 1.0:
            raise ValueError(f"trade_fraction must be in (0, 1], got {trade_fraction}")
        self._trade_fraction = float(trade_fraction)
        # (label, regime) → state
        self._state: Dict[Tuple[str, Regime], _LabelRegimeState] = {}
        # Открытые позиции по (label, sym) — нужно знать для close.
        # Один label может иметь только одну открытую позицию по sym
        # (общепринятое допущение, как в v1).
        self._open: Dict[Tuple[str, str, str], _OpenPosition] = {}
        # Записанные signals — для bookkeeping (avoid double-counting if
        # update_from_signal вызывается отдельно):
        self._seen_signal_ids: set = set()
        self._aggregate_cache: Dict[str, Metrics] = {}

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
            )

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
    ) -> Metrics:
        """Возвращает immutable Metrics.

        regime=None → агрегат по всем регимам.
        """
        if not label:
            return Metrics.empty()
        if regime is not None:
            state = self._state.get((label, regime))
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
    ) -> Mapping[Regime, Metrics]:
        """Полная карта regime → Metrics для одного label.

        Используется QuarantineManager для is_locally_proven /
        is_hopeless_in_all_regimes.
        """
        out: Dict[Regime, Metrics] = {}
        for (lbl, reg), state in self._state.items():
            if lbl == label:
                out[reg] = state.snapshot_as_metrics()
        return out

    def all_labels(self) -> List[str]:
        return sorted({label for (label, _) in self._state.keys()})

    def top_k_for_regime(
        self,
        regime: Regime,
        k: int,
        *,
        scorer,                       # Callable[[Metrics, Regime], float]
        exclude: Optional[set] = None,
    ) -> List[Tuple[str, float, Metrics]]:
        """Top-k labels для регима, отсортированных по убыванию score.

        scorer обычно = panteon_v2.scoring.regime_score (или замыкание
        с другим ScoringConfig).
        """
        exclude = exclude or set()
        candidates: List[Tuple[str, float, Metrics]] = []
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
            "state": {
                f"{label}|{regime.label}": _state_to_dict(s)
                for (label, regime), s in self._state.items()
            },
            "open": {
                _open_key_to_text(scope, label, sym): _open_to_dict(p)
                for (scope, label, sym), p in self._open.items()
            },
            "seen_signal_ids": sorted(self._seen_signal_ids),
        }

    def restore(self, snapshot: dict) -> None:
        """Восстановление из snapshot()."""
        self._trade_fraction = float(snapshot.get("trade_fraction", _DEFAULT_TRADE_FRACTION))
        self._state.clear()
        self._aggregate_cache.clear()
        for key, payload in (snapshot.get("state") or {}).items():
            label, regime_str = key.split("|", 1)
            regime = Regime.from_string(regime_str)
            self._state[(label, regime)] = _state_from_dict(payload)
        self._open.clear()
        for key, payload in (snapshot.get("open") or {}).items():
            scope, label, sym = _open_key_from_text(key)
            self._open[(scope, label, sym)] = _open_from_dict(payload)
        self._seen_signal_ids = set(int(x) for x in snapshot.get("seen_signal_ids", []))

    # ── Внутренние helper-методы ───────────────────────────────────

    def _get_or_create(self, label: str, regime: Regime) -> _LabelRegimeState:
        key = (label, regime)
        self._aggregate_cache.pop(label, None)
        if key not in self._state:
            self._state[key] = _LabelRegimeState()
        return self._state[key]

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
            self._get_or_create(label, signal.regime).signals += 1

    def _labels_for_signal(self, signal: Signal, *, trade: Optional[Trade] = None) -> List[str]:
        if self._is_cash_flat_close(signal):
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
    ) -> None:
        state = self._get_or_create(label, regime)
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
        state = self._get_or_create(label, regime)
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
        )

    def _record_close(
        self,
        label: str,
        regime: Regime,
        trade: Trade,
        signal: Signal,
    ) -> None:
        key = (signal.position_scope, label, trade.sym)
        opened = self._open.pop(key, None)
        # Считаем close в любом случае (для signal counter etc)
        state = self._get_or_create(label, regime)
        state.closed_trades += 1
        if opened is None:
            # Пытаемся закрыть несуществующую позицию — записываем close,
            # но без realized PnL (не знаем entry). В v1 такое случалось.
            return
        # Realized PnL pct (long: (exit - entry)/entry; short: наоборот).
        if opened.side == "long":
            return_pct = (trade.fill_price - opened.entry_price) / opened.entry_price
        else:
            return_pct = (opened.entry_price - trade.fill_price) / opened.entry_price
        # Учитываем фракцию + fees
        entry_notional = max(opened.entry_price * opened.qty, 1e-12)
        gross_pnl = return_pct * self._trade_fraction
        fee_total_pct = (
            (opened.fee_open + trade.fee) / entry_notional * self._trade_fraction
        )
        funding_total_pct = (
            (opened.funding_open + trade.funding)
            / entry_notional
            * self._trade_fraction
        )
        net_pnl = gross_pnl - fee_total_pct - funding_total_pct
        net_pnl_pct = net_pnl * 100.0  # переводим в %

        # Обновляем cumulative PnL и returns
        state.pnl_pct += net_pnl_pct
        state.pnl_gross_pct += gross_pnl * 100.0
        state.fee_pct += fee_total_pct * 100.0
        state.funding_pct += funding_total_pct * 100.0
        state.returns.append(net_pnl_pct)
        if net_pnl_pct > 0:
            state.wins += 1
        elif net_pnl_pct < 0:
            state.losses += 1
        # Wins+Losses == closed_trades (плюс ноль). Ноль не считаем ни win, ни loss
        # — это редкий случай idealized fill.

        # Обновляем equity-curve и max_dd для этой (label, regime) пары.
        # Равитет переменная — стартует с 1.0 = 100%, изменяется на
        # net_pnl (доля). +1% = 1.01.
        state.equity *= (1.0 + net_pnl)
        if state.equity > state.peak:
            state.peak = state.equity
        dd = (state.peak - state.equity) / max(state.peak, 1e-12) * 100.0
        if dd > state.max_dd_pct:
            state.max_dd_pct = dd

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
        "peak":          s.peak,
        "max_dd_pct":    s.max_dd_pct,
        "blocked_signals": s.blocked_signals,
        "rejected_signals": s.rejected_signals,
        "pending_signals": s.pending_signals,
        "execution_failures": s.execution_failures,
    }


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
    )
