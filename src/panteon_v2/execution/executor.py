"""TradeExecutor — единственный путь сигнала на биржу.

Pipeline:
    Signal
    → SymbolHealthMonitor.is_blocked? → BLOCKED
    → RiskLimits.evaluate?            → BLOCKED
    → Exchange.send_order             → FILLED / PENDING / REJECTED
    → on success:
        PositionTracker.on_open/close → events
        PerformanceMemory.update_from_trade
        SymbolHealthMonitor.record_success
        EmitEvents: OrderSent, OrderFilled, PositionOpened/Closed
    → on rejection:
        SymbolHealthMonitor.record_pending_failure
        EmitEvents: OrderSent, OrderRejected
    → on pending:
        SymbolHealthMonitor.record_pending_failure
        EmitEvents: OrderSent (without fill)

Гарантии:
  • Каждый Signal с is_open проверяется на is_blocked ДО send_order.
  • Каждая Trade связана с Signal через signal_id.
  • Каждое успешное исполнение → PerformanceMemory обновляется один раз.
  • Любая ветка эмиттит events для AttributionLedger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..attribution.events import (
    MemoryUpdateFailed,
    OrderFilled,
    OrderRejected,
    OrderSent,
    SymbolBlocked,
)
from ..attribution.event_log import EventLog
from ..domain.types import Action, Signal, Trade
from ..memory import PerformanceMemory
from .exchange import Exchange, OrderResult, OrderStatus
from .order_ledger import OrderLedger
from .position_tracker import PositionTracker
from .risk_limits import RiskLimits
from .symbol_health import SymbolHealthMonitor


# ────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────


class ExecutionStatus(Enum):
    """Итоговый статус попытки исполнения."""

    FILLED   = "filled"      # ордер исполнен, позиция учтена
    PENDING  = "pending"     # биржа приняла, но fill ещё нет
    REJECTED = "rejected"    # биржа отвергла
    BLOCKED  = "blocked"     # отвергнут pre-flight (health/risk)


@dataclass(frozen=True)
class ExecutionResult:
    """Результат TradeExecutor.execute()."""

    status:     ExecutionStatus
    signal:     Signal
    trade:      Optional[Trade] = None
    reason:     str = ""

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.FILLED


# ────────────────────────────────────────────────────────────────────
# TradeExecutor
# ────────────────────────────────────────────────────────────────────


class TradeExecutor:
    """Единственный путь Signal → Exchange → Memory + Events.

    Параметры (всё через DI):
      • exchange        — Exchange Protocol (real или Fake)
      • health          — SymbolHealthMonitor
      • risk_limits     — RiskLimits
      • position_tracker — PositionTracker
      • perf            — PerformanceMemory
      • event_log       — EventLog
    """

    def __init__(
        self,
        *,
        exchange:         Exchange,
        health:           SymbolHealthMonitor,
        risk_limits:      RiskLimits,
        position_tracker: PositionTracker,
        perf:             PerformanceMemory,
        event_log:        EventLog,
        order_ledger:     Optional[OrderLedger] = None,
    ):
        self._exchange = exchange
        self._health = health
        self._risk = risk_limits
        self._tracker = position_tracker
        self._perf = perf
        self._log = event_log
        self._order_ledger = order_ledger or OrderLedger()

    # ── Public API ──────────────────────────────────────────────────

    def execute(
        self,
        signal: Signal,
        *,
        balance_usd: float,
    ) -> ExecutionResult:
        """Главная функция. Принимает signal, возвращает ExecutionResult.

        Не выбрасывает исключений (биржевые ошибки → REJECTED).
        """
        # 0. Hold actions — игнорируем явно
        if signal.action.is_hold:
            self._record_execution_outcome(signal, ExecutionStatus.BLOCKED, "hold action")
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                signal=signal,
                reason="hold action",
            )

        trace_id = self._trace_id(signal)

        # 1. Pre-flight: SymbolHealthMonitor (только для open)
        if signal.action.is_open and self._health.is_blocked(signal.sym):
            reason = f"symbol {signal.sym} is in health blocklist"
            self._record_execution_outcome(signal, ExecutionStatus.BLOCKED, reason)
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                signal=signal,
                reason=reason,
            )

        min_notional_for_sym = 0.0
        if signal.action.is_open:
            try:
                min_notional_for_sym = self._exchange.get_min_notional(signal.sym)
            except Exception as exc:
                self._health.record_pending_failure(signal.sym)
                self._log.emit(OrderRejected(
                    bar=signal.bar,
                    trace_id=trace_id,
                    signal_id=signal.id,
                    sym=signal.sym,
                    reason=f"exchange metadata exception: {type(exc).__name__}: {exc}",
                ))
                self._record_execution_outcome(signal, ExecutionStatus.REJECTED, str(exc))
                return ExecutionResult(
                    status=ExecutionStatus.REJECTED,
                    signal=signal,
                    reason=f"exchange metadata error: {exc}",
                )

        # 2. Pre-flight: RiskLimits
        risk_check = self._risk.evaluate(
            signal,
            balance_usd=balance_usd,
            open_positions=self._tracker.all_open(),
            min_notional_for_sym=min_notional_for_sym,
        )
        if not risk_check.allowed:
            reason = f"risk_limits: {risk_check.reason}"
            self._record_execution_outcome(signal, ExecutionStatus.BLOCKED, reason)
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                signal=signal,
                reason=reason,
            )

        # Вычисляем qty:
        #   • для open — используем risk_check.qty (нотинал × leverage)
        #   • для close — берём qty из существующей позиции
        if signal.action.is_close:
            existing = self._tracker.get(signal.sym)
            if existing is None:
                # double-check (RiskLimits должен был отфильтровать)
                self._record_execution_outcome(
                    signal,
                    ExecutionStatus.BLOCKED,
                    "no position to close (post-risk)",
                )
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    signal=signal,
                    reason="no position to close (post-risk)",
                )
            qty = existing.qty
        else:
            qty = self._quantize_open_qty(signal, risk_check.qty)
            if qty <= 0:
                self._record_execution_outcome(
                    signal,
                    ExecutionStatus.BLOCKED,
                    "risk_limits: quantized qty is below exchange minimum",
                )
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    signal=signal,
                    reason="risk_limits: quantized qty is below exchange minimum",
                )
            quantized_notional = qty * signal.price
            allowed_notional = max(float(risk_check.notional or 0.0), 0.0)
            if allowed_notional > 0 and quantized_notional > allowed_notional + 1e-9:
                cfg = self._risk.config
                upscale = quantized_notional / max(allowed_notional, 1e-12)
                can_accept_quantized_floor = (
                    quantized_notional <= float(cfg.max_notional_usd) + 1e-9
                    and upscale <= float(cfg.max_min_notional_upscale) + 1e-9
                )
                if not can_accept_quantized_floor:
                    reason = (
                        "risk_limits: quantized notional "
                        f"${quantized_notional:.2f} exceeds approved "
                        f"${allowed_notional:.2f}"
                    )
                    self._record_execution_outcome(signal, ExecutionStatus.BLOCKED, reason)
                    return ExecutionResult(
                        status=ExecutionStatus.BLOCKED,
                        signal=signal,
                        reason=reason,
                    )

        # 3. Эмитим OrderSent ДО send_order, чтобы в логе была попытка
        self._log.emit(OrderSent(
            bar=signal.bar,
            trace_id=trace_id,
            signal_id=signal.id,
            sym=signal.sym,
            action=signal.action,
            exchange_order_id="",   # ещё не знаем
        ))

        # 4. Вызываем биржу
        self._order_ledger.record_submitted(signal)
        try:
            order_result = self._exchange.send_order(signal, qty=qty)
        except Exception as exc:
            # Любая ошибка биржи → REJECTED + record_pending_failure
            self._health.record_pending_failure(signal.sym)
            self._log.emit(OrderRejected(
                bar=signal.bar,
                trace_id=trace_id,
                signal_id=signal.id,
                sym=signal.sym,
                reason=f"exchange exception: {type(exc).__name__}: {exc}",
            ))
            self._record_execution_outcome(signal, ExecutionStatus.REJECTED, str(exc))
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                signal=signal,
                reason=f"exchange error: {exc}",
            )

        # 5. Обработка результата
        if order_result.status == OrderStatus.FILLED:
            return self._handle_filled(signal, order_result, trace_id)
        elif order_result.status == OrderStatus.REJECTED:
            return self._handle_rejected(signal, order_result, trace_id)
        else:  # PENDING
            return self._handle_pending(signal, order_result, trace_id)

    # ── Handlers ────────────────────────────────────────────────────

    def recover_pending_order(self, signal: Signal, order_id: str) -> ExecutionResult:
        """Poll and finalize a previously accepted order after process restart."""
        trace_id = f"{self._trace_id(signal)}-recover"
        placeholder = OrderResult(
            status=OrderStatus.PENDING,
            signal_id=signal.id,
            sym=signal.sym,
            exchange_order_id=str(order_id or ""),
            message="restored pending order",
        )
        polled = self._poll_pending(signal, placeholder)
        if polled is None or polled.status == OrderStatus.PENDING:
            if polled is not None:
                self._order_ledger.record_result(signal, polled)
            self._record_execution_outcome(
                signal,
                ExecutionStatus.PENDING,
                "restored order is still pending",
            )
            return ExecutionResult(
                status=ExecutionStatus.PENDING,
                signal=signal,
                reason="restored order is still pending",
            )
        self._order_ledger.record_result(signal, polled)
        if polled.status == OrderStatus.FILLED:
            return self._handle_filled(signal, polled, trace_id)
        return self._handle_rejected(signal, polled, trace_id)

    def expire_stale_pending_orders(self, *, max_age_sec: float) -> int:
        """Mark accepted/submitted orders as timed out after the configured age."""
        expired = self._order_ledger.expire_pending(max_age_sec=max_age_sec)
        for record in expired:
            cancel_reason = self._cancel_pending_if_supported(record.exchange_order_id, record.sym)
            trace_id = (
                f"{record.sym}-{record.signal.bar}-{record.signal_id}-timeout"
                if record.signal is not None
                else f"{record.sym}-{record.signal_id}-timeout"
            )
            self._log.emit(OrderRejected(
                bar=(record.signal.bar if record.signal is not None else -1),
                trace_id=trace_id,
                signal_id=record.signal_id,
                sym=record.sym,
                reason=(
                    f"pending timeout: {record.exchange_order_id or record.signal_id}"
                    + (f" ({cancel_reason})" if cancel_reason else "")
                ),
            ))
            if record.signal is not None:
                self._record_execution_outcome(
                    record.signal,
                    ExecutionStatus.REJECTED,
                    "pending timeout",
                )
        return len(expired)

    def _cancel_pending_if_supported(self, order_id: str, sym: str) -> str:
        if not order_id:
            return ""
        cancel = getattr(self._exchange, "cancel_order", None)
        if not callable(cancel):
            return "cancel unsupported"
        for args in ((order_id, sym), (order_id,), (sym, order_id)):
            try:
                cancel(*args)
                return "cancel requested"
            except TypeError:
                continue
            except Exception as exc:
                return f"cancel failed: {type(exc).__name__}: {exc}"
        return "cancel signature unsupported"

    def _handle_filled(
        self,
        signal: Signal,
        result: OrderResult,
        trace_id: str,
    ) -> ExecutionResult:
        trade = result.trade
        if trade is None:
            # OrderResult.__post_init__ уже это проверил, но parano
            return self._handle_rejected(signal, result, trace_id)

        # Position tracker → events для лога
        self._order_ledger.record_result(signal, result)

        if signal.action.is_open:
            events = self._tracker.on_open(signal=signal, trade=trade)
        else:
            events = self._tracker.on_close(signal=signal, trade=trade)

        # PerformanceMemory обновляется ОДНИМ вызовом — это SSOT
        try:
            self._perf.update_from_trade(trade, signal)
        except Exception as exc:
            # Не блокируем — нам важнее зарегистрировать filled trade
            self._log.emit(MemoryUpdateFailed(
                bar=signal.bar,
                trace_id=trace_id,
                signal_id=signal.id,
                sym=signal.sym,
                reason=f"{type(exc).__name__}: {exc}",
            ))

        # Health → success
        self._health.record_success(signal.sym)

        # OrderFilled + position events
        self._log.emit(OrderFilled(
            bar=signal.bar,
            trace_id=trace_id,
            trade=trade,
        ))
        self._log.emit_many(events)

        return ExecutionResult(
            status=ExecutionStatus.FILLED,
            signal=signal,
            trade=trade,
        )

    def _handle_rejected(
        self,
        signal: Signal,
        result: OrderResult,
        trace_id: str,
    ) -> ExecutionResult:
        # Failure для health
        self._order_ledger.record_result(signal, result)
        newly_blocked = self._health.record_pending_failure(signal.sym)
        self._log.emit(OrderRejected(
            bar=signal.bar,
            trace_id=trace_id,
            signal_id=signal.id,
            sym=signal.sym,
            reason=result.message or "rejected by exchange",
        ))
        self._record_execution_outcome(
            signal,
            ExecutionStatus.REJECTED,
            result.message or "rejected by exchange",
        )
        if newly_blocked:
            status = self._health.status(signal.sym)
            self._log.emit(SymbolBlocked(
                bar=signal.bar,
                trace_id=trace_id,
                sym=signal.sym,
                failures_in_window=status.failures_in_window,
                blocked_until_ts=status.blocked_until_ts or 0.0,
                reason="pending_failures_threshold",
            ))
        return ExecutionResult(
            status=ExecutionStatus.REJECTED,
            signal=signal,
            reason=result.message,
        )

    def _handle_pending(
        self,
        signal: Signal,
        result: OrderResult,
        trace_id: str,
    ) -> ExecutionResult:
        self._order_ledger.record_result(signal, result)
        polled = self._poll_pending(signal, result)
        if polled is not None and polled.status != OrderStatus.PENDING:
            self._order_ledger.record_result(signal, polled)
            if polled.status == OrderStatus.FILLED:
                return self._handle_filled(signal, polled, trace_id)
            if polled.status == OrderStatus.REJECTED:
                return self._handle_rejected(signal, polled, trace_id)

        # Pending тоже считаем failure (как в v1) — после таймаута
        # биржа обычно его сбрасывает.
        newly_blocked = self._health.record_pending_failure(signal.sym)
        if newly_blocked:
            status = self._health.status(signal.sym)
            self._log.emit(SymbolBlocked(
                bar=signal.bar,
                trace_id=trace_id,
                sym=signal.sym,
                failures_in_window=status.failures_in_window,
                blocked_until_ts=status.blocked_until_ts or 0.0,
                reason="pending_failures_threshold",
            ))
        self._record_execution_outcome(
            signal,
            ExecutionStatus.PENDING,
            result.message or "pending",
        )
        return ExecutionResult(
            status=ExecutionStatus.PENDING,
            signal=signal,
            reason=result.message or "pending",
        )

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _trace_id(signal: Signal) -> str:
        return f"{signal.sym}-{signal.bar}-{signal.id}"

    def _record_execution_outcome(
        self,
        signal: Signal,
        status: ExecutionStatus,
        reason: str,
    ) -> None:
        try:
            self._perf.record_execution_outcome(
                signal,
                status=status.value,
                reason=reason,
            )
        except Exception as exc:
            self._log.emit(MemoryUpdateFailed(
                bar=signal.bar,
                trace_id=self._trace_id(signal),
                signal_id=signal.id,
                sym=signal.sym,
                reason=f"execution outcome memory update failed: {type(exc).__name__}: {exc}",
            ))

    def _quantize_open_qty(self, signal: Signal, qty: float) -> float:
        quantize = getattr(self._exchange, "quantize_order_qty", None)
        if not callable(quantize):
            return float(qty or 0.0)
        try:
            return float(quantize(signal, qty) or 0.0)
        except Exception:
            return 0.0

    def _poll_pending(self, signal: Signal, result: OrderResult) -> Optional[OrderResult]:
        order_id = result.exchange_order_id
        if not order_id:
            return None
        poll = getattr(self._exchange, "poll_order", None)
        if not callable(poll):
            return None
        try:
            polled = poll(order_id, signal)
        except TypeError:
            try:
                polled = poll(order_id)
            except Exception:
                return None
        except Exception:
            return None
        return polled if isinstance(polled, OrderResult) else None
