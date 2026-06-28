"""Session-local degradation gate for weak genetic agents.

The gate is deliberately separate from all-time quarantine scoring. It compares
current PerformanceMemory against a captured session baseline and force
quarantines monitored labels when live/shadow behavior degrades enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from ..domain.types import Metrics
from .performance import PerformanceMemory
from .quarantine import QuarantineManager, RecomputeResult


@dataclass(frozen=True)
class DegradationGateConfig:
    enabled: bool = True
    label_prefixes: Tuple[str, ...] = ("Genetics",)
    min_session_signals: int = 8
    min_session_closed_trades: int = 3
    max_session_loss_pct: float = 1.0
    max_session_drawdown_pct: float = 2.0
    drawdown_requires_nonpositive_pnl: bool = True
    max_execution_failure_rate: float = 0.50
    max_blocked_signal_rate: float = 0.90
    reason_prefix: str = "degradation_gate"

    def __post_init__(self) -> None:
        if self.min_session_signals < 0:
            raise ValueError("min_session_signals must be >= 0")
        if self.min_session_closed_trades < 0:
            raise ValueError("min_session_closed_trades must be >= 0")
        if self.max_session_loss_pct < 0:
            raise ValueError("max_session_loss_pct must be >= 0")
        if self.max_session_drawdown_pct < 0:
            raise ValueError("max_session_drawdown_pct must be >= 0")
        if self.max_execution_failure_rate < 0:
            raise ValueError("max_execution_failure_rate must be >= 0")
        if self.max_blocked_signal_rate < 0:
            raise ValueError("max_blocked_signal_rate must be >= 0")


@dataclass(frozen=True)
class DegradationDecision:
    label: str
    should_disable: bool
    reasons: Tuple[str, ...]
    session_metrics: Metrics
    execution_failure_rate: float = 0.0
    blocked_signal_rate: float = 0.0


class DegradationGate:
    """Disable monitored agents when current-session behavior deteriorates."""

    def __init__(self, config: Optional[DegradationGateConfig] = None):
        self._config = config or DegradationGateConfig()
        self._baseline: Dict[str, Metrics] = {}
        self._baseline_captured = False
        self._last_decisions: Tuple[DegradationDecision, ...] = ()

    @property
    def config(self) -> DegradationGateConfig:
        return self._config

    @property
    def baseline_captured(self) -> bool:
        return self._baseline_captured

    @property
    def last_decisions(self) -> Tuple[DegradationDecision, ...]:
        return self._last_decisions

    def capture_baseline(
        self,
        perf: PerformanceMemory,
        *,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        selected = labels if labels is not None else perf.all_labels()
        self._baseline = {
            str(label): perf.get(str(label))
            for label in selected
            if str(label or "")
        }
        self._baseline_captured = True

    def ensure_baseline(
        self,
        perf: PerformanceMemory,
        *,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        if not self._baseline_captured:
            self.capture_baseline(perf, labels=labels)

    def session_metrics(self, perf: PerformanceMemory, label: str) -> Metrics:
        current = perf.get(label)
        baseline = self._baseline.get(label, Metrics.empty())
        return _metrics_delta(current, baseline)

    def evaluate_label(self, perf: PerformanceMemory, label: str) -> DegradationDecision:
        label = str(label or "")
        if not self._config.enabled or not label or not self._is_monitored_label(label):
            return DegradationDecision(
                label=label,
                should_disable=False,
                reasons=(),
                session_metrics=Metrics.empty(),
            )

        metrics = self.session_metrics(perf, label)
        failure_rate = (
            float(metrics.rejected_signals) / max(float(metrics.signals), 1.0)
            if metrics.signals > 0 else 0.0
        )
        blocked_rate = (
            float(metrics.blocked_signals) / max(float(metrics.signals), 1.0)
            if metrics.signals > 0 else 0.0
        )
        reasons = []
        if (
            self._config.max_session_loss_pct > 0
            and metrics.closed_trades >= self._config.min_session_closed_trades
            and metrics.pnl_pct <= -float(self._config.max_session_loss_pct)
        ):
            reasons.append("session_loss_pct")
        if (
            self._config.max_session_drawdown_pct > 0
            and metrics.closed_trades >= self._config.min_session_closed_trades
            and metrics.max_dd_pct >= float(self._config.max_session_drawdown_pct)
            and (
                not self._config.drawdown_requires_nonpositive_pnl
                or metrics.pnl_pct <= 0.0
            )
        ):
            reasons.append("session_drawdown_pct")
        if (
            self._config.max_execution_failure_rate > 0
            and metrics.signals >= self._config.min_session_signals
            and metrics.rejected_signals > 0
            and failure_rate >= float(self._config.max_execution_failure_rate)
        ):
            reasons.append("execution_failure_rate")
        if (
            self._config.max_blocked_signal_rate > 0
            and metrics.signals >= self._config.min_session_signals
            and metrics.blocked_signals > 0
            and blocked_rate >= float(self._config.max_blocked_signal_rate)
        ):
            reasons.append("blocked_signal_rate")

        return DegradationDecision(
            label=label,
            should_disable=bool(reasons),
            reasons=tuple(reasons),
            session_metrics=metrics,
            execution_failure_rate=failure_rate,
            blocked_signal_rate=blocked_rate,
        )

    def evaluate(
        self,
        perf: PerformanceMemory,
        labels: Iterable[str],
    ) -> Tuple[DegradationDecision, ...]:
        labels_tuple = tuple(str(label) for label in labels if str(label or ""))
        self.ensure_baseline(perf, labels=labels_tuple)
        decisions = tuple(
            self.evaluate_label(perf, str(label))
            for label in labels_tuple
        )
        self._last_decisions = decisions
        return decisions

    def apply(
        self,
        perf: PerformanceMemory,
        qm: QuarantineManager,
        *,
        labels: Iterable[str],
        bar: int = 0,
    ) -> RecomputeResult:
        if not self._config.enabled:
            current = qm.all_quarantined()
            self._last_decisions = ()
            return RecomputeResult(
                added=frozenset(),
                removed=frozenset(),
                current=current,
            )

        before = qm.all_quarantined()
        decisions = self.evaluate(perf, labels)
        for decision in decisions:
            if not decision.should_disable:
                continue
            if qm.is_quarantined(decision.label):
                continue
            qm.force_quarantine(
                decision.label,
                reason=f"{self._config.reason_prefix}:{','.join(decision.reasons)}",
                bar=bar,
            )
        after = qm.all_quarantined()
        return RecomputeResult(
            added=frozenset(after - before),
            removed=frozenset(),
            current=after,
        )

    def _is_monitored_label(self, label: str) -> bool:
        prefixes = tuple(p for p in self._config.label_prefixes if p)
        if not prefixes:
            return True
        return any(label.startswith(prefix) for prefix in prefixes)


def _metrics_delta(current: Metrics, baseline: Metrics) -> Metrics:
    closed_trades = max(0, int(current.closed_trades) - int(baseline.closed_trades))
    wins = max(0, int(current.wins) - int(baseline.wins))
    losses = max(0, int(current.losses) - int(baseline.losses))
    if wins + losses > closed_trades:
        wins = min(wins, closed_trades)
        losses = min(losses, max(0, closed_trades - wins))
    return Metrics(
        pnl_pct=float(current.pnl_pct) - float(baseline.pnl_pct),
        closed_trades=closed_trades,
        entries=max(0, int(current.entries) - int(baseline.entries)),
        signals=max(0, int(current.signals) - int(baseline.signals)),
        wins=wins,
        losses=losses,
        max_dd_pct=max(0.0, float(current.max_dd_pct) - float(baseline.max_dd_pct)),
        blocked_signals=max(
            0, int(current.blocked_signals) - int(baseline.blocked_signals)
        ),
        rejected_signals=max(
            0, int(current.rejected_signals) - int(baseline.rejected_signals)
        ),
        pending_signals=max(
            0, int(current.pending_signals) - int(baseline.pending_signals)
        ),
        execution_failures=max(
            0, int(current.execution_failures) - int(baseline.execution_failures)
        ),
        pnl_gross_pct=float(current.pnl_gross_pct) - float(baseline.pnl_gross_pct),
        fee_pct=max(0.0, float(current.fee_pct) - float(baseline.fee_pct)),
        funding_pct=float(current.funding_pct) - float(baseline.funding_pct),
    )
