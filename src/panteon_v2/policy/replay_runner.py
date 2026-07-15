"""Exact research replay for the Pantheon vNext policy route."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from ..attribution.event_log import EventLog
from ..domain.types import Action, MarketSnapshot, Regime, Signal
from ..execution import (
    ExecutionStatus,
    FakeExchange,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from ..memory import PerformanceMemory
from .carryflow_adapter import (
    ActorActivationTrace,
    CarryFlowPolicyAdapter,
    ExitIntent,
    OpenProposal,
)
from .executor import DecisionTrace, PolicyExecutorV1, PolicyTarget, RuntimeContext
from .manifest import LoadedPolicyManifest, format_datetime


@dataclass(frozen=True)
class ReplayTradeOutcome:
    symbol: str
    side: str
    entry_reference_price: float
    exit_reference_price: float
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    notional_usd: float
    opened_bar: int
    closed_bar: int
    close_reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ReplayExecutionAudit:
    signal_id: int
    candidate_signal_id: str
    bar: int
    symbol: str
    action: str
    stage: str
    status: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PolicyReplayReport:
    policy_id: str
    manifest_sha256: str
    actor: str
    target: str
    first_timestamp: str
    last_timestamp: str
    bars: int
    bar_interval_seconds: int
    cadence_tolerance_seconds: int
    required_context_coverage_pct: float
    actor_observations: int
    derivatives_context_complete_observations: int
    derivatives_context_missing_observations: int
    derivatives_context_unknown_observations: int
    derivatives_context_coverage_pct: float
    actor_open_actions: int
    adapter_rejected_open_actions: int
    candidate_signals: int
    candidate_signals_with_derivatives_context: int
    actor_opens_blocked_missing_derivatives_context: int
    policy_allowed: int
    policy_no_trade: int
    open_execution_attempts: int
    open_fills: int
    close_execution_attempts: int
    close_fills: int
    filled_orders: int
    closed_trades: int
    remaining_open_positions: int
    gross_pnl_usd: float
    total_costs_usd: float
    net_pnl_usd: float
    expectancy_after_costs_usd: float
    expectancy_lcb_usd: float | None
    win_rate: float
    profit_factor: float | None
    max_drawdown_usd: float
    max_drawdown_pct: float
    mean_cost_bps: float
    turnover_usd: float
    activation_reasons: tuple[tuple[str, int], ...]
    actor_diagnostic_reasons: tuple[tuple[str, int], ...]
    policy_reasons: tuple[tuple[str, int], ...]
    execution_reasons: tuple[tuple[str, int], ...]
    parity_passed: bool
    parity_failures: tuple[str, ...]
    evidence_eligible: bool
    evidence_failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["activation_reasons"] = dict(self.activation_reasons)
        payload["actor_diagnostic_reasons"] = dict(self.actor_diagnostic_reasons)
        payload["policy_reasons"] = dict(self.policy_reasons)
        payload["execution_reasons"] = dict(self.execution_reasons)
        payload["parity_failures"] = list(self.parity_failures)
        payload["evidence_failures"] = list(self.evidence_failures)
        return payload


class PolicyReplayRunner:
    """Run one immutable replay policy through the production executor stack."""

    def __init__(
        self,
        *,
        loaded_manifest: LoadedPolicyManifest,
        actor_adapter: CarryFlowPolicyAdapter,
        policy_executor: PolicyExecutorV1,
        trade_executor: TradeExecutor,
        tracker: PositionTracker,
        exchange: FakeExchange,
        initial_capital_usd: float,
        assumed_spread_bps: float,
    ) -> None:
        manifest = loaded_manifest.manifest
        if manifest.target != PolicyTarget.REPLAY:
            raise ValueError("PolicyReplayRunner requires target=replay")
        if not policy_executor.ready:
            raise ValueError("PolicyReplayRunner requires a valid policy executor")
        if initial_capital_usd <= 0.0:
            raise ValueError("initial_capital_usd must be positive")
        if assumed_spread_bps < 0.0:
            raise ValueError("assumed_spread_bps must be non-negative")
        self.loaded_manifest = loaded_manifest
        self.manifest = manifest
        self.actor_adapter = actor_adapter
        self.policy_executor = policy_executor
        self.trade_executor = trade_executor
        self.tracker = tracker
        self.exchange = exchange
        self.initial_capital_usd = float(initial_capital_usd)
        self.assumed_spread_bps = float(assumed_spread_bps)
        self.timeframe = _timeframe_label(manifest.data.bar_interval_seconds)
        self.activation_traces: list[ActorActivationTrace] = []
        self.policy_traces: list[DecisionTrace] = []
        self.execution_audit: list[ReplayExecutionAudit] = []
        self.trade_outcomes: list[ReplayTradeOutcome] = []
        self._next_signal_id = 1
        self._realized_pnl_usd = 0.0
        self._daily_pnl: dict[date, float] = {}
        self._entry_reference_prices: dict[str, float] = {}
        self._bars = 0
        self._first_timestamp = ""
        self._last_timestamp = ""
        self._previous_market_timestamp = None

    @classmethod
    def create(
        cls,
        *,
        loaded_manifest: LoadedPolicyManifest,
        agent: Any,
        initial_capital_usd: float = 1000.0,
        assumed_spread_bps: float = 2.0,
        exchange_min_notional_usd: float = 5.0,
        derivatives_context_provider: Any | None = None,
    ) -> "PolicyReplayRunner":
        manifest = loaded_manifest.manifest
        exchange = FakeExchange(name="BITGET_REPLAY")
        # Manifest costs are round-trip values. FakeExchange applies these per
        # fill, so each side receives half of the configured total.
        exchange.set_fee_rate(manifest.costs.round_trip_fee_bps / 2.0 / 10_000.0)
        exchange.set_slippage_pct(manifest.costs.slippage_bps / 2.0 / 10_000.0)
        tracker = PositionTracker()
        risk = RiskLimits(
            config=RiskLimitsConfig(
                max_open_positions=manifest.risk.max_open_positions,
                max_per_symbol=1,
                min_notional_usd=float(exchange_min_notional_usd),
                max_notional_usd=manifest.risk.max_notional_usd,
                max_leverage=1,
                apply_leverage_to_notional=False,
                floor_to_exchange_min_notional=False,
                capital_fraction=manifest.risk.capital_fraction,
            )
        )
        trade_executor = TradeExecutor(
            exchange=exchange,
            health=SymbolHealthMonitor(),
            risk_limits=risk,
            position_tracker=tracker,
            perf=PerformanceMemory(trade_fraction=manifest.risk.capital_fraction),
            event_log=EventLog(),
        )
        return cls(
            loaded_manifest=loaded_manifest,
            actor_adapter=CarryFlowPolicyAdapter(
                agent=agent,
                manifest=manifest,
                derivatives_context_provider=derivatives_context_provider,
            ),
            policy_executor=PolicyExecutorV1(loaded_manifest),
            trade_executor=trade_executor,
            tracker=tracker,
            exchange=exchange,
            initial_capital_usd=initial_capital_usd,
            assumed_spread_bps=assumed_spread_bps,
        )

    def run(
        self,
        snapshots: Iterable[MarketSnapshot],
        *,
        flatten_end: bool = True,
    ) -> PolicyReplayReport:
        last_market: MarketSnapshot | None = None
        for market in snapshots:
            last_market = market
            self._run_bar(market)
        if flatten_end and last_market is not None:
            self._flatten(last_market)
        return self._build_report()

    def _run_bar(self, market: MarketSnapshot) -> None:
        self._validate_market_cadence(market)
        self._bars += 1
        timestamp = format_datetime(market.timestamp)
        self._first_timestamp = self._first_timestamp or timestamp
        self._last_timestamp = timestamp
        step = self.actor_adapter.propose(market, tracker=self.tracker)
        self.activation_traces.extend(step.traces)

        # Exits are safety/lifecycle actions and do not require entry promotion.
        for intent in step.exit_intents:
            self._execute_exit(intent)
        for proposal in step.open_proposals:
            self._evaluate_open(proposal, market)
        self.actor_adapter.reconcile(
            self.tracker,
            market_symbols=tuple(market.prices),
            bar=market.bar,
        )

    def _validate_market_cadence(self, market: MarketSnapshot) -> None:
        current = market.cadence_timestamp or market.timestamp
        if current.tzinfo is None:
            raise ValueError("replay market timestamps must be timezone-aware")
        previous = self._previous_market_timestamp
        if previous is not None:
            delta = (current - previous).total_seconds()
            if delta <= 0.0:
                raise ValueError("replay market timestamps must be strictly increasing")
            expected = float(self.manifest.data.bar_interval_seconds)
            tolerance = float(self.manifest.data.cadence_tolerance_seconds)
            if abs(delta - expected) > tolerance:
                raise ValueError(
                    "replay market cadence mismatch: "
                    f"expected {expected:.0f}s +/- {tolerance:.0f}s, got {delta:.3f}s"
                )
        self._previous_market_timestamp = current

    def _evaluate_open(self, proposal: OpenProposal, market: MarketSnapshot) -> None:
        candidate = proposal.candidate
        daily_loss = max(0.0, -self._daily_pnl.get(market.timestamp.date(), 0.0))
        context = RuntimeContext(
            exchange=self.manifest.exchange,
            mode=PolicyTarget.REPLAY,
            now=market.timestamp,
            spread_bps=self.assumed_spread_bps,
            estimated_slippage_bps=self.manifest.costs.slippage_bps,
            regime_confidence=market.regime_confidence,
            exchange_healthy=True,
            kill_switch_active=False,
            open_positions=self.tracker.open_count,
            daily_loss_usd=daily_loss,
        )
        decision = self.policy_executor.decide(candidate, context)
        self.policy_traces.append(decision.trace)
        if not decision.allowed:
            return

        signal_id = self._allocate_signal_id()
        signal = Signal(
            id=signal_id,
            bar=candidate.bar,
            sym=proposal.execution_symbol,
            action=proposal.action,
            price=candidate.price,
            regime=Regime.from_string(candidate.regime),
            by_player=f"Policy:{self.manifest.policy_id}",
            by_agent=self.manifest.actor,
            position_scope=f"policy_{self.manifest.manifest_sha256[:12]}",
            risk_mult=decision.risk_mult,
            timestamp=candidate.generated_at,
            metadata={
                "policy_id": self.manifest.policy_id,
                "manifest_sha256": self.manifest.manifest_sha256,
                "candidate_signal_id": candidate.signal_id,
                "expected_move_bps": candidate.expected_move_bps,
                "stop_loss_pct": self.manifest.risk.stop_loss_pct,
            },
        )
        self.trade_executor.set_event_context(
            {
                "decision_id": candidate.signal_id,
                "exchange": self.manifest.exchange,
                "symbol": proposal.execution_symbol,
                "timeframe": self.timeframe,
                "mode": "replay",
                "run_id": self.manifest.policy_id,
                "session_id": self.manifest.manifest_sha256[:16],
            }
        )
        result = self.trade_executor.execute(signal, balance_usd=self._balance_usd())
        self.execution_audit.append(
            ReplayExecutionAudit(
                signal_id=signal_id,
                candidate_signal_id=candidate.signal_id,
                bar=candidate.bar,
                symbol=proposal.execution_symbol,
                action=proposal.action.name,
                stage="open",
                status=result.status.value,
                reason=result.reason,
            )
        )
        if result.status == ExecutionStatus.FILLED and result.trade is not None:
            self._entry_reference_prices[proposal.execution_symbol] = float(signal.price)

    def _execute_exit(self, intent: ExitIntent) -> None:
        opened = self.tracker.get(intent.execution_symbol)
        signal_id = self._allocate_signal_id()
        signal = Signal(
            id=signal_id,
            bar=intent.bar,
            sym=intent.execution_symbol,
            action=Action.FUT_CLOSE_ALL,
            price=intent.price,
            regime=Regime.from_string(intent.regime),
            by_player=(opened.by_player if opened is not None else f"Policy:{self.manifest.policy_id}"),
            by_agent=(opened.by_agent if opened is not None else self.manifest.actor),
            position_scope=f"policy_{self.manifest.manifest_sha256[:12]}",
            timestamp=intent.generated_at,
            metadata={
                "policy_id": self.manifest.policy_id,
                "manifest_sha256": self.manifest.manifest_sha256,
                "candidate_signal_id": intent.signal_id,
                "close_reason": intent.reason,
            },
        )
        self.trade_executor.set_event_context(
            {
                "decision_id": intent.signal_id,
                "exchange": self.manifest.exchange,
                "symbol": intent.execution_symbol,
                "timeframe": self.timeframe,
                "mode": "replay",
                "run_id": self.manifest.policy_id,
                "session_id": self.manifest.manifest_sha256[:16],
            }
        )
        result = self.trade_executor.execute(signal, balance_usd=self._balance_usd())
        self.execution_audit.append(
            ReplayExecutionAudit(
                signal_id=signal_id,
                candidate_signal_id=intent.signal_id,
                bar=intent.bar,
                symbol=intent.execution_symbol,
                action=Action.FUT_CLOSE_ALL.name,
                stage="close",
                status=result.status.value,
                reason=result.reason or intent.reason,
            )
        )
        if (
            result.status == ExecutionStatus.FILLED
            and result.trade is not None
            and opened is not None
        ):
            qty = min(float(opened.qty), float(result.trade.qty))
            actual_gross = (
                (result.trade.fill_price - opened.entry_price) * qty
                if opened.side == "long"
                else (opened.entry_price - result.trade.fill_price) * qty
            )
            fees_and_funding = (
                float(opened.fee_open)
                + float(result.trade.fee)
                + float(opened.funding_open)
                + float(result.trade.funding)
            )
            net = actual_gross - fees_and_funding
            entry_reference = self._entry_reference_prices.pop(
                intent.execution_symbol,
                float(opened.entry_price),
            )
            exit_reference = float(intent.price)
            market_gross = (
                (exit_reference - entry_reference) * qty
                if opened.side == "long"
                else (entry_reference - exit_reference) * qty
            )
            costs = market_gross - net
            outcome = ReplayTradeOutcome(
                symbol=intent.execution_symbol,
                side=opened.side,
                entry_reference_price=entry_reference,
                exit_reference_price=exit_reference,
                entry_price=float(opened.entry_price),
                exit_price=float(result.trade.fill_price),
                qty=qty,
                gross_pnl_usd=market_gross,
                costs_usd=costs,
                net_pnl_usd=net,
                notional_usd=entry_reference * qty,
                opened_bar=int(opened.opened_bar),
                closed_bar=intent.bar,
                close_reason=intent.reason,
            )
            self.trade_outcomes.append(outcome)
            self._realized_pnl_usd += net
            day = intent.generated_at.date()
            self._daily_pnl[day] = self._daily_pnl.get(day, 0.0) + net

    def _flatten(self, market: MarketSnapshot) -> None:
        for symbol in sorted(self.tracker.all_open()):
            if symbol not in market.prices:
                continue
            self._execute_exit(
                ExitIntent(
                    signal_id=f"{self.manifest.policy_id}:{market.bar}:{symbol}:flatten",
                    bar=market.bar,
                    execution_symbol=symbol,
                    price=float(market.prices[symbol]),
                    generated_at=market.timestamp,
                    regime=market.regime_for_symbol(symbol).label,
                    reason="replay_end_flatten",
                )
            )
        self.actor_adapter.reconcile(
            self.tracker,
            market_symbols=tuple(market.prices),
            bar=market.bar,
        )

    def _balance_usd(self) -> float:
        return max(0.0, self.initial_capital_usd + self._realized_pnl_usd)

    def _allocate_signal_id(self) -> int:
        signal_id = self._next_signal_id
        self._next_signal_id += 1
        return signal_id

    def _build_report(self) -> PolicyReplayReport:
        activation_reasons = Counter(trace.outcome for trace in self.activation_traces)
        actor_diagnostic_reasons = Counter(
            trace.diagnostic_reason for trace in self.activation_traces
        )
        policy_reasons = Counter(trace.primary_reason.value for trace in self.policy_traces)
        execution_reasons = Counter(
            row.reason or row.status for row in self.execution_audit
        )
        actor_open_actions = sum(
            1
            for trace in self.activation_traces
            if trace.raw_action.startswith("FUT_LONG")
            or trace.raw_action.startswith("FUT_SHORT")
        )
        adapter_rejected = sum(
            count
            for reason, count in activation_reasons.items()
            if reason in {
                "unsupported_open_action",
                "expected_move_feature_missing",
                "expected_move_feature_below_model_gate",
                "expected_move_nonpositive",
                "derivatives_context_missing",
            }
        )
        candidate_signals = activation_reasons.get("candidate_created", 0)
        candidates_with_derivatives_context = sum(
            trace.outcome == "candidate_created"
            and trace.derivatives_context_present is True
            for trace in self.activation_traces
        )
        opens_blocked_missing_derivatives = activation_reasons.get(
            "derivatives_context_missing",
            0,
        )
        complete_context_observations = sum(
            trace.derivatives_context_present is True
            for trace in self.activation_traces
        )
        missing_context_observations = sum(
            trace.derivatives_context_present is False
            for trace in self.activation_traces
        )
        unknown_context_observations = (
            len(self.activation_traces)
            - complete_context_observations
            - missing_context_observations
        )
        derivatives_context_coverage_pct = (
            complete_context_observations / len(self.activation_traces) * 100.0
            if self.activation_traces
            else 0.0
        )
        policy_allowed = policy_reasons.get("allowed", 0)
        policy_no_trade = len(self.policy_traces) - policy_allowed
        open_rows = [row for row in self.execution_audit if row.stage == "open"]
        close_rows = [row for row in self.execution_audit if row.stage == "close"]
        open_fills = sum(row.status == ExecutionStatus.FILLED.value for row in open_rows)
        close_fills = sum(row.status == ExecutionStatus.FILLED.value for row in close_rows)
        filled_orders = open_fills + close_fills

        net_values = [row.net_pnl_usd for row in self.trade_outcomes]
        gross_pnl = sum(row.gross_pnl_usd for row in self.trade_outcomes)
        total_costs = sum(row.costs_usd for row in self.trade_outcomes)
        net_pnl = sum(net_values)
        closed = len(net_values)
        expectancy = net_pnl / closed if closed else 0.0
        lcb = _mean_lcb(net_values)
        wins = sum(value > 0.0 for value in net_values)
        win_rate = wins / closed if closed else 0.0
        positive = sum(value for value in net_values if value > 0.0)
        negative = abs(sum(value for value in net_values if value < 0.0))
        profit_factor = (positive / negative) if negative > 0.0 else None
        max_drawdown = _max_drawdown(net_values)
        max_drawdown_pct = max_drawdown / self.initial_capital_usd * 100.0
        turnover = sum(row.notional_usd * 2.0 for row in self.trade_outcomes)
        mean_cost_bps = (
            statistics.fmean(
                row.costs_usd / row.notional_usd * 10_000.0
                for row in self.trade_outcomes
                if row.notional_usd > 0.0
            )
            if self.trade_outcomes
            else 0.0
        )

        parity_failures: list[str] = []
        if actor_open_actions != candidate_signals + adapter_rejected:
            parity_failures.append("actor_open_to_candidate_mismatch")
        if policy_allowed != len(open_rows):
            parity_failures.append("policy_allow_to_execution_mismatch")
        if open_fills < closed:
            parity_failures.append("closed_trades_exceed_open_fills")
        if activation_reasons.get("close_without_owned_position", 0) > 0:
            parity_failures.append("orphan_actor_close")
        if self.tracker.open_count > 0:
            parity_failures.append("remaining_open_positions")

        evidence_failures: list[str] = []
        if filled_orders < 20:
            evidence_failures.append("fills_below_20")
        if closed < 10:
            evidence_failures.append("closed_trades_below_10")
        if expectancy <= 0.0:
            evidence_failures.append("nonpositive_expectancy")
        if lcb is None or lcb <= 0.0:
            evidence_failures.append("nonpositive_lcb")
        if max_drawdown_pct > 20.0:
            evidence_failures.append("drawdown_above_20pct")
        if mean_cost_bps <= 0.0 and closed > 0:
            evidence_failures.append("zero_cost_attribution")
        if opens_blocked_missing_derivatives > 0:
            evidence_failures.append("derivatives_context_missing")
        if (
            derivatives_context_coverage_pct
            < self.manifest.data.required_context_coverage_pct
        ):
            evidence_failures.append("derivatives_context_coverage_below_required")
        if parity_failures:
            evidence_failures.append("parity_failed")

        return PolicyReplayReport(
            policy_id=self.manifest.policy_id,
            manifest_sha256=self.manifest.manifest_sha256,
            actor=self.manifest.actor,
            target=self.manifest.target.value,
            first_timestamp=self._first_timestamp,
            last_timestamp=self._last_timestamp,
            bars=self._bars,
            bar_interval_seconds=self.manifest.data.bar_interval_seconds,
            cadence_tolerance_seconds=(
                self.manifest.data.cadence_tolerance_seconds
            ),
            required_context_coverage_pct=(
                self.manifest.data.required_context_coverage_pct
            ),
            actor_observations=len(self.activation_traces),
            derivatives_context_complete_observations=(
                complete_context_observations
            ),
            derivatives_context_missing_observations=(
                missing_context_observations
            ),
            derivatives_context_unknown_observations=(
                unknown_context_observations
            ),
            derivatives_context_coverage_pct=derivatives_context_coverage_pct,
            actor_open_actions=actor_open_actions,
            adapter_rejected_open_actions=adapter_rejected,
            candidate_signals=candidate_signals,
            candidate_signals_with_derivatives_context=(
                candidates_with_derivatives_context
            ),
            actor_opens_blocked_missing_derivatives_context=(
                opens_blocked_missing_derivatives
            ),
            policy_allowed=policy_allowed,
            policy_no_trade=policy_no_trade,
            open_execution_attempts=len(open_rows),
            open_fills=open_fills,
            close_execution_attempts=len(close_rows),
            close_fills=close_fills,
            filled_orders=filled_orders,
            closed_trades=closed,
            remaining_open_positions=self.tracker.open_count,
            gross_pnl_usd=gross_pnl,
            total_costs_usd=total_costs,
            net_pnl_usd=net_pnl,
            expectancy_after_costs_usd=expectancy,
            expectancy_lcb_usd=lcb,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_usd=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            mean_cost_bps=mean_cost_bps,
            turnover_usd=turnover,
            activation_reasons=tuple(sorted(activation_reasons.items())),
            actor_diagnostic_reasons=tuple(sorted(actor_diagnostic_reasons.items())),
            policy_reasons=tuple(sorted(policy_reasons.items())),
            execution_reasons=tuple(sorted(execution_reasons.items())),
            parity_passed=not parity_failures,
            parity_failures=tuple(parity_failures),
            evidence_eligible=not evidence_failures,
            evidence_failures=tuple(evidence_failures),
        )


def _mean_lcb(values: list[float], z: float = 1.6448536) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    return mean - z * stderr


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _timeframe_label(seconds: int) -> str:
    value = int(seconds)
    if value % 3600 == 0:
        return f"{value // 3600}h"
    return f"{value // 60}m"
