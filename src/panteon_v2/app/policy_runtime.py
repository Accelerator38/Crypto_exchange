"""Single-policy Bitget micro-live runtime.

This module is intentionally independent from Flash, Strategist and the shadow
tournament.  It converts one sealed CarryFlow policy into domain Signals while
leaving exchange I/O to the existing TradeExecutor.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain.types import Action, MarketSnapshot, Regime, Signal
from ..execution.position_tracker import PositionTracker
from ..policy.carryflow_adapter import ActorActivationTrace, CarryFlowPolicyAdapter
from ..policy.executor import DecisionTrace, PolicyExecutorV1, PolicyTarget, RuntimeContext
from ..policy.manifest import (
    LoadedPolicyManifest,
    PolicyManifest,
    canonical_symbol,
    load_policy_manifest,
)
from ..shadow.adapters import V1AgentAdapter
from .regime_detector import PriceRegimeDetector


DEFAULT_POLICY_MANIFEST_PATH = "Runtime/BITGET/active_policy_manifest_v1.json"
DEFAULT_POLICY_RISK_STATE_PATH = "panteon_v2_state/bitget_policy_risk_v1.json"
POLICY_RISK_STATE_SCHEMA = "panteon.policy_risk_state.v1"


class LivePolicyRuntimeError(RuntimeError):
    """Fail-closed policy runtime configuration or market-data error."""


@dataclass(frozen=True)
class PolicyMarketQuality:
    symbol: str
    spread_bps: float
    estimated_slippage_bps: float
    bid: float
    ask: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "spread_bps": self.spread_bps,
            "estimated_slippage_bps": self.estimated_slippage_bps,
            "bid": self.bid,
            "ask": self.ask,
        }


@dataclass(frozen=True)
class LivePolicyStep:
    status: str
    reason: str
    policy_market: MarketSnapshot | None
    signals: tuple[Signal, ...]
    activation_traces: tuple[ActorActivationTrace, ...] = ()
    decision_traces: tuple[DecisionTrace, ...] = ()
    quality: tuple[PolicyMarketQuality, ...] = ()
    candidate_count: int = 0
    denied_count: int = 0
    next_signal_id: int = 0


@dataclass
class LivePolicyRuntimeV1:
    """Stateful one-actor runtime for a pinned micro-live manifest."""

    loaded_manifest: LoadedPolicyManifest
    actor_adapter: CarryFlowPolicyAdapter
    policy_executor: PolicyExecutorV1
    regime_detector: PriceRegimeDetector = field(
        default_factory=lambda: PriceRegimeDetector(exchange_name="BITGET")
    )
    risk_state_path: Path | None = None
    _last_cadence_timestamp: datetime | None = None
    _policy_bar: int = 0
    _daily_equity_date: str = ""
    _daily_start_equity_usd: float = 0.0
    _current_daily_loss_usd: float = 0.0
    _daily_loss_latched: bool = False
    _daily_loss_reason: str = ""

    def __post_init__(self) -> None:
        manifest = self.loaded_manifest.manifest
        if manifest.target != PolicyTarget.MICRO_LIVE:
            raise LivePolicyRuntimeError("live policy requires target=micro_live")
        if manifest.exchange != "BITGET":
            raise LivePolicyRuntimeError("live policy requires exchange=BITGET")
        if manifest.actor != "CarryFlowAgentV2":
            raise LivePolicyRuntimeError("unsupported live policy actor")
        if not self.policy_executor.ready:
            raise LivePolicyRuntimeError("policy executor is not ready")
        if self.risk_state_path is not None:
            self.risk_state_path = Path(self.risk_state_path)
            self._load_risk_state()

    @property
    def manifest(self) -> PolicyManifest:
        return self.loaded_manifest.manifest

    @property
    def actor(self) -> V1AgentAdapter:
        return self.actor_adapter.agent

    @property
    def label(self) -> str:
        return f"Policy:{self.manifest.policy_id}"

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted({rule.symbol for rule in self.manifest.rules}))

    @property
    def required_warmup_bars(self) -> int:
        config = dict(self.manifest.actor_config)
        return max(
            int(config["EMA_SLOW"]) + 5,
            int(config["EMA_FAST"]) + 3,
            int(config["RSI_N"]) * 3 + 1,
        )

    @property
    def current_daily_loss_usd(self) -> float:
        return float(self._current_daily_loss_usd)

    @property
    def risk_status(self) -> dict[str, Any]:
        return {
            "state_path": str(self.risk_state_path or ""),
            "utc_date": self._daily_equity_date,
            "start_equity_usd": self._daily_start_equity_usd,
            "current_daily_loss_usd": self._current_daily_loss_usd,
            "max_daily_loss_usd": self.manifest.risk.max_daily_loss_usd,
            "daily_loss_latched": self._daily_loss_latched,
            "daily_loss_reason": self._daily_loss_reason,
        }

    def update_equity_guard(
        self,
        *,
        now: datetime,
        current_equity_usd: float,
        initial_equity_usd: float,
    ) -> str:
        """Update the persisted UTC-day loss latch and return its stop reason."""

        timestamp = _aware_utc(now)
        current = _positive_float(current_equity_usd, "current_equity_usd")
        initial = _positive_float(initial_equity_usd, "initial_equity_usd")
        utc_date = timestamp.date().isoformat()
        state_changed = False
        if self._daily_equity_date != utc_date:
            baseline = initial if not self._daily_equity_date else current
            self._daily_equity_date = utc_date
            self._daily_start_equity_usd = baseline
            self._current_daily_loss_usd = 0.0
            self._daily_loss_latched = False
            self._daily_loss_reason = ""
            state_changed = True
        if self._daily_start_equity_usd <= 0.0:
            self._daily_start_equity_usd = initial
            state_changed = True

        observed_loss = max(
            0.0,
            self._daily_start_equity_usd - current,
        )
        self._current_daily_loss_usd = (
            max(self._current_daily_loss_usd, observed_loss)
            if self._daily_loss_latched
            else observed_loss
        )
        limit = float(self.manifest.risk.max_daily_loss_usd)
        if self._current_daily_loss_usd >= limit and not self._daily_loss_latched:
            self._daily_loss_latched = True
            self._daily_loss_reason = (
                "policy max daily loss exceeded: "
                f"loss=${self._current_daily_loss_usd:.2f}, "
                f"limit=${limit:.2f}, current=${current:.2f}"
            )
            state_changed = True
        if state_changed:
            self._save_risk_state(timestamp)
        return self._daily_loss_reason if self._daily_loss_latched else ""

    def _load_risk_state(self) -> None:
        path = self.risk_state_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LivePolicyRuntimeError(
                f"policy risk state is unreadable: {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise LivePolicyRuntimeError("policy risk state must be a JSON object")
        if str(payload.get("schema_version") or "") != POLICY_RISK_STATE_SCHEMA:
            raise LivePolicyRuntimeError("policy risk state schema mismatch")
        raw_date = str(payload.get("utc_date") or "")
        try:
            date.fromisoformat(raw_date)
            baseline = _positive_float(
                payload.get("start_equity_usd"),
                "risk_state_start_equity_usd",
            )
        except (TypeError, ValueError, LivePolicyRuntimeError) as exc:
            raise LivePolicyRuntimeError("policy risk state values are invalid") from exc
        latched = payload.get("daily_loss_latched")
        if not isinstance(latched, bool):
            raise LivePolicyRuntimeError("policy risk state latch is invalid")
        reason = str(payload.get("daily_loss_reason") or "")
        if latched and not reason:
            raise LivePolicyRuntimeError("policy risk state latch reason is missing")
        self._daily_equity_date = raw_date
        self._daily_start_equity_usd = baseline
        self._current_daily_loss_usd = _nonnegative_float(
            payload.get("current_daily_loss_usd", 0.0),
            "risk_state_current_daily_loss_usd",
        )
        self._daily_loss_latched = latched
        self._daily_loss_reason = reason

    def _save_risk_state(self, timestamp: datetime) -> None:
        path = self.risk_state_path
        if path is None:
            return
        payload = {
            "schema_version": POLICY_RISK_STATE_SCHEMA,
            "exchange": self.manifest.exchange,
            "utc_date": self._daily_equity_date,
            "start_equity_usd": self._daily_start_equity_usd,
            "current_daily_loss_usd": self._current_daily_loss_usd,
            "daily_loss_latched": self._daily_loss_latched,
            "daily_loss_reason": self._daily_loss_reason,
            "last_manifest_sha256": self.manifest.manifest_sha256,
            "updated_at": timestamp.isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError as exc:
            raise LivePolicyRuntimeError(
                f"policy risk state write failed: {path}: {exc}"
            ) from exc

    def warmup_from_history(
        self,
        price_rows: Sequence[Mapping[str, float]],
        volume_rows: Sequence[Mapping[str, float]],
        *,
        source_bar_interval_seconds: int,
        restore_derivatives_fetcher: Any | None = None,
    ) -> int:
        """Warm price/regime state at manifest cadence without fake OI history."""

        target_interval = int(self.manifest.data.bar_interval_seconds)
        source_interval = int(source_bar_interval_seconds)
        if source_interval <= 0 or target_interval % source_interval:
            raise LivePolicyRuntimeError("warmup cadence is incompatible with manifest")
        stride = target_interval // source_interval
        if stride <= 0:
            raise LivePolicyRuntimeError("warmup stride must be positive")

        prices = [dict(row) for row in price_rows]
        volumes = [dict(row) for row in volume_rows]
        if not prices:
            return 0
        volume_offset = max(0, len(prices) - len(volumes))
        offset = len(prices) % stride
        start = offset + stride - 1
        warmed = 0

        set_fetcher = None
        try:
            import panteon_agents  # type: ignore

            set_fetcher = getattr(panteon_agents, "set_fetcher", None)
            if callable(set_fetcher):
                set_fetcher(None)

            first_timestamp = datetime.now(timezone.utc) - timedelta(
                seconds=target_interval * max(1, len(prices) // stride)
            )
            for end_index in range(start, len(prices), stride):
                row = _canonical_numeric_row(prices[end_index], self.symbols)
                if len(row) != len(self.symbols):
                    continue
                chunk_start = end_index - stride + 1
                aggregate_volumes = {symbol: 0.0 for symbol in self.symbols}
                for source_index in range(chunk_start, end_index + 1):
                    volume_index = source_index - volume_offset
                    if not 0 <= volume_index < len(volumes):
                        continue
                    for symbol, value in _canonical_numeric_row(
                        volumes[volume_index], self.symbols, allow_zero=True
                    ).items():
                        aggregate_volumes[symbol] += value

                self._policy_bar += 1
                timestamp = first_timestamp + timedelta(
                    seconds=target_interval * self._policy_bar
                )
                regime = self.regime_detector.update(row, volumes=aggregate_volumes)
                market = MarketSnapshot(
                    bar=self._policy_bar,
                    timestamp=timestamp,
                    cadence_timestamp=timestamp,
                    regime=regime,
                    regime_confidence=self.regime_detector.confidence,
                    prices=row,
                    volumes=aggregate_volumes,
                    regimes_by_symbol=self.regime_detector.symbol_regimes,
                    regime_features_by_symbol=self.regime_detector.symbol_stats,
                )
                self.actor.act(market)
                warmed += 1
        finally:
            if callable(set_fetcher):
                set_fetcher(restore_derivatives_fetcher)

        self._last_cadence_timestamp = None
        return warmed

    def evaluate(
        self,
        poll_market: MarketSnapshot,
        *,
        exchange: Any,
        tracker: PositionTracker,
        signal_id_start: int,
        exchange_healthy: bool,
        daily_loss_usd: float,
    ) -> LivePolicyStep:
        """Evaluate one poll. Opens occur only on a complete manifest-cadence bar."""

        next_signal_id = int(signal_id_start)
        safety_signals, next_signal_id = self._safety_exit_signals(
            poll_market,
            tracker=tracker,
            signal_id_start=next_signal_id,
        )
        if safety_signals:
            return LivePolicyStep(
                status="safety_exit",
                reason=str(safety_signals[0].metadata.get("close_reason") or "safety_exit"),
                policy_market=None,
                signals=tuple(safety_signals),
                next_signal_id=next_signal_id,
            )

        policy_market, quality, status, reason = self._prepare_policy_market(
            poll_market,
            exchange=exchange,
        )
        if policy_market is None:
            return LivePolicyStep(
                status=status,
                reason=reason,
                policy_market=None,
                signals=(),
                quality=quality,
                next_signal_id=next_signal_id,
            )

        actor_step = self.actor_adapter.propose(policy_market, tracker=tracker)
        signals: list[Signal] = []
        decisions: list[DecisionTrace] = []

        for intent in actor_step.exit_intents:
            opened = tracker.get(intent.execution_symbol)
            signals.append(
                Signal(
                    id=next_signal_id,
                    bar=poll_market.bar,
                    sym=intent.execution_symbol,
                    action=Action.FUT_CLOSE_ALL,
                    price=intent.price,
                    regime=Regime.from_string(intent.regime),
                    by_player=(opened.by_player if opened is not None else self.label),
                    by_agent=(opened.by_agent if opened is not None else self.manifest.actor),
                    position_scope=self._position_scope(),
                    timestamp=intent.generated_at,
                    metadata=self._signal_metadata(
                        intent.signal_id,
                        close_reason=intent.reason,
                    ),
                )
            )
            next_signal_id += 1

        quality_by_symbol = {item.symbol: item for item in quality}
        allowed_opens = 0
        for proposal in actor_step.open_proposals:
            candidate = proposal.candidate
            market_quality = quality_by_symbol.get(candidate.symbol)
            context = RuntimeContext(
                exchange=self.manifest.exchange,
                mode=PolicyTarget.MICRO_LIVE,
                now=policy_market.timestamp,
                spread_bps=(market_quality.spread_bps if market_quality else 0.0),
                estimated_slippage_bps=(
                    market_quality.estimated_slippage_bps
                    if market_quality
                    else self.manifest.costs.slippage_bps
                ),
                regime_confidence=policy_market.regime_confidence,
                exchange_healthy=bool(exchange_healthy and market_quality is not None),
                kill_switch_active=False,
                open_positions=tracker.open_count + allowed_opens,
                daily_loss_usd=max(0.0, float(daily_loss_usd)),
            )
            decision = self.policy_executor.decide(candidate, context)
            decisions.append(decision.trace)
            if not decision.allowed:
                continue
            signals.append(
                Signal(
                    id=next_signal_id,
                    bar=poll_market.bar,
                    sym=proposal.execution_symbol,
                    action=proposal.action,
                    price=candidate.price,
                    regime=Regime.from_string(candidate.regime),
                    by_player=self.label,
                    by_agent=self.manifest.actor,
                    position_scope=self._position_scope(),
                    risk_mult=decision.risk_mult,
                    timestamp=candidate.generated_at,
                    metadata=self._signal_metadata(
                        candidate.signal_id,
                        expected_move_bps=candidate.expected_move_bps,
                        stop_loss_pct=self.manifest.risk.stop_loss_pct,
                    ),
                )
            )
            next_signal_id += 1
            allowed_opens += 1

        # CarryFlow marks proposed positions before policy/exchange acceptance.
        # Tracker is authoritative, so clear denied or not-yet-filled proposals.
        self.actor_adapter.reconcile(
            tracker,
            market_symbols=tuple(policy_market.prices),
            bar=policy_market.bar,
        )
        denied = sum(1 for trace in decisions if trace.outcome.value == "NO_TRADE")
        return LivePolicyStep(
            status="evaluated",
            reason="",
            policy_market=policy_market,
            signals=tuple(signals),
            activation_traces=actor_step.traces,
            decision_traces=tuple(decisions),
            quality=quality,
            candidate_count=len(actor_step.open_proposals),
            denied_count=denied,
            next_signal_id=next_signal_id,
        )

    def _prepare_policy_market(
        self,
        poll_market: MarketSnapshot,
        *,
        exchange: Any,
    ) -> tuple[
        MarketSnapshot | None,
        tuple[PolicyMarketQuality, ...],
        str,
        str,
    ]:
        current = _aware_utc(poll_market.timestamp)
        interval = int(self.manifest.data.bar_interval_seconds)
        epoch = int(current.timestamp())
        boundary_epoch = epoch - epoch % interval
        boundary = datetime.fromtimestamp(boundary_epoch, tz=timezone.utc)
        if (
            self._last_cadence_timestamp is not None
            and boundary <= self._last_cadence_timestamp
        ):
            return None, (), "waiting_for_bar_close", "cadence_not_due"

        lag_seconds = (current - boundary).total_seconds()
        max_lag = float(self.manifest.data.max_bar_close_lag_seconds)
        if lag_seconds < 0.0:
            return None, (), "market_frame_blocked", "bar_close_from_future"
        if lag_seconds > max_lag:
            self._last_cadence_timestamp = boundary
            return None, (), "market_frame_blocked", "bar_close_lag_exceeded"

        provider = getattr(exchange, "get_policy_market_frame", None)
        if not callable(provider):
            return None, (), "market_frame_blocked", "policy_market_provider_unavailable"
        try:
            frame = provider(
                symbols=self.symbols,
                bar_interval_seconds=interval,
                bar_close_timestamp=boundary,
                max_notional_usd=self.manifest.risk.max_notional_usd,
            )
        except Exception as exc:
            return (
                None,
                (),
                "market_frame_retry",
                f"policy_market_provider_error:{type(exc).__name__}:{exc}",
            )
        if not isinstance(frame, Mapping):
            return None, (), "market_frame_retry", "policy_market_frame_invalid"
        if not bool(frame.get("complete")):
            reason = str(frame.get("reason") or "policy_market_frame_incomplete")
            return None, (), "market_frame_retry", reason

        try:
            observed_at = _aware_utc(frame.get("observed_at"))
            frame_boundary = _aware_utc(frame.get("bar_close_timestamp"))
        except (TypeError, ValueError):
            return None, (), "market_frame_retry", "policy_market_timestamp_invalid"
        if frame_boundary != boundary:
            return None, (), "market_frame_retry", "policy_market_cadence_mismatch"
        observed_lag = (observed_at - boundary).total_seconds()
        if not 0.0 <= observed_lag <= max_lag:
            return None, (), "market_frame_blocked", "policy_market_lag_exceeded"

        raw_rows = frame.get("symbols")
        if not isinstance(raw_rows, Mapping):
            return None, (), "market_frame_retry", "policy_market_symbols_missing"
        prices: dict[str, float] = {}
        volumes: dict[str, float] = {}
        qualities: list[PolicyMarketQuality] = []
        for symbol in self.symbols:
            row = raw_rows.get(symbol)
            if not isinstance(row, Mapping) or not bool(row.get("complete")):
                return None, (), "market_frame_retry", f"policy_market_symbol_incomplete:{symbol}"
            try:
                price = _positive_float(row.get("decision_price"), "decision_price")
                volume = _nonnegative_float(row.get("volume"), "volume")
                bid = _positive_float(row.get("bid"), "bid")
                ask = _positive_float(row.get("ask"), "ask")
                spread_bps = _nonnegative_float(row.get("spread_bps"), "spread_bps")
                slippage_bps = _nonnegative_float(
                    row.get("estimated_slippage_bps"),
                    "estimated_slippage_bps",
                )
            except LivePolicyRuntimeError as exc:
                return None, (), "market_frame_retry", f"{symbol}:{exc}"
            if ask < bid:
                return None, (), "market_frame_retry", f"policy_market_crossed_book:{symbol}"
            prices[symbol] = price
            volumes[symbol] = volume
            qualities.append(
                PolicyMarketQuality(
                    symbol=symbol,
                    spread_bps=spread_bps,
                    estimated_slippage_bps=max(
                        slippage_bps,
                        self.manifest.costs.slippage_bps,
                    ),
                    bid=bid,
                    ask=ask,
                )
            )

        funding = _canonical_numeric_row(
            poll_market.funding,
            self.symbols,
            allow_zero=True,
        )
        regime = self.regime_detector.update(prices, volumes=volumes, funding=funding)
        self._policy_bar += 1
        policy_market = MarketSnapshot(
            bar=self._policy_bar,
            timestamp=observed_at,
            cadence_timestamp=boundary,
            regime=regime,
            regime_confidence=self.regime_detector.confidence,
            prices=prices,
            volumes=volumes,
            funding=funding,
            month=observed_at.month,
            regimes_by_symbol=self.regime_detector.symbol_regimes,
            regime_features_by_symbol=self.regime_detector.symbol_stats,
        )
        self._last_cadence_timestamp = boundary
        return policy_market, tuple(qualities), "evaluated", ""

    def _safety_exit_signals(
        self,
        market: MarketSnapshot,
        *,
        tracker: PositionTracker,
        signal_id_start: int,
    ) -> tuple[list[Signal], int]:
        signals: list[Signal] = []
        next_signal_id = int(signal_id_start)
        now = _aware_utc(market.timestamp)
        expired = now >= self.manifest.expires_at
        for symbol, opened in sorted(tracker.all_open().items()):
            if str(opened.by_agent or "") != self.manifest.actor:
                continue
            price = _market_price(market.prices, symbol)
            if price <= 0.0 or opened.entry_price <= 0.0:
                continue
            pnl_pct = (
                (price / opened.entry_price - 1.0) * 100.0
                if opened.side == "long"
                else (opened.entry_price / price - 1.0) * 100.0
            )
            held_minutes = max(0.0, (now - _aware_utc(opened.opened_at)).total_seconds() / 60.0)
            reason = ""
            if expired:
                reason = "policy_manifest_expired"
            elif pnl_pct <= -float(self.manifest.risk.stop_loss_pct):
                reason = "policy_stop_loss"
            elif held_minutes >= float(self.manifest.risk.max_holding_minutes):
                reason = "policy_max_holding"
            if not reason:
                continue
            signals.append(
                Signal(
                    id=next_signal_id,
                    bar=market.bar,
                    sym=symbol,
                    action=Action.FUT_CLOSE_ALL,
                    price=price,
                    regime=market.regime_for_symbol(symbol),
                    by_player=opened.by_player or self.label,
                    by_agent=opened.by_agent or self.manifest.actor,
                    position_scope=self._position_scope(),
                    timestamp=now,
                    metadata=self._signal_metadata(reason, close_reason=reason),
                )
            )
            next_signal_id += 1
        return signals, next_signal_id

    def _position_scope(self) -> str:
        return f"policy_{self.manifest.manifest_sha256[:12]}"

    def _signal_metadata(self, candidate_signal_id: str, **extra: Any) -> dict[str, Any]:
        return {
            "decision_path": "policy_v1",
            "policy_id": self.manifest.policy_id,
            "manifest_sha256": self.manifest.manifest_sha256,
            "candidate_signal_id": candidate_signal_id,
            **extra,
        }


def load_bitget_micro_live_policy_runtime(
    *,
    project_root: str | Path,
    portfolio_value_fn: Any,
    manifest_path: str | Path | None = None,
    expected_sha256: str = "",
    now: datetime | None = None,
) -> LivePolicyRuntimeV1:
    """Load the pinned manifest and instantiate exactly its one supported actor."""

    root = Path(project_root)
    raw_path = str(
        manifest_path
        or os.getenv("BITGET_POLICY_MANIFEST_V1")
        or DEFAULT_POLICY_MANIFEST_PATH
    ).strip()
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    pin = str(expected_sha256 or os.getenv("BITGET_POLICY_MANIFEST_SHA256", "")).strip()
    if not pin:
        raise LivePolicyRuntimeError("policy manifest SHA-256 pin is missing")
    loaded = load_policy_manifest(
        path,
        project_root=root,
        expected_sha256=pin,
        now=now,
        verify_artifacts=True,
        required_target=PolicyTarget.MICRO_LIVE,
        required_exchange="BITGET",
    )

    try:
        from panteon_agents import CarryFlowAgentV2  # type: ignore
    except ImportError as exc:
        raise LivePolicyRuntimeError("CarryFlowAgentV2 runtime is unavailable") from exc
    wrapped = V1AgentAdapter(
        label="CarryFlowAgentV2",
        v1_agent=CarryFlowAgentV2(),
        portfolio_value_fn=portfolio_value_fn,
    )
    adapter = CarryFlowPolicyAdapter(agent=wrapped, manifest=loaded.manifest)
    return LivePolicyRuntimeV1(
        loaded_manifest=loaded,
        actor_adapter=adapter,
        policy_executor=PolicyExecutorV1(loaded),
        risk_state_path=root / DEFAULT_POLICY_RISK_STATE_PATH,
    )


def _canonical_numeric_row(
    row: Mapping[str, Any],
    allowed_symbols: Sequence[str],
    *,
    allow_zero: bool = False,
) -> dict[str, float]:
    allowed = set(allowed_symbols)
    out: dict[str, float] = {}
    for raw_symbol, raw_value in dict(row or {}).items():
        try:
            symbol = canonical_symbol(str(raw_symbol))
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if symbol not in allowed or not math.isfinite(value):
            continue
        if value > 0.0 or (allow_zero and value >= 0.0):
            out[symbol] = value
    return out


def _market_price(prices: Mapping[str, Any], symbol: str) -> float:
    target = canonical_symbol(symbol)
    for raw_symbol, raw_price in dict(prices or {}).items():
        if canonical_symbol(str(raw_symbol)) != target:
            continue
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return 0.0
        return price if math.isfinite(price) and price > 0.0 else 0.0
    return 0.0


def _aware_utc(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("timestamp must be datetime or ISO-8601")
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive_float(value: Any, name: str) -> float:
    parsed = _nonnegative_float(value, name)
    if parsed <= 0.0:
        raise LivePolicyRuntimeError(f"policy_market_{name}_nonpositive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LivePolicyRuntimeError(f"policy_market_{name}_invalid") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise LivePolicyRuntimeError(f"policy_market_{name}_invalid")
    return parsed
