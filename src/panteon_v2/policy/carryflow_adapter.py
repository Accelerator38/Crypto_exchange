"""CarryFlowAgentV2 adapter for the deterministic policy runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from carryflow_policy import apply_carryflow_profile, get_carryflow_profile

from ..domain.types import Action, MarketSnapshot
from ..execution.position_tracker import PositionTracker
from .executor import CandidateSignal
from .manifest import Direction, PolicyManifest, canonical_symbol, format_datetime


CARRYFLOW_REQUIRED_CONFIG_FIELDS = (
    "CHECK_INT",
    "HOLD",
    "STOP",
    "TARGET",
    "FUNDING_ENTRY",
    "FUNDING_EXIT",
    "OI_SPIKE",
    "CROWD_RATIO",
    "BASIS_ENTRY",
    "EXTREME_EXT",
    "EMA_FAST",
    "EMA_SLOW",
    "RSI_N",
    "RSI_OB",
    "RSI_OS",
    "MAX_DATA_AGE_SEC",
    "MAX_POS",
    "ENTRY_COOLDOWN",
    "ALLOW_LONG",
    "ALLOW_SHORT",
)
CARRYFLOW_OPTIONAL_CONFIG_FIELDS = (
    "SHORT_BASIS_FLOOR",
    "EXIT_ON_NORMALIZATION",
    "MIN_HOLD_BEFORE_NORMALIZATION",
    "FLOW_MODEL",
    "OI_LOOKBACK_BARS",
    "PRICE_LOOKBACK_BARS",
    "MIN_PRICE_RETURN",
    "MAX_PRICE_RETURN",
    "REQUIRED_HISTORY",
)
CARRYFLOW_CONFIG_FIELDS = (
    *CARRYFLOW_REQUIRED_CONFIG_FIELDS,
    *CARRYFLOW_OPTIONAL_CONFIG_FIELDS,
)
CARRYFLOW_PROFILE_CONFIG_FIELD = "PROFILE_ID"


class CarryFlowAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class OpenProposal:
    candidate: CandidateSignal
    execution_symbol: str
    action: Action
    diagnostics: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class ExitIntent:
    signal_id: str
    bar: int
    execution_symbol: str
    price: float
    generated_at: datetime
    regime: str
    reason: str


@dataclass(frozen=True)
class ActorActivationTrace:
    bar: int
    timestamp: datetime
    actor: str
    execution_symbol: str
    policy_symbol: str
    regime: str
    raw_action: str
    diagnostic_reason: str
    derivatives_context_present: bool | None
    feature_value: float | None
    expected_move_bps: float | None
    candidate_signal_id: str
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar,
            "timestamp": format_datetime(self.timestamp),
            "actor": self.actor,
            "execution_symbol": self.execution_symbol,
            "policy_symbol": self.policy_symbol,
            "regime": self.regime,
            "raw_action": self.raw_action,
            "diagnostic_reason": self.diagnostic_reason,
            "derivatives_context_present": self.derivatives_context_present,
            "feature_value": self.feature_value,
            "expected_move_bps": self.expected_move_bps,
            "candidate_signal_id": self.candidate_signal_id,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class ActorStep:
    open_proposals: tuple[OpenProposal, ...]
    exit_intents: tuple[ExitIntent, ...]
    traces: tuple[ActorActivationTrace, ...]


class CarryFlowPolicyAdapter:
    """Convert CarryFlow actions into exact policy proposals and exits.

    CarryFlow mutates its internal position state while proposing an action.
    This adapter reconciles that state from PositionTracker before and after
    every bar, so a denied proposal cannot become a phantom position.
    """

    def __init__(
        self,
        *,
        agent: Any,
        manifest: PolicyManifest,
        derivatives_context_provider: Any | None = None,
    ) -> None:
        if manifest.actor != "CarryFlowAgentV2":
            raise CarryFlowAdapterError("manifest actor must be CarryFlowAgentV2")
        if str(getattr(agent, "label", "")) != manifest.actor:
            raise CarryFlowAdapterError("agent label does not match manifest actor")
        runtime = getattr(agent, "v1_agent", None)
        if runtime is None or type(runtime).__name__ != "CarryFlowAgentV2":
            raise CarryFlowAdapterError("agent must wrap CarryFlowAgentV2")
        configured = dict(manifest.actor_config)
        expected = {
            *CARRYFLOW_CONFIG_FIELDS,
            CARRYFLOW_PROFILE_CONFIG_FIELD,
        }
        compact_profile = set(configured) == {CARRYFLOW_PROFILE_CONFIG_FIELD}
        if compact_profile:
            try:
                profile = apply_carryflow_profile(
                    runtime,
                    str(configured[CARRYFLOW_PROFILE_CONFIG_FIELD]),
                )
            except ValueError as exc:
                raise CarryFlowAdapterError(str(exc)) from exc
            _validate_profile_manifest(profile, manifest)
            self.profile_id = profile.profile_id
        else:
            if manifest.target.value != "replay":
                raise CarryFlowAdapterError(
                    "paper and micro-live CarryFlow require PROFILE_ID-only "
                    "actor_config"
                )
            if CARRYFLOW_PROFILE_CONFIG_FIELD in configured:
                raise CarryFlowAdapterError(
                    "PROFILE_ID cannot be mixed with scalar actor_config fields"
                )
            missing = sorted(
                set(CARRYFLOW_REQUIRED_CONFIG_FIELDS) - set(configured)
            )
            extra = sorted(set(configured) - expected)
            if missing:
                raise CarryFlowAdapterError(
                    "missing CarryFlow actor_config fields: "
                    + ", ".join(missing)
                )
            if extra:
                raise CarryFlowAdapterError(
                    "unknown CarryFlow actor_config fields: " + ", ".join(extra)
                )
            for name in CARRYFLOW_CONFIG_FIELDS:
                if name not in configured:
                    continue
                current = getattr(runtime, name, None)
                value = _coerce_config_value(name, configured[name], current)
                setattr(runtime, name, value)
            self.profile_id = ""
        if int(runtime.MAX_DATA_AGE_SEC) != int(
            manifest.data.max_derivatives_age_seconds
        ):
            raise CarryFlowAdapterError(
                "actor_config MAX_DATA_AGE_SEC must match "
                "data.max_derivatives_age_seconds"
            )
        self.agent = agent
        self.runtime = runtime
        self.manifest = manifest
        self.derivatives_context_provider = derivatives_context_provider

    @property
    def required_warmup_bars(self) -> int:
        if self.profile_id:
            return max(1, int(self.runtime.REQUIRED_HISTORY))
        return max(
            int(self.runtime.EMA_SLOW) + 5,
            int(self.runtime.EMA_FAST) + 3,
            int(self.runtime.RSI_N) * 3 + 1,
        )

    def warmup(self, snapshots: Iterable[MarketSnapshot]) -> int:
        """Prime price indicators without derivatives, entries, or evidence."""

        allow_long = bool(self.runtime.ALLOW_LONG)
        allow_short = bool(self.runtime.ALLOW_SHORT)
        count = 0
        self.runtime.ALLOW_LONG = False
        self.runtime.ALLOW_SHORT = False
        try:
            for market in snapshots:
                actions = self.agent.act(market)
                if any(action != Action.HOLD for action in actions.values()):
                    raise CarryFlowAdapterError(
                        "warm-up generated a non-HOLD actor action"
                    )
                count += 1
        finally:
            self.runtime.ALLOW_LONG = allow_long
            self.runtime.ALLOW_SHORT = allow_short
            self.agent.last_signal_diagnostics = {}
        if any(value is not None for value in self.runtime.pos.values()):
            raise CarryFlowAdapterError("warm-up mutated CarryFlow position state")
        return count

    def propose(
        self,
        market: MarketSnapshot,
        *,
        tracker: PositionTracker,
    ) -> ActorStep:
        if self.derivatives_context_provider is not None:
            advance = getattr(self.derivatives_context_provider, "advance", None)
            if not callable(advance):
                raise CarryFlowAdapterError(
                    "derivatives context provider must implement advance(timestamp)"
                )
            advance(market.timestamp)
        self.reconcile(tracker, market_symbols=tuple(market.prices), bar=market.bar)
        actions = self.agent.act(market)
        diagnostics = getattr(self.agent, "last_signal_diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        opens: list[OpenProposal] = []
        exits: list[ExitIntent] = []
        traces: list[ActorActivationTrace] = []

        for execution_symbol in sorted(market.prices):
            policy_symbol = canonical_symbol(execution_symbol)
            action = actions.get(execution_symbol, Action.HOLD)
            if not isinstance(action, Action):
                try:
                    action = Action(int(action))
                except (TypeError, ValueError):
                    action = Action.HOLD
            raw_diag = diagnostics.get(str(execution_symbol).upper(), {})
            diag = dict(raw_diag) if isinstance(raw_diag, Mapping) else {}
            diagnostic_reason = str(diag.get("reason") or "actor_hold")
            regime = market.regime_for_symbol(execution_symbol).label
            feature = _finite_float(diag.get("edge"))
            derivatives_context_present = (
                bool(diag.get("derivatives_context_complete"))
                if "derivatives_context_complete" in diag
                else None
            )

            if action.is_open:
                expected_move = (
                    self.manifest.signal_model.estimate_bps(feature)
                    if feature is not None
                    else None
                )
                candidate_id = (
                    f"{self.manifest.policy_id}:{market.bar}:"
                    f"{policy_symbol}:{action.side}"
                )
                if action not in (
                    Action.FUT_LONG_HALF,
                    Action.FUT_LONG_FULL,
                    Action.FUT_SHORT_HALF,
                    Action.FUT_SHORT_FULL,
                ):
                    outcome = "unsupported_open_action"
                elif derivatives_context_present is not True:
                    outcome = "derivatives_context_missing"
                elif feature is None:
                    outcome = "expected_move_feature_missing"
                elif expected_move is None:
                    outcome = "expected_move_feature_below_model_gate"
                elif expected_move <= 0.0:
                    outcome = "expected_move_nonpositive"
                else:
                    direction = Direction.LONG if action.is_long_open else Direction.SHORT
                    candidate = CandidateSignal(
                        signal_id=candidate_id,
                        bar=market.bar,
                        actor=self.manifest.actor,
                        symbol=policy_symbol,
                        regime=regime,
                        direction=direction,
                        expected_move_bps=expected_move,
                        price=float(market.prices[execution_symbol]),
                        generated_at=market.timestamp,
                    )
                    opens.append(
                        OpenProposal(
                            candidate=candidate,
                            execution_symbol=execution_symbol,
                            action=action,
                            diagnostics=tuple(sorted(diag.items())),
                        )
                    )
                    outcome = "candidate_created"
                traces.append(
                    ActorActivationTrace(
                        bar=market.bar,
                        timestamp=market.timestamp,
                        actor=self.manifest.actor,
                        execution_symbol=execution_symbol,
                        policy_symbol=policy_symbol,
                        regime=regime,
                        raw_action=action.name,
                        diagnostic_reason=diagnostic_reason,
                        derivatives_context_present=derivatives_context_present,
                        feature_value=feature,
                        expected_move_bps=expected_move,
                        candidate_signal_id=candidate_id,
                        outcome=outcome,
                    )
                )
                continue

            if action.is_close:
                position = tracker.get(execution_symbol)
                if position is not None and position.by_agent == self.manifest.actor:
                    signal_id = (
                        f"{self.manifest.policy_id}:{market.bar}:"
                        f"{policy_symbol}:close"
                    )
                    exits.append(
                        ExitIntent(
                            signal_id=signal_id,
                            bar=market.bar,
                            execution_symbol=execution_symbol,
                            price=float(market.prices[execution_symbol]),
                            generated_at=market.timestamp,
                            regime=regime,
                            reason=str(
                                diag.get("close_trigger")
                                or diagnostic_reason
                            ),
                        )
                    )
                    outcome = "exit_intent_created"
                    candidate_id = signal_id
                else:
                    outcome = "close_without_owned_position"
                    candidate_id = ""
                traces.append(
                    ActorActivationTrace(
                        bar=market.bar,
                        timestamp=market.timestamp,
                        actor=self.manifest.actor,
                        execution_symbol=execution_symbol,
                        policy_symbol=policy_symbol,
                        regime=regime,
                        raw_action=action.name,
                        diagnostic_reason=diagnostic_reason,
                        derivatives_context_present=derivatives_context_present,
                        feature_value=feature,
                        expected_move_bps=None,
                        candidate_signal_id=candidate_id,
                        outcome=outcome,
                    )
                )
                continue

            traces.append(
                ActorActivationTrace(
                    bar=market.bar,
                    timestamp=market.timestamp,
                    actor=self.manifest.actor,
                    execution_symbol=execution_symbol,
                    policy_symbol=policy_symbol,
                    regime=regime,
                    raw_action=Action.HOLD.name,
                    diagnostic_reason=diagnostic_reason,
                    derivatives_context_present=derivatives_context_present,
                    feature_value=feature,
                    expected_move_bps=None,
                    candidate_signal_id="",
                    outcome="actor_hold",
                )
            )

        return ActorStep(
            open_proposals=tuple(opens),
            exit_intents=tuple(exits),
            traces=tuple(traces),
        )

    def reconcile(
        self,
        tracker: PositionTracker,
        *,
        market_symbols: tuple[str, ...],
        bar: int,
    ) -> None:
        symbols = set(str(sym).upper() for sym in market_symbols)
        symbols.update(str(sym).upper() for sym in getattr(self.runtime, "pos", {}))
        for symbol in symbols:
            position = tracker.get(symbol)
            owned = position is not None and position.by_agent == self.manifest.actor
            self.runtime.pos[symbol] = position.side if owned else None
            self.runtime.ep[symbol] = float(position.entry_price) if owned else 0.0
            self.runtime.et[symbol] = int(position.opened_bar) if owned else 0
            if not owned and getattr(self.runtime, "last_entry", {}).get(symbol) == bar:
                self.runtime.last_entry.pop(symbol, None)


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _coerce_config_value(name: str, value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise CarryFlowAdapterError(f"actor_config {name} must be bool")
        return value
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool) or int(value) != float(value):
            raise CarryFlowAdapterError(f"actor_config {name} must be int")
        return int(value)
    if isinstance(current, float):
        parsed = _finite_float(value)
        if parsed is None:
            raise CarryFlowAdapterError(f"actor_config {name} must be finite")
        return parsed
    if isinstance(current, str):
        if not isinstance(value, str) or not value.strip():
            raise CarryFlowAdapterError(f"actor_config {name} must be string")
        return value.strip()
    raise CarryFlowAdapterError(f"unsupported CarryFlow config field: {name}")


def _validate_profile_manifest(profile: Any, manifest: PolicyManifest) -> None:
    interval_minutes = manifest.data.bar_interval_seconds / 60.0
    expected_holding = profile.hold_bars * interval_minutes
    checks = {
        "risk.stop_loss_pct": (
            float(manifest.risk.stop_loss_pct),
            profile.stop_pct * 100.0,
        ),
        "risk.max_holding_minutes": (
            float(manifest.risk.max_holding_minutes),
            expected_holding,
        ),
        "risk.max_open_positions": (
            float(manifest.risk.max_open_positions),
            float(profile.max_positions),
        ),
        "risk.max_signal_age_seconds": (
            float(manifest.risk.max_signal_age_seconds),
            float(profile.max_signal_age_seconds),
        ),
        "data.max_derivatives_age_seconds": (
            float(manifest.data.max_derivatives_age_seconds),
            float(profile.max_data_age_seconds),
        ),
        "signal.slope_bps_per_unit": (
            float(manifest.signal_model.slope_bps_per_unit),
            float(profile.signal_slope_bps_per_unit),
        ),
        "signal.lcb_haircut_bps": (
            float(manifest.signal_model.lcb_haircut_bps),
            float(profile.signal_lcb_haircut_bps),
        ),
        "signal.min_feature_value": (
            float(manifest.signal_model.min_feature_value),
            float(profile.signal_min_feature),
        ),
        "signal.max_expected_move_bps": (
            float(manifest.signal_model.max_expected_move_bps),
            float(profile.signal_max_expected_move_bps),
        ),
    }
    for name, (actual, expected) in checks.items():
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise CarryFlowAdapterError(
                f"profile manifest mismatch: {name}={actual}, expected {expected}"
            )
    if manifest.signal_model.feature != "diagnostic.edge":
        raise CarryFlowAdapterError(
            "profile manifest mismatch: signal.feature must be diagnostic.edge"
        )
    if not math.isclose(
        float(manifest.signal_model.intercept_bps),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CarryFlowAdapterError(
            "profile manifest mismatch: signal.intercept_bps must be zero"
        )
    expected_regimes = set(profile.allowed_regimes)
    expected_move_bps = (
        float(manifest.costs.round_trip_fee_bps)
        + float(manifest.costs.slippage_bps)
        + float(manifest.costs.safety_buffer_bps)
    )
    rules_by_symbol: dict[str, set[str]] = {}
    for rule in manifest.rules:
        if rule.direction.value != "SHORT":
            raise CarryFlowAdapterError(
                "profile manifest mismatch: only SHORT rules are allowed"
            )
        if not math.isclose(
            float(rule.min_regime_confidence),
            float(profile.min_regime_confidence),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CarryFlowAdapterError(
                "profile manifest mismatch: rule min_regime_confidence must "
                f"be {profile.min_regime_confidence}"
            )
        if not math.isclose(
            float(rule.min_expected_move_bps),
            expected_move_bps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CarryFlowAdapterError(
                "profile manifest mismatch: rule min_expected_move_bps must "
                f"equal total modeled costs ({expected_move_bps})"
            )
        rule_checks = {
            "max_spread_bps": (
                float(rule.max_spread_bps),
                float(profile.max_spread_bps),
            ),
            "max_slippage_bps": (
                float(rule.max_slippage_bps),
                float(profile.max_slippage_bps),
            ),
            "risk_mult": (
                float(rule.risk_mult),
                float(profile.risk_mult),
            ),
        }
        for name, (actual, expected) in rule_checks.items():
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise CarryFlowAdapterError(
                    f"profile manifest mismatch: rule {name}={actual}, "
                    f"expected {expected}"
                )
        rules_by_symbol.setdefault(rule.symbol, set()).add(rule.regime)
    for symbol, regimes in sorted(rules_by_symbol.items()):
        if regimes != expected_regimes:
            missing = sorted(expected_regimes - regimes)
            extra = sorted(regimes - expected_regimes)
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise CarryFlowAdapterError(
                f"profile manifest regime coverage mismatch for {symbol}: "
                + "; ".join(details)
            )
