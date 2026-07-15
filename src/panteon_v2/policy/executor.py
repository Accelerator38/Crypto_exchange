"""Deterministic entry authorizer for Pantheon vNext.

The executor has no exchange I/O and no mutable selection state. Its only output
is ALLOW_OPEN or NO_TRADE plus the first failed check. TradeExecutor remains the
single path from an authorized domain Signal to an exchange order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .manifest import (
    Direction,
    LoadedPolicyManifest,
    ManifestError,
    PolicyRule,
    PolicyTarget,
    aware_utc,
    canonical_exchange,
    canonical_regime,
    canonical_symbol,
    format_datetime,
    load_policy_manifest,
)


class DecisionOutcome(str, Enum):
    ALLOW_OPEN = "ALLOW_OPEN"
    NO_TRADE = "NO_TRADE"


class DecisionReason(str, Enum):
    ALLOWED = "allowed"
    MANIFEST_UNAVAILABLE = "manifest_unavailable"
    MANIFEST_INVALID = "manifest_invalid"
    MODE_MISMATCH = "mode_mismatch"
    EXCHANGE_MISMATCH = "exchange_mismatch"
    MANIFEST_EXPIRED = "manifest_expired"
    KILL_SWITCH = "kill_switch"
    EXCHANGE_UNHEALTHY = "exchange_unhealthy"
    INVALID_PRICE = "invalid_price"
    SIGNAL_FROM_FUTURE = "signal_from_future"
    SIGNAL_STALE = "signal_stale"
    ACTOR_NOT_ALLOWED = "actor_not_allowed"
    SYMBOL_NOT_ALLOWED = "symbol_not_allowed"
    REGIME_NOT_ALLOWED = "regime_not_allowed"
    DIRECTION_NOT_ALLOWED = "direction_not_allowed"
    REGIME_CONFIDENCE_LOW = "regime_confidence_low"
    SPREAD_TOO_WIDE = "spread_too_wide"
    SLIPPAGE_TOO_HIGH = "slippage_too_high"
    EXPECTED_MOVE_BELOW_COST = "expected_move_below_cost"
    MAX_OPEN_POSITIONS = "max_open_positions"
    DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass(frozen=True)
class CandidateSignal:
    signal_id: str
    bar: int
    actor: str
    symbol: str
    regime: str
    direction: Direction
    expected_move_bps: float
    price: float
    generated_at: datetime

    def __post_init__(self) -> None:
        if not str(self.signal_id or "").strip():
            raise ValueError("CandidateSignal.signal_id must be non-empty")
        if int(self.bar) < 0:
            raise ValueError("CandidateSignal.bar must be non-negative")
        if not str(self.actor or "").strip():
            raise ValueError("CandidateSignal.actor must be non-empty")
        object.__setattr__(self, "symbol", canonical_symbol(self.symbol))
        object.__setattr__(self, "regime", canonical_regime(self.regime))
        if not isinstance(self.direction, Direction):
            object.__setattr__(self, "direction", Direction(str(self.direction).upper()))
        for name in ("expected_move_bps", "price"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"CandidateSignal.{name} must be finite")
        object.__setattr__(self, "generated_at", aware_utc(self.generated_at))


@dataclass(frozen=True)
class RuntimeContext:
    exchange: str
    mode: PolicyTarget
    now: datetime
    spread_bps: float
    estimated_slippage_bps: float
    regime_confidence: float
    exchange_healthy: bool = True
    kill_switch_active: bool = False
    open_positions: int = 0
    daily_loss_usd: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange", canonical_exchange(self.exchange))
        if not isinstance(self.mode, PolicyTarget):
            object.__setattr__(self, "mode", PolicyTarget(str(self.mode)))
        object.__setattr__(self, "now", aware_utc(self.now))
        for name in ("spread_bps", "estimated_slippage_bps", "daily_loss_usd"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"RuntimeContext.{name} must be finite and non-negative")
        if not 0.0 <= float(self.regime_confidence) <= 1.0:
            raise ValueError("RuntimeContext.regime_confidence must be in [0, 1]")
        if int(self.open_positions) < 0:
            raise ValueError("RuntimeContext.open_positions must be non-negative")


@dataclass(frozen=True)
class PolicyCheck:
    name: str
    passed: bool
    observed: Any
    required: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "required": self.required,
        }


@dataclass(frozen=True)
class DecisionTrace:
    signal_id: str
    bar: int
    timestamp: datetime
    policy_id: str
    manifest_sha256: str
    actor: str
    symbol: str
    regime: str
    direction: str
    outcome: DecisionOutcome
    primary_reason: DecisionReason
    detail: str
    checks: tuple[PolicyCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "bar": self.bar,
            "timestamp": format_datetime(self.timestamp),
            "policy_id": self.policy_id,
            "manifest_sha256": self.manifest_sha256,
            "actor": self.actor,
            "symbol": self.symbol,
            "regime": self.regime,
            "direction": self.direction,
            "outcome": self.outcome.value,
            "primary_reason": self.primary_reason.value,
            "detail": self.detail,
            "checks": [check.as_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class PolicyDecision:
    outcome: DecisionOutcome
    trace: DecisionTrace
    rule: PolicyRule | None = None
    risk_mult: float = 0.0
    max_notional_usd: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.outcome == DecisionOutcome.ALLOW_OPEN


class PolicyExecutorV1:
    """Authorize one candidate signal against one loaded policy manifest."""

    def __init__(
        self,
        loaded: LoadedPolicyManifest | None,
        *,
        invalid_reasons: tuple[str, ...] = (),
    ) -> None:
        self._loaded = loaded
        self._invalid_reasons = tuple(invalid_reasons)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        project_root: str | Path,
        expected_sha256: str,
        now: datetime | None = None,
        required_target: PolicyTarget | str | None = None,
        required_exchange: str = "BITGET",
    ) -> "PolicyExecutorV1":
        try:
            loaded = load_policy_manifest(
                path,
                project_root=project_root,
                expected_sha256=expected_sha256,
                now=now,
                required_target=required_target,
                required_exchange=required_exchange,
            )
        except ManifestError as exc:
            return cls(None, invalid_reasons=exc.reasons)
        return cls(loaded)

    @classmethod
    def unavailable(cls) -> "PolicyExecutorV1":
        return cls(None)

    @property
    def ready(self) -> bool:
        return self._loaded is not None and not self._invalid_reasons

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        return self._invalid_reasons

    def decide(
        self,
        signal: CandidateSignal,
        context: RuntimeContext,
    ) -> PolicyDecision:
        checks: list[PolicyCheck] = []
        if self._loaded is None:
            reason = (
                DecisionReason.MANIFEST_INVALID
                if self._invalid_reasons
                else DecisionReason.MANIFEST_UNAVAILABLE
            )
            detail = ",".join(self._invalid_reasons)
            return self._deny(signal, context, reason, detail, checks)

        manifest = self._loaded.manifest
        failure = self._check(
            checks,
            "mode",
            context.mode == manifest.target,
            context.mode.value,
            manifest.target.value,
            DecisionReason.MODE_MISMATCH,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "exchange",
            context.exchange == manifest.exchange,
            context.exchange,
            manifest.exchange,
            DecisionReason.EXCHANGE_MISMATCH,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "manifest_not_expired",
            context.now < manifest.expires_at,
            format_datetime(context.now),
            f"before {format_datetime(manifest.expires_at)}",
            DecisionReason.MANIFEST_EXPIRED,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "kill_switch",
            not context.kill_switch_active,
            context.kill_switch_active,
            False,
            DecisionReason.KILL_SWITCH,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "exchange_health",
            context.exchange_healthy,
            context.exchange_healthy,
            True,
            DecisionReason.EXCHANGE_UNHEALTHY,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "price",
            signal.price > 0.0,
            signal.price,
            "> 0",
            DecisionReason.INVALID_PRICE,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)

        signal_age = (context.now - signal.generated_at).total_seconds()
        failure = self._check(
            checks,
            "signal_not_from_future",
            signal_age >= -5.0,
            signal_age,
            ">= -5 seconds",
            DecisionReason.SIGNAL_FROM_FUTURE,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "signal_age_seconds",
            signal_age <= manifest.risk.max_signal_age_seconds,
            signal_age,
            f"<= {manifest.risk.max_signal_age_seconds}",
            DecisionReason.SIGNAL_STALE,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "actor",
            signal.actor == manifest.actor,
            signal.actor,
            manifest.actor,
            DecisionReason.ACTOR_NOT_ALLOWED,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)

        symbol_rules = tuple(rule for rule in manifest.rules if rule.symbol == signal.symbol)
        failure = self._check(
            checks,
            "symbol",
            bool(symbol_rules),
            signal.symbol,
            sorted({rule.symbol for rule in manifest.rules}),
            DecisionReason.SYMBOL_NOT_ALLOWED,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        regime_rules = tuple(rule for rule in symbol_rules if rule.regime == signal.regime)
        failure = self._check(
            checks,
            "regime",
            bool(regime_rules),
            signal.regime,
            sorted({rule.regime for rule in symbol_rules}),
            DecisionReason.REGIME_NOT_ALLOWED,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        direction_rules = tuple(
            rule for rule in regime_rules if rule.direction == signal.direction
        )
        failure = self._check(
            checks,
            "direction",
            bool(direction_rules),
            signal.direction.value,
            sorted({rule.direction.value for rule in regime_rules}),
            DecisionReason.DIRECTION_NOT_ALLOWED,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        rule = direction_rules[0]

        failure = self._check(
            checks,
            "regime_confidence",
            context.regime_confidence >= rule.min_regime_confidence,
            context.regime_confidence,
            f">= {rule.min_regime_confidence}",
            DecisionReason.REGIME_CONFIDENCE_LOW,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "spread_bps",
            context.spread_bps <= rule.max_spread_bps,
            context.spread_bps,
            f"<= {rule.max_spread_bps}",
            DecisionReason.SPREAD_TOO_WIDE,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "slippage_bps",
            context.estimated_slippage_bps <= rule.max_slippage_bps,
            context.estimated_slippage_bps,
            f"<= {rule.max_slippage_bps}",
            DecisionReason.SLIPPAGE_TOO_HIGH,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)

        runtime_cost_bps = (
            manifest.costs.round_trip_fee_bps
            + max(manifest.costs.slippage_bps, context.estimated_slippage_bps)
            + manifest.costs.safety_buffer_bps
        )
        required_move_bps = max(rule.min_expected_move_bps, runtime_cost_bps)
        failure = self._check(
            checks,
            "expected_move_bps",
            signal.expected_move_bps >= required_move_bps,
            signal.expected_move_bps,
            f">= {required_move_bps}",
            DecisionReason.EXPECTED_MOVE_BELOW_COST,
        )
        if failure:
            detail = (
                f"fee={manifest.costs.round_trip_fee_bps},"
                f"slippage={max(manifest.costs.slippage_bps, context.estimated_slippage_bps)},"
                f"buffer={manifest.costs.safety_buffer_bps}"
            )
            return self._deny(signal, context, failure, detail, checks)
        failure = self._check(
            checks,
            "open_positions",
            context.open_positions < manifest.risk.max_open_positions,
            context.open_positions,
            f"< {manifest.risk.max_open_positions}",
            DecisionReason.MAX_OPEN_POSITIONS,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)
        failure = self._check(
            checks,
            "daily_loss_usd",
            context.daily_loss_usd < manifest.risk.max_daily_loss_usd,
            context.daily_loss_usd,
            f"< {manifest.risk.max_daily_loss_usd}",
            DecisionReason.DAILY_LOSS_LIMIT,
        )
        if failure:
            return self._deny(signal, context, failure, "", checks)

        checks.append(PolicyCheck("decision", True, "ALLOW_OPEN", "ALLOW_OPEN"))
        trace = self._trace(
            signal,
            context,
            DecisionOutcome.ALLOW_OPEN,
            DecisionReason.ALLOWED,
            "",
            checks,
        )
        return PolicyDecision(
            outcome=DecisionOutcome.ALLOW_OPEN,
            trace=trace,
            rule=rule,
            risk_mult=rule.risk_mult,
            max_notional_usd=manifest.risk.max_notional_usd,
        )

    @staticmethod
    def _check(
        checks: list[PolicyCheck],
        name: str,
        passed: bool,
        observed: Any,
        required: Any,
        reason: DecisionReason,
    ) -> DecisionReason | None:
        checks.append(PolicyCheck(name, bool(passed), observed, required))
        return None if passed else reason

    def _deny(
        self,
        signal: CandidateSignal,
        context: RuntimeContext,
        reason: DecisionReason,
        detail: str,
        checks: list[PolicyCheck],
    ) -> PolicyDecision:
        return PolicyDecision(
            outcome=DecisionOutcome.NO_TRADE,
            trace=self._trace(
                signal,
                context,
                DecisionOutcome.NO_TRADE,
                reason,
                detail,
                checks,
            ),
        )

    def _trace(
        self,
        signal: CandidateSignal,
        context: RuntimeContext,
        outcome: DecisionOutcome,
        reason: DecisionReason,
        detail: str,
        checks: list[PolicyCheck],
    ) -> DecisionTrace:
        manifest = self._loaded.manifest if self._loaded is not None else None
        return DecisionTrace(
            signal_id=signal.signal_id,
            bar=signal.bar,
            timestamp=context.now,
            policy_id=manifest.policy_id if manifest else "",
            manifest_sha256=manifest.manifest_sha256 if manifest else "",
            actor=signal.actor,
            symbol=signal.symbol,
            regime=signal.regime,
            direction=signal.direction.value,
            outcome=outcome,
            primary_reason=reason,
            detail=detail,
            checks=tuple(checks),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
