"""Immutable, fail-closed policy manifest for Pantheon vNext.

This module is intentionally independent from Flash, Strategist and GeneticsCore.
It turns evidence into a fixed runtime contract; it does not discover strategies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "panteon.policy.v1"
EVIDENCE_SCHEMA_VERSION = "panteon.policy_evidence.v1"
RUNTIME_FINGERPRINT_PATHS = (
    "src/panteon_runtime/panteon_agents.py",
    "src/panteon_v2/domain/types.py",
    "src/panteon_v2/shadow/adapters.py",
    "src/panteon_v2/policy/manifest.py",
    "src/panteon_v2/policy/executor.py",
    "src/panteon_v2/policy/carryflow_adapter.py",
    "src/panteon_v2/policy/derivatives_context.py",
    "src/panteon_v2/policy/evidence_tape.py",
    "src/panteon_v2/policy/replay_runner.py",
    "src/panteon_v2/execution/exchange.py",
    "src/panteon_v2/execution/executor.py",
    "src/panteon_v2/execution/order_ledger.py",
    "src/panteon_v2/execution/position_tracker.py",
    "src/panteon_v2/execution/risk_limits.py",
    "src/panteon_v2/execution/symbol_health.py",
    "src/panteon_v2/memory/performance.py",
    "src/panteon_v2/attribution/event_log.py",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")
_KNOWN_REGIMES = frozenset(
    {
        "bullish",
        "bearish",
        "neutral",
        "crash",
        "range_low_vol",
        "choppy_down",
        "choppy_up",
        "mixed_rotational",
        "range_low_vol_transition",
    }
)


class ManifestError(ValueError):
    """A manifest cannot be trusted or used."""

    def __init__(self, reasons: Sequence[str] | str):
        if isinstance(reasons, str):
            clean = (reasons,)
        else:
            clean = tuple(str(item) for item in reasons if str(item))
        self.reasons = clean or ("manifest_invalid",)
        super().__init__("; ".join(self.reasons))


class PolicyTarget(str, Enum):
    REPLAY = "replay"
    PAPER = "paper"
    MICRO_LIVE = "micro_live"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EvidenceKind(str, Enum):
    VALIDATION = "validation"
    OOS = "oos"
    COST_STRESS = "cost_stress"
    SANITY = "sanity"
    SHORT_PAPER_CANARY = "short_paper_canary"
    EXTENDED_PAPER_CANARY = "extended_paper_canary"


@dataclass(frozen=True)
class SignalModel:
    feature: str
    intercept_bps: float
    slope_bps_per_unit: float
    lcb_haircut_bps: float
    min_feature_value: float
    max_expected_move_bps: float

    def __post_init__(self) -> None:
        if str(self.feature or "").strip() != "diagnostic.edge":
            raise ManifestError("signal_model.feature_unsupported")
        for name in (
            "intercept_bps",
            "slope_bps_per_unit",
            "lcb_haircut_bps",
            "min_feature_value",
            "max_expected_move_bps",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ManifestError(f"signal_model.{name}_nonfinite")
        if float(self.slope_bps_per_unit) <= 0.0:
            raise ManifestError("signal_model.slope_bps_per_unit_nonpositive")
        if float(self.lcb_haircut_bps) < 0.0:
            raise ManifestError("signal_model.lcb_haircut_bps_negative")
        if float(self.min_feature_value) < 0.0:
            raise ManifestError("signal_model.min_feature_value_negative")
        if float(self.max_expected_move_bps) <= 0.0:
            raise ManifestError("signal_model.max_expected_move_bps_nonpositive")

    def estimate_bps(self, feature_value: float) -> float | None:
        value = float(feature_value)
        if not math.isfinite(value) or value < self.min_feature_value:
            return None
        estimate = (
            self.intercept_bps
            + self.slope_bps_per_unit * value
            - self.lcb_haircut_bps
        )
        return max(0.0, min(float(self.max_expected_move_bps), estimate))


@dataclass(frozen=True)
class MarketDataPolicy:
    bar_interval_seconds: int
    cadence_tolerance_seconds: int
    max_bar_close_lag_seconds: int
    max_derivatives_age_seconds: int
    required_context_coverage_pct: float

    def __post_init__(self) -> None:
        interval = int(self.bar_interval_seconds)
        tolerance = int(self.cadence_tolerance_seconds)
        close_lag = int(self.max_bar_close_lag_seconds)
        max_age = int(self.max_derivatives_age_seconds)
        coverage = float(self.required_context_coverage_pct)
        if not 60 <= interval <= 24 * 60 * 60 or interval % 60 != 0:
            raise ManifestError("data.bar_interval_seconds_invalid")
        if not 0 <= tolerance <= min(300, interval // 10):
            raise ManifestError("data.cadence_tolerance_seconds_invalid")
        if not 1 <= close_lag <= 300:
            raise ManifestError("data.max_bar_close_lag_seconds_invalid")
        if not 60 <= max_age <= 60 * 60:
            raise ManifestError("data.max_derivatives_age_seconds_invalid")
        if not 95.0 <= coverage <= 100.0:
            raise ManifestError("data.required_context_coverage_pct_invalid")


@dataclass(frozen=True)
class CostModel:
    round_trip_fee_bps: float
    slippage_bps: float
    safety_buffer_bps: float

    def __post_init__(self) -> None:
        _finite_nonnegative("round_trip_fee_bps", self.round_trip_fee_bps)
        _finite_nonnegative("slippage_bps", self.slippage_bps)
        _finite_nonnegative("safety_buffer_bps", self.safety_buffer_bps)
        if self.round_trip_fee_bps <= 0.0:
            raise ManifestError("cost_model.round_trip_fee_bps_nonpositive")
        if self.safety_buffer_bps <= 0.0:
            raise ManifestError("cost_model.safety_buffer_bps_nonpositive")

    @property
    def required_move_bps(self) -> float:
        return (
            float(self.round_trip_fee_bps)
            + float(self.slippage_bps)
            + float(self.safety_buffer_bps)
        )


@dataclass(frozen=True)
class RiskPolicy:
    capital_fraction: float
    max_notional_usd: float
    max_open_positions: int
    max_daily_loss_usd: float
    stop_loss_pct: float
    max_holding_minutes: int
    max_signal_age_seconds: int = 120

    def __post_init__(self) -> None:
        if not 0.0 < float(self.capital_fraction) <= 0.10:
            raise ManifestError("risk.capital_fraction_out_of_range")
        if not 0.0 < float(self.max_notional_usd) <= 1000.0:
            raise ManifestError("risk.max_notional_usd_out_of_range")
        if not 1 <= int(self.max_open_positions) <= 8:
            raise ManifestError("risk.max_open_positions_out_of_range")
        if float(self.max_daily_loss_usd) <= 0.0:
            raise ManifestError("risk.max_daily_loss_usd_nonpositive")
        if not 0.0 < float(self.stop_loss_pct) <= 10.0:
            raise ManifestError("risk.stop_loss_pct_out_of_range")
        if int(self.max_holding_minutes) <= 0:
            raise ManifestError("risk.max_holding_minutes_nonpositive")
        if not 1 <= int(self.max_signal_age_seconds) <= 900:
            raise ManifestError("risk.max_signal_age_seconds_out_of_range")


@dataclass(frozen=True)
class PolicyRule:
    symbol: str
    regime: str
    direction: Direction
    min_expected_move_bps: float
    max_spread_bps: float
    max_slippage_bps: float
    min_regime_confidence: float
    risk_mult: float = 1.0

    def __post_init__(self) -> None:
        symbol = canonical_symbol(self.symbol)
        regime = canonical_regime(self.regime)
        direction = _enum_value(Direction, self.direction, "rule.direction")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "direction", direction)
        if float(self.min_expected_move_bps) <= 0.0:
            raise ManifestError("rule.min_expected_move_bps_nonpositive")
        _finite_nonnegative("rule.max_spread_bps", self.max_spread_bps)
        _finite_nonnegative("rule.max_slippage_bps", self.max_slippage_bps)
        if not 0.0 <= float(self.min_regime_confidence) <= 1.0:
            raise ManifestError("rule.min_regime_confidence_out_of_range")
        if not 0.0 < float(self.risk_mult) <= 1.0:
            raise ManifestError("rule.risk_mult_out_of_range")

    @property
    def key(self) -> tuple[str, str, Direction]:
        return (self.symbol, self.regime, self.direction)


@dataclass(frozen=True)
class EvidenceGate:
    kind: EvidenceKind
    candidate_key: str
    exchange: str
    artifact_path: str
    artifact_sha256: str
    generated_at: datetime
    data_as_of: datetime
    passed: bool
    signals: int = 0
    orders: int = 0
    fills: int = 0
    closed_trades: int = 0
    expectancy_after_costs_usd: float = 0.0
    expectancy_lcb_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    mean_cost_bps: float = 0.0
    direction_collapse: bool = False
    regime_collapse: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum_value(EvidenceKind, self.kind, "evidence.kind"))
        object.__setattr__(self, "exchange", canonical_exchange(self.exchange))
        object.__setattr__(self, "generated_at", aware_utc(self.generated_at))
        object.__setattr__(self, "data_as_of", aware_utc(self.data_as_of))
        if not str(self.candidate_key or "").strip():
            raise ManifestError("evidence.candidate_key_missing")
        if not str(self.artifact_path or "").strip():
            raise ManifestError("evidence.artifact_path_missing")
        digest = str(self.artifact_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ManifestError("evidence.artifact_sha256_invalid")
        object.__setattr__(self, "artifact_sha256", digest)
        for name in ("signals", "orders", "fills", "closed_trades"):
            if int(getattr(self, name)) < 0:
                raise ManifestError(f"evidence.{name}_negative")
        for name in (
            "expectancy_after_costs_usd",
            "expectancy_lcb_usd",
            "max_drawdown_pct",
            "mean_cost_bps",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ManifestError(f"evidence.{name}_nonfinite")


@dataclass(frozen=True)
class PolicyManifest:
    schema_version: str
    policy_id: str
    target: PolicyTarget
    exchange: str
    actor: str
    actor_config: tuple[tuple[str, bool | int | float | str], ...]
    signal_model: SignalModel
    data: MarketDataPolicy
    created_at: datetime
    expires_at: datetime
    source_revision: str
    runtime_fingerprint_sha256: str
    costs: CostModel
    risk: RiskPolicy
    rules: tuple[PolicyRule, ...]
    evidence: tuple[EvidenceGate, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError("manifest.schema_version_unsupported")
        if not str(self.policy_id or "").strip():
            raise ManifestError("manifest.policy_id_missing")
        if not str(self.actor or "").strip():
            raise ManifestError("manifest.actor_missing")
        if not self.actor_config:
            raise ManifestError("manifest.actor_config_empty")
        keys = [str(key) for key, _ in self.actor_config]
        if len(keys) != len(set(keys)):
            raise ManifestError("manifest.actor_config_duplicate_key")
        object.__setattr__(self, "target", _enum_value(PolicyTarget, self.target, "manifest.target"))
        object.__setattr__(self, "exchange", canonical_exchange(self.exchange))
        object.__setattr__(self, "created_at", aware_utc(self.created_at))
        object.__setattr__(self, "expires_at", aware_utc(self.expires_at))
        revision = str(self.source_revision or "").strip().lower()
        if not _REVISION_RE.fullmatch(revision):
            raise ManifestError("manifest.source_revision_invalid")
        object.__setattr__(self, "source_revision", revision)
        runtime_digest = str(self.runtime_fingerprint_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(runtime_digest):
            raise ManifestError("manifest.runtime_fingerprint_sha256_invalid")
        object.__setattr__(self, "runtime_fingerprint_sha256", runtime_digest)
        digest = str(self.manifest_sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ManifestError("manifest.sha256_invalid")
        object.__setattr__(self, "manifest_sha256", digest)
        if not self.rules:
            raise ManifestError("manifest.rules_empty")
        rule_keys = [rule.key for rule in self.rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise ManifestError("manifest.rules_duplicate")
        kinds = [gate.kind for gate in self.evidence]
        if len(kinds) != len(set(kinds)):
            raise ManifestError("manifest.evidence_duplicate_kind")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "target": self.target.value,
            "exchange": self.exchange,
            "actor": self.actor,
            "actor_config": dict(self.actor_config),
            "signal_model": {
                "feature": self.signal_model.feature,
                "intercept_bps": self.signal_model.intercept_bps,
                "slope_bps_per_unit": self.signal_model.slope_bps_per_unit,
                "lcb_haircut_bps": self.signal_model.lcb_haircut_bps,
                "min_feature_value": self.signal_model.min_feature_value,
                "max_expected_move_bps": self.signal_model.max_expected_move_bps,
            },
            "data": {
                "bar_interval_seconds": self.data.bar_interval_seconds,
                "cadence_tolerance_seconds": self.data.cadence_tolerance_seconds,
                "max_bar_close_lag_seconds": (
                    self.data.max_bar_close_lag_seconds
                ),
                "max_derivatives_age_seconds": self.data.max_derivatives_age_seconds,
                "required_context_coverage_pct": (
                    self.data.required_context_coverage_pct
                ),
            },
            "created_at": format_datetime(self.created_at),
            "expires_at": format_datetime(self.expires_at),
            "source_revision": self.source_revision,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "costs": {
                "round_trip_fee_bps": self.costs.round_trip_fee_bps,
                "slippage_bps": self.costs.slippage_bps,
                "safety_buffer_bps": self.costs.safety_buffer_bps,
            },
            "risk": {
                "capital_fraction": self.risk.capital_fraction,
                "max_notional_usd": self.risk.max_notional_usd,
                "max_open_positions": self.risk.max_open_positions,
                "max_daily_loss_usd": self.risk.max_daily_loss_usd,
                "stop_loss_pct": self.risk.stop_loss_pct,
                "max_holding_minutes": self.risk.max_holding_minutes,
                "max_signal_age_seconds": self.risk.max_signal_age_seconds,
            },
            "rules": [
                {
                    "symbol": rule.symbol,
                    "regime": rule.regime,
                    "direction": rule.direction.value,
                    "min_expected_move_bps": rule.min_expected_move_bps,
                    "max_spread_bps": rule.max_spread_bps,
                    "max_slippage_bps": rule.max_slippage_bps,
                    "min_regime_confidence": rule.min_regime_confidence,
                    "risk_mult": rule.risk_mult,
                }
                for rule in self.rules
            ],
            "evidence": [
                {
                    "kind": gate.kind.value,
                    "candidate_key": gate.candidate_key,
                    "exchange": gate.exchange,
                    "artifact_path": gate.artifact_path,
                    "artifact_sha256": gate.artifact_sha256,
                    "generated_at": format_datetime(gate.generated_at),
                    "data_as_of": format_datetime(gate.data_as_of),
                    "passed": gate.passed,
                    "signals": gate.signals,
                    "orders": gate.orders,
                    "fills": gate.fills,
                    "closed_trades": gate.closed_trades,
                    "expectancy_after_costs_usd": gate.expectancy_after_costs_usd,
                    "expectancy_lcb_usd": gate.expectancy_lcb_usd,
                    "max_drawdown_pct": gate.max_drawdown_pct,
                    "mean_cost_bps": gate.mean_cost_bps,
                    "direction_collapse": gate.direction_collapse,
                    "regime_collapse": gate.regime_collapse,
                }
                for gate in self.evidence
            ],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class ManifestValidation:
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedPolicyManifest:
    manifest: PolicyManifest
    path: str
    validation: ManifestValidation


@dataclass(frozen=True)
class _GateRequirement:
    min_signals: int = 0
    min_orders: int = 0
    min_fills: int = 0
    min_closed: int = 0
    require_positive_expectancy: bool = False
    require_positive_lcb: bool = False
    require_costs: bool = False
    max_drawdown_pct: float = 20.0
    max_age_hours: float = 168.0


_GATE_REQUIREMENTS: dict[EvidenceKind, _GateRequirement] = {
    EvidenceKind.VALIDATION: _GateRequirement(
        min_fills=20,
        min_closed=10,
        require_positive_expectancy=True,
        require_positive_lcb=True,
        require_costs=True,
    ),
    EvidenceKind.OOS: _GateRequirement(
        min_fills=20,
        min_closed=10,
        require_positive_expectancy=True,
        require_positive_lcb=True,
        require_costs=True,
    ),
    EvidenceKind.COST_STRESS: _GateRequirement(
        min_fills=20,
        min_closed=10,
        require_positive_expectancy=True,
        require_positive_lcb=True,
        require_costs=True,
    ),
    EvidenceKind.SANITY: _GateRequirement(),
    EvidenceKind.SHORT_PAPER_CANARY: _GateRequirement(
        min_signals=1,
        min_orders=1,
        min_fills=1,
        min_closed=1,
        require_positive_expectancy=True,
        require_costs=True,
        max_age_hours=24.0,
    ),
    EvidenceKind.EXTENDED_PAPER_CANARY: _GateRequirement(
        min_signals=20,
        min_orders=20,
        min_fills=20,
        min_closed=10,
        require_positive_expectancy=True,
        require_positive_lcb=True,
        require_costs=True,
        max_age_hours=24.0,
    ),
}

_REQUIRED_EVIDENCE: dict[PolicyTarget, tuple[EvidenceKind, ...]] = {
    PolicyTarget.REPLAY: (),
    PolicyTarget.PAPER: (
        EvidenceKind.VALIDATION,
        EvidenceKind.OOS,
        EvidenceKind.COST_STRESS,
        EvidenceKind.SANITY,
    ),
    PolicyTarget.MICRO_LIVE: (
        EvidenceKind.VALIDATION,
        EvidenceKind.OOS,
        EvidenceKind.COST_STRESS,
        EvidenceKind.SANITY,
        EvidenceKind.SHORT_PAPER_CANARY,
        EvidenceKind.EXTENDED_PAPER_CANARY,
    ),
}

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "target",
        "exchange",
        "actor",
        "actor_config",
        "signal_model",
        "data",
        "created_at",
        "expires_at",
        "source_revision",
        "runtime_fingerprint_sha256",
        "costs",
        "risk",
        "rules",
        "evidence",
        "manifest_sha256",
    }
)
_SIGNAL_MODEL_KEYS = frozenset(
    {
        "feature",
        "intercept_bps",
        "slope_bps_per_unit",
        "lcb_haircut_bps",
        "min_feature_value",
        "max_expected_move_bps",
    }
)
_DATA_KEYS = frozenset(
    {
        "bar_interval_seconds",
        "cadence_tolerance_seconds",
        "max_bar_close_lag_seconds",
        "max_derivatives_age_seconds",
        "required_context_coverage_pct",
    }
)
_COST_KEYS = frozenset(
    {"round_trip_fee_bps", "slippage_bps", "safety_buffer_bps"}
)
_RISK_KEYS = frozenset(
    {
        "capital_fraction",
        "max_notional_usd",
        "max_open_positions",
        "max_daily_loss_usd",
        "stop_loss_pct",
        "max_holding_minutes",
        "max_signal_age_seconds",
    }
)
_RULE_KEYS = frozenset(
    {
        "symbol",
        "regime",
        "direction",
        "min_expected_move_bps",
        "max_spread_bps",
        "max_slippage_bps",
        "min_regime_confidence",
        "risk_mult",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "kind",
        "candidate_key",
        "exchange",
        "artifact_path",
        "artifact_sha256",
        "generated_at",
        "data_as_of",
        "passed",
        "signals",
        "orders",
        "fills",
        "closed_trades",
        "expectancy_after_costs_usd",
        "expectancy_lcb_usd",
        "max_drawdown_pct",
        "mean_cost_bps",
        "direction_collapse",
        "regime_collapse",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "candidate_key",
        "exchange",
        "generated_at",
        "data_as_of",
        "passed",
        "metrics",
        "source_artifacts",
    }
)
_RECEIPT_METRIC_KEYS = frozenset(
    {
        "signals",
        "orders",
        "fills",
        "closed_trades",
        "expectancy_after_costs_usd",
        "expectancy_lcb_usd",
        "max_drawdown_pct",
        "mean_cost_bps",
        "direction_collapse",
        "regime_collapse",
    }
)
_SOURCE_ARTIFACT_KEYS = frozenset({"path", "sha256"})


def prepare_manifest_payload(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Copy a draft and fill evidence hashes explicitly marked as AUTO."""
    prepared = json.loads(json.dumps(dict(payload)))
    evidence = prepared.get("evidence")
    if isinstance(evidence, list):
        for raw_gate in evidence:
            if not isinstance(raw_gate, dict):
                continue
            digest = str(raw_gate.get("artifact_sha256") or "").strip().upper()
            if digest != "AUTO":
                continue
            path, reason = resolve_artifact_path(
                project_root,
                str(raw_gate.get("artifact_path") or ""),
            )
            if reason:
                raise ManifestError(reason)
            raw_gate["artifact_sha256"] = sha256_file(path)
    return prepared


def seal_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = json.loads(json.dumps(dict(payload)))
    sealed["manifest_sha256"] = compute_manifest_sha256(sealed)
    # Parse immediately so malformed drafts never become sealed artifacts.
    parse_manifest_payload(sealed)
    return sealed


def compute_manifest_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    raw = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_runtime_fingerprint(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for relative in RUNTIME_FINGERPRINT_PATHS:
        path = root / relative
        if not path.is_file():
            raise ManifestError(f"runtime_fingerprint.file_missing:{relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_manifest_payload(payload: Mapping[str, Any]) -> PolicyManifest:
    _require_exact_keys(payload, _MANIFEST_KEYS, "manifest")
    costs = _mapping(payload.get("costs"), "manifest.costs")
    risk = _mapping(payload.get("risk"), "manifest.risk")
    actor_config = _parse_actor_config(payload.get("actor_config"))
    signal_model = _mapping(payload.get("signal_model"), "manifest.signal_model")
    data = _mapping(payload.get("data"), "manifest.data")
    _require_exact_keys(costs, _COST_KEYS, "costs")
    _require_exact_keys(risk, _RISK_KEYS, "risk")
    _require_exact_keys(signal_model, _SIGNAL_MODEL_KEYS, "signal_model")
    _require_exact_keys(data, _DATA_KEYS, "data")

    raw_rules = _sequence(payload.get("rules"), "manifest.rules")
    rules: list[PolicyRule] = []
    for index, raw in enumerate(raw_rules):
        row = _mapping(raw, f"rules[{index}]")
        _require_exact_keys(row, _RULE_KEYS, f"rules[{index}]")
        rules.append(
            PolicyRule(
                symbol=str(row["symbol"]),
                regime=str(row["regime"]),
                direction=Direction(str(row["direction"]).upper()),
                min_expected_move_bps=float(row["min_expected_move_bps"]),
                max_spread_bps=float(row["max_spread_bps"]),
                max_slippage_bps=float(row["max_slippage_bps"]),
                min_regime_confidence=float(row["min_regime_confidence"]),
                risk_mult=float(row["risk_mult"]),
            )
        )

    raw_evidence = _sequence(payload.get("evidence"), "manifest.evidence")
    evidence: list[EvidenceGate] = []
    for index, raw in enumerate(raw_evidence):
        row = _mapping(raw, f"evidence[{index}]")
        _require_exact_keys(row, _EVIDENCE_KEYS, f"evidence[{index}]")
        evidence.append(
            EvidenceGate(
                kind=EvidenceKind(str(row["kind"])),
                candidate_key=str(row["candidate_key"]),
                exchange=str(row["exchange"]),
                artifact_path=str(row["artifact_path"]),
                artifact_sha256=str(row["artifact_sha256"]),
                generated_at=parse_datetime(row["generated_at"]),
                data_as_of=parse_datetime(row["data_as_of"]),
                passed=bool(row["passed"]),
                signals=int(row["signals"]),
                orders=int(row["orders"]),
                fills=int(row["fills"]),
                closed_trades=int(row["closed_trades"]),
                expectancy_after_costs_usd=float(row["expectancy_after_costs_usd"]),
                expectancy_lcb_usd=float(row["expectancy_lcb_usd"]),
                max_drawdown_pct=float(row["max_drawdown_pct"]),
                mean_cost_bps=float(row["mean_cost_bps"]),
                direction_collapse=bool(row["direction_collapse"]),
                regime_collapse=bool(row["regime_collapse"]),
            )
        )

    manifest = PolicyManifest(
        schema_version=str(payload["schema_version"]),
        policy_id=str(payload["policy_id"]),
        target=PolicyTarget(str(payload["target"])),
        exchange=str(payload["exchange"]),
        actor=str(payload["actor"]),
        actor_config=actor_config,
        signal_model=SignalModel(
            feature=str(signal_model["feature"]),
            intercept_bps=float(signal_model["intercept_bps"]),
            slope_bps_per_unit=float(signal_model["slope_bps_per_unit"]),
            lcb_haircut_bps=float(signal_model["lcb_haircut_bps"]),
            min_feature_value=float(signal_model["min_feature_value"]),
            max_expected_move_bps=float(signal_model["max_expected_move_bps"]),
        ),
        data=MarketDataPolicy(
            bar_interval_seconds=int(data["bar_interval_seconds"]),
            cadence_tolerance_seconds=int(data["cadence_tolerance_seconds"]),
            max_bar_close_lag_seconds=int(data["max_bar_close_lag_seconds"]),
            max_derivatives_age_seconds=int(data["max_derivatives_age_seconds"]),
            required_context_coverage_pct=float(
                data["required_context_coverage_pct"]
            ),
        ),
        created_at=parse_datetime(payload["created_at"]),
        expires_at=parse_datetime(payload["expires_at"]),
        source_revision=str(payload["source_revision"]),
        runtime_fingerprint_sha256=str(payload["runtime_fingerprint_sha256"]),
        costs=CostModel(
            round_trip_fee_bps=float(costs["round_trip_fee_bps"]),
            slippage_bps=float(costs["slippage_bps"]),
            safety_buffer_bps=float(costs["safety_buffer_bps"]),
        ),
        risk=RiskPolicy(
            capital_fraction=float(risk["capital_fraction"]),
            max_notional_usd=float(risk["max_notional_usd"]),
            max_open_positions=int(risk["max_open_positions"]),
            max_daily_loss_usd=float(risk["max_daily_loss_usd"]),
            stop_loss_pct=float(risk["stop_loss_pct"]),
            max_holding_minutes=int(risk["max_holding_minutes"]),
            max_signal_age_seconds=int(risk["max_signal_age_seconds"]),
        ),
        rules=tuple(rules),
        evidence=tuple(evidence),
        manifest_sha256=str(payload["manifest_sha256"]),
    )
    expected = compute_manifest_sha256(payload)
    if manifest.manifest_sha256 != expected:
        raise ManifestError("manifest.sha256_mismatch")
    return manifest


def validate_policy_manifest(
    manifest: PolicyManifest,
    *,
    project_root: str | Path,
    now: datetime | None = None,
    verify_artifacts: bool = True,
    verify_runtime_fingerprint: bool = False,
) -> ManifestValidation:
    current = aware_utc(now or datetime.now(timezone.utc))
    reasons: list[str] = []
    if verify_runtime_fingerprint:
        try:
            actual_runtime_fingerprint = compute_runtime_fingerprint(project_root)
        except ManifestError as exc:
            reasons.extend(exc.reasons)
        else:
            if actual_runtime_fingerprint != manifest.runtime_fingerprint_sha256:
                reasons.append("manifest.runtime_fingerprint_mismatch")
    if manifest.created_at > current + timedelta(minutes=5):
        reasons.append("manifest.created_in_future")
    if manifest.expires_at <= current:
        reasons.append("manifest.expired")
    max_lifetime_hours = {
        PolicyTarget.REPLAY: 720,
        PolicyTarget.PAPER: 168,
        PolicyTarget.MICRO_LIVE: 24,
    }[manifest.target]
    max_lifetime = timedelta(hours=max_lifetime_hours)
    if manifest.expires_at <= manifest.created_at:
        reasons.append("manifest.expiry_not_after_creation")
    elif manifest.expires_at - manifest.created_at > max_lifetime:
        reasons.append("manifest.lifetime_too_long")

    by_kind = {gate.kind: gate for gate in manifest.evidence}
    for kind in _REQUIRED_EVIDENCE[manifest.target]:
        gate = by_kind.get(kind)
        if gate is None:
            reasons.append(f"evidence.{kind.value}.missing")
            continue
        _validate_evidence_gate(
            gate,
            manifest=manifest,
            project_root=project_root,
            current=current,
            verify_artifact=verify_artifacts,
            reasons=reasons,
        )

    # No evidence from another candidate or exchange is allowed, including
    # optional gates. This prevents accidental cross-run assembly.
    for gate in manifest.evidence:
        if gate.candidate_key != manifest.policy_id:
            reasons.append(f"evidence.{gate.kind.value}.candidate_mismatch")
        if gate.exchange != manifest.exchange:
            reasons.append(f"evidence.{gate.kind.value}.exchange_mismatch")

    return ManifestValidation(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def load_policy_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    expected_sha256: str = "",
    now: datetime | None = None,
    verify_artifacts: bool = True,
    verify_runtime_fingerprint: bool = True,
    required_target: PolicyTarget | str | None = None,
    required_exchange: str = "",
) -> LoadedPolicyManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError("manifest.file_missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestError(f"manifest.file_invalid:{type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest.file_not_object")
    manifest = parse_manifest_payload(payload)
    pin = str(expected_sha256 or "").strip().lower()
    if pin:
        if not _SHA256_RE.fullmatch(pin):
            raise ManifestError("manifest.expected_sha256_invalid")
        if manifest.manifest_sha256 != pin:
            raise ManifestError("manifest.expected_sha256_mismatch")
    if required_target is not None:
        target = _enum_value(PolicyTarget, required_target, "required_target")
        if manifest.target != target:
            raise ManifestError("manifest.target_mismatch")
    if required_exchange and manifest.exchange != canonical_exchange(required_exchange):
        raise ManifestError("manifest.exchange_mismatch")
    validation = validate_policy_manifest(
        manifest,
        project_root=project_root,
        now=now,
        verify_artifacts=verify_artifacts,
        verify_runtime_fingerprint=verify_runtime_fingerprint,
    )
    if not validation.passed:
        raise ManifestError(validation.reasons)
    return LoadedPolicyManifest(
        manifest=manifest,
        path=str(manifest_path.resolve()),
        validation=validation,
    )


def _validate_evidence_gate(
    gate: EvidenceGate,
    *,
    manifest: PolicyManifest,
    project_root: str | Path,
    current: datetime,
    verify_artifact: bool,
    reasons: list[str],
) -> None:
    prefix = f"evidence.{gate.kind.value}"
    requirement = _GATE_REQUIREMENTS[gate.kind]
    if not gate.passed:
        reasons.append(f"{prefix}.failed")
    if gate.candidate_key != manifest.policy_id:
        reasons.append(f"{prefix}.candidate_mismatch")
    if gate.exchange != manifest.exchange:
        reasons.append(f"{prefix}.exchange_mismatch")
    if gate.generated_at > current + timedelta(minutes=5):
        reasons.append(f"{prefix}.generated_in_future")
    if gate.data_as_of > current + timedelta(minutes=5):
        reasons.append(f"{prefix}.data_in_future")
    age_hours = (current - gate.data_as_of).total_seconds() / 3600.0
    if age_hours > requirement.max_age_hours:
        reasons.append(f"{prefix}.stale")
    if gate.direction_collapse:
        reasons.append(f"{prefix}.direction_collapse")
    if gate.regime_collapse:
        reasons.append(f"{prefix}.regime_collapse")
    if gate.signals < requirement.min_signals:
        reasons.append(f"{prefix}.signals_below_gate")
    if gate.orders < requirement.min_orders:
        reasons.append(f"{prefix}.orders_below_gate")
    if gate.fills < requirement.min_fills:
        reasons.append(f"{prefix}.fills_below_gate")
    if gate.closed_trades < requirement.min_closed:
        reasons.append(f"{prefix}.closed_trades_below_gate")
    if (
        requirement.require_positive_expectancy
        and gate.expectancy_after_costs_usd <= 0.0
    ):
        reasons.append(f"{prefix}.nonpositive_expectancy")
    if requirement.require_positive_lcb and gate.expectancy_lcb_usd <= 0.0:
        reasons.append(f"{prefix}.nonpositive_lcb")
    if requirement.require_costs and gate.mean_cost_bps <= 0.0:
        reasons.append(f"{prefix}.zero_costs")
    if gate.max_drawdown_pct < 0.0:
        reasons.append(f"{prefix}.drawdown_negative")
    elif gate.max_drawdown_pct > requirement.max_drawdown_pct:
        reasons.append(f"{prefix}.drawdown_above_gate")
    if verify_artifact:
        artifact, path_reason = resolve_artifact_path(project_root, gate.artifact_path)
        if path_reason:
            reasons.append(f"{prefix}.{path_reason}")
        elif not artifact.is_file():
            reasons.append(f"{prefix}.artifact_missing")
        elif sha256_file(artifact) != gate.artifact_sha256:
            reasons.append(f"{prefix}.artifact_sha256_mismatch")
        else:
            _validate_evidence_receipt(
                artifact,
                gate=gate,
                project_root=project_root,
                prefix=prefix,
                reasons=reasons,
            )


def _validate_evidence_receipt(
    artifact: Path,
    *,
    gate: EvidenceGate,
    project_root: str | Path,
    prefix: str,
    reasons: list[str],
) -> None:
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception:
        reasons.append(f"{prefix}.receipt_invalid_json")
        return
    if not isinstance(payload, Mapping):
        reasons.append(f"{prefix}.receipt_not_object")
        return
    try:
        _require_exact_keys(payload, _RECEIPT_KEYS, "receipt")
        metrics = _mapping(payload.get("metrics"), "receipt.metrics")
        _require_exact_keys(metrics, _RECEIPT_METRIC_KEYS, "receipt.metrics")
        sources = _sequence(payload.get("source_artifacts"), "receipt.source_artifacts")
    except ManifestError as exc:
        reasons.extend(f"{prefix}.{reason}" for reason in exc.reasons)
        return

    expected_fields: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": gate.kind.value,
        "candidate_key": gate.candidate_key,
        "exchange": gate.exchange,
        "generated_at": format_datetime(gate.generated_at),
        "data_as_of": format_datetime(gate.data_as_of),
        "passed": gate.passed,
    }
    for field, expected in expected_fields.items():
        observed = payload.get(field)
        if field == "exchange":
            observed = str(observed or "").strip().upper()
        if observed != expected:
            reasons.append(f"{prefix}.receipt_mismatch:{field}")

    expected_metrics: dict[str, Any] = {
        "signals": gate.signals,
        "orders": gate.orders,
        "fills": gate.fills,
        "closed_trades": gate.closed_trades,
        "expectancy_after_costs_usd": gate.expectancy_after_costs_usd,
        "expectancy_lcb_usd": gate.expectancy_lcb_usd,
        "max_drawdown_pct": gate.max_drawdown_pct,
        "mean_cost_bps": gate.mean_cost_bps,
        "direction_collapse": gate.direction_collapse,
        "regime_collapse": gate.regime_collapse,
    }
    for field, expected in expected_metrics.items():
        observed = metrics.get(field)
        if isinstance(expected, bool):
            matches = isinstance(observed, bool) and observed is expected
        elif isinstance(expected, int):
            matches = isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        else:
            try:
                matches = math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        if not matches:
            reasons.append(f"{prefix}.receipt_mismatch:metrics.{field}")

    if not sources:
        reasons.append(f"{prefix}.receipt_sources_empty")
        return
    for index, raw_source in enumerate(sources):
        try:
            source = _mapping(raw_source, f"receipt.source_artifacts[{index}]")
            _require_exact_keys(
                source,
                _SOURCE_ARTIFACT_KEYS,
                f"receipt.source_artifacts[{index}]",
            )
        except ManifestError as exc:
            reasons.extend(f"{prefix}.{reason}" for reason in exc.reasons)
            continue
        source_path, path_reason = resolve_artifact_path(
            project_root,
            str(source.get("path") or ""),
        )
        if path_reason:
            reasons.append(f"{prefix}.source_{index}_{path_reason}")
            continue
        digest = str(source.get("sha256") or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            reasons.append(f"{prefix}.source_{index}_sha256_invalid")
        elif not source_path.is_file():
            reasons.append(f"{prefix}.source_{index}_missing")
        elif source_path.resolve() == artifact.resolve():
            reasons.append(f"{prefix}.source_{index}_self_reference")
        elif sha256_file(source_path) != digest:
            reasons.append(f"{prefix}.source_{index}_sha256_mismatch")


def canonical_exchange(value: str) -> str:
    clean = str(value or "").strip().upper()
    if clean != "BITGET":
        raise ManifestError("manifest.exchange_not_bitget")
    return clean


def canonical_symbol(value: str) -> str:
    clean = str(value or "").strip().upper()
    for suffix in ("/USDT:USDT", "/USDT", "USDT"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break
    if not clean or not re.fullmatch(r"[A-Z0-9]{2,12}", clean):
        raise ManifestError("rule.symbol_invalid")
    return clean


def canonical_regime(value: str) -> str:
    clean = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if clean not in _KNOWN_REGIMES:
        raise ManifestError("rule.regime_unknown")
    return clean


def parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ManifestError("manifest.datetime_invalid") from exc
    if parsed.tzinfo is None:
        raise ManifestError("manifest.datetime_timezone_missing")
    return parsed.astimezone(timezone.utc)


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ManifestError("manifest.datetime_timezone_missing")
    return value.astimezone(timezone.utc)


def format_datetime(value: datetime) -> str:
    return aware_utc(value).isoformat().replace("+00:00", "Z")


def resolve_artifact_path(
    project_root: str | Path,
    artifact_path: str,
) -> tuple[Path, str]:
    root = Path(project_root).resolve()
    raw = Path(str(artifact_path or ""))
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return path, "artifact_outside_project"
    return path, ""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    keys = frozenset(str(key) for key in payload)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    reasons = [f"{label}.field_missing:{item}" for item in missing]
    reasons.extend(f"{label}.field_unknown:{item}" for item in extra)
    if reasons:
        raise ManifestError(reasons)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{label}.not_object")
    return value


def _parse_actor_config(value: Any) -> tuple[tuple[str, bool | int | float | str], ...]:
    payload = _mapping(value, "manifest.actor_config")
    out: list[tuple[str, bool | int | float | str]] = []
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", key):
            raise ManifestError(f"manifest.actor_config_key_invalid:{key}")
        if isinstance(raw_value, bool):
            parsed: bool | int | float | str = raw_value
        elif isinstance(raw_value, int):
            parsed = raw_value
        elif isinstance(raw_value, float):
            if not math.isfinite(raw_value):
                raise ManifestError(f"manifest.actor_config_value_nonfinite:{key}")
            parsed = raw_value
        elif isinstance(raw_value, str) and raw_value.strip():
            parsed = raw_value.strip()
        else:
            raise ManifestError(f"manifest.actor_config_value_invalid:{key}")
        out.append((key, parsed))
    return tuple(sorted(out))


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"{label}.not_array")
    return value


def _enum_value(enum_type: type[Enum], value: Any, label: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise ManifestError(f"{label}_invalid") from exc


def _finite_nonnegative(label: str, value: float) -> None:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ManifestError(f"{label}_invalid")
