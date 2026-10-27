"""Additive, metadata-only adapters. No model, collector or runtime imports.

An envelope records what an artifact reports, not independently verified truth.
Unsupported semantics remain unknown; normalization grants no trading authority.
"""

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
import json
import math
import re
from typing import Any


SCHEMA = "exia.genetic_primus.result_catalog_envelope/1"
AUDITED_BATCH_SHA = "4ec6c77c8d533dd4db970da5ffa7769706ccf202225e0cb4ed7e7506154daee7"
HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactKind(StrEnum):
    FINANCIAL_RESULT = "financial_result"
    MODEL = "model_metadata"
    SCALER = "scaler"
    MANIFEST = "manifest"
    REGISTRATION = "registration"
    LEDGER = "research_ledger"
    CHECKPOINT = "checkpoint_metadata"
    CHECK = "engineering_check"
    TRACE = "search_trace"
    BANK = "resample_bank_metadata"
    GATE = "research_gate_decision"
    ORIGIN = "origin_step_result"
    DATA_SUMMARY = "data_summary"
    UNKNOWN = "unknown"


class MetricName(StrEnum):
    NET = "total_net_return_reported"
    GROSS = "total_gross_return_reported"
    MARKED_NET = "marked_net_return"
    TERMINAL_NET = "net_after_hypothetical_terminal_liquidation"
    MEAN_TRADE = "mean_net_return_per_trade"
    LCB_TRADE = "net_return_lcb_per_trade"
    MEAN_DAY = "mean_net_return_per_calendar_day"
    LCB_DAY = "net_return_lcb_per_calendar_day"
    DD = "max_drawdown_reported_sampling_unknown"
    DD_CLOSE = "max_drawdown_bar_close"
    TRADES = "trades_reported"
    FILLS = "fills_reported"
    TRANSITIONS = "position_transition_count"
    FEES = "fees_fraction_initial_capital"


class Unit(StrEnum):
    FRACTION = "fraction"
    PERCENT = "percent"
    BPS = "bps"
    COUNT = "count"


@dataclass(frozen=True)
class Source:
    path: str
    byte_sha256: str
    size_bytes: int
    reported_schema: str | None


@dataclass(frozen=True)
class Metric:
    name: MetricName
    value: float | int | None
    normalized_unit: str
    source_field: str
    source_unit: Unit
    unknown_reason: str | None = None


@dataclass(frozen=True)
class Context:
    interval_start_ms: int | None = None
    interval_end_exclusive_ms: int | None = None
    deployment_horizon_ms: int | None = None
    retrain_cadence_ms: int | None = None
    round_trip_cost_bps: float | None = None
    delay_bars: int | None = None
    accounting: str | None = None
    fingerprints: tuple[tuple[str, str], ...] = ()
    # This is descriptive comparability within ONE pinned, audited source only.
    shared_batch_sha256: str | None = None


@dataclass(frozen=True)
class Evaluation:
    policy: str
    scenario: str
    adapter: str
    context: Context
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class Envelope:
    schema: str
    source: Source
    artifact_kind: ArtifactKind
    evidence_category: str
    reported_evidence: str | None
    reported_hash_claims: tuple[tuple[str, str], ...]
    evaluations: tuple[Evaluation, ...]
    issues: tuple[str, ...]
    execution_allowed: bool = False
    credentials_allowed: bool = False
    orders_enabled: bool = False
    paper: bool = False
    live: bool = False
    promotion_authority: bool = False
    envelope_sha256: str = ""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_json_metadata(raw: bytes) -> Any:
    """Reject ambiguous duplicate fields rather than silently choosing the last."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique_object)


def envelope_dict(envelope: Envelope) -> dict:
    return asdict(envelope)


def _number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def normalize_metric(name: MetricName, raw: Any, unit: Unit, field: str) -> Metric:
    """Normalize explicit units only. Invalid/missing is NOT observed zero."""
    name, unit = MetricName(name), Unit(unit)
    count_names = (MetricName.TRADES, MetricName.FILLS, MetricName.TRANSITIONS)
    if (name in count_names) != (unit == Unit.COUNT):
        raise ValueError("metric/unit semantics mismatch")
    reason = None
    value = None
    if raw is None:
        reason = "missing_or_null"
    elif not _number(raw):
        reason = "not_a_finite_number"
    elif unit == Unit.COUNT:
        if raw < 0 or int(raw) != raw:
            reason = "not_a_nonnegative_integer"
        else:
            value = int(raw)
    else:
        value = float(raw) / {Unit.FRACTION: 1, Unit.PERCENT: 100, Unit.BPS: 10000}[unit]
        if name in (MetricName.DD, MetricName.DD_CLOSE) and not 0 <= value <= 1:
            value, reason = None, "drawdown_out_of_range"
    return Metric(name, value, "count" if unit == Unit.COUNT else "fraction",
                  field, unit, reason)


def metric_value(evaluation: Evaluation, name: MetricName) -> float | int | None:
    matches = [m.value for m in evaluation.metrics if m.name == name]
    return matches[0] if len(matches) == 1 else None


def _metrics(data: dict, mapping: tuple, prefix: str) -> tuple[Metric, ...]:
    # Include every expected field, including explicit unknowns.
    return tuple(normalize_metric(name, data.get(key), unit, f"{prefix}.{key}")
                 for key, name, unit in mapping)


FRACTION_RESULT = (
    ("net_return", MetricName.NET, Unit.FRACTION),
    ("max_drawdown", MetricName.DD, Unit.FRACTION),
    ("trades", MetricName.TRADES, Unit.COUNT),
)
PERCENT_RESULT = (
    ("net_return_pct", MetricName.NET, Unit.PERCENT),
    ("gross_return_pct", MetricName.GROSS, Unit.PERCENT),
    ("max_drawdown_pct", MetricName.DD, Unit.PERCENT),
    ("trades", MetricName.TRADES, Unit.COUNT),
    ("fills", MetricName.FILLS, Unit.COUNT),
)
REPLAY_RESULT = (
    ("marked_net_return", MetricName.MARKED_NET, Unit.FRACTION),
    ("net_after_terminal_liquidation", MetricName.TERMINAL_NET, Unit.FRACTION),
    ("max_drawdown_close", MetricName.DD_CLOSE, Unit.FRACTION),
    ("transition_count", MetricName.TRANSITIONS, Unit.COUNT),
    ("cumulative_fees_initial_nav", MetricName.FEES, Unit.FRACTION),
)


def _hash_claims(data: dict) -> tuple[tuple[str, str], ...]:
    """Retain direct claims with their original names; never relabel identities."""
    claims = [(k, v) for k, v in data.items() if k.endswith("sha256")
              and isinstance(v, str) and HEX_SHA.fullmatch(v)]
    for group in ("sources_sha256", "code_sha256", "source_hashes"):
        children = data.get(group)
        if isinstance(children, dict):
            claims.extend((f"{group}.{k}", v) for k, v in children.items()
                          if isinstance(v, str) and HEX_SHA.fullmatch(v))
    return tuple(sorted(claims))


def _int_or_unknown(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _context(data: dict, accounting: str | None = None, *, legacy_ms: bool = False) -> Context:
    # Bare timestamp aliases are accepted only for a schema whose unit is known.
    start = _int_or_unknown(data.get("interval_start" if legacy_ms else "interval_start_ms"))
    end = _int_or_unknown(data.get("interval_end_exclusive" if legacy_ms else "interval_end_exclusive_ms"))
    if start is not None and end is not None and end <= start:
        start = end = None
    hashes = tuple((k, v) for k, v in _hash_claims(data) if k in (
        "market_rows_sha256", "examples_sha256", "source_snapshot_sha256",
        "source_manifest_sha256", "source_contract_sha256", "train_bank_sha256"))
    return Context(start, end, accounting=accounting, fingerprints=hashes)


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _kind(data: dict, basename: str, schema: str) -> ArtifactKind:
    # Structural hints identify artifact type, never validation quality.
    token = schema.lower()
    if "scaler" in basename or "scaler" in token:
        return ArtifactKind.SCALER
    if "model_manifest" in basename or "model_manifest" in token:
        return ArtifactKind.MODEL
    if "ledger" in token or basename == "ledger.json":
        return ArtifactKind.LEDGER
    if "promotion_gate" in basename or "promotion_gate" in token:
        return ArtifactKind.GATE
    if basename in ("label_summary.json", "cell_summary.json"):
        return ArtifactKind.DATA_SUMMARY
    nested_schema = _dict(data.get("result")).get("schema")
    if nested_schema == "exia.genetic_primus.reversal_rolling_step/1":
        return ArtifactKind.ORIGIN
    if "registration" in token or "registration" in basename or "pre_run_receipt" in basename:
        return ArtifactKind.REGISTRATION
    if "checkpoint" in token or basename.endswith("_started.json"):
        return ArtifactKind.CHECKPOINT
    if ("synthetic" in token or "check_result" in token or ".check/" in token
            or "synthetic" in basename or basename.startswith("check_")):
        return ArtifactKind.CHECK
    if "generation_" in basename or "search_trace" in token or "hypothesis_budget" in basename:
        return ArtifactKind.TRACE
    if "bank" in basename and ("bootstrap" in basename or "resample" in basename):
        return ArtifactKind.BANK
    if "manifest" in basename or "manifest" in token:
        return ArtifactKind.MANIFEST
    if ("result" in token or "metrics" in basename or basename in ("result.json", "summary.json")
            or "mean_bps" in data or "mean_daily_bps" in data):
        return ArtifactKind.FINANCIAL_RESULT
    return ArtifactKind.UNKNOWN


def _evidence(data: dict, kind: ArtifactKind) -> tuple[str, str | None]:
    raw = data.get("evidence_class")
    raw = raw if isinstance(raw, str) else None
    if kind == ArtifactKind.CHECK:
        return "engineering_only", raw
    if raw and any(t in raw.lower() for t in ("revealed", "development", "diagnostic", "calibration")):
        return "reported_development_or_revealed", raw
    if data.get("data_already_revealed") is True:
        return "reported_development_or_revealed", raw
    if data.get("independent_validation") is False:
        return "reported_not_independent", raw
    if raw and "prospective" in raw.lower():
        return "claimed_prospective_unverified", raw
    if data.get("independent_validation") is True or (raw and "independent" in raw.lower()):
        return "claimed_independent_unverified", raw
    return "unknown", raw


def _evaluations(data: dict, source: Source) -> tuple[Evaluation, ...]:
    schema = source.reported_schema or ""
    output = []
    if schema == "exia.genetic_primus.baseline_economics_audit.result/1":
        context = _context(data, legacy_ms=True)
        for policy, payload in sorted(_dict(data.get("policies")).items()):
            if not isinstance(payload, dict):
                continue
            for scenario, entry in sorted(_dict(payload.get("evaluations")).items()):
                if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
                    continue
                metrics = entry["metrics"]
                cost = metrics.get("stress_cost_round_trip_bps")
                cost = float(cost) if _number(cost) and cost >= 0 else None
                batch = source.byte_sha256 if source.byte_sha256 == AUDITED_BATCH_SHA else None
                ctx = replace(context, round_trip_cost_bps=cost, shared_batch_sha256=batch)
                output.append(Evaluation(policy, scenario, "baseline_economics_pct/1", ctx,
                                         _metrics(metrics, PERCENT_RESULT,
                                                  f"policies.{policy}.evaluations.{scenario}.metrics")))
    elif schema == "exia.genetic_primus.rolling_linear_matrix_result/1":
        records = data.get("records", [])
        if isinstance(records, list):
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                policy = f"{record.get('feature_set', 'unknown')}:{record.get('variant', 'unknown')}:{index}"
                for scenario, metrics in sorted(_dict(record.get("evaluations")).items()):
                    if isinstance(metrics, dict):
                        output.append(Evaluation(policy, scenario, "rolling_linear_fraction/1", _context(data),
                                                 _metrics(metrics, FRACTION_RESULT,
                                                          f"records[{index}].evaluations.{scenario}")))
        for scenario, metrics in sorted(_dict(data.get("ema_training_reference")).items()):
            if isinstance(metrics, dict):
                output.append(Evaluation("EMA_training_reference", scenario, "rolling_linear_fraction/1",
                                         _context(data), _metrics(metrics, FRACTION_RESULT,
                                                                 f"ema_training_reference.{scenario}")))
    elif schema == "exia.genetic_primus.reversal_calibration_result/1":
        for scenario, metrics in sorted(_dict(data.get("accounting")).items()):
            if isinstance(metrics, dict):
                output.append(Evaluation("frozen_reversal_pilot", scenario, "continuous_nav_fraction/1",
                                         _context(data, "continuous_nav_marked_and_hypothetical_terminal"),
                                         _metrics(metrics, REPLAY_RESULT, f"accounting.{scenario}")))
    elif schema == "exia.genetic_primus.revealed_3d_summary/1":
        mapping = (("compounded_return", MetricName.NET, Unit.FRACTION),
                   ("max_drawdown", MetricName.DD, Unit.FRACTION),
                   ("trades", MetricName.TRADES, Unit.COUNT))
        for scenario, cost_payload in sorted(_dict(data.get("costs")).items()):
            for policy, metrics in sorted(_dict(_dict(cost_payload).get("candidates")).items()):
                if isinstance(metrics, dict):
                    output.append(Evaluation(policy, scenario, "closed_origin_geometric_chain/1",
                                             _context(data, "geometric_chain_of_closed_origin_windows"),
                                             _metrics(metrics, mapping, f"costs.{scenario}.candidates.{policy}")))
    elif _dict(data.get("result")).get("schema") == "exia.genetic_primus.reversal_rolling_step/1":
        step = data["result"]
        start = _int_or_unknown(step.get("origin_ms"))
        end = _int_or_unknown(_dict(step.get("state")).get("next_origin_ms"))
        if start is None or end is None or end <= start:
            start = end = None
        for scenario, metrics in sorted(_dict(step.get("scenarios")).items()):
            if not isinstance(metrics, dict):
                continue
            cost = metrics.get("round_trip_cost_bps")
            cost = float(cost) if _number(cost) and cost >= 0 else None
            context = Context(start, end, round_trip_cost_bps=cost,
                              delay_bars=_int_or_unknown(metrics.get("delay_bars")),
                              accounting="continuous_nav_per_origin",
                              fingerprints=_hash_claims(step))
            mapping = (("origin_net_return", MetricName.NET, Unit.FRACTION),)
            output.append(Evaluation("frozen_reversal_origin", scenario, "continuous_nav_origin/1", context,
                                     _metrics(metrics, mapping, f"result.scenarios.{scenario}")))
    elif "mean_bps" in data and "lcb_bps" in data:
        mapping = (("mean_bps", MetricName.MEAN_TRADE, Unit.BPS),
                   ("lcb_bps", MetricName.LCB_TRADE, Unit.BPS),
                   ("max_drawdown", MetricName.DD, Unit.FRACTION),
                   ("trades", MetricName.TRADES, Unit.COUNT))
        output.append(Evaluation("reported_policy", "unknown", "legacy_trade_bps/1", _context(data),
                                 _metrics(data, mapping, "$")))
    elif "mean_daily_bps" in data and "lcb_daily_bps" in data:
        mapping = (("mean_daily_bps", MetricName.MEAN_DAY, Unit.BPS),
                   ("lcb_daily_bps", MetricName.LCB_DAY, Unit.BPS),
                   ("max_drawdown", MetricName.DD, Unit.FRACTION),
                   ("trades", MetricName.TRADES, Unit.COUNT),
                   ("turnover_fills", MetricName.FILLS, Unit.COUNT))
        output.append(Evaluation("reported_policy", "unknown", "legacy_daily_bps/1", _context(data),
                                 _metrics(data, mapping, "$")))
    return tuple(output)


def adapt(data: Any, source: Source) -> Envelope:
    if not HEX_SHA.fullmatch(source.byte_sha256) or type(source.size_bytes) is not int or source.size_bytes < 0:
        raise ValueError("invalid source identity")
    if not isinstance(data, dict):
        data = {}
        issues = ["non_object_payload_unsupported"]
    else:
        issues = []
    if (isinstance(data.get("schema"), str) and isinstance(data.get("schema_version"), str)
            and data["schema"] != data["schema_version"]):
        raise ValueError("conflicting reported schemas")
    kind = _kind(data, source.path.rsplit("/", 1)[-1], source.reported_schema or "")
    evaluations = _evaluations(data, source) if kind in (ArtifactKind.FINANCIAL_RESULT, ArtifactKind.ORIGIN) else ()
    if kind == ArtifactKind.FINANCIAL_RESULT and not evaluations:
        issues.append("unsupported_financial_shape_no_metrics_inferred")
    if any(m.value is None for e in evaluations for m in e.metrics):
        issues.append("some_metrics_unknown")
    if any(e.context.interval_start_ms is None or e.context.interval_end_exclusive_ms is None for e in evaluations):
        issues.append("evaluation_period_unknown")
    evidence_data = _dict(data.get("result")) if kind == ArtifactKind.ORIGIN else data
    category, reported = _evidence(evidence_data, kind)
    claims = list(_hash_claims(data))
    if kind == ArtifactKind.ORIGIN:
        claims.extend(("result." + k, v) for k, v in _hash_claims(evidence_data))
        model = _dict(evidence_data.get("model"))
        claims.extend(("result.model." + k, v) for k, v in _hash_claims(model))
    envelope = Envelope(SCHEMA, source, kind, category, reported, tuple(sorted(claims)), evaluations, tuple(issues))
    body = envelope_dict(envelope)
    body.pop("envelope_sha256")
    return replace(envelope, envelope_sha256=sha256(canonical_bytes(body)).hexdigest())


def comparison_reasons(left: Envelope, a: Evaluation, right: Envelope, b: Evaluation,
                       metric: MetricName) -> tuple[str, ...]:
    """Descriptive arithmetic only. Never a ranking/promotion/statistical gate."""
    metric = MetricName(metric)
    reasons = []
    try:
        validate_serialized(envelope_dict(left))
        validate_serialized(envelope_dict(right))
    except (ValueError, TypeError, KeyError):
        reasons.append("envelope_integrity_invalid")
    if a not in left.evaluations or b not in right.evaluations:
        reasons.append("evaluation_not_owned_by_envelope")
    if left.artifact_kind != ArtifactKind.FINANCIAL_RESULT or right.artifact_kind != ArtifactKind.FINANCIAL_RESULT:
        reasons.append("not_financial_results")
    if metric_value(a, metric) is None or metric_value(b, metric) is None:
        reasons.append("metric_missing_or_incompatible_semantics")
    if a.scenario != b.scenario or a.context.round_trip_cost_bps != b.context.round_trip_cost_bps:
        reasons.append("cost_scenario_mismatch")
    if left.evidence_category != right.evidence_category:
        reasons.append("evidence_category_mismatch")
    pinned = (left.source.byte_sha256 == right.source.byte_sha256 == AUDITED_BATCH_SHA
              and a.context.shared_batch_sha256 == b.context.shared_batch_sha256 == AUDITED_BATCH_SHA
              and a.context == b.context and a.adapter == b.adapter == "baseline_economics_pct/1")
    if not pinned:
        # No generic compatibility inferred from field names or equal incomplete contexts.
        reasons.append("common_measurement_contract_not_verified")
    return tuple(reasons)


def validate_serialized(value: dict) -> None:
    """Validate metadata envelope and digest without reconstructing any runtime object."""
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported envelope schema")
    ArtifactKind(value["artifact_kind"])
    for flag in ("execution_allowed", "credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority"):
        if value.get(flag) is not False:
            raise ValueError("catalog safety flag must be false")
    source = value["source"]
    if not HEX_SHA.fullmatch(source["byte_sha256"]) or type(source["size_bytes"]) is not int or source["size_bytes"] < 0:
        raise ValueError("invalid source identity")
    for evaluation in value["evaluations"]:
        seen = set()
        for metric in evaluation["metrics"]:
            name = MetricName(metric["name"])
            source_unit = Unit(metric["source_unit"])
            if name in seen:
                raise ValueError("duplicate metric")
            seen.add(name)
            raw = metric["value"]
            if raw is not None and not _number(raw):
                raise ValueError("invalid normalized metric")
            if (raw is None) != (metric["unknown_reason"] is not None):
                raise ValueError("unknown/value inconsistency")
            if metric["normalized_unit"] not in ("count", "fraction"):
                raise ValueError("unknown normalized unit")
            expected_unit = "count" if name in (MetricName.TRADES, MetricName.FILLS, MetricName.TRANSITIONS) else "fraction"
            if metric["normalized_unit"] != expected_unit:
                raise ValueError("metric/unit semantics mismatch")
            if (expected_unit == "count") != (source_unit == Unit.COUNT):
                raise ValueError("source unit/metric semantics mismatch")
            if raw is not None and expected_unit == "count" and (type(raw) is not int or raw < 0):
                raise ValueError("invalid normalized count")
            if raw is not None and name in (MetricName.DD, MetricName.DD_CLOSE) and not 0 <= raw <= 1:
                raise ValueError("invalid normalized drawdown")
    body = dict(value)
    digest = body.pop("envelope_sha256")
    if sha256(canonical_bytes(body)).hexdigest() != digest:
        raise ValueError("envelope hash mismatch")
