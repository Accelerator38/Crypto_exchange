"""Sealed, gap-tolerant campaign contract for CarryFlow market evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .evidence_tape import CarryFlowEvidenceTape


CAMPAIGN_SCHEMA_VERSION = "panteon.carryflow_evidence_campaign.v1"
ROOT_SCHEMA_VERSION = "panteon.carryflow_evidence_campaign_root.v1"
ROOT_VERDICT_SCHEMA_VERSION = "panteon.carryflow_evidence_root_verdict.v1"
CAMPAIGN_STATUS_SCHEMA_VERSION = "panteon.carryflow_evidence_campaign_status.v1"
BITGET_FULL8 = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK")
EXACT_REPORT_SCHEMA_VERSION = "panteon.carryflow_exact_profile_evaluation.v1"
EVIDENCE_EXTENSION_FAILURES = ("nonpositive_lcb",)
CAMPAIGN_ALIGNMENT_DELAY_SECONDS = 30
CAMPAIGN_WARMUP_BARS = 53

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,40}$")
_CAMPAIGN_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "created_at",
        "source_revision",
        "runtime_fingerprint_sha256",
        "source_exchange",
        "market_type",
        "strategy_config",
        "symbol_set",
        "bar_interval_seconds",
        "max_bar_close_lag_seconds",
        "max_derivatives_age_seconds",
        "alignment_delay_seconds",
        "warmup_bars",
        "screening_report_file",
        "screening_report_sha256",
        "screening_report_generated_at",
        "baseline_evidence",
        "collector_mode",
        "gap_handling",
        "segments_concatenated",
        "allowed_use",
        "orders_enabled",
        "promotion_authority",
        "campaign_sha256",
    }
)
_BASELINE_KEYS = frozenset(
    {
        "filled_orders",
        "closed_trades",
        "mean_net_pnl_usd",
        "expectancy_lcb_95_usd",
        "max_drawdown_usd",
        "max_drawdown_limit_usd",
        "active_roots",
        "root_expectancy_collapses",
        "failures",
    }
)
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_sha256",
        "root_index",
        "root_id",
        "created_at",
        "intended_first_bar_close_timestamp_ms",
        "start_reason",
        "previous_root_id",
        "previous_root_head_sha256",
        "evidence_tape_file",
        "warmup_seed_file",
        "collector_status_file",
        "orders_enabled",
        "promotion_authority",
        "root_sha256",
    }
)
_ROOT_VERDICT_KEYS = frozenset(
    {
        "schema_version",
        "recorded_at",
        "campaign_id",
        "campaign_sha256",
        "root_id",
        "root_sha256",
        "evidence_tape_sha256",
        "evidence_head_sha256",
        "samples",
        "complete_samples",
        "context_coverage_pct",
        "replay_summary_sha256",
        "replay_manifest_sha256",
        "verdict",
        "reason",
        "evidence_failures",
        "metrics",
        "terminal",
        "continuation_allowed",
        "orders_enabled",
        "promotion_authority",
        "verdict_sha256",
    }
)
_ROOT_VERDICT_METRIC_KEYS = frozenset(
    {
        "filled_orders",
        "closed_trades",
        "gross_pnl_usd",
        "total_costs_usd",
        "net_pnl_usd",
        "expectancy_after_costs_usd",
        "expectancy_lcb_usd",
        "max_drawdown_usd",
        "remaining_open_positions",
    }
)


class EvidenceCampaignError(ValueError):
    """The evidence campaign is ambiguous, mutable, or unsafe to run."""


@dataclass(frozen=True)
class CampaignCollectionDecision:
    action: str
    reason: str
    current_bar_close_timestamp_ms: int
    next_collection_at: str


@dataclass(frozen=True)
class CarryFlowEvidenceCampaign:
    path: Path
    payload: Mapping[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "CarryFlowEvidenceCampaign":
        source = Path(path)
        if not source.is_file():
            raise EvidenceCampaignError(f"campaign lock not found: {source}")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceCampaignError(f"invalid campaign lock JSON: {exc}") from exc
        return cls(path=source.resolve(), payload=validate_campaign_payload(raw))

    @property
    def campaign_sha256(self) -> str:
        return str(self.payload["campaign_sha256"])

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.payload["symbol_set"])

    @property
    def profile_id(self) -> str:
        return str(self.payload["strategy_config"]["PROFILE_ID"])


def build_campaign_payload(
    *,
    report: Mapping[str, Any],
    report_file: str,
    report_sha256: str,
    source_revision: str,
    runtime_fingerprint_sha256: str,
    created_at: datetime,
    tape_loader: Callable[[str | Path], CarryFlowEvidenceTape] | None = None,
) -> dict[str, Any]:
    """Build a campaign only from an exact report eligible for LCB extension."""

    candidate = validate_evidence_extension_report(report)
    load_tape = tape_loader or CarryFlowEvidenceTape.from_jsonl
    segment_rows = candidate.get("segment_results")
    if not isinstance(segment_rows, list) or len(segment_rows) < 2:
        raise EvidenceCampaignError("extension candidate needs at least two roots")
    tapes = []
    for index, row in enumerate(segment_rows, start=1):
        if not isinstance(row, Mapping) or not str(row.get("tape") or "").strip():
            raise EvidenceCampaignError(f"candidate segment {index} has no tape")
        tapes.append(load_tape(str(row["tape"])))
    symbols = {tuple(tape.symbols) for tape in tapes}
    intervals = {int(tape.bar_interval_seconds) for tape in tapes}
    lag_limits = {int(tape.max_bar_close_lag_seconds) for tape in tapes}
    age_limits = {int(tape.max_derivatives_age_seconds) for tape in tapes}
    if symbols != {BITGET_FULL8}:
        raise EvidenceCampaignError("extension campaign requires exact Bitget full8")
    if intervals != {3600}:
        raise EvidenceCampaignError("extension campaign requires hourly evidence")
    if len(lag_limits) != 1 or len(age_limits) != 1:
        raise EvidenceCampaignError("candidate roots use different freshness limits")

    revision = str(source_revision).strip().lower()
    runtime_sha = str(runtime_fingerprint_sha256).strip().lower()
    report_sha = str(report_sha256).strip().lower()
    if not _REVISION_RE.fullmatch(revision):
        raise EvidenceCampaignError("source revision is invalid")
    if not _SHA256_RE.fullmatch(runtime_sha):
        raise EvidenceCampaignError("runtime fingerprint is invalid")
    if not _SHA256_RE.fullmatch(report_sha):
        raise EvidenceCampaignError("screening report SHA-256 is invalid")
    created = _as_utc(created_at)
    profile_id = str(candidate["profile_id"])
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": (
            f"carryflow-extension-{profile_id}-{revision[:8]}-"
            f"{created:%Y%m%dT%H%M%SZ}"
        ),
        "created_at": created.isoformat(),
        "source_revision": revision,
        "runtime_fingerprint_sha256": runtime_sha,
        "source_exchange": "BITGET",
        "market_type": "swap",
        "strategy_config": {"PROFILE_ID": profile_id},
        "symbol_set": list(BITGET_FULL8),
        "bar_interval_seconds": 3600,
        "max_bar_close_lag_seconds": next(iter(lag_limits)),
        "max_derivatives_age_seconds": next(iter(age_limits)),
        "alignment_delay_seconds": CAMPAIGN_ALIGNMENT_DELAY_SECONDS,
        "warmup_bars": CAMPAIGN_WARMUP_BARS,
        "screening_report_file": str(report_file),
        "screening_report_sha256": report_sha,
        "screening_report_generated_at": str(report["generated_at"]),
        "baseline_evidence": {
            key: _json_copy(candidate[key])
            for key in _BASELINE_KEYS
        },
        "collector_mode": "hourly_one_shot",
        "gap_handling": "start_new_independent_root",
        "segments_concatenated": False,
        "allowed_use": "no_order_evidence_extension",
        "orders_enabled": False,
        "promotion_authority": False,
    }
    payload["campaign_sha256"] = compute_campaign_sha256(payload)
    return validate_campaign_payload(payload)


def validate_evidence_extension_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    data = _json_copy(report)
    if data.get("schema_version") != EXACT_REPORT_SCHEMA_VERSION:
        raise EvidenceCampaignError("unsupported exact evaluation report")
    for key in (
        "research_only",
        "profile_selection_only",
        "root_robustness_required",
        "fresh_prospective_validation_required",
    ):
        if data.get(key) is not True:
            raise EvidenceCampaignError(f"exact report missing invariant: {key}")
    for key in (
        "orders_enabled",
        "promotion_authority",
        "segments_concatenated",
        "evidence_extension_orders_enabled",
        "evidence_extension_is_promotion",
    ):
        if data.get(key) is not False:
            raise EvidenceCampaignError(f"exact report unsafe flag: {key}")
    if data.get("selected_for_prospective_validation") is not None:
        raise EvidenceCampaignError(
            "passing prospective candidate must not enter evidence extension"
        )
    candidate = data.get("selected_for_evidence_extension")
    if not isinstance(candidate, Mapping):
        raise EvidenceCampaignError("exact report has no extension candidate")
    row = _json_copy(candidate)
    if row.get("evidence_extension_eligible") is not True:
        raise EvidenceCampaignError("candidate is not extension eligible")
    if row.get("screening_passed") is not False:
        raise EvidenceCampaignError("passing candidate does not need extension")
    if tuple(row.get("failures") or ()) != EVIDENCE_EXTENSION_FAILURES:
        raise EvidenceCampaignError("extension is allowed only for nonpositive LCB")
    if int(row.get("filled_orders") or 0) < 20:
        raise EvidenceCampaignError("candidate has fewer than 20 fills")
    if int(row.get("closed_trades") or 0) < 10:
        raise EvidenceCampaignError("candidate has fewer than 10 closed trades")
    if _finite(row.get("mean_net_pnl_usd"), "mean net PnL") <= 0.0:
        raise EvidenceCampaignError("candidate expectancy is not positive")
    lcb = _finite(row.get("expectancy_lcb_95_usd"), "expectancy LCB")
    if lcb > 0.0:
        raise EvidenceCampaignError("positive-LCB candidate must use promotion path")
    drawdown = _finite(row.get("max_drawdown_usd"), "drawdown")
    drawdown_limit = _finite(row.get("max_drawdown_limit_usd"), "drawdown limit")
    if drawdown_limit <= 0.0 or drawdown > drawdown_limit:
        raise EvidenceCampaignError("candidate drawdown is above its fixed limit")
    if int(row.get("active_roots") or 0) < 2:
        raise EvidenceCampaignError("candidate has fewer than two active roots")
    if row.get("root_expectancy_collapses") != []:
        raise EvidenceCampaignError("candidate has a root expectancy collapse")
    if not str(row.get("profile_id") or "").strip():
        raise EvidenceCampaignError("candidate profile is missing")
    _parse_datetime(data.get("generated_at"), "report generated_at")
    return row


def compute_campaign_sha256(payload: Mapping[str, Any]) -> str:
    canonical = _json_copy(payload)
    canonical.pop("campaign_sha256", None)
    return _sha256_json(canonical)


def validate_campaign_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _json_copy(payload)
    _exact_keys(row, _CAMPAIGN_KEYS, "campaign")
    if row["schema_version"] != CAMPAIGN_SCHEMA_VERSION:
        raise EvidenceCampaignError("unsupported campaign schema")
    if not str(row["campaign_id"]).strip():
        raise EvidenceCampaignError("campaign_id is required")
    _parse_datetime(row["created_at"], "campaign created_at")
    if not _REVISION_RE.fullmatch(str(row["source_revision"]).lower()):
        raise EvidenceCampaignError("campaign source revision is invalid")
    for key in ("runtime_fingerprint_sha256", "screening_report_sha256"):
        value = str(row[key]).strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise EvidenceCampaignError(f"campaign {key} is invalid")
        row[key] = value
    if row["source_exchange"] != "BITGET" or row["market_type"] != "swap":
        raise EvidenceCampaignError("campaign market contract is invalid")
    strategy = row["strategy_config"]
    if not isinstance(strategy, Mapping) or set(strategy) != {"PROFILE_ID"}:
        raise EvidenceCampaignError("campaign strategy has independent knobs")
    if not str(strategy["PROFILE_ID"]).strip():
        raise EvidenceCampaignError("campaign PROFILE_ID is required")
    if tuple(row["symbol_set"]) != BITGET_FULL8:
        raise EvidenceCampaignError("campaign symbol set is not Bitget full8")
    if int(row["bar_interval_seconds"]) != 3600:
        raise EvidenceCampaignError("campaign cadence is not hourly")
    if int(row["max_bar_close_lag_seconds"]) < 1:
        raise EvidenceCampaignError("campaign bar-close lag is invalid")
    if int(row["max_derivatives_age_seconds"]) < 1:
        raise EvidenceCampaignError("campaign derivatives age is invalid")
    if int(row["alignment_delay_seconds"]) != CAMPAIGN_ALIGNMENT_DELAY_SECONDS:
        raise EvidenceCampaignError("campaign alignment delay changed")
    if int(row["warmup_bars"]) != CAMPAIGN_WARMUP_BARS:
        raise EvidenceCampaignError("campaign warm-up length changed")
    if Path(str(row["screening_report_file"])).is_absolute():
        raise EvidenceCampaignError("campaign report path must be portable")
    _parse_datetime(row["screening_report_generated_at"], "report generated_at")
    baseline = row["baseline_evidence"]
    if not isinstance(baseline, Mapping):
        raise EvidenceCampaignError("campaign baseline evidence is invalid")
    _exact_keys(baseline, _BASELINE_KEYS, "baseline evidence")
    if tuple(baseline["failures"]) != EVIDENCE_EXTENSION_FAILURES:
        raise EvidenceCampaignError("campaign baseline failure changed")
    if baseline["root_expectancy_collapses"] != []:
        raise EvidenceCampaignError("campaign baseline has a root collapse")
    if row["collector_mode"] != "hourly_one_shot":
        raise EvidenceCampaignError("campaign collector mode changed")
    if row["gap_handling"] != "start_new_independent_root":
        raise EvidenceCampaignError("campaign gap policy changed")
    if row["segments_concatenated"] is not False:
        raise EvidenceCampaignError("campaign roots cannot be concatenated")
    if row["allowed_use"] != "no_order_evidence_extension":
        raise EvidenceCampaignError("campaign allowed use changed")
    if (
        row["orders_enabled"] is not False
        or row["promotion_authority"] is not False
    ):
        raise EvidenceCampaignError("campaign cannot enable orders or promotion")
    expected_sha = compute_campaign_sha256(row)
    actual_sha = str(row["campaign_sha256"]).strip().lower()
    if actual_sha != expected_sha:
        raise EvidenceCampaignError("campaign SHA-256 mismatch")
    row["campaign_sha256"] = actual_sha
    return row


def decide_campaign_collection(
    *,
    now: datetime,
    bar_interval_seconds: int,
    alignment_delay_seconds: int,
    max_bar_close_lag_seconds: int,
    latest_bar_close_timestamp_ms: int | None,
) -> CampaignCollectionDecision:
    current = _as_utc(now)
    interval_ms = int(bar_interval_seconds) * 1000
    timestamp_ms = int(current.timestamp() * 1000)
    close_ms = timestamp_ms - timestamp_ms % interval_ms
    lag_seconds = (timestamp_ms - close_ms) / 1000.0
    next_close_ms = close_ms + interval_ms
    next_collection = datetime.fromtimestamp(
        next_close_ms / 1000.0 + int(alignment_delay_seconds),
        timezone.utc,
    ).isoformat()
    if lag_seconds < int(alignment_delay_seconds):
        return CampaignCollectionDecision(
            action="wait",
            reason="bar_close_settlement_delay",
            current_bar_close_timestamp_ms=close_ms,
            next_collection_at=(
                datetime.fromtimestamp(
                    close_ms / 1000.0 + int(alignment_delay_seconds),
                    timezone.utc,
                ).isoformat()
            ),
        )
    if lag_seconds > int(max_bar_close_lag_seconds):
        return CampaignCollectionDecision(
            action="wait",
            reason="current_collection_window_missed",
            current_bar_close_timestamp_ms=close_ms,
            next_collection_at=next_collection,
        )
    if latest_bar_close_timestamp_ms is None:
        return CampaignCollectionDecision(
            action="new_root",
            reason="campaign_first_root",
            current_bar_close_timestamp_ms=close_ms,
            next_collection_at=next_collection,
        )
    latest = int(latest_bar_close_timestamp_ms)
    if latest == close_ms:
        return CampaignCollectionDecision(
            action="idle",
            reason="current_bar_already_collected",
            current_bar_close_timestamp_ms=close_ms,
            next_collection_at=next_collection,
        )
    if latest > close_ms:
        raise EvidenceCampaignError("system clock precedes latest evidence bar")
    if latest + interval_ms == close_ms:
        return CampaignCollectionDecision(
            action="resume_root",
            reason="exact_next_bar",
            current_bar_close_timestamp_ms=close_ms,
            next_collection_at=next_collection,
        )
    return CampaignCollectionDecision(
        action="new_root",
        reason="real_bar_gap",
        current_bar_close_timestamp_ms=close_ms,
        next_collection_at=next_collection,
    )


def build_root_payload(
    *,
    campaign: CarryFlowEvidenceCampaign,
    root_index: int,
    created_at: datetime,
    intended_first_bar_close_timestamp_ms: int,
    start_reason: str,
    previous_root_id: str = "",
    previous_root_head_sha256: str = "",
) -> dict[str, Any]:
    if root_index < 1:
        raise EvidenceCampaignError("root index must be positive")
    created = _as_utc(created_at)
    root_id = (
        f"{campaign.payload['campaign_id']}-root-{root_index:03d}-"
        f"{created:%Y%m%dT%H%M%SZ}"
    )
    payload = {
        "schema_version": ROOT_SCHEMA_VERSION,
        "campaign_id": campaign.payload["campaign_id"],
        "campaign_sha256": campaign.campaign_sha256,
        "root_index": root_index,
        "root_id": root_id,
        "created_at": created.isoformat(),
        "intended_first_bar_close_timestamp_ms": int(
            intended_first_bar_close_timestamp_ms
        ),
        "start_reason": str(start_reason),
        "previous_root_id": str(previous_root_id),
        "previous_root_head_sha256": str(previous_root_head_sha256),
        "evidence_tape_file": "carryflow_evidence_tape.jsonl",
        "warmup_seed_file": "carryflow_warmup_seed.json",
        "collector_status_file": "collector_status.json",
        "orders_enabled": False,
        "promotion_authority": False,
    }
    payload["root_sha256"] = compute_root_sha256(payload)
    return validate_root_payload(payload, campaign=campaign)


def compute_root_sha256(payload: Mapping[str, Any]) -> str:
    canonical = _json_copy(payload)
    canonical.pop("root_sha256", None)
    return _sha256_json(canonical)


def validate_root_payload(
    payload: Mapping[str, Any],
    *,
    campaign: CarryFlowEvidenceCampaign | None = None,
) -> dict[str, Any]:
    row = _json_copy(payload)
    _exact_keys(row, _ROOT_KEYS, "campaign root")
    if row["schema_version"] != ROOT_SCHEMA_VERSION:
        raise EvidenceCampaignError("unsupported campaign root schema")
    if int(row["root_index"]) < 1 or not str(row["root_id"]).strip():
        raise EvidenceCampaignError("campaign root identity is invalid")
    _parse_datetime(row["created_at"], "root created_at")
    close_ms = int(row["intended_first_bar_close_timestamp_ms"])
    if close_ms <= 0 or close_ms % 3_600_000:
        raise EvidenceCampaignError("campaign root first close is misaligned")
    if row["start_reason"] not in {"campaign_first_root", "real_bar_gap"}:
        raise EvidenceCampaignError("campaign root start reason is invalid")
    for key in (
        "evidence_tape_file",
        "warmup_seed_file",
        "collector_status_file",
    ):
        path = Path(str(row[key]))
        if path.is_absolute() or path.name != str(row[key]):
            raise EvidenceCampaignError(f"campaign root {key} is not local")
    previous_sha = str(row["previous_root_head_sha256"])
    if previous_sha and not _SHA256_RE.fullmatch(previous_sha):
        raise EvidenceCampaignError("previous root head SHA-256 is invalid")
    if (
        row["orders_enabled"] is not False
        or row["promotion_authority"] is not False
    ):
        raise EvidenceCampaignError("campaign root cannot enable orders")
    expected_sha = compute_root_sha256(row)
    if str(row["root_sha256"]).lower() != expected_sha:
        raise EvidenceCampaignError("campaign root SHA-256 mismatch")
    if campaign is not None:
        if row["campaign_id"] != campaign.payload["campaign_id"]:
            raise EvidenceCampaignError("campaign root ID mismatch")
        if row["campaign_sha256"] != campaign.campaign_sha256:
            raise EvidenceCampaignError("campaign root lock mismatch")
    return row


def build_negative_root_verdict(
    *,
    campaign: CarryFlowEvidenceCampaign,
    root_payload: Mapping[str, Any],
    tape_description: Mapping[str, Any],
    evidence_tape_sha256: str,
    replay_summary: Mapping[str, Any],
    replay_summary_sha256: str,
    recorded_at: datetime,
) -> dict[str, Any]:
    """Seal a terminal rejection when prospective costed evidence is negative."""

    root = validate_root_payload(root_payload, campaign=campaign)
    tape_sha = str(evidence_tape_sha256).strip().lower()
    replay_sha = str(replay_summary_sha256).strip().lower()
    if not _SHA256_RE.fullmatch(tape_sha):
        raise EvidenceCampaignError("root verdict tape SHA-256 is invalid")
    if not _SHA256_RE.fullmatch(replay_sha):
        raise EvidenceCampaignError("root verdict replay SHA-256 is invalid")

    summary = _json_copy(replay_summary)
    evidence = summary.get("evidence_tape")
    if not isinstance(evidence, Mapping):
        raise EvidenceCampaignError("root verdict replay has no evidence tape")
    if evidence.get("collector_run_id") != root["root_id"]:
        raise EvidenceCampaignError("root verdict replay/root ID mismatch")
    if str(evidence.get("file_sha256") or "").lower() != tape_sha:
        raise EvidenceCampaignError("root verdict replay/tape SHA-256 mismatch")
    if summary.get("evidence_eligible") is not False:
        raise EvidenceCampaignError("eligible replay cannot be terminally rejected")
    if summary.get("research_only") is not True:
        raise EvidenceCampaignError("root verdict replay is not research-only")
    if summary.get("promotion_authority") is not False:
        raise EvidenceCampaignError("root verdict replay has promotion authority")

    failures = tuple(str(item) for item in summary.get("evidence_failures") or ())
    required_failures = {"nonpositive_expectancy", "nonpositive_lcb"}
    if not required_failures.issubset(failures):
        raise EvidenceCampaignError(
            "negative root verdict requires expectancy and LCB failures"
        )
    metrics = {
        key: _json_copy(summary.get(key))
        for key in _ROOT_VERDICT_METRIC_KEYS
    }
    if _finite(metrics["expectancy_after_costs_usd"], "root expectancy") > 0.0:
        raise EvidenceCampaignError("negative root verdict has positive expectancy")
    if _finite(metrics["expectancy_lcb_usd"], "root expectancy LCB") > 0.0:
        raise EvidenceCampaignError("negative root verdict has positive LCB")
    if int(metrics["remaining_open_positions"] or 0) != 0:
        raise EvidenceCampaignError("negative root verdict has open positions")

    samples = int(tape_description.get("samples") or 0)
    complete_samples = int(tape_description.get("complete_samples") or 0)
    head_sha = str(tape_description.get("head_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(head_sha):
        raise EvidenceCampaignError("root verdict evidence head is invalid")
    if int(evidence.get("samples") or 0) != samples:
        raise EvidenceCampaignError("root verdict replay/tape sample mismatch")
    if int(evidence.get("complete_samples") or 0) != complete_samples:
        raise EvidenceCampaignError(
            "root verdict replay/tape complete-sample mismatch"
        )
    coverage = _finite(
        tape_description.get("context_coverage_pct"),
        "root context coverage",
    )
    replay_coverage = _finite(
        evidence.get("context_coverage_pct"),
        "replay context coverage",
    )
    if not math.isclose(coverage, replay_coverage, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceCampaignError("root verdict replay/tape coverage mismatch")
    manifest_sha = str(summary.get("manifest_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(manifest_sha):
        raise EvidenceCampaignError("root verdict manifest SHA-256 is invalid")

    payload = {
        "schema_version": ROOT_VERDICT_SCHEMA_VERSION,
        "recorded_at": _as_utc(recorded_at).isoformat(),
        "campaign_id": campaign.payload["campaign_id"],
        "campaign_sha256": campaign.campaign_sha256,
        "root_id": root["root_id"],
        "root_sha256": root["root_sha256"],
        "evidence_tape_sha256": tape_sha,
        "evidence_head_sha256": head_sha,
        "samples": samples,
        "complete_samples": complete_samples,
        "context_coverage_pct": coverage,
        "replay_summary_sha256": replay_sha,
        "replay_manifest_sha256": manifest_sha,
        "verdict": "rejected",
        "reason": "negative_costed_expectancy_and_lcb",
        "evidence_failures": list(failures),
        "metrics": metrics,
        "terminal": True,
        "continuation_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    payload["verdict_sha256"] = compute_root_verdict_sha256(payload)
    return validate_root_verdict_payload(
        payload,
        campaign=campaign,
        root_payload=root,
    )


def compute_root_verdict_sha256(payload: Mapping[str, Any]) -> str:
    canonical = _json_copy(payload)
    canonical.pop("verdict_sha256", None)
    return _sha256_json(canonical)


def validate_root_verdict_payload(
    payload: Mapping[str, Any],
    *,
    campaign: CarryFlowEvidenceCampaign | None = None,
    root_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _json_copy(payload)
    _exact_keys(row, _ROOT_VERDICT_KEYS, "root verdict")
    if row["schema_version"] != ROOT_VERDICT_SCHEMA_VERSION:
        raise EvidenceCampaignError("unsupported root verdict schema")
    _parse_datetime(row["recorded_at"], "root verdict recorded_at")
    for key in (
        "campaign_sha256",
        "root_sha256",
        "evidence_tape_sha256",
        "evidence_head_sha256",
        "replay_summary_sha256",
        "replay_manifest_sha256",
        "verdict_sha256",
    ):
        value = str(row[key]).strip().lower()
        if not _SHA256_RE.fullmatch(value):
            raise EvidenceCampaignError(f"root verdict {key} is invalid")
        row[key] = value
    if int(row["samples"]) < 1:
        raise EvidenceCampaignError("root verdict has no samples")
    complete_samples = int(row["complete_samples"])
    if complete_samples < 0 or complete_samples > int(row["samples"]):
        raise EvidenceCampaignError("root verdict complete samples are invalid")
    coverage = _finite(row["context_coverage_pct"], "root verdict coverage")
    if not 0.0 <= coverage <= 100.0:
        raise EvidenceCampaignError("root verdict coverage is invalid")
    if row["verdict"] != "rejected":
        raise EvidenceCampaignError("root verdict must reject the experiment")
    if row["reason"] != "negative_costed_expectancy_and_lcb":
        raise EvidenceCampaignError("root verdict reason is invalid")
    failures = tuple(str(item) for item in row["evidence_failures"])
    if not {"nonpositive_expectancy", "nonpositive_lcb"}.issubset(failures):
        raise EvidenceCampaignError("root verdict failures are incomplete")
    metrics = row["metrics"]
    if not isinstance(metrics, Mapping):
        raise EvidenceCampaignError("root verdict metrics are invalid")
    _exact_keys(metrics, _ROOT_VERDICT_METRIC_KEYS, "root verdict metrics")
    for key in _ROOT_VERDICT_METRIC_KEYS:
        _finite(metrics[key], f"root verdict metric {key}")
    if float(metrics["expectancy_after_costs_usd"]) > 0.0:
        raise EvidenceCampaignError("root verdict expectancy is positive")
    if float(metrics["expectancy_lcb_usd"]) > 0.0:
        raise EvidenceCampaignError("root verdict LCB is positive")
    if int(metrics["remaining_open_positions"]) != 0:
        raise EvidenceCampaignError("root verdict has open positions")
    if (
        row["terminal"] is not True
        or row["continuation_allowed"] is not False
        or row["orders_enabled"] is not False
        or row["promotion_authority"] is not False
    ):
        raise EvidenceCampaignError("root verdict safety flags are invalid")
    if str(row["verdict_sha256"]).lower() != compute_root_verdict_sha256(row):
        raise EvidenceCampaignError("root verdict SHA-256 mismatch")
    if campaign is not None:
        if row["campaign_id"] != campaign.payload["campaign_id"]:
            raise EvidenceCampaignError("root verdict campaign ID mismatch")
        if row["campaign_sha256"] != campaign.campaign_sha256:
            raise EvidenceCampaignError("root verdict campaign SHA-256 mismatch")
    if root_payload is not None:
        if row["root_id"] != root_payload["root_id"]:
            raise EvidenceCampaignError("root verdict root ID mismatch")
        if row["root_sha256"] != root_payload["root_sha256"]:
            raise EvidenceCampaignError("root verdict root SHA-256 mismatch")
    return row


def write_atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceCampaignError("campaign payload is not finite JSON") from exc


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise EvidenceCampaignError(
            f"{label} keys mismatch; missing={missing}, extra={extra}"
        )


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceCampaignError(f"{label} is invalid") from exc
    if not math.isfinite(parsed):
        raise EvidenceCampaignError(f"{label} is not finite")
    return parsed


def _parse_datetime(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceCampaignError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise EvidenceCampaignError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise EvidenceCampaignError("campaign time must be timezone-aware")
    return value.astimezone(timezone.utc)
