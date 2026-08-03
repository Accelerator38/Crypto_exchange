"""Immutable multi-session campaign for Bitget microstructure precursors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .research_dataset import validate_research_dataset
from .research_dataset_v2 import (
    validate_research_dataset_v2,
    validate_research_dataset_v3,
)


CAMPAIGN_SCHEMA_VERSION = "panteon.bitget_precursor_campaign.v1"
LOCK_SCHEMA_VERSION = "panteon.bitget_precursor_campaign_lock.v1"
CAMPAIGN_SCHEMA_VERSION_V2 = "panteon.bitget_precursor_campaign.v2"
LOCK_SCHEMA_VERSION_V2 = "panteon.bitget_precursor_campaign_lock.v2"
CAMPAIGN_SCHEMA_VERSION_V3 = "panteon.bitget_precursor_campaign.v3"
LOCK_SCHEMA_VERSION_V3 = "panteon.bitget_precursor_campaign_lock.v3"
HORIZON_MINUTES = 15
COST_STRESS_BPS = 4.0
MIN_VALIDATION_SESSIONS = 2
MIN_OOS_SESSIONS = 3
MIN_SESSION_DURATION_HOURS = 12.0
MIN_SESSION_COVERAGE_PCT = 95.0
LCB_Z = 1.6448536269514722

CANDIDATE_RULES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "oi_flow_book_quorum_v1",
        "description": "OI expansion confirmed by flow, book and microprice",
        "mode": "oi_expansion",
        "oi_min_bps": 10.0,
        "flow_min": 0.25,
        "book_min": 0.10,
        "microprice_min_bps": 0.05,
        "spread_max_bps": 2.5,
    },
    {
        "candidate_id": "oi_absorption_reversal_v1",
        "description": "OI expansion and order flow against the recent move",
        "mode": "absorption_reversal",
        "oi_min_bps": 15.0,
        "return_min_bps": 10.0,
        "flow_min": 0.25,
        "book_min": 0.08,
        "microprice_min_bps": 0.05,
        "spread_max_bps": 2.5,
    },
    {
        "candidate_id": "flow_book_liquidity_pressure_v1",
        "description": "Aggressive flow aligned with book, depth and microprice",
        "mode": "liquidity_pressure",
        "flow_min": 0.40,
        "book_min": 0.18,
        "depth_imbalance_min": 0.20,
        "microprice_min_bps": 0.08,
        "spread_max_bps": 2.0,
    },
    {
        "candidate_id": "oi_neutral_build_v1",
        "description": "OI builds before price leaves a neutral five-minute range",
        "mode": "neutral_oi_build",
        "oi_min_bps": 20.0,
        "return_max_bps": 5.0,
        "flow_min": 0.18,
        "book_min": 0.08,
        "microprice_min_bps": 0.03,
        "spread_max_bps": 2.5,
    },
    {
        "candidate_id": "oi_unwind_continuation_v1",
        "description": "OI contraction with aligned price, flow and book pressure",
        "mode": "oi_unwind",
        "oi_max_bps": -15.0,
        "return_min_bps": 10.0,
        "flow_min": 0.25,
        "book_min": 0.08,
        "microprice_min_bps": 0.05,
        "spread_max_bps": 2.5,
    },
)

CANDIDATE_RULES_V3: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "oi_change_return_momentum_h120_v1",
        "description": (
            "Follow a large five-minute move when absolute OI change confirms "
            "participation; hold for 120 minutes"
        ),
        "mode": "oi_change_return_momentum",
        "oi_abs_min_bps": 10.0,
        "return_abs_min_bps": 10.0,
        "spread_max_bps": 2.0,
    },
)


def run_precursor_campaign(
    *,
    research_root: str | Path,
    campaign_dir: str | Path,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    return _run_precursor_campaign(
        research_root=research_root,
        campaign_dir=campaign_dir,
        frozen_at=frozen_at,
        campaign_schema_version=CAMPAIGN_SCHEMA_VERSION,
        lock_schema_version=LOCK_SCHEMA_VERSION,
        research_manifest_name="microstructure_research_v1.manifest.json",
        frame_manifest_name="frame_1s_v1.manifest.json",
        research_validator=validate_research_dataset,
        data_contract_id=None,
        candidate_rules=CANDIDATE_RULES,
        horizon_minutes=HORIZON_MINUTES,
    )


def run_precursor_campaign_v2(
    *,
    research_root: str | Path,
    campaign_dir: str | Path,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    return _run_precursor_campaign(
        research_root=research_root,
        campaign_dir=campaign_dir,
        frozen_at=frozen_at,
        campaign_schema_version=CAMPAIGN_SCHEMA_VERSION_V2,
        lock_schema_version=LOCK_SCHEMA_VERSION_V2,
        research_manifest_name="microstructure_research_v2.manifest.json",
        frame_manifest_name="frame_1s_v2.manifest.json",
        research_validator=validate_research_dataset_v2,
        data_contract_id="event_snapshot_continuity_bounded_v2",
        candidate_rules=CANDIDATE_RULES,
        horizon_minutes=HORIZON_MINUTES,
    )


def run_precursor_campaign_v3(
    *,
    research_root: str | Path,
    campaign_dir: str | Path,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    return _run_precursor_campaign(
        research_root=research_root,
        campaign_dir=campaign_dir,
        frozen_at=frozen_at,
        campaign_schema_version=CAMPAIGN_SCHEMA_VERSION_V3,
        lock_schema_version=LOCK_SCHEMA_VERSION_V3,
        research_manifest_name="microstructure_research_v3.manifest.json",
        frame_manifest_name="frame_1s_v2.manifest.json",
        research_validator=validate_research_dataset_v3,
        data_contract_id="event_snapshot_continuity_bounded_v2_h120",
        candidate_rules=CANDIDATE_RULES_V3,
        horizon_minutes=120,
    )


def _run_precursor_campaign(
    *,
    research_root: str | Path,
    campaign_dir: str | Path,
    frozen_at: datetime | None,
    campaign_schema_version: str,
    lock_schema_version: str,
    research_manifest_name: str,
    frame_manifest_name: str,
    research_validator: Callable[[str | Path], Mapping[str, Any]],
    data_contract_id: str | None,
    candidate_rules: Sequence[Mapping[str, Any]],
    horizon_minutes: int,
) -> dict[str, Any]:
    contract_payload = _contract_payload(
        data_contract_id=data_contract_id,
        candidate_rules=candidate_rules,
        horizon_minutes=horizon_minutes,
    )
    if data_contract_id is None:
        sessions = _discover_sessions(research_root)
    else:
        sessions = _discover_sessions(
            research_root,
            research_manifest_name=research_manifest_name,
            frame_manifest_name=frame_manifest_name,
            research_validator=research_validator,
            horizon_minutes=horizon_minutes,
        )
    if not sessions:
        raise ValueError("no validated Bitget research datasets found")
    campaign = Path(campaign_dir).resolve()
    lock, lock_created = _ensure_lock(
        campaign,
        sessions=sessions,
        frozen_at=frozen_at,
        lock_schema_version=lock_schema_version,
        contract_payload=contract_payload,
    )
    classified, classification_failures = _classify_sessions(sessions, lock)
    role_counts = Counter(row["role"] for row in classified)
    campaign_failures = list(classification_failures)
    if role_counts["validation"] < MIN_VALIDATION_SESSIONS:
        campaign_failures.append("validation_sessions_below_2")
    if role_counts["oos"] < MIN_OOS_SESSIONS:
        campaign_failures.append("oos_sessions_below_3")
    bad_quality = [
        row["source_session_id"]
        for row in classified
        if row["role"] != "development"
        and (
            float(row["duration_hours"]) < MIN_SESSION_DURATION_HOURS
            or float(row["eligible_bar_coverage_pct"]) < MIN_SESSION_COVERAGE_PCT
            or float(row["frame_coverage_pct"]) < MIN_SESSION_COVERAGE_PCT
        )
    ]
    if bad_quality:
        campaign_failures.append("prospective_session_quality_failed")

    candidate_reports = []
    for rule in candidate_rules:
        session_reports = []
        selected_by_role: dict[str, list[dict[str, Any]]] = {
            "development": [],
            "validation": [],
            "oos": [],
        }
        for session in classified:
            trades = _select_session_trades(
                session["rows"],
                rule,
                horizon_minutes=horizon_minutes,
            )
            if session["role"] in selected_by_role:
                selected_by_role[session["role"]].extend(trades)
            session_reports.append(
                {
                    "source_session_id": session["source_session_id"],
                    "role": session["role"],
                    "duration_hours": session["duration_hours"],
                    "eligible_bar_coverage_pct": session[
                        "eligible_bar_coverage_pct"
                    ],
                    "frame_coverage_pct": session["frame_coverage_pct"],
                    "metrics": _metrics(trades),
                }
            )
        role_metrics = {
            role: _metrics(rows) for role, rows in selected_by_role.items()
        }
        candidate_failures = _candidate_failures(
            role_metrics=role_metrics,
            session_reports=session_reports,
            enough_sessions=(
                role_counts["validation"] >= MIN_VALIDATION_SESSIONS
                and role_counts["oos"] >= MIN_OOS_SESSIONS
                and not bad_quality
                and not classification_failures
            ),
        )
        early_stop_reason = None
        if data_contract_id == "event_snapshot_continuity_bounded_v2_h120":
            early_stop_reason = _prospective_early_stop_reason(session_reports)
            if early_stop_reason is not None:
                candidate_failures = [early_stop_reason]
        candidate_reports.append(
            {
                "candidate_id": rule["candidate_id"],
                "description": rule["description"],
                "role_metrics": role_metrics,
                "sessions": session_reports,
                "early_stop_reason": early_stop_reason,
                "failures": candidate_failures,
                "eligible_for_strict_paper_canary": not candidate_failures,
            }
        )
    candidate_reports.sort(
        key=lambda row: (
            row["role_metrics"]["oos"]["lcb_95_stress_net_bps"] is not None,
            row["role_metrics"]["oos"]["lcb_95_stress_net_bps"] or -math.inf,
            row["role_metrics"]["development"]["mean_stress_net_bps"]
            or -math.inf,
        ),
        reverse=True,
    )
    passed = [
        row["candidate_id"]
        for row in candidate_reports
        if row["eligible_for_strict_paper_canary"]
    ]
    enough_evidence = (
        role_counts["validation"] >= MIN_VALIDATION_SESSIONS
        and role_counts["oos"] >= MIN_OOS_SESSIONS
        and not bad_quality
        and not classification_failures
    )
    if passed:
        verdict = "eligible_for_strict_paper_canary"
    elif any(row["early_stop_reason"] for row in candidate_reports):
        verdict = "rejected_early"
    elif enough_evidence:
        verdict = "rejected"
    else:
        verdict = "collecting_prospective_evidence"
    report = {
        "schema_version": campaign_schema_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_lock": {
            "path": str(campaign / "campaign_lock.json"),
            "lock_sha256": lock["lock_sha256"],
            "contract_sha256": lock["contract_sha256"],
            "frozen_at": lock["frozen_at"],
            "created_this_run": lock_created,
        },
        "fixed_contract": contract_payload,
        "evidence": {
            "research_root": str(Path(research_root).resolve()),
            "independent_sessions": len(classified),
            "role_counts": dict(sorted(role_counts.items())),
            "minimum_validation_sessions": MIN_VALIDATION_SESSIONS,
            "minimum_oos_sessions": MIN_OOS_SESSIONS,
            "sessions": [_public_session_row(row) for row in classified],
        },
        "candidates": candidate_reports,
        "passed_candidates": passed,
        "campaign_failures": campaign_failures,
        "verdict": verdict,
        "research_only": True,
        "runtime_profile_created": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    campaign.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(campaign / "latest_report.json", report)
    (campaign / "latest_report.md").write_text(
        render_precursor_campaign_markdown(report),
        encoding="utf-8",
    )
    return report


def render_precursor_campaign_markdown(report: Mapping[str, Any]) -> str:
    evidence = report["evidence"]
    lines = [
        "# Bitget precursor evidence campaign v1",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        (
            f"Independent sessions: {evidence['independent_sessions']}; roles: "
            f"{evidence['role_counts']}. Required: 2 validation and 3 OOS."
        ),
        "",
        "| Candidate | Dev trades/net | Validation trades/net/LCB | "
        "OOS trades/net/LCB | Pass |",
        "|---|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        development = candidate["role_metrics"]["development"]
        validation = candidate["role_metrics"]["validation"]
        oos = candidate["role_metrics"]["oos"]
        lines.append(
            f"| {candidate['candidate_id']} | "
            f"{development['closed_trades']} / "
            f"{_fmt(development['mean_stress_net_bps'])} | "
            f"{validation['closed_trades']} / "
            f"{_fmt(validation['mean_stress_net_bps'])} / "
            f"{_fmt(validation['lcb_95_stress_net_bps'])} | "
            f"{oos['closed_trades']} / {_fmt(oos['mean_stress_net_bps'])} / "
            f"{_fmt(oos['lcb_95_stress_net_bps'])} | "
            f"{str(candidate['eligible_for_strict_paper_canary']).lower()} |"
        )
    lines.extend(
        [
            "",
            "Campaign failures: "
            + (", ".join(report["campaign_failures"]) or "none"),
            "",
            (
                "Existing sessions frozen into development never become validation "
                "or OOS. Prospective sessions are evaluated independently; gaps "
                "between them are not stitched."
            ),
            "",
            "No runtime policy, paper order or live order is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _discover_sessions(
    research_root: str | Path,
    *,
    research_manifest_name: str = "microstructure_research_v1.manifest.json",
    frame_manifest_name: str = "frame_1s_v1.manifest.json",
    research_validator: Callable[[str | Path], Mapping[str, Any]] = (
        validate_research_dataset
    ),
    horizon_minutes: int = HORIZON_MINUTES,
) -> list[dict[str, Any]]:
    root = Path(research_root).resolve()
    sessions = []
    seen: set[str] = set()
    for manifest_path in sorted(root.glob(f"*/{research_manifest_name}")):
        dataset_dir = manifest_path.parent
        validation = research_validator(dataset_dir)
        manifest = _read_json(manifest_path)
        source_session_id = str(manifest["source_session_id"])
        if source_session_id in seen:
            raise ValueError(f"duplicate source session: {source_session_id}")
        seen.add(source_session_id)
        database = dataset_dir / str(manifest["database_file"])
        rows, facts = _load_session_rows(
            database,
            horizon_minutes=horizon_minutes,
        )
        frame_manifest = _read_json(
            Path(str(manifest["source_frame_dataset_dir"]))
            / frame_manifest_name
        )
        sessions.append(
            {
                "dataset_dir": str(dataset_dir),
                "source_session_id": source_session_id,
                "manifest_sha256": validation["manifest_sha256"],
                "database_sha256": validation["database_sha256"],
                "start_timestamp_ms": facts["start_timestamp_ms"],
                "end_timestamp_ms": facts["end_timestamp_ms"],
                "duration_hours": facts["duration_hours"],
                "eligible_bar_coverage_pct": (
                    int(manifest["eligible_bars"]) / int(manifest["bars"]) * 100.0
                ),
                "frame_coverage_pct": float(
                    frame_manifest["complete_frame_coverage_pct"]
                ),
                "rows": rows,
            }
        )
    return sorted(sessions, key=lambda row: int(row["start_timestamp_ms"]))


def _load_session_rows(
    database: Path,
    *,
    horizon_minutes: int = HORIZON_MINUTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = """
        SELECT b.bar_timestamp_ms, b.symbol, b.continuity_id,
               b.spread_mean_bps, b.microprice_edge_mean_bps,
               b.book_imbalance_mean, b.bid_depth_close_usd,
               b.ask_depth_close_usd, b.flow_imbalance,
               b.return_5m_bps, b.oi_change_5m_bps,
               l.direction, l.gross_mid_bps, l.net_execution_bps,
               l.total_cost_bps
        FROM bars b
        JOIN labels l
          ON l.entry_bar_timestamp_ms = b.bar_timestamp_ms
         AND l.symbol = b.symbol
        WHERE b.eligible = 1
          AND l.eligible = 1
          AND l.horizon_minutes = ?
          AND b.spread_mean_bps IS NOT NULL
          AND b.microprice_edge_mean_bps IS NOT NULL
          AND b.book_imbalance_mean IS NOT NULL
          AND b.bid_depth_close_usd IS NOT NULL
          AND b.ask_depth_close_usd IS NOT NULL
          AND b.return_5m_bps IS NOT NULL
          AND b.oi_change_5m_bps IS NOT NULL
        ORDER BY b.bar_timestamp_ms, b.symbol, l.direction
    """
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(query, (horizon_minutes,))]
        bounds = connection.execute(
            "SELECT MIN(bar_timestamp_ms), MAX(bar_timestamp_ms) FROM bars"
        ).fetchone()
    if not rows or bounds is None or bounds[0] is None or bounds[1] is None:
        raise ValueError(f"eligible precursor rows are empty: {database}")
    start = int(bounds[0])
    end = int(bounds[1])
    return rows, {
        "start_timestamp_ms": start,
        "end_timestamp_ms": end,
        "duration_hours": (end - start + 60_000) / 3_600_000.0,
    }


def _ensure_lock(
    campaign_dir: Path,
    *,
    sessions: Sequence[Mapping[str, Any]],
    frozen_at: datetime | None,
    lock_schema_version: str = LOCK_SCHEMA_VERSION,
    contract_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    lock_path = campaign_dir / "campaign_lock.json"
    fixed_contract = contract_payload or _contract_payload()
    contract_sha = _sha256_json(fixed_contract)
    if lock_path.is_file():
        lock = _read_json(lock_path)
        claimed = str(lock.pop("lock_sha256", ""))
        if _sha256_json(lock) != claimed:
            raise ValueError("precursor campaign lock hash mismatch")
        lock["lock_sha256"] = claimed
        if lock.get("schema_version") != lock_schema_version:
            raise ValueError("precursor campaign lock schema mismatch")
        if lock.get("contract_sha256") != contract_sha:
            raise ValueError("precursor candidate contract changed after freeze")
        return lock, False

    now = frozen_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("campaign frozen_at must be timezone-aware")
    frozen_iso = now.astimezone(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "schema_version": lock_schema_version,
        "frozen_at": frozen_iso,
        "frozen_at_timestamp_ms": int(now.timestamp() * 1000),
        "contract_sha256": contract_sha,
        "development_sources": [
            {
                "source_session_id": row["source_session_id"],
                "manifest_sha256": row["manifest_sha256"],
                "end_timestamp_ms": row["end_timestamp_ms"],
            }
            for row in sessions
            if int(row["end_timestamp_ms"]) <= int(now.timestamp() * 1000)
        ],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    payload["lock_sha256"] = _sha256_json(payload)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(lock_path, payload)
    return payload, True


def _classify_sessions(
    sessions: Sequence[dict[str, Any]],
    lock: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    frozen_sources = {
        (str(row["source_session_id"]), str(row["manifest_sha256"]))
        for row in lock["development_sources"]
    }
    frozen_at_ms = int(lock["frozen_at_timestamp_ms"])
    development = []
    prospective = []
    failures = []
    for source in sessions:
        row = dict(source)
        identity = (row["source_session_id"], row["manifest_sha256"])
        if identity in frozen_sources:
            row["role"] = "development"
            development.append(row)
        elif int(row["start_timestamp_ms"]) > frozen_at_ms:
            prospective.append(row)
        else:
            row["role"] = "unclassified"
            development.append(row)
            failures.append(
                f"pre_freeze_session_not_in_lock:{row['source_session_id']}"
            )
    prospective.sort(key=lambda row: int(row["start_timestamp_ms"]))
    accepted = []
    quarantined = []
    for row in prospective:
        if (
            float(row["duration_hours"]) >= MIN_SESSION_DURATION_HOURS
            and float(row["eligible_bar_coverage_pct"])
            >= MIN_SESSION_COVERAGE_PCT
            and float(row["frame_coverage_pct"]) >= MIN_SESSION_COVERAGE_PCT
        ):
            accepted.append(row)
        else:
            row["role"] = "quarantined"
            quarantined.append(row)
    for index, row in enumerate(accepted):
        row["role"] = "validation" if index < MIN_VALIDATION_SESSIONS else "oos"
    return sorted(
        development + accepted + quarantined,
        key=lambda row: int(row["start_timestamp_ms"]),
    ), failures


def _select_session_trades(
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    *,
    horizon_minutes: int = HORIZON_MINUTES,
) -> list[dict[str, Any]]:
    by_timestamp: dict[int, list[dict[str, Any]]] = {}
    for source in rows:
        score = _rule_score(source, rule)
        if score <= 0.0:
            continue
        row = dict(source)
        row["score"] = score
        by_timestamp.setdefault(int(row["bar_timestamp_ms"]), []).append(row)
    selected = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        selected.append(best)
        next_allowed = timestamp + horizon_minutes * 60_000
    return selected


def _rule_score(row: Mapping[str, Any], rule: Mapping[str, Any]) -> float:
    flow = float(row["flow_imbalance"])
    book = float(row["book_imbalance_mean"])
    microprice = float(row["microprice_edge_mean_bps"])
    oi_change = float(row["oi_change_5m_bps"])
    recent_return = float(row["return_5m_bps"])
    spread = float(row["spread_mean_bps"])
    bid_depth = float(row["bid_depth_close_usd"])
    ask_depth = float(row["ask_depth_close_usd"])
    depth_imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-9)
    direction_sign = 1.0 if str(row["direction"]) == "LONG" else -1.0
    if spread > float(rule["spread_max_bps"]):
        return 0.0
    mode = str(rule["mode"])
    if mode == "oi_expansion":
        if oi_change < float(rule["oi_min_bps"]):
            return 0.0
        context_sign = _aligned_sign(
            flow,
            book,
            microprice,
            minimums=(rule["flow_min"], rule["book_min"], rule["microprice_min_bps"]),
        )
        magnitude = oi_change + abs(flow) * 20.0 + abs(book) * 10.0
    elif mode == "absorption_reversal":
        if oi_change < float(rule["oi_min_bps"]) or abs(recent_return) < float(rule["return_min_bps"]):
            return 0.0
        context_sign = _aligned_sign(
            flow,
            book,
            microprice,
            minimums=(rule["flow_min"], rule["book_min"], rule["microprice_min_bps"]),
        )
        if context_sign * recent_return >= 0.0:
            return 0.0
        magnitude = oi_change + abs(recent_return) + abs(flow) * 20.0
    elif mode == "liquidity_pressure":
        context_sign = _aligned_sign(
            flow,
            book,
            microprice,
            minimums=(rule["flow_min"], rule["book_min"], rule["microprice_min_bps"]),
        )
        if context_sign == 0.0 or context_sign * depth_imbalance < float(rule["depth_imbalance_min"]):
            return 0.0
        magnitude = abs(flow) * 30.0 + abs(book) * 20.0 + abs(depth_imbalance) * 10.0
    elif mode == "neutral_oi_build":
        if oi_change < float(rule["oi_min_bps"]) or abs(recent_return) > float(rule["return_max_bps"]):
            return 0.0
        context_sign = _aligned_sign(
            flow,
            book,
            microprice,
            minimums=(rule["flow_min"], rule["book_min"], rule["microprice_min_bps"]),
        )
        magnitude = oi_change + abs(flow) * 20.0 + abs(book) * 10.0
    elif mode == "oi_unwind":
        if oi_change > float(rule["oi_max_bps"]) or abs(recent_return) < float(rule["return_min_bps"]):
            return 0.0
        context_sign = _aligned_sign(
            flow,
            book,
            microprice,
            minimums=(rule["flow_min"], rule["book_min"], rule["microprice_min_bps"]),
        )
        if context_sign * recent_return <= 0.0:
            return 0.0
        magnitude = abs(oi_change) + abs(recent_return) + abs(flow) * 20.0
    elif mode == "oi_change_return_momentum":
        oi_min = float(rule["oi_abs_min_bps"])
        return_min = float(rule["return_abs_min_bps"])
        if abs(oi_change) < oi_min or abs(recent_return) < return_min:
            return 0.0
        context_sign = 1.0 if recent_return > 0.0 else -1.0
        magnitude = abs(oi_change) / oi_min + abs(recent_return) / return_min
    else:
        raise ValueError(f"unknown precursor rule mode: {mode}")
    return float(magnitude) if context_sign == direction_sign else 0.0


def _aligned_sign(
    first: float,
    second: float,
    third: float,
    *,
    minimums: Sequence[float],
) -> float:
    values = (first, second, third)
    if any(abs(value) < float(minimum) for value, minimum in zip(values, minimums)):
        return 0.0
    signs = {1.0 if value > 0.0 else -1.0 for value in values}
    return signs.pop() if len(signs) == 1 else 0.0


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net = [float(row["net_execution_bps"]) for row in rows]
    stress = [value - COST_STRESS_BPS for value in net]
    symbols = dict(sorted(Counter(str(row["symbol"]) for row in rows).items()))
    directions = dict(sorted(Counter(str(row["direction"]) for row in rows).items()))
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in net:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "fills": len(rows) * 2,
        "closed_trades": len(rows),
        "mean_gross_bps": _mean(row["gross_mid_bps"] for row in rows),
        "mean_total_cost_bps": _mean(row["total_cost_bps"] for row in rows),
        "mean_net_bps": _mean(net),
        "lcb_95_net_bps": _lcb(net),
        "mean_stress_net_bps": _mean(stress),
        "lcb_95_stress_net_bps": _lcb(stress),
        "net_sum_bps": sum(net),
        "max_drawdown_bps": drawdown,
        "win_rate": sum(value > 0.0 for value in net) / len(net) if net else 0.0,
        "symbols": symbols,
        "directions": directions,
        "top_symbol_trade_share": max(symbols.values()) / len(rows) if rows else 0.0,
        "minimum_direction_trade_share": (
            min(directions.get("LONG", 0), directions.get("SHORT", 0)) / len(rows)
            if rows
            else 0.0
        ),
    }


def _candidate_failures(
    *,
    role_metrics: Mapping[str, Mapping[str, Any]],
    session_reports: Sequence[Mapping[str, Any]],
    enough_sessions: bool,
) -> list[str]:
    if not enough_sessions:
        return ["prospective_session_evidence_incomplete"]
    failures = []
    for role in ("validation", "oos"):
        metrics = role_metrics[role]
        if int(metrics["fills"]) < 20 or int(metrics["closed_trades"]) < 10:
            failures.append(f"{role}_trade_count_failed")
        if metrics["mean_net_bps"] is None or float(metrics["mean_net_bps"]) <= 0.0:
            failures.append(f"{role}_nonpositive_expectancy")
        if metrics["lcb_95_net_bps"] is None or float(metrics["lcb_95_net_bps"]) <= 0.0:
            failures.append(f"{role}_nonpositive_lcb")
        if metrics["mean_stress_net_bps"] is None or float(metrics["mean_stress_net_bps"]) <= 0.0:
            failures.append(f"{role}_cost_stress_failed")
        if metrics["lcb_95_stress_net_bps"] is None or float(metrics["lcb_95_stress_net_bps"]) <= 0.0:
            failures.append(f"{role}_cost_stress_lcb_failed")
        if float(metrics["max_drawdown_bps"]) > 2_000.0:
            failures.append(f"{role}_drawdown_failed")
        if float(metrics["top_symbol_trade_share"]) > 0.50:
            failures.append(f"{role}_symbol_concentration_failed")
        if metrics["closed_trades"] and float(metrics["minimum_direction_trade_share"]) < 0.10:
            failures.append(f"{role}_direction_collapse")
    oos_sessions = [row for row in session_reports if row["role"] == "oos"]
    positive_oos = sum(
        row["metrics"]["mean_stress_net_bps"] is not None
        and float(row["metrics"]["mean_stress_net_bps"]) > 0.0
        for row in oos_sessions
    )
    if positive_oos / len(oos_sessions) < 2.0 / 3.0:
        failures.append("oos_positive_session_share_below_two_thirds")
    return failures


def _prospective_early_stop_reason(
    session_reports: Sequence[Mapping[str, Any]],
) -> str | None:
    validation = [row for row in session_reports if row["role"] == "validation"]
    if not validation:
        return None
    first = validation[0]["metrics"]
    if int(first["closed_trades"]) < 3:
        return "first_validation_low_incidence_early_stop"
    if (
        first["mean_stress_net_bps"] is None
        or float(first["mean_stress_net_bps"]) <= 0.0
    ):
        return "first_validation_nonpositive_stress_early_stop"
    return None


def _contract_payload(
    *,
    data_contract_id: str | None = None,
    candidate_rules: Sequence[Mapping[str, Any]] = CANDIDATE_RULES,
    horizon_minutes: int = HORIZON_MINUTES,
) -> dict[str, Any]:
    payload = {
        "candidate_rules": candidate_rules,
        "horizon_minutes": horizon_minutes,
        "cost_model": "captured_taker_execution_plus_fixed_stress",
        "cost_stress_bps": COST_STRESS_BPS,
        "selection": "one_global_nonoverlapping_position_per_session",
        "minimum_validation_sessions": MIN_VALIDATION_SESSIONS,
        "minimum_oos_sessions": MIN_OOS_SESSIONS,
        "minimum_session_duration_hours": MIN_SESSION_DURATION_HOURS,
        "minimum_session_coverage_pct": MIN_SESSION_COVERAGE_PCT,
        "sessions_are_never_stitched": True,
        "development_never_promotes": True,
    }
    if data_contract_id is not None:
        payload["required_data_contract"] = data_contract_id
    return payload


def _public_session_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "dataset_dir",
            "source_session_id",
            "manifest_sha256",
            "database_sha256",
            "start_timestamp_ms",
            "end_timestamp_ms",
            "duration_hours",
            "eligible_bar_coverage_pct",
            "frame_coverage_pct",
            "role",
        )
    }


def _mean(values: Sequence[float] | Any) -> float | None:
    parsed = [float(value) for value in values]
    return statistics.fmean(parsed) if parsed else None


def _lcb(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.fmean(values) - LCB_Z * statistics.stdev(values) / math.sqrt(len(values))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
