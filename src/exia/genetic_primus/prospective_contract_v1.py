"""Fail-closed validation for the frozen prospective three-day protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATES = (
    "NoTrade",
    "EMA_TrendConsensus",
    "GA_overlay_f7a4ae11",
)
SAFETY_KEYS = (
    "credentials_allowed",
    "orders_enabled",
    "paper_allowed",
    "live_allowed",
    "promotion_authority",
    "runtime_authority",
    "network_allowed",
)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be explicit UTC")
    return parsed.astimezone(timezone.utc)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_origins(contract: dict[str, Any]) -> tuple[tuple[datetime, datetime], ...]:
    outer = contract["outer"]
    start = _utc(outer["start_utc"])
    end = _utc(outer["end_exclusive_utc"])
    horizon = timedelta(days=int(outer["deployment_horizon_days"]))
    cadence = timedelta(days=int(outer["retrain_cadence_days"]))
    if horizon != cadence or outer["overlap_allowed"]:
        raise ValueError("origins must be non-overlapping and cadence must equal horizon")
    if end <= start or (end - start) % cadence:
        raise ValueError("outer interval must contain whole origins")
    origins = tuple(
        (start + index * cadence, start + (index + 1) * horizon)
        for index in range(int((end - start) / cadence))
    )
    if len(origins) != int(outer["origin_count"]):
        raise ValueError("origin_count does not match frozen bounds")
    return origins


def validate_contract(contract: dict[str, Any]) -> tuple[tuple[datetime, datetime], ...]:
    if contract["schema_version"] != "exia.genetic_primus.prospective_3d/1":
        raise ValueError("unexpected schema")
    origins = derive_origins(contract)
    if _utc(contract["frozen_at_utc"]) >= origins[0][0]:
        raise ValueError("contract must be frozen before the outer interval")
    if contract["outer"]["outcomes_available_at_registration"]:
        raise ValueError("outer outcomes were already available")

    training = contract["training"]
    if training["purge_bars"] != training["label_horizon_bars"] or training["purge_bars"] != 7:
        raise ValueError("seven-bar label purge is mandatory")
    if training["timeframe_minutes"] != 240:
        raise ValueError("this protocol is fixed to four-hour bars")
    if training["genetic_search_during_outer"] or training["genome_updates_during_outer"]:
        raise ValueError("outer search or genome mutation is forbidden")

    candidate_ids = tuple(item["candidate_id"] for item in contract["candidates"])
    if candidate_ids != EXPECTED_CANDIDATES or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate set or order changed")
    overlay = contract["candidates"][2]
    if overlay["genome"] != [0, 0, 1, 1, -1]:
        raise ValueError("frozen overlay genome changed")
    if len([weight for weight in overlay["genome"][:-1] if weight]) > 2:
        raise ValueError("overlay is not sparse")

    measurement = contract["measurement"]
    if measurement["daily_observations"] != 30 or len(origins) != 10:
        raise ValueError("qualified first-stage review requires 30 days and 10 origins")
    if not measurement["opportunities_and_trades_are_separate_gates"]:
        raise ValueError("opportunities and trades must remain separate gates")
    if measurement["selection_or_tuning_on_outer"]:
        raise ValueError("outer selection is forbidden")

    commitment = contract["data_commitment"]
    if commitment["future_outer_data_hash_known_at_registration"]:
        raise ValueError("a future data hash cannot be claimed at registration")
    if "sha256" not in commitment["required_before_evaluation"]:
        raise ValueError("evaluation needs an immutable data hash")
    if any(contract["safety"].get(key) is not False for key in SAFETY_KEYS):
        raise ValueError("all runtime and network permissions must be false")
    if contract["decision"]["automatic_promotion"]:
        raise ValueError("automatic promotion is forbidden")
    return origins


def load_and_validate(path: Path) -> tuple[dict[str, Any], tuple[tuple[datetime, datetime], ...]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    return contract, validate_contract(contract)
