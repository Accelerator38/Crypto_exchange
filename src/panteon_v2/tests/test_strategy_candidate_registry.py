from __future__ import annotations

import json
from pathlib import Path

import pytest

from panteon_v2.policy.strategy_candidate_registry import (
    StrategyCandidateRegistry,
    StrategyCandidateRegistryError,
    compute_candidate_profile_sha256,
    validate_strategy_candidate_registry,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "configs" / "strategy_candidates_v1.json"


def _payload() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_project_candidates_are_pre_registered_without_authority():
    registry = StrategyCandidateRegistry.from_json(REGISTRY_PATH)

    assert registry.ready_candidate_ids == (
        "regime_pullback_hourly_v1",
        "ohlcv_compression_transition_hourly_v1",
    )
    assert registry.payload["operational_candidate_id"] is None
    assert all(
        candidate["paper_allowed"] is False
        and candidate["live_allowed"] is False
        and candidate["orders_enabled"] is False
        for candidate in registry.payload["candidates"]
    )


def test_profile_change_breaks_pre_registration_hash():
    payload = _payload()
    payload["candidates"][0]["event_contract"]["ema_fast_bars"] = 25

    with pytest.raises(StrategyCandidateRegistryError, match="SHA-256 mismatch"):
        validate_strategy_candidate_registry(payload)


def test_funding_candidate_stays_blocked_without_settlement_history():
    payload = _payload()
    funding = payload["candidates"][2]
    funding["status"] = "pre_registered"
    funding["historical_evaluation_allowed"] = True

    with pytest.raises(StrategyCandidateRegistryError, match="missing data"):
        validate_strategy_candidate_registry(payload)


def test_open_interest_dependency_is_forbidden():
    payload = _payload()
    candidate = payload["candidates"][0]
    candidate["data_requirements"]["required_fields"].append("open_interest")
    candidate["data_requirements"]["missing_fields"].append("open_interest")
    candidate["profile_sha256"] = compute_candidate_profile_sha256(candidate)

    with pytest.raises(StrategyCandidateRegistryError, match="open interest"):
        validate_strategy_candidate_registry(payload)


def test_promotion_gates_cannot_be_lowered_below_fixed_minimums():
    payload = _payload()
    payload["common_evaluation"]["gates"]["min_total_closed_trades"] = 10

    with pytest.raises(StrategyCandidateRegistryError, match="gates are too weak"):
        validate_strategy_candidate_registry(payload)
