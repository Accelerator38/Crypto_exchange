from __future__ import annotations

import json
from pathlib import Path

import pytest

from panteon_v2.policy.development_candidate import (
    DevelopmentCandidateContract,
    DevelopmentCandidateError,
    validate_development_candidate,
)


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p3_cross_sectional_v1.json"
)
PAIRED_CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p4_market_neutral_pair_v1.json"
)


def _payload():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_project_development_candidate_is_fixed_and_has_no_authority():
    contract = DevelopmentCandidateContract.from_json(CONTRACT_PATH)

    assert contract.payload["protocol"]["stage"] == "development_only"
    assert contract.payload["protocol"]["sealed_windows"] == [
        "validation",
        "oos",
        "sanity",
    ]
    assert contract.payload["orders_enabled"] is False
    assert contract.payload["promotion_authority"] is False


def test_profile_change_breaks_development_registration():
    payload = _payload()
    payload["candidate"]["event_contract"]["momentum_lookback_bars"] = 179

    with pytest.raises(DevelopmentCandidateError, match="profile SHA"):
        validate_development_candidate(payload)


def test_development_contract_cannot_open_validation_window():
    payload = _payload()
    payload["protocol"]["development_end_at"] = (
        "2024-12-31T19:00:00+00:00"
    )

    with pytest.raises(DevelopmentCandidateError, match="sealed data"):
        validate_development_candidate(payload)


def test_development_gates_cannot_be_lowered():
    payload = _payload()
    payload["protocol"]["gates"]["min_closed_trades"] = 5

    with pytest.raises(DevelopmentCandidateError, match="gates are too weak"):
        validate_development_candidate(payload)


def test_paired_development_candidate_is_atomic_and_has_no_authority():
    contract = DevelopmentCandidateContract.from_json(PAIRED_CONTRACT_PATH)

    assert contract.payload["protocol"]["portfolio"]["max_open_positions"] == 2
    assert contract.payload["candidate"]["selection_contract"]["required_legs"] == 2
    assert contract.payload["candidate"]["orders_enabled"] is False
    assert contract.payload["candidate"]["promotion_authority"] is False


def test_paired_contract_cannot_weaken_atomic_fill_gate():
    payload = json.loads(PAIRED_CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["protocol"]["gates"]["paired_fill_atomicity_hard_fail"] = False

    with pytest.raises(DevelopmentCandidateError, match="gates are too weak"):
        validate_development_candidate(payload)
