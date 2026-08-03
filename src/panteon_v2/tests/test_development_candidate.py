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
PORTFOLIO_CONTRACT_PATH = (
    ROOT
    / "configs"
    / "strategy_candidate_p5_market_neutral_portfolio_v1.json"
)
COINTEGRATION_CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p6_cointegration_spread_v1.json"
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


def test_portfolio_development_candidate_has_one_run_budget():
    contract = DevelopmentCandidateContract.from_json(PORTFOLIO_CONTRACT_PATH)
    selection = contract.payload["candidate"]["selection_contract"]

    assert selection["long_legs"] == 2
    assert selection["short_legs"] == 2
    assert selection["development_run_budget"] == 1
    assert selection["parameter_sweep_allowed"] is False
    assert contract.payload["protocol"]["portfolio"]["max_open_positions"] == 4


def test_portfolio_contract_cannot_enable_parameter_sweep():
    payload = json.loads(PORTFOLIO_CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["candidate"]["selection_contract"]["parameter_sweep_allowed"] = True

    with pytest.raises(
        DevelopmentCandidateError,
        match="low-turnover cost-aware",
    ):
        validate_development_candidate(payload)


def test_cointegration_candidate_has_no_symbol_allowlist_or_sweep():
    contract = DevelopmentCandidateContract.from_json(
        COINTEGRATION_CONTRACT_PATH
    )
    candidate = contract.payload["candidate"]

    assert candidate["hypothesis"]["symbol_allowlist"] == []
    assert candidate["selection_contract"]["pair_universe"] == (
        "all_28_unordered_full8_pairs"
    )
    assert candidate["selection_contract"]["development_run_budget"] == 1
    assert candidate["selection_contract"]["parameter_sweep_allowed"] is False
    assert contract.payload["protocol"]["portfolio"][
        "gross_notional_per_trade_usd"
    ] == 20.0


def test_cointegration_candidate_cannot_add_symbol_allowlist():
    payload = json.loads(
        COINTEGRATION_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    payload["candidate"]["hypothesis"]["symbol_allowlist"] = ["BTC", "ETH"]

    with pytest.raises(
        DevelopmentCandidateError,
        match="low-turnover cost-aware",
    ):
        validate_development_candidate(payload)
