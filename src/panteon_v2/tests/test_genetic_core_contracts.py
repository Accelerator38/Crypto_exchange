from __future__ import annotations

import pytest

from panteon_v2.analysis.genetic_core_contracts import (
    GENETIC_CORE_WALK_FORWARD_WINDOWS,
    PROVEN_BC_SEED_AGENTS,
    TEACHER_FEATURE_POLICY,
    TEACHER_RUNTIME_LOGITS_ENABLED,
    exchange_cost_profile,
    genetic_core_training_score,
    validate_genetic_core_training_contract,
)


def test_genetic_core_walk_forward_contract_is_fixed():
    contract = validate_genetic_core_training_contract(
        train_start="2022-01-01",
        train_end="2023-12-31",
        validation_start="2024-01-01",
        validation_end="2024-12-31",
        oos_start="2025-01-01",
        oos_end="2025-12-31",
        sanity_start="2026-01-01",
        sanity_end="2026-06-30",
        bc_seed_agents=PROVEN_BC_SEED_AGENTS,
        exchange="MEXC",
    )

    assert GENETIC_CORE_WALK_FORWARD_WINDOWS["train"] == (
        "2022-01-01",
        "2023-12-31",
    )
    assert contract.windows["sanity"] == ("2026-01-01", "2026-06-30")
    assert contract.bc_seed_agents == PROVEN_BC_SEED_AGENTS


def test_teacher_logits_remain_research_only_until_oos_stabilizes():
    contract = validate_genetic_core_training_contract(
        train_start="2022-01-01",
        train_end="2023-12-31",
        validation_start="2024-01-01",
        validation_end="2024-12-31",
        oos_start="2025-01-01",
        oos_end="2025-12-31",
        sanity_start="2026-01-01",
        sanity_end="2026-06-30",
        bc_seed_agents=PROVEN_BC_SEED_AGENTS,
        exchange="MEXC",
    )

    assert TEACHER_RUNTIME_LOGITS_ENABLED is False
    assert TEACHER_FEATURE_POLICY == "research_after_oos_stabilization"
    assert contract.teacher_runtime_logits_enabled is False
    assert contract.teacher_feature_policy == TEACHER_FEATURE_POLICY


def test_genetic_core_contract_rejects_window_drift():
    with pytest.raises(ValueError, match="train window"):
        validate_genetic_core_training_contract(
            train_start="2021-01-01",
            train_end="2023-12-31",
            validation_start="2024-01-01",
            validation_end="2024-12-31",
            oos_start="2025-01-01",
            oos_end="2025-12-31",
            sanity_start="2026-01-01",
            sanity_end="2026-06-30",
            bc_seed_agents=PROVEN_BC_SEED_AGENTS,
            exchange="MEXC",
        )


def test_genetic_core_contract_rejects_non_proven_bc_seed_agents():
    assert PROVEN_BC_SEED_AGENTS == (
        "LiveOIBreakout",
        "MomentumScalper",
        "ResearchValidatorAgent",
    )

    with pytest.raises(ValueError, match="BC seed agents"):
        validate_genetic_core_training_contract(
            train_start="2022-01-01",
            train_end="2023-12-31",
            validation_start="2024-01-01",
            validation_end="2024-12-31",
            oos_start="2025-01-01",
            oos_end="2025-12-31",
            sanity_start="2026-01-01",
            sanity_end="2026-06-30",
            bc_seed_agents=("Panteon_Flash",),
            exchange="MEXC",
        )


def test_exchange_cost_profile_includes_execution_constraints():
    profile = exchange_cost_profile("MEXC")

    assert profile.exchange == "MEXC"
    assert profile.fees_bps > 0.0
    assert profile.funding_bps >= 0.0
    assert profile.spread_bps > 0.0
    assert profile.slippage_bps > 0.0
    assert profile.min_notional_usd > 0.0
    assert profile.quantity_precision_step > 0.0


def test_training_score_uses_robust_components_and_bad_signal_key_penalty():
    clean = genetic_core_training_score(
        period_returns_pct=[1.2, 0.8, 0.9, 1.1],
        period_lcb_pct=[0.4, 0.3, 0.35, 0.45],
        period_max_drawdown_pct=[2.0, 2.5, 2.2, 2.1],
        period_costs_pct=[0.03, 0.03, 0.03, 0.03],
        turnover_rates=[0.03, 0.04, 0.03, 0.04],
        invalid_open_pressures=[0.0, 0.0, 0.0, 0.0],
        bad_signal_key_rates=[0.0, 0.0, 0.0, 0.0],
        concentration_pct=18.0,
    )
    dirty = genetic_core_training_score(
        period_returns_pct=[1.2, 0.8, 0.9, 1.1],
        period_lcb_pct=[0.4, 0.3, 0.35, 0.45],
        period_max_drawdown_pct=[2.0, 2.5, 2.2, 2.1],
        period_costs_pct=[0.40, 0.35, 0.40, 0.35],
        turnover_rates=[0.32, 0.30, 0.35, 0.31],
        invalid_open_pressures=[0.20, 0.35, 0.20, 0.35],
        bad_signal_key_rates=[0.40, 0.45, 0.50, 0.55],
        concentration_pct=65.0,
    )

    assert clean.score > dirty.score
    assert {"lcb_pct", "calmar", "cvar_5_pct", "positive_window_pct"} <= set(
        clean.components
    )
    assert dirty.penalties["bad_signal_key_penalty"] > 0.0
