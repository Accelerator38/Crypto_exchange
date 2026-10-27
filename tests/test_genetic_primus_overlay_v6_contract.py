from pathlib import Path

import numpy as np

from exia.genetic_primus.benchmarks_v4 import (
    RELATIVE_MOMENTUM_RANK_INDEX,
    relative_momentum_policy,
)
from exia.genetic_primus.contracts import FEATURE_NAMES
from exia.genetic_primus.fitness_recency import recency_ess_threshold
from exia.genetic_primus.objective_registry import load_immutable_objective_registry
from exia.genetic_primus.rolling import (
    build_rolling_origins,
    genetic_simulation_kwargs,
    load_rolling_protocol,
)
from exia.genetic_primus.simulator import overlay_opportunity_mask
from tools.run_genetic_primus_rolling_3d import build_revealed_engineering_override


ROOT = Path(__file__).resolve().parents[1]


def test_v6_contract_has_exactly_sixty_unopened_origins() -> None:
    protocol = load_rolling_protocol(
        ROOT / "configs/genetic_primus_overlay_3d_v6_relative_opportunity_sealed.json"
    )
    origins = build_rolling_origins(protocol)
    assert len(origins) == 60
    assert protocol.payload["sealed_interval"]["status"] == "UNOPENED"
    assert protocol.payload["safety"] == {
        "credentials_allowed": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def test_v6_registry_appends_without_rewriting_history() -> None:
    registry = load_immutable_objective_registry(
        ROOT / "configs/genetic_primus_objective_registry_v3.json", workspace=ROOT
    )
    assert registry["objective_count"] == 21
    assert registry["objective_ids"][-1] == "overlay_3d_v6_relative_opportunity_sealed"


def test_relative_momentum_proposal_uses_fixed_quartile_deadband() -> None:
    policy = relative_momentum_policy()
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[RELATIVE_MOMENTUM_RANK_INDEX] = 0.75
    assert policy.target(features) == 1
    features[RELATIVE_MOMENTUM_RANK_INDEX] = -0.75
    assert policy.target(features) == -1
    features[RELATIVE_MOMENTUM_RANK_INDEX] = 0.25
    assert policy.target(features) == 0


def test_feature_and_return_threshold_units_are_not_mixed() -> None:
    values = np.asarray([0.40, 0.60, -0.70, np.nan])
    assert overlay_opportunity_mask(
        values, threshold_mode="absolute_feature", threshold_value=0.5
    ).tolist() == [False, True, True, False]
    returns = np.asarray([0.001, 0.003])
    assert overlay_opportunity_mask(
        returns, threshold_mode="return_bps", threshold_value=24.0
    ).tolist() == [False, True]


def test_v6_simulation_kwargs_preserve_feature_units() -> None:
    protocol = load_rolling_protocol(
        ROOT / "configs/genetic_primus_overlay_3d_v6_relative_opportunity_sealed.json"
    )
    kwargs = genetic_simulation_kwargs(protocol)
    assert kwargs["overlay_threshold_mode"] == "absolute_feature"
    assert kwargs["overlay_threshold_value"] == 0.5


def test_recency_ess_threshold_adapts_only_to_observed_window_length() -> None:
    config = {
        "minimum_ess_days": {"90": 120.0},
        "minimum_ess_fraction_of_observed_days": 0.75,
    }
    assert recency_ess_threshold(config, half_life_days=90, observed_days=60) == 45.0
    assert recency_ess_threshold(config, half_life_days=90, observed_days=365) == 120.0


def test_v6_revealed_engineering_override_preserves_sealed_base() -> None:
    base = load_rolling_protocol(
        ROOT / "configs/genetic_primus_overlay_3d_v6_relative_opportunity_sealed.json"
    )
    override = build_revealed_engineering_override(
        base,
        validation_start="2025-02-01T00:00:00Z",
        validation_end="2025-02-04T00:00:00Z",
    )
    assert base.payload["sealed_interval"]["status"] == "UNOPENED"
    assert override.payload["sealed_interval"]["status"] == "REVEALED_ENGINEERING"
    assert override.payload["engineering_override"]["non_evidentiary"] is True
    assert override.payload["engineering_override"]["sealed_interval_touched"] is False
    assert "objective_registry" not in override.payload
    assert override.sha256 != base.sha256


def test_v6_revealed_engineering_override_rejects_sealed_overlap() -> None:
    base = load_rolling_protocol(
        ROOT / "configs/genetic_primus_overlay_3d_v6_relative_opportunity_sealed.json"
    )
    try:
        build_revealed_engineering_override(
            base,
            validation_start="2026-01-01T00:00:00Z",
            validation_end="2026-01-04T00:00:00Z",
        )
    except ValueError as exc:
        assert "before the sealed interval" in str(exc)
    else:
        raise AssertionError("sealed-overlap engineering interval was accepted")
