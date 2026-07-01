from __future__ import annotations

from panteon_v2.analysis.genetics_validation import (
    build_fixed_walk_forward_contract,
    build_rolling_year_folds,
    evaluate_fitness_v4_uplift_gate,
    evaluate_fitness_v3_promotion_gate,
    fitness_v4_robust_score,
    fitness_v3_robust_score,
    robust_period_score,
    split_precomp_by_periods,
)


def _monthly_periods(start_year: int, end_year: int) -> list[str]:
    return [
        f"{year}-{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def test_build_rolling_year_folds_keeps_final_test_year_out():
    folds = build_rolling_year_folds(
        _monthly_periods(2018, 2025),
        train_years=4,
        validation_years=1,
        final_test_year=2025,
    )

    assert [fold["fold_id"] for fold in folds] == [
        "train_2018_2021_validate_2022",
        "train_2019_2022_validate_2023",
        "train_2020_2023_validate_2024",
    ]
    assert folds[0]["train_periods"][0] == "2018-01"
    assert folds[0]["train_periods"][-1] == "2021-12"
    assert folds[0]["validation_periods"][0] == "2022-01"
    assert folds[0]["validation_periods"][-1] == "2022-12"
    assert all("2025" not in period for fold in folds for period in fold["validation_periods"])


def test_build_rolling_year_folds_supports_embargo_months():
    folds = build_rolling_year_folds(
        _monthly_periods(2020, 2023),
        train_years=2,
        validation_years=1,
        embargo_months=2,
    )

    assert folds[0]["train_periods"][-1] == "2021-10"
    assert folds[0]["embargo_periods"] == ["2021-11", "2021-12"]
    assert folds[0]["validation_periods"][0] == "2022-01"


def test_split_precomp_by_periods_uses_period_field_at_index_four():
    precomp = [
        ("feat1", "prices1", "syms1", 1, "2024-01", 1.0, "neutral"),
        ("feat2", "prices2", "syms2", 2, "2024-02", 1.0, "bearish"),
        ("feat3", "prices3", "syms3", 3, "2024-03", 1.0, "bullish"),
    ]

    subset = split_precomp_by_periods(precomp, ["2024-03", "2024-01"])

    assert [entry[4] for entry in subset] == ["2024-01", "2024-03"]


def test_robust_period_score_rejects_single_outlier_concentration():
    score = robust_period_score([0.5] * 20 + [100.0])

    assert score["max_positive_contribution_pct"] > 30.0
    assert not score["passes_default_gates"]
    assert "single_period_concentration" in score["failed_gates"]


def test_robust_period_score_accepts_stable_positive_distribution():
    score = robust_period_score([0.6, 0.8, 1.0, 1.1, 0.7, -0.2, 0.9, 1.2, 0.5, 0.8])

    assert score["median_ret"] > 0.0
    assert score["trimmed_mean_ret"] > 0.0
    assert score["positive_period_pct"] >= 80.0
    assert score["passes_default_gates"]


def test_fixed_walk_forward_contract_uses_train_validation_oos_and_final_sanity():
    contract = build_fixed_walk_forward_contract(_monthly_periods(2022, 2026))

    assert contract["train_periods"][0] == "2022-01"
    assert contract["train_periods"][-1] == "2023-12"
    assert contract["validation_periods"] == _monthly_periods(2024, 2024)
    assert contract["oos_periods"] == _monthly_periods(2025, 2025)
    assert contract["final_sanity_periods"][0] == "2026-01"
    assert contract["final_sanity_periods"][-1] == "2026-06"
    assert contract["regime_balance_required"] == ["crash", "bearish", "neutral", "bullish"]


def test_fitness_v3_robust_penalizes_cost_turnover_saturation_and_invalid_pressure():
    clean = fitness_v3_robust_score(
        [1.0, 1.0, 1.0, 1.0],
        period_max_drawdowns_pct=[0.5, 0.5, 0.5, 0.5],
        period_turnover_rates=[0.03, 0.03, 0.03, 0.03],
        period_saturation_rates=[0.00, 0.00, 0.00, 0.00],
        period_invalid_open_pressures=[0.00, 0.00, 0.00, 0.00],
        period_costs_pct=[0.02, 0.02, 0.02, 0.02],
        period_slippage_pct=[0.01, 0.01, 0.01, 0.01],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    expensive = fitness_v3_robust_score(
        [1.0, 1.0, 1.0, 1.0],
        period_max_drawdowns_pct=[2.0, 2.0, 2.0, 2.0],
        period_turnover_rates=[0.30, 0.30, 0.30, 0.30],
        period_saturation_rates=[0.20, 0.20, 0.20, 0.20],
        period_invalid_open_pressures=[0.40, 0.40, 0.40, 0.40],
        period_costs_pct=[0.40, 0.40, 0.40, 0.40],
        period_slippage_pct=[0.20, 0.20, 0.20, 0.20],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    assert clean["fitness_v3_robust"] > expensive["fitness_v3_robust"]
    assert "turnover" in expensive["failed_gates"]
    assert "saturation" in expensive["failed_gates"]
    assert "invalid_open_pressure" in expensive["failed_gates"]


def test_fitness_v3_robust_regime_balance_does_not_hide_missing_crash():
    score = fitness_v3_robust_score(
        [2.0, 2.0, 2.0],
        period_regimes=["bearish", "neutral", "bullish"],
    )

    assert score["mean_ret"] == 2.0
    assert score["regime_balanced_mean_ret"] < score["mean_ret"]
    assert "missing_regime:crash" in score["failed_gates"]


def test_fitness_v4_prefers_regime_balanced_utility_over_single_outlier():
    stable = fitness_v4_robust_score(
        [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5],
        period_turnover_rates=[0.02] * 8,
        period_saturation_rates=[0.01] * 8,
        period_invalid_open_pressures=[0.00] * 8,
        period_regimes=["crash", "bearish", "neutral", "bullish"] * 2,
    )
    spiky = fitness_v4_robust_score(
        [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 8.0],
        period_turnover_rates=[0.02] * 8,
        period_saturation_rates=[0.01] * 8,
        period_invalid_open_pressures=[0.00] * 8,
        period_regimes=["crash", "bearish", "neutral", "bullish"] * 2,
    )

    assert stable["fitness_v4_robust"] > spiky["fitness_v4_robust"]
    assert spiky["max_positive_contribution_pct"] > 30.0
    assert "single_period_concentration" in spiky["failed_gates"]
    assert spiky["passes_default_gates"] is False


def test_fitness_v4_penalizes_one_sided_direction_bias():
    balanced = fitness_v4_robust_score(
        [0.6, 0.6, 0.6, 0.6],
        period_long_slot_rates=[0.10, 0.10, 0.10, 0.10],
        period_short_slot_rates=[0.10, 0.10, 0.10, 0.10],
        period_net_direction_biases=[0.0, 0.0, 0.0, 0.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    short_only = fitness_v4_robust_score(
        [0.6, 0.6, 0.6, 0.6],
        period_long_slot_rates=[0.0, 0.0, 0.0, 0.0],
        period_short_slot_rates=[0.35, 0.35, 0.35, 0.35],
        period_net_direction_biases=[-1.0, -1.0, -1.0, -1.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    assert balanced["fitness_v4_robust"] > short_only["fitness_v4_robust"]
    assert short_only["direction_bias_penalty"] > 0.0
    assert "direction_bias" in short_only["failed_gates"]


def test_fitness_v4_penalizes_empty_period_collapse():
    stable = fitness_v4_robust_score(
        [0.25, 0.25, 0.25, 0.25],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
        max_zero_period_pct=25.0,
        zero_period_penalty_weight=40.0,
    )
    collapsed = fitness_v4_robust_score(
        [0.0, 0.0, 0.0, 1.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
        max_zero_period_pct=25.0,
        zero_period_penalty_weight=40.0,
    )

    assert stable["zero_period_pct"] == 0.0
    assert collapsed["zero_period_pct"] == 75.0
    assert collapsed["zero_period_penalty"] > 0.0
    assert stable["fitness_v4_robust"] > collapsed["fitness_v4_robust"]
    assert "zero_period_collapse" in collapsed["failed_gates"]


def test_fitness_v4_penalizes_persistent_one_sided_direction_bias():
    alternating = fitness_v4_robust_score(
        [0.6, 0.6, 0.6, 0.6],
        period_net_direction_biases=[-1.0, 1.0, -1.0, 1.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
        max_direction_bias_abs=0.50,
        direction_bias_penalty_weight=20.0,
        max_persistent_direction_bias_abs=0.25,
        persistent_direction_bias_penalty_weight=30.0,
    )
    short_only = fitness_v4_robust_score(
        [0.6, 0.6, 0.6, 0.6],
        period_net_direction_biases=[-1.0, -1.0, -1.0, -1.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
        max_direction_bias_abs=0.50,
        direction_bias_penalty_weight=20.0,
        max_persistent_direction_bias_abs=0.25,
        persistent_direction_bias_penalty_weight=30.0,
    )

    assert alternating["direction_bias_penalty"] == short_only["direction_bias_penalty"]
    assert alternating["persistent_direction_bias_penalty"] == 0.0
    assert short_only["persistent_direction_bias_penalty"] > 0.0
    assert alternating["fitness_v4_robust"] > short_only["fitness_v4_robust"]
    assert "persistent_direction_bias" in short_only["failed_gates"]


def test_fitness_v4_penalizes_regime_collapse():
    balanced = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    collapsed = fitness_v4_robust_score(
        [-0.3, -0.2, -0.1, 1.9],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    assert balanced["regime_positive_rate"] == 1.0
    assert collapsed["regime_positive_rate"] == 0.25
    assert balanced["fitness_v4_robust"] > collapsed["fitness_v4_robust"]
    assert collapsed["regime_collapse_penalty"] > 0.0
    assert "regime_collapse" in collapsed["failed_gates"]


def test_fitness_v4_uplift_gate_rejects_oos_and_final_sanity_degradation():
    baseline_validation = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_validation = fitness_v4_robust_score(
        [0.5, 0.5, 0.5, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    baseline_oos = fitness_v4_robust_score(
        [0.3, 0.3, 0.3, 0.3],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_oos = fitness_v4_robust_score(
        [0.3, 0.3, 0.3, -0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    baseline_final = fitness_v4_robust_score(
        [0.2, 0.2, 0.2, 0.2],
        period_turnover_rates=[0.02] * 4,
        period_saturation_rates=[0.02] * 4,
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_final = fitness_v4_robust_score(
        [0.2, 0.2, 0.2, 0.2],
        period_turnover_rates=[0.22] * 4,
        period_saturation_rates=[0.02] * 4,
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    gate = evaluate_fitness_v4_uplift_gate(
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        baseline_oos=baseline_oos,
        candidate_oos=candidate_oos,
        baseline_final_sanity=baseline_final,
        candidate_final_sanity=candidate_final,
    )

    assert gate["promotion_eligible"] is False
    assert gate["validation"]["mean_ret_delta"] > 0.0
    assert gate["oos"]["mean_ret_delta"] < 0.0
    assert "oos_mean_ret" in gate["promotion_failures"]
    assert "oos_min_ret" in gate["promotion_failures"]
    assert "final_sanity_turnover" in gate["promotion_failures"]


def test_fitness_v4_promotion_requires_validation_oos_and_final_sanity():
    baseline_validation = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_validation = fitness_v4_robust_score(
        [0.5, 0.5, 0.5, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    gate = evaluate_fitness_v4_uplift_gate(
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
    )

    assert gate["promotion_eligible"] is False
    assert "missing_oos" in gate["promotion_failures"]
    assert "missing_final_sanity" in gate["promotion_failures"]
    assert gate["oos"] is None
    assert gate["final_sanity"] is None


def test_fitness_v4_promotion_hard_blocks_direction_and_regime_collapse():
    baseline_validation = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_validation = fitness_v4_robust_score(
        [0.6, 0.6, 0.6, 0.6],
        period_net_direction_biases=[-1.0, -1.0, -1.0, -1.0],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    baseline_oos = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_oos = fitness_v4_robust_score(
        [-0.1, -0.1, -0.1, 1.9],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    baseline_final = fitness_v4_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_final = fitness_v4_robust_score(
        [0.5, 0.5, 0.5, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    gate = evaluate_fitness_v4_uplift_gate(
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        baseline_oos=baseline_oos,
        candidate_oos=candidate_oos,
        baseline_final_sanity=baseline_final,
        candidate_final_sanity=candidate_final,
    )

    assert gate["promotion_eligible"] is False
    assert "hard_block:validation_direction_bias" in gate["promotion_failures"]
    assert "hard_block:validation_persistent_direction_bias" in gate["promotion_failures"]
    assert "hard_block:oos_regime_collapse" in gate["promotion_failures"]


def test_fitness_v3_gate_rejects_validation_tie_oos_loss_and_crash_floor_loss():
    baseline_validation = fitness_v3_robust_score(
        [1.0, 0.8, 0.6, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_validation = fitness_v3_robust_score(
        [1.0, 0.8, 0.6, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    baseline_oos = fitness_v3_robust_score(
        [0.5, 0.5, 0.5, 0.5],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )
    candidate_oos = fitness_v3_robust_score(
        [0.4, 0.4, 0.4, 0.4],
        period_regimes=["crash", "bearish", "neutral", "bullish"],
    )

    gate = evaluate_fitness_v3_promotion_gate(
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        baseline_oos=baseline_oos,
        candidate_oos=candidate_oos,
    )

    assert gate["promotion_eligible"] is False
    assert "validation_tie" in gate["promotion_failures"]
    assert "oos_mean_ret" in gate["promotion_failures"]
    assert "crash_floor" in gate["promotion_failures"]
