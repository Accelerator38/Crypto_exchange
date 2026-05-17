from __future__ import annotations

from panteon_v2.analysis.genetics_validation import (
    build_rolling_year_folds,
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
