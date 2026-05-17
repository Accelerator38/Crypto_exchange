from __future__ import annotations

from panteon_v2.analysis.genetics_promotion import evaluate_promotion_gate


def _mode(
    fitness: float,
    robust_utility: float,
    *,
    passes: bool = True,
    turnover: float = 0.20,
    saturation: float = 0.10,
    invalid_open_pressure: float = 0.0,
) -> dict:
    return {
        "mode": "fee_fixed_nextbar",
        "fitness": fitness,
        "period_stats": {
            "median_ret": 0.3,
            "positive_period_pct": 100.0,
        },
        "robust_score": {
            "passes_default_gates": passes,
            "failed_gates": [] if passes else ["single_period_concentration"],
            "robust_utility": robust_utility,
            "median_ret": 0.3,
            "positive_period_pct": 100.0,
            "max_positive_contribution_pct": 15.0,
        },
        "contract_metrics": {
            "mean_turnover_rate": turnover,
            "mean_saturation_rate": saturation,
            "max_saturation_rate": saturation,
            "mean_invalid_open_logit_pressure": invalid_open_pressure,
            "max_invalid_open_logit_pressure": invalid_open_pressure,
        },
    }


def test_promotion_gate_accepts_candidate_with_better_robust_utility():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31)]},
            {"path": "candidate.npy", "modes": [_mode(13.6, 0.37)]},
        ]
    }

    result = evaluate_promotion_gate(report)

    assert result["accepted"] is True
    assert result["candidate_path"] == "candidate.npy"
    assert result["baseline_path"] == "baseline.npy"
    assert result["robust_utility_delta"] > 0.0


def test_promotion_gate_rejects_failed_candidate_robust_gates():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31)]},
            {"path": "candidate.npy", "modes": [_mode(15.0, 0.50, passes=False)]},
        ]
    }

    result = evaluate_promotion_gate(report)

    assert result["accepted"] is False
    assert "candidate_robust_gates" in result["failed_checks"]


def test_promotion_gate_rejects_weaker_robust_utility():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31)]},
            {"path": "candidate.npy", "modes": [_mode(16.0, 0.20)]},
        ]
    }

    result = evaluate_promotion_gate(report)

    assert result["accepted"] is False
    assert "robust_utility_delta" in result["failed_checks"]


def test_promotion_gate_rejects_candidate_with_excess_saturation():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31, saturation=0.12)]},
            {"path": "candidate.npy", "modes": [_mode(16.0, 0.40, saturation=0.34)]},
        ]
    }

    result = evaluate_promotion_gate(report, max_saturation_rate=0.25)

    assert result["accepted"] is False
    assert "max_position_saturation_rate" in result["failed_checks"]
    assert result["candidate_mean_saturation_rate"] == 0.34


def test_promotion_gate_rejects_candidate_with_excess_turnover():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31, turnover=0.18)]},
            {"path": "candidate.npy", "modes": [_mode(16.0, 0.40, turnover=0.42)]},
        ]
    }

    result = evaluate_promotion_gate(report, max_turnover_rate=0.35)

    assert result["accepted"] is False
    assert "max_turnover_rate" in result["failed_checks"]
    assert result["candidate_mean_turnover_rate"] == 0.42


def test_promotion_gate_rejects_candidate_with_excess_invalid_open_pressure():
    report = {
        "genomes": [
            {"path": "baseline.npy", "modes": [_mode(14.1, 0.31, invalid_open_pressure=0.02)]},
            {"path": "candidate.npy", "modes": [_mode(16.0, 0.40, invalid_open_pressure=1.35)]},
        ]
    }

    result = evaluate_promotion_gate(report, max_invalid_open_logit_pressure=0.50)

    assert result["accepted"] is False
    assert "max_invalid_open_logit_pressure" in result["failed_checks"]
    assert result["candidate_mean_invalid_open_logit_pressure"] == 1.35
