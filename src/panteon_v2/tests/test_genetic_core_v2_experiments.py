from __future__ import annotations

from panteon_v2.analysis.genetic_core_v2_experiments import (
    candidate_metrics_from_mode,
    score_specialist_router,
    summarize_singleton_candidate,
)
from tools.report_genetic_core_v2_experiment import build_summary


def _mode(
    returns: list[float],
    regimes: list[str],
    *,
    long_rate: float = 0.10,
    short_rate: float = 0.10,
    direction_bias: float = 0.0,
) -> dict:
    periods = [
        {
            "period": f"2024-{idx + 1:02d}",
            "regime": regime,
            "turnover_rate": 0.02,
            "effective_turnover_rate": 0.02,
            "saturation_rate": 0.01,
            "invalid_open_logit_pressure": 0.0,
            "max_drawdown_pct": 0.2,
            "mean_long_slot_rate": long_rate,
            "mean_short_slot_rate": short_rate,
            "mean_net_direction_bias": direction_bias,
        }
        for idx, regime in enumerate(regimes)
    ]
    return {
        "mode": "fee_fixed_nextbar",
        "period_rets": returns,
        "contract_metrics": {"periods": periods},
    }


def _report(path_to_returns: dict[str, list[float]], regimes: list[str]) -> dict:
    return {
        "genomes": [
            {
                "path": path,
                "modes": [_mode(returns, regimes)],
            }
            for path, returns in path_to_returns.items()
        ]
    }


def _write_report_bundle(run_dir, validation: dict, oos: dict) -> None:
    import json

    for filename, payload in {
        "contract_train_2022_2023.json": validation,
        "contract_validation_2024.json": validation,
        "contract_oos_2025.json": oos,
        "contract_final_sanity_2026_h1.json": oos,
    }.items():
        (run_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )


def test_candidate_metrics_from_mode_passes_directional_fields_to_v2_fitness():
    metrics = candidate_metrics_from_mode(
        _mode(
            [0.5, 0.5, 0.5, 0.5],
            ["crash", "bearish", "neutral", "bullish"],
            long_rate=0.0,
            short_rate=0.35,
            direction_bias=-1.0,
        )
    )

    assert metrics["mean_ret"] == 0.5
    assert metrics["positive_period_pct"] == 100.0
    assert metrics["fitness_v4"]["direction_bias_penalty"] > 0.0
    assert "direction_bias" in metrics["fitness_v4"]["failed_gates"]


def test_report_summary_uses_persisted_guard_router_selection(tmp_path):
    import json

    regimes = ["crash", "bearish", "neutral", "bullish"]
    validation = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [0.6, 0.6, 0.6, 0.6],
        },
        regimes,
    )
    oos = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [0.4, 0.4, 0.4, 0.4],
        },
        regimes,
    )
    _write_report_bundle(tmp_path, validation, oos)
    persisted_map = {regime: "baseline.npy" for regime in regimes}
    (tmp_path / "selection_router_fitness_v4.json").write_text(
        json.dumps(
            {
                "mode": "fee_fixed_nextbar",
                "promotion_eligible": False,
                "promotion_failures": ["baseline_selected", "guard_rejected"],
                "baseline_path": "baseline.npy",
                "selected_regime_map": persisted_map,
                "thresholds": {"min_unique_selected_genomes": 2},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = build_summary(tmp_path, mode="fee_fixed_nextbar")

    assert summary["specialist_router_source"] == "selection_router_fitness_v4.json"
    assert summary["specialist_router"]["selected_regime_map"] == persisted_map
    assert summary["singleton_summary"]["selected_regime_map"] == persisted_map
    assert summary["singleton_summary"]["unique_selected_genomes"] == 1
    assert "guard_rejected" in summary["singleton_summary"]["promotion_failures"]


def test_score_specialist_router_selects_regime_specialists_but_returns_singleton():
    regimes = ["crash", "bearish", "neutral", "bullish"]
    validation = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "defensive.npy": [0.7, 0.6, -0.1, -0.2],
            "growth.npy": [-0.2, -0.1, 0.6, 0.7],
        },
        regimes,
    )
    oos = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "defensive.npy": [0.3, 0.3, 0.2, 0.2],
            "growth.npy": [0.2, 0.2, 0.3, 0.3],
        },
        regimes,
    )

    result = score_specialist_router(
        validation_report=validation,
        oos_reports=[oos],
        baseline_path="baseline.npy",
    )
    summary = summarize_singleton_candidate(result)

    assert result["selected_regime_map"] == {
        "crash": "defensive.npy",
        "bearish": "defensive.npy",
        "neutral": "growth.npy",
        "bullish": "growth.npy",
    }
    assert summary["deployment_shape"] == "singleton"
    assert summary["live_policy"] == "probation_only"
    assert summary["uses_live_ensemble"] is False


def test_score_specialist_router_rejects_oos_degradation():
    regimes = ["crash", "bearish", "neutral", "bullish"]
    validation = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [0.4, 0.4, 0.4, 0.4],
        },
        regimes,
    )
    oos = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [-0.5, -0.4, -0.3, -0.2],
        },
        regimes,
    )

    result = score_specialist_router(
        validation_report=validation,
        oos_reports=[oos],
        baseline_path="baseline.npy",
    )

    assert result["promotion_eligible"] is False
    assert "oos_0_mean_ret_not_positive" in result["promotion_failures"]
    assert "oos_0_fitness_v4_gates" in result["promotion_failures"]


def test_score_specialist_router_rejects_single_genome_router_collapse():
    regimes = ["crash", "bearish", "neutral", "bullish"]
    validation = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [0.4, 0.4, 0.4, 0.4],
        },
        regimes,
    )
    oos = _report(
        {
            "baseline.npy": [0.1, 0.1, 0.1, 0.1],
            "specialist.npy": [0.3, 0.3, 0.3, 0.3],
        },
        regimes,
    )

    result = score_specialist_router(
        validation_report=validation,
        oos_reports=[oos],
        baseline_path="baseline.npy",
    )
    summary = summarize_singleton_candidate(result)

    assert result["selected_regime_map"] == {
        "crash": "specialist.npy",
        "bearish": "specialist.npy",
        "neutral": "specialist.npy",
        "bullish": "specialist.npy",
    }
    assert result["unique_selected_genomes"] == 1
    assert result["promotion_eligible"] is False
    assert "router_collapse" in result["promotion_failures"]
    assert summary["unique_selected_genomes"] == 1
