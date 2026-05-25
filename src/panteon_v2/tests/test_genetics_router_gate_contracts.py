from __future__ import annotations

import importlib

import pytest


def _genome(path: str, rets: list[float], turnovers: list[float], saturations: list[float]) -> dict:
    return {
        "path": path,
        "position_state_features_enabled": False,
        "modes": [
            {
                "mode": "fee_fixed_nextbar",
                "period_rets": [float(value) for value in rets],
                "period_stats": {
                    "mean_ret": float(sum(rets) / len(rets)),
                    "min_ret": float(min(rets)),
                    "max_ret": float(max(rets)),
                    "positive_period_pct": float(
                        sum(1 for value in rets if value > 0.0) * 100.0 / len(rets)
                    ),
                },
                "mean_ret": float(sum(rets) / len(rets)),
                "min_ret": float(min(rets)),
                "max_ret": float(max(rets)),
                "positive_period_pct": float(
                    sum(1 for value in rets if value > 0.0) * 100.0 / len(rets)
                ),
                "contract_metrics": {
                    "mean_turnover_rate": float(sum(turnovers) / len(turnovers)),
                    "max_turnover_rate": float(max(turnovers)),
                    "mean_saturation_rate": float(sum(saturations) / len(saturations)),
                    "max_saturation_rate": float(max(saturations)),
                    "mean_invalid_open_logit_pressure": 0.0,
                    "max_invalid_open_logit_pressure": 0.0,
                    "periods": [
                        {
                            "period": f"2026-{idx + 1:02d}",
                            "regime": regime,
                            "ret": float(ret),
                            "turnover_rate": float(turnovers[idx]),
                            "saturation_rate": float(saturations[idx]),
                            "invalid_open_logit_pressure": 0.0,
                        }
                        for idx, (regime, ret) in enumerate(zip(["bearish", "neutral"], rets))
                    ],
                },
            }
        ],
    }


def _report(
    *,
    baseline_rets: list[float],
    candidate_rets: list[float],
    baseline_turnovers: list[float] | None = None,
    candidate_turnovers: list[float] | None = None,
    baseline_saturations: list[float] | None = None,
    candidate_saturations: list[float] | None = None,
) -> dict:
    n = len(baseline_rets)
    baseline_turnovers = baseline_turnovers or [0.01] * n
    candidate_turnovers = candidate_turnovers or [0.01] * n
    baseline_saturations = baseline_saturations or [0.01] * n
    candidate_saturations = candidate_saturations or [0.01] * n
    return {
        "mode": "fee_fixed_nextbar",
        "genomes": [
            _genome("baseline.npy", baseline_rets, baseline_turnovers, baseline_saturations),
            _genome("candidate.npy", candidate_rets, candidate_turnovers, candidate_saturations),
        ],
    }


def test_regime_router_multisplit_gate_rejects_oos_tie_for_selected_map():
    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
        },
    }
    oos = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[0.20, 0.30],
    )

    gate = selector.evaluate_regime_router_multisplit_gate(selection, [oos])

    assert gate["promotion_eligible"] is False
    assert "oos_0_mean_ret_tie" in gate["promotion_failures"]


def test_regime_router_multisplit_gate_rejects_per_period_max_turnover_and_saturation():
    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
        },
    }
    oos = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[0.20, 0.35],
        baseline_turnovers=[0.01, 0.01],
        candidate_turnovers=[0.01, 0.18],
        baseline_saturations=[0.01, 0.01],
        candidate_saturations=[0.01, 0.17],
    )

    gate = selector.evaluate_regime_router_multisplit_gate(
        selection,
        [oos],
        max_turnover_rate=0.10,
        max_saturation_rate=0.10,
    )

    assert gate["promotion_eligible"] is False
    assert "oos_0_max_turnover" in gate["promotion_failures"]
    assert "oos_0_max_saturation" in gate["promotion_failures"]
    assert gate["holdouts"][0]["candidate"]["mean_turnover_rate"] == pytest.approx(0.095)
    assert gate["holdouts"][0]["candidate"]["max_turnover_rate"] == pytest.approx(0.18)


def _touch_genome(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"genome")
    return str(path)


def test_specialists_manifest_builds_shadow_only_four_regime_map_with_crash_risk_off(tmp_path):
    selector = importlib.import_module("tools.select_genetics_candidate")
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "neutral_run"
    baseline = _touch_genome(run_dir / "warm_start_genome.npy")
    neutral = _touch_genome(run_dir / "archive_rank1.npy")
    risk_off = _touch_genome(results_root / "manifest_specialists" / "risk_off_hold_genome.npy")
    selection_source = results_root / "selection_router.json"
    selection_source.write_text("{}", encoding="utf-8")
    selection = {
        "selected_is_baseline": False,
        "promotion_eligible": True,
        "paper_trading_eligible": True,
        "live_trading_eligible": True,
        "promotion_failures": [],
        "selected_regime_map": {
            "bearish": baseline,
            "neutral": neutral,
            "bullish": baseline,
        },
        "baseline_regime_map": {
            "bearish": baseline,
            "neutral": baseline,
            "bullish": baseline,
        },
    }

    manifest = selector.build_specialists_manifest(
        selection,
        results_root=results_root,
        risk_off_crash_genome=risk_off,
        selection_source=selection_source,
        created_at="2026-05-25",
        baseline_control={"profile": "Round3 Deny8 EntryRegime"},
    )

    assert manifest["created_at"] == "2026-05-25"
    assert manifest["promotion_eligible"] is False
    assert manifest["paper_trading_eligible"] is False
    assert manifest["live_trading_eligible"] is False
    assert "crash_risk_off_fallback" in manifest["promotion_failures"]
    assert "requires_panteon_flash_shadow_matrix" in manifest["promotion_failures"]
    assert manifest["baseline_control"]["profile"] == "Round3 Deny8 EntryRegime"
    assert set(manifest["specialist_genome_map"]) == {
        "GeneticsBest",
        "GeneticsCrash",
        "GeneticsBullish",
        "GeneticsBearish",
        "GeneticsNeutral",
    }
    assert set(manifest["selected_regime_map"]) == {
        "crash",
        "bearish",
        "neutral",
        "bullish",
    }
    assert manifest["selected_regime_map"]["crash"] == risk_off
    assert manifest["selected_regime_map"]["neutral"] == neutral
    assert manifest["baseline_regime_map"]["crash"] == baseline
    assert manifest["specialist_genome_map"]["GeneticsBest"] == baseline
    assert manifest["specialist_genome_map"]["GeneticsCrash"] == risk_off
    assert manifest["specialist_genome_map"]["GeneticsNeutral"] == neutral


def test_specialists_manifest_rejects_paths_outside_neiro_genetics(tmp_path):
    selector = importlib.import_module("tools.select_genetics_candidate")
    results_root = tmp_path / "Results" / "neiro_genetics"
    baseline = _touch_genome(results_root / "run" / "warm_start_genome.npy")
    risk_off = _touch_genome(results_root / "manifest_specialists" / "risk_off_hold_genome.npy")
    outside = _touch_genome(tmp_path / "outside" / "archive_rank1.npy")
    selection = {
        "selected_regime_map": {
            "bearish": baseline,
            "neutral": outside,
            "bullish": baseline,
        },
        "baseline_regime_map": {
            "bearish": baseline,
            "neutral": baseline,
            "bullish": baseline,
        },
    }

    with pytest.raises(ValueError, match="Results/neiro_genetics"):
        selector.build_specialists_manifest(
            selection,
            results_root=results_root,
            risk_off_crash_genome=risk_off,
        )


def test_specialists_manifest_ignores_crash_candidate_without_explicit_oos_gate(tmp_path):
    selector = importlib.import_module("tools.select_genetics_candidate")
    results_root = tmp_path / "Results" / "neiro_genetics"
    baseline = _touch_genome(results_root / "run" / "warm_start_genome.npy")
    crash_candidate = _touch_genome(results_root / "crash_run" / "archive_rank1.npy")
    risk_off = _touch_genome(results_root / "manifest_specialists" / "risk_off_hold_genome.npy")
    selection = {
        "selected_regime_map": {
            "crash": crash_candidate,
            "bearish": baseline,
            "neutral": baseline,
            "bullish": baseline,
        },
        "baseline_regime_map": {
            "crash": baseline,
            "bearish": baseline,
            "neutral": baseline,
            "bullish": baseline,
        },
    }

    manifest = selector.build_specialists_manifest(
        selection,
        results_root=results_root,
        risk_off_crash_genome=risk_off,
    )

    assert manifest["selected_regime_map"]["crash"] == risk_off
    assert manifest["specialist_genome_map"]["GeneticsCrash"] == risk_off
    assert "crash_specialist_not_oos_eligible" in manifest["promotion_failures"]
