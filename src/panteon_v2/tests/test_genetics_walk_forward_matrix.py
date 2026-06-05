from __future__ import annotations

import importlib
import json

import pytest


def _genome(
    path: str,
    rets: list[float],
    *,
    max_turnover: float = 0.03,
    max_saturation: float = 0.03,
    max_invalid_open: float = 0.0,
) -> dict:
    return {
        "path": path,
        "position_state_features_enabled": True,
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
                "robust_score": {
                    "passes_default_gates": True,
                    "failed_gates": [],
                },
                "contract_metrics": {
                    "mean_turnover_rate": max_turnover / 2.0,
                    "max_turnover_rate": max_turnover,
                    "mean_saturation_rate": max_saturation / 2.0,
                    "max_saturation_rate": max_saturation,
                    "mean_invalid_open_logit_pressure": max_invalid_open / 2.0,
                    "max_invalid_open_logit_pressure": max_invalid_open,
                },
            }
        ],
    }


def _report(
    *,
    start_date: str,
    end_date: str,
    baseline_rets: list[float],
    candidate_rets: list[float],
) -> dict:
    return {
        "exchange": "BITGET",
        "start_date": start_date,
        "end_date": end_date,
        "genomes": [
            _genome("baseline.npy", baseline_rets),
            _genome("candidate.npy", candidate_rets),
        ],
    }


def test_walk_forward_gate_rejects_candidate_that_fails_any_fold():
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    fold_a = _report(
        start_date="2026-02-01",
        end_date="2026-03-31",
        baseline_rets=[0.20, 0.25],
        candidate_rets=[0.30, 0.35],
    )
    fold_b = _report(
        start_date="2026-04-01",
        end_date="2026-05-31",
        baseline_rets=[0.20, 0.25],
        candidate_rets=[-0.10, 0.05],
    )

    summary = tool.summarize_walk_forward_reports([fold_a, fold_b])

    assert summary["promotion_eligible"] is False
    assert summary["selected_is_baseline"] is True
    assert summary["selected_genome"] == "baseline.npy"
    candidate = summary["candidates"][0]
    assert candidate["path"] == "candidate.npy"
    assert candidate["promotion_eligible"] is False
    assert candidate["folds"][0]["promotion_eligible"] is True
    assert candidate["folds"][1]["promotion_eligible"] is False
    assert "oos_mean_ret" in candidate["folds"][1]["promotion_failures"]
    assert "oos_min_ret" in candidate["folds"][1]["promotion_failures"]
    assert candidate["mean_ret_delta_avg"] == pytest.approx(-0.075)


def test_cli_writes_source_report_metadata_and_selects_passing_candidate(tmp_path):
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    fold_a = _report(
        start_date="2026-02-01",
        end_date="2026-03-31",
        baseline_rets=[0.20, 0.25],
        candidate_rets=[0.30, 0.35],
    )
    fold_b = _report(
        start_date="2026-04-01",
        end_date="2026-05-31",
        baseline_rets=[0.20, 0.25],
        candidate_rets=[0.26, 0.31],
    )
    report_a = tmp_path / "fold_a.json"
    report_b = tmp_path / "fold_b.json"
    output = tmp_path / "walk_forward_summary.json"
    report_a.write_text(json.dumps(fold_a), encoding="utf-8")
    report_b.write_text(json.dumps(fold_b), encoding="utf-8")

    exit_code = tool.main(
        [
            "--report",
            str(report_a),
            "--report",
            str(report_b),
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["promotion_eligible"] is True
    assert payload["selected_is_baseline"] is False
    assert payload["selected_genome"] == "candidate.npy"
    assert payload["source_reports"] == [str(report_a), str(report_b)]


def test_walk_forward_gate_groups_different_fold_genomes_by_candidate_label():
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    fold_a = {
        "exchange": "BITGET",
        "start_date": "2026-02-01",
        "end_date": "2026-03-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_a_raw.npy", [0.30, 0.35]),
        ],
    }
    fold_b = {
        "exchange": "BITGET",
        "start_date": "2026-04-01",
        "end_date": "2026-05-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_b_raw.npy", [0.26, 0.31]),
        ],
    }

    summary = tool.summarize_walk_forward_reports(
        [fold_a, fold_b],
        candidate_labels=["position_state_raw"],
    )

    assert summary["promotion_eligible"] is True
    assert summary["selected_is_baseline"] is False
    assert summary["selected_strategy_id"] == "position_state_raw"
    assert summary["selected_genome"] == "fold_b_raw.npy"
    candidate = summary["candidates"][0]
    assert candidate["strategy_id"] == "position_state_raw"
    assert candidate["path"] == "position_state_raw"
    assert [fold["path"] for fold in candidate["folds"]] == [
        "fold_a_raw.npy",
        "fold_b_raw.npy",
    ]


def test_cli_accepts_candidate_labels_for_strategy_level_summary(tmp_path):
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    fold_a = {
        "exchange": "BITGET",
        "start_date": "2026-02-01",
        "end_date": "2026-03-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_a_raw.npy", [0.30, 0.35]),
        ],
    }
    fold_b = {
        "exchange": "BITGET",
        "start_date": "2026-04-01",
        "end_date": "2026-05-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_b_raw.npy", [0.26, 0.31]),
        ],
    }
    report_a = tmp_path / "fold_a.json"
    report_b = tmp_path / "fold_b.json"
    output = tmp_path / "walk_forward_summary.json"
    report_a.write_text(json.dumps(fold_a), encoding="utf-8")
    report_b.write_text(json.dumps(fold_b), encoding="utf-8")

    exit_code = tool.main(
        [
            "--report",
            str(report_a),
            "--report",
            str(report_b),
            "--candidate-label",
            "position_state_raw",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_strategy_id"] == "position_state_raw"
    assert payload["selected_genome"] == "fold_b_raw.npy"
    assert payload["candidates"][0]["strategy_id"] == "position_state_raw"


def test_cli_accepts_utf8_bom_report_files(tmp_path):
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    report = _report(
        start_date="2026-02-01",
        end_date="2026-03-31",
        baseline_rets=[0.20, 0.25],
        candidate_rets=[0.30, 0.35],
    )
    report_path = tmp_path / "bom_report.json"
    output = tmp_path / "summary.json"
    report_path.write_text(json.dumps(report), encoding="utf-8-sig")

    exit_code = tool.main(
        [
            "--report",
            str(report_path),
            "--candidate-label",
            "position_state_raw",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["promotion_eligible"] is True
    assert payload["selected_strategy_id"] == "position_state_raw"


def test_walk_forward_gate_selects_candidates_from_strategy_spec_per_fold():
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    fold_a = {
        "exchange": "BITGET",
        "start_date": "2025-12-01",
        "end_date": "2026-01-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_a_open_bias_0p75.npy", [0.32, 0.36]),
            _genome("fold_a_open_bias_0p95.npy", [-0.10, 0.04]),
        ],
    }
    fold_b = {
        "exchange": "BITGET",
        "start_date": "2026-02-01",
        "end_date": "2026-03-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_b_open_bias_0p75.npy", [0.29, 0.35]),
            _genome("fold_b_open_bias_0p95.npy", [0.27, 0.30]),
        ],
    }
    fold_c = {
        "exchange": "BITGET",
        "start_date": "2026-04-01",
        "end_date": "2026-05-31",
        "genomes": [
            _genome("baseline.npy", [0.20, 0.25]),
            _genome("fold_c_open_bias_0p75.npy", [0.205, 0.255]),
            _genome("fold_c_open_bias_0p95.npy", [0.26, 0.31]),
        ],
    }

    summary = tool.summarize_walk_forward_reports(
        [fold_a, fold_b, fold_c],
        strategy_specs=[
            "regime_adaptive_bias=*open_bias_0p75.npy,*open_bias_0p75.npy,*open_bias_0p95.npy"
        ],
        min_oos_mean_delta=0.005,
    )

    assert summary["promotion_eligible"] is True
    assert summary["selected_strategy_id"] == "regime_adaptive_bias"
    candidate = summary["candidates"][0]
    assert candidate["strategy_id"] == "regime_adaptive_bias"
    assert candidate["selection_contract"] == {
        "type": "fold_selectors",
        "selectors": [
            "*open_bias_0p75.npy",
            "*open_bias_0p75.npy",
            "*open_bias_0p95.npy",
        ],
    }
    assert [fold["path"] for fold in candidate["folds"]] == [
        "fold_a_open_bias_0p75.npy",
        "fold_b_open_bias_0p75.npy",
        "fold_c_open_bias_0p95.npy",
    ]


def test_cli_accepts_strategy_specs_for_per_fold_candidate_selection(tmp_path):
    tool = importlib.import_module("tools.run_genetics_walk_forward_matrix")
    reports = [
        {
            "exchange": "BITGET",
            "start_date": "2025-12-01",
            "end_date": "2026-01-31",
            "genomes": [
                _genome("baseline.npy", [0.20, 0.25]),
                _genome("fold_a_open_bias_0p75.npy", [0.32, 0.36]),
                _genome("fold_a_open_bias_0p95.npy", [-0.10, 0.04]),
            ],
        },
        {
            "exchange": "BITGET",
            "start_date": "2026-02-01",
            "end_date": "2026-03-31",
            "genomes": [
                _genome("baseline.npy", [0.20, 0.25]),
                _genome("fold_b_open_bias_0p75.npy", [0.29, 0.35]),
                _genome("fold_b_open_bias_0p95.npy", [0.27, 0.30]),
            ],
        },
        {
            "exchange": "BITGET",
            "start_date": "2026-04-01",
            "end_date": "2026-05-31",
            "genomes": [
                _genome("baseline.npy", [0.20, 0.25]),
                _genome("fold_c_open_bias_0p75.npy", [0.205, 0.255]),
                _genome("fold_c_open_bias_0p95.npy", [0.26, 0.31]),
            ],
        },
    ]
    report_paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"fold_{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)
    output = tmp_path / "summary.json"

    exit_code = tool.main(
        [
            "--report",
            str(report_paths[0]),
            "--report",
            str(report_paths[1]),
            "--report",
            str(report_paths[2]),
            "--strategy-spec",
            "regime_adaptive_bias=*open_bias_0p75.npy,*open_bias_0p75.npy,*open_bias_0p95.npy",
            "--min-oos-mean-delta",
            "0.005",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selected_strategy_id"] == "regime_adaptive_bias"
    assert payload["candidates"][0]["selection_contract"]["type"] == "fold_selectors"
