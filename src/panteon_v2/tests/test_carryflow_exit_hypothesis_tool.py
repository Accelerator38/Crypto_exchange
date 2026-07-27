from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "evaluate_carryflow_exit_hypothesis.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "evaluate_carryflow_exit_hypothesis_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(candidate_values, baseline_values):
    details = []
    for bar, (candidate, baseline) in enumerate(
        zip(candidate_values, baseline_values),
        start=1,
    ):
        details.append(
            {
                "policy_outcome": "ALLOW_OPEN",
                "symbol": "BTC",
                "bar": bar,
                "feature_value": float(bar),
                "edge_components": {
                    "price_divergence": float(bar) / 100.0,
                    "oi_excess": float(bar) / 1000.0,
                },
                "horizons": {
                    "4": {"net_bps": candidate, "gross_bps": candidate + 12.0},
                    "6": {"net_bps": baseline, "gross_bps": baseline + 12.0},
                },
            }
        )
    return {"signal_calibration": {"candidate_details": details}}


def test_exit_hypothesis_rejects_independent_root_collapse():
    tool = _load_tool()
    report = tool.evaluate_exit_hypothesis(
        {
            "selection": _report([20.0], [-5.0]),
            "evaluation_a": _report([10.0, 5.0], [2.0, 3.0]),
            "evaluation_b": _report([-8.0], [1.0]),
        },
        selection_root="selection",
    )

    assert report["verdict"] == "rejected"
    assert report["passed"] is False
    assert report["profile_registration_allowed"] is False
    assert report["independent_evaluation"][
        "candidate_root_collapses"
    ] == ["evaluation_b"]
    assert report["orders_enabled"] is False
    assert report["expected_move_model_registration_allowed"] is False


def test_exit_hypothesis_requires_root_robust_baseline_improvement():
    tool = _load_tool()
    report = tool.evaluate_exit_hypothesis(
        {
            "selection": _report([20.0], [-5.0]),
            "evaluation_a": _report([10.0], [2.0]),
            "evaluation_b": _report([8.0], [3.0]),
        },
        selection_root="selection",
    )

    assert report["verdict"] == "accepted_for_new_prospective_root"
    assert report["passed"] is True
    assert report["profile_registration_allowed"] is True
    assert report["promotion_authority"] is False
