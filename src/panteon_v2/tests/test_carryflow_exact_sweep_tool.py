from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "sweep_carryflow_exact_segments.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "sweep_carryflow_exact_segments_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_sweep_aggregation_keeps_segments_separate_and_costed():
    tool = _load_tool()
    trades = [
        {
            "net_pnl_usd": 0.10,
            "gross_pnl_usd": 0.112,
            "costs_usd": 0.012,
            "notional_usd": 10.0,
        },
        {
            "net_pnl_usd": -0.02,
            "gross_pnl_usd": -0.008,
            "costs_usd": 0.012,
            "notional_usd": 10.0,
        },
    ]
    result = tool._aggregate(
        "test-profile",
        [
            {"candidate_signals": 1, "segment": 1},
            {"candidate_signals": 1, "segment": 2},
        ],
        trades,
    )

    assert result["candidate_signals"] == 2
    assert result["filled_orders"] == 4
    assert result["closed_trades"] == 2
    assert result["net_pnl_usd"] == 0.08
    assert result["mean_cost_bps"] == pytest.approx(12.0)
    assert result["screening_passed"] is False


def test_exact_sweep_uses_censored_end_and_one_profile_knob(tmp_path):
    tool = _load_tool()
    namespace = tool._run_namespace(
        tape=tmp_path / "tape.jsonl",
        seed=tmp_path / "seed.json",
        out_dir=tmp_path / "out",
        profile_id="divergence_short_v1",
    )

    assert namespace.no_flatten_end is True
    assert namespace.profile == "divergence_short_v1"
    assert not hasattr(namespace, "symbols")
    assert not hasattr(namespace, "stride_minutes")
    assert not hasattr(namespace, "oi_spike")
    assert not hasattr(namespace, "hold_bars")


def test_exact_sweep_hard_blocks_root_expectancy_collapse():
    tool = _load_tool()
    trades = [
        {
            "_segment": 1,
            "net_pnl_usd": 0.10,
            "gross_pnl_usd": 0.112,
            "costs_usd": 0.012,
            "notional_usd": 10.0,
            "close_reason": "max_holding",
        },
        {
            "_segment": 2,
            "net_pnl_usd": -0.02,
            "gross_pnl_usd": -0.008,
            "costs_usd": 0.012,
            "notional_usd": 10.0,
            "close_reason": "stop_loss",
        },
    ]
    result = tool._aggregate(
        "root-test",
        [
            {"candidate_signals": 1, "segment": 1},
            {"candidate_signals": 1, "segment": 2},
        ],
        trades,
    )

    assert result["active_roots"] == 2
    assert result["root_expectancy_collapses"] == [2]
    assert "root_expectancy_collapse:segment_2" in result["failures"]
    assert result["exit_reasons"] == {"max_holding": 1, "stop_loss": 1}


def test_exact_sweep_hard_blocks_drawdown_above_fixed_gate():
    tool = _load_tool()
    result = tool._aggregate(
        "drawdown-test",
        [{"candidate_signals": 2, "segment": 1}],
        [
            {
                "_segment": 1,
                "net_pnl_usd": 0.10,
                "gross_pnl_usd": 0.112,
                "costs_usd": 0.012,
                "notional_usd": 10.0,
            },
            {
                "_segment": 1,
                "net_pnl_usd": -0.25,
                "gross_pnl_usd": -0.238,
                "costs_usd": 0.012,
                "notional_usd": 10.0,
            },
        ],
    )

    assert result["max_drawdown_usd"] == pytest.approx(0.25)
    assert result["max_drawdown_limit_usd"] == 0.20
    assert "drawdown_above_limit" in result["failures"]
    assert result["evidence_extension_eligible"] is False


def test_exact_sweep_allows_no_order_extension_when_only_lcb_is_missing():
    tool = _load_tool()
    values = [0.20, -0.15, 0.01, 0.01, 0.01] * 2
    trades = []
    for index, value in enumerate(values):
        trades.append(
            {
                "_segment": 1 if index < 5 else 2,
                "net_pnl_usd": value,
                "gross_pnl_usd": value + 0.012,
                "costs_usd": 0.012,
                "notional_usd": 10.0,
                "close_reason": "max_holding",
            }
        )

    result = tool._aggregate(
        "lcb-extension-test",
        [
            {"candidate_signals": 5, "segment": 1},
            {"candidate_signals": 5, "segment": 2},
        ],
        trades,
    )

    assert result["filled_orders"] == 20
    assert result["closed_trades"] == 10
    assert result["mean_net_pnl_usd"] > 0.0
    assert result["root_expectancy_collapses"] == []
    assert result["failures"] == ["nonpositive_lcb"]
    assert result["screening_passed"] is False
    assert result["evidence_extension_eligible"] is True
