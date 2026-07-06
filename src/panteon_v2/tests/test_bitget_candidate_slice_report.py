from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "build_bitget_candidate_slice_report.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_bitget_candidate_slice_report",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_slices_blocks_range_low_vol_without_transition_confirmation():
    tool = _load_tool()
    summary = {
        "candidate_slices": {
            "LiveOIBreakout|BTC|range_low_vol|SHORT": {
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC",
                "regime": "range_low_vol",
                "direction": "SHORT",
                "filled_signals": 20,
                "closed_trades": 12,
                "expectancy_usd": 0.05,
                "lcb_usd": 0.03,
                "gross_loss": -0.01,
                "promotion_eligible": True,
                "fail_reasons": [],
            }
        }
    }

    report = tool.build_slice_report(summary)

    row = report["ranked_slices"][0]
    assert row["eligible_for_canary"] is False
    assert "range_low_vol_requires_transition_confirmation" in row["fail_reasons"]


def test_rank_slices_accepts_positive_trend_slice():
    tool = _load_tool()
    summary = {
        "candidate_slices": {
            "CarryFlowAgentV2|ETH|bearish|SHORT": {
                "actor_label": "CarryFlowAgentV2",
                "symbol": "ETH",
                "regime": "bearish",
                "direction": "SHORT",
                "filled_signals": 24,
                "closed_trades": 14,
                "expectancy_usd": 0.04,
                "lcb_usd": 0.02,
                "gross_loss": -0.08,
                "max_drawdown_usd": 0.12,
                "promotion_eligible": True,
                "fail_reasons": [],
            }
        }
    }

    report = tool.build_slice_report(
        summary,
        min_closed_trades=10,
        max_drawdown_usd=0.2,
    )

    assert report["ranked_slices"][0]["eligible_for_canary"] is True
    assert report["recommended_slices"][0]["key"] == "CarryFlowAgentV2|ETH|bearish|SHORT"
