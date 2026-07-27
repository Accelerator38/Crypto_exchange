from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "evaluate_carryflow_forward_labels.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "evaluate_carryflow_forward_labels_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(*, reverse_root: str | None = None):
    rows = []
    for root_index, root in enumerate(("a", "b", "c")):
        for bar in range(1, 13):
            feature = float(bar - 6)
            target = feature * (5.0 if root != reverse_root else -5.0)
            rows.append(
                {
                    "root": root,
                    "bar": bar,
                    "symbol": "BTC",
                    "oi_change_bps": feature,
                    "price_return_bps": 0.0,
                    "funding_bps": 0.0,
                    "crowding_pct_points": 0.0,
                    "basis_bps": 0.0,
                    "market_positive_breadth": 0.5,
                    "volume_ratio_6": 1.0,
                    "realized_vol_6_bps": 0.0,
                    "trend_12_bps": 0.0,
                    "bar_range_bps": 10.0,
                    "close_location": 0.5,
                    "is_range_low_vol": 1.0,
                    "net_6h_bps": target + root_index * 0.01,
                }
            )
    return rows


def test_leave_one_root_out_accepts_stable_held_out_signal():
    tool = _load_tool()

    report = tool.evaluate_leave_one_root_out(_rows())

    assert report["passed"] is True
    assert report["aggregate"]["trades"] >= 10
    assert report["aggregate"]["mean_net_bps"] > 0.0
    assert report["aggregate"]["lcb_95_net_bps"] > 0.0
    assert report["root_expectancy_collapses"] == []


def test_leave_one_root_out_rejects_root_sign_reversal():
    tool = _load_tool()

    report = tool.evaluate_leave_one_root_out(_rows(reverse_root="c"))

    assert report["passed"] is False
    assert "c" in report["root_expectancy_collapses"]
    assert "root_expectancy_collapse:c" in report["failures"]


def test_forward_label_uses_conservative_stop_before_target():
    tool = _load_tool()
    samples = [
        {
            "symbols": {
                "BTC": {
                    "market": {
                        "high": 100.0,
                        "low": 100.0,
                        "decision_price": 100.0,
                    }
                }
            }
        }
        for _ in range(7)
    ]
    samples[1]["symbols"]["BTC"]["market"].update(
        {"high": 103.0, "low": 97.0, "decision_price": 99.0}
    )

    gross, reason = tool._forward_short_label(
        samples=samples,
        bar=1,
        symbol="BTC",
        entry_price=100.0,
        stop_pct=0.02,
        target_pct=0.02,
    )

    assert gross == -200.0
    assert reason == "stop_loss"


def test_dataset_ignores_old_actor_position_but_keeps_regime_guard():
    tool = _load_tool()
    samples = [
        {"symbols": {"BTC": {"complete": True}}}
        for _ in range(20)
    ]

    assert tool._row_eligible(
        bar=13,
        symbol="BTC",
        samples=samples,
        sample=samples[12],
        trace={
            "diagnostic_reason": "in_short_position",
            "regime": "range_low_vol",
        },
        required_history=4,
        allowed_regimes=("range_low_vol",),
    )
    assert not tool._row_eligible(
        bar=13,
        symbol="BTC",
        samples=samples,
        sample=samples[12],
        trace={"diagnostic_reason": "actor_hold", "regime": "bullish"},
        required_history=4,
        allowed_regimes=("range_low_vol",),
    )
