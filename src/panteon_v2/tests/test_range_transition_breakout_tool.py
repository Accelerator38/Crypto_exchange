from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "evaluate_range_transition_breakout.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "evaluate_range_transition_breakout_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(price, *, volume=100.0, oi=100.0, high=None, low=None):
    return {
        "observed_at": "2026-01-01T00:00:00Z",
        "symbols": {
            "BTC": {
                "complete": True,
                "market": {
                    "decision_price": price,
                    "high": price if high is None else high,
                    "low": price if low is None else low,
                    "volume": volume,
                },
                "derivatives": {"open_interest_usdt": oi},
            }
        },
    }


def test_detects_confirmed_short_breakout_and_costs_label():
    tool = _load_tool()
    samples = [_sample(100.0, oi=100.0 + index * 0.1) for index in range(20)]
    samples[12] = _sample(99.0, volume=200.0, oi=103.0, high=99.2, low=98.8)
    for index in range(13, 19):
        samples[index] = _sample(98.0, oi=103.0)
    regimes = {
        (bar, "BTC"): "range_low_vol" for bar in range(1, 21)
    }

    result = tool.detect_transition_events(
        root_label="root",
        samples=samples,
        symbols=("BTC",),
        regimes_by_symbol_bar=regimes,
        mean_cost_bps=12.0,
    )

    assert len(result["selected_events"]) == 1
    event = result["selected_events"][0]
    assert event["direction"] == "SHORT"
    assert event["net_bps"] > 0.0


def test_breakout_requires_volume_and_oi_confirmation():
    tool = _load_tool()
    samples = [_sample(100.0) for _ in range(20)]
    samples[12] = _sample(99.0, volume=110.0, oi=100.5)
    regimes = {
        (bar, "BTC"): "range_low_vol" for bar in range(1, 21)
    }

    result = tool.detect_transition_events(
        root_label="root",
        samples=samples,
        symbols=("BTC",),
        regimes_by_symbol_bar=regimes,
        mean_cost_bps=12.0,
    )

    assert result["raw_events"] == []
    assert result["selected_events"] == []


def test_reverse_directional_to_range_transition_is_not_an_entry():
    tool = _load_tool()
    samples = [_sample(100.0, oi=100.0 + index * 0.1) for index in range(20)]
    samples[12] = _sample(99.0, volume=200.0, oi=103.0)
    regimes = {
        (bar, "BTC"): "range_low_vol" for bar in range(1, 21)
    }
    regimes[(12, "BTC")] = "bearish"

    result = tool.detect_transition_events(
        root_label="root",
        samples=samples,
        symbols=("BTC",),
        regimes_by_symbol_bar=regimes,
        mean_cost_bps=12.0,
    )

    assert result["selected_events"] == []


def test_evaluation_rejects_inactive_and_negative_roots():
    tool = _load_tool()
    evaluation = tool.evaluate_roots(
        [
            {
                "root": "a",
                "selected_events": [
                    {"root": "a", "direction": "LONG", "net_bps": 20.0}
                ],
            },
            {
                "root": "b",
                "selected_events": [
                    {"root": "b", "direction": "SHORT", "net_bps": -10.0}
                ],
            },
            {"root": "c", "selected_events": []},
        ]
    )

    assert evaluation["passed"] is False
    assert "b" in evaluation["root_collapses"]
    assert "c" in evaluation["inactive_roots"]
    assert "inactive_root:c" in evaluation["failures"]
    assert "direction_expectancy_collapse:SHORT" in evaluation["failures"]
