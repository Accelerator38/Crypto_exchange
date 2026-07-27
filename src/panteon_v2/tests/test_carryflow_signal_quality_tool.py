from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "analyze_carryflow_signal_quality.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "analyze_carryflow_signal_quality_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTape:
    def __init__(self, samples):
        self.samples = samples
        self.symbols = ("BNB",)
        self.collector_run_id = "quality-test-root"
        self.file_sha256 = "a" * 64
        self.head_sha256 = "b" * 64

    def describe(self):
        return {"complete_samples": len(self.samples)}


def _samples():
    prices = [100.0, 100.2, 100.1, 100.0, 99.8, 99.7, 99.6, 99.7, 99.6, 99.5]
    oi_values = [100.0, 101.0, 102.0, 104.0, 104.5, 105.0, 105.5, 106.0, 106.5, 107.0]
    rows = []
    for index, (price, oi_value) in enumerate(zip(prices, oi_values), start=1):
        rows.append(
            {
                "symbols": {
                    "BNB": {
                        "derivatives": {
                            "open_interest_usdt": oi_value,
                            "funding_rate": 0.0001,
                            "long_ratio": 0.70,
                            "mark_price": price * 1.0005,
                            "index_price": price,
                        },
                        "market": {
                            "decision_price": price,
                            "open": price + 0.05,
                            "high": price + 0.10,
                            "low": price - 0.10,
                            "close": price,
                        },
                    }
                },
                "bar_close_timestamp_ms": index * 3_600_000,
            }
        )
    return rows


def test_signal_quality_funnel_and_exit_diagnostics_are_read_only():
    tool = _load_tool()
    samples = _samples()
    profile = tool.get_carryflow_profile("divergence_short_systemic_guard_v1")
    derived = tool._derive_observations(
        samples=samples,
        symbols=("BNB",),
        profile=profile,
    )
    feature = sum(tool._edge_components(derived[(4, "BNB")], profile).values())
    trace = []
    for bar in range(1, 11):
        if bar < 4:
            reason = "oi_below_threshold"
            outcome = "actor_hold"
            feature_value = None
        elif bar == 4:
            reason = "candidate_short"
            outcome = "candidate_created"
            feature_value = feature
        elif bar < 10:
            reason = "in_short_position"
            outcome = "actor_hold"
            feature_value = None
        else:
            reason = "close_short"
            outcome = "exit_intent_created"
            feature_value = None
        trace.append(
            {
                "bar": bar,
                "execution_symbol": "BNB",
                "diagnostic_reason": reason,
                "regime": "range_low_vol",
                "feature_value": feature_value,
                "outcome": outcome,
            }
        )
    decisions = [
        {
            "bar": 4,
            "symbol": "BNB",
            "outcome": "ALLOW_OPEN",
            "primary_reason": "allowed",
        }
    ]
    trades = [
        {
            "opened_bar": 4,
            "closed_bar": 10,
            "symbol": "BNB",
            "notional_usd": 10.0,
            "gross_pnl_usd": 0.05,
            "net_pnl_usd": 0.038,
            "close_reason": "max_holding",
        }
    ]
    summary = {
        "research_hypothesis": {
            "profile_id": "divergence_short_systemic_guard_v1"
        },
        "mean_cost_bps": 12.0,
        "manifest_sha256": "c" * 64,
    }

    report = tool.analyze_signal_quality(
        tape=_FakeTape(samples),
        activation_trace=trace,
        policy_decisions=decisions,
        closed_trades=trades,
        replay_summary=summary,
    )

    assert report["research_only"] is True
    assert report["promotion_authority"] is False
    assert report["orders_enabled"] is False
    assert report["strategy_changed"] is False
    assert report["activation_funnel"]["oi_below_threshold"] == 3
    assert report["activation_funnel"]["raw_actor_candidates"] == 1
    assert report["activation_funnel"]["closed_trades"] == 1
    assert report["exit_analysis"]["close_reasons"] == {"max_holding": 1}
    assert report["signal_calibration"]["mean_realized_gross_bps"] == pytest.approx(
        50.0
    )


def test_horizon_aggregation_keeps_missing_endpoints_visible():
    tool = _load_tool()
    rows = [
        {
            "horizons": {
                "1": {"net_bps": 10.0},
                "2": {"net_bps": -5.0},
            }
        },
        {"horizons": {"1": {"net_bps": -2.0}}},
    ]

    result = tool._aggregate_horizons(rows)

    assert result["horizons"]["1"]["observations"] == 2
    assert result["horizons"]["1"]["mean_net_bps"] == pytest.approx(4.0)
    assert result["horizons"]["2"]["observations"] == 1
    assert result["horizons"]["2"]["mean_net_bps"] == pytest.approx(-5.0)
    assert result["horizons"]["12"]["observations"] == 0
    assert result["horizons"]["12"]["mean_net_bps"] is None
