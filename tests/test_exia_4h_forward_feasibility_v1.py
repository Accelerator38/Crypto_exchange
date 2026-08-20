from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simple_research.forward_labels import build_event_forward_labels


SPEC_PATH = ROOT / "configs" / "exia_4h_forward_feasibility_v1.json"
RUNNER_PATH = ROOT / "tools" / "run_exia_4h_forward_feasibility_v1.py"
BAR_MS = 14_400_000


def _load_runner():
    spec = importlib.util.spec_from_file_location("exia_4h_forward_feasibility_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame() -> pd.DataFrame:
    bars = [
        (90.0, 92.0, 88.0, 90.0),
        (95.0, 98.0, 93.0, 96.0),
        (100.0, 105.0, 95.0, 102.0),
        (110.0, 115.0, 108.0, 112.0),
        (120.0, 125.0, 117.0, 122.0),
        (130.0, 135.0, 128.0, 132.0),
        (140.0, 145.0, 138.0, 142.0),
    ]
    return pd.DataFrame(
        [
            {
                "timestamp": index * BAR_MS,
                "symbol": "BTC/USDT",
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": 1000.0,
            }
            for index, values in enumerate(bars)
        ]
    )


def test_forward_spec_is_sealed_bounded_and_safe() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    runner = _load_runner()
    event_family, entry_family = runner.validate_spec(payload)

    assert payload["horizons_bars"] == [1, 3, 6, 12]
    assert len(event_family["events"]) == 4
    assert entry_family["timeframe_minutes"] == 240
    assert set(payload["safety"].values()) == {False}
    assert payload["stage_gate"]["minimum_observations"] == 20


def test_forward_label_uses_next_open_terminal_open_costs_and_path_extremes() -> None:
    frame = _frame()
    signal = pd.Series([0, 1, 0, 0, 0, 0, 0], dtype="int8")
    labels = build_event_forward_labels(
        frame,
        signal,
        horizons_bars=[1, 3],
        start_timestamp=0,
        end_timestamp=7 * BAR_MS,
        base_cost_bps=8.0,
        stress_cost_bps=16.0,
    ).set_index("horizon_bars")

    one = labels.loc[1]
    assert one["event_timestamp"] == BAR_MS
    assert one["entry_timestamp"] == 2 * BAR_MS
    assert one["exit_timestamp"] == 3 * BAR_MS
    assert one["entry_price"] == 100.0
    assert one["exit_price"] == 110.0
    assert one["gross_bps"] == pytest.approx(1000.0)
    assert one["base_net_bps"] == pytest.approx(992.0)
    assert one["stress_net_bps"] == pytest.approx(984.0)
    assert one["mfe_bps"] == pytest.approx(500.0)
    assert one["mae_bps"] == pytest.approx(500.0)

    three = labels.loc[3]
    assert three["exit_timestamp"] == 5 * BAR_MS
    assert three["gross_bps"] == pytest.approx(3000.0)
    assert three["mfe_bps"] == pytest.approx(2500.0)
    assert three["mae_bps"] == pytest.approx(500.0)


def test_incomplete_or_cross_window_horizon_is_excluded() -> None:
    frame = _frame()
    signal = pd.Series([0, 1, 0, 0, 0, 0, 0], dtype="int8")
    labels = build_event_forward_labels(
        frame,
        signal,
        horizons_bars=[1, 3, 12],
        start_timestamp=0,
        end_timestamp=5 * BAR_MS,
        base_cost_bps=8.0,
        stress_cost_bps=16.0,
    )

    assert labels["horizon_bars"].tolist() == [1]


def test_completed_forward_label_has_no_dependency_after_terminal_open() -> None:
    frame = _frame()
    signal = pd.Series([0, 1, 0, 0, 0, 0, 0], dtype="int8")
    original = build_event_forward_labels(
        frame,
        signal,
        horizons_bars=[1],
        start_timestamp=0,
        end_timestamp=7 * BAR_MS,
        base_cost_bps=8.0,
        stress_cost_bps=16.0,
    )
    changed = frame.copy()
    changed.loc[changed["timestamp"] > 3 * BAR_MS, ["open", "high", "low", "close"]] *= 5.0
    mutated = build_event_forward_labels(
        changed,
        signal,
        horizons_bars=[1],
        start_timestamp=0,
        end_timestamp=7 * BAR_MS,
        base_cost_bps=8.0,
        stress_cost_bps=16.0,
    )

    pd.testing.assert_frame_equal(original, mutated)


def test_forward_gate_requires_positive_distribution_and_excludes_control() -> None:
    runner = _load_runner()
    gate = json.loads(SPEC_PATH.read_text(encoding="utf-8"))["stage_gate"]
    metric = {
        "comparison_only": False,
        "observations": 30,
        "mean_stress_net_bps": 20.0,
        "median_stress_net_bps": -1.0,
        "stress_lcb_bps": 5.0,
        "familywise_stress_lcb_bps": 2.0,
    }
    assert runner._gate_pass(metric, gate) is False
    metric["median_stress_net_bps"] = 1.0
    assert runner._gate_pass(metric, gate) is True
    metric["comparison_only"] = True
    assert runner._gate_pass(metric, gate) is False
