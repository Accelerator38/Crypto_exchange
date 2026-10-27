"""Small direct regressions: no pytest, collector, files with market data or network."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from exia.genetic_primus.entry_only_overlay_v1 import entry_only_targets


def test_hold_and_delayed_entry():
    ema = np.array([[1], [1], [1], [0], [1], [1]])
    gate = np.array([[True], [False], [False], [True], [False], [True]])
    assert entry_only_targets(ema, gate).ravel().tolist() == [1, 1, 1, 0, 0, 1]


def test_reversal_requires_fresh_permission():
    ema = np.array([[1], [-1], [-1], [0], [-1]])
    gate = np.array([[True], [False], [True], [False], [True]])
    assert entry_only_targets(ema, gate).ravel().tolist() == [1, 0, -1, 0, -1]


def test_symbol_independence_reset_and_extremes():
    ema = np.array([[1, -1], [1, -1], [1, -1]])
    gate = np.array([[True, False], [False, True], [False, False]])
    assert entry_only_targets(ema, gate).tolist() == [[1, 0], [1, -1], [1, -1]]
    assert entry_only_targets(ema, gate, reset_before=np.array([False, False, True]))[-1].tolist() == [0, 0]
    np.testing.assert_array_equal(entry_only_targets(ema, np.ones_like(gate)), ema)
    assert not entry_only_targets(ema, np.zeros_like(gate)).any()


def test_prefix_causality_and_bad_inputs():
    rng = np.random.default_rng(812)
    ema = rng.choice([-1, 0, 1], size=(20, 3))
    gate = rng.random((20, 3)) > .5
    full = entry_only_targets(ema, gate)
    for n in (1, 7, 19):
        np.testing.assert_array_equal(entry_only_targets(ema[:n], gate[:n]), full[:n])
    for bad_ema, bad_gate in ((np.array([[np.nan]]), np.array([[True]])),
                              (np.array([[2]]), np.array([[True]])),
                              (np.array([[1]]), np.array([[1]]))):
        try:
            entry_only_targets(bad_ema, bad_gate)
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


def fixture_window(invalid_middle=False):
    from tools import run_genetic_primus_entry_only_ablation_v1 as r
    dt = 14_400_000
    values = np.array([[1., 1., 2., 0.], [1., 1., 0., 0.], [1., 1., 0., 0.]])
    if invalid_middle:
        values[1, 0] = np.nan
    bars = pd.DataFrame({"timestamp": [0, dt, 2 * dt], "symbol": ["X"] * 3,
                         "open": [100., 110., 120.], "close": [110., 120., 130.]})
    schedule = pd.DataFrame({"execution_timestamp": [0, dt, 2 * dt],
                             "signal_timestamp": [0, dt, 2 * dt], "symbol": ["X"] * 3})
    window = r.cold.PreparedWindow(bars, schedule, values, np.ones(3, dtype=np.int8), ("X",), 0, 3 * dt, dt)
    scaler = r.warm.LabScaler((0.,) * 4, (1.,) * 4, 3, "synthetic")
    return r, window, scaler


def test_accounting_terminal_and_policy_model_identity():
    r, window, scaler = fixture_window()
    sim, model = r.simulate_entry(window, scaler, 16.)
    old = window.simulate(r.cold.FIXED, scaler, 16., details=True)
    assert sim.decisions.desired_position.tolist() == [1, 1, 1]
    assert old.decisions.desired_position.tolist() == [1, 0, 0]
    assert sim.metrics["net_return"] > old.metrics["net_return"]
    assert sim.trades.exit_timestamp.tolist() == [window.end]
    assert sim.trades.exit_reason.tolist() == ["interval_end"]
    assert r.warm.object_sha(model) != r.warm.object_sha(window.model(r.cold.FIXED, scaler))
    assert all(sim.decisions.model_sha256 == r.warm.object_sha(model))


def test_missing_features_force_flat_and_reset_rechecks_gate():
    r, window, scaler = fixture_window(invalid_middle=True)
    schedule, _ = r.entry_schedule(window, scaler)
    assert schedule.target.tolist() == [1, 0, 0]
    r, window, scaler = fixture_window()
    sim, _ = r.simulate_entry(window, scaler, 16., reset_times=(window.timeframe_ms,))
    assert sim.decisions.desired_position.tolist() == [1, 0, 0]
    assert sim.trades.exit_reason.tolist() == ["retrain_forced_exit"]
