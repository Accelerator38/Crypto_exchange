from __future__ import annotations

import copy
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


SPEC_PATH = ROOT / "configs" / "exia_4h_sol_pullback_h12_replication_v1.json"
RUNNER_PATH = ROOT / "tools" / "run_exia_4h_sol_pullback_h12_replication_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exia_4h_sol_replication_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replication_spec_is_single_fixed_and_safe() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    runner = _load_runner()
    feasibility, event_family, entry_family = runner.validate_replication_spec(payload)

    assert payload["candidate"] == {
        "candidate_id": "SOL_PULLBACK_H12_REPLICATION",
        "source_combination_id": "BULL_TREND_FAST_RECLAIM__H12",
        "event_id": "BULL_TREND_FAST_RECLAIM",
        "symbol": "SOL/USDT",
        "direction": "LONG",
        "horizon_bars": 12,
    }
    assert feasibility["horizons_bars"] == [1, 3, 6, 12]
    assert len(event_family["events"]) == 4
    assert entry_family["timeframe_minutes"] == 240
    assert payload["evaluation_windows"] == ["validation", "oos", "sanity"]
    assert set(payload["safety"].values()) == {False}


def test_replication_rejects_candidate_mutation() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["candidate"]["horizon_bars"] = 6

    with pytest.raises(ValueError, match="selected development slice"):
        _load_runner().validate_replication_spec(changed)


def test_symbol_isolation_keeps_only_sol_events() -> None:
    runner = _load_runner()
    frame = pd.DataFrame(
        {
            "symbol": ["SOL/USDT", "BTC/USDT", "SOL/USDT", "ETH/USDT"],
            "timestamp": [0, 0, 1, 1],
        }
    )
    signal = pd.Series([1, 1, 0, 1], dtype="int8")
    isolated = runner.isolate_symbol_signal(frame, signal, symbol="SOL/USDT")

    assert isolated.tolist() == [1, 0, 0, 0]


def test_replication_gate_requires_coverage_and_all_positive_metrics() -> None:
    runner = _load_runner()
    gate = json.loads(SPEC_PATH.read_text(encoding="utf-8"))["stage_gate"]
    metric = {
        "observations": 20,
        "label_coverage": 0.94,
        "mean_stress_net_bps": 100.0,
        "median_stress_net_bps": 50.0,
        "stress_lcb_bps": 20.0,
        "familywise_stress_lcb_bps": 20.0,
        "mfe_surplus_lcb_bps": 30.0,
    }
    assert runner._gate_pass(metric, gate) is False
    metric["label_coverage"] = 0.95
    assert runner._gate_pass(metric, gate) is True
    for field in (
        "mean_stress_net_bps",
        "median_stress_net_bps",
        "stress_lcb_bps",
        "familywise_stress_lcb_bps",
        "mfe_surplus_lcb_bps",
    ):
        changed = dict(metric)
        changed[field] = 0.0
        assert runner._gate_pass(changed, gate) is False
