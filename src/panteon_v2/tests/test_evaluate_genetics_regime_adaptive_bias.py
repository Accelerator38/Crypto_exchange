from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("tools.evaluate_genetics_contract")


def test_regime_adaptive_output_bias_uses_period_regime(monkeypatch):
    tool = _load_tool()
    cg = tool.cg
    source = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
    open_idx = int(tool._open_output_bias_indices()[0])
    precomp = [
        (None, None, ["BTC"], None, "2025-12", None, "bearish"),
        (None, None, ["BTC"], None, "2026-04", None, "neutral"),
    ]

    def fake_evaluate_population(population, period_precomp):
        bias = float(-np.asarray(population)[0, open_idx])
        return np.asarray([bias], dtype=np.float64), [[bias]]

    def fake_contract_metrics_for_genome(*, genome, precomp, execution_lag_bars):
        bias = float(-np.asarray(genome)[open_idx])
        period = str(precomp[0][4])
        regime = tool._contract_regime_label(precomp[0])
        return {
            "execution_lag_bars": execution_lag_bars,
            "turnover_target_rate": 0.10,
            "mean_turnover_rate": bias / 100.0,
            "max_turnover_rate": bias / 100.0,
            "mean_effective_turnover_rate": bias / 100.0,
            "mean_saturation_rate": bias / 200.0,
            "max_saturation_rate": bias / 200.0,
            "mean_invalid_open_logit_pressure": bias / 300.0,
            "max_invalid_open_logit_pressure": bias / 300.0,
            "periods": [
                {
                    "period": period,
                    "regime": regime,
                    "turnover_rate": bias / 100.0,
                    "effective_turnover_rate": bias / 100.0,
                    "saturation_rate": bias / 200.0,
                    "invalid_open_logit_pressure": bias / 300.0,
                }
            ],
        }

    monkeypatch.setattr(tool, "_evaluate_population", fake_evaluate_population)
    monkeypatch.setattr(tool, "_contract_metrics_for_genome", fake_contract_metrics_for_genome)

    report = tool._evaluate_regime_adaptive_output_bias_mode(
        source_genome=source,
        precomp=precomp,
        regime_open_bias={"bearish": 0.75, "neutral": 0.95},
        mode_name="fee_fixed_nextbar",
        execution_lag_bars=1,
        futures_fee=0.0004,
        position_state_features_enabled=True,
    )

    assert report["regime_adaptive_output_bias"]["enabled"] is True
    assert report["period_rets"] == pytest.approx([0.75, 0.95])
    periods = report["regime_adaptive_output_bias"]["periods"]
    assert [item["regime"] for item in periods] == ["bearish", "neutral"]
    assert [item["open_output_bias"] for item in periods] == [0.75, 0.95]
    assert report["contract_metrics"]["max_turnover_rate"] == pytest.approx(0.0095)
