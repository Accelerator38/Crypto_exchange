from __future__ import annotations

import pytest

from panteon_v2.analysis.portfolio_failure_decomposition import (
    PortfolioFailureDecompositionError,
    _drawdown_interval,
    _linear_exposure,
    verify_development_replay_parity,
)


def test_replay_parity_ignores_only_runtime_metadata_and_raw_ledgers():
    sealed = {
        "generated_at": "first",
        "candidate_id": "candidate",
        "candidate": {"metrics": {"mean": 1.0}},
    }
    replay = {
        "generated_at": "second",
        "candidate_id": "candidate",
        "candidate": {
            "metrics": {"mean": 1.0},
            "trades": [{"net_bps": 1.0}],
            "leg_trades": [{"net_bps": 1.0}],
        },
    }

    result = verify_development_replay_parity(sealed, replay)

    assert result["passed"] is True
    assert (
        result["sealed_projection_sha256"]
        == result["replay_projection_sha256"]
    )


def test_replay_parity_rejects_metric_drift():
    sealed = {"candidate": {"metrics": {"mean": 1.0}}}
    replay = {"candidate": {"metrics": {"mean": 2.0}}}

    with pytest.raises(
        PortfolioFailureDecompositionError,
        match="parity mismatch",
    ):
        verify_development_replay_parity(sealed, replay)


def test_linear_exposure_and_drawdown_interval_are_deterministic():
    exposure = _linear_exposure(
        [10.0, 20.0, 30.0],
        [20.0, 40.0, 60.0],
    )
    trades = [
        {
            "net_pnl_usd": 1.0,
            "entry_timestamp_ms": 1,
            "exit_timestamp_ms": 2,
        },
        {
            "net_pnl_usd": -0.5,
            "entry_timestamp_ms": 3,
            "exit_timestamp_ms": 4,
        },
        {
            "net_pnl_usd": -1.0,
            "entry_timestamp_ms": 5,
            "exit_timestamp_ms": 6,
        },
        {
            "net_pnl_usd": 0.25,
            "entry_timestamp_ms": 7,
            "exit_timestamp_ms": 8,
        },
    ]

    drawdown = _drawdown_interval(trades)

    assert exposure["beta"] == pytest.approx(2.0)
    assert exposure["correlation"] == pytest.approx(1.0)
    assert exposure["alpha_intercept_bps"] == pytest.approx(0.0)
    assert drawdown["max_drawdown_usd"] == pytest.approx(1.5)
    assert drawdown["start_trade_index"] == 1
    assert drawdown["end_trade_index"] == 2
    assert drawdown["interval_portfolios"] == 2
