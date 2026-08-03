from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simple_research.confirmation import (
    aggregate_complete_minutes_to_hour,
    apply_regime_allowlist,
    apply_regime_entry_gate,
    moving_block_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]


def _minute_frame(rows: int = 60) -> pd.DataFrame:
    close = np.arange(rows, dtype="float64") + 100.0
    return pd.DataFrame(
        {
            "timestamp": np.arange(rows, dtype="int64") * 60_000,
            "symbol": "BTC/USDT",
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close + 1.0,
            "volume": 3.0,
        }
    )


def test_minute_aggregation_requires_complete_hours() -> None:
    hourly = aggregate_complete_minutes_to_hour(_minute_frame())

    assert len(hourly) == 1
    assert hourly.loc[0, "timestamp"] == 0
    assert hourly.loc[0, "open"] == 100.0
    assert hourly.loc[0, "high"] == 161.0
    assert hourly.loc[0, "low"] == 98.0
    assert hourly.loc[0, "close"] == 160.0
    assert hourly.loc[0, "volume"] == 180.0

    with pytest.raises(ValueError, match="incomplete hour"):
        aggregate_complete_minutes_to_hour(_minute_frame(59))


def test_regime_allowlist_masks_without_looking_forward() -> None:
    frame = pd.DataFrame({"timestamp": [0, 3_600_000, 7_200_000]})
    signal = pd.Series([1, -1, 1], dtype="int8")
    lookup = {0: "neutral", 3_600_000: "bullish", 7_200_000: "neutral"}

    masked, regimes = apply_regime_allowlist(
        frame,
        signal,
        lookup,
        allowed_regimes=("neutral",),
    )

    assert masked.tolist() == [1, 0, 1]
    assert regimes.tolist() == ["neutral", "bullish", "neutral"]


def test_regime_entry_gate_keeps_admitted_position_until_base_exit() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [0, 3_600_000, 7_200_000, 10_800_000],
            "symbol": ["BTC/USDT"] * 4,
        }
    )
    signal = pd.Series([1, 1, -1, -1], dtype="int8")
    lookup = {
        0: "neutral",
        3_600_000: "bullish",
        7_200_000: "bullish",
        10_800_000: "neutral",
    }

    gated, _ = apply_regime_entry_gate(
        frame,
        signal,
        lookup,
        allowed_entry_regimes=("neutral",),
    )

    assert gated.tolist() == [1, 1, 0, -1]


def test_block_bootstrap_is_deterministic() -> None:
    returns = pd.Series(np.tile([0.001, -0.0005], 48))
    first = moving_block_bootstrap(
        returns,
        block_size=24,
        replicates=200,
        seed=7,
        confidence=0.95,
    )
    second = moving_block_bootstrap(
        returns,
        block_size=24,
        replicates=200,
        seed=7,
        confidence=0.95,
    )

    assert first == second
    assert first["observations"] == 96


def test_confirmation_contract_is_fail_closed() -> None:
    path = (
        ROOT
        / "configs"
        / "simple_research"
        / "ema_neutral_confirmation_v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert contract["candidate_id"] == "ema_trend_12_48_neutral_v1"
    assert contract["candidate"]["allowed_market_regimes"] == ["neutral"]
    assert (
        contract["confirmation_dataset"][
            "minimum_calendar_days_for_confirmation"
        ]
        == 30
    )
    assert contract["orders_enabled"] is False
    assert contract["paper_allowed"] is False
    assert contract["live_allowed"] is False
    assert contract["promotion_authority"] is False

    diagnostic_path = (
        ROOT
        / "configs"
        / "simple_research"
        / "ema_neutral_entry_diagnostic_v1.json"
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["independent_confirmation_authority"] is False
    assert diagnostic["orders_enabled"] is False
    assert diagnostic["paper_allowed"] is False
    assert diagnostic["live_allowed"] is False


def test_confirmation_reports_are_terminal_and_fail_closed() -> None:
    reports = {
        "ema_neutral_confirmation_v1": "FAILED_SEALED_CONFIRMATION_EARLY",
        "ema_trend_12_48_neutral_entry_v1": "FAILED_DIAGNOSTIC_REPLAY",
    }
    for directory, expected_status in reports.items():
        path = (
            ROOT
            / "Reports"
            / "SimpleResearch"
            / directory
            / "confirmation_report.json"
        )
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == expected_status
        assert report["passed"] is False
        assert report["orders_enabled"] is False
        assert report["paper_allowed"] is False
        assert report["live_allowed"] is False
        assert report["promotion_authority"] is False
