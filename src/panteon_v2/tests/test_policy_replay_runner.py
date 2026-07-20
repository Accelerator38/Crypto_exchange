from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from panteon_v2.app.agent_bootstrap import _ensure_paths


_ensure_paths()

import panteon_agents
from panteon_agents import CarryFlowAgentV2
from panteon_v2.domain.types import MarketSnapshot, Regime
from panteon_v2.policy import (
    CarryFlowAdapterError,
    DerivativesContextRecord,
    HistoricalDerivativesContext,
    PolicyReplayRunner,
    PolicyTarget,
    load_policy_manifest,
    seal_manifest_payload,
)
from panteon_v2.policy.carryflow_adapter import CARRYFLOW_CONFIG_FIELDS
from panteon_v2.shadow.adapters import V1AgentAdapter


NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _FundingFeed:
    def __init__(self) -> None:
        self.oi = 100.0

    def get(self, _symbol):
        self.oi *= 1.05
        return {
            "funding_rate": 0.001,
            "long_ratio": 0.70,
            "short_ratio": 0.30,
            "mark_price": 101.0,
            "index_price": 100.0,
            "open_interest_usdt": self.oi,
        }


def _actor_config() -> dict[str, object]:
    actor = CarryFlowAgentV2()
    config = {name: getattr(actor, name) for name in CARRYFLOW_CONFIG_FIELDS}
    config.update(
        {
            "CHECK_INT": 1,
            "HOLD": 2,
            "STOP": 0.10,
            "TARGET": 0.10,
            "EMA_FAST": 2,
            "EMA_SLOW": 6,
            "RSI_N": 2,
            "RSI_OB": 0,
            "RSI_OS": 100,
            "EXTREME_EXT": 0.0,
            "MAX_POS": 1,
            "ENTRY_COOLDOWN": 0,
            "ALLOW_LONG": False,
            "ALLOW_SHORT": True,
        }
    )
    return config


def _manifest_payload(*, regime: str = "neutral") -> dict[str, object]:
    return {
        "schema_version": "panteon.policy.v1",
        "policy_id": "carryflow-btc-replay-v1",
        "target": "replay",
        "exchange": "BITGET",
        "actor": "CarryFlowAgentV2",
        "actor_config": _actor_config(),
        "signal_model": {
            "feature": "diagnostic.edge",
            "intercept_bps": 0.0,
            "slope_bps_per_unit": 2.0,
            "lcb_haircut_bps": 2.0,
            "min_feature_value": 0.1,
            "max_expected_move_bps": 100.0,
        },
        "data": {
            "bar_interval_seconds": 60,
            "cadence_tolerance_seconds": 0,
            "max_bar_close_lag_seconds": 30,
            "max_derivatives_age_seconds": 1200,
            "required_context_coverage_pct": 95.0,
        },
        "created_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(days=14)).isoformat(),
        "source_revision": "d863585",
        "runtime_fingerprint_sha256": "a" * 64,
        "costs": {
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 4.0,
        },
        "risk": {
            "capital_fraction": 0.01,
            "max_notional_usd": 10.0,
            "max_open_positions": 1,
            "max_daily_loss_usd": 5.0,
            "stop_loss_pct": 10.0,
            "max_holding_minutes": 180,
            "max_signal_age_seconds": 120,
        },
        "rules": [
            {
                "symbol": "BTC",
                "regime": regime,
                "direction": "SHORT",
                "min_expected_move_bps": 16.0,
                "max_spread_bps": 5.0,
                "max_slippage_bps": 5.0,
                "min_regime_confidence": 0.5,
                "risk_mult": 1.0,
            }
        ],
        "evidence": [],
    }


def _loaded_manifest(tmp_path, *, regime: str = "neutral"):
    payload = seal_manifest_payload(_manifest_payload(regime=regime))
    path = tmp_path / "replay_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_policy_manifest(
        path,
        project_root=tmp_path,
        expected_sha256=payload["manifest_sha256"],
        now=NOW,
        verify_runtime_fingerprint=False,
        required_target=PolicyTarget.REPLAY,
        required_exchange="BITGET",
    )


def _snapshots(
    count: int = 30,
    *,
    global_confidence: float = 0.9,
    local_confidence: float | None = None,
):
    return [
        MarketSnapshot(
            bar=index,
            timestamp=NOW + timedelta(minutes=index),
            regime=Regime.NEUTRAL,
            regime_confidence=global_confidence,
            prices={"BTC/USDT": 100.0},
            volumes={"BTC/USDT": 1000.0},
            bar_opens={"BTC/USDT": 100.0},
            bar_highs={"BTC/USDT": 100.0},
            bar_lows={"BTC/USDT": 100.0},
            bar_closes={"BTC/USDT": 100.0},
            regimes_by_symbol={"BTC/USDT": Regime.NEUTRAL},
            regime_features_by_symbol=(
                {
                    "BTC/USDT": {
                        "regime_confidence": local_confidence,
                    }
                }
                if local_confidence is not None
                else {}
            ),
        )
        for index in range(1, count + 1)
    ]


def _agent():
    return V1AgentAdapter(label="CarryFlowAgentV2", v1_agent=CarryFlowAgentV2())


def test_exact_replay_routes_real_carryflow_through_policy_and_trade_executor(tmp_path):
    panteon_agents.set_fetcher(_FundingFeed())
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path),
            agent=_agent(),
            initial_capital_usd=1000.0,
            assumed_spread_bps=2.0,
        )

        report = runner.run(_snapshots(), flatten_end=True)
    finally:
        panteon_agents.set_fetcher(None)

    assert report.target == "replay"
    assert report.actor_open_actions > 0
    assert report.candidate_signals == report.actor_open_actions
    assert report.policy_allowed == report.candidate_signals
    assert report.open_execution_attempts == report.policy_allowed
    assert report.open_fills > 0
    assert report.closed_trades == report.open_fills
    assert report.remaining_open_positions == 0
    assert report.parity_passed is True
    assert report.derivatives_context_coverage_pct == 100.0
    assert report.mean_cost_bps == pytest.approx(12.0, abs=0.05)
    assert report.net_pnl_usd < 0.0
    assert report.evidence_eligible is False
    assert report.total_costs_usd > 0.0


def test_exact_replay_can_right_censor_open_position_at_segment_end(tmp_path):
    panteon_agents.set_fetcher(_FundingFeed())
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path),
            agent=_agent(),
        )
        report = runner.run(_snapshots(count=18), flatten_end=False)
    finally:
        panteon_agents.set_fetcher(None)

    assert report.closed_trades == 2
    assert report.remaining_open_positions == 1
    assert report.open_fills == report.closed_trades + 1
    assert all(
        outcome.close_reason != "replay_end_flatten"
        for outcome in runner.trade_outcomes
    )
    assert report.parity_passed is True
    assert "end_positions_censored" in report.evidence_failures


def test_exact_replay_advances_point_in_time_derivatives_context(tmp_path):
    records = [
        DerivativesContextRecord(
            timestamp=NOW + timedelta(minutes=index),
            symbol="BTC",
            funding_rate=0.001,
            open_interest_usdt=100.0 * (1.05 ** index),
            long_ratio=0.70,
            short_ratio=0.30,
            mark_price=101.0,
            index_price=100.0,
            last_price=100.0,
            next_funding_ts=0,
            long_short_ratio_ts=0,
            source_updated_ts=0.0,
            context_complete=True,
        )
        for index in range(1, 31)
    ]
    context = HistoricalDerivativesContext(records)
    panteon_agents.set_fetcher(context)
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path),
            agent=_agent(),
            derivatives_context_provider=context,
        )
        report = runner.run(_snapshots(), flatten_end=True)
    finally:
        panteon_agents.set_fetcher(None)

    assert report.actor_open_actions > 0
    assert report.actor_opens_blocked_missing_derivatives_context == 0
    assert report.candidate_signals_with_derivatives_context == report.candidate_signals
    assert context.get("BTC")["age_sec"] == 0.0
    assert report.derivatives_context_coverage_pct == 100.0


def test_exact_replay_uses_symbol_local_regime_confidence(tmp_path):
    panteon_agents.set_fetcher(_FundingFeed())
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path),
            agent=_agent(),
        )
        report = runner.run(
            _snapshots(global_confidence=0.1, local_confidence=0.9),
            flatten_end=True,
        )
    finally:
        panteon_agents.set_fetcher(None)

    assert report.candidate_signals > 0
    assert report.policy_allowed == report.candidate_signals
    assert "regime_confidence_low" not in dict(report.policy_reasons)


def test_exact_replay_executes_manifest_stop_from_intrabar_high(tmp_path):
    snapshots = [
        replace(
            snapshot,
            bar_highs={"BTC/USDT": 120.0},
        )
        for snapshot in _snapshots()
    ]
    panteon_agents.set_fetcher(_FundingFeed())
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path),
            agent=_agent(),
        )
        report = runner.run(snapshots, flatten_end=True)
    finally:
        panteon_agents.set_fetcher(None)

    stopped = [
        outcome
        for outcome in runner.trade_outcomes
        if outcome.close_reason == "policy_stop_loss_intrabar"
    ]
    assert report.open_fills > 0
    assert stopped
    assert all(outcome.exit_reference_price == pytest.approx(110.0) for outcome in stopped)


def test_exact_replay_fails_closed_when_open_position_has_no_ohlc(tmp_path):
    runner = PolicyReplayRunner.create(
        loaded_manifest=_loaded_manifest(tmp_path),
        agent=_agent(),
    )
    runner.tracker.restore(
        {
            "BTC/USDT": {
                "open_signal_id": 1,
                "sym": "BTC/USDT",
                "side": "short",
                "entry_price": 100.0,
                "qty": 0.1,
                "fee_open": 0.0,
                "by_player": "Policy:test",
                "by_agent": "CarryFlowAgentV2",
                "opened_at": NOW.isoformat(),
                "opened_bar": 0,
                "open_action": "FUT_SHORT_FULL",
                "stop_price": 110.0,
                "stop_loss_pct": 10.0,
            }
        }
    )
    snapshot = replace(
        _snapshots(1)[0],
        bar_opens={},
        bar_highs={},
        bar_lows={},
        bar_closes={},
    )

    with pytest.raises(ValueError, match="requires complete OHLC"):
        runner.run((snapshot,))


def test_missing_derivatives_context_fails_coverage_without_needing_a_signal(tmp_path):
    panteon_agents.set_fetcher(None)
    runner = PolicyReplayRunner.create(
        loaded_manifest=_loaded_manifest(tmp_path),
        agent=_agent(),
    )

    report = runner.run(_snapshots(), flatten_end=True)

    assert report.actor_open_actions == 0
    assert report.derivatives_context_complete_observations == 0
    assert report.derivatives_context_missing_observations == report.actor_observations
    assert report.derivatives_context_coverage_pct == 0.0
    assert "derivatives_context_coverage_below_required" in report.evidence_failures
    assert report.parity_passed is True


def test_policy_deny_rolls_back_carryflow_phantom_position(tmp_path):
    panteon_agents.set_fetcher(_FundingFeed())
    try:
        runner = PolicyReplayRunner.create(
            loaded_manifest=_loaded_manifest(tmp_path, regime="bearish"),
            agent=_agent(),
        )

        report = runner.run(_snapshots(), flatten_end=True)
    finally:
        panteon_agents.set_fetcher(None)

    assert report.candidate_signals > 0
    assert report.policy_allowed == 0
    assert report.policy_no_trade == report.candidate_signals
    assert report.open_execution_attempts == 0
    assert report.open_fills == 0
    assert report.closed_trades == 0
    assert dict(report.policy_reasons) == {"regime_not_allowed": report.candidate_signals}
    assert dict(report.activation_reasons).get("close_without_owned_position", 0) == 0
    assert runner.actor_adapter.runtime.pos["BTC/USDT"] is None
    assert report.parity_passed is True


def test_carryflow_adapter_requires_complete_pinned_actor_config(tmp_path):
    payload = _manifest_payload()
    payload["actor_config"].pop("HOLD")
    payload = seal_manifest_payload(payload)
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_policy_manifest(
        path,
        project_root=tmp_path,
        expected_sha256=payload["manifest_sha256"],
        now=NOW,
        verify_runtime_fingerprint=False,
    )

    with pytest.raises(CarryFlowAdapterError, match="missing CarryFlow actor_config"):
        PolicyReplayRunner.create(loaded_manifest=loaded, agent=_agent())


def test_carryflow_adapter_rejects_profile_mixed_with_scalar_knobs(tmp_path):
    payload = _manifest_payload()
    payload["actor_config"] = {
        "PROFILE_ID": "screened_short_v1",
        "HOLD": 6,
    }
    payload = seal_manifest_payload(payload)
    path = tmp_path / "mixed-profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_policy_manifest(
        path,
        project_root=tmp_path,
        expected_sha256=payload["manifest_sha256"],
        now=NOW,
        verify_runtime_fingerprint=False,
    )

    with pytest.raises(CarryFlowAdapterError, match="cannot be mixed"):
        PolicyReplayRunner.create(loaded_manifest=loaded, agent=_agent())


def test_replay_rejects_market_cadence_gap(tmp_path):
    snapshots = _snapshots(3)
    snapshots[1] = MarketSnapshot(
        bar=2,
        timestamp=NOW + timedelta(minutes=3),
        regime=Regime.NEUTRAL,
        regime_confidence=0.9,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 1000.0},
        regimes_by_symbol={"BTC/USDT": Regime.NEUTRAL},
    )
    runner = PolicyReplayRunner.create(
        loaded_manifest=_loaded_manifest(tmp_path),
        agent=_agent(),
    )

    with pytest.raises(ValueError, match="cadence mismatch"):
        runner.run(snapshots)


def test_adapter_rejects_manifest_actor_freshness_mismatch(tmp_path):
    payload = _manifest_payload()
    payload["data"]["max_derivatives_age_seconds"] = 600
    payload = seal_manifest_payload(payload)
    path = tmp_path / "freshness_mismatch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_policy_manifest(
        path,
        project_root=tmp_path,
        expected_sha256=payload["manifest_sha256"],
        now=NOW,
        verify_runtime_fingerprint=False,
    )

    with pytest.raises(CarryFlowAdapterError, match="MAX_DATA_AGE_SEC"):
        PolicyReplayRunner.create(loaded_manifest=loaded, agent=_agent())
