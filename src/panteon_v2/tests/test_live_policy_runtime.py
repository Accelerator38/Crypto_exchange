from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from panteon_v2.app.agent_bootstrap import _ensure_paths


_ensure_paths()

from panteon_agents import CarryFlowAgentV2

from panteon_v2.app.bitget_adapter import BitgetExchangeAdapter
from panteon_v2.app import policy_runtime as policy_runtime_module
from panteon_v2.app import startup
from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.app.main_loop import main_loop
from panteon_v2.app.output_writer import OutputWriter, OutputWriterConfig
from panteon_v2.app.policy_runtime import (
    LivePolicyRuntimeError,
    LivePolicyRuntimeV1,
    LivePolicyStep,
)
from panteon_v2.domain.types import Action, MarketSnapshot, Regime, Signal
from panteon_v2.execution.exchange import FakeExchange
from panteon_v2.execution.position_tracker import PositionTracker
from panteon_v2.execution.risk_limits import RiskLimitsConfig
from panteon_v2.policy.carryflow_adapter import (
    ActorStep,
    CARRYFLOW_CONFIG_FIELDS,
    CarryFlowPolicyAdapter,
)
from panteon_v2.policy.executor import PolicyExecutorV1, PolicyTarget
from panteon_v2.policy.manifest import (
    CostModel,
    Direction,
    LoadedPolicyManifest,
    ManifestValidation,
    MarketDataPolicy,
    PolicyManifest,
    PolicyRule,
    RiskPolicy,
    SCHEMA_VERSION,
    SignalModel,
)
from panteon_v2.selection.agent import AgentRegistry
from panteon_v2.shadow.adapters import V1AgentAdapter
from panteon_v2.shadow.feed import ReplayFeed


NOW = datetime(2026, 7, 15, 12, 0, 30, tzinfo=timezone.utc)


def _manifest(*, expires_at: datetime | None = None) -> PolicyManifest:
    defaults = CarryFlowAgentV2()
    config = {
        name: getattr(defaults, name)
        for name in CARRYFLOW_CONFIG_FIELDS
    }
    config.update({
        "CHECK_INT": 1,
        "EMA_FAST": 12,
        "EMA_SLOW": 48,
        "MAX_POS": 1,
        "ALLOW_LONG": False,
        "ALLOW_SHORT": True,
        "MAX_DATA_AGE_SEC": 1200,
    })
    return PolicyManifest(
        schema_version=SCHEMA_VERSION,
        policy_id="carryflow-test-policy",
        target=PolicyTarget.MICRO_LIVE,
        exchange="BITGET",
        actor="CarryFlowAgentV2",
        actor_config=tuple(sorted(config.items())),
        signal_model=SignalModel(
            feature="diagnostic.edge",
            intercept_bps=0.0,
            slope_bps_per_unit=100.0,
            lcb_haircut_bps=4.0,
            min_feature_value=0.2,
            max_expected_move_bps=250.0,
        ),
        data=MarketDataPolicy(
            bar_interval_seconds=3600,
            cadence_tolerance_seconds=0,
            max_bar_close_lag_seconds=120,
            max_derivatives_age_seconds=1200,
            required_context_coverage_pct=95.0,
        ),
        created_at=NOW - timedelta(days=1),
        expires_at=expires_at or NOW + timedelta(days=1),
        source_revision="a" * 40,
        runtime_fingerprint_sha256="b" * 64,
        costs=CostModel(
            round_trip_fee_bps=8.0,
            slippage_bps=4.0,
            safety_buffer_bps=4.0,
        ),
        risk=RiskPolicy(
            capital_fraction=0.01,
            max_notional_usd=10.0,
            max_open_positions=1,
            max_daily_loss_usd=5.0,
            stop_loss_pct=1.2,
            max_holding_minutes=2880,
            max_signal_age_seconds=120,
        ),
        rules=(
            PolicyRule(
                symbol="BTC",
                regime="neutral",
                direction=Direction.SHORT,
                min_expected_move_bps=16.0,
                max_spread_bps=5.0,
                max_slippage_bps=5.0,
                min_regime_confidence=0.0,
            ),
        ),
        evidence=(),
        manifest_sha256="c" * 64,
    )


def _live_runtime(
    *,
    expires_at: datetime | None = None,
    risk_state_path: Path | None = None,
) -> LivePolicyRuntimeV1:
    manifest = _manifest(expires_at=expires_at)
    loaded = LoadedPolicyManifest(
        manifest=manifest,
        path="Runtime/BITGET/active_policy_manifest_v1.json",
        validation=ManifestValidation(True),
    )
    wrapped = V1AgentAdapter(
        label="CarryFlowAgentV2",
        v1_agent=CarryFlowAgentV2(),
        portfolio_value_fn=lambda: 100.0,
    )
    adapter = CarryFlowPolicyAdapter(agent=wrapped, manifest=manifest)
    return LivePolicyRuntimeV1(
        loaded_manifest=loaded,
        actor_adapter=adapter,
        policy_executor=PolicyExecutorV1(loaded),
        risk_state_path=risk_state_path,
    )


def _market(timestamp: datetime = NOW) -> MarketSnapshot:
    return MarketSnapshot(
        bar=7,
        timestamp=timestamp,
        cadence_timestamp=timestamp.replace(minute=0, second=0, microsecond=0),
        regime=Regime.NEUTRAL,
        regime_confidence=0.8,
        prices={"BTC": 100.0},
        volumes={"BTC": 10.0},
        funding={"BTC": 0.0001},
        regimes_by_symbol={"BTC": Regime.NEUTRAL},
    )


class _PolicyFrameExchange:
    def __init__(self) -> None:
        self.calls = 0

    def get_policy_market_frame(self, **kwargs):
        self.calls += 1
        boundary = kwargs["bar_close_timestamp"]
        return {
            "complete": True,
            "observed_at": boundary + timedelta(seconds=30),
            "bar_close_timestamp": boundary,
            "symbols": {
                "BTC": {
                    "complete": True,
                    "decision_price": 100.0,
                    "volume": 12.0,
                    "bid": 99.99,
                    "ask": 100.01,
                    "spread_bps": 2.0,
                    "estimated_slippage_bps": 3.0,
                },
            },
        }


def test_live_policy_runtime_uses_exact_cadence_once_and_real_market_quality():
    runtime = _live_runtime()
    runtime.actor_adapter.propose = lambda *args, **kwargs: ActorStep((), (), ())
    exchange = _PolicyFrameExchange()
    tracker = PositionTracker()

    first = runtime.evaluate(
        _market(),
        exchange=exchange,
        tracker=tracker,
        signal_id_start=10,
        exchange_healthy=True,
        daily_loss_usd=0.0,
    )
    duplicate = runtime.evaluate(
        _market(NOW + timedelta(seconds=10)),
        exchange=exchange,
        tracker=tracker,
        signal_id_start=first.next_signal_id,
        exchange_healthy=True,
        daily_loss_usd=0.0,
    )

    assert first.status == "evaluated"
    assert first.policy_market is not None
    assert first.policy_market.cadence_timestamp == NOW.replace(
        minute=0, second=0, microsecond=0
    )
    assert first.quality[0].spread_bps == 2.0
    assert first.quality[0].estimated_slippage_bps == 4.0
    assert duplicate.status == "waiting_for_bar_close"
    assert exchange.calls == 1
    assert runtime.required_warmup_bars == 53


def test_live_policy_runtime_emits_expiry_exit_without_waiting_for_cadence():
    runtime = _live_runtime(expires_at=NOW)
    tracker = PositionTracker()
    tracker.restore({
        "BTC": {
            "open_signal_id": 1,
            "sym": "BTC",
            "side": "short",
            "entry_price": 100.0,
            "qty": 0.1,
            "fee_open": 0.0,
            "by_player": runtime.label,
            "by_agent": "CarryFlowAgentV2",
            "opened_at": (NOW - timedelta(hours=1)).isoformat(),
            "opened_bar": 1,
            "open_action": "FUT_SHORT_FULL",
        }
    })
    exchange = _PolicyFrameExchange()

    step = runtime.evaluate(
        _market(NOW + timedelta(seconds=1)),
        exchange=exchange,
        tracker=tracker,
        signal_id_start=20,
        exchange_healthy=True,
        daily_loss_usd=0.0,
    )

    assert step.status == "safety_exit"
    assert step.signals[0].action == Action.FUT_CLOSE_ALL
    assert step.signals[0].metadata["close_reason"] == "policy_manifest_expired"
    assert exchange.calls == 0


def test_policy_daily_loss_latch_survives_same_day_restart(tmp_path):
    state_path = Path(tmp_path) / "policy_risk.json"
    runtime = _live_runtime(risk_state_path=state_path)

    assert runtime.update_equity_guard(
        now=NOW,
        current_equity_usd=100.0,
        initial_equity_usd=100.0,
    ) == ""
    reason = runtime.update_equity_guard(
        now=NOW + timedelta(minutes=1),
        current_equity_usd=94.0,
        initial_equity_usd=100.0,
    )
    restarted = _live_runtime(risk_state_path=state_path)
    restarted_reason = restarted.update_equity_guard(
        now=NOW + timedelta(hours=1),
        current_equity_usd=100.0,
        initial_equity_usd=100.0,
    )

    assert "policy max daily loss exceeded" in reason
    assert restarted_reason == reason
    assert restarted.current_daily_loss_usd == 6.0

    next_day_reason = restarted.update_equity_guard(
        now=NOW + timedelta(days=1),
        current_equity_usd=100.0,
        initial_equity_usd=100.0,
    )
    assert next_day_reason == ""


def test_policy_risk_state_corruption_blocks_runtime_start(tmp_path):
    state_path = Path(tmp_path) / "policy_risk.json"
    state_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(LivePolicyRuntimeError, match="policy risk state is unreadable"):
        _live_runtime(risk_state_path=state_path)


class _PublicSwap:
    def __init__(self, *, omit_last: bool = False) -> None:
        self.omit_last = omit_last
        self.calls: list[tuple[str, str]] = []

    def fetch_ticker(self, symbol):
        self.calls.append(("ticker", symbol))
        return {"bid": 99.99, "ask": 100.01, "last": 100.0}

    def fetch_order_book(self, symbol, limit):
        self.calls.append(("book", symbol))
        return {
            "bids": [[99.99, 100.0]],
            "asks": [[100.01, 100.0]],
        }

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append(("ohlcv", symbol))
        count = int(limit) - (1 if self.omit_last and symbol.startswith("ETH/") else 0)
        interval_ms = {
            "1m": 60_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }[timeframe]
        return [
            [since + index * interval_ms, 99.0, 101.0, 98.0, 100.0, 25.0]
            for index in range(count)
        ]


class _OrderClient:
    def __init__(self, exchange) -> None:
        self.exchange = exchange
        self.order_calls = 0

    @staticmethod
    def _market_symbol(symbol):
        return f"{symbol}/USDT:USDT"

    def place_order(self, *args, **kwargs):
        self.order_calls += 1
        raise AssertionError("public policy reads must never place an order")


def _bitget_adapter(public: _PublicSwap) -> tuple[BitgetExchangeAdapter, _OrderClient]:
    order_client = _OrderClient(public)
    read_client = SimpleNamespace(swap=public)
    return (
        BitgetExchangeAdapter(
            order_client=order_client,
            read_client=read_client,
            leverage=1,
        ),
        order_client,
    )


def test_bitget_policy_market_frame_reads_exact_closed_bar_without_orders():
    public = _PublicSwap()
    adapter, order_client = _bitget_adapter(public)
    boundary = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

    frame = adapter.get_policy_market_frame(
        symbols=("BTC",),
        bar_interval_seconds=3600,
        bar_close_timestamp=boundary,
        max_notional_usd=10.0,
    )

    row = frame["symbols"]["BTC"]
    assert frame["complete"] is True
    assert row["candle_timestamp_ms"] == int(boundary.timestamp() * 1000) - 3_600_000
    assert row["decision_price"] == 100.0
    assert row["estimated_slippage_bps"] > 0.0
    assert order_client.order_calls == 0


def test_bitget_policy_warmup_requires_complete_synchronized_window():
    public = _PublicSwap()
    adapter, order_client = _bitget_adapter(public)

    complete = adapter.get_policy_warmup_history(
        symbols=("BTC", "ETH"),
        bar_interval_seconds=3600,
        required_bars=53,
    )

    assert complete["complete"] is True
    assert len(complete["prices"]) == 53
    assert all(set(row) == {"BTC", "ETH"} for row in complete["prices"])
    assert order_client.order_calls == 0

    incomplete_adapter, _ = _bitget_adapter(_PublicSwap(omit_last=True))
    incomplete = incomplete_adapter.get_policy_warmup_history(
        symbols=("BTC", "ETH"),
        bar_interval_seconds=3600,
        required_bars=53,
    )
    assert incomplete["complete"] is False
    assert "ETH" in incomplete["errors"]


class _NeverCalledAgent:
    label = "CarryFlowAgentV2"

    def act(self, market):
        raise AssertionError("legacy actor execution path was called")


class _SingleSignalRuntime:
    def __init__(self, manifest: PolicyManifest, actor) -> None:
        self.manifest = manifest
        self.actor = actor
        self.label = f"Policy:{manifest.policy_id}"
        self.loaded_manifest = SimpleNamespace(path="Runtime/BITGET/policy.json")

    def evaluate(self, market, **kwargs):
        signal_id = int(kwargs["signal_id_start"])
        policy_market = replace(
            market,
            regime=Regime.NEUTRAL,
            regimes_by_symbol={"BTC": Regime.NEUTRAL},
            cadence_timestamp=market.timestamp,
        )
        signal = Signal(
            id=signal_id,
            bar=market.bar,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=market.prices["BTC"],
            regime=Regime.NEUTRAL,
            by_player=self.label,
            by_agent="CarryFlowAgentV2",
            position_scope="policy_test",
            risk_mult=1.0,
            timestamp=market.timestamp,
            metadata={
                "decision_path": "policy_v1",
                "policy_id": self.manifest.policy_id,
                "manifest_sha256": self.manifest.manifest_sha256,
            },
        )
        return LivePolicyStep(
            status="evaluated",
            reason="",
            policy_market=policy_market,
            signals=(signal,),
            candidate_count=1,
            next_signal_id=signal_id + 1,
        )


def test_main_loop_policy_path_bypasses_legacy_selection_and_reports_policy(tmp_path):
    manifest = _manifest()
    actor = _NeverCalledAgent()
    runtime = _SingleSignalRuntime(manifest, actor)
    registry = AgentRegistry()
    registry.register(actor)
    exchange = FakeExchange(name="BITGET")
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=100.0,
        profiles=(),
        flash_enabled=False,
        policy_runtime_v1=runtime,
    )
    pipeline.mode = "live_futures"
    pipeline.timeframe = "1h"
    pipeline.shadow_tournament = SimpleNamespace(
        run_bar=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shadow tournament was called")
        )
    )
    feed = ReplayFeed()
    feed.append(_market())

    steps = main_loop(pipeline, feed)

    assert len(steps) == 1
    step = steps[0]
    assert step.n_filled == 1
    assert step.executed_leader == runtime.label
    assert step.causal_decision["decision_path"] == "policy_v1"
    assert len(exchange.orders_log) == 1
    assert pipeline.flash_enabled is False

    writer = OutputWriter(
        pipeline,
        OutputWriterConfig(
            output_dir=str(tmp_path),
            compact_causal_entry_decisions=True,
            compact_causal_entry_selected_only=True,
        ),
    )
    writer.write(step)
    status = json.loads((Path(tmp_path) / "status.json").read_text(encoding="utf-8"))
    causal_lines = (
        Path(tmp_path) / "causal_entry_decisions.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert status["policy"]["enabled"] is True
    assert status["policy"]["actor"] == "CarryFlowAgentV2"
    assert status["flash"]["enabled"] is False
    assert len(causal_lines) == 1
    assert json.loads(causal_lines[0])["decision_path"] == "policy_v1"


def test_policy_daily_loss_activates_nonrecovering_manage_only_kill_switch():
    class LossExchange(FakeExchange):
        def get_account_snapshot(self):
            return {
                "current_balance": 94.0,
                "futures_equity": 94.0,
                "available_balance": 94.0,
                "spot_assets": 0.0,
                "total_assets": 94.0,
                "unrealized_pnl": 0.0,
            }

    manifest = _manifest()
    actor = _NeverCalledAgent()
    runtime = _SingleSignalRuntime(manifest, actor)
    runtime.evaluate = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("policy evaluation must stop after daily loss")
    )
    registry = AgentRegistry()
    registry.register(actor)
    exchange = LossExchange(name="BITGET")
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=100.0,
        profiles=(),
        flash_enabled=False,
        policy_runtime_v1=runtime,
    )
    feed = ReplayFeed()
    feed.append(_market())

    steps = main_loop(pipeline, feed)

    assert len(steps) == 1
    assert steps[0].executed_leader == "ManageOnly"
    assert "policy max daily loss exceeded" in pipeline.kill_switch.disabled_reason
    assert pipeline.kill_switch.disabled_kind == "policy_daily_loss"
    assert exchange.orders_log == []


def test_startup_policy_mode_registers_only_manifest_actor(monkeypatch, tmp_path):
    manifest = _manifest()
    actor = _NeverCalledAgent()
    runtime = _SingleSignalRuntime(manifest, actor)
    captured: dict[str, object] = {}

    def resolve_probe(*args, **kwargs):
        captured["leverage_override"] = kwargs.get("leverage_override")
        return FakeExchange(name="BITGET")

    def configure_probe(pipeline):
        captured["pipeline"] = pipeline

    monkeypatch.setattr(
        startup,
        "_direct_bitget_live_preflight_failure",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(startup, "resolve_exchange", resolve_probe)
    monkeypatch.setattr(
        startup,
        "register_all_v1_agents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy registry construction was called")
        ),
    )
    monkeypatch.setattr(
        policy_runtime_module,
        "load_bitget_micro_live_policy_runtime",
        lambda **kwargs: runtime,
    )

    result = startup.start_production(
        exchange="BITGET",
        mode="live_futures",
        initial_capital=100.0,
        max_bars=0,
        use_v1_bridge=False,
        results_root=str(tmp_path),
        sleep_between_polls_sec=0.0,
        enable_policy_runtime=True,
        configure_pipeline=configure_probe,
    )

    pipeline = captured["pipeline"]
    assert result == 3
    assert captured["leverage_override"] == 1
    assert pipeline.registry.all_labels() == ["CarryFlowAgentV2"]
    assert pipeline.policy_runtime_v1 is runtime
    assert pipeline.flash_enabled is False
    assert pipeline.flash_allocator is None
    assert pipeline.profiles == []
    assert pipeline.shadow_tournament is None
    assert pipeline.risk_config.max_notional_usd == 10.0
    assert pipeline.risk_config.max_open_positions == 1
    assert pipeline.risk_config.max_leverage == 1


def test_startup_policy_mode_fails_when_real_exchange_adapter_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        startup,
        "_direct_bitget_live_preflight_failure",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(
        startup,
        "resolve_exchange",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("adapter missing")
        ),
    )

    result = startup.start_production(
        exchange="BITGET",
        mode="live_futures",
        enable_policy_runtime=True,
    )

    assert result == 2


def test_policy_pipeline_invariant_rejects_registry_drift():
    manifest = _manifest()
    actor = _NeverCalledAgent()
    runtime = _SingleSignalRuntime(manifest, actor)
    registry = AgentRegistry()
    registry.register(actor)
    exchange = FakeExchange(name="BITGET")
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange,
        initial_capital=100.0,
        policy_runtime_v1=runtime,
        risk_config=startup._restrict_risk_config_to_policy(
            RiskLimitsConfig(),
            manifest,
        ),
    )

    assert startup._policy_pipeline_invariant_error(
        pipeline,
        runtime,
        exchange,
    ) == ""
    registry.register(SimpleNamespace(label="UnexpectedActor", act=lambda market: {}))
    assert startup._policy_pipeline_invariant_error(
        pipeline,
        runtime,
        exchange,
    ) == "registry is not the single manifest actor"
