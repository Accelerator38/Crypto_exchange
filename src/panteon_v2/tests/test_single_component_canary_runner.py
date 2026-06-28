from __future__ import annotations

import importlib.util
from pathlib import Path

from panteon_v2.domain.types import Action
from panteon_v2.selection import AgentRegistry, FlashAllocatorConfig


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_panteon3_single_component_canary.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("single_component_canary", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Agent:
    def __init__(self, label: str) -> None:
        self.label = label
        self.shadow_only = True
        self.live_trading_eligible = False
        self.paper_trading_eligible = False

    def act(self, market):
        return {symbol: Action.HOLD for symbol in market.prices}


class _Pipeline:
    def __init__(self) -> None:
        self.registry = AgentRegistry()
        self.registry.register(_Agent("LiveOIBreakout"))
        self.registry.register(_Agent("OtherActor"))
        self.profiles = ("profile",)
        self.manual_quarantine_labels = ("OtherActor",)
        self.quarantine_override_labels = ()


def test_single_component_flash_config_limits_and_relaxes_diagnostic_gates():
    tool = _load_tool()
    base = FlashAllocatorConfig(
        min_closed_trades_to_trade=10,
        min_pnl_pct_to_trade=1.0,
        live_real_actor_whitelist=("OtherActor",),
        range_low_vol_real_actor_allowlist=("OtherActor",),
        promotion_derived_router_enabled=False,
        promotion_derived_actor_labels=("OtherActor",),
        shadow_confirmation_enabled=True,
        shadow_symbol_confirmation_enabled=True,
        shadow_actor_fallback_confirmation_enabled=True,
        shadow_base_fallback_confirmation_enabled=True,
        shadow_quality_confirmation_enabled=True,
        global_health_gate_enabled=True,
        regime_edge_gate_enabled=True,
        real_loss_gate_enabled=True,
        fee_aware_admission_enabled=True,
    )

    cfg = tool.single_component_flash_config(
        base,
        "LiveOIBreakout",
        diagnostic=True,
        exploration_risk_mult=0.03,
    )

    assert "LiveOIBreakout" in cfg.live_real_actor_whitelist
    assert "agent:LiveOIBreakout" in cfg.live_real_actor_whitelist
    assert "Solo_LiveOIBreakout" in cfg.live_real_actor_whitelist
    assert "OtherActor" not in cfg.live_real_actor_whitelist
    assert cfg.range_low_vol_real_actor_allowlist == cfg.live_real_actor_whitelist
    assert cfg.promotion_derived_router_enabled is True
    assert cfg.promotion_derived_actor_labels == ("LiveOIBreakout",)
    assert cfg.controlled_exploration_enabled is True
    assert cfg.controlled_exploration_allowed_reasons == ("no_evidence",)
    assert cfg.controlled_exploration_risk_mult == 0.03
    assert cfg.controlled_exploration_min_shadow_score == 0.0
    assert cfg.controlled_exploration_min_shadow_closed == 0
    assert cfg.min_closed_trades_to_trade == 0
    assert cfg.min_pnl_pct_to_trade < 0.0
    assert cfg.shadow_confirmation_enabled is False
    assert cfg.shadow_symbol_confirmation_enabled is False
    assert cfg.shadow_actor_fallback_confirmation_enabled is False
    assert cfg.shadow_base_fallback_confirmation_enabled is False
    assert cfg.shadow_quality_confirmation_enabled is False
    assert cfg.global_health_gate_enabled is False
    assert cfg.regime_edge_gate_enabled is False
    assert cfg.real_loss_gate_enabled is False
    assert cfg.fee_aware_admission_enabled is False


def test_configure_pipeline_keeps_only_selected_actor_and_makes_it_executable():
    tool = _load_tool()
    pipeline = _Pipeline()

    selected = tool.configure_single_component_pipeline(pipeline, "LiveOIBreakout")

    assert selected == "LiveOIBreakout"
    assert pipeline.registry.all_labels() == ["LiveOIBreakout"]
    agent = pipeline.registry.get("LiveOIBreakout")
    assert agent is not None
    assert agent.shadow_only is False
    assert agent.live_trading_eligible is True
    assert agent.paper_trading_eligible is True
    assert pipeline.profiles == ()
    assert pipeline.manual_quarantine_labels == ()
    assert pipeline.quarantine_override_labels == ("LiveOIBreakout",)


def test_run_exchange_uses_paper_live_feed_and_isolated_runtime_paths(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        pipeline = _Pipeline()
        kwargs["configure_pipeline"](pipeline)
        return 0

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(),
    )
    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)

    result = tool.run_exchange(
        "MEXC",
        actor_label="LiveOIBreakout",
        results_root=tmp_path / "results",
        initial_capital=250.0,
        max_bars=1,
        max_idle_polls=2,
        sleep_between_polls_sec=0.0,
        warmup_bars=3,
        diagnostic=True,
        exploration_risk_mult=0.12,
    )

    assert result["exchange"] == "MEXC"
    assert result["actor_label"] == "LiveOIBreakout"
    assert result["return_code"] == 0
    assert captured["mode"] == "paper_live_feed"
    assert captured["initial_capital"] == 250.0
    assert captured["use_v1_bridge"] is True
    assert captured["flash_enabled_override"] is True
    assert captured["flash_allocator_config_override"].controlled_exploration_risk_mult == 0.12
    assert captured["max_bars"] == 1
    assert captured["max_idle_polls"] == 2
    assert captured["warmup_bars"] == 3
    assert str(captured["results_root"]).endswith("results")
    assert str(captured["snapshot_path"]).endswith("results\\state\\mexc_single_component_snapshot.json")
    assert str(captured["jsonl_event_log"]).endswith("results\\logs\\mexc_single_component_events.jsonl")
