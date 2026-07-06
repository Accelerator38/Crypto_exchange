from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from panteon_v2.domain.types import Action, Regime
from panteon_v2.selection import AgentRegistry, FlashAllocatorConfig


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_panteon3_single_component_canary.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("single_component_canary", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_bootstrap_adds_runtime_path_for_exchange_settings():
    tool = _load_tool()

    assert str(ROOT / "src" / "panteon_runtime") in sys.path
    assert tool.RUNTIME == ROOT / "src" / "panteon_runtime"


class _Agent:
    prefers_full_market_snapshot = True

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


class _Market:
    prices = {"BTC": 100.0, "ETH": 200.0}


class _DirectionalMarket:
    prices = {"BTC": 1000.0, "ADA": 0.2}

    def regime_for_symbol(self, sym: str) -> Regime:
        if str(sym).upper() == "ADA":
            return Regime.BULLISH
        return Regime.RANGE_LOW_VOL


class _PaperExchange:
    def __init__(self) -> None:
        self.min_notional: dict[str, float] = {}

    def set_min_notional(self, sym: str, value: float) -> None:
        self.min_notional[sym] = float(value)


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
    assert cfg.promotion_derived_risk_mult == 0.03
    assert cfg.promotion_derived_min_notional_sizing_enabled is True
    assert cfg.promotion_derived_min_notional_max_risk_mult == 1.0
    assert cfg.promotion_derived_default_min_notional_usd == 5.0
    assert cfg.controlled_exploration_enabled is True
    assert cfg.controlled_exploration_allowed_reasons == (
        "no_evidence",
        "score_below_threshold",
    )
    assert cfg.controlled_exploration_risk_mult == 0.03
    assert cfg.controlled_exploration_min_shadow_score == 0.0
    assert cfg.controlled_exploration_min_shadow_closed == 0
    assert cfg.controlled_exploration_min_notional_sizing_enabled is True
    assert cfg.controlled_exploration_min_notional_max_risk_mult == 1.0
    assert cfg.controlled_exploration_default_min_notional_usd == 5.0
    assert cfg.max_signals_per_actor == 8
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


def test_single_component_flash_config_uses_diagnostic_risk_for_promotion_probe():
    tool = _load_tool()
    cfg = tool.single_component_flash_config(
        FlashAllocatorConfig(promotion_derived_risk_mult=0.03),
        "LiveOIBreakout",
        diagnostic=True,
        exploration_risk_mult=1.0,
    )

    assert cfg.promotion_derived_risk_mult == 1.0


def test_single_component_flash_config_merges_terminal_context_deny_keys():
    tool = _load_tool()
    base = FlashAllocatorConfig(
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol",
        ),
    )

    cfg = tool.single_component_flash_config(
        base,
        "LiveOIBreakout",
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        ),
    )

    assert "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert "agent:LiveOIBreakout|BTC/USDT|FUT_SHORT_HALF|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )


def test_single_component_flash_config_bypasses_terminal_denies_for_execution_smoke():
    tool = _load_tool()
    base = FlashAllocatorConfig(
        terminal_denied_signal_keys=(
            "agent:LiveOIBreakout|BTC|FUT_SHORT_FULL",
        ),
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol",
        ),
    )

    cfg = tool.single_component_flash_config(
        base,
        "LiveOIBreakout",
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        ),
        bypass_terminal_denies=True,
    )

    assert cfg.terminal_denied_signal_keys == ()
    assert cfg.terminal_denied_context_signal_keys == ()


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


def test_configure_pipeline_can_wrap_idle_actor_with_paper_probe():
    tool = _load_tool()
    pipeline = _Pipeline()

    selected = tool.configure_single_component_pipeline(
        pipeline,
        "LiveOIBreakout",
        paper_probe_on_idle=True,
    )

    assert selected == "LiveOIBreakout"
    agent = pipeline.registry.get("LiveOIBreakout")
    assert agent is not None
    assert getattr(agent, "prefers_full_market_snapshot", False) is True
    actions = agent.act(_Market())
    assert actions["BTC"] == Action.HOLD
    assert actions["ETH"] == Action.FUT_SHORT_FULL
    assert agent.last_signal_diagnostics["ETH"]["reason"] == "paper_canary_probe_open"
    assert agent.last_signal_diagnostics["ETH"]["base_reason"] == "hold"
    assert agent.last_signal_diagnostics["ETH"]["paper_canary_probe"] is True


def test_paper_probe_prefers_directional_symbol_when_inner_open_is_not_directional():
    tool = _load_tool()

    class _NonDirectionalOpenAgent(_Agent):
        def act(self, market):
            self.last_signal_diagnostics = {
                "BTC": {"reason": "candidate_short"},
                "ADA": {"reason": "check_interval_wait"},
            }
            return {"BTC": Action.FUT_SHORT_FULL, "ADA": Action.HOLD}

    agent = tool.PaperCanaryProbeAgent(
        _NonDirectionalOpenAgent("LiveOIBreakout"),
        label="LiveOIBreakout",
    )

    actions = agent.act(_DirectionalMarket())

    assert actions["BTC"] == Action.FUT_SHORT_FULL
    assert actions["ADA"] == Action.FUT_SHORT_FULL
    assert agent.last_signal_diagnostics["ADA"]["reason"] == "paper_canary_probe_open"
    assert agent.last_signal_diagnostics["ADA"]["base_reason"] == "check_interval_wait"


def test_configure_pipeline_applies_calibration_overrides_to_selected_actor():
    tool = _load_tool()
    pipeline = _Pipeline()

    selected = tool.configure_single_component_pipeline(
        pipeline,
        "LiveOIBreakout",
        actor_overrides={"CHECK_INT": 1, "MOM_MIN": 0.0015},
    )

    agent = pipeline.registry.get("LiveOIBreakout")
    assert selected == "LiveOIBreakout"
    assert agent is not None
    assert agent.CHECK_INT == 1
    assert agent.MOM_MIN == 0.0015
    assert pipeline.single_component_canary_actor_overrides == {
        "CHECK_INT": 1,
        "MOM_MIN": 0.0015,
    }


def test_run_exchange_uses_paper_live_feed_and_isolated_runtime_paths(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        pipeline = _Pipeline()
        pipeline.flash_allocator = type(
            "Allocator",
            (),
            {"_config": FlashAllocatorConfig()},
        )()
        kwargs["configure_pipeline"](pipeline)
        captured["runtime_flash_allocator_config"] = pipeline.flash_allocator._config
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
        paper_probe_on_idle=True,
    )

    assert result["exchange"] == "MEXC"
    assert result["actor_label"] == "LiveOIBreakout"
    assert result["return_code"] == 0
    assert captured["mode"] == "paper_live_feed"
    assert captured["initial_capital"] == 250.0
    assert captured["use_v1_bridge"] is True
    assert captured["live_execution_config_override"].max_new_opens_per_bar == 8
    assert captured["flash_enabled_override"] is True
    assert captured["flash_allocator_config_override"].controlled_exploration_risk_mult == 0.12
    assert captured["max_bars"] == 1
    assert captured["max_idle_polls"] == 2
    assert captured["warmup_bars"] == 141
    assert result["effective_warmup_bars"] == 141
    assert result["paper_probe_on_idle"] is True
    assert str(captured["results_root"]).endswith("results")
    assert str(captured["snapshot_path"]).endswith("results\\state\\mexc_single_component_snapshot.json")
    assert str(captured["jsonl_event_log"]).endswith("results\\logs\\mexc_single_component_events.jsonl")


def test_run_exchange_can_override_exchange_symbols_for_mexc_canary(monkeypatch, tmp_path):
    tool = _load_tool()
    captured_env: dict[str, str | None] = {}

    def fake_start_production(**kwargs):
        captured_env["MEXC_SYMBOLS"] = os.environ.get("MEXC_SYMBOLS")
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
        symbols=("btc", "ETH/USDT", "ada"),
    )

    assert result["exchange"] == "MEXC"
    assert result["symbols"] == ["BTC", "ETH", "ADA"]
    assert captured_env["MEXC_SYMBOLS"] == "BTC,ETH,ADA"
    assert os.environ.get("MEXC_SYMBOLS") is None


def test_run_exchange_sets_paper_exchange_min_notional_for_canary_symbols(monkeypatch, tmp_path):
    tool = _load_tool()
    captured_exchange = _PaperExchange()

    def fake_start_production(**kwargs):
        pipeline = _Pipeline()
        pipeline.executor = type("Executor", (), {"_exchange": captured_exchange})()
        kwargs["configure_pipeline"](pipeline)
        return 0

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(),
    )
    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)

    result = tool.run_exchange(
        "BITGET",
        actor_label="LiveOIBreakout",
        results_root=tmp_path / "results",
        initial_capital=100.0,
        max_bars=1,
        max_idle_polls=2,
        sleep_between_polls_sec=0.0,
        warmup_bars=3,
        diagnostic=True,
        exploration_risk_mult=1.0,
        symbols=("SOL", "BNB/USDT"),
    )

    assert result["return_code"] == 0
    assert result["paper_min_notional_floor_usd"] == 5.0
    assert captured_exchange.min_notional["SOL"] == 5.0
    assert captured_exchange.min_notional["SOL/USDT"] == 5.0
    assert captured_exchange.min_notional["BNB"] == 5.0
    assert captured_exchange.min_notional["BNB/USDT"] == 5.0


def test_run_exchange_diagnostic_does_not_enable_paper_probe_by_default(monkeypatch, tmp_path):
    tool = _load_tool()

    def fake_start_production(**kwargs):
        pipeline = _Pipeline()
        kwargs["configure_pipeline"](pipeline)
        agent = pipeline.registry.get("LiveOIBreakout")
        assert agent is not None
        assert agent.__class__.__name__ != "PaperCanaryProbeAgent"
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

    assert result["return_code"] == 0
    assert result["paper_probe_on_idle"] is False


def test_run_exchange_applies_terminal_context_deny_keys(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        pipeline = _Pipeline()
        pipeline.flash_allocator = type(
            "Allocator",
            (),
            {"_config": FlashAllocatorConfig()},
        )()
        kwargs["configure_pipeline"](pipeline)
        captured["runtime_flash_allocator_config"] = pipeline.flash_allocator._config
        return 0

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(
            terminal_denied_context_signal_keys=(
                "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol",
            ),
        ),
    )
    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)

    result = tool.run_exchange(
        "BITGET",
        actor_label="LiveOIBreakout",
        results_root=tmp_path / "results",
        initial_capital=100.0,
        max_bars=1,
        max_idle_polls=2,
        sleep_between_polls_sec=0.0,
        warmup_bars=3,
        diagnostic=True,
        exploration_risk_mult=1.0,
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        ),
    )

    cfg = captured["flash_allocator_config_override"]
    assert "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )
    runtime_cfg = captured["runtime_flash_allocator_config"]
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in (
        runtime_cfg.terminal_denied_context_signal_keys
    )
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in result[
        "terminal_denied_context_signal_keys"
    ]
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in result[
        "runtime_terminal_denied_context_signal_keys"
    ]


def test_run_exchange_can_bypass_terminal_denies_for_execution_smoke(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        pipeline = _Pipeline()
        pipeline.flash_allocator = type(
            "Allocator",
            (),
            {"_config": FlashAllocatorConfig()},
        )()
        kwargs["configure_pipeline"](pipeline)
        captured["runtime_flash_allocator_config"] = pipeline.flash_allocator._config
        return 0

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(
            terminal_denied_signal_keys=(
                "agent:LiveOIBreakout|BTC|FUT_SHORT_FULL",
            ),
            terminal_denied_context_signal_keys=(
                "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol",
            ),
        ),
    )
    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)

    result = tool.run_exchange(
        "BITGET",
        actor_label="LiveOIBreakout",
        results_root=tmp_path / "results",
        initial_capital=100.0,
        max_bars=1,
        max_idle_polls=2,
        sleep_between_polls_sec=0.0,
        warmup_bars=3,
        diagnostic=True,
        exploration_risk_mult=1.0,
        paper_probe_on_idle=True,
        terminal_denied_context_signal_keys=(
            "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        ),
        bypass_terminal_denies=True,
    )

    cfg = captured["flash_allocator_config_override"]
    assert cfg.terminal_denied_signal_keys == ()
    assert cfg.terminal_denied_context_signal_keys == ()
    runtime_cfg = captured["runtime_flash_allocator_config"]
    assert runtime_cfg.terminal_denied_signal_keys == ()
    assert runtime_cfg.terminal_denied_context_signal_keys == ()
    assert result["execution_smoke_bypass_terminal_denies"] is True
    assert result["runtime_terminal_denied_signal_keys"] == []
    assert result["runtime_terminal_denied_context_signal_keys"] == []


def test_strict_canary_never_bypasses_terminal_denies(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exchange": "BITGET",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
            "runtime_terminal_denied_context_signal_keys": [
                "agent:LiveOIBreakout|*|*|range_low_vol",
            ],
        }

    def fake_build_canary_summary(**kwargs):
        captured["summary_execution_smoke"] = kwargs.get("execution_smoke")
        return {
            "passed": False,
            "execution_smoke": bool(kwargs.get("execution_smoke")),
            "exchanges": {
                "BITGET": {
                    "passed": False,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "expectancy_after_costs": 0.0,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "BITGET",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
        "--strict-gates",
        "--require-positive-expectancy",
    ])

    assert rc == 2
    assert captured["paper_probe_on_idle"] is False
    assert captured["bypass_terminal_denies"] is False
    assert captured["summary_execution_smoke"] is False
    capsys.readouterr()


def test_cli_defaults_exploration_risk_mult_to_validated_canary_cap(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exchange": "MEXC",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
        }

    def fake_build_canary_summary(**kwargs):
        captured["require_positive_expectancy"] = kwargs.get("require_positive_expectancy")
        captured["execution_smoke"] = kwargs.get("execution_smoke")
        return {
            "passed": True,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "MEXC",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "1",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
    ])

    assert rc == 0
    assert captured["exploration_risk_mult"] == 1.0
    assert captured["paper_probe_on_idle"] is True
    assert captured["bypass_terminal_denies"] is True
    assert captured["require_positive_expectancy"] is False
    assert captured["execution_smoke"] is True
    capsys.readouterr()


def test_cli_requires_positive_expectancy_for_extended_diagnostic_canary(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exchange": "MEXC",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
        }

    def fake_build_canary_summary(**kwargs):
        captured["require_positive_expectancy"] = kwargs.get("require_positive_expectancy")
        return {
            "passed": True,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "MEXC",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
    ])

    assert rc == 0
    assert captured["max_bars"] == 65
    assert captured["paper_probe_on_idle"] is False
    assert captured["require_positive_expectancy"] is True
    capsys.readouterr()


def test_cli_marks_actor_overrides_as_calibration_only(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured["actor_overrides"] = kwargs.get("actor_overrides")
        return {
            "exchange": "MEXC",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
            "actor_overrides": kwargs.get("actor_overrides"),
        }

    def fake_build_canary_summary(**kwargs):
        captured["calibration_only"] = kwargs.get("calibration_only")
        captured["summary_actor_overrides"] = kwargs.get("actor_overrides")
        return {
            "passed": True,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 2,
                    "orders": 2,
                    "fills": 2,
                    "expectancy_after_costs": 0.02,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "MEXC",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
        "--actor-check-int", "1",
        "--actor-mom-min", "0.0015",
    ])

    assert rc == 0
    assert captured["actor_overrides"] == {"CHECK_INT": 1, "MOM_MIN": 0.0015}
    assert captured["summary_actor_overrides"] == {"CHECK_INT": 1, "MOM_MIN": 0.0015}
    assert captured["calibration_only"] is True
    capsys.readouterr()


def test_cli_passes_terminal_context_deny_keys_to_runs_and_summary(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured["terminal_denied_context_signal_keys"] = kwargs.get(
            "terminal_denied_context_signal_keys"
        )
        return {
            "exchange": "BITGET",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
            "terminal_denied_context_signal_keys": list(
                kwargs.get("terminal_denied_context_signal_keys") or []
            ),
            "runtime_terminal_denied_context_signal_keys": [
                "agent:LiveOIBreakout|SOL|FUT_SHORT_HALF|range_low_vol",
                "agent:LiveOIBreakout|ADA|FUT_SHORT_FULL|range_low_vol",
            ],
        }

    def fake_build_canary_summary(**kwargs):
        captured["summary_terminal_denied_context_signal_keys"] = kwargs.get(
            "terminal_denied_context_signal_keys"
        )
        return {
            "passed": True,
            "terminal_denied_context_signal_keys": list(
                kwargs.get("terminal_denied_context_signal_keys") or []
            ),
            "exchanges": {
                "BITGET": {
                    "passed": True,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "BITGET",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
        "--terminal-deny-context-signal-key",
        "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        "--terminal-deny-context-signal-key",
        "agent:LiveOIBreakout|SOL|FUT_SHORT_HALF|range_low_vol",
    ])

    assert rc == 0
    expected = (
        "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
        "agent:LiveOIBreakout|SOL|FUT_SHORT_HALF|range_low_vol",
    )
    assert captured["terminal_denied_context_signal_keys"] == expected
    assert captured["summary_terminal_denied_context_signal_keys"] == (
        *expected,
        "agent:LiveOIBreakout|ADA|FUT_SHORT_FULL|range_low_vol",
    )
    capsys.readouterr()


def test_cli_loads_candidate_policy_without_marking_calibration_only(
    monkeypatch,
    tmp_path,
    capsys,
):
    tool = _load_tool()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({
            "exchange": "BITGET",
            "actor": "CarryFlowAgentV2",
            "symbols": ["ETH"],
            "terminal_deny_context_signal_keys": [
                "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*",
            ],
            "policy_sha256": "abc",
        }),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exchange": "BITGET",
            "actor_label": "CarryFlowAgentV2",
            "return_code": 0,
        }

    def fake_build_canary_summary(**kwargs):
        captured["calibration_only"] = kwargs.get("calibration_only")
        captured["candidate_policy"] = kwargs.get("candidate_policy")
        return {
            "passed": True,
            "exchanges": {
                "BITGET": {
                    "passed": True,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "BITGET",
        "--actor", "CarryFlowAgentV2",
        "--candidate-policy", str(policy_path),
        "--strict-gates",
        "--require-positive-expectancy",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
    ])

    assert rc == 0
    assert captured["symbols"] == ("ETH",)
    assert captured["calibration_only"] is False
    assert captured["terminal_denied_context_signal_keys"] == (
        "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*",
    )
    assert captured["candidate_policy"]["policy_sha256"] == "abc"
    capsys.readouterr()
