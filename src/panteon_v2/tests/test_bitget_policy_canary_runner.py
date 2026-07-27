from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from panteon_v2.selection import FlashAllocatorConfig


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_bitget_policy_canary.py"
ANALYZER_PATH = ROOT / "tools" / "analyze_bitget_policy_activation.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bitget_policy_canary", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("bitget_policy_activation", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hypothesis() -> dict[str, object]:
    return {
        "id": "controlled_exploration_loss_budget_deny_richard_btc",
        "actor": "controlled_exploration",
        "candidate_variant": "controlled_exploration",
        "symbols": ["BTC", "ETH"],
        "candidate_runner_args": [
            "--flash-controlled-exploration-loss-budget-usd",
            "0.20",
            "--flash-controlled-exploration-loss-budget-adverse-move-pct",
            "5.0",
            "--flash-controlled-exploration-stop-loss-pct",
            "3.5",
            "--flash-controlled-exploration-session-loss-budget-usd",
            "0.20",
            "--flash-terminal-deny-context-signal-key",
            "agent:RichardDennis|BTC/USDT|FUT_LONG_FULL|neutral",
        ],
    }


def _carryflow_top4_hypothesis() -> dict[str, object]:
    return {
        "id": "carryflow_neutral_short_top4",
        "actor": "CarryFlowAgentV2",
        "single_component_candidate_label": "CarryFlowAgentV2",
        "symbols": ["ADA", "BNB", "DOGE", "SOL"],
        "candidate_runner_args": [
            "--flash-terminal-deny-context-signal-key",
            "agent:CarryFlowAgentV2|*|*|range_low_vol",
            "--flash-terminal-deny-context-signal-key",
            "agent:CarryFlowAgentV2|BTC|*|neutral",
        ],
    }


def test_archived_manifest_cannot_load_a_canary_profile(tmp_path):
    tool = _load_tool()
    manifest_path = tmp_path / "archived.json"
    manifest_path.write_text(
        json.dumps(
            {
                "exchange": "BITGET",
                "operational_status": "archived_research_only",
                "hypotheses": [_hypothesis()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not operational"):
        tool.load_hypothesis(
            manifest_path,
            "controlled_exploration_loss_budget_deny_richard_btc",
        )


def test_policy_flash_config_matches_controlled_exploration_matrix_profile():
    tool = _load_tool()
    base = FlashAllocatorConfig(
        controlled_exploration_enabled=False,
        causal_actor_router_enabled=True,
        causal_actor_router_exploration_enabled=True,
        promotion_derived_router_enabled=True,
        live_real_actor_whitelist=("__NO_LIVE_ACTOR_UNTIL_PROMOTION__",),
        terminal_denied_context_signal_keys=("agent:Old|*|*|*",),
        flash_genetics_core_primary_enabled=True,
        genetics_probation_bypass_terminal_deny_enabled=True,
    )

    cfg = tool.build_policy_flash_config(
        base,
        _hypothesis(),
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
    )

    assert cfg.controlled_exploration_enabled is True
    assert cfg.controlled_exploration_allowed_reasons == (
        "expected_edge_below_cost",
        "range_low_vol_actor_not_allowed",
        "shadow_unconfirmed",
        "insufficient_closed_trades",
    )
    assert cfg.controlled_exploration_allow_range_low_vol_actor_not_allowed is True
    assert cfg.controlled_exploration_range_low_vol_allowed_directions == ("short",)
    assert cfg.controlled_exploration_risk_mult == 0.05
    assert cfg.controlled_exploration_min_shadow_score == 0.0
    assert cfg.controlled_exploration_min_shadow_closed == 0
    assert cfg.controlled_exploration_max_daily_trades == 6
    assert cfg.controlled_exploration_max_open_positions == 2
    assert cfg.controlled_exploration_min_notional_sizing_enabled is True
    assert cfg.controlled_exploration_account_equity_usd == 1000.0
    assert cfg.controlled_exploration_capital_fraction == 0.10
    assert cfg.controlled_exploration_min_notional_max_risk_mult == 0.10
    assert cfg.controlled_exploration_default_min_notional_usd == 5.0
    assert cfg.controlled_exploration_loss_budget_usd == 0.20
    assert cfg.controlled_exploration_loss_budget_adverse_move_pct == 5.0
    assert cfg.controlled_exploration_stop_loss_pct == 3.5
    assert cfg.controlled_exploration_session_loss_budget_usd == 0.20
    assert cfg.causal_actor_router_enabled is False
    assert cfg.causal_actor_router_exploration_enabled is False
    assert cfg.promotion_derived_router_enabled is False
    assert cfg.live_real_actor_whitelist == ()
    assert "agent:Old|*|*|*" not in cfg.terminal_denied_context_signal_keys
    assert "agent:RichardDennis|BTC/USDT|FUT_LONG_FULL|neutral" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert cfg.flash_genetics_core_primary_enabled is False
    assert cfg.genetics_probation_bypass_terminal_deny_enabled is False


def test_policy_flash_config_supports_single_component_matrix_profile():
    tool = _load_tool()
    base = FlashAllocatorConfig(
        controlled_exploration_enabled=False,
        causal_actor_router_enabled=True,
        promotion_derived_router_enabled=False,
        live_real_actor_whitelist=("__NO_LIVE_ACTOR_UNTIL_PROMOTION__",),
        terminal_denied_context_signal_keys=("agent:Old|*|*|*",),
        flash_genetics_core_primary_enabled=True,
        genetics_probation_bypass_terminal_deny_enabled=True,
    )

    cfg = tool.build_policy_flash_config(
        base,
        _carryflow_top4_hypothesis(),
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
    )

    assert cfg.live_real_actor_whitelist == (
        "CarryFlowAgentV2",
        "agent:CarryFlowAgentV2",
        "Solo_CarryFlowAgentV2",
        "ensemble:Solo_CarryFlowAgentV2",
    )
    assert cfg.promotion_derived_router_enabled is True
    assert cfg.promotion_derived_actor_labels == ("CarryFlowAgentV2",)
    assert cfg.min_closed_trades_to_trade == 0
    assert cfg.controlled_exploration_enabled is True
    assert cfg.controlled_exploration_allowed_reasons == (
        "no_evidence",
        "score_below_threshold",
    )
    assert "agent:Old|*|*|*" not in cfg.terminal_denied_context_signal_keys
    assert "agent:CarryFlowAgentV2|*|*|range_low_vol" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert "agent:CarryFlowAgentV2|BTC|*|neutral" in (
        cfg.terminal_denied_context_signal_keys
    )
    assert cfg.flash_genetics_core_primary_enabled is False
    assert cfg.genetics_probation_bypass_terminal_deny_enabled is False


def test_matrix_evidence_requires_pass(tmp_path):
    tool = _load_tool()
    summary_path = tmp_path / "panteon3_pre_live_matrix_summary.json"
    summary_path.write_text(
        json.dumps({
            "promotion_verdict": {
                "passed": False,
                "fail_reasons": ["min_filled"],
            }
        }),
        encoding="utf-8",
    )

    try:
        tool.load_matrix_evidence(
            "controlled_exploration_loss_budget_deny_richard_btc",
            summary_path,
            require_pass=True,
        )
    except ValueError as exc:
        assert "matrix promotion gate is not PASS" in str(exc)
    else:
        raise AssertionError("expected failed matrix summary to block policy canary")


def test_matrix_evidence_exports_promotion_eligible_slice_summary(tmp_path):
    tool = _load_tool()
    summary_path = tmp_path / "panteon3_pre_live_matrix_summary.json"
    summary_path.write_text(
        json.dumps({
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "candidate": {"filled_signals": 24, "closed_trades": 12},
            "candidate_slices": {
                "CarryFlowAgentV2|ADA|neutral|SHORT": {
                    "actor_label": "CarryFlowAgentV2",
                    "symbol": "ADA",
                    "regime": "neutral",
                    "direction": "SHORT",
                    "promotion_eligible": True,
                    "filled_signals": 2,
                    "closed_trades": 2,
                    "expectancy_usd": 0.24,
                    "lcb_usd": 0.22,
                },
                "CarryFlowAgentV2|ADA|range_low_vol|SHORT": {
                    "actor_label": "CarryFlowAgentV2",
                    "symbol": "ADA",
                    "regime": "range_low_vol",
                    "direction": "SHORT",
                    "promotion_eligible": False,
                    "filled_signals": 2,
                    "closed_trades": 2,
                    "expectancy_usd": -0.10,
                    "lcb_usd": -0.12,
                },
            },
        }),
        encoding="utf-8",
    )

    evidence = tool.load_matrix_evidence("carryflow_neutral_short_top4", summary_path)

    slice_summary = evidence["candidate_slice_summary"]
    assert slice_summary["eligible_count"] == 1
    assert slice_summary["eligible_regimes_by_symbol"] == {"ADA": ["neutral"]}
    assert slice_summary["eligible_slices"][0]["key"] == (
        "CarryFlowAgentV2|ADA|neutral|SHORT"
    )


def test_run_exchange_uses_paper_feed_without_smoke_or_genetics(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        captured["BITGET_SYMBOLS"] = os.environ.get("BITGET_SYMBOLS")
        pipeline = type("Pipeline", (), {})()
        pipeline.flash_allocator = type(
            "Allocator",
            (),
            {"_config": FlashAllocatorConfig()},
        )()
        pipeline.executor = type(
            "Executor",
            (),
            {"_exchange": type("Exchange", (), {"set_min_notional": lambda *args: None})()},
        )()
        kwargs["configure_pipeline"](pipeline)
        captured["runtime_flash_allocator_config"] = pipeline.flash_allocator._config
        captured["pipeline_policy"] = pipeline.bitget_policy_canary
        return 0

    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)
    flash_config = tool.build_policy_flash_config(
        FlashAllocatorConfig(live_real_actor_whitelist=("blocked",)),
        _hypothesis(),
        initial_capital=1000.0,
    )

    result = tool.run_exchange(
        results_root=tmp_path / "results",
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        max_bars=65,
        max_idle_polls=5,
        sleep_between_polls_sec=0.0,
        warmup_bars=240,
        symbols=("BTC", "ETH"),
        flash_config=flash_config,
        candidate_policy={"profile_id": "profile"},
    )

    assert result["return_code"] == 0
    assert result["risk_capital_fraction"] == 0.10
    assert result["execution_smoke"] is False
    assert result["calibration_only"] is False
    assert result["include_genetics"] is False
    assert captured["mode"] == "paper_live_feed"
    assert captured["include_genetics"] is False
    assert captured["risk_config_override"].capital_fraction == 0.10
    assert captured["flash_enabled_override"] is True
    assert captured["BITGET_SYMBOLS"] == "BTC,ETH"
    assert captured["runtime_flash_allocator_config"].live_real_actor_whitelist == ()
    assert captured["pipeline_policy"] == {"profile_id": "profile"}


def test_run_exchange_applies_single_component_registry_override(monkeypatch, tmp_path):
    tool = _load_tool()
    captured: dict[str, object] = {}

    def fake_configure_single_component_pipeline(pipeline, actor_label, **kwargs):
        captured["single_component_actor_label"] = actor_label
        captured["single_component_kwargs"] = dict(kwargs)
        setattr(pipeline, "single_component_canary_actor", actor_label)
        return actor_label

    def fake_start_production(**kwargs):
        captured.update(kwargs)
        pipeline = type("Pipeline", (), {})()
        pipeline.flash_allocator = type(
            "Allocator",
            (),
            {"_config": FlashAllocatorConfig()},
        )()
        pipeline.executor = type(
            "Executor",
            (),
            {"_exchange": type("Exchange", (), {"set_min_notional": lambda *args: None})()},
        )()
        kwargs["configure_pipeline"](pipeline)
        captured["runtime_flash_allocator_config"] = pipeline.flash_allocator._config
        captured["pipeline_policy"] = pipeline.bitget_policy_canary
        return 0

    monkeypatch.setattr(tool, "configure_single_component_pipeline", fake_configure_single_component_pipeline)
    monkeypatch.setattr(tool.startup, "start_production", fake_start_production)
    flash_config = tool.build_policy_flash_config(
        FlashAllocatorConfig(),
        _carryflow_top4_hypothesis(),
        initial_capital=1000.0,
    )

    result = tool.run_exchange(
        results_root=tmp_path / "results",
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        max_bars=10,
        max_idle_polls=1,
        sleep_between_polls_sec=0.0,
        warmup_bars=1,
        symbols=("ADA", "BNB", "DOGE", "SOL"),
        flash_config=flash_config,
        candidate_policy={
            "profile_id": "carryflow_neutral_short_top4",
            "candidate_variant": "single_component__CarryFlowAgentV2",
        },
    )

    assert captured["single_component_actor_label"] == "CarryFlowAgentV2"
    assert captured["single_component_kwargs"] == {
        "paper_probe_on_idle": False,
        "actor_overrides": {},
    }
    assert result["runtime_single_component_actor"] == "CarryFlowAgentV2"
    assert captured["pipeline_policy"]["candidate_variant"] == "single_component__CarryFlowAgentV2"


def test_actionable_regime_snapshot_blocks_all_range_low_vol():
    tool = _load_tool()

    snapshot = tool.build_actionable_regime_snapshot(
        {
            "mode": "paper_live_feed",
            "run_state": "stopped",
            "feed_status": "closed",
            "regimes_by_symbol": {
                "BTC": "range_low_vol",
                "ETH": "range_low_vol",
            },
        },
        symbols=("BTC", "ETH"),
        blocked_regimes=("range_low_vol",),
        min_actionable_symbols=1,
    )

    assert snapshot["actionable"] is False
    assert snapshot["actionable_symbols"] == []
    assert snapshot["blocked_symbols"] == ["BTC", "ETH"]


def test_actionable_regime_snapshot_allows_one_non_range_symbol():
    tool = _load_tool()

    snapshot = tool.build_actionable_regime_snapshot(
        {
            "regimes_by_symbol": {
                "BTC": "range_low_vol",
                "ETH": "bullish",
            },
        },
        symbols=("BTC", "ETH"),
        blocked_regimes=("range_low_vol",),
        min_actionable_symbols=1,
    )

    assert snapshot["actionable"] is True
    assert snapshot["actionable_symbols"] == ["ETH"]
    assert snapshot["blocked_symbols"] == ["BTC"]


def test_actionable_regime_snapshot_respects_policy_terminal_deny_keys():
    tool = _load_tool()

    snapshot = tool.build_actionable_regime_snapshot(
        {
            "regimes_by_symbol": {
                "ADA": "range_low_vol",
                "SOL": "bearish",
            },
        },
        symbols=("ADA", "SOL"),
        blocked_regimes=("range_low_vol",),
        min_actionable_symbols=1,
        actor_label="CarryFlowAgentV2",
        terminal_denied_context_signal_keys=(
            "agent:CarryFlowAgentV2|*|*|bearish",
        ),
    )

    assert snapshot["actionable"] is False
    assert snapshot["actionable_symbols"] == []
    assert snapshot["blocked_symbols"] == ["ADA", "SOL"]
    assert snapshot["policy_blocked_symbols"] == ["SOL"]


def test_cli_skips_strict_canary_when_actionable_precheck_fails(
    monkeypatch,
    tmp_path,
    capsys,
):
    tool = _load_tool()
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps({"hypotheses": [_hypothesis()]}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "promotion_gates": {"candidate_variant": "controlled_exploration"},
            "candidate": {"filled_signals": 24, "closed_trades": 12},
            "candidate_slices": {
                "LiveOIBreakout|BTC|neutral|SHORT": {
                    "actor_label": "LiveOIBreakout",
                    "symbol": "BTC",
                    "regime": "neutral",
                    "direction": "SHORT",
                    "promotion_eligible": True,
                    "filled_signals": 2,
                    "closed_trades": 2,
                    "expectancy_usd": 0.10,
                    "lcb_usd": 0.08,
                }
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(),
    )

    def fake_monitor(**kwargs):
        return {
            "runner": "bitget_policy_canary_actionable_regime_monitor",
            "actionable": False,
            "final_snapshot": {
                "actionable": False,
                "regimes_by_symbol": {
                    "BTC": "range_low_vol",
                    "ETH": "range_low_vol",
                },
                "actionable_symbols": [],
                "blocked_symbols": ["BTC", "ETH"],
            },
            "attempts": [
                {
                    "runner": "bitget_policy_canary_actionable_regime_precheck",
                    "snapshot": {
                        "actionable": False,
                        "blocked_symbols": ["BTC", "ETH"],
                    },
                }
            ],
        }

    def fail_run_exchange(**kwargs):
        raise AssertionError("strict canary must not run without actionable regime")

    def fail_summary(**kwargs):
        raise AssertionError("summary must not be built when canary is skipped")

    monkeypatch.setattr(tool, "run_actionable_regime_monitor", fake_monitor)
    monkeypatch.setattr(tool, "run_exchange", fail_run_exchange)
    monkeypatch.setattr(tool, "build_canary_summary", fail_summary)

    rc = tool.main([
        "--manifest",
        str(manifest_path),
        "--matrix-summary",
        str(matrix_path),
        "--results-root",
        str(tmp_path / "results"),
        "--reports-dir",
        str(tmp_path / "reports"),
        "--run-id",
        "skip-run",
        "--require-actionable-regime",
    ])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["canary_skipped"] is True
    assert out["skip_reason"] == "no_actionable_regime"
    assert out["canary_summary"] is None
    assert out["actionable_regime_monitor"]["actionable"] is False
    assert out["activation_gap_report"]["activation_gap_reason"] == (
        "observed_regimes_all_blocked"
    )
    assert out["activation_gap_report"]["continue_same_test"] is False
    report_path = Path(out["activation_gap_report"]["path"])
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["activation_gap_reason"] == "observed_regimes_all_blocked"


def test_cli_runs_strict_canary_after_actionable_monitor_pass(
    monkeypatch,
    tmp_path,
    capsys,
):
    tool = _load_tool()
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps({"hypotheses": [_hypothesis()]}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "promotion_gates": {"candidate_variant": "controlled_exploration"},
            "candidate": {"filled_signals": 24, "closed_trades": 12},
        }),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(),
    )

    def fake_monitor(**kwargs):
        captured["monitor"] = kwargs
        return {
            "runner": "bitget_policy_canary_actionable_regime_monitor",
            "actionable": True,
            "final_snapshot": {
                "actionable": True,
                "regimes_by_symbol": {
                    "BTC": "range_low_vol",
                    "ETH": "bullish",
                },
                "actionable_symbols": ["ETH"],
                "blocked_symbols": ["BTC"],
            },
            "attempts": [
                {
                    "runner": "bitget_policy_canary_actionable_regime_precheck",
                    "snapshot": {"actionable": True, "actionable_symbols": ["ETH"]},
                }
            ],
        }

    def fake_run_exchange(**kwargs):
        captured["run_exchange"] = kwargs
        return {
            "exchange": "BITGET",
            "return_code": 0,
            "runtime_terminal_denied_context_signal_keys": [],
        }

    def fake_summary(**kwargs):
        captured["summary"] = kwargs
        return {"passed": True, "exchanges": {"BITGET": {"passed": True}}}

    monkeypatch.setattr(tool, "run_actionable_regime_monitor", fake_monitor)
    monkeypatch.setattr(tool, "run_exchange", fake_run_exchange)
    monkeypatch.setattr(tool, "build_canary_summary", fake_summary)

    rc = tool.main([
        "--manifest",
        str(manifest_path),
        "--matrix-summary",
        str(matrix_path),
        "--results-root",
        str(tmp_path / "results"),
        "--reports-dir",
        str(tmp_path / "reports"),
        "--run-id",
        "actionable-run",
        "--require-actionable-regime",
        "--actionable-monitor-attempts",
        "2",
        "--actionable-monitor-interval-sec",
        "0",
    ])

    assert rc == 0
    assert captured["monitor"]["attempts"] == 2
    assert captured["run_exchange"]["symbols"] == ("BTC", "ETH")
    assert captured["summary"]["candidate_policy"]["actionable_regime_precheck"][
        "actionable_symbols"
    ] == ["ETH"]
    out = json.loads(capsys.readouterr().out)
    assert out["actionable_regime_monitor"]["actionable"] is True
    assert out["canary_summary"]["passed"] is True


def test_cli_runs_strict_bitget_summary(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps({"hypotheses": [_hypothesis()]}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "promotion_gates": {"candidate_variant": "controlled_exploration"},
            "candidate": {
                "filled_signals": 24,
                "closed_trades": 12,
                "expectancy_usd": 0.0669,
                "max_drawdown_usd": 0.135,
                "cost_attribution_present": True,
            },
        }),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool.startup,
        "_resolve_flash_allocator_config",
        lambda exchange: FlashAllocatorConfig(live_real_actor_whitelist=("blocked",)),
    )

    def fake_run_exchange(**kwargs):
        captured["run_exchange"] = kwargs
        return {
            "exchange": "BITGET",
            "return_code": 0,
            "runtime_terminal_denied_context_signal_keys": [
                "agent:RichardDennis|BTC/USDT|FUT_LONG_FULL|neutral",
            ],
        }

    def fake_build_canary_summary(**kwargs):
        captured["summary"] = kwargs
        return {
            "passed": True,
            "execution_smoke": bool(kwargs.get("execution_smoke")),
            "calibration_only": bool(kwargs.get("calibration_only")),
            "exchanges": {
                "BITGET": {
                    "passed": True,
                    "signals": 2,
                    "orders": 2,
                    "fills": 2,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "run_exchange", fake_run_exchange)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--manifest",
        str(manifest_path),
        "--matrix-summary",
        str(matrix_path),
        "--results-root",
        str(tmp_path / "results"),
        "--reports-dir",
        str(tmp_path / "reports"),
        "--run-id",
        "test-run",
        "--max-bars",
        "65",
        "--max-idle-polls",
        "5",
        "--sleep-between-polls-sec",
        "0",
    ])

    assert rc == 0
    run_kwargs = captured["run_exchange"]
    summary_kwargs = captured["summary"]
    assert run_kwargs["symbols"] == ("BTC", "ETH")
    assert str(run_kwargs["results_root"]).endswith(
        "controlled_exploration_loss_budget_deny_richard_btc\\test-run"
    ) or str(run_kwargs["results_root"]).endswith(
        "controlled_exploration_loss_budget_deny_richard_btc/test-run"
    )
    assert run_kwargs["flash_config"].live_real_actor_whitelist == ()
    assert summary_kwargs["exchanges"] == ("BITGET",)
    assert summary_kwargs["require_positive_expectancy"] is True
    assert summary_kwargs["calibration_only"] is False
    assert summary_kwargs["execution_smoke"] is False
    assert summary_kwargs["candidate_policy"]["matrix_evidence"]["candidate_metrics"][
        "filled_signals"
    ] == 24
    assert summary_kwargs["candidate_policy"]["run_id"] == "test-run"
    capsys.readouterr()


def test_activation_analyzer_cli_reuses_existing_monitor_report(tmp_path, capsys):
    analyzer = _load_analyzer()
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    monitor_path = tmp_path / "monitor.json"
    reports_dir = tmp_path / "reports"
    manifest_path.write_text(
        json.dumps({"exchange": "BITGET", "hypotheses": [_carryflow_top4_hypothesis()]}),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "candidate": {"filled_signals": 24, "closed_trades": 12},
            "candidate_slices": {
                "CarryFlowAgentV2|ADA|neutral|SHORT": {
                    "actor_label": "CarryFlowAgentV2",
                    "symbol": "ADA",
                    "regime": "neutral",
                    "direction": "SHORT",
                    "promotion_eligible": True,
                    "filled_signals": 2,
                    "closed_trades": 2,
                    "expectancy_usd": 0.24,
                    "lcb_usd": 0.22,
                }
            },
        }),
        encoding="utf-8",
    )
    monitor_path.write_text(
        json.dumps({
            "actionable": False,
            "final_snapshot": {
                "actionable": False,
                "regimes_by_symbol": {
                    "ADA": "range_low_vol",
                    "BNB": "range_low_vol",
                    "DOGE": "range_low_vol",
                    "SOL": "range_low_vol",
                },
                "actionable_symbols": [],
                "blocked_symbols": ["ADA", "BNB", "DOGE", "SOL"],
            },
            "attempts": [
                {
                    "snapshot": {
                        "regimes_by_symbol": {
                            "ADA": "range_low_vol",
                            "BNB": "range_low_vol",
                            "DOGE": "range_low_vol",
                            "SOL": "range_low_vol",
                        }
                    }
                }
            ],
        }),
        encoding="utf-8",
    )

    rc = analyzer.main([
        "--profile-id",
        "carryflow_neutral_short_top4",
        "--manifest",
        str(manifest_path),
        "--matrix-summary",
        str(matrix_path),
        "--monitor-report",
        str(monitor_path),
        "--reports-dir",
        str(reports_dir),
    ])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["activation_gap_reason"] == "observed_regimes_all_blocked"
    assert out["continue_same_test"] is False
    assert out["matrix_eligible_regimes_by_symbol"] == {"ADA": ["neutral"]}
    assert Path(out["path"]).exists()
