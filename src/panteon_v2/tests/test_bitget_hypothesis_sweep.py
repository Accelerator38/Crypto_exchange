from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_bitget_hypothesis_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_bitget_hypothesis_sweep", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_commands_for_manifest(tmp_path):
    tool = _load_tool()
    manifest = {
        "exchange": "BITGET",
        "data_dir": "Retrodate/mexc_bitget_futures",
        "default_years": "2026",
        "default_max_bars": 480,
        "default_window_skip_bars": [0, 240],
        "hypotheses": [
            {
                "id": "carryflow_short_funding",
                "single_component_candidate_label": "CarryFlowAgentV2",
                "min_filled": 20,
                "min_closed_trades": 10,
                "include_derivatives_context_actors": True,
                "extra_runner_args": [
                    "--flash-terminal-deny-context-signal-key",
                    "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*",
                ],
            }
        ],
    }

    commands = tool.build_sweep_commands(
        manifest,
        python_executable="python",
        results_root=tmp_path / "Results",
        reports_root=tmp_path / "Reports",
    )

    command = commands[0]["command"]
    assert commands[0]["id"] == "carryflow_short_funding"
    assert "tools/run_panteon3_pre_live_matrix.py" in command
    assert "--single-component-candidate-label" in command
    assert "CarryFlowAgentV2" in command
    assert "--candidate-variant" in command
    assert "single_component__CarryFlowAgentV2" in command
    assert "--include-derivatives-context-actors" in command
    assert "--require-cost-attribution" in command
    assert "--extra-runner-arg=--flash-terminal-deny-context-signal-key" in command
    assert "--extra-runner-arg=agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*" in command


def test_build_commands_supports_policy_level_candidate_runner_args(tmp_path):
    tool = _load_tool()
    manifest = {
        "exchange": "BITGET",
        "data_dir": "Retrodate/bitget_futures_current_20260709",
        "default_years": "2026",
        "default_max_bars": 720,
        "default_window_skip_bars": [0, 240],
        "hypotheses": [
            {
                "id": "controlled_exploration_loss_budget",
                "actor": "controlled_exploration",
                "candidate_variant": "controlled_exploration",
                "data_dir": "Retrodate/bitget_futures_current_20260709",
                "baseline_compare_mode": "pnl_per_drawdown",
                "min_filled": 20,
                "min_closed_trades": 10,
                "max_drawdown_usd": 0.2,
                "candidate_runner_args": [
                    "--flash-controlled-exploration-loss-budget-usd",
                    "0.20",
                    "--flash-terminal-deny-context-signal-key",
                    "agent:RichardDennis|BTC/USDT|FUT_LONG_FULL|neutral",
                ],
            }
        ],
    }

    commands = tool.build_sweep_commands(
        manifest,
        python_executable="python",
        results_root=tmp_path / "Results",
        reports_root=tmp_path / "Reports",
    )

    command = commands[0]["command"]
    assert commands[0]["candidate_variant"] == "controlled_exploration"
    assert "--single-component-candidate-label" not in command
    assert "--candidate-variant" in command
    assert "controlled_exploration" in command
    assert "Retrodate/bitget_futures_current_20260709" in command
    assert "--baseline-compare-mode" in command
    assert "pnl_per_drawdown" in command
    assert "--candidate-runner-arg=--flash-controlled-exploration-loss-budget-usd" in command
    assert "--candidate-runner-arg=0.20" in command
    assert "--candidate-runner-arg=--flash-terminal-deny-context-signal-key" in command
    assert (
        "--candidate-runner-arg=agent:RichardDennis|BTC/USDT|FUT_LONG_FULL|neutral"
        in command
    )


def test_archived_manifest_cannot_start_a_sweep(tmp_path):
    tool = _load_tool()
    manifest = {
        "exchange": "BITGET",
        "operational_status": "archived_research_only",
        "hypotheses": [],
    }

    with pytest.raises(ValueError, match="not operational"):
        tool.build_sweep_commands(
            manifest,
            python_executable="python",
            results_root=tmp_path / "Results",
            reports_root=tmp_path / "Reports",
        )
