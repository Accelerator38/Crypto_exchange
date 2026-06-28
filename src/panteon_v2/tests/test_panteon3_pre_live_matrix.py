from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "run_panteon3_pre_live_matrix.py"
    spec = importlib.util.spec_from_file_location("panteon3_pre_live_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flags(entry: dict) -> list[str]:
    return list(entry["command"])


def test_build_matrix_plan_has_four_safe_replay_variants(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2026",
        max_bars=120,
        stride_minutes=60,
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
    )

    assert [entry["variant"] for entry in plan] == [
        "baseline",
        "controlled_exploration",
        "causal_router",
        "panteon3_candidate",
    ]
    for entry in plan:
        command = _flags(entry)
        assert command[:3] == ["python", "-m", "panteon_v2.analysis.retrodate_market_runner"]
        assert "--enable-flash" in command
        assert "--exchange-live" not in command
        assert "--live" not in command
        assert "--real-orders" not in command
        assert "--max-bars" in command
        assert "--data-dir" in command
        assert "mexc_bitget_futures" in command[command.index("--data-dir") + 1].lower()
        assert str(tmp_path / "runs" / entry["variant"]) in command
        assert "--compact-causal-entry-include-label" in command

    baseline = _flags(plan[0])
    assert "--enable-flash-controlled-exploration" not in baseline
    assert "--enable-flash-causal-actor-router" not in baseline

    exploration = _flags(plan[1])
    assert "--enable-flash-controlled-exploration" in exploration
    assert "--enable-flash-controlled-exploration-min-notional-sizing" in exploration
    assert "--flash-controlled-exploration-min-rolling-expectancy" in exploration
    assert "--enable-flash-causal-actor-router" not in exploration
    assert "--enable-flash-promotion-derived-router" not in exploration

    router = _flags(plan[2])
    assert "--enable-flash-causal-actor-router" in router
    assert "--enable-flash-controlled-exploration" not in router
    assert "--enable-flash-promotion-derived-router" not in router

    candidate = _flags(plan[3])
    assert "--enable-flash-controlled-exploration" in candidate
    assert "--enable-flash-controlled-exploration-min-notional-sizing" in candidate
    assert "--enable-futures-replay-signal-fixes" in candidate
    assert "--enable-flash-causal-actor-router" in candidate
    assert "--enable-flash-promotion-derived-router" in candidate
    assert "--flash-legacy-real-agent-label" in candidate
    assert "--flash-promotion-derived-actor-label" in candidate
    legacy_real_labels = [
        candidate[index + 1]
        for index, arg in enumerate(candidate)
        if arg == "--flash-legacy-real-agent-label"
    ]
    assert legacy_real_labels == [
        "LiveOIBreakout",
        "MomentumScalper",
        "LiveCrashHunter",
        "LiveVolCompress",
        "FundingArb",
    ]
    assert "CarryFlowAgentV2" not in legacy_real_labels
    assert "LiveOIBreakout" in candidate
    assert "MomentumScalper" in candidate
    assert "LiveCrashHunter" in candidate
    assert "LiveVolCompress" in candidate
    assert "FundingArb" in candidate
    promotion_labels = [
        candidate[index + 1]
        for index, arg in enumerate(candidate)
        if arg == "--flash-promotion-derived-actor-label"
    ]
    assert promotion_labels == [
        "LiveOIBreakout",
        "MomentumScalper",
        "LiveCrashHunter",
        "LiveVolCompress",
        "FundingArb",
    ]
    assert "CarryFlowAgentV2" not in promotion_labels
    assert "--flash-promotion-derived-risk-mult" in candidate
    assert "--flash-promotion-derived-dynamic-best" in candidate
    assert "0.03" in candidate
    assert "--enable-flash-promotion-derived-min-notional-sizing" in candidate
    assert "--flash-promotion-derived-account-equity-usd" in candidate
    assert "--flash-promotion-derived-min-notional-max-risk-mult" in candidate
    assert "--flash-controlled-exploration-min-notional-max-risk-mult" in candidate
    diagnostic_labels = [
        candidate[index + 1]
        for index, arg in enumerate(candidate)
        if arg == "--compact-causal-entry-include-label"
    ]
    assert "CarryFlowAgentV2" in diagnostic_labels
    assert "LiveVolCompress" in diagnostic_labels
    assert "0.10" in candidate


def test_build_matrix_plan_can_include_derivatives_context_actors_when_requested(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2026",
        max_bars=120,
        stride_minutes=60,
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        include_derivatives_context_actors=True,
    )

    candidate = _flags(plan[3])
    legacy_real_labels = [
        candidate[index + 1]
        for index, arg in enumerate(candidate)
        if arg == "--flash-legacy-real-agent-label"
    ]
    promotion_labels = [
        candidate[index + 1]
        for index, arg in enumerate(candidate)
        if arg == "--flash-promotion-derived-actor-label"
    ]
    assert "CarryFlowAgentV2" in legacy_real_labels
    assert "CarryFlowAgentV2" in promotion_labels


def test_build_matrix_plan_carries_extra_runner_args(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2026",
        max_bars=None,
        stride_minutes=15,
        initial_capital=500.0,
        risk_capital_fraction=0.05,
        extra_runner_args=["--disable-hard-policy"],
    )

    for entry in plan:
        command = _flags(entry)
        assert "--max-bars" not in command
        assert command[command.index("--stride-minutes") + 1] == "15"
        assert command[command.index("--initial-capital") + 1] == "500.0"
        assert command[command.index("--risk-capital-fraction") + 1] == "0.05"
        assert "--disable-hard-policy" in command


def test_build_matrix_plan_can_expand_multiple_replay_windows(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2025,2026",
        max_bars=1440,
        stride_minutes=1,
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        window_skip_bars=(0, 4320),
    )

    assert [entry["variant"] for entry in plan] == [
        "skip_0__baseline",
        "skip_0__controlled_exploration",
        "skip_0__causal_router",
        "skip_0__panteon3_candidate",
        "skip_4320__baseline",
        "skip_4320__controlled_exploration",
        "skip_4320__causal_router",
        "skip_4320__panteon3_candidate",
    ]
    first_window = _flags(plan[0])
    second_window = _flags(plan[4])
    assert "--skip-bars" not in first_window
    assert second_window[second_window.index("--skip-bars") + 1] == "4320"
    assert str(tmp_path / "runs" / "skip_4320__panteon3_candidate") in _flags(plan[7])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_variant_run(
    root: Path,
    variant: str,
    *,
    selected: int,
    filled: int,
    blocked: int,
    rejected: int,
    closed: int,
    pnl: float,
    gross_profit: float,
    gross_loss: float,
    windows: list[float],
    regimes: list[float],
) -> Path:
    run_dir = root / variant / "RETRODATE_MARKET" / "run_001"
    _write_json(
        run_dir / "flash_attribution_summary.json",
        {
            "summary": {
                "selected_signals": selected,
                "executable_selected_signals": filled,
                "filled_signals": filled,
                "blocked_signals": blocked,
                "rejected_signals": rejected,
                "pending_signals": 0,
                "closed_trades": closed,
                "winning_trades": max(0, closed - 1),
                "losing_trades": 1 if gross_loss < 0 else 0,
                "realized_pnl_usd": pnl,
            }
        },
    )
    _write_json(
        run_dir / "allocation_diagnostics.json",
        {
            "bars": 100,
            "no_trade_share_pct": 0.0,
            "raw_zero_share_pct": 100.0 - filled,
            "filled_zero_share_pct": 100.0 - filled,
        },
    )
    _write_json(
        run_dir / "candidate_diagnostics.json",
        {
            "rejection_summary": {
                "flash_active_reason_counts": {
                    "insufficient_closed_trades": 2,
                }
            }
        },
    )
    _write_json(
        run_dir / "component_benchmark_report.json",
        {
            "summary": {
                "panteon_pnl_usd": pnl,
                "panteon_pnl_pct": pnl / 10.0,
                "best_component_label": "BestActor",
                "best_component_pnl_usd": pnl - 0.1,
                "best_component_pnl_pct": (pnl - 0.1) / 10.0,
                "panteon_alpha_pct": 0.01,
                "panteon_beats_best_component": pnl > 0.0,
            }
        },
    )
    _write_json(
        run_dir / "walk_forward_report.json",
        {
            "temporal_windows": {
                "windows": [
                    {
                        "window": f"window_{index}",
                        "stats": {"closed_trades": 1, "net_pnl": value},
                    }
                    for index, value in enumerate(windows, start=1)
                ]
            },
            "by_regime": {
                f"regime_{index}": {
                    "closed_trades": 1,
                    "net_pnl": value,
                    "gross_profit": max(value, 0.0),
                    "gross_loss": min(value, 0.0),
                    "profit_factor": 2.0 if value > 0.0 else 0.0,
                }
                for index, value in enumerate(regimes, start=1)
            },
            "by_actor": {
                "LiveVolCompress": {
                    "closed_trades": closed,
                    "net_pnl": pnl,
                    "gross_profit": gross_profit,
                    "gross_loss": gross_loss,
                    "profit_factor": 99.0 if gross_loss == 0 else gross_profit / abs(gross_loss),
                    "expectancy": pnl / closed if closed else 0.0,
                }
            },
        },
    )
    return run_dir


def test_collect_matrix_summary_applies_prelive_promotion_gates(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    _write_variant_run(
        results_root,
        "baseline",
        selected=0,
        filled=0,
        blocked=0,
        rejected=0,
        closed=0,
        pnl=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        windows=[0.0, 0.0],
        regimes=[0.0],
    )
    _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=4,
        filled=4,
        blocked=0,
        rejected=0,
        closed=2,
        pnl=0.5,
        gross_profit=0.7,
        gross_loss=-0.2,
        windows=[0.4, -0.1],
        regimes=[0.5, 0.0],
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=4,
        min_closed_trades=2,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    candidate = summary["candidate"]
    assert candidate["variant"] == "panteon3_candidate"
    assert candidate["filled_signals"] == 4
    assert candidate["closed_trades"] == 2
    assert candidate["expectancy_usd"] == 0.25
    assert candidate["profit_factor"] == 3.5
    assert candidate["profitable_windows"] == 1
    assert candidate["positive_regimes"] == 1
    assert summary["promotion_verdict"]["passed"] is True
    assert summary["promotion_verdict"]["fail_reasons"] == []


def test_collect_matrix_summary_reports_failed_prelive_gates(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    _write_variant_run(
        results_root,
        "baseline",
        selected=1,
        filled=1,
        blocked=0,
        rejected=0,
        closed=1,
        pnl=0.1,
        gross_profit=0.1,
        gross_loss=0.0,
        windows=[0.1],
        regimes=[0.1],
    )
    _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=2,
        filled=1,
        blocked=1,
        rejected=1,
        closed=1,
        pnl=-0.2,
        gross_profit=0.0,
        gross_loss=-0.2,
        windows=[-0.2],
        regimes=[-0.2],
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=4,
        min_closed_trades=2,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    assert summary["promotion_verdict"]["passed"] is False
    assert summary["promotion_verdict"]["fail_reasons"] == [
        "filled_signals 1 < min_filled 4",
        "closed_trades 1 < min_closed_trades 2",
        "blocked_signals 1 > max_blocked 0",
        "rejected_signals 1 > max_rejected 0",
        "expectancy_usd -0.2 <= 0",
        "profitable_windows 0 < min_profitable_windows 1",
        "positive_regimes 0 < min_positive_regimes 1",
        "candidate_pnl_usd -0.2 <= baseline_pnl_usd 0.1",
    ]


def test_collect_matrix_summary_aggregates_multi_window_candidate(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    _write_variant_run(
        results_root,
        "skip_0__baseline",
        selected=1,
        filled=1,
        blocked=0,
        rejected=0,
        closed=1,
        pnl=0.1,
        gross_profit=0.1,
        gross_loss=0.0,
        windows=[0.1],
        regimes=[0.1],
    )
    _write_variant_run(
        results_root,
        "skip_4320__baseline",
        selected=1,
        filled=1,
        blocked=0,
        rejected=0,
        closed=1,
        pnl=0.2,
        gross_profit=0.2,
        gross_loss=0.0,
        windows=[0.2],
        regimes=[0.2],
    )
    _write_variant_run(
        results_root,
        "skip_0__panteon3_candidate",
        selected=2,
        filled=2,
        blocked=0,
        rejected=0,
        closed=1,
        pnl=0.4,
        gross_profit=0.4,
        gross_loss=0.0,
        windows=[0.4],
        regimes=[0.4],
    )
    _write_variant_run(
        results_root,
        "skip_4320__panteon3_candidate",
        selected=3,
        filled=3,
        blocked=0,
        rejected=0,
        closed=2,
        pnl=0.8,
        gross_profit=0.9,
        gross_loss=-0.1,
        windows=[0.8],
        regimes=[0.8],
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=5,
        min_closed_trades=3,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=2,
        min_positive_regimes=2,
        require_beats_baseline=True,
    )

    assert summary["candidate"]["variant"] == "panteon3_candidate"
    assert summary["candidate"]["source_variants"] == [
        "skip_0__panteon3_candidate",
        "skip_4320__panteon3_candidate",
    ]
    assert summary["candidate"]["filled_signals"] == 5
    assert summary["candidate"]["closed_trades"] == 3
    assert summary["candidate"]["realized_pnl_usd"] == pytest.approx(1.2)
    assert summary["candidate"]["expectancy_usd"] == pytest.approx(0.4)
    assert summary["baseline"]["realized_pnl_usd"] == 0.30000000000000004
    assert summary["promotion_verdict"]["passed"] is True
