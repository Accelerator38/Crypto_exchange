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


def test_matrix_summary_exports_actor_symbol_regime_direction_slices():
    module = _load_tool()
    rows = [
        {
            "actor_label": "LiveOIBreakout",
            "symbol": "BTC",
            "regime": "bullish",
            "action": "FUT_LONG_FULL",
            "realized_pnl_usd": 0.12,
            "fees_usd": 0.01,
            "filled": True,
            "closed": True,
        },
        {
            "actor_label": "LiveOIBreakout",
            "symbol": "BTC",
            "regime": "range_low_vol",
            "action": "FUT_SHORT_FULL",
            "realized_pnl_usd": -0.05,
            "fees_usd": 0.01,
            "filled": True,
            "closed": True,
        },
    ]

    slices = module.build_candidate_slice_metrics(rows)

    bullish = slices["LiveOIBreakout|BTC|bullish|LONG"]
    assert bullish["closed_trades"] == 1
    assert bullish["expectancy_usd"] == 0.11
    assert bullish["direction"] == "LONG"
    assert bullish["promotion_eligible"] is True

    range_short = slices["LiveOIBreakout|BTC|range_low_vol|SHORT"]
    assert range_short["expectancy_usd"] == -0.06
    assert range_short["promotion_eligible"] is False
    assert "nonpositive_lcb" in range_short["fail_reasons"]


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
    assert (
        "--flash-controlled-exploration-allow-range-low-vol-actor-not-allowed"
        in exploration
    )
    assert (
        exploration[
            exploration.index(
                "--flash-controlled-exploration-range-low-vol-allowed-direction"
            )
            + 1
        ]
        == "short"
    )
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


def test_build_matrix_plan_can_add_single_component_candidate(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2026",
        max_bars=120,
        stride_minutes=60,
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        single_component_candidate_labels=("LiveOIBreakout",),
    )

    assert [entry["variant"] for entry in plan] == [
        "baseline",
        "controlled_exploration",
        "causal_router",
        "panteon3_candidate",
        "single_component__LiveOIBreakout",
    ]

    component = _flags(plan[4])
    assert "--enable-futures-replay-signal-fixes" in component
    assert "--flash-legacy-real-agent-label" in component
    assert "--flash-single-component-replay-actor" in component
    assert component[component.index("--flash-single-component-replay-actor") + 1] == (
        "LiveOIBreakout"
    )
    whitelist_labels = [
        component[index + 1]
        for index, arg in enumerate(component)
        if arg == "--flash-live-real-actor-whitelist"
    ]
    assert whitelist_labels == [
        "LiveOIBreakout",
        "agent:LiveOIBreakout",
        "Solo_LiveOIBreakout",
        "ensemble:Solo_LiveOIBreakout",
    ]
    range_allowlist_labels = [
        component[index + 1]
        for index, arg in enumerate(component)
        if arg == "--flash-range-low-vol-real-actor-allowlist"
    ]
    assert range_allowlist_labels == whitelist_labels
    assert "--enable-flash-controlled-exploration" in component
    assert "--flash-controlled-exploration-allowed-reason" in component
    assert (
        "--flash-controlled-exploration-allow-range-low-vol-actor-not-allowed"
        not in component
    )
    controlled_reasons = [
        component[index + 1]
        for index, arg in enumerate(component)
        if arg == "--flash-controlled-exploration-allowed-reason"
    ]
    assert "no_evidence" in controlled_reasons
    assert "score_below_threshold" in controlled_reasons
    assert "--flash-controlled-exploration-min-notional-max-risk-mult" in component
    assert (
        component[
            component.index("--flash-controlled-exploration-min-notional-max-risk-mult")
            + 1
        ]
        == "1.00"
    )
    assert "--flash-min-closed-trades-to-trade" in component
    assert component[component.index("--flash-min-closed-trades-to-trade") + 1] == "0"
    assert "--flash-min-pnl-pct-to-trade" in component
    assert component[component.index("--flash-min-pnl-pct-to-trade") + 1] == "-999"
    assert "--flash-max-signals-per-actor" in component
    assert component[component.index("--flash-max-signals-per-actor") + 1] == "8"
    assert "--max-new-opens-per-bar" in component
    assert component[component.index("--max-new-opens-per-bar") + 1] == "8"
    assert "--enable-flash-promotion-derived-router" in component
    assert "--flash-promotion-derived-min-closed-trades" in component
    assert component[component.index("--flash-promotion-derived-min-closed-trades") + 1] == "0"
    assert "--flash-promotion-derived-min-expectancy" in component
    assert component[component.index("--flash-promotion-derived-min-expectancy") + 1] == "-999"
    assert "--enable-flash-promotion-derived-min-notional-sizing" in component
    assert "--flash-promotion-derived-min-notional-max-risk-mult" in component
    assert (
        component[
            component.index("--flash-promotion-derived-min-notional-max-risk-mult") + 1
        ]
        == "1.00"
    )
    legacy_real_labels = [
        component[index + 1]
        for index, arg in enumerate(component)
        if arg == "--flash-legacy-real-agent-label"
    ]
    assert legacy_real_labels == ["LiveOIBreakout"]
    assert "--enable-flash-causal-actor-router" not in component


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


def test_build_matrix_plan_applies_candidate_runner_args_only_to_candidate_variant(tmp_path):
    module = _load_tool()

    plan = module.build_matrix_plan(
        python_executable="python",
        results_root=tmp_path / "runs",
        years="2026",
        max_bars=120,
        stride_minutes=60,
        initial_capital=1000.0,
        risk_capital_fraction=0.10,
        candidate_variant="controlled_exploration",
        candidate_runner_args=[
            "--flash-terminal-deny-context-signal-key",
            "agent:PlayerFunding|ETH/USDT|SPOT_BUY_HALF|neutral",
            "--flash-controlled-exploration-capital-fraction",
            "0.052",
        ],
    )

    baseline = _flags(plan[0])
    controlled = _flags(plan[1])
    router = _flags(plan[2])
    panteon = _flags(plan[3])

    assert "--flash-terminal-deny-context-signal-key" not in baseline
    assert "--flash-terminal-deny-context-signal-key" not in router
    assert "--flash-terminal-deny-context-signal-key" not in panteon
    assert "--flash-terminal-deny-context-signal-key" in controlled
    assert "agent:PlayerFunding|ETH/USDT|SPOT_BUY_HALF|neutral" in controlled
    assert controlled[-2:] == [
        "--flash-controlled-exploration-capital-fraction",
        "0.052",
    ]


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
    losing_trades: int | None = None,
    explicit_costs: float = 0.0,
    turnover_notional: float = 0.0,
    max_drawdown_usd: float = 0.0,
    by_symbol: dict[str, float] | None = None,
    context_rows: list[dict] | None = None,
) -> Path:
    loss_count = 1 if gross_loss < 0 else 0
    if losing_trades is not None:
        loss_count = int(losing_trades)
    run_dir = root / variant / "RETRODATE_MARKET" / "run_001"
    flash_payload = {
        "summary": {
            "selected_signals": selected,
            "executable_selected_signals": filled,
            "filled_signals": filled,
            "blocked_signals": blocked,
            "rejected_signals": rejected,
            "pending_signals": 0,
            "closed_trades": closed,
            "winning_trades": max(0, closed - 1),
            "losing_trades": loss_count,
            "realized_pnl_usd": pnl,
        }
    }
    if context_rows is not None:
        flash_payload["context_rows"] = context_rows
    _write_json(
        run_dir / "flash_attribution_summary.json",
        flash_payload,
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
            "totals": {
                "closed_trades": closed,
                "net_pnl": pnl,
                "gross_profit": gross_profit,
                "gross_loss": abs(gross_loss),
                "explicit_costs": explicit_costs,
                "fees": explicit_costs,
                "fee_per_turnover_pct": 0.06 if turnover_notional > 0.0 else 0.0,
                "turnover_notional": turnover_notional,
                "max_drawdown_pnl": max_drawdown_usd,
            },
            "temporal_windows": {
                "windows": [
                    {
                        "window": f"window_{index}",
                        "stats": {"closed_trades": 1, "net_pnl": value},
                        "by_symbol": (
                            {
                                symbol: {"closed_trades": 1, "net_pnl": symbol_pnl}
                                for symbol, symbol_pnl in (by_symbol or {}).items()
                            }
                            if index == 1 and by_symbol
                            else {}
                        ),
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


def test_collect_matrix_summary_exports_candidate_slices_from_context_rows(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    _write_variant_run(
        results_root,
        "skip_0__baseline",
        selected=0,
        filled=0,
        blocked=0,
        rejected=0,
        closed=0,
        pnl=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        windows=[0.0],
        regimes=[0.0],
    )
    _write_variant_run(
        results_root,
        "skip_0__single_component__LiveOIBreakout",
        selected=2,
        filled=2,
        blocked=0,
        rejected=0,
        closed=2,
        pnl=0.05,
        gross_profit=0.12,
        gross_loss=-0.07,
        windows=[0.05],
        regimes=[0.05],
        context_rows=[
            {
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC/USDT",
                "regime": "bullish",
                "action": "FUT_LONG_FULL",
                "closed_trades": 1,
                "filled_signals": 1,
                "realized_pnl_usd": 0.15,
                "fees_usd": 0.01,
            },
            {
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC/USDT",
                "regime": "range_low_vol",
                "action": "FUT_SHORT_FULL",
                "closed_trades": 1,
                "filled_signals": 1,
                "realized_pnl_usd": -0.08,
                "fees_usd": 0.01,
            },
        ],
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="single_component__LiveOIBreakout",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=False,
    )

    assert summary["candidate_slice_source_rows"] == 2
    slices = summary["candidate_slices"]
    assert slices["LiveOIBreakout|BTC|bullish|LONG"]["promotion_eligible"] is True
    assert slices["LiveOIBreakout|BTC|bullish|LONG"]["expectancy_usd"] == 0.14
    assert slices["LiveOIBreakout|BTC|range_low_vol|SHORT"]["promotion_eligible"] is False
    assert "nonpositive_lcb" in slices[
        "LiveOIBreakout|BTC|range_low_vol|SHORT"
    ]["fail_reasons"]


def test_collect_matrix_summary_flags_missing_gross_loss_with_losing_trades(tmp_path):
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
        windows=[0.0],
        regimes=[0.0],
    )
    _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=12,
        filled=12,
        blocked=0,
        rejected=0,
        closed=12,
        pnl=0.25,
        gross_profit=0.37,
        gross_loss=0.0,
        losing_trades=5,
        windows=[0.25],
        regimes=[0.25],
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    assert summary["candidate"]["profit_factor_reliable"] is False
    assert "panteon3_candidate:gross_loss_missing_with_losing_trades" in summary[
        "accounting_warnings"
    ]
    assert summary["promotion_verdict"]["passed"] is False
    assert "accounting_warnings_present" in summary["promotion_verdict"]["fail_reasons"]


def test_collect_matrix_summary_uses_total_walk_forward_gross_loss(tmp_path):
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
    run_dir = _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=2,
        filled=2,
        blocked=0,
        rejected=0,
        closed=2,
        pnl=0.15,
        gross_profit=0.2,
        gross_loss=0.0,
        losing_trades=1,
        windows=[0.15],
        regimes=[0.15],
    )
    _write_json(
        run_dir / "walk_forward_report.json",
        {
            "temporal_windows": {"windows": [{"stats": {"net_pnl": 0.15}}]},
            "by_regime": {"bull": {"net_pnl": 0.15}},
            "by_actor": {
                "Winner": {
                    "closed_trades": 1,
                    "net_pnl": 0.20,
                    "gross_profit": 0.20,
                    "gross_loss": 0.0,
                    "losses": 0,
                    "wins": 1,
                    "profit_factor": 9999.0,
                },
                "Loser": {
                    "closed_trades": 1,
                    "net_pnl": -0.05,
                    "gross_profit": 0.0,
                    "gross_loss": 0.05,
                    "losses": 1,
                    "wins": 0,
                    "profit_factor": 0.0,
                },
            },
        },
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=False,
    )

    assert summary["candidate"]["gross_loss"] == -0.05
    assert summary["candidate"]["profit_factor_reliable"] is True
    assert summary["accounting_warnings"] == []


def test_single_component_summary_reports_activation_gap_when_flash_selected_zero(tmp_path):
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
        windows=[0.0],
        regimes=[0.0],
    )
    run_dir = _write_variant_run(
        results_root,
        "single_component__LiveOIBreakout",
        selected=0,
        filled=0,
        blocked=0,
        rejected=0,
        closed=0,
        pnl=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        windows=[0.0],
        regimes=[0.0],
    )
    _write_json(
        run_dir / "component_benchmark_report.json",
        {
            "summary": {
                "best_component_label": "LiveOIBreakout",
                "best_component_pnl_usd": 24.31,
                "panteon_beats_best_component": False,
            },
            "components": [
                {
                    "actor_type": "agent",
                    "label": "LiveOIBreakout",
                    "pnl_usd": 24.31,
                    "closed_trades": 8,
                    "wins": 8,
                    "win_rate_pct": 100.0,
                }
            ],
        },
    )
    _write_json(
        run_dir / "candidate_diagnostics.json",
        {
            "rejection_summary": {
                "reason_counts": {"flash:inactive": 1920},
                "flash_reason_counts": {"inactive": 1920},
                "flash_active_reason_counts": {},
            },
            "rejections": {
                "LiveOIBreakout": {
                    "count": 1920,
                    "last_reason": "flash:inactive",
                }
            },
        },
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="single_component__LiveOIBreakout",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    assert summary["candidate"]["activation_gap"] == {
        "enabled": True,
        "standalone_component_label": "LiveOIBreakout",
        "standalone_component_pnl_usd": 24.31,
        "standalone_component_closed_trades": 8,
        "flash_selected_signals": 0,
        "flash_filled_signals": 0,
        "flash_inactive_rejections": 1920,
        "first_inactive_examples": [],
        "activation_gap_reason": "standalone_component_not_flash_active",
    }


def test_collect_matrix_summary_flags_empty_skip_windows(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    for variant in ("skip_0__baseline", "skip_0__panteon3_candidate"):
        _write_variant_run(
            results_root,
            variant,
            selected=2,
            filled=2,
            blocked=0,
            rejected=0,
            closed=1,
            pnl=0.2,
            gross_profit=0.2,
            gross_loss=-0.01,
            windows=[0.2],
            regimes=[0.2],
        )
    for variant in ("skip_480__baseline", "skip_480__panteon3_candidate"):
        _write_variant_run(
            results_root,
            variant,
            selected=0,
            filled=0,
            blocked=0,
            rejected=0,
            closed=0,
            pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            windows=[],
            regimes=[],
        )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=False,
    )

    assert "skip_480__baseline:empty_window" in summary["accounting_warnings"]
    assert "skip_480__panteon3_candidate:empty_window" in summary["accounting_warnings"]
    assert summary["promotion_verdict"]["passed"] is False
    assert "accounting_warnings_present" in summary["promotion_verdict"]["fail_reasons"]


def test_collect_matrix_summary_can_ignore_stale_orphan_variants(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    for variant in ("skip_0__baseline", "skip_0__panteon3_candidate"):
        _write_variant_run(
            results_root,
            variant,
            selected=2,
            filled=2,
            blocked=0,
            rejected=0,
            closed=1,
            pnl=0.2,
            gross_profit=0.2,
            gross_loss=-0.01,
            windows=[0.2],
            regimes=[0.2],
        )
    for variant in ("skip_480__baseline", "skip_480__panteon3_candidate"):
        _write_variant_run(
            results_root,
            variant,
            selected=0,
            filled=0,
            blocked=0,
            rejected=0,
            closed=0,
            pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            windows=[],
            regimes=[],
        )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        allowed_variants=["skip_0__baseline", "skip_0__panteon3_candidate"],
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=False,
    )

    assert [row["variant"] for row in summary["rows"]] == [
        "skip_0__baseline",
        "skip_0__panteon3_candidate",
    ]
    assert summary["accounting_warnings"] == []
    assert summary["promotion_verdict"]["passed"] is True


def test_collect_matrix_summary_recommends_single_component_without_unblocking_router(tmp_path):
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
        windows=[0.0],
        regimes=[0.0],
    )
    run_dir = _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=12,
        filled=12,
        blocked=0,
        rejected=0,
        closed=12,
        pnl=0.25,
        gross_profit=0.30,
        gross_loss=-0.05,
        windows=[0.25],
        regimes=[0.25],
    )
    _write_json(
        run_dir / "component_benchmark_report.json",
        {
            "summary": {
                "panteon_pnl_usd": 0.25,
                "best_component_label": "LiveOIBreakout",
                "best_component_pnl_usd": 24.31,
                "panteon_beats_best_component": False,
            }
        },
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    assert summary["single_component_candidate"] == {
        "recommended": True,
        "label": "LiveOIBreakout",
        "pnl_usd": 24.31,
        "router_candidate_pnl_usd": 0.25,
        "requires_separate_matrix_artifact": True,
    }
    assert summary["promotion_verdict"]["passed"] is True


def test_collect_matrix_summary_accepts_selected_single_component_artifact(tmp_path):
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
        windows=[0.0],
        regimes=[0.0],
    )
    run_dir = _write_variant_run(
        results_root,
        "single_component__LiveOIBreakout",
        selected=12,
        filled=12,
        blocked=0,
        rejected=0,
        closed=12,
        pnl=24.31,
        gross_profit=25.0,
        gross_loss=-0.69,
        windows=[24.31],
        regimes=[24.31],
    )
    _write_json(
        run_dir / "component_benchmark_report.json",
        {
            "summary": {
                "panteon_pnl_usd": 24.31,
                "best_component_label": "LiveOIBreakout",
                "best_component_pnl_usd": 24.31,
                "panteon_beats_best_component": True,
            }
        },
    )

    summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="single_component__LiveOIBreakout",
        min_filled=1,
        min_closed_trades=1,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
    )

    assert summary["candidate"]["variant"] == "single_component__LiveOIBreakout"
    assert summary["single_component_candidate"] == {
        "recommended": False,
        "label": "LiveOIBreakout",
        "pnl_usd": 24.31,
        "router_candidate_pnl_usd": 24.31,
        "requires_separate_matrix_artifact": False,
    }
    assert summary["promotion_verdict"]["passed"] is True


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


def test_collect_matrix_summary_applies_costed_strict_gates(tmp_path):
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
        explicit_costs=0.01,
        turnover_notional=10.0,
        by_symbol={"BTC/USDT": 0.1},
    )
    _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=4,
        filled=4,
        blocked=0,
        rejected=0,
        closed=2,
        pnl=0.2,
        gross_profit=0.4,
        gross_loss=-0.4,
        windows=[0.2],
        regimes=[0.2],
        explicit_costs=0.0,
        turnover_notional=0.0,
        max_drawdown_usd=1.5,
        by_symbol={"ADA/USDT": 0.25, "ETH/USDT": -0.05},
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
        min_profit_factor=2.0,
        min_positive_symbols=2,
        max_drawdown_usd=1.0,
        require_cost_attribution=True,
    )

    assert summary["candidate"]["profit_factor"] == 1.0
    assert summary["candidate"]["positive_symbols"] == 1
    assert summary["candidate"]["max_drawdown_usd"] == 1.5
    assert summary["candidate"]["cost_attribution_present"] is False
    assert summary["promotion_verdict"]["passed"] is False
    assert summary["promotion_verdict"]["fail_reasons"] == [
        "profit_factor 1.0 < min_profit_factor 2.0",
        "positive_symbols 1 < min_positive_symbols 2",
        "max_drawdown_usd 1.5 > max_drawdown_usd 1.0",
        "cost_attribution_missing",
    ]


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


def test_collect_matrix_summary_can_compare_baseline_by_pnl_per_drawdown(tmp_path):
    module = _load_tool()
    results_root = tmp_path / "runs"
    _write_variant_run(
        results_root,
        "baseline",
        selected=30,
        filled=30,
        blocked=0,
        rejected=0,
        closed=15,
        pnl=5.0,
        gross_profit=6.0,
        gross_loss=-1.0,
        windows=[5.0],
        regimes=[5.0],
        max_drawdown_usd=5.0,
    )
    _write_variant_run(
        results_root,
        "panteon3_candidate",
        selected=20,
        filled=20,
        blocked=0,
        rejected=0,
        closed=10,
        pnl=1.0,
        gross_profit=1.2,
        gross_loss=-0.2,
        windows=[1.0],
        regimes=[1.0],
        max_drawdown_usd=0.2,
    )

    raw_summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=20,
        min_closed_trades=10,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
        max_drawdown_usd=0.2,
    )
    risk_adjusted_summary = module.collect_matrix_summary(
        results_root,
        candidate_variant="panteon3_candidate",
        min_filled=20,
        min_closed_trades=10,
        max_blocked=0,
        max_rejected=0,
        require_positive_expectancy=True,
        min_profitable_windows=1,
        min_positive_regimes=1,
        require_beats_baseline=True,
        baseline_compare_mode="pnl_per_drawdown",
        max_drawdown_usd=0.2,
    )

    assert raw_summary["promotion_verdict"]["passed"] is False
    assert raw_summary["promotion_verdict"]["fail_reasons"] == [
        "candidate_pnl_usd 1.0 <= baseline_pnl_usd 5.0",
    ]
    assert risk_adjusted_summary["candidate"]["pnl_per_drawdown"] == pytest.approx(5.0)
    assert risk_adjusted_summary["baseline"]["pnl_per_drawdown"] == pytest.approx(1.0)
    assert risk_adjusted_summary["promotion_gates"]["baseline_compare_mode"] == "pnl_per_drawdown"
    assert risk_adjusted_summary["promotion_verdict"]["passed"] is True


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
