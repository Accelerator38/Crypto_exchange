"""Tests for the Panteon Flash profitability matrix runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_matrix_tool():
    path = ROOT / "tools" / "run_panteon_flash_profitability_matrix.py"
    spec = importlib.util.spec_from_file_location("panteon_flash_matrix_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overextension_calibration_dry_run_writes_z_sweep_rows(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--overextension-calibration",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
        "--calibration-max-bars",
        "240",
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    rows = data["experiments"]
    assert [row["name"] for row in rows] == [
        "baseline_no_overextension",
        "raw_overextension_8pct",
        "vol_z_2_0",
        "vol_z_2_5",
        "vol_z_3_0",
        "vol_z_4_0",
        "vol_z_5_0",
        "vol_z_6_0",
    ]
    assert {row["tier"] for row in rows} == {"2026_calibration"}
    baseline_args = module.OVEREXTENSION_CALIBRATION_EXPERIMENTS[0]["args"]
    assert "--flash-promotion-min-full-closed-trades" in baseline_args
    assert "--flash-promotion-min-latest-closed-trades" in baseline_args


def test_default_dry_run_includes_proven_solo_position_handoff_candidate(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    names = [row["name"] for row in data["experiments"]]
    assert "proven_solo_position_handoff_age24_shadow5" in names
    assert "proven_solo_stale_position_exit_age168_shadow5" in names
    assert "proven_solo_stale_exit_quality60_age168_shadow5" in names
    assert "proven_solo_nonpositive_handoff_stale_exit_age168_shadow5" in names
    assert (
        "proven_solo_portfolio_nonpositive_handoff_stale_exit_age168_shadow5"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_nonpositive_handoff_stale_exit_age168_shadow5"
        in names
    )
    assert "proven_solo_portfolio_hard_solo_momentum_cap10" in names
    assert "proven_solo_portfolio_hard_solo_momentum_cap10_shadow_pnl_lcb" in names
    assert "proven_solo_portfolio_hard_solo_momentum_cap10_shadow_pnl_lcb_penalty" in names
    assert "proven_solo_portfolio_hard_solo_momentum_cap10_manifest_pnl_lcb" in names
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_selected_deny_probe"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_selected_deny_probe_v2"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_oos2025"
        in names
    )
    assert (
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_terminal_atom"
        in names
    )
    assert "proven_solo_portfolio_hard_solo_momentum_cap10_no_neutral_open" in names
    assert "proven_solo_portfolio_hard_solo_momentum_cap10_wider_execution" in names
    handoff_args = module.PROVEN_SOLO_POSITION_HANDOFF_AGE24_SHADOW5_ARGS
    assert "--enable-flash-prefer-proven-solo-player-wrappers" in handoff_args
    assert "--enable-v3-shadow-fresh-handoff" in handoff_args
    assert "--v3-shadow-fresh-handoff-max-age-bars" in handoff_args
    shadow_min_index = handoff_args.index("--flash-shadow-min-closed-trades")
    assert handoff_args[shadow_min_index + 1] == "5"
    stale_exit_args = module.PROVEN_SOLO_STALE_POSITION_EXIT_AGE168_SHADOW5_ARGS
    assert "--enable-flash-stale-position-exit" in stale_exit_args
    age_index = stale_exit_args.index("--flash-stale-position-exit-max-age-bars")
    assert stale_exit_args[age_index + 1] == "168"
    quality_args = module.PROVEN_SOLO_STALE_EXIT_QUALITY60_AGE168_SHADOW5_ARGS
    win_rate_index = quality_args.index("--flash-shadow-min-win-rate-pct")
    assert quality_args[win_rate_index + 1] == "60.0"
    nonpositive_args = (
        module.PROVEN_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    )
    assert "--v3-shadow-fresh-handoff-allow-nonpositive-unrealized" in nonpositive_args
    portfolio_args = (
        module.PROVEN_SOLO_PORTFOLIO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    )
    assert portfolio_args.count("--flash-portfolio-actor-key") == 2
    assert "ensemble:Solo_MomentumScalper" in portfolio_args
    assert "ensemble:Solo_LiveCrashHunter" in portfolio_args
    portfolio_hard_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_NONPOSITIVE_HANDOFF_STALE_EXIT_AGE168_SHADOW5_ARGS
    )
    assert "--enable-flash-prefer-solo-player-wrappers" in portfolio_hard_args
    assert "agent:LiveOIBreakout" in portfolio_hard_args
    cap10_args = module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    cap_index = cap10_args.index("--flash-promoted-actor-cap-override")
    assert cap10_args[cap_index + 1] == "ensemble:Solo_MomentumScalper=10"
    shadow_lcb_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_ARGS
    )
    assert "--flash-shadow-min-pnl-per-trade-lcb-usd" in shadow_lcb_args
    assert "--flash-shadow-pnl-per-trade-lcb-z" in shadow_lcb_args
    shadow_penalty_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SHADOW_PNL_LCB_PENALTY_ARGS
    )
    assert "--flash-shadow-pnl-per-trade-lcb-penalty-floor-usd" in shadow_penalty_args
    assert "--flash-shadow-pnl-per-trade-lcb-penalty-weight" in shadow_penalty_args
    pnl_lcb_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_MANIFEST_PNL_LCB_ARGS
    )
    assert "--enable-flash-promotion-manifest" in pnl_lcb_args
    assert "--flash-promotion-min-full-pnl-per-trade-lcb-pct" in pnl_lcb_args
    assert "--flash-promotion-min-latest-pnl-per-trade-lcb-pct" in pnl_lcb_args
    assert "--flash-promotion-pnl-per-trade-lcb-z" in pnl_lcb_args
    selected_deny_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_ARGS
    )
    assert selected_deny_args.count("--flash-deny-signal-key") == 12
    assert (
        "ensemble:Solo_MomentumScalper|TRX/USDT|FUT_SHORT_FULL"
        in selected_deny_args
    )
    selected_deny_v2_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_SELECTED_DENY_PROBE_V2_ARGS
    )
    assert selected_deny_v2_args.count("--flash-deny-signal-key") == 27
    assert (
        "ensemble:Optimal_StaticRotator|DOT/USDT|FUT_LONG_FULL"
        in selected_deny_v2_args
    )
    generated_deny = next(
        experiment
        for experiment in module.EXPERIMENTS
        if experiment["name"]
        == "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny"
    )
    assert generated_deny["args"] == module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_ARGS
    assert (
        generated_deny["selected_deny_source_experiment"]
        == "proven_solo_portfolio_hard_solo_momentum_cap10"
    )
    assert generated_deny["selected_deny_max_keys"] == 50
    generated_deny_round2 = next(
        experiment
        for experiment in module.EXPERIMENTS
        if experiment["name"]
        == "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2"
    )
    assert generated_deny_round2["selected_deny_source_experiments"] == [
        "proven_solo_portfolio_hard_solo_momentum_cap10",
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
    ]
    generated_deny_round2_oos = next(
        experiment
        for experiment in module.EXPERIMENTS
        if experiment["name"]
        == "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_oos2025"
    )
    assert generated_deny_round2_oos["selected_deny_source_tier"] == "2025"
    assert generated_deny_round2_oos["selected_deny_source_experiments"] == [
        "proven_solo_portfolio_hard_solo_momentum_cap10",
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
    ]
    generated_round2_terminal_atom = next(
        experiment
        for experiment in module.EXPERIMENTS
        if experiment["name"]
        == "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny_round2_terminal_atom"
    )
    assert generated_round2_terminal_atom["selected_deny_source_tier"] == "2025"
    assert generated_round2_terminal_atom["selected_deny_source_experiments"] == [
        "proven_solo_portfolio_hard_solo_momentum_cap10",
        "proven_solo_portfolio_hard_solo_momentum_cap10_generated_selected_deny",
    ]
    assert "--flash-terminal-deny-signal-key" in generated_round2_terminal_atom["args"]
    assert (
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL"
        in generated_round2_terminal_atom["args"]
    )
    no_neutral_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_NO_NEUTRAL_ARGS
    )
    regime_index = no_neutral_args.index("--flash-denied-open-regime")
    assert no_neutral_args[regime_index + 1] == "neutral"
    wider_args = (
        module.PROVEN_SOLO_PORTFOLIO_HARD_SOLO_MOMENTUM_CAP10_WIDER_EXECUTION_ARGS
    )
    opens_index = wider_args.index("--max-new-opens-per-bar")
    risk_index = wider_args.index("--risk-max-open-positions")
    assert wider_args[opens_index + 1] == "3"
    assert wider_args[risk_index + 1] == "16"


def test_default_dry_run_includes_hard_solo_actor_guard_candidate(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "solo_hard_actor_guard_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("solo_hard_actor_guard_age24_shadow5", "2026_h1")]

    args = module.SOLO_HARD_ACTOR_GUARD_AGE24_SHADOW5_ARGS
    assert "--enable-flash-prefer-solo-player-wrappers" in args
    assert "--enable-flash-degradation-actor-guard" in args
    assert "--enable-v3-shadow-fresh-handoff" in args


def test_symbol_only_hard_solo_actor_guard_candidate_disables_actor_fallback(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_hard_solo_actor_guard_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_hard_solo_actor_guard_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_HARD_SOLO_ACTOR_GUARD_AGE24_SHADOW5_ARGS
    assert "--enable-flash-prefer-solo-player-wrappers" in args
    assert "--enable-flash-degradation-actor-guard" in args
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_base_fallback_candidate_allows_only_anchor_fallback(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_base_fallback_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_base_fallback_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_AGE24_SHADOW5_ARGS
    assert "--enable-flash-shadow-base-fallback-confirmation" in args
    assert "--flash-shadow-base-fallback-actor-key" in args
    key_index = args.index("--flash-shadow-base-fallback-actor-key")
    assert args[key_index + 1] == "ensemble:Solo_MomentumScalper"
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_base_fallback_min1_candidate_lowers_anchor_floor(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_base_fallback_min1_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_base_fallback_min1_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_BASE_FALLBACK_MIN1_AGE24_SHADOW5_ARGS
    floor_index = args.index("--flash-shadow-actor-fallback-min-base-score")
    assert args[floor_index + 1] == "1.0"
    assert "--flash-shadow-base-fallback-actor-key" in args
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_min_score_2_5_candidate_lowers_trade_gate(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_min_score_2_5_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_min_score_2_5_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_MIN_SCORE_2_5_AGE24_SHADOW5_ARGS
    score_index = args.index("--flash-min-score-to-trade")
    assert args[score_index + 1] == "2.5"
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_dominance_candidate_sets_anchor_contract(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_dominance_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_dominance_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_DOMINANCE_AGE24_SHADOW5_ARGS
    assert "--flash-anchor-actor-key" in args
    key_index = args.index("--flash-anchor-actor-key")
    assert args[key_index + 1] == "ensemble:Solo_MomentumScalper"
    floor_index = args.index("--flash-anchor-min-score-to-trade")
    assert args[floor_index + 1] == "1.0"
    margin_index = args.index("--flash-anchor-min-score-advantage")
    assert args[margin_index + 1] == "1.0"
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_dominance_no_actor_guard_candidate(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_dominance_no_actor_guard_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_dominance_no_actor_guard_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_DOMINANCE_NO_ACTOR_GUARD_AGE24_SHADOW5_ARGS
    assert "--flash-anchor-actor-key" in args
    assert "--enable-flash-degradation-actor-guard" not in args
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_shadow_floor_candidate_relaxes_anchor_shadow_gate(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_shadow_floor_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_shadow_floor_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_SHADOW_FLOOR_AGE24_SHADOW5_ARGS
    assert "--flash-anchor-actor-key" in args
    floor_index = args.index("--flash-anchor-shadow-min-score")
    assert args[floor_index + 1] == "-0.1"
    assert "--enable-flash-degradation-actor-guard" in args
    assert "--enable-flash-shadow-actor-fallback-confirmation" not in args


def test_symbol_only_anchor_actor_cooldown_candidate_limits_actor_kill_switch(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_actor_cooldown_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_actor_cooldown_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_ACTOR_COOLDOWN_AGE24_SHADOW5_ARGS
    cooldown_index = args.index("--flash-degradation-actor-cooldown-bars")
    assert args[cooldown_index + 1] == "72"
    assert "--enable-flash-degradation-actor-guard" in args
    assert "--flash-anchor-shadow-min-score" in args


def test_symbol_only_anchor_actor_regime_guard_candidate_scopes_actor_guard(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "symbol_only_anchor_actor_regime_guard_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("symbol_only_anchor_actor_regime_guard_age24_shadow5", "2026_h1")]

    args = module.SYMBOL_ONLY_ANCHOR_ACTOR_REGIME_GUARD_AGE24_SHADOW5_ARGS
    scope_index = args.index("--flash-degradation-actor-scope")
    assert args[scope_index + 1] == "actor_regime"
    assert "--enable-flash-degradation-actor-guard" in args
    assert "--flash-degradation-actor-cooldown-bars" not in args


def test_dry_run_can_filter_one_experiment_and_tier(tmp_path):
    module = _load_matrix_tool()
    reports_dir = tmp_path / "Reports"
    results_root = tmp_path / "Results"

    rc = module.main([
        "--dry-run",
        "--experiment",
        "proven_solo_position_handoff_age24_shadow5",
        "--tier",
        "2026_h1",
        "--reports-dir",
        str(reports_dir),
        "--results-root",
        str(results_root),
    ])

    assert rc == 0
    data = json.loads(
        (reports_dir / "panteon_flash_profitability_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (row["name"], row["tier"])
        for row in data["experiments"]
    ] == [("proven_solo_position_handoff_age24_shadow5", "2026_h1")]


def test_matrix_stops_experiment_after_walk_forward_control_failure(monkeypatch, tmp_path):
    module = _load_matrix_tool()
    calls: list[tuple[str, str]] = []
    rows_by_tier = {
        "2026_h1": {
            "pnl_pct": 8.0,
            "max_drawdown_pct": 1.0,
            "panteon_alpha_pct": 2.0,
            "beats_best_component": True,
            "output_dir": "h1",
            "bars_processed": 3312,
        },
        "2025": {
            "pnl_pct": -9.0,
            "max_drawdown_pct": 12.0,
            "panteon_alpha_pct": -28.0,
            "beats_best_component": False,
            "output_dir": "2025",
            "bars_processed": 8760,
        },
        "full_2022_2026": {
            "pnl_pct": 99.0,
            "max_drawdown_pct": 1.0,
            "panteon_alpha_pct": 99.0,
            "beats_best_component": True,
            "output_dir": "full",
            "bars_processed": 100,
        },
    }

    def fake_runner_command(*, experiment_name, tier_args, **_kwargs):
        if "--max-bars" in tier_args:
            tier_name = "2026_h1"
        elif "2025" in tier_args:
            tier_name = "2025"
        else:
            tier_name = "full_2022_2026"
        calls.append((experiment_name, tier_name))
        return ["runner", tier_name]

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_extract(_output_dir):
        return dict(rows_by_tier[calls[-1][1]])

    monkeypatch.setattr(module, "_runner_command", fake_runner_command)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Proc())
    monkeypatch.setattr(module, "_find_declared_output_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(module, "_extract_metrics", fake_extract)

    args = module._parse_args([
        "--results-root",
        str(tmp_path / "Results"),
        "--reports-dir",
        str(tmp_path / "Reports"),
    ])
    rows = module._run_matrix(
        args,
        experiments=[{
            "name": "candidate",
            "args": [],
        }],
        tiers=module.TIERS,
    )

    assert [call[1] for call in calls] == ["2026_h1", "2025"]
    assert len(rows) == 2
    assert rows[1]["stopped_early"] is True
    assert rows[1]["stop_reason"] == "panteon_under_best_component_on_2025"
    assert rows[1]["walk_forward_gate"] == "failed"
    assert rows[1]["walk_forward_gate_tier"] == "2025"


def test_extract_metrics_includes_flash_quality_contract(tmp_path):
    module = _load_matrix_tool()
    output = tmp_path / "run"
    output.mkdir()
    (output / "status.json").write_text(
        json.dumps({
            "live_session": {"panteon_owned_pnl_pct": 14.0},
            "panteon_max_drawdown_pct": 6.0,
        }),
        encoding="utf-8",
    )
    (output / "component_benchmark_report.json").write_text(
        json.dumps({
            "summary": {
                "panteon_alpha_pct": 3.0,
                "panteon_beats_best_component": True,
                "best_component_pnl_pct": 11.0,
            },
            "components": [
                {"actor_type": "agent", "label": "AgentA", "pnl_pct": 10.0},
                {"actor_type": "player", "label": "PlayerA", "pnl_pct": 11.0},
            ],
        }),
        encoding="utf-8",
    )
    (output / "flash_attribution_summary.json").write_text(
        json.dumps({
            "summary": {
                "flash_decisions": 10,
                "no_trade_decisions": 4,
                "selected_signals": 5,
                "executable_selected_signals": 3,
                "selected_filtered_before_execution": 2,
                "filled_signals": 2,
                "missing_execution_signals": 1,
                "filter_detail_counts": {"duplicate_open": 2},
            }
        }),
        encoding="utf-8",
    )
    (output / "causal_entry_decisions.jsonl").write_text(
        "\n".join([
            json.dumps({
                "bar": 1,
                "flash_selected_actors_by_symbol": {
                    "BTC": "AgentA",
                    "ETH": "NoTrade",
                },
            }),
            json.dumps({
                "bar": 2,
                "flash_selected_actors_by_symbol": {
                    "BTC": "AgentB",
                    "ETH": "NoTrade",
                },
            }),
            json.dumps({
                "bar": 3,
                "flash_selected_actors_by_symbol": {
                    "BTC": "AgentB",
                    "ETH": "AgentC",
                },
            }),
        ])
        + "\n",
        encoding="utf-8",
    )
    (output / "walk_forward_report.json").write_text(
        json.dumps({
            "by_regime": {
                "bullish": {"closed_trades": 2, "net_pnl": 5.0},
                "bearish": {"closed_trades": 30, "net_pnl": -1.0},
            }
        }),
        encoding="utf-8",
    )
    (output / "standalone_vs_flash_selected_report.json").write_text(
        json.dumps({
            "summary": {
                "total_selection_alpha_pct": -14.4,
                "total_standalone_pnl_pct": 18.15,
                "total_flash_selected_pnl_pct": 3.75,
            }
        }),
        encoding="utf-8",
    )

    metrics = module._extract_metrics(output)

    assert metrics["panteon_advantage"] == 0.5
    assert metrics["dominance_equity_ratio"] == 114.0 / 111.0
    assert metrics["panteon_lcb_dominance"] is True
    assert metrics["actor_switches"] == 2
    assert metrics["no_trade_decision_share_pct"] == 40.0
    assert metrics["flash_selected_signals"] == 5
    assert metrics["flash_executable_selected_signals"] == 3
    assert metrics["flash_selected_filtered_before_execution"] == 2
    assert metrics["flash_missing_execution_signals"] == 1
    assert metrics["flash_filtered_before_execution_share_pct"] == 40.0
    assert metrics["flash_executable_fill_rate_pct"] == pytest.approx(66.6666667)
    assert metrics["flash_filter_detail_counts"] == {"duplicate_open": 2}
    assert metrics["ensemble_lift_pct"] == 1.0
    assert metrics["ensemble_lift_pass"] is True
    assert metrics["regime_floor_pass"] is False
    assert metrics["regime_floor_failed_regimes"] == ["bearish"]
    assert metrics["regime_floor_insufficient_regimes"] == ["bullish"]
    assert metrics["regime_floor_min_closed_trades"] == 30
    assert metrics["churn_budget_pass"] is True
    assert metrics["churn_budget_max_actor_switches_per_day"] == 24.0
    assert metrics["standalone_vs_flash_selection_alpha_pct"] == -14.4
    assert metrics["standalone_vs_flash_standalone_pnl_pct"] == 18.15
    assert metrics["standalone_vs_flash_selected_pnl_pct"] == 3.75


def test_regime_floor_does_not_fail_on_under_sampled_negative_regime():
    module = _load_matrix_tool()

    metrics = module._regime_floor_metrics({
        "by_regime": {
            "bullish": {"closed_trades": 2, "net_pnl": -10.0},
            "bearish": {"closed_trades": 35, "net_pnl": 5.0},
        }
    })

    assert metrics["regime_floor_pass"] is True
    assert metrics["regime_floor_failed_regimes"] == []
    assert metrics["regime_floor_insufficient_regimes"] == ["bullish"]


def test_manifest_baseline_source_routes_pnl_lcb_to_cap10_baseline():
    module = _load_matrix_tool()

    assert module._manifest_baseline_experiment("manifest_cap1") == "baseline_reserve_cap"
    assert (
        module._manifest_baseline_experiment(
            "proven_solo_portfolio_hard_solo_momentum_cap10_manifest_pnl_lcb"
        )
        == "proven_solo_portfolio_hard_solo_momentum_cap10"
    )
    assert module._manifest_baseline_experiment("baseline_reserve_cap") == ""


def test_flash_selected_deny_key_args_from_report_filters_worst_closed_keys():
    module = _load_matrix_tool()

    args = module._flash_selected_deny_key_args_from_report(
        {
            "rows": [
                {
                    "actor_key": "ensemble:Solo_MomentumScalper",
                    "symbol": "DOGE/USDT",
                    "action": "FUT_LONG_FULL",
                    "closed_trades": 1,
                    "realized_pnl_usd": -6.0,
                },
                {
                    "actor_key": "ensemble:Solo_MomentumScalper",
                    "symbol": "ADA/USDT",
                    "action": "FUT_SHORT_FULL",
                    "closed_trades": 2,
                    "realized_pnl_usd": -6.0,
                },
                {
                    "actor_key": "ensemble:Solo_MomentumScalper",
                    "symbol": "DOGE/USDT",
                    "action": "FUT_SHORT_FULL",
                    "closed_trades": 5,
                    "realized_pnl_usd": 34.0,
                },
                {
                    "actor_key": "agent:Bad",
                    "symbol": "ETH/USDT",
                    "action": "FUT_SHORT_HALF",
                    "closed_trades": 0,
                    "realized_pnl_usd": -5.0,
                },
                {
                    "actor_key": "",
                    "symbol": "ETH/USDT",
                    "action": "FUT_LONG_FULL",
                    "closed_trades": 1,
                    "realized_pnl_usd": -10.0,
                },
            ],
        },
        max_realized_pnl_usd=-0.1,
        min_closed_trades=1,
        max_keys=10,
    )

    assert args == [
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ADA/USDT|FUT_SHORT_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|DOGE/USDT|FUT_LONG_FULL",
    ]


def test_flash_selected_deny_key_args_from_report_caps_key_count():
    module = _load_matrix_tool()

    args = module._flash_selected_deny_key_args_from_report(
        {
            "rows": [
                {
                    "actor_key": "ensemble:A",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "closed_trades": 1,
                    "realized_pnl_usd": -1.0,
                },
                {
                    "actor_key": "ensemble:B",
                    "symbol": "ETH/USDT",
                    "action": "FUT_SHORT_FULL",
                    "closed_trades": 1,
                    "realized_pnl_usd": -3.0,
                },
            ],
        },
        max_keys=1,
    )

    assert args == [
        "--flash-deny-signal-key",
        "ensemble:B|ETH/USDT|FUT_SHORT_FULL",
    ]


def test_flash_selected_deny_key_args_from_report_ignores_invalid_payloads():
    module = _load_matrix_tool()

    assert module._flash_selected_deny_key_args_from_report({}) == []
    assert module._flash_selected_deny_key_args_from_report({"rows": {}}) == []


def test_flash_shadow_deny_key_args_from_report_filters_whitelisted_losers():
    module = _load_matrix_tool()

    args = module._flash_shadow_deny_key_args_from_report(
        {
            "rows": [
                {
                    "signal_key": "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
                    "actor_key": "ensemble:Solo_LiveCrashHunter",
                    "full_closed_trades": 2,
                    "full_pnl_usd": -7.1,
                },
                {
                    "signal_key": "agent:MomentumScalper|APT/USDT|FUT_SHORT_FULL",
                    "actor_key": "agent:MomentumScalper",
                    "full_closed_trades": 50,
                    "full_pnl_usd": 21.9,
                },
                {
                    "signal_key": "ensemble:MeanRevResearch|APT/USDT|FUT_SHORT_FULL",
                    "actor_key": "ensemble:MeanRevResearch",
                    "full_closed_trades": 4,
                    "full_pnl_usd": -20.0,
                },
                {
                    "signal_key": "agent:PlayerFunding|ADA/USDT|FUT_SHORT_FULL",
                    "actor_key": "agent:PlayerFunding",
                    "full_closed_trades": 1,
                    "full_pnl_usd": -5.0,
                },
            ]
        },
        max_full_pnl_usd=-0.1,
        min_full_closed_trades=2,
        max_keys=10,
        allowed_actor_keys=[
            "ensemble:Solo_LiveCrashHunter",
            "agent:PlayerFunding",
        ],
    )

    assert args == [
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
    ]


def test_generated_selected_deny_args_loads_source_attribution(tmp_path):
    module = _load_matrix_tool()
    output_dir = tmp_path / "baseline"
    output_dir.mkdir()
    (output_dir / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "actor_key": "ensemble:Solo_MomentumScalper",
                        "symbol": "ADA/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args, warning = module._generated_selected_deny_args(
        {
            "selected_deny_source_experiment": "baseline_exp",
            "selected_deny_max_realized_pnl_usd": -0.1,
            "selected_deny_min_closed_trades": 1,
            "selected_deny_max_keys": 10,
        },
        tier_name="2025",
        output_dirs_by_experiment_tier={("baseline_exp", "2025"): output_dir},
    )

    assert warning == ""
    assert args == [
        "--flash-deny-signal-key",
        "ensemble:Solo_MomentumScalper|ADA/USDT|FUT_LONG_FULL",
    ]


def test_generated_selected_deny_args_warns_when_source_is_missing():
    module = _load_matrix_tool()

    args, warning = module._generated_selected_deny_args(
        {"selected_deny_source_experiment": "baseline_exp"},
        tier_name="2025",
        output_dirs_by_experiment_tier={},
    )

    assert args == []
    assert warning == "selected_deny_source_missing:baseline_exp:2025"


def test_generated_selected_deny_args_unions_multiple_source_reports(tmp_path):
    module = _load_matrix_tool()
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -2.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (source_b / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -1.0,
                    },
                    {
                        "actor_key": "ensemble:B",
                        "symbol": "ETH/USDT",
                        "action": "FUT_SHORT_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -3.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    args, warning = module._generated_selected_deny_args(
        {
            "selected_deny_source_experiments": ["source_a", "source_b"],
            "selected_deny_max_realized_pnl_usd": -0.1,
            "selected_deny_min_closed_trades": 1,
            "selected_deny_max_keys": 10,
        },
        tier_name="2025",
        output_dirs_by_experiment_tier={
            ("source_a", "2025"): source_a,
            ("source_b", "2025"): source_b,
        },
    )

    assert warning == ""
    assert args == [
        "--flash-deny-signal-key",
        "ensemble:A|BTC/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:B|ETH/USDT|FUT_SHORT_FULL",
    ]


def test_generated_selected_deny_args_can_use_fixed_source_tier(tmp_path):
    module = _load_matrix_tool()
    output_dir = tmp_path / "source_2025"
    output_dir.mkdir()
    (output_dir / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args, warning = module._generated_selected_deny_args(
        {
            "selected_deny_source_experiment": "source",
            "selected_deny_source_tier": "2025",
        },
        tier_name="2026_h1",
        output_dirs_by_experiment_tier={("source", "2025"): output_dir},
    )

    assert warning == ""
    assert args == [
        "--flash-deny-signal-key",
        "ensemble:A|BTC/USDT|FUT_LONG_FULL",
    ]


def test_generated_selected_deny_args_can_include_shadow_full_losers(tmp_path):
    module = _load_matrix_tool()
    output_dir = tmp_path / "source_2025"
    output_dir.mkdir()
    (output_dir / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 1,
                        "realized_pnl_usd": -2.0,
                    },
                    {
                        "actor_key": "ensemble:Solo_LiveCrashHunter",
                        "symbol": "ATOM/USDT",
                        "action": "FUT_SHORT_FULL",
                        "closed_trades": 2,
                        "realized_pnl_usd": 4.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "flash_signal_key_shadow_report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "signal_key": "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
                        "actor_key": "ensemble:Solo_LiveCrashHunter",
                        "full_closed_trades": 2,
                        "full_pnl_usd": -7.1,
                    },
                    {
                        "signal_key": "ensemble:MeanRevResearch|APT/USDT|FUT_SHORT_FULL",
                        "actor_key": "ensemble:MeanRevResearch",
                        "full_closed_trades": 4,
                        "full_pnl_usd": -20.0,
                    },
                    {
                        "signal_key": "ensemble:Solo_LiveCrashHunter|XRP/USDT|FUT_SHORT_FULL",
                        "actor_key": "ensemble:Solo_LiveCrashHunter",
                        "full_closed_trades": 3,
                        "full_pnl_usd": -9.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    args, warning = module._generated_selected_deny_args(
        {
            "selected_deny_source_experiment": "source",
            "selected_deny_source_tier": "2025",
            "selected_deny_include_shadow_report": True,
            "selected_deny_shadow_require_selected_history": True,
            "selected_deny_shadow_actor_keys": ["ensemble:Solo_LiveCrashHunter"],
        },
        tier_name="2026_h1",
        output_dirs_by_experiment_tier={("source", "2025"): output_dir},
    )

    assert warning == ""
    assert args == [
        "--flash-deny-signal-key",
        "ensemble:A|BTC/USDT|FUT_LONG_FULL",
        "--flash-deny-signal-key",
        "ensemble:Solo_LiveCrashHunter|ATOM/USDT|FUT_SHORT_FULL",
    ]


def test_signal_key_train_eval_rows_compare_oos_to_training_history():
    module = _load_matrix_tool()

    rows = module._signal_key_train_eval_rows(
        [
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "selected_signals": 2,
                        "closed_trades": 2,
                        "realized_pnl_usd": 6.0,
                    },
                    {
                        "actor_key": "agent:B",
                        "symbol": "ETH/USDT",
                        "action": "FUT_SHORT_FULL",
                        "selected_signals": 1,
                        "closed_trades": 1,
                        "realized_pnl_usd": -3.0,
                    },
                ]
            },
            {
                "rows": [
                    {
                        "actor_key": "ensemble:A",
                        "symbol": "BTC/USDT",
                        "action": "FUT_LONG_FULL",
                        "selected_signals": 1,
                        "closed_trades": 1,
                        "realized_pnl_usd": 4.0,
                    }
                ]
            },
        ],
        {
            "rows": [
                {
                    "actor_key": "ensemble:A",
                    "symbol": "BTC/USDT",
                    "action": "FUT_LONG_FULL",
                    "selected_signals": 1,
                    "closed_trades": 1,
                    "realized_pnl_usd": -2.0,
                },
                {
                    "actor_key": "agent:C",
                    "symbol": "XRP/USDT",
                    "action": "FUT_SHORT_FULL",
                    "selected_signals": 2,
                    "closed_trades": 2,
                    "realized_pnl_usd": 5.0,
                },
            ]
        },
    )

    assert rows == [
        {
            "signal_key": "ensemble:A|BTC/USDT|FUT_LONG_FULL",
            "actor_key": "ensemble:A",
            "symbol": "BTC/USDT",
            "action": "FUT_LONG_FULL",
            "train_selected_signals": 3,
            "train_closed_trades": 3,
            "train_realized_pnl_usd": 10.0,
            "train_pnl_per_trade_usd": pytest.approx(3.3333333333),
            "eval_selected_signals": 1,
            "eval_closed_trades": 1,
            "eval_realized_pnl_usd": -2.0,
            "eval_pnl_per_trade_usd": -2.0,
            "pnl_delta_usd": -12.0,
            "supported_by_train": True,
        },
        {
            "signal_key": "agent:C|XRP/USDT|FUT_SHORT_FULL",
            "actor_key": "agent:C",
            "symbol": "XRP/USDT",
            "action": "FUT_SHORT_FULL",
            "train_selected_signals": 0,
            "train_closed_trades": 0,
            "train_realized_pnl_usd": 0.0,
            "train_pnl_per_trade_usd": 0.0,
            "eval_selected_signals": 2,
            "eval_closed_trades": 2,
            "eval_realized_pnl_usd": 5.0,
            "eval_pnl_per_trade_usd": 2.5,
            "pnl_delta_usd": 5.0,
            "supported_by_train": False,
        },
    ]


def test_should_stop_early_on_contract_failures():
    module = _load_matrix_tool()

    assert (
        module._should_stop_early(
            "2025",
            {
                "beats_best_component": True,
                "panteon_lcb_dominance": False,
            },
        )
        == "panteon_lcb_dominance_failed_on_2025"
    )
    assert (
        module._should_stop_early(
            "2025",
            {
                "beats_best_component": True,
                "panteon_lcb_dominance": True,
                "ensemble_lift_pass": True,
                "regime_floor_pass": False,
            },
        )
        == "regime_floor_failed_on_2025"
    )
    assert (
        module._should_stop_early(
            "2025",
            {
                "beats_best_component": True,
                "panteon_lcb_dominance": True,
                "ensemble_lift_pass": False,
            },
        )
        == "ensemble_lift_failed_on_2025"
    )
    assert (
        module._should_stop_early(
            "2025",
            {
                "beats_best_component": True,
                "panteon_lcb_dominance": True,
                "ensemble_lift_pass": True,
                "regime_floor_pass": True,
                "churn_budget_pass": False,
            },
        )
        == "churn_budget_failed_on_2025"
    )
