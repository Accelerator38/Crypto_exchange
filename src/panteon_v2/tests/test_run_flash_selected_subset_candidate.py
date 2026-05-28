"""Tests for the selected-subset candidate runner wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_candidate_tool():
    path = ROOT / "tools" / "run_flash_selected_subset_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "run_flash_selected_subset_candidate_tool",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_subset_candidate_overrides_shadow_actor_fallback_min_base_score(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(
        module.matrix,
        "ROUND3_DENY8_ENTRY_REGIME_ARGS",
        (
            "--flash-shadow-actor-fallback-min-base-score",
            "3.0",
            "--other-profile-flag",
            "value",
        ),
    )

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--flash-shadow-actor-fallback-min-base-score",
        "10.0",
    ])

    assert rc == 0
    args = captured["args"]
    idx = args.index("--flash-shadow-actor-fallback-min-base-score")
    assert float(args[idx + 1]) == 10.0
    assert args.count("--flash-shadow-actor-fallback-min-base-score") == 1


def test_selected_subset_candidate_overrides_replay_actor_fallback_min_base_score(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    option = "--flash-shadow-position-replay-actor-fallback-min-base-score"
    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(
        module.matrix,
        "ROUND3_DENY8_ENTRY_REGIME_ARGS",
        (
            option,
            "2.0",
            "--other-profile-flag",
            "value",
        ),
    )

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        option,
        "6.75",
    ])

    assert rc == 0
    args = captured["args"]
    idx = args.index(option)
    assert float(args[idx + 1]) == 6.75
    assert args.count(option) == 1


def test_selected_subset_candidate_overrides_replay_actor_fallback_min_shadow_score(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    option = "--flash-shadow-position-replay-actor-fallback-min-shadow-score"
    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(
        module.matrix,
        "ROUND3_DENY8_ENTRY_REGIME_ARGS",
        (
            option,
            "2.0",
            "--other-profile-flag",
            "value",
        ),
    )

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        option,
        "5.25",
    ])

    assert rc == 0
    args = captured["args"]
    idx = args.index(option)
    assert float(args[idx + 1]) == 5.25
    assert args.count(option) == 1


def test_selected_subset_candidate_overrides_shadow_pnl_lcb_floor(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    option = "--flash-shadow-min-pnl-per-trade-lcb-usd"
    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(
        module.matrix,
        "ROUND3_DENY8_ENTRY_REGIME_ARGS",
        (
            option,
            "-1.0",
            "--other-profile-flag",
            "value",
        ),
    )

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        option,
        "0.0",
    ])

    assert rc == 0
    args = captured["args"]
    idx = args.index(option)
    assert float(args[idx + 1]) == 0.0
    assert args.count(option) == 1


def test_selected_subset_candidate_passes_flash_1_1_control_args(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(
        module.matrix,
        "ROUND3_DENY8_ENTRY_REGIME_ARGS",
        (
            "--flash-min-score-to-trade",
            "4.0",
            "--flash-degradation-window-closed-trades",
            "1",
            "--flash-degradation-min-closed-trades",
            "1",
            "--flash-degradation-max-recent-pnl-usd",
            "-1.0",
            "--flash-degradation-signal-cooldown-bars",
            "720",
            "--flash-degradation-actor-cooldown-bars",
            "72",
            "--v3-shadow-fresh-handoff-max-age-bars",
            "24",
            "--flash-anchor-min-score-advantage",
            "0.0",
        ),
    )

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--flash-min-score-to-trade",
        "3.5",
        "--flash-degradation-window-closed-trades",
        "5",
        "--flash-degradation-min-closed-trades",
        "3",
        "--flash-degradation-max-recent-pnl-usd",
        "-50.0",
        "--flash-degradation-signal-cooldown-bars",
        "240",
        "--flash-degradation-actor-cooldown-bars",
        "36",
        "--v3-shadow-fresh-handoff-max-age-bars",
        "12",
        "--flash-anchor-actor-key",
        "ensemble:Optimal_StaticRotator",
        "--flash-portfolio-actor-key",
        "ensemble:Antonius_conservative",
        "--flash-anchor-min-score-advantage",
        "0.3",
    ])

    assert rc == 0
    args = captured["args"]
    expectations = {
        "--flash-min-score-to-trade": 3.5,
        "--flash-degradation-window-closed-trades": 5,
        "--flash-degradation-min-closed-trades": 3,
        "--flash-degradation-max-recent-pnl-usd": -50.0,
        "--flash-degradation-signal-cooldown-bars": 240,
        "--flash-degradation-actor-cooldown-bars": 36,
        "--v3-shadow-fresh-handoff-max-age-bars": 12,
        "--flash-anchor-min-score-advantage": 0.3,
    }
    for option, expected in expectations.items():
        idx = args.index(option)
        assert float(args[idx + 1]) == expected
        assert args.count(option) == 1
    anchor_idx = args.index("--flash-anchor-actor-key")
    portfolio_idx = args.index("--flash-portfolio-actor-key")
    assert args[anchor_idx + 1] == "ensemble:Optimal_StaticRotator"
    assert args[portfolio_idx + 1] == "ensemble:Antonius_conservative"


def test_selected_subset_candidate_passes_shadow_pnl_lcb_risk_sizing_args(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--enable-flash-shadow-pnl-lcb-risk-sizing",
        "--flash-shadow-pnl-lcb-risk-min-mult",
        "0.4",
        "--flash-shadow-pnl-lcb-risk-floor-usd",
        "0.0",
        "--flash-shadow-pnl-lcb-risk-scale-usd",
        "2.0",
    ])

    assert rc == 0
    args = captured["args"]
    assert "--enable-flash-shadow-pnl-lcb-risk-sizing" in args
    min_idx = args.index("--flash-shadow-pnl-lcb-risk-min-mult")
    floor_idx = args.index("--flash-shadow-pnl-lcb-risk-floor-usd")
    scale_idx = args.index("--flash-shadow-pnl-lcb-risk-scale-usd")
    assert float(args[min_idx + 1]) == 0.4
    assert float(args[floor_idx + 1]) == 0.0
    assert float(args[scale_idx + 1]) == 2.0


def test_selected_subset_candidate_passes_actor_risk_sizing_args(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--enable-flash-actor-risk-sizing",
        "--flash-actor-risk-min-mult",
        "0.5",
        "--flash-actor-risk-max-mult",
        "1.25",
        "--flash-actor-risk-edge-scale-pct",
        "1.0",
    ])

    assert rc == 0
    args = captured["args"]
    assert "--enable-flash-actor-risk-sizing" in args
    min_idx = args.index("--flash-actor-risk-min-mult")
    max_idx = args.index("--flash-actor-risk-max-mult")
    scale_idx = args.index("--flash-actor-risk-edge-scale-pct")
    assert float(args[min_idx + 1]) == 0.5
    assert float(args[max_idx + 1]) == 1.25
    assert float(args[scale_idx + 1]) == 1.0


def test_selected_subset_candidate_passes_pretrade_risk_sizing_args(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--flash-funding-risk-mult-weight",
        "1.5",
        "--flash-funding-risk-mult-cap",
        "0.2",
        "--enable-flash-volatility-risk-sizing",
        "--flash-volatility-risk-target-pct",
        "2.5",
        "--flash-volatility-risk-min-volatility-pct",
        "0.4",
        "--flash-volatility-risk-max-mult",
        "1.25",
        "--enable-flash-overextension-guard",
        "--flash-overextension-lookback-bars",
        "24",
        "--enable-flash-overextension-volatility-normalized",
        "--flash-short-overextension-z-floor",
        "-3.0",
        "--flash-long-overextension-z-ceiling",
        "3.0",
        "--flash-overextension-min-volatility-pct",
        "0.2",
    ])

    assert rc == 0
    args = captured["args"]
    assert "--enable-flash-volatility-risk-sizing" in args
    assert "--enable-flash-overextension-guard" in args
    assert "--enable-flash-overextension-volatility-normalized" in args
    funding_idx = args.index("--flash-funding-risk-mult-weight")
    cap_idx = args.index("--flash-funding-risk-mult-cap")
    target_idx = args.index("--flash-volatility-risk-target-pct")
    vol_floor_idx = args.index("--flash-volatility-risk-min-volatility-pct")
    max_idx = args.index("--flash-volatility-risk-max-mult")
    lookback_idx = args.index("--flash-overextension-lookback-bars")
    short_z_idx = args.index("--flash-short-overextension-z-floor")
    long_z_idx = args.index("--flash-long-overextension-z-ceiling")
    min_vol_idx = args.index("--flash-overextension-min-volatility-pct")
    assert float(args[funding_idx + 1]) == 1.5
    assert float(args[cap_idx + 1]) == 0.2
    assert float(args[target_idx + 1]) == 2.5
    assert float(args[vol_floor_idx + 1]) == 0.4
    assert float(args[max_idx + 1]) == 1.25
    assert int(args[lookback_idx + 1]) == 24
    assert float(args[short_z_idx + 1]) == -3.0
    assert float(args[long_z_idx + 1]) == 3.0
    assert float(args[min_vol_idx + 1]) == 0.2


def test_selected_subset_candidate_passes_degradation_symbol_and_earned_cap_args(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--enable-flash-degradation-symbol-guard",
        "--flash-degradation-symbol-cooldown-bars",
        "48",
        "--flash-degradation-symbol-lookback-bars",
        "2160",
        "--flash-degradation-symbol-window-closed-trades",
        "2",
        "--flash-degradation-symbol-min-closed-trades",
        "2",
        "--flash-degradation-symbol-max-recent-pnl-usd",
        "-10.0",
        "--enable-flash-degradation-recovery",
        "--flash-degradation-recovery-min-closed-trades",
        "3",
        "--flash-degradation-recovery-min-recent-pnl-usd",
        "0.0",
        "--enable-flash-earned-cap-overrides",
    ])

    assert rc == 0
    args = captured["args"]
    assert "--enable-flash-degradation-symbol-guard" in args
    assert "--enable-flash-degradation-recovery" in args
    assert "--enable-flash-earned-cap-overrides" in args
    cooldown_idx = args.index("--flash-degradation-symbol-cooldown-bars")
    lookback_idx = args.index("--flash-degradation-symbol-lookback-bars")
    window_idx = args.index("--flash-degradation-symbol-window-closed-trades")
    min_closed_idx = args.index("--flash-degradation-symbol-min-closed-trades")
    max_pnl_idx = args.index("--flash-degradation-symbol-max-recent-pnl-usd")
    recovery_closed_idx = args.index(
        "--flash-degradation-recovery-min-closed-trades"
    )
    recovery_pnl_idx = args.index(
        "--flash-degradation-recovery-min-recent-pnl-usd"
    )
    assert int(args[cooldown_idx + 1]) == 48
    assert int(args[lookback_idx + 1]) == 2160
    assert int(args[window_idx + 1]) == 2
    assert int(args[min_closed_idx + 1]) == 2
    assert float(args[max_pnl_idx + 1]) == -10.0
    assert int(args[recovery_closed_idx + 1]) == 3
    assert float(args[recovery_pnl_idx + 1]) == 0.0


def test_selected_subset_candidate_passes_terminal_deny_signal_key(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    signal_key = "ensemble:Optimal_StaticRotator|TRX/USDT|FUT_SHORT_HALF"
    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--flash-terminal-deny-signal-key",
        signal_key,
    ])

    assert rc == 0
    args = captured["args"]
    idx = args.index("--flash-terminal-deny-signal-key")
    assert args[idx + 1] == signal_key


def test_selected_subset_candidate_passes_context_terminal_deny_signal_keys(
    tmp_path,
    monkeypatch,
):
    module = _load_candidate_tool()
    manifest = tmp_path / "manifest.json"
    manifest_key = "ensemble:Solo_MomentumScalper|APT/USDT|FUT_LONG_FULL|bullish"
    cli_key = "ensemble:Solo_MomentumScalper|DOT/USDT|FUT_LONG_FULL|bearish"
    manifest.write_text(
        json.dumps({
            "score_boosts": [],
            "context_score_boosts": [],
            "context_terminal_deny_signal_keys": [manifest_key],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_risk_mult_overrides": [],
        }),
        encoding="utf-8",
    )
    captured: dict[str, list[str]] = {}

    def fake_runner_main(args):
        captured["args"] = list(args)
        return 0

    monkeypatch.setattr(module.runner, "main", fake_runner_main)
    monkeypatch.setattr(module.matrix, "ROUND3_DENY8_ENTRY_REGIME_ARGS", ())

    rc = module.main([
        "--manifest",
        str(manifest),
        "--results-root",
        str(tmp_path / "Results"),
        "--years",
        "2022",
        "--flash-terminal-deny-context-signal-key",
        cli_key,
    ])

    assert rc == 0
    args = captured["args"]
    values = [
        args[index + 1]
        for index, item in enumerate(args)
        if item == "--flash-terminal-deny-context-signal-key"
    ]
    assert values == [manifest_key, cli_key]
