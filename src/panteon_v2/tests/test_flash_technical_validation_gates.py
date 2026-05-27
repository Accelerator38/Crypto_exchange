from __future__ import annotations

import json

import pytest

from panteon_v2.analysis.flash_technical_validation_gates import (
    MODE_ATR_SIZING,
    MODE_HARD_GATE,
    build_technical_validation_report,
    equity_windows_from_curve,
    evaluate_equity_window_gate,
    evaluate_technical_candidate,
    load_technical_run_metrics,
    write_technical_validation_report,
)


def test_hard_gate_candidate_passes_when_pnl_drawdown_and_winrate_improve():
    baseline = {
        "name": "baseline",
        "mode": "baseline",
        "pnl_pct": -36.976013369013735,
        "max_drawdown_pct": 39.675645963241315,
        "realized_max_drawdown_pct": 39.675645963241315,
        "closed_trades": 2525,
        "win_rate_pct": 42.69,
    }
    candidate = {
        "name": "hard_atr",
        "mode": MODE_HARD_GATE,
        "pnl_pct": 10.098901836157568,
        "max_drawdown_pct": 11.936782670486743,
        "realized_max_drawdown_pct": 11.936782670486743,
        "closed_trades": 1466,
        "win_rate_pct": 45.36,
    }

    result = evaluate_technical_candidate(baseline, candidate)

    assert result["promotion_eligible"] is True
    assert result["failures"] == []
    assert result["deltas"]["pnl_pct"] == pytest.approx(47.0749152051713)
    assert result["deltas"]["max_drawdown_pct"] == pytest.approx(-27.738863292754572)


def test_hard_gate_rejects_when_candidate_only_reduces_trades():
    baseline = {
        "pnl_pct": 12.0,
        "max_drawdown_pct": 18.0,
        "realized_max_drawdown_pct": 18.0,
        "closed_trades": 900,
        "win_rate_pct": 51.0,
    }
    candidate = {
        "name": "thin_hard_gate",
        "mode": MODE_HARD_GATE,
        "pnl_pct": 8.0,
        "max_drawdown_pct": 10.0,
        "realized_max_drawdown_pct": 10.0,
        "closed_trades": 12,
        "win_rate_pct": 50.0,
    }

    result = evaluate_technical_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert "pnl_pct_delta>0" in result["failures"]
    assert "closed_trades>=100" in result["failures"]
    assert "win_rate_delta>=0" in result["failures"]


def test_atr_sizing_candidate_can_pass_with_drawdown_reduction_and_small_pnl_giveback():
    baseline = {
        "pnl_pct": 20.0,
        "max_drawdown_pct": 15.0,
        "realized_max_drawdown_pct": 15.0,
        "closed_trades": 500,
        "win_rate_pct": 50.0,
    }
    candidate = {
        "name": "atr_sizing",
        "mode": MODE_ATR_SIZING,
        "pnl_pct": 19.4,
        "max_drawdown_pct": 9.0,
        "realized_max_drawdown_pct": 9.0,
        "closed_trades": 480,
        "win_rate_pct": 49.0,
    }

    result = evaluate_technical_candidate(baseline, candidate)

    assert result["promotion_eligible"] is True
    assert result["failures"] == []


def test_hard_gate_rejects_candidate_with_bad_sampled_equity_windows():
    baseline = {
        "pnl_pct": 10.0,
        "max_drawdown_pct": 12.0,
        "realized_max_drawdown_pct": 12.0,
        "closed_trades": 800,
        "win_rate_pct": 45.0,
        "equity_curve": [100.0, 101.0, 102.0, 103.0, 104.0],
    }
    candidate = {
        "name": "aggregate_passes_but_path_is_bad",
        "mode": MODE_HARD_GATE,
        "pnl_pct": 18.0,
        "max_drawdown_pct": 9.0,
        "realized_max_drawdown_pct": 9.0,
        "closed_trades": 700,
        "win_rate_pct": 48.0,
        "equity_curve": [100.0, 99.0, 101.0, 100.0, 104.0],
    }

    result = evaluate_technical_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert "equity_window_gate" in result["failures"]
    assert result["window_gate"]["summary"]["windows"] == 4
    assert (
        result["window_gate"]["summary"]["pnl_delta_positive_windows"]
        < result["window_gate"]["summary"]["windows"]
    )


def test_hard_gate_rejects_candidate_with_bad_temporal_trade_windows():
    baseline = {
        "pnl_pct": 10.0,
        "max_drawdown_pct": 12.0,
        "realized_max_drawdown_pct": 12.0,
        "closed_trades": 800,
        "win_rate_pct": 45.0,
        "trade_windows": [
            {"window": "window_01", "net_pnl": 10.0, "max_drawdown_pnl": 1.0},
            {"window": "window_02", "net_pnl": 10.0, "max_drawdown_pnl": 1.0},
            {"window": "window_03", "net_pnl": 10.0, "max_drawdown_pnl": 1.0},
            {"window": "window_04", "net_pnl": 10.0, "max_drawdown_pnl": 1.0},
        ],
    }
    candidate = {
        "name": "aggregate_passes_but_trade_windows_are_bad",
        "mode": MODE_HARD_GATE,
        "pnl_pct": 18.0,
        "max_drawdown_pct": 9.0,
        "realized_max_drawdown_pct": 9.0,
        "closed_trades": 700,
        "win_rate_pct": 48.0,
        "trade_windows": [
            {"window": "window_01", "net_pnl": -5.0, "max_drawdown_pnl": 4.0},
            {"window": "window_02", "net_pnl": -3.0, "max_drawdown_pnl": 3.0},
            {"window": "window_03", "net_pnl": 2.0, "max_drawdown_pnl": 2.0},
            {"window": "window_04", "net_pnl": 20.0, "max_drawdown_pnl": 1.0},
        ],
    }

    result = evaluate_technical_candidate(baseline, candidate)

    assert result["promotion_eligible"] is False
    assert "temporal_trade_window_gate" in result["failures"]
    assert result["window_gate"]["source"] == "walk_forward_temporal_trade_windows"
    assert result["window_gate"]["summary"]["pnl_delta_positive_windows"] == 1


def test_equity_windows_from_curve_calculates_return_and_drawdown():
    windows = equity_windows_from_curve(
        [100.0, 110.0, 105.0, 120.0],
        window_count=3,
    )

    assert len(windows) == 3
    assert windows[0]["window"] == "window_01"
    assert windows[0]["start_equity"] == 100.0
    assert windows[0]["end_equity"] == 110.0
    assert windows[0]["pnl_pct"] == pytest.approx(10.0)
    assert windows[1]["pnl_pct"] == pytest.approx(-4.545454545)
    assert windows[1]["max_drawdown_pct"] == pytest.approx(4.545454545)


def test_equity_window_gate_rejects_mismatched_sample_counts():
    baseline = {"equity_curve": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}
    candidate = {"equity_curve": [100.0, 102.0, 104.0, 106.0, 108.0]}

    result = evaluate_equity_window_gate(baseline, candidate)

    assert result is not None
    assert result["promotion_eligible"] is False
    assert "equity_window_count_match" in result["failures"]


def test_load_technical_run_metrics_prefers_clean_panteon_scope(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps(
            {
                "initial_capital": 100.0,
                "clean_panteon_max_drawdown_pct": 3.0,
                "clean_panteon_realized_max_drawdown_pct": 2.5,
                "clean_panteon_equity_curve": [100.0, 103.0, 111.0],
                "live_session": {
                    "clean_panteon_total_pnl_usd": 11.0,
                    "clean_panteon_pnl_pct": 11.0,
                    "clean_panteon_max_drawdown_pct": 3.0,
                    "clean_panteon_realized_max_drawdown_pct": 2.5,
                    "clean_real_closed_trades": 20,
                    "clean_real_successful_trades": 12,
                    "clean_real_unsuccessful_trades": 8,
                    "panteon_owned_total_pnl_usd": -99.0,
                    "panteon_owned_pnl_pct": -99.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "flash_attribution_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "selected_signals": 44,
                    "filled_signals": 33,
                    "closed_trades": 19,
                    "winning_trades": 10,
                    "losing_trades": 9,
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "run_summary.json").write_text(
        json.dumps({"bars_processed": 123}),
        encoding="utf-8",
    )
    (run / "walk_forward_report.json").write_text(
        json.dumps(
            {
                "temporal_windows": {
                    "windows": [
                        {
                            "window": "window_01",
                            "bar_start": 1,
                            "bar_end": 50,
                            "stats": {
                                "closed_trades": 3,
                                "net_pnl": 4.0,
                                "max_drawdown_pnl": 1.5,
                            },
                        },
                        {
                            "window": "window_02",
                            "bar_start": 51,
                            "bar_end": 100,
                            "stats": {
                                "closed_trades": 2,
                                "net_pnl": -1.0,
                                "max_drawdown_pnl": 2.5,
                            },
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = load_technical_run_metrics(run, name="hard_atr", mode=MODE_HARD_GATE)

    assert metrics["name"] == "hard_atr"
    assert metrics["mode"] == MODE_HARD_GATE
    assert metrics["pnl_usd"] == 11.0
    assert metrics["pnl_pct"] == 11.0
    assert metrics["max_drawdown_pct"] == 3.0
    assert metrics["realized_max_drawdown_pct"] == 2.5
    assert metrics["closed_trades"] == 20
    assert metrics["win_rate_pct"] == pytest.approx(60.0)
    assert metrics["selected_signals"] == 44
    assert metrics["filled_signals"] == 33
    assert metrics["bars_processed"] == 123
    assert metrics["equity_curve"] == [100.0, 103.0, 111.0]
    assert metrics["equity_points"] == 3
    assert metrics["trade_windows"][0]["window"] == "window_01"
    assert metrics["trade_windows"][0]["net_pnl"] == 4.0
    assert metrics["trade_windows"][1]["max_drawdown_pnl"] == 2.5


def test_build_technical_validation_report_summarizes_candidates():
    baseline = {
        "name": "baseline",
        "mode": "baseline",
        "pnl_pct": -5.0,
        "max_drawdown_pct": 20.0,
        "realized_max_drawdown_pct": 20.0,
        "closed_trades": 1000,
        "win_rate_pct": 42.0,
    }
    candidates = [
        {
            "name": "hard_atr",
            "mode": MODE_HARD_GATE,
            "pnl_pct": 4.0,
            "max_drawdown_pct": 11.0,
            "realized_max_drawdown_pct": 11.0,
            "closed_trades": 400,
            "win_rate_pct": 45.0,
        },
        {
            "name": "bad_hard",
            "mode": MODE_HARD_GATE,
            "pnl_pct": -6.0,
            "max_drawdown_pct": 19.0,
            "realized_max_drawdown_pct": 19.0,
            "closed_trades": 20,
            "win_rate_pct": 40.0,
        },
    ]

    report = build_technical_validation_report(baseline, candidates)

    assert report["schema"] == "panteon_flash_technical_validation_report_v1"
    assert report["summary"]["candidates"] == 2
    assert report["summary"]["promotion_eligible"] == 1
    assert report["candidates"][0]["promotion_eligible"] is True
    assert report["candidates"][1]["promotion_eligible"] is False


def test_write_technical_validation_report_includes_window_gate_summary(tmp_path):
    baseline = {
        "name": "baseline",
        "mode": "baseline",
        "pnl_pct": -4.0,
        "max_drawdown_pct": 10.0,
        "realized_max_drawdown_pct": 10.0,
        "closed_trades": 1000,
        "win_rate_pct": 42.0,
        "equity_curve": [100.0, 99.0, 98.0, 97.0, 96.0],
    }
    candidate = {
        "name": "hard_atr",
        "mode": MODE_HARD_GATE,
        "pnl_pct": 4.0,
        "max_drawdown_pct": 6.0,
        "realized_max_drawdown_pct": 6.0,
        "closed_trades": 400,
        "win_rate_pct": 45.0,
        "equity_curve": [100.0, 101.0, 102.0, 103.0, 104.0],
    }
    output_json = tmp_path / "technical_validation_report.json"
    output_md = tmp_path / "technical_validation_report.md"

    write_technical_validation_report(
        output_json=output_json,
        output_md=output_md,
        baseline=baseline,
        candidates=(candidate,),
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    md = output_md.read_text(encoding="utf-8")

    assert report["candidates"][0]["window_gate"]["promotion_eligible"] is True
    assert "Window gate" in md
    assert "| hard_atr | hard_gate | PASS | PASS |" in md
