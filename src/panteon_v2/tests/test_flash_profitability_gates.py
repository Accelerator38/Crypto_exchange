from __future__ import annotations

import json

from panteon_v2.analysis.flash_profitability_gates import (
    PERIOD_2025,
    PERIOD_2026_H1,
    PERIOD_FULL,
    build_profitability_gate_report,
    evaluate_profitability_gates,
    load_flash_run_metrics,
)


def test_profitability_gate_passes_selected_subset_rounded_h1_baseline():
    result = evaluate_profitability_gates(
        {
            PERIOD_2026_H1: {"pnl_usd": 112.71911066205612},
            PERIOD_2025: {"pnl_usd": 221.80116250903365},
            PERIOD_FULL: {
                "pnl_usd": 1063.1031600787285,
                "max_drawdown_pct": 4.115268150939742,
            },
        }
    )

    assert result["promotion_eligible"] is True
    assert result["failures"] == []


def test_profitability_gate_rejects_softlock_full_without_period_support():
    report = build_profitability_gate_report(
        [
            {
                "name": "ultima_softlock_full",
                "periods": {
                    PERIOD_FULL: {
                        "pnl_usd": 463.08,
                        "max_drawdown_pct": 3.94,
                    }
                },
            }
        ]
    )

    candidate = report["candidates"][0]
    assert candidate["promotion_eligible"] is False
    assert f"{PERIOD_2026_H1}.pnl_usd>=$112.72" in candidate["gate"]["failures"]
    assert f"{PERIOD_2025}.pnl_usd>=$221.80" in candidate["gate"]["failures"]
    assert f"{PERIOD_FULL}.pnl_usd>$1063.10" in candidate["gate"]["failures"]


def test_load_flash_run_metrics_prefers_clean_panteon_scope(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(
        json.dumps(
            {
                "initial_capital": 100.0,
                "pnl_usd": 99.0,
                "live_session": {
                    "clean_panteon_total_pnl_usd": 3.0,
                    "clean_panteon_pnl_pct": 3.0,
                    "clean_panteon_max_drawdown_pct": 1.2,
                    "clean_real_closed_trades": 2,
                    "panteon_owned_total_pnl_usd": 9.0,
                    "panteon_owned_pnl_pct": 9.0,
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = load_flash_run_metrics(run, name="candidate", period=PERIOD_2025)

    assert metrics["pnl_usd"] == 3.0
    assert metrics["pnl_pct"] == 3.0
    assert metrics["max_drawdown_pct"] == 1.2
    assert metrics["closed_trades"] == 2
