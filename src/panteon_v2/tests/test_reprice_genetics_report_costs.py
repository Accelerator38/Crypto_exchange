from __future__ import annotations

import json
from pathlib import Path

import pytest


def _report() -> dict:
    return {
        "exchange": "MEXC",
        "genomes": [
            {
                "path": "candidate.npy",
                "modes": [
                    {
                        "mode": "fee_fixed_nextbar",
                        "period_rets": [1.0, -0.5],
                        "period_stats": {
                            "mean_ret": 0.25,
                            "min_ret": -0.5,
                            "max_ret": 1.0,
                            "positive_period_pct": 50.0,
                        },
                        "contract_metrics": {
                            "periods": [
                                {
                                    "period": "2024-01",
                                    "turnover_rate": 1.0,
                                    "effective_turnover_rate": 1.0,
                                    "cost_pct": 0.0,
                                    "slippage_pct": 0.0,
                                },
                                {
                                    "period": "2024-02",
                                    "turnover_rate": 0.5,
                                    "effective_turnover_rate": 0.5,
                                    "cost_pct": 0.0,
                                    "slippage_pct": 0.0,
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }


def test_reprice_report_applies_exchange_fee_slippage_to_zero_cost_periods():
    from tools.reprice_genetics_report_costs import reprice_report_costs

    repriced = reprice_report_costs(_report(), exchange="MEXC")
    mode = repriced["genomes"][0]["modes"][0]
    periods = mode["contract_metrics"]["periods"]

    assert repriced["cost_reprice"]["exchange"] == "MEXC"
    assert repriced["cost_reprice"]["cost_model"] == "turnover"
    assert periods[0]["cost_pct"] == 0.07
    assert periods[0]["slippage_pct"] == 0.01
    assert periods[1]["cost_pct"] == 0.035
    assert periods[1]["slippage_pct"] == 0.005
    assert mode["period_rets"] == [0.92, -0.54]
    assert mode["period_stats"]["n_periods"] == 2
    assert mode["period_stats"]["mean_ret"] == pytest.approx(0.19)
    assert mode["period_stats"]["median_ret"] == pytest.approx(0.19)
    assert mode["period_stats"]["best_period_idx"] == 0
    assert mode["period_stats"]["worst_period_idx"] == 1
    assert mode["contract_metrics"]["mean_cost_pct"] == pytest.approx((0.07 + 0.035) / 2)
    assert mode["contract_metrics"]["mean_slippage_pct"] == pytest.approx((0.01 + 0.005) / 2)


def test_reprice_cli_writes_repriced_report(tmp_path: Path):
    from tools import reprice_genetics_report_costs as tool

    report_path = tmp_path / "report.json"
    out_path = tmp_path / "repriced.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    exit_code = tool.main(
        [
            "--report",
            str(report_path),
            "--exchange",
            "MEXC",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["cost_reprice"]["exchange"] == "MEXC"
    assert payload["genomes"][0]["modes"][0]["period_rets"] == [0.92, -0.54]
