from __future__ import annotations

import json


def test_gate_funnel_analyzer_counts_rejections(tmp_path):
    from tools.analyze_panteon_gate_funnel import analyze_files

    path = tmp_path / "causal_entry_decisions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "t1",
                        "raw_signal_count": 0,
                        "n_signals": 0,
                        "real_universe_symbol_reject_reasons": {
                            "BTC": "min_executable_notional $6 exceeds approved $1"
                        },
                        "flash_decisions": [
                            {
                                "symbol": "BTC",
                                "selected_actor": "NoTrade",
                                "reason": "no_real_admission:expected_edge_below_cost",
                                "candidate_count": 2,
                                "rejected_candidate_count": 1,
                                "eligible_candidate_count": 1,
                                "candidate_rejection_counts": {
                                    "expected_edge_below_cost": 1
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "t2",
                        "raw_signal_count": 1,
                        "executable_signal_count": 1,
                        "n_filled": 1,
                        "flash_decisions": [
                            {
                                "symbol": "ETH",
                                "selected_actor": "DefaultEnsemble",
                                "reason": "selected",
                                "candidate_count": 1,
                                "rejected_candidate_count": 0,
                                "eligible_candidate_count": 1,
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = analyze_files([path])

    assert report["bars"] == 2
    assert report["decisions"] == 2
    assert report["real_signal_bars"] == 1
    assert report["filled_bars"] == 1
    assert report["decision_reasons"]["no_real_admission:expected_edge_below_cost"] == 1
    assert report["candidate_rejection_reasons"]["expected_edge_below_cost"] == 1
    assert report["real_universe_reasons"]["min_executable_notional"] == 1
    assert report["selected_actors"]["NoTrade"] == 1
    assert report["selected_actors"]["DefaultEnsemble"] == 1
    assert report["candidate_totals"]["candidate_count"] == 3
    assert report["candidate_totals"]["rejected_candidate_count"] == 1


def test_gate_funnel_analyzer_cli_writes_json_and_markdown(tmp_path):
    from tools.analyze_panteon_gate_funnel import main

    path = tmp_path / "causal_entry_decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": "t1",
                "raw_signal_count": 0,
                "flash_gate_funnel_by_symbol": {
                    "DOGE": {
                        "candidate_count": 0,
                        "eligible_candidate_count": 0,
                        "rejected_candidate_count": 0,
                        "top_blocker": "min_executable_notional",
                    }
                },
                "flash_decisions": [
                    {
                        "symbol": "DOGE",
                        "selected_actor": "NoTrade",
                        "reason": "min_executable_notional",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "gate_funnel_report.json"

    rc = main([str(path), "--out", str(out)])

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert data["bars"] == 1
    assert data["gate_funnel_top_blockers"]["min_executable_notional"] == 1
    assert "| metric | value |" in markdown
    assert "min_executable_notional" in markdown
