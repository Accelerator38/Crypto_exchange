import json

from panteon_v2.analysis.flash_live_admission_attribution import (
    build_flash_live_admission_attribution,
    write_flash_live_admission_attribution,
)


def test_live_admission_attribution_groups_original_actor_denials():
    rows = [
        {
            "bar": 10,
            "raw_signal_count": 2,
            "executable_signal_count": 0,
            "n_filled": 0,
            "flash_decisions": [
                {
                    "symbol": "BNB",
                    "selected_actor": "NoTrade",
                    "original_selected_actor": "GeneticsCore",
                    "reason": "no_real_admission:score_below_threshold",
                    "candidate_rejection_counts": {
                        "inactive": 2,
                        "score_below_threshold": 1,
                    },
                },
                {
                    "symbol": "ADA",
                    "selected_actor": "NoTrade",
                    "original_selected_actor": "GeneticsRiskTight",
                    "reason": "no_real_admission:shadow_unconfirmed",
                    "top_rejected_candidates": [
                        {"label": "GeneticsRiskTight", "reason": "shadow_unconfirmed"}
                    ],
                    "candidate_rejection_counts": {
                        "shadow_unconfirmed": 1,
                    },
                },
            ],
        },
        {
            "bar": 11,
            "raw_signal_count": 1,
            "executable_signal_count": 1,
            "n_filled": 1,
            "signal_filter_details": [
                "genetics_signal_key_blocked:ETH:GeneticsRegimeAdaptiveBias",
                "genetics_shadow_unconfirmed:BNB:GeneticsCore",
                "genetics_daily_trade_limit:ADA:GeneticsCore:1",
                "genetics_regime_blocked:XRP:GeneticsRegimeAdaptiveBias:mixed_rotational",
            ],
            "flash_decisions": [
                {
                    "symbol": "ADA",
                    "selected_actor": "DefaultEnsemble",
                    "original_selected_actor": "DefaultEnsemble",
                    "reason": "selected",
                },
                {
                    "symbol": "BNB",
                    "selected_actor": "NoTrade",
                    "original_selected_actor": "GeneticsCore",
                    "reason": "no_real_admission:score_below_threshold",
                    "candidate_rejection_counts": {
                        "score_below_threshold": 1,
                    },
                },
            ],
        },
    ]

    report = build_flash_live_admission_attribution(
        rows,
        target_actors=(
            "GeneticsCore",
            "GeneticsRiskTight",
            "GeneticsRegimeAdaptiveBias",
        ),
    )

    assert report["summary"]["rows_read"] == 2
    assert report["summary"]["flash_decisions"] == 4
    assert report["summary"]["raw_signals"] == 3
    assert report["summary"]["executable_signals"] == 1
    assert report["summary"]["filled"] == 1
    assert report["summary"]["denied_original_actor_decisions"] == 3
    assert report["summary"]["target_denied_decisions"] == 3
    assert report["summary"]["candidate_detail_available"] is True
    assert report["actor_summaries"]["GeneticsCore"]["denied_decisions"] == 2
    assert report["actor_summaries"]["GeneticsCore"]["reason_counts"] == {
        "no_real_admission:score_below_threshold": 2,
    }
    assert report["actor_summaries"]["GeneticsRiskTight"]["denied_decisions"] == 1
    assert report["denial_groups"][0]["actor"] == "GeneticsCore"
    assert report["denial_groups"][0]["symbol"] == "BNB"
    assert report["denial_groups"][0]["denied_decisions"] == 2
    assert report["denial_groups"][0]["candidate_rejection_counts"] == {
        "inactive": 2,
        "score_below_threshold": 2,
    }
    assert report["signal_filter_reason_counts"] == {
        "daily_limit": 1,
        "regime_blocked": 1,
        "shadow_unconfirmed": 1,
        "signal_key_blocked": 1,
    }
    assert report["signal_filter_label_reason_counts"]["GeneticsRegimeAdaptiveBias"] == {
        "regime_blocked": 1,
        "signal_key_blocked": 1,
    }


def test_live_admission_attribution_targets_adaptive_by_default():
    rows = [
        {
            "bar": 1,
            "flash_decisions": [
                {
                    "symbol": "ETH",
                    "selected_actor": "NoTrade",
                    "original_selected_actor": "GeneticsRegimeAdaptiveBias",
                    "reason": "no_real_admission:score_below_threshold",
                }
            ],
        }
    ]

    report = build_flash_live_admission_attribution(rows)

    assert report["summary"]["target_denied_decisions"] == 1
    assert "GeneticsRegimeAdaptiveBias" in report["actor_summaries"]


def test_live_admission_attribution_writer_outputs_json_and_markdown(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "bar": 1,
            "flash_decisions": [
                {
                    "symbol": "BTC",
                    "selected_actor": "NoTrade",
                    "original_selected_actor": "GeneticsCore",
                    "reason": "no_real_admission:shadow_unconfirmed",
                }
            ],
        }
    ]
    with (run_dir / "causal_entry_decisions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    json_path, md_path = write_flash_live_admission_attribution(
        run_dir,
        target_actors=("GeneticsCore",),
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["target_denied_decisions"] == 1
    assert md_path.exists()
    assert "GeneticsCore" in md_path.read_text(encoding="utf-8")
