from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "diagnose_promotion_derived_components.py"
    spec = importlib.util.spec_from_file_location("promotion_diagnostics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_promotion_diagnostics_classifies_allowlisted_why_not_promoted_reasons(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "regime": "neutral",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "reason": "selected",
                        "candidate_count": 4,
                        "candidates": [
                            {
                                "label": "CarryFlowAgentV2",
                                "actor_key": "agent:CarryFlowAgentV2",
                                "actor_type": "agent",
                                "action": "HOLD",
                                "reason": "inactive",
                                "rejected": True,
                                "closed_trades": 8,
                                "pnl_net_pct": 1.6,
                            },
                            {
                                "label": "MomentumScalper",
                                "actor_key": "agent:MomentumScalper",
                                "actor_type": "agent",
                                "action": "FUT_LONG_FULL",
                                "reason": "eligible",
                                "rejected": False,
                                "closed_trades": 3,
                                "pnl_net_pct": 0.9,
                            },
                            {
                                "label": "LiveCrashHunter",
                                "actor_key": "agent:LiveCrashHunter",
                                "actor_type": "agent",
                                "action": "FUT_SHORT_FULL",
                                "reason": "eligible",
                                "rejected": False,
                                "closed_trades": 5,
                                "pnl_net_pct": -0.2,
                            },
                        ],
                    }
                ],
            },
            {
                "bar": 2,
                "timestamp": "2026-01-01T01:00:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "LiveCrashHunter",
                        "reason": "promotion_derived_router",
                        "selected_reasons": ["selected", "promotion_derived_router"],
                        "candidate_count": 4,
                        "candidates": [
                            {
                                "label": "CarryFlowAgentV2",
                                "actor_key": "agent:CarryFlowAgentV2",
                                "actor_type": "agent",
                                "action": "FUT_CLOSE_ALL",
                                "reason": "eligible",
                                "rejected": False,
                                "closed_trades": 8,
                                "pnl_net_pct": 1.6,
                            },
                            {
                                "label": "MomentumScalper",
                                "actor_key": "agent:MomentumScalper",
                                "actor_type": "agent",
                                "action": "FUT_LONG_FULL",
                                "reason": "foreign_position_owner",
                                "rejected": True,
                                "closed_trades": 8,
                                "pnl_net_pct": 1.6,
                            },
                            {
                                "label": "LiveCrashHunter",
                                "actor_key": "agent:LiveCrashHunter",
                                "actor_type": "agent",
                                "action": "FUT_SHORT_FULL",
                                "reason": "promotion_derived_router",
                                "rejected": False,
                                "closed_trades": 8,
                                "pnl_net_pct": 1.6,
                            },
                        ],
                    }
                ],
            },
            {
                "bar": 3,
                "timestamp": "2026-01-01T02:00:00+00:00",
                "regime": "neutral",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "reason": "selected",
                        "candidate_count": 12,
                        "candidates": [],
                        "top_rejected_candidates": [
                            {
                                "label": "OtherActor",
                                "actor_type": "agent",
                                "action": "FUT_LONG_FULL",
                                "reason": "score_below_threshold",
                            }
                        ],
                    }
                ],
            },
        ],
    )

    report = module.build_promotion_derived_diagnostics(
        [path],
        actor_labels=("CarryFlowAgentV2", "MomentumScalper", "LiveCrashHunter"),
        min_closed_trades=5,
        min_expectancy=0.0,
    )

    assert report["summary"]["bars_read"] == 3
    assert report["summary"]["detail_rows"] == 9
    assert report["summary"]["promoted_rows"] == 1
    assert report["reason_counts"]["inactive"] == 1
    assert report["reason_counts"]["closed<trades"] == 1
    assert report["reason_counts"]["expectancy<=0"] == 1
    assert report["reason_counts"]["non-open action"] == 1
    assert report["reason_counts"]["foreign owner"] == 1
    assert report["reason_counts"]["promoted"] == 1
    assert report["reason_counts"]["not_observed_in_compact_audit"] == 3
    rows = {
        (row["bar"], row["actor"]): row
        for row in report["details"]
    }
    assert rows[(1, "MomentumScalper")]["closed_trades"] == 3
    assert rows[(1, "LiveCrashHunter")]["rolling_expectancy"] == -0.04
    assert rows[(2, "LiveCrashHunter")]["promoted"] is True


def test_promotion_diagnostics_cli_writes_json_markdown_and_details(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "reason": "selected",
                        "candidate_count": 1,
                        "candidates": [
                            {
                                "label": "CarryFlowAgentV2",
                                "actor_type": "agent",
                                "action": "HOLD",
                                "reason": "inactive",
                                "rejected": True,
                                "closed_trades": 0,
                                "pnl_net_pct": 0.0,
                            }
                        ],
                    }
                ],
            }
        ],
    )
    out = tmp_path / "promotion_diagnostics.json"

    rc = module.main([
        str(path),
        "--out",
        str(out),
        "--actor-label",
        "CarryFlowAgentV2",
    ])

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    details = out.with_suffix(".details.jsonl").read_text(encoding="utf-8")
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert data["reason_counts"]["inactive"] == 1
    assert "CarryFlowAgentV2" in details
    assert "Why Not Promoted" in markdown


def test_promotion_diagnostics_maps_virtual_label_to_active_solo_wrapper(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "Solo_GeneticsCore",
                        "reason": "promotion_derived_router",
                        "selected_reasons": ["selected", "promotion_derived_router"],
                        "candidate_count": 2,
                        "candidates": [
                            {
                                "label": "Solo_GeneticsCore",
                                "actor_key": "ensemble:Solo_GeneticsCore",
                                "actor_type": "ensemble",
                                "action": "FUT_SHORT_FULL",
                                "reason": "promotion_derived_router",
                                "rejected": False,
                                "closed_trades": 8,
                                "pnl_net_pct": 1.6,
                            }
                        ],
                    }
                ],
            }
        ],
    )

    report = module.build_promotion_derived_diagnostics(
        [path],
        actor_labels=("V_GeneticsCore",),
        min_closed_trades=5,
        min_expectancy=0.0,
    )

    assert report["summary"]["promoted_rows"] == 1
    assert report["reason_counts"]["promoted"] == 1
    assert report["details"][0]["actor"] == "V_GeneticsCore"
    assert report["details"][0]["audit_source"] == "candidates"


def test_promotion_diagnostics_prefers_open_family_candidate_over_inactive_wrapper(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "XRP/USDT",
                        "selected_actor": "NoTrade",
                        "reason": "no_real_admission:real_loss_veto",
                        "candidate_count": 3,
                        "top_rejected_candidates": [
                            {
                                "label": "Solo_GeneticsCore",
                                "actor_key": "ensemble:Solo_GeneticsCore",
                                "actor_type": "ensemble",
                                "action": "HOLD",
                                "reason": "inactive",
                                "rank": 2,
                            },
                            {
                                "label": "GeneticsCore",
                                "actor_key": "agent:GeneticsCore",
                                "actor_type": "agent",
                                "action": "FUT_SHORT_FULL",
                                "reason": "real_loss_veto",
                                "rank": 3,
                            },
                        ],
                    }
                ],
            }
        ],
    )

    report = module.build_promotion_derived_diagnostics(
        [path],
        actor_labels=("V_GeneticsCore",),
        min_closed_trades=5,
        min_expectancy=0.0,
    )

    row = report["details"][0]
    assert row["action"] == "FUT_SHORT_FULL"
    assert row["candidate_reason"] == "real_loss_veto"
    assert row["why_not_promoted"] == "real_loss_veto"
