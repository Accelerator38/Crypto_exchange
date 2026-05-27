from __future__ import annotations

import json
from pathlib import Path

from panteon_v2.analysis.flash_context_selected_subset_manifest import (
    build_flash_context_selected_subset_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_context_selected_subset_manifest_preserves_base_and_adds_context_only(tmp_path):
    base = tmp_path / "base.json"
    h1 = tmp_path / "h1.json"
    y2025 = tmp_path / "2025.json"
    stable_context_key = "ensemble:Solo_LiveCrashHunter|LINK/USDT|FUT_SHORT_FULL|bearish"

    _write_json(
        base,
        {
            "schema": "panteon_flash_selected_subset_manifest_v1",
            "score_boosts": [
                {
                    "signal_key": "ensemble:Base|BTC/USDT|FUT_SHORT_FULL",
                    "boost": 0.25,
                }
            ],
            "do_not_demote_signal_keys": [
                "ensemble:Base|BTC/USDT|FUT_SHORT_FULL"
            ],
            "risk_mult_overrides": [
                {
                    "signal_key": "ensemble:Base|BTC/USDT|FUT_SHORT_FULL",
                    "risk_mult": 1.15,
                }
            ],
            "context_score_boosts": [],
            "context_risk_mult_overrides": [],
        },
    )
    for path, pnl in ((h1, 6.0), (y2025, 14.0)):
        _write_json(
            path,
            {
                "rows": [
                    {
                        "actor_key": "ensemble:GeneratedSignalBoost",
                        "symbol": "ETH/USDT",
                        "action": "FUT_LONG_FULL",
                        "closed_trades": 4,
                        "realized_pnl_usd": 20.0,
                    }
                ],
                "context_rows": [
                    {
                        "context_key": stable_context_key,
                        "closed_trades": 1,
                        "realized_pnl_usd": pnl,
                        "selected_signals": 1,
                    }
                ],
            },
        )

    manifest = build_flash_context_selected_subset_manifest(
        base_manifest_path=base,
        confirmation_attribution_paths=(h1, y2025),
        context_min_closed_trades=1,
        context_score_boost=0.15,
        context_risk_positive_mult=1.0,
    )

    assert manifest["score_boosts"] == [
        {
            "signal_key": "ensemble:Base|BTC/USDT|FUT_SHORT_FULL",
            "boost": 0.25,
        }
    ]
    assert manifest["risk_mult_overrides"] == [
        {
            "signal_key": "ensemble:Base|BTC/USDT|FUT_SHORT_FULL",
            "risk_mult": 1.15,
        }
    ]
    assert manifest["context_score_boosts"] == [
        {
            "context_key": stable_context_key,
            "boost": 0.15,
            "reason": "actor_symbol_action_regime_positive_in_all_confirmation_periods",
            "periods": [
                {
                    "closed_trades": 1,
                    "realized_pnl_usd": 6.0,
                    "selected_signals": 1,
                    "regime": "bearish",
                },
                {
                    "closed_trades": 1,
                    "realized_pnl_usd": 14.0,
                    "selected_signals": 1,
                    "regime": "bearish",
                },
            ],
        }
    ]
    assert manifest["context_risk_mult_overrides"] == [
        {
            "context_key": stable_context_key,
            "risk_mult": 1.0,
            "reason": "actor_symbol_action_regime_positive_in_all_confirmation_periods",
        }
    ]


def test_context_selected_subset_manifest_dedupes_existing_context_key(tmp_path):
    base = tmp_path / "base.json"
    h1 = tmp_path / "h1.json"
    y2025 = tmp_path / "2025.json"
    key = "ensemble:Solo_LiveCrashHunter|LINK/USDT|FUT_SHORT_FULL|bearish"

    _write_json(
        base,
        {
            "schema": "panteon_flash_selected_subset_manifest_v1",
            "score_boosts": [],
            "do_not_demote_signal_keys": [],
            "risk_mult_overrides": [],
            "context_score_boosts": [
                {"context_key": key, "boost": 0.05, "reason": "base"}
            ],
            "context_risk_mult_overrides": [
                {"context_key": key, "risk_mult": 0.9, "reason": "base"}
            ],
        },
    )
    for path in (h1, y2025):
        _write_json(
            path,
            {
                "rows": [],
                "context_rows": [
                    {
                        "context_key": key,
                        "closed_trades": 1,
                        "realized_pnl_usd": 1.0,
                        "selected_signals": 1,
                    }
                ],
            },
        )

    manifest = build_flash_context_selected_subset_manifest(
        base_manifest_path=base,
        confirmation_attribution_paths=(h1, y2025),
        context_min_closed_trades=1,
    )

    assert manifest["context_score_boosts"] == [
        {"context_key": key, "boost": 0.05, "reason": "base"}
    ]
    assert manifest["context_risk_mult_overrides"] == [
        {"context_key": key, "risk_mult": 0.9, "reason": "base"}
    ]
