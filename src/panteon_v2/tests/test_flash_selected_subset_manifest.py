from __future__ import annotations

import json
from pathlib import Path

from panteon_v2.analysis.flash_selected_subset_manifest import (
    build_flash_selected_subset_manifest,
)


def _write_attribution(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )


def test_selected_subset_manifest_boosts_only_cross_period_positive_cells(tmp_path):
    y2025 = tmp_path / "2025.json"
    full = tmp_path / "full.json"
    _write_attribution(
        y2025,
        [
            {
                "actor_key": "ensemble:Solo_MomentumScalper",
                "symbol": "APT/USDT",
                "action": "FUT_LONG_FULL",
                "closed_trades": 4,
                "realized_pnl_usd": 24.0,
            },
            {
                "actor_key": "ensemble:Solo_MomentumScalper",
                "symbol": "BNB/USDT",
                "action": "FUT_LONG_FULL",
                "closed_trades": 5,
                "realized_pnl_usd": -10.0,
            },
        ],
    )
    _write_attribution(
        full,
        [
            {
                "actor_key": "ensemble:Solo_MomentumScalper",
                "symbol": "APT/USDT",
                "action": "FUT_LONG_FULL",
                "closed_trades": 12,
                "realized_pnl_usd": 114.0,
            },
            {
                "actor_key": "ensemble:Solo_MomentumScalper",
                "symbol": "BNB/USDT",
                "action": "FUT_LONG_FULL",
                "closed_trades": 15,
                "realized_pnl_usd": 5.0,
            },
        ],
    )

    manifest = build_flash_selected_subset_manifest(
        confirmation_attribution_paths=(y2025, full),
        min_closed_trades=4,
        score_boost=0.25,
    )

    boosts = {item["signal_key"]: item["boost"] for item in manifest["score_boosts"]}
    assert boosts == {
        "ensemble:Solo_MomentumScalper|APT/USDT|FUT_LONG_FULL": 0.25
    }
    risk = {
        item["signal_key"]: item["risk_mult"]
        for item in manifest["risk_mult_overrides"]
    }
    assert risk["ensemble:Solo_MomentumScalper|APT/USDT|FUT_LONG_FULL"] == 1.15
    assert risk["ensemble:Solo_MomentumScalper|BNB/USDT|FUT_LONG_FULL"] == 0.75
    assert manifest["do_not_demote_signal_keys"] == [
        "ensemble:Solo_MomentumScalper|APT/USDT|FUT_LONG_FULL"
    ]


def test_selected_subset_manifest_sparse_positive_cells_are_protected(tmp_path):
    full = tmp_path / "full.json"
    _write_attribution(
        full,
        [
            {
                "actor_key": "agent:LiveOIBreakout",
                "symbol": "ADA/USDT",
                "action": "SPOT_BUY_FULL",
                "closed_trades": 2,
                "realized_pnl_usd": 50.0,
            }
        ],
    )

    manifest = build_flash_selected_subset_manifest(
        confirmation_attribution_paths=(full,),
        min_closed_trades=5,
        sparse_min_pnl_usd=25.0,
    )

    assert manifest["score_boosts"] == []
    assert manifest["do_not_demote_signal_keys"] == [
        "agent:LiveOIBreakout|ADA/USDT|SPOT_BUY_FULL"
    ]


def test_selected_subset_manifest_builds_context_boosts_from_regime_stability(tmp_path):
    h1 = tmp_path / "2026_h1.json"
    y2025 = tmp_path / "2025.json"
    full = tmp_path / "full.json"
    stable_key = "ensemble:Solo_MomentumScalper|BTC/USDT|FUT_SHORT_FULL|bearish"
    unstable_key = "ensemble:Solo_MomentumScalper|ETH/USDT|FUT_LONG_FULL|neutral"
    for path, stable_pnl, unstable_pnl in (
        (h1, 12.0, 4.0),
        (y2025, 18.0, -1.0),
        (full, 40.0, 22.0),
    ):
        path.write_text(
            json.dumps(
                {
                    "context_rows": [
                        {
                            "context_key": stable_key,
                            "closed_trades": 4,
                            "realized_pnl_usd": stable_pnl,
                            "selected_signals": 4,
                        },
                        {
                            "context_key": unstable_key,
                            "closed_trades": 4,
                            "realized_pnl_usd": unstable_pnl,
                            "selected_signals": 4,
                        },
                    ],
                    "rows": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = build_flash_selected_subset_manifest(
        confirmation_attribution_paths=(h1, y2025, full),
        min_closed_trades=4,
        context_score_boost=0.15,
        context_risk_positive_mult=1.0,
    )

    boosts = {
        item["context_key"]: item["boost"]
        for item in manifest["context_score_boosts"]
    }
    risks = {
        item["context_key"]: item["risk_mult"]
        for item in manifest["context_risk_mult_overrides"]
    }
    assert boosts == {stable_key: 0.15}
    assert risks == {stable_key: 1.0}
