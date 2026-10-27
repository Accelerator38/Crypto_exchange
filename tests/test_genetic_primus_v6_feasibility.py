from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from exia.genetic_primus.feasibility_v6 import apply_cooldown, assert_evolution_authority


def test_v6_event_cooldown_prevents_overlapping_h6_positions() -> None:
    mask = pd.Series([True] * 15)
    selected = apply_cooldown(mask, 6)
    assert selected[selected].index.tolist() == [0, 7, 14]


def test_v6_terminal_event_gate_blocks_evolution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "V6_TERMINAL_REJECTED_NO_EVENT_CAPACITY",
                "passing_event_ids": [],
                "evolution_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="evolution blocked"):
        assert_evolution_authority(manifest)


def test_v6_only_explicit_event_acceptance_opens_evolution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {
        "status": "V6_DATA_SOURCE_AND_EVENT_CAPACITY_ACCEPTED",
        "passing_event_ids": ["oi_taker_price_continuation"],
        "evolution_allowed": True,
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert assert_evolution_authority(manifest) == payload
