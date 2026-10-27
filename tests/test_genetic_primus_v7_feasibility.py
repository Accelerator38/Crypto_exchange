from __future__ import annotations

import json
from pathlib import Path

import pytest

from exia.genetic_primus.feasibility_v7 import assert_evolution_authority


def test_v7_terminal_event_result_blocks_evolution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "V7_TERMINAL_REJECTED_NO_EVENT_CAPACITY",
                "passing_event_ids": [],
                "evolution_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="evolution blocked"):
        assert_evolution_authority(manifest)


def test_v7_explicit_event_pass_opens_only_evolution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {
        "status": "V7_DATA_SOURCE_AND_EVENT_CAPACITY_ACCEPTED",
        "passing_event_ids": ["cross_market_cascade_reversal"],
        "evolution_allowed": True,
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert assert_evolution_authority(manifest) == payload
