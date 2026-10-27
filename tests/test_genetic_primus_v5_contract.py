from __future__ import annotations

import json
from pathlib import Path

import pytest

from exia.genetic_primus.contracts_v5 import load_v5_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/genetic_primus_v5_contract.json"


def test_v5_contract_is_fixed_and_fail_closed() -> None:
    contract = load_v5_contract(CONTRACT, root=ROOT)
    assert contract.sha256
    assert [fold["name"] for fold in contract.folds] == ["WF1", "WF2", "WF3", "WF4"]
    assert contract.execution_source["holding_bars"] == 6
    assert contract.feasibility["stress_round_trip_cost_bps"] == 24.0
    assert contract.payload["evolution"]["fallback_selection_allowed"] is False
    assert all(value is False for value in contract.payload["safety"].values())


def test_v5_contract_rejects_search_extension(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["evolution"]["extension_allowed"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be extended"):
        load_v5_contract(changed, root=ROOT)
