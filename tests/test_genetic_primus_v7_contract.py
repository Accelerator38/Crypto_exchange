from __future__ import annotations

from pathlib import Path

from exia.genetic_primus.contracts_v7 import load_v7_contract


ROOT = Path(__file__).resolve().parents[1]


def test_v7_contract_is_fail_closed_and_fixed() -> None:
    contract = load_v7_contract(ROOT / "configs/genetic_primus_v7_contract.json", root=ROOT)
    assert contract.payload["safety"]["orders_enabled"] is False
    assert contract.payload["execution_source"]["holding_bars"] == 3
    assert contract.feasibility["minimum_events_per_fold"] == 30
    assert len(contract.folds) == 5
